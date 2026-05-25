#!/usr/bin/env python3
"""Export the RGB-D ORB-SLAM sparse map to PLY and JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _as_list(values) -> list:
    if values is None:
        return []
    if hasattr(values, "to_list"):
        return values.to_list()
    return list(values)


def _is_bad(obj) -> bool:
    fn = getattr(obj, "is_bad", None)
    if callable(fn):
        return bool(fn())
    return bool(getattr(obj, "_is_bad", False))


def _point_replaced(point) -> bool:
    replacement = getattr(point, "replacement", None)
    if replacement is None and hasattr(point, "get_replacement"):
        try:
            replacement = point.get_replacement()
        except Exception:
            replacement = None
    return replacement is not None and replacement is not point


def _pose_matrix(keyframe, inverse: bool = False) -> np.ndarray:
    method = "Twc" if inverse else "Tcw"
    fn = getattr(keyframe, method, None)
    if callable(fn):
        return np.asarray(fn(), dtype=np.float64).reshape(4, 4)
    Tcw = np.asarray(getattr(keyframe, "Tcw", np.eye(4)), dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(Tcw) if inverse else Tcw


def collect_exportable_points(map_object) -> tuple[np.ndarray, np.ndarray, list[int]]:
    points = []
    colors = []
    ids = []
    for point in _as_list(map_object.get_points() if hasattr(map_object, "get_points") else getattr(map_object, "points", [])):
        if point is None or _is_bad(point) or _point_replaced(point):
            continue
        get_position = getattr(point, "get_position", None)
        position = get_position() if callable(get_position) else getattr(point, "position", None)
        if position is None:
            continue
        position = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(position)):
            continue
        color = getattr(point, "color", None)
        if color is None:
            color = np.array([230, 230, 230], dtype=np.uint8)
        color = np.asarray(color, dtype=np.uint8).reshape(-1)[:3]
        if len(color) < 3:
            color = np.array([230, 230, 230], dtype=np.uint8)
        points.append(position)
        colors.append(color)
        ids.append(int(getattr(point, "id", len(ids))))
    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8), []
    return np.vstack(points), np.vstack(colors).astype(np.uint8), ids


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if colors is None:
        colors = np.full((len(points), 3), 230, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def export_keyframes(map_object, path: str | Path) -> list[dict]:
    keyframes = []
    for keyframe in _as_list(map_object.get_keyframes() if hasattr(map_object, "get_keyframes") else getattr(map_object, "keyframes", [])):
        if keyframe is None or _is_bad(keyframe):
            continue
        Tcw = _pose_matrix(keyframe, inverse=False)
        Twc = _pose_matrix(keyframe, inverse=True)
        keyframes.append(
            {
                "kid": int(getattr(keyframe, "kid", getattr(keyframe, "id", len(keyframes)))),
                "frame_id": int(getattr(keyframe, "id", -1)),
                "img_id": getattr(keyframe, "img_id", None),
                "timestamp": getattr(keyframe, "timestamp", None),
                "num_points": int(len([p for p in getattr(keyframe, "points", []) if p is not None])),
                "is_bad": bool(_is_bad(keyframe)),
                "position": [float(v) for v in Twc[:3, 3]],
                "Tcw": Tcw.tolist(),
                "Twc": Twc.tolist(),
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keyframes, indent=2, sort_keys=True) + "\n")
    return keyframes


def _edge_pair(a, b):
    ka = int(getattr(a, "kid", getattr(a, "id", -1)))
    kb = int(getattr(b, "kid", getattr(b, "id", -1)))
    if ka == kb or ka < 0 or kb < 0:
        return None
    return (min(ka, kb), max(ka, kb))


def export_keyframe_graph(map_object, path: str | Path) -> dict:
    keyframes = [
        kf
        for kf in _as_list(map_object.get_keyframes() if hasattr(map_object, "get_keyframes") else getattr(map_object, "keyframes", []))
        if kf is not None and not _is_bad(kf)
    ]
    nodes = [int(getattr(kf, "kid", getattr(kf, "id", -1))) for kf in keyframes]
    spanning = set()
    covisibility = {}
    loops = set()
    for keyframe in keyframes:
        parent = keyframe.get_parent() if hasattr(keyframe, "get_parent") else getattr(keyframe, "parent", None)
        pair = _edge_pair(keyframe, parent) if parent is not None else None
        if pair:
            spanning.add(pair)
        weights = getattr(keyframe, "connected_keyframes_weights", {})
        for other, weight in getattr(weights, "items", lambda: [])():
            if other is None or _is_bad(other):
                continue
            pair = _edge_pair(keyframe, other)
            if pair:
                covisibility[pair] = max(int(weight), covisibility.get(pair, 0))
        loop_edges = keyframe.get_loop_edges() if hasattr(keyframe, "get_loop_edges") else getattr(keyframe, "loop_edges", [])
        for other in loop_edges:
            if other is None or _is_bad(other):
                continue
            pair = _edge_pair(keyframe, other)
            if pair:
                loops.add(pair)
    graph = {
        "nodes": nodes,
        "spanning_tree_edges": [{"source": a, "target": b} for a, b in sorted(spanning)],
        "covisibility_edges": [
            {"source": a, "target": b, "weight": int(weight)}
            for (a, b), weight in sorted(covisibility.items())
        ],
        "loop_edges": [{"source": a, "target": b} for a, b in sorted(loops)],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
    return graph


def export_orbslam_map(map_or_slam, output_dir: str | Path) -> dict:
    map_object = getattr(map_or_slam, "map", map_or_slam)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    points, colors, point_ids = collect_exportable_points(map_object)
    ply_path = output_dir / "map_points.ply"
    keyframes_path = output_dir / "keyframes.json"
    graph_path = output_dir / "keyframe_graph.json"
    write_ply(ply_path, points, colors)
    keyframes = export_keyframes(map_object, keyframes_path)
    graph = export_keyframe_graph(map_object, graph_path)
    return {
        "map_points_ply": str(ply_path),
        "keyframes_json": str(keyframes_path),
        "keyframe_graph_json": str(graph_path),
        "num_exported_points": int(len(point_ids)),
        "num_exported_keyframes": int(len(keyframes)),
        "num_covisibility_edges": int(len(graph["covisibility_edges"])),
        "num_loop_edges": int(len(graph["loop_edges"])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.parse_args(argv)
    raise SystemExit("Use export_orbslam_map() from an evaluation runner with a live SLAM map.")


if __name__ == "__main__":
    raise SystemExit(main())
