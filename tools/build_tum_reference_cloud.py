#!/usr/bin/env python3
"""Build a GT-reference point cloud from TUM RGB-D depth and ground-truth poses."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class TumGtPose:
    timestamp: float
    translation: np.ndarray
    quaternion: np.ndarray
    Twc: np.ndarray


@dataclass(frozen=True)
class TumRgbDepthPair:
    timestamp: float
    rgb_path: Path
    depth_path: Path


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError(f"Invalid quaternion: {quaternion}")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_tum_groundtruth(path: str | Path) -> list[TumGtPose]:
    path = Path(path).expanduser()
    poses: list[TumGtPose] = []
    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"Invalid groundtruth line {line_number}: expected 8 columns")
            timestamp = float(parts[0])
            values = [float(v) for v in parts[1:8]]
            translation = np.asarray(values[:3], dtype=np.float64)
            quaternion = np.asarray(values[3:7], dtype=np.float64)
            Twc = np.eye(4, dtype=np.float64)
            Twc[:3, :3] = quaternion_to_rotation(quaternion)
            Twc[:3, 3] = translation
            poses.append(TumGtPose(timestamp, translation, quaternion, Twc))
    if not poses:
        raise ValueError(f"No valid TUM ground-truth poses found in {path}")
    return sorted(poses, key=lambda pose: pose.timestamp)


def read_tum_associations(dataset: str | Path) -> list[TumRgbDepthPair]:
    dataset = Path(dataset).expanduser().resolve()
    associations_path = dataset / "associations.txt"
    pairs: list[TumRgbDepthPair] = []
    if associations_path.exists():
        with open(associations_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    pairs.append(TumRgbDepthPair(float(parts[0]), dataset / parts[1], dataset / parts[3]))
                elif len(parts) >= 3:
                    pairs.append(TumRgbDepthPair(float(parts[0]), dataset / parts[1], dataset / parts[2]))
        return pairs

    from visual_slam.orbslam.io import load_tum_rgbd_associations

    return [
        TumRgbDepthPair(frame.timestamp, frame.rgb_path, frame.depth_path)
        for frame in load_tum_rgbd_associations(dataset)
    ]


def find_nearest_pose(poses: list[TumGtPose], timestamp: float, max_time_diff: float = 0.03) -> TumGtPose | None:
    times = np.asarray([pose.timestamp for pose in poses], dtype=np.float64)
    idx = int(np.searchsorted(times, float(timestamp)))
    candidates = []
    if idx < len(poses):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(poses[i].timestamp - timestamp))
    if abs(poses[best].timestamp - timestamp) > max_time_diff:
        return None
    return poses[best]


def backproject_depth(
    depth: np.ndarray,
    rgb: np.ndarray | None,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_factor: float,
    pixel_stride: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth)
    h, w = depth.shape[:2]
    ys = np.arange(0, h, int(pixel_stride), dtype=np.int32)
    xs = np.arange(0, w, int(pixel_stride), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    raw = depth[grid_y, grid_x].astype(np.float64)
    z = raw / float(depth_factor)
    valid = np.isfinite(z) & (z > 0.0) & (z <= float(max_depth))
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    u = grid_x[valid].astype(np.float64)
    v = grid_y[valid].astype(np.float64)
    z = z[valid]
    x = (u - float(cx)) * z / float(fx)
    y = (v - float(cy)) * z / float(fy)
    points = np.column_stack([x, y, z]).astype(np.float64)
    if rgb is None:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    else:
        bgr = rgb[grid_y[valid], grid_x[valid]]
        colors = bgr[:, ::-1].astype(np.uint8)
    return points, colors


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) == 0 or voxel_size <= 0.0:
        return points, colors
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    keep: dict[tuple[int, int, int], int] = {}
    for idx, key in enumerate(map(tuple, keys)):
        keep.setdefault(key, idx)
    indices = np.fromiter(keep.values(), dtype=np.int64)
    return points[indices], colors[indices]


def write_ascii_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if colors is None:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def build_reference_cloud(
    dataset: str | Path,
    output: str | Path,
    *,
    frame_stride: int = 15,
    pixel_stride: int = 8,
    max_depth: float = 4.5,
    voxel_size: float = 0.03,
    max_time_diff: float = 0.03,
    fx: float = 517.3,
    fy: float = 516.5,
    cx: float = 318.6,
    cy: float = 255.3,
    depth_factor: float = 5000.0,
) -> dict:
    dataset = Path(dataset).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    poses = read_tum_groundtruth(dataset / "groundtruth.txt")
    pairs = read_tum_associations(dataset)
    all_points = []
    all_colors = []
    frames_used = 0
    for pair in pairs[:: max(1, int(frame_stride))]:
        pose = find_nearest_pose(poses, pair.timestamp, max_time_diff=max_time_diff)
        if pose is None:
            continue
        depth = cv2.imread(str(pair.depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        rgb = cv2.imread(str(pair.rgb_path), cv2.IMREAD_COLOR)
        points_c, colors = backproject_depth(
            depth,
            rgb,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_factor=depth_factor,
            pixel_stride=pixel_stride,
            max_depth=max_depth,
        )
        if len(points_c) == 0:
            continue
        points_w = (pose.Twc[:3, :3] @ points_c.T + pose.Twc[:3, 3].reshape(3, 1)).T
        finite = np.all(np.isfinite(points_w), axis=1)
        all_points.append(points_w[finite])
        all_colors.append(colors[finite])
        frames_used += 1
    if all_points:
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
    else:
        points = np.empty((0, 3), dtype=np.float64)
        colors = np.empty((0, 3), dtype=np.uint8)
    points, colors = voxel_downsample(points, colors, voxel_size)
    write_ascii_ply(output, points, colors)
    return {
        "dataset": str(dataset),
        "output": str(output),
        "frames_total": len(pairs),
        "frames_used": frames_used,
        "points": int(len(points)),
        "frame_stride": int(frame_stride),
        "pixel_stride": int(pixel_stride),
        "max_depth": float(max_depth),
        "voxel_size": float(voxel_size),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--max-depth", type=float, default=4.5)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    args = parser.parse_args(argv)
    summary = build_reference_cloud(
        args.dataset,
        args.output,
        frame_stride=args.frame_stride,
        pixel_stride=args.pixel_stride,
        max_depth=args.max_depth,
        voxel_size=args.voxel_size,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
