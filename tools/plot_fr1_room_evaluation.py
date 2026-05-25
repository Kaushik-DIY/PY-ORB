#!/usr/bin/env python3
"""Generate fr1_room trajectory, map, graph, and metrics figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.evaluate_tum_trajectory import align_se3, read_tum_poses, transform_positions


EXPECTED_FIGURES = [
    "trajectory_xy_comparison.png",
    "trajectory_3d_comparison.png",
    "trajectory_loop_events.png",
    "estimated_sparse_map_xy.png",
    "estimated_sparse_map_3d.png",
    "reference_cloud_xy.png",
    "map_side_by_side_xy.png",
    "keyframe_graph_xy.png",
    "metrics_table.png",
    "metrics_table.md",
]

ALIGNED_FIGURES = [
    "map_side_by_side_xy_raw.png",
    "map_side_by_side_xy_aligned.png",
    "trajectory_loop_events_aligned.png",
    "estimated_sparse_map_xy_aligned.png",
    "keyframe_graph_xy_aligned.png",
]


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _discover_run_dir(root: Path, run_name: str) -> Path:
    run_dir = root / run_name
    if (run_dir / "run_summary.json").exists():
        return run_dir
    dry = sorted(run_dir.glob("dry_run_*"))
    for candidate in reversed(dry):
        if (candidate / "run_summary.json").exists():
            return candidate
    return run_dir


def _trajectory_file(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("trajectory_*.txt"))
    return candidates[0] if candidates else None


def _load_tum_positions(path: Path | None) -> np.ndarray:
    if path is None or not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    try:
        return np.asarray([pose.translation for pose in read_tum_poses(path)], dtype=np.float64)
    except Exception:
        return np.empty((0, 3), dtype=np.float64)


def _load_aligned_positions(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = run_dir / "trajectory_eval" / "associated_poses.csv"
    if not csv_path.exists():
        return np.empty((0, 3)), np.empty((0, 3))
    gt = []
    est = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt.append([float(row["gt_tx"]), float(row["gt_ty"]), float(row["gt_tz"])])
            est.append([float(row["est_se3_tx"]), float(row["est_se3_ty"]), float(row["est_se3_tz"])])
    return np.asarray(gt, dtype=np.float64), np.asarray(est, dtype=np.float64)


def _load_alignment_transform(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    csv_path = run_dir / "trajectory_eval" / "associated_poses.csv"
    if not csv_path.exists():
        return None
    gt = []
    est = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gt.append([float(row["gt_tx"]), float(row["gt_ty"]), float(row["gt_tz"])])
                est.append([float(row["est_tx"]), float(row["est_ty"]), float(row["est_tz"])])
            except Exception:
                continue
    if len(gt) < 3 or len(est) < 3:
        return None
    rotation, translation, _ = align_se3(np.asarray(est, dtype=np.float64), np.asarray(gt, dtype=np.float64))
    return rotation, translation


def apply_alignment(points: np.ndarray, transform: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if transform is None or len(points) == 0:
        return points.copy()
    rotation, translation = transform
    return transform_positions(points, rotation, translation, scale=1.0)


def read_ply_points(path: str | Path, max_points: int | None = 250000) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    points = []
    in_header = True
    with open(path, "r") as f:
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if max_points is not None and len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, int(max_points), dtype=np.int64)
        pts = pts[idx]
    return pts


def _set_equal_xy(ax, arrays: list[np.ndarray], *, robust: bool = False) -> None:
    finite = [arr[:, :2] for arr in arrays if arr is not None and len(arr) > 0]
    if not finite:
        return
    pts = np.vstack(finite)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return
    if robust and len(pts) >= 20:
        mins = np.percentile(pts, 1.0, axis=0)
        maxs = np.percentile(pts, 99.0, axis=0)
    else:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
    center = (mins + maxs) * 0.5
    span = max(float(np.max(maxs - mins)), 1e-6)
    pad = span * 0.08
    ax.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
    ax.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
    ax.set_aspect("equal", adjustable="box")


def _plot_xy(ax, pts: np.ndarray, label: str, *, color=None, lw=1.8, alpha=1.0, scatter=False, s=2):
    if len(pts) == 0:
        return
    if scatter:
        ax.scatter(pts[:, 0], pts[:, 1], s=s, label=label, color=color, alpha=alpha, linewidths=0)
    else:
        ax.plot(pts[:, 0], pts[:, 1], label=label, color=color, lw=lw, alpha=alpha)


def _load_keyframes(run_dir: Path) -> dict[int, np.ndarray]:
    data = _load_json(run_dir / "keyframes.json", [])
    out = {}
    for row in data:
        try:
            out[int(row["kid"])] = np.asarray(row["position"], dtype=np.float64).reshape(3)
        except Exception:
            continue
    return out


def _load_event_kf_ids(path: Path, column: str) -> list[int]:
    if not path.exists():
        return []
    ids = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ids.append(int(row[column]))
            except Exception:
                continue
    return ids


def write_metrics_table(output_dir: Path, run_dirs: dict[str, Path]) -> list[dict]:
    rows = []
    for label, run_dir in run_dirs.items():
        summary = _load_json(run_dir / "run_summary.json", {})
        metrics = _load_json(run_dir / "trajectory_eval" / "trajectory_metrics.json", {})
        rows.append(
            {
                "run": label,
                "frames": summary.get("frames_attempted", ""),
                "ok": summary.get("tracking_ok_count", ""),
                "lost": summary.get("tracking_lost_count", ""),
                "loops": summary.get("accepted_loops", ""),
                "gba": summary.get("global_ba_success", ""),
                "ate_se3": metrics.get("ate_rmse_se3_m", ""),
                "rpe_trans": metrics.get("rpe_trans_rmse_m", ""),
                "rpe_rot": metrics.get("rpe_rot_rmse_deg", ""),
            }
        )
    md = [
        "| Run | Frames | OK | Lost | Accepted loops | GBA success | ATE RMSE SE(3) m | RPE trans RMSE m | RPE rot RMSE deg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            "| {run} | {frames} | {ok} | {lost} | {loops} | {gba} | {ate_se3} | {rpe_trans} | {rpe_rot} |".format(
                **{k: _fmt_metric(v) for k, v in row.items()}
            )
        )
    (output_dir / "metrics_table.md").write_text("\n".join(md) + "\n")
    return rows


def _fmt_metric(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else str(value)


def generate_plots(root: str | Path, dataset: str | Path | None = None, output: str | Path | None = None) -> list[Path]:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output).expanduser().resolve() if output else root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {
        "A no loop": _discover_run_dir(root, "run_A_no_loop"),
        "B loop only": _discover_run_dir(root, "run_B_loop_only"),
        "C loop+GBA": _discover_run_dir(root, "run_C_loop_plus_gba"),
    }
    gt_path = Path(dataset).expanduser() / "groundtruth.txt" if dataset else None
    gt_raw = _load_tum_positions(gt_path)
    aligned = {label: _load_aligned_positions(run_dir) for label, run_dir in run_dirs.items()}
    raw_est = {label: _load_tum_positions(_trajectory_file(run_dir)) for label, run_dir in run_dirs.items()}
    c_dir = run_dirs["C loop+GBA"]
    est_map = read_ply_points(c_dir / "map_points.ply")
    reference = read_ply_points(root / "reference_map" / "reference_cloud_gt.ply")
    keyframes = _load_keyframes(c_dir)
    graph = _load_json(c_dir / "keyframe_graph.json", {})
    alignment = _load_alignment_transform(c_dir)
    est_map_aligned = apply_alignment(est_map, alignment)
    keyframes_aligned = {
        kid: apply_alignment(position.reshape(1, 3), alignment).reshape(3)
        for kid, position in keyframes.items()
    }

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    if len(gt_raw):
        _plot_xy(ax, gt_raw, "Ground truth", color="black", lw=2.0)
    for label, (gt, est) in aligned.items():
        pts = est if len(est) else raw_est[label]
        _plot_xy(ax, pts, label)
    _set_equal_xy(ax, [gt_raw, *[v[1] if len(v[1]) else raw_est[k] for k, v in aligned.items()]])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_xy_comparison.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax3 = fig.add_subplot(111, projection="3d")
    if len(gt_raw):
        ax3.plot(gt_raw[:, 0], gt_raw[:, 1], gt_raw[:, 2], label="Ground truth", color="black")
    for label, (gt, est) in aligned.items():
        pts = est if len(est) else raw_est[label]
        if len(pts):
            ax3.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=label)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_zlabel("z [m]")
    ax3.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_3d_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    c_pts = aligned["C loop+GBA"][1] if len(aligned["C loop+GBA"][1]) else raw_est["C loop+GBA"]
    _plot_xy(ax, c_pts, "C trajectory", color="#2468b2")
    kf_positions = keyframes_aligned if alignment is not None else keyframes
    loop_ids = _load_event_kf_ids(c_dir / "loop_events.csv", "current_kf_id")
    gba_ids = _load_event_kf_ids(c_dir / "global_ba_events.csv", "trigger_kf_id")
    loop_pts = np.asarray([kf_positions[kid] for kid in loop_ids if kid in kf_positions], dtype=np.float64).reshape(-1, 3)
    gba_pts = np.asarray([kf_positions[kid] for kid in gba_ids if kid in kf_positions], dtype=np.float64).reshape(-1, 3)
    _plot_xy(ax, loop_pts, "Loop events", color="#d04a3a", scatter=True, s=28)
    _plot_xy(ax, gba_pts, "Global BA events", color="#1f9d55", scatter=True, s=34)
    _set_equal_xy(ax, [c_pts, loop_pts, gba_pts])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_loop_events.png")
    fig.savefig(output_dir / "trajectory_loop_events_aligned.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    _plot_xy(ax, est_map, "Estimated sparse map", color="#273746", scatter=True, s=1, alpha=0.65)
    _set_equal_xy(ax, [est_map])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "estimated_sparse_map_xy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    _plot_xy(ax, est_map_aligned, "Estimated sparse map", color="#273746", scatter=True, s=1, alpha=0.65)
    _set_equal_xy(ax, [est_map_aligned], robust=True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "estimated_sparse_map_xy_aligned.png")
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6), dpi=150)
    ax3 = fig.add_subplot(111, projection="3d")
    if len(est_map):
        ax3.scatter(est_map[:, 0], est_map[:, 1], est_map[:, 2], s=1, alpha=0.45)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_zlabel("z [m]")
    fig.tight_layout()
    fig.savefig(output_dir / "estimated_sparse_map_3d.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    _plot_xy(ax, reference, "GT-reference cloud", color="#586f7c", scatter=True, s=0.5, alpha=0.45)
    _set_equal_xy(ax, [reference])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "reference_cloud_xy.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
    _plot_xy(axes[0], est_map, "Estimated sparse", color="#273746", scatter=True, s=1, alpha=0.65)
    _plot_xy(axes[1], reference, "GT-reference", color="#586f7c", scatter=True, s=0.5, alpha=0.45)
    for ax in axes:
        _set_equal_xy(ax, [est_map, reference])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.2)
    axes[0].set_title("Estimated sparse map")
    axes[1].set_title("GT-reference cloud")
    fig.tight_layout()
    fig.savefig(output_dir / "map_side_by_side_xy_raw.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
    _plot_xy(axes[0], est_map_aligned, "Estimated sparse", color="#273746", scatter=True, s=1, alpha=0.65)
    _plot_xy(axes[1], reference, "GT-reference", color="#586f7c", scatter=True, s=0.5, alpha=0.45)
    for ax in axes:
        _set_equal_xy(ax, [est_map_aligned, reference], robust=True)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.2)
    axes[0].set_title("Estimated sparse map")
    axes[1].set_title("GT-reference cloud")
    fig.tight_layout()
    fig.savefig(output_dir / "map_side_by_side_xy_aligned.png")
    fig.savefig(output_dir / "map_side_by_side_xy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    kpts = np.asarray(list(keyframes.values()), dtype=np.float64).reshape(-1, 3)
    _plot_xy(ax, kpts, "Keyframes", color="#1b4f72", scatter=True, s=14)
    for edge_group, color, lw in [
        ("spanning_tree_edges", "#7f8c8d", 0.8),
        ("covisibility_edges", "#b7950b", 0.45),
        ("loop_edges", "#c0392b", 1.6),
    ]:
        for edge in graph.get(edge_group, []):
            a = keyframes.get(int(edge.get("source", -1)))
            b = keyframes.get(int(edge.get("target", -1)))
            if a is not None and b is not None:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, alpha=0.65)
    _set_equal_xy(ax, [kpts])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "keyframe_graph_xy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    kpts_aligned = np.asarray(list(keyframes_aligned.values()), dtype=np.float64).reshape(-1, 3)
    _plot_xy(ax, kpts_aligned, "Keyframes", color="#1b4f72", scatter=True, s=14)
    for edge_group, color, lw in [
        ("spanning_tree_edges", "#7f8c8d", 0.8),
        ("covisibility_edges", "#b7950b", 0.45),
        ("loop_edges", "#c0392b", 1.6),
    ]:
        for edge in graph.get(edge_group, []):
            a = keyframes_aligned.get(int(edge.get("source", -1)))
            b = keyframes_aligned.get(int(edge.get("target", -1)))
            if a is not None and b is not None:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, alpha=0.65)
    _set_equal_xy(ax, [kpts_aligned])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "keyframe_graph_xy_aligned.png")
    plt.close(fig)

    rows = write_metrics_table(output_dir, run_dirs)
    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.45 * (len(rows) + 2))), dpi=150)
    ax.axis("off")
    columns = ["run", "frames", "ok", "lost", "loops", "gba", "ate_se3", "rpe_trans", "rpe_rot"]
    table = ax.table(
        cellText=[[_fmt_metric(row[col]) for col in columns] for row in rows],
        colLabels=["Run", "Frames", "OK", "Lost", "Loops", "GBA", "ATE", "RPE t", "RPE r"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_table.png")
    plt.close(fig)

    return [output_dir / name for name in EXPECTED_FIGURES]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    paths = generate_plots(args.root, dataset=args.dataset, output=args.output)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
