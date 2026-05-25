#!/usr/bin/env python3
"""Evaluate TUM-format trajectories with ATE and simple consecutive-pose RPE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TumPose:
    timestamp: float
    translation: np.ndarray
    quaternion: np.ndarray
    matrix: np.ndarray


@dataclass(frozen=True)
class PoseAssociation:
    gt: TumPose
    estimate: TumPose
    time_diff: float


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError(f"Invalid quaternion: {quaternion}")
    x, y, z, w = q / norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def make_pose(timestamp: float, values: list[float]) -> TumPose:
    translation = np.asarray(values[:3], dtype=np.float64)
    quaternion = np.asarray(values[3:7], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_matrix(quaternion)
    matrix[:3, 3] = translation
    return TumPose(timestamp=float(timestamp), translation=translation, quaternion=quaternion, matrix=matrix)


def read_tum_poses(path: str | Path) -> list[TumPose]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"TUM trajectory file does not exist: {path}")

    poses: list[TumPose] = []
    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"Invalid TUM pose line {line_number} in {path}: expected 8 columns")

            try:
                timestamp = float(parts[0])
                values = [float(v) for v in parts[1:8]]
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value on line {line_number} in {path}") from exc

            poses.append(make_pose(timestamp, values))

    if not poses:
        raise ValueError(f"No valid TUM poses found in {path}")

    return sorted(poses, key=lambda pose: pose.timestamp)


def associate_poses(
    groundtruth: list[TumPose],
    estimates: list[TumPose],
    max_time_diff: float = 0.02,
) -> list[PoseAssociation]:
    if not groundtruth or not estimates:
        return []

    gt_times = np.asarray([pose.timestamp for pose in groundtruth], dtype=np.float64)
    used_gt: set[int] = set()
    associations: list[PoseAssociation] = []

    for estimate in sorted(estimates, key=lambda pose: pose.timestamp):
        insert_idx = int(np.searchsorted(gt_times, estimate.timestamp))
        candidates = []
        if insert_idx < len(gt_times):
            candidates.append(insert_idx)
        if insert_idx > 0:
            candidates.append(insert_idx - 1)

        best_idx = None
        best_diff = float("inf")
        for idx in candidates:
            if idx in used_gt:
                continue
            diff = abs(float(gt_times[idx] - estimate.timestamp))
            if diff < best_diff:
                best_idx = idx
                best_diff = diff

        if best_idx is not None and best_diff <= max_time_diff:
            used_gt.add(best_idx)
            associations.append(PoseAssociation(gt=groundtruth[best_idx], estimate=estimate, time_diff=best_diff))

    return associations


def _positions_from_associations(associations: list[PoseAssociation]) -> tuple[np.ndarray, np.ndarray]:
    gt = np.asarray([assoc.gt.translation for assoc in associations], dtype=np.float64)
    est = np.asarray([assoc.estimate.translation for assoc in associations], dtype=np.float64)
    return gt, est


def align_se3(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape Nx3")
    if len(source) < 3:
        raise ValueError("At least three associated poses are required for SE(3) alignment")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = source_centered.T @ target_centered / len(source)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    translation = target_mean - rotation @ source_mean
    return rotation, translation, 1.0


def align_sim3(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape Nx3")
    if len(source) < 3:
        raise ValueError("At least three associated poses are required for Sim(3) alignment")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_variance <= 0.0:
        raise ValueError("Cannot compute Sim(3) alignment for zero-variance source trajectory")

    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    translation = target_mean - scale * rotation @ source_mean
    return rotation, translation, scale


def transform_positions(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return scale * (np.asarray(points, dtype=np.float64) @ rotation.T) + translation


def transform_pose_matrix(matrix: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float = 1.0) -> np.ndarray:
    transformed = np.eye(4, dtype=np.float64)
    transformed[:3, :3] = rotation @ matrix[:3, :3]
    transformed[:3, 3] = scale * (rotation @ matrix[:3, 3]) + translation
    return transformed


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    trace_value = float(np.trace(rotation))
    cos_angle = max(-1.0, min(1.0, (trace_value - 1.0) * 0.5))
    return math.degrees(math.acos(cos_angle))


def compute_rpe(
    associations: list[PoseAssociation],
    se3_rotation: np.ndarray,
    se3_translation: np.ndarray,
) -> tuple[float, float, int]:
    if len(associations) < 2:
        return float("nan"), float("nan"), 0

    trans_errors: list[float] = []
    rot_errors: list[float] = []
    aligned_estimates = [
        transform_pose_matrix(assoc.estimate.matrix, se3_rotation, se3_translation, scale=1.0)
        for assoc in associations
    ]
    gt_matrices = [assoc.gt.matrix for assoc in associations]

    for idx in range(len(associations) - 1):
        gt_rel = np.linalg.inv(gt_matrices[idx]) @ gt_matrices[idx + 1]
        est_rel = np.linalg.inv(aligned_estimates[idx]) @ aligned_estimates[idx + 1]
        error = np.linalg.inv(gt_rel) @ est_rel
        trans_errors.append(float(np.linalg.norm(error[:3, 3])))
        rot_errors.append(rotation_angle_degrees(error[:3, :3]))

    return rmse(np.asarray(trans_errors)), rmse(np.asarray(rot_errors)), len(trans_errors)


def evaluate_trajectories(
    groundtruth_path: str | Path,
    trajectory_path: str | Path,
    output_dir: str | Path,
    max_time_diff: float = 0.02,
) -> dict:
    groundtruth = read_tum_poses(groundtruth_path)
    estimates = read_tum_poses(trajectory_path)
    associations = associate_poses(groundtruth, estimates, max_time_diff=max_time_diff)
    if len(associations) < 3:
        raise ValueError(
            f"Need at least 3 associated poses for trajectory evaluation, got {len(associations)}"
        )

    gt_positions, est_positions = _positions_from_associations(associations)

    se3_rotation, se3_translation, _ = align_se3(est_positions, gt_positions)
    se3_aligned = transform_positions(est_positions, se3_rotation, se3_translation)
    se3_errors = np.linalg.norm(se3_aligned - gt_positions, axis=1)

    sim3_rotation, sim3_translation, sim3_scale = align_sim3(est_positions, gt_positions)
    sim3_aligned = transform_positions(est_positions, sim3_rotation, sim3_translation, sim3_scale)
    sim3_errors = np.linalg.norm(sim3_aligned - gt_positions, axis=1)

    rpe_trans_rmse, rpe_rot_rmse_deg, rpe_pairs = compute_rpe(associations, se3_rotation, se3_translation)

    metrics = {
        "groundtruth": str(Path(groundtruth_path).expanduser()),
        "trajectory": str(Path(trajectory_path).expanduser()),
        "max_time_diff": float(max_time_diff),
        "num_groundtruth_poses": len(groundtruth),
        "num_estimated_poses": len(estimates),
        "num_associations": len(associations),
        "ate_rmse_se3_m": rmse(se3_errors),
        "ate_mean_se3_m": float(np.mean(se3_errors)),
        "ate_median_se3_m": float(np.median(se3_errors)),
        "ate_max_se3_m": float(np.max(se3_errors)),
        "ate_rmse_sim3_m": rmse(sim3_errors),
        "sim3_scale": float(sim3_scale),
        "rpe_trans_rmse_m": rpe_trans_rmse,
        "rpe_rot_rmse_deg": rpe_rot_rmse_deg,
        "rpe_pairs": rpe_pairs,
        "mean_time_diff_s": float(np.mean([assoc.time_diff for assoc in associations])),
        "max_association_time_diff_s": float(max(assoc.time_diff for assoc in associations)),
    }

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(output_dir / "trajectory_metrics.json", metrics)
    write_metrics_markdown(output_dir / "trajectory_metrics.md", metrics)
    write_associated_poses_csv(output_dir / "associated_poses.csv", associations, se3_aligned, sim3_aligned)
    return metrics


def write_metrics_json(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def write_metrics_markdown(path: Path, metrics: dict) -> None:
    lines = [
        "# TUM Trajectory Metrics",
        "",
        f"Ground truth: `{metrics['groundtruth']}`",
        f"Trajectory: `{metrics['trajectory']}`",
        f"Max time diff: `{metrics['max_time_diff']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Ground-truth poses | {metrics['num_groundtruth_poses']} |",
        f"| Estimated poses | {metrics['num_estimated_poses']} |",
        f"| Associated poses | {metrics['num_associations']} |",
        f"| ATE RMSE SE(3) m | {metrics['ate_rmse_se3_m']:.9f} |",
        f"| ATE RMSE Sim(3) m | {metrics['ate_rmse_sim3_m']:.9f} |",
        f"| Sim(3) scale | {metrics['sim3_scale']:.9f} |",
        f"| RPE translational RMSE m | {metrics['rpe_trans_rmse_m']:.9f} |",
        f"| RPE rotational RMSE deg | {metrics['rpe_rot_rmse_deg']:.9f} |",
        f"| RPE pairs | {metrics['rpe_pairs']} |",
        f"| Mean association diff s | {metrics['mean_time_diff_s']:.9f} |",
        f"| Max association diff s | {metrics['max_association_time_diff_s']:.9f} |",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_associated_poses_csv(
    path: Path,
    associations: list[PoseAssociation],
    se3_aligned_positions: np.ndarray,
    sim3_aligned_positions: np.ndarray,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp_est",
                "timestamp_gt",
                "time_diff",
                "gt_tx",
                "gt_ty",
                "gt_tz",
                "est_tx",
                "est_ty",
                "est_tz",
                "est_se3_tx",
                "est_se3_ty",
                "est_se3_tz",
                "est_sim3_tx",
                "est_sim3_ty",
                "est_sim3_tz",
            ]
        )
        for assoc, se3_pos, sim3_pos in zip(associations, se3_aligned_positions, sim3_aligned_positions):
            writer.writerow(
                [
                    f"{assoc.estimate.timestamp:.9f}",
                    f"{assoc.gt.timestamp:.9f}",
                    f"{assoc.time_diff:.9f}",
                    *[f"{v:.9f}" for v in assoc.gt.translation],
                    *[f"{v:.9f}" for v in assoc.estimate.translation],
                    *[f"{v:.9f}" for v in se3_pos],
                    *[f"{v:.9f}" for v in sim3_pos],
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a TUM-format trajectory against ground truth.")
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-time-diff", type=float, default=0.02)
    args = parser.parse_args(argv)

    try:
        metrics = evaluate_trajectories(
            groundtruth_path=args.groundtruth,
            trajectory_path=args.trajectory,
            output_dir=args.output,
            max_time_diff=args.max_time_diff,
        )
    except Exception as exc:
        print(f"trajectory evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
