"""
Ground-truth loop oracle helpers for TUM RGB-D diagnostics.
This module associates keyframe timestamps to TUM poses and labels loop-like pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import numpy as np

from tools.evaluate_tum_trajectory import TumPose, read_tum_poses, rotation_angle_degrees


@dataclass(frozen=True)
class LoopOraclePairDiagnostics:
    gt_available: bool
    gt_translation_distance: Optional[float]
    gt_rotation_angle_deg: Optional[float]
    gt_loop_like: bool
    gt_near_loop: bool


class TumLoopOracle:
    """Associate timestamps to TUM ground-truth poses for loop diagnostics."""

    def __init__(
        self,
        poses: list[TumPose],
        *,
        max_time_diff: float,
        loop_translation_threshold_m: float = 0.75,
        loop_rotation_threshold_deg: float = 45.0,
        near_loop_translation_threshold_m: float = 1.5,
    ):
        self.poses = list(sorted(poses, key=lambda pose: float(pose.timestamp)))
        self.timestamps = np.asarray([float(pose.timestamp) for pose in self.poses], dtype=np.float64)
        self.max_time_diff = float(max_time_diff)
        self.loop_translation_threshold_m = float(loop_translation_threshold_m)
        self.loop_rotation_threshold_deg = float(loop_rotation_threshold_deg)
        self.near_loop_translation_threshold_m = float(near_loop_translation_threshold_m)

    @classmethod
    def from_tum_groundtruth(
        cls,
        path: str | Path,
        *,
        max_time_diff: float,
        loop_translation_threshold_m: float = 0.75,
        loop_rotation_threshold_deg: float = 45.0,
        near_loop_translation_threshold_m: float = 1.5,
    ) -> "TumLoopOracle":
        poses = read_tum_poses(path)
        return cls(
            poses,
            max_time_diff=max_time_diff,
            loop_translation_threshold_m=loop_translation_threshold_m,
            loop_rotation_threshold_deg=loop_rotation_threshold_deg,
            near_loop_translation_threshold_m=near_loop_translation_threshold_m,
        )

    def has_data(self) -> bool:
        return len(self.poses) > 0

    def find_pose(self, timestamp: float) -> TumPose | None:
        if not self.poses or not np.isfinite(float(timestamp)):
            return None
        idx = int(np.searchsorted(self.timestamps, float(timestamp)))
        candidates = []
        if idx < len(self.timestamps):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            return None
        best_idx = min(candidates, key=lambda i: abs(float(self.timestamps[i] - timestamp)))
        best_diff = abs(float(self.timestamps[best_idx] - timestamp))
        if best_diff > self.max_time_diff:
            return None
        return self.poses[best_idx]

    def describe_pair(
        self,
        current_timestamp: float,
        candidate_timestamp: float,
    ) -> LoopOraclePairDiagnostics:
        current_pose = self.find_pose(current_timestamp)
        candidate_pose = self.find_pose(candidate_timestamp)
        if current_pose is None or candidate_pose is None:
            return LoopOraclePairDiagnostics(False, None, None, False, False)

        current_translation = np.asarray(current_pose.translation, dtype=np.float64).reshape(3)
        candidate_translation = np.asarray(candidate_pose.translation, dtype=np.float64).reshape(3)
        translation_distance = float(np.linalg.norm(current_translation - candidate_translation))

        current_rotation = np.asarray(current_pose.matrix[:3, :3], dtype=np.float64).reshape(3, 3)
        candidate_rotation = np.asarray(candidate_pose.matrix[:3, :3], dtype=np.float64).reshape(3, 3)
        relative_rotation = current_rotation.T @ candidate_rotation
        rotation_angle_deg = float(rotation_angle_degrees(relative_rotation))
        if not math.isfinite(rotation_angle_deg):
            rotation_angle_deg = None

        gt_loop_like = (
            translation_distance <= self.loop_translation_threshold_m
            and rotation_angle_deg is not None
            and rotation_angle_deg <= self.loop_rotation_threshold_deg
        )
        gt_near_loop = translation_distance <= self.near_loop_translation_threshold_m
        return LoopOraclePairDiagnostics(
            True,
            translation_distance,
            rotation_angle_deg,
            bool(gt_loop_like),
            bool(gt_near_loop),
        )
