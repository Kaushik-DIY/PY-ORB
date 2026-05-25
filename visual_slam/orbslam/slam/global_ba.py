"""
Global bundle-adjustment coordinator.
This module runs a full-map optimization pass and applies the validated updates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import g2o

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.optimizer_g2o import global_bundle_adjustment


# Store the summary of one global bundle-adjustment run.
@dataclass
class GlobalBAResult:
    started: bool = False
    success: bool = False
    num_keyframes: int = 0
    num_map_points: int = 0
    num_edges: int = 0
    num_inliers: int = 0
    num_outliers: int = 0
    mean_error_before: float | None = None
    mean_error_after: float | None = None
    elapsed_sec: float = 0.0
    aborted: bool = False
    reason: str = ""
    fixed_keyframe_ids: tuple[int, ...] = ()

    def to_diagnostics(self) -> dict:
        return asdict(self)


# Prepare, execute, and validate a full-map bundle-adjustment pass.
class GlobalBundleAdjuster:
    def __init__(
        self,
        map_object,
        rounds: int = 10,
        use_robust_kernel: bool = True,
        min_point_observations: int = 2,
        min_inlier_edges: int = 10,
        max_translation_jump: float = 50.0,
    ):
        self.map = map_object
        self.rounds = int(rounds)
        self.use_robust_kernel = bool(use_robust_kernel)
        self.min_point_observations = int(min_point_observations)
        self.min_inlier_edges = int(min_inlier_edges)
        self.max_translation_jump = float(max_translation_jump)
        self.last_result = GlobalBAResult()

    def run(self, loop_kf_id: int = 0, stop_flag=None, verbose: bool = False) -> GlobalBAResult:
        start_t = time.time()
        result = GlobalBAResult(started=True)

        if _abort_requested(stop_flag):
            result.aborted = True
            result.reason = "aborted before graph construction"
            result.elapsed_sec = time.time() - start_t
            self.last_result = result
            return result

        keyframes, points = self.collect_graph()
        result.num_keyframes = len(keyframes)
        result.num_map_points = len(points)

        if len(keyframes) == 0:
            result.reason = "no valid keyframes"
            result.elapsed_sec = time.time() - start_t
            self.last_result = result
            return result
        if len(points) == 0:
            result.reason = "no valid map points"
            result.elapsed_sec = time.time() - start_t
            self.last_result = result
            return result

        old_poses = {kf: kf.Tcw().copy() for kf in keyframes}
        old_points = {point: point.get_position().copy() for point in points}
        opt_abort_flag = stop_flag
        updates: dict = {}

        try:
            _, updates = global_bundle_adjustment(
                keyframes=keyframes,
                points=points,
                rounds=self.rounds,
                loop_kf_id=loop_kf_id,
                use_robust_kernel=self.use_robust_kernel,
                abort_flag=opt_abort_flag,
                result_dict=updates,
                write_back=False,
                verbose=verbose,
            )
        except Exception as exc:
            result.reason = f"g2o failed: {exc}"
            result.elapsed_sec = time.time() - start_t
            self.last_result = result
            return result

        result.num_edges = int(updates.get("num_edges", 0))
        result.num_inliers = int(updates.get("num_inliers", 0))
        result.num_outliers = int(updates.get("num_outliers", 0))
        result.mean_error_before = _finite_or_none(updates.get("mean_error_before"))
        result.mean_error_after = _finite_or_none(updates.get("mean_error_after"))
        result.fixed_keyframe_ids = tuple(int(kid) for kid in updates.get("fixed_keyframes", ()))

        total_edges = result.num_inliers + result.num_outliers
        if total_edges > 0:
            outlier_rate = result.num_outliers / total_edges
            if outlier_rate > 0.30:
                result.reason = (
                    f"GBA outlier rate {outlier_rate:.1%} exceeds 30% — "
                    f"map likely corrupted by a false loop; aborting to prevent further damage"
                )
                result.elapsed_sec = time.time() - start_t
                self.last_result = result
                return result

        if _abort_requested(opt_abort_flag):
            result.aborted = True
            result.reason = "aborted during optimization"
            result.elapsed_sec = time.time() - start_t
            self.last_result = result
            return result

        pose_updates = _updates_by_keyframe(keyframes, updates.get("keyframes", {}))
        point_updates = _updates_by_point(points, updates.get("points", {}))

        ok, reason = self.validate_updates(
            old_poses=old_poses,
            old_points=old_points,
            pose_updates=pose_updates,
            point_updates=point_updates,
            num_inliers=result.num_inliers,
        )
        if not ok:
            result.reason = reason
            result.elapsed_sec = time.time() - start_t
            self.last_result = result
            return result

        try:
            self.apply_updates_atomically(old_poses, old_points, pose_updates, point_updates)
        except Exception as exc:
            _restore_state(old_poses, old_points)
            result.reason = f"write-back failed: {exc}"
            result.elapsed_sec = time.time() - start_t
            self.last_result = result
            return result

        result.success = True
        result.reason = "ok"
        result.elapsed_sec = time.time() - start_t
        self.last_result = result
        return result

    def collect_graph(self) -> tuple[list, list]:
        keyframes = []
        for keyframe in _as_list(self.map.get_keyframes()):
            if keyframe is None or (hasattr(keyframe, "is_bad") and keyframe.is_bad()):
                continue
            keyframes.append(keyframe)

        keyframe_set = set(keyframes)
        points = []
        seen = set()
        for point in _as_list(self.map.get_points()):
            if point is None or (hasattr(point, "is_bad") and point.is_bad()):
                continue
            if point in seen or point.num_observations() < self.min_point_observations:
                continue
            point_w = point.get_position()
            if not np.all(np.isfinite(point_w)):
                continue
            observations = [
                (kf, idx)
                for kf, idx in point.observations()
                if kf in keyframe_set and kf is not None and not kf.is_bad()
            ]
            if len(observations) == 0:
                continue
            if not any(_kf_has_positive_depth(kf, point_w) for kf, _ in observations):
                continue
            points.append(point)
            seen.add(point)
        return keyframes, points

    def validate_updates(
        self,
        old_poses: dict,
        old_points: dict,
        pose_updates: dict,
        point_updates: dict,
        num_inliers: int,
    ) -> tuple[bool, str]:
        if num_inliers < self.min_inlier_edges:
            return False, "not enough inlier edges"
        if not pose_updates:
            return False, "no optimized poses"
        if not point_updates:
            return False, "no optimized points"

        for keyframe, Tcw in pose_updates.items():
            Tcw = np.asarray(Tcw, dtype=np.float64)
            if not _is_valid_se3(Tcw):
                return False, f"invalid optimized pose for keyframe {getattr(keyframe, 'kid', None)}"
            old = old_poses.get(keyframe)
            if old is not None:
                jump = float(np.linalg.norm(Tcw[:3, 3] - old[:3, 3]))
                if not np.isfinite(jump) or jump > self.max_translation_jump:
                    return False, "optimized translation jump is unreasonable"

        for point, position in point_updates.items():
            position = np.asarray(position, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(position)):
                return False, f"invalid optimized map point {getattr(point, 'id', None)}"
            if not _positive_depth_in_any_observer(point, position, pose_updates):
                return False, f"optimized map point {getattr(point, 'id', None)} has no positive-depth observer"

        return True, ""

    def apply_updates_atomically(
        self,
        old_poses: dict,
        old_points: dict,
        pose_updates: dict,
        point_updates: dict,
    ) -> None:
        lock = getattr(self.map, "update_lock", None)
        context = lock if lock is not None else _NullLock()
        with context:
            try:
                for keyframe, Tcw in pose_updates.items():
                    keyframe.Tcw_before_GBA = old_poses.get(keyframe)
                    keyframe.Tcw_GBA = np.asarray(Tcw, dtype=np.float64).copy()
                    keyframe.is_Tcw_GBA_valid = True
                    keyframe.update_pose(g2o.Isometry3d(Tcw))

                for point, position in point_updates.items():
                    point.pt_GBA = np.asarray(position, dtype=np.float64).copy()
                    point.is_pt_GBA_valid = True
                    point.update_position(position)
                    point.update_info()

                for keyframe in pose_updates:
                    keyframe.update_connections()
            except Exception:
                _restore_state(old_poses, old_points)
                raise


def _updates_by_keyframe(keyframes: list, updates: dict) -> dict:
    by_kid = {int(getattr(kf, "kid", -1)): kf for kf in keyframes}
    out = {}
    for key, value in updates.items():
        keyframe = by_kid.get(int(key))
        if keyframe is not None:
            out[keyframe] = np.asarray(value, dtype=np.float64).reshape(4, 4)
    return out


def _updates_by_point(points: list, updates: dict) -> dict:
    by_id = {int(getattr(point, "id", -1)): point for point in points}
    out = {}
    for key, value in updates.items():
        point = by_id.get(int(key))
        if point is not None:
            out[point] = np.asarray(value, dtype=np.float64).reshape(3)
    return out


def _positive_depth_in_any_observer(point, position: np.ndarray, pose_updates: dict) -> bool:
    for keyframe, _ in point.observations():
        if keyframe is None or (hasattr(keyframe, "is_bad") and keyframe.is_bad()):
            continue
        Tcw = pose_updates.get(keyframe, keyframe.Tcw())
        if not _is_valid_se3(Tcw):
            continue
        p_c = Tcw[:3, :3] @ position.reshape(3) + Tcw[:3, 3]
        if np.isfinite(p_c[2]) and p_c[2] > Parameters.kMinDepth:
            return True
    return False


def _restore_state(old_poses: dict, old_points: dict) -> None:
    for keyframe, Tcw in old_poses.items():
        keyframe.update_pose(g2o.Isometry3d(Tcw))
    for point, position in old_points.items():
        point.update_position(position)


def _is_valid_se3(T: np.ndarray) -> bool:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4):
        return False
    det = float(np.linalg.det(R))
    return bool(np.isfinite(det) and abs(det - 1.0) <= 1e-3)


def _abort_requested(stop_flag) -> bool:
    if stop_flag is None:
        return False
    value = getattr(stop_flag, "value", stop_flag)
    if callable(value):
        try:
            value = value()
        except TypeError:
            pass
    return bool(value)


def _kf_has_positive_depth(kf, point_w: np.ndarray) -> bool:
    try:
        Tcw = np.asarray(kf.Tcw(), dtype=np.float64).reshape(4, 4)
        p_c = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
        return bool(np.isfinite(p_c[2]) and p_c[2] > Parameters.kMinDepth)
    except Exception:
        return False


def _finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _as_list(values) -> list:
    if values is None:
        return []
    if hasattr(values, "to_list"):
        return values.to_list()
    return list(values)


# Provide a no-op lock interface for code paths that do not need synchronization.
class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
