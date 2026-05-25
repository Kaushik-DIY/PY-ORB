"""
Scale-fixed rigid alignment for loop verification.
This module estimates the relative transform between matched 3D point sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import g2o
import numpy as np


# Store the rigid alignment estimate produced during loop verification.
@dataclass
class Sim3Estimate:
    success: bool
    R: np.ndarray
    t: np.ndarray
    scale: float
    inlier_mask: np.ndarray
    mean_error: float
    sim3: Optional[object] = None
    error: Optional[str] = None

    @property
    def T(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.scale * self.R
        T[:3, 3] = self.t
        return T


def estimate_scale_fixed_sim3(
    points_current: np.ndarray,
    points_loop: np.ndarray,
    max_error: float = 0.08,
    ransac_iterations: int = 200,
    random_seed: int = 226,
) -> Sim3Estimate:
    """Estimate loop-point transform points_loop ~= R * points_current + t."""
    points_current = np.asarray(points_current, dtype=np.float64).reshape(-1, 3)
    points_loop = np.asarray(points_loop, dtype=np.float64).reshape(-1, 3)

    n = min(len(points_current), len(points_loop))
    if n < 3:
        return Sim3Estimate(
            False,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            1.0,
            np.zeros(n, dtype=bool),
            float("inf"),
            error="too few 3D correspondences",
        )

    points_current = points_current[:n]
    points_loop = points_loop[:n]

    finite = np.all(np.isfinite(points_current), axis=1) & np.all(np.isfinite(points_loop), axis=1)
    if np.sum(finite) < 3:
        return Sim3Estimate(
            False,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            1.0,
            finite,
            float("inf"),
            error="non-finite 3D correspondences",
        )

    R, t, inliers, errors = _estimate_ransac(
        points_current,
        points_loop,
        finite,
        max_error=float(max_error),
        ransac_iterations=int(ransac_iterations),
        random_seed=int(random_seed),
    )

    if np.sum(inliers) < 3:
        return Sim3Estimate(
            False,
            R,
            t,
            1.0,
            inliers,
            float(np.mean(errors[finite])) if np.any(finite) else float("inf"),
            error="not enough scale-fixed Sim3 inliers",
        )

    mean_error = float(np.mean(errors[inliers]))
    if not np.isfinite(mean_error) or not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
        return Sim3Estimate(False, R, t, 1.0, inliers, mean_error, error="non-finite Sim3 estimate")

    sim3 = None
    try:
        sim3 = g2o.Sim3(R, t, 1.0)
    except Exception:
        sim3 = None

    return Sim3Estimate(True, R, t, 1.0, inliers, mean_error, sim3=sim3)


def _estimate_ransac(
    source: np.ndarray,
    target: np.ndarray,
    finite: np.ndarray,
    *,
    max_error: float,
    ransac_iterations: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite_idxs = np.flatnonzero(finite)
    best_R = np.eye(3, dtype=np.float64)
    best_t = np.zeros(3, dtype=np.float64)
    best_inliers = np.zeros(len(source), dtype=bool)
    best_errors = np.full(len(source), np.inf, dtype=np.float64)

    if len(finite_idxs) < 3:
        return best_R, best_t, best_inliers, best_errors

    def score_model(R, t):
        transformed = (R @ source.T).T + t.reshape(1, 3)
        errors = np.linalg.norm(transformed - target, axis=1)
        inliers = finite & np.isfinite(errors) & (errors <= max_error)
        return inliers, errors

    # The all-correspondence model is useful for synthetic clean tests and as a
    # deterministic fallback when the match set is already mostly inlier.
    try:
        R_all, t_all = _kabsch(source[finite], target[finite])
        inliers_all, errors_all = score_model(R_all, t_all)
        best_R, best_t, best_inliers, best_errors = R_all, t_all, inliers_all, errors_all
    except Exception:
        pass

    rng = np.random.default_rng(random_seed)
    sample_count = min(max(0, ransac_iterations), _num_unique_triplets(len(finite_idxs)))
    seen_samples: set[tuple[int, int, int]] = set()

    for _ in range(sample_count):
        sample = tuple(sorted(int(i) for i in rng.choice(finite_idxs, size=3, replace=False)))
        if sample in seen_samples:
            continue
        seen_samples.add(sample)

        sample_idxs = np.asarray(sample, dtype=np.int64)
        try:
            R, t = _kabsch(source[sample_idxs], target[sample_idxs])
        except Exception:
            continue
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            continue

        inliers, errors = score_model(R, t)
        if _is_better_model(inliers, errors, best_inliers, best_errors):
            best_R, best_t, best_inliers, best_errors = R, t, inliers, errors

    if np.sum(best_inliers) >= 3:
        try:
            best_R, best_t = _kabsch(source[best_inliers], target[best_inliers])
            best_inliers, best_errors = score_model(best_R, best_t)
        except Exception:
            pass

    return best_R, best_t, best_inliers, best_errors


def _is_better_model(
    inliers: np.ndarray,
    errors: np.ndarray,
    best_inliers: np.ndarray,
    best_errors: np.ndarray,
) -> bool:
    count = int(np.sum(inliers))
    best_count = int(np.sum(best_inliers))
    if count != best_count:
        return count > best_count
    if count == 0:
        return False
    mean_error = float(np.mean(errors[inliers]))
    best_mean_error = float(np.mean(best_errors[best_inliers])) if best_count > 0 else float("inf")
    return mean_error < best_mean_error


def _num_unique_triplets(n: int) -> int:
    if n < 3:
        return 0
    return int(n * (n - 1) * (n - 2) // 6)


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)

    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)

    source_centered = source - source_centroid.reshape(1, 3)
    target_centered = target - target_centroid.reshape(1, 3)

    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = target_centroid - R @ source_centroid
    return R, t
