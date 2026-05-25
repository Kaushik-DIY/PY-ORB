"""
Small geometry helper functions.
This module provides SE3 algebra, angle utilities, normalization, and skew matrices.
"""

from __future__ import annotations

import math
import numpy as np


sign = lambda x: math.copysign(1, x)


def s1_diff_deg(angle1, angle2):
    diff = (angle1 - angle2) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def s1_dist_deg(angle1, angle2):
    return abs(s1_diff_deg(angle1, angle2))


k2pi = 2.0 * math.pi


def s1_diff_rad(angle1, angle2):
    diff = (angle1 - angle2) % k2pi
    if diff > math.pi:
        diff -= k2pi
    return diff


def s1_dist_rad(angle1, angle2):
    return abs(s1_diff_rad(angle1, angle2))


def poseRt(R, t):
    ret = np.eye(4, dtype=np.float64)
    ret[:3, :3] = np.asarray(R, dtype=np.float64)
    ret[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return ret


def inv_poseRt(R, t):
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    ret = np.eye(4, dtype=np.float64)
    ret[:3, :3] = R.T
    ret[:3, 3] = -R.T @ np.ascontiguousarray(t)
    return ret


def inv_T(T):
    T = np.asarray(T, dtype=np.float64)
    ret = np.eye(4, dtype=np.float64)
    R_T = T[:3, :3].T
    t = T[:3, 3]
    ret[:3, :3] = R_T
    ret[:3, 3] = -R_T @ np.ascontiguousarray(t)
    return ret


def normalize_vector(v):
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1.0e-10:
        return v, norm
    return v / norm, norm


def normalize_vector2(v):
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1.0e-10:
        return v
    return v / norm


def add_ones(x):
    x = np.asarray(x)
    if len(x.shape) == 1:
        return add_ones_1D(x)
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)


def add_ones_1D(x):
    x = np.asarray(x)
    return np.array([x[0], x[1], 1], dtype=x.dtype)


def add_ones_numba(uvs):
    uvs = np.asarray(uvs)
    N = uvs.shape[0]
    out = np.ones((N, 3), dtype=uvs.dtype)
    out[:, 0:2] = uvs
    return out


def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]


def skew(w):
    wx, wy, wz = np.asarray(w, dtype=np.float64).ravel()
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=np.float64,
    )
