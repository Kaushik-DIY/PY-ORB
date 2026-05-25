"""Sim3 rigid-body similarity transform, ported from pyslam."""

from __future__ import annotations

import numpy as np


def _pose_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


class Sim3Pose:
    def __init__(
        self,
        R: np.ndarray = None,
        t: np.ndarray = None,
        s: float = 1.0,
    ):
        self.R = np.eye(3, dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)
        self.t = np.zeros((3, 1), dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64).reshape(3, 1)
        assert s > 0
        self.s = float(s)

    def __repr__(self) -> str:
        return f"Sim3Pose(R={self.R}, t={self.t.ravel()}, s={self.s})"

    def from_matrix(self, T: np.ndarray) -> "Sim3Pose":
        T = np.asarray(T, dtype=np.float64)
        R = T[:3, :3]
        row_norms = np.linalg.norm(R, axis=1)
        self.s = float(row_norms.mean())
        self.R = R / self.s
        self.t = T[:3, 3].reshape(3, 1)
        return self

    def from_se3_matrix(self, T: np.ndarray) -> "Sim3Pose":
        T = np.asarray(T, dtype=np.float64)
        self.s = 1.0
        self.R = T[:3, :3].copy()
        self.t = T[:3, 3].reshape(3, 1).copy()
        return self

    def matrix(self) -> np.ndarray:
        return _pose_Rt(self.R * self.s, self.t.ravel())

    def inverse(self) -> "Sim3Pose":
        return Sim3Pose(self.R.T, (-1.0 / self.s) * self.R.T @ self.t, 1.0 / self.s)

    def inverse_matrix(self) -> np.ndarray:
        sR_inv = (1.0 / self.s) * self.R.T
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = sR_inv
        T[:3, 3] = (-sR_inv @ self.t).ravel()
        return T

    def to_se3_matrix(self) -> np.ndarray:
        """Return SE3 form [R | t/s; 0 | 1]."""
        return _pose_Rt(self.R, self.t.ravel() / self.s)

    def copy(self) -> "Sim3Pose":
        return Sim3Pose(self.R.copy(), self.t.copy(), self.s)

    def map(self, p3d: np.ndarray) -> np.ndarray:
        return self.s * self.R @ np.asarray(p3d).reshape(3, 1) + self.t

    def map_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return (self.s * self.R @ points.T + self.t).T

    def __matmul__(self, other: "Sim3Pose") -> "Sim3Pose":
        if isinstance(other, Sim3Pose):
            return Sim3Pose(
                self.R @ other.R,
                self.s * self.R @ other.t + self.t,
                self.s * other.s,
            )
        if isinstance(other, np.ndarray) and other.shape == (4, 4):
            R_other = other[:3, :3]
            s_other = float(np.linalg.norm(R_other, axis=1).mean())
            R_other = R_other / s_other
            t_other = other[:3, 3].reshape(3, 1)
            return Sim3Pose(
                self.R @ R_other,
                self.s * self.R @ t_other + self.t,
                self.s * s_other,
            )
        raise TypeError(f"Unsupported operand for @: Sim3Pose and {type(other)}")

    def __str__(self) -> str:
        return f"Sim3Pose(R={self.R}, t={self.t.ravel()}, s={self.s})"
