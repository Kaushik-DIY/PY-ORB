"""
Camera pose wrapper for SE3 transforms.
This module keeps matrix, rotation, translation, and camera-center views in sync.
"""

from __future__ import annotations

import numpy as np
import g2o


# Wrap a g2o SE3 pose and expose cached geometric views of it.
class CameraPose:
    """Camera pose wrapper around a g2o SE3 object using the Tcw convention."""

    def __init__(self, pose=None):
        if pose is None:
            pose = g2o.Isometry3d()
        self.covariance = np.identity(6, dtype=np.float64)
        self.set(pose)

    def copy(self) -> "CameraPose":
        return CameraPose(self._pose.copy())

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_pose" in state:
            state["_pose"] = self._pose.matrix()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pose = g2o.Isometry3d(self._pose)
        self.set_mat(self._pose.matrix())

    def set(self, pose) -> None:
        if isinstance(pose, g2o.SE3Quat):
            self._pose = g2o.Isometry3d(pose.orientation(), pose.position())
        elif isinstance(pose, g2o.Isometry3d):
            self._pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self._pose = g2o.Isometry3d(np.asarray(pose, dtype=np.float64))
        self.set_mat(self._pose.matrix())

    def update(self, pose) -> None:
        self.set(pose)

    def set_mat(self, Tcw: np.ndarray) -> None:
        Tcw = np.asarray(Tcw, dtype=np.float64)
        if Tcw.shape != (4, 4):
            raise ValueError(f"Expected Tcw shape (4,4), got {Tcw.shape}")

        self.Tcw = np.ascontiguousarray(Tcw)
        self.Rcw = np.ascontiguousarray(self.Tcw[:3, :3])
        self.tcw = np.ascontiguousarray(self.Tcw[:3, 3])
        self.Rwc = np.ascontiguousarray(self.Rcw.T)
        self.Ow = np.ascontiguousarray(-(self.Rwc @ self.tcw))

    def update_mat(self, Tcw: np.ndarray) -> None:
        self.set_from_matrix(Tcw)

    @property
    def isometry3d(self):
        return self._pose

    @property
    def quaternion(self):
        return self._pose.orientation()

    @property
    def orientation(self):
        return self._pose.orientation()

    @property
    def position(self):
        return self._pose.position()

    def get_rotation_matrix(self) -> np.ndarray:
        return self._pose.rotation_matrix()

    def get_rotation_angle_axis(self):
        return g2o.AngleAxis(self._pose.orientation())

    def get_matrix(self) -> np.ndarray:
        return self._pose.matrix()

    def get_inverse_matrix(self) -> np.ndarray:
        return self._pose.inverse().matrix()

    def set_from_quaternion_and_position(self, quaternion, position) -> None:
        self.set(g2o.Isometry3d(quaternion, np.asarray(position, dtype=np.float64)))

    def set_from_matrix(self, Tcw: np.ndarray) -> None:
        self.set(g2o.Isometry3d(np.asarray(Tcw, dtype=np.float64)))

    def set_from_rotation_and_translation(self, Rcw: np.ndarray, tcw: np.ndarray) -> None:
        self.set(g2o.Isometry3d(g2o.Quaternion(np.asarray(Rcw, dtype=np.float64)),
                                np.asarray(tcw, dtype=np.float64)))

    def set_quaternion(self, quaternion) -> None:
        self.set(g2o.Isometry3d(quaternion, self._pose.position()))

    def set_rotation_matrix(self, Rcw: np.ndarray) -> None:
        self.set(g2o.Isometry3d(g2o.Quaternion(np.asarray(Rcw, dtype=np.float64)),
                                self._pose.position()))

    def set_translation(self, tcw: np.ndarray) -> None:
        self.set(g2o.Isometry3d(self._pose.orientation(), np.asarray(tcw, dtype=np.float64)))
