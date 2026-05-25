"""
Motion-model helpers for frame pose prediction.
This module stores constant-velocity estimates and optional damping for tracking.
"""

from __future__ import annotations

import numpy as np
import g2o


def _to_matrix(pose) -> np.ndarray:
    if isinstance(pose, np.ndarray):
        T = np.asarray(pose, dtype=np.float64)
    elif hasattr(pose, "matrix"):
        T = np.asarray(pose.matrix(), dtype=np.float64)
    else:
        T = np.asarray(pose, dtype=np.float64)

    if T.shape != (4, 4):
        raise ValueError(f"Expected SE3 matrix shape (4,4), got {T.shape}")
    return T


def _isometry(T: np.ndarray):
    return g2o.Isometry3d(np.asarray(T, dtype=np.float64))


def _pose_from_orientation_position(orientation, position) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64).reshape(3)

    try:
        iso = g2o.Isometry3d(orientation, position)
        return np.asarray(iso.matrix(), dtype=np.float64)
    except Exception:
        pass

    T = np.eye(4, dtype=np.float64)

    if hasattr(orientation, "matrix"):
        T[:3, :3] = np.asarray(orientation.matrix(), dtype=np.float64)
    elif hasattr(orientation, "rotation_matrix"):
        T[:3, :3] = np.asarray(orientation.rotation_matrix(), dtype=np.float64)
    else:
        # Last-resort fallback: identity orientation.
        T[:3, :3] = np.eye(3, dtype=np.float64)

    T[:3, 3] = position
    return T


# Define the shared interface for motion-model prediction and update.
class MotionModelBase(object):
    def __init__(
        self,
        timestamp=None,
        initial_position=None,
        initial_orientation=None,
        initial_covariance=None,
    ):
        self.timestamp = timestamp

        if initial_position is not None and initial_orientation is not None:
            self.Tcw = _pose_from_orientation_position(initial_orientation, initial_position)
        else:
            self.Tcw = np.eye(4, dtype=np.float64)
            if initial_position is not None:
                self.Tcw[:3, 3] = np.asarray(initial_position, dtype=np.float64).reshape(3)

        self.covariance = initial_covariance
        self.is_ok = False
        self.initialized = False

    @property
    def position(self):
        return self.Tcw[:3, 3].copy()

    @position.setter
    def position(self, value):
        self.Tcw[:3, 3] = np.asarray(value, dtype=np.float64).reshape(3)

    @property
    def orientation(self):
        return _isometry(self.Tcw).orientation()

    @orientation.setter
    def orientation(self, value):
        self.Tcw = _pose_from_orientation_position(value, self.position)

    def current_pose(self):
        return _isometry(self.Tcw), self.covariance

    def predict_pose(self, timestamp, prev_position=None, prev_orientation=None):
        return None

    def update_pose(self, timestamp, new_position, new_orientation, new_covariance=None):
        return None

    def update_pose_from_matrix(self, timestamp, Tcw, new_covariance=None):
        self.timestamp = timestamp
        self.Tcw = _to_matrix(Tcw)
        self.covariance = new_covariance
        self.initialized = True
        self.is_ok = True

    def apply_correction(self, correction):
        return None

    def reset(self):
        self.timestamp = None
        self.Tcw = np.eye(4, dtype=np.float64)
        self.covariance = None
        self.is_ok = False
        self.initialized = False


# Predict the next pose using a constant-velocity rigid motion model.
class MotionModel(MotionModelBase):
    """

    delta_Tcw maps previous camera pose to predicted current camera pose.
    """

    def __init__(
        self,
        timestamp=None,
        initial_position=None,
        initial_orientation=None,
        initial_covariance=None,
    ):
        super().__init__(timestamp, initial_position, initial_orientation, initial_covariance)
        self.delta_Tcw = np.eye(4, dtype=np.float64)

    def predict_pose(self, timestamp, prev_position=None, prev_orientation=None):
        if prev_position is not None and prev_orientation is not None:
            self.Tcw = _pose_from_orientation_position(prev_orientation, prev_position)

        if not self.initialized:
            return _isometry(self.Tcw), self.covariance

        T_pred = self.delta_Tcw @ self.Tcw
        return _isometry(T_pred), self.covariance

    def update_pose(self, timestamp, new_position, new_orientation, new_covariance=None):
        T_new = _pose_from_orientation_position(new_orientation, new_position)
        self.update_pose_from_matrix(timestamp, T_new, new_covariance)

    def update_pose_from_matrix(self, timestamp, Tcw, new_covariance=None):
        T_new = _to_matrix(Tcw)

        if self.initialized:
            self.delta_Tcw = T_new @ np.linalg.inv(self.Tcw)

        self.timestamp = timestamp
        self.Tcw = T_new
        self.covariance = new_covariance
        self.initialized = True
        self.is_ok = True

    def apply_correction(self, correction):
        correction = _to_matrix(correction)
        self.Tcw = correction @ self.Tcw
        self.delta_Tcw = correction @ self.delta_Tcw


# Predict the next pose using a damped constant-velocity model.
class MotionModelDamping(MotionModelBase):
    """
    Timestamp-aware damped motion model.

    preserves the class name and public methods. Translation is damped with dt;
    rotation is propagated through the last delta rotation matrix.
    """

    def __init__(
        self,
        timestamp=None,
        initial_position=None,
        initial_orientation=None,
        initial_covariance=None,
        damping=0.95,
    ):
        super().__init__(timestamp, initial_position, initial_orientation, initial_covariance)
        self.v_linear = np.zeros(3, dtype=np.float64)
        self.delta_R = np.eye(3, dtype=np.float64)
        self.damp = float(damping)

    def predict_pose(self, timestamp, prev_position=None, prev_orientation=None):
        if prev_position is not None and prev_orientation is not None:
            self.Tcw = _pose_from_orientation_position(prev_orientation, prev_position)

        if not self.initialized:
            return _isometry(self.Tcw), self.covariance

        dt = 0.0 if self.timestamp is None else float(timestamp - self.timestamp)

        T_pred = self.Tcw.copy()
        T_pred[:3, :3] = self.delta_R @ T_pred[:3, :3]
        T_pred[:3, 3] = T_pred[:3, 3] + self.v_linear * dt * self.damp

        return _isometry(T_pred), self.covariance

    def update_pose(self, timestamp, new_position, new_orientation, new_covariance=None):
        T_new = _pose_from_orientation_position(new_orientation, new_position)
        self.update_pose_from_matrix(timestamp, T_new, new_covariance)

    def update_pose_from_matrix(self, timestamp, Tcw, new_covariance=None):
        T_new = _to_matrix(Tcw)

        if self.initialized:
            dt = float(timestamp - self.timestamp)
            if abs(dt) > 1e-12:
                self.v_linear = (T_new[:3, 3] - self.Tcw[:3, 3]) / dt
            self.delta_R = T_new[:3, :3] @ self.Tcw[:3, :3].T

        self.timestamp = timestamp
        self.Tcw = T_new
        self.covariance = new_covariance
        self.initialized = True
        self.is_ok = True

    def apply_correction(self, correction):
        correction = _to_matrix(correction)
        self.Tcw = correction @ self.Tcw
        self.v_linear = correction[:3, :3] @ self.v_linear
        self.delta_R = correction[:3, :3] @ self.delta_R
