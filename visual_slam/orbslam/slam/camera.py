"""
Camera models and projection helpers for RGB-D SLAM.
This module stores intrinsics and provides projection, backprojection, and image-bound checks.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

import cv2
import numpy as np

from visual_slam.orbslam.slam.sensor_types import SensorType, get_sensor_type
from visual_slam.orbslam.slam.config_parameters import Parameters


# Enumerate the camera model families supported by the pipeline.
class CameraType(Enum):
    """Supported camera model identifiers."""

    NONE = 0
    PINHOLE = 1


def fov2focal(fov: float, pixels: int | float) -> float:
    return float(pixels) / (2.0 * math.tan(float(fov) / 2.0))


def focal2fov(focal: float, pixels: int | float) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def _add_ones(uvs: np.ndarray) -> np.ndarray:
    uvs = np.asarray(uvs)
    return np.concatenate([uvs, np.ones((uvs.shape[0], 1), dtype=uvs.dtype)], axis=1)


# Group reusable projection and backprojection helper functions.
class CameraUtils:
    """Static camera geometry helpers used across tracking and mapping."""

    @staticmethod
    def backproject_3d(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        depth = np.asarray(depth, dtype=np.float64).reshape(-1)
        uv1 = _add_ones(uv).astype(np.float64)
        p3d = depth.reshape(-1, 1) * (np.linalg.inv(K) @ uv1.T).T
        return p3d.reshape(-1, 3)

    @staticmethod
    def project(xcs: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xcs = np.asarray(xcs, dtype=np.float64)
        if xcs.ndim == 1:
            xcs = xcs.reshape(1, 3)
        projs = K @ xcs.T
        zs = projs[2, :]
        uvs = (projs[:2, :] / zs).T
        return uvs, zs

    @staticmethod
    def project_stereo(xcs: np.ndarray, K: np.ndarray, bf: float) -> tuple[np.ndarray, np.ndarray]:
        xcs = np.asarray(xcs, dtype=np.float64)
        if xcs.ndim == 1:
            xcs = xcs.reshape(1, 3)
        uvs, zs = CameraUtils.project(xcs, K)
        ur = uvs[:, 0] - float(bf) / zs
        return np.concatenate([uvs, ur[:, None]], axis=1), zs

    @staticmethod
    def unproject_points(uvs: np.ndarray, Kinv: np.ndarray) -> np.ndarray:
        uvs = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
        return (Kinv @ _add_ones(uvs).T).T[:, :2]

    @staticmethod
    def unproject_points_3d(uvs: np.ndarray, depths: np.ndarray, Kinv: np.ndarray) -> np.ndarray:
        uvs = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
        depths = np.asarray(depths, dtype=np.float64).reshape(-1)
        return (Kinv @ (_add_ones(uvs).T * depths)).T[:, :3]

    @staticmethod
    def are_in_image(
        uvs: np.ndarray,
        zs: np.ndarray,
        u_min: float,
        u_max: float,
        v_min: float,
        v_max: float,
    ) -> np.ndarray:
        uvs = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
        zs = np.asarray(zs, dtype=np.float64).reshape(-1)
        return (
            (uvs[:, 0] >= u_min)
            & (uvs[:, 0] < u_max)
            & (uvs[:, 1] >= v_min)
            & (uvs[:, 1] < v_max)
            & (zs > 0)
        )


# Hold the common intrinsic and geometric fields shared by all camera models.
class CameraBase:
    def __init__(self):
        self.type = CameraType.NONE
        self.width = None
        self.height = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.K = None
        self.Kinv = None
        self.D = None
        self.is_distorted = None
        self.fps = None
        self.bf = None
        self.b = None
        self.depth_factor = None
        self.depth_threshold = None
        self.u_min = None
        self.u_max = None
        self.v_min = None
        self.v_max = None
        self.fovx = None
        self.fovy = None
        self.sensor_type = SensorType.MONOCULAR
        self.initialized = False


# Represent a calibrated camera with projection and image-bound checks.
class Camera(CameraBase):
    """

    This implementation accepts either:
    - None
    - explicit keyword parameters through subclasses/classmethods
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        if config is not None:
            self.init_from_config_dict(config)

    def init_from_config_dict(self, config: dict[str, Any]) -> None:
        cam_settings = config.get("cam_settings", config)

        self.width = int(cam_settings.get("Camera.width", cam_settings.get("Camera.w")))
        self.height = int(cam_settings.get("Camera.height", cam_settings.get("Camera.h")))
        self.fx = float(cam_settings["Camera.fx"])
        self.fy = float(cam_settings["Camera.fy"])
        self.cx = float(cam_settings["Camera.cx"])
        self.cy = float(cam_settings["Camera.cy"])

        self.D = np.array(
            cam_settings.get("D", cam_settings.get("DistCoef", [0, 0, 0, 0, 0])),
            dtype=np.float64,
        )
        self.is_distorted = np.linalg.norm(self.D) > 1e-10
        self.fps = float(cam_settings.get("Camera.fps", 30.0))
        self.sensor_type = get_sensor_type(cam_settings.get("sensor_type", config.get("sensor_type", "mono")))

        if "Camera.bf" in cam_settings and self.sensor_type != SensorType.MONOCULAR:
            self.bf = float(cam_settings["Camera.bf"])
            self.b = self.bf / self.fx
        elif self.sensor_type != SensorType.MONOCULAR:
            self.b = float(cam_settings.get("Camera.b", Parameters.kDefaultRgbdBaselineMeters))
            self.bf = self.fx * self.b

        depth_map_factor = float(cam_settings.get("DepthMapFactor", 1.0))
        self.depth_factor = 1.0 / depth_map_factor if depth_map_factor != 0 else 1.0

        self.depth_threshold = float("inf")
        if "ThDepth" in cam_settings and self.sensor_type != SensorType.MONOCULAR:
            if self.bf is None:
                raise ValueError("Camera.bf is required when ThDepth is used for RGB-D/stereo.")
            self.depth_threshold = self.bf * float(cam_settings["ThDepth"]) / self.fx

        self.set_intrinsic_matrices()
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)

    def set_intrinsic_matrices(self) -> None:
        self.K = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.Kinv = np.array(
            [
                [1.0 / self.fx, 0.0, -self.cx / self.fx],
                [0.0, 1.0 / self.fy, -self.cy / self.fy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def is_stereo(self) -> bool:
        return self.bf is not None and self.sensor_type != SensorType.MONOCULAR

    def project(self, xcs: np.ndarray):
        raise NotImplementedError

    def project_stereo(self, xcs: np.ndarray):
        raise NotImplementedError

    def is_in_image(self, uv: np.ndarray, z: float) -> bool:
        return (
            (uv[0] >= self.u_min)
            & (uv[0] < self.u_max)
            & (uv[1] >= self.v_min)
            & (uv[1] < self.v_max)
            & (z > 0)
        )

    def are_in_image(self, uvs: np.ndarray, zs: np.ndarray) -> np.ndarray:
        return CameraUtils.are_in_image(uvs, zs, self.u_min, self.u_max, self.v_min, self.v_max)

    def set_fovx(self, fovx: float) -> None:
        self.fx = fov2focal(fovx, self.width)
        self.fovx = fovx
        self.set_intrinsic_matrices()

    def set_fovy(self, fovy: float) -> None:
        self.fy = fov2focal(fovy, self.height)
        self.fovy = fovy
        self.set_intrinsic_matrices()


# Specialize the generic camera model for the pinhole RGB-D setup.
class PinholeCamera(Camera):

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.type = CameraType.PINHOLE
        if config is not None:
            self.u_min = 0
            self.u_max = self.width
            self.v_min = 0
            self.v_max = self.height
            self.init()

    @classmethod
    def from_params(
        cls,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        sensor_type: SensorType = SensorType.RGBD,
        bf: float | None = None,
        baseline: float | None = None,
        depth_map_factor: float = 5000.0,
        th_depth: float | None = 40.0,
        fps: float = 30.0,
        D: np.ndarray | list[float] | None = None,
    ) -> "PinholeCamera":
        if bf is None and baseline is not None:
            bf = fx * baseline
        if bf is None and sensor_type != SensorType.MONOCULAR:
            bf = fx * Parameters.kDefaultRgbdBaselineMeters

        config = {
            "Camera.width": width,
            "Camera.height": height,
            "Camera.fx": fx,
            "Camera.fy": fy,
            "Camera.cx": cx,
            "Camera.cy": cy,
            "Camera.fps": fps,
            "sensor_type": sensor_type,
            "DepthMapFactor": depth_map_factor,
            "D": [0, 0, 0, 0, 0] if D is None else D,
        }
        if bf is not None:
            config["Camera.bf"] = bf
        if th_depth is not None:
            config["ThDepth"] = th_depth

        return cls(config)

    def init(self) -> None:
        if not self.initialized:
            self.initialized = True
            self.undistort_image_bounds()

    def project(self, xcs: np.ndarray):
        return CameraUtils.project(xcs, self.K)

    def project_stereo(self, xcs: np.ndarray):
        if self.bf is None:
            raise ValueError("project_stereo requires camera.bf.")
        return CameraUtils.project_stereo(xcs, self.K, self.bf)

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(1, 2)
        return CameraUtils.unproject_points(uv, self.Kinv).reshape(2)

    def unproject_3d(self, uv: np.ndarray, depth: float) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(1, 2)
        return CameraUtils.unproject_points_3d(uv, np.array([depth], dtype=np.float64), self.Kinv).reshape(3)

    def unproject_points(self, uvs: np.ndarray) -> np.ndarray:
        return CameraUtils.unproject_points(uvs, self.Kinv)

    def unproject_points_3d(self, uvs: np.ndarray, depths: np.ndarray) -> np.ndarray:
        return CameraUtils.unproject_points_3d(uvs, depths, self.Kinv)

    def undistort_points(self, uvs: np.ndarray) -> np.ndarray:
        uvs = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
        if self.D is None or not self.is_distorted:
            return uvs
        pts = uvs.reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.D, P=self.K).reshape(-1, 2)

    def undistort_image_bounds(self) -> None:
        if self.width is None or self.height is None:
            return

        if self.D is None or not self.is_distorted:
            self.u_min = 0
            self.u_max = self.width
            self.v_min = 0
            self.v_max = self.height
            return

        corners = np.array(
            [
                [0, 0],
                [self.width, 0],
                [0, self.height],
                [self.width, self.height],
            ],
            dtype=np.float64,
        )
        undistorted = self.undistort_points(corners)
        self.u_min = float(np.min(undistorted[:, 0]))
        self.u_max = float(np.max(undistorted[:, 0]))
        self.v_min = float(np.min(undistorted[:, 1]))
        self.v_max = float(np.max(undistorted[:, 1]))
