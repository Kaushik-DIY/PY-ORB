"""
=============================================================================
visual_slam/g2o_compat.py

Compatibility helpers for the installed Python g2o binding.

Purpose
-------
pySLAM/ORB-SLAM-style optimizer code often assumes projection edges expose
camera fields directly, e.g. edge.fx, edge.fy, edge.cx, edge.cy, edge.bf.

The installed g2o binding in this workspace does not expose those fields.
Instead, projection edges use g2o.CameraParameters and edge.set_parameter_id().

This module provides a narrow compatibility layer so the visual_slam optimizer
can be implemented in an ORB-SLAM/pySLAM-like way without depending on pySLAM's
exact g2o wrapper.

Important limitation
--------------------
The installed g2o CameraParameters API supports one focal length.
For ORB/RGB-D compatibility we use fx, because stereo/RGB-D virtual stereo
geometry defines bf = fx * baseline. fy is only approximated in the vertical
projection. For TUM Freiburg1 this approximation is small because fx and fy
differ by about 0.15%.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import g2o


@dataclass(frozen=True)
class G2OCamera:
    """Camera model used by the parameter-based g2o projection edges."""

    fx: float
    fy: float
    cx: float
    cy: float
    bf: float = 0.0

    @property
    def focal(self) -> float:
        """Single focal length required by g2o.CameraParameters."""
        #return 0.5 * (float(self.fx) + float(self.fy))
        return float(self.fx)


    @property
    def baseline(self) -> float:
        """
        Baseline in meters/pseudo-meters for g2o.CameraParameters.

        ORB-SLAM typically stores bf = fx * baseline. Therefore:
            baseline = bf / fx
        """
        if abs(self.fx) < 1e-12:
            return 0.0
        return float(self.bf) / float(self.fx)

    @property
    def principal_point(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=np.float64)


def make_optimizer(verbose: bool = False) -> g2o.SparseOptimizer:
    """Create a SparseOptimizer with a robust SE3 solver fallback order."""
    optimizer = g2o.SparseOptimizer()

    linear_solver = None
    for solver_name in (
        "LinearSolverCholmodSE3",
        "LinearSolverCSparseSE3",
        "LinearSolverEigenSE3",
        "LinearSolverDenseSE3",
    ):
        solver_cls = getattr(g2o, solver_name, None)
        if solver_cls is None:
            continue
        try:
            linear_solver = solver_cls()
            break
        except Exception:
            continue

    if linear_solver is None:
        raise RuntimeError("No usable g2o SE3 linear solver found.")

    block_solver = g2o.BlockSolverSE3(linear_solver)
    algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)

    optimizer.set_algorithm(algorithm)
    optimizer.set_verbose(verbose)
    return optimizer


def add_camera_parameters(
    optimizer: g2o.SparseOptimizer,
    camera: G2OCamera,
    parameter_id: int = 0,
) -> g2o.CameraParameters:
    """Add g2o.CameraParameters to optimizer and return the parameter object."""
    params = g2o.CameraParameters(
        float(camera.focal),
        camera.principal_point,
        float(camera.baseline),
    )
    params.set_id(int(parameter_id))
    optimizer.add_parameter(params)
    return params


def se3quat_from_matrix(Tcw: np.ndarray) -> g2o.SE3Quat:
    """Convert a 4x4 SE3 matrix to g2o.SE3Quat."""
    Tcw = np.asarray(Tcw, dtype=np.float64)
    if Tcw.shape != (4, 4):
        raise ValueError(f"Expected Tcw shape (4,4), got {Tcw.shape}")

    R = np.ascontiguousarray(Tcw[:3, :3], dtype=np.float64)
    t = np.ascontiguousarray(Tcw[:3, 3], dtype=np.float64)
    return g2o.SE3Quat(R, t)


def add_pose_vertex(
    optimizer: g2o.SparseOptimizer,
    vertex_id: int,
    Tcw: np.ndarray,
    fixed: bool = False,
) -> g2o.VertexSE3Expmap:
    """Add a camera pose vertex using VertexSE3Expmap."""
    vertex = g2o.VertexSE3Expmap()
    vertex.set_id(int(vertex_id))
    vertex.set_estimate(se3quat_from_matrix(Tcw))
    vertex.set_fixed(bool(fixed))
    optimizer.add_vertex(vertex)
    return vertex


def add_point_vertex(
    optimizer: g2o.SparseOptimizer,
    vertex_id: int,
    point_w: np.ndarray,
    fixed: bool = False,
    marginalized: bool = True,
) -> g2o.VertexSBAPointXYZ:
    """Add a 3D map-point vertex using VertexSBAPointXYZ."""
    point_w = np.asarray(point_w, dtype=np.float64).reshape(3)

    vertex = g2o.VertexSBAPointXYZ()
    vertex.set_id(int(vertex_id))
    vertex.set_estimate(point_w)
    vertex.set_marginalized(bool(marginalized))
    vertex.set_fixed(bool(fixed))
    optimizer.add_vertex(vertex)
    return vertex


def make_huber(delta: float) -> g2o.RobustKernelHuber:
    """Create a Huber robust kernel."""
    kernel = g2o.RobustKernelHuber()
    kernel.set_delta(float(delta))
    return kernel


def add_mono_edge(
    optimizer: g2o.SparseOptimizer,
    edge_id: int,
    point_vertex: g2o.VertexSBAPointXYZ,
    pose_vertex: g2o.VertexSE3Expmap,
    uv: np.ndarray,
    inv_sigma2: float = 1.0,
    parameter_id: int = 0,
    huber_delta: Optional[float] = None,
) -> g2o.EdgeProjectXYZ2UV:
    """
    Add a monocular reprojection edge.

    This uses g2o.EdgeProjectXYZ2UV with CameraParameters, not direct fx/fy
    edge attributes.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(2)

    edge = g2o.EdgeProjectXYZ2UV()
    edge.set_id(int(edge_id))
    edge.set_vertex(0, point_vertex)
    edge.set_vertex(1, pose_vertex)
    edge.set_measurement(uv)
    edge.set_information(np.eye(2, dtype=np.float64) * float(inv_sigma2))
    edge.set_parameter_id(0, int(parameter_id))

    if huber_delta is not None:
        edge.set_robust_kernel(make_huber(huber_delta))

    optimizer.add_edge(edge)
    return edge


def add_stereo_edge(
    optimizer: g2o.SparseOptimizer,
    edge_id: int,
    point_vertex: g2o.VertexSBAPointXYZ,
    pose_vertex: g2o.VertexSE3Expmap,
    uvu: np.ndarray,
    inv_sigma2: float = 1.0,
    parameter_id: int = 0,
    huber_delta: Optional[float] = None,
) -> g2o.EdgeProjectXYZ2UVU:
    """
    Add a stereo/RGB-D virtual-stereo reprojection edge.

    Measurement convention:
        [u_left, v_left, u_right]

    For RGB-D, u_right can be synthesized as:
        u_right = u_left - bf / depth
    """
    uvu = np.asarray(uvu, dtype=np.float64).reshape(3)

    edge = g2o.EdgeProjectXYZ2UVU()
    edge.set_id(int(edge_id))
    edge.set_vertex(0, point_vertex)
    edge.set_vertex(1, pose_vertex)
    edge.set_measurement(uvu)
    edge.set_information(np.eye(3, dtype=np.float64) * float(inv_sigma2))
    edge.set_parameter_id(0, int(parameter_id))

    if huber_delta is not None:
        edge.set_robust_kernel(make_huber(huber_delta))

    optimizer.add_edge(edge)
    return edge


def project_mono_with_g2o_camera(camera: G2OCamera, point_c: np.ndarray) -> np.ndarray:
    """
    Project a camera-frame point using the same single-focal convention as
    g2o.CameraParameters.
    """
    x, y, z = np.asarray(point_c, dtype=np.float64).reshape(3)
    if z <= 0:
        raise ValueError("Point must have positive depth for projection.")

    f = camera.focal
    return np.array(
        [
            f * x / z + camera.cx,
            f * y / z + camera.cy,
        ],
        dtype=np.float64,
    )


def project_stereo_with_g2o_camera(camera: G2OCamera, point_c: np.ndarray) -> np.ndarray:
    """
    Project a camera-frame point into [u_left, v_left, u_right] using the same
    convention as g2o.CameraParameters.
    """
    x, y, z = np.asarray(point_c, dtype=np.float64).reshape(3)
    if z <= 0:
        raise ValueError("Point must have positive depth for projection.")

    f = camera.focal
    u_left = f * x / z + camera.cx
    v_left = f * y / z + camera.cy
    u_right = f * (x - camera.baseline) / z + camera.cx

    return np.array([u_left, v_left, u_right], dtype=np.float64)


def optimize(
    optimizer: g2o.SparseOptimizer,
    iterations: int = 10,
    verbose: bool = False,
) -> int:
    """Initialize and run graph optimization."""
    optimizer.set_verbose(verbose)
    optimizer.initialize_optimization()
    return int(optimizer.optimize(int(iterations)))
