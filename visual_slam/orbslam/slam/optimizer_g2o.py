"""
g2o-based optimization routines for the SLAM back-end.
This module provides pose optimization plus local and global bundle adjustment.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Optional

import g2o
import numpy as np

_BA_PROFILE = os.environ.get("BA_PROFILE", "0") == "1"

# Use the optional C++ backend when available, otherwise stay on the Python implementation.
try:
    import slam_optimizer_core as _SOC
    _SOC_AVAILABLE = True
except ImportError:
    _SOC = None
    _SOC_AVAILABLE = False

from visual_slam.g2o_compat import (
    G2OCamera,
    add_camera_parameters,
    add_mono_edge,
    add_point_vertex,
    add_pose_vertex,
    add_stereo_edge,
    make_optimizer,
    optimize,
)
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.frame import Frame
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.map_point import MapPoint


# Store the outcome and error statistics of one optimizer call.
@dataclass
class OptimizerResult:
    num_edges: int
    num_inliers: int
    num_outliers: int
    mean_squared_error: float
    success: bool
    mean_error_before: float = float("inf")
    mean_error_after: float = float("inf")
    num_keyframes: int = 0
    num_map_points: int = 0
    aborted: bool = False
    reason: str = ""


def _as_list(values) -> list:
    if values is None:
        return []
    if hasattr(values, "to_list"):
        return values.to_list()
    return list(values)


def _is_bad_keyframe(kf: KeyFrame) -> bool:
    return hasattr(kf, "is_bad") and kf.is_bad()


def _is_bad_point(point: MapPoint) -> bool:
    return point is None or (hasattr(point, "is_bad") and point.is_bad())


def _abort_requested(abort_flag) -> bool:
    if abort_flag is None:
        return False
    value = getattr(abort_flag, "value", abort_flag)
    if callable(value):
        try:
            value = value()
        except TypeError:
            pass
    return bool(value)


def _is_finite_pose(Tcw: np.ndarray) -> bool:
    Tcw = np.asarray(Tcw, dtype=np.float64)
    return Tcw.shape == (4, 4) and np.all(np.isfinite(Tcw))


def _point_has_positive_depth(frame_or_kf, point_w: np.ndarray) -> bool:
    try:
        Tcw = np.asarray(frame_or_kf.Tcw(), dtype=np.float64).reshape(4, 4)
    except Exception:
        return False
    point_w = np.asarray(point_w, dtype=np.float64).reshape(3)
    if not (_is_finite_pose(Tcw) and np.all(np.isfinite(point_w))):
        return False
    point_c = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
    return bool(np.isfinite(point_c[2]) and point_c[2] > Parameters.kMinDepth)


def _valid_observation(frame_or_kf, idx: int, point_w: np.ndarray) -> bool:
    try:
        uv = _get_observation_uv(frame_or_kf, idx)
        ur = _get_observation_ur(frame_or_kf, idx)
    except Exception:
        return False
    if uv.shape != (2,) or not np.all(np.isfinite(uv)):
        return False
    if ur >= 0.0 and not np.isfinite(ur):
        return False
    return _point_has_positive_depth(frame_or_kf, point_w)


def _camera_to_g2o(camera) -> G2OCamera:
    bf = getattr(camera, "bf", 0.0)
    if bf is None:
        bf = 0.0
    return G2OCamera(
        fx=float(camera.fx),
        fy=float(camera.fy),
        cx=float(camera.cx),
        cy=float(camera.cy),
        bf=float(bf),
    )


def _get_observation_uv(frame_or_kf, idx: int) -> np.ndarray:
    kps = getattr(frame_or_kf, "kpsu", None)
    if kps is None:
        kps = getattr(frame_or_kf, "kps", getattr(frame_or_kf, "keypoints", None))
    if kps is None:
        raise ValueError("Frame/keyframe has no keypoints.")
    kp = kps[int(idx)]
    if hasattr(kp, "pt"):
        return np.array([kp.pt[0], kp.pt[1]], dtype=np.float64)
    return np.asarray(kp, dtype=np.float64).reshape(2)


def _get_observation_ur(frame_or_kf, idx: int) -> float:
    uRs = getattr(frame_or_kf, "uRs", getattr(frame_or_kf, "kps_ur", None))
    if uRs is None or idx < 0 or idx >= len(uRs):
        return -1.0
    return float(uRs[int(idx)])


def _get_inv_sigma2(frame_or_kf, idx: int) -> float:
    kps = getattr(frame_or_kf, "kps", getattr(frame_or_kf, "keypoints", None))
    if kps is None or idx < 0 or idx >= len(kps):
        return 1.0

    octave = max(0, int(getattr(kps[int(idx)], "octave", 0)))

    feature_manager = FeatureTrackerShared.feature_manager
    if feature_manager is None:
        return 1.0

    octave = min(octave, len(feature_manager.inv_level_sigmas2) - 1)
    return float(feature_manager.inv_level_sigmas2[octave])


def _extract_Tcw_from_pose_vertex(pose_vertex) -> np.ndarray:
    se3_opt = pose_vertex.estimate()
    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, :3] = se3_opt.rotation().matrix()
    Tcw[:3, 3] = se3_opt.translation()
    return Tcw


def _set_frame_pose_from_vertex(frame_or_kf, pose_vertex) -> None:
    Tcw = _extract_Tcw_from_pose_vertex(pose_vertex)
    frame_or_kf.update_pose(g2o.Isometry3d(Tcw))


def _pose_vertex_point_depth(pose_vertex, point_vertex) -> float:
    Tcw = _extract_Tcw_from_pose_vertex(pose_vertex)
    point_w = np.asarray(point_vertex.estimate(), dtype=np.float64).reshape(3)
    point_c = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
    return float(point_c[2])


def _is_depth_positive(edge, pose_vertex, point_vertex) -> bool:
    is_depth_positive = getattr(edge, "is_depth_positive", None)
    if callable(is_depth_positive):
        try:
            return bool(is_depth_positive())
        except Exception:
            pass

    # Binding adaptation: this workspace's g2o projection edges do not expose
    depth = _pose_vertex_point_depth(pose_vertex, point_vertex)
    return bool(np.isfinite(depth) and depth > 0.0)


def _manual_reprojection_chi2(frame_or_kf, idx: int, pose_vertex, point_vertex, is_stereo: bool) -> float:
    Tcw = _extract_Tcw_from_pose_vertex(pose_vertex)
    point_w = np.asarray(point_vertex.estimate(), dtype=np.float64).reshape(3)
    point_c = Tcw[:3, :3] @ point_w + Tcw[:3, 3]

    if not np.all(np.isfinite(point_c)):
        return float("inf")

    z = float(point_c[2])
    if z <= 0.0:
        return float("inf")

    camera = frame_or_kf.camera
    u = float(camera.fx) * float(point_c[0]) / z + float(camera.cx)
    v = float(camera.fy) * float(point_c[1]) / z + float(camera.cy)

    if not np.isfinite(u) or not np.isfinite(v):
        return float("inf")

    uv = _get_observation_uv(frame_or_kf, idx)
    inv_sigma2 = _get_inv_sigma2(frame_or_kf, idx)
    err2 = (u - float(uv[0])) ** 2 + (v - float(uv[1])) ** 2

    if is_stereo:
        ur_obs = _get_observation_ur(frame_or_kf, idx)
        if ur_obs >= 0.0:
            ur = u - float(getattr(camera, "bf", 0.0)) / z
            err2 += (ur - ur_obs) ** 2

    return float(err2 * inv_sigma2)


def _add_reprojection_edge(
    optimizer,
    edge_id: int,
    point_vertex,
    pose_vertex,
    frame_or_kf,
    idx: int,
    parameter_id: int = 0,
    use_robust_kernel: bool = True,
):
    uv = _get_observation_uv(frame_or_kf, idx)
    ur = _get_observation_ur(frame_or_kf, idx)
    inv_sigma2 = _get_inv_sigma2(frame_or_kf, idx)

    if ur >= 0.0:
        measurement = np.array([uv[0], uv[1], ur], dtype=np.float64)
        edge = add_stereo_edge(
            optimizer=optimizer,
            edge_id=edge_id,
            point_vertex=point_vertex,
            pose_vertex=pose_vertex,
            uvu=measurement,
            inv_sigma2=inv_sigma2,
            parameter_id=parameter_id,
            huber_delta=Parameters.kHuberStereo if use_robust_kernel else None,
        )
        chi2_threshold = Parameters.kChi2Stereo
        is_stereo = True
    else:
        edge = add_mono_edge(
            optimizer=optimizer,
            edge_id=edge_id,
            point_vertex=point_vertex,
            pose_vertex=pose_vertex,
            uv=uv,
            inv_sigma2=inv_sigma2,
            parameter_id=parameter_id,
            huber_delta=Parameters.kHuberMono if use_robust_kernel else None,
        )
        chi2_threshold = Parameters.kChi2Mono
        is_stereo = False

    return edge, chi2_threshold, is_stereo


def _pose_optimization_cpp(frame, rounds: int, iters_per_round: int, print=print) -> tuple[int, float]:
    """C++ motion-only BA via slam_optimizer_core.run_pose_optimization()."""
    from visual_slam.orbslam.slam.slam_optimizer_bridge import (
        pack_pose_optimization, unpack_pose_optimization
    )
    feature_manager = FeatureTrackerShared.feature_manager

    frame_pose, observations, camera, valid_indices = pack_pose_optimization(
        frame, feature_manager
    )

    if len(observations) < Parameters.kRelocalizationPoseOpt1MinMatches:
        return 0, float("inf")

    result = _SOC.run_pose_optimization(
        frame_pose, observations, camera,
        rounds=rounds,
        iters_per_round=iters_per_round,
    )

    num_inliers = int(result["num_inliers"])
    mse = float(result["mse"])

    if num_inliers >= Parameters.kRelocalizationPoseOpt1MinMatches:
        unpack_pose_optimization(result, frame, valid_indices)

    return num_inliers, mse


def pose_optimization(
    frame: Frame,
    verbose: bool = False,
    rounds: int = 4,
    iterations_per_round: int = 10,
    print=print,
) -> tuple[int, float]:
    """

    Optimizes only the current frame pose Tcw. Map points are fixed.

    Returns:
        (num_inliers, mean_squared_error)
    """
    if frame is None:
        return 0, float("inf")

    # ---- C++ GIL-free dispatch ----
    if _SOC_AVAILABLE:
        try:
            return _pose_optimization_cpp(frame, rounds, iterations_per_round, print)
        except Exception as _cpp_exc:
            print(f"[slam_optimizer_core] C++ pose_opt failed ({_cpp_exc}), falling back")
    # ---- end C++ dispatch ----

    points = list(getattr(frame, "points", []))
    if len(points) == 0:
        return 0, float("inf")

    optimizer = make_optimizer(verbose=verbose)
    add_camera_parameters(optimizer, _camera_to_g2o(frame.camera), parameter_id=0)

    pose_vertex = add_pose_vertex(
        optimizer=optimizer,
        vertex_id=0,
        Tcw=frame.Tcw(),
        fixed=False,
    )

    edges = []
    vertex_id = 1
    edge_id = 0

    for idx, point in enumerate(points):
        if _is_bad_point(point):
            continue
        point_w = point.get_position()
        if not _valid_observation(frame, idx, point_w):
            continue

        point_vertex = add_point_vertex(
            optimizer=optimizer,
            vertex_id=vertex_id,
            point_w=point_w,
            fixed=True,
            marginalized=True,
        )

        edge, chi2_threshold, is_stereo = _add_reprojection_edge(
            optimizer=optimizer,
            edge_id=edge_id,
            point_vertex=point_vertex,
            pose_vertex=pose_vertex,
            frame_or_kf=frame,
            idx=idx,
            parameter_id=0,
            use_robust_kernel=True,
        )

        edges.append((edge, idx, chi2_threshold, is_stereo))
        vertex_id += 1
        edge_id += 1

    if len(edges) < Parameters.kRelocalizationPoseOpt1MinMatches:
        return 0, float("inf")

    if not hasattr(frame, "outliers") or len(frame.outliers) != len(points):
        frame.outliers = np.zeros(len(points), dtype=bool)

    num_inliers = 0

    for round_idx in range(int(rounds)):
        try:
            optimize(optimizer, iterations=iterations_per_round, verbose=verbose)
        except Exception as exc:
            print(f"pose_optimization: g2o failed: {exc}")
            return 0, float("inf")

        num_inliers = 0

        for edge, idx, chi2_threshold, _ in edges:
            chi2 = float(edge.chi2())

            if not np.isfinite(chi2) or chi2 > chi2_threshold:
                frame.outliers[idx] = True
                edge.set_level(1)
            else:
                frame.outliers[idx] = False
                edge.set_level(0)
                num_inliers += 1

            # ORB-SLAM removes robust kernels in later rounds.
            if round_idx == 2:
                edge.set_robust_kernel(None)

        optimizer.initialize_optimization(0)

    if num_inliers < Parameters.kRelocalizationPoseOpt1MinMatches:
        return num_inliers, float("inf")

    _set_frame_pose_from_vertex(frame, pose_vertex)

    active_chi2 = [
        float(edge.chi2())
        for edge, idx, _, _ in edges
        if idx < len(frame.outliers) and not frame.outliers[idx]
    ]
    mse = float(np.mean(active_chi2)) if active_chi2 else float("inf")

    return num_inliers, mse


def _bundle_adjustment_cpp(
    local_keyframes,
    fixed_keyframes,
    points,
    rounds,
    use_robust_kernel,
    abort_flag,
    map_lock,
    prune_outliers,
    print=print,
) -> "OptimizerResult":
    """C++ BA via slam_optimizer_core — called only when _SOC_AVAILABLE is True."""
    from visual_slam.orbslam.slam.slam_optimizer_bridge import pack_local_ba, unpack_local_ba

    feature_manager = FeatureTrackerShared.feature_manager

    if abort_flag is not None and _abort_requested(abort_flag):
        return OptimizerResult(0, 0, 0, float("inf"), False,
                               aborted=True, reason="aborted before C++ BA")

    # Wire abort into the C++ module flag
    _SOC.set_abort(False)

    (kf_poses, kf_ids, kf_fixed, point_pos,
     observations, camera, kf_list, pt_list, obs_triples) = pack_local_ba(
        local_keyframes, fixed_keyframes, points, feature_manager
    )

    if len(observations) == 0:
        return OptimizerResult(0, 0, 0, float("inf"), False,
                               reason="no valid reprojection edges (C++)")

    # Block solver asserts _sizePoses > 0 — skip if all KFs are fixed (e.g. only KF 0)
    n_free_kfs = int(np.sum(kf_fixed == 0))
    if n_free_kfs == 0:
        return OptimizerResult(0, 0, 0, 0.0, True,
                               reason="no free KFs — nothing to optimize (C++)")

    result = _SOC.run_local_ba(
        kf_poses, kf_ids, kf_fixed, point_pos, observations, camera,
        rounds=rounds,
        use_robust_kernel=use_robust_kernel,
        prune_outliers=prune_outliers,
    )

    n_edges = len(observations)
    n_bad = int(result["n_bad_edges"])
    n_inliers = n_edges - n_bad
    mse = float(result["mse"])

    unpack_local_ba(
        result=result,
        kf_list=kf_list,
        pt_list=pt_list,
        obs_triples=obs_triples,
        local_keyframes=local_keyframes,
        fixed_keyframes=fixed_keyframes,
        fixed_points=False,
        prune_outliers=prune_outliers,
        map_lock=map_lock,
    )

    return OptimizerResult(
        n_edges, n_inliers, n_bad, mse, True,
        num_keyframes=len(kf_list),
        num_map_points=len(pt_list),
    )


def _bundle_adjustment_cpp_deferred(
    local_keyframes,
    fixed_keyframes,
    points,
    rounds,
    use_robust_kernel,
    abort_flag,
    result_dict: dict,
    print=print,
) -> "OptimizerResult":
    """Run global BA through the C++ backend and collect updates before write-back."""
    from visual_slam.orbslam.slam.slam_optimizer_bridge import pack_local_ba

    feature_manager = FeatureTrackerShared.feature_manager

    if abort_flag is not None and _abort_requested(abort_flag):
        return OptimizerResult(0, 0, 0, float("inf"), False,
                               aborted=True, reason="aborted before C++ global BA")

    _SOC.set_abort(False)

    (kf_poses, kf_ids, kf_fixed, point_pos,
     observations, camera, kf_list, pt_list, obs_triples) = pack_local_ba(
        local_keyframes, fixed_keyframes, points, feature_manager
    )

    if len(observations) == 0:
        return OptimizerResult(0, 0, 0, float("inf"), False,
                               reason="no valid reprojection edges (C++ deferred)")

    n_free_kfs = int(np.sum(kf_fixed == 0))
    if n_free_kfs == 0:
        return OptimizerResult(0, 0, 0, 0.0, True,
                               reason="no free KFs — nothing to optimize (C++ deferred)")

    cpp_result = _SOC.run_global_ba(
        kf_poses, kf_ids, point_pos, observations, camera,
        rounds=rounds,
        use_robust_kernel=use_robust_kernel,
        loop_kf_id=0,
    )

    n_edges    = len(observations)
    n_bad      = int(cpp_result["n_bad_edges"])
    n_inliers  = n_edges - n_bad
    initial_mse = float(cpp_result["initial_mse"])
    mse        = float(cpp_result["mse"])

    updated_poses  = cpp_result["updated_poses"]   # (N, 16)
    updated_points = cpp_result["updated_points"]  # (M, 3)
    outlier_mask   = cpp_result["outlier_mask"]    # (K,)

    # Map kf_list rows back to kid-indexed dicts expected by GlobalBundleAdjuster
    local_ids = {id(kf) for kf in local_keyframes}
    fixed_ids = {id(kf) for kf in fixed_keyframes}

    kf_updates: dict = {}
    for i, kf in enumerate(kf_list):
        T = updated_poses[i].reshape(4, 4)
        if not np.all(np.isfinite(T)):
            continue
        kf_updates[kf.kid] = T.copy()

    pt_updates: dict = {}
    for j, p in enumerate(pt_list):
        pos = updated_points[j]
        if np.all(np.isfinite(pos)):
            pt_updates[p.id] = pos.copy()

    fixed_kf_ids = [int(kf_list[i].kid) for i in range(len(kf_list)) if kf_fixed[i]]

    result_dict["keyframes"]         = kf_updates
    result_dict["keyframe_updates"]  = kf_updates
    result_dict["points"]            = pt_updates
    result_dict["point_updates"]     = pt_updates
    result_dict["fixed_keyframes"]   = fixed_kf_ids
    result_dict["num_edges"]         = n_edges
    result_dict["num_inliers"]       = n_inliers
    result_dict["num_outliers"]      = n_bad
    result_dict["mean_error_before"] = initial_mse
    result_dict["mean_error_after"]  = mse

    return OptimizerResult(
        n_edges, n_inliers, n_bad, mse, True,
        num_keyframes=len(kf_list),
        num_map_points=len(pt_list),
    )


def _bundle_adjustment_core(
    local_keyframes: list[KeyFrame],
    fixed_keyframes: list[KeyFrame],
    points: list[MapPoint],
    fixed_points: bool = False,
    rounds: int = 10,
    use_robust_kernel: bool = False,
    abort_flag=None,
    map_lock=None,
    verbose: bool = False,
    result_dict: Optional[dict] = None,
    write_back: bool = True,
    prune_outliers: bool = False,
    print=print,
) -> OptimizerResult:
    _t0 = time.perf_counter() if _BA_PROFILE else 0.0

    local_keyframes = [kf for kf in local_keyframes if kf is not None and not _is_bad_keyframe(kf)]
    fixed_keyframes = [kf for kf in fixed_keyframes if kf is not None and not _is_bad_keyframe(kf)]
    points = [p for p in points if not _is_bad_point(p)]

    if len(local_keyframes) == 0 or len(points) == 0:
        return OptimizerResult(0, 0, 0, float("inf"), False, reason="empty graph input")

    # ---- C++ GIL-free dispatch (slam_optimizer_core) ----
    if _SOC_AVAILABLE and not fixed_points:
        if write_back and result_dict is None:
            # Local BA: write back directly to map objects
            try:
                return _bundle_adjustment_cpp(
                    local_keyframes=local_keyframes,
                    fixed_keyframes=fixed_keyframes,
                    points=points,
                    rounds=rounds,
                    use_robust_kernel=use_robust_kernel,
                    abort_flag=abort_flag,
                    map_lock=map_lock,
                    prune_outliers=prune_outliers,
                    print=print,
                )
            except Exception as _cpp_exc:
                print(f"[slam_optimizer_core] C++ local BA failed ({_cpp_exc}), falling back to Python")
        elif not write_back and result_dict is not None:
            # Global BA deferred: collect updates into result_dict, apply validation + atomic
            # write handled by caller (GlobalBundleAdjuster)
            try:
                return _bundle_adjustment_cpp_deferred(
                    local_keyframes=local_keyframes,
                    fixed_keyframes=fixed_keyframes,
                    points=points,
                    rounds=rounds,
                    use_robust_kernel=use_robust_kernel,
                    abort_flag=abort_flag,
                    result_dict=result_dict,
                    print=print,
                )
            except Exception as _cpp_exc:
                print(f"[slam_optimizer_core] C++ global BA failed ({_cpp_exc}), falling back to Python")
    # ---- end C++ dispatch ----

    optimizer = make_optimizer(verbose=verbose)
    if abort_flag is not None:
        if hasattr(optimizer, "set_force_stop_flag") and abort_flag.__class__.__module__ == "g2o":
            optimizer.set_force_stop_flag(abort_flag)
    add_camera_parameters(optimizer, _camera_to_g2o(local_keyframes[0].camera), parameter_id=0)

    _t1 = time.perf_counter() if _BA_PROFILE else 0.0

    pose_vertices = {}
    point_vertices = {}
    graph_edges = {}

    # makes graph construction traceable across local and global BA windows.
    good_keyframes = []

    for kf in local_keyframes:
        if kf in pose_vertices:
            continue
        v = add_pose_vertex(
            optimizer=optimizer,
            vertex_id=int(kf.kid) * 2,
            Tcw=kf.Tcw(),
            fixed=(kf.kid == 0),
        )
        pose_vertices[kf] = v
        good_keyframes.append(kf)

    for kf in fixed_keyframes:
        if kf in pose_vertices:
            continue
        v = add_pose_vertex(
            optimizer=optimizer,
            vertex_id=int(kf.kid) * 2,
            Tcw=kf.Tcw(),
            fixed=True,
        )
        pose_vertices[kf] = v
        good_keyframes.append(kf)

    _t2 = time.perf_counter() if _BA_PROFILE else 0.0

    for point in points:
        point_w = point.get_position()
        if not np.all(np.isfinite(point_w)):
            continue
        v = add_point_vertex(
            optimizer=optimizer,
            vertex_id=int(point.id) * 2 + 1,
            point_w=point_w,
            fixed=bool(fixed_points),
            marginalized=True,
        )
        point_vertices[point] = v

    _t3 = time.perf_counter() if _BA_PROFILE else 0.0

    edges = []
    edge_id = 0
    points_with_edges = set()

    for point in points:
        point_vertex = point_vertices.get(point)
        if point_vertex is None:
            continue

        for kf, idx in point.observations():
            if _is_bad_keyframe(kf):
                continue
            pose_vertex = pose_vertices.get(kf)
            if pose_vertex is None:
                continue
            if idx < 0 or idx >= len(getattr(kf, "points", [])):
                continue
            if kf.get_point_match(idx) is not point:
                continue
            if not _valid_observation(kf, idx, point.get_position()):
                continue

            edge, chi2_threshold, is_stereo = _add_reprojection_edge(
                optimizer=optimizer,
                edge_id=edge_id,
                point_vertex=point_vertex,
                pose_vertex=pose_vertex,
                frame_or_kf=kf,
                idx=idx,
                parameter_id=0,
                use_robust_kernel=use_robust_kernel,
            )

            edge_data = (point, kf, idx, chi2_threshold, is_stereo, point_vertex, pose_vertex)
            edges.append((edge, *edge_data))
            graph_edges[edge] = edge_data
            points_with_edges.add(point)
            edge_id += 1

    _t4 = time.perf_counter() if _BA_PROFILE else 0.0

    if len(edges) == 0:
        return OptimizerResult(
            0,
            0,
            0,
            float("inf"),
            False,
            num_keyframes=len(pose_vertices),
            num_map_points=len(point_vertices),
            reason="no valid reprojection edges",
        )

    if verbose:
        optimizer.set_verbose(True)

    if _abort_requested(abort_flag):
        return OptimizerResult(
            len(edges),
            0,
            len(edges),
            float("inf"),
            False,
            num_keyframes=len(pose_vertices),
            num_map_points=len(point_vertices),
            aborted=True,
            reason="aborted before optimization",
        )

    _t5 = time.perf_counter() if _BA_PROFILE else 0.0

    num_bad_edges = 0

    try:
        optimizer.initialize_optimization()
        optimizer.compute_active_errors()
        initial_active_chi2 = float(optimizer.active_chi2())

        if use_robust_kernel:
            optimizer.optimize(5)

            for edge, (
                point,
                kf,
                idx,
                chi2_threshold,
                is_stereo,
                point_vertex,
                pose_vertex,
            ) in graph_edges.items():
                chi2 = _manual_reprojection_chi2(kf, idx, pose_vertex, point_vertex, is_stereo)
                is_bad_edge = (
                    (not np.isfinite(chi2))
                    or chi2 > float(chi2_threshold)
                    or not _is_depth_positive(edge, pose_vertex, point_vertex)
                )
                if is_bad_edge:
                    edge.set_level(1)
                    num_bad_edges += 1
                edge.set_robust_kernel(None)

            if _abort_requested(abort_flag):
                return OptimizerResult(
                    len(edges),
                    0,
                    len(edges),
                    float("inf"),
                    False,
                    mean_error_before=initial_active_chi2 / max(len(edges), 1),
                    num_keyframes=len(pose_vertices),
                    num_map_points=len(point_vertices),
                    aborted=True,
                    reason="aborted after robust optimization",
                )

            optimizer.initialize_optimization()
            optimizer.optimize(int(rounds))
        else:
            optimizer.initialize_optimization()
            optimizer.optimize(int(rounds))
    except Exception as exc:
        print(f"bundle_adjustment: g2o failed: {exc}")
        return OptimizerResult(
            len(edges),
            0,
            len(edges),
            float("inf"),
            False,
            num_keyframes=len(pose_vertices),
            num_map_points=len(point_vertices),
            reason=f"g2o failed: {exc}",
        )

    _t6 = time.perf_counter() if _BA_PROFILE else 0.0

    outlier_observations = []
    inlier_chi2 = []

    for edge, point, kf, idx, chi2_threshold, is_stereo, point_vertex, pose_vertex in edges:
        if _is_bad_point(point) or _is_bad_keyframe(kf):
            continue
        if idx < 0 or idx >= len(getattr(kf, "points", [])):
            continue
        if kf.get_point_match(idx) is not point:
            continue
        chi2 = _manual_reprojection_chi2(kf, idx, pose_vertex, point_vertex, is_stereo)
        is_bad_observation = (
            (not np.isfinite(chi2))
            or chi2 > float(chi2_threshold)
            or not _is_depth_positive(edge, pose_vertex, point_vertex)
        )
        if is_bad_observation:
            outlier_observations.append((point, kf, idx, is_stereo))
        else:
            inlier_chi2.append(chi2)

    pose_updates = {}
    point_updates = {}
    for kf, vertex in pose_vertices.items():
        pose_updates[kf] = _extract_Tcw_from_pose_vertex(vertex)
    if not fixed_points:
        for point, vertex in point_vertices.items():
            if point not in points_with_edges:
                continue
            point_updates[point] = np.array(vertex.estimate(), dtype=np.float64, copy=True).reshape(3)

    if write_back:
        lock_context = map_lock if map_lock is not None else nullcontext()
        with lock_context:
            if prune_outliers:
                for point, kf, idx, _ in outlier_observations:
                    if _is_bad_point(point) or _is_bad_keyframe(kf):
                        continue
                    if idx < 0 or idx >= len(getattr(kf, "points", [])):
                        continue
                    if kf.get_point_match(idx) is point:
                        point.remove_observation(kf, idx, map_no_lock=True)

            for kf, Tcw in pose_updates.items():
                if kf in local_keyframes and not _is_bad_keyframe(kf):
                    kf.update_pose(g2o.Isometry3d(Tcw))
                    if hasattr(kf, "lba_count"):
                        kf.lba_count += 1

            if not fixed_points:
                for point, position in point_updates.items():
                    if _is_bad_point(point):
                        continue
                    point.update_position(position)
                    point.update_normal_and_depth()

    if result_dict is not None:
        result_dict["keyframes"] = {kf.kid: Tcw.copy() for kf, Tcw in pose_updates.items()}
        result_dict["keyframe_updates"] = {
            getattr(kf, "id", kf.kid): Tcw.copy() for kf, Tcw in pose_updates.items()
        }
        result_dict["points"] = {p.id: position.copy() for p, position in point_updates.items()}
        result_dict["point_updates"] = {p.id: position.copy() for p, position in point_updates.items()}
        result_dict["fixed_keyframes"] = [kf.kid for kf, vertex in pose_vertices.items() if vertex.fixed()]
        result_dict["num_edges"] = len(edges)
        result_dict["num_inliers"] = len(inlier_chi2)
        result_dict["num_outliers"] = len(outlier_observations)
        result_dict["mean_error_before"] = initial_active_chi2 / max(len(edges), 1)
        result_dict["mean_error_after"] = float(np.mean(inlier_chi2)) if inlier_chi2 else float("inf")

    _t7 = time.perf_counter() if _BA_PROFILE else 0.0

    if _BA_PROFILE:
        nkf = len(local_keyframes)
        nfkf = len(fixed_keyframes)
        npt = len(point_vertices)
        ned = len(edges)
        print(
            f"[BA_PROFILE] kfs={nkf}+{nfkf}fix pts={npt} edges={ned} | "
            f"setup={_t1-_t0:.3f}s pose_v={_t2-_t1:.3f}s pt_v={_t3-_t2:.3f}s "
            f"edges={_t4-_t3:.3f}s abort_chk={_t5-_t4:.3f}s "
            f"optimize={_t6-_t5:.3f}s extract={_t7-_t6:.3f}s "
            f"total={_t7-_t0:.3f}s",
            flush=True,
        )

    mse = float(np.mean(inlier_chi2)) if inlier_chi2 else float("inf")

    return OptimizerResult(
        num_edges=len(edges),
        num_inliers=len(inlier_chi2),
        num_outliers=len(outlier_observations),
        mean_squared_error=mse,
        success=len(inlier_chi2) > 0 and np.isfinite(initial_active_chi2),
        mean_error_before=initial_active_chi2 / max(len(edges), 1),
        mean_error_after=mse,
        num_keyframes=len(pose_vertices),
        num_map_points=len(point_vertices),
        reason="" if len(inlier_chi2) > 0 and np.isfinite(initial_active_chi2) else "no inlier edges",
    )


def bundle_adjustment(
    keyframes,
    points,
    local_window_size=None,
    fixed_points: bool = False,
    rounds: int = 10,
    loop_kf_id: int = 0,
    use_robust_kernel: bool = False,
    abort_flag=None,
    mp_abort_flag=None,
    result_dict: Optional[dict] = None,
    write_back: bool = True,
    verbose: bool = False,
    print=print,
) -> tuple[float, Optional[dict]]:
    """

    Returns:
        (mean_squared_error, result_dict)
    """
    keyframes = _as_list(keyframes)
    points = _as_list(points)

    if local_window_size is None:
        local_keyframes = keyframes
    else:
        local_keyframes = keyframes[-int(local_window_size):]

    # Gauge fixing: keep the first keyframe fixed. This mirrors the role of
    # fixed boundary keyframes in local BA and root fixation in global BA.
    fixed_keyframes = []
    if len(local_keyframes) > 0 and local_keyframes[0].kid != 0:
        fixed_keyframes.append(local_keyframes[0])

    result = _bundle_adjustment_core(
        local_keyframes=local_keyframes,
        fixed_keyframes=fixed_keyframes,
        points=points,
        fixed_points=fixed_points,
        rounds=rounds,
        use_robust_kernel=use_robust_kernel,
        abort_flag=abort_flag,
        map_lock=None,
        verbose=verbose,
        result_dict=result_dict,
        write_back=write_back,
        prune_outliers=False,
        print=print,
    )

    return result.mean_squared_error, result_dict


def local_bundle_adjustment(
    keyframes,
    points: Optional[Iterable[MapPoint]] = None,
    keyframes_ref: Optional[Iterable[KeyFrame]] = None,
    fixed_points: bool = False,
    verbose: bool = False,
    rounds: int = 10,
    abort_flag=None,
    mp_abort_flag=None,
    map_lock=None,
    print=print,
) -> OptimizerResult:
    """

        local_bundle_adjustment(keyframes, points, keyframes_ref, ...)

    Compatibility call retained for existing tests:
        local_bundle_adjustment(reference_keyframe, ...)
    """
    if isinstance(keyframes, KeyFrame):
        keyframe = keyframes
        if keyframe is None or keyframe.is_bad():
            return OptimizerResult(0, 0, 0, float("inf"), False)

        local_keyframes = [keyframe]

        for kf in keyframe.get_covisible_keyframes():
            if kf is not None and not kf.is_bad() and kf not in local_keyframes:
                local_keyframes.append(kf)

        local_points = []
        for kf in local_keyframes:
            for point in kf.get_matched_good_points():
                if point is not None and not point.is_bad() and point not in local_points:
                    local_points.append(point)

        fixed_keyframes = []
        for point in local_points:
            for observing_kf, _ in point.observations():
                if observing_kf not in local_keyframes and observing_kf not in fixed_keyframes:
                    if not observing_kf.is_bad():
                        fixed_keyframes.append(observing_kf)
    else:
        local_keyframes = _as_list(keyframes)
        local_points = _as_list(points)
        fixed_keyframes = _as_list(keyframes_ref)

    if len(local_keyframes) == 0 or len(local_points) == 0:
        return OptimizerResult(0, 0, 0, float("inf"), False)

    # If there are no boundary fixed keyframes and kid=0 is not already local,
    # fix the first local keyframe to remove gauge freedom.
    # Skip if any local KF has kid==0 — pack_local_ba already fixes it via the kid==0 rule.
    if not fixed_keyframes and local_keyframes:
        if not any(kf.kid == 0 for kf in local_keyframes):
            fixed_keyframes.append(local_keyframes[0])

    return _bundle_adjustment_core(
        local_keyframes=local_keyframes,
        fixed_keyframes=fixed_keyframes,
        points=local_points,
        fixed_points=fixed_points,
        rounds=int(rounds),
        use_robust_kernel=True,
        abort_flag=abort_flag,
        map_lock=map_lock,
        verbose=verbose,
        result_dict=None,
        write_back=True,
        prune_outliers=True,
        print=print,
    )


def optimize_sim3(
    kf1: KeyFrame,
    kf2: KeyFrame,
    map_points1,
    map_point_matches12,
    R12: np.ndarray,
    t12: np.ndarray,
    s12: float,
    th2: float,
    fix_scale: bool,
    verbose: bool = False,
):
    """Optimize a Sim3 transformation between two keyframes using g2o.

    Mirrors pyslam optimizer_g2o.py:optimize_sim3 (lines 1226-1417) exactly.
    map_point_matches12[i] = map point of kf2 matched with i-th map point of kf1.
    Returns (num_inliers, R12, t12, s12, delta_err).
    """
    R12 = np.asarray(R12, dtype=np.float64)
    t12 = np.asarray(t12, dtype=np.float64).ravel()
    s12 = float(s12)
    th2 = float(th2)

    cam1 = kf1.camera
    cam2 = kf2.camera
    kf1_Tcw = np.asarray(kf1.Tcw(), dtype=np.float64).reshape(4, 4)
    kf2_Tcw = np.asarray(kf2.Tcw(), dtype=np.float64).reshape(4, 4)
    R1w, t1w = kf1_Tcw[:3, :3], kf1_Tcw[:3, 3]
    R2w, t2w = kf2_Tcw[:3, :3], kf2_Tcw[:3, 3]

    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    sim3 = g2o.Sim3(R12.copy(), t12.copy(), s12)
    sim3_vertex = g2o.VertexSim3Expmap()
    sim3_vertex.set_estimate(sim3)
    sim3_vertex.set_id(0)
    sim3_vertex.set_fixed(False)
    sim3_vertex._fix_scale = fix_scale
    sim3_vertex._principle_point1 = np.array([cam1.cx, cam1.cy])
    sim3_vertex._focal_length1 = np.array([cam1.fx, cam1.fy])
    sim3_vertex._principle_point2 = np.array([cam2.cx, cam2.cy])
    sim3_vertex._focal_length2 = np.array([cam2.fx, cam2.fy])
    optimizer.add_vertex(sim3_vertex)

    if map_points1 is None:
        map_points1 = kf1.get_points()
    num_matches = len(map_point_matches12)
    assert num_matches == len(map_points1)

    edges_12 = []
    edges_21 = []
    vertex_indices = []
    delta_huber = float(np.sqrt(th2))
    inv_level_sigmas2 = (
        FeatureTrackerShared.feature_manager.inv_level_sigmas2
        if FeatureTrackerShared.feature_manager is not None
        else np.ones(8)
    )
    eye2 = np.eye(2)

    num_correspondences = 0
    for i in range(num_matches):
        mp1 = map_points1[i]
        if mp1 is None or mp1.is_bad():
            continue
        mp2 = map_point_matches12[i]
        if mp2 is None or mp2.is_bad():
            continue

        vertex_id1 = 2 * i + 1
        vertex_id2 = 2 * (i + 1)
        index2 = mp2.get_observation_idx(kf2)
        if index2 < 0:
            continue

        v_mp1 = g2o.VertexSBAPointXYZ()
        v_mp1.set_estimate(R1w @ mp1.pt() + t1w)
        v_mp1.set_id(vertex_id1)
        v_mp1.set_fixed(True)
        optimizer.add_vertex(v_mp1)

        v_mp2 = g2o.VertexSBAPointXYZ()
        v_mp2.set_estimate(R2w @ mp2.pt() + t2w)
        v_mp2.set_id(vertex_id2)
        v_mp2.set_fixed(True)
        optimizer.add_vertex(v_mp2)

        kpsu_i = np.array(kf1.kpsu[i].pt if hasattr(kf1.kpsu[i], "pt") else kf1.kpsu[i], dtype=np.float64)
        edge_12 = g2o.EdgeSim3ProjectXYZ()
        edge_12.set_vertex(0, optimizer.vertex(vertex_id2))
        edge_12.set_vertex(1, optimizer.vertex(0))
        edge_12.set_measurement(kpsu_i)
        level_i = int(kf1.octaves[i]) if kf1.octaves is not None else 0
        invSigma2_12 = float(inv_level_sigmas2[min(level_i, len(inv_level_sigmas2) - 1)])
        edge_12.set_information(eye2 * invSigma2_12)
        edge_12.set_robust_kernel(g2o.RobustKernelHuber(delta_huber))
        optimizer.add_edge(edge_12)

        kpsu_j = np.array(kf2.kpsu[index2].pt if hasattr(kf2.kpsu[index2], "pt") else kf2.kpsu[index2], dtype=np.float64)
        edge_21 = g2o.EdgeInverseSim3ProjectXYZ()
        edge_21.set_vertex(0, optimizer.vertex(vertex_id1))
        edge_21.set_vertex(1, optimizer.vertex(0))
        edge_21.set_measurement(kpsu_j)
        level_j = int(kf2.octaves[index2]) if kf2.octaves is not None else 0
        invSigma2_21 = float(inv_level_sigmas2[min(level_j, len(inv_level_sigmas2) - 1)])
        edge_21.set_information(eye2 * invSigma2_21)
        edge_21.set_robust_kernel(g2o.RobustKernelHuber(delta_huber))
        optimizer.add_edge(edge_21)

        edges_12.append(edge_12)
        edges_21.append(edge_21)
        vertex_indices.append(i)
        num_correspondences += 1

    if num_correspondences < 10:
        return 0, None, None, None, 0.0

    optimizer.initialize_optimization()
    if verbose:
        optimizer.set_verbose(True)
    optimizer.optimize(5)
    err = optimizer.active_chi2()

    # First inlier check
    num_bad = 0
    for i, (e12, e21) in enumerate(zip(edges_12, edges_21)):
        if (
            e12.chi2() > th2
            or not e12.is_depth_positive()
            or e21.chi2() > th2
            or not e21.is_depth_positive()
        ):
            idx = vertex_indices[i]
            map_points1[idx] = None
            optimizer.remove_edge(e12)
            optimizer.remove_edge(e21)
            edges_12[i] = None
            edges_21[i] = None
            num_bad += 1

    num_more_iterations = 10 if num_bad > 0 else 5
    if num_correspondences - num_bad < 10:
        return 0, None, None, None, 0.0

    optimizer.initialize_optimization()
    optimizer.optimize(num_more_iterations)
    delta_err = optimizer.active_chi2() - err

    num_inliers = 0
    for i, (e12, e21) in enumerate(zip(edges_12, edges_21)):
        if e12 is None or e21 is None:
            continue
        if (
            e12.chi2() > th2
            or not e12.is_depth_positive()
            or e21.chi2() > th2
            or not e21.is_depth_positive()
        ):
            map_points1[vertex_indices[i]] = None
        else:
            num_inliers += 1

    sim3_recov = optimizer.vertex(0).estimate()
    return (
        num_inliers,
        np.asarray(sim3_recov.rotation().matrix(), dtype=np.float64),
        np.asarray(sim3_recov.translation(), dtype=np.float64).ravel(),
        float(sim3_recov.scale()),
        float(delta_err),
    )


def global_bundle_adjustment(
    keyframes,
    points,
    rounds: int = 10,
    loop_kf_id: int = 0,
    use_robust_kernel: bool = True,
    abort_flag=None,
    mp_abort_flag=None,
    result_dict: Optional[dict] = None,
    write_back: bool = True,
    verbose: bool = False,
    print=print,
) -> tuple[float, Optional[dict]]:
    return bundle_adjustment(
        keyframes=keyframes,
        points=points,
        local_window_size=None,
        fixed_points=False,
        rounds=rounds,
        loop_kf_id=loop_kf_id,
        use_robust_kernel=use_robust_kernel,
        abort_flag=abort_flag,
        mp_abort_flag=mp_abort_flag,
        result_dict=result_dict,
        write_back=write_back,
        verbose=verbose,
        print=print,
    )
