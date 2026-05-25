"""
slam_optimizer_bridge — pack/unpack Python map objects for slam_optimizer_core (C++ BA).

Converts our Python KeyFrame / MapPoint graph into flat numpy arrays accepted by
slam_optimizer_core.run_local_ba() and slam_optimizer_core.run_global_ba(), then
writes the optimized poses and positions back to the live map.
"""
from __future__ import annotations

import numpy as np
from contextlib import nullcontext
from typing import Iterable

from visual_slam.orbslam.slam.map_point import MapPoint
from visual_slam.orbslam.slam.keyframe import KeyFrame


def _get_inv_sigma2(kf: KeyFrame, idx: int, feature_manager) -> float:
    """inv_sigma2 for octave level at idx — matches get_inv_level_sigma2 in optimizer_g2o.py."""
    if feature_manager is None:
        return 1.0
    octaves = getattr(kf, "octaves", None)
    if octaves is None or idx >= len(octaves):
        return 1.0
    oct_level = int(octaves[idx])
    oct_level = min(oct_level, len(feature_manager.inv_level_sigmas2) - 1)
    return float(feature_manager.inv_level_sigmas2[oct_level])


def _get_ur(kf: KeyFrame, idx: int) -> float:
    kps_ur = getattr(kf, "kps_ur", None)
    if kps_ur is None:
        kps_ur = getattr(kf, "uRs", None)
    if kps_ur is None or idx >= len(kps_ur):
        return -1.0
    return float(kps_ur[idx])


def _is_bad_kf(kf) -> bool:
    return kf is None or getattr(kf, "_is_bad", False) or getattr(kf, "to_be_erased", False)


def _is_bad_pt(p) -> bool:
    return p is None or getattr(p, "_is_bad", False)


def pack_local_ba(
    local_keyframes: list[KeyFrame],
    fixed_keyframes: list[KeyFrame],
    points: list[MapPoint],
    feature_manager,
) -> tuple:
    """
    Build numpy arrays for slam_optimizer_core.run_local_ba().

    Returns:
        kf_poses      (N, 16) float64
        kf_ids        (N,)    int64
        kf_fixed      (N,)    uint8
        point_pos     (M, 3)  float64
        observations  (K, 8)  float64  [kf_row, pt_row, u, v, ur, octave, inv_sigma2, is_stereo]
        camera        (5,)    float64  [fx, fy, cx, cy, bf]
        kf_list       list[KeyFrame]   ordered KF list (index = row in kf_poses)
        pt_list       list[MapPoint]   ordered point list (index = row in point_pos)
        obs_triples   list[(point, kf, idx)]   one per observation row, for unpack
    """
    # Deduplicated ordered KF list: local first, then fixed boundary
    kf_list: list[KeyFrame] = []
    seen_kf: set = set()
    for kf in list(local_keyframes) + list(fixed_keyframes):
        if kf is not None and id(kf) not in seen_kf and not _is_bad_kf(kf):
            kf_list.append(kf)
            seen_kf.add(id(kf))

    fixed_ids = {id(kf) for kf in fixed_keyframes}
    local_ids = {id(kf) for kf in local_keyframes}

    kf_index = {id(kf): i for i, kf in enumerate(kf_list)}
    N = len(kf_list)

    kf_poses = np.zeros((N, 16), dtype=np.float64)
    kf_ids_arr = np.zeros(N, dtype=np.int64)
    kf_fixed_arr = np.zeros(N, dtype=np.uint8)

    for i, kf in enumerate(kf_list):
        T = kf.Tcw().flatten()  # row-major 4×4
        kf_poses[i] = T
        kf_ids_arr[i] = int(kf.kid)
        # Fixed if in boundary set OR it's KF 0 (gauge fix)
        kf_fixed_arr[i] = 1 if (id(kf) in fixed_ids or kf.kid == 0) else 0

    # Deduplicated point list
    pt_list = [p for p in points if not _is_bad_pt(p)]
    pt_index = {id(p): i for i, p in enumerate(pt_list)}
    M = len(pt_list)

    point_pos = np.array([p.get_position() for p in pt_list], dtype=np.float64)
    if point_pos.ndim == 1:
        point_pos = point_pos.reshape(0, 3) if M == 0 else point_pos.reshape(M, 3)

    # Observations
    obs_rows = []
    obs_triples = []  # (point, kf, idx) for unpack

    # Pick camera from first local KF
    ref_kf = kf_list[0] if kf_list else None
    cam = getattr(ref_kf, "camera", None) if ref_kf else None
    fx = float(getattr(cam, "fx", 0.0)) if cam else 0.0
    fy = float(getattr(cam, "fy", 0.0)) if cam else 0.0
    cx = float(getattr(cam, "cx", 0.0)) if cam else 0.0
    cy = float(getattr(cam, "cy", 0.0)) if cam else 0.0
    bf = float(getattr(cam, "bf", 0.0)) if cam else 0.0
    camera_arr = np.array([fx, fy, cx, cy, bf], dtype=np.float64)

    for p in pt_list:
        pt_row = pt_index[id(p)]
        pt_pos = p.get_position()
        if not np.all(np.isfinite(pt_pos)):
            continue

        for kf, idx in p.observations():
            if _is_bad_kf(kf):
                continue
            kf_row = kf_index.get(id(kf))
            if kf_row is None:
                continue
            if idx < 0 or idx >= len(getattr(kf, "points", [])):
                continue
            if kf.get_point_match(idx) is not p:
                continue

            kpsu = getattr(kf, "kpsu", None)
            if kpsu is None or idx >= len(kpsu):
                continue
            kp = kpsu[idx]
            if hasattr(kp, 'pt'):  # cv2.KeyPoint
                u, v = float(kp.pt[0]), float(kp.pt[1])
            else:
                u, v = float(kp[0]), float(kp[1])
            if not (np.isfinite(u) and np.isfinite(v)):
                continue

            ur = _get_ur(kf, idx)
            inv_s2 = _get_inv_sigma2(kf, idx, feature_manager)
            octave = int(getattr(kf, "octaves", [0] * (idx + 1))[idx])
            is_stereo = 1.0 if ur >= 0.0 else 0.0

            obs_rows.append([
                float(kf_row), float(pt_row),
                u, v, ur,
                float(octave), inv_s2, is_stereo,
            ])
            obs_triples.append((p, kf, idx))

    if obs_rows:
        observations = np.array(obs_rows, dtype=np.float64)
    else:
        observations = np.empty((0, 8), dtype=np.float64)

    return (
        kf_poses, kf_ids_arr, kf_fixed_arr,
        point_pos, observations, camera_arr,
        kf_list, pt_list, obs_triples,
    )


def pack_pose_optimization(frame, feature_manager) -> tuple:
    """
    Pack frame observations for slam_optimizer_core.run_pose_optimization().

    Returns:
        frame_pose    (16,)  float64   row-major Tcw
        observations  (K, 8) float64   [u, v, ur, inv_sigma2, is_stereo, px, py, pz]
        camera        (5,)   float64   [fx, fy, cx, cy, bf]
        valid_indices list[int]        frame.points indices that produced rows
    """
    cam = getattr(frame, "camera", None)
    fx = float(getattr(cam, "fx", 0.0)) if cam else 0.0
    fy = float(getattr(cam, "fy", 0.0)) if cam else 0.0
    cx_v = float(getattr(cam, "cx", 0.0)) if cam else 0.0
    cy_v = float(getattr(cam, "cy", 0.0)) if cam else 0.0
    bf   = float(getattr(cam, "bf", 0.0)) if cam else 0.0
    camera_arr = np.array([fx, fy, cx_v, cy_v, bf], dtype=np.float64)

    frame_pose = frame.Tcw().flatten().astype(np.float64)

    points = list(getattr(frame, "points", []))
    kpsu   = getattr(frame, "kpsu", None)
    kps_ur = getattr(frame, "kps_ur", None)
    if kps_ur is None:
        kps_ur = getattr(frame, "uRs", None)

    obs_rows = []
    valid_indices = []

    for idx, p in enumerate(points):
        if _is_bad_pt(p):
            continue
        pt_w = p.get_position()
        if not np.all(np.isfinite(pt_w)):
            continue
        if kpsu is None or idx >= len(kpsu):
            continue
        kp = kpsu[idx]
        if hasattr(kp, 'pt'):  # cv2.KeyPoint
            u, v = float(kp.pt[0]), float(kp.pt[1])
        else:
            u, v = float(kp[0]), float(kp[1])
        if not (np.isfinite(u) and np.isfinite(v)):
            continue

        ur = -1.0
        if kps_ur is not None and idx < len(kps_ur):
            ur = float(kps_ur[idx])
        is_stereo = 1.0 if ur >= 0.0 else 0.0

        inv_s2 = _get_inv_sigma2(frame, idx, feature_manager)

        obs_rows.append([u, v, ur, inv_s2, is_stereo,
                         float(pt_w[0]), float(pt_w[1]), float(pt_w[2])])
        valid_indices.append(idx)

    if obs_rows:
        observations = np.array(obs_rows, dtype=np.float64)
    else:
        observations = np.empty((0, 8), dtype=np.float64)

    return frame_pose, observations, camera_arr, valid_indices


def unpack_pose_optimization(result: dict, frame, valid_indices: list) -> None:
    """
    Write updated pose back to frame and update outlier flags.
    Mirrors the write-back in pose_optimization() in optimizer_g2o.py.
    """
    import g2o
    updated_pose = result["updated_pose"].reshape(4, 4)
    if np.all(np.isfinite(updated_pose)):
        frame.update_pose(g2o.Isometry3d(updated_pose))

    outlier_mask = result["outlier_mask"]
    if not hasattr(frame, "outliers") or len(frame.outliers) != len(getattr(frame, "points", [])):
        frame.outliers = np.zeros(len(getattr(frame, "points", [])), dtype=bool)

    for k, idx in enumerate(valid_indices):
        if k < len(outlier_mask) and idx < len(frame.outliers):
            frame.outliers[idx] = bool(outlier_mask[k])


def unpack_local_ba(
    result: dict,
    kf_list: list[KeyFrame],
    pt_list: list[MapPoint],
    obs_triples: list,
    local_keyframes: list[KeyFrame],
    fixed_keyframes: list[KeyFrame],
    fixed_points: bool,
    prune_outliers: bool,
    map_lock,
) -> None:
    """
    Write optimized poses and positions back to the live map.
    Mirrors the write-back block in _bundle_adjustment_core().
    """
    import g2o

    updated_poses = result["updated_poses"]   # (N, 16)
    updated_points = result["updated_points"] # (M, 3)
    outlier_mask = result["outlier_mask"]     # (K,)

    local_ids = {id(kf) for kf in local_keyframes}
    fixed_ids = {id(kf) for kf in fixed_keyframes}

    # Collect updates before acquiring map lock
    pose_updates: dict = {}
    for i, kf in enumerate(kf_list):
        if id(kf) not in local_ids:
            continue
        if _is_bad_kf(kf):
            continue
        T = updated_poses[i].reshape(4, 4)
        if not np.all(np.isfinite(T)):
            continue
        pose_updates[kf] = T

    point_updates: list[tuple] = []
    if not fixed_points:
        for j, p in enumerate(pt_list):
            if _is_bad_pt(p):
                continue
            pos = updated_points[j]
            if not np.all(np.isfinite(pos)):
                continue
            point_updates.append((p, pos))

    outlier_obs: list[tuple] = []
    if prune_outliers:
        for k, (p, kf, idx) in enumerate(obs_triples):
            if k < len(outlier_mask) and outlier_mask[k]:
                outlier_obs.append((p, kf, idx))

    # Write back under map lock
    lock_ctx = map_lock if map_lock is not None else nullcontext()
    with lock_ctx:
        if prune_outliers:
            for p, kf, idx in outlier_obs:
                if _is_bad_pt(p) or _is_bad_kf(kf):
                    continue
                if idx < 0 or idx >= len(getattr(kf, "points", [])):
                    continue
                if kf.get_point_match(idx) is p:
                    p.remove_observation(kf, idx, map_no_lock=True)

        for kf, T in pose_updates.items():
            if not _is_bad_kf(kf):
                kf.update_pose(g2o.Isometry3d(T))
                if hasattr(kf, "lba_count"):
                    kf.lba_count += 1

        for p, pos in point_updates:
            if not _is_bad_pt(p):
                p.update_position(pos)
                p.update_normal_and_depth()
