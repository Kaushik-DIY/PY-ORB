"""
Geometric matching routines for tracking, mapping, and loop correction.
This module implements projection search, epipolar matching, and map-point fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.frame import (
    Frame,
    are_map_points_visible_in_frame,
    ensure_frame_feature_arrays,
)
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.map_point import MapPoint
from visual_slam.orbslam.slam.rotation_histogram import RotationHistogram
from visual_slam.orbslam.utilities.geom_2views import computeF12, check_dist_epipolar_line


kCheckFeaturesOrientation = Parameters.kCheckFeaturesOrientation


# Group projection-based matching routines used across the pipeline.
class ProjectionMatcher:
    @staticmethod
    def search_frame_by_projection(*args, **kwargs):
        return _search_frame_by_projection(*args, **kwargs)

    @staticmethod
    def search_keyframe_by_projection(*args, **kwargs):
        return _search_keyframe_by_projection(*args, **kwargs)

    @staticmethod
    def search_map_by_projection(*args, **kwargs):
        return _search_map_by_projection(*args, **kwargs)

    @staticmethod
    def search_local_frames_by_projection(*args, **kwargs):
        return _search_local_frames_by_projection(*args, **kwargs)

    @staticmethod
    def search_all_map_by_projection(*args, **kwargs):
        return _search_all_map_by_projection(*args, **kwargs)

    @staticmethod
    def search_more_map_points_by_projection(*args, **kwargs):
        return _search_more_map_points_by_projection(*args, **kwargs)

    @staticmethod
    def search_and_fuse(*args, **kwargs):
        return _search_and_fuse(*args, **kwargs)

    @staticmethod
    def search_and_fuse_for_loop_correction(*args, **kwargs):
        return _search_and_fuse_for_loop_correction(*args, **kwargs)

    @staticmethod
    def search_by_sim3(*args, **kwargs):
        return _search_by_sim3(*args, **kwargs)


def _are_map_points_visible_sim3(frame1, frame2, map_points, sR21: np.ndarray, t21: np.ndarray):
    """Project map_points (world frame, observed by frame1) onto frame2 using Sim3 transform.

    Mirrors pyslam frame.py:are_map_points_visible(frame1, frame2, map_points1, sR21, t21).
    sR21 = s21 * R21 (already scaled), t21 = translation from cam1 to cam2.
    Returns (visible_flags, uvs_2, zs_2, dists_2).
    """
    n = len(map_points)
    if n == 0:
        return (
            np.array([], dtype=bool),
            np.empty((0, 2), dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
    points_w = np.array([mp.get_position() for mp in map_points], dtype=np.float64)
    min_dists = np.array([mp.get_min_distance_invariance() for mp in map_points], dtype=np.float64)
    max_dists = np.array([mp.get_max_distance_invariance() for mp in map_points], dtype=np.float64)

    # Transform to camera1 then Sim3 to camera2
    points_c1 = frame1.transform_points(points_w)  # Nx3
    sR21 = np.asarray(sR21, dtype=np.float64)
    t21 = np.asarray(t21, dtype=np.float64).reshape(3)
    points_c2 = (sR21 @ points_c1.T).T + t21.reshape(1, 3)  # Nx3

    uvs_2, zs_2 = frame2.camera.project(points_c2)
    uvs_2 = np.asarray(uvs_2, dtype=np.float64).reshape(-1, 2)
    zs_2 = np.asarray(zs_2, dtype=np.float64).reshape(-1)

    are_in_image = frame2.are_in_image(uvs_2, zs_2)
    dists_2 = np.linalg.norm(points_c2, axis=1)
    are_in_good_dist = (dists_2 > min_dists) & (dists_2 < max_dists)
    out_flags = are_in_image & are_in_good_dist
    return out_flags, uvs_2, zs_2, dists_2


def _search_by_sim3(
    kf1: KeyFrame,
    kf2: KeyFrame,
    idxs1,
    idxs2,
    s12: float,
    R12: np.ndarray,
    t12: np.ndarray,
    max_reproj_distance: float = Parameters.kMaxReprojectionDistanceSim3,
    max_descriptor_distance=None,
    print_fun=None,
):
    """Bidirectional guided Sim3 matching between two keyframes.

    Mirrors pyslam geometry_matchers.py:_search_by_sim3 (lines 946-1106).
    Returns (num_matches_found, new_matches12, new_matches21).
    new_matches12[i] = index of kf2 map point matched to i-th kf1 map point (-1 if none).
    new_matches21[i] = index of kf1 map point matched to i-th kf2 map point (-1 if none).
    """
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance

    R12 = np.asarray(R12, dtype=np.float64)
    t12 = np.asarray(t12, dtype=np.float64).reshape(3)
    s12 = float(s12)

    # Sim3 both directions
    sR12 = s12 * R12
    sR21 = (1.0 / s12) * R12.T
    t21 = -sR21 @ t12

    map_points1 = kf1.get_points()
    n1 = len(map_points1)
    new_matches12 = np.full(n1, -1, dtype=np.int32)
    good_points1 = np.array(
        [True if mp is not None and not mp.is_bad() else False for mp in map_points1]
    )

    map_points2 = kf2.get_points()
    n2 = len(map_points2)
    new_matches21 = np.full(n2, -1, dtype=np.int32)
    good_points2 = np.array(
        [True if mp is not None and not mp.is_bad() else False for mp in map_points2]
    )

    # Seed with input inlier matches
    for idx1, idx2 in zip(idxs1, idxs2):
        idx1, idx2 = int(idx1), int(idx2)
        if good_points1[idx1] and good_points2[idx2]:
            new_matches12[idx1] = idx2
            new_matches21[idx2] = idx1

    map_points1_arr = np.asarray(map_points1, dtype=object)
    map_points2_arr = np.asarray(map_points2, dtype=object)

    # Unmatched map points of kf1 → check visibility on kf2 via inverse Sim3
    unmatched_idxs1 = np.array(
        [i for i in range(n1) if good_points1[i] and new_matches12[i] < 0], dtype=np.int32
    )
    if len(unmatched_idxs1) > 0:
        unmatched_mp1 = map_points1_arr[unmatched_idxs1]
        visible_21, projs_21, _, dists_21 = _are_map_points_visible_sim3(
            kf1, kf2, unmatched_mp1, sR21, t21
        )
        if np.any(visible_21):
            scale_factors = FeatureTrackerShared.feature_manager.scale_factors if FeatureTrackerShared.feature_manager is not None else np.ones(8)
            predicted_levels = MapPoint.predict_detection_levels(unmatched_mp1, dists_21)
            kp_scale_factors = np.asarray(scale_factors, dtype=np.float64)[
                np.clip(predicted_levels, 0, len(scale_factors) - 1)
            ]
            radiuses = max_reproj_distance * kp_scale_factors
            kd2_idxs = kf2.kd.query_ball_point(projs_21[:, :2], radiuses)

            for j, (mp1, vis) in enumerate(zip(unmatched_mp1, visible_21)):
                if not vis:
                    continue
                kd2_idxs_j = kd2_idxs[j]
                predicted_level = int(predicted_levels[j])
                best_dist = np.inf
                best_idx = -1
                for kd2_idx in kd2_idxs_j:
                    kp_level = int(kf2.octaves[kd2_idx])
                    if kp_level < predicted_level - 1 or kp_level > predicted_level:
                        continue
                    dist = mp1.min_des_distance(kf2.des[kd2_idx])
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = kd2_idx
                if best_dist <= max_descriptor_distance and best_idx >= 0:
                    if new_matches21[best_idx] == -1:
                        new_matches12[unmatched_idxs1[j]] = best_idx

    # Unmatched map points of kf2 → check visibility on kf1 via forward Sim3
    unmatched_idxs2 = np.array(
        [i for i in range(n2) if good_points2[i] and new_matches21[i] < 0], dtype=np.int32
    )
    if len(unmatched_idxs2) > 0:
        unmatched_mp2 = map_points2_arr[unmatched_idxs2]
        visible_12, projs_12, _, dists_12 = _are_map_points_visible_sim3(
            kf2, kf1, unmatched_mp2, sR12, t12
        )
        if np.any(visible_12):
            scale_factors = FeatureTrackerShared.feature_manager.scale_factors if FeatureTrackerShared.feature_manager is not None else np.ones(8)
            predicted_levels = MapPoint.predict_detection_levels(unmatched_mp2, dists_12)
            kp_scale_factors = np.asarray(scale_factors, dtype=np.float64)[
                np.clip(predicted_levels, 0, len(scale_factors) - 1)
            ]
            radiuses = max_reproj_distance * kp_scale_factors
            kd1_idxs = kf1.kd.query_ball_point(projs_12[:, :2], radiuses)

            for j, (mp2, vis) in enumerate(zip(unmatched_mp2, visible_12)):
                if not vis:
                    continue
                kd1_idxs_j = kd1_idxs[j]
                predicted_level = int(predicted_levels[j])
                best_dist = np.inf
                best_idx = -1
                for kd1_idx in kd1_idxs_j:
                    kp_level = int(kf1.octaves[kd1_idx])
                    if kp_level < predicted_level - 1 or kp_level > predicted_level:
                        continue
                    dist = mp2.min_des_distance(kf1.des[kd1_idx])
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = kd1_idx
                if best_dist <= max_descriptor_distance and best_idx >= 0:
                    if new_matches12[best_idx] == -1:
                        new_matches21[unmatched_idxs2[j]] = best_idx

    # Cross-check: only keep mutually consistent matches
    num_matches_found = 0
    for i1 in range(n1):
        idx2 = int(new_matches12[i1])
        if idx2 >= 0:
            idx1 = int(new_matches21[idx2])
            if idx1 != i1:
                new_matches12[i1] = -1
                new_matches21[idx2] = -1
            else:
                num_matches_found += 1

    return num_matches_found, new_matches12, new_matches21


# Match feature pairs under epipolar constraints for triangulation.
class EpipolarMatcher:
    @staticmethod
    def search_frame_for_triangulation(
        f1,
        f2,
        idxs1=None,
        idxs2=None,
        max_descriptor_distance=None,
        is_monocular=True,
    ):
        """

        It returns only matches where both keypoints currently have no assigned
        map point, which is the intended local-mapping triangulation input.
        """
        ensure_frame_feature_arrays(f1)
        ensure_frame_feature_arrays(f2)

        max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)

        candidate1 = np.array(
            [i for i, p in enumerate(f1.points) if p is None],
            dtype=np.int32,
        )
        candidate2 = np.array(
            [i for i, p in enumerate(f2.points) if p is None],
            dtype=np.int32,
        )

        if idxs1 is not None and idxs2 is not None:
            idxs1 = np.asarray(idxs1, dtype=np.int32).reshape(-1)
            idxs2 = np.asarray(idxs2, dtype=np.int32).reshape(-1)

            if len(idxs1) != len(idxs2):
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

            pair_candidates = [
                (int(i1), int(i2))
                for i1, i2 in zip(idxs1, idxs2)
                if i1 in set(candidate1) and i2 in set(candidate2)
            ]
        else:
            if len(candidate1) == 0 or len(candidate2) == 0:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

            matches = FeatureTrackerShared.feature_matcher.match(
                f1.img,
                f2.img,
                f1.des[candidate1],
                f2.des[candidate2],
                kps1=[f1.kps[i] for i in candidate1],
                kps2=[f2.kps[i] for i in candidate2],
            )

            if matches.idxs1 is None or matches.idxs2 is None:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

            pair_candidates = [
                (int(candidate1[i1]), int(candidate2[i2]))
                for i1, i2 in zip(matches.idxs1, matches.idxs2)
            ]

        if len(pair_candidates) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

        F12, _ = computeF12(f1, f2)

        out1 = []
        out2 = []

        level_sigmas2 = FeatureTrackerShared.feature_manager.level_sigmas2

        for i1, i2 in pair_candidates:
            if i1 < 0 or i1 >= len(f1.des) or i2 < 0 or i2 >= len(f2.des):
                continue

            d = FeatureTrackerShared.descriptor_distance(f1.des[i1], f2.des[i2])
            if d > max_descriptor_distance:
                continue

            octave2 = max(0, min(int(f2.octaves[i2]), len(level_sigmas2) - 1))
            sigma2 = float(level_sigmas2[octave2])

            if not check_dist_epipolar_line(f1.kpsu[i1].pt, f2.kpsu[i2].pt, F12, sigma2):
                continue

            out1.append(i1)
            out2.append(i2)

        if len(out1) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

        idxs1_out = np.asarray(out1, dtype=np.int32)
        idxs2_out = np.asarray(out2, dtype=np.int32)

        if FeatureTrackerShared.oriented_features:
            valid = RotationHistogram.filter_matches_with_histogram_orientation(
                idxs1_out,
                idxs2_out,
                f1.angles,
                f2.angles,
            )
            idxs1_out = idxs1_out[valid]
            idxs2_out = idxs2_out[valid]

        return idxs1_out, idxs2_out, len(idxs1_out)


# Collect statistics from one projection-fusion step.
@dataclass
class ProjectionFuseDiagnostics:
    projected_points: int = 0
    visible_projected_points: int = 0
    candidate_matches: int = 0
    added_observations: int = 0
    fused_points: int = 0
    replaced_points: int = 0
    rejected_bad_point: int = 0
    rejected_not_visible: int = 0
    rejected_descriptor: int = 0
    rejected_scale: int = 0
    rejected_duplicate: int = 0

    def merge(self, other: "ProjectionFuseDiagnostics") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}






def _map_point_position(mp):
    if mp is None:
        return None

    for name in ("pt", "get_position"):
        if hasattr(mp, name):
            try:
                value = getattr(mp, name)()
                return np.asarray(value, dtype=np.float64).reshape(3)
            except Exception:
                pass

    for name in ("position", "position_world"):
        if hasattr(mp, name):
            try:
                value = getattr(mp, name)
                return np.asarray(value, dtype=np.float64).reshape(3)
            except Exception:
                pass

    return None


def _map_point_visibility_info(mp, fallback_position):
    """
        position, normal, min_distance, max_distance

    that same interface but tolerates partially initialized points.
    """
    position = fallback_position
    normal = None
    min_dist = 0.0
    max_dist = np.inf

    if mp is not None and hasattr(mp, "get_all_pos_info"):
        try:
            info = mp.get_all_pos_info()
            if info is not None and len(info) >= 4:
                position = np.asarray(info[0], dtype=np.float64).reshape(3)
                normal = np.asarray(info[1], dtype=np.float64).reshape(3)
                min_dist = float(info[2])
                max_dist = float(info[3])
        except Exception:
            pass

    if normal is None or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-12:
        normal = None

    if not np.isfinite(min_dist):
        min_dist = 0.0

    if not np.isfinite(max_dist) or max_dist <= 0.0:
        max_dist = np.inf

    return position, normal, min_dist, max_dist


def _prepare_visible_projection_candidates(
    frame,
    idxs_ref,
    map_points,
    do_stereo_project=False,
):
    """

    Invalid/non-finite map points are rejected before transform/project. This
    design where projection search uses only visible candidate map points.
    """
    idxs_ref = np.asarray(idxs_ref, dtype=np.int32).reshape(-1)
    map_points = list(map_points)

    if len(idxs_ref) != len(map_points):
        raise ValueError(
            f"idxs_ref/map_points length mismatch: {len(idxs_ref)} vs {len(map_points)}"
        )

    candidate_idxs = []
    candidate_points = []
    candidate_positions = []
    candidate_normals = []
    candidate_min_dists = []
    candidate_max_dists = []

    for idx_ref, mp in zip(idxs_ref, map_points):
        if mp is None:
            continue
        if hasattr(mp, "is_bad") and mp.is_bad():
            continue

        pw0 = _map_point_position(mp)
        if pw0 is None:
            continue

        pw, normal, min_dist, max_dist = _map_point_visibility_info(mp, pw0)

        if pw is None or pw.shape != (3,):
            continue
        if not np.all(np.isfinite(pw)):
            continue

        # Reject impossible numeric values before matrix multiplication.
        if np.linalg.norm(pw) > 1.0e9:
            continue

        candidate_idxs.append(int(idx_ref))
        candidate_points.append(mp)
        candidate_positions.append(pw)
        candidate_normals.append(normal)
        candidate_min_dists.append(min_dist)
        candidate_max_dists.append(max_dist)

    proj_dim = 3 if do_stereo_project else 2

    if len(candidate_points) == 0:
        return (
            np.empty((0,), dtype=np.int32),
            [],
            np.empty((0, proj_dim), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    points_w = np.ascontiguousarray(candidate_positions, dtype=np.float64)

    # Only finite world points reach transform_points().
    points_c = frame.transform_points(points_w)

    finite_camera = np.all(np.isfinite(points_c), axis=1)
    positive_depth = finite_camera & (points_c[:, 2] > Parameters.kMinDepth)

    projs = np.zeros((len(candidate_points), proj_dim), dtype=np.float64)
    depths = np.zeros((len(candidate_points),), dtype=np.float64)

    if np.any(positive_depth):
        if do_stereo_project:
            projs_good, depths_good = frame.camera.project_stereo(points_c[positive_depth])
        else:
            projs_good, depths_good = frame.camera.project(points_c[positive_depth])

        projs[positive_depth] = np.asarray(projs_good, dtype=np.float64)
        depths[positive_depth] = np.asarray(depths_good, dtype=np.float64).reshape(-1)

    try:
        Ow = np.asarray(frame.Ow(), dtype=np.float64).reshape(3)
    except Exception:
        Twc = np.linalg.inv(frame.pose())
        Ow = Twc[:3, 3]

    PO = points_w - Ow.reshape(1, 3)
    dists = np.linalg.norm(PO, axis=1)

    valid = (
        positive_depth
        & np.all(np.isfinite(projs[:, :2]), axis=1)
        & np.isfinite(depths)
        & np.isfinite(dists)
    )

    try:
        valid &= frame.are_in_image(projs[:, :2], depths)
    except Exception:
        width = getattr(frame.camera, "width", None)
        height = getattr(frame.camera, "height", None)
        if width is not None and height is not None:
            valid &= (
                (projs[:, 0] >= 0.0)
                & (projs[:, 0] < float(width))
                & (projs[:, 1] >= 0.0)
                & (projs[:, 1] < float(height))
            )

    normals_available = np.array([n is not None for n in candidate_normals], dtype=bool)
    if np.any(normals_available):
        safe_dists = np.maximum(dists, 1e-12)
        dirs = PO / safe_dists.reshape(-1, 1)

        cos_view = np.ones(len(candidate_points), dtype=np.float64)
        for i, normal in enumerate(candidate_normals):
            if normal is not None:
                cos_view[i] = float(np.dot(normal, dirs[i]))

        valid &= (~normals_available) | (cos_view > Parameters.kViewingCosLimitForPoint)

    min_dists = np.asarray(candidate_min_dists, dtype=np.float64)
    max_dists = np.asarray(candidate_max_dists, dtype=np.float64)

    valid &= dists > min_dists
    valid &= dists < max_dists

    idxs_ref_valid = np.asarray(candidate_idxs, dtype=np.int32)[valid]
    map_points_valid = [mp for mp, keep in zip(candidate_points, valid) if keep]
    projs_valid = projs[valid]
    depths_valid = depths[valid]
    dists_valid = dists[valid]

    return idxs_ref_valid, map_points_valid, projs_valid, depths_valid, dists_valid


def _make_projection_query_safe(frame, projs, depths, radiuses):
    """

    to the spatial search. Candidate map points are filtered by finite projection,
    positive depth, image bounds, and valid search radius before descriptor
    matching.

    This helper preserves the original candidate indexing while making invalid
    candidates produce no KD-tree matches.
    """
    projs = np.asarray(projs, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    radiuses = np.asarray(radiuses, dtype=np.float64).reshape(-1)

    if projs.ndim != 2 or projs.shape[1] < 2:
        raise ValueError(f"Expected projs shape Nx2/Nx3, got {projs.shape}")

    n = len(projs)
    if len(depths) != n or len(radiuses) != n:
        raise ValueError(
            f"Inconsistent projection arrays: projs={len(projs)}, "
            f"depths={len(depths)}, radiuses={len(radiuses)}"
        )

    valid = (
        np.all(np.isfinite(projs[:, :2]), axis=1)
        & np.isfinite(depths)
        & (depths > Parameters.kMinDepth)
        & np.isfinite(radiuses)
        & (radiuses > 0.0)
    )

    try:
        valid &= frame.are_in_image(projs[:, :2], depths)
    except Exception:
        width = getattr(frame.camera, "width", None)
        height = getattr(frame.camera, "height", None)
        if width is not None and height is not None:
            valid &= (
                (projs[:, 0] >= 0.0)
                & (projs[:, 0] < float(width))
                & (projs[:, 1] >= 0.0)
                & (projs[:, 1] < float(height))
            )

    safe_points = np.ascontiguousarray(projs[:, :2], dtype=np.float64)
    safe_radiuses = np.ascontiguousarray(radiuses, dtype=np.float64)

    invalid = ~valid
    if np.any(invalid):
        # cKDTree requires finite query points. Use a finite out-of-image dummy
        # with zero radius so invalid map-point candidates return no matches.
        safe_points[invalid, :] = -1.0e9
        safe_radiuses[invalid] = 0.0

    return safe_points, safe_radiuses, valid


def _max_descriptor_distance(value):
    return Parameters.kMaxDescriptorDistance if value is None else value


def _as_se3_matrix(pose_like) -> np.ndarray:
    if hasattr(pose_like, "matrix"):
        pose_like = pose_like.matrix()
    elif hasattr(pose_like, "to_homogeneous_matrix"):
        pose_like = pose_like.to_homogeneous_matrix()
    return np.asarray(pose_like, dtype=np.float64).reshape(4, 4)


def _project_points_with_pose(keyframe: KeyFrame, points_w: np.ndarray, Tcw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Tcw = _as_se3_matrix(Tcw)
    points_w = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    points_c = (Tcw[:3, :3] @ points_w.T + Tcw[:3, 3].reshape(3, 1)).T
    depths = points_c[:, 2].astype(np.float64, copy=True)
    projs = np.full((len(points_w), 2), np.nan, dtype=np.float64)

    valid_depth = np.isfinite(depths) & (depths > Parameters.kMinDepth)
    if np.any(valid_depth):
        projected, _ = keyframe.camera.project(points_c[valid_depth])
        projs[valid_depth] = np.asarray(projected, dtype=np.float64).reshape(-1, 2)

    try:
        Ow = np.linalg.inv(Tcw)[:3, 3]
    except np.linalg.LinAlgError:
        Ow = np.zeros(3, dtype=np.float64)
    dists = np.linalg.norm(points_w - Ow.reshape(1, 3), axis=1)
    return projs, depths, dists


def _valid_rotation_filter(idxs_ref, idxs_cur, ref_angles, cur_angles):
    if not (kCheckFeaturesOrientation and FeatureTrackerShared.oriented_features):
        return np.asarray(idxs_ref, dtype=np.int32), np.asarray(idxs_cur, dtype=np.int32)

    if len(idxs_ref) == 0:
        return np.asarray(idxs_ref, dtype=np.int32), np.asarray(idxs_cur, dtype=np.int32)

    rot_histo = RotationHistogram(Parameters.kRotationHistogramLength if hasattr(Parameters, "kRotationHistogramLength") else 12)

    for match_idx, (idx_ref, idx_cur) in enumerate(zip(idxs_ref, idxs_cur)):
        rot = float(ref_angles[idx_ref]) - float(cur_angles[idx_cur])
        rot_histo.push(rot, match_idx)

    valid = rot_histo.get_valid_idxs()

    return np.asarray(idxs_ref, dtype=np.int32)[valid], np.asarray(idxs_cur, dtype=np.int32)[valid]


def _search_frame_by_projection(
    f_ref: Frame,
    f_cur: Frame,
    max_reproj_distance=Parameters.kMaxReprojectionDistanceFrame,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
    is_monocular=True,
    already_matched_ref_idxs=None,
):
    max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)

    ensure_frame_feature_arrays(f_ref)
    ensure_frame_feature_arrays(f_cur)

    matched_ref_idxs = np.array(
        [i for i, p in enumerate(f_ref.points) if p is not None and not f_ref.outliers[i]],
        dtype=np.int32,
    )

    if already_matched_ref_idxs is not None:
        matched_ref_idxs = np.setdiff1d(matched_ref_idxs, already_matched_ref_idxs)

    if len(matched_ref_idxs) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

    matched_ref_points = [f_ref.points[i] for i in matched_ref_idxs]

    matched_ref_idxs, matched_ref_points, projs, depths, dists = _prepare_visible_projection_candidates(
        f_cur,
        matched_ref_idxs,
        matched_ref_points,
        do_stereo_project=f_cur.camera.is_stereo(),
    )

    if len(matched_ref_idxs) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

    is_visible = np.ones(len(matched_ref_idxs), dtype=bool)

    kp_ref_octaves = f_ref.octaves[matched_ref_idxs]
    kp_ref_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[kp_ref_octaves]
    radiuses = max_reproj_distance * kp_ref_scale_factors

    safe_query_points, safe_radiuses, valid_projection_mask = _make_projection_query_safe(
        f_cur,
        projs,
        depths,
        radiuses,
    )
    kd_cur_idxs = f_cur.kd.query_ball_point(safe_query_points, safe_radiuses)

    idxs_ref = []
    idxs_cur = []

    cur_des = f_cur.des
    cur_points = f_cur.points
    cur_octaves = f_cur.octaves

    do_stereo_check = f_cur.uRs is not None and len(f_cur.uRs) > 0

    for j, (ref_idx, p_ref) in enumerate(zip(matched_ref_idxs, matched_ref_points)):
        if not is_visible[j]:
            continue

        kp_ref_octave = f_ref.octaves[ref_idx]
        best_dist = float("inf")
        best_k_idx = -1

        candidate_idxs = kd_cur_idxs[j]

        for h, kd_idx in enumerate(candidate_idxs):
            p_cur = cur_points[kd_idx]
            if p_cur is not None and p_cur.num_observations() > 0:
                continue

            kp_cur_octave = cur_octaves[kd_idx]
            if kp_cur_octave < (kp_ref_octave - 1) or kp_cur_octave > (kp_ref_octave + 1):
                continue

            if do_stereo_check and f_cur.uRs[kd_idx] >= 0:
                err_ur = abs(projs[j, 2] - f_cur.uRs[kd_idx])
                scale = FeatureTrackerShared.feature_manager.scale_factors[kp_cur_octave]
                if err_ur >= max_reproj_distance * scale:
                    continue

            descriptor_dist = p_ref.min_des_distance(cur_des[kd_idx])

            if descriptor_dist < best_dist:
                best_dist = descriptor_dist
                best_k_idx = kd_idx

        if best_k_idx > -1 and best_dist < max_descriptor_distance:
            if p_ref.add_frame_view(f_cur, best_k_idx):
                idxs_ref.append(int(ref_idx))
                idxs_cur.append(int(best_k_idx))

    idxs_ref, idxs_cur = _valid_rotation_filter(idxs_ref, idxs_cur, f_ref.angles, f_cur.angles)

    return idxs_ref, idxs_cur, len(idxs_cur)


def _search_keyframe_by_projection(
    kf_ref: KeyFrame,
    f_cur: Frame,
    max_reproj_distance,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
    already_matched_ref_idxs=None,
):
    max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)

    assert kf_ref.is_keyframe, "[search_keyframe_by_projection] kf_ref must be a KeyFrame"

    ensure_frame_feature_arrays(kf_ref)
    ensure_frame_feature_arrays(f_cur)

    ref_mps = kf_ref.get_matched_points()
    if len(ref_mps) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

    matched_ref_idxs = np.array(
        [i for i, p in enumerate(ref_mps) if p is not None and not p.is_bad()],
        dtype=np.int32,
    )

    if already_matched_ref_idxs is not None:
        matched_ref_idxs = np.setdiff1d(matched_ref_idxs, already_matched_ref_idxs)

    matched_ref_points = [ref_mps[i] for i in matched_ref_idxs]
    if len(matched_ref_points) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

    matched_ref_idxs, matched_ref_points, projs, depths, dists = _prepare_visible_projection_candidates(
        f_cur,
        matched_ref_idxs,
        matched_ref_points,
        do_stereo_project=f_cur.camera.is_stereo(),
    )

    if len(matched_ref_points) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

    visible = np.ones(len(matched_ref_points), dtype=bool)
    predicted_levels = MapPoint.predict_detection_levels(matched_ref_points, dists)
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors

    safe_query_points, safe_radiuses, valid_projection_mask = _make_projection_query_safe(
        f_cur,
        projs,
        depths,
        radiuses,
    )
    kd_cur_idxs = f_cur.kd.query_ball_point(safe_query_points, safe_radiuses)

    idxs_ref = []
    idxs_cur = []

    for j, (ref_idx, mp) in enumerate(zip(matched_ref_idxs, matched_ref_points)):
        if not visible[j]:
            continue

        predicted_level = predicted_levels[j]
        best_dist = float("inf")
        best_dist2 = float("inf")
        best_level = -1
        best_level2 = -1
        best_k_idx = -1

        for idx2 in kd_cur_idxs[j]:
            if f_cur.points[idx2] is not None:
                continue

            kp_level = f_cur.octaves[idx2]
            if kp_level < predicted_level - 1 or kp_level > predicted_level + 1:
                continue

            descriptor_dist = mp.min_des_distance(f_cur.des[idx2])

            if descriptor_dist < best_dist:
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_k_idx = idx2
            elif descriptor_dist < best_dist2:
                best_dist2 = descriptor_dist
                best_level2 = kp_level

        if best_k_idx > -1 and best_dist < max_descriptor_distance:
            if best_level == best_level2 and best_dist > best_dist2 * ratio_test:
                continue
            if mp.add_frame_view(f_cur, best_k_idx):
                idxs_ref.append(int(ref_idx))
                idxs_cur.append(int(best_k_idx))

    idxs_ref, idxs_cur = _valid_rotation_filter(idxs_ref, idxs_cur, kf_ref.angles, f_cur.angles)

    return idxs_ref, idxs_cur, len(idxs_cur)


def _search_map_by_projection(
    points: list[MapPoint],
    f_cur: Frame,
    max_reproj_distance=Parameters.kMaxReprojectionDistanceMap,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
    far_points_threshold=None,
    diagnostics: dict | None = None,
):
    max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)
    input_points = list(points)

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "input_local_points": len(input_points),
                "rejected_bad": sum(
                    1
                    for p in input_points
                    if p is None or (hasattr(p, "is_bad") and p.is_bad())
                ),
                "rejected_already_seen": sum(
                    1
                    for p in input_points
                    if p is not None
                    and not (hasattr(p, "is_bad") and p.is_bad())
                    and getattr(p, "last_frame_id_seen", -1) == f_cur.id
                ),
                "rejected_not_visible": 0,
                "visible_projected_points": 0,
                "kd_candidate_count": 0,
                "descriptor_comparisons": 0,
                "matches": 0,
            }
        )

    candidate_pairs = [
        (idx, p)
        for idx, p in enumerate(input_points)
        if p is not None
        and not (hasattr(p, "is_bad") and p.is_bad())
        and getattr(p, "last_frame_id_seen", -1) != f_cur.id
    ]

    if len(candidate_pairs) == 0:
        return 0, []

    ensure_frame_feature_arrays(f_cur)

    num_candidate_points = len(candidate_pairs)
    input_idxs = np.asarray([idx for idx, _ in candidate_pairs], dtype=np.int32)
    points = [p for _, p in candidate_pairs]
    input_idxs, points, projs, depths, dists = _prepare_visible_projection_candidates(
        f_cur,
        input_idxs,
        points,
        do_stereo_project=f_cur.camera.is_stereo(),
    )

    if diagnostics is not None:
        diagnostics["rejected_not_visible"] = max(
            0,
            int(num_candidate_points) - int(len(points)),
        )
        diagnostics["visible_projected_points"] = int(len(points))

    if len(points) == 0:
        return 0, []

    visibility_flags = np.ones(len(points), dtype=bool)
    predicted_levels = MapPoint.predict_detection_levels(points, dists)

    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors

    safe_query_points, safe_radiuses, valid_projection_mask = _make_projection_query_safe(
        f_cur,
        projs,
        depths,
        radiuses,
    )
    kd_cur_idxs = f_cur.kd.query_ball_point(safe_query_points, safe_radiuses)

    if diagnostics is not None:
        diagnostics["kd_candidate_count"] = int(sum(len(idxs) for idxs in kd_cur_idxs))

    if far_points_threshold is not None:
        visible_before_far_filter = int(np.count_nonzero(visibility_flags))
        visibility_flags = np.logical_and(visibility_flags, depths < far_points_threshold)
        if diagnostics is not None:
            diagnostics["rejected_not_visible"] = int(diagnostics["rejected_not_visible"]) + max(
                0,
                visible_before_far_filter - int(np.count_nonzero(visibility_flags)),
            )

    idxs_and_pts = [
        (i, p)
        for i, p in enumerate(points)
        if visibility_flags[i]
        and p is not None
        and not p.is_bad()
        and p.last_frame_id_seen != f_cur.id
    ]

    found_pts_count = 0
    found_pts_fidxs = []

    for i, p in idxs_and_pts:
        p.increase_visible()
        predicted_level = predicted_levels[i]

        best_dist = float("inf")
        best_dist2 = float("inf")
        best_level = -1
        best_level2 = -1
        best_k_idx = -1

        for kd_idx in kd_cur_idxs[i]:
            p_f = f_cur.points[kd_idx]
            if p_f is not None and p_f.num_observations() > 0:
                continue

            kp_level = f_cur.octaves[kd_idx]
            if kp_level < predicted_level - 1 or kp_level > predicted_level:
                continue

            if diagnostics is not None:
                diagnostics["descriptor_comparisons"] = int(diagnostics["descriptor_comparisons"]) + 1
            descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])

            if descriptor_dist < best_dist:
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_k_idx = kd_idx
            elif descriptor_dist < best_dist2:
                best_dist2 = descriptor_dist
                best_level2 = kp_level

        if best_k_idx > -1 and best_dist < max_descriptor_distance:
            if best_level == best_level2 and best_dist > best_dist2 * ratio_test:
                continue
            if p.add_frame_view(f_cur, best_k_idx):
                p.increase_found()
                found_pts_count += 1
                found_pts_fidxs.append(best_k_idx)

    if diagnostics is not None:
        diagnostics["matches"] = int(found_pts_count)

    return found_pts_count, found_pts_fidxs


def _search_local_frames_by_projection(
    map,
    f_cur,
    local_window_size=Parameters.kLocalBAWindowSize,
    max_descriptor_distance=None,
):
    max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)
    frames = map.get_last_keyframes(local_window_size)
    frame_valid_points = set([p for f in frames for p in f.get_points() if p is not None])
    return _search_map_by_projection(
        list(frame_valid_points),
        f_cur,
        max_descriptor_distance=max_descriptor_distance,
    )


def _search_all_map_by_projection(map, f_cur, max_descriptor_distance=None):
    max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)
    return _search_map_by_projection(
        map.get_points().to_list() if hasattr(map.get_points(), "to_list") else list(map.get_points()),
        f_cur,
        max_descriptor_distance=max_descriptor_distance,
    )


def _search_and_fuse(
    points: list[MapPoint],
    keyframe: KeyFrame,
    max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
):
    max_descriptor_distance = 0.5 * _max_descriptor_distance(max_descriptor_distance)

    if len(points) == 0:
        return 0

    ensure_frame_feature_arrays(keyframe)

    good_points = [p for p in points if p is not None and not p.is_bad_or_is_in_keyframe(keyframe)]

    if len(good_points) == 0:
        return 0

    visible, projs, depths, dists = keyframe.are_visible(good_points, keyframe.camera.is_stereo())

    predicted_levels = MapPoint.predict_detection_levels(good_points, dists)
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors
    kd_idxs = keyframe.kd.query_ball_point(projs[:, :2], radiuses)

    fused_pts_count = 0
    inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2

    for j, point in enumerate(good_points):
        if not visible[j]:
            continue

        predicted_level = predicted_levels[j]
        best_dist = float("inf")
        best_kd_idx = -1

        for kd_idx in kd_idxs[j]:
            kp_level = keyframe.octaves[kd_idx]
            if kp_level < predicted_level - 1 or kp_level > predicted_level:
                continue

            err = projs[j, :2] - np.array(keyframe.kpsu[kd_idx].pt, dtype=np.float64)
            chi2 = float(np.dot(err, err) * inv_level_sigmas2[kp_level])

            if chi2 > Parameters.kChi2Mono:
                continue

            descriptor_dist = point.min_des_distance(keyframe.des[kd_idx])

            if descriptor_dist < best_dist:
                best_dist = descriptor_dist
                best_kd_idx = kd_idx

        if best_kd_idx > -1 and best_dist < max_descriptor_distance:
            existing = keyframe.get_point_match(best_kd_idx)

            if existing is not None:
                if existing.num_observations() > point.num_observations():
                    point.replace_with(existing)
                else:
                    existing.replace_with(point)
                    point.add_observation(keyframe, best_kd_idx)
            else:
                point.add_observation(keyframe, best_kd_idx)

            point.update_info()
            fused_pts_count += 1

    return fused_pts_count


def _search_and_fuse_for_loop_correction(
    keyframe: KeyFrame,
    Scw,
    points,
    replace_points,
    max_reproj_distance=Parameters.kLoopClosingMaxReprojectionDistanceFuse,
    max_descriptor_distance=None,
    diagnostics: ProjectionFuseDiagnostics | None = None,
):
    """

    `replace_points`: when a projected loop-side point lands on an already
    occupied target keypoint, the existing current-side point is returned for
    later replacement by the caller. Missing observations are added directly.
    """
    max_descriptor_distance = 0.5 * _max_descriptor_distance(max_descriptor_distance)

    points = list(points)
    if len(points) != len(replace_points):
        raise ValueError("points and replace_points must have the same length")

    if diagnostics is None:
        diagnostics = ProjectionFuseDiagnostics()

    if len(points) == 0:
        return replace_points

    ensure_frame_feature_arrays(keyframe)

    good_input_idxs = []
    good_points = []
    good_positions = []
    for idx, point in enumerate(points):
        if point is None:
            diagnostics.rejected_bad_point += 1
            continue
        if point.is_bad():
            diagnostics.rejected_bad_point += 1
            continue
        if point.is_in_keyframe(keyframe):
            diagnostics.rejected_duplicate += 1
            continue
        position = _map_point_position(point)
        if position is None or not np.all(np.isfinite(position)):
            diagnostics.rejected_bad_point += 1
            continue
        good_input_idxs.append(idx)
        good_points.append(point)
        good_positions.append(position)

    diagnostics.projected_points += len(good_points)
    if len(good_points) == 0:
        return replace_points

    projs, depths, dists = _project_points_with_pose(
        keyframe,
        np.asarray(good_positions, dtype=np.float64),
        Scw,
    )

    finite_projection = (
        np.all(np.isfinite(projs[:, :2]), axis=1)
        & np.isfinite(depths)
        & (depths > Parameters.kMinDepth)
        & np.isfinite(dists)
    )
    try:
        visible = finite_projection & keyframe.are_in_image(projs[:, :2], depths)
    except Exception:
        width = getattr(keyframe.camera, "width", None)
        height = getattr(keyframe.camera, "height", None)
        if width is None or height is None:
            visible = finite_projection
        else:
            visible = finite_projection & (
                (projs[:, 0] >= 0.0)
                & (projs[:, 0] < float(width))
                & (projs[:, 1] >= 0.0)
                & (projs[:, 1] < float(height))
            )

    diagnostics.visible_projected_points += int(np.sum(visible))

    predicted_levels = MapPoint.predict_detection_levels(good_points, dists)
    feature_manager = FeatureTrackerShared.feature_manager
    if feature_manager is None:
        scale_factors = np.ones(max(1, int(np.max(predicted_levels)) + 1), dtype=np.float64)
        inv_level_sigmas2 = np.ones_like(scale_factors)
    else:
        predicted_levels = np.clip(predicted_levels, 0, feature_manager.num_levels - 1)
        scale_factors = feature_manager.scale_factors
        inv_level_sigmas2 = feature_manager.inv_level_sigmas2

    radiuses = max_reproj_distance * scale_factors[predicted_levels]
    query_points, query_radiuses, safe_visible = _make_projection_query_safe(
        keyframe,
        projs,
        depths,
        radiuses,
    )
    visible &= safe_visible
    kd_idxs = keyframe.kd.query_ball_point(query_points, query_radiuses)

    for local_idx, point in enumerate(good_points):
        input_idx = good_input_idxs[local_idx]
        if not visible[local_idx]:
            diagnostics.rejected_not_visible += 1
            continue

        kd_idxs_local = list(kd_idxs[local_idx])
        if len(kd_idxs_local) == 0:
            diagnostics.rejected_not_visible += 1
            continue

        predicted_level = int(predicted_levels[local_idx])
        best_dist = float("inf")
        best_kd_idx = -1
        saw_scale_reject = False
        saw_descriptor_candidate = False

        for kd_idx in kd_idxs_local:
            kd_idx = int(kd_idx)
            if kd_idx < 0 or kd_idx >= len(keyframe.des):
                continue

            existing = keyframe.get_point_match(kd_idx)
            if existing is point:
                diagnostics.rejected_duplicate += 1
                continue
            if existing is not None and existing.is_bad():
                diagnostics.rejected_bad_point += 1
                continue

            kp_level = int(keyframe.octaves[kd_idx])
            if kp_level < predicted_level - 1 or kp_level > predicted_level:
                saw_scale_reject = True
                continue

            inv_sigma2 = float(inv_level_sigmas2[min(kp_level, len(inv_level_sigmas2) - 1)])
            kp = keyframe.kpsu[kd_idx]
            kp_uv = np.asarray(kp.pt if hasattr(kp, "pt") else kp, dtype=np.float64).reshape(2)
            err = projs[local_idx, :2] - kp_uv
            chi2 = float(np.dot(err, err) * inv_sigma2)
            if not np.isfinite(chi2) or chi2 > Parameters.kChi2Mono:
                diagnostics.rejected_not_visible += 1
                continue

            diagnostics.candidate_matches += 1
            saw_descriptor_candidate = True
            descriptor_dist = point.min_des_distance(keyframe.des[kd_idx])
            if descriptor_dist < best_dist:
                best_dist = descriptor_dist
                best_kd_idx = kd_idx

        if best_kd_idx < 0:
            if saw_scale_reject and not saw_descriptor_candidate:
                diagnostics.rejected_scale += 1
            else:
                diagnostics.rejected_descriptor += 1
            continue

        if best_dist >= max_descriptor_distance:
            diagnostics.rejected_descriptor += 1
            continue

        existing = keyframe.get_point_match(best_kd_idx)
        if existing is None:
            if point.add_observation(keyframe, best_kd_idx):
                point.update_info()
                diagnostics.added_observations += 1
                diagnostics.fused_points += 1
            else:
                diagnostics.rejected_duplicate += 1
        elif existing is point:
            diagnostics.rejected_duplicate += 1
        elif existing.is_bad():
            diagnostics.rejected_bad_point += 1
        else:
            replace_points[input_idx] = existing
            diagnostics.replaced_points += 1
            diagnostics.fused_points += 1

    return replace_points


def _search_more_map_points_by_projection(
    points,
    f_cur: KeyFrame,
    current_keyframe_Tcw_corrected: np.ndarray,
    f_cur_matched_points: list,
    f_cur_matched_points_idxs=None,
    max_reproj_distance: float = Parameters.kLoopClosingMaxReprojectionDistanceMapSearch,
    max_descriptor_distance=None,
    return_diagnostics: bool = False,
):
    """Project corrected loop candidates into the current keyframe and extend loop matches.

    Args:
        current_keyframe_Tcw_corrected: corrected current-keyframe `Tcw` pose used for
            RGB-D SE3 loop projection expansion. This is expected to be
            `Tcw_current @ inv(T12)` for the accepted loop correction.
    """
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5 * float(Parameters.kMaxDescriptorDistance or 256)

    diagnostics = {
        "candidate_input_points": int(len(points) if points is not None else 0),
        "candidate_unique_points": 0,
        "projected_visible_points": 0,
        "new_projection_matches": 0,
    }

    def _ret(found, matches):
        if return_diagnostics:
            return found, matches, diagnostics
        return found, matches

    if not points or len(f_cur_matched_points) == 0:
        return _ret(0, f_cur_matched_points)

    ensure_frame_feature_arrays(f_cur)
    if f_cur.kd is None:
        return _ret(0, f_cur_matched_points)

    Tcw = np.asarray(current_keyframe_Tcw_corrected, dtype=np.float64).reshape(4, 4)
    Rcw = Tcw[:3, :3]
    tcw = Tcw[:3, 3]

    # Compute the current camera center in world coordinates.
    Ow = (-Rcw.T @ tcw).reshape(3)

    already_matched_ids = {mp.id for mp in f_cur_matched_points if mp is not None}

    target_points = [
        p for p in points
        if p is not None and not p.is_bad() and p.id not in already_matched_ids
    ]
    diagnostics["candidate_unique_points"] = int(len(target_points))
    if not target_points:
        return _ret(0, f_cur_matched_points)

    # Gather the map-point data needed by the visibility and projection tests.
    positions_w = np.array([p.get_position() for p in target_points], dtype=np.float64)
    normals = np.array([p.get_normal() for p in target_points], dtype=np.float64)
    min_dists = np.array([p.get_min_distance_invariance() for p in target_points], dtype=np.float64)
    max_dists = np.array([p.get_max_distance_invariance() for p in target_points], dtype=np.float64)

    # Project into corrected camera frame
    positions_c = (Rcw @ positions_w.T).T + tcw
    depths = positions_c[:, 2]

    projs_raw, _ = f_cur.camera.project(positions_c)
    projs = np.asarray(projs_raw, dtype=np.float64).reshape(-1, 2)

    # View direction vectors from camera center to each point (world frame)
    POs = positions_w - Ow
    dists_from_cam = np.linalg.norm(POs, axis=1, keepdims=False)
    POs_unit = POs / np.maximum(dists_from_cam[:, None], 1e-12)
    cos_views = np.einsum("ij,ij->i", normals, POs_unit)

    in_image = f_cur.are_in_image(projs, depths)
    good_depth = depths > Parameters.kMinDepth
    good_angle = cos_views > Parameters.kViewingCosLimitForPoint
    good_dist = (dists_from_cam > min_dists) & (dists_from_cam < max_dists)

    visible_mask = in_image & good_depth & good_angle & good_dist
    diagnostics["projected_visible_points"] = int(np.sum(visible_mask))

    # Scale-aware search radius
    feature_manager = FeatureTrackerShared.feature_manager
    if feature_manager is not None:
        predicted_levels = MapPoint.predict_detection_levels(target_points, dists_from_cam)
        scale_factors = np.asarray(feature_manager.scale_factors, dtype=np.float64)
        radiuses = max_reproj_distance * scale_factors[
            np.clip(predicted_levels, 0, len(scale_factors) - 1)
        ]
    else:
        predicted_levels = np.zeros(len(target_points), dtype=np.int32)
        radiuses = np.full(len(target_points), max_reproj_distance, dtype=np.float64)

    # Batch KD-tree query — scipy supports variable radius per point
    all_nearby = f_cur.kd.query_ball_point(projs, radiuses)

    found_pts_count = 0
    for i, point in enumerate(target_points):
        if not visible_mask[i]:
            continue

        nearby_idxs = all_nearby[i]
        if not nearby_idxs:
            continue

        predicted_level = int(predicted_levels[i])
        best_dist = float("inf")
        best_kp_idx = -1
        for kp_idx in nearby_idxs:
            if f_cur_matched_points[kp_idx] is not None:
                continue
            kp_level = int(f_cur.octaves[kp_idx])
            if kp_level < predicted_level - 1 or kp_level > predicted_level:
                continue
            dist = float(point.min_des_distance(f_cur.des[kp_idx]))
            if dist < best_dist:
                best_dist = dist
                best_kp_idx = kp_idx

        if best_kp_idx >= 0 and best_dist < max_descriptor_distance:
            f_cur_matched_points[best_kp_idx] = point
            found_pts_count += 1

    diagnostics["new_projection_matches"] = int(found_pts_count)
    return _ret(found_pts_count, f_cur_matched_points)
