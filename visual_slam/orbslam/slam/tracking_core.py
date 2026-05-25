"""
Low-level tracking geometry routines.
This module implements pose fitting, match propagation, and RGB-D support helpers.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.frame import Frame, ensure_frame_feature_arrays
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.map import Map
from visual_slam.orbslam.slam.map_point import MapPoint
from visual_slam.orbslam.slam.sensor_types import SensorType
from visual_slam.orbslam.utilities.geom_2views import estimate_pose_ess_mat
from visual_slam.orbslam.utilities.geometry import inv_T
from visual_slam.orbslam.utilities.logging import Printer


kRansacThresholdNormalized = 0.0004
kRansacProb = 0.999
kNumMinInliersEssentialMat = 8
kRansacReprojThreshold = 5
kRansacMinNumInliers = 15


def _as_int_array(values):
    return np.asarray(values, dtype=np.int32).reshape(-1)


# Provide the low-level geometric helpers used by the tracking front-end.
class TrackingCore:
    @staticmethod
    def estimate_pose_by_fitting_ess_mat(
        f_ref: Frame,
        f_cur: Frame,
        idxs_ref: list[int],
        idxs_cur: list[int],
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Estimate inter-frame rotation from an essential matrix and filter outliers.

        - estimate_pose_ess_mat returns Trc such that pr = Trc * pc.
        - Only rotation is used here; translation scale from E is not reliable.
        """
        idxs_ref = _as_int_array(idxs_ref)
        idxs_cur = _as_int_array(idxs_cur)

        if len(idxs_ref) == 0 or len(idxs_cur) != len(idxs_ref):
            Printer.orange("estimate_pose_by_fitting_ess_mat: empty or inconsistent match indices")
            return idxs_ref, idxs_cur, 0

        ensure_frame_feature_arrays(f_ref)
        ensure_frame_feature_arrays(f_cur)

        if len(idxs_ref) < kNumMinInliersEssentialMat:
            return idxs_ref, idxs_cur, 0

        kpn_ref = np.asarray(f_ref.kpsn[idxs_ref], dtype=np.float64)
        kpn_cur = np.asarray(f_cur.kpsn[idxs_cur], dtype=np.float64)

        try:
            ransac_method = cv2.USAC_MAGSAC
        except AttributeError:
            ransac_method = cv2.RANSAC

        Trc, mask_match = estimate_pose_ess_mat(
            kpn_ref,
            kpn_cur,
            method=ransac_method,
            prob=kRansacProb,
            threshold=kRansacThresholdNormalized,
        )

        if Trc is None or mask_match is None:
            return idxs_ref, idxs_cur, 0

        mask_idxs = np.asarray(mask_match).ravel() > 0
        num_inliers = int(np.count_nonzero(mask_idxs))

        idxs_ref_in = idxs_ref[mask_idxs]
        idxs_cur_in = idxs_cur[mask_idxs]

        if num_inliers < kNumMinInliersEssentialMat:
            Printer.orange("Essential mat: not enough inliers")
            return idxs_ref_in, idxs_cur_in, num_inliers

        # Trc maps current camera coordinates to reference camera coordinates.
        # Tcr maps reference to current.
        Tcr = inv_T(Trc)
        estimated_Tcw = Tcr @ f_ref.pose()

        Rcw = estimated_Tcw[:3, :3]
        tcw = f_ref.pose()[:3, 3]
        f_cur.update_rotation_and_translation(Rcw, tcw)

        return idxs_ref_in, idxs_cur_in, num_inliers

    @staticmethod
    def find_homography_with_ransac(
        f_cur: Frame,
        f_ref: Frame,
        idxs_cur: list[int],
        idxs_ref: list[int],
        reproj_threshold=kRansacReprojThreshold,
        min_num_inliers=kRansacMinNumInliers,
    ) -> Tuple[bool, np.ndarray, np.ndarray, int, int]:
        idxs_cur = _as_int_array(idxs_cur)
        idxs_ref = _as_int_array(idxs_ref)

        if len(idxs_cur) == 0 or len(idxs_cur) != len(idxs_ref):
            Printer.orange("find_homography_with_ransac: empty or inconsistent match indices")
            return False, np.array([], dtype=int), np.array([], dtype=int), 0, 0

        try:
            ransac_method = cv2.USAC_MAGSAC
        except AttributeError:
            ransac_method = cv2.RANSAC

        kps_cur = np.array([f_cur.kps[i].pt for i in idxs_cur], dtype=np.float32)
        kps_ref = np.array([f_ref.kps[i].pt for i in idxs_ref], dtype=np.float32)

        _, mask = cv2.findHomography(
            kps_cur,
            kps_ref,
            ransac_method,
            ransacReprojThreshold=float(reproj_threshold),
        )

        if mask is None:
            return False, np.array([], dtype=int), np.array([], dtype=int), 0, 0

        mask = np.asarray(mask, dtype=bool).ravel()
        num_inliers = int(np.count_nonzero(mask))
        num_outliers = int(len(idxs_cur) - num_inliers)

        if num_inliers < int(min_num_inliers):
            return False, np.array([], dtype=int), np.array([], dtype=int), 0, num_outliers

        idxs_cur = idxs_cur[mask]
        idxs_ref = idxs_ref[mask]
        num_matched_kps = len(idxs_cur)

        return True, idxs_cur, idxs_ref, num_matched_kps, num_outliers

    @staticmethod
    def propagate_map_point_matches(
        f_ref: Frame,
        f_cur: Frame,
        idxs_ref,
        idxs_cur,
        max_descriptor_distance=None,
    ):
        if max_descriptor_distance is None:
            max_descriptor_distance = Parameters.kMaxDescriptorDistance

        idxs_ref = _as_int_array(idxs_ref)
        idxs_cur = _as_int_array(idxs_cur)

        idx_ref_out = []
        idx_cur_out = []
        num_matched_map_pts = 0

        for i, idx_ref in enumerate(idxs_ref):
            p_ref = f_ref.points[idx_ref]

            if p_ref is None or f_ref.outliers[idx_ref] or p_ref.is_bad():
                continue

            idx_cur = int(idxs_cur[i])
            p_cur = f_cur.points[idx_cur]

            if p_cur is not None:
                continue

            des_distance = p_ref.min_des_distance(f_cur.des[idx_cur])

            if des_distance > max_descriptor_distance:
                continue

            if p_ref.add_frame_view(f_cur, idx_cur):
                num_matched_map_pts += 1
                idx_ref_out.append(int(idx_ref))
                idx_cur_out.append(int(idx_cur))

        return num_matched_map_pts, idx_ref_out, idx_cur_out

    @staticmethod
    def create_vo_points(
        frame: Frame,
        max_num_points=Parameters.kMaxNumVisualOdometryPoints,
        color=(0, 0, 255),
    ):
        """
        Create temporary VO points on a frame using RGB-D/stereo depth.
        """
        if frame.depths is None or len(frame.depths) == 0:
            return []

        depth_threshold = frame.camera.depth_threshold

        valid_mask = frame.depths > Parameters.kMinDepth
        if not np.any(valid_mask):
            return []

        valid_depths = frame.depths[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        sort_indices = np.argsort(valid_depths)
        sorted_depths = valid_depths[sort_indices]
        sorted_indices = valid_indices[sort_indices]

        mask_depths_smaller_than_th = sorted_depths < depth_threshold
        mask_first_N_points = np.arange(len(sorted_depths)) < int(max_num_points)
        mask_first_selection = np.logical_or(mask_depths_smaller_than_th, mask_first_N_points)

        selected_indices = sorted_indices[mask_first_selection]

        if len(selected_indices) == 0:
            return []

        num_observations = np.array(
            [
                frame.points[i].num_observations() if frame.points[i] is not None else 0
                for i in selected_indices
            ],
            dtype=np.int32,
        )

        final_indices = selected_indices[num_observations < 1]

        if len(final_indices) == 0:
            return []

        pts3d, pts3d_mask = frame.unproject_points_3d(final_indices, transform_in_world=True)

        created_points = []

        for idx, point_w, valid in zip(final_indices, pts3d, pts3d_mask):
            if not bool(valid):
                continue

            mp = MapPoint(point_w[0:3], color=color)
            mp.add_frame_view(frame, int(idx))
            frame.points[int(idx)] = mp
            created_points.append(mp)

        return created_points

    @staticmethod
    def create_and_add_stereo_map_points_on_new_kf(
        f: Frame,
        kf: KeyFrame,
        map: Map,
        img=None,
    ):
        valid_depths_and_idxs = [
            (float(z), i)
            for i, z in enumerate(kf.depths)
            if z > Parameters.kMinDepth
        ]

        valid_depths_and_idxs.sort()

        if len(valid_depths_and_idxs) == 0:
            Printer.orange("[create_and_add_stereo_map_points_on_new_kf] no valid depths found")
            return 0

        sorted_z_values, sorted_idx_values = zip(*valid_depths_and_idxs)
        sorted_z_values = np.array(sorted_z_values, dtype=np.float32)
        sorted_idx_values = np.array(sorted_idx_values, dtype=np.int32)

        N = Parameters.kMaxNumStereoPointsOnNewKeyframe

        mask_depths_smaller_than_th = sorted_z_values < kf.camera.depth_threshold
        mask_first_N_points = np.zeros(len(sorted_z_values), dtype=bool)
        mask_first_N_points[: min(N, len(sorted_z_values))] = True

        mask_first_selection = np.logical_or(mask_depths_smaller_than_th, mask_first_N_points)

        sorted_idx_values = sorted_idx_values[mask_first_selection]

        sorted_points = [kf.points[int(i)] for i in sorted_idx_values]

        vector_num_mp_observations = np.array(
            [p.num_observations() if p is not None else 0 for p in sorted_points],
            dtype=np.int32,
        )

        mask_where_to_create_new_map_points = vector_num_mp_observations < 1
        sorted_idx_values = sorted_idx_values[mask_where_to_create_new_map_points]

        if len(sorted_idx_values) == 0:
            return 0

        pts3d, pts3d_mask = f.unproject_points_3d(sorted_idx_values, transform_in_world=True)

        return map.add_stereo_points(pts3d, pts3d_mask, f, kf, sorted_idx_values, img)

    @staticmethod
    def count_tracked_and_non_tracked_close_points(
        f_cur: Frame,
        sensor_type: SensorType,
    ):
        num_non_tracked_close = 0
        num_tracked_close = 0
        tracked_mask = None

        if sensor_type != SensorType.MONOCULAR:
            points_mask = np.array([p is not None for p in f_cur.points], dtype=bool)
            outliers_mask = ~np.asarray(f_cur.outliers, dtype=bool)
            tracked_mask = points_mask & outliers_mask

            depth_mask = (
                (f_cur.depths > Parameters.kMinDepth)
                & (f_cur.depths < f_cur.camera.depth_threshold)
            )

            num_tracked_close = int(np.sum(depth_mask & tracked_mask))
            num_non_tracked_close = int(np.sum(depth_mask & ~tracked_mask))

        return num_tracked_close, num_non_tracked_close, tracked_mask
