"""
Relocalization logic for tracking recovery.
This module retrieves candidate keyframes, matches observations, and solves a PnP pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.bow_matcher import BoWGuidedMatcher
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.frame import Frame, ensure_frame_feature_arrays
from visual_slam.orbslam.slam.geometry_matchers import ProjectionMatcher
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.map import Map
from visual_slam.orbslam.slam.optimizer_g2o import pose_optimization
from visual_slam.orbslam.slam.rotation_histogram import RotationHistogram


# Store the outcome of one PnP-based relocalization hypothesis.
@dataclass
class PnPResult:
    success: bool
    Tcw: Optional[np.ndarray]
    inlier_mask: np.ndarray
    num_inliers: int
    error: Optional[str] = None


# Provide a simple map-scan fallback when no BoW database is available.
class TemporaryRelocalizationKeyFrameDatabase:
    """Fallback relocalization database that scans the current map keyframes."""

    is_temporary_fallback = True

    def __init__(self, slam_map: Optional[Map] = None):
        self.map = slam_map

    def detect_relocalization_candidates(self, frame: Frame):
        if self.map is None:
            return []

        keyframes = self.map.get_keyframes()
        if hasattr(keyframes, "to_list"):
            keyframes = keyframes.to_list()
        else:
            keyframes = list(keyframes)

        candidates = [
            kf
            for kf in keyframes
            if kf is not None
            and getattr(kf, "id", None) != getattr(frame, "id", None)
            and not (hasattr(kf, "is_bad") and kf.is_bad())
        ]

        # Prefer recent keyframes when falling back to a map scan.
        return sorted(candidates, key=lambda kf: (getattr(kf, "kid", -1), getattr(kf, "id", -1)), reverse=True)


# Recover tracking by retrieving candidate keyframes and solving a camera pose.
class Relocalizer:
    """Relocalization worker that recovers pose from retrieved keyframes."""
    def __init__(self, slam_map: Optional[Map] = None, keyframe_database=None):
        self.map = slam_map
        self.keyframe_database = keyframe_database
        self.fallback_database = TemporaryRelocalizationKeyFrameDatabase(slam_map)

        self.num_relocalization_candidates = 0
        self.num_relocalization_matches = 0
        self.num_relocalization_inliers = 0
        self.last_relocalization_success = False
        self.last_relocalization_kf_id = None
        self.last_relocalization_error = None
        self.last_bow_guided_matching_available = False
        self.last_fallback_descriptor_matching = False
        self.last_match_diagnostics = None
        self._active_keyframe_database = None

    def reset_stats(self) -> None:
        self.num_relocalization_candidates = 0
        self.num_relocalization_matches = 0
        self.num_relocalization_inliers = 0
        self.last_relocalization_success = False
        self.last_relocalization_kf_id = None
        self.last_relocalization_error = None
        self.last_bow_guided_matching_available = False
        self.last_fallback_descriptor_matching = False
        self.last_match_diagnostics = None

    def detect_relocalization_candidates(
        self,
        frame: Frame,
        keyframe_database=None,
        keyframes_map: Optional[dict[int, KeyFrame]] = None,
        detection_output=None,
    ) -> list[KeyFrame]:
        if detection_output is not None and getattr(detection_output, "candidate_idxs", None) is not None:
            candidate_ids = list(getattr(detection_output, "candidate_idxs", []))
            keyframes_map = keyframes_map or {}
            candidates = [keyframes_map[idx] for idx in candidate_ids if idx in keyframes_map]
            return [kf for kf in candidates if kf is not None and not kf.is_bad()]

        database = keyframe_database if keyframe_database is not None else self.keyframe_database
        if database is not None and hasattr(database, "detect_relocalization_candidates"):
            candidates = database.detect_relocalization_candidates(frame)
        else:
            candidates = self.fallback_database.detect_relocalization_candidates(frame)

        return self._normalize_candidates(candidates, keyframes_map)

    def relocalize(
        self,
        frame: Frame,
        detection_output=None,
        keyframes_map: Optional[dict[int, KeyFrame]] = None,
        keyframe_database=None,
    ) -> bool:
        self.reset_stats()

        try:
            ensure_frame_feature_arrays(frame)
            if len(frame.des) == 0 or len(frame.kps) == 0:
                self.last_relocalization_error = "current frame has no descriptors"
                return False

            database = keyframe_database if keyframe_database is not None else self.keyframe_database
            self._active_keyframe_database = database
            candidates = self.detect_relocalization_candidates(
                frame,
                keyframe_database=database,
                keyframes_map=keyframes_map,
                detection_output=detection_output,
            )

            self.num_relocalization_candidates = len(candidates)
            if len(candidates) == 0:
                self.last_relocalization_error = "no relocalization candidates"
                return False

            original_pose = frame.pose()
            original_points = list(getattr(frame, "points", []))
            if len(original_points) != len(frame.kps):
                original_points = [None] * len(frame.kps)

            for keyframe in candidates:
                if keyframe is None or keyframe is frame:
                    continue
                if hasattr(keyframe, "is_bad") and keyframe.is_bad():
                    continue

                candidate_ok = self._try_candidate(frame, keyframe, original_pose, original_points)
                if candidate_ok:
                    frame.kf_ref = keyframe
                    self.last_relocalization_success = True
                    self.last_relocalization_kf_id = getattr(keyframe, "id", getattr(keyframe, "kid", None))
                    self.last_relocalization_error = None
                    return True

                frame.update_pose(g2o.Isometry3d(original_pose))
                frame.points = list(original_points)
                frame.outliers = np.zeros(len(frame.kps), dtype=bool)

            self.last_relocalization_error = "all candidates rejected"
            return False
        except Exception as exc:
            self.last_relocalization_error = f"{type(exc).__name__}: {exc}"
            return False

    def _try_candidate(
        self,
        frame: Frame,
        keyframe: KeyFrame,
        original_pose: np.ndarray,
        original_points: list,
    ) -> bool:
        idxs_frame, idxs_kf = self.match_frame_to_keyframe(frame, keyframe)
        self.num_relocalization_matches = max(self.num_relocalization_matches, len(idxs_frame))

        if len(idxs_frame) < Parameters.kRelocalizationMinKpsMatches:
            return False

        points_3d_w, points_2d, sigmas2, idxs_frame, idxs_kf = self.prepare_input_data_for_pnp(
            frame,
            keyframe,
            idxs_frame,
            idxs_kf,
        )

        if len(points_2d) < 4:
            return False

        pnp = self.estimate_pose_pnp(frame, points_3d_w, points_2d)
        if not pnp.success or pnp.Tcw is None:
            self.last_relocalization_error = pnp.error
            return False

        frame.update_pose(g2o.Isometry3d(pnp.Tcw))
        frame.points = [None] * len(frame.kps)
        frame.outliers = np.zeros(len(frame.kps), dtype=bool)

        inlier_idxs_frame = idxs_frame[pnp.inlier_mask]
        inlier_idxs_kf = idxs_kf[pnp.inlier_mask]

        for idx_frame, idx_kf in zip(inlier_idxs_frame, inlier_idxs_kf):
            if 0 <= idx_frame < len(frame.points) and 0 <= idx_kf < len(keyframe.points):
                frame.points[int(idx_frame)] = keyframe.points[int(idx_kf)]

        self.num_relocalization_inliers = max(self.num_relocalization_inliers, int(pnp.num_inliers))

        num_matched_map_points, mean_error = pose_optimization(frame, verbose=False)
        if not np.isfinite(mean_error) or num_matched_map_points < Parameters.kRelocalizationPoseOpt1MinMatches:
            frame.update_pose(g2o.Isometry3d(original_pose))
            frame.points = list(original_points)
            return False

        self._clear_pose_optimization_outliers(frame)

        if num_matched_map_points < Parameters.kRelocalizationDoPoseOpt2NumInliers:
            already_matched_ref_idxs = self._keyframe_points_to_matched_indices(keyframe, inlier_idxs_kf)
            _, _, num_new_found = ProjectionMatcher.search_keyframe_by_projection(
                keyframe,
                frame,
                max_reproj_distance=Parameters.kRelocalizationMaxReprojectionDistanceMapSearchCoarse,
                max_descriptor_distance=Parameters.kMaxDescriptorDistance,
                ratio_test=Parameters.kRelocalizationFeatureMatchRatioTestLarge,
                already_matched_ref_idxs=already_matched_ref_idxs,
            )

            if num_matched_map_points + num_new_found >= Parameters.kRelocalizationDoPoseOpt2NumInliers:
                pose_before = frame.pose()
                num_matched_map_points, mean_error = pose_optimization(frame, verbose=False)
                if not np.isfinite(mean_error):
                    frame.update_pose(g2o.Isometry3d(pose_before))
                    return False

                self._clear_pose_optimization_outliers(frame)

                if (
                    num_matched_map_points > 30
                    and num_matched_map_points < Parameters.kRelocalizationDoPoseOpt2NumInliers
                ):
                    matched_ref_idxs = frame.get_matched_points_idxs()
                    _, _, num_new_found = ProjectionMatcher.search_keyframe_by_projection(
                        keyframe,
                        frame,
                        max_reproj_distance=Parameters.kRelocalizationMaxReprojectionDistanceMapSearchFine,
                        max_descriptor_distance=0.7 * Parameters.kMaxDescriptorDistance,
                        ratio_test=Parameters.kRelocalizationFeatureMatchRatioTestLarge,
                        already_matched_ref_idxs=matched_ref_idxs,
                    )

                    if num_matched_map_points + num_new_found >= Parameters.kRelocalizationDoPoseOpt2NumInliers:
                        pose_before = frame.pose()
                        num_matched_map_points, mean_error = pose_optimization(frame, verbose=False)
                        if not np.isfinite(mean_error):
                            frame.update_pose(g2o.Isometry3d(pose_before))
                            return False
                        self._clear_pose_optimization_outliers(frame)

        self.num_relocalization_inliers = max(
            self.num_relocalization_inliers,
            int(num_matched_map_points),
        )

        return bool(num_matched_map_points >= Parameters.kRelocalizationDoPoseOpt2NumInliers)

    def match_frame_to_keyframe(self, frame: Frame, keyframe: KeyFrame) -> tuple[np.ndarray, np.ndarray]:
        ensure_frame_feature_arrays(frame)
        ensure_frame_feature_arrays(keyframe)

        valid_kf_idxs = np.asarray(
            [
                idx
                for idx, point in enumerate(getattr(keyframe, "points", []))
                if point is not None and not (hasattr(point, "is_bad") and point.is_bad())
            ],
            dtype=np.int32,
        )

        if len(valid_kf_idxs) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        database = self._active_keyframe_database or self.keyframe_database
        if database is not None and getattr(database, "available", False):
            try:
                database.compute_bow(frame)
                database.compute_bow(keyframe)
                bow_result = BoWGuidedMatcher(database.voc).match(
                    frame,
                    keyframe,
                    valid_idxs2=valid_kf_idxs,
                    max_descriptor_distance=Parameters.kMaxDescriptorDistance,
                    ratio_test=Parameters.kRelocalizationFeatureMatchRatioTest,
                    orientation_check=True,
                )
                self.last_match_diagnostics = bow_result.diagnostics
                self.last_bow_guided_matching_available = bow_result.available
                if bow_result.available:
                    return bow_result.idxs1, bow_result.idxs2
            except Exception as exc:
                self.last_relocalization_error = f"BoW-guided matching unavailable: {exc}"

        self.last_fallback_descriptor_matching = True
        if FeatureTrackerShared.feature_matcher is None:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        matches = FeatureTrackerShared.feature_matcher.match(
            frame.img,
            keyframe.img,
            frame.des,
            keyframe.des[valid_kf_idxs],
            kps1=frame.kps,
            kps2=[keyframe.kps[i] for i in valid_kf_idxs],
            ratio_test=Parameters.kRelocalizationFeatureMatchRatioTest,
        )

        if matches.idxs1 is None or matches.idxs2 is None or len(matches.idxs1) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        idxs_frame = np.asarray(matches.idxs1, dtype=np.int32)
        idxs_kf = valid_kf_idxs[np.asarray(matches.idxs2, dtype=np.int32)]

        if FeatureTrackerShared.oriented_features and len(idxs_frame) > 0:
            valid = RotationHistogram.filter_matches_with_histogram_orientation(
                idxs_frame,
                idxs_kf,
                frame.angles,
                keyframe.angles,
            )
            idxs_frame = idxs_frame[valid]
            idxs_kf = idxs_kf[valid]

        return idxs_frame, idxs_kf

    def prepare_input_data_for_pnp(
        self,
        frame: Frame,
        keyframe: KeyFrame,
        idxs_frame,
        idxs_kf,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxs_frame = np.asarray(idxs_frame, dtype=np.int32).reshape(-1)
        idxs_kf = np.asarray(idxs_kf, dtype=np.int32).reshape(-1)

        points_3d = []
        points_2d = []
        sigmas2 = []
        valid_frame_idxs = []
        valid_kf_idxs = []

        level_sigmas2 = getattr(FeatureTrackerShared.feature_manager, "level_sigmas2", None)

        for idx_frame, idx_kf in zip(idxs_frame, idxs_kf):
            if idx_frame < 0 or idx_frame >= len(frame.kps):
                continue
            if idx_kf < 0 or idx_kf >= len(keyframe.points):
                continue

            point = keyframe.points[int(idx_kf)]
            if point is None or (hasattr(point, "is_bad") and point.is_bad()):
                continue

            try:
                point_w = np.asarray(point.get_position(), dtype=np.float64).reshape(3)
            except Exception:
                continue

            if not np.all(np.isfinite(point_w)):
                continue

            uv = np.asarray(frame.kpsu[int(idx_frame)].pt, dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(uv)):
                continue

            octave = int(frame.octaves[int(idx_frame)]) if len(frame.octaves) > idx_frame else 0
            if level_sigmas2 is not None and len(level_sigmas2) > 0:
                octave = max(0, min(octave, len(level_sigmas2) - 1))
                sigma2 = float(level_sigmas2[octave])
            else:
                sigma2 = 1.0

            points_3d.append(point_w)
            points_2d.append(uv)
            sigmas2.append(sigma2)
            valid_frame_idxs.append(int(idx_frame))
            valid_kf_idxs.append(int(idx_kf))

        return (
            np.ascontiguousarray(points_3d, dtype=np.float64).reshape(-1, 3),
            np.ascontiguousarray(points_2d, dtype=np.float64).reshape(-1, 2),
            np.asarray(sigmas2, dtype=np.float64),
            np.asarray(valid_frame_idxs, dtype=np.int32),
            np.asarray(valid_kf_idxs, dtype=np.int32),
        )

    def estimate_pose_pnp(self, frame: Frame, points_3d_w: np.ndarray, points_2d: np.ndarray) -> PnPResult:
        points_3d_w = np.ascontiguousarray(points_3d_w, dtype=np.float64).reshape(-1, 3)
        points_2d = np.ascontiguousarray(points_2d, dtype=np.float64).reshape(-1, 2)

        if len(points_3d_w) < 4 or len(points_2d) < 4:
            return PnPResult(False, None, np.zeros(len(points_2d), dtype=bool), 0, "too few PnP correspondences")

        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d_w,
                points_2d,
                frame.camera.K,
                None,
                iterationsCount=300,
                reprojectionError=8.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error as exc:
            return PnPResult(False, None, np.zeros(len(points_2d), dtype=bool), 0, str(exc))

        if not ok or inliers is None or len(inliers) < 4:
            return PnPResult(False, None, np.zeros(len(points_2d), dtype=bool), 0, "PnP RANSAC failed")

        inlier_mask = np.zeros(len(points_2d), dtype=bool)
        inlier_mask[np.asarray(inliers, dtype=np.int32).reshape(-1)] = True

        try:
            cv2.solvePnPRefineLM(
                points_3d_w[inlier_mask],
                points_2d[inlier_mask],
                frame.camera.K,
                None,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

        Rcw, _ = cv2.Rodrigues(rvec)
        Tcw = np.eye(4, dtype=np.float64)
        Tcw[:3, :3] = Rcw
        Tcw[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)

        if not np.all(np.isfinite(Tcw)):
            return PnPResult(False, None, inlier_mask, int(np.sum(inlier_mask)), "non-finite PnP pose")

        return PnPResult(True, Tcw, inlier_mask, int(np.sum(inlier_mask)))

    @staticmethod
    def _normalize_candidates(candidates, keyframes_map: Optional[dict[int, KeyFrame]] = None) -> list[KeyFrame]:
        if candidates is None:
            return []

        keyframes_map = keyframes_map or {}
        normalized = []
        for candidate in candidates:
            if isinstance(candidate, KeyFrame):
                keyframe = candidate
            elif isinstance(candidate, (int, np.integer)):
                keyframe = keyframes_map.get(int(candidate))
            else:
                keyframe = candidate

            if keyframe is None:
                continue
            if hasattr(keyframe, "is_bad") and keyframe.is_bad():
                continue
            if keyframe not in normalized:
                normalized.append(keyframe)
        return normalized

    @staticmethod
    def _clear_pose_optimization_outliers(frame: Frame) -> None:
        if not hasattr(frame, "outliers") or len(frame.outliers) != len(frame.points):
            return
        for idx, is_outlier in enumerate(frame.outliers):
            if bool(is_outlier):
                frame.points[idx] = None

    @staticmethod
    def _keyframe_points_to_matched_indices(keyframe: KeyFrame, keyframe_point_indices: Iterable[int]) -> np.ndarray:
        matched_idxs = np.asarray(keyframe.get_matched_points_idxs(), dtype=np.int32).reshape(-1)
        if len(matched_idxs) == 0:
            return np.array([], dtype=np.int32)

        index_map = {int(kf_idx): compact_idx for compact_idx, kf_idx in enumerate(matched_idxs)}
        converted = [
            index_map[int(kf_idx)]
            for kf_idx in np.asarray(list(keyframe_point_indices), dtype=np.int32).reshape(-1)
            if int(kf_idx) in index_map
        ]
        return np.asarray(converted, dtype=np.int32)
