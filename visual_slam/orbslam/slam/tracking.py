"""
Tracking front-end and frame-to-frame state machine.
This module estimates poses, tracks local structure, and decides when to create keyframes.
"""

from __future__ import annotations

from contextlib import nullcontext
from itertools import chain
import time

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.frame import Frame, ensure_frame_feature_arrays
from visual_slam.orbslam.slam.geometry_matchers import ProjectionMatcher
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.map import Map
from visual_slam.orbslam.slam.map_point import MapPoint
from visual_slam.orbslam.slam.motion_model import MotionModel
from visual_slam.orbslam.slam.optimizer_g2o import pose_optimization as g2o_pose_optimization
from visual_slam.orbslam.slam.relocalizer import Relocalizer
from visual_slam.orbslam.slam.rotation_histogram import RotationHistogram
from visual_slam.orbslam.slam.sensor_types import SensorType
from visual_slam.orbslam.slam.slam_commons import SlamState
from visual_slam.orbslam.slam.tracking_core import TrackingCore
from visual_slam.orbslam.utilities.logging import Printer


kVerbose = True
kUseDynamicDesDistanceTh = Parameters.kUseDynamicDesDistanceTh
kUseMotionModel = Parameters.kUseMotionModel or Parameters.kUseSearchFrameByProjection
kUseSearchFrameByProjection = (
    Parameters.kUseSearchFrameByProjection and not Parameters.kUseEssentialMatrixFitting
)
kNumMinInliersPoseOptimizationTrackFrame = 10
kNumMinInliersPoseOptimizationTrackLocalMap = 20
kNumMinInliersTrackLocalMapForNotWaitingLocalMappingIdle = 50
kNumMinObsForKeyFrameDefault = 3
kMinDepth = Parameters.kMinDepth


# Store the per-frame pose history needed for final trajectory reconstruction.
class TrackingHistory(object):
    def __init__(self):
        self.relative_frame_poses = []
        self.kf_references = []
        self.timestamps = []
        self.ids = []
        self.slam_states = []

    def reset(self):
        self.relative_frame_poses.clear()
        self.kf_references.clear()
        self.timestamps.clear()
        self.ids.clear()
        self.slam_states.clear()

    def memory_stats(self) -> dict:
        """Return lightweight counts for memory reporting. Does not store extra data."""
        unique_kfs = set()
        for kf in self.kf_references:
            if kf is not None:
                unique_kfs.add(id(kf))
        return {
            "num_history_entries": len(self.relative_frame_poses),
            "num_timestamps": len(self.timestamps),
            "num_unique_kf_references": len(unique_kfs),
        }


# Run the frame-by-frame tracking front-end and manage keyframe decisions.
class Tracking:
    def __init__(self, slam):
        self.slam = slam
        self.motion_model = MotionModel()
        self.relocalizer = Relocalizer(self.map)

        self.descriptor_distance_sigma = FeatureTrackerShared.feature_manager.max_descriptor_distance
        self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMap
        if self.sensor_type == SensorType.RGBD:
            self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMapRgbd

        _max_kf_override = int(getattr(Parameters, "kMaxFramesBetweenKeyframesRgbd", -1))
        self.max_frames_between_kfs = (
            _max_kf_override if _max_kf_override > 0
            else (int(self.camera.fps) if getattr(self.camera, "fps", None) is not None else 1)
        )
        self.max_frames_between_kfs_after_reloc = self.max_frames_between_kfs
        self.min_frames_between_kfs = 0

        self.far_points_threshold = None
        self.use_fov_centers_based_kf_generation = False
        self.max_fov_centers_distance = -1

        self.state = SlamState.NO_IMAGES_YET

        self.num_matched_kps = None
        self.num_inliers = None
        self.num_matched_map_points = None
        self.num_matched_map_points_in_last_pose_opt = None
        self.num_kf_ref_tracked_points = None

        self.last_num_static_stereo_map_points = None
        self.total_num_static_stereo_map_points = 0
        self.last_reloc_frame_id = -float("inf")

        self.pose_is_ok = False
        self.mean_pose_opt_chi2_error = None
        self.predicted_pose = None
        self.velocity = None

        self.f_cur: Frame | None = None
        self.idxs_cur = None
        self.f_ref: Frame | None = None
        self.idxs_ref = None

        self.kf_ref = None
        self.kf_last = None
        self.kid_last_BA = -1

        self.local_keyframes = []
        self.local_points = []
        self.vo_points: list[MapPoint] = []
        self.tracking_history = TrackingHistory()
        self.profile_local_map = False
        self.local_map_profile_rows: list[dict] = []
        self.profile_keyframes = False
        self.keyframe_decision_rows: list[dict] = []

        self.init_history = True
        self.poses = []
        self.pose_timestamps = []
        self.traj3d_est = []

    @property
    def feature_tracker(self):
        return self.slam.feature_tracker

    @property
    def map(self) -> Map:
        return self.slam.map

    @property
    def camera(self):
        return self.slam.camera

    @property
    def sensor_type(self):
        return self.slam.sensor_type

    @property
    def local_mapping(self):
        return getattr(self.slam, "local_mapping", None)

    def _profile_section(self, name: str):
        profiler = getattr(self.slam, "runtime_profiler", None)
        if profiler is None:
            return nullcontext()
        return profiler.section(name)

    def _append_keyframe_decision(self, **row):
        if not getattr(self, "profile_keyframes", False):
            return
        defaults = {
            "frame_id": getattr(self.f_cur, "id", -1) if self.f_cur is not None else -1,
            "timestamp": getattr(self.f_cur, "timestamp", None) if self.f_cur is not None else None,
            "num_keyframes": self.map.num_keyframes() if getattr(self, "map", None) is not None else 0,
            "last_keyframe_id": self._object_id(self.kf_last),
            "frames_since_last_kf": -1,
            "min_frames_between_kfs": getattr(self, "min_frames_between_kfs", 0),
            "max_frames_between_kfs": getattr(self, "max_frames_between_kfs", 0),
            "sensor_type": getattr(getattr(self, "sensor_type", None), "name", str(getattr(self, "sensor_type", ""))),
            "local_mapping_idle": True,
            "local_mapping_accepting": True,
            "local_mapping_queue_size": 0,
            "local_mapping_abort_requested": False,
            "num_ref_tracked": 0,
            "ref_min_obs": 0,
            "num_matched_cur": 0,
            "num_tracked_close": 0,
            "num_non_tracked_close": 0,
            "need_to_insert_close": False,
            "ref_ratio": 0.0,
            "th_ref_ratio": 0.0,
            "c1a": False,
            "c1b": False,
            "c1c": False,
            "c2": False,
            "inserted": False,
            "insert_reason": "",
            "reject_reason": "",
        }
        defaults.update(row)
        self.keyframe_decision_rows.append(defaults)

    @staticmethod
    def _abort_requested(local_mapping) -> bool:
        core = getattr(local_mapping, "local_mapping_core", None)
        flag = getattr(core, "opt_abort_flag", None)
        return bool(getattr(flag, "value", False))

    def _is_local_mapping_threaded(self) -> bool:
        local_mapping = self.local_mapping
        return bool(
            local_mapping is not None
            and getattr(local_mapping, "_thread", None) is not None
            and getattr(local_mapping._thread, "is_alive", lambda: False)()
        )

    def _compute_min_frames_between_keyframes(self, is_threaded: bool) -> int:
        if self.sensor_type == SensorType.MONOCULAR:
            return int(getattr(self, "min_frames_between_kfs", 0))

        parameter_name = (
            "kMinFramesBetweenKeyframesThreadedRgbd"
            if is_threaded
            else "kMinFramesBetweenKeyframesSequentialRgbd"
        )
        configured = int(getattr(Parameters, parameter_name, -1))
        if configured >= 0:
            return configured

        if getattr(Parameters, "kUseFpsAwareKeyframeSpacing", True):
            fps = getattr(self.camera, "fps", None)
            if fps is not None and fps > 0:
                return max(3, int(float(Parameters.kMinKeyframeSpacingSeconds) * float(fps)))

        return max(3, int(getattr(self, "min_frames_between_kfs", 0)))

    @staticmethod
    def _new_keyframe_ref_min_obs(num_keyframes: int) -> int:
        override = int(getattr(Parameters, "kNewKeyframeRefMinObs", -1))
        if override > 0:
            return override
        return 2 if int(num_keyframes) <= 2 else kNumMinObsForKeyFrameDefault

    @staticmethod
    def _object_id(obj):
        if obj is None:
            return -1
        return int(getattr(obj, "kid", getattr(obj, "id", -1)))

    def _collect_local_keyframe_votes_from_current_frame(self, f_cur) -> dict:
        keyframe_counter: dict = {}
        if f_cur is None:
            return keyframe_counter

        for p in f_cur.get_matched_good_points():
            if p is None or (hasattr(p, "is_bad") and p.is_bad()):
                continue
            for kf, _ in p.observations():
                if kf is None or (hasattr(kf, "is_bad") and kf.is_bad()):
                    continue
                keyframe_counter[kf] = keyframe_counter.get(kf, 0) + 1
        return keyframe_counter

    @staticmethod
    def _is_good_keyframe(kf) -> bool:
        return kf is not None and not (hasattr(kf, "is_bad") and kf.is_bad())

    @staticmethod
    def _is_good_map_point(point) -> bool:
        if point is None:
            return False
        if hasattr(point, "is_bad") and point.is_bad():
            return False
        replacement = point.get_replacement() if hasattr(point, "get_replacement") else getattr(point, "replacement", None)
        return replacement is None

    def _fallback_reference_keyframe(self):
        for reference in (self.kf_ref, self.kf_last):
            if self._is_good_keyframe(reference):
                return reference
        all_kfs = [kf for kf in self.map.get_keyframes() if self._is_good_keyframe(kf)]
        return all_kfs[-1] if all_kfs else None

    def _select_reference_keyframe_from_votes(self, keyframe_votes: dict, fallback_reference=None):
        if keyframe_votes:
            return max(keyframe_votes, key=keyframe_votes.__getitem__)
        if self._is_good_keyframe(fallback_reference):
            return fallback_reference
        return self._fallback_reference_keyframe()

    def _add_tracking_local_keyframe(self, keyframe_counts: dict, local_keyframes: list, keyframe, frame_id: int) -> bool:
        if not self._is_good_keyframe(keyframe):
            return False
        if keyframe in keyframe_counts:
            return False
        keyframe_counts[keyframe] = 1
        setattr(keyframe, "last_tracking_frame_id", int(frame_id))
        setattr(keyframe, "tracking_vote_count", 1)
        local_keyframes.append(keyframe)
        return True

    def _build_local_keyframes_from_votes(self, keyframe_votes: dict, f_cur, *, num_best: int | None = None) -> list:
        if not keyframe_votes:
            return []

        frame_id = int(getattr(f_cur, "id", -1))
        num_best = int(Parameters.kNumBestCovisibilityKeyFramesTracking if num_best is None else num_best)
        max_local_keyframes = int(Parameters.kMaxNumOfKeyframesInLocalMap)

        keyframe_counts = {
            kf: int(count)
            for kf, count in keyframe_votes.items()
            if self._is_good_keyframe(kf)
        }
        local_keyframes_list = list(keyframe_counts.keys())

        for kf, count in keyframe_counts.items():
            setattr(kf, "last_tracking_frame_id", frame_id)
            setattr(kf, "tracking_vote_count", int(count))

        for kf in local_keyframes_list:
            if len(local_keyframes_list) >= max_local_keyframes:
                break

            for neighbor in kf.get_best_covisible_keyframes(num_best):
                if self._add_tracking_local_keyframe(keyframe_counts, local_keyframes_list, neighbor, frame_id):
                    break

            if getattr(Parameters, "kExpandLocalMapWithChildren", True):
                for child in kf.get_children():
                    if self._add_tracking_local_keyframe(keyframe_counts, local_keyframes_list, child, frame_id):
                        break

            if getattr(Parameters, "kExpandLocalMapWithParent", True):
                self._add_tracking_local_keyframe(keyframe_counts, local_keyframes_list, kf.get_parent(), frame_id)

        return [
            kf
            for kf, _ in sorted(
                keyframe_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:max_local_keyframes]
        ]

    def _collect_local_points_from_keyframes(self, local_keyframes: list, f_cur) -> list:
        frame_id = int(getattr(f_cur, "id", -1))
        local_points = []
        for kf in local_keyframes:
            if not self._is_good_keyframe(kf):
                continue
            for point in kf.get_matched_points():
                if not self._is_good_map_point(point):
                    continue
                if getattr(point, "last_track_reference_frame_id", -1) == frame_id:
                    continue
                point.last_track_reference_frame_id = frame_id
                local_points.append(point)
        return local_points

    def _mark_current_frame_matched_points_seen(self, f_cur) -> int:
        marked = 0
        if f_cur is None:
            return marked
        for point in f_cur.get_matched_good_points():
            if not self._is_good_map_point(point):
                continue
            point.increase_visible()
            point.last_frame_id_seen = f_cur.id
            marked += 1
        return marked

    def _make_local_map_profile_row(
        self,
        f_cur,
        *,
        kf_ref_before,
        kf_ref_after,
        keyframe_votes,
        num_current_matched_points,
        num_current_good_matched_points,
        local_map_build_sec,
        search_map_by_projection_sec,
        pose_optimization_sec,
        track_local_map_sec,
        projection_diagnostics,
    ) -> dict:
        top_voted_kf = None
        top_voted_count = 0
        if keyframe_votes:
            top_voted_kf = max(keyframe_votes, key=keyframe_votes.__getitem__)
            top_voted_count = int(keyframe_votes[top_voted_kf])

        num_local_keyframes = len(self.local_keyframes) if self.local_keyframes is not None else 0
        num_local_keyframes_voted = sum(1 for kf in self.local_keyframes if kf in keyframe_votes)
        num_local_keyframes_expanded = max(0, num_local_keyframes - num_local_keyframes_voted)

        projection_diagnostics = projection_diagnostics or {}
        timestamp = getattr(f_cur, "timestamp", -1.0)
        if timestamp is None:
            timestamp = -1.0

        return {
            "frame_id": int(getattr(f_cur, "id", -1)),
            "timestamp": float(timestamp),
            "kf_ref_id_before": self._object_id(kf_ref_before),
            "kf_ref_id_after": self._object_id(kf_ref_after),
            "num_current_matched_points": int(num_current_matched_points),
            "num_current_good_matched_points": int(num_current_good_matched_points),
            "num_voted_keyframes": int(len(keyframe_votes)),
            "top_voted_kf_id": self._object_id(top_voted_kf),
            "top_voted_count": int(top_voted_count),
            "num_local_keyframes": int(num_local_keyframes),
            "num_local_keyframes_voted": int(num_local_keyframes_voted),
            "num_local_keyframes_expanded": int(num_local_keyframes_expanded),
            "num_local_points": int(len(self.local_points) if self.local_points is not None else 0),
            "num_already_seen_points": int(projection_diagnostics.get("rejected_already_seen", -1)),
            "num_bad_points_skipped": int(projection_diagnostics.get("rejected_bad", -1)),
            "num_visible_projected_points": int(projection_diagnostics.get("visible_projected_points", -1)),
            "num_kd_candidates_total": int(projection_diagnostics.get("kd_candidate_count", -1)),
            "num_descriptor_comparisons": int(projection_diagnostics.get("descriptor_comparisons", -1)),
            "num_projection_matches": int(projection_diagnostics.get("matches", -1)),
            "track_local_map_sec": float(track_local_map_sec),
            "search_map_by_projection_sec": float(search_map_by_projection_sec),
            "pose_optimization_sec": float(pose_optimization_sec),
            "local_map_build_sec": float(local_map_build_sec),
        }

    def reset(self):
        Printer.orange("Tracking: reset...")
        self.motion_model.reset()
        self.state = SlamState.NO_IMAGES_YET

        self.num_matched_kps = None
        self.num_inliers = None
        self.num_matched_map_points = None
        self.num_matched_map_points_in_last_pose_opt = None
        self.num_kf_ref_tracked_points = None

        self.last_num_static_stereo_map_points = None
        self.total_num_static_stereo_map_points = 0
        self.pose_is_ok = False
        self.mean_pose_opt_chi2_error = None
        self.predicted_pose = None
        self.velocity = None

        self.f_cur = None
        self.idxs_cur = None
        self.f_ref = None
        self.idxs_ref = None

        self.kf_ref = None
        self.kf_last = None
        self.kid_last_BA = -1

        self.local_keyframes.clear()
        self.local_points.clear()
        self.vo_points.clear()
        self.tracking_history.reset()

        self.init_history = True
        self.poses.clear()
        self.pose_timestamps.clear()
        self.traj3d_est.clear()

    def pose_optimization(self, f_cur: Frame, name=""):
        pose_before = f_cur.pose()

        num_inliers, mean_chi2 = g2o_pose_optimization(f_cur, verbose=False)

        self.num_matched_map_points_in_last_pose_opt = int(num_inliers)
        self.mean_pose_opt_chi2_error = float(mean_chi2)
        self.pose_is_ok = (
            np.isfinite(mean_chi2)
            and num_inliers >= Parameters.kRelocalizationPoseOpt1MinMatches
        )

        if not self.pose_is_ok:
            f_cur.update_pose(g2o.Isometry3d(pose_before))

        return self.pose_is_ok, self.mean_pose_opt_chi2_error

    def track_previous_frame(self, f_ref: Frame, f_cur: Frame):
        with self._profile_section("tracking.track_previous_frame"):
            ensure_frame_feature_arrays(f_ref)
            ensure_frame_feature_arrays(f_cur)

            is_search_frame_by_projection_failure = False

            use_search_frame_by_projection = (
                self.motion_model.is_ok and kUseSearchFrameByProjection and kUseMotionModel
            )

            if use_search_frame_by_projection:
                search_radius = Parameters.kMaxReprojectionDistanceFrame
                if self.sensor_type != SensorType.STEREO:
                    search_radius = Parameters.kMaxReprojectionDistanceFrameNonStereo

                f_cur.reset_points()

                idxs_ref, idxs_cur, num_found_map_pts = ProjectionMatcher.search_frame_by_projection(
                    f_ref,
                    f_cur,
                    max_reproj_distance=search_radius,
                    max_descriptor_distance=self.descriptor_distance_sigma,
                    ratio_test=Parameters.kMatchRatioTestFrameByProjection,
                    is_monocular=(self.sensor_type == SensorType.MONOCULAR),
                )

                self.num_matched_kps = len(idxs_cur)

                if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchFrameByProjection:
                    f_cur.remove_frame_views(idxs_cur)
                    f_cur.reset_points()

                    idxs_ref, idxs_cur, num_found_map_pts = ProjectionMatcher.search_frame_by_projection(
                        f_ref,
                        f_cur,
                        max_reproj_distance=2 * search_radius,
                        max_descriptor_distance=self.descriptor_distance_sigma,
                        ratio_test=Parameters.kMatchRatioTestFrameByProjection,
                        is_monocular=(self.sensor_type == SensorType.MONOCULAR),
                    )
                    self.num_matched_kps = len(idxs_cur)

                if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchFrameByProjection:
                    f_cur.remove_frame_views(idxs_cur)
                    f_cur.reset_points()
                    is_search_frame_by_projection_failure = True
                    Printer.red("Not enough matches in search frame by projection:", self.num_matched_kps)
                else:
                    self.idxs_ref = idxs_ref
                    self.idxs_cur = idxs_cur

                    pose_before_pos_opt = f_cur.pose()

                    self.pose_optimization(f_cur, "proj-frame-frame")
                    self.num_matched_map_points = f_cur.clean_outlier_map_points()

                    if (
                        not self.pose_is_ok
                        or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame
                    ):
                        Printer.red(
                            "failure in tracking previous frame, # matched map points:",
                            self.num_matched_map_points,
                        )
                        self.pose_is_ok = False
                        f_cur.update_pose(g2o.Isometry3d(pose_before_pos_opt))
                        is_search_frame_by_projection_failure = True

            if not use_search_frame_by_projection or is_search_frame_by_projection_failure:
                self.track_reference_frame(f_ref, f_cur, "match-frame-frame")

            return self.pose_is_ok

    def track_reference_frame(self, f_ref: Frame, f_cur: Frame, name=""):
        with self._profile_section("tracking.track_reference_frame"):
            if f_ref is not None:
                ensure_frame_feature_arrays(f_ref)
            ensure_frame_feature_arrays(f_cur)

            if f_ref is None:
                self.pose_is_ok = False
                Printer.red("[track_reference_frame]: f_ref is None")
                return False

            idxs_ref_map_points = np.asarray(f_ref.get_matched_good_points_idxs(), dtype=int)

            if len(idxs_ref_map_points) == 0:
                self.pose_is_ok = False
                Printer.orange("[track_reference_frame]: reference frame has no valid map points")
                return False

            des_ref = f_ref.des[idxs_ref_map_points]
            kps_ref = [f_ref.kps[i] for i in idxs_ref_map_points]

            matching_result = FeatureTrackerShared.feature_matcher.match(
                f_cur.img,
                f_ref.img,
                f_cur.des,
                des_ref,
                kps1=f_cur.kps,
                kps2=kps_ref,
            )

            idxs_cur = (
                np.asarray(matching_result.idxs1, dtype=int)
                if matching_result.idxs1 is not None
                else np.array([], dtype=int)
            )
            idxs_ref_local = (
                np.asarray(matching_result.idxs2, dtype=int)
                if matching_result.idxs2 is not None
                else np.array([], dtype=int)
            )

            idxs_ref = idxs_ref_map_points[idxs_ref_local] if len(idxs_ref_local) > 0 else np.array([], dtype=int)
            self.num_matched_kps = len(idxs_cur)

            if FeatureTrackerShared.oriented_features and len(idxs_cur) > 0 and len(idxs_ref) > 0:
                valid_match_idxs = RotationHistogram.filter_matches_with_histogram_orientation(
                    idxs_cur,
                    idxs_ref,
                    f_cur.angles,
                    f_ref.angles,
                )

                des_distances = FeatureTrackerShared.descriptor_distances(
                    f_cur.des[idxs_cur],
                    f_ref.des[idxs_ref],
                )

                valid_match_idxs = np.intersect1d(
                    valid_match_idxs,
                    np.where(des_distances <= 0.5 * self.descriptor_distance_sigma)[0],
                )

                if len(valid_match_idxs) > 0:
                    idxs_cur = idxs_cur[valid_match_idxs]
                    idxs_ref = idxs_ref[valid_match_idxs]
                else:
                    idxs_cur = np.array([], dtype=int)
                    idxs_ref = np.array([], dtype=int)

            self.num_matched_kps = len(idxs_cur)

            if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchReferenceFrame:
                self.pose_is_ok = False
                Printer.orange("Not enough matches in reference-frame matching:", self.num_matched_kps)
                return False

            if Parameters.kUseEssentialMatrixFitting:
                idxs_ref, idxs_cur, self.num_inliers = TrackingCore.estimate_pose_by_fitting_ess_mat(
                    f_ref,
                    f_cur,
                    idxs_ref,
                    idxs_cur,
                )
                self.num_matched_kps = len(idxs_cur)

            max_descriptor_distance = (
                self.descriptor_distance_sigma
                if not getattr(f_ref, "is_keyframe", False)
                else 0.5 * self.descriptor_distance_sigma
            )

            num_found_map_pts_inter_frame, idx_ref_prop, idx_cur_prop = (
                TrackingCore.propagate_map_point_matches(
                    f_ref,
                    f_cur,
                    idxs_ref,
                    idxs_cur,
                    max_descriptor_distance=max_descriptor_distance,
                )
            )

            self.idxs_ref = idxs_ref
            self.idxs_cur = idxs_cur

            pose_before_pos_opt = f_cur.pose()

            self.pose_optimization(f_cur, name)
            self.num_matched_map_points = f_cur.clean_outlier_map_points()

            if (
                not self.pose_is_ok
                or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame
            ):
                f_cur.remove_frame_views(idxs_cur)
                f_cur.reset_points()
                Printer.red(
                    f"failure in tracking reference {f_ref.id}, # matched map points:",
                    self.num_matched_map_points,
                )
                self.pose_is_ok = False
                f_cur.update_pose(g2o.Isometry3d(pose_before_pos_opt))
                return False

            return True

    def track_keyframe(self, keyframe: KeyFrame, f_cur: Frame, name="match-frame-keyframe"):
        """
        Track current frame against reference keyframe.

        pose, not blindly from the keyframe pose. This avoids a large pose jump
        when the keyframe is older than the previous frame.
        """
        if self.f_ref is not None:
            f_cur.update_pose(self.f_ref.pose())
        else:
            f_cur.update_pose(keyframe.pose())

        return self.track_reference_frame(keyframe, f_cur, name)

    def _elect_best_kf_ref(self, local_keyframes):
        """Return the local KF sharing the most matched map points with f_cur.

        for need_new_keyframe() must stay in sync with what the current frame
        actually sees, not frozen at last KF creation time.
        """
        if self.f_cur is None or not local_keyframes:
            return None
        counter: dict = {}
        for p in self.f_cur.get_matched_good_points():
            if p is None or (hasattr(p, "is_bad") and p.is_bad()):
                continue
            for kf, _ in p.observations():
                if kf is None or (hasattr(kf, "is_bad") and kf.is_bad()):
                    continue
                counter[kf] = counter.get(kf, 0) + 1
        if not counter:
            return None
        return max(counter, key=counter.__getitem__)

    def update_local_map(self):
        self.f_cur.clean_bad_map_points()

        keyframe_votes = self._collect_local_keyframe_votes_from_current_frame(self.f_cur)
        fallback_reference = self._fallback_reference_keyframe()
        reference = self._select_reference_keyframe_from_votes(
            keyframe_votes,
            fallback_reference=fallback_reference,
        )

        if keyframe_votes:
            self.kf_ref = reference
            self.f_cur.kf_ref = reference
            self.local_keyframes = self._build_local_keyframes_from_votes(
                keyframe_votes,
                self.f_cur,
                num_best=Parameters.kNumBestCovisibilityKeyFramesTracking,
            )
            self.local_points = self._collect_local_points_from_keyframes(
                self.local_keyframes,
                self.f_cur,
            )
        elif reference is not None:
            self.map.update_local_map(
                reference,
                num_best=Parameters.kNumBestCovisibilityKeyFramesTracking,
            )
            self.local_keyframes = self.map.get_local_keyframes().to_list()
            self.local_points = self.map.get_local_points().to_list()
            self.kf_ref = reference
            self.f_cur.kf_ref = reference
        else:
            self.local_keyframes = []
            self.local_points = []

    def track_local_map(self):
        with self._profile_section("tracking.track_local_map"):
            track_local_map_start = time.perf_counter()
            local_map_build_sec = 0.0
            search_map_by_projection_sec = 0.0
            pose_optimization_sec = 0.0
            projection_diagnostics = {} if self.profile_local_map else None
            kf_ref_before = self.kf_ref
            keyframe_votes = {}
            num_current_matched_points = 0
            num_current_good_matched_points = 0

            if self.f_cur is not None:
                ensure_frame_feature_arrays(self.f_cur)

            if self.f_cur is None:
                self.pose_is_ok = False
                return False

            if self.profile_local_map:
                num_current_matched_points = sum(1 for p in getattr(self.f_cur, "points", []) if p is not None)
                current_good_points = self.f_cur.get_matched_good_points()
                num_current_good_matched_points = len(current_good_points)
                keyframe_votes = self._collect_local_keyframe_votes_from_current_frame(self.f_cur)

            local_map_build_start = time.perf_counter()
            self.update_local_map()
            local_map_build_sec = time.perf_counter() - local_map_build_start

            if len(self.local_points) > 0:
                self._mark_current_frame_matched_points_seen(self.f_cur)
                search_start = time.perf_counter()
                found_pts_count, found_pts_fidxs = ProjectionMatcher.search_map_by_projection(
                    self.local_points,
                    self.f_cur,
                    max_reproj_distance=self.reproj_err_frame_map_sigma,
                    max_descriptor_distance=self.descriptor_distance_sigma,
                    ratio_test=Parameters.kMatchRatioTestMap,
                    far_points_threshold=self.far_points_threshold,
                    diagnostics=projection_diagnostics,
                )
                search_map_by_projection_sec = time.perf_counter() - search_start
            else:
                found_pts_count = 0

            pose_before_pos_opt = self.f_cur.pose()
            pose_optimization_start = time.perf_counter()
            self.pose_optimization(self.f_cur, "local-map")
            pose_optimization_sec = time.perf_counter() - pose_optimization_start

            self.num_matched_map_points = self.f_cur.clean_outlier_map_points()
            track_local_map_sec = time.perf_counter() - track_local_map_start

            if self.profile_local_map:
                self.local_map_profile_rows.append(
                    self._make_local_map_profile_row(
                        self.f_cur,
                        kf_ref_before=kf_ref_before,
                        kf_ref_after=self.kf_ref,
                        keyframe_votes=keyframe_votes,
                        num_current_matched_points=num_current_matched_points,
                        num_current_good_matched_points=num_current_good_matched_points,
                        local_map_build_sec=local_map_build_sec,
                        search_map_by_projection_sec=search_map_by_projection_sec,
                        pose_optimization_sec=pose_optimization_sec,
                        track_local_map_sec=track_local_map_sec,
                        projection_diagnostics=projection_diagnostics,
                    )
                )

            if (
                not self.pose_is_ok
                or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackLocalMap
            ):
                self.f_cur.update_pose(g2o.Isometry3d(pose_before_pos_opt))
                self.pose_is_ok = False
                return False

            return True

    def clean_vo_points(self):
        self.vo_points = [p for p in self.vo_points if p is not None and not p.is_bad()]

    def need_new_keyframe(self):
        with self._profile_section("tracking.need_new_keyframe"):
            if self.f_cur is None:
                return False

            decision = {
                "frame_id": getattr(self.f_cur, "id", -1),
                "timestamp": getattr(self.f_cur, "timestamp", None),
                "sensor_type": getattr(self.sensor_type, "name", str(self.sensor_type)),
                "min_frames_between_kfs": self.min_frames_between_kfs,
                "max_frames_between_kfs": self.max_frames_between_kfs,
            }

            if self.kf_last is None:
                self._append_keyframe_decision(
                    **decision,
                    num_keyframes=self.map.num_keyframes(),
                    inserted=True,
                    insert_reason="first_keyframe",
                )
                return True

            if self.sensor_type == SensorType.MONOCULAR:
                self._append_keyframe_decision(
                    **decision,
                    num_keyframes=self.map.num_keyframes(),
                    last_keyframe_id=self._object_id(self.kf_last),
                    reject_reason="monocular_not_supported",
                )
                return False

            # Do not insert KFs while LM is stopped (e.g., during loop correction)
            if self.local_mapping is not None:
                if getattr(self.local_mapping, "stopped", False) or getattr(self.local_mapping, "stop_requested", False):
                    self._append_keyframe_decision(
                        **decision,
                        num_keyframes=self.map.num_keyframes(),
                        last_keyframe_id=self._object_id(self.kf_last),
                        local_mapping_idle=getattr(self.local_mapping, "is_idle", lambda: True)(),
                        local_mapping_accepting=getattr(self.local_mapping, "accept_keyframes", lambda: getattr(self.local_mapping, "is_idle", lambda: True)())(),
                        local_mapping_queue_size=getattr(self.local_mapping, "queue_size", lambda: 0)(),
                        local_mapping_abort_requested=self._abort_requested(self.local_mapping),
                        reject_reason="local_mapping_stopped",
                    )
                    return False

            local_mapping = self.local_mapping
            if local_mapping is not None:
                is_idle = getattr(local_mapping, "is_idle", lambda: True)()
                local_mapping_accepting = getattr(local_mapping, "accept_keyframes", lambda: is_idle)()
                local_mapping_queue_size = getattr(
                    local_mapping,
                    "keyframes_in_queue",
                    getattr(local_mapping, "queue_size", lambda: 0),
                )()
                local_mapping_abort_requested = self._abort_requested(local_mapping)
            else:
                is_idle = True
                local_mapping_accepting = True
                local_mapping_queue_size = 0
                local_mapping_abort_requested = False

            num_kfs = self.map.num_keyframes()
            is_threaded = self._is_local_mapping_threaded()
            self.min_frames_between_kfs = self._compute_min_frames_between_keyframes(is_threaded)
            decision.update(
                {
                    "num_keyframes": num_kfs,
                    "last_keyframe_id": self._object_id(self.kf_last),
                    "min_frames_between_kfs": self.min_frames_between_kfs,
                    "local_mapping_idle": is_idle,
                    "local_mapping_accepting": local_mapping_accepting,
                    "local_mapping_queue_size": local_mapping_queue_size,
                    "local_mapping_abort_requested": local_mapping_abort_requested,
                }
            )

            if num_kfs == 0:
                self._append_keyframe_decision(
                    **decision,
                    inserted=True,
                    insert_reason="map_has_no_keyframes",
                )
                return True

            if (
                self.f_cur.id < self.last_reloc_frame_id + self.max_frames_between_kfs_after_reloc
                and num_kfs > self.max_frames_between_kfs
            ):
                self._append_keyframe_decision(
                    **decision,
                    reject_reason="too_close_after_relocalization",
                )
                return False

            frames_since_last_kf = self.f_cur.id - self.kf_last.id
            decision["frames_since_last_kf"] = frames_since_last_kf

            nMinObs = self._new_keyframe_ref_min_obs(num_kfs)
            # Guard: if kf_ref was culled (set_bad clears all point observations -> returns 0),
            # fall back to kf_last, then any valid map KF, so c2 always has a meaningful baseline.
            kf_ref_for_count = self.kf_ref
            if kf_ref_for_count is None or (hasattr(kf_ref_for_count, "is_bad") and kf_ref_for_count.is_bad()):
                kf_ref_for_count = self.kf_last
            if kf_ref_for_count is None or (hasattr(kf_ref_for_count, "is_bad") and kf_ref_for_count.is_bad()):
                all_kfs = [kf for kf in self.map.get_keyframes() if not kf.is_bad()]
                kf_ref_for_count = all_kfs[-1] if all_kfs else None
            num_ref_tracked = (
                kf_ref_for_count.num_tracked_points(nMinObs) if kf_ref_for_count is not None else 1
            )
            num_ref_tracked = max(1, num_ref_tracked)
            # Keep cache in sync for external readers (e.g., local_BA result update)
            self.num_kf_ref_tracked_points = num_ref_tracked

            # Current frame matched inlier map points
            num_matched_cur = self.num_matched_map_points if self.num_matched_map_points is not None else 0

            # Close point starvation check (RGB-D specific)
            num_tracked_close, num_non_tracked_close, _ = (
                TrackingCore.count_tracked_and_non_tracked_close_points(self.f_cur, self.sensor_type)
            )
            need_to_insert_close = (
                num_tracked_close < Parameters.kNumMinTrackedClosePointsForNewKfNonMonocular
                and num_non_tracked_close > Parameters.kNumMaxNonTrackedClosePointsForNewKfNonMonocular
            )

            # Ratio threshold — more permissive during early map build
            ref_ratio = Parameters.kThNewKfRefRatioStereo
            if num_kfs < 2:
                ref_ratio = 0.4
            if self.sensor_type == SensorType.MONOCULAR:
                ref_ratio = Parameters.kThNewKfRefRatioMonocular

            # Condition 1a: time fallback — max_frames_between_kfs elapsed since last KF
            c1a = frames_since_last_kf >= self.max_frames_between_kfs

            # Condition 1b: min_frames elapsed AND LM is idle
            c1b = frames_since_last_kf >= self.min_frames_between_kfs and is_idle

            # Condition 1c: RGB-D idle-bypass — tracking weak or close points starved
            c1c = self.sensor_type != SensorType.MONOCULAR and (
                num_matched_cur < Parameters.kThNewKfRefRatioNonMonocular * num_ref_tracked
                or need_to_insert_close
            )

            # Condition 2: fewer tracked points than threshold of reference KF (+ absolute min floor)
            c2 = (
                (num_matched_cur < ref_ratio * num_ref_tracked or need_to_insert_close)
                and num_matched_cur > Parameters.kNumMinPointsForNewKf
            )

            decision.update(
                {
                    "num_ref_tracked": num_ref_tracked,
                    "ref_min_obs": nMinObs,
                    "num_matched_cur": num_matched_cur,
                    "num_tracked_close": num_tracked_close,
                    "num_non_tracked_close": num_non_tracked_close,
                    "need_to_insert_close": need_to_insert_close,
                    "ref_ratio": ref_ratio,
                    "th_ref_ratio": ref_ratio,
                    "c1a": c1a,
                    "c1b": c1b,
                    "c1c": c1c,
                    "c2": c2,
                }
            )

            # c1a (max_frames elapsed) is a hard time-based override — insert unconditionally.
            # c1b/c1c still require c2 (tracking quality gate).
            if not (c1a or ((c1b or c1c) and c2)):
                self._append_keyframe_decision(
                    **decision,
                    reject_reason="conditions_not_met",
                )
                return False

            if local_mapping_accepting:
                self._append_keyframe_decision(
                    **decision,
                    inserted=True,
                    insert_reason="local_mapping_accepting",
                )
                return True

            if local_mapping is not None and hasattr(local_mapping, "interrupt_optimization"):
                local_mapping.interrupt_optimization()
                decision["local_mapping_abort_requested"] = True

            if (
                self.sensor_type != SensorType.MONOCULAR
                and local_mapping_queue_size < int(Parameters.kLocalMappingMaxQueueForForcedInsert)
            ):
                self._append_keyframe_decision(
                    **decision,
                    inserted=True,
                    insert_reason="busy_rgbd_queue_below_threshold",
                )
                return True

            self._append_keyframe_decision(
                **decision,
                reject_reason="local_mapping_busy_queue_pressure",
            )
            return False

    def create_new_keyframe(self, img=None):
        with self._profile_section("tracking.create_new_keyframe"):
            if self.f_cur is None:
                return None

            kf_new = KeyFrame(self.f_cur, img=img)
            self.map.add_keyframe(kf_new)
            self._add_keyframe_to_database(kf_new)

            kf_new.init_observations()

            if self.sensor_type != SensorType.MONOCULAR:
                count = self.create_and_add_stereo_map_points_on_new_kf(self.f_cur, kf_new, img=img)
                self.last_num_static_stereo_map_points = count
                self.total_num_static_stereo_map_points += count

            # §4.4: release KF depth image after stereo map-point/depth arrays are computed.
            # Per-keypoint depths and uRs are already stored; full depth_img is no longer needed.
            if not Parameters.kStoreKeyFrameDepthImages:
                kf_new.release_heavy_data(release_rgb=not Parameters.kStoreKeyFrameImages,
                                          release_depth=True, release_kd=False)

            kf_new.update_connections()

            self.kf_last = kf_new
            self.kf_ref = kf_new
            self.f_cur.kf_ref = kf_new
            self.clean_vo_points()

            if self.local_mapping is not None and hasattr(self.local_mapping, "insert_keyframe"):
                self.local_mapping.insert_keyframe(kf_new)

            loop_closing = getattr(self.slam, "loop_closing", None)
            if loop_closing is not None and hasattr(loop_closing, "insert_keyframe"):
                loop_closing.insert_keyframe(kf_new)

            return kf_new

    def relocalize(self, frame: Frame):
        Printer.green(f"Relocalizing frame id: {frame.id}...")
        keyframe_database = getattr(self.slam, "keyframe_database", None)
        return self.relocalizer.relocalize(
            frame,
            keyframe_database=keyframe_database,
            keyframes_map=self.map.keyframes_map,
        )

    def create_vo_points_on_last_frame(self):
        if self.f_ref is None:
            return []
        self.vo_points = TrackingCore.create_vo_points(self.f_ref)
        return self.vo_points

    def create_and_add_stereo_map_points_on_new_kf(self, f, kf, img=None):
        return TrackingCore.create_and_add_stereo_map_points_on_new_kf(f, kf, self.map, img)

    def wait_for_local_mapping(self):
        return True

    def update_history(self):
        if self.f_cur is None:
            return

        self.poses.append(self.f_cur.pose())
        self.pose_timestamps.append(self.f_cur.timestamp)

        if self.kf_ref is not None:
            Tcr = self.f_cur.pose() @ np.linalg.inv(self.kf_ref.pose())
            self.tracking_history.relative_frame_poses.append(g2o.Isometry3d(Tcr))
            self.tracking_history.kf_references.append(self.kf_ref)
            self.tracking_history.timestamps.append(self.f_cur.timestamp)
            self.tracking_history.ids.append(self.f_cur.id)
            self.tracking_history.slam_states.append(self.state)

    def _create_initial_rgbd_map(self, f_cur: Frame, img=None):
        f_cur.update_pose(g2o.Isometry3d(np.eye(4, dtype=np.float64)))

        kf0 = KeyFrame(f_cur, img=img)
        self.map.add_keyframe(kf0)
        self._add_keyframe_to_database(kf0)

        num_created = TrackingCore.create_and_add_stereo_map_points_on_new_kf(
            f_cur,
            kf0,
            self.map,
            img=img,
        )

        # §4.4: release initial KF depth image after map-point/depth arrays are computed.
        if not Parameters.kStoreKeyFrameDepthImages:
            kf0.release_heavy_data(release_rgb=not Parameters.kStoreKeyFrameImages,
                                   release_depth=True, release_kd=False)

        kf0.update_connections()

        self.kf_ref = kf0
        self.kf_last = kf0
        self.f_ref = f_cur
        self.f_cur = f_cur

        self.motion_model.update_pose_from_matrix(f_cur.timestamp, f_cur.pose())

        self.state = SlamState.OK
        self.pose_is_ok = True
        self.num_matched_map_points = num_created

        return num_created >= Parameters.kInitializerNumMinTriangulatedPointsStereo

    def _add_keyframe_to_database(self, keyframe: KeyFrame) -> None:
        keyframe_database = getattr(self.slam, "keyframe_database", None)
        if keyframe_database is None:
            return
        if not getattr(keyframe_database, "available", False):
            return
        keyframe_database.add(keyframe)

    def track(
        self,
        img,
        img_right=None,
        depth=None,
        img_id=None,
        timestamp=None,
        mask=None,
        mask_right=None,
    ):
        """

        Main state flow:
          1. Build current frame.
          2. Initialize first RGB-D keyframe if needed.
          3. Use previous frame from the map as f_ref.
          4. Try previous-frame tracking.
          5. If that fails, fall back to reference-keyframe tracking.
          6. If pose is valid, track local map.
          7. Only then update state, motion model, and keyframe insertion.
        """
        f_cur = Frame(
            camera=self.camera,
            img=img,
            depth_img=depth,
            pose=g2o.Isometry3d(np.eye(4, dtype=np.float64)),
            timestamp=timestamp,
            img_id=img_id,
            img_right=img_right,
        )

        self.f_cur = f_cur
        self.idxs_ref = []
        self.idxs_cur = []
        self.pose_is_ok = False
        self.num_matched_map_points = 0
        self.mean_pose_opt_chi2_error = float("inf")

        # First frame: create initial RGB-D keyframe/map.
        if self.state in (SlamState.NO_IMAGES_YET, SlamState.NOT_INITIALIZED):
            self.map.add_frame(f_cur)
            ok = self._create_initial_rgbd_map(f_cur, img=img)
            self.update_history()
            return ok

        f_ref = self.map.get_frame(-1)
        self.f_ref = f_ref

        self.map.add_frame(f_cur)
        f_cur.kf_ref = self.kf_ref

        if self.state == SlamState.OK:
            if self.f_ref is not None and hasattr(self.f_ref, "check_replaced_map_points"):
                self.f_ref.check_replaced_map_points()

            if self.motion_model.is_ok:
                predicted_pose, _ = self.motion_model.predict_pose(timestamp)
                f_cur.update_pose(predicted_pose)
            elif self.f_ref is not None:
                f_cur.update_pose(self.f_ref.pose())
            elif self.kf_ref is not None:
                f_cur.update_pose(self.kf_ref.pose())

            if (not self.motion_model.is_ok) and self.kf_ref is not None:
                self.track_keyframe(self.kf_ref, f_cur, "match-frame-keyframe")
            else:
                if self.f_ref is not None:
                    self.track_previous_frame(self.f_ref, f_cur)

                if (not self.pose_is_ok) and self.kf_ref is not None:
                    self.track_keyframe(self.kf_ref, f_cur, "match-frame-keyframe")

            if self.pose_is_ok:
                self.track_local_map()

        else:
            if self.state != SlamState.INIT_RELOCALIZE:
                self.state = SlamState.RELOCALIZE

            if self.relocalize(f_cur):
                self.last_reloc_frame_id = f_cur.id
                self.state = SlamState.OK
                self.pose_is_ok = True
                self.kf_ref = f_cur.kf_ref
                self.kf_last = self.kf_ref
                self.map.update_local_map(self.kf_ref)
                self.motion_model.reset()
                Printer.green(
                    f"Relocalization successful, frame id {f_cur.id} "
                    f"reconnected to keyframe id {self.kf_ref.id}"
                )
            else:
                self.pose_is_ok = False
                Printer.red("Relocalization failed")

        if self.pose_is_ok:
            self.state = SlamState.OK
            self.motion_model.update_pose_from_matrix(timestamp, f_cur.pose())
            if f_cur.id <= self.last_reloc_frame_id + 1:
                self.motion_model.is_ok = False

            if self.need_new_keyframe():
                self.create_new_keyframe(img=img)
        else:
            self.state = SlamState.LOST
            self.motion_model.is_ok = False

        # Important: do not assign self.f_ref = f_cur here.
        self.update_history()

        return self.pose_is_ok
