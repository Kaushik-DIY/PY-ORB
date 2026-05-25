"""
Top-level RGB-D SLAM system container.
This module wires the front-end, map, local mapping, loop closing, and optimization services.
"""

from __future__ import annotations

from enum import Enum
import time
import traceback

import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.bow import load_default_vocabulary
from visual_slam.orbslam.slam.keyframe_database import KeyFrameDatabase
from visual_slam.orbslam.slam.local_mapping import LocalMapping
from visual_slam.orbslam.slam.map import Map
from visual_slam.orbslam.slam.sensor_types import SensorType
from visual_slam.orbslam.slam.slam_commons import SlamState
from visual_slam.orbslam.slam.tracking import Tracking
from visual_slam.orbslam.utilities.logging import Printer


kVerbose = True


# Enumerate the top-level operating modes of the SLAM system.
class SlamMode(Enum):
    SLAM = 0
    MAP_BROWSER = 1


# Own and connect the full RGB-D SLAM pipeline.
class Slam:
    """

        1. store camera/config/sensor metadata
        2. initialize feature tracker and FeatureTrackerShared
        3. create map
        4. create local mapping
        5. create tracking
        6. distribute optional config parameters
    """

    def __init__(
        self,
        camera,
        feature_tracker_config: dict | None = None,
        loop_detector_config=None,
        semantic_mapping_config=None,
        sensor_type: SensorType = SensorType.RGBD,
        environment_type=None,
        slam_mode: SlamMode = SlamMode.SLAM,
        config=None,
        headless: bool = True,
        viewer3d=None,
        start_local_mapping_thread: bool | None = None,
        enable_loop_closing: bool = False,
        enable_global_ba: bool = False,
        global_ba_after_loop: bool | None = None,
        global_ba_iterations: int = Parameters.kGlobalBAIterations,
    ):
        self.camera = camera
        self.feature_tracker_config = feature_tracker_config or {}
        self.loop_detector_config = loop_detector_config
        self.semantic_mapping_config = semantic_mapping_config
        self.sensor_type = sensor_type
        self.environment_type = environment_type
        self.slam_mode = slam_mode
        self.headless = headless
        self.viewer3d = viewer3d
        self.runtime_profiler = None

        self.feature_tracker = None
        self.init_feature_tracker(self.feature_tracker_config)

        self.map = Map()
        self.bow_vocabulary = load_default_vocabulary()
        self.keyframe_database = KeyFrameDatabase(self.bow_vocabulary)

        self.semantic_mapping = None
        self.loop_closing = None
        self.GBA = None
        self.GBA_on_demand = None
        self.enable_global_ba = bool(enable_global_ba)
        self.global_ba_after_loop = bool(enable_global_ba if global_ba_after_loop is None else global_ba_after_loop)
        self.global_ba_iterations = int(global_ba_iterations)
        self.volumetric_integrator = None

        self.local_mapping = LocalMapping(self)
        self.tracking = Tracking(self)

        if enable_loop_closing:
            from visual_slam.orbslam.slam.loop_closing import LoopClosing

            self.loop_closing = LoopClosing(self, keyframe_database=self.keyframe_database)

        self.reset_requested = False
        self.has_quit = False
        self.config = None

        self.set_config_params(config)

        # Enable threaded local mapping only when the runner requests it.
        self.start_local_mapping_thread = (
            Parameters.kLocalMappingOnSeparateThread
            if start_local_mapping_thread is None
            else bool(start_local_mapping_thread)
        )

        if self.start_local_mapping_thread:
            self.local_mapping.start_thread()

    def set_config_params(self, config):
        self.config = config

        if config is None:
            return

        far_points_threshold = getattr(config, "far_points_threshold", None)
        use_fov_centers = getattr(config, "use_fov_centers_based_kf_generation", False)
        max_fov_centers_distance = getattr(config, "max_fov_centers_distance", -1)

        if self.tracking is not None:
            self.tracking.far_points_threshold = far_points_threshold
            self.tracking.use_fov_centers_based_kf_generation = use_fov_centers
            self.tracking.max_fov_centers_distance = max_fov_centers_distance

        if self.local_mapping is not None:
            self.local_mapping.far_points_threshold = far_points_threshold
            self.local_mapping.use_fov_centers_based_kf_generation = use_fov_centers
            self.local_mapping.max_fov_centers_distance = max_fov_centers_distance

    def init_feature_tracker(self, feature_tracker_config):
        """
        Initialize ORB2 feature tracker and FeatureTrackerShared.

        For now, this ORB-SLAM subset intentionally supports the ORB2 path only.
        Additional feature-manager choices can be added later if they are needed
        for comparison, but the thesis benchmark target is ORB/RGB-D.
        """
        if feature_tracker_config is None:
            feature_tracker_config = {}

        if "feature_tracker" in feature_tracker_config:
            feature_tracker = feature_tracker_config["feature_tracker"]
        else:
            feature_tracker = create_orb2_feature_tracker(**feature_tracker_config)

        self.feature_tracker = feature_tracker

        FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)

        if self.sensor_type == SensorType.STEREO:
            # Reserved for true stereo camera path. RGB-D target does not need it.
            try:
                feature_tracker_right = create_orb2_feature_tracker(**feature_tracker_config)
                FeatureTrackerShared.set_feature_tracker_right(feature_tracker_right, force=True)
            except TypeError:
                pass

    def request_reset(self):
        self.reset_requested = True

    def reset(self):
        if self.local_mapping is not None:
            self.local_mapping.request_reset()

        if self.tracking is not None:
            self.tracking.reset()

        if self.map is not None:
            self.map.reset()

        self.reset_requested = False

    def reset_session(self):
        self.reset()

    def shutdown(self):
        """Stop background threads cleanly. Call before discarding the Slam object."""
        if self.start_local_mapping_thread and self.local_mapping is not None:
            self.local_mapping.stop_thread()

    def quit(self):
        if self.has_quit:
            return

        self.has_quit = True

        # Give the local-mapping thread a moment to exit cleanly if it was started.
        time.sleep(0.01)

    def __del__(self):
        try:
            self.quit()
        except Exception:
            pass

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

        Delegates to Tracking.track() with the same argument order.
        """
        if self.reset_requested:
            self.reset()

        try:
            return self.tracking.track(
                img,
                img_right,
                depth,
                img_id,
                timestamp,
                mask,
                mask_right,
            )
        except Exception:
            Printer.red("Slam.track(): tracking exception")
            Printer.red(traceback.format_exc())
            raise

    def set_tracking_state(self, state: SlamState):
        self.tracking.state = state

    def bundle_adjust(self):
        from visual_slam.orbslam.slam.global_ba import GlobalBundleAdjuster

        adjuster = GlobalBundleAdjuster(
            self.map,
            rounds=self.global_ba_iterations,
            use_robust_kernel=Parameters.kGBAUseRobustKernel,
        )
        return adjuster.run(loop_kf_id=0)

    def get_final_trajectory(self):
        """Reconstruct the final per-frame trajectory from stored frame-to-keyframe poses."""
        history = self.tracking.tracking_history
        poses = []
        for rel_pose, ref_kf in zip(history.relative_frame_poses, history.kf_references):
            keyframe = ref_kf
            Tcr_accum = np.eye(4, dtype=np.float64)
            depth = 0
            while keyframe.is_bad() and depth < 10:
                Tcr_accum = Tcr_accum @ np.asarray(keyframe.Tcp(), dtype=np.float64)
                parent = keyframe.get_parent()
                if parent is None:
                    keyframe = ref_kf
                    Tcr_accum = np.eye(4, dtype=np.float64)
                    break
                keyframe = parent
                depth += 1

            Tcw_ref = np.asarray(keyframe.Tcw(), dtype=np.float64)
            Tcr_frame = np.asarray(rel_pose.matrix(), dtype=np.float64)
            Tcw_frame = Tcr_frame @ Tcr_accum @ Tcw_ref
            poses.append(Tcw_frame)

        return {
            "poses": poses,
            "timestamps": list(history.timestamps),
            "ids": list(history.ids),
            "slam_states": list(history.slam_states),
        }

    def compute_kf_trajectory_consistency(self) -> dict:
        """
        Diagnostic: compare each keyframe's Tcw() against the reconstructed trajectory
        at the matching timestamp. After correct reconstruction the difference should be
        near-zero (numerical noise only, < 0.001 m).
        """
        traj = self.get_final_trajectory()
        ts_to_pose = dict(zip(traj["timestamps"], traj["poses"]))

        diffs = []
        for kf in list(self.map.keyframes_map.values()):
            if kf is None or kf.is_bad():
                continue
            kf_ts = kf.timestamp
            traj_pose = ts_to_pose.get(kf_ts)
            if traj_pose is None:
                closest_ts = min(ts_to_pose, key=lambda t: abs(t - kf_ts), default=None)
                if closest_ts is None or abs(closest_ts - kf_ts) > 0.05:
                    continue
                traj_pose = ts_to_pose[closest_ts]

            t_kf = np.asarray(kf.Tcw(), dtype=np.float64)[:3, 3]
            t_traj = np.asarray(traj_pose, dtype=np.float64)[:3, 3]
            diffs.append(float(np.linalg.norm(t_kf - t_traj)))

        if not diffs:
            return {"n_checked": 0, "max_diff_m": None, "median_diff_m": None}

        return {
            "n_checked": len(diffs),
            "max_diff_m": float(np.max(diffs)),
            "median_diff_m": float(np.median(diffs)),
            "mean_diff_m": float(np.mean(diffs)),
        }

    def get_tracking_state(self):
        return self.tracking.state

    def is_ok(self):
        return self.tracking.state == SlamState.OK
