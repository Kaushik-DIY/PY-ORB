"""
Shared feature front-end state for the SLAM system.
This module stores the active tracker, feature manager, and orientation settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from visual_slam.orbslam.local_features.feature_matcher import FeatureMatcherTypes
from visual_slam.orbslam.slam.config_parameters import Parameters

if TYPE_CHECKING:
    from visual_slam.orbslam.local_features.feature_tracker import FeatureTracker
    from visual_slam.orbslam.local_features.feature_matcher import FeatureMatcher
    from visual_slam.orbslam.local_features.feature_manager import FeatureManager


# Cache the feature-manager properties needed across the SLAM modules.
class SlamFeatureManagerInfo:
    """Minimal information about the feature manager used by SLAM."""

    def __init__(self, slam=None, feature_manager: "FeatureManager | None" = None):
        self.feature_descriptor_type = None
        self.feature_descriptor_norm_type = None
        self.max_descriptor_distance = None

        if slam is not None:
            assert slam.feature_tracker is not None
            assert slam.feature_tracker.feature_manager is not None
            self.feature_descriptor_type = slam.feature_tracker.feature_manager.descriptor_type
            self.feature_descriptor_norm_type = slam.feature_tracker.feature_manager.norm_type
            self.max_descriptor_distance = slam.feature_tracker.feature_manager.max_descriptor_distance
        elif feature_manager is not None:
            self.feature_descriptor_type = feature_manager.descriptor_type
            self.feature_descriptor_norm_type = feature_manager.norm_type
            self.max_descriptor_distance = feature_manager.max_descriptor_distance


# Expose the active front-end components as shared global state.
class FeatureTrackerShared:
    feature_tracker: "FeatureTracker | None" = None
    feature_manager: "FeatureManager | None" = None
    feature_matcher: "FeatureMatcher | None" = None

    descriptor_distance = None
    descriptor_distances = None
    oriented_features = False

    feature_tracker_right: "FeatureTracker | None" = None

    _is_cpp_used = False
    _is_cpp_available = False
    _is_cpp_initialized = False
    _cpp_module_parameters = None

    @staticmethod
    def set_feature_tracker(feature_tracker, force: bool = False) -> None:
        if not force and FeatureTrackerShared.feature_tracker is not None:
            raise Exception("FeatureTrackerShared: Tracker is already set!")

        FeatureTrackerShared.feature_tracker = feature_tracker
        FeatureTrackerShared.feature_manager = feature_tracker.feature_manager
        FeatureTrackerShared.feature_matcher = feature_tracker.matcher
        FeatureTrackerShared.descriptor_distance = feature_tracker.feature_manager.descriptor_distance
        FeatureTrackerShared.descriptor_distances = feature_tracker.feature_manager.descriptor_distances
        FeatureTrackerShared.oriented_features = feature_tracker.feature_manager.oriented_features

        if hasattr(FeatureTrackerShared.feature_manager, "max_descriptor_distance"):
            Parameters.kMaxDescriptorDistance = int(FeatureTrackerShared.feature_manager.max_descriptor_distance)
        else:
            Parameters.kMaxDescriptorDistance = 100

    @staticmethod
    def set_feature_tracker_right(feature_tracker, force: bool = False) -> None:
        if not force and FeatureTrackerShared.feature_tracker_right is not None:
            raise Exception("FeatureTrackerShared: Tracker-right is already set!")
        FeatureTrackerShared.feature_tracker_right = feature_tracker

    @staticmethod
    def init_cpp_module(feature_tracker) -> None:
        FeatureTrackerShared._is_cpp_used = False
        FeatureTrackerShared._is_cpp_available = False
        FeatureTrackerShared._is_cpp_initialized = False

    @staticmethod
    def setup_feature_detection_callbacks(module_type: str, module) -> None:
        raise NotImplementedError("C++ callbacks are not used in this ORB-SLAM Python subset.")

    @staticmethod
    def init_cpp_module_config_parameters() -> None:
        return

    @staticmethod
    def update_cpp_module_dynamic_config_parameters() -> None:
        return

    @staticmethod
    def clear_cpp_module_callbacks() -> None:
        return

    @staticmethod
    def reset() -> None:
        FeatureTrackerShared.feature_tracker = None
        FeatureTrackerShared.feature_manager = None
        FeatureTrackerShared.feature_matcher = None
        FeatureTrackerShared.descriptor_distance = None
        FeatureTrackerShared.descriptor_distances = None
        FeatureTrackerShared.oriented_features = False
        FeatureTrackerShared.feature_tracker_right = None
