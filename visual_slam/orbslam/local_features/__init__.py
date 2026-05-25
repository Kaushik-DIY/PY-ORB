"""
Local feature subsystem exports.
This package exposes detector, descriptor, matcher, and tracker building blocks.
"""

from visual_slam.orbslam.local_features.feature_types import (
    FeatureDescriptorTypes,
    FeatureDetectorTypes,
    FeatureInfo,
)
from visual_slam.orbslam.local_features.feature_matcher import (
    FeatureMatcher,
    FeatureMatcherTypes,
    FeatureMatchingResult,
    feature_matcher_factory,
)
from visual_slam.orbslam.local_features.extractor_backends import (
    BackendUnavailableError,
    DEFAULT_EXTRACTOR_BACKEND,
    FeatureExtractionResult,
    OpenCVORBBackend,
    PySLAMORB2Backend,
    create_extractor_backend,
)
from visual_slam.orbslam.local_features.feature_manager import (
    FeatureManager,
    feature_manager_factory,
)
from visual_slam.orbslam.local_features.feature_tracker import (
    FeatureTracker,
    FeatureTrackerTypes,
    FeatureTrackingResult,
    feature_tracker_factory,
)
from visual_slam.orbslam.local_features.feature_tracker_configs import (
    FeatureTrackerConfigs,
    create_orb2_feature_tracker,
)
from visual_slam.orbslam.local_features.feature_orbslam2 import Orbslam2Feature2D

__all__ = [
    "FeatureDescriptorTypes",
    "FeatureDetectorTypes",
    "FeatureInfo",
    "FeatureMatcher",
    "FeatureMatcherTypes",
    "FeatureMatchingResult",
    "feature_matcher_factory",
    "BackendUnavailableError",
    "DEFAULT_EXTRACTOR_BACKEND",
    "FeatureExtractionResult",
    "OpenCVORBBackend",
    "PySLAMORB2Backend",
    "create_extractor_backend",
    "FeatureManager",
    "feature_manager_factory",
    "FeatureTracker",
    "FeatureTrackerTypes",
    "FeatureTrackingResult",
    "feature_tracker_factory",
    "FeatureTrackerConfigs",
    "create_orb2_feature_tracker",
    "Orbslam2Feature2D",
]
