"""
Predefined feature-tracker configurations.
This module keeps the default ORB settings used by the RGB-D pipeline.
"""

from __future__ import annotations

from visual_slam.orbslam.local_features.extractor_backends import DEFAULT_EXTRACTOR_BACKEND
from visual_slam.orbslam.local_features.feature_manager import feature_manager_factory
from visual_slam.orbslam.local_features.feature_tracker import (
    FeatureTrackerTypes,
    feature_tracker_factory,
)
from visual_slam.orbslam.local_features.feature_types import (
    FeatureDescriptorTypes,
    FeatureDetectorTypes,
)
from visual_slam.orbslam.local_features.feature_matcher import FeatureMatcherTypes
from visual_slam.orbslam.slam.config_parameters import Parameters


kNumFeatures = Parameters.kNumFeatures
kDefaultRatioTest = Parameters.kFeatureMatchDefaultRatioTest
kTrackerType = FeatureTrackerTypes.DES_BF


# Collect named tracker presets used by the RGB-D front-end.
class FeatureTrackerConfigs:
    @staticmethod
    def get_config_from_name(config_name: str):
        config_dict = getattr(FeatureTrackerConfigs, config_name, None)
        if config_dict is None:
            raise ValueError(f"FeatureTrackerConfigs: No configuration found for '{config_name}'")
        return dict(config_dict)

    ORB2 = dict(
        num_features=kNumFeatures,
        num_levels=8,
        scale_factor=1.2,
        detector_type=FeatureDetectorTypes.ORB2,
        descriptor_type=FeatureDescriptorTypes.ORB2,
        sigma_level0=Parameters.kSigmaLevel0,
        match_ratio_test=kDefaultRatioTest,
        tracker_type=kTrackerType,
        matcher_type=FeatureMatcherTypes.DES_BF,
        deterministic=False,
        extractor_backend=DEFAULT_EXTRACTOR_BACKEND,
    )


def create_orb2_feature_tracker(**overrides):
    config = FeatureTrackerConfigs.get_config_from_name("ORB2")
    config.update(overrides)
    return feature_tracker_factory(**config)
