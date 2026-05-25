"""
Feature tracking wrapper for frame-to-frame correspondence.
This module couples feature extraction with descriptor matching for the front-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from visual_slam.orbslam.local_features.feature_manager import (
    FeatureManager,
    feature_manager_factory,
)
from visual_slam.orbslam.local_features.feature_matcher import (
    FeatureMatcher,
    FeatureMatcherTypes,
    feature_matcher_factory,
)
from visual_slam.orbslam.slam.config_parameters import Parameters


# Enumerate the feature tracking modes exposed by the front-end.
class FeatureTrackerTypes(Enum):
    NONE = 0
    LK = 1
    DES_BF = 2
    DES_FLANN = 3


# Store one frame-to-frame feature tracking result.
@dataclass
class FeatureTrackingResult:
    kps_ref: list
    kps_cur: list
    des_ref: np.ndarray
    des_cur: np.ndarray
    idxs_ref: np.ndarray
    idxs_cur: np.ndarray
    matches: list


# Couple feature extraction and descriptor matching for tracking.
class FeatureTracker:
    def __init__(
        self,
        feature_manager: FeatureManager,
        matcher: FeatureMatcher,
        tracker_type: FeatureTrackerTypes = FeatureTrackerTypes.DES_BF,
        match_ratio_test: float = Parameters.kFeatureMatchDefaultRatioTest,
    ):
        self.feature_manager = feature_manager
        self.matcher = matcher
        self.tracker_type = tracker_type
        self.ratio_test = float(match_ratio_test)

        self.num_features = self.feature_manager.num_features
        self.num_levels = self.feature_manager.num_levels
        self.scale_factor = self.feature_manager.scale_factor
        self.norm_type = self.feature_manager.norm_type
        self.descriptor_distance = self.feature_manager.descriptor_distance
        self.descriptor_distances = self.feature_manager.descriptor_distances

    def set_double_num_features(self) -> None:
        self.feature_manager.set_double_num_features()
        self.num_features = self.feature_manager.num_features

    def set_normal_num_features(self) -> None:
        self.feature_manager.set_normal_num_features()
        self.num_features = self.feature_manager.num_features

    def detectAndCompute(self, image: np.ndarray, mask=None):
        return self.feature_manager.detectAndCompute(image, mask)

    def extract(self, image: np.ndarray, mask=None):
        return self.feature_manager.extract(image, mask)

    def track(self, image_ref, image_cur, kps_ref, des_ref):
        kps_cur, des_cur = self.detectAndCompute(image_cur)
        result = self.matcher.match(
            image_ref,
            image_cur,
            des_ref,
            des_cur,
            kps_ref,
            kps_cur,
            ratio_test=self.ratio_test,
        )

        return FeatureTrackingResult(
            kps_ref=kps_ref,
            kps_cur=kps_cur,
            des_ref=des_ref,
            des_cur=des_cur,
            idxs_ref=result.idxs1,
            idxs_cur=result.idxs2,
            matches=result.matches,
        )


def feature_tracker_factory(**config) -> FeatureTracker:
    tracker_type = config.get("tracker_type", FeatureTrackerTypes.DES_BF)
    if tracker_type != FeatureTrackerTypes.DES_BF:
        raise ValueError("This ORB-SLAM subset currently supports only DES_BF tracking.")

    feature_manager = feature_manager_factory(**config)
    matcher = feature_matcher_factory(
        matcher_type=FeatureMatcherTypes.DES_BF,
        norm_type=feature_manager.norm_type,
        ratio_test=config.get("match_ratio_test", Parameters.kFeatureMatchDefaultRatioTest),
    )

    return FeatureTracker(
        feature_manager=feature_manager,
        matcher=matcher,
        tracker_type=tracker_type,
        match_ratio_test=config.get("match_ratio_test", Parameters.kFeatureMatchDefaultRatioTest),
    )
