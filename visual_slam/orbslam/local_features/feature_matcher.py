"""
Descriptor matcher utilities for local feature tracking.
This module scores feature correspondence candidates and returns index-aligned matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


# Enumerate the matcher modes supported by the local feature stack.
class FeatureMatcherTypes(Enum):
    NONE = 0
    BF = 1
    DES_BF = 2


# Store the index-aligned result of one descriptor matching pass.
@dataclass
class FeatureMatchingResult:
    idxs1: np.ndarray
    idxs2: np.ndarray
    distances: np.ndarray
    matches: list


# Define the base interface for descriptor matchers.
class FeatureMatcher:
    def __init__(
        self,
        matcher_type: FeatureMatcherTypes = FeatureMatcherTypes.DES_BF,
        norm_type: int = cv2.NORM_HAMMING,
        ratio_test: float = 0.7,
        cross_check: bool = False,
    ):
        self.matcher_type = matcher_type
        self.norm_type = norm_type
        self.ratio_test = float(ratio_test)
        self.cross_check = bool(cross_check)
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=cross_check)

    def match(
        self,
        image1,
        image2,
        des1: Optional[np.ndarray],
        des2: Optional[np.ndarray],
        kps1=None,
        kps2=None,
        ratio_test: Optional[float] = None,
        row_matching: bool = False,
        max_disparity: float = -1,
    ) -> FeatureMatchingResult:
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return FeatureMatchingResult(
                idxs1=np.array([], dtype=np.int32),
                idxs2=np.array([], dtype=np.int32),
                distances=np.array([], dtype=np.float32),
                matches=[],
            )

        ratio = self.ratio_test if ratio_test is None else float(ratio_test)

        raw = self.matcher.knnMatch(des1, des2, k=2)

        good = []
        for pair in raw:
            if len(pair) < 2:
                continue

            m, n = pair
            if m.distance >= ratio * n.distance:
                continue

            if row_matching and kps1 is not None and kps2 is not None:
                y1 = kps1[m.queryIdx].pt[1]
                y2 = kps2[m.trainIdx].pt[1]
                if abs(y1 - y2) > 1.1:
                    continue

            if max_disparity is not None and max_disparity >= 0 and kps1 is not None and kps2 is not None:
                x1 = kps1[m.queryIdx].pt[0]
                x2 = kps2[m.trainIdx].pt[0]
                disparity = x1 - x2
                if disparity < 0 or disparity > max_disparity:
                    continue

            good.append(m)

        return FeatureMatchingResult(
            idxs1=np.array([m.queryIdx for m in good], dtype=np.int32),
            idxs2=np.array([m.trainIdx for m in good], dtype=np.int32),
            distances=np.array([m.distance for m in good], dtype=np.float32),
            matches=good,
        )


# Run brute-force descriptor matching with ratio and distance filtering.
class BfFeatureMatcher(FeatureMatcher):
    def __init__(self, norm_type: int = cv2.NORM_HAMMING, ratio_test: float = 0.7):
        super().__init__(
            matcher_type=FeatureMatcherTypes.DES_BF,
            norm_type=norm_type,
            ratio_test=ratio_test,
            cross_check=False,
        )


def feature_matcher_factory(
    matcher_type: FeatureMatcherTypes = FeatureMatcherTypes.DES_BF,
    norm_type: int = cv2.NORM_HAMMING,
    ratio_test: float = 0.7,
    **kwargs,
) -> FeatureMatcher:
    if matcher_type in (FeatureMatcherTypes.BF, FeatureMatcherTypes.DES_BF):
        return BfFeatureMatcher(norm_type=norm_type, ratio_test=ratio_test)
    raise ValueError(f"Unsupported matcher_type for ORB-SLAM subset: {matcher_type}")
