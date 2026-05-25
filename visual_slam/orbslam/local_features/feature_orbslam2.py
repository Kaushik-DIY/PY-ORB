"""
ORB feature extractor wrapper with a detector-style interface.
This module exposes a uniform API for the configured ORB extraction backend.
"""

from __future__ import annotations

import numpy as np

from visual_slam.orbslam.local_features.extractor_backends import (
    DEFAULT_EXTRACTOR_BACKEND,
    FeatureExtractionResult,
    create_extractor_backend,
)


# Present the configured ORB extractor as a detector-style feature object.
class Orbslam2Feature2D:
    def __init__(
        self,
        num_features: int = 2000,
        scale_factor: float = 1.2,
        num_levels: int = 8,
        ini_th_fast: int = 20,
        min_th_fast: int = 7,
        deterministic: bool = False,
        backend_name: str = DEFAULT_EXTRACTOR_BACKEND,
    ):
        self.num_features = int(num_features)
        self.scale_factor = float(scale_factor)
        self.num_levels = int(num_levels)
        self.ini_th_fast = int(ini_th_fast)
        self.min_th_fast = int(min_th_fast)
        self.deterministic = bool(deterministic)
        self.backend_name = backend_name

        self.backend = create_extractor_backend(
            backend_name,
            num_features=self.num_features,
            scale_factor=self.scale_factor,
            num_levels=self.num_levels,
            ini_th_fast=self.ini_th_fast,
            min_th_fast=self.min_th_fast,
            deterministic=self.deterministic,
        )

    def setMaxFeatures(self, num_features: int) -> None:
        self.num_features = int(num_features)
        self.backend.setMaxFeatures(self.num_features)

    def detect(self, image: np.ndarray, mask=None):
        return self.backend.detect(image, mask)

    def compute(self, image: np.ndarray, keypoints):
        return self.backend.compute(image, keypoints)

    def detectAndCompute(self, image: np.ndarray, mask=None):
        return self.extract(image, mask).as_tuple()

    def extract(self, image: np.ndarray, mask=None) -> FeatureExtractionResult:
        return self.backend.extract(image, mask)
