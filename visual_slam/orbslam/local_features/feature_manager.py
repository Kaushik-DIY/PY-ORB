"""
Feature extraction manager for the ORB front-end.
This module owns detector settings, pyramid statistics, and extractor dispatch.
"""

from __future__ import annotations

import math
from enum import Enum

import cv2
import numpy as np

from visual_slam.orbslam.local_features.extractor_backends import (
    DEFAULT_EXTRACTOR_BACKEND,
    FeatureExtractionResult,
    normalize_backend_name,
)
from visual_slam.orbslam.local_features.feature_orbslam2 import Orbslam2Feature2D
from visual_slam.orbslam.local_features.feature_types import (
    FeatureDescriptorTypes,
    FeatureDetectorTypes,
    FeatureInfo,
)
from visual_slam.orbslam.slam.config_parameters import Parameters


# Enumerate optional post-detection keypoint filtering modes.
class KeyPointFilterTypes(Enum):
    NONE = 0


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    return int(cv2.norm(a, b, cv2.NORM_HAMMING))


def hamming_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)

    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)), dtype=np.float32)

    out = np.empty((len(a), len(b)), dtype=np.float32)
    for i, da in enumerate(a):
        for j, db in enumerate(b):
            out[i, j] = cv2.norm(da, db, cv2.NORM_HAMMING)
    return out


def _default_max_descriptor_distance(descriptor_type) -> int:
    """Return the descriptor distance threshold used by binary feature matching."""
    name = getattr(descriptor_type, "name", str(descriptor_type)).upper()

    if "ORB" in name or "BRISK" in name or "AKAZE" in name:
        return 100

    # Keep a fallback branch for non-binary descriptors even though ORB is the main path.
    return 0



# Manage detector settings, descriptor scale statistics, and extraction calls.
class FeatureManager:
    def __init__(
        self,
        num_features: int = Parameters.kNumFeatures,
        num_levels: int = Parameters.kORBNumLevels,
        scale_factor: float = Parameters.kORBScaleFactor,
        sigma_level0: float = Parameters.kSigmaLevel0,
        detector_type: FeatureDetectorTypes = FeatureDetectorTypes.ORB2,
        descriptor_type: FeatureDescriptorTypes = FeatureDescriptorTypes.ORB2,
        deterministic: bool = Parameters.kORBDeterministic,
        ini_th_fast: int = 20,
        min_th_fast: int = 7,
        extractor_backend: str = DEFAULT_EXTRACTOR_BACKEND,
    ):
        self.num_features = int(num_features)
        self.num_levels = int(num_levels)
        self.scale_factor = float(scale_factor)
        self.inv_scale_factor = 1.0 / self.scale_factor
        self.log_scale_factor = math.log(self.scale_factor)

        self.sigma_level0 = float(sigma_level0)
        self.detector_type = detector_type
        self.descriptor_type = descriptor_type
        self.max_descriptor_distance = _default_max_descriptor_distance(descriptor_type)
        self.norm_type = cv2.NORM_HAMMING
        self.oriented_features = FeatureInfo.is_oriented_features(detector_type)
        self.deterministic = bool(deterministic)
        self.extractor_backend_name = normalize_backend_name(extractor_backend)

        self.init_sigma_levels()

        self.feature = Orbslam2Feature2D(
            num_features=self.num_features,
            scale_factor=self.scale_factor,
            num_levels=self.num_levels,
            ini_th_fast=ini_th_fast,
            min_th_fast=min_th_fast,
            deterministic=deterministic,
            backend_name=self.extractor_backend_name,
        )
        self.extractor_backend = self.feature.backend

        self.descriptor_distance = hamming_distance
        self.descriptor_distances = hamming_distances

        Parameters.kMaxDescriptorDistance = 100

    def init_sigma_levels(self) -> None:
        self.scale_factors = np.ones(self.num_levels, dtype=np.float32)
        self.level_sigmas2 = np.ones(self.num_levels, dtype=np.float32)

        for level in range(1, self.num_levels):
            self.scale_factors[level] = self.scale_factors[level - 1] * self.scale_factor

        self.inv_scale_factors = 1.0 / self.scale_factors
        self.level_sigmas2 = (self.sigma_level0 * self.scale_factors) ** 2
        self.level_sigmas = np.sqrt(self.level_sigmas2)
        self.inv_level_sigmas2 = 1.0 / self.level_sigmas2

    def set_num_features(self, num_features: int) -> None:
        self.num_features = int(num_features)
        self.feature.setMaxFeatures(self.num_features)

    def set_double_num_features(self) -> None:
        self.set_num_features(2 * self.num_features)

    def set_normal_num_features(self) -> None:
        self.set_num_features(Parameters.kNumFeatures)

    def detect(self, image: np.ndarray, mask=None):
        return self.feature.detect(image, mask)

    def compute(self, image: np.ndarray, keypoints):
        return self.feature.compute(image, keypoints)

    def detectAndCompute(self, image: np.ndarray, mask=None):
        return self.extract(image, mask).as_tuple()

    def extract(self, image: np.ndarray, mask=None) -> FeatureExtractionResult:
        return self.extractor_backend.extract(image, mask)

    def debug_print(self) -> None:
        print(
            "FeatureManager("
            f"detector={self.detector_type}, descriptor={self.descriptor_type}, "
            f"num_features={self.num_features}, levels={self.num_levels}, "
            f"scale_factor={self.scale_factor}, extractor_backend={self.extractor_backend_name})"
        )


def feature_manager_factory(**config) -> FeatureManager:
    detector_type = config.get("detector_type", FeatureDetectorTypes.ORB2)
    descriptor_type = config.get("descriptor_type", FeatureDescriptorTypes.ORB2)

    if detector_type != FeatureDetectorTypes.ORB2 or descriptor_type != FeatureDescriptorTypes.ORB2:
        raise ValueError(
            "This ORB-SLAM subset currently supports only "
            "FeatureDetectorTypes.ORB2 + FeatureDescriptorTypes.ORB2."
        )

    return FeatureManager(
        num_features=config.get("num_features", Parameters.kNumFeatures),
        num_levels=config.get("num_levels", Parameters.kORBNumLevels),
        scale_factor=config.get("scale_factor", Parameters.kORBScaleFactor),
        sigma_level0=config.get("sigma_level0", Parameters.kSigmaLevel0),
        detector_type=detector_type,
        descriptor_type=descriptor_type,
        deterministic=config.get("deterministic", Parameters.kORBDeterministic),
        extractor_backend=config.get(
            "extractor_backend",
            config.get("backend_name", config.get("feature_backend", DEFAULT_EXTRACTOR_BACKEND)),
        ),
    )
