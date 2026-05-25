"""
Feature extractor backend abstraction for ORB-based RGB-D SLAM.
This module normalizes OpenCV and external ORB2 extractors behind one interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Protocol

import cv2
import numpy as np


DEFAULT_EXTRACTOR_BACKEND = "opencv_orb"


# Signal that a requested feature extractor backend cannot be constructed.
class BackendUnavailableError(ImportError):
    """Raised when a requested extractor backend is unavailable."""


# Hold the normalized output of one feature extraction pass.
@dataclass(frozen=True)
class FeatureExtractionResult:
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray
    octaves: np.ndarray | None = None
    angles: np.ndarray | None = None
    sizes: np.ndarray | None = None
    backend_name: str = DEFAULT_EXTRACTOR_BACKEND
    success: bool = True
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "keypoints", list(self.keypoints or []))
        object.__setattr__(self, "descriptors", _normalize_descriptors(self.descriptors))

        kps = self.keypoints
        if self.octaves is None:
            object.__setattr__(
                self,
                "octaves",
                np.array([max(0, int(getattr(kp, "octave", 0))) for kp in kps], dtype=np.int32),
            )
        if self.angles is None:
            object.__setattr__(
                self,
                "angles",
                np.array([float(getattr(kp, "angle", -1.0)) for kp in kps], dtype=np.float32),
            )
        if self.sizes is None:
            object.__setattr__(
                self,
                "sizes",
                np.array([float(getattr(kp, "size", 0.0)) for kp in kps], dtype=np.float32),
            )

    def as_tuple(self) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        return self.keypoints, self.descriptors


# Define the common interface expected from all extractor backends.
class FeatureExtractorBackend(Protocol):
    name: str

    @classmethod
    def is_available(cls) -> bool:
        ...

    def extract(self, image: np.ndarray, mask=None) -> FeatureExtractionResult:
        ...

    def detect(self, image: np.ndarray, mask=None) -> list[cv2.KeyPoint]:
        ...

    def compute(
        self, image: np.ndarray, keypoints: list[cv2.KeyPoint]
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        ...

    def setMaxFeatures(self, num_features: int) -> None:
        ...


# Use OpenCV ORB as the default keypoint detector and descriptor extractor.
class OpenCVORBBackend:
    name = DEFAULT_EXTRACTOR_BACKEND

    def __init__(
        self,
        num_features: int = 2000,
        scale_factor: float = 1.2,
        num_levels: int = 8,
        ini_th_fast: int = 20,
        min_th_fast: int = 7,
        deterministic: bool = False,
    ):
        self.num_features = int(num_features)
        self.scale_factor = float(scale_factor)
        self.num_levels = int(num_levels)
        self.ini_th_fast = int(ini_th_fast)
        self.min_th_fast = int(min_th_fast)
        self.deterministic = bool(deterministic)
        self._orb = self._make_orb()

    @classmethod
    def is_available(cls) -> bool:
        return hasattr(cv2, "ORB_create")

    def _make_orb(self):
        return cv2.ORB_create(
            nfeatures=self.num_features,
            scaleFactor=self.scale_factor,
            nlevels=self.num_levels,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=self.ini_th_fast,
        )

    def setMaxFeatures(self, num_features: int) -> None:
        self.num_features = int(num_features)
        self._orb.setMaxFeatures(self.num_features)

    def detect(self, image: np.ndarray, mask=None) -> list[cv2.KeyPoint]:
        return list(self._orb.detect(_as_gray(image), mask))

    def compute(
        self, image: np.ndarray, keypoints: list[cv2.KeyPoint]
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        kps, des = self._orb.compute(_as_gray(image), keypoints)
        return list(kps or []), _normalize_descriptors(des)

    def extract(self, image: np.ndarray, mask=None) -> FeatureExtractionResult:
        kps, des = self._orb.detectAndCompute(_as_gray(image), mask)
        return FeatureExtractionResult(
            keypoints=list(kps or []),
            descriptors=des,
            backend_name=self.name,
        )


# Wrap the optional external ORB2 extractor backend used by the runner.
class PySLAMORB2Backend:
    name = "pyslam_orb2"

    def __init__(
        self,
        num_features: int = 2000,
        scale_factor: float = 1.2,
        num_levels: int = 8,
        ini_th_fast: int = 20,
        min_th_fast: int = 7,
        deterministic: bool = False,
    ):
        module = _import_orbslam2_features()
        if module is None:
            raise BackendUnavailableError(
                "pyslam_orb2 backend requires the local orbslam2_features module. "
                "Build/install it inside /home/kaushik/slam_ws/.venv before selecting this backend."
            )

        extractor_cls_name = "ORBextractorDeterministic" if deterministic else "ORBextractor"
        extractor_cls = getattr(module, extractor_cls_name)
        self.num_features = int(num_features)
        self.scale_factor = float(scale_factor)
        self.num_levels = int(num_levels)
        self.ini_th_fast = int(ini_th_fast)
        self.min_th_fast = int(min_th_fast)
        self.deterministic = bool(deterministic)
        self._extractor = extractor_cls(
            self.num_features,
            self.scale_factor,
            self.num_levels,
            self.ini_th_fast,
            self.min_th_fast,
        )

    @classmethod
    def is_available(cls) -> bool:
        return _import_orbslam2_features() is not None

    def setMaxFeatures(self, num_features: int) -> None:
        self.num_features = int(num_features)
        self._extractor.SetNumFeatures(self.num_features)

    def detect(self, image: np.ndarray, mask=None) -> list[cv2.KeyPoint]:
        kps_tuples = self._extractor.detect(_as_gray(image))
        return [_keypoint_from_tuple(kp) for kp in kps_tuples]

    def compute(
        self, image: np.ndarray, keypoints: list[cv2.KeyPoint]
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        return self.extract(image).as_tuple()

    def extract(self, image: np.ndarray, mask=None) -> FeatureExtractionResult:
        kps_tuples, des = self._extractor.detectAndCompute(_as_gray(image))
        return FeatureExtractionResult(
            keypoints=[_keypoint_from_tuple(kp) for kp in kps_tuples],
            descriptors=des,
            backend_name=self.name,
        )


def create_extractor_backend(
    backend_name: str | None = None,
    *,
    num_features: int = 2000,
    scale_factor: float = 1.2,
    num_levels: int = 8,
    ini_th_fast: int = 20,
    min_th_fast: int = 7,
    deterministic: bool = False,
) -> FeatureExtractorBackend:
    normalized = normalize_backend_name(backend_name)
    kwargs = dict(
        num_features=num_features,
        scale_factor=scale_factor,
        num_levels=num_levels,
        ini_th_fast=ini_th_fast,
        min_th_fast=min_th_fast,
        deterministic=deterministic,
    )

    if normalized == "auto":
        if PySLAMORB2Backend.is_available():
            return PySLAMORB2Backend(**kwargs)
        return OpenCVORBBackend(**kwargs)
    if normalized == OpenCVORBBackend.name:
        return OpenCVORBBackend(**kwargs)
    if normalized == PySLAMORB2Backend.name:
        return PySLAMORB2Backend(**kwargs)

    raise ValueError(
        f"Unknown extractor backend '{backend_name}'. Expected 'opencv_orb', 'pyslam_orb2', or 'auto'."
    )


def normalize_backend_name(backend_name: str | None) -> str:
    if backend_name is None:
        return DEFAULT_EXTRACTOR_BACKEND
    normalized = str(backend_name).strip().lower().replace("-", "_")
    aliases = {
        "opencv": "opencv_orb",
        "orb": "opencv_orb",
        "orb2": "opencv_orb",
        "orbslam2": "pyslam_orb2",
        "pyslam": "pyslam_orb2",
    }
    return aliases.get(normalized, normalized)


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def _normalize_descriptors(descriptors) -> np.ndarray:
    if descriptors is None:
        return np.empty((0, 32), dtype=np.uint8)
    descriptors = np.asarray(descriptors, dtype=np.uint8)
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(0, 32) if descriptors.size == 0 else descriptors.reshape(1, -1)
    return np.ascontiguousarray(descriptors)


def _import_orbslam2_features():
    try:
        return import_module("orbslam2_features")
    except Exception:
        return None


def _keypoint_from_tuple(kp_tuple) -> cv2.KeyPoint:
    return cv2.KeyPoint(*kp_tuple)
