"""
Bag-of-words vocabulary access for loop and relocalization queries.
This module loads the local vocabulary, exposes transform helpers, and reports backend status.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_PYDBOW3_DIR = REPO_ROOT / "third_party" / "local" / "pydbow3"
LOCAL_VOCAB_DIR = REPO_ROOT / "third_party" / "vocabs"
DEFAULT_DBOW3_VOCAB_PATH = LOCAL_VOCAB_DIR / "ORBvoc.dbow3"
PYSLAM_DBOW3_GDRIVE_URL = "https://drive.google.com/uc?id=13xmRtop_ow3aPtv3qCT5beG19_mlogqI"

_DEFAULT_VOCABULARY_CACHE = None


# Report availability and file locations for the BoW backend.
@dataclass
class BoWBackendStatus:
    backend_name: str
    available: bool
    vocabulary_path: Optional[Path]
    pydbow3_path: Optional[Path]
    reason: Optional[str] = None


def ensure_local_pydbow3_path() -> None:
    local_path = str(LOCAL_PYDBOW3_DIR)
    if LOCAL_PYDBOW3_DIR.exists() and local_path not in sys.path:
        sys.path.insert(0, local_path)


def import_pydbow3():
    ensure_local_pydbow3_path()
    try:
        import pydbow3  # type: ignore
    except Exception as exc:
        return None, exc
    return pydbow3, None


def get_default_vocabulary_path() -> Path:
    return DEFAULT_DBOW3_VOCAB_PATH


def get_bow_backend_status(vocabulary_path: Optional[Path] = None) -> BoWBackendStatus:
    vocabulary_path = Path(vocabulary_path or DEFAULT_DBOW3_VOCAB_PATH)
    pydbow3, import_error = import_pydbow3()

    if pydbow3 is None:
        return BoWBackendStatus(
            backend_name="dbow3",
            available=False,
            vocabulary_path=vocabulary_path,
            pydbow3_path=LOCAL_PYDBOW3_DIR,
            reason=f"pydbow3 import failed: {import_error}",
        )

    if not vocabulary_path.exists():
        return BoWBackendStatus(
            backend_name="dbow3",
            available=False,
            vocabulary_path=vocabulary_path,
            pydbow3_path=Path(pydbow3.__file__),
            reason=f"vocabulary file not found: {vocabulary_path}",
        )

    return BoWBackendStatus(
        backend_name="dbow3",
        available=True,
        vocabulary_path=vocabulary_path,
        pydbow3_path=Path(pydbow3.__file__),
        reason=None,
    )


# Load, cache, and query the visual vocabulary used for BoW conversion.
class DBoW3Vocabulary:
    """Small wrapper around the local DBoW3 Python binding."""

    backend_name = "dbow3"

    def __init__(self, vocabulary_path: Optional[Path] = None, autoload: bool = True):
        self.vocabulary_path = Path(vocabulary_path or DEFAULT_DBOW3_VOCAB_PATH)
        self.pydbow3 = None
        self.voc = None
        self.available = False
        self.error = None
        self.loaded_with_boost = None

        pydbow3, import_error = import_pydbow3()
        if pydbow3 is None:
            self.error = f"pydbow3 import failed: {import_error}"
            return

        self.pydbow3 = pydbow3
        self.voc = pydbow3.Vocabulary()

        if autoload:
            self.load(self.vocabulary_path)

    def load(self, vocabulary_path: Optional[Path] = None) -> bool:
        if self.pydbow3 is None or self.voc is None:
            return False

        self.vocabulary_path = Path(vocabulary_path or self.vocabulary_path)
        if not self.vocabulary_path.exists():
            self.available = False
            self.error = f"vocabulary file not found: {self.vocabulary_path}"
            return False

        last_error = None
        # Try the direct loader first because the bundled binary vocabulary is not boost-serialized.
        for use_boost in (False, True):
            try:
                voc = self.pydbow3.Vocabulary()
                voc.load(str(self.vocabulary_path), use_boost=use_boost)
                if voc.empty():
                    raise RuntimeError("loaded vocabulary is empty")
                self.voc = voc
                self.available = True
                self.error = None
                self.loaded_with_boost = use_boost
                return True
            except Exception as exc:
                last_error = exc

        self.available = False
        self.error = f"failed to load vocabulary {self.vocabulary_path}: {last_error}"
        return False

    def transform(self, descriptors: np.ndarray):
        if not self.available or self.voc is None:
            raise RuntimeError(self.error or "DBoW3 vocabulary is unavailable")
        descriptors = self._normalize_descriptors(descriptors)
        return self.voc.transform(descriptors)

    def transform_with_feature_vector(self, descriptors: np.ndarray, levelsup: int = 4):
        if not self.available or self.voc is None:
            raise RuntimeError(self.error or "DBoW3 vocabulary is unavailable")
        descriptors = self._normalize_descriptors(descriptors)
        if hasattr(self.voc, "transform_with_feature_vector"):
            return self.voc.transform_with_feature_vector(descriptors, int(levelsup))

        bow = self.voc.transform(descriptors)
        return _PythonTransformResult(bow, self.feature_vector(descriptors, bow))

    def score(self, bow_a, bow_b) -> float:
        if not self.available or self.voc is None:
            return 0.0
        if bow_a is None or bow_b is None:
            return 0.0
        return float(self.voc.score(self.to_native_bow(bow_a), self.to_native_bow(bow_b)))

    def size(self) -> int:
        if not self.available or self.voc is None:
            return 0
        return int(self.voc.size())

    def to_native_bow(self, bow):
        if bow is None:
            return None
        if self.pydbow3 is not None and isinstance(bow, self.pydbow3.BowVector):
            return bow
        if self.pydbow3 is not None:
            return self.pydbow3.BowVector(self.bow_to_vec(bow))
        return bow

    @staticmethod
    def bow_to_vec(bow) -> list[tuple[int, float]]:
        if bow is None:
            return []
        if hasattr(bow, "toVec"):
            return [(int(word_id), float(weight)) for word_id, weight in bow.toVec()]
        if isinstance(bow, dict):
            return [(int(word_id), float(weight)) for word_id, weight in bow.items()]
        return [(int(word_id), float(weight)) for word_id, weight in list(bow)]

    def feature_vector(self, descriptors: np.ndarray, bow=None) -> dict[int, list[int]]:
        descriptors = self._normalize_descriptors(descriptors)
        if self.voc is not None and hasattr(self.voc, "transform_with_feature_vector"):
            result = self.voc.transform_with_feature_vector(descriptors, 4)
            return self.feature_vector_to_dict(result.featureVector)
        if bow is None:
            bow = self.transform(descriptors)
        return {int(word_id): [] for word_id, _ in self.bow_to_vec(bow)}

    @staticmethod
    def feature_vector_to_dict(feature_vector) -> dict[int, list[int]]:
        if feature_vector is None:
            return {}
        if hasattr(feature_vector, "to_dict"):
            feature_vector = feature_vector.to_dict()
        return {
            int(node_id): [int(idx) for idx in indices]
            for node_id, indices in dict(feature_vector).items()
        }

    @staticmethod
    def _normalize_descriptors(descriptors: np.ndarray) -> np.ndarray:
        if descriptors is None:
            return np.empty((0, 32), dtype=np.uint8)
        descriptors = np.asarray(descriptors, dtype=np.uint8)
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)
        return np.ascontiguousarray(descriptors)


def load_default_vocabulary(force_reload: bool = False) -> DBoW3Vocabulary:
    global _DEFAULT_VOCABULARY_CACHE
    if _DEFAULT_VOCABULARY_CACHE is None or force_reload:
        _DEFAULT_VOCABULARY_CACHE = DBoW3Vocabulary(DEFAULT_DBOW3_VOCAB_PATH, autoload=True)
    return _DEFAULT_VOCABULARY_CACHE


def compute_bow_for_frame(frame_or_keyframe: Any, vocabulary: DBoW3Vocabulary):
    if vocabulary is None or not getattr(vocabulary, "available", False):
        raise RuntimeError(getattr(vocabulary, "error", "BoW vocabulary is unavailable"))
    result = vocabulary.transform_with_feature_vector(getattr(frame_or_keyframe, "des", None), levelsup=4)
    bow = result.bowVector if hasattr(result, "bowVector") else result[0]
    raw_feature_vector = result.featureVector if hasattr(result, "featureVector") else result[1]
    feature_vector = vocabulary.feature_vector_to_dict(raw_feature_vector)
    frame_or_keyframe.g_des = bow
    frame_or_keyframe.bow_vector = bow
    frame_or_keyframe.f_des = feature_vector
    frame_or_keyframe.feature_vector = feature_vector
    return bow, feature_vector


@dataclass
# Expose BoW and feature-vector outputs through one lightweight result object.
class _PythonTransformResult:
    bowVector: Any
    featureVector: dict[int, list[int]]
