"""
BoW-guided descriptor matching for relocalization and loop verification.
This module narrows descriptor comparisons to shared vocabulary nodes before geometric filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.frame import ensure_frame_feature_arrays
from visual_slam.orbslam.slam.rotation_histogram import RotationHistogram


# Record statistics from one BoW-guided matching attempt.
@dataclass
class BoWMatchDiagnostics:
    bow_guided_matching_available: bool = False
    shared_words: int = 0
    raw_matches: int = 0
    matches_after_ratio: int = 0
    matches_after_orientation: int = 0
    threshold_rejects: int = 0
    ratio_rejects: int = 0
    duplicate_train_rejects: int = 0
    orientation_rejects: int = 0
    fallback_descriptor_matching: bool = False
    unavailable_reason: Optional[str] = None


# Store the matched feature indices and distances returned by BoW matching.
@dataclass
class BoWMatchResult:
    idxs1: np.ndarray
    idxs2: np.ndarray
    distances: np.ndarray
    diagnostics: BoWMatchDiagnostics

    @property
    def available(self) -> bool:
        return self.diagnostics.bow_guided_matching_available


# Restrict descriptor matching to shared visual words before geometric filtering.
class BoWGuidedMatcher:
    def __init__(self, vocabulary=None):
        self.vocabulary = vocabulary

    def match(
        self,
        frame1,
        frame2,
        *,
        valid_idxs1=None,
        valid_idxs2=None,
        max_descriptor_distance: Optional[float] = None,
        ratio_test: float = 0.75,
        orientation_check: bool = True,
    ) -> BoWMatchResult:
        ensure_frame_feature_arrays(frame1)
        ensure_frame_feature_arrays(frame2)

        diagnostics = BoWMatchDiagnostics()
        feature_vector1 = self._feature_vector(frame1)
        feature_vector2 = self._feature_vector(frame2)
        if not feature_vector1 or not feature_vector2:
            diagnostics.unavailable_reason = "missing or empty direct-file feature vector"
            return self._empty(diagnostics)

        descriptors1 = self._descriptors(frame1)
        descriptors2 = self._descriptors(frame2)
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            diagnostics.unavailable_reason = "missing descriptors"
            return self._empty(diagnostics)

        allowed1 = self._allowed_mask(len(descriptors1), valid_idxs1)
        allowed2 = self._allowed_mask(len(descriptors2), valid_idxs2)
        shared_words = sorted(set(feature_vector1).intersection(feature_vector2))
        diagnostics.shared_words = len(shared_words)
        if len(shared_words) == 0:
            diagnostics.bow_guided_matching_available = True
            return self._empty(diagnostics)

        max_distance = (
            float(max_descriptor_distance)
            if max_descriptor_distance is not None
            else float(Parameters.kMaxDescriptorDistance or 100)
        )

        candidates = []
        best_for_train: dict[int, tuple[int, int, float]] = {}

        for node_id in shared_words:
            idxs1 = [idx for idx in feature_vector1[node_id] if 0 <= idx < len(descriptors1) and allowed1[idx]]
            idxs2 = [idx for idx in feature_vector2[node_id] if 0 <= idx < len(descriptors2) and allowed2[idx]]
            if len(idxs1) == 0 or len(idxs2) == 0:
                continue

            train_des = descriptors2[np.asarray(idxs2, dtype=np.int32)]
            for idx1 in idxs1:
                distances = self._hamming_distances(descriptors1[int(idx1)], train_des)
                if len(distances) == 0:
                    continue
                order = np.argsort(distances)
                best_pos = int(order[0])
                best_distance = float(distances[best_pos])
                if best_distance > max_distance:
                    diagnostics.threshold_rejects += 1
                    continue
                if len(order) > 1:
                    second_distance = float(distances[int(order[1])])
                    if best_distance >= float(ratio_test) * second_distance:
                        diagnostics.ratio_rejects += 1
                        continue

                idx2 = int(idxs2[best_pos])
                previous = best_for_train.get(idx2)
                if previous is None or best_distance < previous[2]:
                    if previous is not None:
                        diagnostics.duplicate_train_rejects += 1
                    best_for_train[idx2] = (int(idx1), idx2, best_distance)
                else:
                    diagnostics.duplicate_train_rejects += 1

        candidates = sorted(best_for_train.values(), key=lambda item: item[2])
        diagnostics.raw_matches = len(candidates)
        diagnostics.matches_after_ratio = len(candidates)
        diagnostics.bow_guided_matching_available = True

        idxs1 = np.asarray([item[0] for item in candidates], dtype=np.int32)
        idxs2 = np.asarray([item[1] for item in candidates], dtype=np.int32)
        distances = np.asarray([item[2] for item in candidates], dtype=np.float32)

        if orientation_check and FeatureTrackerShared.oriented_features and len(idxs1) > 0:
            valid = RotationHistogram.filter_matches_with_histogram_orientation(
                idxs1,
                idxs2,
                frame1.angles,
                frame2.angles,
            )
            valid = np.asarray(valid, dtype=np.int32)
            diagnostics.orientation_rejects = int(len(idxs1) - len(valid))
            idxs1 = idxs1[valid]
            idxs2 = idxs2[valid]
            distances = distances[valid]

        diagnostics.matches_after_orientation = len(idxs1)
        return BoWMatchResult(idxs1, idxs2, distances, diagnostics)

    def _feature_vector(self, frame) -> dict[int, list[int]]:
        feature_vector = getattr(frame, "feature_vector", None) or getattr(frame, "f_des", None)
        if (not feature_vector) and self.vocabulary is not None and getattr(self.vocabulary, "available", False):
            frame.compute_bow(self.vocabulary)
            feature_vector = getattr(frame, "feature_vector", None) or getattr(frame, "f_des", None)
        if hasattr(feature_vector, "to_dict"):
            feature_vector = feature_vector.to_dict()
        if not feature_vector:
            return {}
        return {
            int(node_id): [int(idx) for idx in indices]
            for node_id, indices in dict(feature_vector).items()
        }

    @staticmethod
    def _descriptors(frame) -> np.ndarray:
        descriptors = getattr(frame, "des", None)
        if descriptors is None:
            return np.empty((0, 32), dtype=np.uint8)
        descriptors = np.asarray(descriptors, dtype=np.uint8)
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)
        return np.ascontiguousarray(descriptors)

    @staticmethod
    def _allowed_mask(length: int, valid_idxs) -> np.ndarray:
        if valid_idxs is None:
            return np.ones(int(length), dtype=bool)
        mask = np.zeros(int(length), dtype=bool)
        valid_idxs = np.asarray(valid_idxs, dtype=np.int32).reshape(-1)
        valid_idxs = valid_idxs[(valid_idxs >= 0) & (valid_idxs < length)]
        mask[valid_idxs] = True
        return mask

    @staticmethod
    def _hamming_distances(query_descriptor: np.ndarray, train_descriptors: np.ndarray) -> np.ndarray:
        xor = np.bitwise_xor(train_descriptors, np.asarray(query_descriptor, dtype=np.uint8).reshape(1, -1))
        return np.unpackbits(xor, axis=1).sum(axis=1)

    @staticmethod
    def _empty(diagnostics: BoWMatchDiagnostics) -> BoWMatchResult:
        return BoWMatchResult(
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            diagnostics,
        )
