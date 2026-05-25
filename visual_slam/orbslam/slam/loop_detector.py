"""
Loop-candidate retrieval logic.
This module queries the keyframe database and filters candidates by similarity score.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import sys
from typing import Optional

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.keyframe_database import KeyFrameDatabase


# Store the loop candidates and similarity scores returned by one query.
@dataclass
class LoopDetectorOutput:
    keyframe: KeyFrame
    candidate_keyframes: list[KeyFrame]
    candidate_scores: list[float]
    min_score: float
    candidate_source: str = "auto"
    candidate_details: dict[int, dict] | None = None
    retrieval_profile: dict | None = None
    source_comparison: dict | None = None
    trace_rows: dict[str, list[dict]] | None = None
    trace_metadata: dict | None = None
    min_score_context: dict | None = None
    unavailable_reason: Optional[str] = None

    @property
    def candidate_idxs(self):
        return [kf.id for kf in self.candidate_keyframes]


# Query the keyframe database and build candidate sets for loop verification.
class LoopDetector:
    def __init__(self, keyframe_database: Optional[KeyFrameDatabase] = None):
        self.keyframe_database = keyframe_database
        self.last_output: Optional[LoopDetectorOutput] = None
        self.last_min_score_context: dict = {}

    @property
    def available(self) -> bool:
        return self.keyframe_database is not None and self.keyframe_database.available

    def compute_reference_similarity_score(self, keyframe: KeyFrame) -> float:
        self.last_min_score_context = {
            "connected_kf_count": 0,
            "min_score_source_kf_id": -1,
            "connected_scores": [],
        }
        if not self.available:
            return 0.0

        self.keyframe_database.compute_bow(keyframe)
        connected = keyframe.get_connected_keyframes()
        # Mirror pyslam loop_detector_base.py:322: return a very-negative sentinel
        # when there are no covisible neighbours so that the min_score gate is
        # effectively disabled for isolated/early keyframes.
        if len(connected) == 0:
            return -sys.float_info.max

        scores: list[tuple[float, int]] = []
        for connected_keyframe in connected:
            if connected_keyframe is None or connected_keyframe.is_bad():
                continue
            self.keyframe_database.compute_bow(connected_keyframe)
            score = float(self.keyframe_database.score(keyframe.g_des, connected_keyframe.g_des))
            scores.append((score, int(getattr(connected_keyframe, "kid", getattr(connected_keyframe, "id", -1)))))

        if not scores:
            return 0.0
        min_score, min_score_source_kf_id = min(scores, key=lambda item: item[0])
        # [PHASE2-MIN-SCORE-RELAX] (added 2026-05-18) — apply optional
        # multiplicative relaxation. factor=1.0 restores legacy behavior; see
        # config_parameters.py:kLoopClosingMinScoreRelaxFactor and
        # CODEX_LAB_PHASE2_CONNECTED_FILTER_TEMPORAL_WINDOW.md.
        relax_factor = float(getattr(Parameters, "kLoopClosingMinScoreRelaxFactor", 1.0) or 1.0)
        if relax_factor <= 0.0:
            relax_factor = 1.0
        relaxed_min_score = float(min_score) * relax_factor
        self.last_min_score_context = {
            "connected_kf_count": int(len(scores)),
            "min_score_source_kf_id": int(min_score_source_kf_id),
            "connected_scores": [
                {"kf_id": int(kf_id), "score": float(score)}
                for score, kf_id in sorted(scores, key=lambda item: item[0])
            ],
            "connected_scores_json": json.dumps(
                [
                    {"kf_id": int(kf_id), "score": float(score)}
                    for score, kf_id in sorted(scores, key=lambda item: item[0])
                ],
                sort_keys=True,
            ),
            # [PHASE2-MIN-SCORE-RELAX] trace fields for analysis
            "min_score_raw": float(min_score),
            "min_score_relax_factor": float(relax_factor),
            "min_score_effective": float(relaxed_min_score),
        }
        return relaxed_min_score

    def detect(self, keyframe: KeyFrame) -> LoopDetectorOutput:
        if not self.available:
            reason = (
                "keyframe database is not configured"
                if self.keyframe_database is None
                else self.keyframe_database.unavailable_reason()
            )
            output = LoopDetectorOutput(keyframe, [], [], 0.0, unavailable_reason=reason)
            self.last_output = output
            return output

        self.keyframe_database.compute_bow(keyframe)
        min_score = self.compute_reference_similarity_score(keyframe)
        result = self.keyframe_database.detect_loop_candidates(
            keyframe,
            min_score=min_score,
            min_delta_frames=Parameters.kMinDeltaFrameForMeaningfulLoopClosure,
            candidate_source=getattr(Parameters, "kLoopCandidateSource", "auto"),
            min_score_context=dict(self.last_min_score_context),
            return_diagnostics=True,
        )
        candidates = list(getattr(result, "candidates", []) or [])
        scores = list(getattr(result, "candidate_scores", []) or [])
        if len(scores) != len(candidates):
            scores = [self.keyframe_database.score(keyframe.g_des, candidate.g_des) for candidate in candidates]

        # Detector-side temporal filter (pyslam loop_detector_dbow2.py:248-251):
        # abs(other_frame_id - frame_id) > kMinDeltaFrameForMeaningfulLoopClosure.
        # Applied after database retrieval so that word counts are computed over
        # the full candidate pool (Stage 1 inverted-file walk).
        keyframe_frame_id = int(getattr(keyframe, "id", -1))
        min_delta = Parameters.kMinDeltaFrameForMeaningfulLoopClosure
        paired = [
            (cand, score)
            for cand, score in zip(candidates, scores)
            if abs(int(getattr(cand, "id", 0)) - keyframe_frame_id) > min_delta
        ]
        candidates = [cand for cand, _ in paired]
        scores = [score for _, score in paired]

        output = LoopDetectorOutput(
            keyframe,
            candidates,
            scores,
            min_score,
            candidate_source=str(getattr(result, "retrieval_profile", {}).get("candidate_source", "auto")),
            candidate_details=dict(getattr(result, "candidate_details", {}) or {}),
            retrieval_profile=dict(getattr(result, "retrieval_profile", {}) or {}),
            source_comparison=dict(getattr(result, "source_comparison", {}) or {}),
            trace_rows=dict(getattr(result, "trace_rows", {}) or {}),
            trace_metadata=dict(getattr(result, "trace_metadata", {}) or {}),
            min_score_context=dict(self.last_min_score_context),
        )
        self.last_output = output
        return output
