"""
Keyframe database for loop and relocalization queries.
This module indexes keyframes by visual words and scores candidate retrievals.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from threading import Lock
from typing import Optional

from visual_slam.orbslam.slam.bow import DBoW3Vocabulary, compute_bow_for_frame
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.frame import Frame
from visual_slam.orbslam.slam.keyframe import KeyFrame


@dataclass
class LoopCandidateRetrievalResult:
    candidates: list[KeyFrame]
    candidate_scores: list[float]
    candidate_details: dict[int, dict]
    retrieval_profile: dict
    source_comparison: dict
    trace_rows: dict[str, list[dict]] | None = None
    trace_metadata: dict | None = None


@dataclass
class RawDbowQueryResult:
    candidates: list[tuple[KeyFrame, float]]
    trace_rows: list[dict]
    query_k: int
    result_count: int


class KeyFrameDatabase:
    """Inverted-file database used by loop closing and relocalization."""

    def __init__(self, vocabulary: Optional[DBoW3Vocabulary] = None):
        self.voc = vocabulary
        self.inverted_file = defaultdict(list)
        self.mutex = Lock()
        self.dbow_database = None
        self._entry_to_keyframe: dict[int, KeyFrame] = {}
        self._keyframe_to_entry: dict[KeyFrame, int] = {}
        self.loop_retrieval_trace_enabled = False
        self.loop_retrieval_trace_raw_k = 0
        if vocabulary is not None:
            self._reset_dbow_database()

    @property
    def available(self) -> bool:
        return self.voc is not None and bool(getattr(self.voc, "available", False))

    def is_available(self) -> bool:
        return self.available

    def set_vocabulary(self, vocabulary: DBoW3Vocabulary) -> None:
        with self.mutex:
            self.voc = vocabulary
            self.inverted_file.clear()
            self._reset_dbow_database()

    def configure_loop_retrieval_trace(self, *, enabled: bool, raw_k: int = 0) -> None:
        self.loop_retrieval_trace_enabled = bool(enabled)
        self.loop_retrieval_trace_raw_k = max(0, int(raw_k))

    def size(self) -> int:
        with self.mutex:
            if self.dbow_database is not None:
                try:
                    return int(self.dbow_database.size())
                except Exception:
                    pass
            return int(len(self._keyframe_to_entry))

    def add(self, keyframe: KeyFrame) -> None:
        if not self.available or keyframe is None:
            return
        self.compute_bow(keyframe)
        with self.mutex:
            for word_id, _ in self._bow_items(keyframe.g_des):
                if keyframe not in self.inverted_file[word_id]:
                    self.inverted_file[word_id].append(keyframe)
            if keyframe not in self._keyframe_to_entry and self.dbow_database is not None:
                entry_id = int(self.dbow_database.addBowVector(self.voc.to_native_bow(keyframe.g_des)))
                self._keyframe_to_entry[keyframe] = entry_id
                self._entry_to_keyframe[entry_id] = keyframe

    def erase(self, keyframe: KeyFrame) -> None:
        if keyframe is None:
            return
        with self.mutex:
            for word_id, _ in self._bow_items(getattr(keyframe, "g_des", None)):
                kf_list = self.inverted_file.get(word_id, [])
                try:
                    kf_list.remove(keyframe)
                except ValueError:
                    pass

    def clear(self) -> None:
        with self.mutex:
            self.inverted_file.clear()
            self._reset_dbow_database()

    def reset(self) -> None:
        self.clear()

    def compute_bow(self, frame_or_keyframe):
        if not self.available:
            raise RuntimeError(self.unavailable_reason())
        if getattr(frame_or_keyframe, "g_des", None) is None:
            return compute_bow_for_frame(frame_or_keyframe, self.voc)
        return getattr(frame_or_keyframe, "g_des"), getattr(frame_or_keyframe, "f_des", None)

    def unavailable_reason(self) -> str:
        if self.voc is None:
            return "BoW vocabulary is not configured"
        return getattr(self.voc, "error", "BoW vocabulary is unavailable")

    def detect_relocalization_candidates(self, frame: Frame) -> list[KeyFrame]:
        if not self.available:
            return []

        self.compute_bow(frame)
        query_id = getattr(frame, "id", getattr(frame, "mn_id", -1))
        keyframes_sharing_words = []

        with self.mutex:
            for word_id, _ in self._bow_items(frame.g_des):
                for keyframe in self.inverted_file.get(word_id, []):
                    if keyframe is None or keyframe.is_bad():
                        continue
                    if keyframe.reloc_query_id != query_id:
                        keyframe.num_reloc_words = 0
                        keyframe.reloc_query_id = query_id
                        keyframes_sharing_words.append(keyframe)
                    keyframe.num_reloc_words += 1

        if len(keyframes_sharing_words) == 0:
            return []

        max_common_words = max(kf.num_reloc_words for kf in keyframes_sharing_words)
        min_common_words = int(max_common_words * 0.8)

        score_and_match = []
        for keyframe in keyframes_sharing_words:
            if keyframe.num_reloc_words > min_common_words:
                score = self.voc.score(frame.g_des, keyframe.g_des)
                keyframe.reloc_score = score
                score_and_match.append((score, keyframe))

        if len(score_and_match) == 0:
            return []

        acc_score_and_match = []
        best_acc_score = 0.0

        for score, keyframe in score_and_match:
            neighbors = keyframe.get_best_covisible_keyframes(10)
            best_score = score
            acc_score = score
            best_keyframe = keyframe

            for neighbor in neighbors:
                if neighbor.reloc_query_id == query_id:
                    acc_score += neighbor.reloc_score
                    if neighbor.reloc_score > best_score:
                        best_keyframe = neighbor
                        best_score = neighbor.reloc_score

            acc_score_and_match.append((acc_score, best_keyframe))
            best_acc_score = max(best_acc_score, acc_score)

        min_score_to_retain = 0.75 * best_acc_score
        already_added = set()
        candidates = []

        for acc_score, keyframe in acc_score_and_match:
            if acc_score > min_score_to_retain and keyframe not in already_added:
                candidates.append(keyframe)
                already_added.add(keyframe)

        return candidates

    def detect_loop_candidates(
        self,
        keyframe: KeyFrame,
        min_score: float,
        min_delta_frames: int = Parameters.kMinDeltaFrameForMeaningfulLoopClosure,
        candidate_source: str | None = None,
        min_score_context: dict | None = None,
        return_diagnostics: bool = False,
    ) -> list[KeyFrame] | LoopCandidateRetrievalResult:
        if not self.available:
            empty = LoopCandidateRetrievalResult([], [], {}, {}, {}, {}, {})
            return empty if return_diagnostics else []

        self.compute_bow(keyframe)
        requested_source = str(candidate_source or getattr(Parameters, "kLoopCandidateSource", "auto")).lower()
        source_mode = self._normalize_candidate_source_mode(requested_source)
        if source_mode == "auto":
            source_mode = self._resolve_auto_candidate_source_mode()

        runtime_dbow_query_k = self._resolve_dbow_detector_runtime_k()
        trace_dbow_query_k = self._resolve_trace_dbow_query_k(runtime_dbow_query_k)

        runtime_raw_dbow_query = self._detect_loop_candidates_dbow3_raw(
            keyframe,
            min_score=float(min_score),
            min_delta_frames=int(min_delta_frames),
            query_k=runtime_dbow_query_k,
        )
        trace_raw_dbow_query = runtime_raw_dbow_query
        if (
            self.loop_retrieval_trace_enabled
            and trace_dbow_query_k != runtime_dbow_query_k
        ):
            trace_raw_dbow_query = self._detect_loop_candidates_dbow3_raw(
                keyframe,
                min_score=float(min_score),
                min_delta_frames=int(min_delta_frames),
                query_k=trace_dbow_query_k,
            )

        classic_inverted_result = self._detect_loop_candidates_inverted_scored(
            keyframe,
            min_score=float(min_score),
            min_delta_frames=int(min_delta_frames),
            min_score_context=min_score_context,
        )
        dbow_detector_result = self._detect_loop_candidates_dbow_detector(
            keyframe,
            raw_dbow_query=runtime_raw_dbow_query,
            min_score=float(min_score),
            min_delta_frames=int(min_delta_frames),
            min_score_context=min_score_context,
        )
        hybrid_dbow_scored_result = self._detect_loop_candidates_dbow3_scored(
            keyframe,
            raw_dbow_query=runtime_raw_dbow_query,
            min_score=float(min_score),
            min_delta_frames=int(min_delta_frames),
            min_score_context=min_score_context,
        )

        if source_mode == "classic_inverted":
            chosen_source = "classic_inverted"
            chosen_result = classic_inverted_result
        elif source_mode == "dbow_detector":
            chosen_source = "dbow_detector"
            chosen_result = dbow_detector_result
        elif source_mode == "hybrid_dbow_scored":
            chosen_source = "hybrid_dbow_scored"
            chosen_result = hybrid_dbow_scored_result
        else:
            chosen_source = self._resolve_auto_candidate_source_mode()
            chosen_result = classic_inverted_result

        chosen_candidates = list(chosen_result["candidates"])
        chosen_scores = list(chosen_result["scores"])
        candidate_details = dict(chosen_result["details"])
        retrieval_profile = dict(chosen_result["profile"])
        retrieval_profile["candidate_source"] = str(chosen_source)
        retrieval_profile["num_raw_inverted_candidates"] = int(
            classic_inverted_result.get("profile", {}).get("num_raw_inverted_candidates", 0)
        )

        source_comparison = self._build_source_comparison_row(
            keyframe,
            dbow_result=dbow_detector_result,
            inverted_result=classic_inverted_result,
            chosen_candidates=chosen_candidates,
            chosen_source=chosen_source,
        )
        result = LoopCandidateRetrievalResult(
            candidates=chosen_candidates,
            candidate_scores=chosen_scores,
            candidate_details=candidate_details,
            retrieval_profile=retrieval_profile,
            source_comparison=source_comparison,
            trace_rows={
                "raw_dbow": list(getattr(trace_raw_dbow_query, "trace_rows", []) or []),
                "inverted_word": list(chosen_result.get("trace_rows", {}).get("inverted_word", []) or []),
                "score_filter": list(chosen_result.get("trace_rows", {}).get("score_filter", []) or []),
                "accumulation": list(chosen_result.get("trace_rows", {}).get("accumulation", []) or []),
            },
            trace_metadata={
                "trace_enabled": bool(self.loop_retrieval_trace_enabled),
                "trace_raw_k_requested": int(self.loop_retrieval_trace_raw_k),
                "runtime_dbow_query_k": int(runtime_dbow_query_k),
                "trace_dbow_query_k": int(trace_dbow_query_k),
                "raw_dbow_query_k": int(getattr(trace_raw_dbow_query, "query_k", 0) or 0),
                "raw_dbow_result_count": int(getattr(trace_raw_dbow_query, "result_count", 0) or 0),
            },
        )
        return result if return_diagnostics else result.candidates

    @staticmethod
    def _normalize_candidate_source_mode(source_mode: str | None) -> str:
        normalized = str(source_mode or "auto").strip().lower()
        aliases = {
            "inverted_file": "classic_inverted",
            "dbow3": "dbow_detector",
            "dbow3_scored": "hybrid_dbow_scored",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {
            "auto",
            "classic_inverted",
            "dbow_detector",
            "hybrid_dbow_scored",
            "compare",
        }:
            return "auto"
        return normalized

    @staticmethod
    def _resolve_auto_candidate_source_mode() -> str:
        return "classic_inverted"

    @staticmethod
    def _resolve_dbow_detector_runtime_k() -> int:
        configured = int(
            getattr(Parameters, "kLoopDbowDetectorTopK", getattr(Parameters, "kMaxResultsForLoopClosure", 1)) or 1
        )
        return max(1, configured)

    def _resolve_trace_dbow_query_k(self, runtime_query_k: int) -> int:
        requested_trace_k = int(self.loop_retrieval_trace_raw_k or 0)
        if requested_trace_k <= 0:
            return int(runtime_query_k)
        return max(int(runtime_query_k), requested_trace_k)

    # [PHASE2-CONNECTED-TEMPORAL-WINDOW] (added 2026-05-18) — see
    # config_parameters.py:kLoopConnectedFilterTemporalWindowKf and the rollback
    # doc CODEX_LAB_PHASE2_CONNECTED_FILTER_TEMPORAL_WINDOW.md.
    def _select_connected_keyframes_for_filter(self, keyframe) -> set:
        """Return covisibility neighbors that should be excluded from loop
        candidate retrieval.

        Legacy behavior excluded EVERY covisibility neighbor, which silently
        drops genuine spatial revisits when local mapping has stitched their
        map points back into the graph (e.g. the lab dataset's 35
        same-direction revisits that were "is_connected=True" at <100 mm
        spatial distance). Restricting the exclusion to a temporal window
        keeps the original intent (suppress trivially adjacent KFs) while
        letting long-range loop-induced connections through.

        Set kLoopConnectedFilterTemporalWindowKf=0 to revert without code
        changes; the early-return below restores the legacy behavior exactly.
        """
        all_connected = keyframe.get_connected_keyframes()
        window = int(getattr(Parameters, "kLoopConnectedFilterTemporalWindowKf", 0))
        if window <= 0:
            return set(all_connected)
        cur_id = int(getattr(keyframe, "kid", getattr(keyframe, "id", -1)))
        return {
            c
            for c in all_connected
            if abs(int(getattr(c, "kid", getattr(c, "id", -1))) - cur_id) <= window
        }

    def _detect_loop_candidates_inverted_scored(
        self,
        keyframe: KeyFrame,
        *,
        min_score: float,
        min_delta_frames: int,
        min_score_context: dict | None,
    ) -> dict:
        # Mirrors pyslam KeyFrameDatabaseDBow.detect_loop_candidates
        # (third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py:60-125).
        # Temporal min-delta filtering is intentionally NOT applied here; it is
        # the detector's responsibility (Stage 2 of the loop-closure realignment
        # plan).
        candidate_source = "classic_inverted"
        details: dict[int, dict] = {}
        trace_rows: dict[str, list[dict]] = {
            "inverted_word": [],
            "score_filter": [],
            "accumulation": [],
        }

        keyframe_id = int(getattr(keyframe, "id", -1))
        current_kid = int(getattr(keyframe, "kid", keyframe_id))
        # [PHASE2-CONNECTED-TEMPORAL-WINDOW] was: set(keyframe.get_connected_keyframes())
        sp_connected_keyframes = self._select_connected_keyframes_for_filter(keyframe)

        # ---- Step 1: inverted-file walk (pyslam keyframe_database.py:60-78) ----
        # For each query word, increment num_loop_words on every keyframe seen
        # (including connected ones). Only non-connected keyframes are appended
        # to l_kfs_sharing_words and have their loop_query_id stamped, so that
        # accumulation in Step 4 sees only non-connected covisibility neighbors.
        l_kfs_sharing_words: list[KeyFrame] = []
        # Auxiliary diagnostic counts; only used to emit accurate inverted_word
        # trace rows (including for connected/temporal-rejected candidates).
        # Decisions never touch this dict.
        diag_shared_words: dict[int, int] = {}
        diag_kf_by_kid: dict[int, KeyFrame] = {}

        with self.mutex:
            for word_id, _ in self._bow_items(keyframe.g_des):
                for p_kf in self.inverted_file.get(word_id, []):
                    if p_kf is None or p_kf is keyframe:
                        continue
                    if p_kf.loop_query_id != keyframe_id:
                        p_kf.num_loop_words = 0
                        if p_kf not in sp_connected_keyframes:
                            p_kf.loop_query_id = keyframe_id
                            l_kfs_sharing_words.append(p_kf)
                    p_kf.num_loop_words += 1
                    if self.loop_retrieval_trace_enabled:
                        candidate_kid = int(getattr(p_kf, "kid", getattr(p_kf, "id", -1)))
                        diag_shared_words[candidate_kid] = diag_shared_words.get(candidate_kid, 0) + 1
                        diag_kf_by_kid[candidate_kid] = p_kf

        num_raw_inverted_candidates = len(l_kfs_sharing_words)

        if not l_kfs_sharing_words:
            return {
                "candidates": [],
                "scores": [],
                "details": details,
                "trace_rows": trace_rows,
                "profile": {
                    **self._empty_inverted_profile(
                        l_kfs_sharing_words,
                        l_kfs_sharing_words,
                        l_kfs_sharing_words,
                    ),
                    "candidate_source": candidate_source,
                    "num_raw_inverted_candidates": int(num_raw_inverted_candidates),
                },
            }

        # ---- Step 2: common-word threshold (pyslam keyframe_database.py:80-81) ----
        max_common_words = max(p_kf.num_loop_words for p_kf in l_kfs_sharing_words)
        min_common_words = int(max_common_words * Parameters.kLoopClosingCommonWordRatioThreshold)
        common_word_threshold_ratio = float(min_common_words) / float(max_common_words or 1)

        # ---- Diagnostic inverted_word trace rows (all sharing kfs, decisions unaffected) ----
        if self.loop_retrieval_trace_enabled:
            connected_ids = {
                int(getattr(c, "kid", getattr(c, "id", -1))) for c in sp_connected_keyframes
            }
            for candidate_kid, shared_words in sorted(diag_shared_words.items()):
                p_kf = diag_kf_by_kid[candidate_kid]
                temporal_gap = abs(int(getattr(p_kf, "id", 0)) - keyframe_id)
                is_connected = bool(candidate_kid in connected_ids)
                # Common-word filter only applies to non-connected candidates;
                # connected ones are excluded upstream.
                passed_common_word_filter = bool(
                    (not is_connected) and shared_words > min_common_words
                )
                trace_rows["inverted_word"].append(
                    {
                        "current_kf_id": current_kid,
                        "candidate_kf_id": int(candidate_kid),
                        "pair_key": self._pair_key(current_kid, candidate_kid),
                        "current_timestamp": getattr(keyframe, "timestamp", None),
                        "candidate_timestamp": getattr(p_kf, "timestamp", None),
                        "candidate_source": candidate_source,
                        "shared_words": int(shared_words),
                        "max_common_words": int(max_common_words),
                        "common_word_ratio": float(shared_words) / float(max_common_words or 1),
                        "common_word_threshold_ratio": common_word_threshold_ratio,
                        "passed_common_word_filter": passed_common_word_filter,
                        "is_connected": is_connected,
                        "temporal_gap": int(temporal_gap),
                        "passed_connected_filter": bool(not is_connected),
                        "passed_temporal_filter": bool(temporal_gap > int(min_delta_frames)),
                    }
                )

        # ---- Step 3: score + min_score filter (pyslam keyframe_database.py:83-92) ----
        l_score_and_match: list[tuple[float, KeyFrame]] = []
        num_candidates_after_common_words = 0
        score_trace_rows: list[dict] = []

        for p_kf in l_kfs_sharing_words:
            candidate_kid = int(getattr(p_kf, "kid", getattr(p_kf, "id", -1)))
            common_words = int(p_kf.num_loop_words)
            detail = {
                "candidate_source": candidate_source,
                "candidate_rank": -1,
                "min_score": float(min_score),
                "common_words": int(common_words),
                "max_common_words": int(max_common_words),
                "common_word_ratio": float(common_words) / float(max_common_words or 1),
                "raw_dbow_score": None,
                "bow_score_raw": 0.0,
                "bow_score_normalized": 0.0,
                "accumulated_score": 0.0,
                "best_accumulated_score": 0.0,
                "is_connected": False,
                "temporal_gap": abs(int(getattr(p_kf, "id", 0)) - keyframe_id),
                "passed_common_word_filter": bool(common_words > min_common_words),
                "passed_min_score_filter": False,
            }
            details[candidate_kid] = detail
            if common_words > min_common_words:
                num_candidates_after_common_words += 1
                score = float(self.voc.score(keyframe.g_des, p_kf.g_des))
                p_kf.loop_score = score
                detail["bow_score_raw"] = score
                detail["bow_score_normalized"] = score
                detail["passed_min_score_filter"] = bool(score >= min_score)
                if self.loop_retrieval_trace_enabled:
                    score_trace_rows.append(
                        {
                            "current_kf_id": current_kid,
                            "candidate_kf_id": candidate_kid,
                            "pair_key": self._pair_key(current_kid, candidate_kid),
                            "current_timestamp": getattr(keyframe, "timestamp", None),
                            "candidate_timestamp": getattr(p_kf, "timestamp", None),
                            "candidate_source": candidate_source,
                            "bow_score": float(score),
                            "min_score": float(min_score),
                            "score_over_min_score": float(score) / float(min_score if abs(min_score) > 1e-12 else 1.0),
                            "connected_kf_count_for_min_score": int((min_score_context or {}).get("connected_kf_count", 0) or 0),
                            "min_score_source_kf_id": int((min_score_context or {}).get("min_score_source_kf_id", -1) or -1),
                            "passed_min_score_filter": bool(score >= min_score),
                            "connected_scores_json": json.dumps((min_score_context or {}).get("connected_scores", []), sort_keys=True),
                        }
                    )
                if score >= min_score:
                    l_score_and_match.append((score, p_kf))

        trace_rows["score_filter"] = score_trace_rows

        if not l_score_and_match:
            return {
                "candidates": [],
                "scores": [],
                "details": details,
                "trace_rows": trace_rows,
                "profile": {
                    **self._empty_inverted_profile(
                        l_kfs_sharing_words,
                        l_kfs_sharing_words,
                        l_kfs_sharing_words,
                    ),
                    "candidate_source": candidate_source,
                    "num_raw_inverted_candidates": int(num_raw_inverted_candidates),
                    "num_candidates_after_common_words": int(num_candidates_after_common_words),
                },
            }

        # ---- Step 4: covisibility accumulation (pyslam keyframe_database.py:94-113) ----
        l_acc_score_and_match: list[tuple[float, KeyFrame]] = []
        best_acc_score = float(min_score)
        accumulation_payloads: list[dict] = []

        for score, p_kf in l_score_and_match:
            vp_neighs = p_kf.get_best_covisible_keyframes(10)
            best_score = score
            acc_score = score
            p_best_kf = p_kf
            group_ids = [int(getattr(p_kf, "kid", getattr(p_kf, "id", -1)))]
            group_scores = [float(score)]

            for p_kf2 in vp_neighs:
                if (
                    p_kf2.loop_query_id == keyframe_id
                    and p_kf2.num_loop_words > min_common_words
                ):
                    neighbor_score = float(getattr(p_kf2, "loop_score", 0.0))
                    acc_score += neighbor_score
                    group_ids.append(int(getattr(p_kf2, "kid", getattr(p_kf2, "id", -1))))
                    group_scores.append(neighbor_score)
                    if neighbor_score > best_score:
                        p_best_kf = p_kf2
                        best_score = neighbor_score

            l_acc_score_and_match.append((acc_score, p_best_kf))
            if acc_score > best_acc_score:
                best_acc_score = acc_score
            accumulation_payloads.append(
                {
                    "candidate": p_kf,
                    "candidate_kid": int(getattr(p_kf, "kid", getattr(p_kf, "id", -1))),
                    "candidate_score": float(score),
                    "group_ids": group_ids,
                    "group_scores": group_scores,
                    "accumulated_score": float(acc_score),
                    "best_candidate_id_in_group": int(getattr(p_best_kf, "kid", getattr(p_best_kf, "id", -1))),
                }
            )

        # ---- Step 5: retention (pyslam keyframe_database.py:115-125) ----
        # Iterate in INSERTION ORDER (chronological); deduplicate by representative.
        min_score_to_retain = 0.75 * best_acc_score
        sp_already_added_kf: set[KeyFrame] = set()
        candidates: list[KeyFrame] = []
        scores: list[float] = []
        retained_rank_by_best_kid: dict[int, int] = {}

        for acc_score, p_best_kf in l_acc_score_and_match:
            best_kid = int(getattr(p_best_kf, "kid", getattr(p_best_kf, "id", -1)))
            details.setdefault(best_kid, {})
            existing_acc = float(details[best_kid].get("accumulated_score", 0.0))
            if acc_score > existing_acc:
                details[best_kid]["accumulated_score"] = float(acc_score)
            details[best_kid]["best_accumulated_score"] = float(best_acc_score)
            if acc_score > min_score_to_retain and p_best_kf not in sp_already_added_kf:
                candidates.append(p_best_kf)
                scores.append(float(getattr(p_best_kf, "loop_score", 0.0)))
                sp_already_added_kf.add(p_best_kf)

        for rank, p_best_kf in enumerate(candidates, start=1):
            best_kid = int(getattr(p_best_kf, "kid", getattr(p_best_kf, "id", -1)))
            details.setdefault(best_kid, {})
            details[best_kid]["candidate_rank"] = int(rank)
            retained_rank_by_best_kid[best_kid] = int(rank)

        # ---- Accumulation trace rows (decisions unaffected) ----
        if self.loop_retrieval_trace_enabled:
            for payload in accumulation_payloads:
                candidate_kid = int(payload["candidate_kid"])
                best_candidate_id = int(payload["best_candidate_id_in_group"])
                acc_score = float(payload["accumulated_score"])
                passed_accumulated = bool(acc_score > min_score_to_retain)
                retained_candidate = bool(
                    passed_accumulated
                    and best_candidate_id in retained_rank_by_best_kid
                )
                trace_rows["accumulation"].append(
                    {
                        "current_kf_id": current_kid,
                        "candidate_kf_id": candidate_kid,
                        "pair_key": self._pair_key(current_kid, candidate_kid),
                        "current_timestamp": getattr(keyframe, "timestamp", None),
                        "candidate_timestamp": getattr(payload["candidate"], "timestamp", None),
                        "candidate_source": candidate_source,
                        "candidate_group_size": int(len(payload["group_ids"])),
                        "candidate_group_ids": json.dumps(payload["group_ids"]),
                        "candidate_group_scores": json.dumps(payload["group_scores"]),
                        "accumulated_score": acc_score,
                        "best_accumulated_score": float(best_acc_score),
                        "accumulated_score_ratio": acc_score / float(best_acc_score or 1.0),
                        "accumulation_threshold_ratio": 0.75,
                        "passed_accumulated_score_filter": passed_accumulated,
                        "best_candidate_id_in_group": int(best_candidate_id),
                        "retained_candidate": retained_candidate,
                        "retained_rank": int(retained_rank_by_best_kid.get(best_candidate_id, -1)),
                    }
                )

        top_kid = (
            int(getattr(candidates[0], "kid", getattr(candidates[0], "id", -1)))
            if candidates
            else -1
        )
        return {
            "candidates": candidates,
            "scores": scores,
            "details": details,
            "trace_rows": trace_rows,
            "profile": {
                "num_db_keyframes_before_query": int(self.size()),
                "candidate_source": candidate_source,
                "num_raw_dbow_candidates": 0,
                "num_raw_inverted_candidates": int(num_raw_inverted_candidates),
                # Temporal filter not applied at this stage (moves to detector in Stage 2).
                "num_candidates_after_temporal_filter": int(len(l_kfs_sharing_words)),
                "num_candidates_after_connected_filter": int(len(l_kfs_sharing_words)),
                "num_candidates_after_common_words": int(num_candidates_after_common_words),
                "num_candidates_after_min_score": int(len(l_score_and_match)),
                "num_candidates_after_accumulation": int(len(candidates)),
                "num_candidates_after_consistency": -1,
                "top_candidate_id": int(top_kid),
                "top_candidate_score": float(scores[0]) if scores else 0.0,
                "top_candidate_acc_score": float(details.get(top_kid, {}).get("accumulated_score", 0.0)) if candidates else 0.0,
                "top_candidate_consistency": -1,
                "accepted_candidate_id": -1,
            },
        }

    def _detect_loop_candidates_dbow3_scored(
        self,
        keyframe: KeyFrame,
        *,
        raw_dbow_query: RawDbowQueryResult | None,
        min_score: float,
        min_delta_frames: int,
        min_score_context: dict | None,
    ) -> dict:
        raw_dbow_candidates = list(getattr(raw_dbow_query, "candidates", []) or [])
        candidate_pool = [candidate for candidate, _ in raw_dbow_candidates]
        raw_dbow_score_by_kid = {
            int(getattr(candidate, "kid", getattr(candidate, "id", -1))): float(score)
            for candidate, score in raw_dbow_candidates
        }
        return self._score_candidate_pool(
            keyframe,
            candidate_pool=candidate_pool,
            min_score=min_score,
            min_delta_frames=min_delta_frames,
            candidate_source="hybrid_dbow_scored",
            raw_dbow_score_by_kid=raw_dbow_score_by_kid,
            raw_dbow_candidate_count=len(raw_dbow_candidates),
            num_raw_inverted_candidates=0,
            min_score_context=min_score_context,
        )

    def _detect_loop_candidates_dbow_detector(
        self,
        keyframe: KeyFrame,
        *,
        raw_dbow_query: RawDbowQueryResult | None,
        min_score: float,
        min_delta_frames: int,
        min_score_context: dict | None,
    ) -> dict:
        trace_rows = {"inverted_word": [], "score_filter": [], "accumulation": []}
        if raw_dbow_query is None:
            return {
                "candidates": [],
                "scores": [],
                "details": {},
                "trace_rows": trace_rows,
                "profile": {
                    "num_db_keyframes_before_query": int(self.size()),
                    "candidate_source": "dbow_detector",
                    "num_raw_dbow_candidates": 0,
                    "num_raw_inverted_candidates": 0,
                    "num_candidates_after_temporal_filter": 0,
                    "num_candidates_after_connected_filter": 0,
                    "num_candidates_after_common_words": 0,
                    "num_candidates_after_min_score": 0,
                    "num_candidates_after_accumulation": 0,
                    "num_candidates_after_consistency": -1,
                    "top_candidate_id": -1,
                    "top_candidate_score": 0.0,
                    "top_candidate_acc_score": 0.0,
                    "top_candidate_consistency": -1,
                    "accepted_candidate_id": -1,
                },
            }

        candidates: list[KeyFrame] = []
        scores: list[float] = []
        details: dict[int, dict] = {}
        passed_connected_temporal = list(getattr(raw_dbow_query, "candidates", []) or [])
        for rank, (candidate, score) in enumerate(passed_connected_temporal, start=1):
            candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            passed_min_score = bool(float(score) >= float(min_score))
            candidate.loop_score = float(score)
            details[candidate_kid] = {
                "candidate_source": "dbow_detector",
                "candidate_rank": int(rank),
                "min_score": float(min_score),
                "common_words": None,
                "max_common_words": None,
                "common_word_ratio": None,
                "raw_dbow_score": float(score),
                "bow_score_raw": float(score),
                "bow_score_normalized": float(score),
                "accumulated_score": float(score),
                "best_accumulated_score": float(score),
                "is_connected": False,
                "temporal_gap": abs(int(getattr(keyframe, "id", 0)) - int(getattr(candidate, "id", 0))),
                "passed_common_word_filter": None,
                "passed_min_score_filter": passed_min_score,
            }
            if self.loop_retrieval_trace_enabled:
                trace_rows["score_filter"].append(
                    {
                        "current_kf_id": int(getattr(keyframe, "kid", getattr(keyframe, "id", -1))),
                        "candidate_kf_id": int(candidate_kid),
                        "pair_key": self._pair_key(
                            getattr(keyframe, "kid", getattr(keyframe, "id", -1)),
                            candidate_kid,
                        ),
                        "current_timestamp": getattr(keyframe, "timestamp", None),
                        "candidate_timestamp": getattr(candidate, "timestamp", None),
                        "candidate_source": "dbow_detector",
                        "bow_score": float(score),
                        "min_score": float(min_score),
                        "score_over_min_score": float(score) / float(min_score if abs(min_score) > 1e-12 else 1.0),
                        "connected_kf_count_for_min_score": int((min_score_context or {}).get("connected_kf_count", 0) or 0),
                        "min_score_source_kf_id": int((min_score_context or {}).get("min_score_source_kf_id", -1) or -1),
                        "passed_min_score_filter": passed_min_score,
                        "connected_scores_json": json.dumps((min_score_context or {}).get("connected_scores", []), sort_keys=True),
                    }
                )
            if passed_min_score:
                candidates.append(candidate)
                scores.append(float(score))

        best_direct_score = max(scores, default=0.0)
        for retained_rank, candidate in enumerate(candidates, start=1):
            candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            details[candidate_kid]["candidate_rank"] = int(retained_rank)
            details[candidate_kid]["best_accumulated_score"] = float(best_direct_score)

        top_candidate_id = -1
        top_candidate_score = 0.0
        if candidates:
            top_candidate_id = int(getattr(candidates[0], "kid", getattr(candidates[0], "id", -1)))
            top_candidate_score = float(scores[0])

        return {
            "candidates": candidates,
            "scores": scores,
            "details": details,
            "trace_rows": trace_rows,
            "profile": {
                "num_db_keyframes_before_query": int(self.size()),
                "candidate_source": "dbow_detector",
                "num_raw_dbow_candidates": int(len(passed_connected_temporal)),
                "num_raw_inverted_candidates": 0,
                "num_candidates_after_temporal_filter": int(len(passed_connected_temporal)),
                "num_candidates_after_connected_filter": int(len(passed_connected_temporal)),
                "num_candidates_after_common_words": int(len(passed_connected_temporal)),
                "num_candidates_after_min_score": int(len(candidates)),
                "num_candidates_after_accumulation": int(len(candidates)),
                "num_candidates_after_consistency": -1,
                "top_candidate_id": int(top_candidate_id),
                "top_candidate_score": float(top_candidate_score),
                "top_candidate_acc_score": float(top_candidate_score),
                "top_candidate_consistency": -1,
                "accepted_candidate_id": -1,
            },
        }

    def _score_candidate_pool(
        self,
        keyframe: KeyFrame,
        *,
        candidate_pool: list[KeyFrame],
        min_score: float,
        min_delta_frames: int,
        candidate_source: str,
        raw_dbow_score_by_kid: dict[int, float] | None,
        raw_dbow_candidate_count: int,
        num_raw_inverted_candidates: int,
        min_score_context: dict | None,
    ) -> dict:
        # [PHASE2-CONNECTED-TEMPORAL-WINDOW] was: set(keyframe.get_connected_keyframes())
        connected_keyframes = self._select_connected_keyframes_for_filter(keyframe)
        raw_unique: list[KeyFrame] = []
        seen = set()
        for candidate in candidate_pool:
            if candidate is None or candidate.is_bad() or candidate is keyframe or candidate in seen:
                continue
            raw_unique.append(candidate)
            seen.add(candidate)

        temporal_filtered = [
            candidate
            for candidate in raw_unique
            if abs(int(getattr(candidate, "id", 0)) - int(getattr(keyframe, "id", 0))) > int(min_delta_frames)
        ]
        connected_filtered = [candidate for candidate in temporal_filtered if candidate not in connected_keyframes]

        details: dict[int, dict] = {}
        trace_rows = {"inverted_word": [], "score_filter": [], "accumulation": []}
        if len(connected_filtered) == 0:
            return {
                "candidates": [],
                "scores": [],
                "details": details,
                "trace_rows": trace_rows,
                "profile": {
                    **self._empty_inverted_profile(raw_unique, temporal_filtered, connected_filtered),
                    "candidate_source": str(candidate_source),
                    "num_raw_dbow_candidates": int(raw_dbow_candidate_count),
                    "num_raw_inverted_candidates": int(num_raw_inverted_candidates),
                },
            }

        query_word_ids = {word_id for word_id, _ in self._bow_items(keyframe.g_des)}
        common_word_count_by_candidate: dict[int, int] = {}
        common_word_count_all: dict[int, int] = {}
        candidate_by_kid = {
            int(getattr(candidate, "kid", getattr(candidate, "id", -1))): candidate
            for candidate in raw_unique
        }
        for candidate in raw_unique:
            candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            candidate_word_ids = {word_id for word_id, _ in self._bow_items(getattr(candidate, "g_des", None))}
            common_word_count_all[candidate_kid] = int(len(query_word_ids.intersection(candidate_word_ids)))
        for candidate in connected_filtered:
            candidate.loop_query_id = keyframe.id
            candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            common_words = int(common_word_count_all.get(candidate_kid, 0))
            candidate.num_loop_words = common_words
            common_word_count_by_candidate[candidate_kid] = common_words

        max_common_words = max(common_word_count_by_candidate.values(), default=0)
        min_common_words = int(max_common_words * 0.8)
        common_word_threshold_ratio = float(min_common_words) / float(max_common_words or 1)

        if self.loop_retrieval_trace_enabled and str(candidate_source) == "classic_inverted":
            connected_ids = {
                int(getattr(candidate, "kid", getattr(candidate, "id", -1))) for candidate in connected_keyframes
            }
            temporal_ids = {
                int(getattr(candidate, "kid", getattr(candidate, "id", -1))) for candidate in temporal_filtered
            }
            connected_filtered_ids = {
                int(getattr(candidate, "kid", getattr(candidate, "id", -1))) for candidate in connected_filtered
            }
            for candidate_kid, candidate in sorted(candidate_by_kid.items()):
                common_words = int(common_word_count_all.get(candidate_kid, 0))
                temporal_gap = abs(int(getattr(candidate, "id", 0)) - int(getattr(keyframe, "id", 0)))
                trace_rows["inverted_word"].append(
                    {
                        "current_kf_id": int(getattr(keyframe, "kid", getattr(keyframe, "id", -1))),
                        "candidate_kf_id": int(candidate_kid),
                        "pair_key": self._pair_key(
                            getattr(keyframe, "kid", getattr(keyframe, "id", -1)),
                            candidate_kid,
                        ),
                        "current_timestamp": getattr(keyframe, "timestamp", None),
                        "candidate_timestamp": getattr(candidate, "timestamp", None),
                        "candidate_source": str(candidate_source),
                        "shared_words": common_words,
                        "max_common_words": int(max_common_words),
                        "common_word_ratio": float(common_words) / float(max_common_words or 1),
                        "common_word_threshold_ratio": common_word_threshold_ratio,
                        "passed_common_word_filter": bool(
                            candidate_kid in connected_filtered_ids and common_words > min_common_words
                        ),
                        "is_connected": bool(candidate_kid in connected_ids),
                        "temporal_gap": int(temporal_gap),
                        "passed_connected_filter": bool(candidate_kid not in connected_ids),
                        "passed_temporal_filter": bool(candidate_kid in temporal_ids),
                    }
                )

        score_and_match = []
        score_trace_rows: list[dict] = []
        num_candidates_after_common_words = 0
        for candidate in connected_filtered:
            candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            common_words = int(common_word_count_by_candidate.get(candidate_kid, 0))
            detail = {
                "candidate_source": str(candidate_source),
                "candidate_rank": -1,
                "min_score": float(min_score),
                "common_words": int(common_words),
                "max_common_words": int(max_common_words),
                "common_word_ratio": float(common_words) / float(max_common_words or 1),
                "raw_dbow_score": None if raw_dbow_score_by_kid is None else raw_dbow_score_by_kid.get(candidate_kid),
                "bow_score_raw": 0.0,
                "bow_score_normalized": 0.0,
                "accumulated_score": 0.0,
                "best_accumulated_score": 0.0,
                "is_connected": False,
                "temporal_gap": abs(int(getattr(keyframe, "id", 0)) - int(getattr(candidate, "id", 0))),
                "passed_common_word_filter": bool(common_words > min_common_words),
                "passed_min_score_filter": False,
            }
            details[candidate_kid] = detail
            if common_words > min_common_words:
                num_candidates_after_common_words += 1
                score = float(self.voc.score(keyframe.g_des, candidate.g_des))
                candidate.loop_score = score
                detail["bow_score_raw"] = score
                detail["bow_score_normalized"] = score
                detail["passed_min_score_filter"] = bool(score >= min_score)
                if self.loop_retrieval_trace_enabled:
                    score_trace_rows.append(
                        {
                            "current_kf_id": int(getattr(keyframe, "kid", getattr(keyframe, "id", -1))),
                            "candidate_kf_id": int(candidate_kid),
                            "pair_key": self._pair_key(
                                getattr(keyframe, "kid", getattr(keyframe, "id", -1)),
                                candidate_kid,
                            ),
                            "current_timestamp": getattr(keyframe, "timestamp", None),
                            "candidate_timestamp": getattr(candidate, "timestamp", None),
                            "candidate_source": str(candidate_source),
                            "bow_score": float(score),
                            "min_score": float(min_score),
                            "score_over_min_score": float(score) / float(min_score if abs(min_score) > 1e-12 else 1.0),
                            "connected_kf_count_for_min_score": int((min_score_context or {}).get("connected_kf_count", 0) or 0),
                            "min_score_source_kf_id": int((min_score_context or {}).get("min_score_source_kf_id", -1) or -1),
                            "passed_min_score_filter": bool(score >= min_score),
                            "connected_scores_json": json.dumps((min_score_context or {}).get("connected_scores", []), sort_keys=True),
                        }
                    )
                if score >= min_score:
                    score_and_match.append((score, candidate))
        trace_rows["score_filter"] = score_trace_rows

        if len(score_and_match) == 0:
            return {
                "candidates": [],
                "scores": [],
                "details": details,
                "trace_rows": trace_rows,
                "profile": {
                    **self._empty_inverted_profile(raw_unique, temporal_filtered, connected_filtered),
                    "candidate_source": str(candidate_source),
                    "num_raw_dbow_candidates": int(raw_dbow_candidate_count),
                    "num_raw_inverted_candidates": int(num_raw_inverted_candidates),
                    "num_candidates_after_common_words": int(num_candidates_after_common_words),
                    "num_candidates_after_min_score": 0,
                },
            }

        acc_score_and_match = []
        best_acc_score = float(min_score)
        accumulation_payloads: list[dict] = []
        for score, candidate in score_and_match:
            neighbors = candidate.get_best_covisible_keyframes(10)
            best_score = score
            acc_score = score
            best_keyframe = candidate
            group_ids = [int(getattr(candidate, "kid", getattr(candidate, "id", -1)))]
            group_scores = [float(score)]

            for neighbor in neighbors:
                if (
                    neighbor.loop_query_id == keyframe.id
                    and neighbor.num_loop_words > min_common_words
                ):
                    acc_score += float(getattr(neighbor, "loop_score", 0.0))
                    group_ids.append(int(getattr(neighbor, "kid", getattr(neighbor, "id", -1))))
                    group_scores.append(float(getattr(neighbor, "loop_score", 0.0)))
                    if float(getattr(neighbor, "loop_score", 0.0)) > best_score:
                        best_keyframe = neighbor
                        best_score = float(getattr(neighbor, "loop_score", 0.0))

            acc_score_and_match.append((acc_score, best_keyframe))
            best_acc_score = max(best_acc_score, acc_score)
            accumulation_payloads.append(
                {
                    "candidate": candidate,
                    "candidate_kid": int(getattr(candidate, "kid", getattr(candidate, "id", -1))),
                    "candidate_score": float(score),
                    "group_ids": group_ids,
                    "group_scores": group_scores,
                    "accumulated_score": float(acc_score),
                    "best_candidate_id_in_group": int(getattr(best_keyframe, "kid", getattr(best_keyframe, "id", -1))),
                }
            )

        min_score_to_retain = 0.75 * best_acc_score
        already_added = set()
        candidates = []
        scores = []
        retained_rank_by_kid: dict[int, int] = {}
        ranked = sorted(
            acc_score_and_match,
            key=lambda item: (float(item[0]), float(getattr(item[1], "loop_score", 0.0))),
            reverse=True,
        )
        for acc_score, candidate in ranked:
            if acc_score > min_score_to_retain and candidate not in already_added:
                candidates.append(candidate)
                scores.append(float(getattr(candidate, "loop_score", 0.0)))
                already_added.add(candidate)
            candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            details.setdefault(candidate_kid, {})
            details[candidate_kid]["accumulated_score"] = float(acc_score)
            details[candidate_kid]["best_accumulated_score"] = float(best_acc_score)

        for rank, candidate in enumerate(candidates, start=1):
            candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            details.setdefault(candidate_kid, {})
            details[candidate_kid]["candidate_rank"] = int(rank)
            retained_rank_by_kid[candidate_kid] = int(rank)

        if self.loop_retrieval_trace_enabled:
            for payload in accumulation_payloads:
                candidate_kid = int(payload["candidate_kid"])
                best_candidate_id = int(payload["best_candidate_id_in_group"])
                acc_score = float(payload["accumulated_score"])
                passed_accumulated = bool(acc_score > min_score_to_retain)
                retained_candidate = bool(
                    passed_accumulated
                    and best_candidate_id == candidate_kid
                    and candidate_by_kid.get(candidate_kid) in already_added
                )
                trace_rows["accumulation"].append(
                    {
                        "current_kf_id": int(getattr(keyframe, "kid", getattr(keyframe, "id", -1))),
                        "candidate_kf_id": int(candidate_kid),
                        "pair_key": self._pair_key(
                            getattr(keyframe, "kid", getattr(keyframe, "id", -1)),
                            candidate_kid,
                        ),
                        "current_timestamp": getattr(keyframe, "timestamp", None),
                        "candidate_timestamp": getattr(payload["candidate"], "timestamp", None),
                        "candidate_source": str(candidate_source),
                        "candidate_group_size": int(len(payload["group_ids"])),
                        "candidate_group_ids": json.dumps(payload["group_ids"]),
                        "candidate_group_scores": json.dumps(payload["group_scores"]),
                        "accumulated_score": acc_score,
                        "best_accumulated_score": float(best_acc_score),
                        "accumulated_score_ratio": acc_score / float(best_acc_score or 1.0),
                        "accumulation_threshold_ratio": 0.75,
                        "passed_accumulated_score_filter": passed_accumulated,
                        "best_candidate_id_in_group": int(best_candidate_id),
                        "retained_candidate": retained_candidate,
                        "retained_rank": int(retained_rank_by_kid.get(candidate_kid, -1)),
                    }
                )

        return {
            "candidates": candidates,
            "scores": scores,
            "details": details,
            "trace_rows": trace_rows,
            "profile": {
                "num_db_keyframes_before_query": int(self.size()),
                "candidate_source": str(candidate_source),
                "num_raw_dbow_candidates": int(raw_dbow_candidate_count),
                "num_raw_inverted_candidates": int(num_raw_inverted_candidates),
                "num_candidates_after_temporal_filter": int(len(temporal_filtered)),
                "num_candidates_after_connected_filter": int(len(connected_filtered)),
                "num_candidates_after_common_words": int(num_candidates_after_common_words),
                "num_candidates_after_min_score": int(len(score_and_match)),
                "num_candidates_after_accumulation": int(len(candidates)),
                "num_candidates_after_consistency": -1,
                "top_candidate_id": int(getattr(candidates[0], "kid", getattr(candidates[0], "id", -1))) if candidates else -1,
                "top_candidate_score": float(scores[0]) if scores else 0.0,
                "top_candidate_acc_score": float(details.get(int(getattr(candidates[0], "kid", getattr(candidates[0], "id", -1))), {}).get("accumulated_score", 0.0)) if candidates else 0.0,
                "top_candidate_consistency": -1,
                "accepted_candidate_id": -1,
            },
        }

    def _detect_loop_candidates_dbow3_raw(
        self,
        keyframe: KeyFrame,
        *,
        min_score: float,
        min_delta_frames: int,
        query_k: int | None = None,
    ) -> RawDbowQueryResult | None:
        if self.dbow_database is None:
            return None
        query_k = max(1, int(query_k if query_k is not None else self.size()))
        try:
            results = self.dbow_database.query(
                self.voc.to_native_bow(keyframe.g_des),
                query_k,
            )
        except Exception:
            return None

        # [PHASE2-CONNECTED-TEMPORAL-WINDOW] was: set(keyframe.get_connected_keyframes())
        connected = self._select_connected_keyframes_for_filter(keyframe)
        candidates: list[tuple[KeyFrame, float]] = []
        trace_rows: list[dict] = []
        for result in results:
            entry_id = int(getattr(result, "id", -1))
            score = float(getattr(result, "score", 0.0))
            candidate = self._entry_to_keyframe.get(entry_id)
            candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1))) if candidate is not None else -1
            temporal_gap = (
                abs(int(getattr(candidate, "id", 0)) - int(getattr(keyframe, "id", 0)))
                if candidate is not None
                else -1
            )
            is_self = bool(candidate is keyframe) if candidate is not None else False
            is_bad = bool(candidate.is_bad()) if candidate is not None else False
            is_connected = bool(candidate in connected) if candidate is not None else False
            passes_connected = bool(candidate is not None and not is_connected and not is_self and not is_bad)
            passes_temporal = bool(candidate is not None and temporal_gap > int(min_delta_frames))
            if self.loop_retrieval_trace_enabled:
                trace_rows.append(
                    {
                        "current_kf_id": int(getattr(keyframe, "kid", getattr(keyframe, "id", -1))),
                        "candidate_kf_id": int(candidate_kid),
                        "pair_key": self._pair_key(
                            getattr(keyframe, "kid", getattr(keyframe, "id", -1)),
                            candidate_kid,
                        ) if candidate_kid >= 0 else "",
                        "current_timestamp": getattr(keyframe, "timestamp", None),
                        "candidate_timestamp": getattr(candidate, "timestamp", None) if candidate is not None else None,
                        "raw_rank": int(len(trace_rows) + 1),
                        "raw_score": float(score),
                        "raw_source": "dbow3_query",
                        "db_size_before_query": int(self.size()),
                        "raw_query_k": int(query_k),
                        "raw_result_count": int(len(results)),
                        "is_self": bool(is_self),
                        "is_bad": bool(is_bad),
                        "is_connected": bool(is_connected),
                        "temporal_gap": int(temporal_gap),
                        "would_pass_connected_filter": bool(passes_connected),
                        "would_pass_temporal_filter": bool(passes_temporal),
                    }
                )
            if candidate is None or candidate is keyframe or candidate.is_bad():
                continue
            if candidate in connected:
                continue
            if abs(int(getattr(candidate, "id", 0)) - int(getattr(keyframe, "id", 0))) <= int(min_delta_frames):
                continue
            candidate.loop_score = score
            if all(existing is not candidate for existing, _ in candidates):
                candidates.append((candidate, score))
        return RawDbowQueryResult(
            candidates=candidates,
            trace_rows=trace_rows,
            query_k=int(query_k),
            result_count=int(len(results)),
        )

    def score(self, bow_a, bow_b) -> float:
        if not self.available:
            return 0.0
        return self.voc.score(bow_a, bow_b)

    def _reset_dbow_database(self) -> None:
        self.dbow_database = None
        self._entry_to_keyframe = {}
        self._keyframe_to_entry = {}
        if self.voc is None or not getattr(self.voc, "available", False):
            return
        pydbow3 = getattr(self.voc, "pydbow3", None)
        native_voc = getattr(self.voc, "voc", None)
        if pydbow3 is None or native_voc is None:
            return
        try:
            database = pydbow3.Database()
            database.setVocabulary(native_voc)
            self.dbow_database = database
        except Exception:
            self.dbow_database = None

    def _make_raw_dbow_candidate_details(
        self,
        keyframe: KeyFrame,
        raw_dbow_candidates: list[tuple[KeyFrame, float]],
        *,
        min_score: float,
    ) -> dict[int, dict]:
        details = {}
        best_score = max((float(score) for _, score in raw_dbow_candidates), default=0.0)
        for rank, (candidate, score) in enumerate(raw_dbow_candidates, start=1):
            candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            details[candidate_kid] = {
                "candidate_source": "dbow3_raw",
                "candidate_rank": int(rank),
                "min_score": float(min_score),
                "common_words": 0,
                "max_common_words": 0,
                "common_word_ratio": 0.0,
                "raw_dbow_score": float(score),
                "bow_score_raw": float(score),
                "bow_score_normalized": float(score),
                "accumulated_score": float(score),
                "best_accumulated_score": float(best_score),
                "is_connected": False,
                "temporal_gap": abs(int(getattr(keyframe, "id", 0)) - int(getattr(candidate, "id", 0))),
            }
        return details

    def _make_raw_dbow_profile(
        self,
        keyframe: KeyFrame,
        raw_dbow_candidates: list[tuple[KeyFrame, float]],
        *,
        inverted_result: dict,
        candidate_source: str,
    ) -> dict:
        top_candidate_id = -1
        top_candidate_score = 0.0
        if raw_dbow_candidates:
            top_candidate = raw_dbow_candidates[0][0]
            top_candidate_id = int(getattr(top_candidate, "kid", getattr(top_candidate, "id", -1)))
            top_candidate_score = float(raw_dbow_candidates[0][1])
        return {
            "num_db_keyframes_before_query": int(self.size()),
            "candidate_source": str(candidate_source),
            "num_raw_dbow_candidates": int(len(raw_dbow_candidates)),
            "num_raw_inverted_candidates": int(inverted_result.get("profile", {}).get("num_raw_inverted_candidates", 0)),
            "num_candidates_after_temporal_filter": int(len(raw_dbow_candidates)),
            "num_candidates_after_connected_filter": int(len(raw_dbow_candidates)),
            "num_candidates_after_common_words": int(len(raw_dbow_candidates)),
            "num_candidates_after_min_score": int(len(raw_dbow_candidates)),
            "num_candidates_after_accumulation": int(len(raw_dbow_candidates)),
            "num_candidates_after_consistency": -1,
            "top_candidate_id": int(top_candidate_id),
            "top_candidate_score": float(top_candidate_score),
            "top_candidate_acc_score": float(top_candidate_score),
            "top_candidate_consistency": -1,
            "accepted_candidate_id": -1,
        }

    def _empty_inverted_profile(self, raw_unique, temporal_filtered, connected_filtered) -> dict:
        return {
            "num_db_keyframes_before_query": int(self.size()),
            "candidate_source": "classic_inverted",
            "num_raw_dbow_candidates": 0,
            "num_raw_inverted_candidates": int(len(raw_unique)),
            "num_candidates_after_temporal_filter": int(len(temporal_filtered)),
            "num_candidates_after_connected_filter": int(len(connected_filtered)),
            "num_candidates_after_common_words": 0,
            "num_candidates_after_min_score": 0,
            "num_candidates_after_accumulation": 0,
            "num_candidates_after_consistency": -1,
            "top_candidate_id": -1,
            "top_candidate_score": 0.0,
            "top_candidate_acc_score": 0.0,
            "top_candidate_consistency": -1,
            "accepted_candidate_id": -1,
        }

    @staticmethod
    def _pair_key(kf_a: int, kf_b: int) -> str:
        return f"{min(int(kf_a), int(kf_b))}-{max(int(kf_a), int(kf_b))}"

    @staticmethod
    def _build_source_comparison_row(
        keyframe: KeyFrame,
        *,
        dbow_result: dict,
        inverted_result: dict,
        chosen_candidates: list[KeyFrame],
        chosen_source: str,
    ) -> dict:
        dbow_ids = [
            int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            for candidate in dbow_result.get("candidates", [])
        ]
        inverted_ids = [
            int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            for candidate in inverted_result.get("candidates", [])
        ]
        chosen_ids = [int(getattr(candidate, "kid", getattr(candidate, "id", -1))) for candidate in chosen_candidates]
        dbow_set = set(dbow_ids)
        inverted_set = set(inverted_ids)
        return {
            "kf_id": int(getattr(keyframe, "kid", getattr(keyframe, "id", -1))),
            "timestamp": getattr(keyframe, "timestamp", None),
            "candidate_source": str(chosen_source),
            "dbow3_candidates": dbow_ids,
            "inverted_file_candidates": inverted_ids,
            "intersection_candidates": sorted(dbow_set.intersection(inverted_set)),
            "dbow3_only_candidates": sorted(dbow_set.difference(inverted_set)),
            "inverted_only_candidates": sorted(inverted_set.difference(dbow_set)),
            "chosen_candidates": chosen_ids,
        }

    @staticmethod
    def _bow_items(bow) -> list[tuple[int, float]]:
        if bow is None:
            return []
        if hasattr(bow, "toVec"):
            return [(int(word_id), float(weight)) for word_id, weight in bow.toVec()]
        if isinstance(bow, dict):
            return [(int(word_id), float(weight)) for word_id, weight in bow.items()]
        return [(int(word_id), float(weight)) for word_id, weight in list(bow)]
