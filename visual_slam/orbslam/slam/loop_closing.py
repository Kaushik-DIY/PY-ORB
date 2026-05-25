"""
Loop-closing orchestration for the sparse map.
This module detects loops, verifies geometry, corrects drift, and triggers map optimization.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import threading
from typing import Optional

import g2o
import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.bow_matcher import BoWGuidedMatcher
from visual_slam.orbslam.slam.essential_graph import (
    EssentialGraphResult,
    apply_correction_to_map_points,
    optimize_essential_graph_se3,
)
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.frame import ensure_frame_feature_arrays, prepare_input_data_for_sim3solver
from visual_slam.orbslam.slam.geometry_matchers import ProjectionFuseDiagnostics, ProjectionMatcher
from visual_slam.orbslam.slam.global_ba import GlobalBAResult, GlobalBundleAdjuster
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.loop_detector import LoopDetector, LoopDetectorOutput
from visual_slam.orbslam.slam.loop_oracle import TumLoopOracle
from visual_slam.orbslam.slam.rotation_histogram import RotationHistogram
from visual_slam.orbslam.slam.sim3_pose import Sim3Pose
from visual_slam.orbslam.slam.sim3_solver import Sim3Estimate, estimate_scale_fixed_sim3
import visual_slam.orbslam.slam.optimizer_g2o as _optimizer_g2o

try:
    import sim3solver as _sim3solver
    _SIM3SOLVER_AVAILABLE = True
except ImportError:
    _sim3solver = None
    _SIM3SOLVER_AVAILABLE = False


# Track one covisibility group and its accumulated loop-consistency score.
@dataclass
class ConsistencyGroup:
    keyframes: set = field(default_factory=set)
    consistency: int = 0


# Store diagnostics for one loop-detection and correction attempt.
@dataclass
class LoopDiagnostics:
    candidates: int = 0
    accepted: int = 0
    rejected_by_bow: int = 0
    rejected_by_consistency: int = 0
    rejected_by_geometry: int = 0
    corrected_keyframes: int = 0
    corrected_points: int = 0
    fused_points: int = 0
    fusion_diagnostics: Optional[ProjectionFuseDiagnostics] = None
    optimization_result: Optional[EssentialGraphResult] = None
    global_ba_result: Optional[GlobalBAResult] = None
    global_ba_started: bool = False
    global_ba_success: bool = False
    global_ba_aborted: bool = False
    global_ba_reason: str = ""
    global_ba_num_keyframes: int = 0
    global_ba_num_map_points: int = 0
    global_ba_num_edges: int = 0
    global_ba_num_inliers: int = 0
    global_ba_num_outliers: int = 0
    global_ba_mean_error_before: Optional[float] = None
    global_ba_mean_error_after: Optional[float] = None
    global_ba_elapsed_sec: float = 0.0
    unavailable_reason: Optional[str] = None
    loop_debug_records: list[dict] = field(default_factory=list)
    candidate_pair_reports: list[dict] = field(default_factory=list)
    loop_candidate_oracle_rows: list[dict] = field(default_factory=list)
    loop_retrieval_profile_rows: list[dict] = field(default_factory=list)
    loop_candidate_source_comparison_rows: list[dict] = field(default_factory=list)
    loop_keyframe_density_rows: list[dict] = field(default_factory=list)
    loop_raw_dbow_trace_rows: list[dict] = field(default_factory=list)
    loop_inverted_word_trace_rows: list[dict] = field(default_factory=list)
    loop_score_filter_trace_rows: list[dict] = field(default_factory=list)
    loop_accumulation_trace_rows: list[dict] = field(default_factory=list)
    loop_retained_candidate_trace_rows: list[dict] = field(default_factory=list)
    loop_gt_positive_trace_rows: list[dict] = field(default_factory=list)
    loop_consistency_progression_rows: list[dict] = field(default_factory=list)
    loop_geometry_trace_rows: list[dict] = field(default_factory=list)


# Accumulate consistent loop groups across successive loop queries.
class LoopGroupConsistencyChecker:
    def __init__(self, consistency_threshold: int = 3):
        self.consistent_groups: list[ConsistencyGroup] = []
        self.consistency_threshold = int(consistency_threshold)
        self.enough_consistent_candidates: list[KeyFrame] = []
        self.last_candidate_debug: dict[int, dict] = {}

    def clear_consistency_groups(self) -> None:
        self.consistent_groups = []
        self.enough_consistent_candidates = []
        self.last_candidate_debug = {}

    def check_candidates(self, current_keyframe: KeyFrame, candidate_keyframes: list[KeyFrame]) -> bool:
        self.enough_consistent_candidates = []
        self.last_candidate_debug = {}
        current_consistent_groups = []
        group_updated = [False] * len(self.consistent_groups)
        previous_group_ids = [
            sorted(int(getattr(kf, "kid", getattr(kf, "id", -1))) for kf in group.keyframes)
            for group in self.consistent_groups
        ]

        for candidate in candidate_keyframes:
            if candidate is None or candidate.is_bad():
                continue

            candidate_group = set(candidate.get_connected_keyframes())
            candidate_group.add(candidate)
            candidate_group_ids = sorted(
                int(getattr(kf, "kid", getattr(kf, "id", -1))) for kf in candidate_group
            )

            enough_consistent = False
            consistent_for_some_group = False
            best_overlap_count = 0
            best_previous_consistency = 0
            best_consistency = 0

            for idx, previous_group in enumerate(self.consistent_groups):
                overlap = candidate_group.intersection(previous_group.keyframes)
                if overlap:
                    consistent_for_some_group = True
                    best_overlap_count = max(best_overlap_count, len(overlap))
                    best_previous_consistency = max(
                        best_previous_consistency,
                        int(previous_group.consistency),
                    )
                    current_consistency = previous_group.consistency + 1
                    best_consistency = max(best_consistency, current_consistency)

                    if not group_updated[idx]:
                        current_consistent_groups.append(
                            ConsistencyGroup(candidate_group, current_consistency)
                        )
                        group_updated[idx] = True

                    if (
                        current_consistency >= self.consistency_threshold
                        and not enough_consistent
                    ):
                        self.enough_consistent_candidates.append(candidate)
                        enough_consistent = True

            if not consistent_for_some_group:
                current_consistent_groups.append(ConsistencyGroup(candidate_group, 0))

            candidate_key = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            self.last_candidate_debug[candidate_key] = {
                "current_group_kf_ids": candidate_group_ids,
                "candidate_group_kf_ids": candidate_group_ids,
                "previous_consistency_group_ids": previous_group_ids,
                "consistency_overlap_count": best_overlap_count,
                "consistency_score_before": best_previous_consistency,
                "consistency_score_after": best_consistency,
                "consistency_count": best_consistency,
                "consistency_required": self.consistency_threshold,
                "passed_consistency": bool(enough_consistent),
            }

        self.consistent_groups = current_consistent_groups
        return len(self.enough_consistent_candidates) > 0


# Verify a loop candidate through matching, rigid alignment, and refinement.
class LoopGeometryChecker:
    def __init__(
        self,
        min_matches: int = Parameters.kLoopClosingGeometryCheckerMinKpsMatches,
        keyframe_database=None,
        is_monocular: bool = False,
    ):
        self.min_matches = int(min_matches)
        self.keyframe_database = keyframe_database
        self.is_monocular = bool(is_monocular)
        self.success_loop_kf: Optional[KeyFrame] = None
        self.success_sim3: Optional[Sim3Estimate] = None
        self.success_loop_kf_sim3_pose: Optional[Sim3Pose] = None
        self.success_map_point_matches: list = []
        self.success_map_point_matches_idxs = np.array([], dtype=np.int32)
        self.success_loop_map_points = set()
        self.num_last_matches = 0
        self.num_last_inliers = 0
        self.last_error = None
        self.last_bow_guided_matching_available = False
        self.last_fallback_descriptor_matching = False
        self.last_match_diagnostics = None
        self.last_match_distances = np.array([], dtype=np.float32)
        self.last_guided_projection_matches = 0
        self.last_final_matches = 0
        self.last_candidate_reports: dict[int, dict] = {}
        self.runtime_profiler = None

    def _profile_section(self, name: str):
        if self.runtime_profiler is None:
            return nullcontext()
        return self.runtime_profiler.section(name)

    def check_candidates(self, current_keyframe: KeyFrame, candidate_keyframes: list[KeyFrame]) -> bool:
        self.success_loop_kf = None
        self.success_sim3 = None
        self.success_loop_kf_sim3_pose = None
        self.success_map_point_matches = []
        self.success_map_point_matches_idxs = np.array([], dtype=np.int32)
        self.success_loop_map_points = set()
        self.num_last_matches = 0
        self.num_last_inliers = 0
        self.last_error = None
        self.last_bow_guided_matching_available = False
        self.last_fallback_descriptor_matching = False
        self.last_match_diagnostics = None
        self.last_match_distances = np.array([], dtype=np.float32)
        self.last_guided_projection_matches = 0
        self.last_final_matches = 0
        self.last_candidate_reports = {}

        # Stage 1: BoW matching for all candidates
        kp_match_idxs: dict[int, tuple] = {}
        for candidate in candidate_keyframes:
            if candidate is None or candidate is current_keyframe or candidate.is_bad():
                continue
            idxs_cur, idxs_cand = self.match_keyframes(current_keyframe, candidate)
            kp_match_idxs[id(candidate)] = (idxs_cur, idxs_cand)

        # Stage 2: Build sim3 solvers for candidates with enough BoW matches
        solvers: list = []
        solver_mp_idxs: dict[int, tuple] = {}
        considered_candidates: list[KeyFrame] = []
        map_points1 = current_keyframe.get_points()
        n1 = len(map_points1)

        for candidate in candidate_keyframes:
            if candidate is None or candidate is current_keyframe or candidate.is_bad():
                continue

            candidate_key = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            idxs_cur, idxs_cand = kp_match_idxs.get(id(candidate), (np.array([], dtype=np.int32), np.array([], dtype=np.int32)))
            n_bow = int(len(idxs_cur))
            self.num_last_matches = max(self.num_last_matches, n_bow)

            report = self._base_candidate_report(current_keyframe, candidate)
            report["geometry_method"] = "sim3solver+optimize_sim3"
            report["bow_matches_after_orientation"] = n_bow
            report["bow_match_pairs"] = n_bow
            self._merge_match_diagnostics(report)
            self.last_candidate_reports[candidate_key] = report

            if n_bow < self.min_matches:
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = f"too few BoW matches ({n_bow} < {self.min_matches})"
                self.last_error = "too few loop geometry matches"
                continue

            if not _SIM3SOLVER_AVAILABLE:
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = "sim3solver module not available"
                continue

            p3d_w1, p3d_w2, s2_1, s2_2, idxs1, idxs2 = prepare_input_data_for_sim3solver(
                current_keyframe, candidate, idxs_cur, idxs_cand
            )
            n_3d = int(len(idxs1))
            report["geometry_input_correspondences"] = n_3d
            report["bow_matches_with_valid_mappoints"] = n_3d

            if n_3d < self.min_matches:
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = f"too few 3D correspondences ({n_3d} < {self.min_matches})"
                self.last_error = "too few valid 3D loop correspondences"
                continue

            solver_mp_idxs[id(candidate)] = (idxs1, idxs2)
            candidate.set_not_erase()

            si = _sim3solver.Sim3SolverInput()
            si.fix_scale = not self.is_monocular
            cur_Tcw = np.asarray(current_keyframe.Tcw(), dtype=np.float32).reshape(4, 4)
            cand_Tcw = np.asarray(candidate.Tcw(), dtype=np.float32).reshape(4, 4)
            si.K1 = np.asarray(current_keyframe.camera.K, dtype=np.float32)
            si.Rcw1 = np.ascontiguousarray(cur_Tcw[:3, :3])
            si.tcw1 = np.ascontiguousarray(cur_Tcw[:3, 3])
            si.K2 = np.asarray(candidate.camera.K, dtype=np.float32)
            si.Rcw2 = np.ascontiguousarray(cand_Tcw[:3, :3])
            si.tcw2 = np.ascontiguousarray(cand_Tcw[:3, 3])
            si.points_3d_w1 = p3d_w1
            si.points_3d_w2 = p3d_w2
            drift_factor = Parameters.kLoopClosingSim3DriftSigmaFactor
            si.sigmas2_1 = s2_1 * drift_factor
            si.sigmas2_2 = s2_2 * drift_factor

            solver = _sim3solver.Sim3Solver(si)
            # [PHASE2-SIM3-INLIER-RELAX] (added 2026-05-18) was: 0.99, 8, 300
            # Override the RANSAC min-inliers used for convergence. Set
            # kLoopClosingSim3SeedInliersOverride=0 in config_parameters.py to
            # restore the legacy 8-inlier threshold. See
            # CODEX_LAB_PHASE2_CONNECTED_FILTER_TEMPORAL_WINDOW.md.
            _sim3_override = int(getattr(Parameters, "kLoopClosingSim3SeedInliersOverride", 0) or 0)
            _sim3_ransac_min_inliers = _sim3_override if _sim3_override > 0 else 8
            solver.set_ransac_parameters(0.99, _sim3_ransac_min_inliers, 300)
            solvers.append(solver)
            considered_candidates.append(candidate)

        # Stage 3: RANSAC → search_by_sim3 → optimize_sim3
        for i, kf in enumerate(considered_candidates):
            candidate_key = int(getattr(kf, "kid", getattr(kf, "id", -1)))
            report = self.last_candidate_reports[candidate_key]

            # Loop iterate(5) until convergence or budget exhausted (max 300 iters).
            is_converged = False
            is_no_more = False
            inlier_flags = []
            num_inliers = 0
            while not (is_converged or is_no_more):
                _, is_no_more, inlier_flags, num_inliers, is_converged = solvers[i].iterate(5)
            inlier_flags = np.array(inlier_flags, dtype=bool)
            report["geometry_ransac_inliers"] = int(num_inliers)

            # Relaxed acceptance: sparse KF maps (1 KF per ~29 frames) produce 6-17
            # genuine inliers — below the solver's strict is_converged threshold (≥20).
            # Accept the best-found estimate when budget is exhausted AND inliers meet
            # the seed threshold, consistent with pyslam's Python-based approach.
            # [PHASE2-SIM3-INLIER-RELAX] (added 2026-05-18) override min_seed when
            # kLoopClosingSim3SeedInliersOverride > 0. Set it to 0 to restore the
            # legacy kLoopClosingSE3GuidedMinSeedInliers behavior.
            _override = int(getattr(Parameters, "kLoopClosingSim3SeedInliersOverride", 0) or 0)
            min_seed = _override if _override > 0 else Parameters.kLoopClosingSE3GuidedMinSeedInliers
            if not is_converged and not (is_no_more and num_inliers >= min_seed):
                report["rejection_stage"] = "geometry"
                report["rejection_reason"] = (
                    f"Sim3Solver did not converge (inliers={num_inliers} < {min_seed} seed threshold)"
                )
                continue

            R12 = np.asarray(solvers[i].get_estimated_rotation(), dtype=np.float64)
            t12 = np.asarray(solvers[i].get_estimated_translation(), dtype=np.float64).ravel()
            scale12 = float(solvers[i].get_estimated_scale())

            idxs1, idxs2 = solver_mp_idxs[id(kf)]
            idxs1 = idxs1[inlier_flags]
            idxs2 = idxs2[inlier_flags]

            # Stage 7: search_by_sim3 (guided bidirectional projection matching)
            with self._profile_section("loop.search_by_sim3"):
                num_found, matches12, matches21 = ProjectionMatcher.search_by_sim3(
                    current_keyframe, kf, idxs1, idxs2, scale12, R12, t12
                )
            self.last_guided_projection_matches = max(self.last_guided_projection_matches, num_found)
            report["guided_projection_matches"] = int(num_found)

            map_points2 = kf.get_points()
            map_point_matches12 = [
                (map_points2[int(idx)] if 0 <= int(idx) < len(map_points2) else None)
                for idx in matches12
            ]

            # Stage 8: optimize_sim3 with g2o
            with self._profile_section("loop.optimize_sim3"):
                num_opt_inliers, R12_opt, t12_opt, scale12_opt, delta_err = _optimizer_g2o.optimize_sim3(
                    current_keyframe, kf, map_points1, map_point_matches12,
                    R12, t12, scale12,
                    th2=float(Parameters.kLoopClosingTh2) * float(Parameters.kLoopClosingSim3DriftSigmaFactor),
                    fix_scale=not self.is_monocular,
                )

            report["geometry_refined_inliers"] = int(num_opt_inliers) if num_opt_inliers else 0
            self.num_last_inliers = max(self.num_last_inliers, int(num_opt_inliers or 0))

            if (
                num_opt_inliers is not None
                and num_opt_inliers > Parameters.kLoopClosingGeometryCheckerMinKpsMatches
                and delta_err <= 0
            ):
                R12 = R12_opt
                t12 = t12_opt
                scale12 = scale12_opt

                # Compute corrected Sim3 pose: Sc1c2 @ Tc2w = Sc1w
                kf_Tcw = np.asarray(kf.Tcw(), dtype=np.float64).reshape(4, 4)
                sim3_pose = Sim3Pose(R12, t12, scale12) @ Sim3Pose().from_se3_matrix(kf_Tcw)
                self.success_loop_kf_sim3_pose = sim3_pose

                # Build Sim3Estimate for backward-compat with correct_loop SE3 path
                self.success_sim3 = Sim3Estimate(
                    success=True,
                    R=R12, t=t12, scale=scale12,
                    inlier_mask=np.ones(num_opt_inliers, dtype=bool),
                    mean_error=0.0,
                )
                self.success_loop_kf = kf
                self.success_map_point_matches = list(map_point_matches12)
                self.success_map_point_matches_idxs = np.asarray(matches12, dtype=np.int32)
                report["seed_inliers"] = int(num_opt_inliers)
                report["final_inliers"] = int(num_opt_inliers)
                report["accepted"] = True
                report["accepted_or_rejected"] = "accepted"
                report["rejection_stage"] = ""
                report["rejection_reason"] = ""
                break
            else:
                report["rejection_stage"] = "geometry"
                reason = "optimize_sim3 failed"
                if num_opt_inliers is not None:
                    reason = f"optimize_sim3: inliers={num_opt_inliers}, delta_err={delta_err:.4f}"
                report["rejection_reason"] = reason

        # Release non-winning candidates (Stage 11 erase-lock parity)
        for kf in candidate_keyframes:
            if kf is not None and kf is not self.success_loop_kf:
                kf.set_erase()

        if self.success_loop_kf is not None:
            # Covisibility expansion + final gate (Stage 10 basic version)
            success_covisible_group = self.success_loop_kf.get_covisible_keyframes()
            success_covisible_group.append(self.success_loop_kf)
            self.success_loop_map_points = set()
            for kf_cov in success_covisible_group:
                for mp in kf_cov.get_matched_good_points():
                    if mp is not None and not mp.is_bad():
                        self.success_loop_map_points.add(mp)

            # Use the Sim3 pose to derive corrected SE3 for projection search
            Tcw_corrected = np.asarray(
                self.success_loop_kf_sim3_pose.to_se3_matrix(), dtype=np.float64
            ).reshape(4, 4)

            with self._profile_section("loop.search_more_projection"):
                num_new, self.success_map_point_matches, search_more_diag = (
                    ProjectionMatcher.search_more_map_points_by_projection(
                        list(self.success_loop_map_points),
                        current_keyframe,
                        Tcw_corrected,
                        self.success_map_point_matches,
                        self.success_map_point_matches_idxs,
                        max_reproj_distance=Parameters.kLoopClosingMaxReprojectionDistanceMapSearch,
                        return_diagnostics=True,
                    )
                )

            num_matched = sum(m is not None for m in self.success_map_point_matches)
            success_key = int(getattr(self.success_loop_kf, "kid", getattr(self.success_loop_kf, "id", -1)))
            if success_key in self.last_candidate_reports:
                r = self.last_candidate_reports[success_key]
                r["new_projection_matches"] = int(num_new)
                r["total_final_matches"] = int(num_matched)
                r["final_matched_map_points"] = int(num_matched)
                r["final_gate_threshold"] = int(Parameters.kLoopClosingMinNumMatchedMapPoints)
                r["candidate_group_map_points"] = int(len(self.success_loop_map_points))
                r["candidate_covisible_points"] = int(len(self.success_loop_map_points))
                r["candidate_group_size"] = int(len(success_covisible_group))
                r["projected_visible_points"] = int(search_more_diag.get("projected_visible_points", 0))

            if num_matched < Parameters.kLoopClosingMinNumMatchedMapPoints:
                self.last_error = (
                    f"too few final matched map points "
                    f"({num_matched} < {Parameters.kLoopClosingMinNumMatchedMapPoints})"
                )
                if success_key in self.last_candidate_reports:
                    r = self.last_candidate_reports[success_key]
                    r["accepted"] = False
                    r["accepted_or_rejected"] = "rejected"
                    r["rejection_stage"] = "geometry"
                    r["rejection_reason"] = self.last_error
                self.success_loop_kf = None
                for kf in candidate_keyframes:
                    if kf is not None:
                        kf.set_erase()

        return self.success_loop_kf is not None

    def match_keyframes(self, current_keyframe: KeyFrame, candidate: KeyFrame):
        ensure_frame_feature_arrays(current_keyframe)
        ensure_frame_feature_arrays(candidate)

        current_idxs = np.asarray(
            [
                idx
                for idx, point in enumerate(current_keyframe.points)
                if point is not None and not point.is_bad()
            ],
            dtype=np.int32,
        )
        candidate_idxs = np.asarray(
            [
                idx
                for idx, point in enumerate(candidate.points)
                if point is not None and not point.is_bad()
            ],
            dtype=np.int32,
        )

        if len(current_idxs) == 0 or len(candidate_idxs) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        if self.keyframe_database is not None and getattr(self.keyframe_database, "available", False):
            try:
                self.keyframe_database.compute_bow(current_keyframe)
                self.keyframe_database.compute_bow(candidate)
                bow_result = BoWGuidedMatcher(self.keyframe_database.voc).match(
                    current_keyframe,
                    candidate,
                    valid_idxs1=current_idxs,
                    valid_idxs2=candidate_idxs,
                    max_descriptor_distance=Parameters.kMaxDescriptorDistance,
                    ratio_test=Parameters.kLoopClosingFeatureMatchRatioTest,
                    orientation_check=True,
                )
                self.last_match_diagnostics = bow_result.diagnostics
                self.last_match_distances = np.asarray(bow_result.distances, dtype=np.float32)
                self.last_bow_guided_matching_available = bow_result.available
                if bow_result.available:
                    return bow_result.idxs1, bow_result.idxs2
            except Exception as exc:
                self.last_error = f"BoW-guided loop matching unavailable: {exc}"

        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    @staticmethod
    def _base_candidate_report(current_keyframe: KeyFrame, candidate: KeyFrame) -> dict:
        return {
            "current_kf_id": int(getattr(current_keyframe, "kid", getattr(current_keyframe, "id", -1))),
            "candidate_kf_id": int(getattr(candidate, "kid", getattr(candidate, "id", -1))),
            "current_frame_id": int(getattr(current_keyframe, "id", -1)),
            "candidate_frame_id": int(getattr(candidate, "id", -1)),
            "geometry_method": "rgbd_se3_ransac",
            "common_words": 0,
            "bow_matches_raw": 0,
            "bow_matches_after_ratio": 0,
            "bow_matches_after_orientation": 0,
            "bow_matches_with_valid_mappoints": 0,
            "bow_match_pairs": 0,
            "descriptor_distances": 0,
            "geometry_input_correspondences": 0,
            "geometry_ransac_inliers": 0,
            "geometry_refined_inliers": 0,
            "geometry_reprojection_rmse": None,
            "estimated_pose_distance": None,
            "estimated_pose_distance_threshold": float(
                getattr(Parameters, "kLoopClosingMaxEstimatedPoseDistanceForGuidedSE3", 0.0) or 0.0
            ),
            "estimated_pose_rotation_deg": None,
            "estimated_pose_rotation_threshold_deg": float(
                getattr(Parameters, "kLoopClosingMaxEstimatedPoseRotationDegForGuidedSE3", 0.0) or 0.0
            ),
            "guided_projection_matches": 0,
            "guided_projection_total_matches": 0,
            "seed_inliers": 0,
            "passed_pose_distance_gate": True,
            "candidate_covisible_points": 0,
            "final_inliers": 0,
            "accept_threshold_inliers": int(Parameters.kLoopClosingMinNumMatchedMapPoints),
            "accepted": False,
            "rejection_stage": "",
            "rejection_reason": "",
        }

    def _merge_match_diagnostics(self, report: dict) -> None:
        diagnostics = self.last_match_diagnostics
        if diagnostics is None:
            return
        report["common_words"] = int(getattr(diagnostics, "shared_words", 0) or 0)
        report["bow_matches_raw"] = int(getattr(diagnostics, "raw_matches", 0) or 0)
        report["bow_matches_after_ratio"] = int(
            getattr(diagnostics, "matches_after_ratio", getattr(diagnostics, "raw_matches", 0)) or 0
        )
        report["bow_matches_after_orientation"] = int(
            getattr(
                diagnostics,
                "matches_after_orientation",
                report.get("bow_matches_after_orientation", 0),
            )
            or 0
        )
        report["threshold_rejects"] = int(getattr(diagnostics, "threshold_rejects", 0) or 0)
        report["ratio_rejects"] = int(getattr(diagnostics, "ratio_rejects", 0) or 0)
        report["duplicate_train_rejects"] = int(getattr(diagnostics, "duplicate_train_rejects", 0) or 0)
        report["orientation_rejects"] = int(getattr(diagnostics, "orientation_rejects", 0) or 0)

    @staticmethod
    def _guided_projection_refinement(
        current_keyframe: KeyFrame,
        candidate: KeyFrame,
        estimate: Sim3Estimate,
        matches: list,
        match_idxs: np.ndarray,
    ) -> int:
        ensure_frame_feature_arrays(current_keyframe)
        ensure_frame_feature_arrays(candidate)

        if len(current_keyframe.points) == 0 or len(candidate.points) == 0:
            return 0

        Tcw_candidate = np.asarray(candidate.Tcw(), dtype=np.float64).reshape(4, 4)
        Rcw = Tcw_candidate[:3, :3]
        tcw = Tcw_candidate[:3, 3]
        taken_candidate_idxs = {int(idx) for idx in match_idxs.tolist() if int(idx) >= 0}
        max_descriptor_distance = float(Parameters.kMaxDescriptorDistance or 100)
        search_radius = float(Parameters.kLoopClosingMaxReprojectionDistanceMapSearch)
        added = 0

        candidate_uvs = np.asarray(
            [kp.pt if hasattr(kp, "pt") else kp for kp in candidate.kpsu],
            dtype=np.float64,
        ).reshape(-1, 2)

        for idx_cur, point_current in enumerate(current_keyframe.points):
            if idx_cur < len(matches) and matches[idx_cur] is not None:
                continue
            if point_current is None or point_current.is_bad():
                continue
            position = point_current.get_position()
            if not np.all(np.isfinite(position)):
                continue

            aligned_position = estimate.R @ position + estimate.t
            point_c = Rcw @ aligned_position + tcw
            if not np.all(np.isfinite(point_c)) or point_c[2] <= Parameters.kMinDepth:
                continue
            uv, depth = candidate.camera.project(point_c.reshape(1, 3))
            uv = np.asarray(uv, dtype=np.float64).reshape(1, 2)
            depth = np.asarray(depth, dtype=np.float64).reshape(1)
            if not bool(candidate.are_in_image(uv, depth)[0]):
                continue

            pixel_dists = np.linalg.norm(candidate_uvs - uv.reshape(1, 2), axis=1)
            nearby = np.flatnonzero(pixel_dists <= search_radius)
            if len(nearby) == 0:
                continue

            best_idx = -1
            best_distance = float("inf")
            for idx_cand in nearby:
                idx_cand = int(idx_cand)
                if idx_cand in taken_candidate_idxs:
                    continue
                if idx_cand < 0 or idx_cand >= len(candidate.points):
                    continue
                point_loop = candidate.points[idx_cand]
                if point_loop is None or point_loop.is_bad():
                    continue
                descriptor_distance = point_current.min_des_distance(candidate.des[idx_cand])
                if descriptor_distance < best_distance:
                    best_distance = float(descriptor_distance)
                    best_idx = idx_cand

            if best_idx >= 0 and best_distance <= max_descriptor_distance:
                matches[int(idx_cur)] = candidate.points[best_idx]
                match_idxs[int(idx_cur)] = int(best_idx)
                taken_candidate_idxs.add(int(best_idx))
                added += 1

        return added

    @staticmethod
    def _estimated_pose_delta(current_keyframe: KeyFrame, candidate: KeyFrame) -> tuple[Optional[float], Optional[float]]:
        try:
            Twc_current = np.asarray(current_keyframe.Twc(), dtype=np.float64).reshape(4, 4)
            Twc_candidate = np.asarray(candidate.Twc(), dtype=np.float64).reshape(4, 4)
        except Exception:
            return None, None
        position_current = Twc_current[:3, 3]
        position_candidate = Twc_candidate[:3, 3]
        distance = float(np.linalg.norm(position_current - position_candidate))
        if not np.isfinite(distance):
            distance = None
        rotation_deg = _rotation_angle_deg(Twc_current[:3, :3], Twc_candidate[:3, :3])
        return distance, rotation_deg

    @staticmethod
    def _matched_3d_points_from_matches(current_keyframe, candidate, matches, match_idxs):
        idxs_current = []
        idxs_candidate = []
        for idx_cur, point_loop in enumerate(matches):
            if point_loop is None:
                continue
            idx_cand = int(match_idxs[idx_cur]) if idx_cur < len(match_idxs) else -1
            if idx_cand < 0:
                continue
            idxs_current.append(int(idx_cur))
            idxs_candidate.append(idx_cand)
        return LoopGeometryChecker._matched_3d_points(
            current_keyframe,
            candidate,
            np.asarray(idxs_current, dtype=np.int32),
            np.asarray(idxs_candidate, dtype=np.int32),
        )

    @staticmethod
    def _matched_3d_points(current_keyframe, candidate, idxs_current, idxs_candidate):
        points_current = []
        points_loop = []
        valid_current = []
        valid_candidate = []

        for idx_cur, idx_cand in zip(idxs_current, idxs_candidate):
            if idx_cur < 0 or idx_cur >= len(current_keyframe.points):
                continue
            if idx_cand < 0 or idx_cand >= len(candidate.points):
                continue
            point_current = current_keyframe.points[int(idx_cur)]
            point_loop = candidate.points[int(idx_cand)]
            if point_current is None or point_loop is None:
                continue
            if point_current.is_bad() or point_loop.is_bad():
                continue
            p1 = point_current.get_position()
            p2 = point_loop.get_position()
            if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
                continue
            points_current.append(p1)
            points_loop.append(p2)
            valid_current.append(int(idx_cur))
            valid_candidate.append(int(idx_cand))

        return (
            np.asarray(points_current, dtype=np.float64).reshape(-1, 3),
            np.asarray(points_loop, dtype=np.float64).reshape(-1, 3),
            np.asarray(valid_current, dtype=np.int32),
            np.asarray(valid_candidate, dtype=np.int32),
        )


# Apply loop corrections to poses, points, and post-loop optimization.
class LoopCorrector:
    def __init__(self, slam, geometry_checker: LoopGeometryChecker):
        self.slam = slam
        self.loop_geometry_checker = geometry_checker
        self.mean_graph_chi2_error = None
        self.last_result: Optional[EssentialGraphResult] = None
        self.last_num_fused_points = 0
        self.last_num_corrected_points = 0
        self.last_fusion_diagnostics = ProjectionFuseDiagnostics()
        self.last_global_ba_result = GlobalBAResult(started=False, reason="disabled")

    @property
    def map(self):
        return getattr(self.slam, "map", None)

    def _profile_section(self, name: str):
        profiler = getattr(self.slam, "runtime_profiler", None)
        if profiler is None:
            return nullcontext()
        return profiler.section(name)

    def correct_loop(self, current_keyframe: KeyFrame) -> EssentialGraphResult:
        loop_keyframe = self.loop_geometry_checker.success_loop_kf
        estimate = self.loop_geometry_checker.success_sim3
        sim3_pose = getattr(self.loop_geometry_checker, "success_loop_kf_sim3_pose", None)

        has_sim3 = sim3_pose is not None
        has_estimate = estimate is not None and estimate.success
        if loop_keyframe is None or (not has_sim3 and not has_estimate):
            result = EssentialGraphResult(False, float("inf"), float("inf"), 0, "missing loop geometry")
            self.last_result = result
            return result

        if has_sim3:
            # pyslam path: derive correction_T from the Sim3 pose
            # corrected_Tcw = Sc1w.to_se3_matrix() = [R | t/s]  (scale≈1 for RGB-D)
            # correction_T = inv(corrected_Tcw) @ old_Tcw  (Twc_corrected @ Tc1w_old)
            corrected_Tcw = np.asarray(sim3_pose.to_se3_matrix(), dtype=np.float64).reshape(4, 4)
            old_Tcw = np.asarray(current_keyframe.Tcw(), dtype=np.float64).reshape(4, 4)
            correction_T = np.linalg.inv(corrected_Tcw) @ old_Tcw
        else:
            correction_T = estimate.T

        if not np.all(np.isfinite(correction_T)):
            result = EssentialGraphResult(False, float("inf"), float("inf"), 0, "non-finite correction")
            self.last_result = result
            return result

        current_group = current_keyframe.get_connected_keyframes()
        current_group.append(current_keyframe)
        corrected_poses = self._make_corrected_pose_map(current_group, correction_T)

        lock = self.map.update_lock if self.map is not None else _NullLock()
        with lock:
            self.last_fusion_diagnostics = ProjectionFuseDiagnostics()
            self.last_num_corrected_points = 0
            direct_fused = self._fuse_loop_matches(current_keyframe)
            projection_fused = self.search_and_fuse_corrected_keyframes(
                current_keyframe,
                loop_keyframe,
                corrected_poses,
            )
            self.last_num_fused_points = direct_fused + projection_fused

            loop_connections = self._build_loop_connections_after_fusion(current_group)
            if loop_keyframe not in loop_connections.get(current_keyframe, []):
                loop_connections.setdefault(current_keyframe, []).append(loop_keyframe)

            with self._profile_section("essential_graph.optimize"):
                result = optimize_essential_graph_se3(
                    current_group,
                    loop_keyframe,
                    current_keyframe,
                    correction_T,
                    map_object=self.map,
                    corrected_poses=corrected_poses,
                    loop_connections=loop_connections,
                )

            if result.success:
                self.last_num_corrected_points = getattr(result, "corrected_points", 0)
                loop_keyframe.add_loop_edge(current_keyframe)
                current_keyframe.add_loop_edge(loop_keyframe)
                for keyframe in current_group:
                    keyframe.update_connections()
                loop_keyframe.update_connections()
                self.last_global_ba_result = self._run_global_ba_after_loop(loop_keyframe)
                self._reset_tracking_after_loop(current_keyframe)
            else:
                self.last_global_ba_result = GlobalBAResult(started=False, reason="loop correction failed")

        self.last_result = result
        self.mean_graph_chi2_error = result.after_error
        return result

    def _reset_tracking_after_loop(self, current_keyframe: KeyFrame) -> None:
        tracking = getattr(self.slam, "tracking", None)
        if tracking is None:
            return
        motion_model = getattr(tracking, "motion_model", None)
        if motion_model is not None and hasattr(motion_model, "reset"):
            motion_model.reset()
        tracking.kf_ref = current_keyframe
        tracking.kf_last = current_keyframe
        try:
            tracking.map.update_local_map(current_keyframe)
        except Exception:
            pass

    def _run_global_ba_after_loop(self, loop_keyframe: KeyFrame) -> GlobalBAResult:
        enabled = bool(
            getattr(self.slam, "enable_global_ba", False)
            and getattr(self.slam, "global_ba_after_loop", False)
        )
        if not enabled:
            return GlobalBAResult(started=False, reason="disabled")

        adjuster = GlobalBundleAdjuster(
            self.map,
            rounds=int(getattr(self.slam, "global_ba_iterations", Parameters.kGlobalBAIterations)),
            use_robust_kernel=Parameters.kGBAUseRobustKernel,
            min_inlier_edges=Parameters.kGlobalBAMinInlierEdges,
        )
        with self._profile_section("global_ba.run"):
            return adjuster.run(loop_kf_id=int(getattr(loop_keyframe, "kid", 0)))

    def _fuse_loop_matches(self, current_keyframe: KeyFrame) -> int:
        fused = 0
        for idx, loop_point in enumerate(self.loop_geometry_checker.success_map_point_matches):
            if loop_point is None or loop_point.is_bad():
                if loop_point is not None:
                    self.last_fusion_diagnostics.rejected_bad_point += 1
                continue
            current_point = current_keyframe.get_point_match(idx)
            if current_point is None:
                if loop_point.add_observation(current_keyframe, idx):
                    loop_point.update_info()
                    self.last_fusion_diagnostics.added_observations += 1
                    self.last_fusion_diagnostics.fused_points += 1
                    fused += 1
            elif current_point is not loop_point:
                current_point.replace_with(loop_point)
                self.last_fusion_diagnostics.replaced_points += 1
                self.last_fusion_diagnostics.fused_points += 1
                fused += 1
            else:
                self.last_fusion_diagnostics.rejected_duplicate += 1
        return fused

    def search_and_fuse_corrected_keyframes(
        self,
        current_keyframe: KeyFrame,
        loop_keyframe: KeyFrame,
        corrected_poses: dict[KeyFrame, np.ndarray],
    ) -> int:
        loop_map_points = self._collect_loop_map_points(loop_keyframe)
        corrected_keyframes = self._collect_current_corrected_keyframes(current_keyframe)

        fused = 0
        added_before = self.last_fusion_diagnostics.added_observations
        affected_points = set()
        affected_keyframes = set(corrected_keyframes)

        for keyframe in corrected_keyframes:
            corrected_pose = corrected_poses.get(keyframe, keyframe.Tcw())
            replace_points = [None] * len(loop_map_points)
            local_diag = ProjectionFuseDiagnostics()
            ProjectionMatcher.search_and_fuse_for_loop_correction(
                keyframe,
                corrected_pose,
                loop_map_points,
                replace_points,
                diagnostics=local_diag,
            )
            self.last_fusion_diagnostics.merge(local_diag)

            for idx, replacement_source in enumerate(replace_points):
                if replacement_source is None:
                    continue
                loop_point = loop_map_points[idx]
                if loop_point is None or loop_point.is_bad():
                    self.last_fusion_diagnostics.rejected_bad_point += 1
                    continue
                if replacement_source is loop_point:
                    self.last_fusion_diagnostics.rejected_duplicate += 1
                    continue
                if replacement_source.is_bad():
                    self.last_fusion_diagnostics.rejected_bad_point += 1
                    continue
                replacement_source.replace_with(loop_point)
                affected_points.add(loop_point)
                affected_keyframes.update(loop_point.keyframes())
                fused += 1

        for point in loop_map_points:
            if point is not None and not point.is_bad():
                affected_points.add(point)
        for point in affected_points:
            point.update_info()
        for keyframe in affected_keyframes:
            if keyframe is not None and not keyframe.is_bad():
                keyframe.update_connections()

        added_here = self.last_fusion_diagnostics.added_observations - added_before
        return fused + added_here

    def _collect_loop_map_points(self, loop_keyframe: KeyFrame) -> list:
        keyframes = []
        for keyframe in [loop_keyframe] + loop_keyframe.get_best_covisible_keyframes(
            Parameters.kNumBestCovisibilityKeyFrames
        ):
            if keyframe is not None and not keyframe.is_bad() and keyframe not in keyframes:
                keyframes.append(keyframe)

        points = []
        seen = set()
        for keyframe in keyframes:
            for point in keyframe.get_matched_good_points():
                if point is None or point.is_bad() or point in seen:
                    continue
                points.append(point)
                seen.add(point)
        return points

    def _collect_current_corrected_keyframes(self, current_keyframe: KeyFrame) -> list[KeyFrame]:
        keyframes = []
        for keyframe in [current_keyframe] + current_keyframe.get_best_covisible_keyframes(
            Parameters.kNumBestCovisibilityKeyFrames
        ):
            if keyframe is not None and not keyframe.is_bad() and keyframe not in keyframes:
                keyframes.append(keyframe)
        return keyframes

    @staticmethod
    def _make_corrected_pose_map(keyframes: list[KeyFrame], correction_T: np.ndarray) -> dict[KeyFrame, np.ndarray]:
        correction_T = np.asarray(correction_T, dtype=np.float64).reshape(4, 4)
        correction_inv = np.linalg.inv(correction_T)
        corrected = {}
        for keyframe in keyframes:
            Tcw = np.asarray(keyframe.Tcw(), dtype=np.float64).reshape(4, 4)
            corrected[keyframe] = Tcw @ correction_inv
        return corrected

    @staticmethod
    def _build_loop_connections_after_fusion(current_group: list[KeyFrame]) -> dict[KeyFrame, list[KeyFrame]]:
        previous_neighbors = {keyframe: set(keyframe.get_covisible_keyframes()) for keyframe in current_group}
        current_set = set(current_group)
        loop_connections = {}

        for keyframe in current_group:
            keyframe.update_connections()
            new_connections = set(keyframe.get_connected_keyframes())
            new_connections.difference_update(previous_neighbors.get(keyframe, set()))
            new_connections.difference_update(current_set)
            loop_connections[keyframe] = [
                connected for connected in new_connections if connected is not None and not connected.is_bad()
            ]
        return loop_connections


# Manage the loop-closing queue and execute one loop event at a time.
class LoopClosing:
    def __init__(self, slam, keyframe_database=None, consistency_threshold: int = 3):
        self.slam = slam
        self.keyframe_database = keyframe_database or getattr(slam, "keyframe_database", None)
        self.loop_detector = LoopDetector(self.keyframe_database)
        self.loop_consistency_checker = LoopGroupConsistencyChecker(consistency_threshold)
        self.loop_geometry_checker = LoopGeometryChecker(keyframe_database=self.keyframe_database)
        self.loop_corrector = LoopCorrector(slam, self.loop_geometry_checker)
        self.queue = deque()
        self._queue_lock = threading.Lock()
        self.last_loop_kf_id = 0
        self.last_diagnostics = LoopDiagnostics()
        self.mean_graph_chi2_error = None
        self._is_correcting = False
        self._is_correcting_lock = threading.Lock()
        self.loop_oracle: TumLoopOracle | None = None
        self.loop_retrieval_trace_enabled = False
        self.loop_retrieval_trace_raw_k = 0

    def is_correcting(self) -> bool:
        with self._is_correcting_lock:
            return bool(self._is_correcting)

    def _profile_section(self, name: str):
        profiler = getattr(self.slam, "runtime_profiler", None)
        if profiler is None:
            return nullcontext()
        return profiler.section(name)

    def insert_keyframe(self, keyframe: KeyFrame) -> None:
        self.add_keyframe(keyframe)

    def set_loop_oracle(self, oracle: TumLoopOracle | None) -> None:
        self.loop_oracle = oracle

    def set_retrieval_trace_config(self, *, enabled: bool, raw_k: int = 0) -> None:
        self.loop_retrieval_trace_enabled = bool(enabled)
        self.loop_retrieval_trace_raw_k = max(0, int(raw_k))
        if self.keyframe_database is not None and hasattr(self.keyframe_database, "configure_loop_retrieval_trace"):
            self.keyframe_database.configure_loop_retrieval_trace(
                enabled=self.loop_retrieval_trace_enabled,
                raw_k=self.loop_retrieval_trace_raw_k,
            )

    def add_keyframe(self, keyframe: KeyFrame, img=None) -> None:
        if img is not None:
            keyframe.img = img
        with self._queue_lock:
            self.queue.append(keyframe)

    def queue_size(self) -> int:
        with self._queue_lock:
            return len(self.queue)

    def pop_keyframe(self):
        with self._queue_lock:
            if not self.queue:
                return None
            return self.queue.popleft()

    def step(self) -> bool:
        keyframe = self.pop_keyframe()
        if keyframe is None:
            return False
        return self.process_keyframe(keyframe)

    def process_keyframe(self, keyframe: KeyFrame) -> bool:
        diagnostics = LoopDiagnostics()
        self.last_diagnostics = diagnostics
        self.loop_geometry_checker.runtime_profiler = getattr(self.slam, "runtime_profiler", None)

        # Cooldown: discard keyframes that arrive within kMinDeltaFrameForMeaningfulLoopClosure
        # KF-ids of the last accepted loop.  Mirrors pyslam loop_closing.py:1009-1012.
        if keyframe.kid < self.last_loop_kf_id + Parameters.kMinDeltaFrameForMeaningfulLoopClosure:
            keyframe.set_erase()
            if self.keyframe_database is not None:
                self.keyframe_database.add(keyframe)
            return False

        if self.keyframe_database is None or not self.keyframe_database.available:
            diagnostics.unavailable_reason = (
                "keyframe database is not configured"
                if self.keyframe_database is None
                else self.keyframe_database.unavailable_reason()
            )
            diagnostics.rejected_by_bow = 1
            return False

        try:
            with self._profile_section("loop.detect_candidates"):
                detection_output: LoopDetectorOutput = self.loop_detector.detect(keyframe)
            diagnostics.candidates = len(detection_output.candidate_keyframes)
            diagnostics.loop_debug_records = self._build_loop_debug_records(keyframe, detection_output)
            self._populate_retrieval_diagnostics(diagnostics, keyframe, detection_output)

            if len(detection_output.candidate_keyframes) == 0:
                diagnostics.rejected_by_bow = 1
                self.loop_consistency_checker.clear_consistency_groups()
                self._finalize_candidate_diagnostics(
                    diagnostics,
                    current_keyframe=keyframe,
                    accepted_candidate_id=-1,
                )
                return False

            got_consistent = self.loop_consistency_checker.check_candidates(
                keyframe,
                detection_output.candidate_keyframes,
            )
            self._merge_consistency_debug(diagnostics.loop_debug_records)
            consistent_candidates = [
                candidate
                for candidate in self.loop_consistency_checker.enough_consistent_candidates
                if not candidate.is_bad()
            ]
            self._update_retrieval_profile_from_records(
                diagnostics.loop_retrieval_profile_rows,
                diagnostics.loop_debug_records,
                accepted_candidate_id=-1,
                num_candidates_after_consistency=len(consistent_candidates) if got_consistent else 0,
            )

            if not got_consistent:
                diagnostics.rejected_by_consistency = diagnostics.candidates
                for record in diagnostics.loop_debug_records:
                    record["rejection_stage"] = "consistency"
                    record["rejection_reason"] = "rejected_by_consistency"
                self._finalize_candidate_diagnostics(
                    diagnostics,
                    current_keyframe=keyframe,
                    accepted_candidate_id=-1,
                )
                return False

            with self._profile_section("loop.compute_geometry"):
                got_geometry = self.loop_geometry_checker.check_candidates(keyframe, consistent_candidates)
            self._merge_geometry_debug(diagnostics.loop_debug_records)
            diagnostics.candidate_pair_reports = list(self.loop_geometry_checker.last_candidate_reports.values())

            for record in diagnostics.loop_debug_records:
                if not bool(record.get("passed_consistency")) and not record.get("rejection_stage"):
                    record["rejection_stage"] = "consistency"
                    record["rejection_reason"] = "rejected_by_consistency"

            if not got_geometry:
                diagnostics.rejected_by_geometry = len(consistent_candidates)
                for record in diagnostics.loop_debug_records:
                    if bool(record.get("passed_consistency")) and not record.get("rejection_stage"):
                        record["rejection_stage"] = "geometry"
                        record["rejection_reason"] = record.get("rejection_reason") or "rejected_by_geometry"
                self._finalize_candidate_diagnostics(
                    diagnostics,
                    current_keyframe=keyframe,
                    accepted_candidate_id=-1,
                )
                return False

            with self._is_correcting_lock:
                self._is_correcting = True
            try:
                with self._profile_section("loop.correct_loop"):
                    result = self.loop_corrector.correct_loop(keyframe)
            finally:
                with self._is_correcting_lock:
                    self._is_correcting = False
            diagnostics.optimization_result = result
            diagnostics.corrected_keyframes = result.corrected_keyframes
            diagnostics.corrected_points = self.loop_corrector.last_num_corrected_points
            diagnostics.fused_points = self.loop_corrector.last_num_fused_points
            diagnostics.fusion_diagnostics = self.loop_corrector.last_fusion_diagnostics
            diagnostics.global_ba_result = self.loop_corrector.last_global_ba_result
            self._copy_global_ba_diagnostics(diagnostics, diagnostics.global_ba_result)

            if result.success:
                diagnostics.accepted = 1
                accepted_candidate_id = int(
                    getattr(self.loop_geometry_checker.success_loop_kf, "kid", getattr(self.loop_geometry_checker.success_loop_kf, "id", -1))
                )
                for record in diagnostics.loop_debug_records:
                    if int(record.get("candidate_kf_id", -1)) == accepted_candidate_id:
                        record["rejection_stage"] = ""
                        record["rejection_reason"] = ""
                        record["accepted"] = True
                self._finalize_candidate_diagnostics(
                    diagnostics,
                    current_keyframe=keyframe,
                    accepted_candidate_id=accepted_candidate_id,
                )
                self.last_loop_kf_id = keyframe.kid
                self.mean_graph_chi2_error = result.after_error
                return True

            diagnostics.rejected_by_geometry = len(consistent_candidates)
            self._finalize_candidate_diagnostics(
                diagnostics,
                current_keyframe=keyframe,
                accepted_candidate_id=-1,
            )
            return False
        finally:
            self.keyframe_database.add(keyframe)

    def _build_loop_debug_records(self, keyframe: KeyFrame, detection_output: LoopDetectorOutput) -> list[dict]:
        records = []
        scores = list(getattr(detection_output, "candidate_scores", []) or [])
        detail_by_candidate = dict(getattr(detection_output, "candidate_details", {}) or {})
        for rank, candidate in enumerate(detection_output.candidate_keyframes, start=1):
            if candidate is None:
                continue
            score = scores[rank - 1] if rank - 1 < len(scores) else None
            candidate_kid = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            current_kid = int(getattr(keyframe, "kid", getattr(keyframe, "id", -1)))
            detail = dict(detail_by_candidate.get(candidate_kid, {}) or {})
            records.append(
                {
                    "frame_id": int(getattr(keyframe, "id", current_kid)),
                    "current_kf_id": current_kid,
                    "candidate_kf_id": candidate_kid,
                    "current_timestamp": getattr(keyframe, "timestamp", None),
                    "candidate_timestamp": getattr(candidate, "timestamp", None),
                    "candidate_score": score,
                    "candidate_rank": int(detail.get("candidate_rank", rank)),
                    "candidate_source": detail.get("candidate_source", getattr(detection_output, "candidate_source", "keyframe_database")),
                    "temporal_separation_kf": abs(current_kid - candidate_kid),
                    "temporal_separation_frames": abs(int(getattr(keyframe, "id", current_kid)) - int(getattr(candidate, "id", candidate_kid))),
                    "current_group_kf_ids": [],
                    "candidate_group_kf_ids": [],
                    "previous_consistency_group_ids": [],
                    "consistency_overlap_count": 0,
                    "consistency_count": 0,
                    "consistency_required": self.loop_consistency_checker.consistency_threshold,
                    "passed_consistency": False,
                    "min_score": float(getattr(detection_output, "min_score", 0.0) or 0.0),
                    "common_words": int(detail.get("common_words", 0) or 0),
                    "max_common_words": int(detail.get("max_common_words", 0) or 0),
                    "common_word_ratio": float(detail.get("common_word_ratio", 0.0) or 0.0),
                    "raw_dbow_score": detail.get("raw_dbow_score"),
                    "bow_score_raw": detail.get("bow_score_raw", score),
                    "bow_score_normalized": detail.get("bow_score_normalized", score),
                    "accumulated_score": float(detail.get("accumulated_score", score or 0.0) or 0.0),
                    "best_accumulated_score": float(detail.get("best_accumulated_score", score or 0.0) or 0.0),
                    "is_connected": bool(detail.get("is_connected", False)),
                    "temporal_gap": int(detail.get("temporal_gap", abs(int(getattr(keyframe, "id", current_kid)) - int(getattr(candidate, "id", candidate_kid)))) or 0),
                    "consistency_group_id": -1,
                    "bow_matches_raw": 0,
                    "bow_matches_after_ratio": 0,
                    "bow_matches_after_orientation": 0,
                    "bow_matches_with_valid_mappoints": 0,
                    "geometry_method": "rgbd_se3_ransac",
                    "geometry_input_correspondences": 0,
                    "geometry_ransac_inliers": 0,
                    "geometry_refined_inliers": 0,
                    "geometry_reprojection_rmse": None,
                    "estimated_pose_distance": None,
                    "estimated_pose_distance_threshold": float(
                        getattr(Parameters, "kLoopClosingMaxEstimatedPoseDistanceForGuidedSE3", 0.0) or 0.0
                    ),
                    "estimated_pose_rotation_deg": None,
                    "estimated_pose_rotation_threshold_deg": float(
                        getattr(Parameters, "kLoopClosingMaxEstimatedPoseRotationDegForGuidedSE3", 0.0) or 0.0
                    ),
                    "guided_projection_matches": 0,
                    "guided_projection_total_matches": 0,
                    "final_inliers": 0,
                    "accept_threshold_inliers": int(Parameters.kLoopClosingMinNumMatchedMapPoints),
                    "accepted": False,
                    "rejection_stage": "",
                    "rejection_reason": "",
                }
            )
        return records

    def _merge_consistency_debug(self, records: list[dict]) -> None:
        for record in records:
            candidate_kid = int(record.get("candidate_kf_id", -1))
            info = self.loop_consistency_checker.last_candidate_debug.get(candidate_kid)
            if info:
                record.update(info)

    def _merge_geometry_debug(self, records: list[dict]) -> None:
        for record in records:
            candidate_kid = int(record.get("candidate_kf_id", -1))
            info = self.loop_geometry_checker.last_candidate_reports.get(candidate_kid)
            if info:
                record.update({k: v for k, v in info.items() if k not in {"current_kf_id", "candidate_kf_id"}})

    def _populate_retrieval_diagnostics(
        self,
        diagnostics: LoopDiagnostics,
        keyframe: KeyFrame,
        detection_output: LoopDetectorOutput,
    ) -> None:
        profile = dict(getattr(detection_output, "retrieval_profile", {}) or {})
        if profile:
            profile["kf_id"] = int(getattr(keyframe, "kid", getattr(keyframe, "id", -1)))
            profile["timestamp"] = getattr(keyframe, "timestamp", None)
            profile.setdefault("num_candidates_after_consistency", 0)
            profile.setdefault("top_candidate_consistency", -1)
            profile.setdefault("accepted_candidate_id", -1)
            diagnostics.loop_retrieval_profile_rows = [profile]

        source_comparison = dict(getattr(detection_output, "source_comparison", {}) or {})
        if source_comparison:
            diagnostics.loop_candidate_source_comparison_rows = [source_comparison]
        trace_rows = dict(getattr(detection_output, "trace_rows", {}) or {})
        diagnostics.loop_raw_dbow_trace_rows = list(trace_rows.get("raw_dbow", []) or [])
        diagnostics.loop_inverted_word_trace_rows = list(trace_rows.get("inverted_word", []) or [])
        diagnostics.loop_score_filter_trace_rows = list(trace_rows.get("score_filter", []) or [])
        diagnostics.loop_accumulation_trace_rows = list(trace_rows.get("accumulation", []) or [])

    def _finalize_candidate_diagnostics(
        self,
        diagnostics: LoopDiagnostics,
        *,
        current_keyframe: KeyFrame,
        accepted_candidate_id: int,
    ) -> None:
        self._merge_oracle_debug(diagnostics.loop_debug_records)
        self._update_retrieval_profile_from_records(
            diagnostics.loop_retrieval_profile_rows,
            diagnostics.loop_debug_records,
            accepted_candidate_id=accepted_candidate_id,
            num_candidates_after_consistency=sum(
                1 for record in diagnostics.loop_debug_records if bool(record.get("passed_consistency"))
            ),
        )
        diagnostics.loop_candidate_oracle_rows = self._build_loop_candidate_oracle_rows(
            diagnostics.loop_debug_records
        )
        diagnostics.loop_keyframe_density_rows = self._build_loop_keyframe_density_rows(
            diagnostics.loop_debug_records
        )
        diagnostics.loop_retained_candidate_trace_rows = self._build_loop_retained_candidate_trace_rows(
            diagnostics.loop_debug_records
        )
        diagnostics.loop_consistency_progression_rows = self._build_loop_consistency_progression_rows(
            diagnostics.loop_debug_records
        )
        diagnostics.loop_geometry_trace_rows = self._build_loop_geometry_trace_rows(
            diagnostics.loop_debug_records
        )
        diagnostics.loop_gt_positive_trace_rows = self._build_loop_gt_positive_trace_rows(
            current_keyframe,
            diagnostics,
        )

    def _merge_oracle_debug(self, records: list[dict]) -> None:
        oracle = self.loop_oracle
        if oracle is None or not oracle.has_data():
            return
        for record in records:
            current_timestamp = record.get("current_timestamp")
            candidate_timestamp = record.get("candidate_timestamp")
            if current_timestamp is None or candidate_timestamp is None:
                continue
            diag = oracle.describe_pair(float(current_timestamp), float(candidate_timestamp))
            record["gt_available"] = bool(diag.gt_available)
            record["gt_translation_distance"] = diag.gt_translation_distance
            record["gt_rotation_angle_deg"] = diag.gt_rotation_angle_deg
            record["gt_loop_like"] = bool(diag.gt_loop_like)
            record["gt_near_loop"] = bool(diag.gt_near_loop)

    @staticmethod
    def _update_retrieval_profile_from_records(
        profile_rows: list[dict],
        records: list[dict],
        *,
        accepted_candidate_id: int,
        num_candidates_after_consistency: int,
    ) -> None:
        if not profile_rows:
            return
        row = profile_rows[0]
        row["num_candidates_after_consistency"] = int(num_candidates_after_consistency)
        row["accepted_candidate_id"] = int(accepted_candidate_id)
        top_candidate_id = int(row.get("top_candidate_id", -1) or -1)
        top_record = None
        for record in records:
            if int(record.get("candidate_kf_id", -1)) == top_candidate_id:
                top_record = record
                break
        if top_record is not None:
            row["top_candidate_consistency"] = int(top_record.get("consistency_count", -1) or -1)

    def _build_loop_candidate_oracle_rows(self, records: list[dict]) -> list[dict]:
        rows: list[dict] = []
        for record in records:
            current_timestamp = record.get("current_timestamp")
            candidate_timestamp = record.get("candidate_timestamp")
            time_gap_sec = None
            if current_timestamp is not None and candidate_timestamp is not None:
                try:
                    time_gap_sec = abs(float(current_timestamp) - float(candidate_timestamp))
                except (TypeError, ValueError):
                    time_gap_sec = None
            rows.append(
                {
                    "current_kf_id": int(record.get("current_kf_id", -1) or -1),
                    "candidate_kf_id": int(record.get("candidate_kf_id", -1) or -1),
                    "current_timestamp": current_timestamp,
                    "candidate_timestamp": candidate_timestamp,
                    "time_gap_sec": time_gap_sec,
                    "bow_score": record.get("candidate_score", record.get("bow_score_raw")),
                    "min_score": record.get("min_score"),
                    "common_words": int(record.get("common_words", 0) or 0),
                    "max_common_words": int(record.get("max_common_words", 0) or 0),
                    "common_word_ratio": record.get("common_word_ratio"),
                    "accumulated_score": record.get("accumulated_score"),
                    "best_accumulated_score": record.get("best_accumulated_score"),
                    "consistency_score": int(record.get("consistency_count", 0) or 0),
                    "consistency_group_id": int(record.get("consistency_group_id", -1) or -1),
                    "candidate_source": record.get("candidate_source"),
                    "candidate_rank": int(record.get("candidate_rank", -1) or -1),
                    "is_connected": bool(record.get("is_connected", False)),
                    "temporal_gap": int(record.get("temporal_gap", 0) or 0),
                    "rejection_stage": record.get("rejection_stage", ""),
                    "rejection_reason": record.get("rejection_reason", ""),
                    "raw_bow_matches": int(record.get("bow_matches_raw", 0) or 0),
                    "valid_bow_map_point_matches": int(record.get("bow_matches_with_valid_mappoints", 0) or 0),
                    "seed_correspondences": int(record.get("geometry_input_correspondences", 0) or 0),
                    "seed_inliers": int(record.get("seed_inliers", record.get("geometry_ransac_inliers", 0)) or 0),
                    "refined_inliers": int(record.get("geometry_refined_inliers", 0) or 0),
                    "guided_projection_matches": int(record.get("guided_projection_matches", 0) or 0),
                    "final_matched_map_points": int(
                        record.get(
                            "total_final_matches",
                            record.get("guided_projection_total_matches", record.get("final_inliers", 0)),
                        )
                        or 0
                    ),
                    "estimated_pose_distance": record.get("estimated_pose_distance"),
                    "estimated_rotation_deg": record.get("estimated_pose_rotation_deg"),
                    "gt_available": bool(record.get("gt_available", False)),
                    "gt_translation_distance": record.get("gt_translation_distance"),
                    "gt_rotation_angle_deg": record.get("gt_rotation_angle_deg"),
                    "gt_loop_like": bool(record.get("gt_loop_like", False)),
                    "gt_near_loop": bool(record.get("gt_near_loop", False)),
                    "accepted": bool(record.get("accepted", False)),
                }
            )
        return rows

    def _build_loop_keyframe_density_rows(self, records: list[dict]) -> list[dict]:
        slam_map = getattr(self.slam, "map", None)
        if slam_map is None:
            return []
        rows: list[dict] = []
        for record in records:
            current_kf = getattr(slam_map, "keyframes_map", {}).get(int(record.get("current_kf_id", -1) or -1))
            candidate_kf = getattr(slam_map, "keyframes_map", {}).get(int(record.get("candidate_kf_id", -1) or -1))
            if current_kf is None or candidate_kf is None:
                continue
            current_points = [point for point in current_kf.get_matched_good_points() if point is not None and not point.is_bad()]
            candidate_points = [point for point in candidate_kf.get_matched_good_points() if point is not None and not point.is_bad()]
            shared_points = len(set(current_points).intersection(candidate_points))
            rows.append(
                {
                    "current_kf_id": int(record.get("current_kf_id", -1) or -1),
                    "candidate_kf_id": int(record.get("candidate_kf_id", -1) or -1),
                    "current_neighbor_count": int(len(current_kf.get_connected_keyframes())),
                    "candidate_neighbor_count": int(len(candidate_kf.get_connected_keyframes())),
                    "current_local_map_points": int(len(current_points)),
                    "candidate_local_map_points": int(len(candidate_points)),
                    "shared_bow_words": int(record.get("common_words", 0) or 0),
                    "shared_map_points_if_any": int(shared_points),
                    "candidate_group_size": int(len(candidate_kf.get_connected_keyframes()) + 1),
                    "current_group_size": int(len(current_kf.get_connected_keyframes()) + 1),
                    "final_matched_map_points": int(
                        record.get(
                            "total_final_matches",
                            record.get("guided_projection_total_matches", record.get("final_inliers", 0)),
                        )
                        or 0
                    ),
                    "gt_translation_distance": record.get("gt_translation_distance"),
                    "accepted": bool(record.get("accepted", False)),
                    "rejection_reason": record.get("rejection_reason", ""),
                }
            )
        return rows

    @staticmethod
    def _pair_key(kf_a: int, kf_b: int) -> str:
        return f"{min(int(kf_a), int(kf_b))}-{max(int(kf_a), int(kf_b))}"

    def _build_loop_retained_candidate_trace_rows(self, records: list[dict]) -> list[dict]:
        rows: list[dict] = []
        for record in records:
            rows.append(
                {
                    "current_kf_id": int(record.get("current_kf_id", -1) or -1),
                    "candidate_kf_id": int(record.get("candidate_kf_id", -1) or -1),
                    "pair_key": self._pair_key(
                        int(record.get("current_kf_id", -1) or -1),
                        int(record.get("candidate_kf_id", -1) or -1),
                    ),
                    "current_timestamp": record.get("current_timestamp"),
                    "candidate_timestamp": record.get("candidate_timestamp"),
                    "retained_rank": int(record.get("candidate_rank", -1) or -1),
                    "candidate_source": record.get("candidate_source"),
                    "bow_score": record.get("candidate_score", record.get("bow_score_raw")),
                    "accumulated_score": record.get("accumulated_score"),
                    "best_accumulated_score": record.get("best_accumulated_score"),
                    "consistency_score_before": int(record.get("consistency_score_before", 0) or 0),
                    "consistency_score_after": int(record.get("consistency_score_after", 0) or 0),
                    "passed_consistency": bool(record.get("passed_consistency", False)),
                    "rejection_reason_if_any": record.get("rejection_reason", ""),
                    "accepted": bool(record.get("accepted", False)),
                }
            )
        return rows

    def _build_loop_consistency_progression_rows(self, records: list[dict]) -> list[dict]:
        rows: list[dict] = []
        for record in records:
            rows.append(
                {
                    "current_kf_id": int(record.get("current_kf_id", -1) or -1),
                    "candidate_kf_id": int(record.get("candidate_kf_id", -1) or -1),
                    "pair_key": self._pair_key(
                        int(record.get("current_kf_id", -1) or -1),
                        int(record.get("candidate_kf_id", -1) or -1),
                    ),
                    "candidate_group_ids": json.dumps(record.get("candidate_group_kf_ids", [])),
                    "previous_group_ids": json.dumps(record.get("previous_consistency_group_ids", [])),
                    "overlap_count": int(record.get("consistency_overlap_count", 0) or 0),
                    "previous_consistency": int(record.get("consistency_score_before", 0) or 0),
                    "new_consistency": int(record.get("consistency_score_after", 0) or 0),
                    "threshold": int(record.get("consistency_required", 0) or 0),
                    "passed_consistency": bool(record.get("passed_consistency", False)),
                    "gt_loop_like": bool(record.get("gt_loop_like", False)),
                    "gt_translation_distance": record.get("gt_translation_distance"),
                    "gt_rotation_angle_deg": record.get("gt_rotation_angle_deg"),
                }
            )
        return rows

    def _build_loop_geometry_trace_rows(self, records: list[dict]) -> list[dict]:
        rows: list[dict] = []
        for record in records:
            rows.append(
                {
                    "current_kf_id": int(record.get("current_kf_id", -1) or -1),
                    "candidate_kf_id": int(record.get("candidate_kf_id", -1) or -1),
                    "pair_key": self._pair_key(
                        int(record.get("current_kf_id", -1) or -1),
                        int(record.get("candidate_kf_id", -1) or -1),
                    ),
                    "gt_loop_like": bool(record.get("gt_loop_like", False)),
                    "gt_translation_distance": record.get("gt_translation_distance"),
                    "gt_rotation_angle_deg": record.get("gt_rotation_angle_deg"),
                    "raw_bow_matches": int(record.get("bow_matches_raw", 0) or 0),
                    "valid_bow_map_point_matches": int(record.get("bow_matches_with_valid_mappoints", 0) or 0),
                    "seed_correspondences": int(record.get("seed_correspondences", record.get("geometry_input_correspondences", 0)) or 0),
                    "ransac_inliers": int(record.get("geometry_ransac_inliers", 0) or 0),
                    "seed_inliers": int(record.get("seed_inliers", record.get("geometry_ransac_inliers", 0)) or 0),
                    "seed_inlier_ratio": record.get("seed_inlier_ratio"),
                    "initial_se3_translation_norm": record.get("initial_se3_translation_norm"),
                    "initial_se3_rotation_deg": record.get("initial_se3_rotation_deg"),
                    "pose_distance_gate_threshold": record.get("estimated_pose_distance_threshold"),
                    "pose_rotation_gate_threshold": record.get("estimated_pose_rotation_threshold_deg"),
                    "passed_pose_distance_gate": bool(record.get("passed_pose_distance_gate", True)),
                    "guided_projection_matches": int(record.get("guided_projection_matches", 0) or 0),
                    "refined_correspondences": int(record.get("refined_correspondences", 0) or 0),
                    "refined_inliers": int(record.get("geometry_refined_inliers", 0) or 0),
                    "candidate_group_size": int(record.get("candidate_group_size", 0) or 0),
                    "candidate_group_map_points": int(record.get("candidate_group_map_points", record.get("candidate_covisible_points", 0)) or 0),
                    "visible_projected_group_points": int(record.get("visible_projected_group_points", record.get("projected_visible_points", 0)) or 0),
                    "final_matched_map_points": int(record.get("final_matched_map_points", record.get("total_final_matches", 0)) or 0),
                    "final_gate_threshold": int(record.get("final_gate_threshold", 0) or 0),
                    "accepted": bool(record.get("accepted", False)),
                    "rejection_reason": record.get("rejection_reason", ""),
                }
            )
        return rows

    def _build_loop_gt_positive_trace_rows(
        self,
        current_keyframe: KeyFrame,
        diagnostics: LoopDiagnostics,
    ) -> list[dict]:
        oracle = self.loop_oracle
        if oracle is None or not oracle.has_data():
            return []
        current_timestamp = getattr(current_keyframe, "timestamp", None)
        if current_timestamp is None:
            return []
        slam_map = getattr(self.slam, "map", None)
        if slam_map is None:
            return []
        keyframes_map = getattr(slam_map, "keyframes_map", {}) or {}
        current_kf_id = int(getattr(current_keyframe, "kid", getattr(current_keyframe, "id", -1)))
        raw_by_kid = {
            int(row.get("candidate_kf_id", -1) or -1): dict(row)
            for row in diagnostics.loop_raw_dbow_trace_rows
            if int(row.get("candidate_kf_id", -1) or -1) >= 0
        }
        inverted_by_kid = {
            int(row.get("candidate_kf_id", -1) or -1): dict(row)
            for row in diagnostics.loop_inverted_word_trace_rows
            if int(row.get("candidate_kf_id", -1) or -1) >= 0
        }
        score_rows_by_kid: dict[int, list[dict]] = {}
        for row in diagnostics.loop_score_filter_trace_rows:
            candidate_kf_id = int(row.get("candidate_kf_id", -1) or -1)
            if candidate_kf_id < 0:
                continue
            score_rows_by_kid.setdefault(candidate_kf_id, []).append(dict(row))
        accumulation_rows_by_kid: dict[int, list[dict]] = {}
        for row in diagnostics.loop_accumulation_trace_rows:
            candidate_kf_id = int(row.get("candidate_kf_id", -1) or -1)
            if candidate_kf_id < 0:
                continue
            accumulation_rows_by_kid.setdefault(candidate_kf_id, []).append(dict(row))
        retained_by_kid = {
            int(row.get("candidate_kf_id", -1) or -1): dict(row)
            for row in diagnostics.loop_retained_candidate_trace_rows
            if int(row.get("candidate_kf_id", -1) or -1) >= 0
        }
        record_by_kid = {
            int(record.get("candidate_kf_id", -1) or -1): dict(record)
            for record in diagnostics.loop_debug_records
            if int(record.get("candidate_kf_id", -1) or -1) >= 0
        }

        rows: list[dict] = []
        for candidate in sorted(
            keyframes_map.values(),
            key=lambda item: int(getattr(item, "kid", getattr(item, "id", -1))),
        ):
            if candidate is None or candidate is current_keyframe or candidate.is_bad():
                continue
            candidate_kf_id = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
            if candidate_kf_id >= current_kf_id:
                continue
            if abs(int(current_kf_id) - int(candidate_kf_id)) <= int(
                getattr(Parameters, "kMinDeltaFrameForMeaningfulLoopClosure", 10)
            ):
                continue
            candidate_timestamp = getattr(candidate, "timestamp", None)
            if candidate_timestamp is None:
                continue
            gt_diag = oracle.describe_pair(float(current_timestamp), float(candidate_timestamp))
            if not bool(gt_diag.gt_loop_like):
                continue

            raw_row = raw_by_kid.get(candidate_kf_id)
            inverted_row = inverted_by_kid.get(candidate_kf_id)
            score_row = self._select_trace_row(score_rows_by_kid.get(candidate_kf_id, []))
            accumulation_row = self._select_trace_row(accumulation_rows_by_kid.get(candidate_kf_id, []))
            retained_row = retained_by_kid.get(candidate_kf_id)
            record = record_by_kid.get(candidate_kf_id)

            row = {
                "current_kf_id": int(current_kf_id),
                "candidate_kf_id": int(candidate_kf_id),
                "pair_key": self._pair_key(current_kf_id, candidate_kf_id),
                "current_timestamp": current_timestamp,
                "candidate_timestamp": candidate_timestamp,
                "gt_translation_distance": gt_diag.gt_translation_distance,
                "gt_rotation_angle_deg": gt_diag.gt_rotation_angle_deg,
                "gt_loop_like": True,
                "gt_near_loop": bool(gt_diag.gt_near_loop),
                "raw_dbow_present": bool(raw_row is not None),
                "raw_dbow_rank": raw_row.get("raw_rank") if raw_row else None,
                "raw_dbow_score": raw_row.get("raw_score") if raw_row else None,
                "raw_dbow_top_k": raw_row.get("raw_query_k") if raw_row else None,
                "raw_dbow_result_count": raw_row.get("raw_result_count") if raw_row else None,
                "inverted_word_present": bool(inverted_row is not None),
                "shared_words": inverted_row.get("shared_words") if inverted_row else None,
                "max_common_words": inverted_row.get("max_common_words") if inverted_row else None,
                "common_word_ratio": inverted_row.get("common_word_ratio") if inverted_row else None,
                "passed_common_word_filter": (
                    inverted_row.get("passed_common_word_filter") if inverted_row is not None else None
                ),
                "is_connected": (
                    raw_row.get("is_connected")
                    if raw_row is not None
                    else (inverted_row.get("is_connected") if inverted_row is not None else None)
                ),
                "temporal_gap": (
                    raw_row.get("temporal_gap")
                    if raw_row is not None
                    else (inverted_row.get("temporal_gap") if inverted_row is not None else None)
                ),
                "passed_connected_filter": (
                    raw_row.get("would_pass_connected_filter") if raw_row is not None else None
                ),
                "passed_temporal_filter": (
                    raw_row.get("would_pass_temporal_filter") if raw_row is not None else None
                ),
                "bow_score": score_row.get("bow_score") if score_row else None,
                "min_score": score_row.get("min_score") if score_row else None,
                "score_over_min_score": score_row.get("score_over_min_score") if score_row else None,
                "passed_min_score_filter": (
                    score_row.get("passed_min_score_filter") if score_row is not None else None
                ),
                "candidate_group_size": accumulation_row.get("candidate_group_size") if accumulation_row else None,
                "accumulated_score": accumulation_row.get("accumulated_score") if accumulation_row else None,
                "best_accumulated_score": (
                    accumulation_row.get("best_accumulated_score") if accumulation_row else None
                ),
                "accumulated_score_ratio": (
                    accumulation_row.get("accumulated_score_ratio") if accumulation_row else None
                ),
                "passed_accumulated_score_filter": (
                    accumulation_row.get("passed_accumulated_score_filter")
                    if accumulation_row is not None
                    else None
                ),
                "retained_candidate": bool(accumulation_row.get("retained_candidate")) if accumulation_row else False,
                "retained_rank": (
                    accumulation_row.get("retained_rank")
                    if accumulation_row and accumulation_row.get("retained_rank") not in {None, ""}
                    else (retained_row.get("retained_rank") if retained_row else None)
                ),
                "passed_consistency": record.get("passed_consistency") if record is not None else None,
                "passed_geometry_if_available": None,
                "accepted": bool(record.get("accepted", False)) if record is not None else False,
                "rejection_reason": record.get("rejection_reason", "") if record is not None else "",
                "consistency_score": (
                    retained_row.get("consistency_score_after")
                    if retained_row is not None
                    else (record.get("consistency_score_after") if record is not None else None)
                ),
                "final_matched_map_points": (
                    record.get(
                        "total_final_matches",
                        record.get("guided_projection_total_matches", record.get("final_inliers")),
                    )
                    if record is not None
                    else None
                ),
                "score_trace_source": score_row.get("candidate_source") if score_row else None,
                "accumulation_trace_source": accumulation_row.get("candidate_source") if accumulation_row else None,
                "first_failed_stage": "UNKNOWN",
                "diagnostic_confidence": "limited",
            }
            finalize_gt_positive_trace_row(row)
            rows.append(row)
        return rows

    @staticmethod
    def _select_trace_row(rows: list[dict]) -> dict | None:
        if not rows:
            return None
        preferred = sorted(
            rows,
            key=lambda row: (0 if str(row.get("candidate_source")) == "dbow3_scored" else 1),
        )
        return dict(preferred[0])

    @staticmethod
    def _copy_global_ba_diagnostics(diagnostics: LoopDiagnostics, result: Optional[GlobalBAResult]) -> None:
        if result is None:
            result = GlobalBAResult(started=False, reason="not run")
        diagnostics.global_ba_started = bool(result.started)
        diagnostics.global_ba_success = bool(result.success)
        diagnostics.global_ba_aborted = bool(result.aborted)
        diagnostics.global_ba_reason = result.reason
        diagnostics.global_ba_num_keyframes = int(result.num_keyframes)
        diagnostics.global_ba_num_map_points = int(result.num_map_points)
        diagnostics.global_ba_num_edges = int(result.num_edges)
        diagnostics.global_ba_num_inliers = int(result.num_inliers)
        diagnostics.global_ba_num_outliers = int(result.num_outliers)
        diagnostics.global_ba_mean_error_before = result.mean_error_before
        diagnostics.global_ba_mean_error_after = result.mean_error_after
        diagnostics.global_ba_elapsed_sec = float(result.elapsed_sec)


def finalize_gt_positive_trace_row(row: dict) -> dict:
    accepted = bool(row.get("accepted", False))
    if accepted:
        row["first_failed_stage"] = "ACCEPTED"
        row["passed_geometry_if_available"] = True
        row["diagnostic_confidence"] = "high"
        return row

    if not bool(row.get("raw_dbow_present", False)):
        row["first_failed_stage"] = "MISSING_FROM_RAW_DBOW"
        row["diagnostic_confidence"] = "high"
        return row

    if row.get("passed_connected_filter") is False:
        row["first_failed_stage"] = "FAILED_CONNECTED_FILTER"
        row["diagnostic_confidence"] = "high"
        return row

    if row.get("passed_temporal_filter") is False:
        row["first_failed_stage"] = "FAILED_TEMPORAL_FILTER"
        row["diagnostic_confidence"] = "high"
        return row

    if not bool(row.get("inverted_word_present", False)):
        row["first_failed_stage"] = "MISSING_FROM_INVERTED_WORD_SET"
        row["diagnostic_confidence"] = "high"
        return row

    if row.get("passed_common_word_filter") is False:
        row["first_failed_stage"] = "FAILED_COMMON_WORD_FILTER"
        row["diagnostic_confidence"] = "high"
        return row

    if row.get("passed_min_score_filter") is False:
        row["first_failed_stage"] = "FAILED_MIN_SCORE_FILTER"
        row["diagnostic_confidence"] = "high"
        return row

    if row.get("passed_accumulated_score_filter") is False:
        row["first_failed_stage"] = "FAILED_ACCUMULATION_FILTER"
        row["diagnostic_confidence"] = "high"
        return row

    if not bool(row.get("retained_candidate", False)):
        row["first_failed_stage"] = "NOT_RETAINED_AFTER_ACCUMULATION"
        row["diagnostic_confidence"] = "high"
        return row

    if row.get("passed_consistency") is False:
        row["first_failed_stage"] = "FAILED_CONSISTENCY"
        row["diagnostic_confidence"] = "high"
        return row

    rejection_reason = str(row.get("rejection_reason", "") or "").lower()
    if rejection_reason:
        if "matched map points after covisibility expansion" in rejection_reason:
            row["first_failed_stage"] = "FAILED_FINAL_SUPPORT"
            row["passed_geometry_if_available"] = True
            row["diagnostic_confidence"] = "medium"
            return row
        if "refined" in rejection_reason or "guided" in rejection_reason:
            row["first_failed_stage"] = "FAILED_REFINED_GEOMETRY"
            row["passed_geometry_if_available"] = False
            row["diagnostic_confidence"] = "medium"
            return row
        if "seed" in rejection_reason or "ransac" in rejection_reason or "estimated pose" in rejection_reason:
            row["first_failed_stage"] = "FAILED_SEED_GEOMETRY"
            row["passed_geometry_if_available"] = False
            row["diagnostic_confidence"] = "medium"
            return row

    row["first_failed_stage"] = "UNKNOWN"
    row["diagnostic_confidence"] = "limited"
    return row


# Provide a no-op lock interface when the map has no update lock.
class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _estimate_rmse(estimate: Sim3Estimate) -> Optional[float]:
    if estimate is None or estimate.inlier_mask is None:
        return None
    if not np.isfinite(estimate.mean_error):
        return None
    return float(estimate.mean_error)


def _rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> Optional[float]:
    try:
        Ra = np.asarray(rotation_a, dtype=np.float64).reshape(3, 3)
        Rb = np.asarray(rotation_b, dtype=np.float64).reshape(3, 3)
        trace_value = float((np.trace(Ra.T @ Rb) - 1.0) * 0.5)
        trace_value = max(-1.0, min(1.0, trace_value))
        angle = float(np.degrees(np.arccos(trace_value)))
    except Exception:
        return None
    return angle if np.isfinite(angle) else None


def _distance_summary(distances) -> dict:
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }
