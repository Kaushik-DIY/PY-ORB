#!/usr/bin/env python3
"""
Main dataset-agnostic RGB-D SLAM runner.
This script loads a dataset, executes the pipeline, and writes run artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import psutil
    import os
    _PROCESS = psutil.Process(os.getpid())
    def get_rss_mb():
        return _PROCESS.memory_info().rss / (1024 * 1024)
except ImportError:
    try:
        import resource
        def get_rss_mb():
            import sys
            divisor = 1024 * 1024 if sys.platform == 'darwin' else 1024
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    except ImportError:
        def get_rss_mb():
            return 0.0

from tools.export_orbslam_map import export_orbslam_map
from tools.run_fr1_room_full_evaluation import (
    LOOP_DEBUG_COLUMNS,
    _append_loop_debug_records,
    _write_candidate_pair_reports,
    write_csv,
)
from visual_slam.orbslam.io import save_tum_trajectory
from visual_slam.orbslam.io.rgbd_dataset import (
    DATASET_TYPE_AUTO,
    DATASET_TYPE_LAB,
    DATASET_TYPE_TUM,
    detect_dataset_type,
    load_rgbd_associations,
    make_rgbd_camera,
    resolve_camera_metadata,
)
from visual_slam.orbslam.slam import Slam, SlamState, SensorType
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.loop_oracle import TumLoopOracle
from visual_slam.orbslam.slam.runtime_profiler import RuntimeProfiler


FRAME_LOG_COLUMNS = [
    "i",
    "timestamp",
    "ok",
    "state",
    "keyframes",
    "points",
    "frames",
    "poses",
    "history",
    "last_tracked",
    "last_ba_mse",
    "lm_last_fused",
    "lm_last_triangulated",
    "loop_global_ba_started",
    "loop_global_ba_success",
    "loop_global_ba_reason",
    "loop_global_ba_edges",
    "loop_global_ba_inliers",
    "loop_global_ba_mse_after",
]

FRAME_TIMING_COLUMNS = [
    "frame_idx",
    "timestamp",
    "frame_total_sec",
    "load_rgb_sec",
    "load_depth_sec",
    "slam_track_sec",
    "local_mapping_sec",
    "loop_closing_sec",
    "prune_old_frame_views_sec",
    "memory_profile_sec",
    "rss_mb",
    "keyframes",
    "map_points",
    "recent_frames",
    "num_frame_views_total",
    "old_frame_views_total",
]

MEMORY_PROFILE_COLUMNS = [
    "frame_idx",
    "timestamp",
    "rss_mb",
    "keyframes",
    "map_points",
    "recent_frames",
    "num_frame_views_total",
    "old_frame_views_total",
    "keyframe_observations_total",
    "recent_frame_images",
    "recent_frame_depth_images",
    "keyframe_images",
    "keyframe_depth_images",
    "local_mapping_queue_size",
    "estimated_heavy_mb",
]

LOCAL_MAP_PROFILE_COLUMNS = [
    "frame_id",
    "timestamp",
    "kf_ref_id_before",
    "kf_ref_id_after",
    "num_current_matched_points",
    "num_current_good_matched_points",
    "num_voted_keyframes",
    "top_voted_kf_id",
    "top_voted_count",
    "num_local_keyframes",
    "num_local_keyframes_voted",
    "num_local_keyframes_expanded",
    "num_local_points",
    "num_already_seen_points",
    "num_bad_points_skipped",
    "num_visible_projected_points",
    "num_kd_candidates_total",
    "num_descriptor_comparisons",
    "num_projection_matches",
    "track_local_map_sec",
    "search_map_by_projection_sec",
    "pose_optimization_sec",
    "local_map_build_sec",
]

KEYFRAME_DECISION_COLUMNS = [
    "frame_id",
    "timestamp",
    "num_keyframes",
    "last_keyframe_id",
    "frames_since_last_kf",
    "min_frames_between_kfs",
    "max_frames_between_kfs",
    "sensor_type",
    "local_mapping_idle",
    "local_mapping_accepting",
    "local_mapping_queue_size",
    "local_mapping_abort_requested",
    "num_ref_tracked",
    "ref_min_obs",
    "num_matched_cur",
    "num_tracked_close",
    "num_non_tracked_close",
    "need_to_insert_close",
    "ref_ratio",
    "th_ref_ratio",
    "c1a",
    "c1b",
    "c1c",
    "c2",
    "inserted",
    "insert_reason",
    "reject_reason",
]

LOCAL_MAPPING_SCHEDULE_COLUMNS = [
    "kf_id",
    "timestamp",
    "queue_size_before",
    "queue_size_after",
    "accept_keyframes_before",
    "accept_keyframes_after",
    "is_single_thread",
    "processed_new_keyframe",
    "ran_cull_map_points",
    "ran_create_new_map_points",
    "ran_fuse_map_points",
    "ran_local_BA",
    "ran_cull_keyframes",
    "skipped_fuse_reason",
    "skipped_local_BA_reason",
    "local_BA_started",
    "local_BA_completed",
    "local_BA_aborted",
    "local_BA_forced_due_starvation",
    "keyframes_since_last_successful_ba",
    "local_BA_sec",
    "total_step_sec",
]

LOOP_CANDIDATE_ORACLE_COLUMNS = [
    "event_id",
    "current_kf_id",
    "candidate_kf_id",
    "current_timestamp",
    "candidate_timestamp",
    "time_gap_sec",
    "bow_score",
    "min_score",
    "common_words",
    "max_common_words",
    "common_word_ratio",
    "accumulated_score",
    "best_accumulated_score",
    "consistency_score",
    "consistency_group_id",
    "candidate_source",
    "candidate_rank",
    "is_connected",
    "temporal_gap",
    "rejection_stage",
    "rejection_reason",
    "raw_bow_matches",
    "valid_bow_map_point_matches",
    "seed_correspondences",
    "seed_inliers",
    "refined_inliers",
    "guided_projection_matches",
    "final_matched_map_points",
    "estimated_pose_distance",
    "estimated_rotation_deg",
    "gt_available",
    "gt_translation_distance",
    "gt_rotation_angle_deg",
    "gt_loop_like",
    "gt_near_loop",
    "accepted",
]

LOOP_RETRIEVAL_PROFILE_COLUMNS = [
    "kf_id",
    "timestamp",
    "num_db_keyframes_before_query",
    "candidate_source",
    "num_raw_dbow_candidates",
    "num_raw_inverted_candidates",
    "num_candidates_after_temporal_filter",
    "num_candidates_after_connected_filter",
    "num_candidates_after_common_words",
    "num_candidates_after_min_score",
    "num_candidates_after_accumulation",
    "num_candidates_after_consistency",
    "top_candidate_id",
    "top_candidate_score",
    "top_candidate_acc_score",
    "top_candidate_consistency",
    "accepted_candidate_id",
]

LOOP_SOURCE_COMPARISON_COLUMNS = [
    "kf_id",
    "timestamp",
    "candidate_source",
    "dbow3_candidates",
    "inverted_file_candidates",
    "intersection_candidates",
    "dbow3_only_candidates",
    "inverted_only_candidates",
    "chosen_candidates",
]

LOOP_KEYFRAME_DENSITY_COLUMNS = [
    "current_kf_id",
    "candidate_kf_id",
    "current_neighbor_count",
    "candidate_neighbor_count",
    "current_local_map_points",
    "candidate_local_map_points",
    "shared_bow_words",
    "shared_map_points_if_any",
    "candidate_group_size",
    "current_group_size",
    "final_matched_map_points",
    "gt_translation_distance",
    "accepted",
    "rejection_reason",
]

LOOP_RAW_DBOW_TRACE_COLUMNS = [
    "current_kf_id",
    "candidate_kf_id",
    "pair_key",
    "current_timestamp",
    "candidate_timestamp",
    "raw_rank",
    "raw_score",
    "raw_source",
    "db_size_before_query",
    "raw_query_k",
    "raw_result_count",
    "is_self",
    "is_bad",
    "is_connected",
    "temporal_gap",
    "would_pass_connected_filter",
    "would_pass_temporal_filter",
]

LOOP_INVERTED_WORD_TRACE_COLUMNS = [
    "current_kf_id",
    "candidate_kf_id",
    "pair_key",
    "current_timestamp",
    "candidate_timestamp",
    "candidate_source",
    "shared_words",
    "max_common_words",
    "common_word_ratio",
    "common_word_threshold_ratio",
    "passed_common_word_filter",
    "is_connected",
    "temporal_gap",
    "passed_connected_filter",
    "passed_temporal_filter",
]

LOOP_SCORE_FILTER_TRACE_COLUMNS = [
    "current_kf_id",
    "candidate_kf_id",
    "pair_key",
    "current_timestamp",
    "candidate_timestamp",
    "candidate_source",
    "bow_score",
    "min_score",
    "score_over_min_score",
    "connected_kf_count_for_min_score",
    "min_score_source_kf_id",
    "passed_min_score_filter",
    "connected_scores_json",
]

LOOP_ACCUMULATION_TRACE_COLUMNS = [
    "current_kf_id",
    "candidate_kf_id",
    "pair_key",
    "current_timestamp",
    "candidate_timestamp",
    "candidate_source",
    "candidate_group_size",
    "candidate_group_ids",
    "candidate_group_scores",
    "accumulated_score",
    "best_accumulated_score",
    "accumulated_score_ratio",
    "accumulation_threshold_ratio",
    "passed_accumulated_score_filter",
    "best_candidate_id_in_group",
    "retained_candidate",
    "retained_rank",
]

LOOP_RETAINED_CANDIDATE_TRACE_COLUMNS = [
    "current_kf_id",
    "candidate_kf_id",
    "pair_key",
    "current_timestamp",
    "candidate_timestamp",
    "retained_rank",
    "candidate_source",
    "bow_score",
    "accumulated_score",
    "best_accumulated_score",
    "consistency_score_before",
    "consistency_score_after",
    "passed_consistency",
    "rejection_reason_if_any",
    "accepted",
]

LOOP_GT_POSITIVE_TRACE_COLUMNS = [
    "current_kf_id",
    "candidate_kf_id",
    "pair_key",
    "current_timestamp",
    "candidate_timestamp",
    "gt_translation_distance",
    "gt_rotation_angle_deg",
    "gt_loop_like",
    "gt_near_loop",
    "raw_dbow_present",
    "raw_dbow_rank",
    "raw_dbow_score",
    "raw_dbow_top_k",
    "raw_dbow_result_count",
    "inverted_word_present",
    "shared_words",
    "max_common_words",
    "common_word_ratio",
    "passed_common_word_filter",
    "is_connected",
    "temporal_gap",
    "passed_connected_filter",
    "passed_temporal_filter",
    "bow_score",
    "min_score",
    "score_over_min_score",
    "passed_min_score_filter",
    "candidate_group_size",
    "accumulated_score",
    "best_accumulated_score",
    "accumulated_score_ratio",
    "passed_accumulated_score_filter",
    "retained_candidate",
    "retained_rank",
    "passed_consistency",
    "passed_geometry_if_available",
    "accepted",
    "consistency_score",
    "final_matched_map_points",
    "rejection_reason",
    "score_trace_source",
    "accumulation_trace_source",
    "first_failed_stage",
    "diagnostic_confidence",
]

LOOP_CONSISTENCY_PROGRESSION_COLUMNS = [
    "current_kf_id",
    "candidate_kf_id",
    "pair_key",
    "candidate_group_ids",
    "previous_group_ids",
    "overlap_count",
    "previous_consistency",
    "new_consistency",
    "threshold",
    "passed_consistency",
    "gt_loop_like",
    "gt_translation_distance",
    "gt_rotation_angle_deg",
]

LOOP_GEOMETRY_TRACE_COLUMNS = [
    "current_kf_id",
    "candidate_kf_id",
    "pair_key",
    "gt_loop_like",
    "gt_translation_distance",
    "gt_rotation_angle_deg",
    "raw_bow_matches",
    "valid_bow_map_point_matches",
    "seed_correspondences",
    "seed_inliers",
    "seed_inlier_ratio",
    "initial_se3_translation_norm",
    "initial_se3_rotation_deg",
    "pose_distance_gate_threshold",
    "pose_rotation_gate_threshold",
    "passed_pose_distance_gate",
    "guided_projection_matches",
    "refined_correspondences",
    "refined_inliers",
    "candidate_group_size",
    "candidate_group_map_points",
    "visible_projected_group_points",
    "final_matched_map_points",
    "final_gate_threshold",
    "accepted",
    "rejection_reason",
]

PARAMETER_NAMES_TO_SNAPSHOT = (
    "kLocalMappingDebugAndPrintToFile",
    "kLoopClosingDebugAndPrintToFile",
    "kLoopClosingDebugWithLoopDetectionImages",
    "kLoopClosingDebugWithSimmetryMatrix",
    "kLoopClosingDebugWithLoopConsistencyCheckImages",
    "kStoreNormalFrameImages",
    "kStoreKeyFrameImages",
    "kStoreKeyFrameDepthImages",
    "kReleaseNormalFrameImagesAfterUse",
    "kReleaseEvictedFrameFeatureCache",
    "kEnableFrameViewPruning",
    "kFrameViewPruneEveryNFrames",
    "kFrameViewRetention",
    "kMaxLenFrameDeque",
    "kWaitForLocalMappingTimeout",
    "kMinFramesBetweenKeyframesSequentialRgbd",
    "kMinFramesBetweenKeyframesThreadedRgbd",
    "kUseFpsAwareKeyframeSpacing",
    "kMinKeyframeSpacingSeconds",
    "kLocalMappingMaxQueueForForcedInsert",
    "kNewKeyframeRefMinObs",
    "kEnableLocalBAStarvationGuard",
    "kMaxKeyframesWithoutLocalBA",
    "kMaxConsecutiveLocalBAAborts",
    "kForceLocalBAWhenStarved",
    "kLoopCandidateSource",
)


def _load_rgb(path: Path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load RGB image: {path}")
    return img_bgr


def _load_depth(path: Path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not load depth image: {path}")
    return depth


def _state_name(state):
    try:
        return state.name
    except Exception:
        return str(state)


def _finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _json_dump(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_.-")
    return text or "unnamed"


class ParameterSnapshot:
    def __init__(self, names):
        self._names = tuple(names)
        self._values = {name: getattr(Parameters, name) for name in self._names}

    def restore(self) -> None:
        for name, value in self._values.items():
            setattr(Parameters, name, value)


@contextmanager
def temporary_parameters(names):
    snapshot = ParameterSnapshot(names)
    try:
        yield snapshot
    finally:
        snapshot.restore()


def _append_loop_candidate_oracle_rows(rows: list[dict], diagnostics, *, event_start: int) -> int:
    count = 0
    for offset, row in enumerate(list(getattr(diagnostics, "loop_candidate_oracle_rows", []) or [])):
        payload = dict(row)
        payload["event_id"] = int(event_start + offset)
        rows.append(payload)
        count += 1
    return count


def _append_loop_rows(rows: list[dict], values) -> int:
    count = 0
    for row in list(values or []):
        rows.append(dict(row))
        count += 1
    return count


def _apply_runtime_parameter_overrides(
    *,
    lean_memory: bool,
    lm_wait_timeout: float,
    no_heavy_loop_reports: bool,
    no_loop_candidate_pair_reports: bool,
    frame_view_prune_every: int,
) -> tuple[bool, bool]:
    Parameters.kWaitForLocalMappingTimeout = float(lm_wait_timeout)

    if frame_view_prune_every > 0:
        Parameters.kFrameViewPruneEveryNFrames = int(frame_view_prune_every)

    if lean_memory:
        Parameters.kLocalMappingDebugAndPrintToFile = False
        Parameters.kLoopClosingDebugAndPrintToFile = False
        Parameters.kLoopClosingDebugWithLoopDetectionImages = False
        Parameters.kLoopClosingDebugWithSimmetryMatrix = False
        Parameters.kLoopClosingDebugWithLoopConsistencyCheckImages = False
        Parameters.kStoreNormalFrameImages = False
        Parameters.kStoreKeyFrameImages = False
        Parameters.kStoreKeyFrameDepthImages = False
        no_heavy_loop_reports = True
        no_loop_candidate_pair_reports = True

    return bool(no_heavy_loop_reports), bool(no_loop_candidate_pair_reports)


def build_completed_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def build_standardized_output_stem(dataset_type: str, dataset_name: str, completed_timestamp: str) -> str:
    return f"{_slugify(dataset_type)}__{_slugify(dataset_name)}__completed_{completed_timestamp}"


def build_standardized_artifact_paths(output_dir: Path, dataset_type: str, dataset_name: str, completed_timestamp: str) -> dict:
    stem = build_standardized_output_stem(dataset_type, dataset_name, completed_timestamp)
    return {
        "stem": stem,
        "trajectory_file": output_dir / f"trajectory__{stem}.txt",
        "frame_log_file": output_dir / f"frame_log__{stem}.csv",
        "frame_timing_file": output_dir / f"frame_timing__{stem}.csv",
        "map_points_ply": output_dir / f"map_points__{stem}.ply",
        "keyframes_json": output_dir / f"keyframes__{stem}.json",
        "keyframe_graph_json": output_dir / f"keyframe_graph__{stem}.json",
        "effective_run_config_json": output_dir / f"effective_run_config__{stem}.json",
        "run_summary_json": output_dir / f"run_summary__{stem}.json",
        "loop_debug_file": output_dir / f"loop_debug_candidates__{stem}.csv",
        "memory_profile_file": output_dir / f"memory_profile__{stem}.csv",
        "local_map_profile_file": output_dir / f"local_map_profile__{stem}.csv",
        "keyframe_decision_log_file": output_dir / f"keyframe_decision_log__{stem}.csv",
        "local_mapping_schedule_log_file": output_dir / f"local_mapping_schedule_log__{stem}.csv",
        "runtime_profile_csv": output_dir / f"runtime_profile__{stem}.csv",
        "runtime_profile_json": output_dir / f"runtime_profile__{stem}.json",
        "loop_candidate_oracle_file": output_dir / f"loop_candidate_oracle__{stem}.csv",
        "loop_retrieval_profile_file": output_dir / f"loop_retrieval_profile__{stem}.csv",
        "loop_candidate_source_comparison_file": output_dir / f"loop_candidate_source_comparison__{stem}.csv",
        "loop_keyframe_density_profile_file": output_dir / f"loop_keyframe_density_profile__{stem}.csv",
        "loop_raw_dbow_trace_file": output_dir / f"loop_raw_dbow_trace__{stem}.csv",
        "loop_inverted_word_trace_file": output_dir / f"loop_inverted_word_trace__{stem}.csv",
        "loop_score_filter_trace_file": output_dir / f"loop_score_filter_trace__{stem}.csv",
        "loop_accumulation_trace_file": output_dir / f"loop_accumulation_trace__{stem}.csv",
        "loop_retained_candidate_trace_file": output_dir / f"loop_retained_candidate_trace__{stem}.csv",
        "loop_gt_positive_trace_file": output_dir / f"loop_gt_positive_trace__{stem}.csv",
        "loop_consistency_progression_file": output_dir / f"loop_consistency_progression__{stem}.csv",
        "loop_geometry_trace_file": output_dir / f"loop_geometry_trace__{stem}.csv",
    }


def _copy_if_exists(src: Path | None, dst: Path | None) -> str | None:
    if src is None or dst is None or not Path(src).exists():
        return None
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def build_effective_run_config(
    *,
    dataset: Path,
    dataset_name: str,
    dataset_type: str,
    output_dir: Path,
    camera_profile: str,
    camera_config: Path | None,
    associations: Path | None,
    camera_metadata: dict,
    feature_backend: str,
    enable_loop_closing: bool,
    enable_global_ba: bool,
    global_ba_after_loop: bool,
    global_ba_iterations: int,
    max_frames: int,
    start_index: int,
    print_every: int,
    loop_debug: bool,
    loop_retrieval_trace: bool = False,
    loop_retrieval_trace_raw_k: int = 0,
    stop_after_loop_events: int,
    stop_after_accepted_loops: int,
    dump_loop_candidate_reports: bool,
    start_local_mapping_thread: bool,
    lm_wait_timeout: float,
    profile_memory: bool = False,
    memory_profile_every: int = 0,
    memory_profile_mode: str = "cheap",
    memory_limit_gb: float = 0.0,
    profile_runtime: bool = False,
    runtime_profile_every: int = 0,
    profile_local_map: bool = False,
    profile_keyframes: bool = False,
    lean_memory: bool = False,
    no_map_export: bool = False,
    no_heavy_loop_reports: bool = False,
    no_loop_candidate_pair_reports: bool = False,
    loop_candidate_source: str = "auto",
    frame_view_prune_every: int | None = None,
    completed_timestamp: str | None = None,
    standardized_output_stem: str | None = None,
) -> dict:
    return {
        "dataset_path": str(dataset),
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "output_dir": str(output_dir),
        "completed_timestamp": completed_timestamp,
        "standardized_output_stem": standardized_output_stem,
        "camera_profile": camera_profile,
        "camera_config": str(camera_config) if camera_config is not None else None,
        "associations": str(associations) if associations is not None else None,
        "camera": {
            "source": camera_metadata.get("camera_source"),
            "sensor_type": camera_metadata.get("sensor_type"),
            "width": camera_metadata.get("width"),
            "height": camera_metadata.get("height"),
            "fps": camera_metadata.get("fps"),
            "fx": camera_metadata.get("fx"),
            "fy": camera_metadata.get("fy"),
            "cx": camera_metadata.get("cx"),
            "cy": camera_metadata.get("cy"),
            "distortion": camera_metadata.get("distortion"),
            "depth_map_factor": camera_metadata.get("depth_map_factor"),
            "depth_factor": camera_metadata.get("depth_factor"),
            "depth_threshold": camera_metadata.get("depth_threshold"),
            "depth_threshold_source": camera_metadata.get("depth_threshold_source"),
            "baseline_m": camera_metadata.get("baseline_m"),
            "baseline_source": camera_metadata.get("baseline_source"),
            "bf": camera_metadata.get("bf"),
        },
        "feature_backend": feature_backend,
        "enable_loop_closing": bool(enable_loop_closing),
        "enable_global_ba": bool(enable_global_ba),
        "global_ba_after_loop": bool(global_ba_after_loop),
        "global_ba_iterations": int(global_ba_iterations),
        "max_frames": int(max_frames),
        "start_index": int(start_index),
        "print_every": int(print_every),
        "loop_debug": bool(loop_debug),
        "loop_retrieval_trace": bool(loop_retrieval_trace),
        "loop_retrieval_trace_raw_k": int(loop_retrieval_trace_raw_k),
        "stop_after_loop_events": int(stop_after_loop_events),
        "stop_after_accepted_loops": int(stop_after_accepted_loops),
        "dump_loop_candidate_reports": bool(dump_loop_candidate_reports),
        "start_local_mapping_thread": bool(start_local_mapping_thread),
        "lm_wait_timeout": float(lm_wait_timeout),
        "profile_memory": bool(profile_memory),
        "memory_profile_every": int(memory_profile_every),
        "memory_profile_mode": str(memory_profile_mode),
        "memory_limit_gb": float(memory_limit_gb),
        "profile_runtime": bool(profile_runtime),
        "runtime_profile_every": int(runtime_profile_every),
        "profile_local_map": bool(profile_local_map),
        "profile_keyframes": bool(profile_keyframes),
        "lean_memory": bool(lean_memory),
        "no_map_export": bool(no_map_export),
        "no_heavy_loop_reports": bool(no_heavy_loop_reports),
        "no_loop_candidate_pair_reports": bool(no_loop_candidate_pair_reports),
        "loop_candidate_source": str(loop_candidate_source),
        "frame_view_prune_every": int(frame_view_prune_every) if frame_view_prune_every is not None else None,
    }


def write_effective_run_config(output_dir: Path, config: dict) -> Path:
    return _json_dump(output_dir / "effective_run_config.json", config)


def build_run_summary(
    *,
    dataset_name: str,
    dataset_type: str,
    frames_attempted: int,
    tracking_ok_count: int,
    tracking_lost_count: int,
    errors: int,
    final_state: str,
    keyframes: int,
    map_points: int,
    trajectory_poses: int,
    elapsed_sec: float,
    avg_fps: float,
    feature_backend: str,
    enable_loop_closing: bool,
    enable_global_ba: bool,
    global_ba_after_loop: bool,
    loop_debug_events: int,
    accepted_loops: int,
    output_files: dict,
    completed_timestamp: str,
    standardized_output_stem: str,
) -> dict:
    return {
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "completed_timestamp": completed_timestamp,
        "standardized_output_stem": standardized_output_stem,
        "frames_attempted": int(frames_attempted),
        "tracking_ok_count": int(tracking_ok_count),
        "tracking_lost_count": int(tracking_lost_count),
        "errors": int(errors),
        "final_state": final_state,
        "keyframes": int(keyframes),
        "map_points": int(map_points),
        "trajectory_poses": int(trajectory_poses),
        "elapsed_sec": float(elapsed_sec),
        "avg_fps": float(avg_fps),
        "feature_backend": feature_backend,
        "loop_closing_enabled": bool(enable_loop_closing),
        "global_ba_enabled": bool(enable_global_ba),
        "global_ba_after_loop": bool(global_ba_after_loop),
        "loop_debug_events": int(loop_debug_events),
        "accepted_loops": int(accepted_loops),
        "final_keyframes": int(keyframes),
        "final_map_points": int(map_points),
        "trajectory_file": output_files.get("trajectory_file"),
        "frame_log_file": output_files.get("frame_log_file"),
        "frame_timing_file": output_files.get("frame_timing_file"),
        "map_points_ply": output_files.get("map_points_ply"),
        "keyframes_json": output_files.get("keyframes_json"),
        "keyframe_graph_json": output_files.get("keyframe_graph_json"),
        "effective_run_config_json": output_files.get("effective_run_config_json"),
        "loop_debug_file": output_files.get("loop_debug_file"),
        "runtime_profile_csv": output_files.get("runtime_profile_csv"),
        "runtime_profile_json": output_files.get("runtime_profile_json"),
        "loop_candidate_oracle_file": output_files.get("loop_candidate_oracle_file"),
        "loop_retrieval_profile_file": output_files.get("loop_retrieval_profile_file"),
        "loop_candidate_source_comparison_file": output_files.get("loop_candidate_source_comparison_file"),
        "loop_keyframe_density_profile_file": output_files.get("loop_keyframe_density_profile_file"),
        "local_map_profile_file": output_files.get("local_map_profile_file"),
        "keyframe_decision_log_file": output_files.get("keyframe_decision_log_file"),
        "local_mapping_schedule_log_file": output_files.get("local_mapping_schedule_log_file"),
        "standardized_trajectory_file": output_files.get("standardized_trajectory_file"),
        "standardized_frame_log_file": output_files.get("standardized_frame_log_file"),
        "standardized_frame_timing_file": output_files.get("standardized_frame_timing_file"),
        "standardized_map_points_ply": output_files.get("standardized_map_points_ply"),
        "standardized_keyframes_json": output_files.get("standardized_keyframes_json"),
        "standardized_keyframe_graph_json": output_files.get("standardized_keyframe_graph_json"),
        "standardized_loop_debug_file": output_files.get("standardized_loop_debug_file"),
        "standardized_loop_candidate_oracle_file": output_files.get("standardized_loop_candidate_oracle_file"),
        "standardized_loop_retrieval_profile_file": output_files.get("standardized_loop_retrieval_profile_file"),
        "standardized_loop_candidate_source_comparison_file": output_files.get("standardized_loop_candidate_source_comparison_file"),
        "standardized_loop_keyframe_density_profile_file": output_files.get("standardized_loop_keyframe_density_profile_file"),
        "standardized_local_map_profile_file": output_files.get("standardized_local_map_profile_file"),
        "standardized_keyframe_decision_log_file": output_files.get("standardized_keyframe_decision_log_file"),
        "standardized_local_mapping_schedule_log_file": output_files.get("standardized_local_mapping_schedule_log_file"),
        "peak_rss_mb": float(output_files.get("peak_rss_mb", 0.0)),
        "final_rss_mb": float(output_files.get("final_rss_mb", 0.0)),
        "memory_profile_file": output_files.get("memory_profile_file"),
        "num_frame_views_total": int(output_files.get("num_frame_views_total", 0)),
        "old_frame_views_total": int(output_files.get("old_frame_views_total", 0)),
        "local_ba_started_count": int(output_files.get("local_ba_started_count", 0)),
        "local_ba_completed_count": int(output_files.get("local_ba_completed_count", 0)),
        "local_ba_aborted_count": int(output_files.get("local_ba_aborted_count", 0)),
        "local_ba_skipped_due_queue_count": int(output_files.get("local_ba_skipped_due_queue_count", 0)),
        "local_ba_forced_due_starvation_count": int(output_files.get("local_ba_forced_due_starvation_count", 0)),
        "last_successful_local_ba_kid": int(output_files.get("last_successful_local_ba_kid", -1)),
        "keyframes_since_last_successful_ba": int(output_files.get("keyframes_since_last_successful_ba", 0)),
        "consecutive_local_ba_aborts": int(output_files.get("consecutive_local_ba_aborts", 0)),
    }


def write_run_summary(output_dir: Path, summary: dict) -> Path:
    return _json_dump(output_dir / "run_summary.json", summary)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final RGB-D ORB-SLAM pipeline on TUM or lab datasets.")
    parser.add_argument("dataset", type=Path, help="Path to the RGB-D dataset root.")
    parser.add_argument(
        "--dataset-type",
        choices=(DATASET_TYPE_AUTO, DATASET_TYPE_TUM, DATASET_TYPE_LAB),
        default=DATASET_TYPE_AUTO,
        help="Dataset layout selector. Use auto unless detection is ambiguous.",
    )
    parser.add_argument(
        "--camera-profile",
        default="auto",
        help="Camera profile for TUM datasets. Use auto to preserve the current Freiburg detection logic.",
    )
    parser.add_argument("--camera-config", type=Path, default=None, help="Optional camera.yaml for lab_rgbd datasets.")
    parser.add_argument("--associations", type=Path, default=None, help="Optional associations.txt override.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for trajectory, logs, and map export.")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process. Use 0 or -1 for full sequence.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--feature-backend",
        choices=("opencv_orb", "pyslam_orb2", "auto"),
        default="auto",
        help="Feature extractor backend override.",
    )
    loop_group = parser.add_mutually_exclusive_group()
    loop_group.add_argument("--enable-loop-closing", action="store_true", help="Enable loop closing.")
    loop_group.add_argument("--disable-loop-closing", action="store_true", help="Disable loop closing.")
    gba_group = parser.add_mutually_exclusive_group()
    gba_group.add_argument("--enable-global-ba", action="store_true", help="Enable loop-triggered Global BA.")
    gba_group.add_argument("--disable-global-ba", action="store_true", help="Disable loop-triggered Global BA.")
    parser.add_argument("--global-ba-after-loop", action="store_true", help="Run Global BA after accepted loop closures.")
    parser.add_argument("--global-ba-iterations", type=int, default=10)
    parser.add_argument("--loop-debug", action="store_true")
    parser.add_argument("--loop-retrieval-trace", action="store_true")
    parser.add_argument("--loop-retrieval-trace-raw-k", type=int, default=0)
    parser.add_argument("--stop-after-loop-events", type=int, default=0)
    parser.add_argument("--stop-after-accepted-loops", type=int, default=0)
    parser.add_argument("--dump-loop-candidate-reports", action="store_true")
    parser.add_argument(
        "--loop-candidate-source",
        choices=(
            "auto",
            "classic_inverted",
            "dbow_detector",
            "hybrid_dbow_scored",
            "compare",
            "inverted_file",
            "dbow3",
            "dbow3_scored",
        ),
        default=getattr(Parameters, "kLoopCandidateSource", "auto"),
    )
    parser.add_argument("--start-local-mapping-thread", action="store_true")
    parser.add_argument("--lm-wait-timeout", type=float, default=0.5)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--memory-profile-every", type=int, default=1)
    parser.add_argument("--memory-profile-mode", choices=("cheap", "deep"), default="cheap")
    parser.add_argument("--profile-runtime", action="store_true")
    parser.add_argument("--runtime-profile-every", type=int, default=1)
    parser.add_argument("--profile-local-map", action="store_true")
    parser.add_argument("--profile-keyframes", action="store_true")
    parser.add_argument("--memory-limit-gb", type=float, default=0.0)
    parser.add_argument("--frame-view-prune-every", type=int, default=Parameters.kFrameViewPruneEveryNFrames)
    parser.add_argument("--lean-memory", action="store_true")
    parser.add_argument("--no-map-export", action="store_true")
    parser.add_argument("--no-heavy-loop-reports", action="store_true")
    parser.add_argument("--no-loop-candidate-pair-reports", action="store_true")
    return parser


def run_rgbd_slam(
    dataset: Path,
    output_dir: Path,
    *,
    dataset_type: str = DATASET_TYPE_AUTO,
    camera_profile: str = "auto",
    camera_config: Path | None = None,
    associations: Path | None = None,
    max_frames: int = 0,
    start_index: int = 0,
    print_every: int = 1,
    feature_backend: str = "auto",
    enable_loop_closing: bool = False,
    enable_global_ba: bool = False,
    global_ba_after_loop: bool = False,
    global_ba_iterations: int = 10,
    loop_debug: bool = False,
    loop_retrieval_trace: bool = False,
    loop_retrieval_trace_raw_k: int = 0,
    stop_after_loop_events: int = 0,
    stop_after_accepted_loops: int = 0,
    dump_loop_candidate_reports: bool = False,
    loop_candidate_source: str = "auto",
    start_local_mapping_thread: bool = False,
    lm_wait_timeout: float = 0.5,
    profile_memory: bool = False,
    memory_profile_every: int = 1,
    memory_profile_mode: str = "cheap",
    profile_runtime: bool = False,
    runtime_profile_every: int = 1,
    profile_local_map: bool = False,
    profile_keyframes: bool = False,
    memory_limit_gb: float = 0.0,
    frame_view_prune_every: int = Parameters.kFrameViewPruneEveryNFrames,
    lean_memory: bool = False,
    no_map_export: bool = False,
    no_heavy_loop_reports: bool = False,
    no_loop_candidate_pair_reports: bool = False,
) -> dict:
    dataset = Path(dataset).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    global_ba_after_loop = bool(global_ba_after_loop or enable_global_ba)

    effective_dataset_type = dataset_type
    if effective_dataset_type == DATASET_TYPE_AUTO:
        effective_dataset_type = detect_dataset_type(dataset)

    camera = make_rgbd_camera(
        dataset,
        dataset_type=effective_dataset_type,
        camera_profile=camera_profile,
        camera_config=camera_config,
    )
    camera_metadata = resolve_camera_metadata(
        dataset,
        dataset_type=effective_dataset_type,
        camera_profile=camera_profile,
        camera_config=camera_config,
    )
    dataset_name = str(camera_metadata.get("dataset_name") or dataset.name)

    frames = load_rgbd_associations(dataset, associations=associations)
    if len(frames) == 0:
        raise RuntimeError(f"No RGB-D associations found in {dataset}")

    if start_index > 0:
        frames = frames[start_index:]
    if max_frames > 0:
        frames = frames[:max_frames]

    selected_backend = None if feature_backend in {None, "auto"} else feature_backend
    feature_tracker_config = None if selected_backend is None else {"extractor_backend": selected_backend}
    memory_profile_every = max(1, int(memory_profile_every))
    runtime_profile_every = max(1, int(runtime_profile_every))
    memory_profile_mode = "deep" if str(memory_profile_mode).lower() == "deep" else "cheap"
    frame_view_prune_every = max(1, int(frame_view_prune_every))
    effective_profile_local_map = bool(profile_local_map or profile_runtime)
    effective_profile_keyframes = bool(profile_keyframes or profile_runtime)

    with temporary_parameters(PARAMETER_NAMES_TO_SNAPSHOT):
        no_heavy_loop_reports, no_loop_candidate_pair_reports = _apply_runtime_parameter_overrides(
            lean_memory=lean_memory,
            lm_wait_timeout=lm_wait_timeout,
            no_heavy_loop_reports=no_heavy_loop_reports,
            no_loop_candidate_pair_reports=no_loop_candidate_pair_reports,
            frame_view_prune_every=frame_view_prune_every,
        )
        Parameters.kLoopCandidateSource = str(loop_candidate_source or "auto")

        run_config = build_effective_run_config(
            dataset=dataset,
            dataset_name=dataset_name,
            dataset_type=effective_dataset_type,
            output_dir=output_dir,
            camera_profile=camera_profile,
            camera_config=Path(camera_metadata["camera_source"]).resolve() if effective_dataset_type == DATASET_TYPE_LAB else camera_config,
            associations=associations,
            camera_metadata=camera_metadata,
            feature_backend=feature_backend,
            enable_loop_closing=enable_loop_closing,
            enable_global_ba=enable_global_ba,
            global_ba_after_loop=global_ba_after_loop,
            global_ba_iterations=global_ba_iterations,
            max_frames=max_frames,
            start_index=start_index,
            print_every=print_every,
            loop_debug=loop_debug,
            loop_retrieval_trace=bool(loop_retrieval_trace),
            loop_retrieval_trace_raw_k=int(loop_retrieval_trace_raw_k),
            stop_after_loop_events=stop_after_loop_events,
            stop_after_accepted_loops=stop_after_accepted_loops,
            dump_loop_candidate_reports=dump_loop_candidate_reports,
            loop_candidate_source=Parameters.kLoopCandidateSource,
            start_local_mapping_thread=start_local_mapping_thread,
            lm_wait_timeout=Parameters.kWaitForLocalMappingTimeout,
            profile_memory=profile_memory,
            memory_profile_every=memory_profile_every,
            memory_profile_mode=memory_profile_mode,
            memory_limit_gb=memory_limit_gb,
            profile_runtime=profile_runtime,
            runtime_profile_every=runtime_profile_every,
            profile_local_map=effective_profile_local_map,
            profile_keyframes=effective_profile_keyframes,
            lean_memory=lean_memory,
            no_map_export=no_map_export,
            no_heavy_loop_reports=no_heavy_loop_reports,
            no_loop_candidate_pair_reports=no_loop_candidate_pair_reports,
            frame_view_prune_every=Parameters.kFrameViewPruneEveryNFrames,
        )
        effective_run_config_path = write_effective_run_config(output_dir, run_config)

        profiler = RuntimeProfiler(enabled=profile_runtime)
        slam = None
        try:
            slam = Slam(
                camera=camera,
                sensor_type=SensorType.RGBD,
                headless=True,
                start_local_mapping_thread=start_local_mapping_thread,
                feature_tracker_config=feature_tracker_config,
                enable_loop_closing=enable_loop_closing,
                enable_global_ba=enable_global_ba,
                global_ba_after_loop=global_ba_after_loop,
                global_ba_iterations=global_ba_iterations,
            )
            slam.runtime_profiler = profiler
            slam.tracking.profile_local_map = effective_profile_local_map
            slam.tracking.profile_keyframes = effective_profile_keyframes
            slam.local_mapping.profile_keyframes = effective_profile_keyframes
            threaded_lm = slam.start_local_mapping_thread
            groundtruth_path = dataset / "groundtruth.txt"
            if (
                effective_dataset_type == DATASET_TYPE_TUM
                and getattr(slam, "loop_closing", None) is not None
                and groundtruth_path.exists()
            ):
                try:
                    slam.loop_closing.set_loop_oracle(
                        TumLoopOracle.from_tum_groundtruth(
                            groundtruth_path,
                            max_time_diff=float(getattr(Parameters, "kLoopOracleMaxGtTimeDiffSec", 0.05)),
                        )
                    )
                except Exception as exc:
                    print(f"[WARN] Failed to load TUM loop oracle from {groundtruth_path}: {exc}")
            if (
                getattr(slam, "loop_closing", None) is not None
                and hasattr(slam.loop_closing, "set_retrieval_trace_config")
            ):
                slam.loop_closing.set_retrieval_trace_config(
                    enabled=bool(loop_retrieval_trace),
                    raw_k=int(loop_retrieval_trace_raw_k),
                )

            print("=" * 80)
            print("Final RGB-D ORB-SLAM run")
            print("=" * 80)
            print(f"Dataset:             {dataset}")
            print(f"Dataset type:        {effective_dataset_type}")
            print(f"Output:              {output_dir}")
            print(f"Frames loaded:       {len(frames)}")
            print(f"Camera source:       {camera_metadata.get('camera_source')}")
            print(
                f"Camera intrinsics:   fx={camera.fx:.3f}, fy={camera.fy:.3f}, "
                f"cx={camera.cx:.3f}, cy={camera.cy:.3f}"
            )
            print(f"Image size:          {camera.width}x{camera.height}")
            print(f"Depth map factor:    {camera_metadata.get('depth_map_factor')}")
            print(f"Depth factor:        {camera.depth_factor}")
            print(
                f"Baseline (m):        {camera_metadata.get('baseline_m')} "
                f"[{camera_metadata.get('baseline_source', 'unspecified')}]"
            )
            print(
                f"Depth threshold:     {camera_metadata.get('depth_threshold')} "
                f"[{camera_metadata.get('depth_threshold_source', 'unspecified')}]"
            )
            print(f"bf:                  {camera_metadata.get('bf')}")
            print(f"Feature backend:     {feature_backend}")
            print(f"Loop closing:        {'enabled' if enable_loop_closing else 'disabled'}")
            print(f"Loop cand source:    {Parameters.kLoopCandidateSource}")
            print(f"Global BA:           {'enabled' if enable_global_ba else 'disabled'}")
            print(f"LM threading:        {'enabled (wait='+str(Parameters.kWaitForLocalMappingTimeout)+'s)' if threaded_lm else 'disabled (sequential)'}")
            print(f"Memory profile:      {'enabled' if profile_memory else 'disabled'} ({memory_profile_mode})")
            print(f"Runtime profile:     {'enabled' if profile_runtime else 'disabled'}")
            print(f"Local map profile:   {'enabled' if effective_profile_local_map else 'disabled'}")
            print(f"Keyframe profile:    {'enabled' if effective_profile_keyframes else 'disabled'}")
            print("=" * 80)

            start_t = time.perf_counter()
            num_ok = 0
            num_lost = 0
            num_errors = 0
            accepted_loop_count = 0
            stop_requested = False

            per_frame_log: list[dict] = []
            frame_timing_rows: list[dict] = []
            loop_debug_rows: list[dict] = []
            loop_candidate_oracle_rows: list[dict] = []
            loop_retrieval_profile_rows: list[dict] = []
            loop_candidate_source_comparison_rows: list[dict] = []
            loop_keyframe_density_rows: list[dict] = []
            loop_raw_dbow_trace_rows: list[dict] = []
            loop_inverted_word_trace_rows: list[dict] = []
            loop_score_filter_trace_rows: list[dict] = []
            loop_accumulation_trace_rows: list[dict] = []
            loop_retained_candidate_trace_rows: list[dict] = []
            loop_gt_positive_trace_rows: list[dict] = []
            loop_consistency_progression_rows: list[dict] = []
            loop_geometry_trace_rows: list[dict] = []
            memory_profile_rows: list[dict] = []
            peak_rss_mb = 0.0
            pair_report_dir = output_dir / "loop_candidate_pair_reports"
            runtime_profile_live_file = output_dir / "runtime_profile_live.csv"

            for i, entry in enumerate(frames):
                frame_idx = start_index + i
                frame_total_start = time.perf_counter()
                load_rgb_sec = 0.0
                load_depth_sec = 0.0
                slam_track_sec = 0.0
                local_mapping_sec = 0.0
                loop_closing_sec = 0.0
                prune_old_frame_views_sec = 0.0
                memory_profile_sec = 0.0
                frame_rss_mb = get_rss_mb()
                latest_stats = {
                    "num_keyframes": 0,
                    "num_map_points": 0,
                    "num_recent_frames": 0,
                    "num_frame_views_total": 0,
                    "old_frame_views_total": 0,
                }

                try:
                    load_start = time.perf_counter()
                    with profiler.section("frame.load_rgb"):
                        rgb = _load_rgb(entry.rgb_path)
                    load_rgb_sec = time.perf_counter() - load_start

                    load_start = time.perf_counter()
                    with profiler.section("frame.load_depth"):
                        depth = _load_depth(entry.depth_path)
                    load_depth_sec = time.perf_counter() - load_start

                    track_start = time.perf_counter()
                    with profiler.section("slam.track"):
                        ok = slam.track(
                            img=rgb,
                            img_right=None,
                            depth=depth,
                            img_id=frame_idx,
                            timestamp=entry.timestamp,
                        )
                    slam_track_sec = time.perf_counter() - track_start

                    if not threaded_lm:
                        while slam.local_mapping.queue_size() > 0:
                            lm_step_start = time.perf_counter()
                            with profiler.section("local_mapping.step"):
                                slam.local_mapping.step()
                            local_mapping_sec += time.perf_counter() - lm_step_start

                    loop_closing = getattr(slam, "loop_closing", None)
                    if threaded_lm:
                        lm_wait_start = time.perf_counter()
                        with profiler.section("local_mapping.step"):
                            slam.local_mapping.wait_idle(timeout=Parameters.kWaitForLocalMappingTimeout)
                        local_mapping_sec += time.perf_counter() - lm_wait_start

                    while loop_closing is not None and loop_closing.queue_size() > 0:
                        event_start = len(loop_debug_rows) + 1
                        loop_step_start = time.perf_counter()
                        with profiler.section("loop_closing.step"):
                            accepted_loop = bool(loop_closing.step())
                        loop_closing_sec += time.perf_counter() - loop_step_start
                        loop_diag_current = loop_closing.last_diagnostics
                        if loop_debug:
                            _append_loop_debug_records(loop_debug_rows, loop_diag_current)
                            _append_loop_candidate_oracle_rows(
                                loop_candidate_oracle_rows,
                                loop_diag_current,
                                event_start=event_start,
                            )
                            _append_loop_rows(
                                loop_retrieval_profile_rows,
                                getattr(loop_diag_current, "loop_retrieval_profile_rows", []),
                            )
                            _append_loop_rows(
                                loop_candidate_source_comparison_rows,
                                getattr(loop_diag_current, "loop_candidate_source_comparison_rows", []),
                            )
                            _append_loop_rows(
                                loop_keyframe_density_rows,
                                getattr(loop_diag_current, "loop_keyframe_density_rows", []),
                            )
                        if loop_retrieval_trace:
                            _append_loop_rows(
                                loop_raw_dbow_trace_rows,
                                getattr(loop_diag_current, "loop_raw_dbow_trace_rows", []),
                            )
                            _append_loop_rows(
                                loop_inverted_word_trace_rows,
                                getattr(loop_diag_current, "loop_inverted_word_trace_rows", []),
                            )
                            _append_loop_rows(
                                loop_score_filter_trace_rows,
                                getattr(loop_diag_current, "loop_score_filter_trace_rows", []),
                            )
                            _append_loop_rows(
                                loop_accumulation_trace_rows,
                                getattr(loop_diag_current, "loop_accumulation_trace_rows", []),
                            )
                            _append_loop_rows(
                                loop_retained_candidate_trace_rows,
                                getattr(loop_diag_current, "loop_retained_candidate_trace_rows", []),
                            )
                            _append_loop_rows(
                                loop_gt_positive_trace_rows,
                                getattr(loop_diag_current, "loop_gt_positive_trace_rows", []),
                            )
                            _append_loop_rows(
                                loop_consistency_progression_rows,
                                getattr(loop_diag_current, "loop_consistency_progression_rows", []),
                            )
                            _append_loop_rows(
                                loop_geometry_trace_rows,
                                getattr(loop_diag_current, "loop_geometry_trace_rows", []),
                            )
                        if (
                            dump_loop_candidate_reports
                            and not no_loop_candidate_pair_reports
                            and not no_heavy_loop_reports
                        ):
                            _write_candidate_pair_reports(
                                pair_report_dir,
                                loop_diag_current,
                                event_start=event_start,
                                dump_all=True,
                            )
                        if accepted_loop:
                            accepted_loop_count += 1
                        if stop_after_loop_events > 0 and len(loop_debug_rows) >= int(stop_after_loop_events):
                            stop_requested = True
                            break
                        if stop_after_accepted_loops > 0 and accepted_loop_count >= int(stop_after_accepted_loops):
                            stop_requested = True
                            break

                    if Parameters.kEnableFrameViewPruning and (i % Parameters.kFrameViewPruneEveryNFrames == 0):
                        prune_start = time.perf_counter()
                        with profiler.section("memory.prune_old_frame_views"):
                            slam.map.prune_old_frame_views(current_frame_id=frame_idx)
                        prune_old_frame_views_sec = time.perf_counter() - prune_start

                    if profile_memory and (i % memory_profile_every == 0 or i == len(frames) - 1):
                        mem_start = time.perf_counter()
                        with profiler.section("memory.profile_snapshot"):
                            frame_rss_mb = get_rss_mb()
                            peak_rss_mb = max(peak_rss_mb, frame_rss_mb)
                            if memory_limit_gb > 0 and (frame_rss_mb / 1024.0) > memory_limit_gb:
                                print(f"[STOP] Memory limit exceeded: {frame_rss_mb/1024.0:.2f} GB > {memory_limit_gb} GB")
                                stop_requested = True
                            latest_stats = slam.map.memory_stats(mode=memory_profile_mode)
                            lm_stats = slam.local_mapping.queue_memory_stats()
                            mem_row = {
                                "frame_idx": frame_idx,
                                "timestamp": entry.timestamp,
                                "rss_mb": frame_rss_mb,
                                "keyframes": latest_stats["num_keyframes"],
                                "map_points": latest_stats["num_map_points"],
                                "recent_frames": latest_stats["num_recent_frames"],
                                "num_frame_views_total": latest_stats["num_frame_views_total"],
                                "old_frame_views_total": latest_stats["old_frame_views_total"],
                                "keyframe_observations_total": latest_stats["num_keyframe_observations_total"],
                                "recent_frame_images": latest_stats["num_recent_frame_images"],
                                "recent_frame_depth_images": latest_stats["num_recent_frame_depth_images"],
                                "keyframe_images": latest_stats["num_keyframe_images"],
                                "keyframe_depth_images": latest_stats["num_keyframe_depth_images"],
                                "local_mapping_queue_size": lm_stats["queue_size"],
                                "estimated_heavy_mb": (latest_stats["estimated_total_heavy_bytes"] + lm_stats["estimated_queue_heavy_bytes"]) / (1024 * 1024),
                            }
                            memory_profile_rows.append(mem_row)
                        memory_profile_sec = time.perf_counter() - mem_start
                    else:
                        peak_rss_mb = max(peak_rss_mb, frame_rss_mb)

                    latest_stats = slam.map.memory_stats(mode="cheap")
                    frame_rss_mb = get_rss_mb()
                    peak_rss_mb = max(peak_rss_mb, frame_rss_mb)

                    state = slam.get_tracking_state()
                    if ok and state == SlamState.OK:
                        num_ok += 1
                    elif state == SlamState.LOST:
                        num_lost += 1

                    n_kf = slam.map.num_keyframes()
                    n_mp = slam.map.num_points()
                    n_frames = slam.map.num_frames()
                    n_pose = len(slam.tracking.poses)
                    n_hist = len(slam.tracking.tracking_history.timestamps)
                    mean_pose_opt_chi2_error = _finite_or_none(slam.tracking.mean_pose_opt_chi2_error)
                    loop_diag = getattr(getattr(slam, "loop_closing", None), "last_diagnostics", None)

                    frame_total_sec = time.perf_counter() - frame_total_start
                    profiler.record("frame.total", frame_total_sec)
                    with profiler.section("frame.log_write"):
                        row = {
                            "i": frame_idx,
                            "timestamp": entry.timestamp,
                            "ok": bool(ok),
                            "state": _state_name(state),
                            "keyframes": n_kf,
                            "points": n_mp,
                            "frames": n_frames,
                            "poses": n_pose,
                            "history": n_hist,
                            "last_tracked": slam.tracking.num_matched_map_points,
                            "last_ba_mse": mean_pose_opt_chi2_error,
                            "lm_last_fused": slam.local_mapping.last_num_fused_points,
                            "lm_last_triangulated": slam.local_mapping.last_num_triangulated_points,
                            "loop_global_ba_started": bool(getattr(loop_diag, "global_ba_started", False)),
                            "loop_global_ba_success": bool(getattr(loop_diag, "global_ba_success", False)),
                            "loop_global_ba_reason": getattr(loop_diag, "global_ba_reason", ""),
                            "loop_global_ba_edges": int(getattr(loop_diag, "global_ba_num_edges", 0)),
                            "loop_global_ba_inliers": int(getattr(loop_diag, "global_ba_num_inliers", 0)),
                            "loop_global_ba_mse_after": _finite_or_none(getattr(loop_diag, "global_ba_mean_error_after", None)),
                        }
                        per_frame_log.append(row)
                        frame_timing_rows.append(
                            {
                                "frame_idx": frame_idx,
                                "timestamp": entry.timestamp,
                                "frame_total_sec": frame_total_sec,
                                "load_rgb_sec": load_rgb_sec,
                                "load_depth_sec": load_depth_sec,
                                "slam_track_sec": slam_track_sec,
                                "local_mapping_sec": local_mapping_sec,
                                "loop_closing_sec": loop_closing_sec,
                                "prune_old_frame_views_sec": prune_old_frame_views_sec,
                                "memory_profile_sec": memory_profile_sec,
                                "rss_mb": frame_rss_mb,
                                "keyframes": latest_stats["num_keyframes"],
                                "map_points": latest_stats["num_map_points"],
                                "recent_frames": latest_stats["num_recent_frames"],
                                "num_frame_views_total": latest_stats["num_frame_views_total"],
                                "old_frame_views_total": latest_stats["old_frame_views_total"],
                            }
                        )

                    if profile_runtime and (i % runtime_profile_every == 0 or i == len(frames) - 1):
                        profiler.write_csv(runtime_profile_live_file)

                    if print_every > 0 and (i % print_every == 0 or i == len(frames) - 1):
                        print(
                            f"[{i+1:04d}/{len(frames):04d}] "
                            f"idx={frame_idx:05d} "
                            f"state={row['state']} ok={row['ok']} "
                            f"kf={n_kf} mp={n_mp} "
                            f"tracked={row['last_tracked']} "
                            f"ba_mse={row['last_ba_mse'] if row['last_ba_mse'] is not None else 'NA'} "
                            f"frame_sec={frame_total_sec:.3f}"
                        )

                except Exception as exc:
                    num_errors += 1
                    print(f"[ERROR] frame_idx={frame_idx} timestamp={entry.timestamp:.6f}: {type(exc).__name__}: {exc}")
                    raise

                if stop_requested:
                    print("[STOP] stop condition reached")
                    break

            elapsed = time.perf_counter() - start_t

            trajectory = slam.get_final_trajectory()
            ok_pairs = [
                (pose, ts)
                for pose, ts, state in zip(
                    trajectory["poses"],
                    trajectory["timestamps"],
                    trajectory["slam_states"],
                )
                if state == SlamState.OK
            ]
            poses = [pose for pose, _ in ok_pairs]
            timestamps = [stamp for _, stamp in ok_pairs]

            traj_file = output_dir / f"trajectory_{dataset_name}.txt"
            save_tum_trajectory(poses, timestamps, traj_file)

            frame_log_file = output_dir / f"frame_log_{dataset_name}.csv"
            write_csv(frame_log_file, per_frame_log, FRAME_LOG_COLUMNS)

            frame_timing_file = output_dir / "frame_timing.csv"
            write_csv(frame_timing_file, frame_timing_rows, FRAME_TIMING_COLUMNS)

            loop_debug_file = None
            if loop_debug and loop_debug_rows:
                loop_debug_file = output_dir / "loop_debug_candidates.csv"
                write_csv(loop_debug_file, loop_debug_rows, LOOP_DEBUG_COLUMNS)

            loop_candidate_oracle_file = None
            loop_retrieval_profile_file = None
            loop_candidate_source_comparison_file = None
            loop_keyframe_density_profile_file = None
            loop_raw_dbow_trace_file = None
            loop_inverted_word_trace_file = None
            loop_score_filter_trace_file = None
            loop_accumulation_trace_file = None
            loop_retained_candidate_trace_file = None
            loop_gt_positive_trace_file = None
            loop_consistency_progression_file = None
            loop_geometry_trace_file = None
            if loop_debug:
                loop_candidate_oracle_file = output_dir / "loop_candidate_oracle.csv"
                write_csv(loop_candidate_oracle_file, loop_candidate_oracle_rows, LOOP_CANDIDATE_ORACLE_COLUMNS)
                loop_retrieval_profile_file = output_dir / "loop_retrieval_profile.csv"
                write_csv(loop_retrieval_profile_file, loop_retrieval_profile_rows, LOOP_RETRIEVAL_PROFILE_COLUMNS)
                loop_candidate_source_comparison_file = output_dir / "loop_candidate_source_comparison.csv"
                write_csv(
                    loop_candidate_source_comparison_file,
                    loop_candidate_source_comparison_rows,
                    LOOP_SOURCE_COMPARISON_COLUMNS,
                )
                loop_keyframe_density_profile_file = output_dir / "loop_keyframe_density_profile.csv"
                write_csv(
                    loop_keyframe_density_profile_file,
                    loop_keyframe_density_rows,
                    LOOP_KEYFRAME_DENSITY_COLUMNS,
                )
            if loop_retrieval_trace:
                loop_raw_dbow_trace_file = output_dir / "loop_raw_dbow_trace.csv"
                write_csv(loop_raw_dbow_trace_file, loop_raw_dbow_trace_rows, LOOP_RAW_DBOW_TRACE_COLUMNS)
                loop_inverted_word_trace_file = output_dir / "loop_inverted_word_trace.csv"
                write_csv(loop_inverted_word_trace_file, loop_inverted_word_trace_rows, LOOP_INVERTED_WORD_TRACE_COLUMNS)
                loop_score_filter_trace_file = output_dir / "loop_score_filter_trace.csv"
                write_csv(loop_score_filter_trace_file, loop_score_filter_trace_rows, LOOP_SCORE_FILTER_TRACE_COLUMNS)
                loop_accumulation_trace_file = output_dir / "loop_accumulation_trace.csv"
                write_csv(loop_accumulation_trace_file, loop_accumulation_trace_rows, LOOP_ACCUMULATION_TRACE_COLUMNS)
                loop_retained_candidate_trace_file = output_dir / "loop_retained_candidate_trace.csv"
                write_csv(
                    loop_retained_candidate_trace_file,
                    loop_retained_candidate_trace_rows,
                    LOOP_RETAINED_CANDIDATE_TRACE_COLUMNS,
                )
                loop_gt_positive_trace_file = output_dir / "loop_gt_positive_trace.csv"
                write_csv(loop_gt_positive_trace_file, loop_gt_positive_trace_rows, LOOP_GT_POSITIVE_TRACE_COLUMNS)
                loop_consistency_progression_file = output_dir / "loop_consistency_progression.csv"
                write_csv(
                    loop_consistency_progression_file,
                    loop_consistency_progression_rows,
                    LOOP_CONSISTENCY_PROGRESSION_COLUMNS,
                )
                loop_geometry_trace_file = output_dir / "loop_geometry_trace.csv"
                write_csv(
                    loop_geometry_trace_file,
                    loop_geometry_trace_rows,
                    LOOP_GEOMETRY_TRACE_COLUMNS,
                )

            memory_profile_file = None
            if profile_memory:
                memory_profile_file = output_dir / "memory_profile.csv"
                write_csv(memory_profile_file, memory_profile_rows, MEMORY_PROFILE_COLUMNS)

            local_map_profile_file = None
            if effective_profile_local_map:
                local_map_profile_file = output_dir / "local_map_profile.csv"
                write_csv(
                    local_map_profile_file,
                    getattr(slam.tracking, "local_map_profile_rows", []),
                    LOCAL_MAP_PROFILE_COLUMNS,
                )

            keyframe_decision_log_file = None
            local_mapping_schedule_log_file = None
            if effective_profile_keyframes:
                keyframe_decision_log_file = output_dir / "keyframe_decision_log.csv"
                write_csv(
                    keyframe_decision_log_file,
                    getattr(slam.tracking, "keyframe_decision_rows", []),
                    KEYFRAME_DECISION_COLUMNS,
                )
                local_mapping_schedule_log_file = output_dir / "local_mapping_schedule_log.csv"
                write_csv(
                    local_mapping_schedule_log_file,
                    getattr(slam.local_mapping, "schedule_log_rows", []),
                    LOCAL_MAPPING_SCHEDULE_COLUMNS,
                )

            runtime_profile_csv = None
            runtime_profile_json = None
            if profile_runtime:
                runtime_profile_csv = output_dir / "runtime_profile.csv"
                runtime_profile_json = output_dir / "runtime_profile.json"
                profiler.write_csv(runtime_profile_csv)
                profiler.write_json(runtime_profile_json)

            map_export = {"map_points_ply": None, "keyframes_json": None, "keyframe_graph_json": None}
            if not no_map_export:
                map_export = export_orbslam_map(slam, output_dir)

            completed_timestamp = build_completed_timestamp()
            standardized_paths = build_standardized_artifact_paths(
                output_dir,
                effective_dataset_type,
                dataset_name,
                completed_timestamp,
            )
            standardized_output_stem = standardized_paths["stem"]

            kf_consistency = {"n_checked": 0}
            if hasattr(slam, "compute_kf_trajectory_consistency"):
                kf_consistency = slam.compute_kf_trajectory_consistency()

            run_config["completed_timestamp"] = completed_timestamp
            run_config["standardized_output_stem"] = standardized_output_stem
            effective_run_config_path = write_effective_run_config(output_dir, run_config)
            standardized_effective_config = _copy_if_exists(
                effective_run_config_path,
                standardized_paths["effective_run_config_json"],
            )

            standardized_output_files = {
                "trajectory_file": _copy_if_exists(traj_file, standardized_paths["trajectory_file"]),
                "frame_log_file": _copy_if_exists(frame_log_file, standardized_paths["frame_log_file"]),
                "frame_timing_file": _copy_if_exists(frame_timing_file, standardized_paths["frame_timing_file"]),
                "map_points_ply": _copy_if_exists(Path(map_export["map_points_ply"]) if map_export["map_points_ply"] else None, standardized_paths["map_points_ply"]),
                "keyframes_json": _copy_if_exists(Path(map_export["keyframes_json"]) if map_export["keyframes_json"] else None, standardized_paths["keyframes_json"]),
                "keyframe_graph_json": _copy_if_exists(Path(map_export["keyframe_graph_json"]) if map_export["keyframe_graph_json"] else None, standardized_paths["keyframe_graph_json"]),
                "effective_run_config_json": standardized_effective_config,
                "loop_debug_file": _copy_if_exists(loop_debug_file, standardized_paths["loop_debug_file"]) if loop_debug_file is not None else None,
                "loop_candidate_oracle_file": _copy_if_exists(loop_candidate_oracle_file, standardized_paths["loop_candidate_oracle_file"]) if loop_candidate_oracle_file is not None else None,
                "loop_retrieval_profile_file": _copy_if_exists(loop_retrieval_profile_file, standardized_paths["loop_retrieval_profile_file"]) if loop_retrieval_profile_file is not None else None,
                "loop_candidate_source_comparison_file": _copy_if_exists(loop_candidate_source_comparison_file, standardized_paths["loop_candidate_source_comparison_file"]) if loop_candidate_source_comparison_file is not None else None,
                "loop_keyframe_density_profile_file": _copy_if_exists(loop_keyframe_density_profile_file, standardized_paths["loop_keyframe_density_profile_file"]) if loop_keyframe_density_profile_file is not None else None,
                "loop_raw_dbow_trace_file": _copy_if_exists(loop_raw_dbow_trace_file, standardized_paths["loop_raw_dbow_trace_file"]) if loop_raw_dbow_trace_file is not None else None,
                "loop_inverted_word_trace_file": _copy_if_exists(loop_inverted_word_trace_file, standardized_paths["loop_inverted_word_trace_file"]) if loop_inverted_word_trace_file is not None else None,
                "loop_score_filter_trace_file": _copy_if_exists(loop_score_filter_trace_file, standardized_paths["loop_score_filter_trace_file"]) if loop_score_filter_trace_file is not None else None,
                "loop_accumulation_trace_file": _copy_if_exists(loop_accumulation_trace_file, standardized_paths["loop_accumulation_trace_file"]) if loop_accumulation_trace_file is not None else None,
                "loop_retained_candidate_trace_file": _copy_if_exists(loop_retained_candidate_trace_file, standardized_paths["loop_retained_candidate_trace_file"]) if loop_retained_candidate_trace_file is not None else None,
                "loop_gt_positive_trace_file": _copy_if_exists(loop_gt_positive_trace_file, standardized_paths["loop_gt_positive_trace_file"]) if loop_gt_positive_trace_file is not None else None,
                "loop_consistency_progression_file": _copy_if_exists(loop_consistency_progression_file, standardized_paths["loop_consistency_progression_file"]) if loop_consistency_progression_file is not None else None,
                "loop_geometry_trace_file": _copy_if_exists(loop_geometry_trace_file, standardized_paths["loop_geometry_trace_file"]) if loop_geometry_trace_file is not None else None,
                "memory_profile_file": _copy_if_exists(memory_profile_file, standardized_paths["memory_profile_file"]) if memory_profile_file is not None else None,
                "local_map_profile_file": _copy_if_exists(local_map_profile_file, standardized_paths["local_map_profile_file"]) if local_map_profile_file is not None else None,
                "keyframe_decision_log_file": _copy_if_exists(keyframe_decision_log_file, standardized_paths["keyframe_decision_log_file"]) if keyframe_decision_log_file is not None else None,
                "local_mapping_schedule_log_file": _copy_if_exists(local_mapping_schedule_log_file, standardized_paths["local_mapping_schedule_log_file"]) if local_mapping_schedule_log_file is not None else None,
                "runtime_profile_csv": _copy_if_exists(runtime_profile_csv, standardized_paths["runtime_profile_csv"]) if runtime_profile_csv is not None else None,
                "runtime_profile_json": _copy_if_exists(runtime_profile_json, standardized_paths["runtime_profile_json"]) if runtime_profile_json is not None else None,
            }

            final_stats_mode = "deep" if profile_memory and memory_profile_mode == "deep" else "cheap"
            final_m_stats = slam.map.memory_stats(mode=final_stats_mode)

            output_files = {
                "trajectory_file": str(traj_file),
                "frame_log_file": str(frame_log_file),
                "frame_timing_file": str(frame_timing_file),
                "map_points_ply": map_export["map_points_ply"],
                "keyframes_json": map_export["keyframes_json"],
                "keyframe_graph_json": map_export["keyframe_graph_json"],
                "effective_run_config_json": str(effective_run_config_path),
                "loop_debug_file": str(loop_debug_file) if loop_debug_file is not None else None,
                "loop_candidate_oracle_file": str(loop_candidate_oracle_file) if loop_candidate_oracle_file is not None else None,
                "loop_retrieval_profile_file": str(loop_retrieval_profile_file) if loop_retrieval_profile_file is not None else None,
                "loop_candidate_source_comparison_file": str(loop_candidate_source_comparison_file) if loop_candidate_source_comparison_file is not None else None,
                "loop_keyframe_density_profile_file": str(loop_keyframe_density_profile_file) if loop_keyframe_density_profile_file is not None else None,
                "loop_raw_dbow_trace_file": str(loop_raw_dbow_trace_file) if loop_raw_dbow_trace_file is not None else None,
                "loop_inverted_word_trace_file": str(loop_inverted_word_trace_file) if loop_inverted_word_trace_file is not None else None,
                "loop_score_filter_trace_file": str(loop_score_filter_trace_file) if loop_score_filter_trace_file is not None else None,
                "loop_accumulation_trace_file": str(loop_accumulation_trace_file) if loop_accumulation_trace_file is not None else None,
                "loop_retained_candidate_trace_file": str(loop_retained_candidate_trace_file) if loop_retained_candidate_trace_file is not None else None,
                "loop_gt_positive_trace_file": str(loop_gt_positive_trace_file) if loop_gt_positive_trace_file is not None else None,
                "loop_consistency_progression_file": str(loop_consistency_progression_file) if loop_consistency_progression_file is not None else None,
                "loop_geometry_trace_file": str(loop_geometry_trace_file) if loop_geometry_trace_file is not None else None,
                "memory_profile_file": str(memory_profile_file) if memory_profile_file is not None else None,
                "local_map_profile_file": str(local_map_profile_file) if local_map_profile_file is not None else None,
                "keyframe_decision_log_file": str(keyframe_decision_log_file) if keyframe_decision_log_file is not None else None,
                "local_mapping_schedule_log_file": str(local_mapping_schedule_log_file) if local_mapping_schedule_log_file is not None else None,
                "runtime_profile_csv": str(runtime_profile_csv) if runtime_profile_csv is not None else None,
                "runtime_profile_json": str(runtime_profile_json) if runtime_profile_json is not None else None,
                "standardized_trajectory_file": standardized_output_files["trajectory_file"],
                "standardized_frame_log_file": standardized_output_files["frame_log_file"],
                "standardized_frame_timing_file": standardized_output_files["frame_timing_file"],
                "standardized_map_points_ply": standardized_output_files["map_points_ply"],
                "standardized_keyframes_json": standardized_output_files["keyframes_json"],
                "standardized_keyframe_graph_json": standardized_output_files["keyframe_graph_json"],
                "standardized_effective_run_config_json": standardized_output_files["effective_run_config_json"],
                "standardized_loop_debug_file": standardized_output_files["loop_debug_file"],
                "standardized_loop_candidate_oracle_file": standardized_output_files["loop_candidate_oracle_file"],
                "standardized_loop_retrieval_profile_file": standardized_output_files["loop_retrieval_profile_file"],
                "standardized_loop_candidate_source_comparison_file": standardized_output_files["loop_candidate_source_comparison_file"],
                "standardized_loop_keyframe_density_profile_file": standardized_output_files["loop_keyframe_density_profile_file"],
                "standardized_local_map_profile_file": standardized_output_files["local_map_profile_file"],
                "standardized_keyframe_decision_log_file": standardized_output_files["keyframe_decision_log_file"],
                "standardized_local_mapping_schedule_log_file": standardized_output_files["local_mapping_schedule_log_file"],
                "peak_rss_mb": peak_rss_mb,
                "final_rss_mb": get_rss_mb(),
                "num_frame_views_total": final_m_stats["num_frame_views_total"],
                "old_frame_views_total": final_m_stats["old_frame_views_total"],
                "local_ba_started_count": getattr(slam.local_mapping, "local_ba_started_count", 0),
                "local_ba_completed_count": getattr(slam.local_mapping, "local_ba_completed_count", 0),
                "local_ba_aborted_count": getattr(slam.local_mapping, "local_ba_aborted_count", 0),
                "local_ba_skipped_due_queue_count": getattr(slam.local_mapping, "local_ba_skipped_due_queue_count", 0),
                "local_ba_forced_due_starvation_count": getattr(slam.local_mapping, "local_ba_forced_due_starvation_count", 0),
                "last_successful_local_ba_kid": getattr(slam.local_mapping, "last_successful_local_ba_kid", -1),
                "keyframes_since_last_successful_ba": getattr(slam.local_mapping, "keyframes_since_last_successful_ba", 0),
                "consecutive_local_ba_aborts": getattr(slam.local_mapping, "consecutive_local_ba_aborts", 0),
            }
            summary = build_run_summary(
                dataset_name=dataset_name,
                dataset_type=effective_dataset_type,
                frames_attempted=len(per_frame_log),
                tracking_ok_count=num_ok,
                tracking_lost_count=num_lost,
                errors=num_errors,
                final_state=_state_name(slam.get_tracking_state()),
                keyframes=slam.map.num_keyframes(),
                map_points=slam.map.num_points(),
                trajectory_poses=len(poses),
                elapsed_sec=elapsed,
                avg_fps=len(per_frame_log) / max(elapsed, 1e-9),
                feature_backend=feature_backend,
                enable_loop_closing=enable_loop_closing,
                enable_global_ba=enable_global_ba,
                global_ba_after_loop=global_ba_after_loop,
                loop_debug_events=len(loop_debug_rows),
                accepted_loops=accepted_loop_count,
                output_files=output_files,
                completed_timestamp=completed_timestamp,
                standardized_output_stem=standardized_output_stem,
            )
            summary_path = write_run_summary(output_dir, summary)
            standardized_summary_path = _copy_if_exists(summary_path, standardized_paths["run_summary_json"])
            summary["standardized_run_summary_json"] = standardized_summary_path
            summary_path = write_run_summary(output_dir, summary)

            print("=" * 80)
            print("RUN SUMMARY")
            print("=" * 80)
            print(f"frames_attempted:     {summary['frames_attempted']}")
            print(f"tracking_ok_count:    {summary['tracking_ok_count']}")
            print(f"tracking_lost_count:  {summary['tracking_lost_count']}")
            print(f"errors:               {summary['errors']}")
            print(f"final_state:          {summary['final_state']}")
            print(f"keyframes:            {summary['keyframes']}")
            print(f"map_points:           {summary['map_points']}")
            print(f"trajectory_poses:     {summary['trajectory_poses']}")
            print(f"elapsed_sec:          {summary['elapsed_sec']:.3f}")
            print(f"avg_fps:              {summary['avg_fps']:.2f}")
            print(f"trajectory_file:      {traj_file}")
            print(f"frame_log_file:       {frame_log_file}")
            print(f"frame_timing_file:    {frame_timing_file}")
            print(f"map_points_ply:       {map_export['map_points_ply']}")
            print(f"keyframes_json:       {map_export['keyframes_json']}")
            print(f"keyframe_graph_json:  {map_export['keyframe_graph_json']}")
            print(f"effective_config:     {effective_run_config_path}")
            print(f"completed_timestamp:  {completed_timestamp}")
            print(f"standardized_stem:    {standardized_output_stem}")
            print(f"standardized_summary: {standardized_summary_path}")
            if loop_debug_file is not None:
                print(f"loop_debug_file:      {loop_debug_file}")
                print(f"loop_debug_events:    {len(loop_debug_rows)}")
                print(f"accepted_loops:       {accepted_loop_count}")
            if loop_candidate_oracle_file is not None:
                print(f"loop_oracle_file:     {loop_candidate_oracle_file}")
            if loop_retrieval_profile_file is not None:
                print(f"loop_retrieval_file:  {loop_retrieval_profile_file}")
            if loop_candidate_source_comparison_file is not None:
                print(f"loop_compare_file:    {loop_candidate_source_comparison_file}")
            if loop_keyframe_density_profile_file is not None:
                print(f"loop_density_file:    {loop_keyframe_density_profile_file}")
            if loop_raw_dbow_trace_file is not None:
                print(f"loop_raw_dbow_trace:  {loop_raw_dbow_trace_file}")
            if loop_inverted_word_trace_file is not None:
                print(f"loop_inverted_trace:  {loop_inverted_word_trace_file}")
            if loop_score_filter_trace_file is not None:
                print(f"loop_score_trace:     {loop_score_filter_trace_file}")
            if loop_accumulation_trace_file is not None:
                print(f"loop_accum_trace:     {loop_accumulation_trace_file}")
            if loop_retained_candidate_trace_file is not None:
                print(f"loop_retained_trace:  {loop_retained_candidate_trace_file}")
            if loop_gt_positive_trace_file is not None:
                print(f"loop_gt_trace:        {loop_gt_positive_trace_file}")
            if loop_consistency_progression_file is not None:
                print(f"loop_consistency:     {loop_consistency_progression_file}")
            if loop_geometry_trace_file is not None:
                print(f"loop_geometry_trace:  {loop_geometry_trace_file}")
            if memory_profile_file is not None:
                print(f"memory_profile_file:  {memory_profile_file}")
            if local_map_profile_file is not None:
                print(f"local_map_profile:    {local_map_profile_file}")
            if keyframe_decision_log_file is not None:
                print(f"keyframe_log:         {keyframe_decision_log_file}")
            if local_mapping_schedule_log_file is not None:
                print(f"lm_schedule_log:      {local_mapping_schedule_log_file}")
            if runtime_profile_csv is not None:
                print(f"runtime_profile_csv:  {runtime_profile_csv}")
            if runtime_profile_json is not None:
                print(f"runtime_profile_json: {runtime_profile_json}")
            if kf_consistency.get("n_checked", 0) > 0:
                print(
                    f"kf_traj_consistency:  n={kf_consistency['n_checked']} "
                    f"max={kf_consistency['max_diff_m']:.4f}m "
                    f"median={kf_consistency['median_diff_m']:.4f}m"
                )
            print("=" * 80)

            return summary
        finally:
            if slam is not None:
                slam.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = create_arg_parser()
    args = parser.parse_args(argv)

    enable_loop_closing = bool(args.enable_loop_closing and not args.disable_loop_closing)
    enable_global_ba = bool(args.enable_global_ba and not args.disable_global_ba)

    run_rgbd_slam(
        dataset=args.dataset,
        output_dir=args.output,
        dataset_type=args.dataset_type,
        camera_profile=args.camera_profile,
        camera_config=args.camera_config,
        associations=args.associations,
        max_frames=args.max_frames,
        start_index=args.start_index,
        print_every=args.print_every,
        feature_backend=args.feature_backend,
        enable_loop_closing=enable_loop_closing,
        enable_global_ba=enable_global_ba,
        global_ba_after_loop=bool(args.global_ba_after_loop),
        global_ba_iterations=int(args.global_ba_iterations),
        loop_debug=bool(args.loop_debug),
        loop_retrieval_trace=bool(args.loop_retrieval_trace),
        loop_retrieval_trace_raw_k=int(args.loop_retrieval_trace_raw_k),
        stop_after_loop_events=int(args.stop_after_loop_events),
        stop_after_accepted_loops=int(args.stop_after_accepted_loops),
        dump_loop_candidate_reports=bool(args.dump_loop_candidate_reports),
        loop_candidate_source=str(args.loop_candidate_source),
        start_local_mapping_thread=bool(args.start_local_mapping_thread),
        lm_wait_timeout=float(args.lm_wait_timeout),
        profile_memory=bool(args.profile_memory),
        memory_profile_every=int(args.memory_profile_every),
        memory_profile_mode=str(args.memory_profile_mode),
        profile_runtime=bool(args.profile_runtime),
        runtime_profile_every=int(args.runtime_profile_every),
        profile_local_map=bool(args.profile_local_map),
        profile_keyframes=bool(args.profile_keyframes),
        memory_limit_gb=float(args.memory_limit_gb),
        frame_view_prune_every=int(args.frame_view_prune_every),
        lean_memory=bool(args.lean_memory),
        no_map_export=bool(args.no_map_export),
        no_heavy_loop_reports=bool(args.no_heavy_loop_reports),
        no_loop_candidate_pair_reports=bool(args.no_loop_candidate_pair_reports),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
