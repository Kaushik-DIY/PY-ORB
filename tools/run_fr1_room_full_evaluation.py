#!/usr/bin/env python3
"""Run Checkpoint 2.26A fr1_room evaluation and produce benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_tum_reference_cloud import build_reference_cloud
from tools.evaluate_tum_trajectory import evaluate_trajectories
from tools.export_orbslam_map import export_orbslam_map
from tools.plot_fr1_room_evaluation import ALIGNED_FIGURES, EXPECTED_FIGURES, generate_plots
from visual_slam.orbslam.io import load_tum_rgbd_associations, make_tum_rgbd_camera, save_tum_trajectory
from visual_slam.orbslam.slam import SensorType, Slam, SlamState


RUNS = {
    "A": ("run_A_no_loop", False, False),
    "B": ("run_B_loop_only", True, False),
    "C": ("run_C_loop_plus_gba", True, True),
}

LOOP_EVENT_COLUMNS = [
    "event_id",
    "timestamp",
    "current_kf_id",
    "candidate_kf_id",
    "candidate_score",
    "status",
    "reason",
    "num_bow_matches",
    "num_geom_inliers",
    "fused_points",
    "replaced_points",
    "added_observations",
    "essential_graph_success",
    "essential_graph_vertices",
    "essential_graph_edges",
]

GLOBAL_BA_EVENT_COLUMNS = [
    "event_id",
    "timestamp",
    "trigger_kf_id",
    "started",
    "success",
    "aborted",
    "reason",
    "num_keyframes",
    "num_map_points",
    "num_edges",
    "num_inliers",
    "num_outliers",
    "mean_error_before",
    "mean_error_after",
    "elapsed_sec",
]

FRAME_LOG_COLUMNS = [
    "frame_index",
    "timestamp",
    "ok",
    "state",
    "keyframes",
    "map_points",
    "trajectory_poses",
    "tracked_map_points",
    "pose_opt_chi2",
    "local_mapping_fused",
    "local_mapping_triangulated",
    "loop_candidates_total",
    "accepted_loops_total",
    "global_ba_started_total",
]

LOOP_DEBUG_COLUMNS = [
    "event_id",
    "frame_id",
    "current_kf_id",
    "candidate_kf_id",
    "current_timestamp",
    "candidate_timestamp",
    "candidate_score",
    "candidate_rank",
    "candidate_source",
    "temporal_separation_kf",
    "temporal_separation_frames",
    "current_group_kf_ids",
    "candidate_group_kf_ids",
    "previous_consistency_group_ids",
    "consistency_overlap_count",
    "consistency_count",
    "consistency_required",
    "passed_consistency",
    "common_words",
    "bow_score_raw",
    "bow_score_normalized",
    "bow_matches_raw",
    "bow_matches_after_ratio",
    "bow_matches_after_orientation",
    "bow_matches_with_valid_mappoints",
    "geometry_method",
    "geometry_input_correspondences",
    "geometry_ransac_inliers",
    "geometry_refined_inliers",
    "geometry_reprojection_rmse",
    "estimated_pose_distance",
    "estimated_pose_distance_threshold",
    "estimated_pose_rotation_deg",
    "estimated_pose_rotation_threshold_deg",
    "guided_projection_matches",
    "guided_projection_total_matches",
    "final_inliers",
    "accept_threshold_inliers",
    "rejection_stage",
    "rejection_reason",
]


def create_output_structure(output: str | Path) -> dict[str, Path]:
    output = Path(output).expanduser().resolve()
    paths = {
        "root": output,
        "run_A_no_loop": output / "run_A_no_loop",
        "run_B_loop_only": output / "run_B_loop_only",
        "run_C_loop_plus_gba": output / "run_C_loop_plus_gba",
        "reference_map": output / "reference_map",
        "comparison": output / "comparison",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(path: str | Path, rows: list[dict], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _csv_value(row.get(col, "")) for col in columns})


def _csv_value(value):
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def create_empty_event_logs(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    write_csv(output_dir / "loop_events.csv", [], LOOP_EVENT_COLUMNS)
    write_csv(output_dir / "global_ba_events.csv", [], GLOBAL_BA_EVENT_COLUMNS)
    write_csv(output_dir / "loop_debug_candidates.csv", [], LOOP_DEBUG_COLUMNS)


def has_real_loop_triggered_gba(summary: dict) -> bool:
    return int(summary.get("accepted_loops", 0) or 0) > 0 and int(summary.get("global_ba_started", 0) or 0) > 0


def _load_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load RGB image: {path}")
    return img


def _load_depth(path: Path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not load depth image: {path}")
    return depth


def _state_name(state) -> str:
    return getattr(state, "name", str(state))


def _finite_or_empty(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return value if np.isfinite(value) else ""


def _event_reason(diagnostics, accepted: bool) -> str:
    if getattr(diagnostics, "unavailable_reason", None):
        return str(diagnostics.unavailable_reason)
    if accepted:
        return "accepted"
    if getattr(diagnostics, "rejected_by_consistency", 0):
        return "rejected_by_consistency"
    if getattr(diagnostics, "rejected_by_geometry", 0):
        checker_error = ""
        return checker_error or "rejected_by_geometry"
    if getattr(diagnostics, "rejected_by_bow", 0):
        return "rejected_by_bow"
    return "no_candidate"


def _loop_event_from_diagnostics(event_id: int, timestamp: float, keyframe, loop_closing, accepted: bool) -> dict:
    diagnostics = loop_closing.last_diagnostics
    detector_output = getattr(loop_closing.loop_detector, "last_output", None)
    geom = getattr(loop_closing, "loop_geometry_checker", None)
    opt = getattr(diagnostics, "optimization_result", None)
    fusion = getattr(diagnostics, "fusion_diagnostics", None)
    candidate_id = ""
    candidate_score = ""
    if detector_output is not None and getattr(detector_output, "candidate_keyframes", None):
        candidate = detector_output.candidate_keyframes[0]
        candidate_id = int(getattr(candidate, "kid", getattr(candidate, "id", -1)))
        scores = getattr(detector_output, "candidate_scores", [])
        candidate_score = _finite_or_empty(scores[0]) if scores else ""
    success_loop_kf = getattr(geom, "success_loop_kf", None)
    if success_loop_kf is not None:
        candidate_id = int(getattr(success_loop_kf, "kid", getattr(success_loop_kf, "id", -1)))
    return {
        "event_id": event_id,
        "timestamp": f"{float(timestamp):.6f}",
        "current_kf_id": int(getattr(keyframe, "kid", getattr(keyframe, "id", -1))),
        "candidate_kf_id": candidate_id,
        "candidate_score": candidate_score,
        "status": "accepted" if accepted else "rejected",
        "reason": _event_reason(diagnostics, accepted),
        "num_bow_matches": int(getattr(geom, "num_last_matches", 0) or 0),
        "num_geom_inliers": int(getattr(geom, "num_last_inliers", 0) or 0),
        "fused_points": int(getattr(diagnostics, "fused_points", 0) or 0),
        "replaced_points": int(getattr(fusion, "replaced_points", 0) or 0),
        "added_observations": int(getattr(fusion, "added_observations", 0) or 0),
        "essential_graph_success": bool(getattr(opt, "success", False)) if opt is not None else False,
        "essential_graph_vertices": int(getattr(opt, "corrected_keyframes", 0) or 0) if opt is not None else "",
        "essential_graph_edges": int(getattr(opt, "num_edges", 0) or 0) if opt is not None else "",
    }


def _gba_event_from_diagnostics(event_id: int, timestamp: float, keyframe, diagnostics) -> dict:
    return {
        "event_id": event_id,
        "timestamp": f"{float(timestamp):.6f}",
        "trigger_kf_id": int(getattr(keyframe, "kid", getattr(keyframe, "id", -1))),
        "started": bool(getattr(diagnostics, "global_ba_started", False)),
        "success": bool(getattr(diagnostics, "global_ba_success", False)),
        "aborted": bool(getattr(diagnostics, "global_ba_aborted", False)),
        "reason": getattr(diagnostics, "global_ba_reason", ""),
        "num_keyframes": int(getattr(diagnostics, "global_ba_num_keyframes", 0) or 0),
        "num_map_points": int(getattr(diagnostics, "global_ba_num_map_points", 0) or 0),
        "num_edges": int(getattr(diagnostics, "global_ba_num_edges", 0) or 0),
        "num_inliers": int(getattr(diagnostics, "global_ba_num_inliers", 0) or 0),
        "num_outliers": int(getattr(diagnostics, "global_ba_num_outliers", 0) or 0),
        "mean_error_before": _finite_or_empty(getattr(diagnostics, "global_ba_mean_error_before", None)),
        "mean_error_after": _finite_or_empty(getattr(diagnostics, "global_ba_mean_error_after", None)),
        "elapsed_sec": _finite_or_empty(getattr(diagnostics, "global_ba_elapsed_sec", 0.0)),
    }


def _write_candidate_pair_reports(
    report_dir: Path,
    diagnostics,
    *,
    event_start: int,
    dump_all: bool = False,
) -> None:
    reports = list(getattr(diagnostics, "candidate_pair_reports", []) or [])
    debug_records = list(getattr(diagnostics, "loop_debug_records", []) or [])
    if not reports:
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    record_by_candidate = {
        int(record.get("candidate_kf_id", -1)): record
        for record in debug_records
        if record.get("candidate_kf_id", "") != ""
    }
    for offset, report in enumerate(reports):
        candidate_id = int(report.get("candidate_kf_id", -1))
        record = record_by_candidate.get(candidate_id, {})
        if not dump_all and not bool(record.get("passed_consistency", False)):
            continue
        current_id = int(report.get("current_kf_id", -1))
        event_id = int(record.get("event_id", event_start + offset))
        path = report_dir / f"candidate_{event_id}_kf_{current_id}_{candidate_id}.json"
        payload = dict(report)
        payload["debug_record"] = record
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _append_loop_debug_records(
    loop_debug_rows: list[dict],
    diagnostics,
) -> int:
    count = 0
    for record in list(getattr(diagnostics, "loop_debug_records", []) or []):
        row = dict(record)
        row["event_id"] = len(loop_debug_rows) + 1
        loop_debug_rows.append(row)
        count += 1
    return count


def _process_loop_queue(
    loop_closing,
    loop_events: list[dict],
    gba_events: list[dict],
    loop_debug_rows: list[dict] | None = None,
    pair_report_dir: Path | None = None,
    *,
    dump_pair_reports: bool = False,
) -> None:
    while loop_closing is not None and loop_closing.queue_size() > 0:
        keyframe = loop_closing.pop_keyframe()
        if keyframe is None:
            break
        accepted = loop_closing.process_keyframe(keyframe)
        diagnostics = loop_closing.last_diagnostics
        event_start = len(loop_debug_rows or []) + 1
        if loop_debug_rows is not None:
            _append_loop_debug_records(loop_debug_rows, diagnostics)
        if pair_report_dir is not None:
            _write_candidate_pair_reports(
                pair_report_dir,
                diagnostics,
                event_start=event_start,
                dump_all=bool(dump_pair_reports),
            )
        if int(getattr(diagnostics, "candidates", 0) or 0) > 0 or accepted or getattr(diagnostics, "unavailable_reason", None):
            loop_events.append(
                _loop_event_from_diagnostics(
                    len(loop_events) + 1,
                    getattr(keyframe, "timestamp", 0.0) or 0.0,
                    keyframe,
                    loop_closing,
                    bool(accepted),
                )
            )
        if bool(getattr(diagnostics, "global_ba_started", False)) or bool(getattr(diagnostics, "global_ba_success", False)):
            gba_events.append(
                _gba_event_from_diagnostics(
                    len(gba_events) + 1,
                    getattr(keyframe, "timestamp", 0.0) or 0.0,
                    keyframe,
                    diagnostics,
                )
            )


def run_single_evaluation(
    dataset: str | Path,
    output_dir: str | Path,
    *,
    backend: str = "pyslam_orb2",
    max_frames: int = 0,
    loop_closing: bool = False,
    global_ba: bool = False,
    print_every: int = 50,
    loop_debug: bool = False,
    dump_loop_candidate_reports: bool = False,
    stop_after_loop_events: int = 0,
    stop_after_accepted_loops: int = 0,
) -> dict:
    dataset = Path(dataset).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_tum_rgbd_associations(dataset)
    if max_frames and max_frames > 0:
        frames = frames[: int(max_frames)]
    camera = make_tum_rgbd_camera(dataset.name)
    slam = Slam(
        camera=camera,
        sensor_type=SensorType.RGBD,
        headless=True,
        start_local_mapping_thread=False,
        feature_tracker_config={"extractor_backend": backend},
        enable_loop_closing=bool(loop_closing),
        enable_global_ba=bool(global_ba),
        global_ba_after_loop=bool(global_ba),
    )

    frame_rows: list[dict] = []
    loop_events: list[dict] = []
    gba_events: list[dict] = []
    loop_debug_rows: list[dict] = []
    pair_report_dir = output_dir / "loop_candidate_pair_reports"
    num_ok = 0
    num_lost = 0
    num_errors = 0
    start_t = time.time()
    with open(output_dir / "run.log", "w") as run_log:
        run_log.write(f"dataset={dataset}\nbackend={backend}\nloop_closing={loop_closing}\nglobal_ba={global_ba}\n")
        for local_idx, entry in enumerate(frames):
            try:
                rgb = _load_rgb(entry.rgb_path)
                depth = _load_depth(entry.depth_path)
                ok = slam.track(rgb, img_right=None, depth=depth, img_id=local_idx, timestamp=entry.timestamp)
                while slam.local_mapping.queue_size() > 0:
                    slam.local_mapping.step()
                _process_loop_queue(
                    getattr(slam, "loop_closing", None),
                    loop_events,
                    gba_events,
                    loop_debug_rows if loop_debug else None,
                    pair_report_dir if dump_loop_candidate_reports else None,
                    dump_pair_reports=bool(dump_loop_candidate_reports),
                )
                state = slam.get_tracking_state()
                if ok and state == SlamState.OK:
                    num_ok += 1
                elif state == SlamState.LOST:
                    num_lost += 1
                frame_rows.append(
                    {
                        "frame_index": local_idx,
                        "timestamp": f"{float(entry.timestamp):.6f}",
                        "ok": int(bool(ok)),
                        "state": _state_name(state),
                        "keyframes": slam.map.num_keyframes(),
                        "map_points": slam.map.num_points(),
                        "trajectory_poses": len(slam.tracking.poses),
                        "tracked_map_points": int(getattr(slam.tracking, "num_matched_map_points", 0) or 0),
                        "pose_opt_chi2": _finite_or_empty(getattr(slam.tracking, "mean_pose_opt_chi2_error", None)),
                        "local_mapping_fused": int(getattr(slam.local_mapping, "last_num_fused_points", 0) or 0),
                        "local_mapping_triangulated": int(getattr(slam.local_mapping, "last_num_triangulated_points", 0) or 0),
                        "loop_candidates_total": sum(int(e.get("status") in ("accepted", "rejected")) for e in loop_events),
                        "accepted_loops_total": sum(int(e.get("status") == "accepted") for e in loop_events),
                        "global_ba_started_total": sum(int(bool(e.get("started"))) for e in gba_events),
                    }
                )
                if print_every > 0 and (local_idx % print_every == 0 or local_idx == len(frames) - 1):
                    line = (
                        f"[{local_idx + 1:04d}/{len(frames):04d}] state={_state_name(state)} "
                        f"kf={slam.map.num_keyframes()} mp={slam.map.num_points()} "
                        f"loops={sum(int(e.get('status') == 'accepted') for e in loop_events)} "
                        f"gba={sum(int(bool(e.get('success'))) for e in gba_events)}\n"
                    )
                    print(line.rstrip())
                    run_log.write(line)
                    run_log.flush()
                if stop_after_loop_events > 0 and len(loop_debug_rows) >= int(stop_after_loop_events):
                    run_log.write(f"STOP stop_after_loop_events={stop_after_loop_events}\n")
                    break
                if stop_after_accepted_loops > 0 and sum(int(e.get("status") == "accepted") for e in loop_events) >= int(stop_after_accepted_loops):
                    run_log.write(f"STOP stop_after_accepted_loops={stop_after_accepted_loops}\n")
                    break
            except Exception as exc:
                num_errors += 1
                run_log.write(f"ERROR frame={local_idx} timestamp={entry.timestamp}: {type(exc).__name__}: {exc}\n")
                raise

    elapsed = time.time() - start_t
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
    timestamps = [ts for _, ts in ok_pairs]
    trajectory_file = output_dir / f"trajectory_{dataset.name}_smoke.txt"
    if poses:
        save_tum_trajectory(poses, timestamps, trajectory_file)
    write_csv(output_dir / "frame_log.csv", frame_rows, FRAME_LOG_COLUMNS)
    write_csv(output_dir / "loop_events.csv", loop_events, LOOP_EVENT_COLUMNS)
    write_csv(output_dir / "loop_debug_candidates.csv", loop_debug_rows, LOOP_DEBUG_COLUMNS)
    write_csv(output_dir / "global_ba_events.csv", gba_events, GLOBAL_BA_EVENT_COLUMNS)
    map_summary = export_orbslam_map(slam, output_dir)
    metrics = {}
    if trajectory_file.exists():
        try:
            metrics = evaluate_trajectories(dataset / "groundtruth.txt", trajectory_file, output_dir / "trajectory_eval")
        except Exception as exc:
            metrics = {"error": f"{type(exc).__name__}: {exc}"}
            (output_dir / "trajectory_eval").mkdir(parents=True, exist_ok=True)
            (output_dir / "trajectory_eval" / "trajectory_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    summary = {
        "dataset": str(dataset),
        "backend": backend,
        "loop_closing_enabled": bool(loop_closing),
        "global_ba_enabled": bool(global_ba),
        "frames_attempted": len(frames),
        "tracking_ok_count": int(num_ok),
        "tracking_lost_count": int(num_lost),
        "errors": int(num_errors),
        "final_state": _state_name(slam.get_tracking_state()),
        "final_keyframes": int(slam.map.num_keyframes()),
        "final_map_points": int(slam.map.num_points()),
        "trajectory_poses": int(len(poses)),
        "elapsed_sec": float(elapsed),
        "avg_fps": float(len(frames) / max(elapsed, 1e-9)),
        "loop_candidates": int(len(loop_events)),
        "accepted_loops": int(sum(int(e.get("status") == "accepted") for e in loop_events)),
        "rejected_loops": int(sum(int(e.get("status") == "rejected") for e in loop_events)),
        "loop_fused_points": int(sum(int(e.get("fused_points") or 0) for e in loop_events)),
        "loop_replaced_points": int(sum(int(e.get("replaced_points") or 0) for e in loop_events)),
        "loop_added_observations": int(sum(int(e.get("added_observations") or 0) for e in loop_events)),
        "essential_graph_runs": int(sum(int(str(e.get("essential_graph_success")).lower() == "true" or e.get("essential_graph_success") is True) for e in loop_events)),
        "global_ba_started": int(sum(int(bool(e.get("started"))) for e in gba_events)),
        "global_ba_success": int(sum(int(bool(e.get("success"))) for e in gba_events)),
        "global_ba_failed": int(sum(int(bool(e.get("started")) and not bool(e.get("success")) and not bool(e.get("aborted"))) for e in gba_events)),
        "global_ba_aborted": int(sum(int(bool(e.get("aborted"))) for e in gba_events)),
        "trajectory_file": str(trajectory_file) if trajectory_file.exists() else "",
        "trajectory_metrics": metrics,
        "map_export": map_summary,
        "loop_edges": int(map_summary.get("num_loop_edges", 0) or 0),
        "kf_trajectory_consistency": slam.compute_kf_trajectory_consistency(),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _brief(summary: dict) -> str:
    metrics = summary.get("trajectory_metrics", {})
    ate = metrics.get("ate_rmse_se3_m", "")
    rpe = metrics.get("rpe_trans_rmse_m", "")
    return (
        f"frames={summary.get('frames_attempted')} ok={summary.get('tracking_ok_count')} "
        f"lost={summary.get('tracking_lost_count')} state={summary.get('final_state')} "
        f"loops={summary.get('accepted_loops')} gba={summary.get('global_ba_success')} "
        f"ATE={ate} RPE_t={rpe}"
    )


def write_final_report(root: Path, dry_summaries: dict[str, dict], full_summaries: dict[str, dict], reference_summary: dict | None) -> Path:
    comparison = root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    c_summary = full_summaries.get("C") or dry_summaries.get("C") or {}
    exercised = has_real_loop_triggered_gba(c_summary)
    lines = [
        "# FR1 Room Full Evaluation Report",
        "",
        "## Runs",
        "",
        "### 100-frame dry runs",
    ]
    for key in ["A", "B", "C"]:
        if key in dry_summaries:
            lines.append(f"- Run {key}: `{_brief(dry_summaries[key])}`")
    lines += ["", "### Full runs"]
    for key in ["A", "B", "C"]:
        if key in full_summaries:
            lines.append(f"- Run {key}: `{_brief(full_summaries[key])}`")
    if "A" not in full_summaries or "B" not in full_summaries:
        lines.append("- Full A/B ablation: deferred.")
    lines += [
        "",
        "## Loop and Global BA",
        "",
        f"- Accepted loops: `{c_summary.get('accepted_loops', 0)}`",
        f"- Essential graph runs: `{c_summary.get('essential_graph_runs', 0)}`",
        f"- Global BA started/success: `{c_summary.get('global_ba_started', 0)}` / `{c_summary.get('global_ba_success', 0)}`",
    ]
    if not exercised:
        lines.append("- The run was stable, but real loop-triggered Global BA was not exercised.")
    else:
        lines.append("- Real loop-triggered Global BA was exercised.")
    lines += ["", "## Reference Cloud", ""]
    if reference_summary:
        lines.append(f"- File: `{reference_summary.get('output')}`")
        lines.append(f"- Points: `{reference_summary.get('points')}`")
        lines.append(f"- Frames used: `{reference_summary.get('frames_used')}`")
    else:
        lines.append("- Reference cloud was not generated in this invocation.")
    lines += ["", "## Figures", ""]
    for name in EXPECTED_FIGURES:
        lines.append(f"- `{comparison / name}`")
    for name in ALIGNED_FIGURES:
        lines.append(f"- `{comparison / name}`")
    report_path = comparison / "FR1_ROOM_FULL_EVALUATION_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")
    debug_report_path = comparison / "FR1_ROOM_LOOP_CLOSURE_DEBUG_REPORT.md"
    debug_report_path.write_text("\n".join(lines).replace("FR1 Room Full Evaluation Report", "FR1 Room Loop Closure Debug Report") + "\n")
    return report_path


def write_audit(root: Path, dry_summaries: dict[str, dict], full_summaries: dict[str, dict], reference_summary: dict | None, tests_summary: str = "") -> Path:
    audit_dir = Path("visual_slam/reference_audit/checkpoint_2_26A")
    audit_dir.mkdir(parents=True, exist_ok=True)
    c_summary = full_summaries.get("C") or {}
    lines = [
        "# Checkpoint 2.26A FR1 Room Full Evaluation Audit",
        "",
        "## 1. Purpose",
        "Validate the existing RGB-D ORB2 pipeline on full `fr1_room` and generate benchmark logs, sparse map export, a GT-reference point cloud, and thesis-ready plots.",
        "",
        "## 2. Dataset Used",
        f"`{c_summary.get('dataset', '')}`",
        "",
        "## 3. Backend Used",
        "`pyslam_orb2`",
        "",
        "## 4. Tests Run",
        tests_summary or "See terminal report.",
        "",
        "## 5. Dry-run A/B/C Summary",
    ]
    for key in ["A", "B", "C"]:
        lines.append(f"- Run {key}: `{_brief(dry_summaries.get(key, {}))}`")
    lines += ["", "## 6. Full Run C Summary", f"`{_brief(c_summary)}`", ""]
    lines += [
        "## 7. Loop Event Summary",
        f"- Candidates/events: `{c_summary.get('loop_candidates', 0)}`",
        f"- Accepted loops: `{c_summary.get('accepted_loops', 0)}`",
        f"- Rejected loops: `{c_summary.get('rejected_loops', 0)}`",
        "",
        "## 8. Global BA Event Summary",
        f"- Started: `{c_summary.get('global_ba_started', 0)}`",
        f"- Success: `{c_summary.get('global_ba_success', 0)}`",
        f"- Failed: `{c_summary.get('global_ba_failed', 0)}`",
        f"- Aborted: `{c_summary.get('global_ba_aborted', 0)}`",
        "",
        "## 9. Trajectory Metrics",
        f"`{c_summary.get('trajectory_metrics', {})}`",
        "",
        "## 10. Map Export Summary",
        f"`{c_summary.get('map_export', {})}`",
        "",
        "## 11. Reference Cloud Generation Summary",
        f"`{reference_summary or {}}`",
        "",
        "## 12. Visualization Outputs",
    ]
    for name in EXPECTED_FIGURES:
        lines.append(f"- `{root / 'comparison' / name}`")
    for name in ALIGNED_FIGURES:
        lines.append(f"- `{root / 'comparison' / name}`")
    lines += [
        "",
        "## 13. Real Loop+GBA Exercised",
        "Yes." if has_real_loop_triggered_gba(c_summary) else "No. The run was stable, but real loop-triggered Global BA was not exercised.",
        "",
        "## 14. No Accepted Loop Diagnostic",
        "Inspect `loop_events.csv` candidate/rejection reasons before changing any thresholds." if int(c_summary.get("accepted_loops", 0) or 0) == 0 else "Accepted loop closure was observed.",
        "",
        "## 15. Remaining Gaps",
        "Full A/B ablation remains deferred unless explicitly run." if ("A" not in full_summaries or "B" not in full_summaries) else "None for the requested full A/B/C execution.",
        "",
        "## 16. Next Recommended Action",
        "Review loop-event diagnostics and thesis figures; run full A/B ablation if a complete comparison table is required.",
    ]
    path = audit_dir / "FR1_ROOM_FULL_EVALUATION_AUDIT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_workflow(args) -> dict:
    if args.backend != "pyslam_orb2":
        raise ValueError("Checkpoint 2.26A full runs must use backend=pyslam_orb2")
    dataset = Path(args.dataset).expanduser().resolve()
    required = [dataset, dataset / "groundtruth.txt", dataset / "rgb", dataset / "depth"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required dataset path(s): {missing}")
    paths = create_output_structure(args.output)
    dry_summaries = {}
    for key in ["A", "B", "C"]:
        run_dir_name, loop, gba = RUNS[key]
        dry_dir = paths[run_dir_name] / f"dry_run_{int(args.dry_run_frames)}"
        summary_path = dry_dir / "run_summary.json"
        if summary_path.exists() and not args.force_dry_runs:
            print(f"Dry Run {key}: reusing {summary_path}")
            dry_summaries[key] = json.loads(summary_path.read_text())
        else:
            print(f"Dry Run {key}: {dry_dir}")
            dry_summaries[key] = run_single_evaluation(
                dataset,
                dry_dir,
                backend=args.backend,
                max_frames=int(args.dry_run_frames),
                loop_closing=loop,
                global_ba=gba,
                print_every=max(1, int(args.print_every)),
                loop_debug=bool(args.loop_debug),
                dump_loop_candidate_reports=bool(args.dump_loop_candidate_reports),
                stop_after_loop_events=int(args.stop_after_loop_events),
                stop_after_accepted_loops=int(args.stop_after_accepted_loops),
            )
    full_summaries = {}
    if not args.dry_run_only:
        full_keys = ["A", "B", "C"] if args.run_full_ablation else (["C"] if args.run_full_c or not args.run_full_ablation else [])
        for key in full_keys:
            run_dir_name, loop, gba = RUNS[key]
            print(f"Full Run {key}: {paths[run_dir_name]}")
            full_summaries[key] = run_single_evaluation(
                dataset,
                paths[run_dir_name],
                backend=args.backend,
                max_frames=0,
                loop_closing=loop,
                global_ba=gba,
                print_every=max(1, int(args.print_every)),
                loop_debug=bool(args.loop_debug),
                dump_loop_candidate_reports=bool(args.dump_loop_candidate_reports),
                stop_after_loop_events=int(args.stop_after_loop_events),
                stop_after_accepted_loops=int(args.stop_after_accepted_loops),
            )
    reference_summary = None
    if args.build_reference_cloud:
        reference_summary = build_reference_cloud(dataset, paths["reference_map"] / "reference_cloud_gt.ply")
        (paths["reference_map"] / "reference_cloud_summary.json").write_text(json.dumps(reference_summary, indent=2) + "\n")
    figures = []
    if args.plot:
        figures = generate_plots(paths["root"], dataset=dataset, output=paths["comparison"])
    report_path = write_final_report(paths["root"], dry_summaries, full_summaries, reference_summary)
    audit_path = write_audit(paths["root"], dry_summaries, full_summaries, reference_summary)
    return {
        "dry_summaries": dry_summaries,
        "full_summaries": full_summaries,
        "reference_summary": reference_summary,
        "figures": [str(path) for path in figures],
        "report": str(report_path),
        "audit": str(audit_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="pyslam_orb2", choices=["pyslam_orb2"])
    parser.add_argument("--dry-run-frames", type=int, default=100)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--run-full-c", action="store_true")
    parser.add_argument("--run-full-ablation", action="store_true")
    parser.add_argument("--build-reference-cloud", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--force-dry-runs", action="store_true")
    parser.add_argument("--loop-debug", action="store_true")
    parser.add_argument("--dump-loop-candidate-reports", action="store_true")
    parser.add_argument("--stop-after-loop-events", type=int, default=0)
    parser.add_argument("--stop-after-accepted-loops", type=int, default=0)
    args = parser.parse_args(argv)
    result = run_workflow(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
