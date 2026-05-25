"""
Local-mapping worker and queue manager.
This module receives new keyframes, runs local-map updates, and coordinates optional threading.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from queue import Queue
from threading import Condition, RLock
import time
import traceback

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.geometry_matchers import EpipolarMatcher
from visual_slam.orbslam.slam.local_mapping_core import LocalMappingCore
from visual_slam.orbslam.slam.sensor_types import SensorType
from visual_slam.orbslam.utilities.geom_triangulation import triangulate_normalized_points
from visual_slam.orbslam.utilities.logging import Printer

import threading as _threading

try:
    import cpp_slam_core as _cpp_slam_core
    _CppLocalMappingCore = getattr(_cpp_slam_core, "LocalMappingCore", None)
    _CPP_LMC_AVAILABLE = _CppLocalMappingCore is not None
except ImportError:
    _CppLocalMappingCore = None
    _CPP_LMC_AVAILABLE = False

kVerbose = True
kUseLargeWindowBA = Parameters.kUseLargeWindowBA
kLocalMappingSleepTime = 5e-3


# Coordinate keyframe insertion and execution of local-mapping work.
class LocalMapping:
    print = staticmethod(lambda *args, **kwargs: None)

    def __init__(self, slam):
        self.slam = slam
        if _CPP_LMC_AVAILABLE:
            self.local_mapping_core = _CppLocalMappingCore(slam.map, slam.sensor_type.value)
            self._use_cpp_lmc = True
        else:
            self.local_mapping_core = LocalMappingCore(slam.map, slam.sensor_type)
            self._use_cpp_lmc = False

        self.queue = Queue()
        self.queue_condition = Condition()
        self.idle_condition = Condition()
        self.stop_mutex = RLock()
        self.reset_mutex = RLock()

        self.is_running = False
        self._is_idle = True
        self._accept_keyframes = True
        self.stop_requested = False
        self.do_not_stop = False
        self.stopped = False
        self.reset_requested = False

        self.depth_cur = None
        self.img_cur_right = None
        self.img_cur = None

        self.mean_ba_chi2_error = None
        self.time_local_mapping = None

        self.far_points_threshold = None
        self.use_fov_centers_based_kf_generation = False
        self.max_fov_centers_distance = -1

        self.last_processed_kf_img_id = None
        self.last_num_triangulated_points = None
        self.total_num_triangulated_points = 0
        self.last_num_fused_points = None
        self.total_num_fused_points = 0
        self.last_num_culled_points = None
        self.total_num_culled_points = 0
        self.last_num_culled_keyframes = None
        self.total_num_culled_keyframes = 0

        self.profile_keyframes = False
        self.schedule_log_rows: list[dict] = []
        self.local_ba_started_count = 0
        self.local_ba_completed_count = 0
        self.local_ba_aborted_count = 0
        self.local_ba_skipped_due_queue_count = 0
        self.local_ba_forced_due_starvation_count = 0
        self.last_successful_local_ba_kid = -1
        self.keyframes_since_last_successful_ba = 0
        self.consecutive_local_ba_aborts = 0
        self._local_ba_completion_window = False

        self._thread: _threading.Thread | None = None

        self.init_print()

    def init_print(self):
        if kVerbose:
            LocalMapping.print = staticmethod(print)
        if not self._use_cpp_lmc and hasattr(LocalMappingCore, "print"):
            LocalMappingCore.print = LocalMapping.print

    @property
    def map(self):
        return self.slam.map

    @property
    def sensor_type(self):
        return self.slam.sensor_type

    @property
    def kf_cur(self):
        return self.local_mapping_core.kf_cur

    @kf_cur.setter
    def kf_cur(self, value):
        self.local_mapping_core.kf_cur = value

    @property
    def kid_last_BA(self):
        return self.local_mapping_core.kid_last_BA

    @kid_last_BA.setter
    def kid_last_BA(self, value):
        self.local_mapping_core.kid_last_BA = value

    @property
    def descriptor_distance_sigma(self):
        return self.slam.tracking.descriptor_distance_sigma

    def _profile_section(self, name: str):
        profiler = getattr(self.slam, "runtime_profiler", None)
        if profiler is None:
            return nullcontext()
        return profiler.section(name)

    def set_opt_abort_flag(self, value):
        self.local_mapping_core.set_opt_abort_flag(value)

    def interrupt_optimization(self):
        """Signal the BA optimizer to abort early so LM can become idle sooner."""
        self.set_opt_abort_flag(True)

    def accept_keyframes(self) -> bool:
        return bool(self._accept_keyframes)

    def set_accept_keyframes(self, flag: bool) -> None:
        self._accept_keyframes = bool(flag)

    def keyframes_in_queue(self) -> int:
        return self.queue_size()

    def check_new_keyframes(self) -> bool:
        return self.keyframes_in_queue() > 0

    def _is_single_thread(self) -> bool:
        return not (
            self._thread is not None
            and getattr(self._thread, "is_alive", lambda: False)()
        )

    def _append_schedule_row(self, row: dict) -> None:
        if getattr(self, "profile_keyframes", False):
            self.schedule_log_rows.append(row)

    def _should_force_local_ba(self) -> bool:
        if self._local_ba_completion_window:
            return True
        if not Parameters.kEnableLocalBAStarvationGuard:
            return False
        if not Parameters.kForceLocalBAWhenStarved:
            return False
        return self.keyframes_since_last_successful_ba >= int(Parameters.kMaxKeyframesWithoutLocalBA)

    def is_stopped(self) -> bool:
        return bool(getattr(self, "stopped", False))

    def is_stop_requested(self) -> bool:
        return bool(getattr(self, "stop_requested", False))

    def push_keyframe(self, keyframe, img=None, img_right=None, depth=None):
        """Queue a keyframe for local mapping.

        §4.5: Do NOT store redundant image/depth copies in the queue.
        The keyframe already holds img/depth when kStoreKeyFrameImages is True.
        For the queue tuple we store (kf, img, img_right, depth) for backward-compat
        but only pass actual arrays when the caller explicitly provides them AND
        the keyframe does not already hold them.
        """
        with self.queue_condition:
            self.queue.put((keyframe, img, img_right, depth))
            self.queue_condition.notify_all()
        self.set_opt_abort_flag(True)

    def insert_keyframe(self, keyframe, img=None, img_right=None, depth=None):
        self.push_keyframe(keyframe, img=img, img_right=img_right, depth=depth)

    def pop_keyframe(self, timeout=Parameters.kLocalMappingTimeoutPopKeyframe):
        with self.queue_condition:
            if self.queue.empty():
                self.queue_condition.wait(timeout=timeout)
            if self.queue.empty() or self.stop_requested:
                return None
            return self.queue.get(timeout=timeout)

    def queue_size(self):
        return self.queue.qsize()

    def is_idle(self):
        with self.idle_condition:
            return self._is_idle

    def set_idle(self, flag):
        with self.idle_condition:
            self._is_idle = bool(flag)
            self.idle_condition.notify_all()

    def wait_idle(self, print=print, timeout=None):
        with self.idle_condition:
            while not self._is_idle and self.is_running:
                ok = self.idle_condition.wait(timeout=timeout)
                if not ok:
                    Printer.yellow(f"LocalMapping: timeout {timeout}s reached")
                    return

    def request_reset(self):
        with self.reset_mutex:
            self.reset_requested = True

    def reset_if_requested(self):
        with self.reset_mutex:
            if self.reset_requested:
                while not self.queue.empty():
                    self.queue.get()
                self.reset_requested = False
                self.total_num_triangulated_points = 0
                self.total_num_fused_points = 0
                self.total_num_culled_points = 0
                self.total_num_culled_keyframes = 0
                self.last_num_triangulated_points = None
                self.local_mapping_core.reset()

    def step(self):
        if self.map.num_keyframes() <= 0:
            time.sleep(kLocalMappingSleepTime)
            return

        ret = self.pop_keyframe(timeout=0.0)

        if ret is None:
            self.set_idle(True)
            return

        self.kf_cur, self.img_cur, self.img_cur_right, self.depth_cur = ret

        if self.kf_cur is None:
            self.set_idle(True)
            return

        self.last_processed_kf_img_id = getattr(self.kf_cur, "img_id", None)
        self.set_idle(False)

        try:
            self.do_local_mapping()
        except Exception as exc:
            LocalMapping.print(f"LocalMapping: encountered exception: {exc}")
            LocalMapping.print(traceback.format_exc())
            raise
        finally:
            self.img_cur = None
            self.img_cur_right = None
            self.depth_cur = None
            if not self._local_ba_completion_window:
                self.set_accept_keyframes(True)
            self.set_idle(True)
            self.reset_if_requested()

    def do_local_mapping(self):
        LocalMapping.print("local mapping: starting...")
        time_start = time.time()
        self.time_local_mapping = None
        queue_size_before = self.queue_size()
        accept_before = self.accept_keyframes()
        schedule_row = {
            "kf_id": getattr(self.kf_cur, "kid", getattr(self.kf_cur, "id", -1)),
            "timestamp": getattr(self.kf_cur, "timestamp", None),
            "queue_size_before": queue_size_before,
            "queue_size_after": queue_size_before,
            "accept_keyframes_before": accept_before,
            "accept_keyframes_after": self.accept_keyframes(),
            "is_single_thread": self._is_single_thread(),
            "processed_new_keyframe": False,
            "ran_cull_map_points": False,
            "ran_create_new_map_points": False,
            "ran_fuse_map_points": False,
            "ran_local_BA": False,
            "ran_cull_keyframes": False,
            "skipped_fuse_reason": "",
            "skipped_local_BA_reason": "",
            "local_BA_started": False,
            "local_BA_completed": False,
            "local_BA_aborted": False,
            "local_BA_forced_due_starvation": False,
            "keyframes_since_last_successful_ba": self.keyframes_since_last_successful_ba,
            "local_BA_sec": 0.0,
            "total_step_sec": 0.0,
        }
        row_appended = False

        self.set_accept_keyframes(False)
        try:
            if self.kf_cur is None:
                Printer.red("local mapping: no keyframe to process")
                return

            with self._profile_section("local_mapping.process_new_keyframe"):
                self.process_new_keyframe()
            schedule_row["processed_new_keyframe"] = True
            self.keyframes_since_last_successful_ba += 1

            # §4.4 third call site: release KF depth image after process_new_keyframe.
            # depths/uRs are already extracted; depth_img is no longer needed.
            if not Parameters.kStoreKeyFrameDepthImages and self.kf_cur is not None:
                try:
                    self.kf_cur.release_heavy_data(
                        release_rgb=not Parameters.kStoreKeyFrameImages,
                        release_depth=True,
                        release_kd=False,
                    )
                except Exception:
                    pass

            with self._profile_section("local_mapping.cull_map_points"):
                num_culled_points = self.cull_map_points()
            schedule_row["ran_cull_map_points"] = True
            self.last_num_culled_points = num_culled_points
            self.total_num_culled_points += num_culled_points

            with self._profile_section("local_mapping.create_new_map_points"):
                total_new_pts = self.create_new_map_points()
            schedule_row["ran_create_new_map_points"] = True
            self.last_num_triangulated_points = total_new_pts
            self.total_num_triangulated_points += total_new_pts

            queue_pending = self.check_new_keyframes()
            is_single_thread = self._is_single_thread()
            if (not queue_pending) or is_single_thread:
                with self._profile_section("local_mapping.fuse_map_points"):
                    total_fused_pts = self.fuse_map_points()
                schedule_row["ran_fuse_map_points"] = True
                self.last_num_fused_points = total_fused_pts
                self.total_num_fused_points += total_fused_pts
            else:
                schedule_row["skipped_fuse_reason"] = "queue_pending_threaded"

            force_local_ba = self._should_force_local_ba()
            if force_local_ba:
                self.local_ba_forced_due_starvation_count += 1
                schedule_row["local_BA_forced_due_starvation"] = True

            should_run_ba = (
                ((not self.check_new_keyframes()) and not self.is_stop_requested())
                or is_single_thread
                or force_local_ba
            )

            if should_run_ba:
                self.set_opt_abort_flag(False)
                local_ba_start = time.time()
                schedule_row["local_BA_started"] = True
                self.local_ba_started_count += 1
                with self._profile_section("local_mapping.local_BA"):
                    self.local_BA()
                schedule_row["ran_local_BA"] = True
                schedule_row["local_BA_sec"] = time.time() - local_ba_start
                aborted = bool(getattr(getattr(self.local_mapping_core, "opt_abort_flag", None), "value", False))
                schedule_row["local_BA_aborted"] = aborted
                if aborted:
                    self.local_ba_aborted_count += 1
                    self.consecutive_local_ba_aborts += 1
                    if (
                        Parameters.kEnableLocalBAStarvationGuard
                        and self.consecutive_local_ba_aborts >= int(Parameters.kMaxConsecutiveLocalBAAborts)
                    ):
                        self._local_ba_completion_window = True
                        self.set_accept_keyframes(False)
                else:
                    self.local_ba_completed_count += 1
                    self.consecutive_local_ba_aborts = 0
                    self.last_successful_local_ba_kid = int(getattr(self.kf_cur, "kid", getattr(self.kf_cur, "id", -1)))
                    self.keyframes_since_last_successful_ba = 0
                    self._local_ba_completion_window = False
                    schedule_row["local_BA_completed"] = True

                with self._profile_section("local_mapping.cull_keyframes"):
                    num_culled_keyframes = self.cull_keyframes()
                schedule_row["ran_cull_keyframes"] = True
                self.last_num_culled_keyframes = num_culled_keyframes
                self.total_num_culled_keyframes += num_culled_keyframes
            else:
                self.local_ba_skipped_due_queue_count += 1
                schedule_row["skipped_local_BA_reason"] = "queue_pending_threaded"

            self.time_local_mapping = time.time() - time_start
            LocalMapping.print(f"local mapping elapsed time: {self.time_local_mapping}")
        finally:
            if not self._local_ba_completion_window:
                self.set_accept_keyframes(True)
            if self.time_local_mapping is None:
                self.time_local_mapping = time.time() - time_start
            schedule_row["queue_size_after"] = self.queue_size()
            schedule_row["accept_keyframes_after"] = self.accept_keyframes()
            schedule_row["keyframes_since_last_successful_ba"] = self.keyframes_since_last_successful_ba
            schedule_row["total_step_sec"] = self.time_local_mapping
            if not row_appended:
                self._append_schedule_row(schedule_row)

    def local_BA(self):
        if getattr(self.slam, "loop_closing", None) is not None:
            if self.slam.loop_closing.is_correcting():
                return

        err, num_kf_ref_tracked_points = self.local_mapping_core.local_BA()
        self.mean_ba_chi2_error = err

        if getattr(self.slam, "tracking", None) is not None:
            self.slam.tracking.num_kf_ref_tracked_points = num_kf_ref_tracked_points

    def large_window_BA(self):
        result = self.local_mapping_core.large_window_BA()
        if isinstance(result, tuple):
            return result[0]
        return result

    def process_new_keyframe(self):
        self.local_mapping_core.process_new_keyframe()

    def cull_map_points(self):
        return self.local_mapping_core.cull_map_points()

    def cull_keyframes(self):
        if self._use_cpp_lmc:
            return self.local_mapping_core.cull_keyframes()
        return self.local_mapping_core.cull_keyframes(
            self.use_fov_centers_based_kf_generation,
            self.max_fov_centers_distance,
        )

    def _get_local_mapping_neighbors(self):
        if self.sensor_type == SensorType.MONOCULAR:
            num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesMonocular
        else:
            num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesStereo

        if hasattr(self.map, "local_map") and hasattr(self.map.local_map, "get_best_neighbors"):
            return self.map.local_map.get_best_neighbors(self.kf_cur, N=num_neighbors)

        return self.kf_cur.get_best_covisible_keyframes(num_neighbors)

    def create_new_map_points(self):
        total_new_pts = 0

        local_keyframes = [
            kf for kf in self._get_local_mapping_neighbors()
            if kf is not None and kf is not self.kf_cur and not kf.is_bad()
        ]

        for kf in local_keyframes:
            if not self.queue.empty():
                return total_new_pts

            idxs_cur, idxs, num_found_matches = EpipolarMatcher.search_frame_for_triangulation(
                self.kf_cur,
                kf,
                None,
                None,
                max_descriptor_distance=0.5 * self.descriptor_distance_sigma,
                is_monocular=(self.sensor_type == SensorType.MONOCULAR),
            )

            if num_found_matches == 0:
                continue

            pts3d, mask_pts3d = triangulate_normalized_points(
                self.kf_cur.pose(),
                kf.pose(),
                self.kf_cur.kpsn[idxs_cur],
                kf.kpsn[idxs],
            )

            new_pts_count, _, list_added_points = self.map.add_points(
                pts3d,
                mask_pts3d,
                self.kf_cur,
                kf,
                idxs_cur,
                idxs,
                self.img_cur,
                do_check=True,
                far_points_threshold=self.far_points_threshold,
            )

            total_new_pts += new_pts_count
            self.local_mapping_core.add_points(list_added_points)

        return total_new_pts

    def fuse_map_points(self):
        return self.local_mapping_core.fuse_map_points(self.descriptor_distance_sigma)

    # ------------------------------------------------------------------
    # Background-thread support
    # ------------------------------------------------------------------

    def start_thread(self):
        """Start local mapping on a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self.is_running = True
        self.stop_requested = False
        self._thread = _threading.Thread(
            target=self._run_thread, daemon=True, name="LocalMapping"
        )
        self._thread.start()
        Printer.green("LocalMapping: background thread started")

    def stop_thread(self, timeout: float = 10.0):
        """Signal the background thread to stop and wait for it to exit."""
        self.is_running = False
        self.stop_requested = True
        with self.queue_condition:
            self.queue_condition.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                Printer.yellow(f"LocalMapping: thread did not exit within {timeout}s")
        self._thread = None
        Printer.green("LocalMapping: background thread stopped")

    def _run_thread(self):
        """Inner loop run by the background thread."""
        while self.is_running and not self.stop_requested:
            try:
                # Blocking pop with default 0.5 s timeout — thread sleeps when idle
                ret = self.pop_keyframe()
                if ret is None:
                    self.set_idle(True)
                    continue

                self.kf_cur, self.img_cur, self.img_cur_right, self.depth_cur = ret
                if self.kf_cur is None:
                    self.set_idle(True)
                    continue

                self.last_processed_kf_img_id = getattr(self.kf_cur, "img_id", None)
                self.set_idle(False)
                try:
                    self.do_local_mapping()
                except Exception as exc:
                    LocalMapping.print(f"LocalMapping thread: {exc}")
                    LocalMapping.print(traceback.format_exc())
                finally:
                    self.img_cur = None
                    self.img_cur_right = None
                    self.depth_cur = None
                    if not self._local_ba_completion_window:
                        self.set_accept_keyframes(True)
                    self.set_idle(True)
                    self.reset_if_requested()
            except Exception as exc:
                Printer.red(f"LocalMapping thread outer: {exc}")
                time.sleep(kLocalMappingSleepTime)
        self.set_idle(True)
        if not self._local_ba_completion_window:
            self.set_accept_keyframes(True)

    def queue_memory_stats(self) -> dict:
        qsize = self.queue.qsize()
        stats = {
            "queue_size": qsize,
            "estimated_queue_heavy_bytes": 0,
            "active_img": self.img_cur is not None,
            "active_depth": self.depth_cur is not None
        }
        if qsize > 0:
            stats["estimated_queue_heavy_bytes"] = qsize * 1024 * 1024 * 2
        return stats
