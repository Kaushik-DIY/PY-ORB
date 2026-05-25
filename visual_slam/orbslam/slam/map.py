"""
Global sparse map container.
This module stores frames, keyframes, landmarks, and local-covisibility bookkeeping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Iterable, Optional

import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.frame import Frame
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.map_point import MapPoint
from visual_slam.orbslam.slam.optimizer_g2o import local_bundle_adjustment, global_bundle_adjustment


# Provide a lightweight ordered set for keyframes and map points.
class OrderedSetLite:
    """
    Minimal ordered-set replacement.

    small container preserves insertion order and supports the subset used by the
    ORB-SLAM path.
    """

    def __init__(self, values: Optional[Iterable] = None):
        self._items = []
        if values is not None:
            for value in values:
                self.add(value)

    def add(self, value) -> None:
        if value not in self._items:
            self._items.append(value)

    def discard(self, value) -> None:
        try:
            self._items.remove(value)
        except ValueError:
            pass

    def remove(self, value) -> None:
        self._items.remove(value)

    def clear(self) -> None:
        self._items.clear()

    def copy(self) -> "OrderedSetLite":
        return OrderedSetLite(self._items)

    def to_list(self) -> list:
        return list(self._items)

    def __contains__(self, value) -> bool:
        return value in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return OrderedSetLite(self._items[item])
        return self._items[item]

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return f"OrderedSetLite({self._items!r})"


# Store counters recovered from a previously saved map session.
@dataclass
class ReloadedSessionMapInfo:
    num_keyframes: int
    num_points: int
    max_point_id: int
    max_frame_id: int
    max_keyframe_id: int


# Hold cached map-state arrays used by visualization and export code.
class MapStateData:

    def __init__(self):
        self.poses = []
        self.pose_timestamps = []
        self.fov_centers = []
        self.fov_centers_colors = []
        self.points = []
        self.colors = []
        self.semantic_colors = []
        self.covisibility_graph = []
        self.spanning_tree = []
        self.loops = []


# Maintain the local keyframe and point neighborhood around a reference keyframe.
class LocalCovisibilityMap:
    """

    The local map consists of:
    - the reference keyframe
    - its best covisible keyframes
    - map points observed by those keyframes
    """

    def __init__(self, map: "Map"):
        self.map = map
        self.reference_keyframe = None
        self.local_keyframes = OrderedSetLite()
        self.local_points = OrderedSetLite()
        self.ref_keyframes = OrderedSetLite()

    def reset(self) -> None:
        self.reference_keyframe = None
        self.local_keyframes.clear()
        self.local_points.clear()
        self.ref_keyframes.clear()

    def reset_session(self, keyframes_to_remove=None, points_to_remove=None) -> None:
        if keyframes_to_remove:
            for kf in keyframes_to_remove:
                self.local_keyframes.discard(kf)
                self.ref_keyframes.discard(kf)
        if points_to_remove:
            for p in points_to_remove:
                self.local_points.discard(p)

    def update(self, reference_keyframe: KeyFrame, num_best: int = Parameters.kNumBestCovisibilityKeyFrames):
        self.reset()
        self.reference_keyframe = reference_keyframe

        if reference_keyframe is None:
            return self.get_keyframes(), self.get_points(), self.ref_keyframes.copy()

        self.local_keyframes.add(reference_keyframe)

        for kf in reference_keyframe.get_best_covisible_keyframes(num_best):
            if kf is not None and not kf.is_bad():
                self.local_keyframes.add(kf)

        for kf in self.local_keyframes:
            for p in kf.get_matched_good_points():
                if p is not None and not p.is_bad():
                    self.local_points.add(p)

        for p in self.local_points:
            for kf, _ in p.observations():
                if kf is not None and not kf.is_bad() and kf not in self.local_keyframes:
                    self.ref_keyframes.add(kf)

        return self.get_keyframes(), self.get_points(), self.ref_keyframes.copy()

    def get_keyframes(self) -> OrderedSetLite:
        return self.local_keyframes.copy()

    def get_points(self) -> OrderedSetLite:
        return self.local_points.copy()

    def num_keyframes(self) -> int:
        return len(self.local_keyframes)

    def num_points(self) -> int:
        return len(self.local_points)


# Store the global sparse map with frames, keyframes, points, and local neighborhoods.
class Map:

    def __init__(self):
        self._lock = RLock()
        self._update_lock = RLock()

        self.frames: deque[Frame] = deque(maxlen=Parameters.kMaxLenFrameDeque)
        self.keyframes: OrderedSetLite = OrderedSetLite()
        self.points: OrderedSetLite = OrderedSetLite()
        self.keyframe_origins: OrderedSetLite = OrderedSetLite()

        self.keyframes_map: dict[int, KeyFrame] = {}

        self.max_point_id = 0
        self.max_frame_id = 0
        self.max_keyframe_id = 0

        self.reloaded_session_map_info: ReloadedSessionMapInfo | None = None
        self.local_map = LocalCovisibilityMap(map=self)
        self.viewer_scale = -1
        self._frame_view_stats_cache = {
            "num_frame_views_total": 0,
            "old_frame_views_total": 0,
            "oldest_frame_view_id": -1,
            "checked_points": 0,
            "removed_frame_views": 0,
            "remaining_frame_views": 0,
            "oldest_remaining_frame_view_id": -1,
        }

    @property
    def lock(self):
        return self._lock

    @property
    def update_lock(self):
        return self._update_lock

    def is_reloaded(self) -> bool:
        return self.reloaded_session_map_info is not None

    def reset(self) -> None:
        with self._lock:
            with self._update_lock:
                self.frames.clear()
                self.keyframes.clear()
                self.points.clear()
                self.keyframe_origins.clear()
                self.keyframes_map.clear()
                self.local_map.reset()
                self.max_point_id = 0
                self.max_frame_id = 0
                self.max_keyframe_id = 0
                self._reset_frame_view_stats_cache()

    def reset_session(self) -> None:
        # Reset the in-memory map state for a fresh run.
        self.reset()

    def delete(self) -> None:
        with self._lock:
            for frame in self.frames:
                frame.reset_points()
            for keyframe in self.keyframes:
                keyframe.reset_points()

    # ------------------------------------------------------------------
    # Points
    # ------------------------------------------------------------------

    def get_points(self) -> OrderedSetLite:
        with self._lock:
            return self.points.copy()

    def num_points(self) -> int:
        with self._lock:
            return len(self.points)

    def add_point(self, point: MapPoint) -> int:
        with self._lock:
            ret = self.max_point_id
            point.id = ret
            point.map = self
            self.max_point_id += 1
            self.points.add(point)
            return ret

    def remove_point(self, point: MapPoint) -> None:
        with self._lock:
            self.points.discard(point)
            if getattr(point, "map", None) is self:
                point.map = None

    def remove_point_no_lock(self, point: MapPoint) -> None:
        self.points.discard(point)
        if getattr(point, "map", None) is self:
            point.map = None

    # Compatibility alias.
    def add_map_point(self, point: MapPoint) -> int:
        return self.add_point(point)

    def remove_map_point(self, point: MapPoint) -> None:
        self.remove_point(point)

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def get_frame(self, idx: int):
        with self._lock:
            try:
                return self.frames[idx]
            except Exception:
                return None

    def get_frames(self):
        with self._lock:
            return self.frames.copy()

    def num_frames(self) -> int:
        with self._lock:
            return len(self.frames)

    def _cleanup_evicted_frame(self, frame: Frame) -> None:
        frame.remove_frame_views()
        frame.reset_points()
        release_rgb = Parameters.kReleaseNormalFrameImagesAfterUse or not Parameters.kStoreNormalFrameImages
        release_depth = Parameters.kReleaseNormalFrameImagesAfterUse or not Parameters.kStoreNormalFrameImages
        release_desc = Parameters.kReleaseEvictedFrameFeatureCache
        frame.release_heavy_data(release_images=True, release_kd=True, release_descriptors=release_desc, release_points=True)
        frame.release_images(release_rgb=release_rgb, release_right=release_rgb, release_depth=release_depth)

    def add_frame(self, frame: Frame, override_id: bool = False) -> int:
        with self._lock:
            ret = frame.id
            if override_id:
                ret = self.max_frame_id
                frame.id = ret
                self.max_frame_id += 1
            else:
                self.max_frame_id = max(self.max_frame_id, frame.id + 1)

            if Parameters.kEnableFrameEvictionCleanup and len(self.frames) >= getattr(self.frames, 'maxlen', Parameters.kMaxLenFrameDeque):
                old_frame = self.frames.popleft()
                self._cleanup_evicted_frame(old_frame)

            self.frames.append(frame)
            return ret

    def _reset_frame_view_stats_cache(self) -> None:
        self._frame_view_stats_cache = {
            "num_frame_views_total": 0,
            "old_frame_views_total": 0,
            "oldest_frame_view_id": -1,
            "checked_points": 0,
            "removed_frame_views": 0,
            "remaining_frame_views": 0,
            "oldest_remaining_frame_view_id": -1,
        }

    def _update_frame_view_stats_cache(self, stats: dict) -> None:
        self._frame_view_stats_cache.update(
            {
                "num_frame_views_total": int(stats.get("num_frame_views_total", stats.get("remaining_frame_views", 0))),
                "old_frame_views_total": int(stats.get("old_frame_views_total", 0)),
                "oldest_frame_view_id": int(stats.get("oldest_frame_view_id", stats.get("oldest_remaining_frame_view_id", -1))),
                "checked_points": int(stats.get("checked_points", 0)),
                "removed_frame_views": int(stats.get("removed_frame_views", 0)),
                "remaining_frame_views": int(stats.get("remaining_frame_views", stats.get("num_frame_views_total", 0))),
                "oldest_remaining_frame_view_id": int(stats.get("oldest_remaining_frame_view_id", stats.get("oldest_frame_view_id", -1))),
            }
        )

    def remove_frame(self, frame: Frame) -> None:
        with self._lock:
            try:
                self.frames.remove(frame)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Keyframes
    # ------------------------------------------------------------------

    def get_keyframes(self) -> OrderedSetLite:
        with self._lock:
            return self.keyframes.copy()

    def get_first_keyframe(self):
        with self._lock:
            if len(self.keyframes) == 0:
                return None
            return self.keyframes[0]

    def get_last_keyframe(self):
        with self._lock:
            if len(self.keyframes) == 0:
                return None
            return self.keyframes[-1]

    def get_last_keyframes(self, local_window_size: int = Parameters.kLocalBAWindowSize) -> OrderedSetLite:
        with self._lock:
            return self.keyframes[-int(local_window_size):]

    def num_keyframes(self) -> int:
        with self._lock:
            return len(self.keyframes)

    def num_keyframes_session(self) -> int:
        with self._lock:
            if self.reloaded_session_map_info is not None:
                return len(self.keyframes) - self.reloaded_session_map_info.num_keyframes
            return len(self.keyframes)

    def add_keyframe(self, keyframe: KeyFrame) -> int:
        with self._lock:
            assert keyframe.is_keyframe

            ret = self.max_keyframe_id
            keyframe.kid = ret
            keyframe.is_keyframe = True
            keyframe.map = self

            self.keyframes.add(keyframe)
            self.keyframes_map[keyframe.id] = keyframe
            self.max_keyframe_id += 1

            if ret == 0:
                self.keyframe_origins.add(keyframe)

            return ret

    def remove_keyframe(self, keyframe: KeyFrame) -> None:
        with self._lock:
            assert keyframe.is_keyframe
            self.keyframes.discard(keyframe)
            self.keyframe_origins.discard(keyframe)
            self.keyframes_map.pop(keyframe.id, None)
            if getattr(keyframe, "map", None) is self:
                keyframe.map = None

    def get_keyframe_by_frame_id(self, frame_id: int):
        with self._lock:
            return self.keyframes_map.get(int(frame_id), None)

    # ------------------------------------------------------------------
    # Local map
    # ------------------------------------------------------------------

    def update_local_map(self, reference_keyframe: KeyFrame, num_best: int = Parameters.kNumBestCovisibilityKeyFrames) -> None:
        with self._lock:
            self.local_map.update(reference_keyframe, num_best=num_best)

    def get_local_keyframes(self) -> OrderedSetLite:
        return self.local_map.get_keyframes()

    def get_local_points(self) -> OrderedSetLite:
        return self.local_map.get_points()

    # ------------------------------------------------------------------

    def locally_optimize(self, kf_ref, abort_flag=None, mp_abort_flag=None):
        """Run local bundle adjustment around the current reference keyframe."""
        keyframes, points, ref_keyframes = self.local_map.update(kf_ref)
        result = local_bundle_adjustment(
            keyframes,
            points,
            ref_keyframes,
            False,
            False,
            10,
            abort_flag=abort_flag,
            mp_abort_flag=mp_abort_flag,
            map_lock=self.update_lock,
        )
        return result.mean_squared_error

    def optimize(self, local_window_size=None, abort_flag=None):
        """

        Returns:
            (mse, result_dict)
        """
        keyframes = self.get_keyframes().to_list() if hasattr(self.get_keyframes(), "to_list") else list(self.get_keyframes())
        points = self.get_points().to_list() if hasattr(self.get_points(), "to_list") else list(self.get_points())

        return global_bundle_adjustment(
            keyframes=keyframes,
            points=points,
            rounds=10,
            result_dict={},
        )

    def add_points(
        self,
        pts3d,
        pts3d_mask,
        kf1,
        kf2,
        idxs1,
        idxs2,
        img=None,
        do_check=True,
        far_points_threshold=None,
    ):
        """

        Returns:
            (num_added, mask_added, list_added_points)
        """
        pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
        pts3d_mask = np.asarray(pts3d_mask, dtype=bool).reshape(-1)
        idxs1 = np.asarray(idxs1, dtype=np.int32).reshape(-1)
        idxs2 = np.asarray(idxs2, dtype=np.int32).reshape(-1)

        added_mask = np.zeros(len(pts3d), dtype=bool)
        list_added_points = []

        for i, (pw, is_valid, idx1, idx2) in enumerate(zip(pts3d, pts3d_mask, idxs1, idxs2)):
            if not bool(is_valid):
                continue

            if idx1 < 0 or idx1 >= len(kf1.points):
                continue
            if idx2 < 0 or idx2 >= len(kf2.points):
                continue

            if kf1.points[idx1] is not None or kf2.points[idx2] is not None:
                continue

            if far_points_threshold is not None:
                d1 = np.linalg.norm(pw - kf1.Ow().reshape(3))
                d2 = np.linalg.norm(pw - kf2.Ow().reshape(3))
                if d1 > far_points_threshold or d2 > far_points_threshold:
                    continue

            mp = MapPoint(pw, keyframe=kf1, idx=int(idx1))
            mp.add_observation(kf2, int(idx2))
            self.add_point(mp)
            mp.update_info()

            added_mask[i] = True
            list_added_points.append(mp)

        return len(list_added_points), added_mask, list_added_points

    def add_stereo_points(self, pts3d, pts3d_mask, f, kf, idxs, img=None) -> int:
        """

        Creates RGB-D/stereo map points from valid 3D coordinates and attaches
        them to the new keyframe observations.
        """
        count = 0
        idxs = list(np.asarray(idxs, dtype=np.int32).reshape(-1))

        for p, is_valid, idx in zip(pts3d, pts3d_mask, idxs):
            if not bool(is_valid):
                continue
            if idx < 0 or idx >= len(kf.points):
                continue

            existing = kf.points[idx]
            if existing is not None and existing.num_observations() > 0:
                continue

            mp = MapPoint(np.asarray(p, dtype=np.float64).reshape(3), keyframe=kf, idx=int(idx))
            self.add_point(mp)

            if f is not None and hasattr(f, "points") and idx < len(f.points):
                f.points[idx] = mp

            mp.update_info()
            count += 1

        return count

    def prune_old_frame_views(self, current_frame_id: int | None = None, keep_last: int | None = None) -> dict:
        with self._lock:
            if current_frame_id is None:
                current_frame_id = self.max_frame_id - 1
            if keep_last is None:
                keep_last = Parameters.kFrameViewRetention
            min_frame_id = current_frame_id - keep_last + 1

            checked_points = 0
            removed_frame_views = 0
            remaining_frame_views = 0
            oldest_remaining_frame_view_id = -1

            for p in self.points:
                checked_points += 1
                if hasattr(p, 'remove_frame_views_older_than'):
                    removed = p.remove_frame_views_older_than(min_frame_id)
                    removed_frame_views += removed
                
                if hasattr(p, 'get_frame_views'):
                    views = p.get_frame_views()
                    remaining_frame_views += len(views)
                    for frame_id in views.keys():
                        if oldest_remaining_frame_view_id == -1 or frame_id < oldest_remaining_frame_view_id:
                            oldest_remaining_frame_view_id = frame_id

            stats = {
                "checked_points": checked_points,
                "removed_frame_views": removed_frame_views,
                "remaining_frame_views": remaining_frame_views,
                "oldest_remaining_frame_view_id": oldest_remaining_frame_view_id
            }
            self._update_frame_view_stats_cache(stats)
            return stats

    def memory_stats(self, mode: str = "deep") -> dict:
        with self._lock:
            mode = "deep" if str(mode).lower() == "deep" else "cheap"
            stats = {
                "num_recent_frames": len(self.frames),
                "recent_frame_ids_min": min((f.id for f in self.frames), default=-1),
                "recent_frame_ids_max": max((f.id for f in self.frames), default=-1),
                "max_len_frame_deque": getattr(self.frames, "maxlen", -1),
                "num_keyframes": len(self.keyframes),
                "num_map_points": len(self.points),
                "num_frame_views_total": int(self._frame_view_stats_cache.get("num_frame_views_total", 0)),
                "old_frame_views_total": int(self._frame_view_stats_cache.get("old_frame_views_total", 0)),
                "oldest_frame_view_id": int(self._frame_view_stats_cache.get("oldest_frame_view_id", -1)),
                "num_keyframe_observations_total": 0,
                "num_bad_points": 0,
                "num_recent_frame_images": 0,
                "num_recent_frame_depth_images": 0,
                "num_keyframe_images": 0,
                "num_keyframe_depth_images": 0,
                "estimated_frame_heavy_bytes": 0,
                "estimated_keyframe_heavy_bytes": 0,
                "estimated_total_heavy_bytes": 0
            }

            for f in self.frames:
                if f.img is not None:
                    stats["num_recent_frame_images"] += 1
                if f.depth_img is not None:
                    stats["num_recent_frame_depth_images"] += 1
                if hasattr(f, "heavy_memory_bytes"):
                    stats["estimated_frame_heavy_bytes"] += f.heavy_memory_bytes()

            for kf in self.keyframes:
                if kf.img is not None:
                    stats["num_keyframe_images"] += 1
                if kf.depth_img is not None:
                    stats["num_keyframe_depth_images"] += 1
                if hasattr(kf, "heavy_memory_bytes"):
                    stats["estimated_keyframe_heavy_bytes"] += kf.heavy_memory_bytes()

            stats["estimated_total_heavy_bytes"] = stats["estimated_frame_heavy_bytes"] + stats["estimated_keyframe_heavy_bytes"]

            if mode == "cheap":
                return stats

            if len(self.frames) > 0:
                min_recent_id = stats["recent_frame_ids_min"]
            else:
                min_recent_id = self.max_frame_id

            stats["num_frame_views_total"] = 0
            stats["old_frame_views_total"] = 0
            stats["oldest_frame_view_id"] = -1

            for p in self.points:
                if hasattr(p, "is_bad") and p.is_bad():
                    stats["num_bad_points"] += 1

                if hasattr(p, "num_observations"):
                    stats["num_keyframe_observations_total"] += p.num_observations()

                if hasattr(p, "get_frame_views"):
                    views = p.get_frame_views()
                    stats["num_frame_views_total"] += len(views)
                    for f_id in views.keys():
                        if f_id < min_recent_id:
                            stats["old_frame_views_total"] += 1
                        if stats["oldest_frame_view_id"] == -1 or f_id < stats["oldest_frame_view_id"]:
                            stats["oldest_frame_view_id"] = f_id

            self._update_frame_view_stats_cache(stats)
            return stats

    def __repr__(self) -> str:
        return (
            f"Map(frames={self.num_frames()}, "
            f"keyframes={self.num_keyframes()}, "
            f"points={self.num_points()})"
        )
