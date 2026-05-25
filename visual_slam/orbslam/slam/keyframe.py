"""
Keyframe representation and graph structure.
This module stores covisibility, spanning-tree, loop-edge, and observation relationships.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from threading import RLock, Lock
from typing import Optional

import numpy as np

from visual_slam.orbslam.slam.camera_pose import CameraPose
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.frame import Frame


# Store the graph relationships attached to one keyframe.
class KeyFrameGraph:
    """Graph container storing parent, child, loop, and covisibility edges."""
    def __init__(self):
        self._lock_features = RLock()
        self._lock_connections = Lock()

        # Spanning tree
        self.init_parent = False
        self.parent = None
        self.children = set()
        self.is_first_connection = True

        # Loop edges
        self.loop_edges = set()
        self.not_to_erase = False

        # Covisibility graph
        self.connected_keyframes_weights = Counter()
        self.ordered_keyframes_weights = OrderedDict()
        self.last_tracking_frame_id = -1
        self.tracking_vote_count = 0

    # ------------------------------------------------------------------
    # Spanning tree
    # ------------------------------------------------------------------

    def add_child_no_lock_(self, keyframe) -> None:
        self.children.add(keyframe)

    def add_child(self, keyframe) -> None:
        with self._lock_connections:
            self.add_child_no_lock_(keyframe)

    def erase_child_no_lock_(self, keyframe) -> None:
        try:
            self.children.remove(keyframe)
        except KeyError:
            pass

    def erase_child(self, keyframe) -> None:
        with self._lock_connections:
            self.erase_child_no_lock_(keyframe)

    def set_parent_no_lock_(self, keyframe) -> None:
        if keyframe is None or keyframe is self:
            return
        self.parent = keyframe
        self.init_parent = True
        keyframe.add_child(self)

    def set_parent(self, keyframe) -> None:
        with self._lock_connections:
            self.set_parent_no_lock_(keyframe)

    def get_children(self):
        with self._lock_connections:
            return self.children.copy()

    def get_parent(self):
        with self._lock_connections:
            return self.parent

    def has_child(self, keyframe) -> bool:
        with self._lock_connections:
            return keyframe in self.children

    # ------------------------------------------------------------------
    # Loop edges
    # ------------------------------------------------------------------

    def add_loop_edge(self, keyframe) -> None:
        with self._lock_connections:
            self.not_to_erase = True
            if keyframe is not None and keyframe is not self:
                self.loop_edges.add(keyframe)

    def get_loop_edges(self):
        with self._lock_connections:
            return self.loop_edges.copy()

    # ------------------------------------------------------------------
    # Covisibility graph
    # ------------------------------------------------------------------

    def reset_covisibility(self) -> None:
        self.connected_keyframes_weights = Counter()
        self.ordered_keyframes_weights = OrderedDict()

    def update_best_covisibles_no_lock_(self) -> None:
        self.ordered_keyframes_weights = OrderedDict(
            sorted(
                self.connected_keyframes_weights.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def add_connection_no_lock_(self, keyframe, weight: int) -> None:
        if keyframe is None or keyframe is self:
            return
        self.connected_keyframes_weights[keyframe] = int(weight)
        self.update_best_covisibles_no_lock_()

    def add_connection(self, keyframe, weight: int) -> None:
        with self._lock_connections:
            self.add_connection_no_lock_(keyframe, weight)

    def erase_connection_no_lock_(self, keyframe) -> None:
        try:
            del self.connected_keyframes_weights[keyframe]
            self.update_best_covisibles_no_lock_()
        except KeyError:
            pass

    def erase_connection(self, keyframe) -> None:
        with self._lock_connections:
            self.erase_connection_no_lock_(keyframe)

    def get_connected_keyframes_no_lock_(self):
        return list(self.connected_keyframes_weights.keys())

    def get_connected_keyframes(self):
        with self._lock_connections:
            return self.get_connected_keyframes_no_lock_()

    def get_covisible_keyframes_no_lock_(self):
        return list(self.ordered_keyframes_weights.keys())

    def get_covisible_keyframes(self):
        with self._lock_connections:
            return self.get_covisible_keyframes_no_lock_()

    def get_best_covisible_keyframes(self, N: int):
        with self._lock_connections:
            return list(self.ordered_keyframes_weights.keys())[: int(N)]

    def get_covisible_by_weight(self, weight: int):
        with self._lock_connections:
            return [kf for kf, w in self.ordered_keyframes_weights.items() if w > weight]

    def get_weight_no_lock_(self, keyframe) -> int:
        return int(self.connected_keyframes_weights[keyframe])

    def get_weight(self, keyframe) -> int:
        with self._lock_connections:
            return self.get_weight_no_lock_(keyframe)

    def get_connected_keyframes_weights(self):
        with self._lock_connections:
            return {
                getattr(kf, "id", None): w
                for kf, w in self.connected_keyframes_weights.items()
                if kf is not None
            }


# Represent a selected map keyframe with graph and observation state.
class KeyFrame(Frame, KeyFrameGraph):

    def __init__(
        self,
        frame: Frame,
        img=None,
        img_right=None,
        depth=None,
        kid: Optional[int] = None,
    ):
        KeyFrameGraph.__init__(self)

        # Create a Frame shell without recomputing features.
        Frame.__init__(
            self,
            img=None,
            camera=frame.camera,
            pose=frame.pose(),
            id=frame.id,
            timestamp=frame.timestamp,
            img_id=frame.img_id,
        )

        self.img = frame.img if frame.img is not None else img
        self.img_right = frame.img_right if frame.img_right is not None else img_right
        self.depth_img = frame.depth_img if frame.depth_img is not None else depth

        self.map = None
        self.is_keyframe = True
        self.kid = kid if kid is not None else frame.id

        self._is_bad = False
        self.to_be_erased = False
        self.lba_count = 0

        self.is_blurry = getattr(frame, "is_blurry", False)
        self.laplacian_var = getattr(frame, "laplacian_var", None)

        # Pose relative to parent. Computed when the keyframe is marked bad.
        self._pose_Tcp = CameraPose()

        # Share immutable feature information with the source frame.
        self.kps = frame.kps
        self.kpsu = frame.kpsu
        self.des = frame.des
        self.depths = frame.depths
        self.uRs = frame.uRs
        self.kps_ur = frame.uRs

        self.kpsn = getattr(frame, "kpsn", None)
        self.octaves = np.array([max(0, int(getattr(kp, "octave", 0))) for kp in self.kps], dtype=np.int32)
        self.sizes = np.array([float(getattr(kp, "size", 0.0)) for kp in self.kps], dtype=np.float32)
        self.angles = np.array([float(getattr(kp, "angle", -1.0)) for kp in self.kps], dtype=np.float32)

        self.median_depth = frame.median_depth
        self.fov_center_c = frame.fov_center_c
        self.fov_center_w = frame.fov_center_w

        # Loop closing fields
        self.g_des = None
        self.f_des = None
        self.bow_vector = None
        self.feature_vector = None
        self.loop_query_id = None
        self.num_loop_words = 0
        self.loop_score = 0.0

        # Relocalization fields
        self.reloc_query_id = None
        self.num_reloc_words = 0
        self.reloc_score = 0.0

        # GBA fields
        self.GBA_kf_id = 0
        self.is_Tcw_GBA_valid = False
        self.Tcw_GBA = None
        self.Tcw_before_GBA = None

        # Copy map-point associations from source frame.
        self.points = list(frame.points)
        self.outliers = np.zeros(len(self.kps), dtype=bool)

    def init_observations(self) -> None:
        """Associate all currently matched map points as keyframe observations."""
        if not hasattr(self, "_lock_features"):
            self._lock_features = RLock()
        with self._lock_features:
            points = list(self.points)

        for idx, point in enumerate(points):
            if point is None:
                continue
            if hasattr(point, "is_bad") and point.is_bad():
                continue
            if point.add_observation(self, idx):
                point.update_info()

    def update_connections(self) -> None:
        """
        """
        points = self.get_matched_good_points()
        if len(points) == 0:
            return

        viewing_keyframes = Counter()

        for point in points:
            if point is None:
                continue
            for kf in point.keyframes():
                if kf is self:
                    continue
                if getattr(kf, "kid", None) == self.kid:
                    continue
                if hasattr(kf, "is_bad") and kf.is_bad():
                    continue
                viewing_keyframes[kf] += 1

        if not viewing_keyframes:
            return

        covisible_keyframes = viewing_keyframes.most_common()
        kf_max, w_max = covisible_keyframes[0]

        with self._lock_connections:
            self.connected_keyframes_weights = viewing_keyframes

            if w_max >= Parameters.kMinNumOfCovisiblePointsForCreatingConnection:
                self.ordered_keyframes_weights = OrderedDict()
                for kf, w in covisible_keyframes:
                    if w >= Parameters.kMinNumOfCovisiblePointsForCreatingConnection:
                        kf.add_connection_no_lock_(self, w)
                        self.ordered_keyframes_weights[kf] = w
                    else:
                        break
            else:
                self.ordered_keyframes_weights = OrderedDict([(kf_max, w_max)])
                kf_max.add_connection_no_lock_(self, w_max)

            if (
                self.is_first_connection
                and self.kid != 0
                and kf_max is not None
                and kf_max is not self
                and not kf_max.is_bad()
            ):
                self.set_parent_no_lock_(kf_max)
                self.is_first_connection = False

    def Tcp(self):
        with self._lock_connections:
            return self._pose_Tcp.get_matrix()

    def is_bad(self) -> bool:
        with self._lock_connections:
            return self._is_bad

    def compute_bow(self, vocabulary):
        from visual_slam.orbslam.slam.bow import compute_bow_for_frame

        return compute_bow_for_frame(self, vocabulary)

    def set_not_erase(self) -> None:
        with self._lock_connections:
            self.not_to_erase = True

    def set_erase(self) -> None:
        should_set_bad = False
        with self._lock_connections:
            if len(self.loop_edges) == 0:
                self.not_to_erase = False
            if self.to_be_erased:
                should_set_bad = True
        if should_set_bad:
            self.set_bad()

    def set_bad(self) -> None:
        """Mark this keyframe bad and detach its graph and point links."""
        with self._lock_connections:
            if self.kid == 0:
                return

            if self.not_to_erase:
                self.to_be_erased = True
                return

            connected = self.get_connected_keyframes_no_lock_()

        for kf_connected in connected:
            kf_connected.erase_connection(self)

        # Remove observations from map points.
        for idx, point in enumerate(list(self.points)):
            if point is not None:
                point.remove_observation(self, idx)
                self.points[idx] = None

        with self._lock_connections:
            self.reset_covisibility()

            if self.parent is not None:
                try:
                    self._pose_Tcp.update(self.Tcw() @ self.parent.Twc())
                except Exception:
                    pass
                self.parent.erase_child_no_lock_(self)

            self.children.clear()
            self._is_bad = True

        if self.map is not None:
            remove_fn = getattr(self.map, "remove_keyframe", None)
            if remove_fn is not None:
                remove_fn(self)

    def __repr__(self) -> str:
        return f"KeyFrame(kid={self.kid}, frame_id={self.id}, kps={len(self.kps)}, bad={self._is_bad})"

    def get_matched_good_points_and_idxs(self):
        """

        Returns:
            list[(MapPoint, keypoint_idx)]

        The index must be the original keypoint index in self.points, not the
        compact index of get_points().
        """
        pairs = []
        points = getattr(self, "points", [])
        outliers = getattr(self, "outliers", None)

        for idx, p in enumerate(points):
            if p is None:
                continue
            if hasattr(p, "is_bad") and p.is_bad():
                continue
            if outliers is not None and idx < len(outliers) and bool(outliers[idx]):
                continue
            pairs.append((p, idx))

        return pairs

    def get_matched_good_points(self):
        """Return non-bad matched map points."""
        return [p for p, _ in self.get_matched_good_points_and_idxs()]

    def get_matched_good_points_idxs(self):
        """Return original keypoint indices of non-bad matched map points."""
        return np.asarray(
            [idx for _, idx in self.get_matched_good_points_and_idxs()],
            dtype=np.int32,
        )

    def num_tracked_points(self, min_num_observations=0):
        """Count tracked map points with at least min_num_observations."""
        count = 0

        for p, _ in self.get_matched_good_points_and_idxs():
            if p is None:
                continue
            if hasattr(p, "is_bad") and p.is_bad():
                continue
            if min_num_observations > 0 and p.num_observations() < min_num_observations:
                continue
            count += 1

        return count

    def check_replaced_map_points(self):
        """
        Replace frame associations when a MapPoint has been substituted.

        mapping / fusion can replace map points between frames.
        """
        replaced = 0

        points = getattr(self, "points", [])
        for idx, p in enumerate(list(points)):
            if p is None:
                continue

            replacement = None

            if hasattr(p, "get_replacement"):
                try:
                    replacement = p.get_replacement()
                except Exception:
                    replacement = None
            # Fallback: check .replacement attribute even when get_replacement() returned None
            if replacement is None and hasattr(p, "replacement"):
                try:
                    replacement = p.replacement
                except Exception:
                    replacement = None

            if replacement is not None and replacement is not p:
                points[idx] = replacement
                try:
                    replacement.add_frame_view(self, idx)
                except Exception:
                    pass
                replaced += 1

        return replaced

    def release_depth_image(self) -> None:
        self.depth_img = None

    def release_rgb_image(self) -> None:
        self.img = None
        self.img_right = None

    def release_heavy_data(self, release_rgb=False, release_depth=True, release_kd=False) -> None:
        if release_rgb:
            self.release_rgb_image()
        if release_depth:
            self.release_depth_image()
        if release_kd:
            self.kd = None

    def heavy_memory_bytes(self) -> int:
        return super().heavy_memory_bytes()
