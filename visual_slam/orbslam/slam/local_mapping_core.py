"""
Core local-mapping operations.
This module handles triangulation, culling, local BA, and local map fusion.
"""

from __future__ import annotations

from types import SimpleNamespace

import g2o
import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.geometry_matchers import ProjectionMatcher
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.map import Map
from visual_slam.orbslam.slam.map_point import MapPoint
from visual_slam.orbslam.slam.sensor_types import SensorType


kNumMinObsForKeyFrameTrackedPoints = 3


def _make_flag(value=False):
    if hasattr(g2o, "Flag"):
        try:
            return g2o.Flag(bool(value))
        except Exception:
            pass
    return SimpleNamespace(value=bool(value))


def _point_first_kid(point: MapPoint, fallback_kid: int) -> int:
    first_kid = getattr(point, "first_kid", None)
    if first_kid is not None:
        return int(first_kid)

    observations = list(point.observations())
    if len(observations) == 0:
        return int(fallback_kid)

    return min(int(kf.kid) for kf, _ in observations)


# Implement the core map-growth and local-refinement operations.
class LocalMappingCore:
    print = staticmethod(lambda *args, **kwargs: None)

    def __init__(self, map: Map, sensor_type):
        self.map = map
        self.sensor_type = sensor_type
        self.recently_added_points = set()
        self.kf_cur: KeyFrame | None = None
        self.kid_last_BA = -1
        self.opt_abort_flag = _make_flag(False)
        self.mp_opt_abort_flag = SimpleNamespace(value=False)

    def reset(self):
        self.recently_added_points.clear()

    def add_points(self, points: list[MapPoint]):
        self.recently_added_points.update(points)

    def remove_points(self, points: list[MapPoint]):
        self.recently_added_points.difference_update(points)

    def set_opt_abort_flag(self, value):
        value = bool(value)
        if getattr(self.opt_abort_flag, "value", None) != value:
            self.opt_abort_flag.value = value
        self.mp_opt_abort_flag.value = value

    def local_BA(self):
        err = self.map.locally_optimize(
            kf_ref=self.kf_cur,
            abort_flag=self.opt_abort_flag,
            mp_abort_flag=self.mp_opt_abort_flag,
        )
        num_kf_ref_tracked_points = self.kf_cur.num_tracked_points(kNumMinObsForKeyFrameTrackedPoints)
        return err, num_kf_ref_tracked_points

    def large_window_BA(self):
        self.kid_last_BA = self.kf_cur.kid
        err, _ = self.map.optimize(
            local_window_size=Parameters.kLargeBAWindowSize,
            abort_flag=self.opt_abort_flag,
        )
        return err

    def process_new_keyframe(self):
        LocalMappingCore.print(">>>> updating map points ...")

        good_points_and_idxs = self.kf_cur.get_matched_good_points_and_idxs()

        for p, idx in good_points_and_idxs:
            added = p.add_observation(self.kf_cur, idx)
            if added:
                p.update_info()
            else:
                # New RGB-D/stereo points inserted by Tracking may already have this observation.
                self.recently_added_points.add(p)

        LocalMappingCore.print(">>>> updating connections ...")
        self.kf_cur.update_connections()

    def cull_map_points(self):
        th_num_observations = 2
        if self.sensor_type != SensorType.MONOCULAR:
            th_num_observations = 3

        min_found_ratio = 0.25
        current_kid = self.kf_cur.kid
        remove_set = set()

        for p in list(self.recently_added_points):
            first_kid = _point_first_kid(p, current_kid)

            if p.is_bad():
                remove_set.add(p)
            elif p.get_found_ratio() < min_found_ratio:
                p.set_bad()
                self.map.remove_point(p)
                remove_set.add(p)
            elif (current_kid - first_kid) >= 2 and p.num_observations() <= th_num_observations:
                p.set_bad()
                self.map.remove_point(p)
                remove_set.add(p)
            elif (current_kid - first_kid) >= 3:
                remove_set.add(p)

        self.recently_added_points = self.recently_added_points - remove_set
        return len(remove_set)

    @staticmethod
    def check_remaining_fov_centers_max_distance(covisible_kfs, kf_to_remove, dist):
        remaining = [
            getattr(kf, "fov_center_w", None)
            for kf in covisible_kfs
            if kf is not kf_to_remove
        ]
        remaining = [r for r in remaining if r is not None]

        if len(remaining) == 0:
            return False

        pts = np.vstack([np.asarray(r).reshape(1, 3) for r in remaining])

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            distances, _ = tree.query(pts, k=2)
            return bool(np.all(distances[:, 1] < dist))
        except Exception:
            for i in range(len(pts)):
                d = np.linalg.norm(pts[i].reshape(1, 3) - pts, axis=1)
                d = np.sort(d)
                if len(d) < 2 or d[1] >= dist:
                    return False
            return True

    def cull_keyframes(self, use_fov_centers_based_kf_generation=False, max_fov_centers_distance=-1):
        LocalMappingCore.print(">>>> culling keyframes...")

        num_culled_keyframes = 0
        th_num_observations = 3
        covisible_kfs = self.kf_cur.get_covisible_keyframes()

        for kf in covisible_kfs:
            if kf.kid == 0:
                continue
            if kf.is_bad():
                continue

            kf_num_points = 0
            kf_num_redundant_observations = 0

            idxs_and_points = [
                (idx, p)
                for p, idx in kf.get_matched_good_points_and_idxs()
                if p is not None and not p.is_bad()
            ]

            for i, p in idxs_and_points:
                if self.sensor_type != SensorType.MONOCULAR:
                    if kf.depths is not None and (
                        kf.depths[i] > kf.camera.depth_threshold or kf.depths[i] < 0.0
                    ):
                        continue

                kf_num_points += 1

                if p.num_observations() > th_num_observations:
                    scale_level = int(kf.octaves[i])
                    p_num_observations = 0

                    for kf_j, idx in p.observations():
                        if kf_j is kf:
                            continue
                        if kf_j.is_bad():
                            continue

                        scale_level_i = int(kf_j.octaves[idx])

                        if scale_level_i <= scale_level + 1:
                            p_num_observations += 1

                        if p_num_observations >= th_num_observations:
                            break

                    if p_num_observations >= th_num_observations:
                        kf_num_redundant_observations += 1

            remove_kf = (
                kf_num_redundant_observations
                > Parameters.kKeyframeCullingRedundantObsRatio * max(kf_num_points, 1)
            ) and (kf_num_points > Parameters.kKeyframeCullingMinNumPoints)

            if remove_kf:
                parent = getattr(kf, "parent", None)
                if parent is not None:
                    delta_time_parent = abs(kf.timestamp - parent.timestamp)
                    if delta_time_parent < Parameters.kKeyframeMaxTimeDistanceInSecForCulling:
                        remove_kf = False

            if remove_kf and use_fov_centers_based_kf_generation:
                if not LocalMappingCore.check_remaining_fov_centers_max_distance(
                    covisible_kfs,
                    kf,
                    max_fov_centers_distance,
                ):
                    remove_kf = False

            if remove_kf:
                kf.set_bad()
                num_culled_keyframes += 1

        return num_culled_keyframes

    def fuse_map_points(self, descriptor_distance_sigma):
        total_fused_pts = 0

        num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesStereo
        if self.sensor_type == SensorType.MONOCULAR:
            num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesMonocular

        if hasattr(self.map, "local_map") and hasattr(self.map.local_map, "get_best_neighbors"):
            local_keyframes = self.map.local_map.get_best_neighbors(self.kf_cur, N=num_neighbors)
        else:
            local_keyframes = self.kf_cur.get_best_covisible_keyframes(num_neighbors)

        local_keyframes = [
            kf for kf in local_keyframes
            if kf is not None and kf is not self.kf_cur and not kf.is_bad()
        ]

        # 1. Fuse current keyframe points into neighbor keyframes.
        cur_points = self.kf_cur.get_matched_good_points()

        for kf in local_keyframes:
            total_fused_pts += ProjectionMatcher.search_and_fuse(
                cur_points,
                kf,
                max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
                max_descriptor_distance=descriptor_distance_sigma,
                ratio_test=Parameters.kMatchRatioTestMap,
            )

        # 2. Collect neighbor points and fuse them into current keyframe.
        fuse_candidates = []
        seen = set()

        for kf in local_keyframes:
            for p in kf.get_matched_good_points():
                if p is None or p.is_bad() or p in seen:
                    continue
                if p.is_in_keyframe(self.kf_cur):
                    continue
                seen.add(p)
                fuse_candidates.append(p)

        total_fused_pts += ProjectionMatcher.search_and_fuse(
            fuse_candidates,
            self.kf_cur,
            max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
            max_descriptor_distance=descriptor_distance_sigma,
            ratio_test=Parameters.kMatchRatioTestMap,
        )

        for p in self.kf_cur.get_matched_good_points():
            if p is not None and not p.is_bad():
                p.update_info()

        self.kf_cur.update_connections()

        return total_fused_pts
