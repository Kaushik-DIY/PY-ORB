"""
Frame representation for RGB-D tracking and mapping.
This module stores image-derived data, pose state, and projection/visibility helpers.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Optional

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.slam.camera import Camera
from visual_slam.orbslam.slam.camera_pose import CameraPose
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.config_parameters import Parameters


kMinDepth = Parameters.kMinDepth

# Provide a minimal radius-search structure when scipy is unavailable.
class SimpleKDTree:
    """Small query_ball_point-compatible fallback for scipy.spatial.cKDTree."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    def query_ball_point(self, x, r):
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 1:
            rr = float(r)
            d = np.linalg.norm(self.points - x.reshape(1, 2), axis=1)
            return list(np.flatnonzero(d <= rr))

        if np.isscalar(r):
            radii = np.full(len(x), float(r), dtype=np.float64)
        else:
            radii = np.asarray(r, dtype=np.float64).reshape(-1)

        results = []
        for xi, ri in zip(x, radii):
            d = np.linalg.norm(self.points - xi.reshape(1, 2), axis=1)
            results.append(list(np.flatnonzero(d <= ri)))
        return results


def make_kdtree_from_keypoints(kps):
    pts = np.array([kp.pt for kp in kps], dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        pts = np.empty((0, 2), dtype=np.float64)

    try:
        from scipy.spatial import cKDTree
        return cKDTree(pts)
    except Exception:
        return SimpleKDTree(pts)


def ensure_frame_feature_arrays(frame) -> None:
    kps = getattr(frame, "kps", getattr(frame, "keypoints", []))

    if not hasattr(frame, "octaves") or len(getattr(frame, "octaves", [])) != len(kps):
        frame.octaves = np.array([max(0, int(getattr(kp, "octave", 0))) for kp in kps], dtype=np.int32)

    if not hasattr(frame, "angles") or len(getattr(frame, "angles", [])) != len(kps):
        frame.angles = np.array([float(getattr(kp, "angle", -1.0)) for kp in kps], dtype=np.float32)

    if not hasattr(frame, "kps_ur"):
        frame.kps_ur = getattr(frame, "uRs", np.full(len(kps), -1.0, dtype=np.float32))

    if getattr(frame, "kd", None) is None:
        frame.kd = make_kdtree_from_keypoints(kps)



def detect_and_compute(img: np.ndarray, left: bool = True, mask=None):
    if left:
        if FeatureTrackerShared.feature_tracker is None:
            raise RuntimeError("FeatureTrackerShared.feature_tracker is not set.")
        return FeatureTrackerShared.feature_tracker.detectAndCompute(img, mask)

    if FeatureTrackerShared.feature_tracker_right is None:
        raise RuntimeError("FeatureTrackerShared.feature_tracker_right is not set.")
    return FeatureTrackerShared.feature_tracker_right.detectAndCompute(img, mask)


def _as_points_array(points_or_map_points) -> np.ndarray:
    pts = []
    for p in points_or_map_points:
        if p is None:
            continue
        if hasattr(p, "pt") and callable(p.pt):
            pts.append(np.asarray(p.pt(), dtype=np.float64).reshape(3))
        elif hasattr(p, "position"):
            pts.append(np.asarray(p.position, dtype=np.float64).reshape(3))
        elif hasattr(p, "position_world"):
            pts.append(np.asarray(p.position_world, dtype=np.float64).reshape(3))
        else:
            pts.append(np.asarray(p, dtype=np.float64).reshape(3))
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.ascontiguousarray(pts, dtype=np.float64)


# Store the common camera, pose, and projection state shared by frame types.
class FrameBase:
    """Base object for camera intrinsics, pose, projection, and visibility."""

    _id = 0
    _id_lock = Lock()

    def __init__(
        self,
        camera: Camera,
        pose=None,
        id: Optional[int] = None,
        timestamp: Optional[float] = None,
        img_id: Optional[int] = None,
    ):
        self._lock_pose = Lock()
        self.camera = camera

        if pose is None:
            self._pose = CameraPose()
        else:
            self._pose = CameraPose(pose)

        if id is not None:
            self.id = int(id)
        else:
            with FrameBase._id_lock:
                self.id = FrameBase._id
                FrameBase._id += 1

        self.timestamp = timestamp
        self.img_id = img_id
        self.median_depth = -1.0
        self.fov_center_c = None
        self.fov_center_w = None

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return isinstance(rhs, FrameBase) and self.id == rhs.id

    def __lt__(self, rhs):
        return self.id < rhs.id

    def __le__(self, rhs):
        return self.id <= rhs.id

    @staticmethod
    def next_id() -> int:
        with FrameBase._id_lock:
            return FrameBase._id

    @staticmethod
    def set_id(id_value: int) -> None:
        with FrameBase._id_lock:
            FrameBase._id = int(id_value)

    @property
    def width(self):
        return self.camera.width

    @property
    def height(self):
        return self.camera.height

    def isometry3d(self):
        with self._lock_pose:
            return self._pose.isometry3d

    def Tcw(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.Tcw.copy()

    def Twc(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.get_inverse_matrix().copy()

    def Rcw(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.Rcw.copy()

    def Rwc(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.Rwc.copy()

    def tcw(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.tcw.copy()

    def Ow(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.Ow.copy()

    def pose(self) -> np.ndarray:
        return self.Tcw()

    def quaternion(self):
        with self._lock_pose:
            return self._pose.quaternion

    def orientation(self):
        with self._lock_pose:
            return self._pose.orientation

    def position(self):
        with self._lock_pose:
            return self._pose.position

    def update_pose(self, pose) -> None:
        with self._lock_pose:
            self._pose.set(pose)
            self._update_fov_center_world_no_lock()

    def update_translation(self, tcw: np.ndarray) -> None:
        with self._lock_pose:
            self._pose.set_translation(tcw)
            self._update_fov_center_world_no_lock()

    def update_rotation_and_translation(self, Rcw: np.ndarray, tcw: np.ndarray) -> None:
        with self._lock_pose:
            self._pose.set_from_rotation_and_translation(Rcw, tcw)
            self._update_fov_center_world_no_lock()

    def _update_fov_center_world_no_lock(self) -> None:
        if self.fov_center_c is not None:
            self.fov_center_w = (
                self._pose.Rwc @ np.asarray(self.fov_center_c, dtype=np.float64).reshape(3, 1)
                + self._pose.Ow.reshape(3, 1)
            )

    def transform_point(self, pw: np.ndarray) -> np.ndarray:
        with self._lock_pose:
            pw = np.asarray(pw, dtype=np.float64).reshape(3)
            return (self._pose.Rcw @ pw) + self._pose.tcw

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        with self._lock_pose:
            points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
            return (self._pose.Rcw @ points.T + self._pose.tcw.reshape(3, 1)).T

    def project_points(self, points: np.ndarray, do_stereo_project: bool = False):
        pcs = self.transform_points(points)
        if do_stereo_project:
            return self.camera.project_stereo(pcs)
        return self.camera.project(pcs)

    def project_map_points(self, map_points, do_stereo_project: bool = False):
        points = _as_points_array(map_points)
        if len(points) == 0:
            if do_stereo_project:
                return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
        return self.project_points(points, do_stereo_project=do_stereo_project)

    def project_point(self, pw: np.ndarray, do_stereo_project: bool = False):
        pc = self.transform_point(pw)
        if do_stereo_project:
            proj, zs = self.camera.project_stereo(pc.reshape(1, 3))
        else:
            proj, zs = self.camera.project(pc.reshape(1, 3))
        return proj.reshape(-1), float(zs[0])

    def is_in_image(self, uv: np.ndarray, z: float) -> bool:
        return bool(self.camera.is_in_image(uv, z))

    def are_in_image(self, uvs: np.ndarray, zs: np.ndarray) -> np.ndarray:
        return self.camera.are_in_image(uvs, zs)

    def are_visible(self, map_points, do_stereo_project: bool = False):
        projs, depths = self.project_map_points(map_points, do_stereo_project=do_stereo_project)
        pts = _as_points_array(map_points)
        if len(pts) == 0:
            return np.empty((0,), dtype=bool), projs, depths, np.empty((0,), dtype=np.float64)
        Ow = self.Ow()
        dists = np.linalg.norm(pts - Ow.reshape(1, 3), axis=1)
        visible = self.are_in_image(projs[:, :2], depths) & (depths > kMinDepth)
        return visible, projs, depths, dists


# Represent one RGB-D frame together with extracted features and map links.
class Frame(FrameBase):
    """

    Important fields preserved for later porting:
    - kps / kpsu
    - des
    - depths
    - uRs
    - points
    - outliers
    - idxs
    - kd placeholder
    """

    def __init__(
        self,
        camera: Camera,
        img: Optional[np.ndarray] = None,
        depth_img: Optional[np.ndarray] = None,
        pose=None,
        id: Optional[int] = None,
        timestamp: Optional[float] = None,
        img_id: Optional[int] = None,
        img_right: Optional[np.ndarray] = None,
        mask=None,
    ):
        super().__init__(camera=camera, pose=pose, id=id, timestamp=timestamp, img_id=img_id)

        self.img = img
        self.img_right = img_right
        self.depth_img = depth_img
        self.mask = mask

        self.kps: list[cv2.KeyPoint] = []
        self.kpsu: list[cv2.KeyPoint] = []
        self.des: np.ndarray = np.empty((0, 32), dtype=np.uint8)

        self.kps_r: list[cv2.KeyPoint] = []
        self.des_r: np.ndarray = np.empty((0, 32), dtype=np.uint8)

        self.depths: np.ndarray = np.empty((0,), dtype=np.float32)
        self.uRs: np.ndarray = np.empty((0,), dtype=np.float32)
        self.octaves: np.ndarray = np.empty((0,), dtype=np.int32)
        self.angles: np.ndarray = np.empty((0,), dtype=np.float32)
        self.sizes: np.ndarray = np.empty((0,), dtype=np.float32)

        self.points: list[Any | None] = []
        self.outliers: np.ndarray = np.empty((0,), dtype=bool)
        self.idxs: np.ndarray = np.empty((0,), dtype=np.int32)

        self.kd = None
        self._is_deleted = False
        self.is_keyframe = False
        self.kf_ref = None

        # BoW/global descriptor fields used by relocalization and loop closing.
        self.g_des = None
        self.f_des = None
        self.bow_vector = None
        self.feature_vector = None
        self.is_blurry = False
        self.laplacian_var = None

        if img is not None:
            self.extract_features(mask=mask)

        if depth_img is not None and len(self.kps) > 0:
            self.set_depth_img(depth_img)

    @property
    def num_kps(self) -> int:
        return len(self.kps)

    @property
    def keypoints(self):
        return self.kps

    @property
    def descriptors(self):
        return self.des

    def ensure_contiguous_arrays(self) -> None:
        self.des = np.ascontiguousarray(self.des)
        self.depths = np.ascontiguousarray(self.depths)
        self.uRs = np.ascontiguousarray(self.uRs)
        self.octaves = np.ascontiguousarray(self.octaves)
        self.angles = np.ascontiguousarray(self.angles)
        self.sizes = np.ascontiguousarray(self.sizes)
        self.outliers = np.ascontiguousarray(self.outliers)
        self.idxs = np.ascontiguousarray(self.idxs)

    def extract_features(self, mask=None) -> None:
        kps, des = detect_and_compute(self.img, left=True, mask=mask)
        self.kps = list(kps)
        self.kpsu = list(kps)
        self.des = des if des is not None else np.empty((0, 32), dtype=np.uint8)

        n = len(self.kps)
        self.depths = np.full(n, -1.0, dtype=np.float32)
        self.uRs = np.full(n, -1.0, dtype=np.float32)
        self.points = [None] * n
        self.outliers = np.zeros(n, dtype=bool)
        self.idxs = np.arange(n, dtype=np.int32)
        self.octaves = np.array([max(0, int(getattr(kp, 'octave', 0))) for kp in self.kps], dtype=np.int32)
        self.angles = np.array([float(getattr(kp, 'angle', -1.0)) for kp in self.kps], dtype=np.float32)
        self.sizes = np.array([float(getattr(kp, 'size', 0.0)) for kp in self.kps], dtype=np.float32)
        pts = np.array([kp.pt for kp in self.kps], dtype=np.float64).reshape(-1, 2)
        self.kpsn = self.camera.unproject_points(pts) if len(pts) > 0 else np.empty((0, 2), dtype=np.float64)
        self.kps_ur = self.uRs
        self.kd = make_kdtree_from_keypoints(self.kps)
        self.ensure_contiguous_arrays()

    def set_img_right(self, img_right: np.ndarray, mask=None) -> None:
        self.img_right = img_right
        kps_r, des_r = detect_and_compute(img_right, left=False, mask=mask)
        self.kps_r = list(kps_r)
        self.des_r = des_r if des_r is not None else np.empty((0, 32), dtype=np.uint8)

    def set_depth_img(self, depth_img: np.ndarray) -> None:
        self.depth_img = depth_img
        self.compute_depths_from_depth_img()

    def compute_depths_from_depth_img(self) -> None:
        n = len(self.kps)
        self.depths = np.full(n, -1.0, dtype=np.float32)
        self.uRs = np.full(n, -1.0, dtype=np.float32)

        if self.depth_img is None or n == 0:
            return

        h, w = self.depth_img.shape[:2]

        for i, kp in enumerate(self.kps):
            u = int(round(kp.pt[0]))
            v = int(round(kp.pt[1]))

            if u < 0 or u >= w or v < 0 or v >= h:
                continue

            raw_depth = float(self.depth_img[v, u])
            depth_m = raw_depth * float(self.camera.depth_factor)

            if not np.isfinite(depth_m) or depth_m <= kMinDepth:
                continue

            self.depths[i] = depth_m

            if self.camera.bf is not None:
                self.uRs[i] = float(kp.pt[0] - self.camera.bf / depth_m)

        self.ensure_contiguous_arrays()

    def get_point_match(self, idx: int):
        return self.points[idx]

    def set_point_match(self, p, idx: int) -> None:
        self.points[idx] = p

    def remove_point_match(self, idx: int) -> None:
        self.points[idx] = None

    def replace_point_match(self, old_point, new_point) -> int:
        count = 0
        for i, p in enumerate(self.points):
            if p is old_point:
                self.points[i] = new_point
                count += 1
        return count

    def remove_point(self, point) -> int:
        count = 0
        for i, p in enumerate(self.points):
            if p is point:
                self.points[i] = None
                count += 1
        return count

    def reset_points(self) -> None:
        self.points = [None] * len(self.kps)
        self.outliers = np.zeros(len(self.kps), dtype=bool)

    def remove_frame_views(self, idxs=None) -> int:
        if idxs is None:
            idxs = range(len(self.points))
        idxs = list(np.asarray(idxs, dtype=np.int32).reshape(-1))

        count = 0
        for idx in idxs:
            if idx < 0 or idx >= len(self.points):
                continue
            p = self.points[idx]
            if p is not None:
                try:
                    p.remove_frame_view(self, idx)
                except Exception:
                    self.points[idx] = None
                count += 1
        return count

    def clean_outlier_map_points(self) -> int:
        num_valid = 0

        for idx, p in enumerate(list(self.points)):
            if p is None:
                continue

            is_outlier = idx < len(self.outliers) and bool(self.outliers[idx])
            is_bad = hasattr(p, "is_bad") and p.is_bad()

            if is_outlier or is_bad:
                try:
                    p.remove_frame_view(self, idx)
                except Exception:
                    self.points[idx] = None
            else:
                num_valid += 1

        return num_valid

    def clean_bad_map_points(self) -> int:
        removed = 0

        for idx, p in enumerate(list(self.points)):
            if p is not None and hasattr(p, "is_bad") and p.is_bad():
                self.points[idx] = None
                removed += 1

        return removed

    def get_points(self):
        # Return full list preserving keypoint-index alignment (pyslam frame.py:938).
        # None entries mean "no map point for this keypoint" — callers must handle None.
        return list(self.points)

    def get_matched_points(self):
        return [p for p in self.points if p is not None]

    def get_matched_points_idxs(self) -> np.ndarray:
        return np.array([i for i, p in enumerate(self.points) if p is not None], dtype=np.int32)

    def get_unmatched_points_idxs(self) -> np.ndarray:
        return np.array([i for i, p in enumerate(self.points) if p is None], dtype=np.int32)

    def compute_bow(self, vocabulary):
        from visual_slam.orbslam.slam.bow import compute_bow_for_frame

        return compute_bow_for_frame(self, vocabulary)


    def get_matched_good_points_and_idxs(self):
        """

        Returns:
            list[(MapPoint, keypoint_idx)]

        The index is the original keypoint index in self.points, not the compact
        index of get_points().
        """
        pairs = []
        outliers = getattr(self, "outliers", None)

        for idx, p in enumerate(getattr(self, "points", [])):
            if p is None:
                continue
            if hasattr(p, "is_bad") and p.is_bad():
                continue
            if outliers is not None and idx < len(outliers) and bool(outliers[idx]):
                continue
            pairs.append((p, idx))

        return pairs

    def get_matched_good_points(self):
        """Return non-bad, non-outlier matched map points."""
        return [p for p, _ in self.get_matched_good_points_and_idxs()]

    def get_matched_good_points_idxs(self) -> np.ndarray:
        """Return original keypoint indices of non-bad, non-outlier matched map points."""
        return np.asarray(
            [idx for _, idx in self.get_matched_good_points_and_idxs()],
            dtype=np.int32,
        )

    def get_matched_inlier_points(self):
        return self.get_matched_good_points()

    def num_tracked_points(self, min_num_observations=0):
        """
        Count tracked map points with at least min_num_observations.

        reference-keyframe points remain supported.
        """
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

        mapping/fusion can replace map points between frames.
        """
        replaced = 0
        points = getattr(self, "points", [])

        for idx, p in enumerate(list(points)):
            if p is None:
                continue

            replacement = None

            if hasattr(p, "get_replaced"):
                try:
                    replacement = p.get_replaced()
                except Exception:
                    replacement = None
            elif hasattr(p, "get_replacement"):
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

            if replacement is None or replacement is p:
                continue
            if hasattr(replacement, "is_bad") and replacement.is_bad():
                continue

            points[idx] = replacement

            try:
                replacement.add_frame_view(self, idx)
            except Exception:
                # frame-side replacement. add_frame_view may no-op if the
                # replacement already owns the frame view.
                pass

            replaced += 1

        return replaced

    def unproject_points(self, idxs):
        idxs = np.asarray(idxs, dtype=np.int32).reshape(-1)
        pts = np.array([self.kpsu[int(i)].pt for i in idxs], dtype=np.float64).reshape(-1, 2)
        return self.camera.unproject_points(pts)

    def unproject_points_3d(self, idxs, transform_in_world: bool = True):
        idxs = np.asarray(idxs, dtype=np.int32).reshape(-1)

        pts3d = np.zeros((len(idxs), 3), dtype=np.float64)
        valid = np.zeros(len(idxs), dtype=bool)

        if len(idxs) == 0:
            return pts3d, valid

        for out_i, idx in enumerate(idxs):
            if idx < 0 or idx >= len(self.kpsu):
                continue
            if idx >= len(self.depths):
                continue

            depth = float(self.depths[idx])
            if not np.isfinite(depth) or depth <= kMinDepth:
                continue

            uv = np.array(self.kpsu[idx].pt, dtype=np.float64)
            pc = self.camera.unproject_3d(uv, depth)

            if transform_in_world:
                Rwc = self.Rwc()
                Ow = self.Ow()
                pw = Rwc @ pc.reshape(3) + Ow.reshape(3)
                pts3d[out_i] = pw
            else:
                pts3d[out_i] = pc.reshape(3)

            valid[out_i] = True

        return pts3d, valid

    def release_images(self, release_rgb=True, release_right=True, release_depth=True, release_mask=True) -> None:
        if release_rgb:
            self.img = None
        if release_right:
            self.img_right = None
        if release_depth:
            self.depth_img = None
        if release_mask:
            self.mask = None

    def release_heavy_data(self, release_images=True, release_kd=True, release_descriptors=False, release_points=False) -> None:
        if release_images:
            self.release_images()
        if release_kd:
            self.kd = None
        if release_descriptors:
            self.des = np.empty((0, 32), dtype=np.uint8)
            self.g_des = None
            self.f_des = None
        if release_points:
            self.reset_points()

    def heavy_memory_bytes(self) -> int:
        total = 0
        if self.img is not None:
            total += self.img.nbytes
        if self.img_right is not None:
            total += self.img_right.nbytes
        if self.depth_img is not None:
            total += self.depth_img.nbytes
        if hasattr(self, 'des') and self.des is not None:
            total += self.des.nbytes
        if hasattr(self, 'depths') and self.depths is not None:
            total += self.depths.nbytes
        if hasattr(self, 'uRs') and self.uRs is not None:
            total += self.uRs.nbytes
        if hasattr(self, 'octaves') and self.octaves is not None:
            total += self.octaves.nbytes
        if hasattr(self, 'angles') and self.angles is not None:
            total += self.angles.nbytes
        if hasattr(self, 'sizes') and self.sizes is not None:
            total += self.sizes.nbytes
        if hasattr(self, 'points') and self.points is not None:
            total += len(self.points) * 8
        if hasattr(self, 'outliers') and self.outliers is not None:
            total += self.outliers.nbytes
        return total

    def delete(self) -> None:
        self._is_deleted = True
        self.img = None
        self.img_right = None
        self.depth_img = None
        self.kd = None

    def __repr__(self) -> str:
        return f"Frame(id={self.id}, t={self.timestamp}, kps={len(self.kps)})"


def are_map_points_visible_in_frame(
    frame: FrameBase,
    map_points,
    do_stereo_project: bool = False,
    check_positive_depth: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project map points into a frame and return visibility mask, projections, depths.
    """
    if len(map_points) == 0:
        dim = 3 if do_stereo_project else 2
        return (
            np.empty((0,), dtype=bool),
            np.empty((0, dim), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    projs, depths = frame.project_map_points(map_points, do_stereo_project=do_stereo_project)
    visible = frame.are_in_image(projs[:, :2], depths)

    if check_positive_depth:
        visible &= depths > kMinDepth

    return visible, projs, depths


def are_map_points_visible(frame: FrameBase, map_points, **kwargs):
    return are_map_points_visible_in_frame(frame, map_points, **kwargs)


def prepare_input_data_for_sim3solver(f1: "FrameBase", f2: "FrameBase", idxs1, idxs2):
    """Build 3D correspondences and per-keypoint sigma2 arrays for Sim3Solver.

    Mirrors pyslam frame.py:1878 exactly.
    """
    level_sigmas2 = FeatureTrackerShared.feature_manager.level_sigmas2 if FeatureTrackerShared.feature_manager is not None else [1.0]
    points_3d_1 = []
    points_3d_2 = []
    sigmas2_1 = []
    sigmas2_2 = []
    idxs1_out = []
    idxs2_out = []
    for i1, i2 in zip(idxs1, idxs2):
        p1 = f1.get_point_match(int(i1))
        if p1 is None or p1.is_bad():
            continue
        p2 = f2.get_point_match(int(i2))
        if p2 is None or p2.is_bad():
            continue
        points_3d_1.append(p1.pt())
        points_3d_2.append(p2.pt())
        lv1 = int(f1.octaves[int(i1)]) if hasattr(f1, "octaves") and f1.octaves is not None else 0
        lv2 = int(f2.octaves[int(i2)]) if hasattr(f2, "octaves") and f2.octaves is not None else 0
        sigmas2_1.append(float(level_sigmas2[min(lv1, len(level_sigmas2) - 1)]))
        sigmas2_2.append(float(level_sigmas2[min(lv2, len(level_sigmas2) - 1)]))
        idxs1_out.append(int(i1))
        idxs2_out.append(int(i2))
    return (
        np.ascontiguousarray(points_3d_1, dtype=np.float32) if points_3d_1 else np.zeros((0, 3), dtype=np.float32),
        np.ascontiguousarray(points_3d_2, dtype=np.float32) if points_3d_2 else np.zeros((0, 3), dtype=np.float32),
        np.ascontiguousarray(sigmas2_1, dtype=np.float32) if sigmas2_1 else np.zeros(0, dtype=np.float32),
        np.ascontiguousarray(sigmas2_2, dtype=np.float32) if sigmas2_2 else np.zeros(0, dtype=np.float32),
        np.ascontiguousarray(idxs1_out, dtype=np.int32) if idxs1_out else np.zeros(0, dtype=np.int32),
        np.ascontiguousarray(idxs2_out, dtype=np.int32) if idxs2_out else np.zeros(0, dtype=np.int32),
    )


def match_frames(frame_ref: Frame, frame_cur: Frame, ratio_test: float | None = None):
    """
    Minimal descriptor matching helper.

    subset delegates to the shared ORB matcher and returns index arrays.
    """
    if FeatureTrackerShared.feature_matcher is None:
        raise RuntimeError("FeatureTrackerShared.feature_matcher is not set.")

    ratio = Parameters.kFeatureMatchDefaultRatioTest if ratio_test is None else ratio_test

    return FeatureTrackerShared.feature_matcher.match(
        frame_ref.img,
        frame_cur.img,
        frame_ref.des,
        frame_cur.des,
        frame_ref.kps,
        frame_cur.kps,
        ratio_test=ratio,
    )
