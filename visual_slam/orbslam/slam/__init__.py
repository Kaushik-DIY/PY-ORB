"""
ORB-SLAM RGB-D core exports.
This package re-exports the camera, map, tracking, mapping, and optimization modules.
"""

from visual_slam.orbslam.slam.sensor_types import (
    DatasetEnvironmentType,
    DatasetType,
    SensorType,
    get_sensor_type,
    is_depth_available,
    is_monocular,
    is_rgbd,
    is_stereo,
)
from visual_slam.orbslam.slam.slam_commons import SlamState
from visual_slam.orbslam.slam.config_parameters import Parameters, OrbSlamSettings
from visual_slam.orbslam.slam.camera_pose import CameraPose
from visual_slam.orbslam.slam.camera import CameraType, CameraUtils, Camera, PinholeCamera, fov2focal, focal2fov
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared, SlamFeatureManagerInfo
from visual_slam.orbslam.slam.frame import FrameBase, Frame, detect_and_compute, match_frames, are_map_points_visible_in_frame
from visual_slam.orbslam.slam.map_point import MapPointBase, MapPoint
from visual_slam.orbslam.slam.keyframe import KeyFrameGraph, KeyFrame
from visual_slam.orbslam.slam.map import Map, LocalCovisibilityMap, OrderedSetLite, MapStateData
from visual_slam.orbslam.slam.optimizer_g2o import OptimizerResult, pose_optimization, bundle_adjustment, local_bundle_adjustment, global_bundle_adjustment
from visual_slam.orbslam.slam.global_ba import GlobalBAResult, GlobalBundleAdjuster
from visual_slam.orbslam.slam.motion_model import MotionModelBase, MotionModel, MotionModelDamping
from visual_slam.orbslam.slam.rotation_histogram import RotationHistogram
from visual_slam.orbslam.slam.geometry_matchers import ProjectionMatcher, EpipolarMatcher
from visual_slam.orbslam.slam.bow import DBoW3Vocabulary, BoWBackendStatus, get_bow_backend_status, get_default_vocabulary_path, load_default_vocabulary
from visual_slam.orbslam.slam.bow_matcher import BoWGuidedMatcher, BoWMatchDiagnostics, BoWMatchResult
from visual_slam.orbslam.slam.keyframe_database import KeyFrameDatabase
from visual_slam.orbslam.slam.relocalizer import Relocalizer, TemporaryRelocalizationKeyFrameDatabase, PnPResult
from visual_slam.orbslam.slam.tracking_core import TrackingCore
from visual_slam.orbslam.slam.tracking import TrackingHistory, Tracking
from visual_slam.orbslam.slam.local_mapping_core import LocalMappingCore
from visual_slam.orbslam.slam.local_mapping import LocalMapping


def __getattr__(name):
    if name in {"Slam", "SlamMode"}:
        from visual_slam.orbslam.slam.slam import Slam, SlamMode

        globals()["Slam"] = Slam
        globals()["SlamMode"] = SlamMode
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DatasetEnvironmentType",
    "DatasetType",
    "SensorType",
    "SlamState",
    "Parameters",
    "OrbSlamSettings",
    "CameraPose",
    "CameraType",
    "CameraUtils",
    "Camera",
    "PinholeCamera",
    "fov2focal",
    "focal2fov",
    "FeatureTrackerShared",
    "SlamFeatureManagerInfo",
    "FrameBase",
    "Frame",
    "detect_and_compute",
    "match_frames",
    "are_map_points_visible_in_frame",
    "MapPointBase",
    "MapPoint",
    "KeyFrameGraph",
    "KeyFrame",
    "Map",
    "LocalCovisibilityMap",
    "OrderedSetLite",
    "MapStateData",
    "OptimizerResult",
    "pose_optimization",
    "bundle_adjustment",
    "local_bundle_adjustment",
    "global_bundle_adjustment",
    "GlobalBAResult",
    "GlobalBundleAdjuster",
    "MotionModelBase",
    "MotionModel",
    "MotionModelDamping",
    "RotationHistogram",
    "ProjectionMatcher",
    "EpipolarMatcher",
    "DBoW3Vocabulary",
    "BoWBackendStatus",
    "BoWGuidedMatcher",
    "BoWMatchDiagnostics",
    "BoWMatchResult",
    "get_bow_backend_status",
    "get_default_vocabulary_path",
    "load_default_vocabulary",
    "KeyFrameDatabase",
    "Relocalizer",
    "TemporaryRelocalizationKeyFrameDatabase",
    "PnPResult",
    "TrackingCore",
    "TrackingHistory",
    "Tracking",
    "LocalMappingCore",
    "LocalMapping",
    "Slam",
    "SlamMode",
    "get_sensor_type",
    "is_depth_available",
    "is_monocular",
    "is_rgbd",
    "is_stereo",
]
