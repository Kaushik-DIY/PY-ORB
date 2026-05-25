"""
RGB-D dataset IO exports.
This package exposes dataset readers, camera builders, and trajectory writers.
"""

from visual_slam.orbslam.io.tum_rgbd import (
    TumRgbdFrame,
    associate_tum_rgbd,
    load_tum_rgbd_associations,
    make_tum_rgbd_camera,
    save_tum_trajectory,
)
from visual_slam.orbslam.io.rgbd_dataset import (
    DATASET_TYPE_AUTO,
    DATASET_TYPE_LAB,
    DATASET_TYPE_TUM,
    detect_dataset_type,
    detect_tum_camera_profile,
    load_lab_camera_config,
    load_rgbd_associations,
    make_lab_rgbd_camera,
    make_rgbd_camera,
    resolve_camera_metadata,
    resolve_lab_camera_config_path,
)

__all__ = [
    "DATASET_TYPE_AUTO",
    "DATASET_TYPE_LAB",
    "DATASET_TYPE_TUM",
    "TumRgbdFrame",
    "associate_tum_rgbd",
    "detect_dataset_type",
    "detect_tum_camera_profile",
    "load_lab_camera_config",
    "load_rgbd_associations",
    "load_tum_rgbd_associations",
    "make_lab_rgbd_camera",
    "make_rgbd_camera",
    "make_tum_rgbd_camera",
    "resolve_camera_metadata",
    "resolve_lab_camera_config_path",
    "save_tum_trajectory",
]
