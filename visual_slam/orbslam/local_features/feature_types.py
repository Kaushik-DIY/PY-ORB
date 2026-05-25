"""
Feature type definitions for the local feature stack.
This module collects the detector and descriptor enums used across the pipeline.
"""

from __future__ import annotations

from enum import Enum


# Enumerate the keypoint detector families used by the front-end.
class FeatureDetectorTypes(Enum):
    NONE = 0
    SHI_TOMASI = 1
    FAST = 2
    ORB = 3
    ORB2 = 4


# Enumerate the descriptor families used by the front-end.
class FeatureDescriptorTypes(Enum):
    NONE = 0
    ORB = 1
    ORB2 = 2


# Provide small capability checks for detector and descriptor types.
class FeatureInfo:
    """Small metadata helper for detector and descriptor capabilities."""

    @staticmethod
    def is_binary_descriptor(descriptor_type: FeatureDescriptorTypes) -> bool:
        return descriptor_type in (
            FeatureDescriptorTypes.ORB,
            FeatureDescriptorTypes.ORB2,
        )

    @staticmethod
    def is_oriented_features(detector_type: FeatureDetectorTypes) -> bool:
        return detector_type in (
            FeatureDetectorTypes.ORB,
            FeatureDetectorTypes.ORB2,
        )
