"""
Shared SLAM state definitions.
This module defines the high-level states reported by the tracking pipeline.
"""

from __future__ import annotations

from enum import Enum


# Enumerate the high-level states reported by the SLAM pipeline.
class SlamState(Enum):
    """High-level states reported by the SLAM system."""
    NO_IMAGES_YET = 0
    NOT_INITIALIZED = 1
    OK = 2
    LOST = 3
    RELOCALIZE = 4
    INIT_RELOCALIZE = 5
