"""
TUM RGB-D dataset helpers.
This module loads RGB-depth pairs, builds camera intrinsics, and saves trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from visual_slam.orbslam.slam import PinholeCamera, SensorType


 # Store one synchronized RGB-D frame entry from a TUM-style dataset.
@dataclass(frozen=True)
class TumRgbdFrame:
    timestamp: float
    rgb_path: Path
    depth_path: Path


def _read_tum_file(path: Path) -> list[tuple[float, str]]:
    entries: list[tuple[float, str]] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            entries.append((float(parts[0]), parts[1]))

    return entries


def associate_tum_rgbd(
    rgb_entries: list[tuple[float, str]],
    depth_entries: list[tuple[float, str]],
    max_difference: float = 0.02,
) -> list[tuple[float, str, float, str]]:
    """Match RGB and depth files by nearest timestamp."""
    rgb_entries = sorted(rgb_entries, key=lambda x: x[0])
    depth_entries = sorted(depth_entries, key=lambda x: x[0])

    associations: list[tuple[float, str, float, str]] = []
    used_depth = set()

    depth_times = np.array([t for t, _ in depth_entries], dtype=np.float64)

    for rgb_t, rgb_file in rgb_entries:
        if len(depth_times) == 0:
            break

        j = int(np.argmin(np.abs(depth_times - rgb_t)))

        if j in used_depth:
            continue

        depth_t, depth_file = depth_entries[j]
        dt = abs(depth_t - rgb_t)

        if dt <= max_difference:
            associations.append((rgb_t, rgb_file, depth_t, depth_file))
            used_depth.add(j)

    return associations


def load_tum_rgbd_associations(dataset_path: str | Path) -> list[TumRgbdFrame]:
    """Load RGB-depth pairs from an associations file or from stream timestamps."""
    dataset_path = Path(dataset_path).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    associations_path = dataset_path / "associations.txt"

    frames: list[TumRgbdFrame] = []

    if associations_path.exists():
        with open(associations_path, "r") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                # Parse the standard four-column association layout first.
                if len(parts) >= 4:
                    timestamp = float(parts[0])
                    rgb_rel = parts[1]
                    depth_rel = parts[3]
                # Accept the compact three-column variant as a fallback.
                elif len(parts) >= 3:
                    timestamp = float(parts[0])
                    rgb_rel = parts[1]
                    depth_rel = parts[2]
                else:
                    continue

                frames.append(
                    TumRgbdFrame(
                        timestamp=timestamp,
                        rgb_path=dataset_path / rgb_rel,
                        depth_path=dataset_path / depth_rel,
                    )
                )

        return frames

    rgb_path = dataset_path / "rgb.txt"
    depth_path = dataset_path / "depth.txt"

    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing rgb.txt: {rgb_path}")

    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth.txt: {depth_path}")

    rgb_entries = _read_tum_file(rgb_path)
    depth_entries = _read_tum_file(depth_path)

    associations = associate_tum_rgbd(rgb_entries, depth_entries)

    for rgb_t, rgb_rel, depth_t, depth_rel in associations:
        frames.append(
            TumRgbdFrame(
                timestamp=rgb_t,
                rgb_path=dataset_path / rgb_rel,
                depth_path=dataset_path / depth_rel,
            )
        )

    return frames


def make_tum_rgbd_camera(dataset_name: str | Path) -> PinholeCamera:
    """Build the standard Freiburg camera model for a TUM RGB-D sequence."""
    name = str(dataset_name).lower()

    if "freiburg2" in name or "freiburg_2" in name or "fr2" in name:
        fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
    elif "freiburg3" in name or "freiburg_3" in name or "fr3" in name:
        fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
    else:
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3

    return PinholeCamera.from_params(
        width=640,
        height=480,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        sensor_type=SensorType.RGBD,
        baseline=0.08,
        depth_map_factor=5000.0,
        th_depth=40.0,
        fps=30.0,
    )


def save_tum_trajectory(poses: Iterable, timestamps: Iterable[float], output_path: str | Path) -> None:
    """Write camera poses to the standard TUM trajectory text format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from scipy.spatial.transform import Rotation

    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")

        for Tcw, timestamp in zip(poses, timestamps):
            Tcw = np.asarray(Tcw, dtype=np.float64)
            Twc = np.linalg.inv(Tcw)

            t = Twc[:3, 3]
            q = Rotation.from_matrix(Twc[:3, :3]).as_quat()

            f.write(
                f"{float(timestamp):.6f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )
