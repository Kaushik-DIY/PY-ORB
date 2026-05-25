"""
RGB-D dataset and camera configuration helpers.
This module resolves dataset type, associations, and camera metadata for the runners.
"""

from __future__ import annotations

import ast
from pathlib import Path

from visual_slam.orbslam.io.tum_rgbd import (
    TumRgbdFrame,
    load_tum_rgbd_associations,
    make_tum_rgbd_camera,
    save_tum_trajectory,
)
from visual_slam.orbslam.slam import PinholeCamera, SensorType


DATASET_TYPE_TUM = "tum_rgbd"
DATASET_TYPE_LAB = "lab_rgbd"
DATASET_TYPE_AUTO = "auto"

TUM_CAMERA_PROFILES = {
    "tum_fr1": "rgbd_dataset_freiburg1",
    "tum_fr2": "rgbd_dataset_freiburg2",
    "tum_fr3": "rgbd_dataset_freiburg3",
}


def _parse_yaml_scalar(value: str):
    value = value.strip()
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


def _load_simple_yaml(path: Path) -> dict:
    """Read the small camera YAML files used by the project datasets."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("%"):
                continue

            indent = len(line) - len(line.lstrip(" "))
            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()

            key, sep, value = stripped.partition(":")
            if not sep:
                continue

            current = stack[-1][1]
            key = key.strip()
            value = value.strip()

            if value == "":
                node: dict = {}
                current[key] = node
                stack.append((indent, node))
            else:
                current[key] = _parse_yaml_scalar(value)

    return root


def _flatten_mapping(data: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_mapping(value, path))
        else:
            flat[path] = value
    return flat


def _tum_name_hint(dataset: Path) -> bool:
    return dataset.name.lower().startswith("rgbd_dataset_freiburg")


def _tum_files_hint(dataset: Path) -> bool:
    return (dataset / "groundtruth.txt").exists()


def _lab_files_hint(dataset: Path) -> bool:
    return (
        (dataset / "camera.yaml").exists()
        and (dataset / "rgb").is_dir()
        and (dataset / "depth").is_dir()
        and (dataset / "associations.txt").exists()
    )


def detect_dataset_type(dataset: str | Path) -> str:
    """Infer the dataset family from its name and on-disk structure."""
    dataset = Path(dataset).expanduser().resolve()
    tum_hint = _tum_name_hint(dataset) or _tum_files_hint(dataset)
    lab_hint = _lab_files_hint(dataset)

    if tum_hint and lab_hint:
        raise ValueError(
            f"Ambiguous RGB-D dataset type for {dataset}. "
            "It looks like both TUM and lab RGB-D. Pass --dataset-type explicitly."
        )
    if lab_hint:
        return DATASET_TYPE_LAB
    if tum_hint:
        return DATASET_TYPE_TUM
    raise ValueError(
        f"Could not auto-detect RGB-D dataset type for {dataset}. "
        "Pass --dataset-type tum_rgbd or --dataset-type lab_rgbd."
    )


def detect_tum_camera_profile(dataset: str | Path) -> str:
    name = Path(dataset).name.lower()
    if "freiburg2" in name or "freiburg_2" in name or "fr2" in name:
        return "tum_fr2"
    if "freiburg3" in name or "freiburg_3" in name or "fr3" in name:
        return "tum_fr3"
    return "tum_fr1"


def resolve_lab_camera_config_path(dataset: str | Path, camera_config: str | Path | None = None) -> Path:
    dataset = Path(dataset).expanduser().resolve()
    if camera_config is not None:
        path = Path(camera_config).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Lab RGB-D camera config does not exist: {path}")
        return path

    candidate = dataset / "camera.yaml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Lab RGB-D dataset requires camera.yaml or --camera-config. Missing: {candidate}"
    )


def load_lab_camera_config(camera_config: str | Path) -> dict:
    path = Path(camera_config).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Camera config does not exist: {path}")

    raw = _load_simple_yaml(path)
    flat = _flatten_mapping(raw)

    if "camera.fx" in flat:
        fx = float(flat["camera.fx"])
        fy = float(flat["camera.fy"])
        cx = float(flat["camera.cx"])
        cy = float(flat["camera.cy"])
        width = int(flat.get("image.width", 640))
        height = int(flat.get("image.height", 480))
        fps = float(flat.get("image.fps", 30.0))
        distortion = list(flat.get("camera.distortion", [0.0, 0.0, 0.0, 0.0, 0.0]))
        depth_map_factor = float(flat.get("depth.depth_map_factor", 1000.0))
        has_depth_threshold = "depth.depth_threshold" in flat
        has_baseline = "depth.baseline_m" in flat
        depth_threshold = float(flat.get("depth.depth_threshold", 40.0))
        baseline_m = float(flat.get("depth.baseline_m", 0.08))
        dataset_name = str(flat.get("dataset_name", path.parent.name))
        sensor_type = str(flat.get("sensor_type", "RGBD"))
    else:
        fx = float(flat["Camera.fx"])
        fy = float(flat["Camera.fy"])
        cx = float(flat["Camera.cx"])
        cy = float(flat["Camera.cy"])
        width = int(flat.get("Camera.width", 640))
        height = int(flat.get("Camera.height", 480))
        fps = float(flat.get("Camera.fps", 30.0))
        distortion = [
            float(flat.get("Camera.k1", 0.0)),
            float(flat.get("Camera.k2", 0.0)),
            float(flat.get("Camera.p1", 0.0)),
            float(flat.get("Camera.p2", 0.0)),
            float(flat.get("Camera.k3", 0.0)),
        ]
        depth_map_factor = float(flat.get("DepthMapFactor", 1000.0))
        has_depth_threshold = "ThDepth" in flat
        has_baseline = "Camera.b" in flat
        depth_threshold = float(flat.get("ThDepth", 40.0))
        baseline_m = float(flat.get("Camera.b", 0.08))
        dataset_name = str(flat.get("dataset_name", path.parent.name))
        sensor_type = str(flat.get("sensor_type", "RGBD"))

    if depth_map_factor <= 0.0:
        raise ValueError(f"depth_map_factor must be positive in {path}")

    return {
        "dataset_name": dataset_name,
        "sensor_type": sensor_type,
        "width": width,
        "height": height,
        "fps": fps,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion": [float(v) for v in distortion],
        "depth_map_factor": depth_map_factor,
        "depth_factor": 1.0 / depth_map_factor,
        "depth_threshold": depth_threshold,
        "depth_threshold_source": "camera_yaml" if has_depth_threshold else "default_th_depth_40",
        "baseline_m": baseline_m,
        "baseline_source": "camera_yaml" if has_baseline else "default_rgbd_virtual_baseline_0p08m",
        "bf": fx * baseline_m,
        "camera_source": str(path),
    }


def make_lab_rgbd_camera(dataset: str | Path, camera_config: str | Path | None = None) -> PinholeCamera:
    config_path = resolve_lab_camera_config_path(dataset, camera_config)
    config = load_lab_camera_config(config_path)
    return PinholeCamera.from_params(
        width=int(config["width"]),
        height=int(config["height"]),
        fx=float(config["fx"]),
        fy=float(config["fy"]),
        cx=float(config["cx"]),
        cy=float(config["cy"]),
        sensor_type=SensorType.RGBD,
        baseline=float(config["baseline_m"]),
        depth_map_factor=float(config["depth_map_factor"]),
        th_depth=float(config["depth_threshold"]),
        fps=float(config["fps"]),
        D=list(config["distortion"]),
    )


def make_rgbd_camera(
    dataset: str | Path,
    dataset_type: str = DATASET_TYPE_AUTO,
    camera_profile: str = "auto",
    camera_config: str | Path | None = None,
) -> PinholeCamera:
    dataset = Path(dataset).expanduser().resolve()
    if dataset_type == DATASET_TYPE_AUTO:
        dataset_type = detect_dataset_type(dataset)

    if dataset_type == DATASET_TYPE_LAB:
        return make_lab_rgbd_camera(dataset, camera_config=camera_config)
    if dataset_type != DATASET_TYPE_TUM:
        raise ValueError(f"Unsupported RGB-D dataset type: {dataset_type}")

    profile = str(camera_profile or "auto").lower()
    if profile in {"auto", "default"}:
        return make_tum_rgbd_camera(dataset)
    if profile not in TUM_CAMERA_PROFILES:
        raise ValueError(
            f"Unsupported TUM camera profile '{camera_profile}'. "
            f"Choose from auto, default, {', '.join(sorted(TUM_CAMERA_PROFILES))}."
        )
    return make_tum_rgbd_camera(TUM_CAMERA_PROFILES[profile])


def resolve_camera_metadata(
    dataset: str | Path,
    dataset_type: str = DATASET_TYPE_AUTO,
    camera_profile: str = "auto",
    camera_config: str | Path | None = None,
) -> dict:
    dataset = Path(dataset).expanduser().resolve()
    if dataset_type == DATASET_TYPE_AUTO:
        dataset_type = detect_dataset_type(dataset)

    if dataset_type == DATASET_TYPE_LAB:
        metadata = load_lab_camera_config(resolve_lab_camera_config_path(dataset, camera_config))
        metadata["camera_profile"] = "camera_yaml"
        return metadata

    if dataset_type != DATASET_TYPE_TUM:
        raise ValueError(f"Unsupported RGB-D dataset type: {dataset_type}")

    profile = str(camera_profile or "auto").lower()
    if profile in {"auto", "default"}:
        profile = detect_tum_camera_profile(dataset)
        source = f"{profile}_auto"
    else:
        if profile not in TUM_CAMERA_PROFILES:
            raise ValueError(
                f"Unsupported TUM camera profile '{camera_profile}'. "
                f"Choose from auto, default, {', '.join(sorted(TUM_CAMERA_PROFILES))}."
            )
        source = profile
    camera = make_rgbd_camera(dataset, dataset_type=DATASET_TYPE_TUM, camera_profile=profile)
    baseline_m = float(camera.bf / camera.fx) if getattr(camera, "bf", None) is not None else None
    return {
        "dataset_name": dataset.name,
        "sensor_type": "RGBD",
        "width": int(camera.width),
        "height": int(camera.height),
        "fps": float(camera.fps),
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "distortion": [float(v) for v in list(camera.D)],
        "depth_map_factor": float(1.0 / camera.depth_factor),
        "depth_factor": float(camera.depth_factor),
        "depth_threshold": float(camera.depth_threshold) if camera.depth_threshold is not None else None,
        "depth_threshold_source": "tum_rgbd_default_th_depth_40",
        "baseline_m": baseline_m,
        "baseline_source": "tum_rgbd_default_virtual_baseline_0p08m",
        "bf": float(camera.bf) if getattr(camera, "bf", None) is not None else None,
        "camera_source": source,
        "camera_profile": profile,
    }


def load_rgbd_associations(
    dataset: str | Path,
    associations: str | Path | None = None,
) -> list[TumRgbdFrame]:
    dataset = Path(dataset).expanduser().resolve()
    if associations is None:
        return load_tum_rgbd_associations(dataset)

    associations_path = Path(associations).expanduser().resolve()
    if not associations_path.exists():
        raise FileNotFoundError(f"Associations file does not exist: {associations_path}")

    frames: list[TumRgbdFrame] = []
    with open(associations_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"Invalid associations line in {associations_path}: '{line}'. "
                    "Expected: rgb_timestamp rgb/file depth_timestamp depth/file"
                )
            frames.append(
                TumRgbdFrame(
                    timestamp=float(parts[0]),
                    rgb_path=dataset / parts[1],
                    depth_path=dataset / parts[3],
                )
            )
    return frames


__all__ = [
    "DATASET_TYPE_AUTO",
    "DATASET_TYPE_LAB",
    "DATASET_TYPE_TUM",
    "TumRgbdFrame",
    "detect_dataset_type",
    "detect_tum_camera_profile",
    "load_lab_camera_config",
    "load_rgbd_associations",
    "make_lab_rgbd_camera",
    "make_rgbd_camera",
    "resolve_camera_metadata",
    "resolve_lab_camera_config_path",
    "save_tum_trajectory",
]
