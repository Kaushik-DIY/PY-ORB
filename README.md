# PY-ORB

Python-based visual SLAM pipeline built around ORB features for RGB-D input and sparse map reconstruction.

This repository contains a structured Python implementation of an ORB-style visual SLAM pipeline with clear separation between feature extraction, tracking, local mapping, loop closing, optimization, dataset IO, and evaluation helpers. It is intended to make the full pipeline easier to read, understand, modify, and experiment with.

The system operates on RGB-D sequences, estimates camera trajectory, maintains a sparse map, and includes optional loop closing and global bundle adjustment components. The TUM RGB-D benchmark is used in this repository to evaluate and benchmark the pipeline results, especially with Freiburg sequences such as `freiburg1_desk` and `freiburg1_room`.

## What Is Included

- `visual_slam/orbslam/`: main pipeline, SLAM core, local features, IO, and utilities
- `visual_slam/g2o_compat.py`: compatibility helpers for the installed `g2o` binding
- `tools/export_orbslam_map.py`: sparse map export
- `tools/evaluate_tum_trajectory.py`: offline trajectory evaluation utilities
- `tools/build_tum_reference_cloud.py`: TUM reference cloud helper
- `tools/plot_fr1_room_evaluation.py`: plotting helpers
- `tools/run_fr1_room_full_evaluation.py`: evaluation/report helper code imported by the runner
- `third_party/vocabs/ORBvoc.dbow3`: bundled DBoW3 vocabulary
- `third_party/local/pydbow3/pydbow3...so`: local DBoW3 Python binding
- `third_party/local/orbslam2_features/orbslam2_features...so`: optional ORB-SLAM2-style extractor backend

## What Is Not Included

- datasets
- large third-party source trees such as `third_party/g2opy/` and `third_party/pyslam_reference/`
- generated outputs, logs, archives, and caches

## Recommended Environment

This repo is easiest to use on Linux with Python 3.11.

The repository includes prebuilt local `.so` files for:

- `pydbow3`
- `orbslam2_features`

Those binaries are platform-specific. If they do not load on your machine, use the more portable OpenCV backend first:

- `--feature-backend opencv_orb`

## Required Python Packages

Install the common Python dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy opencv-python scipy matplotlib psutil
```

You also need a compatible Python `g2o` binding available in the same environment. The pipeline imports `g2o` directly, so make sure this works before running:

```bash
python -c "import g2o, cv2, numpy, scipy; print('environment looks OK')"
```

## Dataset

This repository uses the **TUM RGB-D SLAM Dataset and Benchmark** to evaluate and benchmark the pipeline, especially the **Freiburg** sequences.

Official dataset pages:

- Main benchmark page: <https://cvg.cit.tum.de/data/datasets/rgbd-dataset>
- Official download page: <https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download>
- File formats and camera notes: <https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats>

Recommended starting sequences:

- `rgbd_dataset_freiburg1_desk`
- `rgbd_dataset_freiburg1_room`

The code auto-detects TUM datasets from the standard folder structure and uses Freiburg camera intrinsics internally.

### Expected Dataset Layout

After extracting a TUM sequence, the runner expects a folder like:

```text
rgbd_dataset_freiburg1_desk/
├── rgb/
├── depth/
├── rgb.txt
├── depth.txt
├── groundtruth.txt
└── associations.txt   # optional
```

If `associations.txt` is missing, the loader can associate `rgb.txt` and `depth.txt` automatically.

## Quick Start

Clone the repo and enter it:

```bash
git clone git@github.com:Kaushik-DIY/PY-ORB.git
cd PY-ORB
```

Create and activate your environment, install the Python packages, and make sure `g2o` imports successfully.

Then run from the repository root.

## Basic Run

Example on TUM Freiburg 1 desk with the OpenCV ORB backend:

```bash
python -m visual_slam.orbslam.run_rgbd_slam \
  /path/to/rgbd_dataset_freiburg1_desk \
  --dataset-type tum_rgbd \
  --output outputs/fr1_desk_opencv \
  --feature-backend opencv_orb \
  --max-frames 300
```

Example using the bundled ORB-SLAM2-style extractor backend if the local module loads on your machine:

```bash
python -m visual_slam.orbslam.run_rgbd_slam \
  /path/to/rgbd_dataset_freiburg1_desk \
  --dataset-type tum_rgbd \
  --output outputs/fr1_desk_orb2 \
  --feature-backend pyslam_orb2 \
  --max-frames 300
```

## Loop Closing And Global BA

Example on TUM Freiburg 1 room with loop closing and global bundle adjustment enabled:

```bash
python -m visual_slam.orbslam.run_rgbd_slam \
  /path/to/rgbd_dataset_freiburg1_room \
  --dataset-type tum_rgbd \
  --output outputs/fr1_room_loop_gba \
  --feature-backend opencv_orb \
  --enable-loop-closing \
  --enable-global-ba \
  --global-ba-after-loop \
  --loop-debug \
  --start-local-mapping-thread
```

## Useful Runner Options

- `--feature-backend {opencv_orb,pyslam_orb2,auto}`
- `--enable-loop-closing`
- `--enable-global-ba`
- `--global-ba-after-loop`
- `--loop-debug`
- `--loop-retrieval-trace`
- `--start-local-mapping-thread`
- `--profile-memory`
- `--profile-runtime`
- `--profile-local-map`
- `--profile-keyframes`
- `--no-map-export`

See the built-in CLI help for the full list:

```bash
python -m visual_slam.orbslam.run_rgbd_slam --help
```

## Outputs

A typical run writes:

- TUM-format trajectory `.txt`
- `run_summary.json`
- frame logs and timing CSVs
- sparse map export as `map_points.ply`
- `keyframes.json`
- `keyframe_graph.json`
- optional loop diagnostics and profiling CSVs

These outputs are written under the folder you pass with `--output`.

## Notes On Backends

### `opencv_orb`

- most portable option
- recommended first run on a new machine
- does not depend on the bundled `orbslam2_features` binary

### `pyslam_orb2`

- uses the included `orbslam2_features` module when available
- closer to the ORB-SLAM2-style extractor path used in the source workspace
- may fail on machines that do not match the binary build environment

## Notes On BoW / Vocabulary

Loop closing and relocalization depend on the bundled DBoW3 assets:

- `third_party/vocabs/ORBvoc.dbow3`
- `third_party/local/pydbow3/pydbow3...so`

If `pydbow3` does not load on your machine, the pipeline may still run in reduced configurations, but loop retrieval and related functionality will be limited.

## Troubleshooting

### `ImportError: No module named g2o`

Install a compatible Python `g2o` binding into your environment. This repository does not vendor the `g2o` package itself.

### `ImportError` or loader errors for `pydbow3` or `orbslam2_features`

The bundled `.so` files are platform-specific. Start with:

```bash
--feature-backend opencv_orb
```

If `pydbow3` is the failing module, loop-closing features may still require rebuilding that dependency for your machine.

### Dataset not detected

Pass the dataset type explicitly:

```bash
--dataset-type tum_rgbd
```

### Missing `rgb.txt` or `depth.txt`

Make sure you extracted the original TUM sequence archive and pointed the runner at the dataset root directory, not a nested subfolder.

## Citation And Dataset License

Please refer to the TUM RGB-D benchmark publication when using the dataset:

J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers,  
"A Benchmark for the Evaluation of RGB-D SLAM Systems,"  
IROS 2012.

According to the official TUM benchmark page, the dataset is licensed under **CC BY 4.0** unless stated otherwise.
