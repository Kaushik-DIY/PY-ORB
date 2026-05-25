# PY-ORB Manifest

This staged repository includes the visual ORB-SLAM pipeline and the direct helper files it imports at runtime.

Included:
- `visual_slam/g2o_compat.py`
- `visual_slam/orbslam/` except `run_tum_rgbd_smoke.py`
- `tools/export_orbslam_map.py`
- `tools/run_fr1_room_full_evaluation.py`
- `tools/evaluate_tum_trajectory.py`
- `tools/build_tum_reference_cloud.py`
- `tools/plot_fr1_room_evaluation.py`
- `third_party/local/pydbow3/pydbow3.cpython-311-x86_64-linux-gnu.so`
- `third_party/local/orbslam2_features/orbslam2_features.cpython-311-x86_64-linux-gnu.so`
- `third_party/vocabs/ORBvoc.dbow3`

Excluded on purpose:
- `datasets/`
- `visual_slam/reference_audit/`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`
- `third_party/pyslam_reference/`
- `third_party/g2opy/`
- build directories, caches, outputs, logs, and zip archives

Notes:
- `g2o` itself is not bundled here. Install it separately in the target environment.
- `orbslam2_features` is optional if you run with the OpenCV ORB backend instead of `pyslam_orb2`.
- The included `.so` files are platform-specific build artifacts for this environment.
