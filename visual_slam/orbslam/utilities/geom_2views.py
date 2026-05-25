"""
Two-view geometry helpers.
This module estimates essential-matrix motion and epipolar relations between frames.
"""

from __future__ import annotations

import cv2
import numpy as np

from visual_slam.orbslam.utilities.geometry import inv_poseRt, poseRt, skew


def computeF12_numba(R1w, t1w, R2w, t2w, K1inv, K2, K2inv):
    R12 = R1w @ R2w.T
    t12 = -R1w @ (R2w.T @ t2w) + t1w
    t12x = skew(t12)

    R21 = R12.T
    H21 = (K2 @ R21) @ K1inv
    F12 = ((K1inv.T @ t12x) @ R12) @ K2inv

    return F12, H21


def computeF12_(f1, f2):
    f1_Tcw = f1.Tcw()
    f2_Tcw = f2.Tcw()

    R1w = f1_Tcw[:3, :3]
    t1w = f1_Tcw[:3, 3]
    R2w = f2_Tcw[:3, :3]
    t2w = f2_Tcw[:3, 3]

    R12 = R1w @ R2w.T
    t12 = -R1w @ (R2w.T @ t2w) + t1w
    t12x = skew(t12)

    K1Tinv = f1.camera.Kinv.T

    R21 = R12.T
    H21 = (f2.camera.K @ R21) @ f1.camera.Kinv
    F12 = ((K1Tinv @ t12x) @ R12) @ f2.camera.Kinv

    return F12, H21


def computeF12(f1, f2):
    f1_Tcw = f1.Tcw()
    f2_Tcw = f2.Tcw()

    R1w = np.ascontiguousarray(f1_Tcw[:3, :3])
    t1w = np.ascontiguousarray(f1_Tcw[:3, 3])
    R2w = np.ascontiguousarray(f2_Tcw[:3, :3])
    t2w = np.ascontiguousarray(f2_Tcw[:3, 3])

    return computeF12_numba(
        R1w,
        t1w,
        R2w,
        t2w,
        f1.camera.Kinv,
        f2.camera.K,
        f2.camera.Kinv,
    )


def check_dist_epipolar_line(kp1, kp2, F12, sigma2_kp2):
    l = np.dot(F12.T, np.array([kp1[0], kp1[1], 1.0], dtype=np.float64))

    num = l[0] * kp2[0] + l[1] * kp2[1] + l[2]
    den = l[0] * l[0] + l[1] * l[1]

    if den == 0:
        return False

    dist_sqr = num * num / den
    return dist_sqr < 3.84 * sigma2_kp2


def estimate_pose_ess_mat(
    kpn_ref,
    kpn_cur,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=0.0004,
):
    """
    Fit essential matrix on normalized keypoint coordinates.

    Args:
        kpn_ref: Nx2 normalized coordinates in reference frame.
        kpn_cur: Nx2 normalized coordinates in current frame.

    Returns:
        Trc:
            4x4 transform where pr = Trc * pc.
        mask_match:
            Boolean/integer inlier mask.
    """
    kpn_ref = np.asarray(kpn_ref, dtype=np.float64).reshape(-1, 2)
    kpn_cur = np.asarray(kpn_cur, dtype=np.float64).reshape(-1, 2)

    if len(kpn_ref) < 5 or len(kpn_cur) != len(kpn_ref):
        return None, None

    try:
        E, mask_match = cv2.findEssentialMat(
            kpn_ref,
            kpn_cur,
            focal=1.0,
            pp=(0.0, 0.0),
            method=method,
            prob=prob,
            threshold=threshold,
        )
    except Exception:
        return None, None

    if E is None or mask_match is None:
        return None, None

    try:
        _, Rcr, tcr, mask_pose = cv2.recoverPose(
            E,
            kpn_ref,
            kpn_cur,
            focal=1.0,
            pp=(0.0, 0.0),
            mask=mask_match,
        )
    except Exception:
        return None, None

    # OpenCV gives current-from-reference: pc = Tcr * pr.
    Trc = inv_poseRt(Rcr, np.asarray(tcr).reshape(3))

    if mask_pose is not None:
        mask_match = mask_pose

    return Trc, mask_match
