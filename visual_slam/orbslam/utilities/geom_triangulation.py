"""
Triangulation helpers for normalized keypoints.
This module reconstructs world points from two camera poses and matched rays.
"""

from __future__ import annotations

import cv2
import numpy as np


def triangulate_normalized_points(Tcw1, Tcw2, kpn1, kpn2):
    Tcw1 = np.asarray(Tcw1, dtype=np.float64)
    Tcw2 = np.asarray(Tcw2, dtype=np.float64)
    kpn1 = np.asarray(kpn1, dtype=np.float64).reshape(-1, 2)
    kpn2 = np.asarray(kpn2, dtype=np.float64).reshape(-1, 2)

    if len(kpn1) == 0 or len(kpn1) != len(kpn2):
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=bool)

    P1 = np.ascontiguousarray(Tcw1[:3, :], dtype=np.float64)
    P2 = np.ascontiguousarray(Tcw2[:3, :], dtype=np.float64)

    pts4 = cv2.triangulatePoints(P1, P2, kpn1.T, kpn2.T).T

    pts3 = np.zeros((len(pts4), 3), dtype=np.float64)
    mask = np.zeros(len(pts4), dtype=bool)

    for i, ph in enumerate(pts4):
        w = ph[3]
        if abs(w) < 1e-12:
            continue

        pw = ph[:3] / w

        pc1 = Tcw1[:3, :3] @ pw + Tcw1[:3, 3]
        pc2 = Tcw2[:3, :3] @ pw + Tcw2[:3, 3]

        if not np.all(np.isfinite(pw)):
            continue
        if pc1[2] <= 0.0 or pc2[2] <= 0.0:
            continue

        pts3[i] = pw
        mask[i] = True

    return pts3, mask
