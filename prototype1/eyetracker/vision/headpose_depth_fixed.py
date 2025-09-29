"""
Modified depth‑assisted head pose solver with positive depth and full centre fields.

This module wraps the depth‑based head pose estimation from the original
``eyetracker/vision/headpose_depth.py`` but addresses two deficiencies:

* **Positive Z coordinate:** The original implementation stored the nose‑tip
  translation in ``HeadPose.head_xyz_mm`` without populating ``center_mm``
  and the per‑axis coordinates ``head_x_mm``, ``head_y_mm`` and
  ``head_z_mm``.  Moreover, the Z component could be negative if the
  RealSense coordinate frame was inverted.  Here we explicitly set
  ``center_mm`` to the nose translation vector and assign the per‑axis
  coordinates, taking the absolute value of the Z component so that the
  reported depth is always positive.

* **Documentation:** Added docstrings and comments for clarity.

Drop this file in place of the existing ``eyetracker/vision/headpose_depth.py``
to obtain the fixed behaviour.  It reuses the ``HeadPose`` class and
face model from ``headpose.py``.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import cv2
from .headpose import HeadPose, euler_zyx_from_R, MODEL_3D, MP


def _median_depth(depth_m: np.ndarray, u: float, v: float, r: int = 2) -> Optional[float]:
    """
    Compute the median depth (in metres) in a square ROI around the given pixel.

    Parameters
    ----------
    depth_m : np.ndarray
        A 2‑D array of depth values in metres.
    u, v : float
        Pixel coordinates (horizontal, vertical).
    r : int, optional
        Radius of the square neighbourhood to consider.  Defaults to 2.

    Returns
    -------
    Optional[float]
        The median depth value if it is finite and positive, otherwise ``None``.
    """
    h, w = depth_m.shape[:2]
    x0, y0 = int(round(u)), int(round(v))
    x1, y1 = max(0, x0 - r), max(0, y0 - r)
    x2, y2 = min(w, x0 + r + 1), min(h, y0 + r + 1)
    roi = depth_m[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    d = np.median(roi)
    return None if not np.isfinite(d) or d <= 0 else float(d)


def _umeyama(X: np.ndarray, Y: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the rigid (and optionally scaling) transform that best aligns X to Y.

    Implements the Umeyama algorithm to find scale, rotation and translation
    such that ``Y ≈ s R X + t``.  Used here to fit the 3‑D face template to
    measured points from a depth camera.
    """
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    C = (Yc.T @ Xc) / X.shape[0]
    U, S, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    if with_scale:
        var = (Xc**2).sum() / X.shape[0]
        s = S.sum() / var
    else:
        s = 1.0
    t = Y.mean(axis=0) - s * (R @ X.mean(axis=0))
    return s, R, t


def solve_head_pose_with_depth(
    face_landmarks: np.ndarray,
    K: np.ndarray,
    depth_m: np.ndarray,
) -> HeadPose:
    """
    Estimate head pose from facial landmarks and a depth map.

    This variant uses depth to deproject selected 2‑D landmarks into 3‑D
    space, fits a rigid transform to the 3‑D face template and returns
    yaw, pitch, roll and the 3‑D head centre.  On failure, ``ok`` will
    be ``False`` and all other fields remain NaN.

    Parameters
    ----------
    face_landmarks : np.ndarray
        An array of MediaPipe FaceMesh landmarks of shape (468, 3).
    K : np.ndarray
        3×3 intrinsic camera matrix for the colour stream.
    depth_m : np.ndarray
        A 2‑D depth image (metres) aligned with the RGB frame.

    Returns
    -------
    HeadPose
        The estimated head pose with positive ``head_z_mm`` and populated
        ``center_mm``, ``head_x_mm``, ``head_y_mm``, ``head_z_mm``.
    """
    # Use the same subset of landmarks as the PnP solver
    idxs = [MP[k] for k in (
        "nose_tip", "chin",
        "l_eye_outer", "l_eye_inner",
        "r_eye_inner", "r_eye_outer",
    )]
    pts2d = face_landmarks[idxs, :2].astype(np.float64)

    # Deproject each selected landmark using a local median depth
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xyz = []
    valid = True
    for (u, v) in pts2d:
        z = _median_depth(depth_m, u, v, r=2)
        if z is None:
            valid = False
            break
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        xyz.append([X, Y, z])
    if not valid:
        return HeadPose(ok=False)

    # Measured 3‑D points in millimetres
    Y_meas = np.asarray(xyz, dtype=np.float64) * 1000.0
    X_model = MODEL_3D.astype(np.float64)

    # Fit rigid transform (with scale) model→measured
    s, R, t = _umeyama(X_model, Y_meas, with_scale=True)

    # Convert rotation to vector form
    rvec, _ = cv2.Rodrigues(R)

    # Euler angles (degrees) using the same convention as solve_head_pose
    yaw, pitch, roll = euler_zyx_from_R(R)
    dist_mm = float(np.linalg.norm(t))

    # Small reprojection error (for debugging)
    proj = (s * (R @ X_model.T)).T + t
    err = float(np.sqrt(np.mean(np.sum((proj - Y_meas) ** 2, axis=1))))

    # Construct HeadPose; note centre is the translation t (nose tip) in mm
    hp = HeadPose(
        ok=True,
        rvec=rvec,
        tvec=t.reshape(3, 1),
        R=R,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        distance_mm=dist_mm,
        reproj_err=err,
        center_mm=t.astype(float),
        head_x_mm=float(t[0]),
        head_y_mm=float(t[1]),
        head_z_mm=float(abs(t[2])),
    )

    return hp