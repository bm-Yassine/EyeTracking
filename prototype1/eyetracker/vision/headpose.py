"""
Updated head pose estimation module for Eye Tracking Part A.

This module defines a ``HeadPose`` dataclass and provides functions to estimate
the 3‑D orientation and position of a person’s head from 2‑D facial
landmarks using OpenCV’s Perspective‑n‑Point (PnP) solver.  The implementation
closely follows the original ``Eyetracker`` project but makes a few
improvements:

* Rotation and translation vectors are stored internally as NumPy arrays
  instead of Python lists.  This makes them compatible with other NumPy and
  OpenCV functions.  Serialisable list representations are exposed via
  properties (``rvec_list``, ``tvec_list`` and ``center_list``) so they
  can still be logged or converted to JSON.
* Additional attributes ``head_x_mm``, ``head_y_mm`` and ``head_z_mm``
  capture the Cartesian coordinates of the head centre (in millimetres)
  relative to the camera.  Alias properties without the ``_mm`` suffix
  (``head_x``, ``head_y``, ``head_z``) are provided for convenience.
* A helper ``smart_angles`` resolves the 180° ambiguity in yaw/pitch/roll
  angles and smoothly unwraps them across frames.  It optionally uses the
  2‑D eye‑line (outer eye corner to outer eye corner) to disambiguate
  roll.

The main entrypoint is ``solve_head_pose``.  It accepts an ``(N,3)`` array
of facial landmarks, a camera intrinsic matrix ``K`` and optional
distortion coefficients ``dist``.  It returns a ``HeadPose`` instance
containing the estimated rotation and translation vectors, rotation
matrix, Euler angles (yaw/pitch/roll in degrees), head distance,
reprojection error and head centre coordinates.  If PnP fails, the
``ok`` flag will be ``False`` and the remaining fields are ``NaN``.

Example usage::

    hp = solve_head_pose(face_landmarks, K, dist)
    if hp.ok:
        print(hp.yaw, hp.pitch, hp.roll)
        print(hp.head_x_mm, hp.head_y_mm, hp.head_z_mm)

This module is intended to replace the original ``eyetracker/vision/headpose.py``
file in the project.  To use it, copy the contents into that file or
import it in place of the old module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

__all__ = ["HeadPose", "solve_head_pose", "smart_angles"]

# MediaPipe FaceMesh indices (subject's LEFT corresponds to image RIGHT)
MP = {
    "nose_tip": 1,
    "chin": 152,
    "l_eye_outer": 263, "l_eye_inner": 362,  # subject left eye
    "r_eye_inner": 133, "r_eye_outer": 33,   # subject right eye
    "mouth_l": 61, "mouth_r": 291,
}

# 3‑D template points in millimetres (approximate human face proportions).
# These anchor points correspond to the landmarks above and are relative to
# an arbitrary head coordinate frame.  Distances are in millimetres.
MODEL_3D = np.array([
    (0.0,   0.0,   0.0),   # nose tip
    (0.0, 110.0, -20.0),   # chin
    (-60.0, 35.0, -30.0),  # left eye outer
    (-30.0, 35.0, -30.0),  # left eye inner
    (30.0, 35.0, -30.0),   # right eye inner
    (60.0, 35.0, -30.0),   # right eye outer
    (-45.0, -10.0, -20.0), # mouth left
    (45.0, -10.0, -20.0),  # mouth right
], dtype=np.float64)


@dataclass
class HeadPose:
    """Data structure holding the result of head pose estimation."""
    ok: bool
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    yaw_deg: float = np.nan
    pitch_deg: float = np.nan
    roll_deg: float = np.nan
    distance_mm: float = np.nan
    reproj_err: float = np.nan
    center_mm: Optional[np.ndarray] = None
    head_x_mm: float = np.nan
    head_y_mm: float = np.nan
    head_z_mm: float = np.nan

    # convenience aliases without the unit suffix
    @property
    def yaw(self) -> float:
        return self.yaw_deg

    @property
    def pitch(self) -> float:
        return self.pitch_deg

    @property
    def roll(self) -> float:
        return self.roll_deg

    @property
    def distance(self) -> float:
        return self.distance_mm

    @property
    def center(self) -> Optional[np.ndarray]:
        return self.center_mm

    @property
    def head_x(self) -> float:
        return self.head_x_mm

    @property
    def head_y(self) -> float:
        return self.head_y_mm

    @property
    def head_z(self) -> float:
        return self.head_z_mm

    # serialisable list views
    @property
    def rvec_list(self) -> Optional[list]:
        return self.rvec.tolist() if self.rvec is not None else None

    @property
    def tvec_list(self) -> Optional[list]:
        return self.tvec.tolist() if self.tvec is not None else None

    @property
    def R_list(self) -> Optional[list]:
        return self.R.tolist() if self.R is not None else None

    @property
    def center_list(self) -> Optional[list]:
        return self.center_mm.tolist() if self.center_mm is not None else None


def _collect_points(face_landmarks: np.ndarray) -> Optional[np.ndarray]:
    """Select the eight 2‑D image points needed for PnP from MediaPipe landmarks.

    Parameters
    ----------
    face_landmarks : np.ndarray
        An ``(N,3)`` array of face mesh landmarks where columns are ``x,y,z``.

    Returns
    -------
    Optional[np.ndarray]
        An ``(8,2)`` array of 2‑D coordinates for the nose tip, chin, eyes and
        mouth corners, or ``None`` if the input is invalid.
    """
    try:
        xy = face_landmarks[:, :2].astype(np.float64)
        idxs = [MP[k] for k in (
            "nose_tip", "chin",
            "l_eye_outer", "l_eye_inner",
            "r_eye_inner", "r_eye_outer",
            "mouth_l", "mouth_r",
        )]
        pts2d = xy[idxs, :]
        return pts2d if np.isfinite(pts2d).all() else None
    except Exception:
        return None


def euler_zyx_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert a rotation matrix to ZYX Euler angles (yaw, pitch, roll).

    OpenCV uses a coordinate system where x→right, y→down and z→forward.  The
    returned angles follow that convention: yaw is rotation about the camera’s
    Y‑axis (left/right), pitch about the X‑axis (up/down) and roll about the
    Z‑axis (tilt).

    Parameters
    ----------
    R : np.ndarray
        A 3×3 rotation matrix.

    Returns
    -------
    Tuple[float, float, float]
        The yaw, pitch and roll angles in degrees.
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
        roll = 0.0
    return float(yaw), float(pitch), float(roll)


def solve_head_pose(
    face_landmarks: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    rvec0: np.ndarray | None = None,
    tvec0: np.ndarray | None = None,
) -> HeadPose:
    """Estimate head pose from 2‑D facial landmarks using solvePnP.

    Parameters
    ----------
    face_landmarks : np.ndarray
        An ``(N,3)`` array of face mesh landmarks from MediaPipe.
    K : np.ndarray
        The camera intrinsic matrix.
    dist : np.ndarray | None, optional
        Distortion coefficients (5‑vector).  If None, zeros are assumed.
    rvec0, tvec0 : np.ndarray | None, optional
        Optional initial guesses for the rotation and translation vectors.

    Returns
    -------
    HeadPose
        A dataclass containing the head pose estimation results.  If the
        estimation fails, ``ok`` is False and other fields are ``NaN``.
    """
    img_pts = _collect_points(face_landmarks)
    if img_pts is None or img_pts.shape != (8, 2):
        return HeadPose(ok=False)

    obj_pts = MODEL_3D
    K = np.asarray(K, dtype=np.float64)
    dist = (
        np.zeros((5,), dtype=np.float64)
        if dist is None
        else np.asarray(dist, np.float64).reshape(-1)
    )

    # First attempt: PnP with EPnP solver and RANSAC to handle outliers
    try:
        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            K,
            dist,
            iterationsCount=300,
            reprojectionError=2.0,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except Exception:
        ok = False
        rvec, tvec = None, None

    # Fallback: use the iterative solver, optionally with an initial guess
    if not ok:
        try:
            if rvec0 is not None and tvec0 is not None:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    K,
                    dist,
                    rvec=np.asarray(rvec0, np.float64),
                    tvec=np.asarray(tvec0, np.float64),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
        except Exception:
            ok = False
            rvec, tvec = None, None
        if not ok:
            return HeadPose(ok=False)

    # Refine the solution using Levenberg–Marquardt if available
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
    except Exception:
        pass

    # Ensure rvec and tvec are column vectors of shape (3,1)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = euler_zyx_from_R(R)

    # Compute head centre in camera coordinates: mean of transformed template points
    pts_cam = (R @ obj_pts.T).T + tvec.reshape(3)
    center = np.mean(pts_cam, axis=0)

    # Distance from the camera origin to the head centre
    dist_mm = float(np.linalg.norm(tvec))

    # Compute reprojection error
    reproj_points, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (reproj_points.reshape(-1, 2) - img_pts) ** 2,
                    axis=1,
                )
            )
        )
    )

    return HeadPose(
        ok=True,
        rvec=rvec,
        tvec=tvec,
        R=R,
        yaw_deg=float(yaw),
        pitch_deg=float(pitch),
        roll_deg=float(roll),
        distance_mm=dist_mm,
        reproj_err=err,
        center_mm=center,
        head_x_mm=float(center[0]),
        head_y_mm=float(center[1]),
        head_z_mm=float(center[2]),
    )


def _wrap180(a: float) -> float:
    """Wrap an angle to the range [-180, 180)."""
    return (float(a) + 180.0) % 360.0 - 180.0


def _unwrap_to_prev(a: float, prev: float) -> float:
    """Unwrap angle ``a`` to be closest to ``prev`` by adding multiples of 360°."""
    a = float(a)
    prev = float(prev)
    while a - prev > 180.0:
        a -= 360.0
    while a - prev < -180.0:
        a += 360.0
    return a


def _closest_equivalent(
    yaw: float,
    pitch: float,
    roll: float,
    prev: Optional[Tuple[float, float, float]] = None,
    roll_2d: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Resolve the 180° flip ambiguity using eye‑line roll and continuity.

    There are two equivalent Euler representations for a given rotation
    matrix when using a ZYX decomposition: ``(yaw, pitch, roll)`` and
    ``(yaw+180, -pitch, roll+180)``.  This helper chooses between them
    by comparing the 2‑D eye‑line roll (if provided) and/or the previous
    frame’s angles.

    Parameters
    ----------
    yaw, pitch, roll : float
        Base Euler angles in degrees (already wrapped to [-180,180)).
    prev : Optional[Tuple[float, float, float]]
        Previous frame’s angles for continuity.  If provided, the candidate
        closest (in L1 sense) to ``prev`` is preferred.
    roll_2d : Optional[float]
        Roll estimated from the 2‑D eye‑line.  If provided, the candidate
        whose roll is closer to ``roll_2d`` is preferred.

    Returns
    -------
    Tuple[float, float, float]
        The chosen yaw, pitch and roll angles in the range [-180, 180).
    """
    # Candidate A: current angles
    ya, pa, ra = _wrap180(yaw), _wrap180(pitch), _wrap180(roll)
    # Candidate B: rotated by 180° around camera Z
    yb, pb, rb = _wrap180(ya + 180.0), _wrap180(-pa), _wrap180(ra + 180.0)

    prefer_B = False
    if roll_2d is not None:
        dA = abs(_wrap180(ra - roll_2d))
        dB = abs(_wrap180(rb - roll_2d))
        prefer_B = dB < dA - 1e-6

    if prev is not None:
        # Compare L1 distance on wrapped angles
        def L1(a, b):
            return (
                abs(_wrap180(a[0] - b[0])) +
                abs(_wrap180(a[1] - b[1])) +
                abs(_wrap180(a[2] - b[2]))
            )
        dA_prev = L1((ya, pa, ra), prev)
        dB_prev = L1((yb, pb, rb), prev)
        if dB_prev + 5.0 < dA_prev:
            prefer_B = True
        elif dA_prev + 5.0 < dB_prev:
            prefer_B = False

    y, p, r = (yb, pb, rb) if prefer_B else (ya, pa, ra)
    return _wrap180(y), _wrap180(p), _wrap180(r)


def smart_angles(
    hp: HeadPose,
    left_outer: Optional[np.ndarray],
    right_outer: Optional[np.ndarray],
    prev_angles_deg: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, float]:
    """Compute robust yaw/pitch/roll angles, resolving 180° flips and unwrapping.

    Given a ``HeadPose`` instance, this function computes yaw/pitch/roll in
    degrees that are centred near 0° (i.e., no clamp to [-90,90] but
    continuity is preserved).  It takes into account the 2‑D eye‑line
    (outer‑eye to outer‑eye) to disambiguate roll and uses the previous
    frame’s angles to maintain continuity across frames.

    Parameters
    ----------
    hp : HeadPose
        A head pose returned by ``solve_head_pose``.
    left_outer, right_outer : Optional[np.ndarray]
        The 2‑D coordinates of the subject’s left and right outer eye corners.
        If both are provided, they are used to estimate roll.
    prev_angles_deg : Optional[Tuple[float, float, float]]
        The previous frame’s yaw/pitch/roll angles in degrees.  If provided,
        helps unwrap the current angles smoothly.

    Returns
    -------
    Tuple[float, float, float]
        The yaw, pitch and roll angles (in degrees) after resolving the
        180° ambiguity and unwrapping.
    """
    if hp is None or not getattr(hp, "ok", False) or hp.R is None:
        return (np.nan, np.nan, np.nan)

    # Base Euler angles from the rotation matrix
    yaw, pitch, roll = euler_zyx_from_R(hp.R)
    yaw, pitch, roll = _wrap180(yaw), _wrap180(pitch), _wrap180(roll)

    # Estimate roll from the 2‑D eye‑line (outer→outer) if available
    roll_2d = None
    if left_outer is not None and right_outer is not None:
        try:
            dx = float(right_outer[0] - left_outer[0])
            dy = float(right_outer[1] - left_outer[1])
            if abs(dx) + abs(dy) > 1e-6:
                roll_2d = _wrap180(np.degrees(np.arctan2(dy, dx)))
        except Exception:
            roll_2d = None

    # Choose the best equivalent angles (handles 180° flips)
    y, p, r = _closest_equivalent(yaw, pitch, roll, prev_angles_deg, roll_2d)

    # Unwrap to previous frame for continuity
    if prev_angles_deg is not None and all(np.isfinite(prev_angles_deg)):
        y = _unwrap_to_prev(y, prev_angles_deg[0])
        p = _unwrap_to_prev(p, prev_angles_deg[1])
        r = _unwrap_to_prev(r, prev_angles_deg[2])

    return float(y), float(p), float(r)