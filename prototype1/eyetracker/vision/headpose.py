"""
Enhanced head pose estimation with orientation fixes.

This module implements a robust head pose solver for the EyeTracker project.
It corrects common mirroring/behind‑camera ambiguities returned by OpenCV's
``solvePnP`` and exposes convenient properties for yaw, pitch, roll and head
position.  The code keeps rotation and translation vectors as ``numpy.ndarray``
objects, computes the head centre from the translation vector (nose tip), and
provides helpers to stabilise angles across frames.

Features:
  * Right‑handed rotation matrix enforcement (determinant > 0).
  * Positive z translation enforcement (head is in front of the camera).
  * Robust Euler angle extraction and unwrapping via ``smart_angles``.

Example usage::

    hp = solve_head_pose(landmarks, K, dist)
    if hp.ok:
        yaw, pitch, roll = smart_angles(hp, left_outer, right_outer, prev_angles)
        print("Head position (mm):", hp.head_x_mm, hp.head_y_mm, hp.head_z_mm)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

__all__ = ["HeadPose", "solve_head_pose", "smart_angles"]

# MediaPipe FaceMesh indices (subject's LEFT = image RIGHT)
MP = {
    "nose_tip": 1,
    "chin": 152,
    "l_eye_outer": 263, "l_eye_inner": 362,  # subject left eye
    "r_eye_inner": 133, "r_eye_outer": 33,   # subject right eye
    "mouth_l": 61, "mouth_r": 291,
}

# 3D template in millimetres (approximate human face proportions)
MODEL_3D = np.array([
    (0.0,   0.0,   0.0),    # nose tip
    (0.0, 110.0, -20.0),    # chin
    (-60.0, 35.0, -30.0),    # left eye outer
    (-30.0, 35.0, -30.0),    # left eye inner
    (30.0, 35.0, -30.0),    # right eye inner
    (60.0, 35.0, -30.0),    # right eye outer
    (-45.0, -10.0, -20.0),  # mouth left
    (45.0, -10.0, -20.0),   # mouth right
], dtype=np.float64)


@dataclass
class HeadPose:
    """Container for PnP head pose results."""

    ok: bool
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    yaw_deg: float = np.nan
    pitch_deg: float = np.nan
    roll_deg: float = np.nan
    distance_mm: float = np.nan
    reproj_err: float = np.nan
    head_x_mm: float = np.nan
    head_y_mm: float = np.nan
    head_z_mm: float = np.nan
    center_mm: Optional[np.ndarray] = None

    # Aliases (without unit suffix)
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
    def head_x(self) -> float:
        return self.head_x_mm

    @property
    def head_y(self) -> float:
        return self.head_y_mm

    @property
    def head_z(self) -> float:
        return self.head_z_mm

    @property
    def center(self) -> Optional[np.ndarray]:
        return self.center_mm

    # Serialisable list views (for JSON logging)
    @property
    def rvec_list(self) -> Optional[list]:
        return self.rvec.reshape(-1).tolist() if self.rvec is not None else None

    @property
    def tvec_list(self) -> Optional[list]:
        return self.tvec.reshape(-1).tolist() if self.tvec is not None else None

    @property
    def R_list(self) -> Optional[list]:
        return self.R.tolist() if self.R is not None else None

    @property
    def center_list(self) -> Optional[list]:
        return self.center_mm.tolist() if self.center_mm is not None else None


def _collect_points(face_landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Pick the eight 2D image points required for PnP from MediaPipe landmarks.
    If the landmarks array is invalid or missing points, returns None.
    """
    try:
        xy = face_landmarks[:, :2].astype(np.float64)
        idxs = [MP[k] for k in ("nose_tip", "chin",
                                "l_eye_outer", "l_eye_inner",
                                "r_eye_inner", "r_eye_outer",
                                "mouth_l", "mouth_r")]
        pts2d = xy[idxs, :]
        return pts2d if np.isfinite(pts2d).all() else None
    except Exception:
        return None


def _wrap180(a: float) -> float:
    """Wrap an angle to the range [-180, 180) degrees."""
    return (a + 180.0) % 360.0 - 180.0


def _unwrap_to_prev(a: float, prev: float) -> float:
    """
    Return the angle equivalent to ``a`` that is closest to ``prev`` by adding
    or subtracting multiples of 360°.
    """
    while a - prev > 180.0:
        a -= 360.0
    while a - prev < -180.0:
        a += 360.0
    return a


def euler_zyx_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Euler angles (yaw, pitch, roll) from a rotation matrix using the
    ZYX convention employed by the EyeTracker project.

    Returns ``(yaw, pitch, roll)`` in degrees where:

      * yaw   → rotation about camera Y‑axis (left/right)
      * pitch → rotation about camera X‑axis (up/down)
      * roll  → rotation about camera Z‑axis (tilt)
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
        roll  = 0.0
    return float(yaw), float(pitch), float(roll)


def _fix_orientation(R: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure the rotation matrix is right‑handed and the translation vector
    indicates the head lies in front of the camera (positive z).

    If ``det(R) < 0``, flips the rotation to maintain a right‑handed coordinate
    system.  If ``tvec[2] < 0``, reflects the solution about the camera's X–Y
    plane to bring the head in front of the camera.

    Returns the corrected ``(R, tvec)``.
    """
    R_corrected = R.copy()
    tvec_corrected = tvec.reshape(3, 1).copy()
    # Fix left‑handed rotation
    if np.linalg.det(R_corrected) < 0:
        R_corrected[:, 2] *= -1
        R_corrected[2, :] *= -1
    # Ensure positive z translation
    if tvec_corrected[2, 0] < 0:
        # Mirror along X and Y axes (flip yaw/roll) and make z positive
        R_corrected = R_corrected @ np.diag([-1, -1, 1])
        tvec_corrected[0, 0] *= -1
        tvec_corrected[1, 0] *= -1
        tvec_corrected[2, 0] = abs(tvec_corrected[2, 0])
    return R_corrected, tvec_corrected


def solve_head_pose(face_landmarks: np.ndarray,
                    K: np.ndarray,
                    dist: np.ndarray | None = None,
                    rvec0: np.ndarray | None = None,
                    tvec0: np.ndarray | None = None) -> HeadPose:
    """
    Estimate head pose from 2D facial landmarks using solvePnP.

    :param face_landmarks: Array ``(N,3)`` of MediaPipe face mesh landmarks.
    :param K: Camera intrinsic matrix (3×3).
    :param dist: Distortion coefficients (vector of length 5).  If None, zeros are used.
    :param rvec0: Optional initial rotation vector guess for iterative solver.
    :param tvec0: Optional initial translation vector guess for iterative solver.
    :return: ``HeadPose`` dataclass with estimated yaw/pitch/roll and head position.
    """
    img_pts = _collect_points(face_landmarks)
    if img_pts is None or img_pts.shape != (8, 2):
        return HeadPose(ok=False)

    obj_pts = MODEL_3D
    K = np.asarray(K, dtype=np.float64)
    if dist is None:
        dist = np.zeros((5,), dtype=np.float64)
    else:
        dist = np.asarray(dist, dtype=np.float64).reshape(-1)

    
    if rvec0 is not None and tvec0 is not None:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist,
            rvec=np.asarray(rvec0, np.float64),
            tvec=np.asarray(tvec0, np.float64),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_P3P
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return HeadPose(ok=False)

    # Step 3: Optional Levenberg–Marquardt refinement (if available)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
    except Exception:
        pass

    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)
    # Correct orientation (right‑handed and positive z)
    R_corr, tvec_corr = _fix_orientation(R, tvec)

    # Compute Euler angles and wrap to [-180, 180)
    yaw, pitch, roll = euler_zyx_from_R(R_corr)
    yaw = _wrap180(yaw); pitch = _wrap180(pitch); roll = _wrap180(roll)

    # Head position is the translation vector (nose tip origin)
    center_mm = tvec_corr.reshape(3)
    dist_mm = float(np.linalg.norm(center_mm))

    # RMS reprojection error
    proj, _ = cv2.projectPoints(obj_pts, cv2.Rodrigues(R_corr)[0], tvec_corr, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts)**2, axis=1))))

    return HeadPose(
        ok=True,
        rvec=rvec.reshape(3, 1),
        tvec=tvec_corr.reshape(3, 1),
        R=R_corr,
        yaw_deg=float(yaw),
        pitch_deg=float(pitch),
        roll_deg=float(roll),
        distance_mm=dist_mm,
        reproj_err=err,
        head_x_mm=float(center_mm[0]),
        head_y_mm=float(center_mm[1]),
        head_z_mm=float(center_mm[2]),
        center_mm=center_mm,
    )


def _closest_equivalent(yaw: float, pitch: float, roll: float,
                        prev: Optional[Tuple[float, float, float]] = None,
                        roll_2d: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Resolve the 180° flip ambiguity for Euler angles by choosing between the
    current angles and the alternative angles (yaw+180, pitch→-pitch, roll+180).

    If ``prev`` is provided, picks the candidate closest (in L1 sense) to
    ``prev``.  If ``roll_2d`` is provided, picks the candidate whose roll is
    closer to ``roll_2d``.
    """
    # Candidate A: current
    ya, pa, ra = _wrap180(yaw), _wrap180(pitch), _wrap180(roll)
    # Candidate B: equivalent after 180° rotation around camera Z
    yb, pb, rb = _wrap180(ya + 180.0), _wrap180(-pa), _wrap180(ra + 180.0)
    prefer_B = False
    # Use 2D roll if available
    if roll_2d is not None:
        dA = abs(_wrap180(ra - roll_2d))
        dB = abs(_wrap180(rb - roll_2d))
        prefer_B = dB < dA - 1e-6
    # Use previous angles for continuity
    if prev is not None and all(np.isfinite(prev)):
        def L1(a, b): return (abs(_wrap180(a[0] - b[0])) + abs(_wrap180(a[1] - b[1])) + abs(_wrap180(a[2] - b[2])))
        dA_prev = L1((ya, pa, ra), prev)
        dB_prev = L1((yb, pb, rb), prev)
        if dB_prev + 5.0 < dA_prev:
            prefer_B = True
        elif dA_prev + 5.0 < dB_prev:
            prefer_B = False
    y, p, r = (yb, pb, rb) if prefer_B else (ya, pa, ra)
    return _wrap180(y), _wrap180(p), _wrap180(r)


def smart_angles(hp: HeadPose,
                 left_outer: Optional[np.ndarray],
                 right_outer: Optional[np.ndarray],
                 prev_angles_deg: Optional[Tuple[float, float, float]] = None) -> Tuple[float, float, float]:
    """
    Return robust yaw, pitch and roll angles in degrees.

    Given a ``HeadPose`` from ``solve_head_pose``, this function resolves the
    180° Euler ambiguity by considering a 2‑D eye line and optional previous
    angles.  If ``left_outer`` and ``right_outer`` are provided (each a 2‑D point
    for the outer eye corners), roll from the eye line is used to disambiguate
    roll; otherwise, the raw roll from ``hp`` is used.
    """
    if hp is None or not hp.ok or hp.R is None:
        return np.nan, np.nan, np.nan
    yaw = _wrap180(hp.yaw_deg)
    pitch = _wrap180(hp.pitch_deg)
    roll = _wrap180(hp.roll_deg)
    # Estimate roll from 2D eye line if available
    roll_2d = None
    if left_outer is not None and right_outer is not None:
        try:
            dx = float(right_outer[0] - left_outer[0])
            dy = float(right_outer[1] - left_outer[1])
            if abs(dx) + abs(dy) > 1e-6:
                roll_2d = _wrap180(np.degrees(np.arctan2(dy, dx)))
        except Exception:
            roll_2d = None
    # Choose the closest equivalent
    y, p, r = _closest_equivalent(yaw, pitch, roll, prev=prev_angles_deg, roll_2d=roll_2d)
    # Unwrap to previous angles for continuity
    if prev_angles_deg is not None and all(np.isfinite(prev_angles_deg)):
        y = _unwrap_to_prev(y, prev_angles_deg[0])
        p = _unwrap_to_prev(p, prev_angles_deg[1])
        r = _unwrap_to_prev(r, prev_angles_deg[2])
    return float(y), float(p), float(r)