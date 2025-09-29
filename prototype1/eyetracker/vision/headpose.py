# eyetracker/vision/headpose.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2

# MediaPipe FaceMesh indices (subject's LEFT = image RIGHT)
MP = {
    "nose_tip": 1,
    "chin": 152,
    "l_eye_outer": 263, "l_eye_inner": 362,   # subject left eye
    "r_eye_inner": 133, "r_eye_outer": 33,    # subject right eye
    "mouth_l": 61, "mouth_r": 291
}

# 3D template in millimeters (approx human face proportions)
MODEL_3D = np.array([
    (  0.0,    0.0,    0.0 ),    # nose tip
    (  0.0, -110.0,  -20.0),    # chin
    (-60.0,   35.0,  -30.0),    # left eye outer
    (-30.0,   35.0,  -30.0),    # left eye inner
    ( 30.0,   35.0,  -30.0),    # right eye inner
    ( 60.0,   35.0,  -30.0),    # right eye outer
    (-45.0,  -10.0,  -20.0),   # mouth left
    ( 45.0,  -10.0,  -20.0),   # mouth right
], dtype=np.float64)

@dataclass
class HeadPose:
    ok: bool
    rvec: list | None = None       # [rx, ry, rz] in radians
    tvec: list | None = None       # [tx, ty, tz] in mm
    R: list | None = None          # 3×3 rotation matrix
    yaw_deg: float = np.nan
    pitch_deg: float = np.nan
    roll_deg: float = np.nan
    distance_mm: float = np.nan
    reproj_err: float = np.nan
    center_mm: list | None = None  # [cx, cy, cz] in mm
    head_x_mm: float = np.nan
    head_y_mm: float = np.nan
    head_z_mm: float = np.nan

    @property
    def yaw(self) -> float: return self.yaw_deg
    @property
    def pitch(self) -> float: return self.pitch_deg
    @property
    def roll(self) -> float: return self.roll_deg
    @property
    def distance(self) -> float: return self.distance_mm
    @property
    def center(self) -> list | None: return self.center_mm
    @property
    def head_x(self) -> float: return self.head_x_mm
    @property
    def head_y(self) -> float: return self.head_y_mm
    @property
    def head_z(self) -> float: return self.head_z_mm


def _collect_points(face_landmarks: np.ndarray) -> Optional[np.ndarray]:
    try:
        xy = face_landmarks[:, :2].astype(np.float64)
        idxs = [MP[k] for k in ("nose_tip","chin",
                                "l_eye_outer","l_eye_inner",
                                "r_eye_inner","r_eye_outer",
                                "mouth_l","mouth_r")]
        pts2d = xy[idxs, :]
        return pts2d if np.isfinite(pts2d).all() else None
    except Exception:
        return None

def euler_zyx_from_R(R: np.ndarray):
    # OpenCV: x→right, y→down, z→forward
    # Returns yaw (Y), pitch (X), roll (Z) in degrees.
    sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(-R[2,0], sy))      # X
        yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))   # Y
        roll  = np.degrees(np.arctan2(R[2,1], R[2,2]))   # Z
    else:
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw   = np.degrees(np.arctan2(-R[0,1], R[1,1]))
        roll  = 0.0
    return yaw, pitch, roll

def roll_from_eye_line_deg(left_eye_outer: np.ndarray, right_eye_outer: np.ndarray) -> Optional[float]:
    # Roll from the 2D eye line: atan2(dy, dx). Near 0 when eyes are horizontal.
    try:
        dx = float(right_eye_outer[0] - left_eye_outer[0])
        dy = float(right_eye_outer[1] - left_eye_outer[1])
        if abs(dx) + abs(dy) < 1e-6:
            return None
        return np.degrees(np.arctan2(dy, dx))
    except Exception:
        return None

def solve_head_pose(face_landmarks: np.ndarray,
                    K: np.ndarray,
                    dist: np.ndarray | None = None,
                    rvec0: np.ndarray | None = None,
                    tvec0: np.ndarray | None = None) -> HeadPose:
    img_pts = _collect_points(face_landmarks)
    if img_pts is None or img_pts.shape != (8, 2):
        return HeadPose(ok=False)

    obj_pts = MODEL_3D
    K = np.asarray(K, dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64) if dist is None else np.asarray(dist, np.float64).reshape(-1, 1)

    # estimate pose (same as before) …
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        obj_pts, img_pts, K, dist,
        iterationsCount=300, reprojectionError=2.0,
        flags=cv2.SOLVEPNP_EPNP
    )

    # 2 fallback: ITERATIVE
    if not ok:
        if rvec0 is not None and tvec0 is not None:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, dist,
                rvec=np.asarray(rvec0, np.float64),
                tvec=np.asarray(tvec0, np.float64),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
        if not ok:
            return HeadPose(ok=False)

    # refine if available
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
    except Exception:
        pass

    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = euler_zyx_from_R(R)

    # compute approximate head centre in camera coordinates
    pts_cam = (R @ obj_pts.T).T + tvec.reshape(3)
    center = np.mean(pts_cam, axis=0)  # np.ndarray of shape (3,)

    # convert everything to Python built‑ins for easy serialisation
    rvec_list = rvec.astype(float).reshape(-1).tolist()
    tvec_list = tvec.astype(float).reshape(-1).tolist()
    R_list = R.astype(float).tolist()
    center_list = center.astype(float).tolist()

    dist_mm = float(np.linalg.norm(tvec))
    reproj_points, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((reproj_points.reshape(-1, 2) - img_pts)**2, axis=1))))

    return HeadPose(
        ok=True,
        rvec=rvec_list,
        tvec=tvec_list,
        R=R_list,
        yaw_deg=float(yaw),
        pitch_deg=float(pitch),
        roll_deg=float(roll),
        distance_mm=dist_mm,
        reproj_err=err,
        center_mm=center_list,
        head_x_mm=center_list[0],
        head_y_mm=center_list[1],
        head_z_mm=center_list[2],
    )

def smart_angles(
    hp,
    left_eye_outer: Optional[np.ndarray] = None,
    right_eye_outer: Optional[np.ndarray] = None,
    prev_angles_deg: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, float]:
    """
    Robust yaw/pitch/roll in degrees centered near 0 (no clamp).
    - Get base Euler from hp.R
    - Estimate roll from the 2D eye-line (outer→outer) if available
    - Resolve the 180°-flip ambiguity by choosing the candidate whose roll matches the eye-line
      and/or is closest to previous angles
    - Unwrap to previous to keep continuity
    """
    if hp is None or not getattr(hp, "ok", False) or hp.R is None:
        return (np.nan, np.nan, np.nan)

    # 1) Base Euler from rotation matrix (your existing helper)
    yaw, pitch, roll = euler_zyx_from_R(hp.R)  # Y, X, Z order as in your project
    yaw, pitch, roll = _wrap180(yaw), _wrap180(pitch), _wrap180(roll)

    # 2) Roll from 2D eye-line (outer→outer)
    roll_2d = None
    if left_eye_outer is not None and right_eye_outer is not None:
        dx = float(right_eye_outer[0] - left_eye_outer[0])
        dy = float(right_eye_outer[1] - left_eye_outer[1])
        if abs(dx) + abs(dy) > 1e-6:
            roll_2d = _wrap180(np.degrees(np.arctan2(dy, dx)))

    # Candidate A: as-is
    ya, pa, ra = yaw, pitch, roll
    # Candidate B: the 180°-equivalent (rotate around camera Z by 180°)
    # In ZYX Euler, this maps to yaw+180, pitch→-pitch, roll+180.
    yb, pb, rb = _wrap180(yaw + 180.0), _wrap180(-pitch), _wrap180(roll + 180.0)

    # 3) Choose candidate using eye-line roll if present
    prefer_B = False
    if roll_2d is not None:
        dA = abs(_wrap180(ra - roll_2d))
        dB = abs(_wrap180(rb - roll_2d))
        prefer_B = dB < dA - 1e-6

    # 4) Refine choice using continuity to previous frame (if provided)
    if prev_angles_deg is not None and all(np.isfinite(prev_angles_deg)):
        def L1(a,b):
            return (abs(_wrap180(a[0]-b[0])) +
                    abs(_wrap180(a[1]-b[1])) +
                    abs(_wrap180(a[2]-b[2])))
        dA_prev = L1((ya,pa,ra), prev_angles_deg)
        dB_prev = L1((yb,pb,rb), prev_angles_deg)
        if dB_prev + 3.0 < dA_prev:
            prefer_B = True
        elif dA_prev + 3.0 < dB_prev:
            prefer_B = False

    y, p, r = (yb, pb, rb) if prefer_B else (ya, pa, ra)

    # 5) Unwrap to previous (keeps values smoothly around 0 instead of drifting to ±180)
    if prev_angles_deg is not None and all(np.isfinite(prev_angles_deg)):
        y = _unwrap_to_prev(y, prev_angles_deg[0])
        p = _unwrap_to_prev(p, prev_angles_deg[1])
        r = _unwrap_to_prev(r, prev_angles_deg[2])

    return (float(y), float(p), float(r))


def _wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0

def _unwrap_to_prev(a: float, prev: float) -> float:
    """Return angle equivalent to a that is closest to prev (adds ±360 as needed)."""
    a = float(a); prev = float(prev)
    while a - prev > 180.0: a -= 360.0
    while a - prev < -180.0: a += 360.0
    return a


def _closest_equivalent(yaw: float, pitch: float, roll: float,
                        prev: Optional[Tuple[float, float, float]] = None,
                        roll_2d: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Resolve the 180°-flip ambiguity using eye-line roll (if available) + continuity.
    Returns the candidate (as-is or Z-rotated by 180°: yaw+180, pitch->-pitch, roll+180)
    that best matches roll_2d and/or previous frame angles.
    """
    # Candidate A: current
    ya, pa, ra = _wrap180(yaw), _wrap180(pitch), _wrap180(roll)
    # Candidate B: equivalent after 180° around camera Z
    yb, pb, rb = _wrap180(ya + 180.0), _wrap180(-pa), _wrap180(ra + 180.0)

    prefer_B = False
    if roll_2d is not None:
        dA = abs(_wrap180(ra - roll_2d))
        dB = abs(_wrap180(rb - roll_2d))
        prefer_B = dB < dA - 1e-6

    if prev is not None:
        def L1(a, b):  # L1 on wrapped angles
            return (abs(_wrap180(a[0] - b[0])) +
                    abs(_wrap180(a[1] - b[1])) +
                    abs(_wrap180(a[2] - b[2])))
        dA_prev = L1((ya, pa, ra), prev)
        dB_prev = L1((yb, pb, rb), prev)
        if dB_prev + 5.0 < dA_prev:  # small hysteresis
            prefer_B = True
        elif dA_prev + 5.0 < dB_prev:
            prefer_B = False

    y, p, r = (yb, pb, rb) if prefer_B else (ya, pa, ra)
    return _wrap180(y), _wrap180(p), _wrap180(r)