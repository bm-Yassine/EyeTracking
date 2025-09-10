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
}

# 3D template in millimeters (approx human face proportions)
MODEL_3D = np.array([
    (  0.0,    0.0,    0.0 ),    # nose tip
    (  0.0, -110.0,  -20.0),    # chin
    (-60.0,   35.0,  -30.0),    # left eye outer
    (-30.0,   35.0,  -30.0),    # left eye inner
    ( 30.0,   35.0,  -30.0),    # right eye inner
    ( 60.0,   35.0,  -30.0),    # right eye outer
], dtype=np.float64)

@dataclass
class HeadPose:
    ok: bool
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    yaw_deg: float = np.nan     #left/right
    pitch_deg: float = np.nan   #up/down
    roll_deg: float = np.nan    #titlr
    distance_mm: float = np.nan
    reproj_err: float = np.nan

def _collect_points(face_landmarks: np.ndarray) -> Optional[np.ndarray]:
    try:
        xy = face_landmarks[:, :2].astype(np.float64)
        idxs = [MP[k] for k in ("nose_tip","chin",
                                "l_eye_outer","l_eye_inner",
                                "r_eye_inner","r_eye_outer")]
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

def solve_head_pose(
    face_landmarks: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    rvec0: np.ndarray | None = None,
    tvec0: np.ndarray | None = None,
) -> HeadPose:
    img_pts = _collect_points(face_landmarks)
    if img_pts is None or img_pts.shape != (6,2):
        return HeadPose(ok=False)

    obj_pts = MODEL_3D
    K = np.asarray(K, dtype=np.float64)
    dist = np.zeros((5,1), dtype=np.float64) if dist is None else np.asarray(dist, np.float64).reshape(-1,1)

    # 1 RANSAC EPNP
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

    yaw, pitch, roll =  euler_zyx_from_R(R)

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1,2) - img_pts)**2, axis=1))))
    dist_mm = float(np.linalg.norm(tvec))

    return HeadPose(ok=True, rvec=rvec, tvec=tvec, R=R,
                    yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                    distance_mm=dist_mm, reproj_err=err)

def smart_angles(
    hp: HeadPose,
    left_eye_outer: Optional[np.ndarray] = None,
    right_eye_outer: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Returns (yaw, pitch, roll) degrees. If valid eye corners are provided,
    uses the 2D eye line for roll; otherwise, roll from R.
    """
    roll = hp.roll_deg
    if left_eye_outer is not None and right_eye_outer is not None:
        dx = float(right_eye_outer[0] - left_eye_outer[0])
        dy = float(right_eye_outer[1] - left_eye_outer[1])
        if abs(dx) + abs(dy) > 1e-6:
            roll = np.degrees(np.arctan2(dy, dx))
    return hp.yaw_deg, hp.pitch_deg, roll