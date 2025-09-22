from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import cv2
from .headpose import HeadPose, euler_zyx_from_R, MODEL_3D, MP

def _median_depth(depth_m: np.ndarray, u: float, v: float, r: int = 2) -> Optional[float]:
    h, w = depth_m.shape[:2]
    x0, y0 = int(round(u)), int(round(v))
    x1, y1 = max(0, x0-r), max(0, y0-r)
    x2, y2 = min(w, x0+r+1), min(h, y0+r+1)
    roi = depth_m[y1:y2, x1:x2]
    if roi.size == 0: return None
    d = np.median(roi)
    return None if not np.isfinite(d) or d <= 0 else float(d)

def _umeyama(X: np.ndarray, Y: np.ndarray, with_scale: bool = True):
    # X -> Y ; returns (s,R,t) such that Y ≈ s R X + t
    # X: (N,3) model, Y: (N,3) measured
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
    # Pick same subset as PnP
    idxs = [MP[k] for k in ("nose_tip","chin","l_eye_outer","l_eye_inner","r_eye_inner","r_eye_outer")]
    pts2d = face_landmarks[idxs, :2].astype(np.float64)

    # Deproject each selected landmark using local median depth
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    xyz = []
    valid = True
    for (u, v) in pts2d:
        z = _median_depth(depth_m, u, v, r=2)
        if z is None: valid = False; break
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        xyz.append([X, Y, z])
    if not valid:
        return HeadPose(ok=False)

    Y_meas = np.asarray(xyz, dtype=np.float64) * 1000.0  # -> mm
    X_model = MODEL_3D.astype(np.float64)                # mm

    # Fit rigid transform model->measured (Umeyama)
    s, R, t = _umeyama(X_model, Y_meas, with_scale=True)

    # by definition MODEL_3D origin is nose tip; so t ≈ nose 3-D in camera coords (mm)
    rvec, _ = cv2.Rodrigues(R)
    yaw, pitch, roll = euler_zyx_from_R(R)
    dist_mm = float(np.linalg.norm(t))

    # small reprojection error (check)
    proj = (s * (R @ X_model.T)).T + t
    err = float(np.sqrt(np.mean(np.sum((proj - Y_meas)**2, axis=1))))

    hp = HeadPose(ok=True, rvec=rvec, tvec=t.reshape(3,1), R=R,
                  yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                  distance_mm=dist_mm, reproj_err=err)
    # Attach Cartesian head centre (nose origin) for logger convenience
    hp.head_xyz_mm = t.astype(float)  # x,y,z in mm
    return hp
