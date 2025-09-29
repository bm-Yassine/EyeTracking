from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np
import cv2


# ===========================
# Utilities / math
# ===========================

def _wrap180(a: float) -> float:
    """Wrap angle in degrees to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return a if a != -180.0 else 180.0

def _deg(x: float) -> float:
    return float(np.degrees(x))

def _rad(x: float) -> float:
    return float(np.radians(x))

def _euler_zyx_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles (yaw Z, pitch Y, roll X) in degrees from rotation matrix R
    following ZYX convention consistent with OpenCV solvePnP usage.

    NOTE:
      - yaw   (Z) +: turn to subject's right
      - pitch (Y) +: look down (because image y grows downward)
      - roll  (X) +: clockwise head tilt from camera viewpoint

    Handles gimbal singularity safely.
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-8
    if not singular:
        yaw   = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        roll  = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw   = np.arctan2(-R[0, 1], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll  = 0.0
    return _deg(yaw), _deg(pitch), _deg(roll)

def _R_from_rvec(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    return R

def _rvec_from_R(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec

def _rotz(deg: float) -> np.ndarray:
    z = _rad(deg)
    cz, sz = np.cos(z), np.sin(z)
    return np.array([[cz, -sz, 0],
                     [sz,  cz, 0],
                     [ 0,   0, 1]], dtype=np.float64)

def _median_abs(x: np.ndarray) -> float:
    return float(np.median(np.abs(x)))

def _reproj_err(pts2d: np.ndarray, proj2d: np.ndarray) -> np.ndarray:
    d = pts2d - proj2d
    return np.sqrt((d ** 2).sum(axis=1))


# ===========================
# Canonical head model (mm)
# ===========================

# A small, stable set of 3D points (approximate mean face, mm).
# Chosen to be compatible with typical MediaPipe/SPIGA keypoints.
#        name             (X,     Y,      Z)
_CANONICAL = {
    "nose_tip":           (   0.0,    0.0,    0.0),
    "chin":               (   0.0, -330.0,  -65.0),
    "left_eye_outer":     (-225.0,  170.0, -135.0),
    "right_eye_outer":    ( 225.0,  170.0, -135.0),
    "left_mouth":         (-150.0, -150.0, -125.0),
    "right_mouth":        ( 150.0, -150.0, -125.0),
}

# Average inter-pupillary distance (can be user-calibrated later).
DEFAULT_IPD_MM = 63.0


# ===========================
# Landmark extraction
# ===========================

def _extract_2d_landmarks(landmarks) -> Dict[str, Tuple[float, float]]:
    """
    Try to extract the 2D landmarks we need from various possible formats.

    Accepts:
      - dict with named keys (ideal)
      - dict providing eye corners under 'left_eye_corners'/'right_eye_corners'
      - np.ndarray with known indices (not used here; extend if needed)

    Returns dict: name -> (x, y) in pixels. Missing keys removed.
    """
    if landmarks is None:
        return {}

    if isinstance(landmarks, dict):
        # Direct named points first
        direct_keys = ["nose_tip", "chin", "left_eye_outer", "right_eye_outer", "left_mouth", "right_mouth"]
        got = {}
        for k in direct_keys:
            v = landmarks.get(k)
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                got[k] = (float(v[0]), float(v[1]))

        # Derive from eye corners array if present
        lec = landmarks.get("left_eye_corners")
        rec = landmarks.get("right_eye_corners")
        if "left_eye_outer" not in got and isinstance(lec, np.ndarray) and lec.shape == (2, 2):
            # corners order expected: [inner, outer]
            got["left_eye_outer"] = (float(lec[1, 0]), float(lec[1, 1]))
        if "right_eye_outer" not in got and isinstance(rec, np.ndarray) and rec.shape == (2, 2):
            got["right_eye_outer"] = (float(rec[1, 0]), float(rec[1, 1]))

        # Heuristic mouth corners, nose tip if available in the dict
        if "left_mouth" not in got and "mouth_left" in landmarks:
            v = landmarks["mouth_left"]; got["left_mouth"] = (float(v[0]), float(v[1]))
        if "right_mouth" not in got and "mouth_right" in landmarks:
            v = landmarks["mouth_right"]; got["right_mouth"] = (float(v[0]), float(v[1]))
        if "nose_tip" not in got and "nose" in landmarks:
            v = landmarks["nose"]; got["nose_tip"] = (float(v[0]), float(v[1]))
        if "chin" not in got and "chin_tip" in landmarks:
            v = landmarks["chin_tip"]; got["chin"] = (float(v[0]), float(v[1]))

        return {k: v for k, v in got.items() if k in _CANONICAL}

    # TODO: extend if you pass a flat array with specific indices.
    return {}


def _vector_ipd_mm() -> float:
    return DEFAULT_IPD_MM


# ===========================
# HeadPose dataclass
# ===========================

@dataclass
class HeadPose:
    ok: bool
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    # Camera-frame head centre (mm)
    head_x_mm: float = np.nan
    head_y_mm: float = np.nan
    head_z_mm: float = np.nan
    distance_mm: float = np.nan
    # Diagnostics
    n_inliers: int = 0
    reproj_px_med: float = np.nan


# ===========================
# Depth helpers
# ===========================

def _unproject(u: float, v: float, z_m: float, K: np.ndarray) -> np.ndarray:
    """Pixel (u,v) with depth z_m -> 3D camera coords (mm)."""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X = (u - cx) * z_m / fx
    Y = (v - cy) * z_m / fy
    Z = z_m
    return np.array([X, Y, Z], dtype=np.float64) * 1000.0  # m -> mm


def _procrustes_umeyama(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    R,t from Umeyama (no scaling): min || R P + t - Q ||.
    P: (3,N) canonical model points in mm
    Q: (3,N) observed 3D camera points in mm
    Returns R (3x3), t (3,)
    """
    muP = P.mean(axis=1, keepdims=True)
    muQ = Q.mean(axis=1, keepdims=True)
    X = P - muP
    Y = Q - muQ
    S = Y @ X.T / P.shape[1]
    U, _, Vt = np.linalg.svd(S)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    t = (muQ - R @ muP).reshape(3)
    return R, t


# ===========================
# Robust PnP (webcam)
# ===========================

def _solve_pnp_robust(landmarks, K: np.ndarray, dist: np.ndarray,
                      rvec0=None, tvec0=None,
                      ipd_mm: float = DEFAULT_IPD_MM) -> HeadPose:
    """EPnP + iterative refinement with outlier pruning and IPD-based metric scale."""
    lm2d = _extract_2d_landmarks(landmarks)
    needed = ["nose_tip", "chin", "left_eye_outer", "right_eye_outer", "left_mouth", "right_mouth"]
    pts2d, obj3d = [], []
    for k in needed:
        if k in lm2d:
            pts2d.append(lm2d[k]); obj3d.append(_CANONICAL[k])
    if len(pts2d) < 4:
        return HeadPose(ok=False)

    pts2d = np.array(pts2d, dtype=np.float64)
    obj3d = np.array(obj3d, dtype=np.float64)

    # EPnP initial
    ok, rvec, tvec = cv2.solvePnP(obj3d, pts2d, K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return HeadPose(ok=False)

    # Reprojection error + prune (robustify)
    for _ in range(2):
        proj, _ = cv2.projectPoints(obj3d, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        err = _reproj_err(pts2d, proj)
        med = np.median(err)
        keep = err < (3.0 * med + 1.0)  # loose Huber-like gate
        if keep.sum() >= 4 and keep.sum() < len(pts2d):
            obj3d = obj3d[keep]; pts2d = pts2d[keep]
            ok, rvec, tvec = cv2.solvePnP(obj3d, pts2d, K, dist, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return HeadPose(ok=False)
        else:
            break

    # Final iterative refine
    ok, rvec, tvec = cv2.solvePnP(obj3d, pts2d, K, dist, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return HeadPose(ok=False)

    R = _R_from_rvec(rvec)

    # Metric scaling via IPD (inter-ocular).
    # Measure model IPD in object space and align to desired IPD to get scale s.
    # Our object model is already in mm-ish units; still, enforce scale for consistency.
    model_ipd = np.linalg.norm(np.array(_CANONICAL["right_eye_outer"]) - np.array(_CANONICAL["left_eye_outer"]))
    s = float(ipd_mm / model_ipd) if model_ipd > 1e-6 else 1.0

    # Head center: midpoint between eye outers in object -> transform -> mm
    eyeL = np.array(_CANONICAL["left_eye_outer"])
    eyeR = np.array(_CANONICAL["right_eye_outer"])
    head_center_obj = (eyeL + eyeR) * 0.5 * s
    tvec_mm = (tvec.reshape(3) * s)
    head_center_cam = (R @ head_center_obj.reshape(3, 1)).reshape(3) + tvec_mm

    hx, hy, hz = float(head_center_cam[0]), float(head_center_cam[1]), float(head_center_cam[2])
    dist_mm = float(np.linalg.norm(head_center_cam))

    # Enforce positive Z (camera forward)
    hz = abs(hz)

    # Diagnostics
    proj, _ = cv2.projectPoints(obj3d, rvec, tvec, K, dist)
    reproj = _reproj_err(pts2d, proj.reshape(-1, 2))

    return HeadPose(ok=True, rvec=rvec, tvec=tvec_mm.reshape(3, 1), R=R,
                    head_x_mm=hx, head_y_mm=hy, head_z_mm=hz, distance_mm=dist_mm,
                    n_inliers=int(len(pts2d)), reproj_px_med=float(np.median(reproj)))


# ===========================
# Depth-assisted (RealSense)
# ===========================

def _solve_depth_umeyama(landmarks, K: np.ndarray, depth_m: np.ndarray) -> HeadPose:
    """
    Build 3D camera points by unprojection with depth for the same canonical keys,
    then solve rigid transform R,t via Umeyama (no scale ambiguity).
    """
    lm2d = _extract_2d_landmarks(landmarks)
    keys = ["nose_tip", "chin", "left_eye_outer", "right_eye_outer", "left_mouth", "right_mouth"]
    pts3d_cam, pts3d_obj = [], []
    for k in keys:
        if k not in lm2d:
            continue
        u, v = lm2d[k]
        v_i = int(round(v)); u_i = int(round(u))
        if v_i < 0 or u_i < 0 or v_i >= depth_m.shape[0] or u_i >= depth_m.shape[1]:
            continue
        z_m = float(depth_m[v_i, u_i])
        if not np.isfinite(z_m) or z_m <= 0.0:
            continue
        P = _unproject(u, v, z_m, K)  # mm
        pts3d_cam.append(P)
        pts3d_obj.append(np.array(_CANONICAL[k], dtype=np.float64))

    if len(pts3d_cam) < 4:
        return HeadPose(ok=False)

    P = np.stack(pts3d_obj, axis=1)  # (3,N), mm
    Q = np.stack(pts3d_cam, axis=1)  # (3,N), mm

    R, t = _procrustes_umeyama(P, Q)
    rvec = _rvec_from_R(R)

    # Head center and distance
    head_center_obj = 0.5 * (np.array(_CANONICAL["left_eye_outer"]) + np.array(_CANONICAL["right_eye_outer"]))
    head_center_cam = (R @ head_center_obj.reshape(3, 1)).reshape(3) + t.reshape(3)
    hx, hy, hz = float(head_center_cam[0]), float(head_center_cam[1]), float(abs(head_center_cam[2]))
    dist_mm = float(np.linalg.norm(head_center_cam))

    return HeadPose(ok=True, rvec=rvec, tvec=t.reshape(3, 1), R=R,
                    head_x_mm=hx, head_y_mm=hy, head_z_mm=hz, distance_mm=dist_mm,
                    n_inliers=int(P.shape[1]), reproj_px_med=np.nan)


# ===========================
# Public API
# ===========================

def solve_head_pose(landmarks,
                    K: np.ndarray,
                    dist: Optional[np.ndarray] = None,
                    *,
                    rvec0=None, tvec0=None,
                    roll_hint_deg: Optional[float] = None,
                    depth_m: Optional[np.ndarray] = None,
                    ipd_mm: float = DEFAULT_IPD_MM) -> Optional[HeadPose]:
    """
    Unified head-pose entrypoint.
      - If depth map provided -> depth-assisted solver.
      - Else -> robust PnP solver with outlier pruning and IPD scaling.

    After solving, if roll_hint_deg is provided (from 2D eye-line),
    we gently fuse it by rotating R around Z to match the hint (minimize roll error).
    """
    if depth_m is not None:
        hp = _solve_depth_umeyama(landmarks, K, depth_m)
    else:
        if dist is None:
            dist = np.zeros((5,), dtype=np.float64)
        hp = _solve_pnp_robust(landmarks, K, dist, rvec0=rvec0, tvec0=tvec0, ipd_mm=ipd_mm)

    if not hp.ok or hp.R is None:
        return hp

    # Fuse roll from 2D eye-line if provided
    if roll_hint_deg is not None and np.isfinite(roll_hint_deg):
        yaw0, pitch0, roll0 = _euler_zyx_from_R(hp.R)
        d_roll = _wrap180(roll_hint_deg - roll0)
        # small corrective rotation around Z
        Rcorr = _rotz(d_roll)
        Rf = Rcorr @ hp.R
        hp.R = Rf
        hp.rvec = _rvec_from_R(Rf)

    # Ensure positive Z
    hp.head_z_mm = abs(hp.head_z_mm)
    hp.distance_mm = float(np.linalg.norm([hp.head_x_mm, hp.head_y_mm, hp.head_z_mm]))
    return hp


def smart_angles(hp: HeadPose,
                 left_outer: Optional[np.ndarray] = None,
                 right_outer: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    Extract yaw/pitch/roll from R and prefer the candidate closest to zero yaw/pitch
    (helps resolve the 180° ambiguity if flip remains).
    """
    if hp is None or not hp.ok or hp.R is None:
        return (np.nan, np.nan, np.nan)

    yaw, pitch, roll = _euler_zyx_from_R(hp.R)

    # If both eyes provided, refine roll with the eye-line directly.
    if left_outer is not None and right_outer is not None:
        dx = float(right_outer[0] - left_outer[0])
        dy = float(right_outer[1] - left_outer[1])
        if abs(dx) + abs(dy) > 1e-6:
            roll2d = _wrap180(np.degrees(np.arctan2(dy, dx)))
            # Blend (vernier) with small gain to avoid jumps
            alpha = 0.35
            roll = _wrap180((1 - alpha) * roll + alpha * roll2d)

    # Ambiguity: consider roll±180 with yaw→-yaw, pitch→-pitch equivalents
    # Pick the one closer to (0,0) yaw/pitch.
    yaw2, pitch2, roll2 = _wrap180(yaw + 180), _wrap180(-pitch), _wrap180(roll + 180)
    if (abs(yaw2) + abs(pitch2)) < (abs(yaw) + abs(pitch)):
        yaw, pitch, roll = yaw2, pitch2, roll2

    return (yaw, pitch, roll)


def _closest_equivalent(yaw0: float, pitch0: float, roll0: float,
                        prev_angles: Optional[Tuple[float, float, float]],
                        roll_2d_hint: Optional[float]) -> Tuple[float, float, float]:
    """
    Keep temporal continuity with previous angles and eye-line roll hint.
    """
    yaw, pitch, roll = yaw0, pitch0, roll0

    if prev_angles is not None and all(np.isfinite(p) for p in prev_angles):
        py, pp, pr = prev_angles
        # pick equivalent that minimizes total delta
        cands = [
            (yaw, pitch, roll),
            (_wrap180(yaw + 180), _wrap180(-pitch), _wrap180(roll + 180)),
        ]
        best = min(cands, key=lambda a: abs(_wrap180(a[0]-py)) + abs(_wrap180(a[1]-pp)) + abs(_wrap180(a[2]-pr)))
        yaw, pitch, roll = best

    if roll_2d_hint is not None and np.isfinite(roll_2d_hint):
        # gentle correction
        d = _wrap180(roll_2d_hint - roll)
        roll = _wrap180(roll + 0.25 * d)

    # Optionally clamp for display (not strictly necessary for logging)
    yaw = max(min(yaw,  90.0), -90.0)
    pitch = max(min(pitch, 90.0), -90.0)
    roll = max(min(roll, 90.0), -90.0)
    return (yaw, pitch, roll)
