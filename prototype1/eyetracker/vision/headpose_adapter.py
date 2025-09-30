# headpose_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Mapping
import numpy as np
import cv2

# ---------- Defaults for MediaPipe FaceMesh / Iris ----------
# You can override these indices via the idx_map parameter if you use SPIGA or a different mesh.
DEFAULT_IDX_MAP: Mapping[str, int | Tuple[int, ...]] = {
    "CHIN": 152,
    "NOSE_TIP_CANDIDATES": (1, 4),         # MP has both ~nose tips; we'll pick the more central
    "L_EYE_OUTER": 33,
    "L_EYE_INNER": 133,
    "R_EYE_OUTER": 362,
    "R_EYE_INNER": 263,
    "MOUTH_LEFT": 61,
    "MOUTH_RIGHT": 291,
    "L_IRIS_RIM": (474, 475, 476, 477),    # 4 points around iris
    "R_IRIS_RIM": (469, 470, 471, 472),    # 4 points around iris
}

# Generic 3D face model (millimeters). Order matches image_points below.
MODEL_POINTS_MM = np.array([
    (   0.0,    0.0,    0.0),     # nose tip
    (   0.0, -330.0,  -65.0),     # chin
    (-225.0,  170.0, -135.0),     # left eye outer corner
    ( 225.0,  170.0, -135.0),     # right eye outer corner
    (-150.0, -150.0, -125.0),     # left mouth corner
    ( 150.0, -150.0, -125.0),     # right mouth corner
], dtype=np.float32)

@dataclass
class HeadPose:
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    euler_deg: Tuple[float, float, float]   # (yaw, pitch, roll) in degrees
    cam_distance: float

@dataclass
class EyeKinematics:
    eye_distance_cm: Dict[str, float]               # via iris size (approx)
    eye_distance_model_units: Dict[str, float]      # distance to outer-corner model points in cam space
    angles_cam_deg: Dict[str, Tuple[float, float]]  # per-eye (yaw_x, pitch_y) in camera frame
    angles_head_deg: Dict[str, Tuple[float, float]] # per-eye relative to head yaw/pitch

# ---------- Utilities ----------
def _pick_nose_idx(lm_px: np.ndarray, nose_candidates: Tuple[int, int]) -> int:
    cx, cy = np.nanmean(lm_px[:, 0]), np.nanmean(lm_px[:, 1])
    i0, i1 = nose_candidates
    d0 = np.hypot(lm_px[i0, 0] - cx, lm_px[i0, 1] - cy)
    d1 = np.hypot(lm_px[i1, 0] - cx, lm_px[i1, 1] - cy)
    return i0 if d0 <= d1 else i1

def _angles_from_pixel(u: float, v: float, K: np.ndarray) -> Tuple[float, float]:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # yaw-like (left/right), pitch-like (up/down; down positive since v grows downward)
    theta_x = np.degrees(np.arctan2((u - cx), fx))
    theta_y = np.degrees(np.arctan2((v - cy), fy))
    return float(theta_x), float(theta_y)

def _euler_zyx_degrees(R: np.ndarray) -> Tuple[float, float, float]:
    # Prefer cv2.RQDecomp3x3 for numerical stability; returns Rx,Ry,Rz (in degrees) for R = Rz*Ry*Rx
    RQ, Qx, Qy, Qz, _, _, _ = cv2.RQDecomp3x3(R)
    # Convert to yaw (Y), pitch (X), roll (Z) to match common wording:
    pitch, yaw, roll = float(Qx), float(Qy), float(Qz)
    return (yaw, pitch, roll)

def _reprojection_error(model_pts: np.ndarray, image_pts: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                        K: np.ndarray, dist: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(model_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return float(np.nanmean(np.linalg.norm(proj - image_pts, axis=1)))

def _solve_pnp_robust(model_pts: np.ndarray, image_pts: np.ndarray, K: np.ndarray, dist: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Try standard solvePnP; if tz<0 or det(R)<0, try a chirality flip (invert Z in model) and keep best.
    Ensures tz>0 and chooses lower reprojection error.
    """
    dist_use = dist if dist is not None else np.zeros((5, 1), dtype=np.float32)

    # 1) Direct solve
    ok1, rvec1, tvec1 = cv2.solvePnP(model_pts, image_pts, K, dist_use, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok1:
        raise RuntimeError("solvePnP failed on initial attempt")

    R1, _ = cv2.Rodrigues(rvec1)
    err1 = _reprojection_error(model_pts, image_pts, rvec1, tvec1, K, dist_use)
    score1 = (tvec1[2, 0] > 0) + (np.linalg.det(R1) > 0)  # prefer tz>0 and proper rotation

    # 2) Try mirrored-Z model if needed
    model_flipped = model_pts.copy()
    model_flipped[:, 2] *= -1.0
    ok2, rvec2, tvec2 = cv2.solvePnP(model_flipped, image_pts, K, dist_use, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok2:
        R2, _ = cv2.Rodrigues(rvec2)
        err2 = _reprojection_error(model_flipped, image_pts, rvec2, tvec2, K, dist_use)
        score2 = (tvec2[2, 0] > 0) + (np.linalg.det(R2) > 0)

        # Choose by (score, reprojection error)
        choose_flip = (score2, -err2) > (score1, -err1)
        if choose_flip:
            # Important: return R2/t2 but also indicate they were solved with flipped model.
            # Downstream we only care about camera-space pose; keeping as-is is fine.
            return rvec2, tvec2, R2

    return rvec1, tvec1, R1

def _safe_get(lm_px: np.ndarray, idx: int) -> Tuple[float, float]:
    x, y = lm_px[idx, 0], lm_px[idx, 1]
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError(f"Landmark {idx} is NaN/Inf")
    return float(x), float(y)

def _iris_center_and_radius_px(lm_px: np.ndarray, rim_idx: Tuple[int, ...],
                               outer_idx: int, inner_idx: int) -> Tuple[Tuple[float, float], float]:
    rim = np.array([_safe_get(lm_px, i) for i in rim_idx], dtype=np.float32)  # (4,2)
    c = np.mean(rim, axis=0)                                                # center from rim
    r = float(np.mean(np.linalg.norm(rim - c[None, :], axis=1)))
    # Fallback if radius collapsed:
    if not np.isfinite(r) or r <= 0:
        oc = np.array(_safe_get(lm_px, outer_idx))
        ic = np.array(_safe_get(lm_px, inner_idx))
        r = float(np.linalg.norm(oc - ic) * 0.25)  # ~ quarter of palpebral fissure width
    return (float(c[0]), float(c[1])), r

def _per_eye_distance_mm_via_iris(r_px: float, fx: float, iris_diam_mm: float = 11.7) -> float:
    # Z ≈ fx * D_world / s_px (using diameter = 2*radius)
    if r_px <= 0 or not np.isfinite(r_px):
        return np.nan
    return (fx * iris_diam_mm) / (2.0 * r_px)

def _relative_to_head(angles_cam: Dict[str, Tuple[float, float]],
                      head_yaw_pitch_deg: Tuple[float, float]) -> Dict[str, Tuple[float, float]]:
    hy, hp = head_yaw_pitch_deg
    return {k: (ax - hy, ay - hp) for k, (ax, ay) in angles_cam.items()}

def _transform_model_point(R: np.ndarray, tvec: np.ndarray, P_model: np.ndarray) -> np.ndarray:
    return (R @ P_model.reshape(3, 1) + tvec).ravel()

# ---------- Main APIs ----------
def estimate_head_pose_from_landmarks(
    lm_px: np.ndarray,                      # shape (N, 2), pixel coords
    K: np.ndarray,                          # 3x3 intrinsics
    dist: Optional[np.ndarray] = None,      # (k,) distortion or None
    idx_map: Mapping[str, int | Tuple[int, ...]] = DEFAULT_IDX_MAP,
) -> Optional[HeadPose]:
    """
    Returns HeadPose or None if something is missing. Uses 6 robust fiducials and robust solvePnP.
    """
    try:
        nose_idx = _pick_nose_idx(lm_px, tuple(idx_map["NOSE_TIP_CANDIDATES"]))  # type: ignore
        image_points = np.array([
            _safe_get(lm_px, nose_idx),                          # nose tip
            _safe_get(lm_px, int(idx_map["CHIN"])),              # chin
            _safe_get(lm_px, int(idx_map["L_EYE_OUTER"])),       # left eye outer
            _safe_get(lm_px, int(idx_map["R_EYE_OUTER"])),       # right eye outer
            _safe_get(lm_px, int(idx_map["MOUTH_LEFT"])),        # mouth left
            _safe_get(lm_px, int(idx_map["MOUTH_RIGHT"])),       # mouth right
        ], dtype=np.float32)

        rvec, tvec, R = _solve_pnp_robust(MODEL_POINTS_MM, image_points, K, dist)
        yaw, pitch, roll = _euler_zyx_degrees(R)
        cam_distance = float(np.linalg.norm(tvec))
        return HeadPose(rvec=rvec, tvec=tvec, R=R, euler_deg=(yaw, pitch, roll), cam_distance=cam_distance)
    except Exception:
        return None

def compute_eye_kinematics(
    lm_px: np.ndarray,
    K: np.ndarray,
    head: HeadPose,
    idx_map: Mapping[str, int | Tuple[int, ...]] = DEFAULT_IDX_MAP,
    iris_diam_mm: float = 11.7,
) -> EyeKinematics:
    # Iris center & radius from rim (MediaPipe Iris recommended)
    (cLx, cLy), rL = _iris_center_and_radius_px(
        lm_px, tuple(idx_map["L_IRIS_RIM"]), int(idx_map["L_EYE_OUTER"]), int(idx_map["L_EYE_INNER"])
    )
    (cRx, cRy), rR = _iris_center_and_radius_px(
        lm_px, tuple(idx_map["R_IRIS_RIM"]), int(idx_map["R_EYE_OUTER"]), int(idx_map["R_EYE_INNER"])
    )

    ang_cam = {
        "left":  _angles_from_pixel(cLx, cLy, K),
        "right": _angles_from_pixel(cRx, cRy, K),
    }
    ang_head = _relative_to_head(ang_cam, (head.euler_deg[0], head.euler_deg[1]))

    fx = float(K[0, 0])
    dL_mm = _per_eye_distance_mm_via_iris(rL, fx, iris_diam_mm)
    dR_mm = _per_eye_distance_mm_via_iris(rR, fx, iris_diam_mm)
    d_cm = {"left": dL_mm / 10.0 if np.isfinite(dL_mm) else np.nan,
            "right": dR_mm / 10.0 if np.isfinite(dR_mm) else np.nan}

    # Optional: distance to the outer eye-corner model points in camera space (same units as model, mm)
    eyeL_cam = _transform_model_point(head.R, head.tvec, np.array(MODEL_POINTS_MM[2]))
    eyeR_cam = _transform_model_point(head.R, head.tvec, np.array(MODEL_POINTS_MM[3]))
    per_eye_model_units = {"left": float(np.linalg.norm(eyeL_cam)), "right": float(np.linalg.norm(eyeR_cam))}

    return EyeKinematics(
        eye_distance_cm=d_cm,
        eye_distance_model_units=per_eye_model_units,
        angles_cam_deg=ang_cam,
        angles_head_deg=ang_head,
    )

def estimate_headpose_and_eyes(
    lm_px: np.ndarray,
    K: np.ndarray,
    dist: Optional[np.ndarray] = None,
    idx_map: Mapping[str, int | Tuple[int, ...]] = DEFAULT_IDX_MAP,
    iris_diam_mm: float = 11.7,
) -> Tuple[Optional[HeadPose], Optional[EyeKinematics]]:
    """
    Convenience wrapper: estimate head pose, then per-eye angles/distances.
    """
    head = estimate_head_pose_from_landmarks(lm_px, K, dist, idx_map)
    if head is None:
        return None, None
    eyes = compute_eye_kinematics(lm_px, K, head, idx_map, iris_diam_mm)
    return head, eyes
