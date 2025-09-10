from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

# MediaPipe eyelid landmarks (pixels). See common mappings:
# Right eye: horiz (33,133), upper (159,158), lower (145,153)
# Left  eye: horiz (263,362), upper (386,385), lower (374,380)
R_OUT, R_IN = 33, 133
R_UP = [159, 158]; R_LO = [145, 153]
L_OUT, L_IN = 263, 362
L_UP = [386, 385]; L_LO = [374, 380]

def _eye_width(corners: np.ndarray) -> float:
    return float(np.linalg.norm(corners[0].astype(np.float64) - corners[1].astype(np.float64)))

def _ear(face_xy: np.ndarray, out_idx: int, in_idx: int, up_idxs, lo_idxs) -> float:
    horiz = np.linalg.norm(face_xy[out_idx,:2] - face_xy[in_idx,:2])
    up = face_xy[up_idxs, :2].mean(axis=0)
    lo = face_xy[lo_idxs, :2].mean(axis=0)
    vert = np.linalg.norm(up - lo)
    if horiz <= 1.0: return 0.0
    return float(vert / horiz)

def face_present(backend_out: Dict[str, Any]) -> bool:
    return bool(backend_out.get("ok", False))

def blink_surrogate(backend_out: Dict[str, Any]) -> Tuple[bool, float, float]:
    """
    Blink is TRUE only when face is present, while eyelids are closed (low EAR)
    OR irises/pupils are unreliable (tiny radius vs eye width).
    Returns (blink, ear_left, ear_right).
    """
    if not backend_out.get("ok", False):
        return False, 0.0, 0.0  # no face → not a blink; it's a tracking drop

    lcorn = backend_out.get("left_eye_corners")    # (2,2)
    rcorn = backend_out.get("right_eye_corners")   # (2,2)
    face_xy = backend_out.get("face_landmarks")    # (N,3)
    pupils = backend_out.get("pupils", {})
    l = pupils.get("left", (np.nan, np.nan, np.nan))
    r = pupils.get("right", (np.nan, np.nan, np.nan))

    # EAR using eyelids
    try:
        ear_r = _ear(face_xy, R_OUT, R_IN, R_UP, R_LO)
        ear_l = _ear(face_xy, L_OUT, L_IN, L_UP, L_LO)
    except Exception:
        ear_r = ear_l = 0.0

    # Iris-radius vs eye-width ratio (0..~0.4 open; near 0 when closed)
    def _ratio(corners, radius):
        if corners is None or not np.isfinite(radius): return 0.0
        w = _eye_width(corners)
        return float(radius / (0.5 * w)) if (w and w > 1.0) else 0.0

    lr = float(l[2]) if np.isfinite(l[2]) else np.nan
    rr = float(r[2]) if np.isfinite(r[2]) else np.nan
    lratio = _ratio(lcorn, lr)
    rratio = _ratio(rcorn, rr)

    # Thresholds (tuneable):
    # EAR closed ~0.12–0.18 (varies by face/cam); start conservative at 0.18
    EAR_T = 0.18
    # Iris ratio tiny when eyes closed or occluded
    IRIS_T = 0.10

    blink = (ear_l < EAR_T) or (ear_r < EAR_T) or (lratio < IRIS_T) or (rratio < IRIS_T)
    return bool(blink), float(ear_l), float(ear_r)
