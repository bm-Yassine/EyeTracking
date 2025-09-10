from __future__ import annotations
import numpy as np
from typing import Tuple

def fit_circle_2d(points_xy: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle fit. points_xy: (N,2) → (cx, cy, r) in pixels."""
    x = points_xy[:, 0].astype(np.float64)
    y = points_xy[:, 1].astype(np.float64)
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = c
    r = np.sqrt(max(c0 + cx**2 + cy**2, 0.0))
    return float(cx), float(cy), float(r)

def eye_ref(inner_corner: np.ndarray, outer_corner: np.ndarray) -> tuple[float, float]:
    """Midpoint of the two eye corners → per-eye reference."""
    p = (inner_corner.astype(np.float64) + outer_corner.astype(np.float64)) * 0.5
    return float(p[0]), float(p[1])

def pupil_angles_from_offsets(
    pupil_xy: tuple[float, float],
    eye_ref_xy: tuple[float, float],
    fx: float, fy: float,
    invert_y: bool = True
) -> tuple[float, float]:
    """
    Convert pixel offset to small-angle yaw/pitch (radians) via pinhole:
      θ_yaw  = atan(Δx / fx)
      θ_pitch= atan(Δy / fy)  (image y is down → invert to make up=+)
    """
    dx = float(pupil_xy[0] - eye_ref_xy[0])
    dy = float(pupil_xy[1] - eye_ref_xy[1])
    if invert_y:
        dy = -dy
    yaw = np.arctan2(dx, fx)
    pitch = np.arctan2(dy, fy)
    return float(yaw), float(pitch)

def norm_screen(x_px: float, y_px: float, sw: int, sh: int) -> tuple[float, float]:
    """Normalize screen coords to [0,1]."""
    return float(x_px / max(sw, 1)), float(y_px / max(sh, 1))

def norm_image_plane_centered(x_px: float, y_px: float, w: int, h: int) -> tuple[float, float]:
    """Normalize image coords to [-0.5, 0.5] with (0,0) at center."""
    return float(x_px / max(w,1) - 0.5), float(y_px / max(h,1) - 0.5)

def eye_width(inner_corner: np.ndarray, outer_corner: np.ndarray) -> float:
    return float(np.linalg.norm(inner_corner.astype(np.float64) - outer_corner.astype(np.float64)))

