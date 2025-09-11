# vision/screen_plane.py
"""
geometry/screen_plane.py

Screen plane builder for the "parallel planes" assumption:
- The screen plane is parallel to the camera image plane (normal along camera +Z).
- We need distance Z0 (camera-to-screen), physical screen size (Wm, Hm), and optional
  lateral offsets (cx, cy) expressing that the camera principal point is not exactly
  at the screen center (in meters, camera X right, Y down).
- Optionally auto-probe Z0 from a RealSense depth frame aligned to color.

Also includes helpers to:
- Save/load plane YAML
- Intersect a gaze ray with the plane
- Project a 3D plane point to screen pixels
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import yaml
import math
from pathlib import Path


# ============ Data structures ============

@dataclass
class ScreenPlane:
    """Screen plane in camera coordinates, with 3 anchor points."""
    n: np.ndarray             # unit normal (3,)
    d: float                  # plane offset: n·X + d = 0
    A: np.ndarray             # top-left corner (3,)
    B: np.ndarray             # top-right corner (3,)
    C: np.ndarray             # bottom-left corner (3,)
    center: np.ndarray        # screen center (3,)
    size_m: Tuple[float,float]# (Wm, Hm)
    res_px: Tuple[int,int]    # (Wpx, Hpx)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n.tolist(),
            "d": float(self.d),
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "C": self.C.tolist(),
            "center": self.center.tolist(),
            "size_m": [float(self.size_m[0]), float(self.size_m[1])],
            "res_px": [int(self.res_px[0]), int(self.res_px[1])],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ScreenPlane":
        return ScreenPlane(
            n=np.array(d["n"], dtype=np.float64),
            d=float(d["d"]),
            A=np.array(d["A"], dtype=np.float64),
            B=np.array(d["B"], dtype=np.float64),
            C=np.array(d["C"], dtype=np.float64),
            center=np.array(d["center"], dtype=np.float64),
            size_m=(float(d["size_m"][0]), float(d["size_m"][1])),
            res_px=(int(d["res_px"][0]), int(d["res_px"][1])),
        )


# ============ Core builder ============

def _median_depth(depth_m: np.ndarray, uv: Tuple[int,int], win: int = 10) -> float:
    """
    Robust median depth (meters) at pixel uv in a (2*win+1)^2 window.
    depth_m must be float meters, aligned to the color frame.
    """
    if depth_m is None:
        return float("nan")
    u, v = int(uv[0]), int(uv[1])
    h, w = depth_m.shape[:2]
    u0, v0 = max(0, u - win), max(0, v - win)
    u1, v1 = min(w, u + win + 1), min(h, v + win + 1)
    roi = depth_m[v0:v1, u0:u1]
    vals = roi[np.isfinite(roi) & (roi > 0)]
    return float(np.median(vals)) if vals.size else float("nan")


def build_parallel_screen_plane(
    *,
    screen_size_m: Tuple[float,float],
    res_px: Tuple[int,int],
    camera_to_screen_m: Optional[float],
    camera_offset_xy_m: Tuple[float,float] = (0.0, 0.0),
    depth_m: Optional[np.ndarray] = None,
    depth_probe_uv: Optional[Tuple[int,int]] = None,
    z_sign: int = +1,
) -> ScreenPlane:
    """
    Construct a ScreenPlane assuming the screen plane is parallel to the camera image plane.

    Arguments
    ---------
    screen_size_m        : (Wm, Hm) physical size in meters.
    res_px               : (Wpx, Hpx) resolution in pixels (what your UI uses).
    camera_to_screen_m   : Z0 (meters). If None and depth_m is provided, it will be probed.
    camera_offset_xy_m   : (cx, cy) meters, camera principal point offset from the screen center.
                           X is right, Y is down (match your image convention).
    depth_m              : optional float32/float64 depth map (meters), aligned to color.
    depth_probe_uv       : pixel (u,v) where the "flush board" is seen to probe Z0. Defaults to image center.
    z_sign               : +1 if +Z is forward (OpenCV’s usual camera frame).

    Returns
    -------
    ScreenPlane
    """
    Wm, Hm = float(screen_size_m[0]), float(screen_size_m[1])
    Wpx, Hpx = int(res_px[0]), int(res_px[1])

    # Determine Z0
    if camera_to_screen_m is None:
        if depth_m is None:
            raise ValueError("camera_to_screen_m is None and no depth_m provided to probe it.")
        if depth_probe_uv is None:
            depth_probe_uv = (depth_m.shape[1] // 2, depth_m.shape[0] // 2)
        Z0 = _median_depth(depth_m, depth_probe_uv, win=10)
        if not math.isfinite(Z0) or Z0 <= 0:
            raise ValueError("Failed to probe camera_to_screen_m (Z0) from depth.")
    else:
        Z0 = float(camera_to_screen_m)

    Z0 *= float(z_sign)

    # Basis aligned to camera axes (parallel-plane assumption)
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # along camera X (right)
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # along camera Y (down)
    n  = np.array([0.0, 0.0, 1.0 * z_sign], dtype=np.float64)  # plane normal
    n /= np.linalg.norm(n)

    cx, cy = float(camera_offset_xy_m[0]), float(camera_offset_xy_m[1])
    P0 = np.array([cx, cy, Z0], dtype=np.float64)  # screen center in camera coords

    A = P0 + (-0.5 * Wm) * ex + (-0.5 * Hm) * ey  # top-left
    B = P0 + (+0.5 * Wm) * ex + (-0.5 * Hm) * ey  # top-right
    C = P0 + (-0.5 * Wm) * ex + (+0.5 * Hm) * ey  # bottom-left

    d = -float(n @ A)  # plane offset so that n·X + d = 0

    return ScreenPlane(n=n, d=d, A=A, B=B, C=C, center=P0, size_m=(Wm, Hm), res_px=(Wpx, Hpx))


# ============ YAML I/O ============

def save_screen_plane_yaml(plane: ScreenPlane, path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(plane.to_dict(), f, sort_keys=False)


def load_screen_plane_yaml(path: str | Path) -> ScreenPlane:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return ScreenPlane.from_dict(d)


# ============ Geometry helpers ============

def ray_plane_intersection(o: np.ndarray, g_dir: np.ndarray, n: np.ndarray, d: float) -> Optional[np.ndarray]:
    """
    Intersect ray p(t) = o + t * g_dir with plane n·X + d = 0.
    Returns 3D point or None if nearly parallel or behind the origin.
    """
    g = g_dir.astype(np.float64)
    g /= max(np.linalg.norm(g), 1e-12)
    denom = float(n @ g)
    if abs(denom) < 1e-8:
        return None
    t = float(-(n @ o + d) / denom)
    if t <= 0:
        return None
    return o + t * g


def screen_basis_and_size(plane: ScreenPlane) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Returns:
      ex_hat, ey_hat : unit basis vectors on the plane (meters)
      Wx,   Hy       : side lengths in meters along ex and ey
    """
    ex = plane.B - plane.A
    ey = plane.C - plane.A
    Wx = float(np.linalg.norm(ex))
    Hy = float(np.linalg.norm(ey))
    ex_hat = ex / max(Wx, 1e-12)
    ey_hat = ey / max(Hy, 1e-12)
    return ex_hat, ey_hat, Wx, Hy


def plane_point_to_screen_pixels(P: np.ndarray, plane: ScreenPlane) -> Tuple[float,float]:
    """
    Convert a 3D point P (on the plane) to screen pixel coordinates (u, v).
    """
    ex_hat, ey_hat, Wx, Hy = screen_basis_and_size(plane)
    um = float(ex_hat @ (P - plane.A))   # meters along width
    vm = float(ey_hat @ (P - plane.A))   # meters along height
    u = um * (plane.res_px[0] / Wx)
    v = vm * (plane.res_px[1] / Hy)
    return (u, v)


def screen_pixels_to_plane_point(u: float, v: float, plane: ScreenPlane) -> np.ndarray:
    """
    Convert screen pixel coordinates (u, v) to a 3D point P on the plane.
    """
    ex_hat, ey_hat, Wx, Hy = screen_basis_and_size(plane)
    um = (u / plane.res_px[0]) * Wx
    vm = (v / plane.res_px[1]) * Hy
    P = plane.A + um * ex_hat + vm * ey_hat
    return P


# ============ Builder from config dict ============

def build_plane_from_config(
    cfg: Dict[str, Any],
    *,
    depth_m: Optional[np.ndarray] = None,
    depth_probe_uv: Optional[Tuple[int,int]] = None,
) -> ScreenPlane:
    """
    Expect a config structure like:

    screen:
      physical_width_m: 0.344
      physical_height_m: 0.193
      camera_to_screen_m: 0.00        # optional if depth probing is used
      camera_offset_x_m: 0.00
      camera_offset_y_m: 0.00
      res_w: 1920
      res_h: 1080
      out_plane_path: "cam_calibration/screen_plane.yaml"

    Returns a ScreenPlane (you can then save it).
    """
    sc = cfg.get("screen", {})
    Wm = float(sc["physical_width_m"])
    Hm = float(sc["physical_height_m"])
    Wpx = int(sc["res_w"])
    Hpx = int(sc["res_h"])

    Z0 = sc.get("camera_to_screen_m", None)
    cx = float(sc.get("camera_offset_x_m", 0.0))
    cy = float(sc.get("camera_offset_y_m", 0.0))

    plane = build_parallel_screen_plane(
        screen_size_m=(Wm, Hm),
        res_px=(Wpx, Hpx),
        camera_to_screen_m=Z0,
        camera_offset_xy_m=(cx, cy),
        depth_m=depth_m,
        depth_probe_uv=depth_probe_uv,
        z_sign=+1,
    )
    return plane
