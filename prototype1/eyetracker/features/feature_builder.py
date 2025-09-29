from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..io.schemas import FrameRow
from .geometry import (
    pupil_angles_from_offsets, norm_image_plane_centered, eye_ref
)

@dataclass
class BuilderContext:
    screen_w: int
    screen_h: int
    cam_w: int
    cam_h: int
    fx: float
    fy: float
    K: Optional[np.ndarray] = None
    dist: Optional[np.ndarray] = None

class FeatureBuilder:
    def __init__(self, ctx: BuilderContext):
        self.ctx = ctx
        self.frame_id = 0
        self._prev_angles = None

    def build(self, t_mono: float, backend_out: Dict[str, Any], target_xy: Optional[Tuple[int,int]] = None) -> FrameRow:
        sw, sh = self.ctx.screen_w, self.ctx.screen_h
        cw, ch = self.ctx.cam_w, self.ctx.cam_h
        fx, fy = self.ctx.fx, self.ctx.fy
        K, dist = self.ctx.K, self.ctx.dist

        # defaults
        yaw_deg = pitch_deg = roll_deg = dist_mm = np.nan
        l_yaw = l_pitch = r_yaw = r_pitch = np.nan
        l_rad = r_rad = np.nan
        l_inner = l_outer = r_inner = r_outer = (np.nan, np.nan)

        ok = backend_out.get("ok", False)
        
        if ok:
            # --- head pose ---
            if K is not None:
                from ..vision.headpose import smart_angles, solve_head_pose
                hp = solve_head_pose(backend_out["face_landmarks"], K, dist)
            if hp.ok:
                # pick outer eye corners if provided
                le = backend_out.get("left_eye_corners")
                re = backend_out.get("right_eye_corners")
                left_outer = le[1] if isinstance(le, np.ndarray) and le.shape == (2,2) else None
                right_outer = re[1] if isinstance(re, np.ndarray) and re.shape == (2,2) else None
                # robust yaw, pitch, roll centred near 0°
                yaw_deg, pitch_deg, roll_deg = smart_angles(hp, left_outer, right_outer, self._prev_angles)
                # update previous angles for continuity
                if all(np.isfinite([yaw_deg, pitch_deg, roll_deg])):
                    self._prev_angles = (yaw_deg, pitch_deg, roll_deg)
                dist_mm = float(hp.distance_mm)

            # --- per-eye pupil angles ---
            lcorn = backend_out.get("left_eye_corners")   # (2,2)
            rcorn = backend_out.get("right_eye_corners")
            pupils = backend_out.get("pupils", {})
            l_p = pupils.get("left")   # (cx,cy,r)
            r_p = pupils.get("right")

            if lcorn is not None and l_p is not None and l_p[0] is not None:
                lref = eye_ref(lcorn[0], lcorn[1])
                l_yaw, l_pitch = pupil_angles_from_offsets((l_p[0], l_p[1]), lref, fx, fy)
                l_rad = float(l_p[2])

                # corners normalized to [-0.5,0.5]
                l_inner = norm_image_plane_centered(float(lcorn[0,0]), float(lcorn[0,1]), cw, ch)
                l_outer = norm_image_plane_centered(float(lcorn[1,0]), float(lcorn[1,1]), cw, ch)

            if rcorn is not None and r_p is not None and r_p[0] is not None:
                rref = eye_ref(rcorn[0], rcorn[1])
                r_yaw, r_pitch = pupil_angles_from_offsets((r_p[0], r_p[1]), rref, fx, fy)
                r_rad = float(r_p[2])
                r_inner = norm_image_plane_centered(float(rcorn[0,0]), float(rcorn[0,1]), cw, ch)
                r_outer = norm_image_plane_centered(float(rcorn[1,0]), float(rcorn[1,1]), cw, ch)

        # pack row
        row = FrameRow(
            t_mono=float(t_mono),
            frame_id=int(self.frame_id),
            
            head_yaw_deg=float(yaw_deg),
            head_pitch_deg=float(pitch_deg),
            head_roll_deg=float(roll_deg),
            head_dist_mm=float(dist_mm),
            head_x_mm=float(hp.head_x_mm if ok and hp.ok else np.nan),
            head_y_mm=float(hp.head_y_mm if ok and hp.ok else np.nan),
            head_z_mm=float(hp.head_z_mm if ok and hp.ok else np.nan),
            left_yaw=float(l_yaw),
            left_pitch=float(l_pitch),
            right_yaw=float(r_yaw),
            right_pitch=float(r_pitch),
            face_present=bool(ok),
            blink=False,  # set by caller after quality gate
            target_x=float(np.nan if target_xy is None else target_xy[0]),
            target_y=float(np.nan if target_xy is None else target_xy[1]),
        )
        self.frame_id += 1
        return row
