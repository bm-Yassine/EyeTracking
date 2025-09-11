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
        score = float(backend_out.get("score", 0.0))

        _hp_solve = None
        if ok:
            # --- head pose ---
            if K is not None:
                try:
                    from ..vision.headpose import solve_head_pose as _hp_solve
                except Exception:
                    _hp_solve = None
                if _hp_solve is not None:
                    hp = _hp_solve(backend_out["face_landmarks"], K, dist)
                    if hp.ok:
                        yaw_deg, pitch_deg, roll_deg, dist_mm = hp.yaw_deg, hp.pitch_deg, hp.roll_deg, hp.distance_mm

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
            screen_w=int(sw), screen_h=int(sh),
            cam_w=int(cw), cam_h=int(ch),
            head_yaw_deg=float(yaw_deg),
            head_pitch_deg=float(pitch_deg),
            head_roll_deg=float(roll_deg),
            head_dist_mm=float(dist_mm),
            left_yaw=float(l_yaw),
            left_pitch=float(l_pitch),
            right_yaw=float(r_yaw),
            right_pitch=float(r_pitch),
            left_radius_px=float(l_rad),
            right_radius_px=float(r_rad),
            l_inner_xc=float(l_inner[0]),
            l_inner_yc=float(l_inner[1]),
            l_outer_xc=float(l_outer[0]),
            l_outer_yc=float(l_outer[1]),
            r_inner_xc=float(r_inner[0]),
            r_inner_yc=float(r_inner[1]),
            r_outer_xc=float(r_outer[0]),
            r_outer_yc=float(r_outer[1]),
            face_present=bool(ok),
            blink=False,  # set by caller after quality gates
            landmark_score=float(score),
            target_x=float(np.nan if target_xy is None else target_xy[0]),
            target_y=float(np.nan if target_xy is None else target_xy[1]),
        )
        self.frame_id += 1
        return row
