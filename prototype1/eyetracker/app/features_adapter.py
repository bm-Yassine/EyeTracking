# File: eyetracker/app/features_adapter.py
from __future__ import annotations
import math, os
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pygame
except Exception:
    pygame = None  # not needed for parquet replay

# ---------- Parquet loader (Part A bridge)

def _load_session_features(session_path: str) -> pd.DataFrame:
    """
    Load and bridge Part A logs into features + targets.
    - Filters out face_present==False and blink==True frames.
    - Forward-fills targets from events (last click) to subsequent frames.
    - Maps columns to the features expected by Part B.
    Returns a DataFrame with columns:
      left_u,left_v,right_u,right_v,left_conf,right_conf,
      yaw_deg,pitch_deg,roll_deg,head_x_mm,head_y_mm,head_z_mm,ipd_norm,
      target_x,target_y
    """
    # Accept either a direct frames.parquet path or a session folder
    if session_path.endswith(".parquet"):
        frame_pq = session_path
        base = os.path.dirname(session_path)
        event_pq = os.path.join(base, "events.parquet")
    else:
        base = session_path
        frame_pq = os.path.join(base, "frames.parquet")
        event_pq = os.path.join(base, "events.parquet")

    frames = pd.read_parquet(frame_pq).sort_values("t_mono").reset_index(drop=True)
    events = pd.read_parquet(event_pq).sort_values("t_mono")

    # 1) filter bad frames
    good = (frames["face_present"] == True) & (frames["blink"] == False)
    frames = frames.loc[good].copy()

    # 2) forward-fill targets from events
    frames = pd.merge_asof(
        frames,
        events[["t_mono", "target_x", "target_y"]],
        on="t_mono",
        direction="backward",
    )
    frames = frames.dropna(subset=["target_x", "target_y"])

    # 3) map to regressor features
    # eye angles (radians) already logged per-eye
    frames["left_u"]  = frames["left_yaw"]
    frames["left_v"]  = frames["left_pitch"]
    frames["right_u"] = frames["right_yaw"]
    frames["right_v"] = frames["right_pitch"]

    # head angles (degrees)
    rename = {
        "head_yaw_deg": "yaw_deg",
        "head_pitch_deg": "pitch_deg",
        "head_roll_deg": "roll_deg",
    }
    for k, v in rename.items():
        if k in frames.columns:
            frames[v] = frames[k]
        elif v not in frames.columns:
            frames[v] = 0.0

    # head xyz (mm)
    if "head_x_mm" not in frames.columns:
        frames["head_x_mm"] = 0.0
    if "head_y_mm" not in frames.columns:
        frames["head_y_mm"] = 0.0
    if "head_z_mm" in frames.columns:
        pass
    elif "head_dist_mm" in frames.columns:
        frames["head_z_mm"] = frames["head_dist_mm"]
    else:
        frames["head_z_mm"] = 600.0

    # simple binocular scale; if missing, set constant
    if {"r_inner_xc", "l_outer_xc"}.issubset(frames.columns):
        frames["ipd_norm"] = frames["r_inner_xc"] - frames["l_outer_xc"]
    else:
        frames["ipd_norm"] = 0.065

    frames["left_conf"] = 1.0
    frames["right_conf"] = 1.0

    cols = [
        "left_u","left_v","right_u","right_v",
        "left_conf","right_conf",
        "yaw_deg","pitch_deg","roll_deg",
        "head_x_mm","head_y_mm","head_z_mm",
        "ipd_norm","target_x","target_y",
    ]
    return frames[cols].reset_index(drop=True)

# ---------- Live sources

class ParquetFeatureSource:
    """Replay features from a recorded session."""
    def __init__(self, session_path: str):
        self.df = _load_session_features(session_path)
        self.i = 0
        self.n = len(self.df)

    def get_frame_features(self) -> Dict[str, Any]:
        if self.n == 0:
            return {}
        row = self.df.iloc[self.i]
        self.i = (self.i + 1) % self.n
        return {
            "left_u":  float(row.left_u),  "left_v":  float(row.left_v),
            "right_u": float(row.right_u), "right_v": float(row.right_v),
            "left_conf":  float(row.left_conf), "right_conf": float(row.right_conf),
            "yaw_deg": float(row.yaw_deg), "pitch_deg": float(row.pitch_deg), "roll_deg": float(row.roll_deg),
            "head_x_mm": float(row.head_x_mm), "head_y_mm": float(row.head_y_mm), "head_z_mm": float(row.head_z_mm),
            "ipd_norm": float(row.ipd_norm),
            # also return target for convenience if caller wants it
            "target_x": float(row.target_x), "target_y": float(row.target_y),
        }

class MouseDemoFeatures:
    """Fallback live source driven by mouse position (for quick UI tests)."""
    def __init__(self, screen_size: Tuple[int,int], ipd_norm: float = 0.065):
        self.W, self.H = screen_size
        self.ipd_norm = ipd_norm

    def get_frame_features(self) -> Dict[str, Any]:
        if pygame is None:
            # head-neutral, centered gaze
            u = v = 0.0
            disp = 0.03
        else:
            x, y = pygame.mouse.get_pos()
            u = (x / max(1, self.W)) * 2.0 - 1.0  # [-1,1]
            v = (y / max(1, self.H)) * 2.0 - 1.0
            disp = 0.06 * math.sin(pygame.time.get_ticks() * 0.002)

        return {
            "left_u":  u - disp, "left_v":  v,
            "right_u": u + disp, "right_v": v,
            "left_conf":  0.9, "right_conf": 0.9,
            "yaw_deg":   0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
            "head_x_mm": 0.0, "head_y_mm": 0.0, "head_z_mm": 600.0,
            "ipd_norm":  self.ipd_norm,
        }

# ---------- Factory expected by liveprediction.py

def build_live_feature_source(screen_size: Tuple[int,int], replay_session: Optional[str] = None):
    """
    If replay_session is provided (folder or frames.parquet path), returns ParquetFeatureSource.
    Otherwise returns MouseDemoFeatures.
    """
    if replay_session is not None and os.path.exists(replay_session):
        return ParquetFeatureSource(replay_session)
    return MouseDemoFeatures(screen_size)
