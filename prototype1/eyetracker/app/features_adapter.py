from typing import Dict, Any, Tuple
import os
import pandas as pd
import numpy as np

class ParquetFeatureSource:
    def __init__(self, session_dir: str):
        self.df = load_session_features(session_dir)
        self.index = 0

    def get_frame_features(self) -> Dict[str, Any]:
        row = self.df.iloc[self.index]
        self.index = (self.index + 1) % len(self.df)
        return {
            "left_u":  row.left_u,  "left_v":  row.left_v,
            "right_u": row.right_u, "right_v": row.right_v,
            "left_conf":  row.left_conf, "right_conf": row.right_conf,
            "yaw_deg": row.yaw_deg,   "pitch_deg": row.pitch_deg, "roll_deg": row.roll_deg,
            "head_x_mm": row.head_x_mm, "head_y_mm": row.head_y_mm, "head_z_mm": row.head_z_mm,
            "ipd_norm": row.ipd_norm,
            "target_x": row.target_x, "target_y": row.target_y,
        }



def load_session_features(session_dir: str) -> pd.DataFrame:
    # Load the recorded Parquet files
    frames = pd.read_parquet(os.path.join(session_dir, "frames.parquet"))
    events = pd.read_parquet(os.path.join(session_dir, "events.parquet"))
    frames = frames.sort_values("t_mono").reset_index(drop=True)
    events = events.sort_values("t_mono")

    # Filter out bad frames
    good = (frames["face_present"] == True) & (frames["blink"] == False)
    frames = frames.loc[good].copy()

    # Forward‑fill targets: merge the last event whose t_mono <= frame.t_mono
    events_ff = events[["t_mono", "target_x", "target_y"]]
    frames = pd.merge_asof(
        frames, events_ff,
        left_on="t_mono", right_on="t_mono",
        direction="backward"
    )
    frames = frames.dropna(subset=["target_x", "target_y"])

    # Compute feature columns expected by your regressor
    frames["left_u"] = frames["left_yaw"]      # horizontal angle (radians)
    frames["left_v"] = frames["left_pitch"]    # vertical angle (radians)
    frames["right_u"] = frames["right_yaw"]
    frames["right_v"] = frames["right_pitch"]
    frames = frames.rename(columns={
        "head_yaw_deg": "yaw_deg",
        "head_pitch_deg": "pitch_deg",
        "head_roll_deg": "roll_deg"
    })

    # If head_x_mm/head_y_mm are missing in FrameRow, fill with zeros and map head_dist_mm to head_z_mm
    for col in ["head_x_mm", "head_y_mm"]:
        if col not in frames.columns:
            frames[col] = 0.0
    frames["head_z_mm"] = frames.get("head_z_mm", frames["head_dist_mm"])

    # Simple inter‑pupil distance from normalised eye corners; if unavailable, set constant
    frames["ipd_norm"] = frames["r_inner_xc"] - frames["l_outer_xc"]
    frames["left_conf"] = 1.0
    frames["right_conf"] = 1.0

    # Select the feature columns and the targets
    feature_cols = [
        "left_u", "left_v", "right_u", "right_v",
        "left_conf", "right_conf",
        "yaw_deg", "pitch_deg", "roll_deg",
        "head_x_mm", "head_y_mm", "head_z_mm",
        "ipd_norm", "target_x", "target_y"
    ]
    return frames[feature_cols].reset_index(drop=True)
