from __future__ import annotations
import numpy as np
import cv2

def ensure_pose_diversity(prev_corners: list[np.ndarray], new_corners: np.ndarray, min_motion_px: float = 20.0) -> bool:
    """
    Accept new corners only if average displacement vs the last accepted sample
    is ≥ min_motion_px (avoid many near-identical views).
    """
    if not prev_corners:
        return True
    last = prev_corners[-1].reshape(-1, 2)
    curr = new_corners.reshape(-1, 2)
    if last.shape != curr.shape:
        return True
    d = np.linalg.norm(curr - last, axis=1).mean()
    return d >= min_motion_px

def draw_found_corners(frame: np.ndarray, corners: np.ndarray, valid: bool) -> np.ndarray:
    out = frame.copy()
    color = (0, 255, 0) if valid else (0, 0, 255)
    for p in corners.reshape(-1, 2):
        cv2.circle(out, tuple(int(x) for x in p), 3, color, -1, lineType=cv2.LINE_AA)
    return out
