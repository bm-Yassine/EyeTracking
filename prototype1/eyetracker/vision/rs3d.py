# vision/backends/rs3d.py
import numpy as np
import cv2

from mediapipe_iris import MediaPipeIris

def _median_depth(depth_m, uv, win=3):
    """Robust depth at pixel uv from a small window; returns meters or np.nan"""
    if depth_m is None: return np.nan
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = depth_m.shape[:2]
    u0, v0 = max(0, u - win), max(0, v - win)
    u1, v1 = min(w, u + win + 1), min(h, v + win + 1)
    roi = depth_m[v0:v1, u0:u1]
    roi = roi[np.isfinite(roi) & (roi > 0)]
    if roi.size == 0: return np.nan
    return float(np.median(roi))

class RealSense3DIris:
    """
    Color landmarks from MediaPipe; depth used for distance estimates / sanity metrics.
    We keep outputs 2D for Part A, but add 'depth_mm_eye' for logging.
    """
    def __init__(self):
        self.mp = MediaPipeIris()

    def process(self, frame_bgr, depth=None, K=None, dist=None):
        out = self.mp.process(frame_bgr)
        if not out.get("ok", False):
            return {"ok": False, "score": float(out.get("score", 0.0))}

        # Estimate a robust eye-region depth using periocular skin around iris centers
        depth_mm_eye = np.nan
        ic = out.get("iris_centers", {})
        cand = []
        for k in ("left", "right"):
            if k in ic:
                d = _median_depth(depth, ic[k], win=4)
                if np.isfinite(d): cand.append(d)
        if len(cand):
            depth_mm_eye = float(np.median(cand) * 1000.0)

        return {
            "ok": True,
            "score": float(out.get("score", 1.0)),
            "face_landmarks": out.get("face_landmarks"),
            "left_eye_corners": out.get("left_eye_corners"),
            "right_eye_corners": out.get("right_eye_corners"),
            "iris_centers": out.get("iris_centers", {}),
            "pupils": out.get("pupils", {}),
            "depth_mm_eye": depth_mm_eye
        }
