# vision/backends/webcam2d.py
import numpy as np
from mediapipe_iris import MediaPipeIris  # uses your existing wrapper

class Webcam2DIris:
    """
    Thin adapter on top of MediaPipeIris, returning a normalized dict.
    """
    def __init__(self):
        self.mp = MediaPipeIris()

    def process(self, frame_bgr, depth=None, K=None, dist=None):
        out = self.mp.process(frame_bgr)  # expect keys: ok, score, face_landmarks, eye corners, iris centers, pupils...
        if not out.get("ok", False):
            return {"ok": False, "score": float(out.get("score", 0.0))}
        # Normalize keys
        return {
            "ok": True,
            "score": float(out.get("score", 1.0)),
            "face_landmarks": out.get("face_landmarks"),
            "left_eye_corners": out.get("left_eye_corners"),
            "right_eye_corners": out.get("right_eye_corners"),
            "iris_centers": out.get("iris_centers", {}),
            "pupils": out.get("pupils", {})
        }
