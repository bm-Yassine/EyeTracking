from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional

try:
    import mediapipe as mp
except Exception as e:  # pragma: no cover
    mp = None

# FaceMesh landmark indices (MediaPipe canonical mesh)
# NOTE: Values below follow commonly used indices for eye corners & iris when refine_landmarks=True
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
LEFT_EYE_INNER  = 362
LEFT_EYE_OUTER  = 263

# Iris landmarks (5 per eye) when FaceMesh(refine_landmarks=True)
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS  = [473, 474, 475, 476, 477]

# A few nose/bridge anchors helpful for head pose proxy (use proper PnP later)
NOSE_BRIDGE = [6, 168, 197, 5]  # tip/mid/bridge-ish points

def _fit_circle_2d(points_xy: np.ndarray) -> tuple[float, float, float]:
    """
    Algebraic circle fit (Taubin-like least squares, small N fine).
    points_xy: Nx2
    returns (cx, cy, r)
    """
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    c, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = c
    r = np.sqrt(max(c0 + cx**2 + cy**2, 0.0))
    return float(cx), float(cy), float(r)

class MediaPipeIris:
    """
    Thin wrapper that returns pixel-space landmarks including iris rings and corners.
    Does NOT compute angles; do that with camera intrinsics upstream.
    """
    def __init__(self, max_faces: int = 1, min_det_conf: float = 0.5, min_track_conf: float = 0.5):
        if mp is None:
            raise RuntimeError("MediaPipe not available. Install 'mediapipe' or use the SPIGA adapter.")
        self._fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,  # required for iris points
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

    
    
    def process(self, frame_bgr) -> Dict[str, Any]:
        """
        Returns a standardized dict:
        {
        'ok': bool,
          'face_landmarks': (N,3) in pixels,
          'left_iris': (5,2), 'right_iris': (5,2),
          'left_eye_corners': (2,2), 'right_eye_corners': (2,2),
          'nose_bridge': (K,2),
          'pupils': {'left': (cx,cy,r), 'right': (cx,cy,r)}  # ellipse fit on 4 rim pts
          'iris_centers': {'left': (cx,cy), 'right': (cx,cy)} # landmark 473/468 directly
          'score': float
        """
        h, w = frame_bgr.shape[:2]

        # MediaPipe expects RGB
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._fm.process(frame_rgb)

        if not res.multi_face_landmarks:
            return {'ok': False, 'score': 0.0}

        # Landmarks to pixel coords
        lms = res.multi_face_landmarks[0].landmark
        arr = np.array([[lm.x * w, lm.y * h, lm.z] for lm in lms], dtype=np.float32)

        def pick(idx_list):
            # returns (k,2) float32
            return arr[idx_list, :2].astype(np.float32)

        left_iris = pick(LEFT_IRIS)
        right_iris = pick(RIGHT_IRIS)
        left_eye_corners = pick([LEFT_EYE_INNER, LEFT_EYE_OUTER])
        right_eye_corners = pick([RIGHT_EYE_INNER, RIGHT_EYE_OUTER])
        nose_bridge = pick(NOSE_BRIDGE)

        iris_center_right = arr[RIGHT_IRIS[0], :2].astype(np.float32)  # 468
        iris_center_left  = arr[LEFT_IRIS[0],  :2].astype(np.float32)  # 473


        # ---- robust pupil circle fit ----
        def _try_fit_circle(pts: np.ndarray) -> tuple[float, float, float]:
            try:
                if pts is None or len(pts) < 3:
                    return (float("nan"), float("nan"), float("nan"))
                if not np.isfinite(pts).all():
                    return (float("nan"), float("nan"), float("nan"))
                return _fit_circle_2d(pts)
            except Exception:
                return (float("nan"), float("nan"), float("nan"))

        lx, ly, lr = _try_fit_circle(left_iris[1:])
        rx, ry, rr = _try_fit_circle(right_iris[1:])

        # ---- eye-quality proxy (0..1) ----
        def _ratio(corners: np.ndarray | None, radius: float) -> float:
            if corners is None or not np.isfinite(radius):
                return 0.0
            a = corners[0].astype(np.float32)
            b = corners[1].astype(np.float32)
            w_eye = float(np.linalg.norm(a - b))
            if w_eye <= 1.0:
                return 0.0
            return float(np.clip(radius / (0.5 * w_eye), 0.0, 1.0))

        lratio = _ratio(left_eye_corners, lr)
        rratio = _ratio(right_eye_corners, rr)
        score = float((lratio + rratio) * 0.5)

        return {
            'ok': True,
            'face_landmarks': arr,                # (478, 3) when refine_landmarks=True
            'left_iris': left_iris,               # (5, 2) typically
            'right_iris': right_iris,
            'left_eye_corners': left_eye_corners, # (2, 2)
            'right_eye_corners': right_eye_corners,
            'nose_bridge': nose_bridge,
            'iris_centers': {'left': tuple(map(float, iris_center_left)),
                             'right': tuple(map(float, iris_center_right))},
            'pupils': {'left': (lx, ly, lr), 'right': (rx, ry, rr)},
            'score': score
        }
