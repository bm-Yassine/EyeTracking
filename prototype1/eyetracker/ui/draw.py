from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional

Color = Tuple[int, int, int]  # BGR

WHITE: Color = (255, 255, 255)
RED: Color = (0, 0, 255)
GREEN: Color = (0, 255, 0)
YELLOW: Color = (0, 255, 255)
CYAN: Color = (255, 255, 0)

def draw_cross(img: np.ndarray, center: Tuple[int, int], size: int = 18, color: Color = WHITE, thickness: int = 2) -> None:
    x, y = int(center[0]), int(center[1])
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)

def draw_circle(img: np.ndarray, center: Tuple[int, int], radius: int = 12, color: Color = RED, thickness: int = 2) -> None:
    cv2.circle(img, (int(center[0]), int(center[1])), radius, color, thickness, lineType=cv2.LINE_AA)

def draw_hud(
    img: np.ndarray,
    fps_ui: Optional[float] = None,
    fps_cam: Optional[float] = None,
    backend: Optional[str] = None,
    idx: Optional[int] = None,
    total: Optional[int] = None,
    face_present: Optional[bool] = None,
    blink: Optional[bool] = None,
) -> None:
    """Light HUD on top-left; pass any subset of fields."""
    lines = []
    if fps_ui is not None: lines.append(f"UI {fps_ui:5.1f} fps")
    if fps_cam is not None: lines.append(f"Cam {fps_cam:5.1f} fps")
    if backend: lines.append(f"Backend: {backend}")
    if idx is not None and total is not None: lines.append(f"Point {idx+1}/{total}")
    if face_present is not None: lines.append(f"Face: {'YES' if face_present else '—'}")
    if blink is not None: lines.append(f"Blink: {'YES' if blink else '—'}")

    y = 28
    for line in lines:
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        y += 26
