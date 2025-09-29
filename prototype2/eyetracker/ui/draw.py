import cv2 as cv
import numpy as np
from typing import Tuple

def draw_cross(img, center: Tuple[int, int], color_bgr=(255, 255, 255),
               r: int = 14, t: int = 2):
    x, y = int(center[0]), int(center[1])
    cv.line(img, (x - r, y), (x + r, y), color_bgr, t, lineType=cv.LINE_AA)
    cv.line(img, (x, y - r), (x, y + r), color_bgr, t, lineType=cv.LINE_AA)

def _text_size(text, font, scale, thickness):
    size, _ = cv.getTextSize(text, font, scale, thickness)
    return size  # (w, h)

def draw_text_centered(img, center: Tuple[int, int], text: str,
                       font=cv.FONT_HERSHEY_SIMPLEX, scale=0.9,
                       color=(0, 0, 255), thickness=2):
    w, h = _text_size(text, font, scale, thickness)
    x = int(center[0] - w / 2)
    y = int(center[1] + h / 2)
    cv.putText(img, text, (x, y), font, scale, color, thickness, lineType=cv.LINE_AA)

def draw_countdown_cross(
    img,
    center: Tuple[int, int],
    *,
    curr_cross_color=(255, 255, 255),
    ring_color=(0, 0, 255),
    cross_radius=14,
    cross_thickness=2,
    ring_radius=40,
    ring_thickness=3,
    total_seconds_left: float = 40.0,
    font_scale: float = 0.9
):
    # white cross
    draw_cross(img, center, curr_cross_color, cross_radius, cross_thickness)
    # red circle around it
    cv.circle(img, (int(center[0]), int(center[1])), ring_radius, ring_color, ring_thickness, lineType=cv.LINE_AA)
    # red total countdown number inside the cross
    secs = max(0, int(np.ceil(total_seconds_left)))
    draw_text_centered(img, center, str(secs), scale=font_scale, color=ring_color, thickness=2)
