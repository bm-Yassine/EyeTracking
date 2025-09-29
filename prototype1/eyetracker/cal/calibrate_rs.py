"""
Calibration script for the EyeTracker using an Intel RealSense camera.

This module mirrors the original ``eyetracker.app.calibrate`` but simplifies
it for depth‑assisted calibration: only the MediaPipe landmark backend is
supported and the camera source is fixed to Intel RealSense (RGB and
depth).  Unnecessary command‑line options have been removed.  The script
records calibration points, estimates head pose from depth, logs frames
and events and optionally records video.

Usage:

    python -m eyetracker.app.calibrate_rs --hud --record

Command‑line arguments allow specifying RealSense serial number and
stream resolution/fps.  A configuration file may provide camera and
calibration grid settings.  If RealSense SDK is unavailable an error
is raised.

"""

from __future__ import annotations
import argparse
import time
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from ..ui.grid import GridSpec, sequence
from ..ui.draw import draw_cross, WHITE, RED
from ..video.rs_capture import RealSenseCapture
from ..video.writer import VideoWriterMP4
from ..vision.mediapipe_iris import MediaPipeIris
from ..vision.headpose import smart_angles, _closest_equivalent, _wrap180
from ..vision.headpose_depth import solve_head_pose_with_depth
from ..io.logger import frames_logger, events_logger, session_logger
from ..features.feature_builder import FeatureBuilder, BuilderContext
from ..quality.gates import face_present as gate_face, blink_surrogate as blink_surrogate_fn


class ClickState:
    """Simple container to track mouse clicks for calibration targets."""

    def __init__(self) -> None:
        self.clicked = False
        self.pos = (0, 0)

    def reset(self) -> None:
        self.clicked = False


_click_state = ClickState()


def _mouse_cb(event: int, x: int, y: int, flags: int, userdata) -> None:
    """OpenCV mouse callback to record left‑clicks."""
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_state.clicked = True
        _click_state.pos = (x, y)


def _screen_size_tk() -> Tuple[int, int]:
    """Return the screen resolution using a hidden Tkinter window."""
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return int(w), int(h)


def _load_config(path: Optional[str]) -> dict:
    """Load a YAML configuration file if it exists; otherwise return an empty dict."""
    if path and Path(path).exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for RealSense calibration."""
    ap = argparse.ArgumentParser(
        description="Calibration UI using an Intel RealSense camera and depth for head pose."
    )
    ap.add_argument("--config", type=str, default="configs/default.yaml",
                    help="Path to YAML configuration file.")
    ap.add_argument("--seed", type=int, default=17,
                    help="Random seed for calibration point order.")
    ap.add_argument("--hud", action="store_true",
                    help="Show on‑screen HUD with quality and pose information.")
    ap.add_argument("--record", action="store_true",
                    help="Record session video to MP4 in the session folder.")
    # RealSense specific parameters
    ap.add_argument("--rs-serial", type=str, default=None,
                    help="Optional RealSense camera serial number to select a specific device.")
    ap.add_argument("--rs-width", type=int, default=1280,
                    help="RealSense RGB/depth stream width (pixels).")
    ap.add_argument("--rs-height", type=int, default=720,
                    help="RealSense RGB/depth stream height (pixels).")
    ap.add_argument("--rs-fps", type=int, default=30,
                    help="RealSense stream frame rate.")
    return ap.parse_args()


def eye_angles_deg(iris_xy, eye_ctr_xy, fx, fy) -> Tuple[float, float]:
    """Convert pixel offsets between an iris centre and eye midpoint into yaw and pitch angles."""
    dx = (float(iris_xy[0]) - float(eye_ctr_xy[0])) / float(fx)
    dy = (float(iris_xy[1]) - float(eye_ctr_xy[1])) / float(fy)
    yaw_rad = np.arctan2(dx, 1.0)
    pitch_rad = np.arctan2(dy, 1.0)
    return float(np.degrees(yaw_rad)), float(np.degrees(pitch_rad))


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    # Screen and calibration grid
    sw, sh = _screen_size_tk()
    grid_cfg = cfg.get("calib_grid", {})
    spec = GridSpec(cols=grid_cfg.get("cols", 5),
                    rows=grid_cfg.get("rows", 5),
                    margin_ratio=grid_cfg.get("margin_ratio", 0.08))
    order_pts = sequence(sw, sh, spec, seed=args.seed)
    total_pts = len(order_pts)

    # Session folder
    ts_run = time.strftime("%Y%m%d_%H%M%S")
    sess_dir = Path(cfg.get("paths", {}).get("sessions_dir", "data/sessions")) / f"calib_{ts_run}"
    sess_dir.mkdir(parents=True, exist_ok=True)
    # Save snapshot of config and calibration points
    with open(sess_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    meta = {
        "order": [{"idx": i, "x": int(p[0]), "y": int(p[1])} for i, p in enumerate(order_pts)],
        "screen": {"w": sw, "h": sh, "dpi": None},
        "seed": args.seed,
    }
    with open(sess_dir / "calib_points.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Start RealSense camera
    cam: Optional[RealSenseCapture] = None
    cam = RealSenseCapture(width=args.rs_width, height=args.rs_height, fps=args.rs_fps,
                           serial=args.rs_serial).start()

    # Grab first frame to know actual camera size
    frame0 = None
    depth0 = None
    for _ in range(100):
        _, f0, d0 = cam.get_latest()
        if f0 is not None and d0 is not None:
            frame0 = f0
            depth0 = d0
            break
        time.sleep(0.02)
    if frame0 is None:
        raise RuntimeError("No frames from RealSense camera; cannot continue.")
    cam_h_actual, cam_w_actual = frame0.shape[:2]
    print(f"[i] RealSense camera size: {cam_w_actual}x{cam_h_actual}")

    # Intrinsics from RealSense
    K = cam.color_K().astype(np.float64)
    dist = cam.color_dist().astype(np.float64).reshape(-1,)
    fx, fy = float(K[0, 0]), float(K[1, 1])

    # Instantiate MediaPipe backend
    try:
        backend = MediaPipeIris()
    except Exception as e:
        raise RuntimeError(f"MediaPipe backend unavailable: {e}")

    # Prepare logging
    ev_logger = events_logger(str(sess_dir / "events.parquet"))
    fr_logger = frames_logger(str(sess_dir / "frames.parquet"))
    bctx = BuilderContext(screen_w=sw, screen_h=sh, cam_w=cam_w_actual, cam_h=cam_h_actual,
                          fx=fx, fy=fy, K=K, dist=dist)
    builder = FeatureBuilder(bctx)
    sess_log = session_logger(str(sess_dir / "session.parquet"))
    sess_log.write({
        "camera_model": "realsense",
        "screen_w": int(sw), "screen_h": int(sh),
        "cam_w": int(cam_w_actual), "cam_h": int(cam_h_actual),
    })

    # Optional video recorder
    writer: Optional[VideoWriterMP4] = None
    if args.record:
        writer = VideoWriterMP4(str(sess_dir / "video.mp4"), size=(cam_w_actual, cam_h_actual), fps=args.rs_fps)
        print(f"[i] Recording to: {writer.actual_path}")

    # Setup UI
    win = "EyeTracker Calibration (RealSense)"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win, _mouse_cb)

    idx = 0
    prev_angles = None

    try:
        while idx < total_pts:
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            current_pt = order_pts[idx]
            next_pt = order_pts[idx + 1] if idx + 1 < total_pts else order_pts[idx]
            draw_cross(canvas, current_pt, size=22, color=WHITE, thickness=3)
            draw_cross(canvas, next_pt, size=16, color=RED, thickness=2)

            # Fetch latest frames
            t_cam, frame, depth_m = cam.get_latest()
            out = {"ok": False, "score": 0.0}
            if frame is not None:
                out = backend.process(frame)

            # Quality and blink surrogate
            score = float(out.get("score", 0.0))
            blink_flag, _, _ = blink_surrogate_fn(out) if score >= 0 else (False, 0.0, 0.0)
            if score < 0.10:
                blink_flag = True

            # Head pose using depth
            hp = None
            yaw = pitch = roll = np.nan
            roll_2d = None
            if out.get("ok", False) and depth_m is not None:
                le = out.get("left_eye_corners")
                re = out.get("right_eye_corners")
                left_outer = le[1] if (isinstance(le, np.ndarray) and le.shape == (2, 2)) else None
                right_outer = re[1] if (isinstance(re, np.ndarray) and re.shape == (2, 2)) else None
                if left_outer is not None and right_outer is not None:
                    dx = float(right_outer[0] - left_outer[0])
                    dy = float(right_outer[1] - left_outer[1])
                    if abs(dx) + abs(dy) > 1e-6:
                        roll_2d = _wrap180(np.degrees(np.arctan2(dy, dx)))
                hp = solve_head_pose_with_depth(out["face_landmarks"], K, depth_m)
                if hp is not None and getattr(hp, "ok", False):
                    y0, p0, r0 = smart_angles(hp, left_outer, right_outer, prev_angles)
                    yaw, pitch, roll = _closest_equivalent(y0, p0, r0, prev_angles, roll_2d)
                    prev_angles = (yaw, pitch, roll)

            # Per‑eye angles
            left_angles = (np.nan, np.nan)
            right_angles = (np.nan, np.nan)
            if out.get("ok", False):
                lec = out.get("left_eye_corners")
                rec = out.get("right_eye_corners")
                lic = out.get("iris_centers", {}).get("left", None)
                ric = out.get("iris_centers", {}).get("right", None)
                if isinstance(lec, np.ndarray) and lec.shape == (2, 2) and lic is not None:
                    left_center = 0.5 * (lec[0] + lec[1])
                    left_angles = eye_angles_deg(lic, left_center, fx, fy)
                if isinstance(rec, np.ndarray) and rec.shape == (2, 2) and ric is not None:
                    right_center = 0.5 * (rec[0] + rec[1])
                    right_angles = eye_angles_deg(ric, right_center, fx, fy)

            # Build frame record and log
            t = t_cam if t_cam is not None else time.monotonic()
            if fr_logger is not None and builder is not None:
                row = {
                    "t_mono": float(t), "frame_id": int(idx),
                    "head_yaw_deg": float(yaw), "head_pitch_deg": float(pitch), "head_roll_deg": float(roll),
                    "head_dist_mm": float(getattr(hp, "distance_mm", np.nan)) if hp is not None else float("nan"),
                    "head_x_mm": float(hp.head_x_mm) if hp is not None and hp.ok else float("nan"),
                    "head_y_mm": float(hp.head_y_mm) if hp is not None and hp.ok else float("nan"),
                    "head_z_mm": float(hp.head_z_mm) if hp is not None and hp.ok else float("nan"),
                    "left_yaw": float(left_angles[0]), "left_pitch": float(left_angles[1]),
                    "right_yaw": float(right_angles[0]), "right_pitch": float(right_angles[1]),
                    "face_present": bool(out.get("ok", False)), "blink": bool(blink_flag),
                    "target_x": None, "target_y": None,
                }
                fr_logger.write(row)

            # Draw simple HUD if requested
            if args.hud:
                lines = [
                    f"Point {idx + 1}/{total_pts}",
                    f"Face score: {score:4.2f}",
                    f"Blink: {'YES' if blink_flag else 'NO'}",
                    f"Yaw/Pitch/Roll: {yaw:5.1f}/{pitch:5.1f}/{roll:5.1f}",
                ]
                y0 = 28
                for line in lines:
                    cv2.putText(canvas, line, (16, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(canvas, line, (16, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    y0 += 22

            # Overlay calibration crosses onto camera preview for display
            if frame is not None:
                fr_small = cv2.resize(frame, (sw, sh)) if frame.shape[1] != sw or frame.shape[0] != sh else frame
                display = cv2.addWeighted(fr_small, 0.5, canvas, 0.5, 0)
            else:
                display = canvas
            cv2.imshow(win, display)

            # Write video if enabled
            if writer is not None and frame is not None:
                writer.write(frame)

            # Handle mouse click events
            if _click_state.clicked:
                ev_logger.write({
                    "t_mono": float(t),
                    "frame_id": int(idx),
                    "target_idx": int(idx),
                    "x": int(current_pt[0]),
                    "y": int(current_pt[1]),
                    "click_x": int(_click_state.pos[0]),
                    "click_y": int(_click_state.pos[1]),
                })
                _click_state.reset()
                idx += 1

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cv2.destroyAllWindows()
        cam.stop()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()