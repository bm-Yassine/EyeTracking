""""
Calibrate script for Eye Tracking Part A with timed calibration points.

This script guides the user through a 20‑point calibration procedure.  After
clicking a single **Start EyeTracking** button, the cross will march through
the first 20 points of the generated calibration grid automatically, spending
exactly 2 seconds on each point.  A global countdown from 40 seconds to 0 is
displayed to indicate progress.  On every frame the current target’s screen
coordinates are logged together with head pose, distance and per‑eye angles.
The resulting parquet files (``frames.parquet`` and ``events.parquet``) live in
``data/sessions/calibN`` where ``N`` is incremented per run (``calib1``,
``calib2``, …).  An optional video recording (``video.mp4``) mirrors the
overlays you see on screen.

The implementation is deliberately self‑contained: only the MediaPipe backend
for 2D iris detection is used.  Head pose is estimated via PnP, and per‑eye
yaw/pitch angles are computed using the camera intrinsics.  RealSense and
SPIGA backends are not supported here (use ``calibrate_webcam.py`` for full
functionality).  See the ``logger.py`` module for the parquet schemas.

Usage::

    python -m eyetracker.app.calibrate --config configs/default.yaml --record

"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

from ..ui.grid import GridSpec, sequence
from ..ui.draw import draw_cross, draw_hud, WHITE, RED
from ..video.capture import VideoCaptureThread
from ..video.writer import VideoWriterMP4
from ..vision.mediapipe_iris import MediaPipeIris
from ..vision.camera_model import load_intrinsics
from ..vision.headpose import solve_head_pose, smart_angles
from ..io.logger import frames_logger, events_logger

from ..quality.gates import blink_surrogate


###

class OnlineRLS2D:
    """
    Recursive Least Squares for y in R^2 with forgetting factor.
    Learns W in R^{D x 2}:  y_hat = x^T W
    """
    def __init__(self, D: int, lam_init: float = 1000.0, ff: float = 0.995):
        self.D = int(D)
        self.W = np.zeros((self.D, 2), dtype=np.float64)
        # P ~ inverse covariance; large diagonal = high initial uncertainty
        self.P = (1.0 / lam_init) * np.eye(self.D, dtype=np.float64)
        self.ff = float(ff)  # forgetting factor in (0,1]; lower = faster adaptation
        self.n_obs = 0

    def update(self, x: np.ndarray, y: np.ndarray):
        """ x: (D,), y: (2,) """
        x = np.asarray(x, np.float64).reshape(-1, 1)
        y = np.asarray(y, np.float64).reshape(1, 2)
        # Gain
        denom = self.ff + float(x.T @ self.P @ x)
        K = (self.P @ x) / denom  # (D,1)        
        # Residual using current W
        y_hat = (x.T @ self.W)    # (1,2)
        # Update W and P
        self.W = self.W + K @ (y - y_hat)  # (D,2)
        self.P = (self.P - (K @ x.T @ self.P)) / self.ff
        self.n_obs += 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64).reshape(-1)
        return (x @ self.W).reshape(2,)


###


class ClickState:
    """Track the most recent left mouse click."""
    def __init__(self) -> None:
        self.clicked: bool = False
        self.pos: Tuple[int, int] = (0, 0)
    def reset(self) -> None:
        self.clicked = False


_click_state = ClickState()


def _mouse_cb(event: int, x: int, y: int, flags: int, userdata) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_state.clicked = True
        _click_state.pos = (x, y)


def _screen_size_tk() -> Tuple[int, int]:
    """Return the width and height of the primary screen using tkinter."""
    import tkinter as tk
    root = tk.Tk(); root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return int(w), int(h)


def _load_config(path: Optional[str]) -> dict:
    if path and Path(path).exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Timed calibration with head/eye logging")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--hud", action="store_true",
                    help="Display FPS and detector status overlay")
    ap.add_argument("--record", action="store_true",
                    help="Write an MP4 alongside the calibration session")
    ap.add_argument(
        "--live-regress", action="store_true",
        help="Enable online linear regressor and draw live predicted gaze point"
    )
    ap.add_argument(
        "--ff", type=float, default=0.995,
        help="RLS forgetting factor (0<ff<=1); smaller adapts faster (default: 0.995)"
    )
    ap.add_argument(
        "--rls-lam", type=float, default=1000.0,
        help="Initial inverse covariance scale for RLS (bigger = more plastic at start)."
    )
    return ap.parse_args()


def eye_angles_deg(iris_xy: Tuple[float, float], eye_ctr_xy: np.ndarray, fx: float, fy: float) -> Tuple[float, float]:
    """Convert pixel offsets to per‑eye yaw/pitch in degrees via small‑angle model."""
    dx = (float(iris_xy[0]) - float(eye_ctr_xy[0])) / float(fx)
    dy = (float(iris_xy[1]) - float(eye_ctr_xy[1])) / float(fy)
    yaw_rad   = np.arctan2(dx, 1.0)
    pitch_rad = np.arctan2(dy, 1.0)
    return float(np.degrees(yaw_rad)), float(np.degrees(pitch_rad))


def next_session_dir(base: Path) -> Path:
    """Return the next calibration directory under ``base``.

    Looks for existing folders named ``calibN`` and returns ``calib{N+1}``.
    Creates the directory before returning.
    """
    base.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("calib")]
    idx = 1
    while f"calib{idx}" in existing:
        idx += 1
    out = base / f"calib{idx}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    live_reg = args.live_regress
    rls = None

    # Screen & grid
    sw, sh = _screen_size_tk()
    grid_cfg = cfg.get("calib_grid", {})
    spec = GridSpec(cols=grid_cfg.get("cols", 5), rows=grid_cfg.get("rows", 5), margin_ratio=grid_cfg.get("margin_ratio", 0.08))
    order_pts = sequence(sw, sh, spec, seed=args.seed)
    # Restrict to first 20 points (2 seconds per point)
    order_pts = order_pts[:20]
    total_pts = len(order_pts)

    # Prepare session directory (calib1, calib2, …)
    sess_base = Path(cfg.get("paths", {}).get("sessions_dir", "data/sessions"))
    sess_dir = next_session_dir(sess_base)
    with open(sess_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    meta = {
        "order": [{"idx": i, "x": int(p[0]), "y": int(p[1])} for i, p in enumerate(order_pts)],
        "screen": {"w": sw, "h": sh},
        "seed": args.seed
    }
    with open(sess_dir / "calib_points.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Camera setup (OpenCV only)
    cam_cfg = cfg.get("camera", {})
    cam_id = int(cam_cfg.get("id", 0))
    req_w = int(cam_cfg.get("width", 1280))
    req_h = int(cam_cfg.get("height", 720))
    fps   = int(cam_cfg.get("fps", 30))
    intr_path = cam_cfg.get("intrinsics_path", None)

    cam: Optional[VideoCaptureThread] = None
    try:
        cam = VideoCaptureThread(cam_id, req_w, req_h, fps).start()
    except Exception as e:
        raise RuntimeError(f"Camera not started: {e}")

    # Acquire one frame to determine actual size
    frame0 = None
    for _ in range(100):
        _, f0 = cam.get_latest()
        if f0 is not None:
            frame0 = f0
            break
        time.sleep(0.02)
    if frame0 is None:
        raise RuntimeError("No frames from camera; cannot continue.")
    cam_h_actual, cam_w_actual = frame0.shape[:2]
    print(f"[i] Camera actual size: {cam_w_actual}x{cam_h_actual}")

    # Load intrinsics & fallback
    K: np.ndarray = np.eye(3, dtype=np.float64)
    dist: np.ndarray = np.zeros((5,), dtype=np.float64)
    fx = float(cam_w_actual)
    fy = float(cam_h_actual)
    if intr_path and Path(intr_path).exists():
        ci = load_intrinsics(intr_path)
        K = ci.K_mat().astype(np.float64).copy()
        dist = ci.dist_vec().astype(np.float64).copy()
        if (ci.width, ci.height) != (cam_w_actual, cam_h_actual):
            sx = cam_w_actual / ci.width
            sy = cam_h_actual / ci.height
            K[0, 0] *= sx; K[1, 1] *= sy
            K[0, 2] *= sx; K[1, 2] *= sy
            print(f"[i] Scaled intrinsics {ci.width}x{ci.height} → {cam_w_actual}x{cam_h_actual} (sx={sx:.3f}, sy={sy:.3f})")
        fx, fy = float(K[0, 0]), float(K[1, 1])
    else:
        # simple pinhole model if no calibration
        K = np.array([[fx, 0, cam_w_actual * 0.5], [0, fy, cam_h_actual * 0.5], [0, 0, 1.0]], dtype=np.float64)
        dist = np.zeros((5,), dtype=np.float64)

    # Backend: use MediaPipeIris exclusively
    try:
        backend = MediaPipeIris()
    except Exception as e:
        raise RuntimeError(f"MediaPipe backend unavailable: {e}")

    # Loggers
    ev_logger = events_logger(str(sess_dir / "events.parquet"))
    fr_logger = frames_logger(str(sess_dir / "frames.parquet"))

    # Optional recorder
    writer: Optional[VideoWriterMP4] = None
    if args.record:
        writer = VideoWriterMP4(str(sess_dir / "video.mp4"), size=(cam_w_actual, cam_h_actual), fps=fps)
        print(f"[i] Recording to: {writer.actual_path}")
   

    # UI
    win = "EyeTracker Calibration"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win, _mouse_cb)

    # Calibration state
    phase = "wait_start"
    calibration_start_time: Optional[float] = None
    last_logged_idx = -1
    frame_counter = 0
    prev_rvec: Optional[np.ndarray] = None
    prev_tvec: Optional[np.ndarray] = None

    try:
        while True:
            # Blank canvas for UI overlay
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

            # Acquire frame and compute features
            t_cam: Optional[float] = None
            frame: Optional[np.ndarray] = None
            out = {"ok": False, "score": 0.0}
            if cam is not None:
                t_cam, frame = cam.get_latest()
                if frame is not None:
                    out = backend.process(frame)
            # Quality gates
            face_ok = bool(out.get("ok", False))
            blink = False
            if face_ok:
                blink, ear_left, ear_right = blink_surrogate(out)

            # Head pose
            hp = None
            yaw = pitch = roll = np.nan
            if face_ok and K is not None:
                try:
                    hp = solve_head_pose(out["face_landmarks"], K, dist, rvec0=prev_rvec, tvec0=prev_tvec)
                    if hp.ok:
                        le = out.get("left_eye_corners")
                        re = out.get("right_eye_corners")
                        left_outer  = le[1] if (isinstance(le, np.ndarray) and le.shape == (2, 2)) else None
                        right_outer = re[1] if (isinstance(re, np.ndarray) and re.shape == (2, 2)) else None
                        yaw, pitch, roll = smart_angles(hp, left_outer, right_outer)
                        prev_rvec, prev_tvec = hp.rvec, hp.tvec
                except Exception:
                    hp = None


            # Per‑eye angles
            left_angles = (np.nan, np.nan)
            right_angles = (np.nan, np.nan)
            if face_ok:
                lec = out.get("left_eye_corners")
                rec = out.get("right_eye_corners")
                lic = out.get("iris_centers", {}).get("left")
                ric = out.get("iris_centers", {}).get("right")
                if isinstance(lec, np.ndarray) and lec.shape == (2, 2) and lic is not None:
                    left_center = 0.5 * (lec[0] + lec[1])
                    left_angles = eye_angles_deg(lic, left_center, fx, fy)
                if isinstance(rec, np.ndarray) and rec.shape == (2, 2) and ric is not None:
                    right_center = 0.5 * (rec[0] + rec[1])
                    right_angles = eye_angles_deg(ric, right_center, fx, fy)

     
                

            # Build row; target_x/target_y will be filled per phase
            t_row = float(t_cam if t_cam is not None else time.monotonic())
            row = {
                "t_mono": t_row,
                "frame_id": int(frame_counter),
                "head_yaw_deg": float(yaw),
                "head_pitch_deg": float(pitch),
                "head_roll_deg": float(roll),
                "head_dist_mm": float(hp.distance_mm) if hp is not None and hp.ok else float("nan"),
                "head_x_mm": float(hp.head_x_mm) if hp is not None and hp.ok else float("nan"),
                "head_y_mm": float(hp.head_y_mm) if hp is not None and hp.ok else float("nan"),
                "head_z_mm": float(hp.head_z_mm) if hp is not None and hp.ok else float("nan"),
                "left_yaw": float(left_angles[0]),
                "left_pitch": float(left_angles[1]),
                "right_yaw": float(right_angles[0]),
                "right_pitch": float(right_angles[1]),
                "face_present": bool(face_ok),
                "blink": bool(blink),
                "target_x": None,
                "target_y": None,
            }

            # Optional video overlay
            if writer is not None and frame is not None:
                fr = frame.copy()
                # Pupils
                pupils = out.get("pupils", {})
                for (cx, cy, r), color in (
                    (pupils.get("left",  (np.nan, np.nan, np.nan)),  (0, 255, 255)),
                    (pupils.get("right", (np.nan, np.nan, np.nan)),  (0, 165, 255))
                ):
                    if np.all(np.isfinite([cx, cy, r])):
                        cv2.circle(fr, (int(round(cx)), int(round(cy))), max(2, int(round(r))), color, 2, lineType=cv2.LINE_AA)
                        cv2.circle(fr, (int(round(cx)), int(round(cy))), 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                # Eye midpoints
                for ec in (out.get("left_eye_corners"), out.get("right_eye_corners")):
                    if isinstance(ec, np.ndarray) and ec.shape == (2, 2):
                        c = 0.5 * (ec[0] + ec[1])
                        cv2.circle(fr, (int(round(c[0])), int(round(c[1]))), 3, (0, 255, 0), -1, cv2.LINE_AA)
                # Head pose axes
                if hp is not None and hp.ok:
                    axis = np.float32([[80, 0, 0], [0, 80, 0], [0, 0, 80]]).reshape(-1, 3)
                    origin = np.float32([[0, 0, 0]]).reshape(-1, 3)
                    try:
                        pts, _ = cv2.projectPoints(np.vstack([origin, axis]), hp.rvec, hp.tvec, K, dist)
                        o, x_pt, y_pt, z_pt = pts.reshape(-1, 2).astype(np.float32)
                        o = (int(round(o[0])), int(round(o[1])))
                        x_i = (int(round(x_pt[0])), int(round(x_pt[1])))
                        y_i = (int(round(y_pt[0])), int(round(y_pt[1])))
                        z_i = (int(round(z_pt[0])), int(round(z_pt[1])))
                        cv2.line(fr, o, x_i, (0, 0, 255), 2)
                        cv2.line(fr, o, y_i, (0, 255, 0), 2)
                        cv2.line(fr, o, z_i, (255, 0, 0), 2)
                    except Exception:
                        pass
                    # Add text overlays
                    cv2.putText(fr, f"Head y/p/r: {yaw:+5.1f} {pitch:+5.1f} {roll:+5.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    if np.isfinite(row["head_dist_mm"]):
                        cv2.putText(fr, f"dist: {row['head_dist_mm']:.0f} mm", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                # Eye angles
                cv2.putText(fr, f"L (yaw,pitch): {left_angles[0]:+4.1f}, {left_angles[1]:+4.1f} deg", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2, cv2.LINE_AA)
                cv2.putText(fr, f"R (yaw,pitch): {right_angles[0]:+4.1f}, {right_angles[1]:+4.1f} deg", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2, cv2.LINE_AA)
                writer.write(fr, info={"idx": last_logged_idx})

            # Phase logic
            if phase == "wait_start":
                # Draw start button in the centre
                start_centre = (sw // 2, sh // 2)
                bx1, by1, bx2, by2 = start_centre[0] - 150, start_centre[1] - 40, start_centre[0] + 150, start_centre[1] + 40
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (220, 220, 220), -1)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (50, 50, 50), 2)
                (tw, th), baseline = cv2.getTextSize("Start EyeTracking", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(canvas, "Start EyeTracking", (start_centre[0] - tw // 2, start_centre[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                if _click_state.clicked:
                    cx_click, cy_click = _click_state.pos
                    _click_state.reset()
                    if bx1 <= cx_click <= bx2 and by1 <= cy_click <= by2:
                        # Log start event as idx -1
                        t_mono = time.monotonic()
                        ev_logger.write({
                            "t_mono": float(t_mono),
                            "idx": -1,
                            "target_x": int(start_centre[0]),
                            "target_y": int(start_centre[1]),
                            "click_x": int(cx_click),
                            "click_y": int(cy_click),
                        })
                        phase = "calibrating"
                        calibration_start_time = time.monotonic()
                        last_logged_idx = -1
                        # Write the initial frame row tagged with the centre
                        row["target_x"] = float(start_centre[0])
                        row["target_y"] = float(start_centre[1])
                        fr_logger.write(row)
                        frame_counter += 1
                        continue

            elif phase == "calibrating":
                if calibration_start_time is None:
                    calibration_start_time = time.monotonic()
                # Determine current index based on elapsed time (2 seconds per point)
                elapsed_total = time.monotonic() - calibration_start_time
                cur_idx = int(elapsed_total / 2.0)
                if cur_idx >= total_pts:
                    phase = "wait_end"
                else:
                    current_pt = order_pts[cur_idx]
                    # Only log point transitions once
                    if cur_idx > last_logged_idx:
                        t_mono = time.monotonic()
                        ev_logger.write({
                            "t_mono": float(t_mono),
                            "idx": int(cur_idx),
                            "target_x": int(current_pt[0]),
                            "target_y": int(current_pt[1]),
                            "click_x": 0,
                            "click_y": 0,
                        })
                        last_logged_idx = cur_idx
                    # Draw the current cross with a surrounding red circle and the remaining time in red.
                    next_pt = order_pts[cur_idx + 1] if cur_idx + 1 < total_pts else current_pt
                    # Compute remaining time (global countdown)
                    remaining = max(0.0, 2.0 * total_pts - elapsed_total)
                    # Format remaining seconds as an integer for display (e.g. 40, 39, …)
                    remaining_int = int(np.ceil(remaining)) if remaining > 0 else 0
                    remaining_str = f"{remaining_int}"
                    # Draw white cross
                    draw_cross(canvas, current_pt, size=22, color=WHITE, thickness=3)
                    # Draw red circle around the cross
                    cv2.circle(canvas, (int(current_pt[0]), int(current_pt[1])), 28, RED, 3)
                    # Draw the remaining time in red centred on the cross
                    (tw, th), baseline = cv2.getTextSize(remaining_str, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    tx = int(current_pt[0] - tw / 2)
                    ty = int(current_pt[1] + th / 2)
                    cv2.putText(canvas, remaining_str, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2, cv2.LINE_AA)
                    # Draw next point as a small red cross
                    draw_cross(canvas, next_pt, size=16, color=RED, thickness=2)
                    # Fill current point coordinates in row
                    row["target_x"] = float(current_pt[0])
                    row["target_y"] = float(current_pt[1])

            elif phase == "wait_end":
                # Draw end button
                end_centre = (sw // 2, sh // 2)
                bx1, by1, bx2, by2 = end_centre[0] - 150, end_centre[1] - 40, end_centre[0] + 150, end_centre[1] + 40
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (220, 220, 220), -1)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (50, 50, 50), 2)
                (tw, th), baseline = cv2.getTextSize("End Calibration", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(canvas, "End Calibration", (end_centre[0] - tw // 2, end_centre[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                if _click_state.clicked:
                    cx_click, cy_click = _click_state.pos
                    _click_state.reset()
                    if bx1 <= cx_click <= bx2 and by1 <= cy_click <= by2:
                        # Log end event
                        t_mono = time.monotonic()
                        ev_logger.write({
                            "t_mono": float(t_mono),
                            "idx": int(total_pts),
                            "target_x": int(end_centre[0]),
                            "target_y": int(end_centre[1]),
                            "click_x": int(cx_click),
                            "click_y": int(cy_click),
                        })
                        # Write final row tagged with end centre
                        row["target_x"] = float(end_centre[0])
                        row["target_y"] = float(end_centre[1])
                        fr_logger.write(row)
                        frame_counter += 1
                        break

            # HUD overlay (optional)
            if args.hud:
                # Use built‑in draw_hud to show FPS and status
                now = time.monotonic()
                draw_hud(canvas, fps_ui=None, fps_cam=None, backend="mediapipe", idx=(last_logged_idx if phase == "calibrating" else None), total=total_pts, face_present=face_ok, blink=blink)

            # Show UI
            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            # Write row for each frame except during wait_start
            if phase != "wait_start":
                fr_logger.write(row)
                frame_counter += 1
        # End main loop
        cv2.destroyWindow(win)
    finally:
        # Close loggers and release resources
        try:
            fr_logger.close()
        except Exception:
            pass
        try:
            ev_logger.close()
        except Exception:
            pass
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        try:
            if cam is not None:
                cam.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()