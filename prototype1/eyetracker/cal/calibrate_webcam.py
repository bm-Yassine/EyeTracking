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
from ..ui.draw import draw_cross, draw_hud, draw_circle, WHITE, RED, YELLOW
from ..video.capture import VideoCaptureThread
from ..video.writer import VideoWriterMP4
from ..vision.mediapipe_iris import MediaPipeIris
from ..vision.spiga_adapter import SpigaAdapter
from ..vision.camera_model import load_intrinsics
from ..vision.headpose import solve_head_pose, smart_angles, _closest_equivalent, _wrap180

from ..io.logger import frames_logger, events_logger, session_logger


# --- helper drawing functions ------------------------------------------------
def _draw_button(img: np.ndarray, centre: Tuple[int, int], text: str, size: Tuple[int, int]=(250, 80)) -> Tuple[int,int,int,int]:
    """Draw a rectangular button with centred text.

    Returns the bounding box (x1, y1, x2, y2) for hit‑testing.
    """
    cx, cy = centre
    w, h = size
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)
    # Draw filled rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (220, 220, 220), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), 2)
    # Draw centred text
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    tx = int(cx - tw / 2)
    ty = int(cy + th / 2)
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return x1, y1, x2, y2


def _draw_cross_with_countdown(
    img: np.ndarray,
    centre: Tuple[int, int],
    number: Optional[int],
    *,
    cross_size: int = 20,
    circle_radius: int = 28,
    cross_color=WHITE,
    circle_color=RED,
    text_color=RED,
    thickness: int = 3,
) -> None:
    """Draw a calibration cross with a surrounding circle and a number."""
    # Draw cross
    draw_cross(img, centre, size=cross_size, color=cross_color, thickness=thickness)
    # Draw circle
    draw_circle(img, centre, radius=circle_radius, color=circle_color, thickness=thickness)
    # Draw number
    if number is not None:
        text = str(int(number))
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        tx = int(centre[0] - tw / 2)
        ty = int(centre[1] + th / 2)
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)


class ClickState:
    """Track the most recent left mouse click."""
    def __init__(self):
        self.clicked = False
        self.pos = (0, 0)
    def reset(self):
        self.clicked = False


_click_state = ClickState()


def _mouse_cb(event, x, y, flags, userdata):
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


def parse_args():
    ap = argparse.ArgumentParser(description="Calibration UI with start/end buttons and countdown")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--hud", action="store_true")
    ap.add_argument("--backend", choices=["mediapipe", "spiga", "none"], default="mediapipe")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--camera", choices=["opencv", "realsense"], default="opencv",
                    help="Select RGB webcam via OpenCV (PnP) or RealSense (depth‑assisted).")
    ap.add_argument("--rs-serial", type=str, default=None, help="Optional RealSense serial")
    ap.add_argument("--rs-width",  type=int, default=1280)
    ap.add_argument("--rs-height", type=int, default=720)
    ap.add_argument("--rs-fps",    type=int, default=30)
    return ap.parse_args()


def eye_angles_deg(iris_xy, eye_ctr_xy, fx, fy):
    dx = (float(iris_xy[0]) - float(eye_ctr_xy[0])) / float(fx)
    dy = (float(iris_xy[1]) - float(eye_ctr_xy[1])) / float(fy)
    yaw_rad   = np.arctan2(dx, 1.0)
    pitch_rad = np.arctan2(dy, 1.0)
    return float(np.degrees(yaw_rad)), float(np.degrees(pitch_rad))


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    # Screen & grid
    sw, sh = _screen_size_tk()
    grid_cfg = cfg.get("calib_grid", {})
    spec = GridSpec(cols=grid_cfg.get("cols", 5), rows=grid_cfg.get("rows", 5), margin_ratio=grid_cfg.get("margin_ratio", 0.08))
    order_pts = sequence(sw, sh, spec, seed=args.seed)
    total_pts = len(order_pts)

    # Session folder setup
    ts_run = time.strftime("%Y%m%d_%H%M%S")
    sess_dir = Path(cfg.get("paths", {}).get("sessions_dir", "data/sessions")) / f"calib_{ts_run}"
    sess_dir.mkdir(parents=True, exist_ok=True)
    with open(sess_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    meta = {"order": [{"idx": i, "x": int(p[0]), "y": int(p[1])} for i, p in enumerate(order_pts)],
            "screen": {"w": sw, "h": sh, "dpi": None}, "seed": args.seed}
    with open(sess_dir / "calib_points.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Camera & backend
    cam_cfg = cfg.get("camera", {})
    cam_id = int(cam_cfg.get("id", 0))
    req_w = int(cam_cfg.get("width", 1280))
    req_h = int(cam_cfg.get("height", 720))
    fps   = int(cam_cfg.get("fps", 30))
    intr_path = cam_cfg.get("intrinsics_path", None)
    cam_backend = args.camera

    cam = None
    rs = None
    try:
        if cam_backend == "opencv":
            cam = VideoCaptureThread(cam_id, req_w, req_h, fps).start()
        else:
            _RS_OK = False
            if cam_backend == "realsense":
                try:
                    from ..video.rs_capture import RealSenseCapture
                    from ..vision.headpose_depth import solve_head_pose_with_depth
                    _RS_OK = True
                except Exception as e:
                    raise RuntimeError(f"RealSense selected but backend unavailable: {e}")
            rs = RealSenseCapture(width=args.rs_width, height=args.rs_height, fps=args.rs_fps, serial=args.rs_serial).start()
            cam = rs
    except Exception as e:
        raise RuntimeError(f"No camera started: {e}")

    # Acquire first frame to determine camera resolution
    frame0 = None
    for _ in range(100):
        if cam_backend == "realsense":
            _, f0, _ = cam.get_latest()
        else:
            _, f0 = cam.get_latest()
        if f0 is not None:
            frame0 = f0; break
        time.sleep(0.02)
    if frame0 is None:
        raise RuntimeError("No frames from camera; cannot continue.")
    cam_h_actual, cam_w_actual = frame0.shape[:2]
    print(f"[i] Camera actual size: {cam_w_actual}x{cam_h_actual}")

    # Load intrinsics and scale to actual size
    K = None; dist = None; fx = float(cam_w_actual); fy = float(cam_h_actual)
    if cam_backend == "realsense":
        K = cam.color_K().astype(np.float64)
        dist = cam.color_dist().astype(np.float64).reshape(-1,)
        fx, fy = float(K[0,0]), float(K[1,1])
    else:
        if intr_path and Path(intr_path).exists():
            ci = load_intrinsics(intr_path)
            K = ci.K_mat().astype(np.float64).copy()
            dist = ci.dist_vec().astype(np.float64).copy()
            if (ci.width, ci.height) != (cam_w_actual, cam_h_actual):
                sx = cam_w_actual / ci.width
                sy = cam_h_actual / ci.height
                K[0,0] *= sx; K[1,1] *= sy
                K[0,2] *= sx; K[1,2] *= sy
                print(f"[i] Scaled intrinsics {ci.width}x{ci.height} → {cam_w_actual}x{cam_h_actual} (sx={sx:.3f}, sy={sy:.3f})")
            fx, fy = float(K[0,0]), float(K[1,1])
        else:
            K = np.array([[fx, 0, cam_w_actual*0.5], [0, fy, cam_h_actual*0.5], [0, 0, 1.0]], dtype=np.float64)
            dist = np.zeros((5,), dtype=np.float64)

    # Backend selection
    backend_name = args.backend
    backend = None
    if backend_name == "mediapipe":
        try:
            backend = MediaPipeIris()
        except Exception as e:
            print(f"[!] MediaPipe backend unavailable: {e}")
            backend_name = "none"
    elif backend_name == "spiga":
        try:
            backend = SpigaAdapter()
        except Exception as e:
            print(f"[!] SPIGA backend unavailable: {e}")
            backend_name = "none"

    blink_surrogate_fn = lambda _out: (False, 0.0, 0.0)
    blink = None
    prev_rvec = None
    prev_tvec = None
    prev_angles = None
    frame_counter = 0

    ev_logger = events_logger(str(sess_dir / "events.parquet"))
    fr_logger = None
    builder = None
    sess_log = None
    if backend_name != "none":
        from ..features.feature_builder import FeatureBuilder, BuilderContext
        from ..quality.gates import face_present as gate_face, blink_surrogate as blink_surrogate_fn
        bctx = BuilderContext(screen_w=sw, screen_h=sh, cam_w=cam_w_actual, cam_h=cam_h_actual, fx=fx, fy=fy, K=K, dist=dist)
        builder = FeatureBuilder(bctx)
        fr_logger = frames_logger(str(sess_dir / "frames.parquet"))
        camera_model = cam_cfg.get("model")
        sess_log = session_logger(str(sess_dir / "session.parquet"))
        sess_log.write({"camera_model": str(camera_model), "screen_w": int(sw), "screen_h": int(sh), "cam_w": int(cam_w_actual), "cam_h": int(cam_h_actual)})

    # Optional recorder
    writer = None
    if args.record and cam is not None:
        writer = VideoWriterMP4(str(sess_dir / "video.mp4"), size=(cam_w_actual, cam_h_actual), fps=fps)
        print(f"[i] Recording to: {writer.actual_path}")

    # UI window
    win = "EyeTracker Calibration"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win, _mouse_cb)

    # Calibration state
    phase = "wait_start"  # wait_start → calibrating → wait_end → done
    idx = 0
    last_ui_tick = time.monotonic(); ui_frames = 0; ui_fps = 0.0
    cross_start_time: Optional[float] = None
    COUNTDOWN_SECS = 2
    COUNTDOWN_NUMBERS = 10

    try:
        while True:
            # Blank canvas
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            current_pt = order_pts[idx] if idx < total_pts else order_pts[-1]
            next_pt    = order_pts[idx+1] if (idx+1) < total_pts else current_pt

            # Acquire camera frame and process backend
            t_cam = None; frame = None; out = {"ok": False, "score": 0.0}
            cam_fps = None; depth_m = None
            if cam is not None:
                cam_fps = getattr(cam, "get_fps", lambda: None)()
                if cam_backend == "realsense":
                    t_cam, frame, depth_m = cam.get_latest()
                else:
                    t_cam, frame = cam.get_latest()
                if frame is not None and backend is not None:
                    out = backend.process(frame)
                # Quality gates
                score = float(out.get("score", 0.0))
                blink, _, _ = blink_surrogate_fn(out) if backend_name != "none" else (False, 0.0, 0.0)
                if score < 0.10:
                    blink = True
                # Head pose
                hp = None
                yaw = pitch = roll = np.nan
                roll_2d = None
                if K is not None and out.get("ok", False):
                    # 2D roll from eye corners
                    le = out.get("left_eye_corners")
                    re = out.get("right_eye_corners")
                    left_outer  = le[1] if (isinstance(le, np.ndarray) and le.shape == (2,2)) else None
                    right_outer = re[1] if (isinstance(re, np.ndarray) and re.shape == (2,2)) else None
                    if left_outer is not None and right_outer is not None:
                        dx = float(right_outer[0] - left_outer[0])
                        dy = float(right_outer[1] - left_outer[1])
                        if abs(dx) + abs(dy) > 1e-6:
                            roll_2d = _wrap180(np.degrees(np.arctan2(dy, dx)))
                    # Choose solver
                    if cam_backend == "realsense" and depth_m is not None:
                        try:
                            from ..vision.headpose_depth import solve_head_pose_with_depth
                            hp = solve_head_pose_with_depth(out["face_landmarks"], K, depth_m)
                        except Exception:
                            hp = None
                    else:
                        hp = solve_head_pose(out["face_landmarks"], K, dist, rvec0=prev_rvec, tvec0=prev_tvec)
                    if hp is not None and getattr(hp, "ok", False):
                        y0, p0, r0 = smart_angles(hp, left_outer, right_outer)
                        yaw, pitch, roll = _closest_equivalent(y0, p0, r0, prev_angles, roll_2d)
                        prev_rvec, prev_tvec = hp.rvec, hp.tvec
                        prev_angles = (yaw, pitch, roll)
                # Per eye angles
                left_angles = (np.nan, np.nan)
                right_angles = (np.nan, np.nan)
                if out.get("ok", False):
                    lec = out.get("left_eye_corners")
                    rec = out.get("right_eye_corners")
                    lic = out.get("iris_centers", {}).get("left")
                    ric = out.get("iris_centers", {}).get("right")
                    if isinstance(lec, np.ndarray) and lec.shape == (2,2) and lic is not None:
                        left_center = 0.5*(lec[0] + lec[1])
                        left_angles = eye_angles_deg(lic, left_center, fx, fy)
                    if isinstance(rec, np.ndarray) and rec.shape == (2,2) and ric is not None:
                        right_center = 0.5*(rec[0] + rec[1])
                        right_angles = eye_angles_deg(ric, right_center, fx, fy)
                # Build row (not yet written)
                row = None
                t = t_cam if t_cam is not None else time.monotonic()
                if fr_logger is not None and builder is not None:
                    row = {
                        "t_mono": float(t), "frame_id": int(frame_counter),
                        "head_yaw_deg": float(yaw), "head_pitch_deg": float(pitch), "head_roll_deg": float(roll),
                        "head_dist_mm": float(getattr(hp, "distance_mm", np.nan)) if hp is not None else float("nan"),
                        "head_x_mm": float(hp.head_x_mm) if hp is not None and hp.ok else float("nan"),
                        "head_y_mm": float(hp.head_y_mm) if hp is not None and hp.ok else float("nan"),
                        "head_z_mm": float(hp.head_z_mm) if hp is not None and hp.ok else float("nan"),
                        "left_yaw": float(left_angles[0]), "left_pitch": float(left_angles[1]),
                        "right_yaw": float(right_angles[0]), "right_pitch": float(right_angles[1]),
                        "face_present": bool(out.get("ok", False)), "blink": bool(blink),
                        "target_x": None, "target_y": None,
                    }
            # End of camera/processing block

            # After building `row` and computing hp, left_angles, right_angles…
            if writer is not None and frame is not None:
                fr = frame.copy()
                # Pupils
                pupils = out.get("pupils", {})
                for (cx, cy, r), color in (
                    (pupils.get("left",  (np.nan, np.nan, np.nan)), (0, 255, 255)),
                    (pupils.get("right", (np.nan, np.nan, np.nan)), (0, 165, 255))
                ):
                    if np.isfinite([cx, cy, r]).all():
                        cv2.circle(fr, (int(cx), int(cy)), max(2, int(r)), color, 2, lineType=cv2.LINE_AA)
                        cv2.circle(fr, (int(cx), int(cy)), 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                # Eye centres
                lec = out.get("left_eye_corners")
                rec = out.get("right_eye_corners")
                if isinstance(lec, np.ndarray) and lec.shape == (2, 2):
                    cL = (lec[0] + lec[1]) * 0.5
                    cv2.circle(fr, tuple(np.int32(cL)), 3, (0, 255, 0), -1, cv2.LINE_AA)
                if isinstance(rec, np.ndarray) and rec.shape == (2, 2):
                    cR = (rec[0] + rec[1]) * 0.5
                    cv2.circle(fr, tuple(np.int32(cR)), 3, (0, 255, 0), -1, cv2.LINE_AA)
                # Axes overlay if head pose is valid
                if hp is not None and hp.ok:
                    axis = np.float32([[80,0,0],[0,80,0],[0,0,80]]).reshape(-1,3)
                    origin = np.float32([[0,0,0]]).reshape(-1,3)
                    pts, _ = cv2.projectPoints(np.vstack([origin, axis]), hp.rvec, hp.tvec, K, dist)
                    o, x, y, z = pts.reshape(-1,2)
                    o = (int(round(o[0])), int(round(o[1])))
                    x = (int(round(x[0])), int(round(x[1])))
                    y = (int(round(y[0])), int(round(y[1])))
                    z = (int(round(z[0])), int(round(z[1])))
                    cv2.line(fr, o, x, (0, 0, 255), 2)
                    cv2.line(fr, o, y, (0, 255, 0), 2)
                    cv2.line(fr, o, z, (255, 0, 0), 2)
                    # Text overlay
                    cv2.putText(fr, f"Head y/p/r: {yaw:+5.1f} {pitch:+5.1f} {roll:+5.1f} deg",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(fr, f"dist {hp.distance_mm:5.0f}mm",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                writer.write(fr, info={"idx": idx})

            
            # Phase dependent drawing and interaction
            if phase == "wait_start":
                # Draw start button in the centre
                start_centre = (sw // 2, sh // 2)
                bx1, by1, bx2, by2 = _draw_button(canvas, start_centre, "Start EyeTracking")
                # If clicked anywhere check if inside button
                if _click_state.clicked:
                    cx_click, cy_click = _click_state.pos
                    _click_state.reset()
                    if bx1 <= cx_click <= bx2 and by1 <= cy_click <= by2:
                        # Log start event with idx -1 and centre target
                        t_mono = time.monotonic()
                        ev_logger.write({
                            "t_mono": float(t_mono),
                            "idx": -1,
                            "target_x": int(start_centre[0]),
                            "target_y": int(start_centre[1]),
                            "click_x": int(cx_click),
                            "click_y": int(cy_click),
                        })
                        # Begin calibration
                        phase = "calibrating"
                        cross_start_time = time.monotonic()
                        # write this frame row as the first entry
                        if fr_logger is not None and row is not None:
                            row["target_x"] = start_centre[0]
                            row["target_y"] = start_centre[1]
                            fr_logger.write(row)
                            frame_counter += 1
                        continue

            elif phase == "calibrating":
                if idx >= total_pts:
                    # Completed all points, move to end phase
                    phase = "wait_end"
                    continue
                # Draw current and next crosses with countdown
                # Determine countdown number based on elapsed time since cross_start_time
                if cross_start_time is None:
                    cross_start_time = time.monotonic()
                elapsed = time.monotonic() - cross_start_time
                number = int(min(max((elapsed / COUNTDOWN_SECS) * COUNTDOWN_NUMBERS, 0), COUNTDOWN_NUMBERS))
                _draw_cross_with_countdown(canvas, current_pt, number, cross_size=20, circle_radius=28,
                                           cross_color=WHITE, circle_color=RED, text_color=RED, thickness=3)
                # Draw next point as small red cross
                draw_cross(canvas, next_pt, size=16, color=RED, thickness=2)
                # If user clicked, treat as target selection
                if _click_state.clicked:
                    cx_click, cy_click = _click_state.pos
                    _click_state.reset()
                    t_mono = time.monotonic()
                    # Log click event for this point
                    ev_logger.write({
                        "t_mono": float(t_mono),
                        "idx": int(idx),
                        "target_x": int(current_pt[0]),
                        "target_y": int(current_pt[1]),
                        "click_x": int(cx_click),
                        "click_y": int(cy_click),
                    })
                    # Mutate current row's target coordinates if available
                    if fr_logger is not None and row is not None:
                        row["target_x"] = float(current_pt[0])
                        row["target_y"] = float(current_pt[1])
                    # Advance to next point and reset countdown
                    idx += 1
                    cross_start_time = time.monotonic()

            elif phase == "wait_end":
                end_centre = (sw // 2, sh // 2)
                bx1, by1, bx2, by2 = _draw_button(canvas, end_centre, "End Calibration")
                if _click_state.clicked:
                    cx_click, cy_click = _click_state.pos
                    _click_state.reset()
                    if bx1 <= cx_click <= bx2 and by1 <= cy_click <= by2:
                        # Log final event
                        t_mono = time.monotonic()
                        ev_logger.write({
                            "t_mono": float(t_mono),
                            "idx": int(total_pts),
                            "target_x": int(end_centre[0]),
                            "target_y": int(end_centre[1]),
                            "click_x": int(cx_click),
                            "click_y": int(cy_click),
                        })
                        # Write row with end target if available
                        if fr_logger is not None and row is not None:
                            row["target_x"] = float(end_centre[0])
                            row["target_y"] = float(end_centre[1])
                            fr_logger.write(row)
                            frame_counter += 1
                        break  # Exit loop

            # Draw HUD if requested
            if args.hud:
                now = time.monotonic()
                ui_frames += 1
                if now - last_ui_tick >= 1.0:
                    ui_fps = ui_frames / (now - last_ui_tick)
                    ui_frames, last_ui_tick = 0, now
                draw_hud(canvas, fps_ui=ui_fps, fps_cam=cam_fps, backend=backend_name,
                         idx=idx if phase == "calibrating" else None,
                         total=total_pts,
                         face_present=out.get("ok", False), blink=blink if 'blink' in locals() else None)

            # Show UI
            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            # Write frame row if applicable (skip when waiting for start)
            if row is not None and phase != "wait_start":
                fr_logger.write(row)
                frame_counter += 1

        # Clean up UI
        cv2.destroyWindow(win)
    finally:
        if fr_logger is not None:
            fr_logger.close()
        if ev_logger is not None:
            ev_logger.close()
        if sess_log is not None:
            sess_log.close()
        if writer is not None:
            writer.release()
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
