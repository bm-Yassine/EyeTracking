from __future__ import annotations
import argparse, time, json, math, threading
from pathlib import Path
from typing import Optional, Tuple ,Dict, Any

import cv2
import numpy as np
import yaml

# --- UI and drawing ---
from ..ui.grid import GridSpec, sequence
from ..ui.draw import draw_cross, draw_hud, WHITE, RED
# --- Video I/O ---
from ..video.capture import VideoCaptureThread  # OpenCV webcam
try:
    from ..video.realsense_capture import RealSenseCaptureThread  # provided below
except Exception:
    RealSenseCaptureThread = None

from ..video.writer import VideoWriterMP4
from ..video.camera_model import load_intrinsics
# --- Backends ---
from ..vision.mediapipe_iris import MediaPipeIris            # your existing wrapper (2D landmarks)
try:
    from ..vision.backends.webcam2d import Webcam2DIris      # provided below
except Exception:
    Webcam2DIris = None
try:
    from ..vision.backends.rs3d import RealSense3DIris       # provided below
except Exception:
    RealSense3DIris = None
    
# --- Head pose ---
from ..vision.headpose import solve_head_pose, smart_angles   # your existing functions

# --- Logging ---
from ..io.logger import frames_logger, events_logger


# -----------------------
# Mouse handling (click-to-advance)
# -----------------------
class ClickState:
    def __init__(self): self.clicked, self.pos = False, (0, 0)
    def reset(self): self.clicked = False
_click_state = ClickState()
def _mouse_cb(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_state.clicked = True
        _click_state.pos = (x, y)

# -----------------------
# Screen resolution
# -----------------------
def _screen_size_tk() -> Tuple[int, int]:
    import tkinter as tk
    root = tk.Tk(); root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return int(w), int(h)

# -----------------------
# Config helpers
# -----------------------
def _load_config(path: Optional[str]) -> dict:
    if path and Path(path).exists():
        with open(path, "r") as f: return yaml.safe_load(f) or {}
    return {}

def _load_screen_plane(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Optional: load a precomputed screen plane in camera coords:
      { "A": [x,y,z], "B":[...], "C":[...], "res_px":[W,H] }
    Not strictly needed for Part A, but we store metadata if present.
    """
    if path and Path(path).exists():
        with open(path, "r") as f: return yaml.safe_load(f)
    return None



# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Calibration UI (current+next), full frame/event logging.")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--hud", action="store_true")
    ap.add_argument("--record", action="store_true")

    # camera + backend choices
    ap.add_argument("--camera", choices=["opencv", "realsense"], default=None,
                    help="Override config camera.type. If omitted, uses config.")
    ap.add_argument("--backend", choices=["mediapipe", "webcam2d", "rs3d", "spiga", "none"], default=None,
                    help="Override config.backend. If omitted, uses config.")
    return ap.parse_args()

def eye_angles_deg(iris_xy, eye_ctr_xy, fx, fy):
    """
    Convert pixel offsets to per-eye yaw/pitch in degrees.
    Uses small-angle model with atan2 for numerical stability.
    """
    dx = (float(iris_xy[0]) - float(eye_ctr_xy[0])) / float(fx)
    dy = (float(iris_xy[1]) - float(eye_ctr_xy[1])) / float(fy)
    yaw_rad   = np.arctan2(dx, 1.0)   # positive = looking to subject's right
    pitch_rad = np.arctan2(dy, 1.0)   # positive = looking down (opencv y grows downward)
    return float(np.degrees(yaw_rad)), float(np.degrees(pitch_rad))


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    cfg = _load_config(args.config)

    # --- screen & calibration grid ---
    sw, sh = _screen_size_tk()
    grid_cfg = cfg.get("calib_grid", {})
    spec = GridSpec(cols=grid_cfg.get("cols", 5),
                    rows=grid_cfg.get("rows", 5),
                    margin_ratio=grid_cfg.get("margin_ratio", 0.08))
    order_pts = sequence(sw, sh, spec, seed=args.seed)
    total_pts = len(order_pts)

    # --- session folder & metadata ---
    ts_run = time.strftime("%Y%m%d_%H%M%S")
    sess_dir = Path(cfg.get("paths", {}).get("sessions_dir", "data/sessions")) / f"calib_{ts_run}"
    sess_dir.mkdir(parents=True, exist_ok=True)
    with open(sess_dir / "config_snapshot.yaml", "w") as f: yaml.safe_dump(cfg, f, sort_keys=False)
    session_meta = {
        "screen": {"w": int(sw), "h": int(sh)},
        "camera": {
            "type": str(cam_type),
            "w": int(cam_w_actual),
            "h": int(cam_h_actual),
            "fps": int(fps),
            "rs_intrinsics": {
                "color_K": rs_intr["color_K"].tolist(),
                "color_dist": rs_intr["color_dist"].tolist(),
            } if (cam_type == "realsense" and rs_intr is not None)      
            else {"K": K.tolist()} if intr_path and Path(intr_path).exists() else None,
            "dist": dist.tolist() if dist is not None else None,
            "intrinsics_source": "realsense" if cam_type=="realsense" else ("file" if intr_path else "fallback"),
        },
        "backend": str(backend_cfg_name),
        "grid": {"cols": spec.cols, "rows": spec.rows, "margin_ratio": spec.margin_ratio},
        "seed": int(args.seed),
        "screen_plane": None  # will be replaced by a dict with A,B,C,res_px
    }
    with open(sess_dir / "session_meta.yaml", "w") as f:yaml.safe_dump(session_meta, f, sort_keys=False)

    screen_plane_path = cfg.get("screen", {}).get("plane_path", None)
    screen_plane = _load_screen_plane(screen_plane_path)
    meta = {"order": [{"idx": i, "x": int(p[0]), "y": int(p[1])} for i, p in enumerate(order_pts)],
            "screen": {"w": sw, "h": sh, "plane": screen_plane}, "seed": args.seed}
    with open(sess_dir / "calib_points.json", "w") as f: json.dump(meta, f, indent=2)

    # --- camera selection ---
    cam_cfg = cfg.get("camera", {})
    cam_type = args.camera or cam_cfg.get("type", "opencv")   # "opencv" | "realsense"
    req_w = int(cam_cfg.get("width", 1280))
    req_h = int(cam_cfg.get("height", 720))
    fps   = int(cam_cfg.get("fps", 30))
    cam_id = int(cam_cfg.get("id", 0))

    intr_path = cam_cfg.get("intrinsics_path", None)

    # start camera
    cam = None
    rs_intr = None
    try:
        if cam_type == "realsense":
            if RealSenseCaptureThread is None:
                raise RuntimeError("RealSenseCaptureThread not found. Did you add video/realsense_capture.py?")
            cam = RealSenseCaptureThread(width=req_w, height=req_h, fps=fps).start()
        else:
            cam = VideoCaptureThread(cam_id, req_w, req_h, fps).start()
    except Exception as e:
        raise RuntimeError(f"Camera not started: {e}")

    # first frame(s) → determine sizes; also collect realsense intrinsics
    frame0 = None; depth0 = None
    for _ in range(100):
        out_cam = cam.get_latest()
        if cam_type == "realsense":
            t0, f0, d0, intr_bundle = out_cam  # (t, color, depth, intrinsics dict)
            rs_intr = intr_bundle
            if f0 is not None:
                frame0, depth0 = f0, d0
                break
        else:
            t0, f0 = out_cam
            if f0 is not None:
                frame0 = f0
                break
        time.sleep(0.02)
    if frame0 is None:
        raise RuntimeError("No frames from camera; cannot continue.")
    cam_h_actual, cam_w_actual = frame0.shape[:2]
    print(f"[i] Camera actual size: {cam_w_actual}x{cam_h_actual}")

    # ---- load intrinsics K,dist; handle scaling; or take from realsense ----
    K = None; dist = None; fx = float(cam_w_actual); fy = float(cam_h_actual)
    if cam_type == "realsense" and rs_intr is not None and rs_intr.get("color_K") is not None:
        K = rs_intr["color_K"].astype(np.float64).copy()
        dist = rs_intr["color_dist"].astype(np.float64).copy()
        fx, fy = float(K[0,0]), float(K[1,1])
        print("[i] Using RealSense color intrinsics from SDK.")
    elif intr_path and Path(intr_path).exists():
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
        # fallback pinhole
        K = np.array([[fx, 0, cam_w_actual*0.5],
                      [0, fy, cam_h_actual*0.5],
                      [0,  0,               1.0]], dtype=np.float64)
        dist = np.zeros((5,), dtype=np.float64)

    # ---- backend selection ----
    # Unify outputs across backends:
    # out = {
    #   "ok": bool, "score": float,
    #   "face_landmarks": np.ndarray[(N,2)] (2D),           # for head pose PnP
    #   "left_eye_corners": np.ndarray[(2,2)],              # [inner, outer]
    #   "right_eye_corners": np.ndarray[(2,2)],
    #   "iris_centers": {"left": (x,y), "right": (x,y)},
    #   "pupils": {"left": (cx,cy,r), "right": (cx,cy,r)},  # optional
    #   "depth_mm_eye": float | None,                       # optional RS3D distance
    # }
    backend_cfg_name = args.backend or cfg.get("backend", "mediapipe")

    backend = None
    if backend_cfg_name == "mediapipe":
        backend = MediaPipeIris()  # your existing (2D) implementation
    elif backend_cfg_name == "webcam2d":
        if Webcam2DIris is None:
            raise RuntimeError("Webcam2DIris not found (add vision/backends/webcam2d.py).")
        backend = Webcam2DIris()   # minimal 2D using MediaPipe under the hood
    elif backend_cfg_name == "rs3d":
        if RealSense3DIris is None:
            raise RuntimeError("RealSense3DIris not found (add vision/backends/rs3d.py).")
        if cam_type != "realsense":
            print("[!] rs3d backend expects --camera realsense; continuing but depth will be None.")
        backend = RealSense3DIris()
    elif backend_cfg_name == "none":
        backend = None

    # ---- loggers ----
    ev_logger = events_logger(str(sess_dir / "events.parquet"))
    fr_logger = frames_logger(str(sess_dir / "frames.parquet"))

    # ---- optional recorder ----
    writer = None
    if args.record:
        writer = VideoWriterMP4(str(sess_dir / "video.mp4"), size=(cam_w_actual, cam_h_actual), fps=fps)
        print(f"[i] Recording to: {writer.actual_path}")

    # ---- UI window ----
    win = "EyeTracker Calibration"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win, _mouse_cb)

    idx = 0
    frame_counter = 0
    last_ui_tick = time.monotonic(); ui_frames = 0; ui_fps = 0.0

    prev_rvec = None
    prev_tvec = None

    try:
        while idx < total_pts:
            # Draw current+next cross on a full-screen canvas
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            current_pt = order_pts[idx]
            next_pt = order_pts[idx+1] if idx+1 < total_pts else order_pts[idx]
            draw_cross(canvas, current_pt, size=22, color=WHITE, thickness=3)
            draw_cross(canvas, next_pt, size=16, color=RED, thickness=2)

            # --- get camera frame(s) ---
            t_cam = None; frame = None; depth = None; cam_fps = None
            out_cam = cam.get_latest()
            if cam_type == "realsense":
                t_cam, frame, depth, _ = out_cam
            else:
                t_cam, frame = out_cam
            if frame is None:
                # show UI anyway, keep loop responsive
                cv2.imshow(win, canvas)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
                continue

            # --- run backend ---
            out = {"ok": False, "score": 0.0}
            if backend is not None:
                try:
                    out = backend.process(frame, depth=depth, K=K, dist=dist)
                except Exception as e:
                    print(f"[!] backend.process failed: {e}")
                    out = {"ok": False, "score": 0.0}

            # --- quality (very light) ---
            face_ok = bool(out.get("ok", False))
            blink = False
            score = float(out.get("score", 0.0))
            if score < 0.10:
                blink = True  # crude surrogate (you have a better blink gate in your codebase)

            # --- Head pose PnP, stabilized ---
            hp = None
            yaw = pitch = roll = np.nan
            if face_ok and K is not None:
                try:
                    hp = solve_head_pose(out["face_landmarks"], K, dist, rvec0=prev_rvec, tvec0=prev_tvec)
                    if hp.ok:
                        le = out.get("left_eye_corners")
                        re = out.get("right_eye_corners")
                        left_outer  = le[1] if (isinstance(le, np.ndarray) and le.shape == (2,2)) else None
                        right_outer = re[1] if (isinstance(re, np.ndarray) and re.shape == (2,2)) else None
                        yaw, pitch, roll = smart_angles(hp, left_outer, right_outer)
                        prev_rvec, prev_tvec = hp.rvec, hp.tvec
                except Exception as e:
                    print(f"[!] head pose failed: {e}")

            # --- per-eye angles from 2D geometry (kept for Part A logs) ---
            left_angles = (np.nan, np.nan); right_angles = (np.nan, np.nan)
            if face_ok:
                lec = out.get("left_eye_corners")
                rec = out.get("right_eye_corners")
                lic = out.get("iris_centers", {}).get("left")
                ric = out.get("iris_centers", {}).get("right")
                if isinstance(lec, np.ndarray) and lec.shape == (2,2) and isinstance(lic, (tuple, list, np.ndarray)):
                    left_center = 0.5*(lec[0] + lec[1])
                    left_angles = eye_angles_deg(lic, left_center, fx, fy)
                if isinstance(rec, np.ndarray) and rec.shape == (2,2) and isinstance(ric, (tuple, list, np.ndarray)):
                    right_center = 0.5*(rec[0] + rec[1])
                    right_angles = eye_angles_deg(ric, right_center, fx, fy)

            # --- build log row ---
            t = float(t_cam if t_cam is not None else time.monotonic())
            row = {
                "t_mono": t, "frame_id": int(frame_counter),
                "head_yaw_deg": float(yaw), "head_pitch_deg": float(pitch), "head_roll_deg": float(roll),
                "head_dist_mm": float(getattr(hp, "distance_mm", np.nan)) if hp is not None else float("nan"),
                "depth_mm_eye": float(out.get("depth_mm_eye", np.nan)) if out.get("depth_mm_eye", None) is not None else float("nan"),
                "left_yaw": float(left_angles[0]), "left_pitch": float(left_angles[1]),
                "right_yaw": float(right_angles[0]), "right_pitch": float(right_angles[1]),
                "face_present": bool(face_ok), "blink": bool(blink),
                # "landmark_score": float(score),  --- IGNORE ---   
                "target_x": None, "target_y": None,
            }

            # --- drawing + optional recording (pupils, eye centers, head axes) ---
            if writer is not None:
                fr = frame.copy()

                # draw pupils if provided
                pupils = out.get("pupils", {})
                for (cx, cy, r), color in (
                    (pupils.get("left",  (np.nan,np.nan,np.nan)),  (0,255,255)),
                    (pupils.get("right", (np.nan,np.nan,np.nan)),  (0,165,255))
                ):
                    if np.all(np.isfinite([cx,cy,r])):
                        cv2.circle(fr, (int(round(cx)), int(round(cy))), max(2, int(round(r))), color, 2, lineType=cv2.LINE_AA)
                        cv2.circle(fr, (int(round(cx)), int(round(cy))), 2, (0,0,0), -1, lineType=cv2.LINE_AA)

                # draw per-eye midpoints
                for ec in (out.get("left_eye_corners"), out.get("right_eye_corners")):
                    if isinstance(ec, np.ndarray) and ec.shape == (2,2):
                        c = 0.5*(ec[0] + ec[1])
                        cv2.circle(fr, (int(round(c[0])), int(round(c[1]))), 3, (0,255,0), -1, cv2.LINE_AA)

                # draw head axes if available
                if hp is not None and hp.ok:
                    axis = np.float32([[80,0,0],[0,80,0],[0,0,80]]).reshape(-1,3)
                    origin = np.float32([[0,0,0]]).reshape(-1,3)
                    try:
                        pts, _ = cv2.projectPoints(np.vstack([origin, axis]), hp.rvec, hp.tvec, K, dist)
                        o, x, y, z = pts.reshape(-1,2).astype(np.float32)
                        o = (int(round(o[0])), int(round(o[1])))
                        x = (int(round(x[0])), int(round(x[1])))
                        y = (int(round(y[0])), int(round(y[1])))
                        z = (int(round(z[0])), int(round(z[1])))
                        cv2.line(fr, o, x, (0,0,255), 2)
                        cv2.line(fr, o, y, (0,255,0), 2)
                        cv2.line(fr, o, z, (255,0,0), 2)
                    except Exception as e:
                        # Keep running even if projection fails for a frame
                        pass

                    cv2.putText(fr, f"Head y/p/r: {yaw:5.1f}/{pitch:5.1f}/{roll:5.1f} deg",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
                    if hasattr(hp, "distance_mm") and hp.distance_mm is not None and np.isfinite(hp.distance_mm):
                        cv2.putText(fr, f"dist: {hp.distance_mm:5.0f} mm", (10,55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

                # eye angles
                cv2.putText(fr, f"L (yaw,pitch): {left_angles[0]:+.1f}, {left_angles[1]:+.1f} deg",
                            (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2, cv2.LINE_AA)
                cv2.putText(fr, f"R (yaw,pitch): {right_angles[0]:+.1f}, {right_angles[1]:+.1f} deg",
                            (10,105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2, cv2.LINE_AA)

                writer.write(fr, info={"idx": idx})

            # --- HUD overlay on canvas (not the video) ---
            if args.hud:
                now = time.monotonic()
                ui_frames += 1
                if now - last_ui_tick >= 1.0:
                    ui_fps = ui_frames / (now - last_ui_tick)
                    ui_frames, last_ui_tick = 0, now
                draw_hud(canvas, fps_ui=ui_fps, fps_cam=getattr(cam, "fps", None),
                         backend=backend_cfg_name, idx=idx, total=total_pts,
                         face_present=face_ok, blink=blink)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break

            # ---- click-to-advance: annotate CURRENT target into the last built row ----
            if _click_state.clicked:
                _click_state.reset()
                t_mono = time.monotonic()
                ev_logger.write({
                    "t_mono": float(t_mono),
                    "idx": int(idx),
                    "target_x": int(current_pt[0]),
                    "target_y": int(current_pt[1]),
                    "click_x": int(_click_state.pos[0]),
                    "click_y": int(_click_state.pos[1]),
                })
                row["target_x"] = int(current_pt[0])
                row["target_y"] = int(current_pt[1])
                idx += 1

            # ---- write one row per frame ----
            fr_logger.write(row)
            frame_counter += 1

        cv2.destroyWindow(win)
    finally:
        try: fr_logger.close()
        except: pass
        try: ev_logger.close()
        except: pass
        try:
            if writer is not None: writer.release()
        except: pass
        try:
            if cam is not None: cam.stop()
        except: pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()