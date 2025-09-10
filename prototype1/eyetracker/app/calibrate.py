from __future__ import annotations
import argparse, time, json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from ..ui.grid import GridSpec, sequence
from ..ui.draw import draw_cross, draw_hud, WHITE, RED
from ..video.capture import VideoCaptureThread
from ..video.writer import VideoWriterMP4
from ..vision.mediapipe_iris import MediaPipeIris
from ..vision.spiga_adapter import SpigaAdapter
from ..vision.camera_model import load_intrinsics
from ..vision.headpose import solve_head_pose, smart_angles

from ..io.logger import frames_logger, events_logger

# --- mouse handling ---
class ClickState:
    def __init__(self):
        self.clicked = False
        self.pos = (0, 0)
    def reset(self): self.clicked = False
_click_state = ClickState()
def _mouse_cb(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_state.clicked = True
        _click_state.pos = (x, y)

def _screen_size_tk() -> Tuple[int, int]:
    import tkinter as tk
    root = tk.Tk(); root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return int(w), int(h)

def _load_config(path: Optional[str]) -> dict:
    if path and Path(path).exists():
        with open(path, "r") as f: return yaml.safe_load(f) or {}
    return {}

def parse_args():
    ap = argparse.ArgumentParser(description="Calibration UI (current+next), full frame/event logging.")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--hud", action="store_true")
    ap.add_argument("--backend", choices=["mediapipe", "spiga", "none"], default="mediapipe")
    ap.add_argument("--record", action="store_true")
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



def main():
    args = parse_args()
    cfg = _load_config(args.config)

    # --- screen & grid ---
    sw, sh = _screen_size_tk()
    grid_cfg = cfg.get("calib_grid", {})
    spec = GridSpec(cols=grid_cfg.get("cols", 5), rows=grid_cfg.get("rows", 5), margin_ratio=grid_cfg.get("margin_ratio", 0.08))
    order_pts = sequence(sw, sh, spec, seed=args.seed)
    total_pts = len(order_pts)

    # --- session folder ---
    ts_run = time.strftime("%Y%m%d_%H%M%S")
    sess_dir = Path(cfg.get("paths", {}).get("sessions_dir", "data/sessions")) / f"calib_{ts_run}"
    sess_dir.mkdir(parents=True, exist_ok=True)
    # snapshot config
    snap_path = sess_dir / "config_snapshot.yaml"
    with open(snap_path, "w") as f: yaml.safe_dump(cfg, f, sort_keys=False)
    # save calib points meta
    meta = {"order": [{"idx": i, "x": int(p[0]), "y": int(p[1])} for i, p in enumerate(order_pts)],
            "screen": {"w": sw, "h": sh, "dpi": None}, "seed": args.seed}
    with open(sess_dir / "calib_points.json", "w") as f: json.dump(meta, f, indent=2)

    # --- camera & backend ---
    cam_cfg = cfg.get("camera", {})
    cam_id = int(cam_cfg.get("id", 0))
    req_w = int(cam_cfg.get("width", 1280))
    req_h = int(cam_cfg.get("height", 720))
    fps   = int(cam_cfg.get("fps", 30))

    intr_path = cam_cfg.get("intrinsics_path", None)

    # start camera
    try:
        cam = VideoCaptureThread(cam_id, req_w, req_h, fps).start()
    except Exception as e:
        print(f"[!] Camera not started: {e}")
        cam = None

    # ---- get first frame to know actual camera size ----
    if cam is None:
        raise RuntimeError("No camera running.")
    frame0 = None
    for _ in range(100):
        _, f0 = cam.get_latest()
        if f0 is not None:
            frame0 = f0; break
        time.sleep(0.02)
    if frame0 is None:
        raise RuntimeError("No frames from camera; cannot continue.")
    cam_h_actual, cam_w_actual = frame0.shape[:2]
    print(f"[i] Camera actual size: {cam_w_actual}x{cam_h_actual}")

    # ---- load intrinsics and scale to actual size ----
    K = None; dist = None; fx = float(cam_w_actual); fy = float(cam_h_actual)
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
        # sensible fallback (assume fx≈w, fy≈h)
        K = np.array([[fx, 0, cam_w_actual*0.5],
                      [0, fy, cam_h_actual*0.5],
                      [0,  0,               1.0]], dtype=np.float64)
        dist = np.zeros((5,), dtype=np.float64)

    # ---- backend ----
    backend_name = args.backend
    backend = None
    if backend_name == "mediapipe":
        try: backend = MediaPipeIris()
        except Exception as e: print(f"[!] MediaPipe backend unavailable: {e}"); backend_name = "none"
    elif backend_name == "spiga":
        try: backend = SpigaAdapter()
        except Exception as e: print(f"[!] SPIGA backend unavailable: {e}"); backend_name = "none"


    #--pipeline--
    # feature builder + loggers+ gate  if backend active
    gate_face = lambda _out: False
    blink_surrogate_fn = lambda _out: (False, 0.0, 0.0)
    blink = None
    prev_rvec = None
    prev_tvec = None

    ev_logger = events_logger(str(sess_dir / "events.parquet"))
    fr_logger = None         # only if backend active
    builder = None           # only if backend active
    if backend_name != "none":
        from ..features.feature_builder import FeatureBuilder, BuilderContext
        from ..quality.gates import face_present as gate_face, blink_surrogate as blink_surrogate_fn
        
        bctx = BuilderContext(screen_w=sw, screen_h=sh, cam_w=cam_w_actual, cam_h=cam_h_actual, fx=fx, fy=fy, K=K, dist=dist)
        builder = FeatureBuilder(bctx)
        fr_logger = frames_logger(str(sess_dir / "frames.parquet"))

    else:
    # harmless no-op stubs
        gate_face = lambda _out: False
        blink_surrogate_fn = lambda _out: (False, 0.0, 0.0)   
        
    #-- --

    # optional recorder
    writer = None
    if args.record and cam is not None:
        writer = VideoWriterMP4(str(sess_dir / "video.mp4"), size=(cam_w_actual, cam_h_actual), fps=fps)
        print(f"[i] Recording to: {writer.actual_path}")

    # UI window
    win = "EyeTracker Calibration"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win, _mouse_cb)

    idx = 0
    frame_counter = 0
    last_ui_tick = time.monotonic(); ui_frames = 0; ui_fps = 0.0
    

    try:
        while idx < total_pts:
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            current_pt = order_pts[idx]
            next_pt = order_pts[idx+1] if idx+1 < total_pts else order_pts[idx]
            draw_cross(canvas, current_pt, size=22, color=WHITE, thickness=3)
            draw_cross(canvas, next_pt, size=16, color=RED, thickness=2)

            t_cam = None; frame = None; out = {"ok": False, "score": 0.0}
            cam_fps = None
            if cam is not None:
                cam_fps = cam.get_fps()
                t_cam, frame = cam.get_latest()
                if frame is not None and backend is not None:
                    out = backend.process(frame)

                # Quality gates
                score = float(out.get("score", 0.0))
                blink, _, _ = blink_surrogate_fn(out)  if backend_name != "none" else (False,0.0,0.0)
                if score < 0.10: blink = True
                        
                # ---- Head pose PnP ----
                hp = None
                yaw = pitch = roll = np.nan
                if K is not None and out.get("ok", False):
                    hp = solve_head_pose(out["face_landmarks"], K, dist, rvec0=prev_rvec, tvec0=prev_tvec)
                    if hp.ok:
                        # roll from eye line if available
                        le = out.get("left_eye_corners")   # (2,2) [inner, outer]
                        re = out.get("right_eye_corners")  # (2,2) [inner, outer]
                        left_outer  = le[1] if (le is not None and len(le)==2) else None
                        right_outer = re[1] if (re is not None and len(re)==2) else None
                        yaw, pitch, roll = smart_angles(hp, left_outer, right_outer)
                        prev_rvec, prev_tvec = hp.rvec, hp.tvec

                # --- per-eye angles (degrees) from iris centers vs eye midpoints ---
                left_angles = (np.nan, np.nan)
                right_angles = (np.nan, np.nan)
                if out.get("ok", False):
                    # eye centers as midpoint between corners
                    lec = out.get("left_eye_corners")
                    rec = out.get("right_eye_corners")
                    lic = out.get("iris_centers", {}).get("left", None)
                    ric = out.get("iris_centers", {}).get("right", None)
                    if isinstance(lec, np.ndarray) and lec.shape == (2,2) and lic is not None:
                        left_center = 0.5*(lec[0] + lec[1])
                        left_angles = eye_angles_deg(lic, left_center, fx, fy)
                    if isinstance(rec, np.ndarray) and rec.shape == (2,2) and ric is not None:
                        right_center = 0.5*(rec[0] + rec[1])
                        right_angles = eye_angles_deg(ric, right_center, fx, fy)
                # ---- Build row but do not write yet (so clicks can annotate it) ----
                row = None
                if fr_logger is not None and builder is not None:
                    t = t_cam if t_cam is not None else time.monotonic()
                row = {
                    "t_mono": float(t), "frame_id": int(frame_counter), 
                    "screen_w": int(sw), "screen_h": int(sh),
                    "cam_w": int(cam_w_actual), "cam_h": int(cam_h_actual),
                    "head_yaw_deg": float(yaw), "head_pitch_deg": float(pitch), "head_roll_deg": float(roll),
                    "head_dist_mm": float(getattr(hp, "distance_mm", np.nan)) if hp is not None else float("nan"),
                    "left_yaw": float(left_angles[0]), "left_pitch": float(left_angles[1]),
                    "right_yaw": float(right_angles[0]), "right_pitch": float(right_angles[1]),
                    "face_present": bool(out.get("ok", False)), "blink": bool(blink),
                    "landmark_score": float(out.get("score", 0.0)),
                    "target_x": None, "target_y": None,
                }

                # ---- Draw overlays & record (axes + pupils) ----
                if writer is not None and frame is not None:
                    fr = frame.copy()
                    # pupils
                    pupils = out.get("pupils", {})
                    for (cx, cy, r), color in (
                        (pupils.get("left",  (np.nan,np.nan,np.nan)),  (0,255,255)),
                        (pupils.get("right", (np.nan,np.nan,np.nan)),  (0,165,255))
                    ):
                        if np.isfinite([cx,cy,r]).all():
                            cv2.circle(fr, (int(cx), int(cy)), max(2, int(r)), color, 2, lineType=cv2.LINE_AA)
                            cv2.circle(fr, (int(cx), int(cy)), 2, (0,0,0), -1, lineType=cv2.LINE_AA)
                    # draw per-eye centers
                    lec = out.get("left_eye_corners")
                    rec = out.get("right_eye_corners")
                    if isinstance(lec, np.ndarray) and lec.shape == (2,2):
                        cL = (lec[0] + lec[1]) * 0.5
                        cv2.circle(fr, tuple(np.int32(cL)), 3, (0,255,0), -1, cv2.LINE_AA)
                    if isinstance(rec, np.ndarray) and rec.shape == (2,2):
                        cR = (rec[0] + rec[1]) * 0.5
                        cv2.circle(fr, tuple(np.int32(cR)), 3, (0,255,0), -1, cv2.LINE_AA)

                    # axes
                    if hp is not None and hp.ok:
                        axis = np.float32([[80,0,0],[0,80,0],[0,0,80]]).reshape(-1,3)
                        origin = np.float32([[0,0,0]]).reshape(-1,3)
                        pts, _ = cv2.projectPoints(np.vstack([origin, axis]), hp.rvec, hp.tvec, K, dist)
                        pts2 = np.squeeze(pts, axis=1)             # (4,2)
                        p_o, p_x, p_y, p_z = [tuple(int(round(v)) for v in p) for p in pts2]

                        cv2.line(fr, p_o, p_x, (0,0,255), 2)
                        cv2.line(fr, p_o, p_y, (0,255,0), 2)
                        cv2.line(fr, p_o, p_z, (255,0,0), 2)
                        cv2.putText(fr, f"Head y/p/r: {yaw:5.1f} {pitch:5.1f} {roll:5.1f} deg",
                                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
                        cv2.putText(fr, f"dist {hp.distance_mm:5.0f}mm",
                                    (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
                        cv2.putText(fr, f"L eye (yaw,pitch): {left_angles[0]:+.1f}, {left_angles[1]:+.1f} deg",
                            (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2, cv2.LINE_AA)
                        cv2.putText(fr, f"R eye (yaw,pitch): {right_angles[0]:+.1f}, {right_angles[1]:+.1f} deg",
                            (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2, cv2.LINE_AA)
                    writer.write(fr, info={"idx": idx})

            # ---- HUD ----
            if args.hud:
                now = time.monotonic()
                ui_frames += 1
                if now - last_ui_tick >= 1.0:
                    ui_fps = ui_frames / (now - last_ui_tick)
                    ui_frames, last_ui_tick = 0, now
                draw_hud(canvas, fps_ui=ui_fps, fps_cam=cam_fps, backend=backend_name,
                         idx=idx, total=total_pts,
                         face_present=out.get("ok", False), blink=blink if 'blink' in locals() else None)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break

            # ---- click-to-advance: mutate CURRENT frame row with target ----
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
                if fr_logger is not None and row is not None:
                    row["target_x"] = int(current_pt[0])
                    row["target_y"] = int(current_pt[1])
                idx += 1

            # ---- finally write one row for this frame ----
            if row is not None:
                fr_logger.write(row)
                frame_counter += 1

        cv2.destroyWindow(win)
    finally:
        if fr_logger is not None:fr_logger.close()
        if ev_logger is not None:ev_logger.close()
        if writer is not None: writer.release()
        if cam is not None: cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

