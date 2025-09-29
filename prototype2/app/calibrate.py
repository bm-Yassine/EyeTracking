import cv2 as cv, numpy as np, time, yaml, os
from pathlib import Path
from eyetracker.ui.draw import draw_cross, draw_countdown_cross
from eyetracker.core.mediapipe_features import MPFaceIris
from eyetracker.core.headpose import HeadPose, build_camera_matrix
from eyetracker.io.logging import SessionLogger

def load_cfg(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def make_grid(w, h, rows, cols):
    xs = np.linspace(int(0.1*w), int(0.9*w), cols, dtype=int)
    ys = np.linspace(int(0.12*h), int(0.88*h), rows, dtype=int)
    pts = [(x,y) for y in ys for x in xs]
    rng = np.random.default_rng(42)
    order = list(range(len(pts)))
    rng.shuffle(order)
    return pts, order

def draw_cross(img, center, color, r=14, t=2):
    x,y = center
    cv.line(img, (x-r,y), (x+r,y), color, t)
    cv.line(img, (x,y-r), (x,y+r), color, t)

def try_load_intrinsics(yaml_path):
    if not yaml_path or not Path(yaml_path).exists(): return None, None
    data = yaml.safe_load(Path(yaml_path).read_text())
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data.get("dist", [0,0,0,0,0]), dtype=np.float64).reshape(-1,1)
    return K, dist

def main():
    cfg = load_cfg(os.environ.get("ET_CFG", "./config/default.yaml"))
    W, H = cfg["session"]["width"], cfg["session"]["height"]
    cap = cv.VideoCapture(cfg["session"]["camera_index"])
    cap.set(cv.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv.CAP_PROP_FPS, cfg["session"]["fps"])

    mpf = MPFaceIris()
    hp = HeadPose()
    logger = SessionLogger(cfg)

    # camera intrinsics
    K, dist = try_load_intrinsics(cfg["camera_model"]["intrinsics_yaml"])
    if K is None:
        K = build_camera_matrix(W, H, None)
        dist = None

    # calibration target grid on the SCREEN (not camera frame)
    SW, SH = cfg["screen"]["width_px"], cfg["screen"]["height_px"]
    pts, order = make_grid(SW, SH, cfg["screen"]["grid_rows"], cfg["screen"]["grid_cols"])
    idx = 0

    # video recorder (on camera frames with overlays)
    writer = None
    if cfg["session"]["record_video"]:
        fourcc = cv.VideoWriter_fourcc(*cfg["session"]["video_codec"])
        writer = cv.VideoWriter(logger.video_path(), fourcc, cfg["session"]["fps"], (W,H))

    # mouse callback for clicks
    click_flag = {"clicked": False}
    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            click_flag["clicked"] = True
    cv.namedWindow("Calibration", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Calibration", on_mouse)

    frame_id = 0
    prev_time = time.perf_counter()
    next_idx = order[idx+1] if idx+1 < len(order) else order[idx]
    
    global_start = time.perf_counter()
    per_target_start = global_start

    total_limit_s = float(cfg["calibration_ui"]["countdown_total_s"])
    per_target_s = float(cfg["calibration_ui"]["per_target_s"])
    auto_advance = bool(cfg["calibration_ui"].get("auto_advance", False))

    while cap.isOpened() and idx < len(order):
        ok, frame = cap.read()
        if not ok: break
        t_mono = time.perf_counter()
        dt = (t_mono - prev_time)
        prev_time = t_mono

        feat = mpf.process(frame)
        face_present = feat.get("face_present", False)

        # head pose if possible
        head = {}
        if face_present and feat.get("pnp_px") is not None:
            res = hp.solve(feat["pnp_px"], K, dist)
            if res is not None:
                head = res

        # EAR → blink
        ear_l, ear_r = feat.get("ear_l", np.nan), feat.get("ear_r", np.nan)
        ear_thresh = cfg["filters"]["blink_ear_thresh"]
        blink = float(ear_l < ear_thresh and ear_r < ear_thresh)

        # overlay metrics
        ov = frame.copy()
        if face_present:
            pl = tuple(np.round(feat["pupil_l"]).astype(int))
            pr = tuple(np.round(feat["pupil_r"]).astype(int))
            cv.circle(ov, pl, 3, (0,255,0), -1)
            cv.circle(ov, pr, 3, (0,255,0), -1)
        if "yaw" in head:
            cv.putText(ov, f"YPR: {head['yaw']:.1f} {head['pitch']:.1f} {head['roll']:.1f}",
                       (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # draw current+next targets at top-left HUD (preview), plus instruction
        vis = ov
        # draw cross markers on a small HUD rectangle (top-right) is optional; main targets
        # are drawn in a separate "targets" panel below using screen coords.

        elapsed_total = t_mono - global_start
        secs_left_total = max(0.0, total_limit_s - elapsed_total)

        elapsed_target = t_mono - per_target_start
        if auto_advance and elapsed_target >= per_target_s:
            # simulate a click to log and advance
            click_flag["clicked"] = True

        # Compose a targets canvas (1080p) to show current+next positions
        target_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        curr = pts[order[idx]]
        nxt  = pts[next_idx]
        draw_cross(target_canvas, (int(curr[0]*1280/SW), int(curr[1]*720/SH)),
                   tuple(cfg["calibration_ui"]["curr_cross_color_bgr"]),
                   cfg["calibration_ui"]["cross_radius"],
                   cfg["calibration_ui"]["cross_thickness"])
        if cfg["calibration_ui"]["show_next"]:
            draw_cross(target_canvas, (int(nxt[0]*1280/SW), int(nxt[1]*720/SH)),
                       tuple(cfg["calibration_ui"]["next_cross_color_bgr"]),
                       cfg["calibration_ui"]["cross_radius"],
                       cfg["calibration_ui"]["cross_thickness"])
        cv.putText(target_canvas, "Look at WHITE cross and CLICK", (30,40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

        # show
        cv.imshow("Calibration", vis)
        cv.imshow("Targets", target_canvas)
        if writer is not None: writer.write(ov)

        # handle click → log one sample tied to current target
        if click_flag["clicked"]:
            target_x, target_y = curr
            row = {
                "t_mono": float(t_mono),
                "frame_id": int(frame_id),
                "face_present": int(face_present),
                "blink": int(blink),
                "target_x": int(target_x),
                "target_y": int(target_y),
                "pupil_l_x": float(feat.get("pupil_l", [np.nan, np.nan])[0]),
                "pupil_l_y": float(feat.get("pupil_l", [np.nan, np.nan])[1]),
                "pupil_r_x": float(feat.get("pupil_r", [np.nan, np.nan])[0]),
                "pupil_r_y": float(feat.get("pupil_r", [np.nan, np.nan])[1]),
                "eye_pitch_l": float(feat.get("eye_pitch_l", np.nan)),
                "eye_yaw_l": float(feat.get("eye_yaw_l", np.nan)),
                "eye_pitch_r": float(feat.get("eye_pitch_r", np.nan)),
                "eye_yaw_r": float(feat.get("eye_yaw_r", np.nan)),
                "head_yaw": float(head.get("yaw", np.nan)),
                "head_pitch": float(head.get("pitch", np.nan)),
                "head_roll": float(head.get("roll", np.nan)),
                "head_X": float(head.get("t", [np.nan, np.nan, np.nan])[0] if "t" in head else np.nan),
                "head_Y": float(head.get("t", [np.nan, np.nan, np.nan])[1] if "t" in head else np.nan),
                "head_Z": float(head.get("t", [np.nan, np.nan, np.nan])[2] if "t" in head else np.nan),
                "head_dist_z": float(head.get("t", [np.nan, np.nan, np.nan])[2] if "t" in head else np.nan),
            }
            logger.add(row)
            click_flag["clicked"] = False
            # advance to next
            idx += 1
            next_idx = order[idx+1] if idx+1 < len(order) else order[idx]

        k = cv.waitKey(1) & 0xFF
        if k == ord('q'): break
        frame_id += 1

    logger.flush()
    if writer is not None: writer.release()
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
