Install:
 pip install -e .
Calibration:
    # ChArUco (default)
    eye-cam-calibrate --camera-id 0 --width 1280 --height 720
    # Checkerboard
    eye-cam-calibrate --pattern checkerboard --cb-cols 9 --cb-rows 6 --cb-square-mm 25
Run:
    python -m eyetracker.app.calibrate --backend none --hud 
    python -m eyetracker.app.calibrate --backend mediapipe --hud --record
  CLI flags (calibration UI)
    --config <path>
        Path to YAML config. Default: configs/default.yaml.
    --seed <int>
        Seed for the grid order. Default: 17.
    --hud
        Show on-screen HUD (FPS, backend, point idx, flags).
    --backend {mediapipe,spiga,none}
        Landmark backend. Default: mediapipe.
    --record
        Save the camera video to the session folder (with axes + pupil overlays).

    

A0: deps + a camera-intrinsics module with a CLI to calibrate your webcam (checkerboard or ChArUco), save K & dist to data/camera/, and utilities to load/undistort later.
A1: grid order + current/next rendering + click logging + HUD.
A2: threaded capture with get_latest() and MP4 writer with overlay hook.
A3: MediaPipe FaceMesh+Iris backend (standardized outputs); SPIGA adapter stub for later SOTA integration.
A4 – vision/headpose.py: robust PnP from a fixed 6-point model → yaw/pitch/roll (deg) + distance_mm.
A5 – features/geometry.py: iris circle fit (already used), eye reference (corner midpoint), and the exact pinhole mapping(with image-y inverted so “up” is +pitch).
A6 – features/feature_builder.py + io/schemas.py + io/logger.py: per-frame struct with timestamps, head pose, left/right pupil angles, iris radii, normalized eye-corner geometry, quality flags, and target at click frames; streamed to Parquet (frames.parquet, events.parquet).
A7 – quality/gates.py: lightweight real-time gates—face present and a blink surrogate based on iris-radius-to-eye-width ratio.
A8 – app/calibrate.py writes a full session folder:
data/sessions/calib_YYYYmmdd_HHMMSS/
  ├─ calib_points.json      # order, screen, seed
  ├─ frames.parquet         # every frame
  ├─ events.parquet         # one row per click
  ├─ video.mp4              # if --record
  └─ config_snapshot.yaml



1) Smoke tests (UI, clicks, session artifacts)
Run UI only (no webcam) to validate order + logging:
python -m eyetracker.app.calibrate --backend none --hud
What you should see:
Full-screen black canvas with current (white) and next (red) crosses.
Click the white cross to advance through all 25 points.
At the end (or if you press q), a session folder appears:
data/sessions/calib_<timestamp>/
  calib_points.json
  frames.parquet        # will exist but be mostly NaNs since backend=none
  events.parquet        # 25 rows if you clicked all points
  config_snapshot.yaml
Quick checks:
events.parquet has 25 rows (5×5).
calib_points.json order matches the on-screen sequence (seed saved).
frames.parquet row count roughly equals run-time × UI FPS (backend=none ⇒ fewer fields valid; that’s OK).
2) Webcam + backend sanity
Run with MediaPipe + recording:
python -m eyetracker.app.calibrate --backend mediapipe --hud --record
What to watch in the HUD (top-left):
UI FPS: typically near your display refresh; anything ≥30 is fine.
Cam FPS: close to your configured value (e.g., ~30).
Backend: mediapipe.
Face: flips to YES when your face is in frame; NO otherwise.
Blink: becomes YES when you blink.
Artifacts:
data/sessions/calib_<ts>/
  video.mp4             # raw webcam (no overlays in default run)
  frames.parquet        # dense per-frame features
  events.parquet        # one row per click
(If you want per-frame overlays drawn into the video, change the VideoWriterMP4 call in calibrate.py to pass an overlay function; optional for QA.)
3) Head-pose reality checks (A4)
While the UI runs (you can ignore the targets for this test):
Look at the center of the camera and keep your head still.
Expected: yaw, pitch, roll ≈ 0 ± (0.5–2.0)° jitter.
distance_mm ≈ 400–800 mm for a typical laptop webcam.
Yaw test: rotate your head left ↔ right ~20°.
Expected: |yaw| grows smoothly to ~20°. Sign depends on convention; just confirm it’s consistent (it shouldn’t flip randomly).
Pitch test: nod up/down ~15°.
Expected: |pitch| grows smoothly; returns near 0 when you’re neutral.
Roll test: tilt ear toward shoulder ~10°.
Expected: |roll| grows smoothly.
Distance: lean in/out by ~10 cm.
Expected: distance_mm changes by roughly that amount; jitter should be small (<30 mm) when still.
If these aren’t true:
Make sure you’re using calibrated intrinsics (camera.intrinsics_path in configs/default.yaml).
Ensure good, even lighting, face fully visible, and 720p or better capture.
4) Pupil/iris angles sanity (A5)
Stare at the camera and move only your eyes (keep head steady).
Left/right gaze: left_yaw and right_yaw change symmetrically; center gaze ≈ 0 rad; ±0.03–0.08 rad (≈2–5°) for moderate eye motions.
Up/down gaze: *_pitch changes; “look up” should produce opposite sign to “look down”. (We invert image-Y so up = positive pitch.)
Blink: close your eyes briefly.
Expected: blink=True frames around the blink; you’ll see short runs in frames.parquet.
If the magnitudes seem too small or too large: verify fx, fy (from your K). Using defaults (no intrinsics) still “works” but scales angles less accurately.
5) Quick data checks (frames/events)
Use pandas (or pyarrow) to spot-check logs:
import pandas as pd
import numpy as np

sess = "data/sessions/calib_YYYYmmdd_HHMMSS"
f = pd.read_parquet(f"{sess}/frames.parquet")
e = pd.read_parquet(f"{sess}/events.parquet")

print("frames:", len(f), "events:", len(e))
print("face_present ratio:", f.face_present.mean().round(3))
print("blink ratio:", f.blink.mean().round(3))
print("yaw/pitch (deg) mean±std:", 
      round(f.head_yaw_deg.dropna().mean(),2), round(f.head_yaw_deg.dropna().std(),2),
      round(f.head_pitch_deg.dropna().mean(),2), round(f.head_pitch_deg.dropna().std(),2))

# nearest-frame join to stamp targets post-hoc (if you want to inspect around clicks)
idx = np.searchsorted(f.t_mono.values, e.t_mono.values)
idx = np.clip(idx, 0, len(f)-1)
f.loc[idx, "target_x"] = e.target_x.values
f.loc[idx, "target_y"] = e.target_y.values
print("frames with stamped targets:", f.target_x.notna().sum())
Healthy ranges (typical laptop, decent light):
face_present > 0.85 during active calibration.
blink ratio ≈ 0.02–0.06 (people blink 10–20/min).
Head angle std when still: ≤2°.
Pupil angle std when fixating: ≤0.02 rad (~1.1°).
6) Visual QA in the MP4
Open video.mp4:
Tracking should not drop out when you slightly rotate or move.
Blinks/half-closure should temporarily reduce iris confidence; you’ll see brief instability in angles (those frames get blink=True).
Strong backlight or glasses glare may cause intermittent face_present=False.
7) Grid & click behavior
The red “next” cross should minimize big saccades (order is serpentine with small shifts). If you want to stress long saccades, re-run with a different --seed.
Each click should immediately advance and log an event; if you mis-click off the cross, it still logs (that’s intentional—ground truth is the target, not the click).
8) Performance & latency
On a modest CPU, MediaPipe at 720p should give >20 FPS. If you see <15 FPS:
Drop capture to 960×540 or increase min_tracking_confidence.
Close heavy apps; ensure you aren’t mirroring multiple 4K displays.
End-to-end UI responsiveness (click → advance) should feel instant; if it lags, check that the UI FPS is healthy.
9) Common failure modes (and quick fixes)
Angles noisy / wrong scale → use camera intrinsics; ensure the correct intrinsics_path points to your .yaml.
Pose flips or spikes → keep at least the 6 PnP landmarks visible; tie back hair; avoid extreme roll.
Blink misclassification → thresholds are conservative; fine later. For now, it’s OK if some “half-blinks” slip through.
Black/empty video → check --record resolution matches camera frames (our writer resizes, but confirm cw×ch are sane).
10) “Did I pass?”—a simple acceptance checklist
 events.parquet has exactly 25 rows; order matches calib_points.json.
 frames.parquet has thousands of rows for a short run; most rows have face_present=True.
 Head pose behaves smoothly; neutral near zero, sensible ranges (±30° yaw/pitch).
 Pupil angles change when eyes move, even if the head doesn’t.
 Blink spikes to True when you blink.
 distance_mm is plausible and stable when you’re still.
 Camera/UI FPS are close to expected.
If you hit anything weird, tell me which of the checks above failed and paste a few sample rows from frames.parquet + your config_snapshot.yaml—I’ll help you zero in on it.



eye-tracker/ 
├─ pyproject.toml # deps + entry points 
├─ README.md
├─ LICENSE 
├─ .env.example 
├─ configs/ 
│ └─ default.yaml # camera id, screen size, MP/Spiga toggle, paths, calibration 
├─ data/ 
│ ├─ sessions/ # per-session logs (Parquet/CSV + MP4) 
│ └─ camera/ # camera intrinsics (json/yaml) if available 
├─ eyetracker/ # Python package 
│ ├─ __init__.py 
│ ├─ app/ 
│ │ ├─ calibrate.py # CLI entry for Part A 
│ │ └─ cam_calibrate.py # generating k file for the camera 
│ ├─ ui/ 
│ │ ├─ grid.py # 5×5 target generator, show current+next cross 
│ │ └─ draw.py # cross/circle rendering, HUD overlays 
│ ├─ video/ 
│ │ ├─ capture.py # cv2.VideoCapture thread (timestamped frames) 
│ │ └─ writer.py # cv2.VideoWriter with overlay hooks 
│ ├─ vision/ 
│ │ ├─ camera_models.py
│ │ ├─ camera_utils.py
│ │ ├─ mediapipe_iris.py # FaceMesh/Iris wrapper → landmarks (468+iris) 
│ │ ├─ spiga_adapter.py # Optional SPIGA frontend → landmarks & head pose 
│ │ ├─ landmarks.py # helper ops (eye corners, iris center fit, blink) 
│ │ └─ headpose.py # solvePnP, yaw/pitch/roll, distance 
│ ├─ features/ 
│ │ ├─ feature_builder.py # build per-frame feature vector 
│ │ └─ geometry.py # angles from pixel offsets, normalization 
│ ├─ io/ 
│ │ ├─ logger.py # structured logging (Parquet/CSV + schema) 
│ │ └─ schemas.py # dataclasses / pydantic models for rows 
│ ├─ quality/ 
│ │ └─ gates.py # face present, blink, motion sanity flags 
│ └─ utils/ 
│ ├─ timing.py # monotonic timestamps, NTP drift note 
│ └─ screen.py # get screen size, DPI scaling, coordinates 
├─ scripts/ 
│ ├─ demo_calib.sh # runs Part A with chosen backend 
│ └─ export_session_to_csv.py 
├─ tests/ 
│ ├─ test_grid.py 
│ ├─ test_headpose.py 
│ └─ test_features.py