Eye Tracker
A webcam-based eye tracking and calibration toolkit with support for MediaPipe FaceMesh/Iris, SPIGA (adapter-ready), and ChArUco/Checkerboard camera calibration.
It provides full-session logging (frames, events, video), head pose estimation, iris/pupil features, and real-time quality checks.
📦 Installation
pip install -e .
🎯 Calibration
ChArUco (default)
eye-cam-calibrate --camera-id 0 --width 1280 --height 720
Checkerboard
eye-cam-calibrate --pattern checkerboard --cb-cols 9 --cb-rows 6 --cb-square-mm 25
🚀 Run the Calibration UI
python -m eyetracker.app.calibrate --backend none --hud
python -m eyetracker.app.calibrate --backend mediapipe --hud --record
CLI Flags
  Flag          Description	                                   Default
--config <path>	Path to YAML config	                            configs/default.yaml
--seed <int>	  Seed for grid order	                            17
--hud	Show      on-screen HUD (FPS, backend, point idx, flags)	Off
--backend       {mediapipe... ,none}	Landmark backend	        mediapipe
--record	      Save webcam video to session folder	            Off
🗂 Project Structure
eye-tracker/
├─ pyproject.toml        # Dependencies + entry points
├─ README.md
├─ LICENSE
├─ configs/
│   └─ default.yaml      # Camera ID, screen size, backend toggle
├─ data/
│   ├─ sessions/         # Per-session logs (Parquet/CSV + MP4)
│   └─ camera/           # Camera intrinsics (json/yaml)
├─ eyetracker/           # Python package
│   ├─ app/              # CLI entry points
│   ├─ ui/               # Calibration grid + HUD rendering
│   ├─ video/            # Capture + writer with overlays
│   ├─ vision/           # Landmark backends + headpose
│   ├─ features/         # Iris, pupil, gaze geometry
│   ├─ io/               # Logging + schemas
│   ├─ quality/          # Real-time quality gates
│   └─ utils/            # Timing, screen helpers
├─ scripts/              # Demos + exports
└─ tests/                # Unit tests
✅ Quick Tests
1. UI Only (no webcam)
python -m eyetracker.app.calibrate --backend none --hud
Full-screen black canvas with a white (current) and red (next) cross.
Click through 25 points → session saved in data/sessions/.
2. Webcam + MediaPipe
python -m eyetracker.app.calibrate --backend mediapipe --hud --record
HUD shows FPS, face presence, blink detection.
Session folder includes frames.parquet, events.parquet, video.mp4.
🧪 Features & Checks
Camera Calibration: ChArUco or checkerboard intrinsics.
Head Pose: Robust PnP → yaw, pitch, roll, distance.
Eye Features: Iris circle fit, pupil angles, blinks.
Session Logging: Frames/events saved as Parquet, optional MP4.
Quality Gates: Face present, blink detection, motion sanity.



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
