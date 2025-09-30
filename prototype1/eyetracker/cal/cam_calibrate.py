from __future__ import annotations
import argparse, os, sys, time
import yaml
import cv2
import numpy as np
from pathlib import Path

from ..video.capture import CameraCapture
from ..vision.camera_model import (
    calibrate_checkerboard, calibrate_charuco,
    save_intrinsics, undistort_image, ensure_pose_diversity, draw_found_corners
)

def _put_hud(frame, text_lines, pos):
    y = 24
    if pos == "r":
        pos = frame.shape[1] - 10 - max(200, max(len(line) for line in text_lines)*10) 
    else:
        pos = 12
    for line in text_lines:
        cv2.putText(frame, line, (pos, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (pos, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return frame

def parse_args():
    ap = argparse.ArgumentParser(description="Calibrate a webcam (checkerboard or ChArUco) and save intrinsics.")
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pattern", choices=["checkerboard", "charuco"], default="charuco")

    # Checkerboard
    ap.add_argument("--cb-cols", type=int, default=9, help="inner corners horizontal")
    ap.add_argument("--cb-rows", type=int, default=6, help="inner corners vertical")
    ap.add_argument("--cb-square-mm", type=float, default=25.0)
    ap.add_argument("--fisheye", action="store_true", help="Use fisheye model for checkerboard")

    # Charuco
    ap.add_argument("--cx", type=int, default=5, help="ChArUco squares_x")
    ap.add_argument("--cy", type=int, default=7, help="ChArUco squares_y")
    ap.add_argument("--charuco-square-mm", type=float, default=30.0)
    ap.add_argument("--charuco-marker-mm", type=float, default=22.0)
    ap.add_argument("--aruco-dict", type=str, default="DICT_5X5_1000")

    # Sampling
    ap.add_argument("--min-frames", type=int, default=18)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--min-motion-px", type=float, default=20.0)
    ap.add_argument("--auto", action="store_true", default=True)
    ap.add_argument("--out", type=str, default="data/camera")
    return ap.parse_args()

def make_charuco(dictionary, cx, cy, square_len, marker_len):
    # OpenCV 4.x cross-version safe creator
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(cx, cy, float(square_len), float(marker_len), dictionary)
    else:
        # Older API had a class but still exposes _create
        board = cv2.aruco.CharucoBoard((cx, cy), float(square_len), float(marker_len), dictionary)
    return board

def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Make sure cv2.aruco exists (opencv-contrib) – fail fast with a clear message
    if args.pattern == "charuco" and not hasattr(cv2, "aruco"):
        print("[!] OpenCV ArUco module not found. Install: pip install opencv-contrib-python")
        sys.exit(1)

    # Create dictionary/board once (not every frame)
    aruco = getattr(cv2, "aruco", None)
    dictionary = None
    board = None
    if args.pattern == "charuco" and aruco is not None:
        dict_enum = getattr(aruco, args.aruco_dict, None)
        if dict_enum is None:
            print(f"[!] Unknown ArUco dict name: {args.aruco_dict}. Example: DICT_5X5_1000")
            sys.exit(1)
        dictionary = aruco.getPredefinedDictionary(dict_enum)
        board = make_charuco(
            dictionary,
            args.cx, args.cy,
            args.charuco_square_mm,   # units are arbitrary but must be consistent
            args.charuco_marker_mm
        )

    # Try to open camera
    with CameraCapture(args.camera_id, args.width, args.height, args.fps) as cam:
        ok, test = cam.read()
        if not ok or test is None:
            print(f"[!] Could not read from camera id {args.camera_id} at {args.width}x{args.height}@{args.fps}")
            sys.exit(2)

        img_size = (args.width, args.height)
        collected_frames, last_corners = [], []

        cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
        print("[i] SPACE: capture | ENTER: finish | q: abort")
        if args.auto:
            print("[i] Auto-capture ON (pose diversity gating)")

        # Pre-build detector params once
        params = None
        detector = None
        if aruco is not None:
            if hasattr(aruco, "DetectorParameters"):
                params = aruco.DetectorParameters()
            else:
                params = aruco.DetectorParameters_create()
            if hasattr(aruco, "ArucoDetector"):
                detector = aruco.ArucoDetector(dictionary, params)

        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                continue  # keep trying; USB cams hiccup

            view = frame.copy()
            valid, corners_vis = False, None

            if args.pattern == "checkerboard":
                # ... your checkerboard code unchanged ...
                pass
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if detector is not None:
                    corners, ids, _ = detector.detectMarkers(gray)
                else:
                    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

                nmk = 0 if ids is None else len(ids)
                if nmk > 0:
                    aruco.drawDetectedMarkers(view, corners, ids)
                    # refineDetectedMarkers signature differs across versions; guard it
                    try:
                        aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)
                    except Exception:
                        pass

                    # interpolate ChArUco corners (board must be valid)
                    try:
                        ok_ch, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
                    except Exception:
                        ok_ch, ch_corners, ch_ids = False, None, None

                    cnt = 0 if ch_ids is None else len(ch_ids)
                    if ok_ch and cnt >= 12:
                        valid = True
                        corners_vis = ch_corners
                        for p in ch_corners.reshape(-1, 2):
                            cv2.circle(view, (int(p[0]), int(p[1])), 3, (0,255,0), -1, lineType=cv2.LINE_AA)

                    _put_hud(view, [f"Dict: {args.aruco_dict} | markers: {nmk} | ch_corners: {cnt}"], "r")
                else:
                    _put_hud(view, [f"Dict: {args.aruco_dict} | markers: 0"], "r")

            hud = [
                f"Pattern: {args.pattern} | collected: {len(collected_frames)}/{args.max_frames}",
                "SPACE: capture | ENTER: finish | q: abort",
                "Auto-capture: ON" if args.auto else "Auto-capture: OFF"
            ]
            _put_hud(view, hud, "l")
            cv2.imshow("Camera Calibration", view)
            key = cv2.waitKey(1) & 0xFF

            should_capture = False
            if valid and corners_vis is not None:
                if args.auto and ensure_pose_diversity(last_corners, corners_vis, args.min_motion_px):
                    should_capture = True
                elif key == ord(' '):
                    should_capture = True

            if should_capture:
                collected_frames.append(frame.copy())
                last_corners.append(corners_vis.copy())
                print(f"[+] Captured sample #{len(collected_frames)}")
                time.sleep(0.35)

            if key in (13, 10):  # ENTER
                break
            if key == ord('q') or len(collected_frames) >= args.max_frames:
                break

        cv2.destroyAllWindows()

    if len(collected_frames) < max(8, args.min_frames):
        print(f"[!] Not enough valid frames: {len(collected_frames)}. Need at least {max(8, args.min_frames)}.")
        sys.exit(2)

    # --- calibration call stays exactly like your code ---

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")