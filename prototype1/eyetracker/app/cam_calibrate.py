from __future__ import annotations
import argparse, os, sys, time
import yaml
import cv2
import numpy as np
from pathlib import Path

from ..video.capture import CameraCapture
from ..video.camera_model import (
    calibrate_checkerboard, calibrate_charuco,
    save_intrinsics, undistort_image
)
from ..video.camera_utils import ensure_pose_diversity, draw_found_corners

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

def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    with CameraCapture(args.camera_id, args.width, args.height, args.fps) as cam:
        img_size = (args.width, args.height)
        collected_frames = []
        last_corners = []

        print("[i] Press SPACE to capture a sample (if detection OK), ENTER to finish, 'q' to abort.")
        if args.auto:
            print("[i] Auto-capture ON: frames will be taken when detection is good and pose changed.")

        while True:
            _, frame = cam.read()
            view = frame.copy()

            valid = False
            corners_vis = None
            if args.pattern == "checkerboard":
                cols, rows = args.cb_cols, args.cb_rows
                ret, corners = cv2.findChessboardCorners(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    (cols, rows),
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                if ret:
                    corners2 = cv2.cornerSubPix(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        corners, (11,11), (-1,-1),
                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    valid = True
                    corners_vis = corners2
                    view = draw_found_corners(view, corners2, True)
                else:
                    view = draw_found_corners(view, corners if corners is not None else np.zeros((0,1,2)), False)

            else:  # Charuco
                if not hasattr(cv2, "aruco"):
                    _put_hud(view, ["OpenCV ArUco not found. Install opencv-contrib-python."], "l")
                else:
                    aruco = cv2.aruco
                    dict_enum = getattr(aruco, args.aruco_dict)
                    dictionary = aruco.getPredefinedDictionary(dict_enum)

                    if hasattr(aruco, "DetectorParameters"):
                        params = aruco.DetectorParameters()
                    else:
                        params = aruco.DetectorParameters_create()
                    # (Optionally copy from YAML detector config here)

                    detector = aruco.ArucoDetector(dictionary, params) if hasattr(aruco, "ArucoDetector") else None
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if detector is not None:
                        corners, ids, _ = detector.detectMarkers(gray)
                    else:
                        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

                    nmk = 0 if ids is None else len(ids)
                    if nmk > 0:
                        aruco.drawDetectedMarkers(view, corners, ids)

                        try:
                            board = aruco.CharucoBoard(
                                (args.cx, args.cy),
                                args.charuco_square_mm / 1000.0,
                                args.charuco_marker_mm / 1000.0,
                                dictionary
                            )
                            aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)
                        except Exception:
                            pass

                        ok, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
                        cnt = 0 if ch_ids is None else len(ch_ids)
                        if ok and cnt >= 12:
                            valid = True
                            corners_vis = ch_corners
                            # draw refined ChArUco corners for feedback
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
                time.sleep(0.4)  # small debounce

            if key in (13, 10):  # ENTER
                break
            if key == ord('q') or len(collected_frames) >= args.max_frames:
                break

        cv2.destroyAllWindows()

    if len(collected_frames) < max(8, args.min_frames):
        print(f"[!] Not enough valid frames: {len(collected_frames)}. Need at least {max(8, args.min_frames)}.")
        sys.exit(2)

    # Run calibration
    if args.pattern == "checkerboard":
        ci = calibrate_checkerboard(
            frames=collected_frames,
            grid_size=(args.cb_cols, args.cb_rows),
            square_size_mm=args.cb_square_mm,
            img_size=img_size,
            fisheye=args.fisheye
        )
    else:
        ci = calibrate_charuco(
            frames=collected_frames,
            squares_x=args.cx, squares_y=args.cy,
            square_size_mm=args.charuco_square_mm,
            marker_size_mm=args.charuco_marker_mm,
            img_size=img_size,
            aruco_dict_name=args.aruco_dict
        )

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) / f"cam{args.camera_id}_{ci.model}_{img_size[0]}x{img_size[1]}_{ts}.yaml"
    save_intrinsics(ci, str(out_path))
    print(f"[✓] Saved intrinsics → {out_path}")
    print(f"    RMS reprojection error: {ci.rms_reprojection_error:.4f}  (frames used: {ci.frames_used})")

    # Quick preview: undistort first frame
    try:
        preview = undistort_image(collected_frames[0], ci)
        side = np.hstack([collected_frames[0], preview])
        cv2.imshow("Undistort preview (left=orig, right=undistorted)", side)
        print("[i] Press any key to close preview.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[!] Preview failed: {e}")
