from __future__ import annotations
import yaml
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from . import camera_utils 
from ..io.schemas import CameraIntrinsics

def load_intrinsics(path: str) -> CameraIntrinsics:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return CameraIntrinsics(**d)

def save_intrinsics(ci: CameraIntrinsics, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(ci.model_dump(), f, sort_keys=False)

def undistort_image(img: np.ndarray, ci: CameraIntrinsics) -> np.ndarray:
    if ci.model == "fisheye":
        K = ci.K_mat()
        D = ci.dist_vec()
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (ci.width, ci.height), np.eye(3), balance=0.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (ci.width, ci.height), cv2.CV_16SC2)
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    else:
        K = ci.K_mat()
        D = ci.dist_vec()
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (ci.width, ci.height), alpha=0.0)
        return cv2.undistort(img, K, D, None, new_K)

def undistort_points(pts: np.ndarray, ci: CameraIntrinsics) -> np.ndarray:
    """pts: Nx2 pixel coords -> Nx2 undistorted."""
    K = ci.K_mat()
    D = ci.dist_vec()
    pts = pts.reshape(-1, 1, 2).astype(np.float64)
    if ci.model == "fisheye":
        und = cv2.fisheye.undistortPoints(pts, K, D, P=K)
    else:
        und = cv2.undistortPoints(pts, K, D, P=K)
    return und.reshape(-1, 2)

# ---- Calibration helpers (checkerboard / Charuco) ----

def calibrate_checkerboard(
    frames: list[np.ndarray],
    grid_size: Tuple[int, int],
    square_size_mm: float,
    img_size: Tuple[int, int],
    fisheye: bool = False
) -> CameraIntrinsics:
    cols, rows = grid_size  # inner corners
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= (square_size_mm / 1000.0)  # meters (scale irrelevant, consistency matters)

    objpoints = []
    imgpoints = []

    # detect corners
    for im in frames:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows),
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret:
            continue
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)
        objpoints.append(objp)

    if len(objpoints) < 8:
        raise RuntimeError(f"Not enough valid frames for calibration (got {len(objpoints)}, need ≥8).")

    if fisheye:
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs, tvecs = [], []
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            [objp] * len(imgpoints), [ip.reshape(-1, 1, 2) for ip in imgpoints],
            img_size, K, D, rvecs, tvecs, flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
        )
        model = "fisheye"
    else:
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None
        )
        rms = float(ret)
        model = "opencv"

    return CameraIntrinsics.from_opencv(
        width=img_size[0], height=img_size[1], K=K, dist=D, rms=rms,
        frames_used=len(objpoints), model=model, notes=f"checkerboard {cols}x{rows} square={square_size_mm}mm"
    )

def calibrate_charuco(
    frames: list[np.ndarray],
    squares_x: int, squares_y: int,
    square_size_mm: float, marker_size_mm: float,
    img_size: Tuple[int, int],
    aruco_dict_name: str = "DICT_5X5_1000",
    detector_params: dict | None = None
) -> CameraIntrinsics:
    """
    Robust Charuco calibration that works across OpenCV 4.x variants.
    - Uses ArucoDetector if available (OpenCV >=4.7); else falls back to detectMarkers.
    - Checks dictionary and prints a friendly error if it's unknown.
    """
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not found. Install opencv-contrib-python.")

    aruco = cv2.aruco

    # --- dictionary (int enum -> dictionary object) ---
    if not hasattr(aruco, aruco_dict_name):
        raise RuntimeError(f"Unknown ArUco dictionary '{aruco_dict_name}'. "
                           f"Common options: DICT_4X4_50, DICT_5X5_100, DICT_5X5_1000, DICT_6X6_250, ...")
    dict_enum = getattr(aruco, aruco_dict_name)
    dictionary = aruco.getPredefinedDictionary(dict_enum)

    # --- detector params (API differs across versions) ---
    def make_params():
        # OpenCV ≥4.7 often exposes a class constructor; older builds use _create()
        if hasattr(aruco, "DetectorParameters"):
            p = aruco.DetectorParameters()
        else:
            p = aruco.DetectorParameters_create()
        if detector_params:
            for k, v in detector_params.items():
                if hasattr(p, k):
                    setattr(p, k, v)
        # Good defaults for printouts under room light:
        # lower adaptive thresholding window can help on small printouts
        if hasattr(p, "adaptiveThreshWinSizeMin") and p.adaptiveThreshWinSizeMin < 3:
            p.adaptiveThreshWinSizeMin = 3
        if hasattr(p, "adaptiveThreshWinSizeMax") and p.adaptiveThreshWinSizeMax < 23:
            p.adaptiveThreshWinSizeMax = 23
        if hasattr(p, "minMarkerPerimeterRate") and p.minMarkerPerimeterRate < 0.03:
            p.minMarkerPerimeterRate = 0.03
        return p

    params = make_params()

    # OpenCV ≥4.7: ArucoDetector; else legacy detectMarkers
    use_detector = hasattr(aruco, "ArucoDetector")
    detector = aruco.ArucoDetector(dictionary, params) if use_detector else None

    # --- board geometry (units: meters; scale consistent, not critical) ---
    square_len_m = float(square_size_mm) / 1000.0
    marker_len_m = float(marker_size_mm) / 1000.0
    board = aruco.CharucoBoard((squares_x, squares_y), square_len_m, marker_len_m, dictionary)

    all_charuco_corners = []
    all_charuco_ids = []

    for im in frames:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        if use_detector:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

        if ids is None or len(ids) == 0:
            continue

        # Helpful: refine detections in-place using the board layout
        try:
            aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)
        except Exception:
            pass  # not all builds support refine

        ok, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if ok and ch_corners is not None and ch_ids is not None and len(ch_ids) >= 12:
            all_charuco_corners.append(ch_corners)
            all_charuco_ids.append(ch_ids)

    if len(all_charuco_ids) < 8:
        raise RuntimeError(
            f"Not enough valid ChArUco views (got {len(all_charuco_ids)}, need ≥8). "
            "Common causes: dictionary mismatch, glare/blur, board too close/small in view."
        )

    # Calibrate
    try:
        ret, K, D, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None
        )
    except Exception:
        # Some builds only have calibrateCameraCharucoExtended signature
        ret, K, D, rvecs, tvecs, *_ = aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None
        )

    rms = float(ret)
    return CameraIntrinsics.from_opencv(
        width=img_size[0], height=img_size[1], K=K, dist=D, rms=rms,
        frames_used=len(all_charuco_ids),
        model="opencv",
        notes=f"charuco {squares_x}x{squares_y} square={square_size_mm}mm marker={marker_size_mm}mm dict={aruco_dict_name}"
    )
