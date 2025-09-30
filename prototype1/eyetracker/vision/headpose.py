# headpose.py
# Robust head-pose estimation using P3P+RANSAC with refinement and safe fallbacks.
# Compatible with OpenCV 4.x / 5.x
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "HeadPose",
    "HeadPoseEstimator",
    "estimate_headpose_p3p",
    "project_points",
]

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------- Data structures ----------

@dataclass
class HeadPose:
    ok: bool
    rvec: Optional[np.ndarray] = None  # (3,1) float64
    tvec: Optional[np.ndarray] = None  # (3,1) float64
    R: Optional[np.ndarray] = None     # (3,3) rotation matrix
    yaw: Optional[float] = None        # degrees
    pitch: Optional[float] = None      # degrees
    roll: Optional[float] = None       # degrees
    n_inliers: int = 0
    reproj_err: Optional[float] = None
    detR: Optional[float] = None
    tz: Optional[float] = None

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.rvec is None or self.tvec is None:
            raise ValueError("Pose is not valid.")
        return self.rvec, self.tvec


# ---------- Utilities ----------

def _as_float(arr, last_dim: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    a = a.reshape(-1, last_dim)
    return a


def _mean_reproj_error(obj_pts: np.ndarray, img_pts: np.ndarray,
                       rvec: np.ndarray, tvec: np.ndarray,
                       K: np.ndarray, dist: Optional[np.ndarray]) -> float:
    proj = project_points(obj_pts, rvec, tvec, K, dist)
    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.mean(err))


def _rvec_tvec_to_R_euler(rvec: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
    """Return (R, yaw, pitch, roll, detR) in degrees using a common camera convention:
    - yaw (Y)  : left/right rotation around +Y
    - pitch (X): up/down rotation around +X
    - roll (Z) : tilt around +Z
    The decomposition below is standard for many OpenCV head pose pipelines.
    """
    R, _ = cv2.Rodrigues(rvec)
    detR = float(np.linalg.det(R))
    sy = math.hypot(R[0, 0], R[1, 0])

    singular = sy < 1e-6
    if not singular:
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))        # around Y
        pitch = math.degrees(math.atan2(-R[2, 0], sy))          # around X
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))       # around Z
    else:
        # Gimbal lock fallback
        yaw = math.degrees(math.atan2(-R[0, 1], R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        roll = 0.0

    return R, yaw, pitch, roll, detR


def project_points(obj_pts: np.ndarray,
                   rvec: np.ndarray,
                   tvec: np.ndarray,
                   K: np.ndarray,
                   dist: Optional[np.ndarray]) -> np.ndarray:
    """Projects 3D points to 2D using OpenCV, returns (N,2) float64."""
    obj = _as_float(obj_pts, 3)
    img, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return img.reshape(-1, 2).astype(np.float64)


# ---------- Core solver ----------

def estimate_headpose_p3p(obj_pts: np.ndarray,
                          img_pts: np.ndarray,
                          K: np.ndarray,
                          dist: Optional[np.ndarray],
                          prev_rvec: Optional[np.ndarray] = None,
                          prev_tvec: Optional[np.ndarray] = None,
                          ransac_iters: int = 800,
                          ransac_err_px: float = 3.0,
                          ransac_conf: float = 0.999,
                          allow_mirrored: bool = False) -> HeadPose:
    """
    Robust head pose with P3P+RANSAC and LM refinement.
    - Uses P3P as minimal solver under RANSAC (needs >= 4 correspondences).
    - Refines with solvePnPRefineLM if available, else ITERATIVE with extrinsic guess.
    - Falls back to EPnP->ITERATIVE if P3P fails (e.g., near-coplanar points).
    - Sanity checks to avoid mirrored (detR<0) or tz<=0 solutions; if so, keep previous pose.
    """
    obj = _as_float(obj_pts, 3)
    img = _as_float(img_pts, 2)

    if obj.shape[0] != img.shape[0] or obj.shape[0] < 4:
        logger.warning("Need >= 4 correspondences, got %d.", obj.shape[0])
        return HeadPose(ok=False)

    # 1) RANSAC with P3P (no extrinsic guess allowed)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj,
        imagePoints=img,
        cameraMatrix=K,
        distCoeffs=dist,
        useExtrinsicGuess=False,
        iterationsCount=ransac_iters,
        reprojectionError=ransac_err_px,
        confidence=ransac_conf,
        flags=cv2.SOLVEPNP_P3P
    )

    # 2) Fallback if needed: EPnP -> ITERATIVE
    if not ok or inliers is None or len(inliers) < 4:
        ok_epnp, rvec_e, tvec_e = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_EPNP)
        if ok_epnp:
            ok_it, rvec_i, tvec_i = cv2.solvePnP(
                obj, img, K, dist,
                rvec=rvec_e, tvec=tvec_e, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok_it:
                ok, rvec, tvec = True, rvec_i, tvec_i
                inliers = None  # unknown
            else:
                ok = False
        else:
            ok = False

    if not ok:
        # Last-resort fallback: keep previous pose if available
        if prev_rvec is not None and prev_tvec is not None:
            R, yaw, pitch, roll, detR = _rvec_tvec_to_R_euler(prev_rvec)
            tz_prev = float(prev_tvec[2])
            return HeadPose(
                ok=True, rvec=np.asarray(prev_rvec, np.float64), tvec=np.asarray(prev_tvec, np.float64),
                R=R, yaw=yaw, pitch=pitch, roll=roll, n_inliers=0, reproj_err=None, detR=detR, tz=tz_prev
            )
        return HeadPose(ok=False)

    # 3) Optional refinement on inliers if present
    try:
        if inliers is not None and len(inliers) >= 4 and hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                obj[inliers[:, 0]], img[inliers[:, 0]], K, dist, rvec, tvec
            )
        else:
            # Fallback refinement (ITERATIVE) with the current estimate as guess
            ok_it, rvec_i, tvec_i = cv2.solvePnP(
                obj, img, K, dist,
                rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok_it:
                rvec, tvec = rvec_i, tvec_i
    except cv2.error as e:
        logger.debug("Refinement skipped due to OpenCV error: %s", e)

    # 4) Sanity checks: mirrored / behind camera
    R, yaw, pitch, roll, detR = _rvec_tvec_to_R_euler(rvec)
    tz = float(tvec[2])

    if not allow_mirrored and detR < 0:
        logger.debug("Rejecting mirrored solution (detR < 0).")
        if prev_rvec is not None and prev_tvec is not None:
            R0, yaw0, pitch0, roll0, detR0 = _rvec_tvec_to_R_euler(prev_rvec)
            return HeadPose(
                ok=True, rvec=np.asarray(prev_rvec, np.float64), tvec=np.asarray(prev_tvec, np.float64),
                R=R0, yaw=yaw0, pitch=pitch0, roll=roll0, n_inliers=int(len(inliers) if inliers is not None else 0),
                reproj_err=None, detR=detR0, tz=float(prev_tvec[2])
            )
        # if no previous, still return but flag as suspicious
        logger.warning("det(R)<0 with no previous pose; returning current solution.")
    if tz <= 0 and prev_tvec is not None and float(prev_tvec[2]) > 0:
        logger.debug("Rejecting tz<=0; keeping previous forward-facing pose.")
        R0, yaw0, pitch0, roll0, detR0 = _rvec_tvec_to_R_euler(prev_rvec)
        return HeadPose(
            ok=True, rvec=np.asarray(prev_rvec, np.float64), tvec=np.asarray(prev_tvec, np.float64),
            R=R0, yaw=yaw0, pitch=pitch0, roll=roll0, n_inliers=int(len(inliers) if inliers is not None else 0),
            reproj_err=None, detR=detR0, tz=float(prev_tvec[2])
        )

    # 5) Metrics
    reproj = _mean_reproj_error(obj, img, rvec, tvec, K, dist)
    n_inl = int(len(inliers) if inliers is not None else 0)

    return HeadPose(
        ok=True, rvec=rvec, tvec=tvec, R=R,
        yaw=yaw, pitch=pitch, roll=roll,
        n_inliers=n_inl, reproj_err=reproj, detR=detR, tz=tz
    )


# ---------- Estimator class (stateful) ----------

class HeadPoseEstimator:
    """
    Stateful wrapper that remembers the previous pose and exposes a simple API:
        estimator = HeadPoseEstimator(K, dist)
        pose = estimator.estimate(obj_pts, img_pts)
    """

    def __init__(self,
                 K: np.ndarray,
                 dist: Optional[np.ndarray] = None,
                 ransac_iters: int = 800,
                 ransac_err_px: float = 3.0,
                 ransac_conf: float = 0.999,
                 allow_mirrored: bool = False) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = None if dist is None else np.asarray(dist, dtype=np.float64)
        self.ransac_iters = int(ransac_iters)
        self.ransac_err_px = float(ransac_err_px)
        self.ransac_conf = float(ransac_conf)
        self.allow_mirrored = bool(allow_mirrored)

        self._prev_rvec: Optional[np.ndarray] = None
        self._prev_tvec: Optional[np.ndarray] = None

    def estimate(self,
                 obj_pts: np.ndarray,
                 img_pts: np.ndarray) -> HeadPose:
        pose = estimate_headpose_p3p(
            obj_pts=obj_pts,
            img_pts=img_pts,
            K=self.K,
            dist=self.dist,
            prev_rvec=self._prev_rvec,
            prev_tvec=self._prev_tvec,
            ransac_iters=self.ransac_iters,
            ransac_err_px=self.ransac_err_px,
            ransac_conf=self.ransac_conf,
            allow_mirrored=self.allow_mirrored
        )
        if pose.ok:
            self._prev_rvec = pose.rvec
            self._prev_tvec = pose.tvec
        return pose

    def reset(self) -> None:
        """Forget previous pose (e.g., on big head motion or tracker reset)."""
        self._prev_rvec, self._prev_tvec = None, None

    @property
    def prev_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._prev_rvec is None or self._prev_tvec is None:
            return None
        return self._prev_rvec, self._prev_tvec
