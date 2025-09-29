import numpy as np
import cv2 as cv

# 3D model points (generic head model in mm; robust for PnP)
_MODEL_3D = np.array([
    [0.0,   0.0,    0.0],    # nose tip
    [0.0,  -330.0, -65.0],   # chin
    [-225.0, 170.0, -135.0], # left eye corner (outer)
    [225.0,  170.0, -135.0], # right eye corner (outer)
    [-150.0,-150.0, -125.0], # left mouth corner
    [150.0, -150.0, -125.0], # right mouth corner
], dtype=np.float64)

def _rotation_matrix_to_euler(R):
    # ZYX convention → yaw(Z), pitch(Y), roll(X) in degrees
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        yaw = np.degrees(np.arctan2(R[1,0], R[0,0]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        roll = np.degrees(np.arctan2(R[2,1], R[2,2]))
    else:
        yaw = np.degrees(np.arctan2(-R[0,1], R[1,1]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        roll = 0.0
    return yaw, pitch, roll

def _unwrap(prev, curr):
    # unwrap to keep continuity around ±180
    if prev is None: return curr
    delta = curr - prev
    if delta > 180: curr -= 360
    elif delta < -180: curr += 360
    return curr

class HeadPose:
    def __init__(self):
        self.prev_ypr = None

    def solve(self, img_points_2d, K, dist=None, rvec=None, tvec=None):
        # img_points_2d: 6x2 np.float64 aligned to _MODEL_3D order
        success, rvec, tvec = cv.solvePnP(
            _MODEL_3D, img_points_2d, K, dist,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=(rvec is not None),
            flags=cv.SOLVEPNP_ITERATIVE
        )
        if not success: return None

        R, _ = cv.Rodrigues(rvec)
        yaw, pitch, roll = _rotation_matrix_to_euler(R)

        # unwrap for stability
        if self.prev_ypr is not None:
            yaw  = _unwrap(self.prev_ypr[0], yaw)
            pitch= _unwrap(self.prev_ypr[1], pitch)
            roll = _unwrap(self.prev_ypr[2], roll)
        self.prev_ypr = (yaw, pitch, roll)

        # ensure forward +Z (OpenCV camera convention); reject mirrored
        if tvec[2] <= 0:
            # try flip reflection if R is improper
            if np.linalg.det(R) < 0:
                R[:,2] *= -1
                yaw, pitch, roll = _rotation_matrix_to_euler(R)
                if self.prev_ypr is not None:
                    yaw  = _unwrap(self.prev_ypr[0], yaw)
                    pitch= _unwrap(self.prev_ypr[1], pitch)
                    roll = _unwrap(self.prev_ypr[2], roll)
                # keep original tvec but mark invalid Z
            # leave as-is; caller can mark invalid
        return dict(R=R, t=tvec.reshape(-1), yaw=yaw, pitch=pitch, roll=roll)

def build_camera_matrix(w, h, intrinsics=None):
    if intrinsics is not None:
        return intrinsics
    # Fallback: assume fx=fy ~ max(w,h) * 1.2, principal at center
    f = max(w, h) * 1.2
    return np.array([[f, 0, w/2.0],
                     [0, f, h/2.0],
                     [0, 0, 1.0]], dtype=np.float64)
