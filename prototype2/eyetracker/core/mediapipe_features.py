import numpy as np
import mediapipe as mp

# FaceMesh with iris refinement (478 landmarks)
_FM = mp.solutions.face_mesh
_IRIS_L = [468, 469, 470, 471, 472]
_IRIS_R = [473, 474, 475, 476, 477]

# eyelid landmark sets for EAR (blink)
# standard choice on MediaPipe topology
L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [263, 387, 385, 362, 380, 373]

# Outer eye corners for PnP points
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP        = 1
CHIN            = 152
MOUTH_L         = 61
MOUTH_R         = 291

PNP_LMKS = [NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, MOUTH_L, MOUTH_R]

def _to_px(lmks, w, h):
    arr = np.array([(p.x*w, p.y*h) for p in lmks], dtype=np.float64)
    return arr

def _centroid(px_coords):
    return np.mean(px_coords, axis=0)

def _ear(eye_pts):
    # eye_pts: 6x2 [p1,p2,p3,p4,p5,p6]
    p2p6 = np.linalg.norm(eye_pts[1]-eye_pts[5])
    p3p5 = np.linalg.norm(eye_pts[2]-eye_pts[4])
    p1p4 = np.linalg.norm(eye_pts[0]-eye_pts[3])
    return (p2p6 + p3p5) / (2.0 * p1p4 + 1e-6)

class MPFaceIris:
    def __init__(self, max_faces=1):
        self.mesh = _FM.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(frame_bgr[:, :, ::-1])  # RGB
        if not res.multi_face_landmarks:
            return dict(face_present=False)
        lmks = res.multi_face_landmarks[0].landmark

        # pupils (iris centroids in pixels)
        irisL = _to_px([lmks[i] for i in _IRIS_L], w, h)
        irisR = _to_px([lmks[i] for i in _IRIS_R], w, h)
        pupil_l = _centroid(irisL)
        pupil_r = _centroid(irisR)

        # eye ROIs for simple angle proxy (normalized offsets → small-angle approx)
        # compute per-eye yaw/pitch ~ atan(offset/scale)
        # scale by eye width (outer corners to inner corners proxy)
        l_outer = np.array([lmks[33].x*w, lmks[33].y*h])
        l_inner = np.array([lmks[133].x*w, lmks[133].y*h])
        r_outer = np.array([lmks[263].x*w, lmks[263].y*h])
        r_inner = np.array([lmks[362].x*w, lmks[362].y*h])

        l_center = 0.5*(l_outer + l_inner)
        r_center = 0.5*(r_outer + r_inner)
        l_w = np.linalg.norm(l_outer - l_inner) + 1e-6
        r_w = np.linalg.norm(r_outer - r_inner) + 1e-6

        # small-angle proxy in degrees
        def _angles(pupil, center, width):
            dx = (pupil[0] - center[0]) / width
            dy = (pupil[1] - center[1]) / width
            yaw = np.degrees(np.arctan(dx))   # left negative, right positive (camera coords)
            pitch = np.degrees(np.arctan(-dy))# up positive
            return pitch, yaw

        l_pitch, l_yaw = _angles(pupil_l, l_center, l_w)
        r_pitch, r_yaw = _angles(pupil_r, r_center, r_w)

        # blink via EAR
        l_eye = _to_px([lmks[i] for i in L_EYE], w, h)
        r_eye = _to_px([lmks[i] for i in R_EYE], w, h)
        ear_l, ear_r = _ear(l_eye), _ear(r_eye)

        # 2D points for PnP, in px
        pnp_px = _to_px([lmks[i] for i in PNP_LMKS], w, h)

        return dict(
            face_present=True,
            pupil_l=pupil_l, pupil_r=pupil_r,
            eye_pitch_l=l_pitch, eye_yaw_l=l_yaw,
            eye_pitch_r=r_pitch, eye_yaw_r=r_yaw,
            ear_l=ear_l, ear_r=ear_r,
            pnp_px=pnp_px
        )
