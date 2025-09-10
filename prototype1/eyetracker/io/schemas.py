from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional
import numpy as np

CameraModelKind = Literal["opencv", "fisheye"]

class CameraIntrinsics(BaseModel):
    model: CameraModelKind = "opencv"
    width: int
    height: int
    K: List[float] = Field(..., description="Row-major 3x3")
    dist: List[float] = Field(..., description="Distortion coefficients")
    rms_reprojection_error: float
    frames_used: int
    notes: Optional[str] = None

    @field_validator("K")
    @classmethod
    def _check_k(cls, v: List[float]) -> List[float]:
        if len(v) != 9:
            raise ValueError("K must have 9 elements (3x3)")
        return v

    def K_mat(self) -> np.ndarray:
        return np.array(self.K, dtype=np.float64).reshape(3, 3)

    def dist_vec(self) -> np.ndarray:
        return np.array(self.dist, dtype=np.float64).reshape(-1,)

    @classmethod
    def from_opencv(
        cls, width: int, height: int, K: np.ndarray, dist: np.ndarray,
        rms: float, frames_used: int, model: CameraModelKind = "opencv", notes: str | None = None
    ) -> "CameraIntrinsics":
        return cls(
            model=model, width=width, height=height,
            K=K.astype(float).reshape(-1).tolist(),
            dist=dist.astype(float).reshape(-1).tolist(),
            rms_reprojection_error=float(rms),
            frames_used=int(frames_used),
            notes=notes
        )

# ---- frame & event rows for Parquet ----

class FrameRow(BaseModel):
    # timing
    t_mono: float
    frame_id: int

    # screen meta (for normalization reproducibility)
    screen_w: int
    screen_h: int

    # camera meta
    cam_w: int
    cam_h: int

    # head pose
    head_yaw_deg: float
    head_pitch_deg: float
    head_roll_deg: float
    head_dist_mm: float

    # eye geometry (angles in deg)
    left_yaw: float
    left_pitch: float
    right_yaw: float
    right_pitch: float


    # quality
    face_present: bool
    blink: bool
    landmark_score: float

    # target (screen px) at click frames; else NaN
    target_x: float
    target_y: float

class EventRow(BaseModel):
    t_mono: float
    idx: int
    target_x: int
    target_y: int
    click_x: int
    click_y: int