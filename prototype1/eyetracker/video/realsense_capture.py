# video/realsense_capture.py
import time, threading
from typing import Tuple, Optional, Dict
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    rs = None

class RealSenseCaptureThread:
    """
    Returns (t_mono, color_bgr, depth_frame_as_np, intrinsics_dict)
    depth array is float32 meters aligned to color.
    """
    def __init__(self, width=1280, height=720, fps=30):
        if rs is None:
            raise RuntimeError("pyrealsense2 not installed")
        self.width, self.height, self.fps = width, height, fps
        self.pipe = rs.pipeline()
        self.cfg  = rs.config()
        # RGB + depth (use 848x480 depth for speed if you want)
        self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)
        # Align depth->color
        self.align = rs.align(rs.stream.color)

        self.depth_scale = None
        self.color_intr = None
        self.color_dist = None

        self.lock = threading.Lock()
        self.latest = (None, None, None, None)  # (t, color, depth, intrinsics)
        self.running = False
        self.thread = None

    def start(self):
        prof = self.pipe.start(self.cfg)
        # intrinsics from SDK
        prof_color = prof.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intr = prof_color.get_intrinsics()  # .fx, .fy, .ppx, .ppy
        color_K = np.array([[self.color_intr.fx, 0, self.color_intr.ppx],
                            [0, self.color_intr.fy, self.color_intr.ppy],
                            [0, 0, 1]], dtype=np.float32)
        # distortion params as 5-vector where possible
        d = self.color_intr.coeffs
        color_dist = np.zeros((5,), np.float32)
        for i in range(min(5, len(d))): color_dist[i] = d[i]
        self.color_dist = color_dist

        self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()  # meters per unit

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while self.running:
            fs = self.pipe.wait_for_frames()
            fs = self.align.process(fs)
            color = fs.get_color_frame()
            depth = fs.get_depth_frame()
            if not color: continue
            # convert
            color_np = np.asanyarray(color.get_data())
            if depth:
                # depth in meters as float32 aligned to color resolution
                h, w = color_np.shape[:2]
                dep_m = np.zeros((h, w), np.float32)
                dep_raw = np.asanyarray(depth.get_data())  # z16; aligned dims may match color
                # If shape differs (e.g., 848x480 depth → 1280x720 color), let SDK align handle resize.
                dep_m = dep_raw.astype(np.float32) * self.depth_scale
            else:
                dep_m = None
            intr_bundle = {
                "color_K": np.array([[self.color_intr.fx, 0, self.color_intr.ppx],
                                     [0, self.color_intr.fy, self.color_intr.ppy],
                                     [0, 0, 1]], dtype=np.float32),
                "color_dist": self.color_dist.copy()
            }
            with self.lock:
                self.latest = (time.monotonic(), color_np, dep_m, intr_bundle)

    def get_latest(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.running = False
        try:
            if self.thread is not None:
                self.thread.join(timeout=0.5)
        except: pass
        try:
            self.pipe.stop()
        except: pass
