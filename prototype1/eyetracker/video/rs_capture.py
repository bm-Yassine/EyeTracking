import threading, time
import numpy as np

class RealSenseCapture:
    def __init__(self, width=848, height=480, fps=30, serial=None):
        import pyrealsense2 as rs
        self.rs = rs
        self.width, self.height, self.fps = width, height, fps
        self.serial = serial
        self._pipe = None
        self._align = None
        self._th = None
        self._stop = False
        self._latest = (None, None, None)  # (t_mono, color_bgr, depth_m)
        self._K = None
        self._dist = None
        self._depth_scale = None

    def start(self):
        rs = self.rs
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self._pipe = rs.pipeline()
        prof = self._pipe.start(cfg)

        # Align depth to color
        self._align = rs.align(rs.stream.color)

        # Color intrinsics -> K, dist
        color_stream = prof.get_stream(rs.stream.color).as_video_stream_profile()
        c_intr = color_stream.get_intrinsics()
        K = np.array([[c_intr.fx, 0, c_intr.ppx],
                      [0, c_intr.fy, c_intr.ppy],
                      [0, 0, 1]], dtype=np.float64)
        # RealSense gives Brown-Conrady (k1,k2,p1,p2,k3)
        dist = np.array([c_intr.coeffs[i] for i in range(5)], dtype=np.float64)

        self._K, self._dist = K, dist

        # Depth scale (to meters)
        d_stream = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        d_sens = d_stream.get_device().first_depth_sensor()
        self._depth_scale = float(d_sens.get_depth_scale())

        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def _loop(self):
        rs = self.rs
        while not self._stop:
            try:
                frameset = self._pipe.wait_for_frames(1000)
                frameset = self._align.process(frameset)
                c = frameset.get_color_frame()
                d = frameset.get_depth_frame()
                if not c or not d:
                    continue
                color = np.asanyarray(c.get_data())  # BGR
                depth = np.asanyarray(d.get_data()).astype(np.float32) * self._depth_scale  # meters
                t = time.monotonic()
                self._latest = (t, color, depth)
            except Exception:
                continue

    def get_latest(self):
        return self._latest  # (t, color_bgr, depth_m)

    def color_K(self):   return self._K
    def color_dist(self):return self._dist
    def depth_scale(self): return self._depth_scale

    def stop(self):
        self._stop = True
        if self._th:
            self._th.join(timeout=1.0)
        if self._pipe:
            try: self._pipe.stop()
            except Exception: pass

    # for 'with' compatibility
    def __enter__(self): return self.start()
    def __exit__(self, *args): self.stop()
