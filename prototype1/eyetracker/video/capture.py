from __future__ import annotations
import cv2, time, threading
from typing import Optional, Tuple

__all__ = ["CameraCapture", "VideoCaptureThread"]

class CameraCapture:
    """Simple, blocking capture (used by A0 camera calibrator)."""
    def __init__(self, cam_id: int = 0, width: int | None = None, height: int | None = None, fps: int | None = None):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_ANY)
        if width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:    self.cap.set(cv2.CAP_PROP_FPS,          fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_id}")

    def read(self) -> tuple[float, any]:
        ok, frame = self.cap.read()
        ts = time.monotonic()
        if not ok:
            raise RuntimeError("Camera read failed")
        return ts, frame

    def release(self): self.cap.release()
    def __enter__(self): return self
    def __exit__(self, *args): self.release()

    
class VideoCaptureThread:
    """
    Threaded OpenCV capture with monotonic timestamps.
    get_latest() returns (ts_monotonic, frame_bgr) or (None, None) if not ready.
    """
    def __init__(self, cam_id: int = 0, width: int | None = None, height: int | None = None, fps: int | None = None):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_ANY)
        if width: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps: self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_id}")

        self._lock = threading.Lock()
        self._latest: Optional[Tuple[float, any]] = (None, None)  # (ts, frame)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

        # fps tracking
        self._fps = 0.0
        self._tick = time.monotonic()
        self._count = 0

    def start(self) -> "VideoCaptureThread":
        self._stop.clear()
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            t = time.monotonic()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = (t, frame)
            # fps
            self._count += 1
            if t - self._tick >= 1.0:
                self._fps = self._count / (t - self._tick)
                self._count, self._tick = 0, t

    def get_latest(self) -> Tuple[Optional[float], Optional[any]]:
        with self._lock:
            return self._latest

    def get_fps(self) -> float:
        return self._fps

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()


    def __enter__(self): return self.start()
    def __exit__(self, exc_type, exc, tb): self.stop()
