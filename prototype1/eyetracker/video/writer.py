from __future__ import annotations
import cv2, os
from pathlib import Path
from typing import Callable, Optional, Tuple

OverlayFn = Callable[[any, dict], None]  # overlay(frame_bgr, info) -> None (in-place)

class VideoWriterMP4:
    """
 OpenCV video writer:
    - Tries multiple codecs/containers in order.
    - Falls back to .avi/MJPG when .mp4 isn't available on the system build.
    - Resizes frames on write if needed.
    - Exposes .actual_path with the final filename used.
    """
    def __init__(self, path: str, size: Tuple[int, int], fps: float = 30.0,
                 fourcc: str | None = None, overlay: Optional[OverlayFn] = None):
        # Ensure dir exists
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

        self.size = size
        self.fps = fps
        self.overlay = overlay
        self._vw: Optional[cv2.VideoWriter] = None
        self.actual_path = None

        # Candidate (fourcc, extension) pairs to try, in order
        # MP4 first, then AVI fallbacks. Add/remove as needed for your system.
        candidates = []
        if fourcc:
            # If user forces a specific fourcc, respect it with the given extension
            ext = (Path(path).suffix or ".mp4")
            candidates.append((fourcc, ext))
        else:
            candidates = [
                ("mp4v", ".mp4"),  # MPEG-4 part 2
                ("avc1", ".mp4"),  # H.264 if available
                ("H264", ".mp4"),
                ("FMP4", ".mp4"),
                ("XVID", ".avi"),
                ("MJPG", ".avi"),
            ]

        stem = str(Path(path).with_suffix(""))  # path without ext

        open_ok = False
        for fcc, ext in candidates:
            trial_path = stem + ext
            writer = cv2.VideoWriter(trial_path, cv2.VideoWriter_fourcc(*fcc), fps, size)
            if writer.isOpened():
                self._vw = writer
                self.actual_path = trial_path
                # Small console hint so users know what succeeded
                print(f"[video] Using {fcc} → {self.actual_path}")
                open_ok = True
                break

        if not open_ok:
            raise RuntimeError(f"Could not open any VideoWriter for: {path} "
                               f"(tried: {', '.join([f'{fcc}{ext}' for fcc,ext in candidates])})")

    def write(self, frame_bgr, info: Optional[dict] = None):
        if frame_bgr is None or self._vw is None:
            return
        if self.overlay is not None:
            self.overlay(frame_bgr, info or {})
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.size:
            frame_bgr = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA)
        self._vw.write(frame_bgr)

    def release(self):
        if self._vw is not None:
            self._vw.release()
            self._vw = None

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.release()
