from __future__ import annotations
import os, time, threading
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

OverlayFn = Callable[[any, dict], None]  # overlay(frame_bgr, info) -> None (in-place)


def _fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)


class VideoWriterMP4:
    """
    OpenCV video writer with:
      - Adaptive FPS: measure real capture FPS from timestamps (first ~1 s or ≥15 frames)
        and open the container with that FPS so playback matches reality.
      - Codec/extension fallbacks (mp4v/avc1 → XVID/MJPG .avi).
      - Early .actual_path so UI never shows "None" (will update if fallback occurs).
      - Background close to avoid AVFoundation warning about finishWriting on main thread (macOS).

    API-compatible with your previous class (overlay, size enforcement, .actual_path).
    """

    def __init__(
        self,
        path: str,
        size: Tuple[int, int],
        fps: float = 30.0,
        fourcc: str | None = None,
        overlay: Optional[OverlayFn] = None,
        *,
        adaptive_fps: bool = True,
        min_seconds_for_estimate: float = 1.0,
        min_frames_for_estimate: int = 15,
        fps_floor: float = 5.0,
        fps_ceil: float = 120.0,
        override_threshold_rel: float = 0.05,  # 5% difference → prefer measured fps
    ):
        # Ensure dir exists
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

        self.size = (int(size[0]), int(size[1]))
        self._fps_hint = float(fps) if fps is not None else None
        self.overlay = overlay
        self._vw: Optional[cv2.VideoWriter] = None

        # Codec candidates (preserve your order/behavior)
        if fourcc:
            ext = (Path(path).suffix or ".mp4")
            self._candidates = [(fourcc, ext)]
        else:
            self._candidates = [
                ("mp4v", ".mp4"),
                ("avc1", ".mp4"),
                ("H264", ".mp4"),
                ("FMP4", ".mp4"),
                ("XVID", ".avi"),
                ("MJPG", ".avi"),
            ]
        self._stem = str(Path(path).with_suffix(""))

        # Give a *planned* path immediately so UI doesn't print None.
        self.actual_path: str = self._stem + self._candidates[0][1]

        # Adaptive-FPS config
        self._adaptive = bool(adaptive_fps)
        self._t_min = float(min_seconds_for_estimate)
        self._n_min = int(min_frames_for_estimate)
        self._fps_floor = float(fps_floor)
        self._fps_ceil = float(fps_ceil)
        self._rel_thresh = float(override_threshold_rel)

        # State
        self._opened = False
        self._pending: List[Tuple[np.ndarray, float]] = []  # (frame_resized, ts)
        self._fps_final: Optional[float] = None
        self.actual_fps: Optional[float] = None

        # If adaptive disabled: open immediately with hint (or default 30)
        if not self._adaptive:
            chosen = self._fps_hint if self._fps_hint is not None else 30.0
            self._open_with_fps(chosen)

    # ---------------- internal helpers ----------------

    def _try_open_cv_writer(self, fps: float) -> bool:
        for fcc, ext in self._candidates:
            trial_path = self._stem + ext
            writer = cv2.VideoWriter(trial_path, _fourcc(fcc), float(fps), self.size)
            if writer.isOpened():
                self._vw = writer
                self.actual_path = trial_path  # update to the *real* path we opened
                print(f"[video] Using {fcc}@{fps:.2f} fps → {self.actual_path}")
                return True
        return False

    def _open_with_fps(self, fps: float) -> None:
        fps = float(np.clip(fps, self._fps_floor, self._fps_ceil))
        if not self._try_open_cv_writer(fps):
            tried = ", ".join([f"{fcc}{ext}" for fcc, ext in self._candidates])
            raise RuntimeError(f"Could not open any VideoWriter (tried: {tried})")
        self._opened = True
        self._fps_final = fps
        self.actual_fps = fps

        # Flush any pending frames (already resized/overlayed)
        if self._pending:
            for fr, _ts in self._pending:
                self._vw.write(fr)
            self._pending.clear()

    def _maybe_open(self) -> None:
        if self._opened:
            return
        if not self._adaptive:
            self._open_with_fps(self._fps_hint if self._fps_hint is not None else 30.0)
            return

        # Need enough frames & span to estimate
        if len(self._pending) < self._n_min:
            return
        t_first = self._pending[0][1]
        t_last = self._pending[-1][1]
        span = max(0.0, t_last - t_first)
        if span < self._t_min:
            return

        n = len(self._pending)
        if span <= 1e-6:
            fps_est = self._fps_hint if self._fps_hint is not None else 30.0
        else:
            fps_est = (n - 1) / span

        # Compare with hint
        if self._fps_hint is not None:
            rel = abs(fps_est - self._fps_hint) / max(1e-6, self._fps_hint)
            fps_final = fps_est if rel >= self._rel_thresh else self._fps_hint
        else:
            fps_final = fps_est

        self._open_with_fps(fps_final)

    # ---------------- public API ----------------

    def write(self, frame_bgr, info: Optional[dict] = None):
        if frame_bgr is None:
            return

        # Timestamp from caller (preferred) or local clock
        ts = None
        if info is not None:
            ts = info.get("t_mono", None)
        if ts is None:
            ts = time.monotonic()

        # Overlay first (in-place)
        if self.overlay is not None:
            self.overlay(frame_bgr, info or {})

        # Resize to target writer size before buffering/writing
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.size:
            frame_bgr = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA)

        if not self._opened:
            # Buffer until we can estimate FPS
            self._pending.append((frame_bgr.copy(), ts))
            self._maybe_open()
        else:
            self._vw.write(frame_bgr)

    def release(self):
        """
        Finalize the file. On macOS/AVFoundation, finishWriting should not be called
        on the main thread. We therefore move the actual release to a background thread.
        """
        # Ensure the writer is opened so pending frames are flushed
        if not self._opened and self._pending:
            if len(self._pending) >= 2:
                t_first = self._pending[0][1]
                t_last = self._pending[-1][1]
                span = max(0.0, t_last - t_first)
                n = len(self._pending)
                if span > 1e-6:
                    fps_est = (n - 1) / span
                else:
                    fps_est = self._fps_hint if self._fps_hint is not None else 30.0
                self._open_with_fps(fps_est)
            else:
                self._open_with_fps(self._fps_hint if self._fps_hint is not None else 30.0)

        vw = self._vw
        self._vw = None
        self._opened = False

        if vw is None:
            return

        def _do_release():
            try:
                vw.release()
            except Exception:
                # Avoid crashing on shutdown due to AVFoundation quirks
                pass

        # Always release off the main thread to satisfy AVFoundation
        if threading.current_thread() is threading.main_thread():
            t = threading.Thread(target=_do_release, daemon=False)
            t.start()
            t.join()  # wait for clean close so file isn't truncated
        else:
            _do_release()

    # Context manager helpers
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.release()
