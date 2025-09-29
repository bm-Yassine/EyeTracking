# Heatmap helper (NumPy + pygame; OpenCV optional for colormap)
from __future__ import annotations
import numpy as np

try:
    import cv2  # optional
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

import pygame

class HeatmapCanvas:
    def __init__(self, width: int, height: int, downsample: int = 2,
                 sigma_px: float = 60.0, decay: float = 0.97, vmax_auto: bool = True):
        self.W = int(width)
        self.H = int(height)
        self.ds = max(1, int(downsample))
        self.w = self.W // self.ds
        self.h = self.H // self.ds
        self.decay = float(decay)
        self.grid = np.zeros((self.h, self.w), dtype=np.float32)
        self._kernel = self._make_kernel(int(6*sigma_px/self.ds+1)|1, sigma_px/self.ds)
        self._vmax = 1.0
        self._vmax_auto = vmax_auto

    def _make_kernel(self, ksize: int, sigma: float) -> np.ndarray:
        ax = np.arange(-(ksize//2), ksize//2+1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        ker = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        ker /= (ker.sum() + 1e-12)
        return ker

    def reset(self) -> None:
        self.grid.fill(0.0)

    def add_point(self, x_px: float, y_px: float, strength: float = 1.0) -> None:
        ix = int(x_px / self.ds)
        iy = int(y_px / self.ds)
        if ix < 0 or iy < 0 or ix >= self.w or iy >= self.h:
            return
        kh, kw = self._kernel.shape
        rx = kw // 2
        ry = kh // 2

        x0 = max(0, ix - rx); x1 = min(self.w, ix + rx + 1)
        y0 = max(0, iy - ry); y1 = min(self.h, iy + ry + 1)

        kx0 = rx - (ix - x0); kx1 = rx + (x1 - ix)
        ky0 = ry - (iy - y0); ky1 = ry + (y1 - iy)

        self.grid[y0:y1, x0:x1] *= self.decay
        self.grid[y0:y1, x0:x1] += strength * self._kernel[ky0:ky1, kx0:kx1]
        if self._vmax_auto:
            self._vmax = max(self._vmax*0.999, float(self.grid.max()))

    def to_surface(self) -> pygame.Surface:
        g = self.grid / (self._vmax + 1e-9)
        g = np.clip(g, 0.0, 1.0)
        img = (g * 255.0).astype(np.uint8)

        if _HAS_CV2:
            cm = cv2.applyColorMap(img, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
        else:
            # simple blue→red gradient fallback
            r = img
            gch = (255 - img) // 2
            b = 255 - img
            cm = np.stack([r, gch, b], axis=-1).astype(np.uint8)

        cm_up = np.repeat(np.repeat(cm, self.ds, axis=0), self.ds, axis=1)
        surf = pygame.image.frombuffer(cm_up.tobytes(), (self.W, self.H), "RGB")
        return surf.convert()
