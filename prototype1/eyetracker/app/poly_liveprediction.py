#!/usr/bin/env python3
"""
poly_liveprediction.py
~~~~~~~~~~~~~~~~~~~~~~

Run a live gaze prediction demo using polynomial ridge regression.

This script mirrors the behaviour of the ``liveprediction.py`` that uses
random Fourier features, but instead employs the polynomial models
defined in ``eyetracker.learners.polynomial``.  It can run in one of
three modes: single–eye (left or right), binocular without head
compensation, and binocular with head compensation.  During a run the
user sees a scrolling block of text covering the screen with a heat map
overlay of the predicted gaze positions.  Optional calibration targets
can be displayed and, by holding the space bar, the model can be
recalibrated on the fly.

Key bindings during the demo:

  * 1: switch to single‑eye (left) mode
  * 2: switch to binocular mode
  * 3: switch to binocular + head mode
  * t: toggle display of calibration target
  * TAB: cycle through calibration targets
  * SPACE (hold): recalibrate at current target
  * r: reset the heat map
  * s: save a screenshot
  * q or ESC: quit the application

Example usage:

.. code-block:: bash

    # Use the most recent calibration in ./data/sessions
    python3 -m eyetracker.app.poly_liveprediction \
      --sessions-dir ./data/sessions --mode 2eyes_head --degree 2

    # Explicit calibration path with normalised targets
    python3 -m eyetracker.app.poly_liveprediction \
      --parquet ./data/session_001/frames.parquet --mode 2eyes \
      --degree 3 --targets-space norm

Dependencies: numpy, pandas, pygame, and optionally opencv-python for
coloured heat maps.  See the README for installation instructions.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pygame  # type: ignore
except Exception:
    pygame = None  # pygame may not be installed; handled later

try:
    import yaml  # type: ignore  # optional for reading config_snapshot.yaml
    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False

try:
    import cv2  # type: ignore  # optional for coloured heat maps
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

from eyetracker.learners.polynomial import (
    FeatureSpec,
    PolyRegressorManager,
)

# -----------------------------------------------------------------------------
# Helper classes and functions


class HeatmapCanvas:
    """
    Accumulate gaze points into a decaying heat map and render to a
    pygame surface.  Downsampling and a Gaussian kernel are used for
    efficiency.  If OpenCV is available the jet colormap is applied,
    otherwise a simple red–blue gradient is used.
    """
    def __init__(
        self,
        width: int,
        height: int,
        downsample: int = 2,
        sigma_px: float = 60.0,
        decay: float = 0.97,
        vmax_auto: bool = True,
    ):
        self.W = int(width)
        self.H = int(height)
        self.ds = max(1, int(downsample))
        self.w = self.W // self.ds
        self.h = self.H // self.ds
        self.decay = float(decay)
        self.grid = np.zeros((self.h, self.w), dtype=np.float32)
        # Create a square Gaussian kernel with odd dimension
        ksize = int(max(3, 6 * sigma_px / self.ds)) | 1
        self._kernel = self._make_kernel(ksize, sigma_px / self.ds)
        self._vmax = 1.0
        self._vmax_auto = vmax_auto

    def _make_kernel(self, ksize: int, sigma: float) -> np.ndarray:
        ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        ker = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        ker /= (ker.sum() + 1e-9)
        return ker

    def reset(self) -> None:
        """Clear the heat map."""
        self.grid.fill(0.0)

    def add_point(self, x_px: float, y_px: float, strength: float = 1.0) -> None:
        """Add a gaze point with optional strength (weight)."""
        ix = int(x_px / self.ds)
        iy = int(y_px / self.ds)
        if ix < 0 or iy < 0 or ix >= self.w or iy >= self.h:
            return
        kh, kw = self._kernel.shape
        rx = kw // 2
        ry = kh // 2
        x0 = max(0, ix - rx)
        x1 = min(self.w, ix + rx + 1)
        y0 = max(0, iy - ry)
        y1 = min(self.h, iy + ry + 1)
        kx0 = rx - (ix - x0)
        kx1 = rx + (x1 - ix)
        ky0 = ry - (iy - y0)
        ky1 = ry + (y1 - iy)
        # Apply decay and add kernel around the point
        self.grid[y0:y1, x0:x1] *= self.decay
        self.grid[y0:y1, x0:x1] += strength * self._kernel[ky0:ky1, kx0:kx1]
        if self._vmax_auto:
            self._vmax = max(self._vmax * 0.999, float(self.grid.max()))

    def to_surface(self) -> pygame.Surface:
        """Render the heat map to a pygame surface with alpha transparency."""
        g = self.grid / (self._vmax + 1e-9)
        g = np.clip(g, 0.0, 1.0)
        img = (g * 255.0).astype(np.uint8)
        if _HAS_CV2:
            cm = cv2.applyColorMap(img, cv2.COLORMAP_JET)[:, :, ::-1]
        else:
            # Simple gradient: red for high density, blue for low
            r = img
            gch = (255 - img) // 2
            b = 255 - img
            cm = np.stack([r, gch, b], axis=-1).astype(np.uint8)
        cm_up = np.repeat(np.repeat(cm, self.ds, axis=0), self.ds, axis=1)
        surf = pygame.image.frombuffer(cm_up.tobytes(), (self.W, self.H), "RGB")
        return surf.convert()


def render_wrapped_text(
    text: str,
    font: pygame.font.Font,
    max_w: int,
    color: Tuple[int, int, int] = (230, 230, 230),
) -> pygame.Surface:
    """
    Render multi‑line text to a surface, wrapping at ``max_w`` pixels.
    A margin of 20 pixels is applied on the left.
    """
    words = text.split()
    lines: List[str] = []
    current = ""
    for w in words:
        test = (current + " " + w).strip()
        if font.size(test)[0] <= max_w - 40:
            current = test
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    line_h = font.get_linesize()
    surf_h = line_h * (len(lines) + 3)
    surf = pygame.Surface((max_w, surf_h), pygame.SRCALPHA)
    y = 20
    for ln in lines:
        txt = font.render(ln, True, color)
        surf.blit(txt, (20, y))
        y += line_h
    return surf.convert_alpha()


def make_grid_targets(
    W: int,
    H: int,
    cols: int = 5,
    rows: int = 3,
    margin: float = 0.1,
) -> List[Tuple[int, int]]:
    """
    Create a list of evenly spaced calibration points on the screen.  The
    margin parameter reserves a border around the edges of the screen.
    """
    mx = int(W * margin)
    my = int(H * margin)
    xs = np.linspace(mx, W - mx, cols)
    ys = np.linspace(my, H - my, rows)
    pts: List[Tuple[int, int]] = []
    for y in ys:
        for x in xs:
            pts.append((int(x), int(y)))
    return pts


def find_sessions(parquet_dir: str) -> List[str]:
    """
    Locate candidate session directories or parquet files within a
    directory.  Returns a sorted list (most recent first) of .parquet
    files.
    """
    pats = ["*.parquet", "**/*.parquet"]
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(os.path.join(parquet_dir, p), recursive=True))
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return files


def load_screen_from_config(
    session_path: str,
    default_size: Tuple[int, int] = (1280, 720),
) -> Tuple[int, int]:
    """
    Attempt to read ``config_snapshot.yaml`` adjacent to a session to
    determine the screen width and height.  If not found or YAML
    parsing fails returns ``default_size``.
    """
    base = os.path.dirname(session_path) if os.path.isfile(session_path) else session_path
    candidates = [
        os.path.join(base, "config_snapshot.yaml"),
        os.path.join(base, "..", "config_snapshot.yaml"),
        os.path.join(base, "session", "config_snapshot.yaml"),
    ]
    for c in candidates:
        if os.path.isfile(c) and _HAS_YAML:
            try:
                with open(c, "r") as f:
                    cfg = yaml.safe_load(f)
                w = int(cfg.get("screen", {}).get("width", default_size[0]))
                h = int(cfg.get("screen", {}).get("height", default_size[1]))
                return (w, h)
            except Exception:
                continue
    return default_size


def auto_target_space(Y: np.ndarray, screen_size: Tuple[int, int]) -> str:
    """
    Heuristically determine whether target coordinates are pixel or
    normalised units.  If values exceed typical normalised ranges the
    'px' space is returned, otherwise 'norm'.
    """
    W, H = screen_size
    if np.nanmax(Y[:, 0]) > 1.2 or np.nanmax(Y[:, 1]) > 1.2:
        return "px"
    return "norm"


def to_px(
    xy: np.ndarray,
    screen_size: Tuple[int, int],
    target_space: str,
) -> np.ndarray:
    """Convert (x,y) from normalised space to pixels if necessary."""
    W, H = screen_size
    if target_space == "px":
        return xy
    return np.stack([xy[:, 0] * W, xy[:, 1] * H], axis=1)


def clamp_xy(x: float, y: float, W: int, H: int) -> Tuple[int, int]:
    """Clamp a coordinate pair to the screen dimensions."""
    return max(0, min(W - 1, int(x))), max(0, min(H - 1, int(y)))


def _load_session_features(session_path: str) -> pd.DataFrame:
    """
    Load a recorded session from ``frames.parquet`` and
    ``events.parquet`` into a DataFrame of features plus targets.

    The DataFrame columns include:
      left_u,left_v,right_u,right_v,left_conf,right_conf,
      yaw_deg,pitch_deg,roll_deg,head_x_mm,head_y_mm,head_z_mm,
      ipd_norm,target_x,target_y
    """
    # Accept either frames.parquet path or session directory
    if session_path.endswith(".parquet"):
        frame_pq = session_path
        base = os.path.dirname(session_path)
        event_pq = os.path.join(base, "events.parquet")
    else:
        base = session_path
        frame_pq = os.path.join(base, "frames.parquet")
        event_pq = os.path.join(base, "events.parquet")
    frames = pd.read_parquet(frame_pq).sort_values("t_mono").reset_index(drop=True)
    events = pd.read_parquet(event_pq).sort_values("t_mono")
    # Filter out bad frames
    good = (frames.get("face_present", True) == True) & (frames.get("blink", False) == False)
    frames = frames.loc[good].copy()
    # Forward fill target_x and target_y from events using merge_asof
    frames = pd.merge_asof(
        frames,
        events[["t_mono", "target_x", "target_y"]],
        on="t_mono",
        direction="backward",
    )
    frames = frames.dropna(subset=["target_x", "target_y"])
    # Compute feature columns expected by the polynomial models
    frames["left_u"] = frames["left_yaw"]
    frames["left_v"] = frames["left_pitch"]
    frames["right_u"] = frames["right_yaw"]
    frames["right_v"] = frames["right_pitch"]
    # Rename head angles to match FeatureSpec
    frames["yaw_deg"] = frames.get("head_yaw_deg", 0.0)
    frames["pitch_deg"] = frames.get("head_pitch_deg", 0.0)
    frames["roll_deg"] = frames.get("head_roll_deg", 0.0)
    # Head position (mm)
    if "head_x_mm" not in frames.columns:
        frames["head_x_mm"] = 0.0
    if "head_y_mm" not in frames.columns:
        frames["head_y_mm"] = 0.0
    if "head_z_mm" in frames.columns:
        pass
    elif "head_dist_mm" in frames.columns:
        frames["head_z_mm"] = frames["head_dist_mm"]
    else:
        frames["head_z_mm"] = 600.0
    # Approximate inter‑pupil distance in normalised space
    if {"r_inner_xc", "l_outer_xc"}.issubset(frames.columns):
        frames["ipd_norm"] = frames["r_inner_xc"] - frames["l_outer_xc"]
    else:
        frames["ipd_norm"] = 0.065
    frames["left_conf"] = 1.0
    frames["right_conf"] = 1.0
    cols = [
        "left_u",
        "left_v",
        "right_u",
        "right_v",
        "left_conf",
        "right_conf",
        "yaw_deg",
        "pitch_deg",
        "roll_deg",
        "head_x_mm",
        "head_y_mm",
        "head_z_mm",
        "ipd_norm",
        "target_x",
        "target_y",
    ]
    return frames[cols].reset_index(drop=True)


class ParquetFeatureSource:
    """
    Replay features from a recorded session.  Iterating through this
    source yields dictionaries matching the keys expected by
    ``PolyRegressorManager.predict``.  The target_x/target_y values are
    also included for convenience but not used by prediction.
    """
    def __init__(self, session_path: str):
        self.df = _load_session_features(session_path)
        self.idx = 0
        self.n = len(self.df)

    def get_frame_features(self) -> Dict[str, Any]:
        """Return the next frame's features as a dict."""
        if self.n == 0:
            return {}
        row = self.df.iloc[self.idx]
        self.idx = (self.idx + 1) % self.n
        return {
            "left_u": float(row.left_u),
            "left_v": float(row.left_v),
            "right_u": float(row.right_u),
            "right_v": float(row.right_v),
            "left_conf": float(row.left_conf),
            "right_conf": float(row.right_conf),
            "yaw_deg": float(row.yaw_deg),
            "pitch_deg": float(row.pitch_deg),
            "roll_deg": float(row.roll_deg),
            "head_x_mm": float(row.head_x_mm),
            "head_y_mm": float(row.head_y_mm),
            "head_z_mm": float(row.head_z_mm),
            "ipd_norm": float(row.ipd_norm),
            # pass through targets for convenience (unused by model)
            "target_x": float(row.target_x),
            "target_y": float(row.target_y),
        }


class MouseDemoFeatures:
    """
    Fallback feature source using the mouse position as a proxy for gaze.
    Produces plausible binocular features; head pose remains neutral.
    """
    def __init__(self, screen_size: Tuple[int, int], ipd_norm: float = 0.065):
        self.W, self.H = screen_size
        self.ipd_norm = ipd_norm

    def get_frame_features(self) -> Dict[str, Any]:
        if pygame is None:
            # If pygame is unavailable just return centred gaze
            u = v = 0.0
            disp = 0.03
        else:
            x, y = pygame.mouse.get_pos()
            u = (x / max(1, self.W)) * 2.0 - 1.0
            v = (y / max(1, self.H)) * 2.0 - 1.0
            disp = 0.06 * math.sin(pygame.time.get_ticks() * 0.002)
        return {
            "left_u": u - disp,
            "left_v": v,
            "right_u": u + disp,
            "right_v": v,
            "left_conf": 0.9,
            "right_conf": 0.9,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "head_x_mm": 0.0,
            "head_y_mm": 0.0,
            "head_z_mm": 600.0,
            "ipd_norm": self.ipd_norm,
        }


def build_live_feature_source(
    screen_size: Tuple[int, int],
    replay_session: Optional[str] = None,
) -> Any:
    """
    Return a feature source appropriate for live prediction.  If
    ``replay_session`` is provided the returned object will stream
    features from the recorded session.  Otherwise the fallback mouse
    demo is used.
    """
    if replay_session is not None and os.path.exists(replay_session):
        return ParquetFeatureSource(replay_session)
    return MouseDemoFeatures(screen_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live gaze prediction demo using polynomial ridge models"
    )
    parser.add_argument(
        "--sessions-dir",
        type=str,
        default=".",
        help="Directory containing calibration parquet(s)",
    )
    parser.add_argument(
        "--session-idx",
        type=int,
        default=0,
        help="Pick nth most recent parquet (0=latest)",
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="",
        help="Explicit frames.parquet (or session folder) to use for calibration",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="2eyes_head",
        choices=["1eye-left", "1eye-right", "2eyes", "2eyes_head"],
        help="Model mode to use",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Polynomial degree (1, 2, or 3)",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Ridge regularisation strength",
    )
    parser.add_argument(
        "--targets-space",
        type=str,
        default="auto",
        choices=["auto", "px", "norm"],
        help="Interpretation of target coords: auto, px, or norm",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Run in a window instead of fullscreen",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=("The quick brown fox jumps over the lazy dog. " * 200).strip(),
        help="Scrolling text to display",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=36,
        help="Font size for the scrolling text",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target frames per second",
    )
    args = parser.parse_args()

    # Determine calibration file
    if args.parquet and os.path.exists(args.parquet):
        parq_path = args.parquet
    else:
        files = find_sessions(args.sessions_dir)
        if not files:
            print(f"[!] No .parquet files found in {args.sessions_dir}")
            sys.exit(1)
        idx = max(0, min(len(files) - 1, args.session_idx))
        parq_path = files[idx]
    print(f"[i] Using calibration: {parq_path}")

    # Determine screen size
    screen_size = load_screen_from_config(parq_path, default_size=(1280, 720))
    W, H = screen_size
    print(f"[i] Screen size: {W}x{H}")

    # Input dimensions for each mode (matching FeatureBuilder)
    input_dims = {
        "1eye-left": 3,
        "1eye-right": 3,
        "2eyes": 8,
        "2eyes_head": 15,
    }
    # Create polynomial model manager
    spec = FeatureSpec()
    mgr = PolyRegressorManager(
        input_dims,
        degree=args.degree,
        ridge=args.ridge,
        feature_spec=spec,
    )

    # Fit the appropriate model(s)
    if args.mode in ("1eye-left", "1eye-right"):
        which = "left" if args.mode.endswith("left") else "right"
        mgr.fit_from_parquet(parq_path, mode="1eye", which_eye=which)
    elif args.mode == "2eyes":
        mgr.fit_from_parquet(parq_path, mode="2eyes")
    elif args.mode == "2eyes_head":
        mgr.fit_from_parquet(parq_path, mode="2eyes_head")
    else:
        raise ValueError(f"Unknown mode '{args.mode}'")

    # Determine target coordinate space automatically if requested
    target_space = args.targets_space
    if target_space == "auto":
        # Read targets from calibration file to decide
        try:
            df = pd.read_parquet(parq_path)
            Y = df[[spec.target_xy[0], spec.target_xy[1]]].to_numpy(dtype=np.float64)
            target_space = auto_target_space(Y, screen_size)
        except Exception:
            target_space = "norm"
        print(f"[i] Target space auto-detected: {target_space}")

    # Build live feature source; use recorded session for replay if desired
    # The default is a mouse-driven demo.  To replay the calibration session
    # itself, pass replay_session=os.path.dirname(parq_path).
    live_src = build_live_feature_source(screen_size)

    # Initialise pygame
    if pygame is None:
        print("[!] pygame is not installed; cannot run live demo", file=sys.stderr)
        sys.exit(1)
    pygame.init()
    flags = pygame.FULLSCREEN if not args.windowed else 0
    screen = pygame.display.set_mode((W, H), flags)
    pygame.display.set_caption("Poly Live Gaze Prediction")

    try:
        pygame.font.init()
        font = pygame.font.SysFont("Arial", args.font_size)
    except Exception:
        font = pygame.font.Font(None, args.font_size)

    clock = pygame.time.Clock()
    # Configure heat map parameters relative to screen size
    sigma_px = max(40, min(W, H) // 15)
    heat = HeatmapCanvas(W, H, downsample=2, sigma_px=sigma_px, decay=0.975)

    # Render the scrolling text into a surface
    text_surf = render_wrapped_text(args.text, font, W)
    scroll_y = 0

    # Calibration grid points and target state
    grid_pts = make_grid_targets(W, H, cols=5, rows=3, margin=0.12)
    grid_idx = 0
    show_target = False

    current_mode = args.mode
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_r:
                    heat.reset()
                elif ev.key == pygame.K_s:
                    # Save a screenshot
                    filename = "poly_live_heatmap_screenshot.png"
                    pygame.image.save(screen, filename)
                    print(f"[i] Saved screenshot to {filename}")
                elif ev.key == pygame.K_t:
                    show_target = not show_target
                elif ev.key == pygame.K_TAB:
                    grid_idx = (grid_idx + 1) % len(grid_pts)
                elif ev.key == pygame.K_1:
                    current_mode = "1eye-left"
                    mgr.fit_from_parquet(parq_path, mode="1eye", which_eye="left")
                    print("[i] Switched to 1eye-left mode")
                elif ev.key == pygame.K_2:
                    current_mode = "2eyes"
                    mgr.fit_from_parquet(parq_path, mode="2eyes")
                    print("[i] Switched to 2eyes mode")
                elif ev.key == pygame.K_3:
                    current_mode = "2eyes_head"
                    mgr.fit_from_parquet(parq_path, mode="2eyes_head")
                    print("[i] Switched to 2eyes_head mode")

        # Acquire features for the current frame
        frame_rec = live_src.get_frame_features()

        # Predict gaze coordinate using the current mode
        try:
            xy_pred, info = mgr.predict(frame_rec, mode=current_mode)
        except Exception as e:
            # In case of prediction failure, skip this frame
            print(f"[!] Prediction error: {e}", file=sys.stderr)
            continue

        xy_px = to_px(xy_pred.reshape(1, 2), screen_size, target_space)[0]
        px, py = clamp_xy(float(xy_px[0]), float(xy_px[1]), W, H)

        # Add gaze point to heatmap
        heat.add_point(px, py, strength=1.0)

        # Clear screen and draw background
        screen.fill((15, 15, 18))
        # Draw scrolling text
        screen.blit(text_surf, (20, -scroll_y))
        scroll_y = (scroll_y + 1) % max(1, text_surf.get_height() - H + 40)

        # Draw calibration target if enabled
        if show_target:
            tx, ty = grid_pts[grid_idx]
            pygame.draw.circle(screen, (255, 255, 255), (tx, ty), 12, width=2)
            pygame.draw.circle(screen, (255, 0, 0), (tx, ty), 4, width=0)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                # Recalibrate towards known target; convert to model units
                if target_space == "norm":
                    tx_norm = tx / W
                    ty_norm = ty / H
                else:
                    tx_norm = tx
                    ty_norm = ty
                mgr.recalibrate_event(
                    frame_rec,
                    known_xy=(tx_norm, ty_norm),
                    mode=current_mode,
                )

        # Draw heatmap overlay with partial transparency
        hm = heat.to_surface()
        hm.set_alpha(155)
        screen.blit(hm, (0, 0))

        # Draw the current predicted point
        pygame.draw.circle(screen, (0, 255, 0), (px, py), 4, width=0)

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()