#!/usr/bin/env python3
from __future__ import annotations
import os, sys, glob, json, math, argparse
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pygame

# Optional YAML (for reading config_snapshot.yaml)
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

from eyetracker.learners.rff_rls import (
    RFFConfig, RLSConfig, FeatureSpec, GazeRegressorManager
)
from eyetracker.app.heatmap import HeatmapCanvas
from eyetracker.app.features_adapter import build_live_feature_source


# ---------- Utilities

def find_sessions(parquet_dir: str) -> List[str]:
    pats = ["*.parquet", "**/*.parquet"]
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(os.path.join(parquet_dir, p), recursive=True))
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return files

def load_screen_from_config(session_path: str, default_size=(1280, 720)) -> Tuple[int,int]:
    # search for a sibling/nearby config_snapshot.yaml
    base = os.path.dirname(session_path) if os.path.isfile(session_path) else session_path
    cand = [
        os.path.join(base, "config_snapshot.yaml"),
        os.path.join(base, "..", "config_snapshot.yaml"),
        os.path.join(base, "session", "config_snapshot.yaml"),
    ]
    for c in cand:
        if os.path.isfile(c) and _HAS_YAML:
            try:
                with open(c, "r") as f:
                    cfg = yaml.safe_load(f)
                w = int(cfg.get("screen", {}).get("width", default_size[0]))
                h = int(cfg.get("screen", {}).get("height", default_size[1]))
                return (w, h)
            except Exception:
                pass
    return default_size

def auto_target_space(Y: np.ndarray, screen_size: Tuple[int,int]) -> str:
    # Heuristic: if values exceed 1.5x screen dims, assume normalized; else if <=1.2, assume normalized.
    W, H = screen_size
    ymax = float(np.max(np.abs(Y)))
    # If targets look like pixels (bigger than 1.2 and similar to screen dims), treat as px.
    if np.nanmax(Y[:,0]) > 1.2 or np.nanmax(Y[:,1]) > 1.2:
        return "px"
    return "norm"

def to_px(xy: np.ndarray, screen_size: Tuple[int,int], target_space: str) -> np.ndarray:
    W, H = screen_size
    if target_space == "px":
        return xy
    # normalized [0,1]
    return np.stack([xy[:,0] * W, xy[:,1] * H], axis=1)

def clamp_xy(x: float, y: float, W: int, H: int) -> Tuple[int,int]:
    return max(0, min(W-1, int(x))), max(0, min(H-1, int(y)))

# ---------- Main App

def main():
    ap = argparse.ArgumentParser("Live gaze prediction + heatmap (RFF-RLS)")
    ap.add_argument("--sessions-dir", type=str, default=".", help="Directory containing calibration parquet(s)")
    ap.add_argument("--session-idx", type=int, default=0, help="Pick nth most-recent parquet (0=latest)")
    ap.add_argument("--parquet", type=str, default="", help="Explicit parquet file (overrides --sessions-dir)")
    ap.add_argument("--mode", type=str, default="2eyes_head",
                    choices=["1eye-left","1eye-right","2eyes","2eyes_head"])
    ap.add_argument("--targets-space", type=str, default="auto", choices=["auto","px","norm"])
    ap.add_argument("--rff-dim", type=int, default=384)
    ap.add_argument("--rff-len", type=float, default=0.8)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--forget", type=float, default=1.0)
    ap.add_argument("--huber", type=float, default=0.02)
    ap.add_argument("--irls", type=int, default=2)
    ap.add_argument("--windowed", action="store_true", help="Run windowed instead of fullscreen")
    ap.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog. " * 200)
    ap.add_argument("--font-size", type=int, default=36)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    # -------- Locate session/parquet
    if args.parquet and os.path.isfile(args.parquet):
        parq_path = args.parquet
    else:
        files = find_sessions(args.sessions_dir)
        if not files:
            print(f"[!] No parquet found under: {args.sessions_dir}")
            sys.exit(1)
        idx = max(0, min(len(files)-1, args.session_idx))
        parq_path = files[idx]
    print(f"[i] Using calibration: {parq_path}")

    # -------- Read screen size from config_snapshot.yaml if available
    screen_size = load_screen_from_config(parq_path, default_size=(1280, 720))
    W, H = screen_size
    print(f"[i] Screen: {W}x{H}")

    # -------- Init regressor manager
    from eyetracker.learners.rff_rls import FeatureSpec
    input_dims = {
        "1eye-left": 3,
        "1eye-right": 3,
        "2eyes": 8,
        "2eyes_head": 15,
    }
    rff_cfg = RFFConfig(dim=args.rff_dim, lengthscale=args.rff_len, seed=0)
    rls_cfg = RLSConfig(ridge=args.ridge, forgetting=args.forget, huber_delta=args.huber, irls_iters=args.irls)
    spec = FeatureSpec()  # adjust here if your parquet columns differ
    from eyetracker.learners.rff_rls import GazeRegressorManager
    mgr = GazeRegressorManager(input_dims, rff_cfg, rls_cfg, feature_spec=spec)

    # Fit all necessary models for chosen mode
    if args.mode in ("1eye-left","1eye-right"):
        which = "left" if args.mode.endswith("left") else "right"
        mgr.fit_from_parquet(parq_path, mode="1eye", which_eye=which)
    elif args.mode == "2eyes":
        # fit both eye models
        mgr.fit_from_parquet(parq_path, mode="1eye", which_eye="left")
        mgr.fit_from_parquet(parq_path, mode="1eye", which_eye="right")
    elif args.mode == "2eyes_head":
        mgr.fit_from_parquet(parq_path, mode="2eyes_head")
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    # If auto, peek targets to decide norm vs px
    target_space = args.targets_space
    if target_space == "auto":
        import pandas as pd
        df = pd.read_parquet(parq_path)
        Y = df[[spec.target_xy[0], spec.target_xy[1]]].to_numpy(dtype=np.float64)
        target_space = auto_target_space(Y, (W,H))
        print(f"[i] Target space auto-detected: {target_space}")

    # -------- Build live feature source
    live_src = build_live_feature_source((W,H))

    # -------- Pygame setup
    pygame.init()
    flags = pygame.FULLSCREEN if not args.windowed else 0
    screen = pygame.display.set_mode((W, H), flags)
    pygame.display.set_caption("Live Gaze + Heatmap")

    try:
        pygame.font.init()
        font = pygame.font.SysFont("Arial", args.font_size)
    except Exception:
        font = pygame.font.Font(None, args.font_size)

    clock = pygame.time.Clock()
    heat = HeatmapCanvas(W, H, downsample=2, sigma_px=max(40, min(W,H)//20), decay=0.975)

    # Prepare scrolling text surface (simple wrap)
    text_surf = render_wrapped_text(args.text, font, W, color=(230,230,230))
    scroll_y = 0

    # Calibration targets (grid)
    grid_pts = make_grid_targets(W, H, cols=5, rows=3, margin=0.12)
    grid_idx = 0
    show_target = False

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
                elif ev.key == pygame.K_1:
                    args.mode = "1eye-left"
                    mgr.fit_from_parquet(parq_path, mode="1eye", which_eye="left")
                    print("[i] Switched to 1eye-left")
                elif ev.key == pygame.K_2:
                    args.mode = "2eyes"
                    mgr.fit_from_parquet(parq_path, mode="1eye", which_eye="left")
                    mgr.fit_from_parquet(parq_path, mode="1eye", which_eye="right")
                    print("[i] Switched to 2eyes")
                elif ev.key == pygame.K_3:
                    args.mode = "2eyes_head"
                    mgr.fit_from_parquet(parq_path, mode="2eyes_head")
                    print("[i] Switched to 2eyes_head")
                elif ev.key == pygame.K_t:
                    show_target = not show_target
                elif ev.key == pygame.K_TAB:
                    grid_idx = (grid_idx + 1) % len(grid_pts)
                elif ev.key == pygame.K_s:
                    pygame.image.save(screen, "live_heatmap_screenshot.png")
                    print("[i] Saved screenshot to live_heatmap_screenshot.png")

        # Get live features
        frame_rec = live_src.get_frame_features()

        # Predict
        xy_pred, _info = mgr.predict(frame_rec, mode=args.mode)
        xy_px = to_px(xy_pred.reshape(1,2), (W,H), target_space)[0]
        px, py = clamp_xy(float(xy_px[0]), float(xy_px[1]), W, H)

        # Add to heatmap
        heat.add_point(px, py, strength=1.0)

        # Background (dark)
        screen.fill((15, 15, 18))

        # Draw scrolling text
        screen.blit(text_surf, (20, -scroll_y))
        scroll_y = (scroll_y + 1) % max(1, text_surf.get_height() - H + 40)

        # Optional calibration target
        if show_target:
            tx, ty = grid_pts[grid_idx]
            pygame.draw.circle(screen, (255, 255, 255), (tx, ty), 12, width=2)
            pygame.draw.circle(screen, (255, 0, 0), (tx, ty), 4, width=0)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                # mid-run robust recalibration towards known target
                tx_norm = tx / W if target_space == "norm" else tx
                ty_norm = ty / H if target_space == "norm" else ty
                mgr.recalibrate_event(frame_rec, (tx_norm, ty_norm), mode=args.mode, robust=True, base_weight=1.0)

        # Draw heatmap overlay (alpha blend)
        hm = heat.to_surface()
        hm.set_alpha(155)
        screen.blit(hm, (0, 0))

        # Draw current prediction marker
        pygame.draw.circle(screen, (0, 255, 0), (px, py), 4, width=0)

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()

# ---------- Helpers: text and targets

def render_wrapped_text(text: str, font: pygame.font.Font, max_w: int, color=(240,240,240)) -> pygame.Surface:
    words = text.split()
    lines: List[str] = []
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if font.size(test)[0] <= max_w - 40:
            line = test
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    # render
    line_h = font.get_linesize()
    surf = pygame.Surface((max_w, line_h * (len(lines) + 3)), pygame.SRCALPHA)
    y = 20
    for ln in lines:
        txt = font.render(ln, True, color)
        surf.blit(txt, (20, y))
        y += line_h
    return surf.convert_alpha()

def make_grid_targets(W: int, H: int, cols: int = 5, rows: int = 3, margin: float = 0.1) -> List[Tuple[int,int]]:
    mx = int(W * margin); my = int(H * margin)
    xs = np.linspace(mx, W - mx, cols)
    ys = np.linspace(my, H - my, rows)
    pts: List[Tuple[int,int]] = []
    for y in ys:
        for x in xs:
            pts.append((int(x), int(y)))
    return pts

if __name__ == "__main__":
    main()
