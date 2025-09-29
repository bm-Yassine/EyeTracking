"""
Part B — Trainer & Adapter (with blink/face filtering)
Reads Part A session folder (frames parquet + config_snapshot.yaml),
trains ALL three regressors, runs K-fold cross‑validation, writes a report,
OOF predictions, and saves final models for live use.

Place this file at: eyetracker/regression/train_all.py

Dependencies: numpy, pandas, pyarrow (for parquet), pyyaml.

CLI:
  python -m eyetracker.regression.train_all \
      --data-dir data/session_001/ \
      --kfold 5 --degree 3 --alpha 1e-2 --min-conf 0.0 \
      --outdir models/session_001 --triangulate \
      --filter-face 1 --filter-blinks 1 \
      --face-col face_present --blink-cols blink_L,blink_R

Filtering behavior:
- By default, rows WITHOUT a detected face are dropped (if a face column is found),
  and rows marked as a blink are dropped (if blink columns are found).
- You can disable each filter with --filter-face 0 or --filter-blinks 0.
- If you don't pass column names, the trainer will try to auto-detect typical
  column names from Part A logs.

Outputs (inside --outdir):
  models/
    poly_left_eye_left_deg3_a1e-2.npz
    poly_left_eye_right_deg3_a1e-2.npz
    poly_both_eyes_deg3_a1e-2_concat_or_tri.npz
    poly_both_eyes_head_deg3_a1e-2_tri.npz
  reports/
    partB_metrics.json
    partB_report.md
  oof/
    oof_LEFT_EYE_left.csv
    oof_LEFT_EYE_right.csv
    oof_BOTH_EYES.csv
    oof_BOTH_EYES_HEAD.csv
  live_profile.json  # summary picking the best variant for runtime default
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .polynomial import GazeRegressor, Mode, ColumnMap, auto_map_columns


# ------------------------------- IO Utilities ------------------------------- #

def _find_frames_parquet(root: Path) -> Path:
    # Prefer explicit names; else fallback to first parquet containing target columns
    candidates = [
        root / "frames.parquet",
        root / "calibration_frames.parquet",
        root / "calib_frames.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: any .parquet in folder
    for p in root.glob("*.parquet"):
        return p
    raise FileNotFoundError(f"No parquet found in {root}")


def _load_config(root: Path) -> Dict:
    for name in ("config_snapshot.yaml", "config.yaml", "session_config.yaml"):
        p = root / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def _ensure_outdirs(outdir: Path) -> Tuple[Path, Path, Path]:
    models = outdir / "models"
    reports = outdir / "reports"
    oof = outdir / "oof"
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    oof.mkdir(parents=True, exist_ok=True)
    return models, reports, oof


# ----------------------------- Filtering Helpers ---------------------------- #

FACE_GUESSES = [
    "face_present", "has_face", "face_ok", "face_detected", "mp_face_ok", "face_found",
]
BLINK_GUESSES = [
    "blink", "is_blink", "blink_any", "blinked",
    "L_blink", "R_blink", "left_blink", "right_blink",
    "blink_left", "blink_right", "eye_blink",
]

def _to_bool_array(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if s.dtype.kind in "biu":
        return s.astype(int) != 0
    if s.dtype.kind == "f":
        return s.fillna(0.0).astype(float) != 0.0
    # strings/object
    return s.astype(str).str.lower().isin({"1", "true", "yes", "y", "t"})


def _guess_face_col(df: pd.DataFrame, explicit: Optional[str]) -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    for c in FACE_GUESSES:
        if c in df.columns:
            return c
    return None


def _guess_blink_cols(df: pd.DataFrame, explicit: Optional[List[str]]) -> List[str]:
    cols: List[str] = []
    if explicit:
        for c in explicit:
            if c in df.columns:
                cols.append(c)
    else:
        for c in BLINK_GUESSES:
            if c in df.columns:
                cols.append(c)
    # unique preserve order
    seen = set()
    uniq = []
    for c in cols:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _apply_row_filters(
    df: pd.DataFrame,
    face_col: Optional[str],
    blink_cols: Optional[List[str]],
    enable_face: bool = True,
    enable_blinks: bool = True,
) -> Tuple[Dict, pd.DataFrame]:
    """Return (stats, filtered_df)."""
    stats = {
        "rows_in": int(len(df)),
        "face_col": face_col,
        "blink_cols": blink_cols or [],
        "removed_no_face": 0,
        "removed_blinks": 0,
    }
    keep = pd.Series(True, index=df.index)

    if enable_face and face_col and face_col in df.columns:
        face_ok = _to_bool_array(df[face_col])
        stats["removed_no_face"] = int((~face_ok).sum())
        keep &= face_ok

    if enable_blinks and blink_cols:
        blink_any = pd.Series(False, index=df.index)
        for c in blink_cols:
            if c in df.columns:
                blink_any |= _to_bool_array(df[c])
        stats["removed_blinks"] = int((blink_any).sum())
        keep &= ~blink_any

    df2 = df[keep].reset_index(drop=True)
    stats["rows_out"] = int(len(df2))
    return stats, df2


# --------------------------------- Metrics --------------------------------- #

def _finite_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, screen_wh: Optional[Tuple[int, int]] = None) -> Dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    assert y_true.shape == y_pred.shape and y_true.shape[1] == 2
    m = _finite_mask(y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1])
    Y = y_true[m]
    P = y_pred[m]
    dx = P[:, 0] - Y[:, 0]
    dy = P[:, 1] - Y[:, 1]
    r = np.hypot(dx, dy)
    def p(x, q):
        return float(np.nanpercentile(x, q))
    W = H = None
    diag = None
    if screen_wh:
        W, H = screen_wh
        diag = float(np.hypot(W, H))
    out = {
        "count": int(len(r)),
        "mae_x": float(np.nanmean(np.abs(dx))),
        "mae_y": float(np.nanmean(np.abs(dy))),
        "mae_r": float(np.nanmean(r)),
        "rmse_r": float(np.sqrt(np.nanmean(r ** 2))),
        "median_r": float(np.nanmedian(r)),
        "p68_r": p(r, 68),
        "p90_r": p(r, 90),
        "p95_r": p(r, 95),
        "r2_x": float(1.0 - np.nanvar(dx) / (np.nanvar(Y[:, 0]) + 1e-12)),
        "r2_y": float(1.0 - np.nanvar(dy) / (np.nanvar(Y[:, 1]) + 1e-12)),
    }
    if diag and diag > 0:
        out.update({
            "mae_r_norm": out["mae_r"] / diag,
            "median_r_norm": out["median_r"] / diag,
            "p95_r_norm": out["p95_r"] / diag,
        })
    return out


# ------------------------------ Cross-Validation --------------------------- #

def kfold_indices(n: int, k: int, shuffle: bool = True, seed: int = 42) -> List[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        rs = np.random.RandomState(seed)
        rs.shuffle(idx)
    folds = np.array_split(idx, k)
    return [fold for fold in folds if len(fold) > 0]


def cross_validate(
    df: pd.DataFrame,
    build_reg,
    mode_key: str,
    k: int,
    shuffle: bool,
    seed: int,
) -> Tuple[Dict, np.ndarray]:
    """
    Returns (metrics, oof_pred) where oof_pred aligns with df rows (x_hat,y_hat).
    build_reg: lambda -> configured GazeRegressor (fresh instance per fold).
    """
    n = len(df)
    oof = np.full((n, 2), np.nan, dtype=float)
    folds = kfold_indices(n, k, shuffle=shuffle, seed=seed)
    for i, val_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), val_idx, assume_unique=False)
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        reg = build_reg()
        reg.fit_from_parquet(df_train)
        preds = []
        for _, r in df_val.iterrows():
            y, _ = reg.predict_from_row(r, return_std=False)
            preds.append(y)
        oof[val_idx] = np.asarray(preds)
    # we need reg to resolve target columns; rebuild a temp for cm
    tmp = build_reg()
    cm = auto_map_columns(df)
    Y = df[[cm.target_x, cm.target_y]].values
    screen_wh = None
    if "screen_w" in df.columns and "screen_h" in df.columns:
        screen_wh = (int(df["screen_w"].iloc[0]), int(df["screen_h"].iloc[0]))
    mets = compute_metrics(Y, oof, screen_wh=screen_wh)
    mets["mode"] = mode_key
    mets["kfold"] = k
    return mets, oof


# --------------------------------- Trainer --------------------------------- #

def train_all(
    data_dir: Path,
    outdir: Path,
    degree: int = 3,
    alpha: float = 1e-2,
    kfold: int = 5,
    triangulate: bool = True,
    min_conf: float = 0.0,
    shuffle: bool = True,
    seed: int = 42,
    one_eye_variants: Optional[List[str]] = None,  # ["left", "right"]
    filter_face: bool = True,
    filter_blinks: bool = True,
    face_col: Optional[str] = None,
    blink_cols: Optional[List[str]] = None,
):
    one_eye_variants = one_eye_variants or ["left", "right"]
    models_dir, reports_dir, oof_dir = _ensure_outdirs(outdir)

    frames_pq = _find_frames_parquet(data_dir)
    df = pd.read_parquet(frames_pq)

    # Attach screen info from config if available (for normalized metrics)
    cfg = _load_config(data_dir)
    screen_wh = None
    if isinstance(cfg, dict):
        W = cfg.get("screen", {}).get("width_px") if cfg.get("screen") else cfg.get("screen_width_px")
        H = cfg.get("screen", {}).get("height_px") if cfg.get("screen") else cfg.get("screen_height_px")
        if W and H:
            screen_wh = (int(W), int(H))
            if "screen_w" not in df.columns:
                df["screen_w"] = int(W)
                df["screen_h"] = int(H)

    # Confidence filtering if column exists
    cm = auto_map_columns(df)
    if min_conf > 0 and cm.frame_conf and cm.frame_conf in df.columns:
        df = df[df[cm.frame_conf] >= min_conf].reset_index(drop=True)

    # Face/blink filtering (before CV)
    face_col_eff = _guess_face_col(df, face_col)
    blink_cols_eff = _guess_blink_cols(df, blink_cols)
    filter_stats, df = _apply_row_filters(df, face_col_eff, blink_cols_eff, enable_face=filter_face, enable_blinks=filter_blinks)

    # Ensure targets exist
    if cm.target_x not in df.columns or cm.target_y not in df.columns:
        raise KeyError("Target columns not found. Ensure Part A logged target_x/target_y.")

    # Build variant registry
    variants = []
    # 1) LEFT_EYE (both left and right to choose later)
    for side in one_eye_variants:
        key = f"LEFT_EYE_{side}"
        def mk_left(side=side):
            return GazeRegressor(mode=Mode.LEFT_EYE, eye=side, degree=degree, alpha=alpha,
                                 triangulate=False, head_comp=False)
        variants.append((key, mk_left))
    # 2) BOTH_EYES (triangulate by default)
    def mk_both():
        return GazeRegressor(mode=Mode.BOTH_EYES, degree=degree, alpha=alpha,
                             triangulate=triangulate, head_comp=False)
    variants.append(("BOTH_EYES", mk_both))
    # 3) BOTH_EYES_HEAD
    def mk_both_head():
        return GazeRegressor(mode=Mode.BOTH_EYES_HEAD, degree=degree, alpha=alpha,
                             triangulate=triangulate, head_comp=True)
    variants.append(("BOTH_EYES_HEAD", mk_both_head))

    # Run CV per variant
    all_metrics: Dict[str, Dict] = {}
    all_oof_paths: Dict[str, str] = {}

    for key, ctor in variants:
        mets, oof = cross_validate(df, ctor, key, k=kfold, shuffle=shuffle, seed=seed)
        # If screen_wh provided externally, recompute normalized metrics with that
        if screen_wh and not ("mae_r_norm" in mets):
            mets.update({
                "mae_r_norm": mets["mae_r"] / float(np.hypot(*screen_wh)),
                "median_r_norm": mets["median_r"] / float(np.hypot(*screen_wh)),
                "p95_r_norm": mets["p95_r"] / float(np.hypot(*screen_wh)),
            })
        all_metrics[key] = mets
        oof_path = oof_dir / f"oof_{key}.csv"
        pd.DataFrame({
            "x_hat": oof[:, 0],
            "y_hat": oof[:, 1],
            cm.target_x: df[cm.target_x].values,
            cm.target_y: df[cm.target_y].values,
        }).to_csv(oof_path, index=False)
        all_oof_paths[key] = str(oof_path)

    # Pick best variant (lowest median_r) for live default
    best_key = min(all_metrics.keys(), key=lambda k: all_metrics[k]["median_r"])

    # Train final models on all data and save
    saved_models: Dict[str, str] = {}
    for key, ctor in variants:
        reg = ctor()
        reg.fit_from_parquet(df)
        # File naming
        if key.startswith("LEFT_EYE_"):
            side = key.split("_")[-1]
            name = f"poly_left_eye_{side}_deg{degree}_a{alpha}.npz"
        elif key == "BOTH_EYES":
            name = f"poly_both_eyes_deg{degree}_a{alpha}_{'tri' if triangulate else 'concat'}.npz"
        else:  # BOTH_EYES_HEAD
            name = f"poly_both_eyes_head_deg{degree}_a{alpha}_{'tri' if triangulate else 'concat'}.npz"
        model_path = models_dir / name
        reg.save(str(model_path))
        saved_models[key] = str(model_path)

    # Write reports
    metrics_path = reports_dir / "partB_metrics.json"
    report_path = reports_dir / "partB_report.md"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "kfold": kfold,
            "degree": degree,
            "alpha": alpha,
            "triangulate": triangulate,
            "screen_wh": screen_wh,
            "filter": {
                "enable_face": filter_face,
                "enable_blinks": filter_blinks,
                "face_col": face_col_eff,
                "blink_cols": blink_cols_eff,
                **filter_stats,
            },
            "variants": all_metrics,
            "oof_paths": all_oof_paths,
            "models": saved_models,
            "best_variant": best_key,
        }, f, indent=2)

    # Pretty markdown report
    def fmt_m(m):
        return (
            f"N={m['count']} | MAE_r={m['mae_r']:.1f}px | median={m['median_r']:.1f}px | "
            f"p95={m['p95_r']:.1f}px | RMSE={m['rmse_r']:.1f}px | "
            f"R2x={m['r2_x']:.3f} R2y={m['r2_y']:.3f}" +
            (f" | median_norm={m['median_r_norm']*100:.2f}% diag" if 'median_r_norm' in m else "")
        )

    lines = [
        "# Part B — Cross‑Validation Report\n",
        f"**Data dir:** {data_dir}\n",
        f"**K-fold:** {kfold} | **degree:** {degree} | **alpha:** {alpha} | **triangulate:** {triangulate}\n",
        f"**Screen:** {screen_wh if screen_wh else 'n/a'}\n",
        "\n## Filtering\n",
        f"- Face column: {face_col_eff if face_col_eff else 'n/a'}  ",
        f"- Blink columns: {', '.join(blink_cols_eff) if blink_cols_eff else 'n/a'}  ",
        f"- Rows in: {filter_stats['rows_in']} → Rows out: {filter_stats['rows_out']}  ",
        f"- Removed (no face): {filter_stats['removed_no_face']}  ",
        f"- Removed (blinks): {filter_stats['removed_blinks']}\n",
        "\n## Variants\n",
    ]
    for key in [k for k, _ in variants]:
        m = all_metrics[key]
        lines.append(f"- **{key}** → {fmt_m(m)}  ")
    lines += [
        "\n## Winner\n",
        f"**Best variant:** {best_key}  \n",
        f"Model: `{saved_models[best_key]}`\n",
        "\n## Files\n",
        f"- Metrics JSON: `{metrics_path}`\n",
        f"- Report MD: `{report_path}`\n",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Live profile for runtime bootstrap
    live_profile = {
        "default_variant": best_key,
        "models": saved_models,
        "screen_wh": screen_wh,
        "poly": {"degree": degree, "alpha": alpha, "triangulate": triangulate},
        "filter": {
            "enable_face": filter_face,
            "enable_blinks": filter_blinks,
            "face_col": face_col_eff,
            "blink_cols": blink_cols_eff,
        },
    }
    with open(outdir / "live_profile.json", "w", encoding="utf-8") as f:
        json.dump(live_profile, f, indent=2)

    return {
        "metrics_json": str(metrics_path),
        "report_md": str(report_path),
        "models": saved_models,
        "best_variant": best_key,
        "oof_paths": all_oof_paths,
        "live_profile": str(outdir / "live_profile.json"),
    }


# ------------------------------------ CLI ---------------------------------- #

def _cli():
    ap = argparse.ArgumentParser(description="Train all Part B regressors with K-fold CV")
    ap.add_argument("--data-dir", type=str, required=True, help="Folder with frames.parquet and config_snapshot.yaml")
    ap.add_argument("--outdir", type=str, required=True, help="Where to write models/reports")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--triangulate", action="store_true")
    ap.add_argument("--no-triangulate", dest="triangulate", action="store_false")
    ap.set_defaults(triangulate=True)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--shuffle", type=int, default=1, help="Shuffle before K-fold split (1/0)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--one-eye", type=str, default="left,right", help="Comma list from {left,right}")
    # Filtering flags/columns
    ap.add_argument("--filter-face", type=int, default=1, help="Drop rows where face absent (1/0)")
    ap.add_argument("--filter-blinks", type=int, default=1, help="Drop rows where blink is true (1/0)")
    ap.add_argument("--face-col", type=str, default=None, help="Column name for face presence (truthy means face present)")
    ap.add_argument("--blink-cols", type=str, default=None, help="Comma-separated blink columns; any truthy means blink")

    args = ap.parse_args()

    out = train_all(
        data_dir=Path(args.data_dir),
        outdir=Path(args.outdir),
        degree=args.degree,
        alpha=args.alpha,
        kfold=args.kfold,
        triangulate=args.triangulate,
        min_conf=args.min_conf,
        shuffle=bool(args.shuffle),
        seed=args.seed,
        one_eye_variants=[s.strip() for s in args.one_eye.split(",") if s.strip()],
        filter_face=bool(args.filter_face),
        filter_blinks=bool(args.filter_blinks),
        face_col=args.face_col,
        blink_cols=[s.strip() for s in args.blink_cols.split(",")] if args.blink_cols else None,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _cli()
