"""
Part B — Polynomial Gaze Regressor (NumPy-only)

Drop-in regressor module for Part A parquet logs.

Features:
- Three switchable modes:
  1) LEFT_EYE: train/predict from one eye only (choose 'left' or 'right').
  2) BOTH_EYES: train/predict from concatenated features of both eyes; optional
     two-branch fusion where each eye has its own head-normalized model and
     predictions are fused ("triangulation" fallback = weighted midpoint).
  3) BOTH_EYES_HEAD: like BOTH_EYES but adds explicit head-pose compensation
     (yaw/pitch/roll + translation) as inputs and normalization terms.
- Polynomial expansion + ridge regularization (no scikit-learn).
- Online weighted updates (incremental re-calibration) via normal-equation
  accumulators (Φᵀ W Φ and Φᵀ W Y). Call `update_with_known_point(...)` at any time.
- Works directly with parquet from Part A (pandas/pyarrow). Flexible column mapping.
- Outputs per-axis uncertainty estimates from ridge posterior (approx.).

Minimal deps: numpy, pandas (for parquet). Optional: pyarrow (usually installed
when writing parquet). If pandas is unavailable, you can pass numpy arrays
manually.

Usage (CLI examples):

Train and save a BOTH_EYES_HEAD model from parquet:
    python -m eyetracker.regression.poly_gaze \
        --parquet data/session.parquet \
        --mode BOTH_EYES_HEAD \
        --degree 3 --alpha 1e-2 \
        --out models/partB_both_head_deg3_alpha1e-2.npz

Load a model and run predictions for a parquet (dry-run):
    python -m eyetracker.regression.poly_gaze \
        --model models/partB_both_head_deg3_alpha1e-2.npz \
        --parquet data/quick_check.parquet --predict-only

Online update (within your realtime loop):
    reg.update_with_known_point(feature_dict_or_row, target_xy=(x,y), weight=5.0)

Integration notes:
- This file is self-contained. Place it at: eyetracker/regression/poly_gaze.py
- Part A should log columns similar to those listed in DEFAULT_COLUMN_GUESSES.
  If your names differ, pass an explicit ColumnMap when constructing.

"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # allow numpy-only usage


# ------------------------------ Column Mapping ------------------------------ #

@dataclass
class ColumnMap:
    """Describes where to find features/targets in the Part A parquet logs.

    You can pass a customized instance if your column names differ. Otherwise
    the loader will try to auto-detect using DEFAULT_COLUMN_GUESSES.
    """
    # Targets (screen coordinates in pixels)
    target_x: str = "target_x"
    target_y: str = "target_y"

    # Left eye keypoints (image coords, pixels)
    L_pupil_x: str = "L_pupil_x"
    L_pupil_y: str = "L_pupil_y"
    L_in_x: str = "L_in_x"       # inner (nasal) canthus
    L_in_y: str = "L_in_y"
    L_out_x: str = "L_out_x"     # outer (temporal) canthus
    L_out_y: str = "L_out_y"

    # Right eye keypoints (image coords, pixels)
    R_pupil_x: str = "R_pupil_x"
    R_pupil_y: str = "R_pupil_y"
    R_in_x: str = "R_in_x"
    R_in_y: str = "R_in_y"
    R_out_x: str = "R_out_x"
    R_out_y: str = "R_out_y"

    # Optional eyelid openness / blink proxy (if available)
    L_ear: Optional[str] = None  # Eye Aspect Ratio or similar
    R_ear: Optional[str] = None

    # Head pose (rvec in degrees; tvec in same linear units as calibration)
    head_yaw: Optional[str] = "head_yaw_deg"
    head_pitch: Optional[str] = "head_pitch_deg"
    head_roll: Optional[str] = "head_roll_deg"
    head_tx: Optional[str] = "head_tx"
    head_ty: Optional[str] = "head_ty"
    head_tz: Optional[str] = "head_tz"

    # Optional per-frame quality/confidence (0..1)
    frame_conf: Optional[str] = None

    # Optional per-eye gaze angles if Part A logged them (degrees or radians)
    # If present, we include them as features; if absent, we derive angles
    # from keypoints normalized by eye width/height.
    L_gaze_yaw: Optional[str] = None
    L_gaze_pitch: Optional[str] = None
    R_gaze_yaw: Optional[str] = None
    R_gaze_pitch: Optional[str] = None

    # Optional screen meta (width/height) if logged per-row; otherwise pass at fit()
    screen_w: Optional[str] = None
    screen_h: Optional[str] = None


# Reasonable default guesses for auto-detection. Add aliases here to match Part A.
DEFAULT_COLUMN_GUESSES: Dict[str, List[str]] = {
    "target_x": ["target_x", "gx", "screen_x", "gt_x"],
    "target_y": ["target_y", "gy", "screen_y", "gt_y"],
    "L_pupil_x": ["L_pupil_x", "left_pupil_x", "pupilL_x", "L_iris_x"],
    "L_pupil_y": ["L_pupil_y", "left_pupil_y", "pupilL_y", "L_iris_y"],
    "L_in_x": ["L_in_x", "left_inner_x", "L_inner_x"],
    "L_in_y": ["L_in_y", "left_inner_y", "L_inner_y"],
    "L_out_x": ["L_out_x", "left_outer_x", "L_outer_x"],
    "L_out_y": ["L_out_y", "left_outer_y", "L_outer_y"],
    "R_pupil_x": ["R_pupil_x", "right_pupil_x", "pupilR_x", "R_iris_x"],
    "R_pupil_y": ["R_pupil_y", "right_pupil_y", "pupilR_y", "R_iris_y"],
    "R_in_x": ["R_in_x", "right_inner_x", "R_inner_x"],
    "R_in_y": ["R_in_y", "right_inner_y", "R_inner_y"],
    "R_out_x": ["R_out_x", "right_outer_x", "R_outer_x"],
    "R_out_y": ["R_out_y", "right_outer_y", "R_outer_y"],
    "head_yaw_deg": ["head_yaw_deg", "yaw", "head_yaw"],
    "head_pitch_deg": ["head_pitch_deg", "pitch", "head_pitch"],
    "head_roll_deg": ["head_roll_deg", "roll", "head_roll"],
    "head_tx": ["head_tx", "tx"],
    "head_ty": ["head_ty", "ty"],
    "head_tz": ["head_tz", "tz", "depth_mm", "z_mm"],
    "frame_conf": ["frame_conf", "quality", "conf"],
}


def auto_map_columns(df: "pd.DataFrame") -> ColumnMap:
    """Heuristic column auto-mapper using DEFAULT_COLUMN_GUESSES."""
    def pick(name: str, guesses: List[str]) -> Optional[str]:
        for g in guesses:
            if g in df.columns:
                return g
        return None

    cmap = ColumnMap()
    for field in ColumnMap.__dataclass_fields__.keys():
        if getattr(cmap, field) is not None and field in DEFAULT_COLUMN_GUESSES:
            guess = pick(field, DEFAULT_COLUMN_GUESSES[field])
            if guess:
                setattr(cmap, field, guess)
    # Targets
    cmap.target_x = pick("target_x", DEFAULT_COLUMN_GUESSES["target_x"]) or cmap.target_x
    cmap.target_y = pick("target_y", DEFAULT_COLUMN_GUESSES["target_y"]) or cmap.target_y
    return cmap


# ------------------------------ Poly Features ------------------------------ #

class PolyFeatures:
    """Polynomial feature expansion with optional standardization.

    - degree: max polynomial degree (>=1). Includes bias by default.
    - include_interactions: if True, include cross terms.
    - feature_names_: names after expansion (set after fit_names) for debugging.
    """

    def __init__(self, degree: int = 2, include_interactions: bool = True, include_bias: bool = True):
        assert degree >= 1
        self.degree = degree
        self.include_interactions = include_interactions
        self.include_bias = include_bias
        self.mu_: Optional[np.ndarray] = None
        self.sigma_: Optional[np.ndarray] = None
        self.feature_idx_: Optional[List[Tuple[int, ...]]] = None
        self.feature_names_: Optional[List[str]] = None

    def _monomials(self, n_in: int) -> List[Tuple[int, ...]]:
        """Return list of exponent tuples e.g., (2,0,1) for x0^2 * x2^1.
        If include_interactions=False, only powers of single variables.
        """
        exps: List[Tuple[int, ...]] = []
        if self.include_interactions:
            # Generate all combinations with replacement up to degree
            # Using recursive generation for clarity (n small)
            def gen(current, start, remaining_degree):
                if remaining_degree == 0:
                    exps.append(tuple(current))
                    return
                # distribute degrees across variables
                for i in range(start, n_in):
                    current[i] += 1
                    gen(current, i, remaining_degree - 1)
                    current[i] -= 1
            # All degrees from 0..degree
            for d in range(0, self.degree + 1):
                v = [0] * n_in
                gen(v, 0, d)
        else:
            # Only single-variable powers
            exps.append(tuple([0] * n_in))  # degree 0 (bias)
            for i in range(n_in):
                for p in range(1, self.degree + 1):
                    v = [0] * n_in
                    v[i] = p
                    exps.append(tuple(v))
        # Drop bias later if include_bias=False
        return exps

    def fit_names(self, in_names: List[str]) -> None:
        n_in = len(in_names)
        idx = self._monomials(n_in)
        if not self.include_bias:
            # Remove the all-zero exponent monomial
            idx = [t for t in idx if any(e > 0 for e in t)]
        # Build readable feature names
        names: List[str] = []
        for tup in idx:
            if sum(tup) == 0:
                names.append("bias")
                continue
            parts = []
            for j, p in enumerate(tup):
                if p == 0:
                    continue
                parts.append(f"{in_names[j]}^{p}" if p > 1 else in_names[j])
            names.append("*".join(parts))
        self.feature_idx_ = idx
        self.feature_names_ = names

    def _standardize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if self.mu_ is None or fit:
            self.mu_ = X.mean(axis=0, keepdims=True)
            self.sigma_ = X.std(axis=0, keepdims=True)
            self.sigma_[self.sigma_ < 1e-8] = 1.0
        return (X - self.mu_) / self.sigma_

    def transform(self, X: np.ndarray, in_names: Optional[List[str]] = None, standardize: bool = True, fit: bool = False) -> np.ndarray:
        """Expand inputs X (N×D) into polynomial features Φ (N×M)."""
        X = np.asarray(X, dtype=np.float64)
        if in_names is not None and self.feature_idx_ is None:
            self.fit_names(in_names)
        assert self.feature_idx_ is not None, "Call fit_names once with input feature names."
        Xs = self._standardize(X, fit=fit) if standardize else X
        N, D = Xs.shape
        M = len(self.feature_idx_)
        Phi = np.empty((N, M), dtype=np.float64)
        for m, exp in enumerate(self.feature_idx_):
            # Compute product over features j: Xs[:, j] ** exp[j]
            col = np.ones(N, dtype=np.float64)
            for j, p in enumerate(exp):
                if p:
                    col *= Xs[:, j] ** p
            Phi[:, m] = col
        if not self.include_bias:
            pass  # bias already excluded in idx
        return Phi


# ------------------------------ Online Ridge ------------------------------ #

class OnlineRidge:
    """Ridge regression solved from normal equations with incremental updates.

    Y can be 1-D (N,) or 2-D (N×T) for multi-output. We maintain:
      S = Φᵀ W Φ,   T = Φᵀ W Y
    and solve (S + αI) β = T for β via Cholesky when predicting or after updates.

    - alpha: ridge strength α (applied to all coefficients; bias optionally exempt).
    - bias_index: set to the column index of the bias term in Φ to avoid regularizing it.
    - eps: numerical jitter for Cholesky.
    - track_rss: accumulate residual sum of squares to estimate σ² (uncertainty).
    """

    def __init__(self, n_features: int, n_targets: int = 2, alpha: float = 1e-2, bias_index: Optional[int] = 0, eps: float = 1e-9):
        self.n_features = int(n_features)
        self.n_targets = int(n_targets)
        self.alpha = float(alpha)
        self.bias_index = bias_index
        self.eps = float(eps)

        self.S = np.zeros((self.n_features, self.n_features), dtype=np.float64)
        self.T = np.zeros((self.n_features, self.n_targets), dtype=np.float64)
        self.n_eff = 0.0  # effective weight sum
        self.rss = np.zeros((self.n_targets,), dtype=np.float64)

        self.coef_: Optional[np.ndarray] = None  # (M×T)
        self.L_: Optional[np.ndarray] = None     # Cholesky factor of (S + A)

    def _ridge_A(self) -> np.ndarray:
        A = np.eye(self.n_features, dtype=np.float64) * self.alpha
        if self.bias_index is not None:
            A[self.bias_index, self.bias_index] = 0.0
        return A

    def _factorize(self) -> None:
        A = self._ridge_A()
        M = self.S + A + np.eye(self.n_features) * self.eps
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            # Fallback to eigh if not PD
            w, V = np.linalg.eigh(M)
            w[w < self.eps] = self.eps
            M_pd = (V * w) @ V.T
            L = np.linalg.cholesky(M_pd)
        self.L_ = L
        # Solve for coefficients: (S + A) β = T  => β = M^{-1} T using chol solves
        self.coef_ = self._solve_matrix(T=self.T)

    def _solve_matrix(self, T: np.ndarray) -> np.ndarray:
        assert self.L_ is not None
        # Solve L Lᵀ X = T  => first solve L Y = T, then Lᵀ X = Y
        Y = np.linalg.solve(self.L_, T)
        X = np.linalg.solve(self.L_.T, Y)
        return X

    def fit(self, Phi: np.ndarray, Y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        Phi = np.asarray(Phi, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        assert Phi.shape[0] == Y.shape[0]
        N, M = Phi.shape
        if sample_weight is None:
            W = np.ones((N, 1), dtype=np.float64)
        else:
            W = np.asarray(sample_weight, dtype=np.float64).reshape(N, 1)
        # Accumulate normal equations
        self.S = Phi.T @ (W * Phi)
        self.T = Phi.T @ (W * Y)
        self.n_eff = float(W.sum())
        # Residuals for sigma^2 estimation
        self._factorize()
        Yhat = Phi @ self.coef_
        resid = Y - Yhat
        self.rss = (W[:, 0] * (resid ** 2)).sum(axis=0)

    def partial_update(self, phi: np.ndarray, y: np.ndarray, w: float = 1.0) -> None:
        phi = np.asarray(phi, dtype=np.float64).reshape(-1, 1)  # (M×1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)         # (T,)
        if y.ndim == 1:
            y = y[:, None]
        self.S += w * (phi @ phi.T)
        self.T += w * (phi @ y.T)
        self.n_eff += w
        # Invalidate factorization; recompute lazily on predict or explicit call
        self.L_ = None
        self.coef_ = None

    def refactor(self) -> None:
        self._factorize()

    def predict(self, Phi: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        Phi = np.asarray(Phi, dtype=np.float64)
        if self.coef_ is None or self.L_ is None:
            self._factorize()
        Yhat = Phi @ self.coef_
        if not return_std:
            return Yhat
        # Approximate predictive std per-target: σ^2 * diag(Φ (S+A)^{-1} Φᵀ)
        # Compute v = solve(M, φᵀ) efficiently for each row
        stds = []
        sigma2 = (self.rss / max(self.n_eff - self.n_features, 1.0)).clip(min=1e-12)
        for i in range(Phi.shape[0]):
            phi = Phi[i : i + 1, :]
            v = self._solve_matrix(T=phi.T)  # (M×1)
            d = (phi @ v).reshape(-1)  # scalar
            stds.append(np.sqrt(sigma2 * float(d)))
        std = np.vstack(stds)  # (N×T)
        return Yhat, std


# --------------------------- Feature Engineering --------------------------- #

@dataclass
class FeatureConfig:
    degree: int = 2
    include_interactions: bool = True
    include_bias: bool = True
    add_eye_geom: bool = True        # derived dx, dy normalized by eye width/height
    add_eye_ratio: bool = True       # pupil offset ratios, eyelid EAR if present
    add_head_pose: bool = False      # yaw/pitch/roll (+ tx/ty/tz if available)
    standardize: bool = True


def _eye_center_and_size(p_in: float, p_out: float) -> Tuple[float, float, float]:
    cx = 0.5 * (p_in + p_out)
    w = abs(p_out - p_in) + 1e-6
    scale = 1.0 / w
    return cx, w, scale


def _compose_eye_features(row: Dict[str, float], side: str, cm: ColumnMap, cfg: FeatureConfig) -> Tuple[List[str], List[float]]:
    assert side in ("L", "R")
    names: List[str] = []
    vals: List[float] = []

    # Raw pupil (px)
    px = float(row[getattr(cm, f"{side}_pupil_x")])
    py = float(row[getattr(cm, f"{side}_pupil_y")])

    # Eye corners
    inx = float(row[getattr(cm, f"{side}_in_x")])
    iny = float(row[getattr(cm, f"{side}_in_y")])
    ox = float(row[getattr(cm, f"{side}_out_x")])
    oy = float(row[getattr(cm, f"{side}_out_y")])

    # Basic raw features (image space)
    names += [f"{side}_pupil_x", f"{side}_pupil_y", f"{side}_in_x", f"{side}_in_y", f"{side}_out_x", f"{side}_out_y"]
    vals  += [px, py, inx, iny, ox, oy]

    if cfg.add_eye_geom:
        cx, w, sx = _eye_center_and_size(inx, ox)
        cy, h, sy = _eye_center_and_size(iny, oy)
        # Normalize pupil displacement by eye width/height (head-scale invariant)
        dx = (px - cx) * sx
        dy = (py - cy) * sy
        names += [f"{side}_dx_norm", f"{side}_dy_norm", f"{side}_w", f"{side}_h"]
        vals  += [dx, dy, w, h]

    if cfg.add_eye_ratio:
        # Aspect ratio proxy
        ear_name = getattr(cm, f"{side}_ear")
        if ear_name and ear_name in row:
            try:
                ear = float(row[ear_name])
            except Exception:
                ear = 0.0
        else:
            ear = 0.0
        names += [f"{side}_ear"]
        vals  += [ear]

    # Optional logged gaze angles
    for ang in ("gaze_yaw", "gaze_pitch"):
        cname = getattr(cm, f"{side}_{ang}")
        if cname and cname in row:
            names.append(f"{side}_{ang}")
            vals.append(float(row[cname]))

    return names, vals


def _compose_head_features(row: Dict[str, float], cm: ColumnMap) -> Tuple[List[str], List[float]]:
    names: List[str] = []
    vals: List[float] = []
    for key in ("head_yaw", "head_pitch", "head_roll", "head_tx", "head_ty", "head_tz"):
        cname = getattr(cm, key, None)
        if cname and cname in row:
            names.append(key)
            vals.append(float(row[cname]))
    return names, vals


# ------------------------------ Gaze Regressor ----------------------------- #

class Mode:
    LEFT_EYE = "LEFT_EYE"
    BOTH_EYES = "BOTH_EYES"
    BOTH_EYES_HEAD = "BOTH_EYES_HEAD"


class GazeRegressor:
    """Unified wrapper for the three Part B regressors.

    Params
    ------
    mode: one of Mode.LEFT_EYE, Mode.BOTH_EYES, Mode.BOTH_EYES_HEAD
    eye: 'left' or 'right' when using LEFT_EYE mode
    degree: polynomial degree
    alpha: ridge regularization strength
    triangulate: if True in BOTH_EYES*, fuse two single-eye branches; otherwise
                 use a single concatenated feature vector model.
    head_comp: include/compensate head pose (enabled automatically in BOTH_EYES_HEAD)
    screen_size: (W,H) pixels for normalization/consistency if not in logs
    column_map: ColumnMap to resolve column names
    """

    def __init__(
        self,
        mode: str = Mode.BOTH_EYES_HEAD,
        eye: str = "left",
        degree: int = 2,
        alpha: float = 1e-2,
        triangulate: bool = True,
        head_comp: Optional[bool] = None,
        screen_size: Optional[Tuple[int, int]] = None,
        column_map: Optional[ColumnMap] = None,
    ):
        assert mode in (Mode.LEFT_EYE, Mode.BOTH_EYES, Mode.BOTH_EYES_HEAD)
        assert eye in ("left", "right")
        self.mode = mode
        self.eye = eye
        self.degree = degree
        self.alpha = alpha
        self.triangulate = triangulate if mode != Mode.LEFT_EYE else False
        self.head_comp = (mode == Mode.BOTH_EYES_HEAD) if head_comp is None else bool(head_comp)
        self.screen_size = screen_size
        self.cm = column_map or ColumnMap()

        # Feature builders per branch
        self.cfg_left = FeatureConfig(
            degree=degree, include_interactions=True, include_bias=True,
            add_eye_geom=True, add_eye_ratio=True, add_head_pose=self.head_comp, standardize=True,
        )
        self.cfg_right = FeatureConfig(
            degree=degree, include_interactions=True, include_bias=True,
            add_eye_geom=True, add_eye_ratio=True, add_head_pose=self.head_comp, standardize=True,
        )

        self.poly_left: Optional[PolyFeatures] = None
        self.poly_right: Optional[PolyFeatures] = None
        self.poly_joint: Optional[PolyFeatures] = None

        self.ridge_left: Optional[OnlineRidge] = None
        self.ridge_right: Optional[OnlineRidge] = None
        self.ridge_joint: Optional[OnlineRidge] = None

        # Names remembered for expansion
        self._in_names_left: Optional[List[str]] = None
        self._in_names_right: Optional[List[str]] = None
        self._in_names_joint: Optional[List[str]] = None

    # -------------------------- Data Loading Helpers ------------------------- #

    def _ensure_df(self, data: Union[str, "pd.DataFrame"]) -> "pd.DataFrame":
        assert pd is not None, "pandas is required to read parquet. Install pandas/pyarrow."
        if isinstance(data, str):
            if data.lower().endswith((".parquet", ".pq")):
                return pd.read_parquet(data)
            else:
                return pd.read_csv(data)
        return data

    def _ensure_column_map(self, df: "pd.DataFrame") -> None:
        # Fill missing names by auto-guess
        self.cm = auto_map_columns(df)

    # ---------------------------- Feature Builders --------------------------- #

    def _row_to_features(self, row: Dict[str, float]) -> Tuple[List[str], np.ndarray, Tuple[float, float]]:
        """Compose input features per current mode. Returns (names, values, target_xy)."""
        # Targets
        tx = float(row[self.cm.target_x])
        ty = float(row[self.cm.target_y])

        # Single-eye mode
        if self.mode == Mode.LEFT_EYE:
            side = "L" if self.eye == "left" else "R"
            names, vals = _compose_eye_features(row, side, self.cm, self.cfg_left)
            if self.head_comp:  # normally False in LEFT_EYE, but keep option
                hn, hv = _compose_head_features(row, self.cm)
                names += hn
                vals += hv
            return names, np.asarray(vals, dtype=np.float64), (tx, ty)

        # Both-eyes modes
        nL, vL = _compose_eye_features(row, "L", self.cm, self.cfg_left)
        nR, vR = _compose_eye_features(row, "R", self.cm, self.cfg_right)
        if self.head_comp:
            hn, hv = _compose_head_features(row, self.cm)
            # Attach head features to both branches and/or joint
            nLh = nL + hn
            vLh = vL + hv
            nRh = nR + hn
            vRh = vR + hv
        else:
            nLh, vLh, nRh, vRh = nL, vL, nR, vR

        if self.triangulate:
            # Two-branch design: keep per-eye feature sets
            # Joint feature set is not used for prediction in this mode
            # (we fuse two predictions later).
            names = ["<two-branch>"]
            vals = np.array([0.0])  # placeholder not used
            # Store separately for internal use
            return (names, vals, (tx, ty))  # not actually used directly
        else:
            # Concatenate and learn a single joint mapping
            names = [f"L.{n}" for n in nLh] + [f"R.{n}" for n in nRh]
            vals = np.asarray(vLh + vRh, dtype=np.float64)
            return names, vals, (tx, ty)

    # ------------------------------ Fitting API ------------------------------ #

    def fit_from_parquet(self, data: Union[str, "pd.DataFrame"], min_conf: float = 0.0, sample_weight_col: Optional[str] = None) -> None:
        df = self._ensure_df(data)
        self._ensure_column_map(df)

        # Optional confidence filtering
        if min_conf > 0 and self.cm.frame_conf and self.cm.frame_conf in df.columns:
            df = df[df[self.cm.frame_conf] >= min_conf]

        # Prepare input matrices
        if self.mode == Mode.LEFT_EYE:
            X, Y, names = self._build_XY_single(df)
            self._setup_single(names)
            Phi = self.poly_left.transform(X, in_names=names, fit=True)
            self.ridge_left = OnlineRidge(n_features=Phi.shape[1], n_targets=2, alpha=self.alpha, bias_index=0)
            self.ridge_left.fit(Phi, Y, sample_weight=df[sample_weight_col].values if sample_weight_col else None)
            return

        if self.triangulate:
            # Two separate branches
            X_L, X_R, Y, names_L, names_R = self._build_XY_branches(df)
            self._setup_branches(names_L, names_R)
            PhiL = self.poly_left.transform(X_L, in_names=names_L, fit=True)
            PhiR = self.poly_right.transform(X_R, in_names=names_R, fit=True)
            self.ridge_left = OnlineRidge(PhiL.shape[1], 2, alpha=self.alpha, bias_index=0)
            self.ridge_right = OnlineRidge(PhiR.shape[1], 2, alpha=self.alpha, bias_index=0)
            W = df[sample_weight_col].values if sample_weight_col else None
            self.ridge_left.fit(PhiL, Y, sample_weight=W)
            self.ridge_right.fit(PhiR, Y, sample_weight=W)
            return
        else:
            # Single joint model
            X, Y, names = self._build_XY_joint(df)
            self._setup_joint(names)
            Phi = self.poly_joint.transform(X, in_names=names, fit=True)
            self.ridge_joint = OnlineRidge(Phi.shape[1], 2, alpha=self.alpha, bias_index=0)
            self.ridge_joint.fit(Phi, Y, sample_weight=df[sample_weight_col].values if sample_weight_col else None)
            return

    def _build_XY_single(self, df: "pd.DataFrame") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X_list: List[List[float]] = []
        names_ref: Optional[List[str]] = None
        Y = df[[self.cm.target_x, self.cm.target_y]].values.astype(np.float64)
        for _, row in df.iterrows():
            n, v, _ = self._row_to_features(row)
            if names_ref is None:
                names_ref = n
            X_list.append(v.tolist())
        assert names_ref is not None
        X = np.array(X_list, dtype=np.float64)
        return X, Y, names_ref

    def _build_XY_branches(self, df: "pd.DataFrame") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        XL: List[List[float]] = []
        XR: List[List[float]] = []
        Y = df[[self.cm.target_x, self.cm.target_y]].values.astype(np.float64)
        namesL: Optional[List[str]] = None
        namesR: Optional[List[str]] = None
        for _, r in df.iterrows():
            # Build explicit branch features
            nL, vL = _compose_eye_features(r, "L", self.cm, self.cfg_left)
            nR, vR = _compose_eye_features(r, "R", self.cm, self.cfg_right)
            if self.head_comp:
                hn, hv = _compose_head_features(r, self.cm)
                nL = nL + hn
                vL = vL + hv
                nR = nR + hn
                vR = vR + hv
            if namesL is None:
                namesL = nL
                namesR = nR
            XL.append(vL)
            XR.append(vR)
        assert namesL is not None and namesR is not None
        return np.array(XL, dtype=np.float64), np.array(XR, dtype=np.float64), Y, namesL, namesR

    def _build_XY_joint(self, df: "pd.DataFrame") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X_list: List[List[float]] = []
        names_ref: Optional[List[str]] = None
        Y = df[[self.cm.target_x, self.cm.target_y]].values.astype(np.float64)
        for _, row in df.iterrows():
            n, v, _ = self._row_to_features(row)
            if names_ref is None:
                names_ref = n
            X_list.append(v.tolist())
        assert names_ref is not None
        X = np.array(X_list, dtype=np.float64)
        return X, Y, names_ref

    def _setup_single(self, names: List[str]) -> None:
        self._in_names_left = names
        self.poly_left = PolyFeatures(self.degree, include_interactions=True, include_bias=True)
        self.poly_left.fit_names(names)

    def _setup_branches(self, names_left: List[str], names_right: List[str]) -> None:
        self._in_names_left = names_left
        self._in_names_right = names_right
        self.poly_left = PolyFeatures(self.degree, include_interactions=True, include_bias=True)
        self.poly_right = PolyFeatures(self.degree, include_interactions=True, include_bias=True)
        self.poly_left.fit_names(names_left)
        self.poly_right.fit_names(names_right)

    def _setup_joint(self, names_joint: List[str]) -> None:
        self._in_names_joint = names_joint
        self.poly_joint = PolyFeatures(self.degree, include_interactions=True, include_bias=True)
        self.poly_joint.fit_names(names_joint)

    # ------------------------------ Prediction ------------------------------ #

    def predict_from_row(self, row: Dict[str, float], return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict (x,y) from a single row dict-like (e.g., pandas.Series)."""
        if self.mode == Mode.LEFT_EYE:
            names, vals, _ = self._row_to_features(row)
            Phi = self.poly_left.transform(vals.reshape(1, -1), in_names=self._in_names_left)
            Yhat, std = self.ridge_left.predict(Phi, return_std=True)
            return Yhat[0], (std[0] if return_std else None)

        if self.triangulate:
            # Two-branch
            nL, vL = _compose_eye_features(row, "L", self.cm, self.cfg_left)
            nR, vR = _compose_eye_features(row, "R", self.cm, self.cfg_right)
            if self.head_comp:
                hn, hv = _compose_head_features(row, self.cm)
                nL = nL + hn
                vL = vL + hv
                nR = nR + hn
                vR = vR + hv
            PhiL = self.poly_left.transform(np.asarray(vL, dtype=np.float64).reshape(1, -1), in_names=self._in_names_left)
            PhiR = self.poly_right.transform(np.asarray(vR, dtype=np.float64).reshape(1, -1), in_names=self._in_names_right)
            yL, sL = self.ridge_left.predict(PhiL, return_std=True)
            yR, sR = self.ridge_right.predict(PhiR, return_std=True)
            # Fuse ("triangulate"): inverse-variance weighting
            varL = np.maximum(sL[0] ** 2, 1e-12)
            varR = np.maximum(sR[0] ** 2, 1e-12)
            wL = 1.0 / varL
            wR = 1.0 / varR
            y = (wL * yL[0] + wR * yR[0]) / (wL + wR)
            if return_std:
                s = np.sqrt(1.0 / (wL + wR))
                return y, s
            return y, None
        else:
            # Joint concatenated model
            names, vals, _ = self._row_to_features(row)
            Phi = self.poly_joint.transform(vals.reshape(1, -1), in_names=self._in_names_joint)
            Yhat, std = self.ridge_joint.predict(Phi, return_std=True)
            return Yhat[0], (std[0] if return_std else None)

    # -------------------------- Online Recalibration ------------------------- #

    def update_with_known_point(self, row: Dict[str, float], target_xy: Tuple[float, float], weight: float = 1.0) -> None:
        """Perform a weighted online update with a frame and known (x,y) target.
        Call this during runtime when the user presses a button while fixating.
        """
        y = np.asarray(target_xy, dtype=np.float64).reshape(2,)
        if self.mode == Mode.LEFT_EYE:
            names, vals, _ = self._row_to_features(row)
            phi = self.poly_left.transform(vals.reshape(1, -1), in_names=self._in_names_left)
            self.ridge_left.partial_update(phi.ravel(), y, w=weight)
            return

        if self.triangulate:
            # Update both branches toward the same target
            nL, vL = _compose_eye_features(row, "L", self.cm, self.cfg_left)
            nR, vR = _compose_eye_features(row, "R", self.cm, self.cfg_right)
            if self.head_comp:
                hn, hv = _compose_head_features(row, self.cm)
                vL = vL + hv
                vR = vR + hv
            phiL = self.poly_left.transform(np.asarray(vL, dtype=np.float64).reshape(1, -1), in_names=self._in_names_left)
            phiR = self.poly_right.transform(np.asarray(vR, dtype=np.float64).reshape(1, -1), in_names=self._in_names_right)
            self.ridge_left.partial_update(phiL.ravel(), y, w=weight)
            self.ridge_right.partial_update(phiR.ravel(), y, w=weight)
        else:
            names, vals, _ = self._row_to_features(row)
            phi = self.poly_joint.transform(vals.reshape(1, -1), in_names=self._in_names_joint)
            self.ridge_joint.partial_update(phi.ravel(), y, w=weight)

    # ------------------------------ Persistence ------------------------------ #

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        state = {
            "mode": self.mode,
            "eye": self.eye,
            "degree": self.degree,
            "alpha": self.alpha,
            "triangulate": self.triangulate,
            "head_comp": self.head_comp,
            "screen_size": self.screen_size,
            "column_map": asdict(self.cm),
            # Poly stats
            "poly_left": {
                "mu": getattr(self.poly_left, "mu_", None),
                "sigma": getattr(self.poly_left, "sigma_", None),
                "idx": getattr(self.poly_left, "feature_idx_", None),
                "names": getattr(self.poly_left, "feature_names_", None),
                "in_names": self._in_names_left,
            } if self.poly_left else None,
            "poly_right": {
                "mu": getattr(self.poly_right, "mu_", None),
                "sigma": getattr(self.poly_right, "sigma_", None),
                "idx": getattr(self.poly_right, "feature_idx_", None),
                "names": getattr(self.poly_right, "feature_names_", None),
                "in_names": self._in_names_right,
            } if self.poly_right else None,
            "poly_joint": {
                "mu": getattr(self.poly_joint, "mu_", None),
                "sigma": getattr(self.poly_joint, "sigma_", None),
                "idx": getattr(self.poly_joint, "feature_idx_", None),
                "names": getattr(self.poly_joint, "feature_names_", None),
                "in_names": self._in_names_joint,
            } if self.poly_joint else None,
            # Ridge params
            "ridge_left": {
                "S": getattr(self.ridge_left, "S", None),
                "T": getattr(self.ridge_left, "T", None),
                "n_eff": getattr(self.ridge_left, "n_eff", None),
                "rss": getattr(self.ridge_left, "rss", None),
                "coef": getattr(self.ridge_left, "coef_", None),
            } if self.ridge_left else None,
            "ridge_right": {
                "S": getattr(self.ridge_right, "S", None),
                "T": getattr(self.ridge_right, "T", None),
                "n_eff": getattr(self.ridge_right, "n_eff", None),
                "rss": getattr(self.ridge_right, "rss", None),
                "coef": getattr(self.ridge_right, "coef_", None),
            } if self.ridge_right else None,
            "ridge_joint": {
                "S": getattr(self.ridge_joint, "S", None),
                "T": getattr(self.ridge_joint, "T", None),
                "n_eff": getattr(self.ridge_joint, "n_eff", None),
                "rss": getattr(self.ridge_joint, "rss", None),
                "coef": getattr(self.ridge_joint, "coef_", None),
            } if self.ridge_joint else None,
        }
        np.savez_compressed(path, state=json.dumps(state, default=lambda o: None))

    @staticmethod
    def load(path: str) -> "GazeRegressor":
        blob = np.load(path, allow_pickle=True)
        state = json.loads(str(blob["state"]))
        gr = GazeRegressor(
            mode=state["mode"], eye=state["eye"], degree=state["degree"], alpha=state["alpha"],
            triangulate=state["triangulate"], head_comp=state["head_comp"],
            screen_size=tuple(state["screen_size"]) if state["screen_size"] else None,
            column_map=ColumnMap(**state["column_map"]),
        )
        # Restore poly/ridge
        def restore_poly(poly_state):
            if not poly_state:
                return None
            pf = PolyFeatures(degree=gr.degree, include_interactions=True, include_bias=True)
            pf.feature_idx_ = [tuple(t) for t in poly_state["idx"]] if poly_state["idx"] else None
            pf.feature_names_ = poly_state["names"]
            pf.mu_ = np.array(poly_state["mu"]) if poly_state["mu"] is not None else None
            pf.sigma_ = np.array(poly_state["sigma"]) if poly_state["sigma"] is not None else None
            return pf

        gr.poly_left = restore_poly(state["poly_left"]) if state["poly_left"] else None
        gr.poly_right = restore_poly(state["poly_right"]) if state["poly_right"] else None
        gr.poly_joint = restore_poly(state["poly_joint"]) if state["poly_joint"] else None
        gr._in_names_left = state.get("poly_left", {}).get("in_names") if state.get("poly_left") else None
        gr._in_names_right = state.get("poly_right", {}).get("in_names") if state.get("poly_right") else None
        gr._in_names_joint = state.get("poly_joint", {}).get("in_names") if state.get("poly_joint") else None

        def restore_ridge(r_state):
            if not r_state:
                return None
            S = np.array(r_state["S"]) if r_state["S"] is not None else None
            T = np.array(r_state["T"]) if r_state["T"] is not None else None
            n_eff = float(r_state["n_eff"]) if r_state["n_eff"] is not None else 0.0
            rss = np.array(r_state["rss"]) if r_state["rss"] is not None else None
            coef = np.array(r_state["coef"]) if r_state["coef"] is not None else None
            if S is None or T is None:
                return None
            rr = OnlineRidge(n_features=S.shape[0], n_targets=T.shape[1], alpha=gr.alpha, bias_index=0)
            rr.S = S
            rr.T = T
            rr.n_eff = n_eff
            rr.rss = rss if rss is not None else np.zeros((T.shape[1],))
            rr.coef_ = coef
            rr.L_ = None  # will refactor when first predict
            return rr

        gr.ridge_left = restore_ridge(state["ridge_left"]) if state["ridge_left"] else None
        gr.ridge_right = restore_ridge(state["ridge_right"]) if state["ridge_right"] else None
        gr.ridge_joint = restore_ridge(state["ridge_joint"]) if state["ridge_joint"] else None
        return gr


# ------------------------------ CLI Utilities ------------------------------ #

def _cli():
    import argparse

    p = argparse.ArgumentParser(description="Part B: Polynomial Gaze Regressor (NumPy-only)")
    p.add_argument("--parquet", type=str, help="Input parquet/CSV for training or prediction")
    p.add_argument("--mode", type=str, default=Mode.BOTH_EYES_HEAD, choices=[Mode.LEFT_EYE, Mode.BOTH_EYES, Mode.BOTH_EYES_HEAD])
    p.add_argument("--eye", type=str, default="left", choices=["left", "right"], help="Eye to use in LEFT_EYE mode")
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1e-2)
    p.add_argument("--triangulate", action="store_true", help="Use two-branch fusion in BOTH_EYES*")
    p.add_argument("--no-triangulate", dest="triangulate", action="store_false")
    p.set_defaults(triangulate=True)
    p.add_argument("--min-conf", type=float, default=0.0, help="Drop frames below this confidence if available")
    p.add_argument("--predict-only", action="store_true", help="Skip fitting and only run predictions with --model")
    p.add_argument("--model", type=str, default=None, help="Path to load/save model .npz")
    p.add_argument("--out", type=str, default=None, help="Output model path (.npz). If omitted, don't save.")
    args = p.parse_args()

    if args.predict_only:
        assert args.model, "--model is required with --predict-only"
        reg = GazeRegressor.load(args.model)
        df = pd.read_parquet(args.parquet)
        preds = []
        for _, r in df.iterrows():
            y, s = reg.predict_from_row(r, return_std=True)
            preds.append([y[0], y[1], s[0], s[1]])
        out = os.path.splitext(args.parquet)[0] + ".pred.csv"
        pd.DataFrame(preds, columns=["x_hat", "y_hat", "sx", "sy"]).to_csv(out, index=False)
        print(f"Wrote predictions to {out}")
        return

    # Train
    reg = GazeRegressor(mode=args.mode, eye=args.eye, degree=args.degree, alpha=args.alpha, triangulate=args.triangulate)
    reg.fit_from_parquet(args.parquet, min_conf=args.min_conf)
    if args.out:
        reg.save(args.out)
        print(f"Saved model to {args.out}")


if __name__ == "__main__":
    _cli()
