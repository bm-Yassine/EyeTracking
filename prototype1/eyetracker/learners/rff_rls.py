# File: eyetracker/learners/rff_rls.py
# NumPy-only core (RFF + Ridge + RLS + Huber IRLS) for per-user gaze mapping.
# Optional pandas-based parquet loader for Part A calibration data.
# Author: You (integrated for Part B)
# License: same as your project

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Any

# ---------- Utilities

def _to_numpy(x: Any, copy: bool = False) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.copy() if copy else x
    return np.asarray(x, dtype=np.float64).copy() if copy else np.asarray(x, dtype=np.float64)

def _safe_cholesky(A: np.ndarray, jitter: float = 1e-8, max_tries: int = 5) -> np.ndarray:
    # For stabilized ridge closed-form fit
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            A = A + jitter * np.eye(A.shape[0], dtype=A.dtype)
            jitter *= 10.0
    # Last resort
    return np.linalg.cholesky(A + 1e-6 * np.eye(A.shape[0], dtype=A.dtype))

def _sin_cos_deg(deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(deg)
    return np.stack([np.sin(rad), np.cos(rad)], axis=-1).reshape(deg.shape[0], -1) if deg.ndim == 2 else np.array([np.sin(rad), np.cos(rad)], dtype=np.float64)

# ---------- Random Fourier Features

@dataclass
class RFFConfig:
    dim: int = 256           # number of random features (per output mapping)
    lengthscale: float = 1.0 # kernel lengthscale (RBF sigma)
    seed: Optional[int] = 0

class RFFProjector:
    """Random Fourier Features for an RBF kernel approximation."""
    def __init__(self, input_dim: int, cfg: RFFConfig):
        self.input_dim = int(input_dim)
        self.cfg = cfg
        rs = np.random.RandomState(cfg.seed if cfg.seed is not None else None)
        # w ~ N(0, 1/l^2 I)
        self.W = rs.normal(loc=0.0, scale=1.0/cfg.lengthscale, size=(self.input_dim, cfg.dim))
        self.b = rs.uniform(low=0.0, high=2.0*np.pi, size=(cfg.dim,))
        self.scale = np.sqrt(2.0 / cfg.dim)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _to_numpy(X)
        Z = X @ self.W + self.b
        return self.scale * np.cos(Z)  # [N, D]

# ---------- RLS (2D outputs) with robust (Huber) weighted updates

@dataclass
class RLSConfig:
    ridge: float = 1e-3        # L2 on weights (applies at batch fit time)
    forgetting: float = 1.0    # >=1.0 ; 1.0=no forgetting; >1.0 forgets history
    huber_delta: float = 0.02  # robust threshold in target units (e.g., normalized screen)
    irls_iters: int = 2        # passes for robust update events

class RFFRLSRegressor:
    """
    Two-output regressor y in R^2 (screen x,y), using:
      - RFF feature map phi(x) in R^D
      - Batch ridge closed-form fit
      - Online RLS updates
      - Robust updates via Huber IRLS (per-sample)
    """
    def __init__(self, input_dim: int, rff: RFFConfig = RFFConfig(), rls: RLSConfig = RLSConfig()):
        self.input_dim = int(input_dim)
        self.rff = rff
        self.rls = rls

        self.proj = RFFProjector(self.input_dim, rff)
        D = rff.dim
        self.W = np.zeros((D, 2), dtype=np.float64)    # weights for (x,y)
        self.P = (1.0 / rls.ridge) * np.eye(D, dtype=np.float64)  # inverse covariance for RLS

    # ----- Feature transform
    def _phi(self, X: np.ndarray) -> np.ndarray:
        return self.proj.transform(X)

    # ----- Batch fit (use your Part A calibration)
    def fit_batch(self, X: np.ndarray, Y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Closed-form ridge on transformed features.
        X: [N, F], Y: [N, 2], weights optional [N]
        """
        X = _to_numpy(X); Y = _to_numpy(Y)
        Phi = self._phi(X)  # [N, D]
        if sample_weight is not None:
            w = _to_numpy(sample_weight).reshape(-1, 1)  # [N,1]
            Phi_w = Phi * w
            A = Phi_w.T @ Phi + self.rls.ridge * np.eye(Phi.shape[1], dtype=np.float64)
            B = Phi_w.T @ Y
        else:
            A = Phi.T @ Phi + self.rls.ridge * np.eye(Phi.shape[1], dtype=np.float64)
            B = Phi.T @ Y
        # Solve A W = B
        L = _safe_cholesky(A)
        self.W = np.linalg.solve(L.T, np.linalg.solve(L, B))
        # Initialize P ~ (A)^-1 for RLS continuity
        self.P = np.linalg.inv(A)

    # ----- Predict
    def predict(self, X: np.ndarray, return_var: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = _to_numpy(X)
        Phi = self._phi(X)  # [N, D]
        Yhat = Phi @ self.W  # [N, 2]
        if not return_var:
            return Yhat, None
        # Approx predictive variance proxy: diag( Phi P Phi^T ) per output (use same scalar for x,y)
        v = np.einsum("nd,dk,nk->n", Phi, self.P, Phi)  # [N]
        # Broadcast to 2 dims so we can weight per-axis similarly
        V = np.stack([v, v], axis=1)
        return Yhat, V

    # ----- One-sample weighted RLS update
    def update_rls(self, x: np.ndarray, y: np.ndarray, weight: float = 1.0) -> None:
        """
        Weighted RLS (per sample):
            s = (forgetting / weight) + phi^T P phi
            K = P phi / s
            W <- W + K (y - phi^T W)
            P <- (P - K phi^T P) / forgetting
        """
        phi = self._phi(x.reshape(1, -1)).reshape(-1, 1)  # [D,1]
        y = _to_numpy(y).reshape(1, 2)                    # [1,2]

        denom = (self.rls.forgetting / max(weight, 1e-8)) + float(phi.T @ self.P @ phi)
        K = (self.P @ phi) / denom                        # [D,1]
        resid = y - (phi.T @ self.W)                      # [1,2]
        self.W = self.W + K @ resid                       # [D,2]
        self.P = (self.P - K @ (phi.T @ self.P)) / self.rls.forgetting

        # Symmetrize small numeric drift
        self.P = 0.5 * (self.P + self.P.T)

    # ----- Robust (Huber) IRLS wrapper for a single event
    def robust_update(self, x: np.ndarray, y: np.ndarray, base_weight: float = 1.0) -> None:
        """
        For a known-gaze event (x -> y), run IRLS passes:
            alpha = 1 if ||e|| <= delta else delta / ||e||
        and call weighted RLS with weight = base_weight * alpha.
        """
        x = _to_numpy(x); y = _to_numpy(y)
        delta = self.rls.huber_delta
        iters = max(1, int(self.rls.irls_iters))
        for _ in range(iters):
            yhat, _ = self.predict(x.reshape(1, -1), return_var=False)
            e = y - yhat.reshape(-1)
            e_norm = float(np.linalg.norm(e) + 1e-12)
            alpha = 1.0 if e_norm <= delta else (delta / e_norm)
            self.update_rls(x, y, weight=base_weight * alpha)

# ---------- Head compensation & feature building

def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float64)

def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)

def compensate_eye_uv_by_head(u: float, v: float, yaw_deg: float, pitch_deg: float) -> Tuple[float, float]:
    """
    Approx inverse head rotation for eye-centered (u,v) with pinhole z=1.
    Apply R = R_y(-yaw) R_x(-pitch), then project back to (u', v').
    """
    vec = np.array([u, v, 1.0], dtype=np.float64)
    Ry = _rot_y(-np.deg2rad(yaw_deg))
    Rx = _rot_x(-np.deg2rad(pitch_deg))
    v3 = Ry @ (Rx @ vec)
    if abs(v3[2]) < 1e-6:
        return u, v
    return float(v3[0] / v3[2]), float(v3[1] / v3[2])

@dataclass
class FeatureSpec:
    """
    Configure which columns from Part A frames/calibration to use.
    Adjust names if your parquet schema differs.
    """
    # Required target columns in calibration parquet
    target_xy: Tuple[str, str] = ("target_x", "target_y")

    # Eye features (normalized per-eye)
    left_uv: Tuple[str, str] = ("left_u", "left_v")
    right_uv: Tuple[str, str] = ("right_u", "right_v")

    # Optional confidences (0..1)
    left_conf: Optional[str] = "left_conf"
    right_conf: Optional[str] = "right_conf"

    # Head pose (degrees + 3D mm)
    head_angles: Tuple[str, str, str] = ("yaw_deg", "pitch_deg", "roll_deg")
    head_xyz_mm: Tuple[str, str, str] = ("head_x_mm", "head_y_mm", "head_z_mm")

    # Optional inter-pupil distance (normalized or px/mm)
    ipd_name: Optional[str] = "ipd_norm"

class FeatureBuilder:
    """
    Build F_i feature vectors for different modes.
    Modes:
      - '1eye-left' or '1eye-right'
      - '2eyes'
      - '2eyes_head' (inverse head rotation of eye (u,v) + append angle sin/cos + ipd/head_z if available)
    """
    def __init__(self, spec: FeatureSpec = FeatureSpec()):
        self.spec = spec

    def _grab(self, rec: Dict[str, Any], name: str, default: float = 0.0) -> float:
        val = rec.get(name, default)
        try:
            return float(val)
        except Exception:
            return default

    def build_one(self, rec: Dict[str, Any], mode: Literal["1eye-left","1eye-right","2eyes","2eyes_head"]) -> Tuple[np.ndarray, Dict[str, float]]:
        s = self.spec
        meta: Dict[str, float] = {}

        # Eye UV
        lu, lv = self._grab(rec, s.left_uv[0]), self._grab(rec, s.left_uv[1])
        ru, rv = self._grab(rec, s.right_uv[0]), self._grab(rec, s.right_uv[1])

        # Conf
        lconf = self._grab(rec, s.left_conf, 1.0) if s.left_conf else 1.0
        rconf = self._grab(rec, s.right_conf, 1.0) if s.right_conf else 1.0
        meta["lconf"] = lconf; meta["rconf"] = rconf

        # Head
        yaw = self._grab(rec, s.head_angles[0], 0.0)
        pitch = self._grab(rec, s.head_angles[1], 0.0)
        roll = self._grab(rec, s.head_angles[2], 0.0)
        hx = self._grab(rec, s.head_xyz_mm[0], 0.0)
        hy = self._grab(rec, s.head_xyz_mm[1], 0.0)
        hz = self._grab(rec, s.head_xyz_mm[2], 600.0)  # assume ~60cm default
        meta.update(dict(yaw=yaw, pitch=pitch, roll=roll, hx=hx, hy=hy, hz=hz))

        ipd = self._grab(rec, s.ipd_name, 0.065) if s.ipd_name else 0.065
        meta["ipd"] = ipd

        if mode == "1eye-left":
            f = np.array([lu, lv, 1.0], dtype=np.float64)
            return f, meta

        if mode == "1eye-right":
            f = np.array([ru, rv, 1.0], dtype=np.float64)
            return f, meta

        if mode == "2eyes":
            # plain concat + simple invariants
            f = np.array([lu, lv, ru, rv,
                          ru - lu, rv - lv,  # binocular disparity proxy
                          ipd, 1.0], dtype=np.float64)
            return f, meta

        if mode == "2eyes_head":
            # compensate (inverse) yaw/pitch on each eye's (u,v)
            clu, clv = compensate_eye_uv_by_head(lu, lv, yaw, pitch)
            cru, crv = compensate_eye_uv_by_head(ru, rv, yaw, pitch)
            # append sin/cos of angles and coarse depth
            sincos = np.array([
                np.sin(np.deg2rad(yaw)),  np.cos(np.deg2rad(yaw)),
                np.sin(np.deg2rad(pitch)),np.cos(np.deg2rad(pitch)),
                np.sin(np.deg2rad(roll)), np.cos(np.deg2rad(roll)),
            ], dtype=np.float64)
            f = np.array([clu, clv, cru, crv,
                          cru - clu, crv - clv,
                          ipd, hz * 1e-3, 1.0], dtype=np.float64)
            f = np.concatenate([f, sincos], axis=0)
            return f, meta

        raise ValueError(f"Unknown mode: {mode}")

# ---------- Two-eye fusion ("triangulation" by precision weighting; optional geometric hook)

def fuse_two_predictions(pL: np.ndarray, vL: np.ndarray, pR: np.ndarray, vR: np.ndarray,
                         wL: float = 1.0, wR: float = 1.0, eps: float = 1e-9) -> np.ndarray:
    """
    Precision-weighted fusion of two 2D predictions.
    pL, pR: [2], vL, vR: [2] variance proxies; wL, wR: extra confidences (0..1).
    """
    invL = wL / (vL + eps)
    invR = wR / (vR + eps)
    num = invL * pL + invR * pR
    den = invL + invR + eps
    return num / den

# Placeholder for true 3D triangulation if you later pass both eye origins/rays + camera model.
# For now we rely on precision-weighting above to remain NumPy-only.

# ---------- Manager: load calibration, keep 3 variants, support online updates

class CalibrationLoader:
    """
    Optional parquet -> arrays loader. Requires pandas/pyarrow if you use it.
    Otherwise pass arrays directly to RFFRLSRegressor.fit_batch.
    """
    def __init__(self, spec: FeatureSpec = FeatureSpec()):
        self.spec = spec
        try:
            import pandas as _pd  # noqa
            self._pandas_available = True
        except Exception:
            self._pandas_available = False

    def load_parquet(self, path: str, mode: str, which_eye: Literal["left","right"] = "left",
                     feature_builder: Optional[FeatureBuilder] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict[str,float]]]:
        if not self._pandas_available:
            raise RuntimeError("pandas/pyarrow are required to read parquet files. Install or pass arrays directly.")
        import pandas as pd
        df = pd.read_parquet(path)
        fb = feature_builder or FeatureBuilder(self.spec)

        X_list: List[np.ndarray] = []
        M_list: List[Dict[str,float]] = []
        tx, ty = self.spec.target_xy

        # Allow '1eye' with chosen eye
        if mode == "1eye":
            true_mode = "1eye-left" if which_eye == "left" else "1eye-right"
        else:
            true_mode = mode

        for rec in df.to_dict("records"):
            f, meta = fb.build_one(rec, true_mode)
            X_list.append(f)
            M_list.append(meta)

        X = np.stack(X_list, axis=0)
        Y = df[[tx, ty]].to_numpy(dtype=np.float64)
        return X, Y, M_list

class GazeRegressorManager:
    """
    Holds up to 3 models:
      - '1eye-left' and/or '1eye-right' (you choose which one to instantiate)
      - '2eyes'
      - '2eyes_head'
    Exposes:
      - fit_from_calibration(...)
      - predict(frame)
      - recalibrate_event(frame, known_xy, mode=..., which_eye=..., robust=True, weight=1.0)
    """
    def __init__(self,
                 input_dims: Dict[str, int],
                 rff_cfg: RFFConfig = RFFConfig(),
                 rls_cfg: RLSConfig = RLSConfig(),
                 feature_spec: FeatureSpec = FeatureSpec()):
        """
        input_dims: dict like
            {
              "1eye-left": 3,
              "1eye-right": 3,
              "2eyes": 8,
              "2eyes_head": 14
            }
        """
        self.rff_cfg = rff_cfg
        self.rls_cfg = rls_cfg
        self.spec = feature_spec
        self.fb = FeatureBuilder(self.spec)

        self.models: Dict[str, RFFRLSRegressor] = {}
        for mode, dim in input_dims.items():
            self.models[mode] = RFFRLSRegressor(dim, rff_cfg, rls_cfg)

        self.cal_loader = CalibrationLoader(self.spec)

    def fit_from_parquet(self, path: str, mode: str, which_eye: Literal["left","right"]="left",
                         sample_weight: Optional[np.ndarray] = None) -> None:
        X, Y, _ = self.cal_loader.load_parquet(path, mode, which_eye, self.fb)
        key = ( "1eye-left" if (mode=="1eye" and which_eye=="left") else
                "1eye-right" if (mode=="1eye" and which_eye=="right") else
                mode )
        if key not in self.models:
            raise KeyError(f"Model '{key}' not initialized in manager.")
        self.models[key].fit_batch(X, Y, sample_weight=sample_weight)

    def predict(self, frame: Dict[str, Any],
                mode: Literal["1eye-left","1eye-right","2eyes","2eyes_head"]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns (xy, info) where xy is [2] and info includes sub-preds/variances.
        """
        if mode in ("1eye-left", "1eye-right"):
            f, meta = self.fb.build_one(frame, mode)
            y, v = self.models[mode].predict(f.reshape(1, -1), return_var=True)
            return y.reshape(-1), {"var": v.reshape(-1), "meta": meta}

        if mode == "2eyes":
            fL, meta = self.fb.build_one(frame, "1eye-left")
            fR, _    = self.fb.build_one(frame, "1eye-right")
            yL, vL = self.models["1eye-left"].predict(fL.reshape(1, -1), return_var=True)
            yR, vR = self.models["1eye-right"].predict(fR.reshape(1, -1), return_var=True)
            lconf = meta.get("lconf", 1.0); rconf = meta.get("rconf", 1.0)
            fused = fuse_two_predictions(yL.reshape(-1), vL.reshape(-1), yR.reshape(-1), vR.reshape(-1), wL=lconf, wR=rconf)
            return fused, {"left": (yL.reshape(-1), vL.reshape(-1)),
                           "right": (yR.reshape(-1), vR.reshape(-1)),
                           "meta": meta}

        if mode == "2eyes_head":
            f, meta = self.fb.build_one(frame, "2eyes_head")
            y, v = self.models["2eyes_head"].predict(f.reshape(1, -1), return_var=True)
            return y.reshape(-1), {"var": v.reshape(-1), "meta": meta}

        raise ValueError(f"Unknown mode '{mode}'")

    def recalibrate_event(self, frame: Dict[str, Any], known_xy: Tuple[float, float],
                          mode: Literal["1eye-left","1eye-right","2eyes","2eyes_head"],
                          robust: bool = True, base_weight: float = 1.0) -> None:
        """
        Apply an online update when you know the user's gaze (e.g., button-held frame).
        """
        y = np.asarray(known_xy, dtype=np.float64)

        if mode in ("1eye-left","1eye-right"):
            f, _ = self.fb.build_one(frame, mode)
            if robust:
                self.models[mode].robust_update(f, y, base_weight)
            else:
                self.models[mode].update_rls(f, y, base_weight)
            return

        if mode == "2eyes":
            # Update each eye model with its own features and the same ground-truth y
            fL, _ = self.fb.build_one(frame, "1eye-left")
            fR, _ = self.fb.build_one(frame, "1eye-right")
            if robust:
                self.models["1eye-left"].robust_update(fL, y, base_weight)
                self.models["1eye-right"].robust_update(fR, y, base_weight)
            else:
                self.models["1eye-left"].update_rls(fL, y, base_weight)
                self.models["1eye-right"].update_rls(fR, y, base_weight)
            return

        if mode == "2eyes_head":
            f, _ = self.fb.build_one(frame, "2eyes_head")
            if robust:
                self.models["2eyes_head"].robust_update(f, y, base_weight)
            else:
                self.models["2eyes_head"].update_rls(f, y, base_weight)
            return

        raise ValueError(f"Unknown mode '{mode}'")
