"""HDF5 I/O for Stage 1: trace loading (with ΔF/F₀) and output writing."""
from __future__ import annotations

import h5py
import numpy as np
import warnings

from .config import Stage1Config

__all__ = ["load_traces_and_regressor", "write_stage1_outputs"]


def _ensure_TN(X: np.ndarray) -> np.ndarray:
    """Ensure traces are (T, N); transpose if obviously flipped."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"trace array must be 2D, got {X.shape}")
    if X.shape[0] < 400 and X.shape[1] >= 400 and X.shape[1] / max(X.shape[0], 1) > 2:
        return X.T
    return X


def _ensure_TD(L: np.ndarray | None) -> np.ndarray | None:
    """Ensure stimulus regressor is (T, D)."""
    if L is None:
        return None
    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        L = L[:, None]
    if L.ndim != 2:
        raise ValueError(f"stimulus regressor must be 2D, got {L.shape}")
    if L.shape[0] < 50 and L.shape[1] >= 50:
        return L.T
    return L


def _quantile_cols(x: np.ndarray, q: float) -> np.ndarray:
    """Column-wise nanquantile, NaN for all-NaN columns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanquantile(x, q, axis=0)
    out = np.asarray(out, dtype=float)
    out[np.all(~np.isfinite(x), axis=0)] = np.nan
    return out


def _rolling_quantile_baseline(x: np.ndarray, q: float, hw: int) -> np.ndarray:
    T, N = x.shape
    f0 = np.full((T, N), np.nan, dtype=float)
    for t in range(T):
        f0[t] = _quantile_cols(x[max(0, t - hw):min(T, t + hw + 1)], q)
    return f0


def _non_stim_mask(f: h5py.File, T: int) -> np.ndarray:
    """Boolean mask: True where no optogenetic stimulus is active."""
    mask = np.ones(T, dtype=bool)
    try:
        if "optogenetics/stim_matrix" in f:
            sm = _ensure_TD(np.array(f["optogenetics/stim_matrix"][()], dtype=float))
            if sm.shape[0] == T:
                mask = ~(np.nansum(np.abs(sm), axis=1) > 0)
    except Exception:
        pass
    return mask


def _apply_dff(f, x, sample_rate_hz, method, q, window_sec, eps):
    """Compute ΔF/F₀ using the chosen baseline method."""
    q = float(np.clip(q, 1e-6, 1 - 1e-6))
    eps = float(max(eps, 1e-12))
    method = str(method).strip().lower()

    if method == "quantile":
        f0 = _quantile_cols(x, q)
        f0 = np.where(np.isfinite(f0), f0, np.nanmedian(x, axis=0))
        return (x - f0[None, :]) / np.maximum(np.abs(f0[None, :]), eps)

    if method == "pre_stim":
        mask = _non_stim_mask(f, x.shape[0])
        f0 = _quantile_cols(x[mask] if np.any(mask) else x, q)
        f0 = np.where(np.isfinite(f0), f0, np.nanmedian(x, axis=0))
        return (x - f0[None, :]) / np.maximum(np.abs(f0[None, :]), eps)

    if method == "rolling_quantile":
        hw = max(1, int(round(0.5 * float(window_sec) * float(sample_rate_hz))))
        f0 = _rolling_quantile_baseline(x, q, hw)
        return (x - f0) / np.maximum(np.abs(f0), eps)

    raise ValueError(f"Unknown f0_method='{method}'")


# ── Public API ──

def _apply_f_over_f0(x, sample_rate_hz, method, q, window_sec, eps):
    """Compute F/F₀ — ratio baseline correction (signal stays positive)."""
    q = float(np.clip(q, 1e-6, 1 - 1e-6))
    eps = float(max(eps, 1e-12))
    method = str(method).strip().lower()

    if method == "quantile":
        f0 = _quantile_cols(x, q)
        f0 = np.where(np.isfinite(f0), f0, np.nanmedian(x, axis=0))
        return x / np.maximum(np.abs(f0[None, :]), eps)

    if method == "rolling_quantile":
        hw = max(1, int(round(0.5 * float(window_sec) * float(sample_rate_hz))))
        f0 = _rolling_quantile_baseline(x, q, hw)
        return x / np.maximum(np.abs(f0), eps)

    raise ValueError(f"Unknown f0_method='{method}' for F/F₀")


def load_traces_and_regressor(cfg: Stage1Config) -> np.ndarray:
    with h5py.File(cfg.h5_path, "r") as f:
        if cfg.trace_dataset not in f:
            raise KeyError(f"'{cfg.trace_dataset}' not in {cfg.h5_path}")
        X = _ensure_TN(np.array(f[cfg.trace_dataset][()], dtype=float))
        if cfg.use_dff:
            X = _apply_dff(f, X, cfg.sample_rate_hz, cfg.f0_method,
                           cfg.f0_quantile, cfg.f0_window_sec, cfg.f0_eps)
        elif getattr(cfg, 'use_f_over_f0', False):
            X = _apply_f_over_f0(X, cfg.sample_rate_hz, cfg.f0_method,
                                 cfg.f0_quantile, cfg.f0_window_sec, cfg.f0_eps)
    return X


# ── HDF5 write helpers ──

def _put(parent, name: str, arr, overwrite: bool) -> None:
    """Create or overwrite a gzip-compressed dataset."""
    if name in parent:
        if overwrite:
            del parent[name]
        else:
            return
    if "/" in name:
        parent.require_group("/".join(name.split("/")[:-1]))
    parent.create_dataset(name, data=arr, compression="gzip", compression_opts=4)


def _put_scalar_or_array(g, key: str, val, overwrite: bool) -> None:
    """Write *val* as dataset (ndarray) or attribute (scalar)."""
    if isinstance(val, np.ndarray):
        _put(g, key, val.astype(np.float32), overwrite)
        if key in g.attrs:
            del g.attrs[key]
    else:
        g.attrs[key] = float(val)
        if key in g:
            del g[key]


def write_stage1_outputs(cfg: Stage1Config, out: dict, overwrite: bool = True) -> None:
    with h5py.File(cfg.h5_path, "a") as f:
        # Smoothed states
        for ds, key in [(cfg.out_u_mean, "u_mean"), (cfg.out_u_var, "u_var"),
                        (cfg.out_c_mean, "c_mean"), (cfg.out_c_var, "c_var")]:
            _put(f, ds, out[key], overwrite)

        _put(f, "stage1/u_std", np.sqrt(np.clip(out["u_var"].astype(np.float32), 0, None)), overwrite)
        _put(f, "stage1/c_std", np.sqrt(np.clip(out["c_var"].astype(np.float32), 0, None)), overwrite)

        if out.get("fit_mask") is not None:
            _put(f, "stage1/fit_mask", np.asarray(out["fit_mask"], dtype=np.uint8), overwrite)

        # Parameter group
        if cfg.out_params in f and overwrite:
            del f[cfg.out_params]
        g = f.require_group(cfg.out_params)

        for key in ("alpha", "beta", "sigma_y", "sigma_u"):
            _put(g, key, out[key], overwrite)
        for key in ("sigma_y", "sigma_u"):
            _put(g, f"{key}2", np.square(out[key].astype(np.float32)), overwrite)

        if out.get("ll_hist") is not None:
            _put(g, "ll_hist", np.asarray(out["ll_hist"], dtype=np.float32), overwrite)
        if out.get("trace_mean") is not None:
            _put(g, "trace_mean", np.asarray(out["trace_mean"], dtype=np.float32), overwrite)

        # Scalar-or-array parameters
        for key in ("rho", "lambda_c", "sigma_c"):
            _put_scalar_or_array(g, key, out[key], overwrite)

        # sigma_c² (mirrors sigma_c storage format)
        sc = out["sigma_c"]
        if isinstance(sc, np.ndarray):
            _put(g, "sigma_c2", np.square(sc.astype(np.float32)), overwrite)
            if "sigma_c2" in g.attrs:
                del g.attrs["sigma_c2"]
        else:
            g.attrs["sigma_c2"] = float(sc) ** 2
            if "sigma_c2" in g:
                del g["sigma_c2"]

        # Config attrs
        for attr, val in [
            ("sample_rate_hz", cfg.sample_rate_hz),
            ("tau_c_init_sec", cfg.tau_c_init_sec),
            ("tau_u_init_sec", cfg.tau_u_init_sec),
            ("sigma_u_scale_init", cfg.sigma_u_scale_init),
            ("sigma_c_init", cfg.sigma_c_init),
            ("sigma_y_floor", cfg.sigma_y_floor),
            ("eps_var", cfg.eps_var),
            ("fix_alpha", bool(cfg.fix_alpha)),
            ("alpha_value", float(cfg.alpha_value)),
            ("em_max_iters", cfg.em_max_iters),
            ("em_tol_rel_ll", cfg.em_tol_rel_ll),
            ("share_rho", cfg.share_rho),
            ("share_lambda_c", cfg.share_lambda_c),
            ("share_sigma_c", cfg.share_sigma_c),
            ("center_traces", bool(cfg.center_traces)),
            ("use_dff", bool(cfg.use_dff)),
            ("f0_method", str(cfg.f0_method)),
            ("f0_quantile", float(cfg.f0_quantile)),
            ("f0_window_sec", float(cfg.f0_window_sec)),
            ("f0_eps", float(cfg.f0_eps)),
        ]:
            g.attrs[attr] = val
        g.attrs["rho_clip"] = np.asarray(cfg.rho_clip, dtype=np.float32)
        g.attrs["lambda_clip"] = np.asarray(cfg.lambda_clip, dtype=np.float32)

        if out.get("ll_hist") is not None:
            ll = np.asarray(out["ll_hist"], dtype=float)
            if ll.size >= 2:
                g.attrs["ll_rel_change_last"] = abs(float(ll[-1]) - float(ll[-2])) / max(abs(float(ll[-2])), 1e-12)
