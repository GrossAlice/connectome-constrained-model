from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List

from .behavior_decoder_eval import (
    behaviour_all_neurons_prediction,
    evaluate_training_decoder,
)
from . import get_stage2_logger
from ._utils import _r2, _cfg_val
from .model import Stage2ModelPT

__all__ = [
    "compute_onestep",
    "compute_cv_reg",
    "compute_free_run",
    "free_run_stochastic",
    "loo_forward_simulate",
    "run_loo_all",
    "choose_loo_subset",
    "compute_current_decomposition",
    "evaluate_training_decoder",
    "behaviour_all_neurons_prediction",
    "run_full_evaluation",
]

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xv, yv = x[m], y[m]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m].astype(np.float64) - y_pred[m].astype(np.float64)) ** 2)))


def _per_neuron_metrics(
    u_true: np.ndarray, u_pred: np.ndarray, indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = u_true.shape[1]
    r2, corr, rmse = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    for i in indices:
        r2[i]   = _r2(u_true[:, i], u_pred[:, i])
        corr[i] = _pearson(u_true[:, i], u_pred[:, i])
        rmse[i] = _rmse(u_true[:, i], u_pred[:, i])
    return r2, corr, rmse

def _get_clip_bounds(model) -> Tuple[Optional[float], Optional[float]]:
    clip = getattr(model, "u_clip", (None, None))
    if isinstance(clip, (tuple, list)) and len(clip) == 2:
        lo, hi = clip
    else:
        lo, hi = None, None
    return (float(lo) if lo is not None else None,
            float(hi) if hi is not None else None)


def _clamp(x: torch.Tensor, lo: Optional[float], hi: Optional[float]) -> torch.Tensor:
    if lo is not None and hi is not None:
        return x.clamp(lo, hi)
    if lo is not None:
        return x.clamp(min=lo)
    if hi is not None:
        return x.clamp(max=hi)
    return x


def _teacher_forced_prior(
    model: Stage2ModelPT, u: torch.Tensor, gating, stim,
) -> torch.Tensor:
    T, N = u.shape
    device = u.device
    ones = torch.ones(N, device=device)
    prior_mu = torch.zeros_like(u)
    prior_mu[0] = u[0]
    s_sv  = torch.zeros(N, model.r_sv,  device=device)
    s_dcv = torch.zeros(N, model.r_dcv, device=device)
    with torch.no_grad():
        for t in range(1, T):
            g = gating[t - 1] if gating is not None else ones
            s = stim[t - 1]   if stim   is not None else None
            prior_mu[t], s_sv, s_dcv = model.prior_step(u[t - 1], s_sv, s_dcv, g, s)
    return prior_mu


def _resolve_motor_indices(data: Dict[str, Any], N: int) -> list[int]:
    # 1. Pre-resolved indices stored during data loading
    pre = data.get("motor_neurons")
    if pre is not None and len(pre) > 0:
        return sorted({int(i) for i in pre if 0 <= int(i) < N})

    cfg = data.get("_cfg")
    if cfg is None or getattr(cfg, "motor_neurons", None) is None:
        return []

    motor = cfg.motor_neurons
    idx: list[int] = []
    unresolved: list[str] = []
    for m in motor:
        try:
            v = int(m)
            if 0 <= v < N:
                idx.append(v)
        except (ValueError, TypeError):
            unresolved.append(str(m).strip())

    # 3. Name-based resolution
    if unresolved:
        labels = data.get("neuron_labels", [])
        if labels:
            lbl_upper = [str(l).strip().upper() for l in labels[:N]]
            for name in unresolved:
                key = name.upper()
                if key in lbl_upper:
                    idx.append(lbl_upper.index(key))

    return sorted(set(idx))


def _ar1_smooth(u: torch.Tensor, lam: torch.Tensor, I0: torch.Tensor) -> np.ndarray:
    T, N = u.shape
    out = torch.zeros_like(u)
    out[0] = u[0]
    with torch.no_grad():
        for t in range(1, T):
            out[t] = (1.0 - lam) * u[t - 1] + lam * I0
    return out.cpu().numpy()


def compute_onestep(model: Stage2ModelPT, data: Dict[str, Any]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    N = u.shape[1]

    prior_mu = _teacher_forced_prior(model, u, data.get("gating"), data.get("stim"))
    u_np, mu_np = u.cpu().numpy(), prior_mu.cpu().numpy()

    # AR(1)-only baseline (leak smoothing, no network)
    ar1_np = _ar1_smooth(u, model.lambda_u, model.I0)

    r2, corr, rmse = _per_neuron_metrics(u_np[1:], mu_np[1:], list(range(N)))

    du_true = u_np[1:] - u_np[:-1]
    du_pred = mu_np[1:] - u_np[:-1]
    r2_delta   = np.array([_r2(du_true[:, i], du_pred[:, i])      for i in range(N)])
    corr_delta = np.array([_pearson(du_true[:, i], du_pred[:, i]) for i in range(N)])

    return {
        "prior_mu": mu_np,
        "ar1_mu": ar1_np,
        "r2": r2, "corr": corr, "rmse": rmse,
        "r2_delta": r2_delta, "corr_delta": corr_delta,
    }


# --------------------------------------------------------------------------- #
#  Per-neuron shrinkage CV-reg                                                  #
# --------------------------------------------------------------------------- #

def _fit_ar1_ols(x: np.ndarray, y: np.ndarray):
    """Fit y = a*x + b by OLS.  Returns (a, b)."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return 0.0, float(np.nanmean(y)) if np.any(np.isfinite(y)) else 0.0
    xv, yv = x[valid].astype(np.float64), y[valid].astype(np.float64)
    x_m, y_m = xv.mean(), yv.mean()
    var_x = ((xv - x_m) ** 2).mean()
    cov_xy = ((xv - x_m) * (yv - y_m)).mean()
    a = cov_xy / max(var_x, 1e-12)
    b = y_m - a * x_m
    return float(a), float(b)


def compute_cv_reg(
    u_np: np.ndarray,
    model_mu: np.ndarray,
    *,
    n_folds: int = 5,
    log_reg_min: float = -3.0,
    log_reg_max: float = 5.0,
    n_grid: int = 50,
) -> Dict[str, Any]:
    """Per-neuron shrinkage CV-reg: blend AR(1) <-> full-model prediction.

    For each neuron *i*, find optimal alpha_i in [0, 1] via temporal-block
    k-fold CV:

        u_reg = u_ar1 + alpha_i * (u_model - u_ar1)

    where alpha_i = 1 / (1 + reg_i*) and reg_i* minimises held-out
    one-step MSE.

    Parameters
    ----------
    u_np : (T, N) ground-truth neural activity
    model_mu : (T, N) teacher-forced one-step model prediction
    n_folds : number of contiguous temporal blocks
    log_reg_min, log_reg_max : log10 bounds for regularisation grid
    n_grid : number of grid points (log-spaced)

    Returns
    -------
    dict with keys:
        cv_reg_mu : (T, N)  blended prediction using best alpha
        alpha     : (N,)    per-neuron optimal alpha (0 = AR1, 1 = model)
        best_reg  : (N,)    per-neuron optimal regularisation strength
        reg_grid  : (n_grid,) regularisation values searched
        alpha_grid: (n_grid,) corresponding alpha = 1/(1+reg)
        cv_mse_all: (N, n_grid) mean CV-MSE at each grid point
        r2        : (N,)    R^2 of the blended prediction
        r2_improvement : (N,)  R^2(cv_reg) - R^2(raw model)
    """
    T, N = u_np.shape

    # Regularisation grid (log-spaced)
    reg_grid = np.logspace(log_reg_min, log_reg_max, n_grid)
    alpha_grid = 1.0 / (1.0 + reg_grid)  # high reg -> alpha near 0 (AR1)

    # Temporal transitions: t -> t+1
    u_prev = u_np[:-1]          # (T-1, N)
    u_next = u_np[1:]           # (T-1, N)
    model_pred = model_mu[1:]   # (T-1, N)
    n_trans = T - 1

    # Create contiguous temporal-block fold assignments
    fold_ids = np.zeros(n_trans, dtype=int)
    fold_size = n_trans // n_folds
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else n_trans
        fold_ids[start:end] = k

    # Per-neuron CV
    best_alpha = np.zeros(N, dtype=np.float64)
    best_reg = np.full(N, np.inf, dtype=np.float64)
    cv_mse_all = np.full((N, n_grid), np.nan, dtype=np.float64)

    for i in range(N):
        fold_mse = np.full((n_folds, n_grid), np.nan, dtype=np.float64)

        for k in range(n_folds):
            train_mask = fold_ids != k
            test_mask = fold_ids == k
            if test_mask.sum() == 0:
                continue

            # Fit AR(1) on training portion: u_{t+1,i} = a * u_{t,i} + b
            a, b = _fit_ar1_ols(u_prev[train_mask, i], u_next[train_mask, i])
            ar1_test = a * u_prev[test_mask, i] + b
            model_test = model_pred[test_mask, i]
            target_test = u_next[test_mask, i]

            # Correction vector (model - AR1) on held-out block
            correction = model_test - ar1_test

            # Evaluate each regularisation level
            for gi, alpha in enumerate(alpha_grid):
                blend = ar1_test + alpha * correction
                resid = target_test - blend
                valid = np.isfinite(resid)
                if valid.sum() > 0:
                    fold_mse[k, gi] = float(np.mean(resid[valid] ** 2))

        # Average across folds and pick best
        mean_mse = np.nanmean(fold_mse, axis=0)
        cv_mse_all[i] = mean_mse

        if np.all(np.isnan(mean_mse)):
            best_alpha[i] = 0.0
            best_reg[i] = np.inf
        else:
            best_idx = int(np.nanargmin(mean_mse))
            best_alpha[i] = alpha_grid[best_idx]
            best_reg[i] = reg_grid[best_idx]

    # Build final blended prediction using full-data AR(1)
    cv_reg_mu = np.zeros_like(u_np)
    cv_reg_mu[0] = u_np[0]
    for i in range(N):
        a, b = _fit_ar1_ols(u_prev[:, i], u_next[:, i])
        ar1_full = a * u_prev[:, i] + b
        cv_reg_mu[1:, i] = ar1_full + best_alpha[i] * (model_pred[:, i] - ar1_full)

    # Per-neuron R^2 of blended prediction
    r2_cv = np.array([_r2(u_np[1:, i], cv_reg_mu[1:, i]) for i in range(N)])
    r2_raw = np.array([_r2(u_np[1:, i], model_mu[1:, i]) for i in range(N)])

    _logger = get_stage2_logger()
    _n_shrunk = int((best_alpha < 0.5).sum())
    _med_alpha = float(np.median(best_alpha))
    _med_reg = float(np.median(best_reg[np.isfinite(best_reg)]))
    _logger.info("cv_reg_done", N=N, n_folds=n_folds,
                 median_alpha=_med_alpha, n_shrunk=_n_shrunk,
                 median_reg=_med_reg,
                 r2_raw_median=float(np.nanmedian(r2_raw)),
                 r2_cv_median=float(np.nanmedian(r2_cv)))

    return {
        "cv_reg_mu": cv_reg_mu,
        "alpha": best_alpha,
        "best_reg": best_reg,
        "reg_grid": reg_grid,
        "alpha_grid": alpha_grid,
        "cv_mse_all": cv_mse_all,
        "r2": r2_cv,
        "r2_raw": r2_raw,
        "r2_improvement": r2_cv - r2_raw,
    }


def compute_free_run(model: Stage2ModelPT, data: Dict[str, Any]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    u0 = data["u_stage1"].to(device)
    T, N = u0.shape
    gating, stim = data.get("gating"), data.get("stim")
    lo, hi = _get_clip_bounds(model)
    ones = torch.ones(N, device=device)

    motor_idx = _resolve_motor_indices(data, N)
    conditioned = 0 < len(motor_idx) < N
    motor_mask = torch.zeros(N, dtype=torch.bool, device=device)
    if motor_idx:
        motor_mask[motor_idx] = True

    with torch.no_grad():
        u_free = torch.zeros_like(u0)
        u_free[0] = u0[0]
        s_sv  = torch.zeros(N, model.r_sv,  device=device)
        s_dcv = torch.zeros(N, model.r_dcv, device=device)
        for t in range(1, T):
            g = gating[t - 1] if gating is not None else ones
            s = stim[t - 1]   if stim   is not None else None
            u_prev = u0[t - 1].clone() if conditioned else u_free[t - 1]
            if conditioned:
                u_prev[motor_mask] = u_free[t - 1, motor_mask]
            u_next, s_sv, s_dcv = model.prior_step(u_prev, s_sv, s_dcv, g, s)
            u_next = _clamp(u_next, lo, hi)
            if conditioned:
                u_free[t] = u0[t]
                u_free[t, motor_mask] = u_next[motor_mask]
            else:
                u_free[t] = u_next

    u_np, uf_np = u0.cpu().numpy(), u_free.cpu().numpy()
    eval_idx = motor_idx if conditioned else list(range(N))
    r2, corr, rmse = _per_neuron_metrics(u_np[1:], uf_np[1:], eval_idx)

    return {
        "u_free": uf_np,
        "r2": r2, "corr": corr, "rmse": rmse,
        "motor_idx": np.array(motor_idx, dtype=int),
        "mode": "motor_conditioned" if conditioned else "autonomous",
    }


def free_run_stochastic(
    model: Stage2ModelPT,
    data: Dict[str, Any],
    n_samples: int = 5,
) -> Dict[str, Any]:
    """Full-brain stochastic free-run: all neurons evolve autonomously with
    learned process noise.

    At each time step *every* neuron receives noise::

        u_{t+1,i} = mu_{t+1,i} + sigma_{t,i} * eps,  eps ~ N(0,1)

    where ``sigma_{t,i}`` is state-dependent (heteroscedastic) or constant
    (homoscedastic) depending on the model.

    Parameters
    ----------
    model : Stage2ModelPT
    data  : dict with ``u_stage1``, optional ``gating``, ``stim``
    n_samples : int
        Number of independent sample trajectories.

    Returns
    -------
    dict with keys:
        samples    : np.ndarray, shape ``(n_samples, T, N)``
        mean       : np.ndarray, shape ``(T, N)`` — ensemble mean
        std        : np.ndarray, shape ``(T, N)`` — ensemble std
        ci_lo      : np.ndarray, shape ``(T, N)`` — 2.5th percentile
        ci_hi      : np.ndarray, shape ``(T, N)`` — 97.5th percentile
        r2_per_sample : np.ndarray, shape ``(n_samples, N)`` — per-sample R²
        r2_mean    : np.ndarray, shape ``(N,)`` — R² of ensemble mean
    """
    device = next(model.parameters()).device
    u0 = data["u_stage1"].to(device)
    T, N = u0.shape
    gating = data.get("gating")
    stim = data.get("stim")
    lo, hi = _get_clip_bounds(model)
    ones = torch.ones(N, device=device)
    is_hetero = (getattr(model, "_noise_mode", "homoscedastic") == "heteroscedastic")

    # Pre-compute constant sigma for homoscedastic models
    if not is_hetero:
        sigma_const = model.sigma_at().detach()  # (N,)

    samples_np = np.zeros((n_samples, T, N), dtype=np.float32)

    with torch.no_grad():
        for k in range(n_samples):
            u_cur = u0[0].clone()
            samples_np[k, 0] = u_cur.cpu().numpy()
            s_sv = torch.zeros(N, model.r_sv, device=device)
            s_dcv = torch.zeros(N, model.r_dcv, device=device)

            for t in range(T - 1):
                g = gating[t] if gating is not None else ones
                s = stim[t] if stim is not None else None
                mu_next, s_sv, s_dcv = model.prior_step(
                    u_cur, s_sv, s_dcv, g, s,
                )
                # Inject process noise into all neurons
                if is_hetero:
                    sigma_t = model.sigma_at(u_cur).detach()  # (N,)
                else:
                    sigma_t = sigma_const
                eps = torch.randn(N, device=device)
                u_next = mu_next + sigma_t * eps
                u_next = _clamp(u_next, lo, hi)
                u_cur = u_next
                samples_np[k, t + 1] = u_cur.cpu().numpy()

    u_np = u0.cpu().numpy()

    # Ensemble statistics
    ens_mean = samples_np.mean(axis=0)           # (T, N)
    ens_std = samples_np.std(axis=0)             # (T, N)
    ci_lo = np.percentile(samples_np, 2.5, axis=0)   # (T, N)
    ci_hi = np.percentile(samples_np, 97.5, axis=0)  # (T, N)

    # Per-sample R² and ensemble-mean R²
    r2_per_sample = np.full((n_samples, N), np.nan)
    for k in range(n_samples):
        for j in range(N):
            r2_per_sample[k, j] = _r2(u_np[1:, j], samples_np[k, 1:, j])
    r2_mean = np.array([_r2(u_np[1:, j], ens_mean[1:, j]) for j in range(N)])

    return {
        "samples": samples_np,
        "mean": ens_mean,
        "std": ens_std,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "r2_per_sample": r2_per_sample,
        "r2_mean": r2_mean,
    }

def loo_forward_simulate(
    model: Stage2ModelPT, u_all: torch.Tensor,
    held_out: int, gating, stim,
) -> np.ndarray:
    T, N = u_all.shape
    device = u_all.device
    i = held_out
    lo, hi = _get_clip_bounds(model)
    ones = torch.ones(N, device=device)

    u_pred = torch.zeros(T, device=device)
    u_pred[0] = u_all[0, i]
    s_sv  = torch.zeros(N, model.r_sv,  device=device)
    s_dcv = torch.zeros(N, model.r_dcv, device=device)

    with torch.no_grad():
        for t in range(T - 1):
            u_t = u_all[t].clone()
            u_t[i] = u_pred[t]
            g = gating[t] if gating is not None else ones
            s = stim[t]   if stim   is not None else None
            mu_next, s_sv, s_dcv = model.prior_step(u_t, s_sv, s_dcv, g, s)
            u_pred[t + 1] = _clamp(mu_next[i : i + 1], lo, hi).squeeze()

    return u_pred.cpu().numpy()


def loo_forward_simulate_stochastic(
    model: Stage2ModelPT, u_all: torch.Tensor,
    held_out: int, gating, stim,
    n_samples: int = 5,
) -> np.ndarray:
    """Sample *n_samples* stochastic LOO trajectories for neuron *held_out*.

    At each time-step, instead of using the deterministic mean prediction,
    we sample::

        u_{t+1,i} ~ N( mu_{t+1,i}, sigma_{t,i}^2 )

    where sigma is state-dependent (heteroscedastic) or constant
    (homoscedastic) depending on the model configuration.
    All other neurons are clamped to ground truth.

    Returns
    -------
    samples : np.ndarray, shape ``(n_samples, T)``
    """
    T, N = u_all.shape
    device = u_all.device
    i = held_out
    lo, hi = _get_clip_bounds(model)
    ones = torch.ones(N, device=device)

    # For homoscedastic models we can pre-compute sigma once
    is_hetero = (getattr(model, "_noise_mode", "homoscedastic") == "heteroscedastic")
    if not is_hetero:
        sigma_i_const = model.sigma_at().detach()[i]

    samples = np.zeros((n_samples, T), dtype=np.float32)
    with torch.no_grad():
        for k in range(n_samples):
            u_pred = torch.zeros(T, device=device)
            u_pred[0] = u_all[0, i]
            s_sv  = torch.zeros(N, model.r_sv,  device=device)
            s_dcv = torch.zeros(N, model.r_dcv, device=device)

            for t in range(T - 1):
                u_t = u_all[t].clone()
                u_t[i] = u_pred[t]
                g = gating[t] if gating is not None else ones
                s = stim[t]   if stim   is not None else None
                mu_next, s_sv, s_dcv = model.prior_step(u_t, s_sv, s_dcv, g, s)
                # State-dependent or constant noise
                if is_hetero:
                    sigma_i = model.sigma_at(u_t).detach()[i]
                else:
                    sigma_i = sigma_i_const
                noise = torch.randn(1, device=device) * sigma_i
                u_pred[t + 1] = _clamp(
                    mu_next[i : i + 1] + noise, lo, hi
                ).squeeze()

            samples[k] = u_pred.cpu().numpy()
    return samples


def _calibrate_loo_sigma(
    u_true: np.ndarray, loo_pred: np.ndarray,
) -> float:
    """Compute per-neuron empirical LOO residual std (excluding frame 0).

    This is the RMS of the deterministic LOO prediction error — the "true"
    scale of LOO uncertainty, which is much larger than the learned one-step σ.
    """
    resid = u_true[1:] - loo_pred[1:]
    m = np.isfinite(resid)
    if m.sum() < 2:
        return 1.0
    return float(np.std(resid[m]))


def loo_forward_simulate_calibrated(
    model: Stage2ModelPT, u_all: torch.Tensor,
    held_out: int, gating, stim,
    sigma_empirical: float,
    n_samples: int = 20,
) -> np.ndarray:
    """Calibrated stochastic LOO: inject noise scaled to the *empirical*
    LOO residual std, not the (too-small) learned σ.

    This produces stochastic sample trajectories whose 95% CI matches
    the actual spread of LOO prediction errors.

    Parameters
    ----------
    sigma_empirical : float
        Per-neuron empirical LOO residual std from `_calibrate_loo_sigma`.
    """
    T, N = u_all.shape
    device = u_all.device
    i = held_out
    lo, hi = _get_clip_bounds(model)
    ones = torch.ones(N, device=device)

    # Use empirical sigma for the held-out neuron, model sigma for others
    sigma_emp_t = torch.tensor(sigma_empirical, device=device, dtype=torch.float32)

    samples = np.zeros((n_samples, T), dtype=np.float32)
    with torch.no_grad():
        for k in range(n_samples):
            u_pred = torch.zeros(T, device=device)
            u_pred[0] = u_all[0, i]
            s_sv  = torch.zeros(N, model.r_sv,  device=device)
            s_dcv = torch.zeros(N, model.r_dcv, device=device)

            for t in range(T - 1):
                u_t = u_all[t].clone()
                u_t[i] = u_pred[t]
                g = gating[t] if gating is not None else ones
                s = stim[t]   if stim   is not None else None
                mu_next, s_sv, s_dcv = model.prior_step(u_t, s_sv, s_dcv, g, s)
                # Inject noise at the *empirical* LOO scale
                noise = torch.randn(1, device=device) * sigma_emp_t
                u_pred[t + 1] = _clamp(
                    mu_next[i : i + 1] + noise, lo, hi
                ).squeeze()

            samples[k] = u_pred.cpu().numpy()
    return samples


def run_loo_all(
    model: Stage2ModelPT, data: Dict[str, Any],
    subset: Optional[List[int]] = None,
    n_sample_trajectories: int = 0,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    gating, stim = data.get("gating"), data.get("stim")
    indices = subset if subset is not None else list(range(u.shape[1]))
    u_np = u.cpu().numpy()

    # Decide whether to run stochastic sampling
    do_stochastic = (
        n_sample_trajectories > 0
        and getattr(model, "_learn_noise", False)
    )

    preds = {}
    samples = {}  # idx -> (n_samples, T) array
    calibrated_samples = {}  # idx -> (n_samples, T) array — calibrated to empirical LOO residuals
    labels = data.get("neuron_labels", [])
    _logger = get_stage2_logger()
    _logger.info("loo_start", n_neurons=len(indices),
                 stochastic=do_stochastic, n_samples=n_sample_trajectories)
    for cnt, i in enumerate(indices):
        preds[i] = loo_forward_simulate(model, u, i, gating, stim)
        if do_stochastic:
            samples[i] = loo_forward_simulate_stochastic(
                model, u, i, gating, stim,
                n_samples=n_sample_trajectories,
            )
            # Calibrated sampling: use empirical LOO residual std
            sigma_emp = _calibrate_loo_sigma(u_np[:, i], preds[i])
            calibrated_samples[i] = loo_forward_simulate_calibrated(
                model, u, i, gating, stim,
                sigma_empirical=sigma_emp,
                n_samples=n_sample_trajectories,
            )
        lbl = labels[i] if i < len(labels) else f"#{i}"
        r2_i = float(_r2(u_np[1:, i], preds[i][1:]))
        if cnt == 0 or (cnt + 1) % 10 == 0 or len(indices) <= 10:
            _logger.info("loo_progress", done=cnt + 1, total=len(indices),
                         neuron=int(i), name=lbl, r2=r2_i)

    pred_full = np.column_stack([preds.get(i, u_np[:, i]) for i in range(u.shape[1])])
    r2, corr, rmse = _per_neuron_metrics(u_np[1:], pred_full[1:], indices)

    result = {"pred": preds, "r2": r2, "corr": corr, "rmse": rmse}
    if samples:
        result["samples"] = samples
    if calibrated_samples:
        result["calibrated_samples"] = calibrated_samples
    return result


def choose_loo_subset(
    data: Dict[str, Any], onestep: Dict[str, Any], *,
    subset_size: int = 0, subset_mode: str = "variance",
    explicit_indices: Optional[List[int]] = None, seed: int = 0,
) -> Optional[List[int]]:
    u_np = data["u_stage1"].cpu().numpy()
    N = u_np.shape[1]

    if explicit_indices is not None and len(explicit_indices) > 0:
        keep = sorted({int(i) for i in explicit_indices if 0 <= int(i) < N})
        return keep or None

    k = int(subset_size)
    if k <= 0 or k >= N:
        return None

    mode = str(subset_mode).strip().lower()

    # Resolve candidate pool: motor neurons only for motor* modes, else all
    candidates = None
    if mode.startswith("motor"):
        _resolved = _resolve_motor_indices(data, N)
        candidates = _resolved if _resolved else None
        if not candidates:
            candidates = list(range(N))  # fallback

    if mode == "motor_best_onestep":
        r2 = np.asarray(onestep["r2"], dtype=float)
        scores = np.where(np.isfinite(r2), r2, -np.inf)
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
        return ranked[:k]
    if mode == "motor":
        return candidates[:k]
    if mode == "variance":
        return [int(i) for i in np.argsort(np.nanvar(u_np, axis=0))[::-1][:k]]
    if mode in ("worst_onestep", "best_onestep"):
        r2 = np.asarray(onestep["r2"], dtype=float)
        fill = np.inf if mode == "worst_onestep" else -np.inf
        score = np.where(np.isfinite(r2), r2, fill)
        order = np.argsort(score) if mode == "worst_onestep" else np.argsort(score)[::-1]
        return [int(i) for i in order[:k]]
    if mode == "named":
        # Look up neurons by name from cfg.eval_loo_subset_names
        cfg = data.get("_cfg")
        names = getattr(cfg, "eval_loo_subset_names", None) if cfg else None
        if names:
            labels = data.get("neuron_labels", [])
            lbl_lower = [str(l).strip().lower() for l in labels]
            found = []
            for nm in names:
                key = str(nm).strip().lower()
                if key in lbl_lower:
                    found.append(lbl_lower.index(key))
            if found:
                return sorted(set(found))
        # fallback to all neurons if names not resolved
        return None

    return sorted(int(i) for i in np.random.default_rng(seed).choice(N, size=k, replace=False))

def compute_current_decomposition(
    model: Stage2ModelPT, data: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    T, N = u.shape
    gating, stim = data.get("gating"), data.get("stim")

    out = {k: np.zeros((T - 1, N)) for k in ("I_leak", "I_gap", "I_sv", "I_dcv", "I_stim", "I_bias")}

    with torch.no_grad():
        s_sv_state  = torch.zeros(N, model.r_sv,  device=device)
        s_dcv_state = torch.zeros(N, model.r_dcv, device=device)
        lam, dt = model.lambda_u.detach(), model.dt
        L = model.laplacian()
        ones = torch.ones(N, device=device)

        syn_terms = []
        if model.r_sv > 0:
            syn_terms.append({
                "prefix": "I_sv",
                "state_name": "sv",
                "gamma": torch.exp(-dt / (model.tau_sv + 1e-12)),
                "T_eff": model.T_sv * model._get_W("W_sv"),
                "a": model.a_sv,
                "E": model.E_sv,
            })
        if model.r_dcv > 0:
            syn_terms.append({
                "prefix": "I_dcv",
                "state_name": "dcv",
                "gamma": torch.exp(-dt / (model.tau_dcv + 1e-12)),
                "T_eff": model.T_dcv * model._get_W("W_dcv"),
                "a": model.a_dcv,
                "E": model.E_dcv,
            })

        for t in range(T - 1):
            u_prev = u[t]
            g = gating[t] if gating is not None else ones
            s = stim[t]   if stim   is not None else None

            out["I_leak"][t] = ((1.0 - lam) * u_prev).cpu().numpy()
            out["I_bias"][t] = (lam * model.I0).cpu().numpy()
            out["I_gap"][t]  = (lam * (L @ u_prev)).cpu().numpy()

            phi_prev = model.phi(u_prev) * g

            for term in syn_terms:
                prefix = term["prefix"]
                s_state = s_sv_state if term["state_name"] == "sv" else s_dcv_state
                gamma = term["gamma"]
                T_eff = term["T_eff"]
                a_param = term["a"]
                E_param = term["E"]
                s_state = gamma.view(1, -1) * s_state + phi_prev.unsqueeze(1)
                a = a_param.unsqueeze(0) if a_param.dim() == 1 else a_param
                # Pool presynaptic states first, then apply postsynaptic amplitudes
                # (sa = s_state * a is wrong when a is per-neuron: it mixes
                # presynaptic index with postsynaptic amplitude).
                pool = torch.matmul(T_eff.t(), s_state)           # (N_post, r)
                g_syn = (pool * a).sum(dim=1)

                E = E_param
                if E.dim() == 0:
                    I_t = lam * g_syn * (E - u_prev)
                elif E.dim() == 1:
                    e_pool = torch.matmul(T_eff.t(), s_state * E.view(-1, 1))
                    e_drive = (e_pool * a).sum(dim=1)
                    I_t = lam * (e_drive - g_syn * u_prev)
                else:
                    e_pool = torch.matmul((T_eff * E).t(), s_state)
                    e_drive = (e_pool * a).sum(dim=1)
                    I_t = lam * (e_drive - g_syn * u_prev)
                out[prefix][t] = I_t.cpu().numpy()

                if prefix == "I_sv":
                    s_sv_state = s_state
                else:
                    s_dcv_state = s_state

            if model.d_ell > 0 and s is not None:
                if getattr(model, "stim_diagonal_only", False):
                    I_stim_t = lam * (model.b * s.view(N))
                else:
                    I_stim_t = lam * torch.matmul(model.b, s)
                out["I_stim"][t] = I_stim_t.cpu().numpy()

    return out

def run_full_evaluation(
    model: Stage2ModelPT, data: Dict[str, Any], cfg,
    decoder=None, beh_all_baseline=None,
) -> Dict[str, Any]:
    data["_cfg"] = cfg

    onestep = compute_onestep(model, data)

    subset = choose_loo_subset(
        data, onestep,
        subset_size=_cfg_val(cfg, "eval_loo_subset_size", 0, int),
        subset_mode=str(getattr(cfg, "eval_loo_subset_mode", "variance")),
        explicit_indices=(
            None if getattr(cfg, "eval_loo_subset_neurons", None) is None
            else list(cfg.eval_loo_subset_neurons)
        ),
        seed=_cfg_val(cfg, "eval_loo_subset_seed", 0, int),
    )

    # Per-neuron shrinkage CV-reg
    cv_reg = None
    if _cfg_val(cfg, "cv_reg_enabled", True, bool):
        u_np = data["u_stage1"].cpu().numpy()
        cv_reg = compute_cv_reg(
            u_np, onestep["prior_mu"],
            n_folds=_cfg_val(cfg, "cv_reg_n_folds", 5, int),
            log_reg_min=_cfg_val(cfg, "cv_reg_log_min", -3.0, float),
            log_reg_max=_cfg_val(cfg, "cv_reg_log_max", 5.0, float),
            n_grid=_cfg_val(cfg, "cv_reg_n_grid", 50, int),
        )
    onestep["cv_reg"] = cv_reg

    loo      = run_loo_all(
        model, data, subset=subset,
        n_sample_trajectories=_cfg_val(cfg, "n_sample_trajectories", 0, int),
    )
    currents = compute_current_decomposition(model, data)
    free_run = compute_free_run(model, data)

    # Full-brain stochastic free-run sampling
    _n_fr = _cfg_val(cfg, "n_freerun_samples", 0, int)
    freerun_stoch = None
    if _n_fr > 0 and getattr(model, "_learn_noise", False):
        _logger = get_stage2_logger()
        _logger.info("freerun_stochastic_start", n_samples=_n_fr)
        freerun_stoch = free_run_stochastic(model, data, n_samples=_n_fr)
        _logger.info("freerun_stochastic_done",
                     r2_mean_median=float(np.nanmedian(freerun_stoch["r2_mean"])))

    beh     = evaluate_training_decoder(decoder, data, onestep) if decoder is not None else None
    beh_all = (
        beh_all_baseline if beh_all_baseline is not None
        else behaviour_all_neurons_prediction(data)
    )

    r2_model_mean = None
    if beh is not None:
        r2 = beh.get("r2_model")
        if r2 is not None and np.any(np.isfinite(r2)):
            r2_model_mean = float(np.nanmean(r2))

    return {
        "onestep": onestep, "loo": loo, "currents": currents,
        "free_run": free_run, "freerun_stoch": freerun_stoch,
        "cv_reg": cv_reg,
        "beh": beh, "beh_all": beh_all,
        "beh_r2_model": r2_model_mean,
    }
