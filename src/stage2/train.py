from __future__ import annotations

import dataclasses
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from pathlib import Path

from .behavior_decoder_eval import (
    behaviour_all_neurons_prediction,
    compute_behaviour_loss,
    init_behaviour_decoder,
    _log_ridge_grid,
    _make_contiguous_folds,
)
from .config import Stage2PTConfig
from .io_h5 import load_data_pt, save_results_pt
from .model import Stage2ModelPT
from .plot_eval import generate_eval_loo_plots

__all__ = [
    "train_stage2",
    "effective_variance",
    "compute_dynamics_loss",
    "compute_rollout_loss",
    "compute_teacher_forced_states",
    "apply_training_step",
    "snapshot_model_state",
    # Ridge-CV alpha solver (merged from ridge_alpha.py)
    "ridge_cv_solve_alpha",
    "inject_alpha_into_model",
]


# --------------------------------------------------------------------------- #
#  Variance and loss utilities (merged from train_utils.py)                     #
# --------------------------------------------------------------------------- #

def effective_variance(
    sigma_u: torch.Tensor,
    T: int,
    *,
    u_var: Optional[torch.Tensor] = None,
    use_u_var_weighting: bool = False,
    u_var_scale: float = 1.0,
    u_var_floor: float = 1e-8,
) -> torch.Tensor:
    sigma_u2 = sigma_u.view(1, -1) ** 2
    if (not bool(use_u_var_weighting)) or (u_var is None):
        return sigma_u2.expand(T, -1)

    uv = torch.clamp(u_var, min=float(u_var_floor))
    return sigma_u2 + float(u_var_scale) * uv


def compute_dynamics_loss(
    target: torch.Tensor,
    prior_mu: torch.Tensor,
    sigma_u: torch.Tensor,
    *,
    u_var: Optional[torch.Tensor] = None,
    use_u_var_weighting: bool = False,
    u_var_scale: float = 1.0,
    u_var_floor: float = 1e-8,
) -> torch.Tensor:
    T = target.shape[0]
    var_eff = effective_variance(
        sigma_u,
        T,
        u_var=u_var,
        use_u_var_weighting=use_u_var_weighting,
        u_var_scale=u_var_scale,
        u_var_floor=u_var_floor,
    )
    resid2 = (target - prior_mu) ** 2
    valid = torch.isfinite(resid2) & torch.isfinite(var_eff) & (var_eff > 0)
    if not valid.any():
        return torch.tensor(0.0, device=target.device)
    weighted = torch.where(valid, resid2 / var_eff, torch.zeros_like(resid2))
    return weighted.sum() / valid.sum().clamp(min=1)


def compute_rollout_loss(
    model: "Stage2ModelPT",
    u_target: torch.Tensor,
    sigma_u: torch.Tensor,
    *,
    rollout_steps: int,
    rollout_starts: int,
    gating_data: Optional[torch.Tensor] = None,
    stim_data: Optional[torch.Tensor] = None,
    cached_states: Optional[dict] = None,
) -> torch.Tensor:
    """Short-horizon rollout loss: MSE over K-step free-running predictions.

    Picks *rollout_starts* random starting times, runs the model for
    *rollout_steps* steps without teacher forcing, and computes the
    variance-normalised MSE against the ground truth trajectory.

    If *cached_states* is provided (from :func:`compute_teacher_forced_states`),
    synaptic states are warm-started from the teacher-forced trajectory
    instead of being initialized to zero.
    """
    T, N = u_target.shape
    device = u_target.device
    K = rollout_steps
    if K <= 0 or T <= K + 1:
        return torch.tensor(0.0, device=device)

    # Choose random starting indices in [0, T - K - 1]
    max_start = T - K - 1
    n_starts = min(rollout_starts, max_start + 1)
    t0s = torch.randperm(max_start + 1, device=device)[:n_starts]

    sigma2 = (sigma_u ** 2).squeeze()  # (N,)
    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for t0_idx in range(n_starts):
        t0 = int(t0s[t0_idx].item())
        u_cur = u_target[t0]

        # Warm-start synaptic states from cached teacher-forced trajectory
        if cached_states is not None:
            s_sv = cached_states["s_sv"][t0].clone()
            s_dcv = cached_states["s_dcv"][t0].clone()
        else:
            s_sv = torch.zeros(N, model.r_sv, device=device)
            s_dcv = torch.zeros(N, model.r_dcv, device=device)

        seg_loss = torch.tensor(0.0, device=device)
        seg_count = 0

        for k in range(K):
            t = t0 + k
            g = gating_data[t] if gating_data is not None else torch.ones(N, device=device)
            s = stim_data[t] if stim_data is not None else None
            u_next, s_sv, s_dcv = model.prior_step(u_cur, s_sv, s_dcv, g, s)

            target_next = u_target[t + 1]
            resid2 = (u_next - target_next) ** 2
            ok = torch.isfinite(resid2)
            if ok.any():
                seg_loss = seg_loss + (resid2[ok] / sigma2[ok]).mean()
                seg_count += 1
            u_cur = u_next  # free-running: use own prediction

        if seg_count > 0:
            total_loss = total_loss + seg_loss / seg_count
            n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / n_valid


def compute_teacher_forced_states(
    model: "Stage2ModelPT",
    u: torch.Tensor,
    *,
    gating_data: Optional[torch.Tensor] = None,
    stim_data: Optional[torch.Tensor] = None,
) -> dict:
    """Run a teacher-forced pass and cache synaptic states at each time step.

    Returns a dict with:
        s_sv  : (T, N, r_sv)  — synaptic SV states *before* each step
        s_dcv : (T, N, r_dcv) — synaptic DCV states *before* each step

    These can be used to warm-start rollout segments or neuron-dropout
    segments so that the synaptic trace state is realistic rather than zero.
    """
    T, N = u.shape
    device = u.device
    r_sv, r_dcv = model.r_sv, model.r_dcv

    s_sv_cache = torch.zeros(T, N, r_sv, device=device)
    s_dcv_cache = torch.zeros(T, N, r_dcv, device=device)

    s_sv = torch.zeros(N, r_sv, device=device)
    s_dcv = torch.zeros(N, r_dcv, device=device)

    with torch.no_grad():
        for t in range(T - 1):
            s_sv_cache[t] = s_sv
            s_dcv_cache[t] = s_dcv
            g = gating_data[t] if gating_data is not None else torch.ones(N, device=device)
            s = stim_data[t] if stim_data is not None else None
            _, s_sv, s_dcv = model.prior_step(u[t], s_sv, s_dcv, g, s)
        # Last time step gets the final states
        s_sv_cache[T - 1] = s_sv
        s_dcv_cache[T - 1] = s_dcv

    return {"s_sv": s_sv_cache, "s_dcv": s_dcv_cache}


def snapshot_model_state(model: Stage2ModelPT) -> Dict[str, torch.Tensor]:
    params_final = {name: p.detach().cpu() for name, p in model.named_parameters()}
    with torch.no_grad():
        params_final["W_sv"] = model.W_sv.detach().cpu()
        params_final["W_dcv"] = model.W_dcv.detach().cpu()
        # Store constrained values alongside raw parameters
        for attr in ("lambda_u", "G", "a_sv", "a_dcv", "tau_sv", "tau_dcv"):
            params_final[attr] = getattr(model, attr).detach().cpu()
    return params_final


# --------------------------------------------------------------------------- #
#  Per-neuron ridge-CV solver for kernel amplitudes (merged from ridge_alpha.py)
# --------------------------------------------------------------------------- #

def _compute_alpha_features(
    model,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Forward pass collecting per-rank, per-neuron synaptic features."""
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    T, N = u.shape
    gating = data.get("gating")
    stim = data.get("stim")

    with torch.no_grad():
        lam = model.lambda_u.detach()
        I0 = model.I0.detach()
        L = model.laplacian().detach()

        T_eff_sv = (model.T_sv * model.W_sv).detach()
        T_eff_dcv = (model.T_dcv * model.W_dcv).detach()
        tau_sv = model.tau_sv.detach()
        tau_dcv = model.tau_dcv.detach()
        E_sv = model.E_sv.detach()
        E_dcv = model.E_dcv.detach()
        dt = model.dt

        gamma_sv = torch.exp(-dt / (tau_sv + 1e-12))
        gamma_dcv = torch.exp(-dt / (tau_dcv + 1e-12))

        r_sv = model.r_sv
        r_dcv = model.r_dcv

        feat_sv = torch.zeros(T - 1, N, r_sv, device=device)
        feat_dcv = torch.zeros(T - 1, N, r_dcv, device=device)
        target = torch.zeros(T - 1, N, device=device)

        s_sv = torch.zeros(N, r_sv, device=device)
        s_dcv = torch.zeros(N, r_dcv, device=device)

        for t in range(1, T):
            u_prev = u[t - 1]
            g = gating[t - 1] if gating is not None else torch.ones(N, device=device)
            phi_gated = torch.sigmoid(u_prev) * g.view(N)

            s_sv = gamma_sv.view(1, -1) * s_sv + phi_gated.unsqueeze(1)
            s_dcv = gamma_dcv.view(1, -1) * s_dcv + phi_gated.unsqueeze(1)

            pool_sv = T_eff_sv.t() @ s_sv
            pool_dcv = T_eff_dcv.t() @ s_dcv

            _fill_alpha_features(feat_sv, t - 1, pool_sv, s_sv, T_eff_sv, E_sv, u_prev)
            _fill_alpha_features(feat_dcv, t - 1, pool_dcv, s_dcv, T_eff_dcv, E_dcv, u_prev)

            I_gap = L @ u_prev
            ar1 = (1.0 - lam) * u_prev + lam * I0
            residual = u[t] - ar1 - lam * I_gap

            if model.d_ell > 0 and stim is not None:
                s_t = stim[t - 1]
                if s_t is not None:
                    if model.stim_diagonal_only:
                        I_stim = model.b.detach() * s_t.view(N)
                    else:
                        I_stim = model.b.detach() @ s_t
                    residual = residual - lam * I_stim

            target[t - 1] = residual / lam.clamp(min=1e-8)

    return {
        "features_sv": feat_sv.cpu().numpy().astype(np.float64),
        "features_dcv": feat_dcv.cpu().numpy().astype(np.float64),
        "target": target.cpu().numpy().astype(np.float64),
    }


def _fill_alpha_features(
    out: torch.Tensor, t_idx: int, pool: torch.Tensor, s: torch.Tensor,
    T_eff: torch.Tensor, E: torch.Tensor, u_prev: torch.Tensor,
) -> None:
    """Write per-(neuron, rank) features into out[t_idx]."""
    if E.dim() == 0:
        out[t_idx] = (E - u_prev).unsqueeze(1) * pool
    elif E.dim() == 1:
        E_pool = T_eff.t() @ (s * E.view(-1, 1))
        out[t_idx] = E_pool - u_prev.unsqueeze(1) * pool
    else:
        E_pool = (T_eff * E).t() @ s
        out[t_idx] = E_pool - u_prev.unsqueeze(1) * pool


def _solve_neuron_ridge_cv(
    X: np.ndarray, y: np.ndarray, ridge_grid: np.ndarray, n_folds: int,
) -> Dict[str, Any]:
    T_eff, p = X.shape
    folds = _make_contiguous_folds(np.arange(T_eff), n_folds)

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std = np.where(x_std > 1e-12, x_std, 1.0)
    Xs = (X - x_mean) / x_std
    y_mean = float(y.mean())
    yc = y - y_mean

    cv_mse = np.full(len(ridge_grid), np.inf, dtype=np.float64)

    for lam_idx, lam in enumerate(ridge_grid):
        fold_errors = []
        for fold_idx in folds:
            train_mask = np.ones(T_eff, dtype=bool)
            train_mask[fold_idx] = False
            tr = np.where(train_mask)[0]
            if tr.size < max(p + 1, 5) or fold_idx.size == 0:
                continue

            Xs_tr = Xs[tr]
            yc_tr = yc[tr]
            gram_tr = Xs_tr.T @ Xs_tr
            if lam > 0:
                gram_tr = gram_tr + lam * np.eye(p, dtype=np.float64)
            rhs_tr = Xs_tr.T @ yc_tr
            try:
                coef_s = np.linalg.solve(gram_tr, rhs_tr)
            except np.linalg.LinAlgError:
                coef_s, *_ = np.linalg.lstsq(gram_tr, rhs_tr, rcond=None)

            mse = np.mean((yc[fold_idx] - Xs[fold_idx] @ coef_s) ** 2)
            if np.isfinite(mse):
                fold_errors.append(mse)

        if fold_errors:
            cv_mse[lam_idx] = float(np.mean(fold_errors))

    if not np.any(np.isfinite(cv_mse)):
        best_idx = 0  # all folds failed; fall back to unregularized
    else:
        best_idx = int(np.nanargmin(np.where(np.isfinite(cv_mse), cv_mse, np.inf)))
    best_lambda = float(ridge_grid[best_idx])

    gram_full = Xs.T @ Xs
    if best_lambda > 0:
        gram_full = gram_full + best_lambda * np.eye(p, dtype=np.float64)
    rhs = Xs.T @ yc
    try:
        coef_s = np.linalg.solve(gram_full, rhs)
    except np.linalg.LinAlgError:
        coef_s, *_ = np.linalg.lstsq(gram_full, rhs, rcond=None)

    coef = coef_s / x_std
    intercept = y_mean - float(x_mean @ coef)

    return {
        "coef": coef,
        "intercept": intercept,
        "best_lambda": best_lambda,
        "cv_mse": cv_mse,
        "at_upper": bool(best_idx == len(ridge_grid) - 1),
    }


def ridge_cv_solve_alpha(
    model,
    data: Dict[str, Any],
    cfg=None,
    *,
    n_folds: int = 5,
    log_lambda_min: float = -2.0,
    log_lambda_max: float = 4.0,
    n_grid: int = 30,
) -> Dict[str, Any]:
    """Solve per-neuron kernel amplitudes via ridge-CV."""
    if cfg is not None:
        n_folds = int(getattr(cfg, "alpha_cv_n_folds", n_folds) or n_folds)
        log_lambda_min = float(getattr(cfg, "alpha_cv_log_min", log_lambda_min))
        log_lambda_max = float(getattr(cfg, "alpha_cv_log_max", log_lambda_max))
        n_grid = int(getattr(cfg, "alpha_cv_n_grid", n_grid) or n_grid)

    ridge_grid = _log_ridge_grid(log_lambda_min, log_lambda_max, n_grid)
    n_grid_actual = len(ridge_grid)
    N = model.N
    r_sv = model.r_sv
    r_dcv = model.r_dcv
    p_total = r_sv + r_dcv

    feats = _compute_alpha_features(model, data)
    feat_sv = feats["features_sv"]
    feat_dcv = feats["features_dcv"]
    target = feats["target"]

    with torch.no_grad():
        a_sv_cur = model.a_sv.detach().cpu().numpy()
        a_dcv_cur = model.a_dcv.detach().cpu().numpy()

    if a_sv_cur.ndim == 1:
        a_sv_out = np.tile(a_sv_cur, (N, 1))
    else:
        a_sv_out = a_sv_cur.copy()
    if a_dcv_cur.ndim == 1:
        a_dcv_out = np.tile(a_dcv_cur, (N, 1))
    else:
        a_dcv_out = a_dcv_cur.copy()

    lambdas = np.full(N, np.nan, dtype=np.float64)
    intercepts = np.zeros(N, dtype=np.float64)
    cv_mse_all = np.full((N, n_grid_actual), np.nan, dtype=np.float64)
    fit_r2 = np.full(N, np.nan, dtype=np.float64)
    coef_raw_all = np.zeros((N, p_total), dtype=np.float64)
    at_upper_flags = np.zeros(N, dtype=bool)
    n_updated = 0
    n_at_upper = 0

    for i in range(N):
        parts = []
        if r_sv > 0:
            parts.append(feat_sv[:, i, :])
        if r_dcv > 0:
            parts.append(feat_dcv[:, i, :])

        X_i = np.concatenate(parts, axis=1)
        y_i = target[:, i]

        valid = np.all(np.isfinite(X_i), axis=1) & np.isfinite(y_i)
        n_valid = int(valid.sum())
        if n_valid < max(p_total + 5, 20):
            continue

        x_norms = np.abs(X_i[valid]).max(axis=0)
        if x_norms.max() < 1e-10:
            continue

        result = _solve_neuron_ridge_cv(X_i[valid], y_i[valid], ridge_grid, n_folds)
        coef_raw = result["coef"]
        coef = np.maximum(coef_raw, 0.0)
        coef_raw_all[i] = coef_raw

        cv_mse_i = result["cv_mse"]
        cv_mse_all[i, :len(cv_mse_i)] = cv_mse_i

        y_pred_i = X_i[valid] @ coef + (float(y_i[valid].mean()) - float(X_i[valid].mean(axis=0) @ coef))
        ss_res = float(np.sum((y_i[valid] - y_pred_i) ** 2))
        ss_tot = float(np.sum((y_i[valid] - y_i[valid].mean()) ** 2))
        fit_r2[i] = 1.0 - ss_res / max(ss_tot, 1e-12)

        y_mean_i = float(y_i[valid].mean())
        x_mean_i = X_i[valid].mean(axis=0)
        intercepts[i] = y_mean_i - float(x_mean_i @ coef)

        if r_sv > 0:
            a_sv_out[i] = coef[:r_sv]
        if r_dcv > 0:
            a_dcv_out[i] = coef[r_sv:]

        lambdas[i] = result["best_lambda"]
        n_updated += 1
        if result["at_upper"]:
            n_at_upper += 1
            at_upper_flags[i] = True

    device = next(model.parameters()).device
    return {
        "alpha_sv": torch.tensor(a_sv_out, dtype=torch.float32, device=device),
        "alpha_dcv": torch.tensor(a_dcv_out, dtype=torch.float32, device=device),
        "intercepts": torch.tensor(intercepts, dtype=torch.float32, device=device),
        "lambdas": lambdas,
        "n_updated": n_updated,
        "n_at_upper": n_at_upper,
        "ridge_grid": ridge_grid,
        "cv_mse_all": cv_mse_all,
        "fit_r2": fit_r2,
        "coef_raw_all": coef_raw_all,
        "at_upper_flags": at_upper_flags,
    }


def inject_alpha_into_model(
    model,
    result: Dict[str, Any],
    blend: float = 1.0,
    max_ratio: float = 0.0,
) -> None:
    """Copy ridge-CV solved alpha back into the model's a_sv / a_dcv."""
    blend = max(0.0, min(1.0, blend))
    with torch.no_grad():
        for attr, key in [("a_sv", "alpha_sv"), ("a_dcv", "alpha_dcv")]:
            cur = getattr(model, attr, None)
            if cur is None or cur.numel() == 0:
                continue
            target = result[key]
            blended = (1.0 - blend) * cur + blend * target
            if max_ratio > 0:
                old = cur.clamp(min=1e-12)
                ratio = blended / old
                ratio_clamped = ratio.clamp(min=1.0 / max_ratio, max=max_ratio)
                blended = old * ratio_clamped
            model.set_param_constrained(attr, blended)

        intercepts = result.get("intercepts")
        if intercepts is not None:
            model.I0.data.add_(blend * intercepts)


# --------------------------------------------------------------------------- #
#  Formatting helpers                                                           #
# --------------------------------------------------------------------------- #

def _fmt(x: torch.Tensor) -> str:
    x = x.detach().float().cpu().flatten()
    if x.numel() == 0:
        return "empty"
    xf = x[torch.isfinite(x)]
    if xf.numel() == 0:
        return f"shape={tuple(x.shape)} all-nonfinite"
    if xf.numel() == 1:
        return f"{xf.item():.6g}"
    return f"mean={xf.mean().item():.6g} min={xf.min().item():.6g} max={xf.max().item():.6g}"


def _fmt_masked(x: torch.Tensor, mask: torch.Tensor) -> str:
    sel = x.detach().float().cpu()[mask.detach().float().cpu() > 0]
    if sel.numel() == 0:
        return "empty"
    if sel.numel() == 1:
        return f"{sel.item():.6g}"
    return f"mean={sel.mean().item():.6g} min={sel.min().item():.6g} max={sel.max().item():.6g}"


def _tag(p: torch.Tensor) -> str:
    return "train" if getattr(p, "requires_grad", False) else "fixed"


def apply_training_step(
    loss: torch.Tensor,
    optimizer: optim.Optimizer,
    params: list[torch.Tensor],
    model: Stage2ModelPT,
    cfg: Stage2PTConfig,
    *,
    grad_clip: float = 0.0,
) -> None:
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()
    with torch.no_grad():
        _clamp_params(model, cfg)


# --------------------------------------------------------------------------- #
#  Parameter clamping                                                           #
# --------------------------------------------------------------------------- #

def _clamp_params(model: Stage2ModelPT, cfg: Stage2PTConfig) -> None:
    # Reparameterized parameters (lambda_u, G, a_sv/dcv, tau_sv/dcv, W_sv/dcv)
    # are bounded by construction via sigmoid / softplus — no clamping needed.

    # Stimulus weights are plain nn.Parameters → clamp if cfg specifies bounds.
    if model.d_ell > 0:
        b_min = getattr(cfg, "b_min", None)
        b_max = getattr(cfg, "b_max", None)
        lo = float(b_min) if b_min is not None else (-float(b_max) if b_max is not None else None)
        hi = float(b_max) if b_max is not None else None
        if lo is not None or hi is not None:
            model.b.data.clamp_(min=lo, max=hi)


# --------------------------------------------------------------------------- #
#  Logging                                                                      #
# --------------------------------------------------------------------------- #

class _TeeWriter:
    """Duplicate stdout to a file, preserving original stdout."""

    def __init__(self, log_path: str | Path):
        self._file = open(log_path, "w", buffering=1)
        self._stdout = sys.stdout

    def write(self, msg: str):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._stdout


def _config_to_dict(cfg: Stage2PTConfig) -> dict:
    """Serialize full config to a JSON-safe dict."""
    out: dict = {}
    for sec_name in ("data", "dynamics", "stimulus", "behavior", "train", "eval", "output"):
        sub = getattr(cfg, sec_name, None)
        if sub is None or not dataclasses.is_dataclass(sub):
            continue
        sec_dict: dict = {}
        for f in dataclasses.fields(sub):
            v = getattr(sub, f.name)
            # Convert non-serializable types
            if isinstance(v, Path):
                v = str(v)
            elif isinstance(v, tuple):
                v = list(v)
            sec_dict[f.name] = v
        out[sec_name] = sec_dict
    return out


def _save_run_config(cfg: Stage2PTConfig, save_dir: str | Path) -> Path:
    """Write the complete run config as JSON into *save_dir*/run_config.json."""
    p = Path(save_dir) / "run_config.json"
    with open(p, "w") as f:
        json.dump(_config_to_dict(cfg), f, indent=2, default=str)
    return p

def _log_config(cfg: Stage2PTConfig, d_ell: int) -> None:
    sep = "=" * 60
    print(
        f"\n{sep}\n"
        f"[Stage2] Config\n"
        f"  device={cfg.device}  lr={cfg.learning_rate}  epochs={cfg.num_epochs}\n"
        f"  masks: T_e={cfg.T_e_dataset}  T_sv={cfg.T_sv_dataset}  T_dcv={cfg.T_dcv_dataset}\n"
        f"  silencing={cfg.silencing_dataset}  stim={cfg.stim_dataset}  d_ell={d_ell}  "
        f"ridge_b={getattr(cfg, 'ridge_b', 0.0)}\n"
        f"  lambda_u: learn={getattr(cfg, 'learn_lambda_u', False)}\n"
        f"  fix_taus: sv={getattr(cfg, 'fix_tau_sv', False)} dcv={getattr(cfg, 'fix_tau_dcv', False)}\n"
        f"  ranks: r_sv={len(cfg.tau_sv_init)} r_dcv={len(cfg.tau_dcv_init)}\n"
        f"  learn_W: sv={getattr(cfg, 'learn_W_sv', False)} dcv={getattr(cfg, 'learn_W_dcv', False)}"
        f"  W_sv_init_mode={getattr(cfg, 'W_sv_init_mode', 'uniform')}"
        f"  W_sv_normalize={getattr(cfg, 'W_sv_normalize', False)}\n"
        f"  gap_junctions: edge_specific_G={getattr(cfg, 'edge_specific_G', False)}\n"
        f"  reversals: learn={getattr(cfg, 'learn_reversals', False)} "
        f"mode={getattr(cfg, 'reversal_mode', 'per_neuron')}\n"
        f"  u_var: weighting={getattr(cfg, 'use_u_var_weighting', False)} "
        f"scale={getattr(cfg, 'u_var_scale', 1.0)} floor={getattr(cfg, 'u_var_floor', 0.0)}\n"
        f"{sep}"
    )


def _log_init_params(model: Stage2ModelPT, cfg: Stage2PTConfig) -> None:
    print("[Stage2] Model parameters (init)")
    print(f"  lambda_u ({_tag(model.lambda_u)}): {_fmt(model.lambda_u)}")
    if model.edge_specific_G:
        G_edges = model.G[model.T_e > 0]
        print(f"  G ({_tag(model.G)}, edge-specific): mean={G_edges.mean():.6g} min={G_edges.min():.6g} max={G_edges.max():.6g}")
    else:
        print(f"  G ({_tag(model.G)}): {_fmt(model.G)}")
    print(f"  I0 ({_tag(model.I0)}): {_fmt(model.I0)}")
    print(f"  a_sv ({_tag(model.a_sv)}): {_fmt(model.a_sv)}")
    if model._W_sv_raw.requires_grad:
        print(f"  W_sv (train, softplus): {_fmt_masked(model.W_sv, model.T_sv)}")
    if model._W_dcv_raw.requires_grad:
        print(f"  W_dcv (train, softplus): {_fmt_masked(model.W_dcv, model.T_dcv)}")
    if model.d_ell > 0:
        print(f"  b ({_tag(model.b)}): shape={tuple(model.b.shape)} L2={model.b.detach().norm().item():.6g}")
    print(f"  tau_sv ({_tag(model.tau_sv)}): {_fmt(model.tau_sv)}")
    print(f"  a_dcv ({_tag(model.a_dcv)}): {_fmt(model.a_dcv)}")
    print(f"  tau_dcv ({_tag(model.tau_dcv)}): {_fmt(model.tau_dcv)}")
    rev_tag = "train" if cfg.learn_reversals else "fixed"
    print(f"  E_sv ({rev_tag}): {_fmt(model.E_sv)}")
    print(f"  E_dcv ({rev_tag}): {_fmt(model.E_dcv)}")
    # beta_interaction frozen at 1.0
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"[Stage2] Trainable: {', '.join(trainable) or '(none)'}")


# --------------------------------------------------------------------------- #
#  Parameter snapshot (for trajectory plots)                                    #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _snapshot_params(model) -> dict:
    """Return a dict of scalar summaries of every tracked model parameter."""
    snap: dict = {}

    # Gap-junction conductance
    G = model.G
    snap["G"] = float(G.mean())

    # Synaptic kernel amplitudes (RMS)
    snap["a_sv_rms"] = float(model.a_sv.pow(2).mean().sqrt()) if model.a_sv.numel() > 0 else 0.0
    snap["a_dcv_rms"] = float(model.a_dcv.pow(2).mean().sqrt()) if model.a_dcv.numel() > 0 else 0.0

    # Leak rate λ_u
    lam = model.lambda_u
    if lam.numel() > 0:
        snap["lambda_u_min"] = float(lam.min())
        snap["lambda_u_max"] = float(lam.max())
        snap["lambda_u_med"] = float(lam.median())
        snap["lambda_u_mean"] = float(lam.mean())
    else:
        snap["lambda_u_min"] = snap["lambda_u_max"] = 0.0
        snap["lambda_u_med"] = snap["lambda_u_mean"] = 0.0

    # Tonic drive I0
    I0 = model.I0
    snap["I0_rms"] = float(I0.pow(2).mean().sqrt()) if I0.numel() > 0 else 0.0
    snap["I0_absmax"] = float(I0.abs().max()) if I0.numel() > 0 else 0.0
    if I0.numel() > 0:
        snap["I0_min"] = float(I0.min())
        snap["I0_max"] = float(I0.max())
        snap["I0_med"] = float(I0.median())
        snap["I0_mean"] = float(I0.mean())
    else:
        snap["I0_min"] = snap["I0_max"] = 0.0
        snap["I0_med"] = snap["I0_mean"] = 0.0

    def _active_weight_stats(W: torch.Tensor, mask: torch.Tensor, prefix: str) -> None:
        active = W[mask > 0]
        if active.numel() > 0:
            snap[f"{prefix}_min"] = float(active.min())
            snap[f"{prefix}_max"] = float(active.max())
            snap[f"{prefix}_med"] = float(active.median())
            snap[f"{prefix}_mean"] = float(active.mean())
        else:
            snap[f"{prefix}_min"] = snap[f"{prefix}_max"] = 0.0
            snap[f"{prefix}_med"] = snap[f"{prefix}_mean"] = 0.0

    _active_weight_stats(model.W_sv, model.T_sv, "W_sv")
    _active_weight_stats(model.W_dcv, model.T_dcv, "W_dcv")

    # Reversal potentials
    E_sv = model.E_sv
    if E_sv.numel() > 1:
        snap["E_sv_mean"] = float(E_sv.mean())
        snap["E_sv_min"] = float(E_sv.min())
        snap["E_sv_max"] = float(E_sv.max())
    else:
        val = float(E_sv)
        snap["E_sv_mean"] = snap["E_sv_min"] = snap["E_sv_max"] = val
    snap["E_dcv"] = float(model.E_dcv.mean())

    # Time constants τ (already in seconds) — store as lists for multi-rank
    snap["tau_sv"] = model.tau_sv.tolist() if model.tau_sv.numel() > 0 else []
    snap["tau_dcv"] = model.tau_dcv.tolist() if model.tau_dcv.numel() > 0 else []

    # Stimulus weights ‖b‖
    snap["b_norm"] = float(model.b.pow(2).sum().sqrt()) if model.b.numel() > 0 else 0.0

    return snap


# --------------------------------------------------------------------------- #
#  Main entry point                                                             #
# --------------------------------------------------------------------------- #

def train_stage2(
    cfg: Stage2PTConfig,
    save_dir: str | None = None,
    show: bool = False,
) -> Optional[dict]:
    data = load_data_pt(cfg)
    data["_cfg"] = cfg

    u_stage1 = data.get("u_stage1")
    if u_stage1 is None:
        raise ValueError("Stage 1 u_mean is required. Run Stage 1 first.")
    u_var_stage1 = data.get("u_var_stage1")
    sigma_u = data["sigma_u"]

    if cfg.behavior_dataset is None and "behavior_dataset" in data:
        cfg.behavior_dataset = data["behavior_dataset"]
    if cfg.motor_neurons is None and "motor_neurons" in data:
        cfg.motor_neurons = tuple(data["motor_neurons"])

    T, N = u_stage1.shape
    device = torch.device(cfg.device)
    d_ell = data.get("d_ell", 0)
    gating_data = data.get("gating")
    stim_data = data.get("stim")
    b_seq = data.get("b")

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _save_run_config(cfg, save_dir)
        _tee = _TeeWriter(Path(save_dir) / "run.log")
        sys.stdout = _tee
    else:
        _tee = None

    try:  # ensure tee is closed even on crash
        from .init_from_data import init_lambda_u, init_all_from_data
        lambda_u_init = init_lambda_u(u_stage1, cfg)
        beh_all_baseline = behaviour_all_neurons_prediction(data)

        # ---- neurotransmitter sign data for reversals ----------------------------
        sign_t = data.get("sign_t")

        # ---- build model ---------------------------------------------------------
        model = Stage2ModelPT(
            N, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
            cfg, device, d_ell=d_ell,
            lambda_u_init=lambda_u_init,
            sign_t=sign_t,
        ).to(device)

        init_all_from_data(model, u_stage1, cfg)

        # ---- init decomposition diagnostic --------------------------------------
        from .plot_eval import _compute_input_decomposition
        with torch.no_grad():
            decomp, per_neuron = _compute_input_decomposition(model, data)
            # Pretty-print with plain labels for terminal
            _label_map = {
                "Target\nresidual": "AR1_resid",
                "$\\lambda I_{gap}$": "λI_gap",
                "$\\lambda I_{sv}$": "λI_sv",
                "$\\lambda I_{dcv}$": "λI_dcv",
                "$\\lambda I_{stim}$": "λI_stim",
                "Unexplained": "unexpl",
            }
            parts = "  ".join(
                f"{_label_map.get(k, k)}={v:.4f}" for k, v in decomp.items()
            )
            print(f"[Stage2][init-decomp] RMS:  {parts}")
            # Per-neuron: flag neurons where unexplained >> target residual
            resid_rms = per_neuron[:, 0]  # target residual per neuron
            gap_rms = per_neuron[:, 1]
            net_rms = np.sqrt(per_neuron[:, 1]**2 + per_neuron[:, 2]**2 + per_neuron[:, 3]**2)
            ratio = net_rms / np.maximum(resid_rms, 1e-12)
            n_over = int((ratio > 2.0).sum())
            n_under = int((ratio < 0.1).sum())
            print(f"[Stage2][init-decomp] Per-neuron: "
                  f"{n_over}/{len(ratio)} have network/resid > 2×  "
                  f"{n_under}/{len(ratio)} have network/resid < 0.1×")

        _log_config(cfg, d_ell)
        with torch.no_grad():
            _log_init_params(model, cfg)

        # ---- optimiser -----------------------------------------------------------
        syn_lr_mult = float(getattr(cfg, "synaptic_lr_multiplier", 1.0) or 1.0)
        syn_names = {"_a_sv_raw", "_a_dcv_raw", "_W_sv_raw", "_W_dcv_raw"}
        syn_params = [p for n, p in model.named_parameters() if n in syn_names and p.requires_grad]
        other_params = [p for n, p in model.named_parameters() if n not in syn_names and p.requires_grad]
        param_groups = [{"params": other_params, "lr": cfg.learning_rate}]
        if syn_params:
            param_groups.append({"params": syn_params, "lr": cfg.learning_rate * syn_lr_mult})
            print(f"[Stage2] Synaptic LR multiplier: {syn_lr_mult:.1f}x "
                  f"({len(syn_params)} synaptic params @ lr={cfg.learning_rate * syn_lr_mult:.5f})")
        params = other_params + syn_params
        optimiser = optim.Adam(param_groups)

        z_obs = u_stage1.to(device)
        z_target = z_obs
        uvar = u_var_stage1.to(device) if u_var_stage1 is not None else None

        use_uvar = bool(getattr(cfg, "use_u_var_weighting", False))
        uvar_scale = float(getattr(cfg, "u_var_scale", 1.0))
        uvar_floor = float(getattr(cfg, "u_var_floor", 1e-8))
        if use_uvar and uvar is None:
            print("[Stage2][warn] use_u_var_weighting requested but u_var missing; using sigma_u^2 only.")

        print(f"[Stage2] One-step training: T={T}, N={N}, dt={data['dt']:.4f}s")

        # ---- rollout helper ------------------------------------------------------
        def compute_prior(u: torch.Tensor) -> torch.Tensor:
            s_sv = torch.zeros((N, model.r_sv), device=device)
            s_dcv = torch.zeros((N, model.r_dcv), device=device)
            preds = [u[0]]
            for t in range(1, T):
                g = gating_data[t - 1] if gating_data is not None else torch.ones(N, device=device)
                s = stim_data[t - 1] if stim_data is not None else None
                u_next, s_sv, s_dcv = model.prior_step(u[t - 1], s_sv, s_dcv, g, s)
                preds.append(u_next)
            return torch.stack(preds)

        def posterior_check(prior_mu: torch.Tensor, target: torch.Tensor) -> None:
            if uvar is None:
                return
            ok = torch.isfinite(target) & torch.isfinite(prior_mu) & torch.isfinite(uvar) & (uvar > 0)
            n = int(ok.sum().item())
            if n == 0:
                return
            zs = ((target - prior_mu) / torch.sqrt(uvar + 1e-8))[ok]
            print(
                f"[Stage2][PosteriorCheck] z-score mean={zs.mean():+.3f} "
                f"std={zs.std(unbiased=False):.3f} "
                f"P(|z|>2)={(zs.abs() > 2).float().mean():.3f} "
                f"P(|z|>3)={(zs.abs() > 3).float().mean():.3f} (n={n})"
            )

        # ---- behaviour decoder ---------------------------------------------------
        behavior_weight = float(getattr(cfg, "behavior_weight", 0.0) or 0.0)
        beh_decoder = None
        if behavior_weight > 0.0 and b_seq is not None:
            print("[Stage2] Initialising learnable behaviour decoder ...")
            beh_decoder = init_behaviour_decoder(data)
            if beh_decoder is None:
                print("[Stage2][warn] Behaviour decoder init failed; behaviour loss disabled.")
            elif beh_decoder["type"] == "mlp":
                mlp_params = list(beh_decoder["model"].parameters())
                params.extend(mlp_params)
                optimiser.add_param_group({"params": mlp_params})
            else:
                params.append(beh_decoder["W"])
                optimiser.add_param_group({"params": [beh_decoder["W"]]})

        epoch_losses: list[dict] = []
        grad_clip = float(getattr(cfg, "grad_clip_norm", 0.0) or 0.0)
        ridge_b = float(getattr(cfg, "ridge_b", 0.0) or 0.0)
        dynamics_l2 = float(getattr(cfg, "dynamics_l2", 0.0) or 0.0)
        dynamics_objective = str(getattr(cfg, "dynamics_objective", "one_step") or "one_step").strip().lower()
        rollout_steps = int(getattr(cfg, "rollout_steps", 0) or 0)
        rollout_weight = float(getattr(cfg, "rollout_weight", 0.0) or 0.0)
        rollout_starts = int(getattr(cfg, "rollout_starts", 0) or 0)
        warmstart_rollout = bool(getattr(cfg, "warmstart_rollout", False))
        interaction_l2 = float(getattr(cfg, "interaction_l2", 0.0) or 0.0)
        ridge_W_sv = float(getattr(cfg, "ridge_W_sv", 0.0) or 0.0)
        ridge_W_dcv = float(getattr(cfg, "ridge_W_dcv", 0.0) or 0.0)
        plot_every = int(getattr(cfg, "plot_every", 0) or 0)

        # Interaction L2 penalty (penalises network-driven component beyond AR(1))
        if interaction_l2 > 0:
            print(f"[Stage2] Interaction L2 penalty = {interaction_l2:.4g}")
        if ridge_W_sv > 0:
            print(f"[Stage2] Ridge W_sv = {ridge_W_sv:.4g}")
        if ridge_W_dcv > 0:
            print(f"[Stage2] Ridge W_dcv = {ridge_W_dcv:.4g}")
        if dynamics_l2 > 0:
            print(f"[Stage2] Dynamics L2 = {dynamics_l2:.4g}")
        if dynamics_objective == "rollout":
            print(f"[Stage2] Dynamics objective = rollout ({rollout_steps} steps, {rollout_starts} starts)")
        elif rollout_weight > 0 and rollout_steps > 0:
            print(f"[Stage2] Rollout auxiliary = {rollout_weight:.4g} × rollout_loss ({rollout_steps} steps, {rollout_starts} starts)")
        alpha_cv_every = int(getattr(cfg, "alpha_cv_every", 0) or 0)
        alpha_cv_blend = float(getattr(cfg, "alpha_cv_blend", 1.0) or 1.0)
        alpha_cv_max_ratio = float(getattr(cfg, "alpha_cv_max_ratio", 0.0) or 0.0)
        if alpha_cv_every > 0:
            print(f"[Stage2] Alpha ridge-CV every {alpha_cv_every} epochs (blend={alpha_cv_blend:.2f})")

        # ---- training loop -------------------------------------------------------
        for epoch in range(cfg.num_epochs):
            if (alpha_cv_every > 0
                    and epoch > 0
                    and epoch % alpha_cv_every == 0):
                try:
                    alpha_result = ridge_cv_solve_alpha(model, data, cfg)
                    inject_alpha_into_model(
                        model, alpha_result,
                        blend=alpha_cv_blend,
                        max_ratio=alpha_cv_max_ratio,
                    )
                    print(f"[Stage2] Alpha-CV @ epoch {epoch}: "
                          f"{alpha_result['n_updated']}/{N} updated, "
                          f"{alpha_result['n_at_upper']} at upper boundary")
                except Exception as e:
                    print(f"[Stage2][warn] Alpha-CV failed at epoch {epoch}: {e}")

            optimiser.zero_grad()

            prior_mu = compute_prior(z_target)
            one_step_loss = compute_dynamics_loss(
                z_target,
                prior_mu,
                sigma_u,
                u_var=uvar,
                use_u_var_weighting=use_uvar,
                u_var_scale=uvar_scale,
                u_var_floor=uvar_floor,
            )

            use_rollout_main = dynamics_objective == "rollout" and rollout_steps > 0 and rollout_starts > 0
            use_rollout_aux = dynamics_objective != "rollout" and rollout_weight > 0 and rollout_steps > 0 and rollout_starts > 0
            cached_states = None
            if warmstart_rollout and (use_rollout_main or use_rollout_aux):
                cached_states = compute_teacher_forced_states(
                    model,
                    z_target,
                    gating_data=gating_data,
                    stim_data=stim_data,
                )

            rollout_loss_val = None
            if use_rollout_main or use_rollout_aux:
                rollout_loss_val = compute_rollout_loss(
                    model,
                    z_target,
                    sigma_u,
                    rollout_steps=rollout_steps,
                    rollout_starts=rollout_starts,
                    gating_data=gating_data,
                    stim_data=stim_data,
                    cached_states=cached_states if warmstart_rollout else None,
                )

            dynamics_loss = rollout_loss_val if use_rollout_main and rollout_loss_val is not None else one_step_loss

            if epoch in (0, cfg.num_epochs - 1):
                with torch.no_grad():
                    posterior_check(prior_mu, z_target)

            loss = dynamics_loss
            if use_rollout_aux and rollout_loss_val is not None:
                loss = loss + rollout_weight * rollout_loss_val
            if ridge_b > 0 and model.d_ell > 0:
                loss = loss + ridge_b * model.b.pow(2).mean()
            if dynamics_l2 > 0:
                dyn_l2_terms = [p.pow(2).mean() for p in model.parameters() if p.requires_grad]
                if dyn_l2_terms:
                    loss = loss + dynamics_l2 * torch.stack(dyn_l2_terms).mean()

            # Ridge on synaptic edge weights (shrinks toward prior)
            if ridge_W_sv > 0 and model._W_sv_raw.requires_grad:
                W_sv_active = model.W_sv[model.T_sv > 0]
                if W_sv_active.numel() > 0:
                    loss = loss + ridge_W_sv * W_sv_active.pow(2).mean()
            if ridge_W_dcv > 0 and model._W_dcv_raw.requires_grad:
                W_dcv_active = model.W_dcv[model.T_dcv > 0]
                if W_dcv_active.numel() > 0:
                    loss = loss + ridge_W_dcv * W_dcv_active.pow(2).mean()

            # Interaction L2: penalise network-driven component beyond AR(1)
            if interaction_l2 > 0:
                with torch.no_grad():
                    lam = model.lambda_u.detach()
                    I0_det = model.I0.detach()
                    ar1_mu = (1.0 - lam) * z_target[:-1] + lam * I0_det
                interaction = prior_mu[1:] - ar1_mu
                loss = loss + interaction_l2 * interaction.pow(2).mean()

            beh_loss_val = None
            if beh_decoder is not None and behavior_weight > 0.0:
                beh_loss_val = compute_behaviour_loss(prior_mu, beh_decoder)
                loss = loss + behavior_weight * beh_loss_val

            apply_training_step(
                loss,
                optimiser,
                params,
                model,
                cfg,
                grad_clip=grad_clip,
            )

            rec = {"dynamics": dynamics_loss.item(), "total": loss.item()}
            rec["one_step_loss"] = one_step_loss.item()
            if rollout_loss_val is not None:
                rec["rollout_loss"] = rollout_loss_val.item()
            if beh_loss_val is not None:
                rec["behaviour_loss"] = beh_loss_val.item()
            rec.update(_snapshot_params(model))
            epoch_losses.append(rec)

            parts = [f"dynamics={dynamics_loss.item():.4f}", f"total={loss.item():.4f}"]
            if rollout_loss_val is not None:
                parts.append(f"rollout={rollout_loss_val.item():.4f}")
            if beh_loss_val is not None:
                parts.append(f"beh_loss={beh_loss_val.item():.4f}")
            print(f"[Stage2] Epoch {epoch + 1}/{cfg.num_epochs}: {'  '.join(parts)}")

            if save_dir and plot_every > 0 and (epoch + 1) % plot_every == 0:
                try:
                    epoch_dir = Path(save_dir) / f"epoch_{epoch + 1:04d}"
                    ev = generate_eval_loo_plots(
                        model=model, data=data, cfg=cfg,
                        epoch_losses=epoch_losses, save_dir=str(epoch_dir), show=False,
                        decoder=beh_decoder,
                        beh_all_baseline=beh_all_baseline,
                        include_ridge_diagnostics=False,
                    )
                    if ev is not None:
                        br2 = ev.get("beh_r2_model")
                        if br2 is not None:
                            epoch_losses[-1]["beh_r2_eval"] = br2
                    print(f"[Stage2] Plots saved to {epoch_dir}/")
                except Exception as e:
                    print(f"[Stage2][warn] plotting failed at epoch {epoch + 1}: {e}")

        # ---- save ----------------------------------------------------------------
        params_final = snapshot_model_state(model)
        results_h5 = None if save_dir is None else str(Path(save_dir) / "stage2_results.h5")
        save_results_pt(cfg, z_target.detach(), params_final, output_path=results_h5)
        if results_h5 is not None:
            print(f"[Stage2] Training complete. Results saved to {results_h5}")
        else:
            print(f"[Stage2] Training complete. Results saved to {cfg.h5_path}")

        # ---- final evaluation ----------------------------------------------------
        eval_result = None
        if save_dir or show:
            eval_result = generate_eval_loo_plots(
                model=model, data=data, cfg=cfg,
                epoch_losses=epoch_losses,
                save_dir=save_dir or "plots/eval_loo", show=show,
                decoder=beh_decoder,
                beh_all_baseline=beh_all_baseline,
            )
            if eval_result is not None:
                br2 = eval_result.get("beh_r2_model")
                if br2 is not None:
                    epoch_losses[-1]["beh_r2_eval"] = br2

            return eval_result

    finally:  # close log tee
        if _tee is not None:
            _tee.close()
