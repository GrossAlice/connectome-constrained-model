from __future__ import annotations

import dataclasses
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # Used by train_multi.py:
    "compute_dynamics_loss",
    "snapshot_model_state",
    "joint_cv_solve",
    "inject_joint_cv",
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
    model_sigma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dynamics loss: weighted MSE *or* Gaussian NLL when *model_sigma* is given.

    When *model_sigma* (shape ``(N,)`` or ``(T, N)``) is provided the loss is
    the proper Gaussian negative-log-likelihood::

        NLL = 0.5 * [ (y - mu)^2 / sigma^2  +  log(sigma^2) ]

    averaged over valid elements.  The ``log(sigma^2)`` term lets the model
    learn per-neuron noise levels: neurons with large irreducible residuals
    get larger sigma, reducing their gradient magnitude; well-predicted
    neurons get tighter sigma, sharpening the loss.

    For *heteroscedastic* models, *model_sigma* has shape ``(T, N)`` where
    each row is the state-dependent noise for that time step.
    """
    T = target.shape[0]

    if model_sigma is not None:
        # --- Gaussian NLL path (learnable noise) ---
        # model_sigma is (N,) [homoscedastic] or (T, N) [heteroscedastic]
        if model_sigma.dim() == 1:
            sigma2 = (model_sigma.view(1, -1) ** 2).expand(T, -1)
        else:
            sigma2 = model_sigma ** 2
        resid2 = (target - prior_mu) ** 2
        valid = (torch.isfinite(resid2) & torch.isfinite(sigma2)
                 & (sigma2 > 0))
        if not valid.any():
            return torch.tensor(0.0, device=target.device)
        nll = 0.5 * (resid2 / sigma2 + torch.log(sigma2))
        return torch.where(valid, nll, torch.zeros_like(nll)).sum() / valid.sum().clamp(min=1)

    # --- Original weighted-MSE path ---
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
    use_nll: bool = False,
) -> torch.Tensor:
    """Short-horizon rollout loss over K-step free-running predictions.

    Picks *rollout_starts* random starting times, runs the model for
    *rollout_steps* steps without teacher forcing, and computes the
    variance-normalised MSE against the ground truth trajectory.

    When *use_nll* is True (and the model has a learnable noise head),
    uses Gaussian NLL with ``model.sigma_at(u)`` so that the noise
    parameters are trained on multi-step rollout residuals — not only
    on one-step teacher-forced residuals.  This yields wider (and more
    calibrated) uncertainty bands during evaluation.

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
                if use_nll:
                    sig = model.sigma_at(u_cur)          # (N,)
                    var = sig ** 2
                    nll = 0.5 * (resid2 / var + var.log())
                    seg_loss = seg_loss + nll[ok].mean()
                else:
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


def compute_loo_aux_loss(
    model: "Stage2ModelPT",
    u_target: torch.Tensor,
    sigma_u: torch.Tensor,
    *,
    loo_steps: int = 20,
    loo_neurons: int = 4,
    loo_starts: int = 1,
    gating_data: Optional[torch.Tensor] = None,
    stim_data: Optional[torch.Tensor] = None,
    cached_states: Optional[dict] = None,
    use_nll: bool = False,
) -> torch.Tensor:
    """LOO auxiliary loss: hold out random neurons and free-run them.

    For each sampled neuron *i* and start time *t0*, all neurons except *i*
    are clamped to ground truth while neuron *i* runs autonomously for
    *loo_steps* steps.  The loss is the variance-normalised MSE of the
    held-out neuron's trajectory — exactly what the LOO evaluation measures,
    but differentiable and short-horizon.

    When *use_nll* is True, uses Gaussian NLL with ``model.sigma_at(u)``
    so that the noise parameters learn from multi-step LOO residuals.
    """
    T, N = u_target.shape
    device = u_target.device
    K = min(loo_steps, T - 2)
    if K <= 0 or N == 0:
        return torch.tensor(0.0, device=device)

    sigma2 = (sigma_u ** 2).squeeze()

    # Sample held-out neurons and start times
    n_neurons = min(loo_neurons, N)
    neuron_ids = torch.randperm(N, device=device)[:n_neurons]
    max_start = max(T - K - 1, 0)
    n_starts = min(loo_starts, max_start + 1)
    t0s = torch.randperm(max_start + 1, device=device)[:n_starts]

    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for ni in range(n_neurons):
        i = int(neuron_ids[ni].item())
        for si in range(n_starts):
            t0 = int(t0s[si].item())

            # Warm-start synaptic states from teacher-forced trajectory
            if cached_states is not None:
                s_sv = cached_states["s_sv"][t0].clone()
                s_dcv = cached_states["s_dcv"][t0].clone()
            else:
                s_sv = torch.zeros(N, model.r_sv, device=device)
                s_dcv = torch.zeros(N, model.r_dcv, device=device)

            u_pred_i = u_target[t0, i]  # seed with true IC
            seg_loss = torch.tensor(0.0, device=device)
            seg_count = 0

            for k in range(K):
                t = t0 + k
                # Clamp all neurons to GT, substitute held-out neuron
                u_t = u_target[t].clone()
                u_t[i] = u_pred_i

                g = gating_data[t] if gating_data is not None else torch.ones(N, device=device)
                s = stim_data[t] if stim_data is not None else None
                u_next, s_sv, s_dcv = model.prior_step(u_t, s_sv, s_dcv, g, s)

                # Only compute loss for the held-out neuron
                target_i = u_target[t + 1, i]
                resid2 = (u_next[i] - target_i) ** 2
                if torch.isfinite(resid2):
                    if use_nll:
                        sig_i = model.sigma_at(u_t)[i]     # scalar
                        var_i = sig_i ** 2
                        nll_i = 0.5 * (resid2 / var_i + var_i.log())
                        seg_loss = seg_loss + nll_i
                    else:
                        seg_loss = seg_loss + resid2 / sigma2[i].clamp(min=1e-8)
                    seg_count += 1

                u_pred_i = u_next[i]  # free-running for the held-out neuron

            if seg_count > 0:
                total_loss = total_loss + seg_loss / seg_count
                n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / n_valid


def snapshot_model_state(model: Stage2ModelPT) -> Dict[str, torch.Tensor]:
    params_final = {name: p.detach().cpu() for name, p in model.named_parameters()}
    with torch.no_grad():
        params_final["W_sv"] = model.W_sv.detach().cpu()
        params_final["W_dcv"] = model.W_dcv.detach().cpu()
        # Store constrained values alongside raw parameters
        for attr in ("lambda_u", "G", "a_sv", "a_dcv", "tau_sv", "tau_dcv"):
            params_final[attr] = getattr(model, attr).detach().cpu()
        # Per-neuron process noise (always present, but only meaningful
        # when learn_noise is True)
        params_final["sigma_process"] = model.sigma_process.detach().cpu()
    return params_final


# --------------------------------------------------------------------------- #
#  Per-neuron ridge-CV solver for kernel amplitudes (merged from ridge_alpha.py)
# --------------------------------------------------------------------------- #

def _compute_joint_cv_features(
    model,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute joint features (gap + synaptic) and target for dynamics CV.

    Target for neuron *i* at time *t*::

        y_i(t) = [u_i(t+1) - (1-lam_i)*u_i(t)] / lam_i  -  I_stim_i(t)
               = I0_i  +  G * (L_struct @ u(t))_i  +  I_sv_i  +  I_dcv_i

    Features per neuron:
      - gap:  (L_struct @ u(t))_i   (structural Laplacian, G≡1)
      - sv:   per-rank synaptic pooling patterns   (r_sv features)
      - dcv:  per-rank DCV pooling patterns         (r_dcv features)

    The intercept captures I0.  All parameters compete in a single
    per-neuron ridge regression, avoiding the sequential gap-then-synaptic
    decomposition that caused catastrophic shrinkage.
    """
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    T, N = u.shape
    gating = data.get("gating")
    stim = data.get("stim")

    with torch.no_grad():
        lam = model.lambda_u.detach()
        # Structural Laplacian (G≡1) so the gap coefficient directly
        # estimates the optimal absolute G value.
        L_struct = model.laplacian_with_G(torch.ones(1, device=device)).detach()

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

        gap_feat = torch.zeros(T - 1, N, device=device)
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

            # Gap feature from structural Laplacian (G≡1)
            gap_feat[t - 1] = L_struct @ u_prev

            # Target: [u(t+1) - (1-λ)*u(t)] / λ  minus stimulus
            # This equals I0 + G*(L_struct @ u) + I_sv + I_dcv
            residual = u[t] - (1.0 - lam) * u_prev

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
        "gap_features": gap_feat.cpu().numpy().astype(np.float64),
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
    nonneg_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Solve a single-target ridge regression with k-fold CV for λ selection.

    Within each CV fold, column means / stds and the target mean are
    computed on the *training* rows only, then applied to both training
    and held-out rows — avoiding validation-set leakage.

    Parameters
    ----------
    nonneg_mask : optional bool array of shape (p,)
        If provided, coefficients at True positions are clamped to ≥ 0
        during CV evaluation and final fit.  Other positions are
        unconstrained.  If None, no non-negativity is applied.
    """
    T_eff, p = X.shape
    folds = _make_contiguous_folds(np.arange(T_eff), n_folds)
    n_lam = len(ridge_grid)
    eye_p = np.eye(p, dtype=np.float64)

    # ---- Cross-validation (fold-outer, λ-inner for efficiency) -------
    fold_errors: list[list[float]] = [[] for _ in range(n_lam)]

    for fold_idx in folds:
        train_mask = np.ones(T_eff, dtype=bool)
        train_mask[fold_idx] = False
        tr = np.where(train_mask)[0]
        if tr.size < max(p + 1, 5) or fold_idx.size == 0:
            continue

        # Per-fold standardisation (training rows only)
        X_tr, y_tr = X[tr], y[tr]
        x_mean_tr = X_tr.mean(axis=0)
        x_std_tr  = X_tr.std(axis=0)
        x_std_tr  = np.where(x_std_tr > 1e-12, x_std_tr, 1.0)
        y_mean_tr = float(y_tr.mean())

        Xs_tr = (X_tr - x_mean_tr) / x_std_tr
        yc_tr = y_tr - y_mean_tr

        gram_base = Xs_tr.T @ Xs_tr
        rhs       = Xs_tr.T @ yc_tr

        # Apply training stats to held-out fold
        Xs_val = (X[fold_idx] - x_mean_tr) / x_std_tr
        yc_val = y[fold_idx] - y_mean_tr

        for lam_idx, lam in enumerate(ridge_grid):
            gram = gram_base + lam * eye_p if lam > 0 else gram_base
            try:
                coef_s = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                coef_s, *_ = np.linalg.lstsq(gram, rhs, rcond=None)

            # Apply the same constraint that will be used at injection time.
            coef_nn = coef_s.copy()
            if nonneg_mask is not None:
                coef_nn[nonneg_mask] = np.maximum(coef_nn[nonneg_mask], 0.0)
            mse = float(np.mean((yc_val - Xs_val @ coef_nn) ** 2))
            if np.isfinite(mse):
                fold_errors[lam_idx].append(mse)

    cv_mse = np.full(n_lam, np.inf, dtype=np.float64)
    for lam_idx in range(n_lam):
        if fold_errors[lam_idx]:
            cv_mse[lam_idx] = float(np.mean(fold_errors[lam_idx]))

    if not np.any(np.isfinite(cv_mse)):
        best_idx = 0
    else:
        best_idx = int(np.nanargmin(np.where(np.isfinite(cv_mse), cv_mse, np.inf)))
    best_lambda = float(ridge_grid[best_idx])

    # ---- Final fit on all data (full-data standardisation) -----------
    x_mean = X.mean(axis=0)
    x_std  = X.std(axis=0)
    x_std  = np.where(x_std > 1e-12, x_std, 1.0)
    y_mean = float(y.mean())

    Xs = (X - x_mean) / x_std
    yc = y - y_mean

    gram_full = Xs.T @ Xs
    if best_lambda > 0:
        gram_full = gram_full + best_lambda * eye_p
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


# --------------------------------------------------------------------------- #
#  Joint dynamics CV: solve I0 + G + a_sv + a_dcv per neuron simultaneously   #
# --------------------------------------------------------------------------- #

def joint_cv_solve(
    model,
    data: Dict[str, Any],
    cfg=None,
    *,
    n_folds: int = 5,
    log_lambda_min: float = -2.0,
    log_lambda_max: float = 6.0,
    n_grid: int = 50,
    r2_gate: float = 0.01,
) -> Dict[str, Any]:
    """Solve per-neuron I0, G_coef, a_sv, a_dcv jointly via ridge-CV.

    Instead of separate alpha-CV (synaptic only) and backbone-CV (gap only),
    this puts all features into a single per-neuron regression.  The gap
    and synaptic features compete on equal footing, so the solver allocates
    variance correctly even when gap junctions dominate.

    Features per neuron i (total = 1 + r_sv + r_dcv):
      col 0:          gap feature  (L_struct @ u)_i   →  coefficient = G_scale
      cols 1..r_sv:   SV synaptic pooling patterns     →  coefficients = a_sv[i,:]
      cols r_sv+1..:  DCV synaptic pooling patterns    →  coefficients = a_dcv[i,:]
      intercept (via centering):                        →  I0_i
    """
    if cfg is not None:
        n_folds = int(getattr(cfg, "dynamics_cv_n_folds", n_folds) or n_folds)
        log_lambda_min = float(getattr(cfg, "dynamics_cv_log_min", log_lambda_min))
        log_lambda_max = float(getattr(cfg, "dynamics_cv_log_max", log_lambda_max))
        n_grid = int(getattr(cfg, "dynamics_cv_n_grid", n_grid) or n_grid)
        r2_gate = float(getattr(cfg, "dynamics_cv_r2_gate", r2_gate))

    ridge_grid = _log_ridge_grid(log_lambda_min, log_lambda_max, n_grid)
    n_grid_actual = len(ridge_grid)
    N = model.N
    r_sv = model.r_sv
    r_dcv = model.r_dcv
    p_syn = r_sv + r_dcv          # synaptic feature count
    p_total = 1 + p_syn           # gap(1) + sv(r_sv) + dcv(r_dcv)
    # Intercept is handled by mean-centering inside _solve_neuron_ridge_cv

    feats = _compute_joint_cv_features(model, data)
    gap = feats["gap_features"]       # (T-1, N)
    feat_sv = feats["features_sv"]    # (T-1, N, r_sv)
    feat_dcv = feats["features_dcv"]  # (T-1, N, r_dcv)
    target = feats["target"]          # (T-1, N)

    # Non-negativity mask: synaptic coefficients ≥ 0, gap coefficient unconstrained
    nonneg = np.zeros(p_total, dtype=bool)
    nonneg[1:] = True   # indices 1.. are synaptic

    # Current values as defaults for neurons that aren't updated
    with torch.no_grad():
        a_sv_cur = model.a_sv.detach().cpu().numpy()
        a_dcv_cur = model.a_dcv.detach().cpu().numpy()
        I0_cur = model.I0.detach().cpu().numpy()

    if a_sv_cur.ndim == 1:
        a_sv_out = np.tile(a_sv_cur, (N, 1))
    else:
        a_sv_out = a_sv_cur.copy()
    if a_dcv_cur.ndim == 1:
        a_dcv_out = np.tile(a_dcv_cur, (N, 1))
    else:
        a_dcv_out = a_dcv_cur.copy()
    I0_out = I0_cur.copy()
    G_coefs = np.full(N, np.nan, dtype=np.float64)

    lambdas = np.full(N, np.nan, dtype=np.float64)
    fit_r2 = np.full(N, np.nan, dtype=np.float64)
    cv_mse_all = np.full((N, n_grid_actual), np.nan, dtype=np.float64)
    at_upper_flags = np.zeros(N, dtype=bool)
    n_updated = 0
    n_at_upper = 0
    n_gated = 0

    for i in range(N):
        # Build feature matrix: [gap | sv_ranks | dcv_ranks]
        parts = [gap[:, i : i + 1]]    # (T-1, 1)
        if r_sv > 0:
            parts.append(feat_sv[:, i, :])   # (T-1, r_sv)
        if r_dcv > 0:
            parts.append(feat_dcv[:, i, :])  # (T-1, r_dcv)

        X_i = np.concatenate(parts, axis=1)  # (T-1, p_total)
        y_i = target[:, i]

        valid = np.all(np.isfinite(X_i), axis=1) & np.isfinite(y_i)
        n_valid = int(valid.sum())
        if n_valid < max(p_total + 5, 20):
            continue

        x_norms = np.abs(X_i[valid]).max(axis=0)
        if x_norms.max() < 1e-10:
            continue

        result = _solve_neuron_ridge_cv(
            X_i[valid], y_i[valid], ridge_grid, n_folds, nonneg_mask=nonneg,
        )
        coef = result["coef"]

        # Compute fit R² (with non-negativity applied to synaptic only)
        coef_constrained = coef.copy()
        coef_constrained[nonneg] = np.maximum(coef_constrained[nonneg], 0.0)
        y_pred = X_i[valid] @ coef_constrained + result["intercept"]
        ss_res = float(np.sum((y_i[valid] - y_pred.ravel()) ** 2))
        ss_tot = float(np.sum((y_i[valid] - y_i[valid].mean()) ** 2))
        fit_r2[i] = 1.0 - ss_res / max(ss_tot, 1e-12)

        cv_mse_i = result["cv_mse"]
        cv_mse_all[i, :len(cv_mse_i)] = cv_mse_i
        lambdas[i] = result["best_lambda"]
        if result["at_upper"]:
            n_at_upper += 1
            at_upper_flags[i] = True

        # R² quality gate: skip neuron if the joint fit has no skill
        if fit_r2[i] < r2_gate:
            n_gated += 1
            continue

        # Extract coefficients
        G_coefs[i] = coef_constrained[0]   # gap (unconstrained)
        idx = 1
        if r_sv > 0:
            a_sv_out[i] = coef_constrained[idx : idx + r_sv]  # already ≥ 0
            idx += r_sv
        if r_dcv > 0:
            a_dcv_out[i] = coef_constrained[idx : idx + r_dcv]
            idx += r_dcv
        I0_out[i] = result["intercept"]
        n_updated += 1

    # Robust G scale: median of per-neuron gap coefficients (positive only)
    valid_G = G_coefs[np.isfinite(G_coefs) & (G_coefs > 0)]
    G_scale = float(np.median(valid_G)) if len(valid_G) > 0 else 0.0

    device = next(model.parameters()).device
    return {
        "I0": torch.tensor(I0_out, dtype=torch.float32, device=device),
        "alpha_sv": torch.tensor(a_sv_out, dtype=torch.float32, device=device),
        "alpha_dcv": torch.tensor(a_dcv_out, dtype=torch.float32, device=device),
        "G_scale": G_scale,
        "G_coefs": G_coefs,
        "fit_r2": fit_r2,
        "lambdas": lambdas,
        "ridge_grid": ridge_grid,
        "cv_mse_all": cv_mse_all,
        "n_updated": n_updated,
        "n_at_upper": n_at_upper,
        "n_gated": n_gated,
        "at_upper_flags": at_upper_flags,
    }


def inject_joint_cv(
    model,
    result: Dict[str, Any],
    blend: float = 1.0,
) -> None:
    """Inject joint-CV solved I0, a_sv, a_dcv, and G back into the model."""
    blend = max(0.0, min(1.0, blend))
    with torch.no_grad():
        # ---- I0: blend between current and solved ----
        I0_new = result["I0"]
        model.I0.data.copy_((1.0 - blend) * model.I0 + blend * I0_new)

        # ---- Synaptic amplitudes: blend between current and solved ----
        for attr, key in [("a_sv", "alpha_sv"), ("a_dcv", "alpha_dcv")]:
            cur = getattr(model, attr, None)
            if cur is None or cur.numel() == 0:
                continue
            target_val = result[key]
            # Handle shared amplitudes: if model param is (r,) but
            # CV solved per-neuron (N, r), reduce to (r,) via mean.
            if cur.ndim == 1 and target_val.ndim == 2:
                target_val = target_val.mean(dim=0)
            blended = (1.0 - blend) * cur + blend * target_val
            model.set_param_constrained(attr, blended)

        # ---- G: blend toward absolute solved scale ----
        G_scale = result["G_scale"]
        if G_scale > 1e-8:
            G_cur = model.G
            if model.edge_specific_G:
                # Preserve per-edge structure, rescale overall level
                G_nonzero = G_cur[G_cur > 0]
                median_cur = float(G_nonzero.median()) if G_nonzero.numel() > 0 else 1.0
                ratio = G_scale / max(median_cur, 1e-8)
                effective_ratio = 1.0 + blend * (ratio - 1.0)
                if effective_ratio > 0.01:   # safety floor
                    model.set_param_constrained("G", G_cur * effective_ratio)
            else:
                # Scalar G: direct blend
                G_new = (1.0 - blend) * G_cur + blend * torch.tensor(
                    G_scale, device=G_cur.device, dtype=G_cur.dtype)
                if G_new.item() > 1e-4:     # safety floor
                    model.set_param_constrained("G", G_new)


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
    for sec_name in ("data", "dynamics", "stimulus", "behavior", "train", "eval", "output", "multi"):
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
    if model.stim_kernel is not None:
        h = F.softplus(model.stim_kernel.detach())
        print(f"  stim_kernel (train): len={model.stim_kernel_len}  "
              f"peak={h.max():.4f} sum={h.sum():.4f}")
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

    # Stimulus kernel
    if model.stim_kernel is not None:
        h = F.softplus(model.stim_kernel)
        snap["stim_kernel_peak"] = float(h.max())
        snap["stim_kernel_sum"] = float(h.sum())
    else:
        snap["stim_kernel_peak"] = 0.0
        snap["stim_kernel_sum"] = 0.0

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
    stim_data_raw = data.get("stim")          # raw (delta-pulse) stimulus
    stim_data = stim_data_raw                  # will be overwritten by convolved version each epoch
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

        # ---- snapshot init values for regularization anchors ---------------------
        # These fixed tensors serve as targets for optional targeted
        # regularizers (lambda_u_reg, I0_reg, G_reg, tau_reg).
        # They break the degeneracy between lambda_u, I0, G, and alpha
        # amplitudes that would otherwise let parameters trade off freely.
        with torch.no_grad():
            lambda_u_init_anchor = model.lambda_u.detach().clone()
            lambda_u_raw_init_anchor = model._lambda_u_raw.detach().clone()
            I0_init_anchor = model.I0.detach().clone()
            G_init_anchor = model.G.detach().clone()
            tau_sv_init_anchor = model.tau_sv.detach().clone()
            tau_dcv_init_anchor = model.tau_dcv.detach().clone()
            # Snapshot the init network strength product for floor penalty
            _init_G_rms = float(model.G.pow(2).mean().sqrt())
            _init_a_sv_rms = float(model.a_sv.pow(2).mean().sqrt())
            _init_strength = _init_G_rms * _init_a_sv_rms

        # ---- init decomposition diagnostic --------------------------------------
        from .plot_eval import _compute_input_decomposition
        with torch.no_grad():
            decomp, per_neuron = _compute_input_decomposition(model, data)
            # Pretty-print with plain labels for terminal
            _label_map = {
                "AR(1) residual\n(network should explain)": "AR1_resid",
                "$\\lambda I_{gap}$": "λI_gap",
                "$\\lambda I_{sv}$": "λI_sv",
                "$\\lambda I_{dcv}$": "λI_dcv",
                "$\\lambda I_{stim}$": "λI_stim",
                "Network total": "net_total",
                "Model error\n(predicted \u2212 actual)": "model_err",
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
            # Slice uvar to match the (possibly shorter) input tensors
            _uvar = uvar[:prior_mu.shape[0]] if uvar.shape[0] != prior_mu.shape[0] else uvar
            ok = torch.isfinite(target) & torch.isfinite(prior_mu) & torch.isfinite(_uvar) & (_uvar > 0)
            n = int(ok.sum().item())
            if n == 0:
                return
            zs = ((target - prior_mu) / torch.sqrt(_uvar + 1e-8))[ok]
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
                optimiser.add_param_group({"params": mlp_params, "lr": cfg.learning_rate})
            else:
                params.append(beh_decoder["W"])
                optimiser.add_param_group({"params": [beh_decoder["W"]], "lr": cfg.learning_rate})

        epoch_losses: list[dict] = []
        grad_clip = float(getattr(cfg, "grad_clip_norm", 0.0) or 0.0)
        ridge_b = float(getattr(cfg, "ridge_b", 0.0) or 0.0)
        dynamics_l2 = float(getattr(cfg, "dynamics_l2", 0.0) or 0.0)
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
        I0_reg = float(getattr(cfg, "I0_reg", 0.0) or 0.0)
        lambda_u_reg = float(getattr(cfg, "lambda_u_reg", 0.0) or 0.0)
        G_reg = float(getattr(cfg, "G_reg", 0.0) or 0.0)
        tau_reg = float(getattr(cfg, "tau_reg", 0.0) or 0.0)
        if I0_reg > 0:
            print(f"[Stage2] I0 regularization toward init = {I0_reg:.4g}")
        if lambda_u_reg > 0:
            print(f"[Stage2] lambda_u regularization (logit space) toward init = {lambda_u_reg:.4g}")
        if G_reg > 0:
            print(f"[Stage2] G regularization toward init = {G_reg:.4g}")
        if tau_reg > 0:
            print(f"[Stage2] tau regularization (log space) toward init = {tau_reg:.4g}")
        net_floor_weight = float(getattr(cfg, "network_strength_floor", 0.0) or 0.0)
        net_floor_target = float(getattr(cfg, "network_strength_target", 0.8) or 0.8)
        if net_floor_weight > 0:
            print(f"[Stage2] Network strength floor = {net_floor_weight:.4g} "
                  f"(target ≥ {net_floor_target:.0%} of init = {_init_strength * net_floor_target:.4f})")
        if rollout_weight > 0 and rollout_steps > 0:
            print(f"[Stage2] Rollout auxiliary = {rollout_weight:.4g} × rollout_loss ({rollout_steps} steps, {rollout_starts} starts)")
        # LOO auxiliary loss
        loo_aux_weight = float(getattr(cfg, "loo_aux_weight", 0.0) or 0.0)
        loo_aux_steps  = int(getattr(cfg, "loo_aux_steps", 20) or 20)
        loo_aux_neurons = int(getattr(cfg, "loo_aux_neurons", 4) or 4)
        loo_aux_starts = int(getattr(cfg, "loo_aux_starts", 1) or 1)
        if loo_aux_weight > 0:
            print(f"[Stage2] LOO auxiliary = {loo_aux_weight:.4g} \u00d7 loo_loss "
                  f"({loo_aux_steps} steps, {loo_aux_neurons} neurons, {loo_aux_starts} starts)")
        dynamics_cv_every = int(getattr(cfg, "dynamics_cv_every", 0) or 0)
        dynamics_cv_blend = float(getattr(cfg, "dynamics_cv_blend", 0.5) or 0.5)
        dynamics_cv_warmup = max(1, int(getattr(cfg, "dynamics_cv_warmup", 3) or 1))
        if dynamics_cv_every > 0:
            print(f"[Stage2] Dynamics-CV (joint I0+G+a_sv+a_dcv) every {dynamics_cv_every} epochs "
                  f"(blend={dynamics_cv_blend:.2f}, warmup={dynamics_cv_warmup})")

        # Injection counter for warmup ramp
        _dynamics_cv_count = 0

        # ---- training loop -------------------------------------------------------
        for epoch in range(cfg.num_epochs):
            # ---- Re-convolve stimulus with learnable kernel ------------------
            if stim_data_raw is not None and model.stim_kernel is not None:
                stim_data = model.convolve_stimulus(stim_data_raw)
                # Update data dict so CV solvers see the convolved version
                data["stim"] = stim_data
            else:
                stim_data = stim_data_raw

            # ---- Joint dynamics-CV re-solve ---------------------------------
            _do_cv = (dynamics_cv_every > 0
                      and epoch > 0
                      and epoch % dynamics_cv_every == 0)

            if _do_cv:
                _dynamics_cv_count += 1
                cv_result = None
                try:
                    cv_result = joint_cv_solve(model, data, cfg)
                except Exception as e:
                    print(f"[Stage2][warn] Dynamics-CV solve failed at epoch {epoch}: {e}")

                if cv_result is not None:
                    _cv_ramp = min(_dynamics_cv_count, dynamics_cv_warmup) / dynamics_cv_warmup
                    _cv_blend = dynamics_cv_blend * _cv_ramp

                    inject_joint_cv(model, cv_result, blend=_cv_blend)

                    # Reset Adam momentum for all injected parameters
                    for _p in [model.I0, model._G_raw,
                               model._a_sv_raw, model._a_dcv_raw]:
                        _st = optimiser.state.get(_p, {})
                        _st.pop("exp_avg", None)
                        _st.pop("exp_avg_sq", None)
                        _st.pop("step", None)

                    # Diagnostics
                    valid_r2 = cv_result['fit_r2'][np.isfinite(cv_result['fit_r2'])]
                    r2_str = f"  fit R\u00b2 med={float(np.median(valid_r2)):.3f}" if len(valid_r2) > 0 else ""
                    valid_lams = cv_result['lambdas'][np.isfinite(cv_result['lambdas'])]
                    lam_str = ""
                    if len(valid_lams) > 0:
                        lam_str = (f"  \u03bb med={np.median(valid_lams):.2g}"
                                   f" IQR=[{np.percentile(valid_lams, 25):.2g},"
                                   f" {np.percentile(valid_lams, 75):.2g}]")
                    print(f"[Stage2] Dynamics-CV @ epoch {epoch}: "
                          f"{cv_result['n_updated']}/{N} updated, "
                          f"{cv_result['n_gated']} gated (R\u00b2<thr), "
                          f"{cv_result['n_at_upper']} at upper"
                          f"  G_scale={cv_result['G_scale']:.4f}"
                          f"{r2_str}{lam_str}"
                          f"  blend={_cv_blend:.3f}")

            optimiser.zero_grad()

            prior_mu = compute_prior(z_target)
            # Skip frame 0: prior_mu[0] == z_target[0] by construction,
            # giving a trivially perfect prediction that biases the loss.
            # model_sigma is passed only when learn_noise is True;
            # compute_dynamics_loss then uses Gaussian NLL instead of
            # weighted MSE.  For heteroscedastic models sigma_at(u)
            # returns (T-1, N); for homoscedastic it returns (N,).
            if model._learn_noise:
                if model._noise_mode == "heteroscedastic":
                    _model_sigma = model.sigma_at(z_target[:-1])  # (T-1, N)
                else:
                    _model_sigma = model.sigma_at()               # (N,)
                # When noise_sigma_source="rollout", detach sigma from the
                # one-step NLL so that only the rollout/LOO NLL terms shape
                # the noise magnitude.  This prevents the (tiny) one-step
                # residuals from pulling sigma down to ~0.1 and yields much
                # wider, better-calibrated confidence intervals.
                _sigma_source = str(getattr(cfg, "noise_sigma_source", "all"))
                _has_multistep = (rollout_weight > 0 and rollout_steps > 0) or (loo_aux_weight > 0)
                if _sigma_source == "rollout" and _has_multistep:
                    _model_sigma = _model_sigma.detach()
            else:
                _model_sigma = None
            one_step_loss = compute_dynamics_loss(
                z_target[1:],
                prior_mu[1:],
                sigma_u,
                u_var=uvar,
                use_u_var_weighting=use_uvar,
                u_var_scale=uvar_scale,
                u_var_floor=uvar_floor,
                model_sigma=_model_sigma,
            )

            use_rollout = rollout_weight > 0 and rollout_steps > 0 and rollout_starts > 0
            use_loo_aux = loo_aux_weight > 0 and loo_aux_steps > 0
            cached_states = None
            if warmstart_rollout and (use_rollout or use_loo_aux):
                cached_states = compute_teacher_forced_states(
                    model,
                    z_target,
                    gating_data=gating_data,
                    stim_data=stim_data,
                )

            rollout_loss_val = None
            if use_rollout:
                rollout_loss_val = compute_rollout_loss(
                    model,
                    z_target,
                    sigma_u,
                    rollout_steps=rollout_steps,
                    rollout_starts=rollout_starts,
                    gating_data=gating_data,
                    stim_data=stim_data,
                    cached_states=cached_states if warmstart_rollout else None,
                    use_nll=model._learn_noise,
                )

            loo_loss_val = None
            if use_loo_aux:
                loo_loss_val = compute_loo_aux_loss(
                    model,
                    z_target,
                    sigma_u,
                    loo_steps=loo_aux_steps,
                    loo_neurons=loo_aux_neurons,
                    loo_starts=loo_aux_starts,
                    gating_data=gating_data,
                    stim_data=stim_data,
                    cached_states=cached_states if warmstart_rollout else None,
                    use_nll=model._learn_noise,
                )

            dynamics_loss = one_step_loss

            if epoch in (0, cfg.num_epochs - 1):
                with torch.no_grad():
                    # Exclude frame 0 from posterior check (same reason
                    # as the loss: prior_mu[0] == z_target[0] by construction).
                    posterior_check(prior_mu[1:], z_target[1:])

            loss = dynamics_loss
            if rollout_loss_val is not None:
                loss = loss + rollout_weight * rollout_loss_val
            if loo_loss_val is not None:
                loss = loss + loo_aux_weight * loo_loss_val
            if ridge_b > 0 and model.d_ell > 0:
                loss = loss + ridge_b * model.b.pow(2).mean()
            if dynamics_l2 > 0:
                dyn_l2_terms = [p.pow(2).mean() for p in model.parameters() if p.requires_grad]
                if dyn_l2_terms:
                    loss = loss + dynamics_l2 * torch.stack(dyn_l2_terms).mean()

            # Noise regularisation: prevent sigma collapse to tiny values
            _noise_reg = float(getattr(cfg, "noise_reg", 0.0) or 0.0)
            if _noise_reg > 0 and model._learn_noise:
                if model._noise_mode == "heteroscedastic":
                    loss = loss + _noise_reg * (
                        model._sigma_w.pow(2).mean()
                        + model._sigma_b.pow(2).mean()
                    )
                else:
                    loss = loss + _noise_reg * model._log_sigma_u.pow(2).mean()

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

            # Regularization toward init values (breaks lambda_u / I0 / alpha degeneracy)
            if I0_reg > 0:
                loss = loss + I0_reg * (model.I0 - I0_init_anchor).pow(2).mean()
            if lambda_u_reg > 0:
                # Penalise in raw-parameter space (≈ logit of normalised λ).
                # Using the raw parameter directly avoids precision loss from
                # computing logit(sigmoid(raw)) near the boundaries.
                loss = loss + lambda_u_reg * (model._lambda_u_raw - lambda_u_raw_init_anchor).pow(2).mean()
            if G_reg > 0 and model._G_raw.requires_grad:
                # Regularize gap-junction conductance toward init.
                # G errors can trade against synaptic alpha terms in the
                # alpha-CV target (which subtracts L @ u_prev using current G).
                loss = loss + G_reg * (model.G - G_init_anchor).pow(2).mean()
            if tau_reg > 0:
                # Regularize time constants in log space to keep ranks
                # separated and near plausible initial values.
                _tau_eps = 1e-6
                if model.tau_sv.requires_grad and model.tau_sv.numel() > 0:
                    loss = loss + tau_reg * (
                        torch.log(model.tau_sv.clamp(min=_tau_eps))
                        - torch.log(tau_sv_init_anchor.clamp(min=_tau_eps))
                    ).pow(2).mean()
                if model.tau_dcv.requires_grad and model.tau_dcv.numel() > 0:
                    loss = loss + tau_reg * (
                        torch.log(model.tau_dcv.clamp(min=_tau_eps))
                        - torch.log(tau_dcv_init_anchor.clamp(min=_tau_eps))
                    ).pow(2).mean()

            # Network strength floor: one-sided penalty on G_rms * a_sv_rms
            # Uses ratio-based (scale-invariant) shortfall so the penalty
            # is effective regardless of the absolute magnitude of the target.
            # penalty = weight * relu(1 - current/target)²
            if net_floor_weight > 0 and net_floor_target > 0:
                _cur_strength = model.G.pow(2).mean().sqrt() * model.a_sv.pow(2).mean().sqrt()
                _target = _init_strength * net_floor_target
                _ratio = _cur_strength / max(_target, 1e-12)
                _rel_shortfall = torch.relu(1.0 - _ratio)
                loss = loss + net_floor_weight * _rel_shortfall.pow(2)

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
            if loo_loss_val is not None:
                rec["loo_loss"] = loo_loss_val.item()
            if beh_loss_val is not None:
                rec["behaviour_loss"] = beh_loss_val.item()
            if model._learn_noise:
                _sp = model.sigma_process.detach()
                rec["sigma_process_median"] = float(_sp.median())
                rec["sigma_process_mean"] = float(_sp.mean())
            rec.update(_snapshot_params(model))
            epoch_losses.append(rec)

            parts = [f"dynamics={dynamics_loss.item():.4f}", f"total={loss.item():.4f}"]
            if rollout_loss_val is not None:
                parts.append(f"rollout={rollout_loss_val.item():.4f}")
            if loo_loss_val is not None:
                parts.append(f"loo={loo_loss_val.item():.4f}")
            if beh_loss_val is not None:
                parts.append(f"beh_loss={beh_loss_val.item():.4f}")
            if model._learn_noise:
                _sp = model.sigma_process.detach()
                parts.append(f"σ_med={float(_sp.median()):.4f}")

            # Log current decomposition (G, a_sv, network drive) every CV epoch
            if _do_cv or (epoch + 1) % max(plot_every, 10) == 0:
                with torch.no_grad():
                    _G_rms = float(model.G.pow(2).mean().sqrt())
                    _a_sv_rms = float(model.a_sv.pow(2).mean().sqrt()) if model.a_sv.numel() > 0 else 0.0
                    _decomp, _ = _compute_input_decomposition(model, data)
                    _dm = {
                        "AR(1) residual\n(network should explain)": "AR1",
                        "$\\lambda I_{gap}$": "λI_gap",
                        "$\\lambda I_{sv}$": "λI_sv",
                        "$\\lambda I_{dcv}$": "λI_dcv",
                        "Network total": "net",
                        "Model error\n(predicted \u2212 actual)": "err",
                    }
                    _dp = "  ".join(f"{_dm.get(k, k)}={v:.4f}" for k, v in _decomp.items() if k in _dm)
                    parts.append(f"G_rms={_G_rms:.4f}  a_sv_rms={_a_sv_rms:.4f}  [{_dp}]")

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
