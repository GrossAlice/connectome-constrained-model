"""Multi-worm two-level MAP training loop (Layer 5).

Implements the EPFL two-level alternating optimisation::

    Step 1 (inner, ``u_unobs_inner_steps`` gradient steps):
        Fix θ, ψ  →  update u_U  (trajectory inference, opt_u only)

    Step 2 (outer, 1 gradient step):
        Fix u_U   →  update θ, ψ  (model learning, opt_theta + opt_psi)

MAP loss per worm
-----------------
For a worm with observed set O and unobserved set U, both enter a **single**
variance-normalised MSE::

    J_w = Σ_t Σ_i (û_i(t) - μ_i(t))² / σ²_{u,i}

where σ²_{u,i} is small for i∈O (stage-1 noise) and inflated by
``sigma_u_unobs_scale`` for i∈U.  No mask is applied — the loss naturally
weights unobserved neurons less, while still back-propagating through their
trajectory free variables ``u_unobs``.

Three Adam optimiser groups
---------------------------
opt_theta : shared model params (all ``Stage2ModelPT`` parameters with
    ``requires_grad=True``).  Includes ``_lambda_u_raw`` and ``I0``
    (shared across worms, same as single-worm training).
opt_psi   : per-worm model params (all worms):
    optionally ``_G_raw``, ``b``
    from ``WormState.param_groups()[0]``.  Also the behaviour decoder params.
opt_u     : per-worm trajectory free variables (all worms):
    ``u_unobs`` from ``WormState.param_groups()[1]``.

Public API
----------
``train_multi_worm(cfg, save_dir, show)``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .behavior_decoder_eval import (
    build_lagged_features_np,
    build_lagged_features_torch,
    MLPBehaviourDecoder,
    _log_ridge_grid,
    _ridge_cv_single_target,
)
from .config import Stage2PTConfig
from .init_from_data import init_all_from_data
from .io_multi import load_multi_worm_data
from .model import Stage2ModelPT
from .train import (
    compute_dynamics_loss,
    inject_alpha_into_model,
    snapshot_model_state,
    _TeeWriter,
    _snapshot_params,
    _solve_neuron_ridge_cv,
    _fill_alpha_features,
)
from .worm_state import WormState, build_worm_states

__all__ = ["train_multi_worm"]



def _forward_pass_worm(
    model: Stage2ModelPT,
    worm:  WormState,
    u_full: torch.Tensor,   # (T, N_atlas) — may carry grad through u_unobs
) -> torch.Tensor:          # (T, N_atlas) prior_mu
    """Teacher-forced dynamics for one worm with per-worm overrides.

    ``prior_mu[0] = u_full[0]`` (trivial).
    ``prior_mu[t] = f(u_full[t-1])`` for t ≥ 1, using the shared model's
    ``lambda_u`` and ``I0`` (same as single-worm), plus per-worm
    ``worm.G`` and ``worm.b``.
    """
    T, N = u_full.shape
    device = u_full.device
    s_sv  = torch.zeros(N, model.r_sv,  device=device, dtype=u_full.dtype)
    s_dcv = torch.zeros(N, model.r_dcv, device=device, dtype=u_full.dtype)
    preds: List[torch.Tensor] = [u_full[0]]

    G   = worm.G          # scalar or None
    b   = worm.b          # (N,) or None

    for t in range(T - 1):
        g = worm.gating[t]                                        # (N,)
        s = worm.stim[t] if worm.stim is not None else None       # (N,) or None
        u_next, s_sv, s_dcv = model.prior_step(
            u_full[t], s_sv, s_dcv, g, s,
            G=G, b=b,
        )
        preds.append(u_next)

    return torch.stack(preds)   # (T, N)



def _compute_alpha_features_worm(
    model:   Stage2ModelPT,
    u:       torch.Tensor,                   # (T, N) — no grad needed
    gating:  Optional[torch.Tensor],
    stim:    Optional[torch.Tensor],
    lam_w:   torch.Tensor,                   # (N,) per-worm lambda_u (detached)
    I0_w:    torch.Tensor,                   # (N,) per-worm I0 (detached)
    G_w:     Optional[torch.Tensor],         # per-worm G scalar or None
    b_w:     Optional[torch.Tensor],         # (N,) per-worm stim weight or None
) -> Dict[str, Any]:
    """Feature matrix for ridge-CV alpha (kernel amplitudes), per worm.

    Target: ``(u[t+1] - ar1 - λ·I_gap - λ·I_stim) / λ``
    Features: ``_fill_alpha_features`` identical to single-worm code.

    Returns dict with:
        ``features_sv``  : ndarray (T-1, N, r_sv)
        ``features_dcv`` : ndarray (T-1, N, r_dcv)
        ``target``       : ndarray (T-1, N)
    """
    T, N   = u.shape
    device = u.device

    with torch.no_grad():
        L         = model.laplacian_with_G(G_w).detach()
        T_eff_sv  = (model.T_sv * model.W_sv).detach()
        T_eff_dcv = (model.T_dcv * model.W_dcv).detach()
        tau_sv    = model.tau_sv.detach()
        tau_dcv   = model.tau_dcv.detach()
        E_sv      = model.E_sv.detach()
        E_dcv     = model.E_dcv.detach()
        dt        = model.dt

        gamma_sv  = torch.exp(-dt / (tau_sv  + 1e-12))
        gamma_dcv = torch.exp(-dt / (tau_dcv + 1e-12))
        r_sv, r_dcv = model.r_sv, model.r_dcv

        feat_sv  = torch.zeros(T - 1, N, r_sv,  device=device)
        feat_dcv = torch.zeros(T - 1, N, r_dcv, device=device)
        target   = torch.zeros(T - 1, N, device=device)

        s_sv  = torch.zeros(N, r_sv,  device=device)
        s_dcv = torch.zeros(N, r_dcv, device=device)

        for t in range(1, T):
            u_prev    = u[t - 1]
            g         = gating[t - 1] if gating is not None else torch.ones(N, device=device)
            phi_gated = torch.sigmoid(u_prev) * g.view(N)

            s_sv  = gamma_sv.view(1, -1)  * s_sv  + phi_gated.unsqueeze(1)
            s_dcv = gamma_dcv.view(1, -1) * s_dcv + phi_gated.unsqueeze(1)

            pool_sv  = T_eff_sv.t()  @ s_sv
            pool_dcv = T_eff_dcv.t() @ s_dcv

            _fill_alpha_features(feat_sv,  t - 1, pool_sv,  s_sv,  T_eff_sv,  E_sv,  u_prev)
            _fill_alpha_features(feat_dcv, t - 1, pool_dcv, s_dcv, T_eff_dcv, E_dcv, u_prev)

            I_gap    = L @ u_prev
            I_stim   = torch.zeros(N, device=device)
            s_ext    = stim[t - 1] if stim is not None else None
            if b_w is not None and s_ext is not None:
                I_stim = b_w.to(device) * s_ext.view(N)

            ar1      = (1.0 - lam_w) * u_prev + lam_w * (I0_w + I_gap + I_stim)
            residual = u[t] - ar1
            target[t - 1] = residual / lam_w.clamp(min=1e-8)

    return {
        "features_sv":  feat_sv.cpu().numpy().astype(np.float64),
        "features_dcv": feat_dcv.cpu().numpy().astype(np.float64),
        "target":       target.cpu().numpy().astype(np.float64),
    }


def _multi_ridge_cv_alpha(
    model:       Stage2ModelPT,
    worm_states: List[WormState],
    cfg:         Stage2PTConfig,
) -> None:
    """Solve per-neuron kernel amplitudes via ridge-CV stacked over all worms.

    Features are computed for each worm using its own λ_u / I0 / G / b, then
    concatenated along the time axis.  Only training time-points are used.
    The fitted amplitudes are injected back into the shared model.  The
    per-worm I0 values are **not** updated (that is handled by opt_psi).
    """
    n_folds       = int(getattr(cfg, "alpha_cv_n_folds",        5)  or 5)
    log_lam_min   = float(getattr(cfg, "alpha_cv_log_min", -2.0))
    log_lam_max   = float(getattr(cfg, "alpha_cv_log_max",  4.0))
    n_grid        = int(getattr(cfg, "alpha_cv_n_grid",          30)  or 30)
    ridge_grid    = _log_ridge_grid(log_lam_min, log_lam_max, n_grid)

    N              = model.N
    r_sv, r_dcv    = model.r_sv, model.r_dcv
    p_total        = r_sv + r_dcv
    n_grid_actual  = len(ridge_grid)

    all_feat_sv, all_feat_dcv, all_target = [], [], []

    with torch.no_grad():
        lam_shared = model.lambda_u.detach()
        I0_shared  = model.I0.detach()
    for worm in worm_states:
        u = worm.assemble_detached().detach()
        feats = _compute_alpha_features_worm(
            model, u, worm.gating, worm.stim,
            lam_shared, I0_shared, worm.G, worm.b,
        )
        # Align with training time: feats[k] corresponds to transition t→t+1,
        # so use worm.train_mask()[1:] (outcome side).
        feat_mask = worm.train_mask().cpu().numpy()[1:]   # (T-1,) bool
        all_feat_sv.append(feats["features_sv"][feat_mask])
        all_feat_dcv.append(feats["features_dcv"][feat_mask])
        all_target.append(feats["target"][feat_mask])

    if not all_feat_sv:
        print("[MultiWorm] Alpha-CV: no valid worm features — skipping.")
        return

    feat_sv_all  = np.concatenate(all_feat_sv,  axis=0)   # (T_tot, N, r_sv)
    feat_dcv_all = np.concatenate(all_feat_dcv, axis=0)
    target_all   = np.concatenate(all_target,   axis=0)   # (T_tot, N)
    T_tot        = feat_sv_all.shape[0]

    with torch.no_grad():
        a_sv_cur  = model.a_sv.detach().cpu().numpy()
        a_dcv_cur = model.a_dcv.detach().cpu().numpy()

    a_sv_out  = (np.tile(a_sv_cur, (N, 1)) if a_sv_cur.ndim == 1
                 else a_sv_cur.copy())
    a_dcv_out = (np.tile(a_dcv_cur, (N, 1)) if a_dcv_cur.ndim == 1
                 else a_dcv_cur.copy())

    n_updated, n_at_upper = 0, 0

    for i in range(N):
        parts = []
        if r_sv  > 0: parts.append(feat_sv_all[:, i, :])
        if r_dcv > 0: parts.append(feat_dcv_all[:, i, :])
        X_i = np.concatenate(parts, axis=1) if parts else np.zeros((T_tot, 0))
        y_i = target_all[:, i]

        valid = np.all(np.isfinite(X_i), axis=1) & np.isfinite(y_i)
        n_v   = int(valid.sum())
        if n_v < max(p_total + 5, 20):
            continue
        if p_total == 0 or np.abs(X_i[valid]).max() < 1e-10:
            continue

        result = _solve_neuron_ridge_cv(X_i[valid], y_i[valid], ridge_grid, n_folds)
        coef   = np.maximum(result["coef"], 0.0)
        if r_sv  > 0: a_sv_out[i]  = coef[:r_sv]
        if r_dcv > 0: a_dcv_out[i] = coef[r_sv:]
        n_updated += 1
        if result["at_upper"]: n_at_upper += 1

    device = next(model.parameters()).device
    result_dict = {
        "alpha_sv":  torch.tensor(a_sv_out,  dtype=torch.float32, device=device),
        "alpha_dcv": torch.tensor(a_dcv_out, dtype=torch.float32, device=device),
        "intercepts": None,   # do not touch shared model I0 in multi-worm mode
    }
    inject_alpha_into_model(model, result_dict)
    print(
        f"[MultiWorm] Alpha-CV (stacked {T_tot} frames, {len(worm_states)} worms): "
        f"{n_updated}/{N} updated, {n_at_upper} at upper λ bound"
    )



def _get_motor_atlas_idx(cfg: Stage2PTConfig, atlas_labels: List[str]) -> List[int]:
    """Map motor neuron names (from cfg) to atlas integer indices."""
    names = getattr(cfg, "motor_neurons", None)
    if not names:
        return []
    idx = []
    for name in names:
        try:
            idx.append(atlas_labels.index(str(name)))
        except ValueError:
            pass   # neuron not in atlas — skip
    return idx


def _init_beh_decoder_multi(
    worm_states:     List[WormState],
    motor_atlas_idx: List[int],
    cfg:             Stage2PTConfig,
    device:          torch.device,
) -> Optional[Dict[str, Any]]:
    """Fit a shared ridge-CV linear behaviour decoder from all Atanas worms.

    The decoder maps lagged atlas motor-neuron activity to eigenworm
    amplitudes.  The weight matrix ``W`` is returned as a learnable
    ``torch.Tensor`` and added to ``opt_psi``.

    Returns ``None`` when no worm with behaviour data is found or when
    ``motor_atlas_idx`` is empty.
    """
    if not motor_atlas_idx:
        return None

    n_lags      = int(getattr(cfg, "behavior_lag_steps",          8) or 8)
    log_lam_min = float(getattr(cfg, "train_behavior_ridge_log_lambda_min", -3.0))
    log_lam_max = float(getattr(cfg, "train_behavior_ridge_log_lambda_max", 10.0))
    n_grid      = int(getattr(cfg, "train_behavior_ridge_n_grid",  50) or 50)
    n_folds     = int(getattr(cfg, "train_behavior_ridge_folds",    5) or 5)
    ridge_grid  = _log_ridge_grid(log_lam_min, log_lam_max, n_grid)

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for worm in worm_states:
        if worm.behaviour is None:
            continue
        # Use the GT observed data (not assembled — u_unobs irrelevant for beh init)
        u_np = worm.u_obs.detach().cpu().numpy()[:, motor_atlas_idx]  # (T, n_motor)
        b_np = worm.behaviour.detach().cpu().numpy()                   # (T, n_modes)
        X    = build_lagged_features_np(u_np, n_lags)                 # (T, n_feat)
        train_t = worm.train_mask().cpu().numpy()
        if n_lags > 0:
            train_t[:n_lags] = False   # cannot form full lag at the start
        ok = (np.all(np.isfinite(X), axis=1)
              & np.all(np.isfinite(b_np), axis=1)
              & train_t)
        if ok.sum() < 10:
            continue
        X_parts.append(X[ok])
        y_parts.append(b_np[ok])

    if not X_parts:
        return None

    X_all    = np.concatenate(X_parts, axis=0)  # (T_total, n_feat)
    y_all    = np.concatenate(y_parts, axis=0)  # (T_total, n_modes)
    n_feat   = X_all.shape[1]
    n_modes  = y_all.shape[1]
    eval_idx = np.arange(len(X_all))

    W_np = np.zeros((n_feat + 1, n_modes), dtype=np.float64)
    for j in range(n_modes):
        r = _ridge_cv_single_target(X_all, y_all[:, j], eval_idx, ridge_grid, n_folds)
        if np.all(np.isfinite(r["coef"])):
            W_np[:n_feat, j] = r["coef"]
            W_np[n_feat,  j] = r["intercept"]

    n_atanas = sum(1 for ws in worm_states if ws.behaviour is not None)

    decoder_type = str(getattr(cfg, "behavior_decoder_type", "mlp") or "mlp").strip().lower()

    if decoder_type == "mlp":
        hidden  = int(getattr(cfg, "behavior_decoder_hidden", 32) or 32)
        dropout = float(getattr(cfg, "behavior_decoder_dropout", 0.1) or 0.0)
        model = MLPBehaviourDecoder(n_feat + 1, n_modes, hidden, dropout).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[MultiWorm] Beh MLP decoder: {n_feat + 1}\u2192{hidden}\u2192{n_modes}  "
            f"({n_params:,} params, stacked {len(X_all)} frames from {n_atanas} Atanas worm(s))"
        )
        return {
            "type":            "mlp",
            "model":           model,
            "motor_atlas_idx": motor_atlas_idx,
            "n_lags":          n_lags,
            "n_modes":         n_modes,
        }

    # Linear: warm-start from ridge-CV
    print(
        f"[MultiWorm] Beh decoder: {n_feat} features \u2192 {n_modes} modes  "
        f"(stacked {len(X_all)} frames from {n_atanas} Atanas worm(s))"
    )
    W = nn.Parameter(
        torch.tensor(W_np, dtype=torch.float32, device=device)
    )
    return {
        "type":            "linear",
        "W":               W,
        "motor_atlas_idx": motor_atlas_idx,
        "n_lags":          n_lags,
        "n_modes":         n_modes,
    }


def _beh_loss_worm(
    prior_mu:   torch.Tensor,   # (T, N_atlas)
    worm:       WormState,
    decoder:    Dict[str, Any],
    train_mask: torch.Tensor,   # (T,) bool
) -> torch.Tensor:
    """Behaviour MSE loss for one Atanas worm."""
    if worm.behaviour is None:
        return torch.tensor(0.0, device=prior_mu.device)

    motor_idx = decoder["motor_atlas_idx"]
    n_lags    = decoder["n_lags"]

    u_motor = prior_mu[:, motor_idx]                                # (T, n_motor)
    X       = build_lagged_features_torch(u_motor, n_lags)         # (T, n_feat)
    X_aug   = torch.cat(
        [X, torch.ones(X.shape[0], 1, device=X.device)], dim=1
    )                                                               # (T, n_feat+1)

    if decoder.get("type") == "mlp":
        b_pred = decoder["model"](X_aug)                            # (T, n_modes)
    else:
        W     = decoder["W"]                                        # (n_feat+1, n_modes)
        b_pred = X_aug @ W                                          # (T, n_modes)

    b_true  = worm.behaviour.to(prior_mu.device)                   # (T, n_modes)

    b_pred_t = b_pred[train_mask]
    b_true_t = b_true[train_mask]
    ok = torch.isfinite(b_pred_t).all(dim=1) & torch.isfinite(b_true_t).all(dim=1)
    if ok.sum() < 3:
        return torch.tensor(0.0, device=prior_mu.device)
    return F.mse_loss(b_pred_t[ok], b_true_t[ok])



def _rollout_loss_worm(
    model:          Stage2ModelPT,
    worm:           WormState,
    u_full:         torch.Tensor,   # (T, N) detached trajectory
    rollout_steps:  int,
    rollout_starts: int,
) -> torch.Tensor:
    """K-step free-running rollout loss (shared lambda_u/I0, per-worm G/b)."""
    T, N   = u_full.shape
    device = u_full.device
    sigma2 = worm.sigma_u.pow(2)

    if rollout_steps <= 0 or T <= rollout_steps + 1:
        return torch.tensor(0.0, device=device)

    max_start = T - rollout_steps - 1
    n_starts  = min(rollout_starts, max_start + 1)
    t0s       = torch.randperm(max_start + 1, device=device)[:n_starts]

    G   = worm.G
    b   = worm.b

    total = torch.tensor(0.0, device=device)
    n_ok  = 0

    for idx in range(n_starts):
        t0    = int(t0s[idx].item())
        u_cur = u_full[t0].detach()
        s_sv  = torch.zeros(N, model.r_sv,  device=device, dtype=u_full.dtype)
        s_dcv = torch.zeros(N, model.r_dcv, device=device, dtype=u_full.dtype)

        seg_loss  = torch.tensor(0.0, device=device)
        seg_count = 0

        for k in range(rollout_steps):
            t = t0 + k
            g = worm.gating[t]
            s = worm.stim[t] if worm.stim is not None else None
            u_next, s_sv, s_dcv = model.prior_step(
                u_cur, s_sv, s_dcv, g, s,
                G=G, b=b,
            )
            target_next = u_full[t + 1].detach()
            resid2      = (u_next - target_next) ** 2
            ok          = torch.isfinite(resid2)
            if ok.any():
                seg_loss  = seg_loss + (resid2[ok] / sigma2[ok].clamp(min=1e-8)).mean()
                seg_count += 1
            u_cur = u_next   # free-running

        if seg_count > 0:
            total = total + seg_loss / seg_count
            n_ok += 1

    return total / max(n_ok, 1)



def _log_multi_config(cfg: Stage2PTConfig, N_atlas: int, n_worms: int) -> None:
    mc = cfg.multi
    print(
        f"[MultiWorm] N_atlas={N_atlas}  n_worms={n_worms}\n"
        f"[MultiWorm] dt={mc.common_dt:.3f}  per_worm_G={mc.per_worm_G}"
        f"  G_cons={mc.G_consistency_weight:.4g}\n"
        f"[MultiWorm] infer_unobs={mc.infer_unobserved}"
        f"  inner_steps={mc.u_unobs_inner_steps}"
        f"  u_lr={mc.u_unobs_lr:.4g}"
        f"  smoothness={mc.u_unobs_smoothness:.4g}\n"
        f"[MultiWorm] sigma_unobs_scale={mc.sigma_u_unobs_scale}"
        f"  val_frac={mc.val_frac}"
        f"  weight_mode={mc.worm_weight_mode}"
    )


def _log_epoch(
    epoch:       int,
    step2_loss:  float,
    step1_loss:  Optional[float],
    val_losses:  Dict[str, float],
    tr_losses:   Dict[str, float],
    model:       Stage2ModelPT,
    print_every: int,
) -> None:
    if epoch % print_every != 0:
        return
    snap = _snapshot_params(model)
    val_mean = float(np.nanmean(list(val_losses.values()))) if val_losses else float("nan")
    tr_mean  = float(np.nanmean(list(tr_losses.values())))  if tr_losses  else float("nan")
    u1 = f"  step1={step1_loss:.5f}" if step1_loss is not None else ""
    print(
        f"[MultiWorm] ep {epoch:04d}"
        f"  step2={step2_loss:.5f}{u1}"
        f"  val={val_mean:.5f}  tr={tr_mean:.5f}"
        f"  G={snap.get('G', float('nan')):.4g}"
        f"  lambda_u={snap.get('lambda_u_mean', float('nan')):.4g}"
    )



@torch.no_grad()
def _val_loss_worm(
    model: Stage2ModelPT, worm: WormState
) -> float:
    """One-step MSE on validation time-points (detached, no grad)."""
    u_full   = worm.assemble_detached()
    prior_mu = _forward_pass_worm(model, worm, u_full)
    val_mask = worm.val_mask
    if not val_mask.any():
        return float("nan")
    loss = compute_dynamics_loss(u_full[val_mask], prior_mu[val_mask], worm.sigma_u)
    return float(loss.item())



def train_multi_worm(
    cfg:      Stage2PTConfig,
    save_dir: Optional[str] = None,
    show:     bool = False,
) -> Optional[Dict[str, Any]]:
    """Two-level MAP joint training across multiple worms.

    Parameters
    ----------
    cfg : Stage2PTConfig
        Must have ``cfg.multi.multi_worm = True`` and ``cfg.multi.h5_paths``
        populated.
    save_dir : str, optional
        Directory to write checkpoints and logs.  Created if absent.
    show : bool
        If ``True``, matplotlib figures are shown (currently unused).

    Returns
    -------
    dict  (or None on error)
        ``model_state`` : snapshot of shared model parameters
        ``worm_results`` : per-worm dict with final val loss
    """
    mc = cfg.multi
    if not mc.multi_worm:
        raise ValueError("cfg.multi.multi_worm must be True for train_multi_worm.")

    tee: Optional[_TeeWriter] = None
    if save_dir is not None:
        sd = Path(save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        tee = _TeeWriter(sd / "train_multi.log")
        sys.stdout = tee

    try:
        return _train_multi_worm_inner(cfg, save_dir)
    finally:
        if tee is not None:
            tee.close()
            sys.stdout = tee._stdout


def _train_multi_worm_inner(
    cfg:      Stage2PTConfig,
    save_dir: Optional[str],
) -> Optional[Dict[str, Any]]:
    mc     = cfg.multi
    device = torch.device(getattr(cfg, "device", "cpu") or "cpu")

    # 1. Load multi-worm data
    print("[MultiWorm] Loading data …")
    data       = load_multi_worm_data(cfg)
    N_atlas    = data["atlas_size"]
    atlas_lbl  = data["atlas_labels"]    # List[str] length N_atlas (subsetted)
    worm_dicts = data["worms"]           # list of per-worm dicts
    n_worms    = len(worm_dicts)

    print(f"[MultiWorm] Atlas size: {N_atlas}  worms loaded: {n_worms}")
    for wd in worm_dicts:
        print(
            f"  {wd['worm_id']}  dataset={wd['dataset_type']}"
            f"  T={wd['T']}  N_obs={wd['N_obs']}  N_unobs={wd['N_unobs']}"
            f"  weight={wd['weight']:.3f}"
        )

    # 2. Build shared Stage2 model
    # Median lambda_u_init across all worms
    all_lu = [torch.as_tensor(wd["lambda_u_init"]) for wd in worm_dicts]
    lu_stack  = torch.stack([lu.mean() for lu in all_lu])
    lu_median = float(lu_stack.median().item())
    lu_init   = torch.stack(all_lu).median(dim=0).values   # (N_atlas,)

    common_dt = mc.common_dt
    T_e   = data["T_e"]
    T_sv  = data["T_sv"]
    T_dcv = data["T_dcv"]
    sign_t = data["sign_t"].to(device) if data["sign_t"] is not None else None

    model = Stage2ModelPT(
        N_atlas, T_e, T_sv, T_dcv,
        common_dt, cfg, device,
        d_ell=0,              # per-worm stim is handled through WormState.b overrides
        lambda_u_init=lu_init.to(device),
        sign_t=sign_t,
    )

    print(
        f"[MultiWorm] Shared model: N={N_atlas}  "
        f"r_sv={model.r_sv}  r_dcv={model.r_dcv}  dt={common_dt}"
    )

    # 3. init_all_from_data — use best representative worm
    # Pick the Atanas worm with the most observed neurons and stage-1 data.
    rep_idx = max(
        range(n_worms),
        key=lambda i: worm_dicts[i]["N_obs"]
                      + (0 if not worm_dicts[i]["has_stage1"] else 1000),
    )
    rep_u = torch.as_tensor(
        worm_dicts[rep_idx]["u"], dtype=torch.float32
    ).to(device)                    # (T, N_atlas) — NaN where unobserved
    # Replace NaN with 0 for init purposes
    rep_u = torch.nan_to_num(rep_u, nan=0.0)
    init_all_from_data(model, rep_u, cfg)

    # 4. Build WormState objects
    worm_states = build_worm_states(data, cfg)
    _log_multi_config(cfg, N_atlas, n_worms)

    # 5. Behaviour decoder
    motor_atlas_idx = _get_motor_atlas_idx(cfg, atlas_lbl)
    beh_decoder: Optional[Dict[str, Any]] = None
    behavior_weight = float(getattr(cfg, "behavior_weight", 0.0) or 0.0)

    if motor_atlas_idx and behavior_weight > 0.0:
        beh_decoder = _init_beh_decoder_multi(
            worm_states, motor_atlas_idx, cfg, device
        )
        if beh_decoder is None:
            print("[MultiWorm] No Atanas worm with behaviour — decoder disabled.")
    else:
        if not motor_atlas_idx:
            print("[MultiWorm] motor_neurons not configured — behaviour decoder disabled.")
        else:
            print("[MultiWorm] behavior_weight=0 — behaviour decoder disabled.")

    # 6. Build optimisers
    lr        = float(getattr(cfg, "learning_rate",   1e-3))
    u_lr      = float(mc.u_unobs_lr)
    grad_clip = float(getattr(cfg, "grad_clip_norm",  1.0) or 0.0)

    # opt_theta: shared model (frozen params have requires_grad=False → ignored by Adam)
    theta_params = [p for p in model.parameters() if p.requires_grad]

    # opt_psi: per-worm {lambda_u, I0, G, b}
    psi_params: List[nn.Parameter] = []
    for ws in worm_states:
        psi, _ = ws.param_groups()
        psi_params.extend(psi)

    # opt_u: per-worm {u_unobs}
    u_params: List[nn.Parameter] = []
    for ws in worm_states:
        _, u_list = ws.param_groups()
        u_params.extend(u_list)

    # Add behaviour decoder weights to opt_psi
    if beh_decoder is not None:
        if beh_decoder.get("type") == "mlp":
            psi_params.extend(list(beh_decoder["model"].parameters()))
        else:
            psi_params.append(beh_decoder["W"])

    opt_theta = optim.Adam(theta_params, lr=lr)
    opt_psi   = optim.Adam(psi_params,   lr=lr)    if psi_params   else None
    opt_u     = optim.Adam(u_params,     lr=u_lr) if u_params     else None

    print(
        f"[MultiWorm] Optimiser groups:"
        f"  opt_theta={len(theta_params)} params"
        f"  opt_psi={len(psi_params)} params"
        f"  opt_u={len(u_params)} params"
    )

    # 7. Initial alpha ridge-CV
    alpha_cv_every = int(getattr(cfg, "alpha_cv_every", 5) or 0)
    if alpha_cv_every > 0:
        print("[MultiWorm] Running initial alpha-CV …")
        _multi_ridge_cv_alpha(model, worm_states, cfg)

    # 8. Training config
    num_epochs         = int(getattr(cfg, "num_epochs",          40))
    inner_steps        = int(mc.u_unobs_inner_steps)
    smoothness_w       = float(mc.u_unobs_smoothness)
    infer_unobs        = bool(mc.infer_unobserved) and (opt_u is not None)
    interaction_l2     = float(getattr(cfg, "interaction_l2",    0.0) or 0.0)
    rollout_weight     = float(getattr(cfg, "rollout_weight",     0.0) or 0.0)
    rollout_steps      = int(getattr(cfg, "rollout_steps",        0)   or 0)
    rollout_starts     = int(getattr(cfg, "rollout_starts",       4)   or 4)
    G_cons_w           = float(mc.G_consistency_weight)
    per_worm_G         = bool(mc.per_worm_G)
    print_every        = int(getattr(cfg, "print_every",          1)   or 1)

    if interaction_l2 > 0:
        print(f"[MultiWorm] Interaction L2 = {interaction_l2:.4g}")
    if rollout_weight > 0:
        print(f"[MultiWorm] Rollout: K={rollout_steps} starts={rollout_starts} w={rollout_weight:.4g}")
    if infer_unobs:
        print(f"[MultiWorm] Trajectory inference: {inner_steps} inner steps / epoch")

    # 9. Training loop
    best_val   = float("inf")
    best_state = None

    for epoch in range(num_epochs):

        if alpha_cv_every > 0 and epoch > 0 and epoch % alpha_cv_every == 0:
            _multi_ridge_cv_alpha(model, worm_states, cfg)

        # Step 1 — Trajectory inference (fix θ, ψ; update u_U)
        step1_loss_val: Optional[float] = None
        if infer_unobs:
            last_u_loss = torch.tensor(0.0, device=device)

            for _inner in range(inner_steps):
                opt_u.zero_grad()
                u_loss = torch.tensor(0.0, device=device)

                for worm in worm_states:
                    u_full  = worm.assemble()       # differentiable through u_unobs
                    pm      = _forward_pass_worm(model, worm, u_full)
                    tm      = worm.train_mask()     # (T,) bool
                    dyn_w   = compute_dynamics_loss(u_full[tm], pm[tm], worm.sigma_u)
                    smooth_w = worm.smoothness_loss(smoothness_w)
                    u_loss   = u_loss + worm.weight * (dyn_w + smooth_w)

                u_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(u_params, grad_clip)
                opt_u.step()
                last_u_loss = u_loss

            step1_loss_val = float(last_u_loss.item())

        # Step 2 — Model learning (fix u_U; update θ, ψ)
        opt_theta.zero_grad()
        if opt_psi is not None:
            opt_psi.zero_grad()

        total_loss  = torch.tensor(0.0, device=device)
        tr_losses: Dict[str, float] = {}

        for worm in worm_states:
            u_full = worm.assemble_detached()           # u_unobs treated as fixed data
            pm     = _forward_pass_worm(model, worm, u_full)
            tm     = worm.train_mask()
            w      = worm.weight

            worm_loss = compute_dynamics_loss(u_full[tm], pm[tm], worm.sigma_u)
            tr_losses[worm.worm_id] = float(worm_loss.item())

            # Interaction L2: penalise network contribution beyond pure AR(1)
            if interaction_l2 > 0:
                with torch.no_grad():
                    lam_d = model.lambda_u.detach()
                    I0_d  = model.I0.detach()
                    ar1_mu = (1.0 - lam_d) * u_full[:-1] + lam_d * I0_d
                interaction  = pm[1:] - ar1_mu
                worm_loss    = worm_loss + interaction_l2 * interaction.pow(2).mean()

            # Rollout loss
            if rollout_weight > 0 and rollout_steps > 0:
                rl_w       = _rollout_loss_worm(model, worm, u_full, rollout_steps, rollout_starts)
                worm_loss  = worm_loss + rollout_weight * rl_w

            # Behaviour loss (Atanas worms only)
            if beh_decoder is not None and worm.behaviour is not None and behavior_weight > 0:
                beh_w     = _beh_loss_worm(pm, worm, beh_decoder, tm)
                worm_loss = worm_loss + behavior_weight * beh_w

            total_loss = total_loss + w * worm_loss

        # G consistency penalty (cross-worm)
        if per_worm_G and G_cons_w > 0:
            G_vals = [ws.G for ws in worm_states if ws.G is not None]
            if len(G_vals) > 1:
                G_stack = torch.stack(G_vals)
                G_mean  = G_stack.mean().detach()   # fixed target for each worm
                g_cons  = ((G_stack - G_mean) ** 2).mean()
                total_loss = total_loss + G_cons_w * g_cons

        total_loss.backward()

        # Gradient clipping over all learnable params in Step 2
        all_step2_params = theta_params + psi_params
        if grad_clip > 0 and all_step2_params:
            torch.nn.utils.clip_grad_norm_(all_step2_params, grad_clip)

        opt_theta.step()
        if opt_psi is not None:
            opt_psi.step()

        val_losses = {ws.worm_id: _val_loss_worm(model, ws) for ws in worm_states}
        _log_epoch(
            epoch, float(total_loss.item()), step1_loss_val,
            val_losses, tr_losses, model, print_every
        )

        val_mean = float(np.nanmean(list(val_losses.values())))
        if val_mean < best_val:
            best_val   = val_mean
            best_state = snapshot_model_state(model)

    # 10. Final alpha-CV
    if alpha_cv_every > 0:
        _multi_ridge_cv_alpha(model, worm_states, cfg)

    # 11. Save results
    worm_results: Dict[str, Any] = {}
    for ws in worm_states:
        worm_results[ws.worm_id] = {
            "val_loss":   _val_loss_worm(model, ws),
            "N_obs":      int((~torch.isnan(ws.u_obs[0])).sum().item()),
            "N_unobs":    int(ws.u_unobs.shape[1]) if ws.u_unobs is not None else 0,
            "T":          int(ws.u_obs.shape[0]),
            "weight":     float(ws.weight),
            "dataset":    ws.worm_id.split("_")[0] if "_" in ws.worm_id else "unknown",
        }

    final_model_state = snapshot_model_state(model)

    results = {
        "model_state":  final_model_state,
        "best_state":   best_state,
        "worm_results": worm_results,
        "best_val":     best_val,
    }

    if save_dir is not None:
        sd = Path(save_dir)
        torch.save(final_model_state, sd / "model_final.pt")
        if best_state is not None:
            torch.save(best_state, sd / "model_best.pt")
        with open(sd / "worm_results.json", "w") as fh:
            json.dump(
                {k: {kk: (float(vv) if isinstance(vv, (float, int)) else vv)
                     for kk, vv in v.items()}
                 for k, v in worm_results.items()},
                fh, indent=2
            )
        # Save per-worm parameters (lambda_u, I0, optionally G and b)
        worm_states_state = {
            ws.worm_id: {name: p.detach().cpu()
                         for name, p in ws.named_parameters()}
            for ws in worm_states
        }
        torch.save(worm_states_state, sd / "worm_states_final.pt")
        print(f"[MultiWorm] Results saved to {sd}")

    # 12. Per-worm evaluation (one-step + LOO on val time-points)
    try:
        from .evaluate_multi import run_multi_worm_evaluation
        print("\n[MultiWorm] Running per-worm evaluation …")
        eval_results = run_multi_worm_evaluation(
            model, worm_states, cfg,
            atlas_labels=atlas_lbl,
            loo_subset_size=getattr(mc, "loo_subset_size", 20),
        )
        results["eval"] = eval_results
    except Exception as exc:
        print(f"[MultiWorm] Evaluation failed (non-fatal): {exc}")
        eval_results = {}

    # 14. Per-worm single-worm diagnostic plots
    if save_dir is not None:
        try:
            from .plot_eval import generate_eval_loo_plots
            print("\n[MultiWorm] Generating per-worm single-worm diagnostic plots …")
            # Resolve cfg.motor_neurons from string names → atlas integer indices
            motor_neurons_raw = getattr(cfg, "motor_neurons", None)
            if motor_neurons_raw is not None:
                lbl_lower = [str(l).strip().lower() for l in atlas_lbl]
                resolved: list[int] = []
                for mn in motor_neurons_raw:
                    try:
                        idx = int(mn)
                        if 0 <= idx < len(atlas_lbl):
                            resolved.append(idx)
                    except (ValueError, TypeError):
                        key = str(mn).strip().lower()
                        if key in lbl_lower:
                            resolved.append(lbl_lower.index(key))
                cfg.motor_neurons = tuple(resolved) if resolved else None
            # Save original shared model parameters so we can restore them
            orig_lambda_u_raw = model._lambda_u_raw.data.clone()
            orig_I0 = model.I0.data.clone()
            for ws in worm_states:
                worm_save = Path(save_dir) / ws.worm_id
                worm_save.mkdir(parents=True, exist_ok=True)
                try:
                    # Temporarily patch shared model with this worm's per-worm params
                    with torch.no_grad():
                        model._lambda_u_raw.data.copy_(ws._lambda_u_raw.data)
                        model.I0.data.copy_(ws.I0.data)

                    # Build single-worm data dict compatible with run_full_evaluation
                    u_full = ws.assemble_detached().detach()
                    beh_t = ws.behaviour  # (T, n_modes) or None
                    worm_data = {
                        "u_stage1":     u_full,
                        "sigma_u":      ws.sigma_u,
                        "gating":       ws.gating,
                        "stim":         ws.stim,
                        "dt":           getattr(mc, "common_dt", 0.6),
                        "neuron_labels": atlas_lbl,
                        # Behaviour fields for Atanas worms
                        "b":      torch.nan_to_num(beh_t, nan=0.0) if beh_t is not None else None,
                        "b_mask": torch.isfinite(beh_t).float() if beh_t is not None else None,
                    }
                    print(f"[MultiWorm]   Plotting {ws.worm_id} → {worm_save}")
                    generate_eval_loo_plots(
                        model, worm_data, cfg, [],
                        str(worm_save), show=False,
                    )
                except Exception as exc:
                    print(f"[MultiWorm]   Single-worm plots failed for {ws.worm_id}: {exc}")
                finally:
                    # Always restore original shared model params
                    with torch.no_grad():
                        model._lambda_u_raw.data.copy_(orig_lambda_u_raw)
                        model.I0.data.copy_(orig_I0)
        except Exception as exc:
            print(f"[MultiWorm] Per-worm plot generation failed (non-fatal): {exc}")

    # 15. Cross-worm R² summary plot (one-step / LOO / free-run)
    if save_dir is not None and eval_results:
        try:
            from .plot_eval import plot_r2_three_metrics
            plot_r2_three_metrics(eval_results, save_dir=save_dir, show=False)
            print("[MultiWorm] 00_r2_summary.png  ✓")
        except Exception as exc:
            print(f"[MultiWorm] R² summary plot failed (non-fatal): {exc}")

    print(
        f"\n[MultiWorm] Done.  Best val loss = {best_val:.5f}  "
        f"({len(worm_states)} worms, {num_epochs} epochs)"
    )
    return results


# ── CLI entry point ──────────────────────────────────────────────────────

def _cli(argv=None):
    """Command-line interface for multi-worm training.

    Usage::

        .venv/bin/python -u -m stage2.train_multi \\
            --h5 data/used/.../worm1.h5 data/used/.../worm2.h5 \\
            --save_dir output_plots/stage2/multi-pair1

        # Full atlas (do not restrict to intersection neurons):
        .venv/bin/python -u -m stage2.train_multi \\
            --h5 file1.h5 file2.h5 file3.h5 \\
            --save_dir output_plots/stage2/trio \\
            --atlas_min_worm_count 0 --num_epochs 200
    """
    import argparse
    from .config import make_config

    parser = argparse.ArgumentParser(
        description="Multi-worm Stage 2 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--h5", dest="h5_paths", nargs="+", required=True,
        metavar="PATH", help="One or more HDF5 worm files",
    )
    parser.add_argument("--save_dir", default=None,
                        help="Output directory for checkpoints and plots")
    parser.add_argument("--atlas_min_worm_count", type=int, default=2,
                        help="Min worms a neuron must appear in (0 = full 302)")
    parser.add_argument("--num_epochs",    type=int,   default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--show",          action="store_true",
                        help="Show plots interactively")

    args = parser.parse_args(argv)
    h5_paths = tuple(args.h5_paths)

    cfg = make_config(
        h5_path=h5_paths[0],
        multi_worm=True,
        h5_paths=h5_paths,
        atlas_min_worm_count=args.atlas_min_worm_count,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    train_multi_worm(cfg, save_dir=args.save_dir, show=args.show)


if __name__ == "__main__":
    _cli()
