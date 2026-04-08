from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ._utils import _get_clip_bounds, _clamp, _per_neuron_metrics
from .model import Stage2ModelPT
from .worm_state import WormState

__all__ = [
    "compute_onestep_worm",
    "run_loo_worm",
    "run_multi_worm_evaluation",
]



def _teacher_forced_prior_worm(
    model: Stage2ModelPT,
    ws:    WormState,
    u:     torch.Tensor,   # (T, N_atlas) — full assembled trajectory
) -> torch.Tensor:         # (T, N_atlas)
    """Teacher-forced prior with per-worm λ_u / I0 / G / b overrides.

    ``prior_mu[0] = u[0]`` (trivial, no prediction).
    ``prior_mu[t] = f(u[t-1])``  for  t ≥ 1.
    """
    T, N   = u.shape
    device = u.device
    prior_mu = torch.zeros_like(u)
    prior_mu[0] = u[0]
    s_sv  = torch.zeros(N, model.r_sv,  device=device, dtype=u.dtype)
    s_dcv = torch.zeros(N, model.r_dcv, device=device, dtype=u.dtype)

    with torch.no_grad():
        lam = model.lambda_u.detach()    # shared on model, not per-worm
        I0  = model.I0.detach()          # shared on model, not per-worm
        G   = ws.G.detach() if ws.G is not None else None
        b   = ws.b.detach() if ws.b is not None else None
        model.reset_lag_history()
        for t in range(1, T):
            g = ws.gating[t - 1]
            s = ws.stim[t - 1] if ws.stim is not None else None
            prior_mu[t], s_sv, s_dcv = model.prior_step(
                u[t - 1], s_sv, s_dcv, g, s,
                lambda_u=lam, I0=I0, G=G, b=b,
            )
    return prior_mu



def _loo_simulate_worm(
    model:     Stage2ModelPT,
    ws:        WormState,
    u_all:     torch.Tensor,   # (T, N_atlas)
    held_out:  int,            # atlas index to hold out
) -> np.ndarray:               # (T,) recursive prediction for neuron held_out
    T, N   = u_all.shape
    device = u_all.device
    lo, hi = _get_clip_bounds(model)

    lam = model.lambda_u   # shared on model, not per-worm
    I0  = model.I0          # shared on model, not per-worm
    G   = ws.G
    b   = ws.b

    u_pred = torch.zeros(T, device=device, dtype=u_all.dtype)
    u_pred[0] = u_all[0, held_out]
    s_sv  = torch.zeros(N, model.r_sv,  device=device, dtype=u_all.dtype)
    s_dcv = torch.zeros(N, model.r_dcv, device=device, dtype=u_all.dtype)

    with torch.no_grad():
        model.reset_lag_history()
        for t in range(T - 1):
            u_t      = u_all[t].clone()
            u_t[held_out] = u_pred[t]               # replace with recursive pred
            g = ws.gating[t]
            s = ws.stim[t] if ws.stim is not None else None
            mu_next, s_sv, s_dcv = model.prior_step(
                u_t, s_sv, s_dcv, g, s,
                lambda_u=lam, I0=I0, G=G, b=b,
            )
            u_pred[t + 1] = _clamp(mu_next[held_out:held_out + 1], lo, hi).squeeze()

    return u_pred.cpu().numpy()   # (T,)



def compute_onestep_worm(
    model:   Stage2ModelPT,
    ws:      WormState,
    use_val: bool = True,
) -> Dict[str, Any]:
    """One-step teacher-forced evaluation for one worm.

    Parameters
    ----------
    model : Stage2ModelPT   — shared model
    ws    : WormState        — per-worm state (provides overrides + val_mask)
    use_val : bool           — if True, evaluate only on held-out val time-steps;
                               if False, evaluate on all time steps.

    Returns
    -------
    dict with:
        ``prior_mu``   (T, N_atlas) numpy array
        ``r2``         (N_atlas,)  — NaN at unobserved atlas positions
        ``corr``       (N_atlas,)
        ``rmse``       (N_atlas,)
        ``obs_idx``    (n_obs,)    — indices of observed neurons
        ``val_t``      (T_val,)    — evaluation time indices used
    """
    device  = next(model.parameters()).device
    u       = ws.assemble(detach=True).to(device)  # (T, N_atlas)
    prior_mu = _teacher_forced_prior_worm(model, ws, u)

    u_np  = u.cpu().numpy()
    mu_np = prior_mu.cpu().numpy()
    T     = u_np.shape[0]

    # Choose evaluation time indices:
    # prior_mu[t] is the prediction for time t (using u[t-1] as input).
    # t=0 is trivial (prior_mu[0] = u[0]), so we always skip it.
    if use_val:
        val_t = np.where(ws.val_mask.cpu().numpy())[0]
    else:
        val_t = np.arange(1, T)
    val_t = val_t[val_t > 0]   # guarantee t > 0

    obs  = ws.obs_idx.cpu().numpy().tolist()
    r2, corr, rmse = _per_neuron_metrics(u_np[val_t], mu_np[val_t], obs)

    return {
        "prior_mu": mu_np,
        "r2":       r2,
        "corr":     corr,
        "rmse":     rmse,
        "obs_idx":  np.array(obs, dtype=int),
        "val_t":    val_t,
    }



def run_loo_worm(
    model:   Stage2ModelPT,
    ws:      WormState,
    subset:  Optional[List[int]] = None,
    use_val: bool = True,
) -> Dict[str, Any]:
    """Leave-one-neuron-out evaluation for one worm (observed neurons only).

    Parameters
    ----------
    subset : list of atlas indices to evaluate (must be a subset of ws.obs_idx).
             None → use all observed neurons.

    Returns
    -------
    dict with:
        ``pred``     dict  neuron_idx → (T,) predicted trace
        ``r2``       (N_atlas,)  — NaN at positions not in subset
        ``corr``     (N_atlas,)
        ``rmse``     (N_atlas,)
        ``obs_idx``  (n_obs,)
        ``val_t``    (T_val,)
    """
    device  = next(model.parameters()).device
    u       = ws.assemble(detach=True).to(device)
    u_np    = u.cpu().numpy()
    N_atlas = u.shape[1]
    T       = u.shape[0]

    obs_set = set(ws.obs_idx.cpu().numpy().tolist())
    obs     = sorted(obs_set)
    indices = [i for i in (subset if subset is not None else obs) if i in obs_set]

    if use_val:
        val_t = np.where(ws.val_mask.cpu().numpy())[0]
    else:
        val_t = np.arange(1, T)
    val_t = val_t[val_t > 0]

    preds: Dict[int, np.ndarray] = {}
    for i in indices:
        preds[i] = _loo_simulate_worm(model, ws, u, i)

    # Build a full (T, N_atlas) pred array for metric computation
    # Non-evaluated positions get the ground-truth to yield NaN R²
    pred_full = u_np.copy()
    for i, pred_i in preds.items():
        pred_full[:, i] = pred_i

    r2, corr, rmse = _per_neuron_metrics(
        u_np[val_t], pred_full[val_t], indices
    )

    return {
        "pred":    preds,
        "r2":      r2,
        "corr":    corr,
        "rmse":    rmse,
        "obs_idx": np.array(obs, dtype=int),
        "val_t":   val_t,
    }



def compute_freerun_worm(
    model:   Stage2ModelPT,
    ws:      WormState,
    use_val: bool = True,
) -> Dict[str, Any]:
    """Autonomous free-run evaluation for one worm with per-worm overrides.

    Returns
    -------
    dict with:
        ``u_free``  (T, N_atlas) numpy array
        ``r2``      (N_atlas,)  — NaN at unobserved positions
        ``corr``    (N_atlas,)
        ``rmse``    (N_atlas,)
        ``obs_idx`` (n_obs,)
        ``val_t``   (T_val,)
    """
    device = next(model.parameters()).device
    u      = ws.assemble(detach=True).to(device)   # (T, N_atlas)
    T, N   = u.shape
    lo, hi = _get_clip_bounds(model)

    lam = model.lambda_u.detach()    # shared on model, not per-worm
    I0  = model.I0.detach()          # shared on model, not per-worm
    G   = ws.G.detach() if ws.G is not None else None
    b   = ws.b.detach() if ws.b is not None else None

    u_free = torch.zeros_like(u)
    u_free[0] = u[0]
    s_sv  = torch.zeros(N, model.r_sv,  device=device, dtype=u.dtype)
    s_dcv = torch.zeros(N, model.r_dcv, device=device, dtype=u.dtype)

    with torch.no_grad():
        model.reset_lag_history()
        for t in range(1, T):
            g = ws.gating[t - 1]
            s = ws.stim[t - 1] if ws.stim is not None else None
            u_next, s_sv, s_dcv = model.prior_step(
                u_free[t - 1], s_sv, s_dcv, g, s,
                lambda_u=lam, I0=I0, G=G, b=b,
            )
            u_free[t] = _clamp(u_next, lo, hi)

    u_np    = u.cpu().numpy()
    uf_np   = u_free.cpu().numpy()

    if use_val:
        val_t = np.where(ws.val_mask.cpu().numpy())[0]
    else:
        val_t = np.arange(1, T)
    val_t = val_t[val_t > 0]

    obs   = ws.obs_idx.cpu().numpy().tolist()
    r2, corr, rmse = _per_neuron_metrics(u_np[val_t], uf_np[val_t], obs)

    return {
        "u_free":  uf_np,
        "r2":      r2,
        "corr":    corr,
        "rmse":    rmse,
        "obs_idx": np.array(obs, dtype=int),
        "val_t":   val_t,
    }



def run_multi_worm_evaluation(
    model:            Stage2ModelPT,
    worm_states:      List[WormState],
    cfg               = None,
    atlas_labels:     Optional[List[str]] = None,
    loo_subset_size:  int = 20,
) -> Dict[str, Any]:
    """Run per-worm one-step + LOO evaluation across all worms.

    Parameters
    ----------
    loo_subset_size : int
        Maximum number of neurons to run LOO on per worm (top-k by variance).
        0 = all observed neurons.

    Returns
    -------
    dict with:
        ``per_worm``   dict worm_id → {onestep, loo, r2_*_median, ...}
        ``summary``    aggregate cross-worm statistics
        ``atlas_labels`` list[str] | None
    """
    from . import get_stage2_logger
    _logger = get_stage2_logger()

    N_atlas = worm_states[0].N if worm_states else 0
    per_worm: Dict[str, Any] = {}

    for ws in worm_states:
        _logger.info("eval_worm_start", worm_id=ws.worm_id, N_obs=ws.N_obs)

        onestep = compute_onestep_worm(model, ws, use_val=True)

        obs = list(ws.obs_idx.cpu().numpy())
        if loo_subset_size > 0 and len(obs) > loo_subset_size:
            u_full = ws.assemble(detach=True).cpu().numpy()
            var_obs = np.nanvar(u_full[:, obs], axis=0)
            top_k   = np.argsort(var_obs)[::-1][:loo_subset_size]
            loo_subset = [obs[k] for k in top_k]
        else:
            loo_subset = obs

        loo = run_loo_worm(model, ws, subset=loo_subset, use_val=True)

        freerun = compute_freerun_worm(model, ws, use_val=True)

        obs_arr              = np.array(obs, dtype=int)
        r2_obs_onestep       = onestep["r2"][obs_arr]
        r2_obs_loo           = loo["r2"][obs_arr]
        r2_obs_freerun       = freerun["r2"][obs_arr]
        r2_onestep_median    = float(np.nanmedian(r2_obs_onestep))
        r2_loo_median        = float(np.nanmedian(r2_obs_loo))
        r2_freerun_median    = float(np.nanmedian(r2_obs_freerun))

        # Per-worm lambda_u and G for cross-worm comparison
        with torch.no_grad():
            lam_w = model.lambda_u.detach().cpu().numpy()       # (N_atlas,) shared
            G_w   = float(ws.G.item()) if ws.G is not None else None

        per_worm[ws.worm_id] = {
            "onestep":             onestep,
            "loo":                 loo,
            "freerun":             freerun,
            "r2_onestep_median":   r2_onestep_median,
            "r2_loo_median":       r2_loo_median,
            "r2_freerun_median":   r2_freerun_median,
            "N_obs":               ws.N_obs,
            "N_unobs":             ws.N_unobs,
            "dataset":             ws.dataset_type,
            "weight":              ws.weight,
            "lambda_u":            lam_w,
            "G_worm":              G_w,
        }

        _logger.info(
            "eval_worm_done",
            worm_id=ws.worm_id,
            r2_onestep=r2_onestep_median,
            r2_loo=r2_loo_median,
            r2_freerun=r2_freerun_median,
        )
        print(
            f"[MultiWorm Eval] {ws.worm_id}"
            f"  one-step R²={r2_onestep_median:.3f}"
            f"  LOO R²={r2_loo_median:.3f}"
            f"  free-run R²={r2_freerun_median:.3f}"
            f"  N_obs={ws.N_obs}"
        )

    r2_os_list  = [v["r2_onestep_median"] for v in per_worm.values()
                   if np.isfinite(v["r2_onestep_median"])]
    r2_loo_list = [v["r2_loo_median"] for v in per_worm.values()
                   if np.isfinite(v["r2_loo_median"])]
    r2_fr_list  = [v["r2_freerun_median"] for v in per_worm.values()
                   if np.isfinite(v["r2_freerun_median"])]

    # Per-neuron mean one-step R² across worms that observed each neuron
    neuron_r2_os_sum  = np.zeros(N_atlas)
    neuron_r2_os_cnt  = np.zeros(N_atlas, dtype=int)
    neuron_r2_loo_sum = np.zeros(N_atlas)
    neuron_r2_loo_cnt = np.zeros(N_atlas, dtype=int)
    neuron_r2_fr_sum  = np.zeros(N_atlas)
    neuron_r2_fr_cnt  = np.zeros(N_atlas, dtype=int)
    neuron_obs_count  = np.zeros(N_atlas, dtype=int)   # how many worms observed each neuron

    for ws, ws_data in zip(worm_states, per_worm.values()):
        obs_arr = ws_data["onestep"]["obs_idx"]
        neuron_obs_count[obs_arr] += 1

        r2_os  = ws_data["onestep"]["r2"]    # (N_atlas,), NaN at unobserved
        r2_loo = ws_data["loo"]["r2"]
        r2_fr  = ws_data["freerun"]["r2"]

        for i in obs_arr:
            if np.isfinite(r2_os[i]):
                neuron_r2_os_sum[i] += r2_os[i]
                neuron_r2_os_cnt[i] += 1
            if np.isfinite(r2_loo[i]):
                neuron_r2_loo_sum[i] += r2_loo[i]
                neuron_r2_loo_cnt[i] += 1
            if np.isfinite(r2_fr[i]):
                neuron_r2_fr_sum[i] += r2_fr[i]
                neuron_r2_fr_cnt[i] += 1

    neuron_r2_os  = np.full(N_atlas, np.nan)
    neuron_r2_loo = np.full(N_atlas, np.nan)
    neuron_r2_fr  = np.full(N_atlas, np.nan)
    pos_os  = neuron_r2_os_cnt  > 0
    pos_loo = neuron_r2_loo_cnt > 0
    pos_fr  = neuron_r2_fr_cnt  > 0
    neuron_r2_os[pos_os]   = neuron_r2_os_sum[pos_os]  / neuron_r2_os_cnt[pos_os]
    neuron_r2_loo[pos_loo] = neuron_r2_loo_sum[pos_loo] / neuron_r2_loo_cnt[pos_loo]
    neuron_r2_fr[pos_fr]   = neuron_r2_fr_sum[pos_fr]  / neuron_r2_fr_cnt[pos_fr]

    # Coverage matrix: (N_atlas, n_worms) binary
    n_worms = len(worm_states)
    coverage_matrix = np.zeros((N_atlas, n_worms), dtype=bool)
    for w_idx, ws in enumerate(worm_states):
        coverage_matrix[ws.obs_idx.cpu().numpy(), w_idx] = True

    summary = {
        "r2_onestep_mean":   float(np.nanmean(r2_os_list))  if r2_os_list  else float("nan"),
        "r2_onestep_median": float(np.nanmedian(r2_os_list)) if r2_os_list else float("nan"),
        "r2_loo_mean":       float(np.nanmean(r2_loo_list))  if r2_loo_list else float("nan"),
        "r2_loo_median":     float(np.nanmedian(r2_loo_list)) if r2_loo_list else float("nan"),
        "r2_freerun_mean":   float(np.nanmean(r2_fr_list))   if r2_fr_list  else float("nan"),
        "r2_freerun_median": float(np.nanmedian(r2_fr_list)) if r2_fr_list  else float("nan"),
        "neuron_r2_onestep": neuron_r2_os,
        "neuron_r2_loo":     neuron_r2_loo,
        "neuron_r2_freerun": neuron_r2_fr,
        "neuron_obs_count":  neuron_obs_count,
        "coverage_matrix":   coverage_matrix,
        "worm_ids":          [ws.worm_id for ws in worm_states],
    }

    print(
        f"\n[MultiWorm Eval] {n_worms} worms  "
        f"One-step R² median={summary['r2_onestep_median']:.3f}  "
        f"LOO R² median={summary['r2_loo_median']:.3f}  "
        f"Free-run R² median={summary['r2_freerun_median']:.3f}"
    )

    return {
        "per_worm":     per_worm,
        "summary":      summary,
        "atlas_labels": atlas_labels,
    }
