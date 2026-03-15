"""Stage-2 evaluation helpers and public evaluation entry points."""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List

from .behavior import (
    behaviour_all_neurons_prediction,
    evaluate_e2e_decoder,
    evaluate_training_decoder,
)
from . import get_stage2_logger
from .model import Stage2ModelPT

__all__ = [
    "compute_onestep",
    "compute_free_run",
    "loo_forward_simulate",
    "run_loo_all",
    "choose_loo_subset",
    "compute_current_decomposition",
    "evaluate_training_decoder",
    "evaluate_e2e_decoder",
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


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m].astype(np.float64), y_pred[m].astype(np.float64)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((yt - yp) ** 2) / ss_tot)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m].astype(np.float64) - y_pred[m].astype(np.float64)) ** 2)))


def _per_neuron_metrics(
    u_true: np.ndarray, u_pred: np.ndarray, indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute R², Pearson r, RMSE for *indices* columns; others stay NaN."""
    N = u_true.shape[1]
    r2, corr, rmse = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    for i in indices:
        r2[i]   = _r2(u_true[:, i], u_pred[:, i])
        corr[i] = _pearson(u_true[:, i], u_pred[:, i])
        rmse[i] = _rmse(u_true[:, i], u_pred[:, i])
    return r2, corr, rmse

def _get_clip_bounds(model) -> Tuple[Optional[float], Optional[float]]:
    lo = getattr(model, "u_next_clip_min", -50.0)
    hi = getattr(model, "u_next_clip_max", 50.0)
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


def _cfg_val(cfg, attr: str, default, cast=float):
    return cast(getattr(cfg, attr, default) or default) if cfg is not None else cast(default)


def _teacher_forced_prior(
    model: Stage2ModelPT, u: torch.Tensor, gating, stim,
) -> torch.Tensor:
    """One-step teacher-forced prior: u(t-1) → u(t) with ground-truth input.

    Despite being named ``_rollout_prior`` historically, this function
    uses teacher forcing (ground-truth u at each step), **not** true
    free-running rollout.  Renamed for clarity.
    """
    T, N = u.shape
    device = u.device
    prior_mu = torch.zeros_like(u)
    prior_mu[0] = u[0]
    s_sv  = torch.zeros(N, model.r_sv,  device=device)
    s_dcv = torch.zeros(N, model.r_dcv, device=device)
    with torch.no_grad():
        for t in range(1, T):
            g = gating[t - 1] if gating is not None else torch.ones(N, device=device)
            s = stim[t - 1]   if stim   is not None else None
            prior_mu[t], s_sv, s_dcv = model.prior_step(u[t - 1], s_sv, s_dcv, g, s)
    return prior_mu


def _ar1_smooth(u: torch.Tensor, lam: torch.Tensor, I0: torch.Tensor) -> np.ndarray:
    """Apply per-neuron AR(1) smoothing without any network coupling.

    u(t+1) = (1-λ)*u(t) + λ*I₀

    This isolates the temporal-smoothing effect of the leak so that a
    behaviour decoder trained on these traces can be compared fairly
    against one trained on the full-model traces.
    """
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


def compute_free_run(model: Stage2ModelPT, data: Dict[str, Any]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    u0 = data["u_stage1"].to(device)
    T, N = u0.shape
    cfg = data.get("_cfg")
    gating, stim = data.get("gating"), data.get("stim")
    lo, hi = _get_clip_bounds(model)

    motor_idx: list[int] = []
    if cfg is not None and getattr(cfg, "motor_neurons", None) is not None:
        motor_idx = [int(i) for i in cfg.motor_neurons if 0 <= int(i) < N]
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
            g = gating[t - 1] if gating is not None else torch.ones(N, device=device)
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

def loo_forward_simulate(
    model: Stage2ModelPT, u_all: torch.Tensor,
    held_out: int, gating, stim,
) -> np.ndarray:
    T, N = u_all.shape
    device = u_all.device
    i = held_out
    lo, hi = _get_clip_bounds(model)

    u_pred = torch.zeros(T, device=device)
    u_pred[0] = u_all[0, i]
    s_sv  = torch.zeros(N, model.r_sv,  device=device)
    s_dcv = torch.zeros(N, model.r_dcv, device=device)

    with torch.no_grad():
        for t in range(T - 1):
            u_t = u_all[t].clone()
            u_t[i] = u_pred[t]
            g = gating[t] if gating is not None else torch.ones(N, device=device)
            s = stim[t]   if stim   is not None else None
            mu_next, s_sv, s_dcv = model.prior_step(u_t, s_sv, s_dcv, g, s)
            u_pred[t + 1] = _clamp(mu_next[i : i + 1], lo, hi).squeeze()

    return u_pred.cpu().numpy()


def run_loo_all(
    model: Stage2ModelPT, data: Dict[str, Any],
    subset: Optional[List[int]] = None,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    gating, stim = data.get("gating"), data.get("stim")
    indices = subset if subset is not None else list(range(u.shape[1]))
    u_np = u.cpu().numpy()

    preds = {}
    _logger = get_stage2_logger()
    _logger.info("loo_start", n_neurons=len(indices))
    for cnt, i in enumerate(indices):
        preds[i] = loo_forward_simulate(model, u, i, gating, stim)
        if cnt == 0 or (cnt + 1) % 10 == 0:
            _logger.info("loo_progress", done=cnt + 1, total=len(indices),
                         neuron=int(i), r2=float(_r2(u_np[:, i], preds[i])))

    pred_full = np.column_stack([preds.get(i, u_np[:, i]) for i in range(u.shape[1])])
    r2, corr, rmse = _per_neuron_metrics(u_np, pred_full, indices)
    return {"pred": preds, "r2": r2, "corr": corr, "rmse": rmse}


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
    if mode == "variance":
        return [int(i) for i in np.argsort(np.nanvar(u_np, axis=0))[::-1][:k]]
    if mode in ("worst_onestep", "best_onestep"):
        r2 = np.asarray(onestep["r2"], dtype=float)
        fill = np.inf if mode == "worst_onestep" else -np.inf
        score = np.where(np.isfinite(r2), r2, fill)
        order = np.argsort(score) if mode == "worst_onestep" else np.argsort(score)[::-1]
        return [int(i) for i in order[:k]]

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

        for t in range(T - 1):
            u_prev = u[t]
            g = gating[t] if gating is not None else torch.ones(N, device=device)
            s = stim[t]   if stim   is not None else None

            out["I_leak"][t] = ((1.0 - lam) * u_prev).cpu().numpy()
            out["I_bias"][t] = (lam * model.I0).cpu().numpy()
            out["I_gap"][t]  = (lam * (model.laplacian() @ u_prev)).cpu().numpy()

            phi_prev = model.phi(u_prev) * g

            for prefix, r, s_state, T_mask, W_param, a_param, tau_param, E_param in [
                ("I_sv",  model.r_sv,  s_sv_state,  model.T_sv,  model._get_W("W_sv"),
                 model.a_sv,  model.tau_sv,  model.E_sv),
                ("I_dcv", model.r_dcv, s_dcv_state, model.T_dcv, model._get_W("W_dcv"),
                 model.a_dcv, model.tau_dcv, model.E_dcv),
            ]:
                if r == 0:
                    continue
                gamma = torch.exp(-dt / (tau_param + 1e-12))
                s_state = gamma.view(1, -1) * s_state + phi_prev.unsqueeze(1)
                a = a_param.unsqueeze(0) if a_param.dim() == 1 else a_param
                sa = s_state * a
                T_eff = T_mask * W_param
                g_syn = torch.matmul(T_eff.t(), sa).sum(dim=1)

                E = E_param
                if E.dim() == 0:
                    I_t = lam * g_syn * (E - u_prev)
                elif E.dim() == 1:
                    e_drive = torch.matmul(T_eff.t(), E.view(-1, 1) * sa).sum(dim=1)
                    I_t = lam * (e_drive - g_syn * u_prev)
                else:
                    e_drive = torch.matmul((T_eff * E).t(), sa).sum(dim=1)
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
    decoder=None, e2e_decoder=None, beh_all_baseline=None, *,
    skip_beh_all: bool = False,
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

    loo      = run_loo_all(model, data, subset=subset)
    currents = compute_current_decomposition(model, data)
    free_run = compute_free_run(model, data)

    beh     = evaluate_training_decoder(decoder, data, onestep) if decoder is not None else None
    beh_e2e = evaluate_e2e_decoder(e2e_decoder, data, onestep)  if e2e_decoder is not None else None
    beh_all = (
        beh_all_baseline if (not skip_beh_all and beh_all_baseline is not None)
        else behaviour_all_neurons_prediction(data) if not skip_beh_all
        else None
    )

    r2_model_mean = None
    for src in (beh_e2e, beh):
        if src is not None:
            r2 = src.get("r2_model_heldout")
            if r2 is None or not np.any(np.isfinite(r2)):
                r2 = src["r2_model"]
            r2_model_mean = float(np.nanmean(r2))
            break

    return {
        "onestep": onestep, "loo": loo, "currents": currents,
        "free_run": free_run,
        "beh": beh, "beh_e2e": beh_e2e, "beh_all": beh_all,
        "beh_r2_model": r2_model_mean,
    }
