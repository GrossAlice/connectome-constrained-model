from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List

from . import get_stage2_logger
from ._utils import (
    _r2,
    _per_neuron_metrics,
    _get_clip_bounds,
    _clamp,
    _resolve_motor_indices,
    _ar1_smooth,
)
from .model import Stage2ModelPT

__all__ = [
    "compute_onestep",
    "compute_free_run",
    "free_run_stochastic",
    "loo_forward_simulate_batched",
    "run_loo_all",
    "choose_loo_subset",
    "compute_current_decomposition",
]


def _teacher_forced_prior(
    model: Stage2ModelPT, u: torch.Tensor, gating, stim,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        params = model.precompute_params()
        return model.forward_sequence(u, gating_data=gating,
                                      stim_data=stim, params=params)


def compute_onestep(model: Stage2ModelPT, data: Dict[str, Any]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    N = u.shape[1]

    prior_mu = _teacher_forced_prior(model, u, data.get("gating"), data.get("stim"))
    u_np, mu_np = u.cpu().numpy(), prior_mu.cpu().numpy()
    ar1_np = _ar1_smooth(u, model.lambda_u, model.I0)

    r2, corr, rmse = _per_neuron_metrics(u_np[1:], mu_np[1:], list(range(N)))

    du_true = u_np[1:] - u_np[:-1]
    du_pred = mu_np[1:] - u_np[:-1]
    r2_delta, corr_delta, _ = _per_neuron_metrics(du_true, du_pred, list(range(N)))

    return {
        "prior_mu": mu_np, "ar1_mu": ar1_np,
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
    ones = torch.ones(N, device=device)
    seed_steps = max(1, min(int(getattr(cfg, "eval_free_run_seed_steps", 16) or 16), T))

    motor_idx = _resolve_motor_indices(data, N)
    conditioned = 0 < len(motor_idx) < N
    motor_mask = torch.zeros(N, dtype=torch.bool, device=device)
    if motor_idx:
        motor_mask[motor_idx] = True

    with torch.no_grad():
        u_free = torch.zeros_like(u0)
        u_free[:seed_steps] = u0[:seed_steps]
        s_sv = torch.zeros(N, model.r_sv, device=device)
        s_dcv = torch.zeros(N, model.r_dcv, device=device)
        model.reset_lag_history()
        if hasattr(model, 'reset_synapse_lag_history'):
            model.reset_synapse_lag_history()
        if hasattr(model, 'reset_iir_delay_history'):
            model.reset_iir_delay_history()
        for t in range(1, seed_steps):
            g = gating[t - 1] if gating is not None else ones
            s = stim[t - 1] if stim is not None else None
            _, s_sv, s_dcv = model.prior_step(u0[t - 1], s_sv, s_dcv, g, s)

        for t in range(seed_steps, T):
            g = gating[t - 1] if gating is not None else ones
            s = stim[t - 1] if stim is not None else None
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
    if seed_steps >= T:
        r2 = np.full(N, np.nan)
        corr = np.full(N, np.nan)
        rmse = np.full(N, np.nan)
    else:
        r2, corr, rmse = _per_neuron_metrics(u_np[seed_steps:], uf_np[seed_steps:], eval_idx)

    return {
        "u_free": uf_np,
        "r2": r2, "corr": corr, "rmse": rmse,
        "motor_idx": np.array(motor_idx, dtype=int),
        "seed_steps": seed_steps,
        "eval_range": (seed_steps, T),
        "mode": "motor_conditioned" if conditioned else "autonomous",
    }


def free_run_stochastic(
    model: Stage2ModelPT, data: Dict[str, Any], n_samples: int = 5,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    u0 = data["u_stage1"].to(device)
    T, N = u0.shape
    gating, stim = data.get("gating"), data.get("stim")
    lo, hi = _get_clip_bounds(model)
    ones = torch.ones(N, device=device)
    is_hetero = (getattr(model, "_noise_mode", "homoscedastic") == "heteroscedastic")
    if not is_hetero:
        sigma_const = model.sigma_at().detach()

    samples_np = np.zeros((n_samples, T, N), dtype=np.float32)

    with torch.no_grad():
        for k in range(n_samples):
            u_cur = u0[0].clone()
            samples_np[k, 0] = u_cur.cpu().numpy()
            s_sv = torch.zeros(N, model.r_sv, device=device)
            s_dcv = torch.zeros(N, model.r_dcv, device=device)
            model.reset_lag_history()
            if hasattr(model, 'reset_synapse_lag_history'):
                model.reset_synapse_lag_history()
            if hasattr(model, 'reset_iir_delay_history'):
                model.reset_iir_delay_history()
            for t in range(T - 1):
                g = gating[t] if gating is not None else ones
                s = stim[t] if stim is not None else None
                mu_next, s_sv, s_dcv = model.prior_step(u_cur, s_sv, s_dcv, g, s)
                sigma_t = model.sigma_at(u_cur).detach() if is_hetero else sigma_const
                if getattr(model, 'noise_corr_rank', 0) > 0:
                    u_next = mu_next + model.sample_correlated_noise(sigma_t)
                else:
                    u_next = mu_next + sigma_t * torch.randn(N, device=device)
                u_cur = _clamp(u_next, lo, hi)
                samples_np[k, t + 1] = u_cur.cpu().numpy()

    u_np = u0.cpu().numpy()
    ens_mean = samples_np.mean(axis=0)

    u_ref = u_np[1:].astype(np.float64)
    if np.isfinite(u_ref).all() and np.isfinite(samples_np[:, 1:]).all():
        u_ref_mean = u_ref.mean(axis=0)
        ss_tot_safe = np.maximum(((u_ref - u_ref_mean) ** 2).sum(axis=0), 1e-12)
        r2_per_sample = np.empty((n_samples, N))
        for k in range(n_samples):
            ss_res = ((u_ref - samples_np[k, 1:].astype(np.float64)) ** 2).sum(axis=0)
            r2_per_sample[k] = 1.0 - ss_res / ss_tot_safe
        ens_ss = ((u_ref - ens_mean[1:].astype(np.float64)) ** 2).sum(axis=0)
        r2_mean = 1.0 - ens_ss / ss_tot_safe
    else:
        r2_per_sample = np.full((n_samples, N), np.nan)
        for k in range(n_samples):
            for j in range(N):
                r2_per_sample[k, j] = _r2(u_np[1:, j], samples_np[k, 1:, j])
        r2_mean = np.array([_r2(u_np[1:, j], ens_mean[1:, j]) for j in range(N)])

    return {
        "samples": samples_np,
        "mean": ens_mean,
        "std": samples_np.std(axis=0),
        "ci_lo": np.percentile(samples_np, 2.5, axis=0),
        "ci_hi": np.percentile(samples_np, 97.5, axis=0),
        "r2_per_sample": r2_per_sample,
        "r2_mean": r2_mean,
    }


def _batched_synaptic_current(u_prev, phi_gated, s, T_eff, a, tau, E, dt):
    gamma = torch.exp(-dt / (tau + 1e-12))
    s_next = gamma.view(1, 1, -1) * s + phi_gated.unsqueeze(-1)
    pool = torch.einsum("mn,bnr->bmr", T_eff.t(), s_next)
    a_post = a.unsqueeze(0) if a.dim() == 1 else a
    g = (pool * a_post.unsqueeze(0)).sum(-1)

    if E.dim() == 0:
        I = g * (E - u_prev)
    elif E.dim() == 1:
        pool_E = torch.einsum("mn,bnr->bmr", T_eff.t(), s_next * E.view(1, -1, 1))
        I = (pool_E * a_post.unsqueeze(0)).sum(-1) - g * u_prev
    else:
        pool_E = torch.einsum("mn,bnr->bmr", (T_eff * E).t(), s_next)
        I = (pool_E * a_post.unsqueeze(0)).sum(-1) - g * u_prev
    return I, s_next


def loo_forward_simulate_batched(
    model: Stage2ModelPT,
    u_all: torch.Tensor,
    held_out_indices: List[int],
    gating,
    stim,
    window_size: int = 0,
    warmup_steps: int = 0,
) -> Dict[int, np.ndarray]:
    T, N = u_all.shape
    device = u_all.device
    B = len(held_out_indices)
    if B == 0:
        return {}

    idx = torch.tensor(held_out_indices, device=device, dtype=torch.long)
    lo, hi = _get_clip_bounds(model)
    ones = torch.ones(N, device=device)

    u_pred = torch.zeros(B, T, device=device)
    u_pred[:, 0] = u_all[0, idx]

    lambda_u = model.lambda_u.detach()
    I0 = model.I0.detach()
    L = model.laplacian().detach()
    has_sv = model.r_sv > 0
    has_dcv = model.r_dcv > 0
    if has_sv:
        T_sv_eff = (model.T_sv * model._get_W("W_sv")).detach()
        a_sv, tau_sv, E_sv = model.a_sv.detach(), model.tau_sv.detach(), model.E_sv.detach()
    if has_dcv:
        T_dcv_eff = (model.T_dcv * model._get_W("W_dcv")).detach()
        a_dcv, tau_dcv, E_dcv = model.a_dcv.detach(), model.tau_dcv.detach(), model.E_dcv.detach()

    has_lr = getattr(model, 'lowrank_rank', 0) > 0
    if has_lr:
        U_lr = model.U_lowrank.detach()  # (K, N)
        V_lr = model.V_lowrank.detach()  # (N, K)

    gp_order = getattr(model, 'graph_poly_order', 1)
    if gp_order > 1:
        gp_alpha = torch.tanh(model.graph_poly_alpha).detach()  # (order-1,)

    lag_K = getattr(model, '_lag_order', 0)
    lag_nbr = lag_K > 0 and getattr(model, '_lag_neighbor', False)
    lag_nbr_per_type = lag_K > 0 and getattr(model, '_lag_nbr_per_type', False)
    lag_nbr_act = str(getattr(model, '_lag_nbr_act', 'none')) if lag_K > 0 else 'none'
    if lag_K > 0:
        lag_alpha = model._lag_alpha.detach()              # (K, N)
        if lag_nbr:
            if lag_nbr_per_type:
                # Per-type: list of (prefix, G_eff) tuples
                _lag_per_type_list = []
                for prefix in getattr(model, '_lag_nbr_types', []):
                    G_p = getattr(model, f'_lag_G_{prefix}').detach()
                    mask_p = getattr(model, f'_lag_nbr_mask_{prefix}').detach()
                    G_eff_p = (G_p * mask_p.unsqueeze(0))  # (K, N, N)
                    _lag_per_type_list.append((prefix, G_eff_p))
            else:
                lag_nbr_mask = model._lag_nbr_mask.detach()    # (N, N)
                lag_G_eff = (model._lag_G * lag_nbr_mask.unsqueeze(0)).detach()  # (K, N, N)

    batch_range = torch.arange(B, device=device)
    ws = T if window_size <= 0 else window_size

    def _warmup_state(burn_start, burn_end):
        sv = torch.zeros(B, N, model.r_sv, device=device) if has_sv else None
        dcv = torch.zeros(B, N, model.r_dcv, device=device) if has_dcv else None
        # Lag buffer: fill from ground truth at end of burn-in
        lb = torch.zeros(B, lag_K, N, device=device) if lag_K > 0 else None
        if lag_K > 0:
            for k in range(lag_K):
                t_k = burn_end - 1 - k
                if 0 <= t_k < T:
                    lb[:, k, :] = u_all[t_k]
        for tb in range(burn_start, burn_end):
            u_gt = u_all[tb].unsqueeze(0).expand(B, -1)
            g = gating[tb] if gating is not None else ones
            phi_g = model.phi(u_gt) * g.unsqueeze(0)
            if has_sv:
                _, sv = _batched_synaptic_current(u_gt, phi_g, sv, T_sv_eff, a_sv, tau_sv, E_sv, model.dt)
            if has_dcv:
                _, dcv = _batched_synaptic_current(u_gt, phi_g, dcv, T_dcv_eff, a_dcv, tau_dcv, E_dcv, model.dt)
        return sv, dcv, lb

    def _step(u_t, phi_gated, s_sv, s_dcv, stim_t, lag_buf):
        I_sv_t = I_dcv_t = torch.zeros(B, N, device=device)
        if has_sv:
            I_sv_t, s_sv = _batched_synaptic_current(u_t, phi_gated, s_sv, T_sv_eff, a_sv, tau_sv, E_sv, model.dt)
        if has_dcv:
            I_dcv_t, s_dcv = _batched_synaptic_current(u_t, phi_gated, s_dcv, T_dcv_eff, a_dcv, tau_dcv, E_dcv, model.dt)

        I_gap = u_t @ L.T  # (B, N)

        if gp_order > 1:
            L_pow_u = I_gap
            for p in range(gp_order - 1):
                L_pow_u = L_pow_u @ L.T
                I_gap = I_gap + gp_alpha[p] * L_pow_u

        I_lr = torch.zeros(B, N, device=device)
        if has_lr:
            I_lr = torch.tanh(u_t @ U_lr.T) @ V_lr.T  # (B,N)@(N,K)->(B,K)->(B,N)

        # Lag: push u_t into buffer, compute weighted sum
        I_lag = torch.zeros(B, N, device=device)
        if lag_K > 0:
            # Push: shift buffer right, insert u_t at position 0
            lag_buf = torch.cat([u_t.unsqueeze(1), lag_buf[:, :-1, :]], dim=1)  # (B,K,N)
            # Self-lag: weighted sum over history
            I_lag = (lag_alpha.unsqueeze(0) * lag_buf).sum(1)  # (1,K,N)*(B,K,N)->sum->(B,N)
            # Neighbor-lag: sparse connectome matmul per lag order
            if lag_nbr:
                if lag_nbr_per_type:
                    # Apply activation to buffer for chemical types
                    if lag_nbr_act in ('none', 'identity', ''):
                        buf_act = lag_buf
                    elif lag_nbr_act == 'sigmoid':
                        buf_act = torch.sigmoid(lag_buf)
                    elif lag_nbr_act == 'softplus':
                        import torch.nn.functional as _F
                        buf_act = _F.softplus(lag_buf)
                    elif lag_nbr_act == 'tanh':
                        buf_act = torch.tanh(lag_buf)
                    elif lag_nbr_act == 'relu':
                        import torch.nn.functional as _F
                        buf_act = _F.relu(lag_buf)
                    else:
                        buf_act = lag_buf
                    for prefix, G_eff_p in _lag_per_type_list:
                        buf_p = lag_buf if prefix == 'e' else buf_act
                        I_lag = I_lag + torch.einsum('kij,bkj->bi', G_eff_p, buf_p)
                else:
                    I_lag = I_lag + torch.einsum('kij,bkj->bi', lag_G_eff, lag_buf)

        I_stim = torch.zeros(B, N, device=device)
        if model.d_ell > 0 and stim_t is not None:
            if model.stim_diagonal_only:
                I_stim = (model.b * stim_t).unsqueeze(0).expand(B, -1)
            else:
                I_stim = (model.b @ stim_t).unsqueeze(0).expand(B, -1)

        u_next = (1.0 - lambda_u) * u_t + lambda_u * (I0 + I_gap + I_sv_t + I_dcv_t + I_lr + I_lag + I_stim)
        if lo is not None or hi is not None:
            u_next = u_next.clamp(min=lo, max=hi)
        return u_next, s_sv, s_dcv, lag_buf

    u_t_buf = torch.empty(B, N, device=device)

    with torch.no_grad():
        for w_start in range(0, T, ws):
            w_end = min(w_start + ws, T)
            u_pred[:, w_start] = u_all[w_start, idx]

            if warmup_steps > 0 and w_start > 0:
                s_sv, s_dcv, lag_buf = _warmup_state(max(0, w_start - warmup_steps), w_start)
            else:
                s_sv = torch.zeros(B, N, model.r_sv, device=device) if has_sv else None
                s_dcv = torch.zeros(B, N, model.r_dcv, device=device) if has_dcv else None
                # Cold-start lag buffer from ground truth
                lag_buf = None
                if lag_K > 0:
                    lag_buf = torch.zeros(B, lag_K, N, device=device)
                    for k in range(lag_K):
                        t_k = w_start - 1 - k
                        if 0 <= t_k < T:
                            lag_buf[:, k, :] = u_all[t_k]

            for t in range(w_start, w_end - 1):
                u_t_buf[:] = u_all[t]
                u_t_buf[batch_range, idx] = u_pred[:, t]
                g = gating[t] if gating is not None else ones
                phi_gated = model.phi(u_t_buf) * g.unsqueeze(0)
                stim_t = stim[t] if stim is not None else None
                u_next, s_sv, s_dcv, lag_buf = _step(u_t_buf, phi_gated, s_sv, s_dcv, stim_t, lag_buf)
                u_pred[:, t + 1] = u_next[batch_range, idx]

    u_pred_np = u_pred.cpu().numpy()
    return {int(held_out_indices[b]): u_pred_np[b] for b in range(B)}


loo_forward_simulate_batched_windowed = loo_forward_simulate_batched


def run_loo_all(
    model: Stage2ModelPT, data: Dict[str, Any],
    subset: Optional[List[int]] = None,
    window_size: int = 0,
    warmup_steps: int = 0,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    gating, stim = data.get("gating"), data.get("stim")
    indices = subset if subset is not None else list(range(u.shape[1]))
    u_np = u.cpu().numpy()
    use_windowed = window_size > 0
    labels = data.get("neuron_labels", [])
    _logger = get_stage2_logger()

    _logger.info("loo_start", n_neurons=len(indices),
                 windowed=use_windowed, window_size=window_size)

    preds = loo_forward_simulate_batched(
        model, u, indices, gating, stim,
        window_size=window_size, warmup_steps=warmup_steps)
    variant = "windowed" if use_windowed else "deterministic"
    _logger.info("loo_batched_done", n_neurons=len(indices), variant=variant,
                 warmup_steps=warmup_steps)

    for cnt, i in enumerate(indices):
        lbl = labels[i] if i < len(labels) else f"#{i}"
        r2_i = float(_r2(u_np[1:, i], preds[i][1:]))
        if cnt == 0 or (cnt + 1) % 10 == 0 or len(indices) <= 10:
            _logger.info("loo_progress", done=cnt + 1, total=len(indices),
                         neuron=int(i), name=lbl, r2=r2_i)

    pred_full = np.column_stack([preds.get(i, u_np[:, i]) for i in range(u.shape[1])])
    r2, corr, rmse = _per_neuron_metrics(u_np[1:], pred_full[1:], indices)

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

    candidates = None
    if mode.startswith("motor"):
        candidates = _resolve_motor_indices(data, N) or list(range(N))

    if mode == "motor_best_onestep":
        r2 = np.asarray(onestep["r2"], dtype=float)
        scores = np.where(np.isfinite(r2), r2, -np.inf)
        return sorted(candidates, key=lambda i: scores[i], reverse=True)[:k]
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
        cfg = data.get("_cfg")
        names = getattr(cfg, "eval_loo_subset_names", None) if cfg else None
        if names:
            labels = data.get("neuron_labels", [])
            lbl_lower = [str(l).strip().lower() for l in labels]
            found = [lbl_lower.index(str(nm).strip().lower())
                     for nm in names if str(nm).strip().lower() in lbl_lower]
            if found:
                return sorted(set(found))
        return None

    return sorted(int(i) for i in np.random.default_rng(seed).choice(N, size=k, replace=False))


def compute_current_decomposition(
    model: Stage2ModelPT, data: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    T, N = u.shape
    gating, stim = data.get("gating"), data.get("stim")

    keys = ["I_leak", "I_gap", "I_sv", "I_dcv", "I_stim", "I_bias",
            "I_lr", "I_lag"]
    out = {k: np.zeros((T - 1, N)) for k in keys}

    with torch.no_grad():
        s_sv_state = torch.zeros(N, model.r_sv, device=device)
        s_dcv_state = torch.zeros(N, model.r_dcv, device=device)
        model.reset_lag_history()
        if hasattr(model, 'reset_synapse_lag_history'):
            model.reset_synapse_lag_history()
        if hasattr(model, 'reset_iir_delay_history'):
            model.reset_iir_delay_history()
        lam, dt = model.lambda_u.detach(), model.dt
        L = model.laplacian()
        ones = torch.ones(N, device=device)

        syn_terms = []
        if model.r_sv > 0:
            syn_terms.append(("I_sv", "sv",
                torch.exp(-dt / (model.tau_sv + 1e-12)),
                model.T_sv * model._get_W("W_sv"), model.a_sv, model.E_sv))
        if model.r_dcv > 0:
            syn_terms.append(("I_dcv", "dcv",
                torch.exp(-dt / (model.tau_dcv + 1e-12)),
                model.T_dcv * model._get_W("W_dcv"), model.a_dcv, model.E_dcv))

        for t in range(T - 1):
            u_prev = u[t]
            g = gating[t] if gating is not None else ones
            s = stim[t] if stim is not None else None

            out["I_leak"][t] = ((1.0 - lam) * u_prev).cpu().numpy()
            out["I_bias"][t] = (lam * model.I0).cpu().numpy()
            out["I_gap"][t] = (lam * (L @ u_prev)).cpu().numpy()

            phi_prev = model.phi(u_prev) * g

            for prefix, state_name, gamma, T_eff, a_param, E_param in syn_terms:
                s_state = s_sv_state if state_name == "sv" else s_dcv_state
                s_state = gamma.view(1, -1) * s_state + phi_prev.unsqueeze(1)
                a = a_param.unsqueeze(0) if a_param.dim() == 1 else a_param
                pool = torch.matmul(T_eff.t(), s_state)
                g_syn = (pool * a).sum(dim=1)

                E = E_param
                if E.dim() == 0:
                    I_t = lam * g_syn * (E - u_prev)
                elif E.dim() == 1:
                    e_pool = torch.matmul(T_eff.t(), s_state * E.view(-1, 1))
                    I_t = lam * ((e_pool * a).sum(dim=1) - g_syn * u_prev)
                else:
                    e_pool = torch.matmul((T_eff * E).t(), s_state)
                    I_t = lam * ((e_pool * a).sum(dim=1) - g_syn * u_prev)
                out[prefix][t] = I_t.cpu().numpy()

                if state_name == "sv":
                    s_sv_state = s_state
                else:
                    s_dcv_state = s_state

            if model.d_ell > 0 and s is not None:
                if getattr(model, "stim_diagonal_only", False):
                    out["I_stim"][t] = (lam * (model.b * s.view(N))).cpu().numpy()
                else:
                    out["I_stim"][t] = (lam * torch.matmul(model.b, s)).cpu().numpy()

            if getattr(model, 'lowrank_rank', 0) > 0:
                out["I_lr"][t] = (lam * (model.V_lowrank @ torch.tanh(
                    model.U_lowrank @ u_prev))).cpu().numpy()

            if getattr(model, '_lag_order', 0) > 0:
                out["I_lag"][t] = (lam * model._lag_push_and_compute(
                    u_prev)).cpu().numpy()

    return out



