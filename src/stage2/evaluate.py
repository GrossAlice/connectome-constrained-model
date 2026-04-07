from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
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
    "compute_free_run",
    "free_run_stochastic",
    "loo_forward_simulate",
    "loo_forward_simulate_windowed",
    "loo_forward_simulate_batched",
    "loo_forward_simulate_batched_windowed",
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
    model.eval()
    with torch.no_grad():
        params = model.precompute_params()
        prior_mu = model.forward_sequence(u, gating_data=gating,
                                          stim_data=stim, params=params)
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


def compute_free_run(model: Stage2ModelPT, data: Dict[str, Any]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    u0 = data["u_stage1"].to(device)
    T, N = u0.shape
    cfg = data.get("_cfg")
    gating, stim = data.get("gating"), data.get("stim")
    lo, hi = _get_clip_bounds(model)
    ones = torch.ones(N, device=device)
    seed_steps = int(getattr(cfg, "eval_free_run_seed_steps", 16) or 16)
    seed_steps = max(1, min(seed_steps, T))

    motor_idx = _resolve_motor_indices(data, N)
    conditioned = 0 < len(motor_idx) < N
    motor_mask = torch.zeros(N, dtype=torch.bool, device=device)
    if motor_idx:
        motor_mask[motor_idx] = True

    with torch.no_grad():
        u_free = torch.zeros_like(u0)
        u_free[:seed_steps] = u0[:seed_steps]
        s_sv  = torch.zeros(N, model.r_sv,  device=device)
        s_dcv = torch.zeros(N, model.r_dcv, device=device)

        for t in range(1, seed_steps):
            g = gating[t - 1] if gating is not None else ones
            s = stim[t - 1]   if stim   is not None else None
            _, s_sv, s_dcv = model.prior_step(u0[t - 1], s_sv, s_dcv, g, s)

        for t in range(seed_steps, T):
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
    if seed_steps >= T:
        r2 = np.full(N, np.nan)
        corr = np.full(N, np.nan)
        rmse = np.full(N, np.nan)
    else:
        r2, corr, rmse = _per_neuron_metrics(
            u_np[seed_steps:],
            uf_np[seed_steps:],
            eval_idx,
        )

    return {
        "u_free": uf_np,
        "r2": r2, "corr": corr, "rmse": rmse,
        "motor_idx": np.array(motor_idx, dtype=int),
        "seed_steps": seed_steps,
        "eval_range": (seed_steps, T),
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
                # Use correlated noise if low-rank factor is present
                if getattr(model, 'noise_corr_rank', 0) > 0:
                    eps_full = model.sample_correlated_noise(sigma_t)
                    u_next = mu_next + eps_full
                else:
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


def loo_forward_simulate_windowed(
    model: Stage2ModelPT, u_all: torch.Tensor,
    held_out: int, gating, stim,
    window_size: int = 200,
    warmup_steps: int = 0,
) -> np.ndarray:
    """LOO prediction with periodic re-seeding to reduce compounding error.

    Every *window_size* frames, the held-out neuron's state is reset to its
    ground-truth value.  If *warmup_steps* > 0, synaptic states are
    warm-started by running *warmup_steps* teacher-forced steps on GT data
    before each window (using the frames immediately preceding the window).
    When *warmup_steps* is 0, synaptic states are initialised to zero (the
    original behaviour).

    Parameters
    ----------
    window_size : int
        Number of frames per free-running window.  The held-out neuron is
        re-seeded from GT at the start of each window.  E.g. 200 frames
        at dt=0.6 s ≈ 2 min windows.
    warmup_steps : int
        Number of teacher-forced GT steps to run before each window to
        build up realistic synaptic states.  0 = cold-start (zeros).
        A good default is the longest DCV tau in frames, e.g. 30–50.
    """
    T, N = u_all.shape
    device = u_all.device
    i = held_out
    lo, hi = _get_clip_bounds(model)
    ones = torch.ones(N, device=device)

    u_pred = torch.zeros(T, device=device)
    u_pred[0] = u_all[0, i]

    with torch.no_grad():
        # Process windows
        for w_start in range(0, T, window_size):
            w_end = min(w_start + window_size, T)
            # Re-seed held-out neuron at window start
            u_pred[w_start] = u_all[w_start, i]

            # ── Warm-start synaptic states ──
            s_sv  = torch.zeros(N, model.r_sv,  device=device)
            s_dcv = torch.zeros(N, model.r_dcv, device=device)
            if warmup_steps > 0 and w_start > 0:
                burn_start = max(0, w_start - warmup_steps)
                for tb in range(burn_start, w_start):
                    g = gating[tb] if gating is not None else ones
                    s = stim[tb]   if stim   is not None else None
                    _, s_sv, s_dcv = model.prior_step(
                        u_all[tb], s_sv, s_dcv, g, s)

            for t in range(w_start, w_end - 1):
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


# ── Batched LOO helpers ──────────────────────────────────────────────────────

def _batched_synaptic_current(u_prev, phi_gated, s, T_eff, a, tau, E, dt):
    """Batched version of ``Stage2ModelPT._synaptic_current``.

    Parameters
    ----------
    u_prev    : (B, N)
    phi_gated : (B, N)
    s         : (B, N, r)
    T_eff     : (N, N)  – topology * W matrix
    a         : (r,) or (N, r)
    tau       : (r,)
    E         : scalar, (N,), or (N, N)
    dt        : float
    """
    gamma = torch.exp(-dt / (tau + 1e-12))           # (r,)
    s_next = gamma.view(1, 1, -1) * s + phi_gated.unsqueeze(-1)  # (B, N, r)

    # pool[b] = T_eff^T @ s_next[b]   →  (B, N, r)
    pool = torch.einsum("mn,bnr->bmr", T_eff.t(), s_next)

    a_post = a.unsqueeze(0) if a.dim() == 1 else a   # (1,r) or (N,r)
    g = (pool * a_post.unsqueeze(0)).sum(-1)          # (B, N)

    if E.dim() == 0:
        I = g * (E - u_prev)
    elif E.dim() == 1:
        weighted = s_next * E.view(1, -1, 1)         # (B, N, r)
        pool_E = torch.einsum("mn,bnr->bmr", T_eff.t(), weighted)
        I = (pool_E * a_post.unsqueeze(0)).sum(-1) - g * u_prev
    else:
        # E is (N, N)
        pool_E = torch.einsum("mn,bnr->bmr", (T_eff * E).t(), s_next)
        I = (pool_E * a_post.unsqueeze(0)).sum(-1) - g * u_prev

    return I, s_next


def loo_forward_simulate_batched(
    model: Stage2ModelPT,
    u_all: torch.Tensor,
    held_out_indices: List[int],
    gating,
    stim,
    warmup_steps: int = 0,
) -> Dict[int, np.ndarray]:
    """Simulate LOO for many neurons **in parallel** on GPU.

    Instead of ``len(held_out_indices)`` sequential calls to
    ``loo_forward_simulate`` (each iterating T steps), this runs a *single*
    batched loop of T steps with all held-out neurons processed together.
    Speedup ≈ ``len(held_out_indices)``× (typically ~100×).

    If *warmup_steps* > 0, the first ``warmup_steps`` frames are run with
    ALL neurons at ground truth (teacher-forced) to build up realistic
    synaptic kernel states.  The held-out neuron's prediction is set to GT
    during warmup and actual LOO divergence starts at ``warmup_steps``.

    Returns dict  ``{neuron_index: np.ndarray of shape (T,)}``.
    """
    T, N = u_all.shape
    device = u_all.device
    B = len(held_out_indices)
    if B == 0:
        return {}

    idx = torch.tensor(held_out_indices, device=device, dtype=torch.long)
    lo, hi = _get_clip_bounds(model)
    ones = torch.ones(N, device=device)

    # Running predictions per LOO trial: (B, T)
    u_pred = torch.zeros(B, T, device=device)
    u_pred[:, 0] = u_all[0, idx]

    # Batched synaptic states: (B, N, r)
    s_sv  = torch.zeros(B, N, model.r_sv,  device=device) if model.r_sv  > 0 else None
    s_dcv = torch.zeros(B, N, model.r_dcv, device=device) if model.r_dcv > 0 else None

    # Pre-compute constant model parameters (avoids repeated property access)
    lambda_u = model.lambda_u.detach()       # (N,)
    I0       = model.I0.detach()             # (N,)
    L        = model.laplacian().detach()    # (N, N)
    if model.r_sv > 0:
        T_sv_eff = (model.T_sv * model._get_W("W_sv")).detach()
        a_sv     = model.a_sv.detach()
        tau_sv   = model.tau_sv.detach()
        E_sv     = model.E_sv.detach()
    if model.r_dcv > 0:
        T_dcv_eff = (model.T_dcv * model._get_W("W_dcv")).detach()
        a_dcv     = model.a_dcv.detach()
        tau_dcv   = model.tau_dcv.detach()
        E_dcv     = model.E_dcv.detach()

    # Coupling gate and residual MLP
    has_gate = getattr(model, 'has_coupling_gate', False)
    gate_vals = model.coupling_gate_values.detach() if has_gate else None  # (N,)
    has_mlp = model.residual_mlp is not None
    mlp_ctx_K = getattr(model, '_mlp_context_K', 1)

    # Pre-compute MLP context inputs from u_all: (T, N*K)
    mlp_ctx_inputs = None
    if has_mlp and mlp_ctx_K > 1:
        u_pad = F.pad(u_all, (0, 0, mlp_ctx_K - 1, 0))
        mlp_ctx_inputs = torch.cat(
            [u_pad[mlp_ctx_K - 1 - k : mlp_ctx_K - 1 - k + T] for k in range(mlp_ctx_K)],
            dim=1,
        )  # (T, N*K)

    # Pre-compute linear lag terms from u_all
    lag_order = getattr(model, '_lag_order', 0)
    lag_neighbor = getattr(model, '_lag_neighbor', False)
    lag_all = None  # (T, N) pre-computed I_lag for each timestep
    if lag_order > 0:
        K_lag = lag_order
        lag_alpha = model._lag_alpha.detach()  # (K, N)
        u_lag_pad = F.pad(u_all, (0, 0, K_lag, 0))  # (T+K, N)
        lag_buf = torch.stack(
            [u_lag_pad[K_lag - 1 - k : K_lag - 1 - k + T] for k in range(K_lag)],
            dim=1,
        )  # (T, K, N)
        lag_all = (lag_alpha.unsqueeze(0) * lag_buf).sum(1)  # (T, N)
        if lag_neighbor and hasattr(model, '_lag_G'):
            lag_G = model._lag_G.detach()  # (K, N, N)
            mask = model._lag_nbr_mask  # (N, N)
            for k in range(K_lag):
                G_k = lag_G[k] * mask
                lag_all = lag_all + (lag_buf[:, k, :] @ G_k.T)

    batch_range = torch.arange(B, device=device)
    warmup = min(max(warmup_steps, 0), T - 1)

    with torch.no_grad():
        # ── Warmup: teacher-force ALL neurons (including held-out) ────
        # This builds up realistic synaptic kernel states before LOO
        # divergence.  Predictions during warmup = GT.
        if warmup > 0:
            for t in range(warmup):
                u_gt = u_all[t].unsqueeze(0).expand(B, -1)  # all GT
                g = gating[t] if gating is not None else ones
                phi_gated = model.phi(u_gt) * g.unsqueeze(0)
                if model.r_sv > 0:
                    _, s_sv = _batched_synaptic_current(
                        u_gt, phi_gated, s_sv, T_sv_eff, a_sv, tau_sv, E_sv, model.dt)
                if model.r_dcv > 0:
                    _, s_dcv = _batched_synaptic_current(
                        u_gt, phi_gated, s_dcv, T_dcv_eff, a_dcv, tau_dcv, E_dcv, model.dt)
                # Prediction = GT during warmup
                u_pred[:, t] = u_all[t, idx]
                if t + 1 < T:
                    u_pred[:, t + 1] = u_all[t + 1, idx]

        # ── Main LOO loop: held-out neuron diverges ──────────────────
        for t in range(max(warmup, 0), T - 1):
            # (B, N) – each row starts as ground truth
            u_t = u_all[t].unsqueeze(0).expand(B, -1).clone()
            # Replace held-out neuron in each row
            u_t[batch_range, idx] = u_pred[:, t]

            g = gating[t] if gating is not None else ones          # (N,)
            phi_gated = model.phi(u_t) * g.unsqueeze(0)           # (B, N)

            # Synaptic currents
            I_sv_t = torch.zeros(B, N, device=device)
            I_dcv_t = torch.zeros(B, N, device=device)
            if model.r_sv > 0:
                I_sv_t, s_sv = _batched_synaptic_current(
                    u_t, phi_gated, s_sv, T_sv_eff, a_sv, tau_sv, E_sv, model.dt)
            if model.r_dcv > 0:
                I_dcv_t, s_dcv = _batched_synaptic_current(
                    u_t, phi_gated, s_dcv, T_dcv_eff, a_dcv, tau_dcv, E_dcv, model.dt)

            # Gap junctions:  L @ u_t^T → (N, B) → transpose → (B, N)
            I_gap = u_t @ L.T

            # Stimulus (broadcast — same across batch)
            I_stim = torch.zeros(B, N, device=device)
            if model.d_ell > 0 and stim is not None:
                s_t = stim[t]
                if model.stim_diagonal_only:
                    I_stim = (model.b * s_t).unsqueeze(0).expand(B, -1)
                else:
                    I_stim = (model.b @ s_t).unsqueeze(0).expand(B, -1)

            # Coupling gate
            I_coupling = I_gap + I_sv_t + I_dcv_t
            if has_gate:
                I_coupling = gate_vals * I_coupling

            # Residual MLP correction (K-step context from ground-truth u_all)
            I_mlp = torch.zeros(B, N, device=device)
            if has_mlp:
                if mlp_ctx_inputs is not None:
                    # Each LOO trial shares the same GT-based context
                    # (the held-out neuron is a single entry — negligible
                    # bias vs. re-computing per-trial context)
                    mlp_in = mlp_ctx_inputs[t]            # (N*K,)
                    I_mlp = model.residual_mlp(mlp_in).unsqueeze(0).expand(B, -1)
                else:
                    I_mlp = model.residual_mlp(u_all[t]).unsqueeze(0).expand(B, -1)

            # Linear lag terms (pre-computed from GT trajectory)
            I_lag = torch.zeros(B, N, device=device)
            if lag_all is not None:
                I_lag = lag_all[t].unsqueeze(0).expand(B, -1)

            # Euler step
            u_next = ((1.0 - lambda_u) * u_t
                      + lambda_u * (I0 + I_coupling + I_stim + I_mlp + I_lag))

            if lo is not None or hi is not None:
                u_next = u_next.clamp(min=lo, max=hi)

            # Each trial extracts its own neuron
            u_pred[:, t + 1] = u_next[batch_range, idx]

    # Pack results
    u_pred_np = u_pred.cpu().numpy()
    return {int(held_out_indices[b]): u_pred_np[b] for b in range(B)}


def loo_forward_simulate_batched_windowed(
    model: Stage2ModelPT,
    u_all: torch.Tensor,
    held_out_indices: List[int],
    gating,
    stim,
    window_size: int = 200,
    warmup_steps: int = 0,
) -> Dict[int, np.ndarray]:
    """Batched windowed LOO — periodic re-seeding, all neurons in parallel."""
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

    # Pre-compute constants
    lambda_u = model.lambda_u.detach()
    I0       = model.I0.detach()
    L        = model.laplacian().detach()
    has_sv  = model.r_sv  > 0
    has_dcv = model.r_dcv > 0
    if has_sv:
        T_sv_eff = (model.T_sv * model._get_W("W_sv")).detach()
        a_sv, tau_sv, E_sv = model.a_sv.detach(), model.tau_sv.detach(), model.E_sv.detach()
    if has_dcv:
        T_dcv_eff = (model.T_dcv * model._get_W("W_dcv")).detach()
        a_dcv, tau_dcv, E_dcv = model.a_dcv.detach(), model.tau_dcv.detach(), model.E_dcv.detach()

    # Coupling gate and residual MLP
    has_gate = getattr(model, 'has_coupling_gate', False)
    gate_vals = model.coupling_gate_values.detach() if has_gate else None
    has_mlp = model.residual_mlp is not None
    mlp_ctx_K = getattr(model, '_mlp_context_K', 1)

    # Pre-compute MLP context inputs from u_all: (T, N*K)
    mlp_ctx_inputs = None
    if has_mlp and mlp_ctx_K > 1:
        u_pad = F.pad(u_all, (0, 0, mlp_ctx_K - 1, 0))
        mlp_ctx_inputs = torch.cat(
            [u_pad[mlp_ctx_K - 1 - k : mlp_ctx_K - 1 - k + T] for k in range(mlp_ctx_K)],
            dim=1,
        )

    # Pre-compute linear lag terms from u_all
    lag_order = getattr(model, '_lag_order', 0)
    lag_neighbor = getattr(model, '_lag_neighbor', False)
    lag_all = None  # (T, N)
    if lag_order > 0:
        K_lag = lag_order
        lag_alpha = model._lag_alpha.detach()
        u_lag_pad = F.pad(u_all, (0, 0, K_lag, 0))
        lag_buf = torch.stack(
            [u_lag_pad[K_lag - 1 - k : K_lag - 1 - k + T] for k in range(K_lag)],
            dim=1,
        )
        lag_all = (lag_alpha.unsqueeze(0) * lag_buf).sum(1)
        if lag_neighbor and hasattr(model, '_lag_G'):
            lag_G = model._lag_G.detach()
            mask = model._lag_nbr_mask
            for k in range(K_lag):
                G_k = lag_G[k] * mask
                lag_all = lag_all + (lag_buf[:, k, :] @ G_k.T)

    batch_range = torch.arange(B, device=device)

    with torch.no_grad():
        for w_start in range(0, T, window_size):
            w_end = min(w_start + window_size, T)
            u_pred[:, w_start] = u_all[w_start, idx]

            # Warm-start synaptic states
            s_sv  = torch.zeros(B, N, model.r_sv,  device=device) if has_sv  else None
            s_dcv = torch.zeros(B, N, model.r_dcv, device=device) if has_dcv else None
            if warmup_steps > 0 and w_start > 0:
                burn_start = max(0, w_start - warmup_steps)
                for tb in range(burn_start, w_start):
                    u_gt = u_all[tb].unsqueeze(0).expand(B, -1)
                    g = gating[tb] if gating is not None else ones
                    phi_g = model.phi(u_gt) * g.unsqueeze(0)
                    if has_sv:
                        _, s_sv = _batched_synaptic_current(
                            u_gt, phi_g, s_sv, T_sv_eff, a_sv, tau_sv, E_sv, model.dt)
                    if has_dcv:
                        _, s_dcv = _batched_synaptic_current(
                            u_gt, phi_g, s_dcv, T_dcv_eff, a_dcv, tau_dcv, E_dcv, model.dt)

            for t in range(w_start, w_end - 1):
                u_t = u_all[t].unsqueeze(0).expand(B, -1).clone()
                u_t[batch_range, idx] = u_pred[:, t]

                g = gating[t] if gating is not None else ones
                phi_gated = model.phi(u_t) * g.unsqueeze(0)

                I_sv_t = torch.zeros(B, N, device=device)
                I_dcv_t = torch.zeros(B, N, device=device)
                if has_sv:
                    I_sv_t, s_sv = _batched_synaptic_current(
                        u_t, phi_gated, s_sv, T_sv_eff, a_sv, tau_sv, E_sv, model.dt)
                if has_dcv:
                    I_dcv_t, s_dcv = _batched_synaptic_current(
                        u_t, phi_gated, s_dcv, T_dcv_eff, a_dcv, tau_dcv, E_dcv, model.dt)

                I_gap = u_t @ L.T
                I_stim = torch.zeros(B, N, device=device)
                if model.d_ell > 0 and stim is not None:
                    s_t = stim[t]
                    if model.stim_diagonal_only:
                        I_stim = (model.b * s_t).unsqueeze(0).expand(B, -1)
                    else:
                        I_stim = (model.b @ s_t).unsqueeze(0).expand(B, -1)

                # Coupling gate
                I_coupling = I_gap + I_sv_t + I_dcv_t
                if has_gate:
                    I_coupling = gate_vals * I_coupling

                # Residual MLP correction
                I_mlp = torch.zeros(B, N, device=device)
                if has_mlp:
                    if mlp_ctx_inputs is not None:
                        mlp_in = mlp_ctx_inputs[t]
                        I_mlp = model.residual_mlp(mlp_in).unsqueeze(0).expand(B, -1)
                    else:
                        I_mlp = model.residual_mlp(u_all[t]).unsqueeze(0).expand(B, -1)

                # Linear lag terms (pre-computed from GT trajectory)
                I_lag = torch.zeros(B, N, device=device)
                if lag_all is not None:
                    I_lag = lag_all[t].unsqueeze(0).expand(B, -1)

                u_next = ((1.0 - lambda_u) * u_t
                          + lambda_u * (I0 + I_coupling + I_stim + I_mlp + I_lag))
                if lo is not None or hi is not None:
                    u_next = u_next.clamp(min=lo, max=hi)
                u_pred[:, t + 1] = u_next[batch_range, idx]

    u_pred_np = u_pred.cpu().numpy()
    return {int(held_out_indices[b]): u_pred_np[b] for b in range(B)}


def run_loo_all(
    model: Stage2ModelPT, data: Dict[str, Any],
    subset: Optional[List[int]] = None,
    n_sample_trajectories: int = 0,
    window_size: int = 0,
    warmup_steps: int = 0,
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

    use_windowed = window_size > 0

    labels = data.get("neuron_labels", [])
    _logger = get_stage2_logger()
    _logger.info("loo_start", n_neurons=len(indices),
                 stochastic=do_stochastic, n_samples=n_sample_trajectories,
                 windowed=use_windowed, window_size=window_size)

    # ── Batched deterministic LOO (all neurons in parallel) ──────────
    preds = loo_forward_simulate_batched(model, u, indices, gating, stim,
                                         warmup_steps=warmup_steps)
    _logger.info("loo_batched_done", n_neurons=len(indices), variant="deterministic",
                 warmup_steps=warmup_steps)

    preds_windowed = {}
    if use_windowed:
        preds_windowed = loo_forward_simulate_batched_windowed(
            model, u, indices, gating, stim,
            window_size=window_size, warmup_steps=warmup_steps,
        )
        _logger.info("loo_batched_done", n_neurons=len(indices), variant="windowed")

    # ── Sequential stochastic sampling (depends on deterministic preds) ──
    samples = {}
    if do_stochastic:
        for cnt, i in enumerate(indices):
            samples[i] = loo_forward_simulate_stochastic(
                model, u, i, gating, stim,
                n_samples=n_sample_trajectories,
            )
            lbl = labels[i] if i < len(labels) else f"#{i}"
            if cnt == 0 or (cnt + 1) % 10 == 0:
                _logger.info("loo_stochastic_progress", done=cnt + 1,
                             total=len(indices), neuron=int(i), name=lbl)

    # Log per-neuron R² summary
    for cnt, i in enumerate(indices):
        lbl = labels[i] if i < len(labels) else f"#{i}"
        r2_i = float(_r2(u_np[1:, i], preds[i][1:]))
        if cnt == 0 or (cnt + 1) % 10 == 0 or len(indices) <= 10:
            extra = {}
            if use_windowed:
                extra["r2_windowed"] = float(_r2(u_np[1:, i], preds_windowed[i][1:]))
            _logger.info("loo_progress", done=cnt + 1, total=len(indices),
                         neuron=int(i), name=lbl, r2=r2_i, **extra)

    pred_full = np.column_stack([preds.get(i, u_np[:, i]) for i in range(u.shape[1])])
    r2, corr, rmse = _per_neuron_metrics(u_np[1:], pred_full[1:], indices)

    result = {"pred": preds, "r2": r2, "corr": corr, "rmse": rmse}
    if use_windowed:
        pred_w_full = np.column_stack([preds_windowed.get(i, u_np[:, i]) for i in range(u.shape[1])])
        r2_w, corr_w, rmse_w = _per_neuron_metrics(u_np[1:], pred_w_full[1:], indices)
        result["pred_windowed"] = preds_windowed
        result["r2_windowed"] = r2_w
        result["corr_windowed"] = corr_w
        result["rmse_windowed"] = rmse_w
    if samples:
        result["samples"] = samples
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

    loo      = run_loo_all(
        model, data, subset=subset,
        n_sample_trajectories=_cfg_val(cfg, "n_sample_trajectories", 0, int),
        window_size=_cfg_val(cfg, "eval_loo_window_size", 0, int),
        warmup_steps=_cfg_val(cfg, "eval_loo_warmup_steps", 0, int),
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
        "beh": beh, "beh_all": beh_all,
        "beh_r2_model": r2_model_mean,
    }
