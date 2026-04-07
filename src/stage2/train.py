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
)
from .config import Stage2PTConfig
from .io_h5 import load_data_pt, save_results_pt
from .model import Stage2ModelPT
from .plot_eval import generate_eval_loo_plots

__all__ = [
    "train_stage2",
    "train_stage2_cv",
    # Used by train_multi.py:
    "compute_dynamics_loss",
    "snapshot_model_state",
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
    train_mask: Optional[torch.Tensor] = None,
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

    Parameters
    ----------
    train_mask : optional bool tensor, shape ``(T,)``
        If provided, only include time steps where ``train_mask[t]`` is True
        in the loss.  Used for k-fold CV training where held-out time steps
        must be excluded from the gradient.
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
        if train_mask is not None:
            valid = valid & train_mask.view(-1, 1)
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
    if train_mask is not None:
        valid = valid & train_mask.view(-1, 1)
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

        # Warm-start MLP context history from teacher-forced trajectory
        if hasattr(model, 'init_mlp_history'):
            model.init_mlp_history(u_target, t0)
        # Warm-start lag history from teacher-forced trajectory
        if hasattr(model, 'init_lag_history'):
            model.init_lag_history(u_target, t0)

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
    cached_params: Optional[dict] = None,
) -> dict:
    """Run a teacher-forced pass and cache synaptic states at each time step.

    Returns a dict with:
        s_sv  : (T, N, r_sv)  — synaptic SV states *before* each step
        s_dcv : (T, N, r_dcv) — synaptic DCV states *before* each step

    These can be used to warm-start rollout segments or neuron-dropout
    segments so that the synaptic trace state is realistic rather than zero.

    If *cached_params* is provided (from ``model.precompute_params()``),
    reparameterized parameters are reused instead of recomputed per step.
    """
    T, N = u.shape
    device = u.device
    r_sv, r_dcv = model.r_sv, model.r_dcv

    s_sv_cache = torch.zeros(T, N, r_sv, device=device)
    s_dcv_cache = torch.zeros(T, N, r_dcv, device=device)

    s_sv = torch.zeros(N, r_sv, device=device)
    s_dcv = torch.zeros(N, r_dcv, device=device)

    ones = torch.ones(N, device=device)

    # Reset MLP history so prior_step builds it up from t=0
    if hasattr(model, 'reset_mlp_history'):
        model.reset_mlp_history()
    # Reset lag history so prior_step builds it up from t=0
    if hasattr(model, 'reset_lag_history'):
        model.reset_lag_history()

    with torch.no_grad():
        for t in range(T - 1):
            s_sv_cache[t] = s_sv
            s_dcv_cache[t] = s_dcv
            g = gating_data[t] if gating_data is not None else ones
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

            # Warm-start MLP context history
            if hasattr(model, 'init_mlp_history'):
                model.init_mlp_history(u_target, t0)
            # Warm-start lag history
            if hasattr(model, 'init_lag_history'):
                model.init_lag_history(u_target, t0)

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
        # Low-rank correlated noise factor
        if getattr(model, 'noise_corr_rank', 0) > 0:
            params_final["noise_V"] = model._noise_V.detach().cpu()
        # Config flags
        params_final["linear_chemical_synapses"] = torch.tensor(
            int(model.linear_chemical_synapses))
    return params_final


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

    def isatty(self):
        return False

    def fileno(self):
        return self._stdout.fileno()

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
        f"  learn_W: sv={getattr(cfg, 'learn_W_sv', False)} dcv={getattr(cfg, 'learn_W_dcv', False)}\n"
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
#  K-fold temporal CV for one-step R²                                           #
# --------------------------------------------------------------------------- #

def _make_temporal_folds(T: int, n_folds: int):
    """Partition prediction frames 1..T-1 into *n_folds* contiguous blocks.

    Returns list of (test_start, test_end) with 1-based frame indices,
    i.e. each block covers prediction targets u[test_start : test_end].
    """
    n_frames = T - 1  # frames 1..T-1
    fold_size = n_frames // n_folds
    remainder = n_frames - fold_size * n_folds
    folds = []
    cursor = 1
    for fi in range(n_folds):
        size = fold_size + (1 if fi < remainder else 0)
        folds.append((cursor, cursor + size))
        cursor += size
    return folds


def train_stage2_cv(
    cfg: Stage2PTConfig,
    save_dir: str | None = None,
    show: bool = False,
    warm_start_dir: str | None = None,
) -> Optional[dict]:
    """Train Stage 2 with k-fold temporal CV for fair one-step **and LOO** R².

    For each fold the model is trained from scratch with the held-out
    time steps excluded from the dynamics loss (via ``train_mask``).
    Teacher-forced predictions on the held-out frames are stitched into
    a full ``(T, N)`` array and used to compute cross-validated R².

    After all folds are trained, LOO forward simulations are run per-fold
    on each fold's model, collecting held-out frame predictions.  The
    stitched LOO trajectories give properly cross-validated LOO R².

    The best-fold model (lowest held-out MSE) is kept for free-run /
    current-decomposition evaluation, giving a complete set of metrics
    comparable to the Transformer baseline.
    """
    from .init_from_data import init_lambda_u, init_all_from_data
    from .evaluate import (
        _teacher_forced_prior, compute_onestep,
        loo_forward_simulate, loo_forward_simulate_windowed,
        loo_forward_simulate_batched, loo_forward_simulate_batched_windowed,
        choose_loo_subset,
    )
    from ._utils import _r2, _cfg_val

    n_folds = cfg.cv_folds
    assert n_folds >= 2, f"cv_folds must be >= 2, got {n_folds}"

    # ---- reproducibility seed ------------------------------------------------
    _seed = int(getattr(cfg, "seed", 0) or 0)
    if _seed > 0:
        import random
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Force ALL ops deterministic (catches backward-pass atomics)
        torch.use_deterministic_algorithms(True, warn_only=True)
        import os as _os
        _os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        print(f"[Stage2-CV] Deterministic mode: seed={_seed}")

    # ---- data ----------------------------------------------------------------
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
    stim_data_raw = data.get("stim")
    b_seq = data.get("b")

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _save_run_config(cfg, save_dir)
        _tee = _TeeWriter(Path(save_dir) / "run.log")
        sys.stdout = _tee
    else:
        _tee = None

    try:
        # ---- fold setup ------------------------------------------------------
        folds = _make_temporal_folds(T, n_folds)
        pred_u_full = np.full((T, N), np.nan, dtype=np.float32)
        fold_test_mse = []
        fold_states = []

        print(f"\n{'='*60}")
        print(f"[Stage2-CV] {n_folds}-fold temporal cross-validation")
        print(f"[Stage2-CV] T={T}, N={N}, epochs/fold={cfg.num_epochs}")
        for fi, (s, e) in enumerate(folds):
            print(f"  fold {fi}: test=[{s}, {e})  ({e-s} frames)")
        print(f"{'='*60}\n")

        if cfg.behavior_weight > 0:
            beh_all_baseline = behaviour_all_neurons_prediction(data)
        else:
            beh_all_baseline = None

        # ==== per-fold training loop ==========================================
        for fi, (te_s, te_e) in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"[Stage2-CV] Fold {fi+1}/{n_folds}  "
                  f"test=[{te_s},{te_e})  ({te_e - te_s} frames held out)")
            print(f"{'='*60}")

            # ---- train_mask --------------------------------------------------
            # Shape (T-1,): mask[i] corresponds to prediction target frame i+1.
            train_mask = torch.ones(T - 1, dtype=torch.bool, device=device)
            train_mask[te_s - 1 : te_e - 1] = False
            n_train = int(train_mask.sum())
            print(f"[fold {fi}] training on {n_train}/{T-1} frames")

            # ---- build model -------------------------------------------------
            sign_t = data.get("sign_t")
            lambda_u_init = init_lambda_u(u_stage1, cfg)
            model = Stage2ModelPT(
                N, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
                cfg, device, d_ell=d_ell,
                lambda_u_init=lambda_u_init,
                sign_t=sign_t,
            ).to(device)
            init_all_from_data(model, u_stage1, cfg)

            # ---- warm-start from a previous run (phase 1 → phase 2) ---------
            if warm_start_dir is not None:
                ws_path = Path(warm_start_dir) / f"fold_{fi}_state.pt"
                if ws_path.exists():
                    ws_state = torch.load(ws_path, map_location=device,
                                          weights_only=True)
                    with torch.no_grad():
                        for name, val in ws_state.items():
                            if hasattr(model, name):
                                param = getattr(model, name)
                                if isinstance(param, torch.Tensor) and param.shape == val.shape:
                                    param.copy_(val.to(device))
                    print(f"[fold {fi}] Warm-started from {ws_path}")
                else:
                    print(f"[fold {fi}] No warm-start file found at {ws_path}, "
                          f"training from scratch")

            # ---- snapshot anchors for regularisation -------------------------
            with torch.no_grad():
                lambda_u_raw_init_anchor = model._lambda_u_raw.detach().clone()
                I0_init_anchor = model.I0.detach().clone()
                tau_sv_init_anchor = model.tau_sv.detach().clone()
                tau_dcv_init_anchor = model.tau_dcv.detach().clone()
                # Snapshot the init network strength product for floor penalty
                _init_G_rms = float(model.G.pow(2).mean().sqrt())
                _init_a_sv_rms = float(model.a_sv.pow(2).mean().sqrt())
                _init_strength = _init_G_rms * _init_a_sv_rms

            # ---- optimiser ---------------------------------------------------
            syn_lr_mult = float(getattr(cfg, "synaptic_lr_multiplier", 1.0) or 1.0)
            syn_names = {"_a_sv_raw", "_a_dcv_raw", "_W_sv_raw", "_W_dcv_raw"}
            syn_params = [p for n, p in model.named_parameters()
                          if n in syn_names and p.requires_grad]
            other_params = [p for n, p in model.named_parameters()
                            if n not in syn_names and p.requires_grad]
            param_groups = [{"params": other_params, "lr": cfg.learning_rate}]
            if syn_params:
                param_groups.append(
                    {"params": syn_params, "lr": cfg.learning_rate * syn_lr_mult})
            params = other_params + syn_params
            optimiser = optim.Adam(param_groups)

            z_target = u_stage1.to(device)
            uvar = (u_var_stage1.to(device) if u_var_stage1 is not None
                    else None)
            stim_data = stim_data_raw

            use_uvar = bool(getattr(cfg, "use_u_var_weighting", False))
            uvar_scale = float(getattr(cfg, "u_var_scale", 1.0))
            uvar_floor = float(getattr(cfg, "u_var_floor", 1e-8))

            # ---- hyperparams for regularisation / aux losses -----------------
            grad_clip = float(getattr(cfg, "grad_clip_norm", 0.0) or 0.0)
            rollout_steps = int(getattr(cfg, "rollout_steps", 0) or 0)
            rollout_weight = float(getattr(cfg, "rollout_weight", 0.0) or 0.0)
            rollout_starts = int(getattr(cfg, "rollout_starts", 0) or 0)
            warmstart_rollout = bool(getattr(cfg, "warmstart_rollout", False))
            loo_aux_weight = float(getattr(cfg, "loo_aux_weight", 0.0) or 0.0)
            loo_aux_steps = int(getattr(cfg, "loo_aux_steps", 20) or 20)
            loo_aux_neurons = int(getattr(cfg, "loo_aux_neurons", 4) or 4)
            loo_aux_starts = int(getattr(cfg, "loo_aux_starts", 1) or 1)
            ridge_b = float(getattr(cfg, "ridge_b", 0.0) or 0.0)
            dynamics_l2 = float(getattr(cfg, "dynamics_l2", 0.0) or 0.0)
            I0_reg = float(getattr(cfg, "I0_reg", 0.0) or 0.0)
            lambda_u_reg = float(getattr(cfg, "lambda_u_reg", 0.0) or 0.0)
            tau_reg = float(getattr(cfg, "tau_reg", 0.0) or 0.0)
            coupling_gate_reg = float(getattr(cfg, "coupling_gate_reg", 0.0) or 0.0)
            net_floor_weight = float(getattr(cfg, "network_strength_floor", 0.0) or 0.0)
            net_floor_target = float(getattr(cfg, "network_strength_target", 0.8) or 0.8)
            behavior_weight = float(getattr(cfg, "behavior_weight", 0.0) or 0.0)

            # ---- behaviour decoder (per fold) --------------------------------
            beh_decoder = None
            if behavior_weight > 0.0 and b_seq is not None:
                beh_decoder = init_behaviour_decoder(data)
                if beh_decoder is not None:
                    if beh_decoder["type"] == "mlp":
                        mlp_params = list(beh_decoder["model"].parameters())
                        params.extend(mlp_params)
                        optimiser.add_param_group(
                            {"params": mlp_params, "lr": cfg.learning_rate})
                    else:
                        params.append(beh_decoder["W"])
                        optimiser.add_param_group(
                            {"params": [beh_decoder["W"]], "lr": cfg.learning_rate})

            # ---- try to torch.compile the inner step for speed --------
            try:
                _dynamo = getattr(torch, '_dynamo', None)
                if _dynamo is not None:
                    _dynamo.config.suppress_errors = True
                model._synaptic_current = torch.compile(
                    model._synaptic_current, mode="reduce-overhead",
                    disable=not torch.cuda.is_available())
            except Exception:
                pass  # graceful fallback if compile not supported

            # ---- training loop (same losses/regularisations as train_stage2) -
            for epoch in range(cfg.num_epochs):
                # Re-convolve stimulus with learnable kernel
                if stim_data_raw is not None and model.stim_kernel is not None:
                    stim_data = model.convolve_stimulus(stim_data_raw)
                    data["stim"] = stim_data
                else:
                    stim_data = stim_data_raw

                optimiser.zero_grad()

                # ---- noise injection on teacher-forced inputs --------
                _input_noise = float(getattr(cfg, 'input_noise_sigma', 0.0) or 0.0)
                if _input_noise > 0:
                    z_noisy = z_target + _input_noise * torch.randn_like(z_target)
                else:
                    z_noisy = z_target

                # ---- curriculum rollout scheduling --------
                _curriculum = bool(getattr(cfg, 'rollout_curriculum', False))
                if _curriculum and rollout_weight > 0:
                    _K_start = int(getattr(cfg, 'rollout_K_start', 5))
                    _K_end   = int(getattr(cfg, 'rollout_K_end', 30))
                    _cur_s   = int(getattr(cfg, 'rollout_curriculum_start_epoch', 0))
                    _cur_e   = int(getattr(cfg, 'rollout_curriculum_end_epoch', 100))
                    if epoch < _cur_s:
                        _K_cur = _K_start
                    elif epoch >= _cur_e:
                        _K_cur = _K_end
                    else:
                        frac = (epoch - _cur_s) / max(_cur_e - _cur_s, 1)
                        _K_cur = int(_K_start + frac * (_K_end - _K_start))
                    rollout_steps = _K_cur  # dynamically update for this epoch

                # Use sequential prior_step loop (matches old compute_prior graph)
                prior_mu = model._forward_sequence_fallback(
                    z_noisy, gating_data=gating_data,
                    stim_data=stim_data)

                # Model sigma (learnable noise)
                if model._learn_noise:
                    if model._noise_mode == "heteroscedastic":
                        _model_sigma = model.sigma_at(z_target[:-1])
                    else:
                        _model_sigma = model.sigma_at()
                    use_rollout = (rollout_weight > 0 and rollout_steps > 0)
                    use_loo_aux = loo_aux_weight > 0 and loo_aux_steps > 0
                else:
                    _model_sigma = None
                    use_rollout = (rollout_weight > 0 and rollout_steps > 0
                                   and rollout_starts > 0)
                    use_loo_aux = loo_aux_weight > 0 and loo_aux_steps > 0

                # ---- one-step loss with train_mask ---------------------------
                one_step_loss = compute_dynamics_loss(
                    z_target[1:], prior_mu[1:], sigma_u,
                    u_var=uvar, use_u_var_weighting=use_uvar,
                    u_var_scale=uvar_scale, u_var_floor=uvar_floor,
                    model_sigma=_model_sigma,
                    train_mask=train_mask,
                )

                # ---- auxiliary losses (rollout, LOO) -------------------------
                cached_states = None
                if warmstart_rollout and (use_rollout or use_loo_aux):
                    cached_states = compute_teacher_forced_states(
                        model, z_target,
                        gating_data=gating_data, stim_data=stim_data)

                rollout_loss_val = None
                if use_rollout:
                    rollout_loss_val = compute_rollout_loss(
                        model, z_target, sigma_u,
                        rollout_steps=rollout_steps,
                        rollout_starts=rollout_starts,
                        gating_data=gating_data, stim_data=stim_data,
                        cached_states=(cached_states if warmstart_rollout
                                       else None),
                        use_nll=model._learn_noise)

                loo_loss_val = None
                if use_loo_aux:
                    loo_loss_val = compute_loo_aux_loss(
                        model, z_target, sigma_u,
                        loo_steps=loo_aux_steps,
                        loo_neurons=loo_aux_neurons,
                        loo_starts=loo_aux_starts,
                        gating_data=gating_data, stim_data=stim_data,
                        cached_states=(cached_states if warmstart_rollout
                                       else None),
                        use_nll=model._learn_noise)

                # ---- total loss + regularisation -----------------------------
                loss = one_step_loss
                if rollout_loss_val is not None:
                    loss = loss + rollout_weight * rollout_loss_val
                if loo_loss_val is not None:
                    loss = loss + loo_aux_weight * loo_loss_val
                if ridge_b > 0 and model.d_ell > 0:
                    loss = loss + ridge_b * model.b.pow(2).mean()
                if dynamics_l2 > 0:
                    dyn_l2 = [p.pow(2).mean() for p in model.parameters()
                              if p.requires_grad]
                    if dyn_l2:
                        loss = loss + dynamics_l2 * torch.stack(dyn_l2).mean()
                # Noise regularisation
                _noise_reg = float(getattr(cfg, "noise_reg", 0.0) or 0.0)
                if _noise_reg > 0 and model._learn_noise:
                    if model._noise_mode == "heteroscedastic":
                        loss = loss + _noise_reg * (
                            model._sigma_w.pow(2).mean()
                            + model._sigma_b.pow(2).mean())
                    else:
                        loss = loss + _noise_reg * model._log_sigma_u.pow(2).mean()
                # Targeted regularisers
                if I0_reg > 0:
                    loss = loss + I0_reg * (
                        model.I0 - I0_init_anchor).pow(2).mean()
                if lambda_u_reg > 0:
                    loss = loss + lambda_u_reg * (
                        model._lambda_u_raw
                        - lambda_u_raw_init_anchor).pow(2).mean()
                if tau_reg > 0:
                    _eps = 1e-6
                    if (model.tau_sv.requires_grad
                            and model.tau_sv.numel() > 0):
                        loss = loss + tau_reg * (
                            torch.log(model.tau_sv.clamp(min=_eps))
                            - torch.log(tau_sv_init_anchor.clamp(min=_eps))
                        ).pow(2).mean()
                    if (model.tau_dcv.requires_grad
                            and model.tau_dcv.numel() > 0):
                        loss = loss + tau_reg * (
                            torch.log(model.tau_dcv.clamp(min=_eps))
                            - torch.log(tau_dcv_init_anchor.clamp(min=_eps))
                        ).pow(2).mean()
                # Coupling gate L2 (pull logits toward 0 → gate≈0.5)
                if coupling_gate_reg > 0 and hasattr(model, '_coupling_gate_raw'):
                    loss = loss + coupling_gate_reg * model._coupling_gate_raw.pow(2).mean()
                # Network strength floor (+ optional growth encouragement)
                if net_floor_weight > 0 and net_floor_target > 0:
                    _cur = (model.G.pow(2).mean().sqrt()
                            * model.a_sv.pow(2).mean().sqrt())
                    _tgt = _init_strength * net_floor_target
                    _ratio = _cur / max(_tgt, 1e-12)
                    # Collapse penalty: fires when strength drops below target
                    loss = loss + net_floor_weight * torch.relu(
                        1.0 - _ratio).pow(2)
                    # Growth encouragement: gentle pull toward target (symmetric)
                    # This prevents the network from stalling at a small fraction
                    # of its initial strength.  Weight is 0.1× the floor weight
                    # so it nudges without dominating the loss.
                    loss = loss + 0.1 * net_floor_weight * (
                        1.0 - _ratio).pow(2)
                # Behaviour loss
                beh_loss_val = None
                if beh_decoder is not None and behavior_weight > 0.0:
                    beh_loss_val = compute_behaviour_loss(
                        prior_mu, beh_decoder)
                    loss = loss + behavior_weight * beh_loss_val
                # Behavior weight cap (Test 7: coupling gain ≈0.5-1%, weakest term)
                _beh_cap = float(getattr(cfg, "behavior_weight_cap", 0.0) or 0.0)
                if _beh_cap > 0 and hasattr(model, "b") and model.b.requires_grad:
                    loss = loss + _beh_cap * model.b.abs().mean()
                # Noise correlation factor regularisation
                if hasattr(model, '_noise_V') and model.noise_corr_rank > 0:
                    _noise_V_reg = float(getattr(cfg, "noise_reg", 0.0) or 0.0)
                    if _noise_V_reg > 0:
                        loss = loss + _noise_V_reg * model._noise_V.pow(2).mean()

                apply_training_step(
                    loss, optimiser, params, model, cfg,
                    grad_clip=grad_clip)

                # ---- logging (sparse) ----------------------------------------
                if epoch == 0 or (epoch + 1) == cfg.num_epochs or (epoch + 1) % 10 == 0:
                    parts = [f"dyn={one_step_loss.item():.4f}",
                             f"total={loss.item():.4f}"]
                    if rollout_loss_val is not None:
                        parts.append(f"roll={rollout_loss_val.item():.4f}")
                        if _curriculum:
                            parts.append(f"K={rollout_steps}")
                    if loo_loss_val is not None:
                        parts.append(f"loo={loo_loss_val.item():.4f}")
                    if beh_loss_val is not None:
                        parts.append(f"beh={beh_loss_val.item():.4f}")
                    if _input_noise > 0:
                        parts.append(f"noise={_input_noise:.3f}")
                    print(f"[fold {fi}] Epoch {epoch+1}/{cfg.num_epochs}: "
                          f"{'  '.join(parts)}")

            # ---- collect held-out predictions --------------------------------
            model.eval()
            with torch.no_grad():
                prior_mu_full = _teacher_forced_prior(
                    model,
                    u_stage1.to(device),
                    gating_data,
                    stim_data if stim_data is not None else stim_data_raw,
                )
            pred_np = prior_mu_full.cpu().numpy()
            pred_u_full[te_s:te_e] = pred_np[te_s:te_e]

            # Held-out MSE for model selection
            u_np = u_stage1.cpu().numpy()
            test_mse = float(np.mean(
                (u_np[te_s:te_e] - pred_np[te_s:te_e]) ** 2))
            fold_test_mse.append(test_mse)
            fold_states.append(snapshot_model_state(model))

            # Save per-fold state for warm-starting (phase 1 → phase 2)
            if save_dir is not None:
                _fold_pt = Path(save_dir) / f"fold_{fi}_state.pt"
                torch.save(fold_states[-1], _fold_pt)

            # Per-fold R² on held-out frames
            _fold_r2 = np.array([
                _r2(u_np[te_s:te_e, i], pred_np[te_s:te_e, i])
                for i in range(N)])
            print(f"[fold {fi}] held-out MSE={test_mse:.6f}  "
                  f"R² mean={np.nanmean(_fold_r2):.4f}  "
                  f"median={np.nanmedian(_fold_r2):.4f}")

        # ==== aggregate CV results ============================================
        valid_mask = ~np.isnan(pred_u_full[:, 0])
        u_np = u_stage1.cpu().numpy()
        cv_r2 = np.array([
            _r2(u_np[valid_mask, i], pred_u_full[valid_mask, i])
            for i in range(N)])

        print(f"\n{'='*60}")
        print(f"[Stage2-CV] Cross-validated one-step R² ({n_folds} folds)")
        print(f"  mean  = {np.nanmean(cv_r2):.4f}")
        print(f"  median= {np.nanmedian(cv_r2):.4f}")
        print(f"  std   = {np.nanstd(cv_r2):.4f}")
        print(f"  min   = {np.nanmin(cv_r2):.4f}")
        print(f"  max   = {np.nanmax(cv_r2):.4f}")
        n_neg = int((cv_r2 < 0).sum())
        print(f"  #negative = {n_neg}/{N}")
        print(f"{'='*60}")

        # ---- rebuild best-fold model for final evaluation --------------------
        best_fi = int(np.argmin(fold_test_mse))
        print(f"\n[Stage2-CV] Best fold = {best_fi} "
              f"(held-out MSE = {fold_test_mse[best_fi]:.6f})")

        lambda_u_init = init_lambda_u(u_stage1, cfg)

        def _rebuild_model_from_state(state_dict):
            """Rebuild a Stage2ModelPT and load *state_dict* parameters."""
            m = Stage2ModelPT(
                N, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
                cfg, device, d_ell=d_ell,
                lambda_u_init=lambda_u_init,
                sign_t=data.get("sign_t"),
            ).to(device)
            init_all_from_data(m, u_stage1, cfg)
            with torch.no_grad():
                for name, val in state_dict.items():
                    if hasattr(m, name):
                        param = getattr(m, name)
                        if isinstance(param, torch.Tensor) and param.shape == val.shape:
                            param.copy_(val)
            m.eval()
            return m

        best_model = _rebuild_model_from_state(fold_states[best_fi])

        # ==== Cross-validated LOO evaluation ==================================
        _cv_onestep_for_subset = {"r2": cv_r2, "prior_mu": pred_u_full}
        subset = choose_loo_subset(
            data, _cv_onestep_for_subset,
            subset_size=_cfg_val(cfg, "eval_loo_subset_size", 0, int),
            subset_mode=str(getattr(cfg, "eval_loo_subset_mode", "variance")),
            explicit_indices=(
                None if getattr(cfg, "eval_loo_subset_neurons", None) is None
                else list(cfg.eval_loo_subset_neurons)
            ),
            seed=_cfg_val(cfg, "eval_loo_subset_seed", 0, int),
        )

        cv_loo_r2 = None
        cv_loo_r2_windowed = None
        cv_loo_preds = None

        if subset is not None and len(subset) > 0:
            window_size = _cfg_val(cfg, "eval_loo_window_size", 0, int)
            warmup_steps = _cfg_val(cfg, "eval_loo_warmup_steps", 40, int)
            u_device = u_stage1.to(device)
            gating_eval = data.get("gating")
            stim_eval = data.get("stim")

            # Allocate per-neuron prediction arrays (NaN = not yet filled)
            loo_pred_full = {i: np.full(T, np.nan, dtype=np.float32)
                            for i in subset}
            loo_pred_w_full = (
                {i: np.full(T, np.nan, dtype=np.float32) for i in subset}
                if window_size > 0 else None
            )

            print(f"\n{'='*60}")
            print(f"[Stage2-CV] Cross-validated LOO ({n_folds} folds, "
                  f"{len(subset)} neurons) — BATCHED")
            print(f"{'='*60}")

            for fi, (te_s, te_e) in enumerate(folds):
                fold_model = _rebuild_model_from_state(fold_states[fi])

                # Batched: all LOO neurons in one pass (~N× faster)
                batch_preds = loo_forward_simulate_batched(
                    fold_model, u_device, subset, gating_eval, stim_eval,
                    warmup_steps=warmup_steps)
                for i in subset:
                    loo_pred_full[i][te_s:te_e] = batch_preds[i][te_s:te_e]

                if window_size > 0:
                    batch_preds_w = loo_forward_simulate_batched_windowed(
                        fold_model, u_device, subset, gating_eval, stim_eval,
                        window_size=window_size, warmup_steps=warmup_steps)
                    for i in subset:
                        loo_pred_w_full[i][te_s:te_e] = batch_preds_w[i][te_s:te_e]

                print(f"[Stage2-CV] LOO fold {fi+1}/{n_folds} done "
                      f"(test=[{te_s},{te_e}), {len(subset)} neurons)")

            # Compute CV LOO R²
            cv_loo_r2 = np.full(N, np.nan, dtype=np.float32)
            for i in subset:
                valid = ~np.isnan(loo_pred_full[i])
                if valid.sum() > 1:
                    cv_loo_r2[i] = _r2(u_np[valid, i], loo_pred_full[i][valid])

            cv_loo_preds = loo_pred_full
            loo_valid = cv_loo_r2[np.isfinite(cv_loo_r2)]
            print(f"\n{'='*60}")
            print(f"[Stage2-CV] Cross-validated LOO R² ({n_folds} folds)")
            if len(loo_valid) > 0:
                print(f"  mean  = {np.nanmean(loo_valid):.4f}")
                print(f"  median= {np.nanmedian(loo_valid):.4f}")
                print(f"  std   = {np.nanstd(loo_valid):.4f}")
                print(f"  min   = {np.nanmin(loo_valid):.4f}")
                print(f"  max   = {np.nanmax(loo_valid):.4f}")
                n_neg_loo = int((loo_valid < 0).sum())
                print(f"  #negative = {n_neg_loo}/{len(loo_valid)}")
            else:
                print(f"  (no valid LOO neurons)")
            print(f"{'='*60}")

            if window_size > 0 and loo_pred_w_full is not None:
                cv_loo_r2_windowed = np.full(N, np.nan, dtype=np.float32)
                for i in subset:
                    valid = ~np.isnan(loo_pred_w_full[i])
                    if valid.sum() > 1:
                        cv_loo_r2_windowed[i] = _r2(
                            u_np[valid, i], loo_pred_w_full[i][valid])
                loo_w_valid = cv_loo_r2_windowed[
                    np.isfinite(cv_loo_r2_windowed)]
                if len(loo_w_valid) > 0:
                    print(f"[Stage2-CV] Windowed LOO R² (window={window_size})")
                    print(f"  mean  = {np.nanmean(loo_w_valid):.4f}")
                    print(f"  median= {np.nanmedian(loo_w_valid):.4f}")

        # ---- save results ----------------------------------------------------
        results_h5 = (None if save_dir is None
                      else str(Path(save_dir) / "stage2_results.h5"))
        params_final = fold_states[best_fi]
        save_results_pt(cfg, u_stage1.to(device).detach(),
                        params_final, output_path=results_h5)

        # Save CV predictions & metrics
        if save_dir is not None:
            cv_path = Path(save_dir) / "cv_onestep.npz"
            save_data = dict(
                pred_u_full=pred_u_full,
                cv_r2=cv_r2,
                fold_test_mse=np.array(fold_test_mse),
                folds=np.array(folds),
                best_fold=best_fi,
            )
            if cv_loo_r2 is not None:
                save_data["cv_loo_r2"] = cv_loo_r2
                for i, pred in cv_loo_preds.items():
                    save_data[f"loo_pred_{i}"] = pred
            if cv_loo_r2_windowed is not None:
                save_data["cv_loo_r2_windowed"] = cv_loo_r2_windowed
            np.savez(cv_path, **save_data)
            print(f"[Stage2-CV] CV predictions saved to {cv_path}")

        # ---- final evaluation with best-fold model ---------------------------
        eval_result = None
        _skip_final = bool(getattr(cfg, "skip_final_eval", False))
        if (save_dir or show) and not _skip_final:
            eval_result = generate_eval_loo_plots(
                model=best_model, data=data, cfg=cfg,
                epoch_losses=[],
                save_dir=save_dir or "plots/eval_cv", show=show,
                decoder=None,
                beh_all_baseline=beh_all_baseline,
            )

        # Inject CV metrics into eval_result
        if eval_result is None:
            eval_result = {}
        eval_result["cv_onestep_r2"] = cv_r2
        eval_result["cv_onestep_r2_mean"] = float(np.nanmean(cv_r2))
        eval_result["cv_onestep_r2_median"] = float(np.nanmedian(cv_r2))
        eval_result["cv_pred_u_full"] = pred_u_full
        eval_result["fold_test_mse"] = fold_test_mse
        eval_result["best_fold_idx"] = best_fi
        if cv_loo_r2 is not None:
            loo_valid = cv_loo_r2[np.isfinite(cv_loo_r2)]
            eval_result["cv_loo_r2"] = cv_loo_r2
            eval_result["cv_loo_r2_mean"] = float(np.nanmean(loo_valid)) if len(loo_valid) > 0 else float("nan")
            eval_result["cv_loo_r2_median"] = float(np.nanmedian(loo_valid)) if len(loo_valid) > 0 else float("nan")
        if cv_loo_r2_windowed is not None:
            loo_w_valid = cv_loo_r2_windowed[np.isfinite(cv_loo_r2_windowed)]
            eval_result["cv_loo_r2_windowed"] = cv_loo_r2_windowed
            eval_result["cv_loo_r2_windowed_mean"] = float(np.nanmean(loo_w_valid)) if len(loo_w_valid) > 0 else float("nan")
            eval_result["cv_loo_r2_windowed_median"] = float(np.nanmedian(loo_w_valid)) if len(loo_w_valid) > 0 else float("nan")

        # Coupling gate diagnostics (if enabled)
        if hasattr(best_model, 'has_coupling_gate') and best_model.has_coupling_gate:
            eval_result["coupling_gate_values"] = best_model.coupling_gate_values.detach().cpu().numpy()

        return eval_result

    finally:
        if _tee is not None:
            _tee.close()


# --------------------------------------------------------------------------- #
#  Main entry point                                                             #
# --------------------------------------------------------------------------- #


def train_stage2(
    cfg: Stage2PTConfig,
    save_dir: str | None = None,
    show: bool = False,
    warm_start_dir: str | None = None,
) -> Optional[dict]:
    """Train Stage 2 with k-fold temporal CV for fair one-step and LOO R².

    This is the main entry point.  It always uses temporal cross-validation
    (``cfg.cv_folds`` folds, default 5) to produce out-of-sample one-step
    and LOO R² metrics, matching the evaluation protocol of the Transformer
    baseline.
    """
    return train_stage2_cv(cfg, save_dir=save_dir, show=show,
                           warm_start_dir=warm_start_dir)
