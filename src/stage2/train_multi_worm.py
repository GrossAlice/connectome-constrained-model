"""Multi-worm Stage 2 training with shared atlas-space dynamics.

Architecture
------------
**Shared** (species-level, learned from all worms):
    G, W_sv, W_dcv, a_sv, a_dcv, tau_sv, tau_dcv, E_sv, E_dcv

**Per-worm** (animal-level, ~2N params each):
    lambda_u(N_atlas,), I0(N_atlas,)

The model operates in the full N_atlas (302) space, but each worm's loss
is masked to its *observed* neurons.  This way:
  - The same AVAL→AVAR weight gets gradient from every worm that observes
    both neurons.
  - Per-worm I0 absorbs the mean drive from unobserved presynaptic neurons.
  - No latent inference is needed (Phase 1: observed-only).

Training loop
-------------
Each epoch iterates over all worms.  Per worm:
  1. Teacher-forced one-step predictions on the full atlas.
  2. Dynamics loss masked to that worm's observed neurons + observed time.
  3. Regularisation (init-anchored, strength floor, etc.).
  4. Shared params get gradient from every worm; per-worm params from theirs.

LOO evaluation is run per-worm at the end, using the shared model with
that worm's per-worm params.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import Stage2PTConfig
from .io_multi import load_multi_worm_data
from .model import Stage2ModelPT, _reparam_inv
from .init_from_data import (
    init_lambda_u, init_I0, init_reversals, init_network_scale,
    init_W_from_config, init_G_from_config,
)
from .train import (
    compute_dynamics_loss,
    compute_teacher_forced_states,
    compute_rollout_loss,
    compute_loo_aux_loss,
    snapshot_model_state,
    _TeeWriter,
    _config_to_dict,
    _make_temporal_folds,
)
from .evaluate import (
    _teacher_forced_prior,
    compute_onestep,
    loo_forward_simulate_batched_windowed,
    choose_loo_subset,
    run_loo_all,
)
from ._utils import _r2

__all__ = ["train_multi_worm"]


# ─── Helpers ────────────────────────────────────────────────────────────────


def _build_atlas_model(
    multi_data: Dict[str, Any],
    cfg: Stage2PTConfig,
    device: torch.device,
) -> Stage2ModelPT:
    """Build a single Stage2ModelPT in atlas space (N = N_atlas)."""
    N_atlas = multi_data["atlas_size"]
    T_e = multi_data["T_e"]
    T_sv = multi_data["T_sv"]
    T_dcv = multi_data["T_dcv"]
    sign_t = multi_data.get("sign_t")
    dt = float(cfg.common_dt)

    # Population-average lambda_u init from all worms
    lam_inits = []
    for w in multi_data["worms"]:
        lam_inits.append(w["lambda_u_init"])
    lam_avg = torch.stack(lam_inits).mean(dim=0)

    model = Stage2ModelPT(
        N=N_atlas,
        T_e=T_e,
        T_sv=T_sv,
        T_dcv=T_dcv,
        dt=dt,
        cfg=cfg,
        device=device,
        d_ell=0,  # no per-neuron stimulus in multi-worm mode
        lambda_u_init=lam_avg,
        sign_t=sign_t,
    ).to(device)

    return model


def _init_shared_from_population(
    model: Stage2ModelPT,
    worms: List[Dict[str, Any]],
    cfg: Stage2PTConfig,
) -> None:
    """Initialise shared model params from population-averaged currents."""
    device = model.I0.device
    N = model.N

    # Population-average I0 from per-worm OLS
    I0_sum = torch.zeros(N, device=device)
    I0_count = torch.zeros(N, device=device)
    for w in worms:
        u = w["u"]                          # (T, N_atlas)
        obs = w["obs_mask"]                 # (N_atlas,) bool
        lam = model.lambda_u.detach()
        x, y = u[:-1], u[1:]
        b = y.mean(0) - (1 - lam) * x.mean(0)
        I0_worm = b / lam.clamp(min=0.01)
        I0_sum[obs] += I0_worm[obs]
        I0_count[obs] += 1.0
    I0_avg = torch.where(I0_count > 0, I0_sum / I0_count.clamp(min=1),
                         torch.zeros_like(I0_sum))
    with torch.no_grad():
        model.I0.data.copy_(I0_avg)
    print(f"[init] I0 (population): mean={I0_avg.mean():.4f}, "
          f"min={I0_avg.min():.4f}, max={I0_avg.max():.4f}")

    # Use the first worm's u for reversal/network-scale init (representative)
    # Actually, use the worm with most observed neurons for best coverage
    best_worm = max(worms, key=lambda w: w["N_obs"])
    u_best = best_worm["u"]
    init_reversals(model, u_best, I0_avg, cfg)

    # W / G structure init
    init_W_from_config(model, cfg)
    init_G_from_config(model, cfg)

    # Network scale from population-average residuals
    # Use population-averaged teacher-forced residuals for OLS scale
    init_network_scale(model, u_best, cfg)


class PerWormParams(nn.Module):
    """Per-worm learnable parameters: lambda_u and I0."""

    def __init__(self, N: int, lambda_u_init: torch.Tensor,
                 I0_init: torch.Tensor, cfg, device: torch.device):
        super().__init__()
        lo = float(getattr(cfg, "lambda_u_lo", 0.0))
        hi = float(getattr(cfg, "lambda_u_hi", 0.9999))
        self._lambda_u_lo = lo
        self._lambda_u_hi = hi
        self._lambda_u_raw = nn.Parameter(
            _reparam_inv(lambda_u_init.to(device), lo, hi),
            requires_grad=bool(getattr(cfg, "learn_lambda_u", True)),
        )
        self.I0 = nn.Parameter(
            I0_init.to(device).clone(),
            requires_grad=bool(getattr(cfg, "learn_I0", True)),
        )

    @property
    def lambda_u(self):
        from .model import _reparam_fwd
        return _reparam_fwd(self._lambda_u_raw, self._lambda_u_lo,
                            self._lambda_u_hi)


def _init_per_worm_params(
    model: Stage2ModelPT,
    worms: List[Dict[str, Any]],
    cfg: Stage2PTConfig,
    device: torch.device,
) -> List[PerWormParams]:
    """Create per-worm lambda_u and I0 initialised from each worm's data."""
    per_worm = []
    for w in worms:
        u = w["u"]           # (T, N_atlas) — zeros for unobs
        lam_init = w["lambda_u_init"].to(device)

        # Per-worm I0 via OLS
        lam = lam_init.clone()
        x, y = u[:-1].to(device), u[1:].to(device)
        b = y.mean(0) - (1 - lam) * x.mean(0)
        I0_init = b / lam.clamp(min=0.01)
        # For unobserved neurons, use the shared model's I0
        obs = w["obs_mask"].to(device)
        I0_init = torch.where(obs, I0_init, model.I0.detach())

        pw = PerWormParams(model.N, lam_init, I0_init, cfg, device)
        per_worm.append(pw)
    return per_worm


# ─── Teacher-forced forward pass (masked to observed) ──────────────────────


def _compute_prior_masked(
    model: Stage2ModelPT,
    u: torch.Tensor,
    obs_mask: torch.Tensor,
    gating: torch.Tensor,
    lambda_u: torch.Tensor,
    I0: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced one-step predictions, returns (prior_mu, obs_loss_mask).

    The model runs on the full atlas, but the loss should be computed only
    on observed neurons.  We return a (T-1, N) bool mask for this.
    """
    T, N = u.shape
    device = u.device
    ones = torch.ones(N, device=device)
    prior_mu = torch.zeros_like(u)
    prior_mu[0] = u[0]

    s_sv = torch.zeros(N, model.r_sv, device=device)
    s_dcv = torch.zeros(N, model.r_dcv, device=device)

    for t in range(1, T):
        g = gating[t - 1] if gating is not None else ones
        u_next, s_sv, s_dcv = model.prior_step(
            u[t - 1], s_sv, s_dcv, g, stim=None,
            lambda_u=lambda_u, I0=I0,
        )
        prior_mu[t] = u_next

    # Observation mask: (N,) → expand to (T-1, N)
    obs_loss_mask = obs_mask.unsqueeze(0).expand(T - 1, -1)  # bool
    return prior_mu, obs_loss_mask


# ─── Main training entry point ─────────────────────────────────────────────


def train_multi_worm(
    cfg: Stage2PTConfig,
    save_dir: str | None = None,
    show: bool = False,
) -> Optional[Dict[str, Any]]:
    """Train Stage 2 with shared dynamics across multiple worms.

    Parameters
    ----------
    cfg : Stage2PTConfig
        Must have ``multi.multi_worm = True`` and ``multi.h5_paths`` set.
    save_dir : str, optional
        Directory for logs, checkpoints, plots.
    show : bool
        Whether to display matplotlib figures.

    Returns
    -------
    dict with trained model, per-worm params, and evaluation results.
    """
    device = torch.device(cfg.device)

    # ── Logging ─────────────────────────────────────────────────────────
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "run_config.json", "w") as f:
            json.dump(_config_to_dict(cfg), f, indent=2, default=str)
        tee = _TeeWriter(Path(save_dir) / "run.log")
        sys.stdout = tee
    else:
        tee = None

    try:
        return _train_multi_worm_inner(cfg, device, save_dir, show)
    finally:
        if tee is not None:
            tee.close()


def _train_multi_worm_inner(
    cfg: Stage2PTConfig,
    device: torch.device,
    save_dir: Optional[str],
    show: bool,
) -> Dict[str, Any]:
    t0_total = time.time()

    # ── Load data ───────────────────────────────────────────────────────
    print("=" * 70)
    print("[MultiWorm] Loading worm data...")
    multi_data = load_multi_worm_data(cfg)
    worms = multi_data["worms"]
    N_atlas = multi_data["atlas_size"]
    W = len(worms)
    print(f"[MultiWorm] {W} worms, {N_atlas}-neuron atlas")

    # ── Build shared model ──────────────────────────────────────────────
    print("[MultiWorm] Building atlas-space model...")
    model = _build_atlas_model(multi_data, cfg, device)
    _init_shared_from_population(model, worms, cfg)

    # ── Per-worm parameters ─────────────────────────────────────────────
    print("[MultiWorm] Initialising per-worm params...")
    per_worm_params = _init_per_worm_params(model, worms, cfg, device)


    # ── Optimiser ───────────────────────────────────────────────────────
    syn_lr_mult = float(getattr(cfg, "synaptic_lr_multiplier", 1.0) or 1.0)
    syn_names = {"_a_sv_raw", "_a_dcv_raw", "_W_sv_raw", "_W_dcv_raw"}

    # Shared params
    syn_params = [p for n, p in model.named_parameters()
                  if n in syn_names and p.requires_grad]
    other_shared = [p for n, p in model.named_parameters()
                    if n not in syn_names and p.requires_grad]

    # Per-worm params (all into one group)
    pw_params = []
    for pw in per_worm_params:
        pw_params.extend(p for p in pw.parameters() if p.requires_grad)

    param_groups = [
        {"params": other_shared, "lr": cfg.learning_rate},
        {"params": pw_params, "lr": cfg.learning_rate},
    ]
    if syn_params:
        param_groups.append(
            {"params": syn_params, "lr": cfg.learning_rate * syn_lr_mult})

    all_params = other_shared + pw_params + syn_params
    optimizer = optim.Adam(param_groups)

    # ── Hyperparameters ─────────────────────────────────────────────────
    num_epochs = cfg.num_epochs
    grad_clip = float(getattr(cfg, "grad_clip_norm", 1.0) or 1.0)
    dynamics_l2 = float(getattr(cfg, "dynamics_l2", 0.0) or 0.0)

    n_total_params = sum(p.numel() for p in all_params if p.requires_grad)
    n_shared = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_per_worm_total = sum(p.numel() for p in pw_params if p.requires_grad)
    print(f"[MultiWorm] Trainable params: {n_total_params:,} "
          f"(shared={n_shared:,}, per-worm={n_per_worm_total:,})")
    print(f"[MultiWorm] Epochs={num_epochs}, lr={cfg.learning_rate}, "
          f"syn_lr×{syn_lr_mult}")
    print("=" * 70)

    # ── Training loop ───────────────────────────────────────────────────
    # Memory-efficient: process one worm at a time, backward immediately,
    # accumulate gradients, then optimizer.step() once per epoch.
    history: List[Dict[str, float]] = []

    for epoch in range(num_epochs):
        t0_ep = time.time()
        optimizer.zero_grad()

        epoch_dyn = 0.0

        # Shuffle worm order each epoch
        order = torch.randperm(W).tolist()

        for wi in order:
            w = worms[wi]
            pw = per_worm_params[wi]
            u = w["u"]                           # already on device
            sigma_u = w["sigma_u"]
            obs_mask = w["obs_mask"]
            gating = w.get("gating")
            weight = w.get("weight", 1.0 / W)

            # Teacher-forced one-step
            prior_mu, obs_loss_mask = _compute_prior_masked(
                model, u, obs_mask, gating,
                lambda_u=pw.lambda_u,
                I0=pw.I0,
            )

            # Variance-weighted MSE on observed neurons only
            target = u[1:]
            pred = prior_mu[1:]
            sigma2 = (sigma_u ** 2).unsqueeze(0).expand_as(target)
            resid2 = (target - pred) ** 2
            valid = (torch.isfinite(resid2)
                     & (sigma2 > 0)
                     & obs_loss_mask)
            if valid.any():
                weighted = resid2[valid] / sigma2[valid]
                dyn_loss = weighted.mean()
            else:
                dyn_loss = torch.tensor(0.0, device=device)

            # Backward per-worm (gradient accumulation)
            (weight * dyn_loss).backward()
            epoch_dyn += weight * dyn_loss.item()

        # ── Shared regularisation (added after per-worm gradient accum) ──
        reg_loss = torch.tensor(0.0, device=device)

        if dynamics_l2 > 0:
            dl2 = sum(p.pow(2).mean() for p in model.parameters()
                      if p.requires_grad)
            reg_loss = reg_loss + dynamics_l2 * dl2

        if reg_loss.requires_grad:
            reg_loss.backward()

        # ── Clip + step ─────────────────────────────────────────────
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
        optimizer.step()

        dt_ep = time.time() - t0_ep
        epoch_total = epoch_dyn + (reg_loss.item() if reg_loss.requires_grad
                                   else 0.0)
        history.append({"epoch": epoch + 1, "loss": epoch_total,
                        "dyn": epoch_dyn, "time": dt_ep})

        # ── Logging ─────────────────────────────────────────────────
        if epoch == 0 or (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            G_rms = float(model.G.pow(2).mean().sqrt())
            a_sv_rms = float(model.a_sv.pow(2).mean().sqrt())
            print(f"[Epoch {epoch+1:3d}/{num_epochs}] "
                  f"loss={epoch_total:.4f}  dyn={epoch_dyn:.4f}  "
                  f"G_rms={G_rms:.4f}  a_sv_rms={a_sv_rms:.4f}  "
                  f"time={dt_ep:.1f}s")

    total_time = time.time() - t0_total
    print(f"\n[MultiWorm] Training complete in {total_time/60:.1f} min")

    # ── Save checkpoint ─────────────────────────────────────────────────
    if save_dir is not None:
        ckpt = {
            "model_state": model.state_dict(),
            "per_worm": [pw.state_dict() for pw in per_worm_params],
            "cfg": _config_to_dict(cfg),
            "atlas_labels": multi_data["atlas_labels"],
            "worm_ids": [w["worm_id"] for w in worms],
        }
        torch.save(ckpt, Path(save_dir) / "checkpoint.pt")
        print(f"[MultiWorm] Checkpoint saved to {save_dir}/checkpoint.pt")

    # ── Per-worm evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[MultiWorm] Per-worm evaluation...")

    results_all = []
    for wi, w in enumerate(worms):
        pw = per_worm_params[wi]
        u = w["u"].to(device)
        obs_mask = w["obs_mask"].to(device)
        obs_idx = w["obs_idx"].cpu().numpy()
        gating = w.get("gating")
        sigma_u = w["sigma_u"].to(device)
        labels = multi_data["atlas_labels"]

        # One-step R² on observed neurons
        prior_mu, _ = _compute_prior_masked(
            model, u, obs_mask, gating,
            lambda_u=pw.lambda_u, I0=pw.I0,
        )
        u_np = u.cpu().numpy()
        pred_np = prior_mu.detach().cpu().numpy()
        r2_1step = np.array([
            _r2(u_np[1:, j], pred_np[1:, j])
            for j in obs_idx
        ])
        med_1step = float(np.nanmedian(r2_1step))

        # LOO on AVAL/AVAR if present
        loo_subset = []
        for name in ("AVAL", "AVAR"):
            if name in labels:
                idx = labels.index(name)
                if obs_mask[idx]:
                    loo_subset.append(idx)

        med_loo = float("nan")
        if len(loo_subset) >= 1:
            # Temporarily swap model's per-neuron params for this worm
            orig_lambda = model._lambda_u_raw.data.clone()
            orig_I0 = model.I0.data.clone()
            with torch.no_grad():
                model._lambda_u_raw.data.copy_(pw._lambda_u_raw.data)
                model.I0.data.copy_(pw.I0.data)

            try:
                window_size = int(getattr(cfg, "eval_loo_window_size", 50) or 50)
                warmup_steps = int(getattr(cfg, "eval_loo_warmup_steps", 40) or 40)
                loo_preds = loo_forward_simulate_batched_windowed(
                    model, u, loo_subset, gating, w.get("stim"),
                    window_size=window_size, warmup_steps=warmup_steps,
                )
                r2_loo = [float(_r2(u_np[1:, j], loo_preds[j][1:]))
                          for j in loo_subset]
                med_loo = float(np.nanmedian(r2_loo))
            finally:
                # Restore shared params
                with torch.no_grad():
                    model._lambda_u_raw.data.copy_(orig_lambda)
                    model.I0.data.copy_(orig_I0)

        worm_result = {
            "worm_id": w["worm_id"],
            "N_obs": w["N_obs"],
            "T": w["T"],
            "r2_1step_median": med_1step,
            "r2_loo_median": med_loo,
        }
        results_all.append(worm_result)

        if wi < 5 or (wi + 1) % 10 == 0 or (wi + 1) == W:
            loo_str = f"{med_loo:.3f}" if np.isfinite(med_loo) else "n/a"
            print(f"  [{wi+1:2d}/{W}] {w['worm_id']:20s}  "
                  f"N={w['N_obs']:3d}  "
                  f"1step_med={med_1step:.3f}  LOO_med={loo_str}")

    # ── Summary ─────────────────────────────────────────────────────────
    all_1step = [r["r2_1step_median"] for r in results_all]
    all_loo = [r["r2_loo_median"] for r in results_all
               if np.isfinite(r["r2_loo_median"])]

    print(f"\n{'='*70}")
    print(f"[MultiWorm] Summary across {W} worms:")
    print(f"  1-step R² median-of-medians: {np.nanmedian(all_1step):.4f} "
          f"(mean={np.nanmean(all_1step):.4f})")
    if all_loo:
        print(f"  LOO R² median-of-medians:    {np.nanmedian(all_loo):.4f} "
              f"(mean={np.nanmean(all_loo):.4f})")
    print(f"  Total time: {(time.time()-t0_total)/60:.1f} min")
    print(f"{'='*70}")

    # Save CSV
    if save_dir is not None:
        import csv
        csv_path = Path(save_dir) / "per_worm_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_all[0].keys())
            writer.writeheader()
            writer.writerows(results_all)
        print(f"[MultiWorm] Results saved to {csv_path}")

    return {
        "model": model,
        "per_worm_params": per_worm_params,
        "per_worm_results": results_all,
        "history": history,
        "multi_data": multi_data,
    }
