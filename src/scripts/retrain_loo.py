#!/usr/bin/env python
"""
Retrain-LOO (gold standard):  for each LOO neuron, retrain Stage2 from scratch
with that neuron *excluded from the training loss*, then evaluate whether the
model can reconstruct the neuron purely from connectome coupling.

This eliminates the distribution-shift concern of standard LOO (where the model
was trained to predict *all* neurons including the held-out one).

Produces a comparison plot vs:
  • standard LOO (from saved npz)
  • baselines (Ridge, EN, MLP, Conn-Ridge, Conn-MLP, Conn-Trf) — all K-context

Usage
─────
    TORCHDYNAMO_DISABLE=1 python -u -m scripts.retrain_loo \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
        --save_dir output_plots/stage2/retrain_loo \
        --epochs 30 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import h5py
from sklearn.linear_model import Ridge, ElasticNet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stage2.config import make_config
from stage2.io_h5 import load_data_pt
from stage2.model import Stage2ModelPT
from stage2.init_from_data import init_lambda_u, init_all_from_data
from stage2.evaluate import (
    loo_forward_simulate_batched_windowed,
    choose_loo_subset,
)


def loo_forward_simulate_windowed(model, u_all, held_out, gating, stim,
                                  window_size=0, warmup_steps=0):
    """Thin wrapper: single-neuron LOO via the batched function."""
    result = loo_forward_simulate_batched_windowed(
        model, u_all, [held_out], gating, stim,
        window_size=window_size, warmup_steps=warmup_steps,
    )
    return result[held_out]
from stage2.train import compute_dynamics_loss, snapshot_model_state


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _r2(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m].astype(np.float64), y_pred[m].astype(np.float64)
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-15))


class JointMLP(nn.Module):
    """K*N → N joint MLP baseline."""

    def __init__(self, d_in: int, d_out: int,
                 hidden: Tuple[int, ...] = (256,),
                 dropout: float = 0.0):
        super().__init__()
        layers: list = []
        d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _teacher_forced_prior(model, u_all, gating, stim):
    """Run teacher-forced forward pass, return (T, N) predictions."""
    model.eval()
    with torch.no_grad():
        params = model.precompute_params()
        prior_mu = model.forward_sequence(
            u_all, gating_data=gating, stim_data=stim, params=params)
    return prior_mu


# ═══════════════════════════════════════════════════════════════════════
#  Transformer baseline (K-step context → one-step prediction)
# ═══════════════════════════════════════════════════════════════════════

def _train_transformer_cv(
    u: np.ndarray,
    loo_neurons: List[int],
    K: int = 5,
    n_folds: int = 2,
    device: str = "cuda",
    window_size: int = 50,
    max_epochs: int = 200,
    patience: int = 20,
    verbose: bool = True,
):
    """Train a causal Transformer baseline with 2-fold temporal CV.

    Uses K past frames [t-K+1 … t-1] to predict u(t).
    Neural-only (no behaviour), Gaussian output head.

    Returns
    -------
    onestep_r2 : (N,) per-neuron one-step R²
    loo_r2_w   : (n_loo,) windowed LOO R² for loo_neurons
    """
    from baseline_transformer.config import TransformerBaselineConfig
    from baseline_transformer.model import build_model
    from baseline_transformer.train import (
        _train_epoch, _eval_epoch, scheduled_sampling_prob,
    )
    from baseline_transformer.dataset import SlidingWindowDataset
    from torch.utils.data import DataLoader, ConcatDataset
    import copy

    T, N = u.shape
    dev = torch.device(device)

    # Config — matched to Conn-Transformer (small arch)
    cfg = TransformerBaselineConfig(
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.1,
        context_length=K,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=128,
        max_epochs=max_epochs,
        patience=patience,
        grad_clip=1.0,
        device=device,
        predict_beh=False,
        include_beh_input=False,
        w_beh=0.0,
    )

    # 2-fold temporal CV — match Stage2 convention:
    # _make_temporal_folds partitions prediction frames 1..T-1.
    from stage2.train import _make_temporal_folds as _mtf
    raw_folds = _mtf(T, n_folds)          # [(te_s, te_e), ...]
    folds = []                            # (tr_s, tr_e, te_s, te_e)
    for te_s, te_e in raw_folds:
        # train = everything outside test range
        if te_s == raw_folds[0][0]:       # first fold: train on later half
            folds.append((te_e, T, te_s, te_e))
        else:
            folds.append((0, te_s, te_s, te_e))

    x = u.astype(np.float32)  # (T, N) — neural only
    os_pred = np.full((T, N), np.nan, dtype=np.float32)
    loo_pred = np.full((T, N), np.nan, dtype=np.float32)

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        if verbose:
            print(f"  [Transformer fold {fi+1}/2]  train=[{tr_s},{tr_e})  test=[{te_s},{te_e})")

        # ── Datasets ──
        tr_len = tr_e - tr_s
        val_size = max(1, int(tr_len * 0.15))
        val_s = tr_e - val_size
        train_ds = SlidingWindowDataset(x, n_neural=N, context_length=K,
                                        start=tr_s, end=val_s)
        val_ds = SlidingWindowDataset(x, n_neural=N, context_length=K,
                                      start=val_s, end=tr_e)
        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"    ⚠ empty dataset, skipping fold {fi}")
            continue

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                                shuffle=False, drop_last=False)

        # ── Model ──
        model = build_model(N, cfg, device=device, n_beh=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                      weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epochs, eta_min=cfg.lr * 0.01)

        best_val, best_state, wait = float("inf"), None, 0
        for epoch in range(1, cfg.max_epochs + 1):
            ss_p = scheduled_sampling_prob(epoch, cfg)
            _train_epoch(model, train_loader, optimizer, dev,
                         grad_clip=cfg.grad_clip, ss_p_gt=ss_p, w_beh=0.0)
            val_m = _eval_epoch(model, val_loader, dev, w_beh=0.0)
            scheduler.step()
            vl = val_m["loss"]
            if vl < best_val - 1e-6:
                best_val = vl
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        if verbose:
            print(f"    trained {epoch} epochs, best val={best_val:.4f}")

        # ── One-step predictions on test fold ──
        x_t = torch.tensor(x, dtype=torch.float32, device=dev)
        with torch.no_grad():
            first = max(K, te_s)
            for t in range(first, te_e):
                ctx = x_t[t - K : t].unsqueeze(0)  # (1, K, N)
                mu_u, _ = model.predict_mean_split(ctx)
                os_pred[t] = mu_u.squeeze(0).cpu().numpy()

        # ── Windowed LOO on test fold ──
        with torch.no_grad():
            for ni in loo_neurons:
                u_te = x[te_s:te_e].copy()  # (T_te, N)
                T_te = u_te.shape[0]
                pred = np.full(T_te, np.nan, dtype=np.float32)
                for s in range(0, T_te, window_size):
                    e = min(s + window_size, T_te)
                    pred[s] = u_te[s, ni]  # re-seed
                    for t_local in range(s, e - 1):
                        t_abs = te_s + t_local
                        if t_abs < K:
                            continue
                        # Build context from global array, replace ALL K
                        # frames of neuron ni with predictions (where avail)
                        ctx_np = x[t_abs - K : t_abs].copy()
                        for k in range(K):
                            pa_idx = t_local - (K - 1) + k
                            if pa_idx >= s and pa_idx <= t_local:
                                ctx_np[k, ni] = pred[pa_idx]
                        ctx = torch.tensor(ctx_np, dtype=torch.float32,
                                           device=dev).unsqueeze(0)
                        mu_u, _ = model.predict_mean_split(ctx)
                        pred[t_local + 1] = mu_u[0, ni].cpu().item()
                loo_pred[te_s:te_e, ni] = pred

        del model, optimizer
        torch.cuda.empty_cache()

    # ── Compute R² ──
    onestep_r2 = np.full(N, np.nan)
    for i in range(N):
        m = np.isfinite(os_pred[:, i])
        if m.sum() > 3:
            onestep_r2[i] = _r2(u[m, i], os_pred[m, i])

    loo_r2_w = np.full(len(loo_neurons), np.nan)
    for ki, ni in enumerate(loo_neurons):
        m = np.isfinite(loo_pred[:, ni])
        if m.sum() > 3:
            loo_r2_w[ki] = _r2(u[m, ni], loo_pred[m, ni])

    return onestep_r2, loo_r2_w


# ═══════════════════════════════════════════════════════════════════════
#  Train a single model with one neuron masked from loss
# ═══════════════════════════════════════════════════════════════════════

def train_single_retrain_loo(
    cfg, data, held_out: int,
    n_folds: int = 2,
    window_size: int = 50,
    warmup_steps: int = 40,
    verbose: bool = True,
):
    """Train Stage2 with neuron `held_out` excluded from loss, then LOO-eval it.

    Returns windowed LOO R² for the held-out neuron (cross-validated).
    """
    from stage2.train import _make_temporal_folds, apply_training_step

    u_stage1 = data["u_stage1"]
    sigma_u = data["sigma_u"]
    T, N = u_stage1.shape
    device = torch.device(cfg.device)

    gating_data = data.get("gating")
    stim_data = data.get("stim")
    d_ell = data.get("d_ell", 0)

    folds = _make_temporal_folds(T, n_folds)

    loo_pred_full = np.full(T, np.nan, dtype=np.float32)

    for fi, (te_s, te_e) in enumerate(folds):
        # ── train mask: exclude held-out time AND held-out neuron ──
        train_mask_t = torch.ones(T - 1, dtype=torch.bool, device=device)
        train_mask_t[te_s - 1 : te_e - 1] = False

        # Neuron mask: loss weight = 0 for held-out neuron
        neuron_mask = torch.ones(N, dtype=torch.float32, device=device)
        neuron_mask[held_out] = 0.0  # exclude from loss

        # ── build model ──
        sign_t = data.get("sign_t")
        lambda_u_init = init_lambda_u(u_stage1, cfg)
        model = Stage2ModelPT(
            N, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
            cfg, device, d_ell=d_ell,
            lambda_u_init=lambda_u_init, sign_t=sign_t,
        ).to(device)
        init_all_from_data(model, u_stage1, cfg)

        # ── optimiser ──
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
        optimiser = torch.optim.Adam(param_groups)

        z_target = u_stage1.to(device)
        grad_clip = float(getattr(cfg, "grad_clip_norm", 0.0) or 0.0)

        # ── training loop ──
        for epoch in range(cfg.num_epochs):
            optimiser.zero_grad()

            cached_params = model.precompute_params()
            prior_mu = model.forward_sequence(
                z_target, gating_data=gating_data,
                stim_data=stim_data, params=cached_params)

            # One-step loss: masked by both time (CV) and neuron (LOO)
            diff = z_target[1:] - prior_mu[1:]  # (T-1, N)
            # Variance weighting
            sigma2 = sigma_u.view(1, -1) ** 2
            weighted = diff ** 2 / sigma2.expand_as(diff)  # (T-1, N)
            # Apply neuron mask (zero out held-out neuron's contribution)
            weighted = weighted * neuron_mask.unsqueeze(0)
            # Apply time mask (exclude held-out fold)
            weighted = weighted * train_mask_t.unsqueeze(1).float()
            n_active = train_mask_t.sum() * (N - 1)  # 1 neuron excluded
            loss = weighted.sum() / max(n_active, 1)

            # Basic regularisation (coupling gate)
            coupling_gate_reg = float(getattr(cfg, "coupling_gate_reg", 0.0) or 0.0)
            if coupling_gate_reg > 0 and hasattr(model, '_coupling_gate_raw'):
                loss = loss + coupling_gate_reg * model._coupling_gate_raw.pow(2).mean()

            apply_training_step(loss, optimiser, params, model, cfg,
                                grad_clip=grad_clip)

            if verbose and (epoch == 0 or (epoch + 1) == cfg.num_epochs
                            or (epoch + 1) % 10 == 0):
                print(f"    [fold {fi}] Epoch {epoch+1}/{cfg.num_epochs}: "
                      f"loss={loss.item():.4f}")

        # ── LOO evaluation for this fold ──
        model.eval()
        u_device = u_stage1.to(device)
        pred = loo_forward_simulate_windowed(
            model, u_device, held_out, gating_data, stim_data,
            window_size=window_size, warmup_steps=warmup_steps,
        )
        loo_pred_full[te_s:te_e] = pred[te_s:te_e]

        del model, optimiser
        torch.cuda.empty_cache()

    # Compute CV LOO R²
    valid = ~np.isnan(loo_pred_full)
    u_np = u_stage1.cpu().numpy()
    r2 = _r2(u_np[valid, held_out], loo_pred_full[valid])
    return r2, loo_pred_full


# ═══════════════════════════════════════════════════════════════════════
#  Stage2 inline training (same folds as all baselines)
# ═══════════════════════════════════════════════════════════════════════

def _train_stage2_cv(
    cfg, data, loo_neurons: List[int],
    n_folds: int = 2,
    window_size: int = 50,
    warmup_steps: int = 40,
    verbose: bool = True,
):
    """Train Stage2 from scratch with k-fold temporal CV (NO neuron masking).

    Uses the same `_make_temporal_folds` as every other model so that
    train/test splits are identical.

    Returns
    -------
    onestep_r2 : (N,) per-neuron one-step R²  (teacher-forced, test folds)
    loo_r2_w   : (n_loo,) windowed LOO R² for loo_neurons
    """
    from stage2.train import _make_temporal_folds, apply_training_step

    u_stage1 = data["u_stage1"]
    sigma_u = data["sigma_u"]
    T, N = u_stage1.shape
    device = torch.device(cfg.device)

    gating_data = data.get("gating")
    stim_data = data.get("stim")
    d_ell = data.get("d_ell", 0)

    folds = _make_temporal_folds(T, n_folds)

    os_pred_full = np.full((T, N), np.nan, dtype=np.float32)
    loo_pred_full = np.full((T, N), np.nan, dtype=np.float32)

    for fi, (te_s, te_e) in enumerate(folds):
        t_fold = time.time()
        if verbose:
            print(f"\n  [Stage2 fold {fi+1}/{n_folds}]  test=[{te_s},{te_e})")

        # ── train mask: exclude held-out time ──
        train_mask_t = torch.ones(T - 1, dtype=torch.bool, device=device)
        train_mask_t[te_s - 1 : te_e - 1] = False

        # ── build model ──
        sign_t = data.get("sign_t")
        lambda_u_init = init_lambda_u(u_stage1, cfg)
        model = Stage2ModelPT(
            N, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
            cfg, device, d_ell=d_ell,
            lambda_u_init=lambda_u_init, sign_t=sign_t,
        ).to(device)
        init_all_from_data(model, u_stage1, cfg)

        # ── optimiser ──
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
        optimiser = torch.optim.Adam(param_groups)

        z_target = u_stage1.to(device)
        grad_clip = float(getattr(cfg, "grad_clip_norm", 0.0) or 0.0)

        # ── training loop (ALL neurons in loss — no masking) ──
        for epoch in range(cfg.num_epochs):
            optimiser.zero_grad()

            cached_params = model.precompute_params()
            prior_mu = model.forward_sequence(
                z_target, gating_data=gating_data,
                stim_data=stim_data, params=cached_params)

            diff = z_target[1:] - prior_mu[1:]
            sigma2 = sigma_u.view(1, -1) ** 2
            weighted = diff ** 2 / sigma2.expand_as(diff)
            weighted = weighted * train_mask_t.unsqueeze(1).float()
            n_active = train_mask_t.sum() * N
            loss = weighted.sum() / max(n_active, 1)

            coupling_gate_reg = float(getattr(cfg, "coupling_gate_reg", 0.0) or 0.0)
            if coupling_gate_reg > 0 and hasattr(model, '_coupling_gate_raw'):
                loss = loss + coupling_gate_reg * model._coupling_gate_raw.pow(2).mean()

            apply_training_step(loss, optimiser, params, model, cfg,
                                grad_clip=grad_clip)

            if verbose and (epoch == 0 or (epoch + 1) == cfg.num_epochs
                            or (epoch + 1) % 10 == 0):
                print(f"    Epoch {epoch+1}/{cfg.num_epochs}: loss={loss.item():.4f}")

        # ── One-step R² on test fold (teacher-forced) ──
        model.eval()
        prior_mu = _teacher_forced_prior(
            model, z_target, gating_data, stim_data)
        os_pred_full[te_s:te_e] = prior_mu[te_s:te_e].cpu().numpy()

        # ── Windowed LOO on test fold ──
        u_device = u_stage1.to(device)
        preds = loo_forward_simulate_batched_windowed(
            model, u_device, loo_neurons, gating_data, stim_data,
            window_size=window_size, warmup_steps=warmup_steps,
        )
        for ni in loo_neurons:
            loo_pred_full[te_s:te_e, ni] = preds[ni][te_s:te_e]

        fold_time = time.time() - t_fold
        if verbose:
            print(f"    Fold {fi+1} done ({fold_time:.1f}s)")

        del model, optimiser
        torch.cuda.empty_cache()

    # ── Compute R² ──
    u_np = u_stage1.cpu().numpy()
    onestep_r2 = np.full(N, np.nan, dtype=np.float32)
    for i in range(N):
        m = np.isfinite(os_pred_full[:, i])
        if m.sum() > 3:
            onestep_r2[i] = _r2(u_np[m, i], os_pred_full[m, i])

    loo_r2_w = np.full(len(loo_neurons), np.nan, dtype=np.float32)
    for ki, ni in enumerate(loo_neurons):
        m = np.isfinite(loo_pred_full[:, ni])
        if m.sum() > 3:
            loo_r2_w[ki] = _r2(u_np[m, ni], loo_pred_full[m, ni])

    return onestep_r2, loo_r2_w


# ═══════════════════════════════════════════════════════════════════════
#  BATCHED retrain-LOO: one model per fold, all neurons masked at once
# ═══════════════════════════════════════════════════════════════════════

def train_batched_retrain_loo(
    cfg, data, loo_neurons: List[int],
    n_folds: int = 2,
    window_size: int = 50,
    warmup_steps: int = 40,
    verbose: bool = True,
):
    """Train Stage2 with ALL loo_neurons excluded from loss at once.

    Instead of 30 separate models, we train ONE model per fold where
    all 30 LOO neurons have loss weight = 0.  Then evaluate all 30
    neurons in parallel using batched windowed LOO.

    Returns
    -------
    retrain_loo_r2 : (N,) array with R² for LOO neurons (NaN elsewhere)
    """
    from stage2.train import _make_temporal_folds, apply_training_step

    u_stage1 = data["u_stage1"]
    sigma_u = data["sigma_u"]
    T, N = u_stage1.shape
    device = torch.device(cfg.device)

    gating_data = data.get("gating")
    stim_data = data.get("stim")
    d_ell = data.get("d_ell", 0)

    folds = _make_temporal_folds(T, n_folds)

    # Collect per-fold LOO predictions
    loo_pred_full = np.full((T, N), np.nan, dtype=np.float32)

    for fi, (te_s, te_e) in enumerate(folds):
        t_fold = time.time()
        if verbose:
            print(f"\n  [Batched fold {fi+1}/{n_folds}]  test=[{te_s},{te_e})")

        # ── train mask: exclude held-out time ──
        train_mask_t = torch.ones(T - 1, dtype=torch.bool, device=device)
        train_mask_t[te_s - 1 : te_e - 1] = False

        # Neuron mask: loss weight = 0 for ALL LOO neurons
        neuron_mask = torch.ones(N, dtype=torch.float32, device=device)
        for ni in loo_neurons:
            neuron_mask[ni] = 0.0
        n_masked = len(loo_neurons)
        if verbose:
            print(f"    Masking {n_masked}/{N} neurons from loss")

        # ── build model ──
        sign_t = data.get("sign_t")
        lambda_u_init = init_lambda_u(u_stage1, cfg)
        model = Stage2ModelPT(
            N, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
            cfg, device, d_ell=d_ell,
            lambda_u_init=lambda_u_init, sign_t=sign_t,
        ).to(device)
        init_all_from_data(model, u_stage1, cfg)

        # ── optimiser ──
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
        optimiser = torch.optim.Adam(param_groups)

        z_target = u_stage1.to(device)
        grad_clip = float(getattr(cfg, "grad_clip_norm", 0.0) or 0.0)

        # ── training loop ──
        for epoch in range(cfg.num_epochs):
            optimiser.zero_grad()

            cached_params = model.precompute_params()
            prior_mu = model.forward_sequence(
                z_target, gating_data=gating_data,
                stim_data=stim_data, params=cached_params)

            diff = z_target[1:] - prior_mu[1:]
            sigma2 = sigma_u.view(1, -1) ** 2
            weighted = diff ** 2 / sigma2.expand_as(diff)
            weighted = weighted * neuron_mask.unsqueeze(0)
            weighted = weighted * train_mask_t.unsqueeze(1).float()
            n_active = train_mask_t.sum() * (N - n_masked)
            loss = weighted.sum() / max(n_active, 1)

            coupling_gate_reg = float(getattr(cfg, "coupling_gate_reg", 0.0) or 0.0)
            if coupling_gate_reg > 0 and hasattr(model, '_coupling_gate_raw'):
                loss = loss + coupling_gate_reg * model._coupling_gate_raw.pow(2).mean()

            apply_training_step(loss, optimiser, params, model, cfg,
                                grad_clip=grad_clip)

            if verbose and (epoch == 0 or (epoch + 1) == cfg.num_epochs
                            or (epoch + 1) % 10 == 0):
                print(f"    Epoch {epoch+1}/{cfg.num_epochs}: loss={loss.item():.4f}")

        # ── Batched LOO evaluation for ALL neurons on this fold ──
        model.eval()
        u_device = u_stage1.to(device)
        preds = loo_forward_simulate_batched_windowed(
            model, u_device, loo_neurons, gating_data, stim_data,
            window_size=window_size, warmup_steps=warmup_steps,
        )
        for ni in loo_neurons:
            loo_pred_full[te_s:te_e, ni] = preds[ni][te_s:te_e]

        fold_time = time.time() - t_fold
        if verbose:
            print(f"    Fold {fi+1} done ({fold_time:.1f}s)")

        del model, optimiser
        torch.cuda.empty_cache()

    # ── Compute R² per neuron ──
    u_np = u_stage1.cpu().numpy()
    retrain_loo_r2 = np.full(N, np.nan, dtype=np.float32)
    for ni in loo_neurons:
        valid = ~np.isnan(loo_pred_full[:, ni])
        if valid.sum() > 3:
            retrain_loo_r2[ni] = _r2(u_np[valid, ni], loo_pred_full[valid, ni])

    return retrain_loo_r2


# ═══════════════════════════════════════════════════════════════════════
#  Comparison plot
# ═══════════════════════════════════════════════════════════════════════

COLORS = {
    "Stage2":       "#d62728",
    "Retrain-LOO":  "#ff6961",
    "Ridge":        "#4682b4",
    "EN":           "#228b22",
    "MLP":          "#ff7f0e",
    "Conn-Ridge":   "#1e90ff",
    "Conn-MLP":     "#9932cc",
    "Transformer":  "#17becf",
    "Conn-Trf":     "#bcbd22",
}

ORDER_B = ["Stage2", "Ridge", "EN", "MLP",
           "Transformer", "Conn-Ridge", "Conn-MLP", "Conn-Trf"]
ORDER_C = ["Stage2", "Retrain-LOO", "Ridge", "EN", "MLP",
           "Conn-Ridge", "Conn-MLP", "Transformer", "Conn-Trf"]

def plot_comparison(
    standard_loo: np.ndarray,
    retrain_loo: np.ndarray,
    baselines: Dict[str, np.ndarray],
    loo_neurons: List[int],
    save_path: Path,
    worm_name: str,
    trf_loo: Optional[np.ndarray] = None,
):
    """Paired scatter + strip-box comparison of standard vs retrain LOO."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel A: paired scatter (standard vs retrain) ──
    ax = axes[0]
    sl = standard_loo[loo_neurons]
    rl = retrain_loo[loo_neurons]
    valid = np.isfinite(sl) & np.isfinite(rl)
    sl_v, rl_v = sl[valid], rl[valid]

    ax.scatter(sl_v, rl_v, s=40, alpha=0.7, color="#2ca02c",
               edgecolors="white", linewidths=0.5, zorder=3)
    lims = [min(sl_v.min(), rl_v.min()) - 0.05,
            max(sl_v.max(), rl_v.max()) + 0.05]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Standard LOO R² (windowed)", fontsize=11)
    ax.set_ylabel("Retrain-LOO R² (windowed)", fontsize=11)
    ax.set_title("A.  Paired comparison (per neuron)", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    mu_s = float(np.nanmean(sl_v))
    mu_r = float(np.nanmean(rl_v))
    ax.annotate(f"Standard mean={mu_s:.3f}\nRetrain mean={mu_r:.3f}",
                xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Panel B: strip-box across models ──
    ax = axes[1]
    models = {}
    models["Standard LOO"] = sl_v
    models["Retrain-LOO"] = rl_v
    if trf_loo is not None:
        models["Transformer"] = trf_loo[np.isfinite(trf_loo)]
    for name in ["EN", "MLP", "Conn-Ridge"]:
        if name in baselines:
            models[name] = baselines[name]

    ordered = list(models.keys())
    positions = np.arange(len(ordered))
    bp_data = []
    for xi, name in enumerate(ordered):
        vals = models[name]
        v = vals[np.isfinite(vals)]
        bp_data.append(v)
        c = COLORS.get(name, "#888")
        jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(v))
        ax.scatter(xi + jitter, v, s=18, alpha=0.55, color=c,
                   edgecolors="white", linewidths=0.3, zorder=3)

    bp = ax.boxplot(bp_data, positions=positions, widths=0.45,
                    patch_artist=True, showfliers=False, zorder=4,
                    medianprops=dict(color="black", linewidth=1.8),
                    whiskerprops=dict(color="gray", linewidth=1.0),
                    capprops=dict(color="gray", linewidth=1.0))
    for patch, name in zip(bp["boxes"], ordered):
        c = COLORS.get(name, "#888")
        patch.set_facecolor(c); patch.set_alpha(0.25)
        patch.set_edgecolor(c); patch.set_linewidth(1.2)

    for xi, (name, v) in enumerate(zip(ordered, bp_data)):
        if len(v) == 0: continue
        mu = float(np.mean(v))
        ax.plot(xi, mu, marker="D", color=COLORS.get(name, "#888"),
                markersize=7, markeredgecolor="black", markeredgewidth=0.8,
                zorder=5)
        ax.annotate(f"{mu:.3f}", (xi, mu), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, fontweight="bold",
                    color=COLORS.get(name, "#888"))

    ax.set_xticks(positions)
    ax.set_xticklabels(ordered, fontsize=9, fontweight="bold", rotation=15)
    ax.set_ylabel("LOO R² (windowed, w=50)", fontsize=11)
    ax.set_title("B.  LOO R² distribution", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Retrain-LOO vs Standard LOO — {worm_name}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Connectome loading + Conn-Transformer (per-neuron)
# ═══════════════════════════════════════════════════════════════════════

def load_connectome(worm_labels: List[str]) -> Dict[int, List[int]]:
    """Load combined connectome → partners dict (presynaptic indices)."""
    d = ROOT / "data/used/masks+motor neurons"
    atlas_names = np.load(d / "neuron_names.npy")
    n2a = {str(n): i for i, n in enumerate(atlas_names)}
    T_all = sum(
        np.abs(np.load(d / f"{t}.npy")) for t in ("T_e", "T_sv", "T_dcv")
    )
    N = len(worm_labels)
    wa = [n2a.get(lab, -1) for lab in worm_labels]
    adj = np.zeros((N, N), np.float64)
    for i in range(N):
        for j in range(N):
            if wa[i] >= 0 and wa[j] >= 0:
                adj[j, i] = T_all[wa[j], wa[i]]
    partners: Dict[int, List[int]] = {}
    for i in range(N):
        partners[i] = sorted(j for j in range(N) if j != i and adj[j, i] > 0)
    np_arr = [len(partners[i]) for i in range(N)]
    print(f"  Connectome: partners/neuron — min={min(np_arr)}, "
          f"median={int(np.median(np_arr))}, max={max(np_arr)}")
    return partners


class ConnTransformer(nn.Module):
    """Per-neuron causal transformer: (K, d_in) → scalar."""

    def __init__(self, d_in: int, d_model: int = 32, n_heads: int = 2,
                 n_layers: int = 1, K: int = 5, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, K, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.full((K, K), float("-inf")), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K_in = x.size(1)
        h = self.proj(x) + self.pos_emb[:, :K_in, :]
        mask = self._causal_mask[:K_in, :K_in]
        h = self.encoder(h, mask=mask)
        return self.head(h[:, -1, :]).squeeze(-1)


class ConnLinearK(nn.Module):
    """Per-neuron linear model: flatten (K, d_in) → scalar."""

    def __init__(self, d_in: int, K: int = 5):
        super().__init__()
        self.linear = nn.Linear(K * d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.reshape(x.size(0), -1)).squeeze(-1)


class ConnMLP(nn.Module):
    """Per-neuron MLP: flatten (K, d_in) → hidden → scalar."""

    def __init__(self, d_in: int, K: int = 5, hidden: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        layers: list = [
            nn.Linear(K * d_in, hidden),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden // 2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.size(0), -1)).squeeze(-1)


def _train_conn_trf_cv(
    u: np.ndarray,
    labels: List[str],
    loo_neurons: List[int],
    K: int = 5,
    n_folds: int = 2,
    device: str = "cuda",
    window_size: int = 50,
    max_epochs: int = 100,
    patience: int = 15,
    verbose: bool = True,
    model_type: str = "transformer",
    weight_decay: float = 1e-4,
):
    """Per-neuron connectome-constrained Transformer (K-step context).

    For each neuron i, trains a tiny causal transformer using only
    connectome neighbours + self as input features over K time steps.

    Returns
    -------
    onestep_r2 : (N,) per-neuron one-step R²
    loo_r2_w   : (n_loo,) windowed LOO R² for loo_neurons
    """
    T, N = u.shape
    dev = torch.device(device)
    partners = load_connectome(labels)

    from stage2.train import _make_temporal_folds as _mtf
    raw_folds = _mtf(T, n_folds)
    folds = []
    for te_s, te_e in raw_folds:
        if te_s == raw_folds[0][0]:
            folds.append((te_e, T, te_s, te_e))
        else:
            folds.append((0, te_s, te_s, te_e))
    x = u.astype(np.float32)
    os_pred = np.full((T, N), np.nan, dtype=np.float32)
    loo_pred = np.full((T, N), np.nan, dtype=np.float32)
    loo_set = set(loo_neurons)
    tag = {"transformer": "Conn-Trf", "linear": "Conn-Ridge",
           "mlp": "Conn-MLP"}.get(model_type, model_type)

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        if verbose:
            print(f"  [{tag} fold {fi+1}/2]  train=[{tr_s},{tr_e})  "
                  f"test=[{te_s},{te_e})")
        u_tr = x[tr_s:tr_e]
        T_te = te_e - te_s

        for i in range(N):
            feats = sorted(partners[i] + [i])
            if not feats:
                feats = [i]
            d_in = len(feats)

            # Sliding-window training data
            indices = np.arange(K, len(u_tr))
            X_np = np.stack([u_tr[t - K : t, feats] for t in indices])
            Y_np = u_tr[indices, i]
            X_t = torch.tensor(X_np, dtype=torch.float32, device=dev)
            Y_t = torch.tensor(Y_np, dtype=torch.float32, device=dev)

            nv = max(int(len(X_t) * 0.1), 1)
            nf = len(X_t) - nv

            if model_type == "linear":
                model = ConnLinearK(d_in, K=K).to(dev)
            elif model_type == "mlp":
                model = ConnMLP(d_in, K=K, dropout=0.5).to(dev)
            else:
                model = ConnTransformer(d_in, d_model=32, n_heads=2,
                                        n_layers=1, K=K, dropout=0.1).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                                   weight_decay=weight_decay)
            bvl, bst, w = float("inf"), None, 0

            for ep in range(max_epochs):
                model.train()
                perm = torch.randperm(nf, device=dev)
                for bs in range(0, nf, 128):
                    idx = perm[bs : bs + 128]
                    loss = nn.functional.mse_loss(model(X_t[idx]), Y_t[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vl = nn.functional.mse_loss(
                        model(X_t[nf:]), Y_t[nf:]).item()
                if vl < bvl - 1e-6:
                    bvl = vl
                    bst = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
                    w = 0
                else:
                    w += 1
                    if w >= patience:
                        break

            if bst:
                model.load_state_dict(
                    {k: v.to(dev) for k, v in bst.items()})
            model.eval()

            # one-step predictions on test fold (batched)
            with torch.no_grad():
                first = max(K, te_s)
                n_te = te_e - first
                if n_te > 0:
                    ctx_all = np.stack(
                        [x[t - K : t, feats]
                         for t in range(first, te_e)])
                    ctx_t = torch.tensor(
                        ctx_all, dtype=torch.float32, device=dev)
                    os_pred[first:te_e, i] = model(ctx_t).cpu().numpy()

            # windowed LOO (only for LOO neurons)
            if i in loo_set:
                sp = feats.index(i) if i in feats else None
                u_te = x[te_s:te_e]
                pred_arr = np.full(T_te, np.nan, dtype=np.float32)
                with torch.no_grad():
                    for s in range(0, T_te, window_size):
                        e = min(s + window_size, T_te)
                        pred_arr[s] = u_te[s, i]
                        for t_loc in range(s, e - 1):
                            t_abs = te_s + t_loc
                            # Context [t_abs-K+1 .. t_abs] → predict t_abs+1
                            ctx_start = t_abs - K + 1
                            if ctx_start < 0:
                                continue
                            ctx = x[ctx_start : t_abs + 1, feats].copy()
                            # Replace ALL neuron-i positions with predictions
                            if sp is not None:
                                for k in range(K):
                                    pa_idx = t_loc - (K - 1) + k
                                    if (pa_idx >= s and pa_idx <= t_loc
                                            and np.isfinite(pred_arr[pa_idx])):
                                        ctx[k, sp] = pred_arr[pa_idx]
                            ctx_t = torch.tensor(
                                ctx, dtype=torch.float32, device=dev
                            ).unsqueeze(0)
                            pred_arr[t_loc + 1] = (
                                model(ctx_t).cpu().item())
                loo_pred[te_s:te_e, i] = pred_arr

            del model, opt
            if (i + 1) % 30 == 0 or i == N - 1:
                if verbose:
                    print(f"    {i + 1}/{N} neurons")

        torch.cuda.empty_cache()

    # compute R²
    onestep_r2 = np.full(N, np.nan)
    for i in range(N):
        m = np.isfinite(os_pred[:, i])
        if m.sum() > 3:
            onestep_r2[i] = _r2(u[m, i], os_pred[m, i])

    loo_r2_w = np.full(len(loo_neurons), np.nan)
    for ki, ni in enumerate(loo_neurons):
        m = np.isfinite(loo_pred[:, ni])
        if m.sum() > 3:
            loo_r2_w[ki] = _r2(u[m, ni], loo_pred[m, ni])

    return onestep_r2, loo_r2_w


# ═══════════════════════════════════════════════════════════════════════
#  Joint baselines with K-context window (Ridge / EN / MLP)
# ═══════════════════════════════════════════════════════════════════════

def _train_joint_baseline_cv(
    u: np.ndarray,
    loo_neurons: List[int],
    K: int = 5,
    model_type: str = "ridge",
    n_folds: int = 2,
    device: str = "cuda",
    window_size: int = 50,
    max_epochs: int = 200,
    patience: int = 20,
    verbose: bool = True,
):
    """Train joint (all-neuron) baseline with K-step context window.

    X = flatten(u[t-K:t]) ∈ R^(K*N)  →  Y = u[t] ∈ R^N

    model_type : "ridge" | "en" | "mlp"

    Returns
    -------
    onestep_r2 : (N,) per-neuron one-step R²
    loo_r2_w   : (n_loo,) windowed LOO R² for loo_neurons
    """
    T, N = u.shape
    dev = torch.device(device)
    from stage2.train import _make_temporal_folds as _mtf
    raw_folds = _mtf(T, n_folds)
    folds = []
    for te_s, te_e in raw_folds:
        if te_s == raw_folds[0][0]:
            folds.append((te_e, T, te_s, te_e))
        else:
            folds.append((0, te_s, te_s, te_e))

    x = u.astype(np.float32)
    os_pred = np.full((T, N), np.nan, dtype=np.float32)
    loo_pred = np.full((T, N), np.nan, dtype=np.float32)

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        if verbose:
            print(f"  [{model_type.upper()} fold {fi+1}/2]  "
                  f"train=[{tr_s},{tr_e})  test=[{te_s},{te_e})")

        u_tr = x[tr_s:tr_e]
        T_te = te_e - te_s

        # ── Sliding-window training data ──
        indices = np.arange(K, len(u_tr))
        X_np = np.stack([u_tr[t - K : t].flatten() for t in indices])
        Y_np = u_tr[indices]

        # ── Fit model ──
        sk_model = None
        W_en: Optional[np.ndarray] = None
        b_en: Optional[np.ndarray] = None
        mlp_model: Optional[nn.Module] = None
        opt = None

        if model_type == "ridge":
            import warnings
            from scipy.linalg import LinAlgWarning
            alphas = np.logspace(-3, 6, 20)
            best_a, best_s = 1.0, -np.inf
            n_s = X_np.shape[0]
            fs = n_s // 3
            for a in alphas:
                sc = []
                for k in range(3):
                    s_cv = k * fs
                    e_cv = (k + 1) * fs if k < 2 else n_s
                    Xtr = np.concatenate([X_np[:s_cv], X_np[e_cv:]])
                    Ytr = np.concatenate([Y_np[:s_cv], Y_np[e_cv:]])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", LinAlgWarning)
                        sc.append(Ridge(alpha=a).fit(Xtr, Ytr).score(
                            X_np[s_cv:e_cv], Y_np[s_cv:e_cv]))
                if np.mean(sc) > best_s:
                    best_s, best_a = np.mean(sc), a
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LinAlgWarning)
                sk_model = Ridge(alpha=best_a).fit(X_np, Y_np)
            if verbose:
                print(f"    Ridge: alpha={best_a:.2e}")

        elif model_type == "en":
            W_en = np.zeros((N, K * N), np.float64)
            b_en = np.zeros(N, np.float64)
            for j in range(N):
                en = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=5000)
                en.fit(X_np, Y_np[:, j])
                W_en[j], b_en[j] = en.coef_, en.intercept_
            if verbose:
                print(f"    EN: fitted {N} per-neuron models")

        elif model_type == "mlp":
            din = K * N
            nv = max(int(X_np.shape[0] * 0.15), 1)
            nf = X_np.shape[0] - nv
            Xf = torch.tensor(X_np[:nf], dtype=torch.float32, device=dev)
            Xv = torch.tensor(X_np[nf:], dtype=torch.float32, device=dev)
            Yf = torch.tensor(Y_np[:nf], dtype=torch.float32, device=dev)
            Yv = torch.tensor(Y_np[nf:], dtype=torch.float32, device=dev)

            mlp_model = JointMLP(din, N, hidden=(64, 32),
                                dropout=0.5).to(dev)
            opt = torch.optim.Adam(mlp_model.parameters(), lr=1e-3,
                                   weight_decay=0.01)
            bvl, bst, w = float("inf"), None, 0
            for ep in range(max_epochs):
                mlp_model.train()
                perm = torch.randperm(nf, device=dev)
                for bs in range(0, nf, 128):
                    idx = perm[bs : bs + 128]
                    loss = nn.functional.mse_loss(mlp_model(Xf[idx]),
                                                  Yf[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                mlp_model.eval()
                with torch.no_grad():
                    vl = nn.functional.mse_loss(mlp_model(Xv), Yv).item()
                if vl < bvl - 1e-6:
                    bvl = vl
                    bst = {k: v.cpu().clone()
                           for k, v in mlp_model.state_dict().items()}
                    w = 0
                else:
                    w += 1
                    if w >= patience:
                        break
            if bst:
                mlp_model.load_state_dict(
                    {k: v.to(dev) for k, v in bst.items()})
            mlp_model.eval()
            if verbose:
                print(f"    MLP: {ep + 1} epochs, best val={bvl:.6f}")

        # ── Batched one-step predictions on test fold ──
        first = max(K, te_s)
        n_te = te_e - first
        if n_te > 0:
            X_te = np.stack([x[t - K : t].flatten()
                             for t in range(first, te_e)])
            if model_type == "ridge":
                os_pred[first:te_e] = sk_model.predict(X_te)
            elif model_type == "en":
                os_pred[first:te_e] = X_te @ W_en.T + b_en
            elif model_type == "mlp":
                with torch.no_grad():
                    for bs in range(0, n_te, 512):
                        be = min(bs + 512, n_te)
                        ct = torch.tensor(X_te[bs:be],
                                          dtype=torch.float32, device=dev)
                        os_pred[first + bs : first + be] = (
                            mlp_model(ct).cpu().numpy())

        # ── Windowed LOO ──
        for ni in loo_neurons:
            pred_arr = np.full(T_te, np.nan, dtype=np.float32)
            for s in range(0, T_te, window_size):
                e = min(s + window_size, T_te)
                pred_arr[s] = x[te_s + s, ni]
                for t_loc in range(s, e - 1):
                    t_abs = te_s + t_loc
                    ctx_start = t_abs - K + 1
                    if ctx_start < 0:
                        continue
                    ctx = x[ctx_start : t_abs + 1].copy()  # (K, N)
                    # Replace ALL K positions of held-out neuron
                    for k in range(K):
                        pa_idx = t_loc - (K - 1) + k
                        if (pa_idx >= s and pa_idx <= t_loc
                                and np.isfinite(pred_arr[pa_idx])):
                            ctx[k, ni] = pred_arr[pa_idx]
                    ctx_flat = ctx.flatten()
                    if model_type == "ridge":
                        pred_arr[t_loc + 1] = sk_model.predict(
                            ctx_flat.reshape(1, -1))[0][ni]
                    elif model_type == "en":
                        pred_arr[t_loc + 1] = (
                            ctx_flat @ W_en[ni] + b_en[ni])
                    elif model_type == "mlp":
                        with torch.no_grad():
                            ct = torch.tensor(
                                ctx_flat, dtype=torch.float32,
                                device=dev).unsqueeze(0)
                            pred_arr[t_loc + 1] = (
                                mlp_model(ct).cpu().numpy()[0, ni])
            loo_pred[te_s:te_e, ni] = pred_arr

        # cleanup
        if model_type == "mlp" and mlp_model is not None:
            del mlp_model, opt
            torch.cuda.empty_cache()

    # ── Compute R² ──
    onestep_r2 = np.full(N, np.nan)
    for i in range(N):
        m = np.isfinite(os_pred[:, i])
        if m.sum() > 3:
            onestep_r2[i] = _r2(u[m, i], os_pred[m, i])

    loo_r2_w = np.full(len(loo_neurons), np.nan)
    for ki, ni in enumerate(loo_neurons):
        m = np.isfinite(loo_pred[:, ni])
        if m.sum() > 3:
            loo_r2_w[ki] = _r2(u[m, ni], loo_pred[m, ni])

    return onestep_r2, loo_r2_w


# ═══════════════════════════════════════════════════════════════════════
#  Two-panel plot:  B (one-step R²)  +  C (LOO R²)
# ═══════════════════════════════════════════════════════════════════════

def _strip_box(ax, ordered, data_dict, ylabel):
    """Draw strip (jittered dots) + box for each model on one axis."""
    positions = np.arange(len(ordered))
    bp_data = []

    for xi, name in enumerate(ordered):
        vals = data_dict.get(name)
        v = vals[np.isfinite(vals)] if vals is not None else np.array([])
        bp_data.append(v)
        if len(v) == 0:
            continue
        c = COLORS.get(name, "#888")
        jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(v))
        ax.scatter(xi + jitter, v, s=18, alpha=0.55, color=c,
                   edgecolors="white", linewidths=0.3, zorder=3)

    bp = ax.boxplot(
        bp_data, positions=positions, widths=0.45,
        patch_artist=True, showfliers=False, zorder=4,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(color="gray", linewidth=1.0),
        capprops=dict(color="gray", linewidth=1.0),
    )
    for patch, name in zip(bp["boxes"], ordered):
        c = COLORS.get(name, "#888")
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)

    for xi, (name, v) in enumerate(zip(ordered, bp_data)):
        if len(v) == 0:
            continue
        mu = float(np.mean(v))
        ax.plot(xi, mu, marker="D", color=COLORS.get(name, "#888"),
                markersize=7, markeredgecolor="black",
                markeredgewidth=0.8, zorder=5)
        ax.annotate(
            f"{mu:.3f}", (xi, mu), textcoords="offset points",
            xytext=(0, 10), ha="center", fontsize=8, fontweight="bold",
            color=COLORS.get(name, "#888"),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(ordered, fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def plot_bc(
    onestep_data: Dict[str, np.ndarray],
    loo_data: Dict[str, np.ndarray],
    save_path: Path,
    worm_name: str,
    N: int,
    n_loo: int,
):
    """Two-panel strip+box: B (one-step R²) + C (LOO R²)."""
    ordered_b = [n for n in ORDER_B if n in onestep_data]
    ordered_c = [n for n in ORDER_C if n in loo_data]

    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(18, 6))

    _strip_box(ax_b, ordered_b, onestep_data, "One-step R²")
    ax_b.set_title(f"B.  One-step R²   (all {N} neurons)",
                   fontsize=13, fontweight="bold")
    ax_b.set_ylim(-0.3, 1.05)

    _strip_box(ax_c, ordered_c, loo_data, "LOO R² (windowed)")
    ax_c.set_title(f"C.  LOO R²   ({n_loo} neurons, w=50)",
                   fontsize=13, fontweight="bold")
    ax_c.set_ylim(-0.8, 1.05)

    fig.suptitle(f"Per-neuron R² Distributions — {worm_name}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Retrain-LOO evaluation")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cv_folds", type=int, default=2)
    ap.add_argument("--loo_subset", type=int, default=30)
    ap.add_argument("--window_size", type=int, default=50)
    ap.add_argument("--warmup_steps", type=int, default=40)
    ap.add_argument("--K", type=int, default=5,
                    help="Context window length (past frames) for all models")
    ap.add_argument("--trf_epochs", type=int, default=200,
                    help="Max epochs for Transformer training")
    ap.add_argument("--baseline_epochs", type=int, default=200,
                    help="Max epochs for MLP / Conn-MLP training")
    args = ap.parse_args()

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    worm = Path(args.h5).stem

    # ── Build config & load data ──
    # Use Stage2 defaults (sigmoid, 50ep, 3-fold CV, seed=42)
    cfg = make_config(
        args.h5,
        num_epochs=args.epochs,
        device=args.device,
        cv_folds=args.cv_folds,
        eval_loo_subset_size=args.loo_subset,
        eval_loo_subset_mode="variance",
        eval_loo_window_size=args.window_size,
        eval_loo_warmup_steps=args.warmup_steps,
        skip_final_eval=True,
    )
    data = load_data_pt(cfg)
    u_stage1 = data["u_stage1"]
    T, N = u_stage1.shape
    print(f"  {worm}  T={T}  N={N}")

    # ── Determine LOO neurons (top-30 by variance) ──
    u_np = u_stage1.cpu().numpy()
    var = np.var(u_np, axis=0)
    loo_neurons = sorted(np.argsort(var)[::-1][:args.loo_subset].tolist())
    print(f"  LOO neurons ({len(loo_neurons)}): {loo_neurons}")

    # ── Load neuron labels for connectome ──
    with h5py.File(args.h5, "r") as f:
        labels = [
            l.decode() if isinstance(l, bytes) else str(l)
            for l in f["gcamp/neuron_labels"][:]
        ]

    K = args.K

    # ══════════════════════════════════════════════════════════════════
    #  Stage2 (trained inline, same folds as all baselines)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  Stage2 (connectome-constrained ODE)  {args.cv_folds}-fold CV")
    print(f"  {args.epochs} epochs × {args.cv_folds} folds")
    print(f"{'═'*60}")
    t0_s2 = time.time()
    s2_onestep, s2_loo_w = _train_stage2_cv(
        cfg, data, loo_neurons,
        n_folds=args.cv_folds,
        window_size=args.window_size,
        warmup_steps=args.warmup_steps,
        verbose=True,
    )
    s2_time = time.time() - t0_s2
    print(f"  Stage2 done ({s2_time:.0f}s)")
    print(f"  One-step R²: mean={np.nanmean(s2_onestep):.4f}  "
          f"median={np.nanmedian(s2_onestep):.4f}")
    print(f"  LOO R² (w): mean={np.nanmean(s2_loo_w):.4f}  "
          f"median={np.nanmedian(s2_loo_w):.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  Ridge baseline (joint, K-context window, sklearn)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  Ridge baseline  K={K}  (2-fold CV)")
    print(f"{'═'*60}")
    t0 = time.time()
    ridge_onestep, ridge_loo_w = _train_joint_baseline_cv(
        u_np, loo_neurons, K=K, model_type="ridge",
        n_folds=args.cv_folds, device=args.device,
        window_size=args.window_size, verbose=True,
    )
    print(f"  Ridge done ({time.time() - t0:.0f}s)")
    print(f"  One-step R²: mean={np.nanmean(ridge_onestep):.4f}  "
          f"median={np.nanmedian(ridge_onestep):.4f}")
    print(f"  LOO R² (w): mean={np.nanmean(ridge_loo_w):.4f}  "
          f"median={np.nanmedian(ridge_loo_w):.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  EN baseline (per-neuron ElasticNet, K-context window)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  EN baseline  K={K}  (2-fold CV)")
    print(f"{'═'*60}")
    t0 = time.time()
    en_onestep, en_loo_w = _train_joint_baseline_cv(
        u_np, loo_neurons, K=K, model_type="en",
        n_folds=args.cv_folds, device=args.device,
        window_size=args.window_size, verbose=True,
    )
    print(f"  EN done ({time.time() - t0:.0f}s)")
    print(f"  One-step R²: mean={np.nanmean(en_onestep):.4f}  "
          f"median={np.nanmedian(en_onestep):.4f}")
    print(f"  LOO R² (w): mean={np.nanmean(en_loo_w):.4f}  "
          f"median={np.nanmedian(en_loo_w):.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  MLP baseline (joint MLP-256, K-context window)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  MLP baseline  K={K}  (2-fold CV)")
    print(f"{'═'*60}")
    t0 = time.time()
    mlp_onestep, mlp_loo_w = _train_joint_baseline_cv(
        u_np, loo_neurons, K=K, model_type="mlp",
        n_folds=args.cv_folds, device=args.device,
        window_size=args.window_size,
        max_epochs=args.baseline_epochs, patience=20, verbose=True,
    )
    print(f"  MLP done ({time.time() - t0:.0f}s)")
    print(f"  One-step R²: mean={np.nanmean(mlp_onestep):.4f}  "
          f"median={np.nanmedian(mlp_onestep):.4f}")
    print(f"  LOO R² (w): mean={np.nanmean(mlp_loo_w):.4f}  "
          f"median={np.nanmedian(mlp_loo_w):.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  Transformer baseline (K-context → one-step, NLL)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  Transformer baseline  K={K}  (2-fold CV)")
    print(f"{'═'*60}")
    t0_trf = time.time()
    trf_onestep, trf_loo_w = _train_transformer_cv(
        u_np, loo_neurons, K=K, n_folds=args.cv_folds,
        device=args.device, window_size=args.window_size,
        max_epochs=args.trf_epochs, patience=20, verbose=True,
    )
    trf_time = time.time() - t0_trf
    print(f"  Transformer done ({trf_time:.0f}s)")
    print(f"  One-step R²: mean={np.nanmean(trf_onestep):.4f}  "
          f"median={np.nanmedian(trf_onestep):.4f}")
    print(f"  LOO R² (w): mean={np.nanmean(trf_loo_w):.4f}  "
          f"median={np.nanmedian(trf_loo_w):.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  Retrain-LOO (batched): one model per fold, all neurons masked
    # ══════════════════════════════════════════════════════════════════
    t_total = time.time()

    print(f"\n{'═'*60}")
    print(f"  Retrain-LOO (BATCHED): {len(loo_neurons)} neurons masked × "
          f"{args.cv_folds} folds × {args.epochs} epochs")
    print(f"  Training {args.cv_folds} models (not {len(loo_neurons)}!)")
    print(f"{'═'*60}")

    retrain_loo_r2 = train_batched_retrain_loo(
        cfg, data, loo_neurons,
        n_folds=args.cv_folds,
        window_size=args.window_size,
        warmup_steps=args.warmup_steps,
        verbose=True,
    )

    total_time = time.time() - t_total
    rl_vals = retrain_loo_r2[loo_neurons]
    print(f"\n{'═'*60}")
    print(f"  Retrain-LOO complete  ({total_time:.0f}s = {total_time/60:.1f}min)")
    print(f"  R² mean  = {np.nanmean(rl_vals):.4f}")
    print(f"  R² median= {np.nanmedian(rl_vals):.4f}")
    delta = np.nanmean(rl_vals) - np.nanmean(s2_loo_w)
    print(f"  Stage2 LOO: mean={np.nanmean(s2_loo_w):.4f}")
    print(f"  Δ mean = {delta:+.4f}")
    print(f"{'═'*60}")

    # ══════════════════════════════════════════════════════════════════
    #  Conn-Ridge (per-neuron linear, connectome-restricted, K-context)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  Conn-Ridge (linear K={K})  (2-fold CV)")
    print(f"{'═'*60}")
    t0_cl = time.time()
    cl_onestep, cl_loo_w = _train_conn_trf_cv(
        u_np, labels, loo_neurons, K=K, n_folds=args.cv_folds,
        device=args.device, window_size=args.window_size,
        max_epochs=100, patience=15, verbose=True,
        model_type="linear", weight_decay=1e-4,
    )
    cl_time = time.time() - t0_cl
    print(f"  Conn-Ridge done ({cl_time:.0f}s)")
    print(f"  One-step R²: mean={np.nanmean(cl_onestep):.4f}  "
          f"median={np.nanmedian(cl_onestep):.4f}")
    print(f"  LOO R² (w): mean={np.nanmean(cl_loo_w):.4f}  "
          f"median={np.nanmedian(cl_loo_w):.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  Conn-MLP (per-neuron MLP, connectome-restricted, K-context)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  Conn-MLP (K={K})  (2-fold CV)")
    print(f"{'═'*60}")
    t0_cm = time.time()
    cm_onestep, cm_loo_w = _train_conn_trf_cv(
        u_np, labels, loo_neurons, K=K, n_folds=args.cv_folds,
        device=args.device, window_size=args.window_size,
        max_epochs=args.baseline_epochs, patience=15, verbose=True,
        model_type="mlp", weight_decay=0.01,
    )
    cm_time = time.time() - t0_cm
    print(f"  Conn-MLP done ({cm_time:.0f}s)")
    print(f"  One-step R²: mean={np.nanmean(cm_onestep):.4f}  "
          f"median={np.nanmedian(cm_onestep):.4f}")
    print(f"  LOO R² (w): mean={np.nanmean(cm_loo_w):.4f}  "
          f"median={np.nanmedian(cm_loo_w):.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  Conn-Transformer (per-neuron, connectome-restricted, K-context)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  Conn-Transformer  K={K}  wd=1e-4  (2-fold CV)")
    print(f"{'═'*60}")
    t0_ct = time.time()
    ct_onestep, ct_loo_w = _train_conn_trf_cv(
        u_np, labels, loo_neurons, K=K, n_folds=args.cv_folds,
        device=args.device, window_size=args.window_size,
        max_epochs=args.trf_epochs, patience=20, verbose=True,
        model_type="transformer", weight_decay=1e-4,
    )
    ct_time = time.time() - t0_ct
    print(f"  Conn-Trf done ({ct_time:.0f}s)")
    print(f"  One-step R²: mean={np.nanmean(ct_onestep):.4f}  "
          f"median={np.nanmedian(ct_onestep):.4f}")
    print(f"  LOO R² (w): mean={np.nanmean(ct_loo_w):.4f}  "
          f"median={np.nanmedian(ct_loo_w):.4f}")

    # Save intermediate results
    np.savez(save / "retrain_loo_progress.npz",
             retrain_loo_r2=retrain_loo_r2,
             loo_neurons=np.array(loo_neurons),
             completed=len(loo_neurons))

    # ── Save final results ──
    results = {
        "worm": worm,
        "K": K,
        "epochs": args.epochs,
        "cv_folds": args.cv_folds,
        "window_size": args.window_size,
        "loo_neurons": loo_neurons,
        # Stage2
        "stage2_onestep_r2": s2_onestep.tolist(),
        "stage2_loo_w": s2_loo_w.tolist(),
        "mean_stage2_loo": float(np.nanmean(s2_loo_w)),
        # Retrain-LOO
        "retrain_loo_r2": retrain_loo_r2.tolist(),
        "mean_retrain_loo": float(np.nanmean(rl_vals)),
        "median_retrain_loo": float(np.nanmedian(rl_vals)),
        # Ridge
        "ridge_onestep_r2": ridge_onestep.tolist(),
        "ridge_loo_w": ridge_loo_w.tolist(),
        "mean_ridge_loo": float(np.nanmean(ridge_loo_w)),
        # EN
        "en_onestep_r2": en_onestep.tolist(),
        "en_loo_w": en_loo_w.tolist(),
        "mean_en_loo": float(np.nanmean(en_loo_w)),
        # MLP
        "mlp_onestep_r2": mlp_onestep.tolist(),
        "mlp_loo_w": mlp_loo_w.tolist(),
        "mean_mlp_loo": float(np.nanmean(mlp_loo_w)),
        # Transformer
        "transformer_onestep_r2": trf_onestep.tolist(),
        "transformer_loo_w": trf_loo_w.tolist(),
        "mean_transformer_loo": float(np.nanmean(trf_loo_w)),
        # Conn-Ridge
        "conn_ridge_onestep_r2": cl_onestep.tolist(),
        "conn_ridge_loo_w": cl_loo_w.tolist(),
        "mean_conn_ridge_loo": float(np.nanmean(cl_loo_w)),
        # Conn-MLP
        "conn_mlp_onestep_r2": cm_onestep.tolist(),
        "conn_mlp_loo_w": cm_loo_w.tolist(),
        "mean_conn_mlp_loo": float(np.nanmean(cm_loo_w)),
        # Conn-Transformer
        "conn_trf_onestep_r2": ct_onestep.tolist(),
        "conn_trf_loo_w": ct_loo_w.tolist(),
        "mean_conn_trf_loo": float(np.nanmean(ct_loo_w)),
    }
    with open(save / "retrain_loo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {save / 'retrain_loo_results.json'}")

    # ── Assemble data for two-panel plot ──
    onestep_data: Dict[str, np.ndarray] = {}
    loo_data: Dict[str, np.ndarray] = {}

    onestep_data["Stage2"] = s2_onestep
    loo_data["Stage2"] = s2_loo_w

    loo_data["Retrain-LOO"] = retrain_loo_r2[loo_neurons]

    onestep_data["Ridge"] = ridge_onestep
    loo_data["Ridge"] = ridge_loo_w
    onestep_data["EN"] = en_onestep
    loo_data["EN"] = en_loo_w
    onestep_data["MLP"] = mlp_onestep
    loo_data["MLP"] = mlp_loo_w
    onestep_data["Transformer"] = trf_onestep
    loo_data["Transformer"] = trf_loo_w
    onestep_data["Conn-Ridge"] = cl_onestep
    loo_data["Conn-Ridge"] = cl_loo_w
    onestep_data["Conn-MLP"] = cm_onestep
    loo_data["Conn-MLP"] = cm_loo_w
    onestep_data["Conn-Trf"] = ct_onestep
    loo_data["Conn-Trf"] = ct_loo_w

    plot_bc(onestep_data, loo_data,
            save / "retrain_loo_comparison.png", worm, N, len(loo_neurons))


if __name__ == "__main__":
    main()
