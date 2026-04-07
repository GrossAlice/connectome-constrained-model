#!/usr/bin/env python3
"""
Stochastic autonomous free-run: 3 strategies × 3 models, distributional evaluation.

STRATEGIES:
  1. JOINT:         Predict [neural, beh] jointly, feed back both.
  2. NEURAL-AR+DEC: AR model on neural only → decode behaviour from predicted neural.
  3. CASCADED:      Neural AR (independent) + beh model (predicted neural + beh history).

MODELS (for each strategy):
  - TRF  (Transformer — native Gaussian σ for noise)
  - MLP  (residual σ from training residuals)
  - Ridge (residual σ from training residuals)

All free-runs are STOCHASTIC: x_{t+1} ~ N(μ, (T·σ)²).
Evaluation is distributional (PSD, ACF, Wasserstein, KS, VarRatio) — not R².

Usage:
  python -m scripts.free_run.run \\
      --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-06-14-01.h5" \\
      --out_dir output_plots/free_run --device cpu
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.config import TransformerBaselineConfig
from stage2.posture_videos import make_posture_compare_video
from baseline_transformer.dataset import load_worm_data
from baseline_transformer.train import train_single_worm_cv


def _make_B_wide_config(context_length: int = 15) -> TransformerBaselineConfig:
    """Create B_wide_256h8 transformer config (larger model)."""
    cfg = TransformerBaselineConfig()
    cfg.d_model = 256
    cfg.n_heads = 8
    cfg.n_layers = 2
    cfg.d_ff = 512
    cfg.dropout = 0.1
    cfg.context_length = context_length
    cfg.lr = 1e-3
    cfg.weight_decay = 1e-4
    cfg.sigma_min = 1e-4
    cfg.sigma_max = 10.0
    cfg.max_epochs = 200
    cfg.patience = 25
    return cfg
from scripts.free_run.utils import (
    build_lagged, train_mlp, train_mlp_rollout, train_mlp_beh_rollout, train_ridge, estimate_residual_std,
    compute_distributional_metrics, ensemble_median_metrics,
    compute_psd, compute_autocorr,
    plot_ensemble_traces, plot_psd_comparison, plot_autocorr_comparison,
    plot_marginals, plot_summary_bars, plot_temperature_sweep,
    NumpyEncoder,
)


# ═════════════════════════════════════════════════════════════════════════════
# Stochastic free-run loops (one function per strategy)
# ═════════════════════════════════════════════════════════════════════════════

# ── STRATEGY 1: JOINT ────────────────────────────────────────────────────────

@torch.no_grad()
def stochastic_joint_trf(model, seed, n_neural, n_steps, device, temperature=1.0,
                         clamp_range=None):
    """TRF Joint: native Gaussian head → sample N(μ, (T·σ)²).
    seed: (K, D) in the model's own scale (raw).
    clamp_range: optional (lo, hi) tensor/array to clamp each sample.
    Returns (pred_neural, pred_beh) each (n_steps, ·).
    """
    model = model.to(device).eval()
    K = model.cfg.context_length
    D = seed.shape[1]
    buf = torch.tensor(seed.copy(), dtype=torch.float32, device=device).unsqueeze(0)
    if clamp_range is not None:
        cl_lo = torch.tensor(clamp_range[0], dtype=torch.float32, device=device)
        cl_hi = torch.tensor(clamp_range[1], dtype=torch.float32, device=device)

    pred = np.zeros((n_steps, D), dtype=np.float32)
    for t in range(n_steps):
        mu_u, sig_u, mu_b, sig_b = model.forward(buf)
        mu = torch.cat([mu_u, mu_b], dim=-1) if mu_b is not None else mu_u
        sigma = torch.cat([sig_u, sig_b], dim=-1) if sig_b is not None else sig_u
        sample = mu + temperature * sigma * torch.randn_like(mu)
        if clamp_range is not None:
            sample = sample.clamp(cl_lo, cl_hi)
        pred[t] = sample[0].cpu().numpy()
        buf = torch.cat([buf[:, 1:, :], sample.unsqueeze(1)], dim=1)

    model.cpu()
    return pred[:, :n_neural], pred[:, n_neural:]


@torch.no_grad()
def stochastic_joint_mlp(mlp, seed, n_steps, n_neural, device,
                          residual_std, temperature=1.0, clamp_range=None):
    """MLP Joint: predict [neural, beh], add N(0, (T·σ_res)²)."""
    mlp = mlp.to(device).eval()
    K, D = seed.shape
    hist = [seed[K - 1 - lag].copy() for lag in range(K)]
    cl_lo = clamp_range[0] if clamp_range is not None else None
    cl_hi = clamp_range[1] if clamp_range is not None else None

    pred = np.zeros((n_steps, D), dtype=np.float32)
    for t in range(n_steps):
        x_lags = np.concatenate(hist)
        x_t = torch.tensor(x_lags, dtype=torch.float32, device=device).unsqueeze(0)
        mu = mlp(x_t)[0].cpu().numpy()
        noise = temperature * residual_std * np.random.randn(D).astype(np.float32)
        sample = mu + noise
        if clamp_range is not None:
            sample = np.clip(sample, cl_lo, cl_hi)
        pred[t] = sample
        hist = [sample.copy()] + hist[:-1]

    mlp.cpu()
    return pred[:, :n_neural], pred[:, n_neural:]


def stochastic_joint_ridge(model, seed, n_steps, n_neural,
                            residual_std, temperature=1.0, clamp_range=None):
    """Ridge Joint: predict [neural, beh], add N(0, (T·σ_res)²)."""
    K, D = seed.shape
    hist = [seed[K - 1 - lag].copy() for lag in range(K)]
    cl_lo = clamp_range[0] if clamp_range is not None else None
    cl_hi = clamp_range[1] if clamp_range is not None else None

    pred = np.zeros((n_steps, D), dtype=np.float32)
    for t in range(n_steps):
        x_lags = np.concatenate(hist)
        mu = model.predict(x_lags.reshape(1, -1))[0]
        noise = temperature * residual_std * np.random.randn(D).astype(np.float32)
        sample = mu + noise
        if clamp_range is not None:
            sample = np.clip(sample, cl_lo, cl_hi)
        pred[t] = sample
        hist = [sample.copy()] + hist[:-1]

    return pred[:, :n_neural], pred[:, n_neural:]


# ── STRATEGY 2: NEURAL-AR + DECODER ─────────────────────────────────────────

@torch.no_grad()
def stochastic_ar_dec_trf(neural_model, beh_decoder_model,
                           u_seed, b_seed, n_steps, n_neural, n_beh,
                           device, temperature=1.0, clamp_range_u=None, clamp_range_b=None):
    """TRF Neural-AR + TRF Beh Decoder.
    neural_model: TRF trained on neural-only → (μ_u, σ_u).
    beh_decoder_model: TRF trained on neural→beh → (_, _, μ_b, σ_b).
    """
    neural_model = neural_model.to(device).eval()
    beh_decoder_model = beh_decoder_model.to(device).eval()
    K = neural_model.cfg.context_length
    if clamp_range_u is not None:
        cl_u_lo = torch.tensor(clamp_range_u[0], dtype=torch.float32, device=device)
        cl_u_hi = torch.tensor(clamp_range_u[1], dtype=torch.float32, device=device)
    if clamp_range_b is not None:
        cl_b_lo = torch.tensor(clamp_range_b[0], dtype=torch.float32, device=device)
        cl_b_hi = torch.tensor(clamp_range_b[1], dtype=torch.float32, device=device)

    # Neural context: (K, N) padded to (K, N+0) — neural-only model
    neural_buf = torch.tensor(u_seed.copy(), dtype=torch.float32, device=device).unsqueeze(0)
    # Decoder context: (K, N+Kw) with predicted neural, zero beh (beh not fed back)
    dec_seed = np.concatenate([u_seed, np.zeros((K, n_beh), dtype=np.float32)], axis=1)
    dec_buf = torch.tensor(dec_seed, dtype=torch.float32, device=device).unsqueeze(0)

    pred_neural = np.zeros((n_steps, n_neural), dtype=np.float32)
    pred_beh = np.zeros((n_steps, n_beh), dtype=np.float32)

    for t in range(n_steps):
        # Step 1: neural AR
        mu_u, sig_u, _, _ = neural_model.forward(neural_buf)
        u_sample = mu_u + temperature * sig_u * torch.randn_like(mu_u)
        if clamp_range_u is not None:
            u_sample = u_sample.clamp(cl_u_lo, cl_u_hi)
        pred_neural[t] = u_sample[0].cpu().numpy()

        # Step 2: decode behaviour from predicted neural
        # Build decoder context frame: [predicted_neural, zeros_beh]
        dec_frame = torch.zeros(1, 1, n_neural + n_beh, device=device)
        dec_frame[0, 0, :n_neural] = u_sample[0]
        dec_buf = torch.cat([dec_buf[:, 1:, :], dec_frame], dim=1)

        _, _, mu_b, sig_b = beh_decoder_model.forward(dec_buf)
        b_sample = mu_b + temperature * sig_b * torch.randn_like(mu_b)
        if clamp_range_b is not None:
            b_sample = b_sample.clamp(cl_b_lo, cl_b_hi)
        pred_beh[t] = b_sample[0].cpu().numpy()

        # Update neural buffer
        neural_buf = torch.cat([neural_buf[:, 1:, :], u_sample.unsqueeze(1)], dim=1)

    neural_model.cpu()
    beh_decoder_model.cpu()
    return pred_neural, pred_beh


@torch.no_grad()
def stochastic_ar_dec_mlp(neural_ar, beh_decoder, u_seed, n_steps,
                           n_neural, n_beh, device,
                           res_std_u, res_std_b, temperature=1.0,
                           clamp_range_u=None, clamp_range_b=None):
    """MLP Neural-AR + MLP Decoder."""
    neural_ar = neural_ar.to(device).eval()
    beh_decoder = beh_decoder.to(device).eval()
    K = u_seed.shape[0]
    cl_u_lo = clamp_range_u[0] if clamp_range_u is not None else None
    cl_u_hi = clamp_range_u[1] if clamp_range_u is not None else None
    cl_b_lo = clamp_range_b[0] if clamp_range_b is not None else None
    cl_b_hi = clamp_range_b[1] if clamp_range_b is not None else None

    neural_hist = [u_seed[K - 1 - lag].copy() for lag in range(K)]

    pred_neural = np.zeros((n_steps, n_neural), dtype=np.float32)
    pred_beh = np.zeros((n_steps, n_beh), dtype=np.float32)

    for t in range(n_steps):
        u_lags = np.concatenate(neural_hist)
        u_t = torch.tensor(u_lags, dtype=torch.float32, device=device).unsqueeze(0)
        mu_u = neural_ar(u_t)[0].cpu().numpy()
        u_sample = mu_u + temperature * res_std_u * np.random.randn(n_neural).astype(np.float32)
        if clamp_range_u is not None:
            u_sample = np.clip(u_sample, cl_u_lo, cl_u_hi)
        pred_neural[t] = u_sample

        # Decode beh from (current + history) neural lags
        u_dec_lags = np.concatenate([u_sample] + neural_hist[:-1])
        u_dec_t = torch.tensor(u_dec_lags, dtype=torch.float32, device=device).unsqueeze(0)
        mu_b = beh_decoder(u_dec_t)[0].cpu().numpy()
        b_sample = mu_b + temperature * res_std_b * np.random.randn(n_beh).astype(np.float32)
        if clamp_range_b is not None:
            b_sample = np.clip(b_sample, cl_b_lo, cl_b_hi)
        pred_beh[t] = b_sample

        neural_hist = [u_sample.copy()] + neural_hist[:-1]

    neural_ar.cpu(); beh_decoder.cpu()
    return pred_neural, pred_beh


def stochastic_ar_dec_ridge(neural_ar, beh_decoder, u_seed, n_steps,
                             n_neural, n_beh,
                             res_std_u, res_std_b, temperature=1.0,
                             clamp_range_u=None, clamp_range_b=None):
    """Ridge Neural-AR + Ridge Decoder."""
    K = u_seed.shape[0]
    neural_hist = [u_seed[K - 1 - lag].copy() for lag in range(K)]
    cl_u_lo = clamp_range_u[0] if clamp_range_u is not None else None
    cl_u_hi = clamp_range_u[1] if clamp_range_u is not None else None
    cl_b_lo = clamp_range_b[0] if clamp_range_b is not None else None
    cl_b_hi = clamp_range_b[1] if clamp_range_b is not None else None

    pred_neural = np.zeros((n_steps, n_neural), dtype=np.float32)
    pred_beh = np.zeros((n_steps, n_beh), dtype=np.float32)

    for t in range(n_steps):
        u_lags = np.concatenate(neural_hist)
        mu_u = neural_ar.predict(u_lags.reshape(1, -1))[0]
        u_sample = mu_u + temperature * res_std_u * np.random.randn(n_neural).astype(np.float32)
        if clamp_range_u is not None:
            u_sample = np.clip(u_sample, cl_u_lo, cl_u_hi)
        pred_neural[t] = u_sample

        u_dec_lags = np.concatenate([u_sample] + neural_hist[:-1])
        mu_b = beh_decoder.predict(u_dec_lags.reshape(1, -1))[0]
        b_sample = mu_b + temperature * res_std_b * np.random.randn(n_beh).astype(np.float32)
        if clamp_range_b is not None:
            b_sample = np.clip(b_sample, cl_b_lo, cl_b_hi)
        pred_beh[t] = b_sample

        neural_hist = [u_sample.copy()] + neural_hist[:-1]

    return pred_neural, pred_beh


# ── STRATEGY 3: CASCADED ────────────────────────────────────────────────────

@torch.no_grad()
def stochastic_cascaded_trf(neural_model, beh_model,
                             u_seed, b_seed, n_steps, n_neural, n_beh,
                             device, temperature=1.0,
                             clamp_range_u=None, clamp_range_b=None):
    """TRF Cascaded: Neural AR (independent) + Beh model (neural + beh history).
    neural_model: TRF neural-only.
    beh_model: TRF that takes [neural, beh] context and predicts beh.
    """
    neural_model = neural_model.to(device).eval()
    beh_model = beh_model.to(device).eval()
    K = neural_model.cfg.context_length
    if clamp_range_u is not None:
        cl_u_lo = torch.tensor(clamp_range_u[0], dtype=torch.float32, device=device)
        cl_u_hi = torch.tensor(clamp_range_u[1], dtype=torch.float32, device=device)
    if clamp_range_b is not None:
        cl_b_lo = torch.tensor(clamp_range_b[0], dtype=torch.float32, device=device)
        cl_b_hi = torch.tensor(clamp_range_b[1], dtype=torch.float32, device=device)

    neural_buf = torch.tensor(u_seed.copy(), dtype=torch.float32, device=device).unsqueeze(0)
    # beh_model context: [predicted_neural, predicted_beh] joint
    joint_seed = np.concatenate([u_seed, b_seed], axis=1)
    beh_buf = torch.tensor(joint_seed.copy(), dtype=torch.float32, device=device).unsqueeze(0)

    pred_neural = np.zeros((n_steps, n_neural), dtype=np.float32)
    pred_beh = np.zeros((n_steps, n_beh), dtype=np.float32)

    for t in range(n_steps):
        # Step 1: neural AR (independent)
        mu_u, sig_u, _, _ = neural_model.forward(neural_buf)
        u_sample = mu_u + temperature * sig_u * torch.randn_like(mu_u)
        if clamp_range_u is not None:
            u_sample = u_sample.clamp(cl_u_lo, cl_u_hi)
        pred_neural[t] = u_sample[0].cpu().numpy()

        # Step 2: beh from [predicted_neural, beh_history]
        _, _, mu_b, sig_b = beh_model.forward(beh_buf)
        b_sample = mu_b + temperature * sig_b * torch.randn_like(mu_b)
        if clamp_range_b is not None:
            b_sample = b_sample.clamp(cl_b_lo, cl_b_hi)
        pred_beh[t] = b_sample[0].cpu().numpy()

        # Update buffers
        neural_buf = torch.cat([neural_buf[:, 1:, :], u_sample.unsqueeze(1)], dim=1)
        joint_frame = torch.cat([u_sample, b_sample], dim=-1).unsqueeze(1)
        beh_buf = torch.cat([beh_buf[:, 1:, :], joint_frame], dim=1)

    neural_model.cpu(); beh_model.cpu()
    return pred_neural, pred_beh


@torch.no_grad()
def stochastic_cascaded_mlp(neural_ar, beh_model, u_seed, b_seed, n_steps,
                             n_neural, n_beh, device,
                             res_std_u, res_std_b, temperature=1.0, K_beh=None,
                             clamp_range_u=None, clamp_range_b=None):
    """MLP Cascaded with separate K_beh for longer behavior context."""
    neural_ar = neural_ar.to(device).eval()
    beh_model = beh_model.to(device).eval()
    K = u_seed.shape[0]  # Neural lag
    if K_beh is None:
        K_beh = K  # Fall back to neural K
    cl_u_lo = clamp_range_u[0] if clamp_range_u is not None else None
    cl_u_hi = clamp_range_u[1] if clamp_range_u is not None else None
    cl_b_lo = clamp_range_b[0] if clamp_range_b is not None else None
    cl_b_hi = clamp_range_b[1] if clamp_range_b is not None else None

    neural_hist = [u_seed[K - 1 - lag].copy() for lag in range(K)]
    # Behavior history needs K_beh length - extend b_seed if needed
    if b_seed.shape[0] < K_beh:
        # Pad with zeros at the beginning
        b_pad = np.zeros((K_beh - b_seed.shape[0], n_beh), dtype=np.float32)
        b_seed_ext = np.vstack([b_pad, b_seed])
    else:
        b_seed_ext = b_seed[-K_beh:]
    beh_hist = [b_seed_ext[K_beh - 1 - lag].copy() for lag in range(K_beh)]

    pred_neural = np.zeros((n_steps, n_neural), dtype=np.float32)
    pred_beh = np.zeros((n_steps, n_beh), dtype=np.float32)

    for t in range(n_steps):
        u_lags = np.concatenate(neural_hist)
        u_t = torch.tensor(u_lags, dtype=torch.float32, device=device).unsqueeze(0)
        mu_u = neural_ar(u_t)[0].cpu().numpy()
        u_sample = mu_u + temperature * res_std_u * np.random.randn(n_neural).astype(np.float32)
        if clamp_range_u is not None:
            u_sample = np.clip(u_sample, cl_u_lo, cl_u_hi)
        pred_neural[t] = u_sample

        b_lags = np.concatenate(beh_hist)
        nb_input = np.concatenate([u_sample, b_lags])
        nb_t = torch.tensor(nb_input, dtype=torch.float32, device=device).unsqueeze(0)
        mu_b = beh_model(nb_t)[0].cpu().numpy()
        b_sample = mu_b + temperature * res_std_b * np.random.randn(n_beh).astype(np.float32)
        if clamp_range_b is not None:
            b_sample = np.clip(b_sample, cl_b_lo, cl_b_hi)
        pred_beh[t] = b_sample

        neural_hist = [u_sample.copy()] + neural_hist[:-1]
        beh_hist = [b_sample.copy()] + beh_hist[:-1]

    neural_ar.cpu(); beh_model.cpu()
    return pred_neural, pred_beh


def stochastic_cascaded_ridge(neural_ar, beh_model, u_seed, b_seed, n_steps,
                               n_neural, n_beh,
                               res_std_u, res_std_b, temperature=1.0,
                               clamp_range_u=None, clamp_range_b=None):
    """Ridge Cascaded."""
    K_neural = u_seed.shape[0]
    K_beh = b_seed.shape[0]
    neural_hist = [u_seed[K_neural - 1 - lag].copy() for lag in range(K_neural)]
    beh_hist = [b_seed[K_beh - 1 - lag].copy() for lag in range(K_beh)]
    cl_u_lo = clamp_range_u[0] if clamp_range_u is not None else None
    cl_u_hi = clamp_range_u[1] if clamp_range_u is not None else None
    cl_b_lo = clamp_range_b[0] if clamp_range_b is not None else None
    cl_b_hi = clamp_range_b[1] if clamp_range_b is not None else None

    pred_neural = np.zeros((n_steps, n_neural), dtype=np.float32)
    pred_beh = np.zeros((n_steps, n_beh), dtype=np.float32)

    for t in range(n_steps):
        u_lags = np.concatenate(neural_hist)
        mu_u = neural_ar.predict(u_lags.reshape(1, -1))[0]
        u_sample = mu_u + temperature * res_std_u * np.random.randn(n_neural).astype(np.float32)
        if clamp_range_u is not None:
            u_sample = np.clip(u_sample, cl_u_lo, cl_u_hi)
        pred_neural[t] = u_sample

        b_lags = np.concatenate(beh_hist)
        nb_input = np.concatenate([u_sample, b_lags])
        mu_b = beh_model.predict(nb_input.reshape(1, -1))[0]
        b_sample = mu_b + temperature * res_std_b * np.random.randn(n_beh).astype(np.float32)
        if clamp_range_b is not None:
            b_sample = np.clip(b_sample, cl_b_lo, cl_b_hi)
        pred_beh[t] = b_sample

        neural_hist = [u_sample.copy()] + neural_hist[:-1]
        beh_hist = [b_sample.copy()] + beh_hist[:-1]

    return pred_neural, pred_beh


# ═════════════════════════════════════════════════════════════════════════════
# TRF training wrappers (for neural-only and decoder models)
# ═════════════════════════════════════════════════════════════════════════════

def _train_trf_neural_only(u_train, K, device):
    """Train a TRF on neural-only (no behaviour channels)."""
    cfg = _make_B_wide_config(context_length=K)
    cfg.device = device
    dummy_b = np.zeros((u_train.shape[0], 0), dtype=np.float32)
    result = train_single_worm_cv(
        u=u_train, cfg=cfg, device=device, verbose=False,
        save_dir=None, worm_id="_neural_only", b=None, b_mask=None,
    )
    return result["best_model"], result


def _train_trf_decoder(u_train, b_train, K, device):
    """Train a TRF that takes [neural, beh=0] context → predicts behaviour.

    During training the neural channels are GT, beh channels are zeroed so
    the model learns a neural→beh mapping without leaking future beh.
    """
    from baseline_transformer.model import TemporalTransformerGaussian

    N, Kw = u_train.shape[1], b_train.shape[1]
    T = u_train.shape[0]

    # Normalize
    mu_u, sig_u = u_train[K:].mean(0), u_train[K:].std(0) + 1e-8
    mu_b, sig_b = b_train[K:].mean(0), b_train[K:].std(0) + 1e-8
    u_n = ((u_train - mu_u) / sig_u).astype(np.float32)
    b_n = ((b_train - mu_b) / sig_b).astype(np.float32)

    # Context: [u_normalized, zeros]; target: b_normalized
    x_ctx = np.concatenate([u_n, np.zeros_like(b_n)], axis=1)
    tr_idx = np.arange(K, T)
    X_win = np.stack([x_ctx[t - K:t] for t in tr_idx])
    y_tr = b_n[tr_idx]

    cfg = _make_B_wide_config(context_length=K)
    model = TemporalTransformerGaussian(n_neural=N, n_beh=Kw, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    nv = max(10, int(len(tr_idx) * 0.15))
    Xt = torch.tensor(X_win[:-nv], dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr[:-nv], dtype=torch.float32, device=device)
    Xv = torch.tensor(X_win[-nv:], dtype=torch.float32, device=device)
    yv = torch.tensor(y_tr[-nv:], dtype=torch.float32, device=device)

    bvl, bs, pat = float("inf"), None, 0
    for ep in range(200):
        model.train()
        mu_u_out, sig_u_out, mu_b_out, sig_b_out = model.forward(Xt)
        loss = nn.functional.gaussian_nll_loss(mu_b_out, yt, sig_b_out ** 2)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            _, _, mu_bv, sig_bv = model.forward(Xv)
            vl = nn.functional.gaussian_nll_loss(mu_bv, yv, sig_bv ** 2).item()
        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
        if pat > 25:
            break

    if bs:
        model.load_state_dict(bs)
    model.eval().cpu()
    return model, mu_u, sig_u, mu_b, sig_b


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out_dir", default="output_plots/free_run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--K", type=int, default=15, help="Neural context length")
    ap.add_argument("--K_beh", type=int, default=30, help="Behavior context length (longer for oscillations)")
    ap.add_argument("--rollout_beh", type=int, default=30, help="Rollout steps for behavior models")
    ap.add_argument("--neurons", choices=["motor", "all"], default="motor")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--temperatures", type=str, default="0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--video", action="store_true", help="Generate posture comparison video")
    ap.add_argument("--video_frames", type=int, default=500, help="Max frames for video")
    ap.add_argument("--strategy", choices=["all", "joint", "ar_dec", "cascaded"], default="all",
                    help="Which strategy to run (default: all)")
    args = ap.parse_args()

    device = args.device
    dt = 0.6
    K = args.K
    N_SAMPLES = args.n_samples
    temperatures = [float(t) for t in args.temperatures.split(",")]
    strategy = args.strategy  # "all", "joint", "ar_dec", "cascaded"

    # ── Load data ────────────────────────────────────────────────────────
    worm_data = load_worm_data(args.h5, n_beh_modes=6)
    u, b, worm_id = worm_data["u"], worm_data["b"], worm_data["worm_id"]
    motor_idx = worm_data.get("motor_idx")

    if args.neurons == "motor" and motor_idx is not None:
        u_sel = u[:, motor_idx]
        neuron_label = "motor"
    else:
        u_sel = u
        neuron_label = "all"

    T, N, Kw = u_sel.shape[0], u_sel.shape[1], b.shape[1]

    # ── Strict temporal split ────────────────────────────────────────────
    train_end = int(args.train_frac * T)
    n_test = T - train_end

    out_dir = Path(args.out_dir) / worm_id / neuron_label
    out_dir.mkdir(parents=True, exist_ok=True)

    K_beh = args.K_beh
    rollout_beh = args.rollout_beh
    print(f"Worm: {worm_id}  T={T}  N={N} ({neuron_label})  Kw={Kw}")
    print(f"SPLIT: train=[0,{train_end})  test=[{train_end},{T})")
    print(f"  K_neural={K}  K_beh={K_beh}  rollout_beh={rollout_beh}")
    print(f"  N_SAMPLES={N_SAMPLES}  Temperatures={temperatures}")
    print(f"Output: {out_dir}\n")

    # ── Normalisation (train only) ───────────────────────────────────────
    u_train, b_train = u_sel[:train_end], b[:train_end]
    mu_u = u_train[K:].mean(0).astype(np.float32)
    sig_u = u_train[K:].std(0).astype(np.float32) + 1e-8
    mu_b = b_train[K:].mean(0).astype(np.float32)
    sig_b = b_train[K:].std(0).astype(np.float32) + 1e-8

    u_norm = ((u_sel - mu_u) / sig_u).astype(np.float32)
    b_norm = ((b - mu_b) / sig_b).astype(np.float32)

    gt_neural_test = u_sel[train_end:T]
    gt_beh_test = b[train_end:T]

    # Normalised training targets
    u_norm_train = u_norm[:train_end]
    b_norm_train = b_norm[:train_end]
    D_joint = N + Kw

    # Lagged features (training)
    x_joint_train = np.concatenate([u_norm_train, b_norm_train], axis=1)
    X_u_lag = build_lagged(u_norm_train, K)
    X_b_lag = build_lagged(b_norm_train, K)
    X_j_lag = build_lagged(x_joint_train, K)

    tr_idx = np.arange(K, train_end)
    y_joint = x_joint_train[tr_idx]
    y_neural = u_norm_train[tr_idx]
    y_beh = b_norm_train[tr_idx]

    # Seeds (last K/K_beh of train for neural/beh respectively)
    u_seed = u_norm[train_end - K : train_end].copy()
    b_seed = b_norm[train_end - K_beh : train_end].copy()  # Use longer K_beh for behavior seed
    joint_seed = np.concatenate([u_seed, b_seed[-K:]], axis=1)  # Joint uses same K

    # Shared variables that may be needed by multiple strategy branches
    X_nb_casc = None
    y_beh_ext = None

    # ═════════════════════════════════════════════════════════════════════
    # TRAIN ALL MODELS
    # ═════════════════════════════════════════════════════════════════════

    # ---- TRF Joint ----
    print("=" * 60); print("Training: TRF Joint (B_wide)"); print("=" * 60)
    t0 = time.time()
    cfg = _make_B_wide_config(context_length=K); cfg.device = device
    trf_joint_result = train_single_worm_cv(
        u=u_sel[:train_end], cfg=cfg, device=device, verbose=False,
        save_dir=None, worm_id=worm_id, b=b[:train_end], b_mask=None,
    )
    trf_joint = trf_joint_result["best_model"]
    n_neural_trf = trf_joint_result["n_neural"]
    x_trf = trf_joint_result["x"]
    trf_seed = x_trf[train_end - K : train_end].copy()
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ---- TRF Neural-only (for strategies 2 & 3) ----
    trf_neural = trf_neural_result = trf_neural_seed = None
    if strategy in ("all", "ar_dec", "cascaded"):
        print("\nTraining: TRF Neural-only")
        t0 = time.time()
        trf_neural, trf_neural_result = _train_trf_neural_only(u_sel[:train_end], K, device)
        trf_neural_seed = trf_neural_result["x"][train_end - K : train_end].copy()
        print(f"  Done ({time.time()-t0:.1f}s)")

    # ---- TRF Decoder: neural→beh (for strategy 2) ----
    trf_decoder = dec_mu_u = dec_sig_u = dec_mu_b = dec_sig_b = None
    if strategy in ("all", "ar_dec"):
        print("\nTraining: TRF Decoder (neural→beh)")
        t0 = time.time()
        trf_decoder, dec_mu_u, dec_sig_u, dec_mu_b, dec_sig_b = _train_trf_decoder(
            u_sel[:train_end], b[:train_end], K, device)
        print(f"  Done ({time.time()-t0:.1f}s)")

    # ---- TRF Cascaded Beh model (for strategy 3) — same arch as joint ----
    trf_cascaded_beh = None
    if strategy in ("all", "cascaded"):
        print("\nTraining: TRF Cascaded Beh (joint arch, trained on full [u,b])")
        trf_cascaded_beh = trf_joint  # shares weights — same model, different loop

    # ---- MLP Joint (rollout training) ----
    print("\nTraining: MLP Joint (rollout=15)")
    t0 = time.time()
    mlp_joint = train_mlp_rollout(x_joint_train, K, device, rollout_steps=15)
    res_std_joint_mlp = estimate_residual_std(mlp_joint, X_j_lag[tr_idx], y_joint, device)
    print(f"  Done ({time.time()-t0:.1f}s)  σ_res neural={res_std_joint_mlp[:N].mean():.4f}  beh={res_std_joint_mlp[N:].mean():.4f}")

    # ---- MLP Neural AR (rollout training) ----
    mlp_neural_ar = res_std_u_mlp = None
    if strategy in ("all", "ar_dec", "cascaded"):
        print("\nTraining: MLP Neural AR (rollout=15)")
        t0 = time.time()
        mlp_neural_ar = train_mlp_rollout(u_norm_train, K, device, rollout_steps=15)
        res_std_u_mlp = estimate_residual_std(mlp_neural_ar, X_u_lag[tr_idx], y_neural, device)
        print(f"  Done ({time.time()-t0:.1f}s)  σ_res={res_std_u_mlp.mean():.4f}")

    # ---- MLP Beh Decoder (neural lags → beh) ----
    mlp_beh_dec = res_std_b_mlp_dec = None
    if strategy in ("all", "ar_dec"):
        print("\nTraining: MLP Beh Decoder (neural→beh)")
        t0 = time.time()
        mlp_beh_dec = train_mlp(X_u_lag[tr_idx], y_beh, device)
        res_std_b_mlp_dec = estimate_residual_std(mlp_beh_dec, X_u_lag[tr_idx], y_beh, device)
        print(f"  Done ({time.time()-t0:.1f}s)  σ_res={res_std_b_mlp_dec.mean():.4f}")

    # ---- MLP Cascaded Beh (neural_current + beh_lags → beh) with ROLLOUT ----
    mlp_beh_casc = res_std_b_mlp_casc = None
    if strategy in ("all", "cascaded"):
        print(f"\nTraining: MLP Cascaded Beh (neural + beh_lags → beh, K_beh={K_beh}, rollout={rollout_beh})")
        t0 = time.time()
        X_b_lag_ext = build_lagged(b_norm_train, K_beh)
        X_nb_casc = np.concatenate([u_norm_train[K_beh:], X_b_lag_ext[K_beh:]], axis=1)
        y_beh_ext = b_norm_train[K_beh:]
        mlp_beh_casc = train_mlp_beh_rollout(
            u_norm_train, b_norm_train, K_beh, device, 
            rollout_steps=rollout_beh, input_mode="cascaded"
        )
        res_std_b_mlp_casc = estimate_residual_std(mlp_beh_casc, X_nb_casc, y_beh_ext, device)
        print(f"  Done ({time.time()-t0:.1f}s)  σ_res={res_std_b_mlp_casc.mean():.4f}")

    # ---- Ridge Joint ----
    print("\nTraining: Ridge Joint")
    t0 = time.time()
    ridge_joint = train_ridge(X_j_lag[tr_idx], y_joint)
    res_std_joint_ridge = estimate_residual_std(ridge_joint, X_j_lag[tr_idx], y_joint, is_torch=False)
    print(f"  Done ({time.time()-t0:.1f}s)  σ_res neural={res_std_joint_ridge[:N].mean():.4f}  beh={res_std_joint_ridge[N:].mean():.4f}")

    # ---- Ridge Neural AR ----
    ridge_neural_ar = res_std_u_ridge = None
    if strategy in ("all", "ar_dec", "cascaded"):
        print("\nTraining: Ridge Neural AR")
        t0 = time.time()
        ridge_neural_ar = train_ridge(X_u_lag[tr_idx], y_neural)
        res_std_u_ridge = estimate_residual_std(ridge_neural_ar, X_u_lag[tr_idx], y_neural, is_torch=False)
        print(f"  Done ({time.time()-t0:.1f}s)  σ_res={res_std_u_ridge.mean():.4f}")

    # ---- Ridge Beh Decoder ----
    ridge_beh_dec = res_std_b_ridge_dec = None
    if strategy in ("all", "ar_dec"):
        print("\nTraining: Ridge Beh Decoder")
        t0 = time.time()
        ridge_beh_dec = train_ridge(X_u_lag[tr_idx], y_beh)
        res_std_b_ridge_dec = estimate_residual_std(ridge_beh_dec, X_u_lag[tr_idx], y_beh, is_torch=False)
        print(f"  Done ({time.time()-t0:.1f}s)  σ_res={res_std_b_ridge_dec.mean():.4f}")

    # ---- Ridge Cascaded Beh ----
    ridge_beh_casc = res_std_b_ridge_casc = None
    if strategy in ("all", "cascaded"):
        print("\nTraining: Ridge Cascaded Beh")
        t0 = time.time()
        if X_nb_casc is None:
            X_b_lag_ext = build_lagged(b_norm_train, K_beh)
            X_nb_casc = np.concatenate([u_norm_train[K_beh:], X_b_lag_ext[K_beh:]], axis=1)
            y_beh_ext = b_norm_train[K_beh:]
        ridge_beh_casc = train_ridge(X_nb_casc, y_beh_ext)
        res_std_b_ridge_casc = estimate_residual_std(ridge_beh_casc, X_nb_casc, y_beh_ext, is_torch=False)
        print(f"  Done ({time.time()-t0:.1f}s)  σ_res={res_std_b_ridge_casc.mean():.4f}")

    # ═════════════════════════════════════════════════════════════════════
    # GENERATE STOCHASTIC SAMPLES  (T=1.0)
    # ═════════════════════════════════════════════════════════════════════

    # Registry:  label → (gen_fn, gen_kwargs)
    configs = {}

    # ---- Strategy 1: Joint ----
    if strategy in ("all", "joint"):
        configs["TRF Joint"] = dict(
            fn=stochastic_joint_trf,
            kw=dict(model=trf_joint, seed=trf_seed, n_neural=n_neural_trf,
                    n_steps=n_test, device=device),
            denorm_u=(1, 0), denorm_b=(1, 0),
        )
        configs["MLP Joint"] = dict(
            fn=stochastic_joint_mlp,
            kw=dict(mlp=mlp_joint, seed=joint_seed, n_steps=n_test,
                    n_neural=N, device=device, residual_std=res_std_joint_mlp),
            denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b),
        )
        configs["Ridge Joint"] = dict(
            fn=stochastic_joint_ridge,
            kw=dict(model=ridge_joint, seed=joint_seed, n_steps=n_test,
                    n_neural=N, residual_std=res_std_joint_ridge),
            denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b),
        )

    # ---- Strategy 2: Neural-AR + Decoder ----
    if strategy in ("all", "ar_dec"):
        configs["MLP AR+Dec"] = dict(
            fn=stochastic_ar_dec_mlp,
            kw=dict(neural_ar=mlp_neural_ar, beh_decoder=mlp_beh_dec,
                    u_seed=u_seed, n_steps=n_test,
                    n_neural=N, n_beh=Kw, device=device,
                    res_std_u=res_std_u_mlp, res_std_b=res_std_b_mlp_dec),
            denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b),
        )
        configs["Ridge AR+Dec"] = dict(
            fn=stochastic_ar_dec_ridge,
            kw=dict(neural_ar=ridge_neural_ar, beh_decoder=ridge_beh_dec,
                    u_seed=u_seed, n_steps=n_test,
                    n_neural=N, n_beh=Kw,
                    res_std_u=res_std_u_ridge, res_std_b=res_std_b_ridge_dec),
            denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b),
        )

    # ---- Strategy 3: Cascaded ----
    if strategy in ("all", "cascaded"):
        configs["MLP Cascaded"] = dict(
            fn=stochastic_cascaded_mlp,
            kw=dict(neural_ar=mlp_neural_ar, beh_model=mlp_beh_casc,
                    u_seed=u_seed, b_seed=b_seed, n_steps=n_test,
                    n_neural=N, n_beh=Kw, device=device,
                    res_std_u=res_std_u_mlp, res_std_b=res_std_b_mlp_casc,
                    K_beh=K_beh),
            denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b),
        )
        configs["Ridge Cascaded"] = dict(
            fn=stochastic_cascaded_ridge,
            kw=dict(neural_ar=ridge_neural_ar, beh_model=ridge_beh_casc,
                    u_seed=u_seed, b_seed=b_seed, n_steps=n_test,
                    n_neural=N, n_beh=Kw,
                    res_std_u=res_std_u_ridge, res_std_b=res_std_b_ridge_casc),
            denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b),
        )

    # ── TRF AR+Dec and TRF Cascaded require special seed handling ────────
    if strategy in ("all", "ar_dec") and trf_neural is not None and trf_decoder is not None:
        dec_u_seed_norm = ((u_sel[train_end - K:train_end] - dec_mu_u) / dec_sig_u).astype(np.float32)
        configs["TRF AR+Dec"] = dict(
            fn=stochastic_ar_dec_trf,
            kw=dict(neural_model=trf_neural, beh_decoder_model=trf_decoder,
                    u_seed=trf_neural_seed,
                    b_seed=b_seed,
                    n_steps=n_test, n_neural=trf_neural_result["n_neural"],
                    n_beh=Kw, device=device),
            denorm_u=(1, 0),
            denorm_b=(dec_sig_b, dec_mu_b),
        )
    if strategy in ("all", "cascaded") and trf_neural is not None and trf_cascaded_beh is not None:
        configs["TRF Cascaded"] = dict(
            fn=stochastic_cascaded_trf,
            kw=dict(neural_model=trf_neural, beh_model=trf_cascaded_beh,
                    u_seed=trf_neural_seed,
                    b_seed=trf_seed[:, n_neural_trf:],
                    n_steps=n_test, n_neural=trf_neural_result["n_neural"],
                    n_beh=Kw, device=device),
            denorm_u=(1, 0),
            denorm_b=(1, 0),
        )

    # ── Run all configs at T=1.0 ─────────────────────────────────────────
    all_metrics = {}
    all_samples_beh = {}   # label → list of (n_test, Kw)
    all_samples_neural = {}

    print("\n" + "=" * 60)
    print(f"Generating {N_SAMPLES} stochastic samples per config (T=1.0)")
    print("=" * 60)

    for label, spec in configs.items():
        fn, kw = spec["fn"], spec["kw"]
        s_u, off_u = spec["denorm_u"]
        s_b, off_b = spec["denorm_b"]

        print(f"\n  {label}:")
        samples_neural, samples_beh = [], []
        for i in range(N_SAMPLES):
            pn, pb = fn(temperature=1.0, **kw)
            # Denormalise to raw scale
            pn_raw = np.asarray(pn, dtype=np.float64) * s_u + off_u
            pb_raw = np.asarray(pb, dtype=np.float64) * s_b + off_b
            samples_neural.append(pn_raw.astype(np.float32))
            samples_beh.append(pb_raw.astype(np.float32))
            if (i + 1) % 5 == 0:
                print(f"    sample {i+1}/{N_SAMPLES}")

        all_samples_neural[label] = samples_neural
        all_samples_beh[label] = samples_beh

        # Distributional metrics (median over ensemble)
        met_list = []
        for s in samples_beh:
            m, _ = compute_distributional_metrics(gt_beh_test, s)
            met_list.append(m)
        med = ensemble_median_metrics(met_list)
        all_metrics[label] = med
        compute_distributional_metrics(gt_beh_test, samples_beh[0], label=label)

    # ═════════════════════════════════════════════════════════════════════
    # TEMPERATURE SWEEP for TRF Joint
    # ═════════════════════════════════════════════════════════════════════
    trf_temp_results = {}
    if "TRF Joint" in configs:
        print("\n" + "=" * 60)
        print(f"Temperature sweep (TRF Joint): {temperatures}")
        print("=" * 60)

        for temp in temperatures:
            print(f"\n  T={temp}:")
            temp_samples = []
            for i in range(N_SAMPLES):
                _, pb = stochastic_joint_trf(
                    trf_joint, trf_seed, n_neural_trf, n_test, device,
                    temperature=temp)
                temp_samples.append(pb)
            met_list = [compute_distributional_metrics(gt_beh_test, s)[0] for s in temp_samples]
            trf_temp_results[temp] = ensemble_median_metrics(met_list)
            compute_distributional_metrics(gt_beh_test, temp_samples[0],
                                           label=f"TRF Joint T={temp}")

    # ═════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Generating plots")
    print("=" * 60)

    n_show = min(300, n_test)
    fs = 1.0 / dt
    model_names_ordered = list(configs.keys())

    # Ensemble traces (one per config)
    for label, samples in all_samples_beh.items():
        plot_ensemble_traces(gt_beh_test, samples, out_dir, label.replace(" ", "_"),
                             worm_id, n_show=n_show, dt=dt)

    # PSD comparison (all configs together)
    f_gt, psd_gt = compute_psd(gt_beh_test, fs=fs)
    all_psd_gen = []
    for label in model_names_ordered:
        psds = [compute_psd(s, fs=fs)[1] for s in all_samples_beh[label]]
        all_psd_gen.append(psds)
    plot_psd_comparison(f_gt, psd_gt, all_psd_gen, model_names_ordered,
                        out_dir, worm_id)

    # Autocorrelation comparison
    acf_gt = compute_autocorr(gt_beh_test)
    all_acf_gen = []
    for label in model_names_ordered:
        acfs = [compute_autocorr(s) for s in all_samples_beh[label]]
        all_acf_gen.append(acfs)
    plot_autocorr_comparison(acf_gt, all_acf_gen, model_names_ordered,
                             out_dir, worm_id, dt=dt)

    # Marginals
    plot_marginals(gt_beh_test,
                   [all_samples_beh[n] for n in model_names_ordered],
                   model_names_ordered, out_dir, worm_id)

    # Summary bars
    plot_summary_bars(all_metrics, out_dir, worm_id)

    # Temperature sweep
    plot_temperature_sweep(trf_temp_results, out_dir, worm_id, "TRF_Joint")

    # ═════════════════════════════════════════════════════════════════════
    # POSTURE VIDEO (optional)
    # ═════════════════════════════════════════════════════════════════════
    if args.video:
        print("\n" + "=" * 60)
        print("Generating posture comparison videos")
        print("=" * 60)
        
        # Generate videos for best models (TRF and MLP variants)
        video_configs = [
            ("TRF Joint", all_samples_beh.get("TRF Joint", [])),
            ("TRF AR+Dec", all_samples_beh.get("TRF AR+Dec", [])),
            ("TRF Cascaded", all_samples_beh.get("TRF Cascaded", [])),
            ("MLP Joint", all_samples_beh.get("MLP Joint", [])),
            ("MLP AR+Dec", all_samples_beh.get("MLP AR+Dec", [])),
            ("MLP Cascaded", all_samples_beh.get("MLP Cascaded", [])),
        ]
        
        for model_name, samples in video_configs:
            if not samples:
                continue
            # Use first sample for video
            beh_gen = samples[0]  # (n_test, Kw)
            
            # Trim to video_frames
            n_vid = min(args.video_frames, beh_gen.shape[0], gt_beh_test.shape[0])
            
            ew_gt = gt_beh_test[:n_vid]      # ground truth eigenworms
            ew_gen = beh_gen[:n_vid]          # generated eigenworms
            
            video_path = str(out_dir / f"{model_name.replace(' ', '_')}_posture.mp4")
            try:
                make_posture_compare_video(
                    h5_path=args.h5,
                    out_path=video_path,
                    ew_stage1=ew_gt,      # GT as "motor raw"
                    ew_model_cv=ew_gen,   # Generated as "motor model"
                    max_frames=n_vid,
                    fps=15,
                )
                print(f"  ✓ {model_name}: {video_path}")
            except Exception as e:
                print(f"  ✗ {model_name}: {e}")

    # ═════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═════════════════════════════════════════════════════════════════════
    save_data = {
        "worm_id": worm_id,
        "T": T, "N": N, "Kw": Kw, "K": K,
        "K_beh": K_beh, "rollout_beh": rollout_beh,
        "train_end": train_end, "n_test": n_test,
        "n_samples": N_SAMPLES,
        "metrics_T1": all_metrics,
        "temperature_sweep_TRF_Joint": trf_temp_results,
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {json_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY (behaviour, median over ensemble, T=1.0)")
    print("=" * 60)
    print(f"{'Config':<22} {'PSD↓':>8} {'ACF↓':>8} {'W1↓':>8} {'KS↓':>8} {'VarR→1':>8}")
    print("-" * 64)
    for name in model_names_ordered:
        m = all_metrics[name]
        print(f"{name:<22} {m['psd_log_distance']:>8.3f} {m['autocorr_rmse']:>8.3f} "
              f"{m['wasserstein_1']:>8.3f} {m['ks_statistic']:>8.3f} "
              f"{m['variance_ratio_mean']:>8.3f}")

    print(f"\nDone! Plots in {out_dir}")


if __name__ == "__main__":
    main()
