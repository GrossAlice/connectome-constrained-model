#!/usr/bin/env python3
"""
Set Transformer Joint Free-Run — batch over all worms.

Architecture
------------
Set Transformer with neuron identity embeddings (from 302-neuron atlas)
for joint (neural + behaviour) prediction.

Key features:
  - Neurons processed as a *set* (permutation-invariant via self-attention)
  - Per-neuron Gaussian heads (μ, σ) for neural activity
  - PMA-pooled Gaussian head for behaviour (eigenworms)
  - Fully autoregressive: predictions fed back as next context

Per worm
--------
  1. Train SetTransformerJoint on first 50 % of data
  2. Stochastic free-run on second 50 % (N_SAMPLES trajectories)
  3. Generate:
       • 20-second posture comparison video  (GT vs generated)
       • Neural activity trace plots         (GT vs generated ensemble)
       • Eigenworm trace plots               (GT vs generated ensemble)
       • Distributional metrics              (PSD, ACF, Wasserstein, KS, VarRatio)

Usage
-----
    # Single worm
    python -m scripts.free_run.run_set_transformer_joint \\
        --h5 "data/used/.../2022-06-14-01.h5" --device cuda

    # All worms
    python -m scripts.free_run.run_set_transformer_joint \\
        --h5_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \\
        --device cuda
"""
from __future__ import annotations

import argparse, glob, json, sys, time, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.dataset import load_worm_data
from stage2.posture_videos import make_posture_compare_video
from stage2.io_multi import _load_full_atlas
from stage2.io_h5 import _recover_labels_to_atlas
from scripts.free_run.utils import (
    compute_distributional_metrics, ensemble_median_metrics,
    compute_psd, compute_autocorr,
    plot_psd_comparison, plot_autocorr_comparison,
    plot_marginals, plot_summary_bars,
    NumpyEncoder,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SetTRFJointConfig:
    """Configuration for Set Transformer Joint model."""
    # Atlas / identity embeddings
    n_atlas: int = 302
    neuron_embed_dim: int = 64

    # Architecture (B_wide–scale)
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1

    # Context length
    context_length: int = 15

    # Gaussian heads
    sigma_min: float = 1e-4
    sigma_max: float = 10.0

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 25
    batch_size: int = 64


# ══════════════════════════════════════════════════════════════════════════════
# SET TRANSFORMER BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class _MAB(nn.Module):
    """Multihead Attention Block (pre-norm)."""
    def __init__(self, d: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d), nn.Dropout(dropout),
        )

    def forward(self, Q, K, mask=None):
        Qn, Kn = self.n1(Q), self.n1(K)
        H = Q + self.attn(Qn, Kn, Kn, key_padding_mask=mask)[0]
        return H + self.ff(self.n2(H))


class _SAB(nn.Module):
    """Set Attention Block — self-attention over set elements."""
    def __init__(self, d, h, d_ff, dropout=0.1):
        super().__init__()
        self.mab = _MAB(d, h, d_ff, dropout)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask)


class _PMA(nn.Module):
    """Pooling by Multihead Attention — aggregate set to fixed size."""
    def __init__(self, d, h, d_ff, n_out=1, dropout=0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_out, d) * 0.02)
        self.mab = _MAB(d, h, d_ff, dropout)

    def forward(self, X, mask=None):
        S = self.seeds.expand(X.size(0), -1, -1)
        return self.mab(S, X, mask)


# ══════════════════════════════════════════════════════════════════════════════
# SET TRANSFORMER JOINT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class SetTransformerJoint(nn.Module):
    """Set Transformer for **joint** neural + behaviour prediction.

    Inputs
    ------
    neural : (B, N, K)  — K-step history per neuron
    beh    : (B, K, Kw) — K-step behaviour history

    Outputs
    -------
    mu_n, sig_n : (B, N)  — per-neuron Gaussian params
    mu_b, sig_b : (B, Kw) — per-mode  Gaussian params
    """

    def __init__(self, n_neurons: int, n_beh: int,
                 atlas_indices: np.ndarray, cfg: SetTRFJointConfig):
        super().__init__()
        self.cfg = cfg
        self.n_neurons = n_neurons
        self.n_beh = n_beh
        K = cfg.context_length

        # Store atlas indices (constant per worm)
        self.register_buffer("atlas_idx",
                             torch.tensor(atlas_indices, dtype=torch.long))

        # Neuron identity embeddings (shared 302-neuron atlas)
        self.neuron_embed = nn.Embedding(cfg.n_atlas, cfg.neuron_embed_dim)

        # Project [activity(K) ‖ embed] → d_model
        self.neural_proj = nn.Sequential(
            nn.Linear(K + cfg.neuron_embed_dim, cfg.d_model),
            nn.GELU(), nn.LayerNorm(cfg.d_model),
        )

        # Behaviour context → d_model
        self.beh_proj = nn.Sequential(
            nn.Linear(K * n_beh, cfg.d_model),
            nn.GELU(), nn.LayerNorm(cfg.d_model),
        )

        # Encoder: self-attention over neuron set
        self.encoder = nn.ModuleList([
            _SAB(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_encoder_layers)
        ])

        # ── Per-neuron Gaussian head (neural prediction) ──
        self.neural_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2), nn.GELU(),
            nn.Linear(cfg.d_model // 2, 2),  # (mu, log_sigma)
        )

        # ── PMA pooling → behaviour head ──
        self.pma = _PMA(cfg.d_model, cfg.n_heads, cfg.d_ff,
                        n_out=1, dropout=cfg.dropout)
        self.beh_head = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.d_model, cfg.d_model),
            nn.GELU(), nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 2 * n_beh),  # (mu, log_sigma) per mode
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, neural: torch.Tensor, beh: torch.Tensor):
        B, N, K = neural.shape
        Kw = self.n_beh
        smin, smax = self.cfg.sigma_min, self.cfg.sigma_max

        # ── Neuron embedding ──
        emb = self.neuron_embed(self.atlas_idx)        # (N, E)
        emb = emb.unsqueeze(0).expand(B, -1, -1)      # (B, N, E)
        x = torch.cat([neural, emb], dim=-1)           # (B, N, K+E)
        x = self.neural_proj(x)                        # (B, N, d)

        # ── Encoder ──
        for layer in self.encoder:
            x = layer(x)                               # (B, N, d)

        # ── Neural predictions (per-neuron) ──
        n_out = self.neural_head(x)                    # (B, N, 2)
        mu_n = n_out[:, :, 0]                          # (B, N)
        sig_n = n_out[:, :, 1].exp().clamp(smin, smax) # (B, N)

        # ── Behaviour prediction ──
        pooled = self.pma(x).squeeze(1)                # (B, d)
        beh_enc = self.beh_proj(beh.reshape(B, -1))    # (B, d)
        b_out = self.beh_head(torch.cat([pooled, beh_enc], -1))  # (B, 2Kw)
        mu_b = b_out[:, :Kw]
        sig_b = b_out[:, Kw:].exp().clamp(smin, smax)

        return mu_n, sig_n, mu_b, sig_b


# ══════════════════════════════════════════════════════════════════════════════
# ATLAS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_atlas_indices(labels: List[str]) -> np.ndarray:
    """Map neuron labels → 302-neuron atlas indices (deterministic fallback)."""
    try:
        full_atlas = _load_full_atlas()
    except FileNotFoundError:
        return np.arange(len(labels), dtype=np.int64) % 302

    recovered = _recover_labels_to_atlas(labels, full_atlas)
    atlas_map = {n: i for i, n in enumerate(full_atlas)}
    indices = []
    for lab in recovered:
        if lab in atlas_map:
            indices.append(atlas_map[lab])
        else:
            # Deterministic fallback — use hash mod 302
            indices.append(abs(hash(lab)) % 302)
    return np.array(indices, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_set_trf_joint(
    u_train: np.ndarray,   # (T, N) normalised
    b_train: np.ndarray,   # (T, Kw) normalised
    atlas_idx: np.ndarray, # (N,) int64
    cfg: SetTRFJointConfig,
    device: str,
) -> SetTransformerJoint:
    """Train the Set Transformer Joint model on one worm."""
    T, N = u_train.shape
    Kw = b_train.shape[1]
    K = cfg.context_length

    # ── Build windows ──
    t_idx = np.arange(K, T)
    n_samples = len(t_idx)
    neural_ctx = np.stack([u_train[t - K:t].T for t in t_idx])   # (S, N, K)
    beh_ctx    = np.stack([b_train[t - K:t]   for t in t_idx])   # (S, K, Kw)
    neural_tgt = u_train[t_idx]                                   # (S, N)
    beh_tgt    = b_train[t_idx]                                   # (S, Kw)

    # ── Train / val split (temporal) ──
    nv = max(10, int(n_samples * 0.15))
    Xn_t = torch.tensor(neural_ctx[:-nv], dtype=torch.float32, device=device)
    Xb_t = torch.tensor(beh_ctx[:-nv],    dtype=torch.float32, device=device)
    Yn_t = torch.tensor(neural_tgt[:-nv], dtype=torch.float32, device=device)
    Yb_t = torch.tensor(beh_tgt[:-nv],    dtype=torch.float32, device=device)

    Xn_v = torch.tensor(neural_ctx[-nv:], dtype=torch.float32, device=device)
    Xb_v = torch.tensor(beh_ctx[-nv:],    dtype=torch.float32, device=device)
    Yn_v = torch.tensor(neural_tgt[-nv:], dtype=torch.float32, device=device)
    Yb_v = torch.tensor(beh_tgt[-nv:],    dtype=torch.float32, device=device)

    n_train = Xn_t.shape[0]

    # ── Model ──
    model = SetTransformerJoint(N, Kw, atlas_idx, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    SetTRF Joint: {n_params:,} params  "
          f"(d={cfg.d_model}, h={cfg.n_heads}, L={cfg.n_encoder_layers})")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    best_vl, best_sd, patience_ctr = float("inf"), None, 0

    for ep in range(cfg.max_epochs):
        # ── Train (mini-batch) ──
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            mu_n, sig_n, mu_b, sig_b = model(Xn_t[idx], Xb_t[idx])
            loss_n = F.gaussian_nll_loss(mu_n, Yn_t[idx], sig_n ** 2)
            loss_b = F.gaussian_nll_loss(mu_b, Yb_t[idx], sig_b ** 2)
            loss = loss_n + loss_b
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train
        sched.step()

        # ── Val ──
        model.eval()
        with torch.no_grad():
            mu_n_v, sig_n_v, mu_b_v, sig_b_v = model(Xn_v, Xb_v)
            vl = (F.gaussian_nll_loss(mu_n_v, Yn_v, sig_n_v ** 2)
                  + F.gaussian_nll_loss(mu_b_v, Yb_v, sig_b_v ** 2)).item()

        if vl < best_vl - 1e-6:
            best_vl = vl
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (ep + 1) % 25 == 0:
            print(f"      ep {ep+1:3d}  train={epoch_loss:.4f}  "
                  f"val={vl:.4f}  best={best_vl:.4f}")

        if patience_ctr >= cfg.patience:
            print(f"      Early stop at epoch {ep+1}")
            break

    if best_sd:
        model.load_state_dict(best_sd)
    model.eval().cpu()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STOCHASTIC FREE-RUN
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def stochastic_joint_settrf(
    model: SetTransformerJoint,
    u_seed: np.ndarray,   # (K, N) normalised
    b_seed: np.ndarray,   # (K, Kw) normalised
    n_steps: int,
    device: str,
    temperature: float = 1.0,
    clamp_range_u=None,
    clamp_range_b=None,
):
    """Autoregressive stochastic free-run with SetTransformerJoint.

    Returns (pred_neural, pred_beh)  each (n_steps, ·).
    """
    model = model.to(device).eval()
    K = model.cfg.context_length
    N = model.n_neurons
    Kw = model.n_beh

    u_hist = u_seed.copy()   # (K, N)
    b_hist = b_seed.copy()   # (K, Kw)

    pred_n = np.zeros((n_steps, N),  dtype=np.float32)
    pred_b = np.zeros((n_steps, Kw), dtype=np.float32)

    for t in range(n_steps):
        # Context tensors — (1, N, K) and (1, K, Kw)
        n_in = torch.tensor(u_hist.T[np.newaxis],
                            dtype=torch.float32, device=device)
        b_in = torch.tensor(b_hist[np.newaxis],
                            dtype=torch.float32, device=device)

        mu_n, sig_n, mu_b, sig_b = model(n_in, b_in)

        # Sample
        u_samp = (mu_n + temperature * sig_n *
                  torch.randn_like(mu_n))[0].cpu().numpy()
        b_samp = (mu_b + temperature * sig_b *
                  torch.randn_like(mu_b))[0].cpu().numpy()

        # Clamp
        if clamp_range_u is not None:
            u_samp = np.clip(u_samp, clamp_range_u[0], clamp_range_u[1])
        if clamp_range_b is not None:
            b_samp = np.clip(b_samp, clamp_range_b[0], clamp_range_b[1])

        pred_n[t] = u_samp
        pred_b[t] = b_samp

        # Shift context
        u_hist = np.vstack([u_hist[1:], u_samp[np.newaxis]])
        b_hist = np.vstack([b_hist[1:], b_samp[np.newaxis]])

    model.cpu()
    return pred_n, pred_b


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING — neural activity traces
# ══════════════════════════════════════════════════════════════════════════════

def plot_neural_traces(
    gt: np.ndarray,            # (T, N)
    samples: list[np.ndarray], # list of (T, N)
    out_dir: Path,
    worm_id: str,
    labels: list[str] | None = None,
    n_show_neurons: int = 8,
    n_show_frames: int = 200,
    dt: float = 0.6,
):
    """GT vs generated ensemble for select neurons."""
    T, N = gt.shape
    show_idx = np.linspace(0, N - 1, min(n_show_neurons, N), dtype=int)
    n_frames = min(n_show_frames, T, samples[0].shape[0])
    t_sec = np.arange(n_frames) * dt

    n_rows = len(show_idx)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for row, ni in enumerate(show_idx):
        ax = axes[row]
        # Ensemble
        for s in samples[:10]:
            ax.plot(t_sec, s[:n_frames, ni], color="#E24A33",
                    alpha=0.12, lw=0.6)
        ens_mean = np.mean([s[:n_frames, ni] for s in samples], axis=0)
        ax.plot(t_sec, ens_mean, color="#E24A33", lw=1.5, ls="--",
                alpha=0.8, label="SetTRF (mean)")
        # GT
        ax.plot(t_sec, gt[:n_frames, ni], color="#333",
                lw=1.2, alpha=0.9, label="GT")

        nlab = labels[ni] if labels and ni < len(labels) else f"n{ni}"
        ax.set_ylabel(nlab, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15)
        if row == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(f"Neural Activity Traces — SetTRF Joint — {worm_id}\n"
                 f"{len(samples)} samples, {N} neurons",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"neural_traces_{worm_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING — eigenworm traces
# ══════════════════════════════════════════════════════════════════════════════

def plot_ew_traces(
    gt: np.ndarray,            # (T, Kw)
    samples: list[np.ndarray], # list of (T, Kw)
    out_dir: Path,
    worm_id: str,
    n_show_frames: int = 200,
    dt: float = 0.6,
):
    """GT vs generated ensemble for all eigenworm modes."""
    T, Kw = gt.shape
    n_frames = min(n_show_frames, T, samples[0].shape[0])
    t_sec = np.arange(n_frames) * dt

    fig, axes = plt.subplots(Kw, 1, figsize=(14, 2 * Kw), sharex=True)
    if Kw == 1:
        axes = [axes]

    for mode in range(Kw):
        ax = axes[mode]
        for s in samples[:10]:
            ax.plot(t_sec, s[:n_frames, mode], color="#348ABD",
                    alpha=0.12, lw=0.6)
        ens_mean = np.mean([s[:n_frames, mode] for s in samples], axis=0)
        ax.plot(t_sec, ens_mean, color="#348ABD", lw=1.5, ls="--",
                alpha=0.8, label="SetTRF (mean)")
        ax.plot(t_sec, gt[:n_frames, mode], color="#333",
                lw=1.2, alpha=0.9, label="GT")

        ax.set_ylabel(f"EW{mode+1}", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15)
        if mode == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(f"Eigenworm Traces — SetTRF Joint — {worm_id}\n"
                 f"{len(samples)} samples",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"ew_traces_{worm_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-WORM PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_one_worm(
    h5_path: str,
    cfg: SetTRFJointConfig,
    device: str,
    base_out: Path,
    neurons: str = "motor",
    n_samples: int = 20,
    train_frac: float = 0.5,
    video_frames: int = 300,
    video_fps: int = 15,
):
    dt = 0.6
    K = cfg.context_length

    # ── Load data ────────────────────────────────────────────────────────
    worm_data = load_worm_data(h5_path, n_beh_modes=6)
    u_all = worm_data["u"]
    b      = worm_data["b"]
    worm_id = worm_data["worm_id"]
    motor_idx = worm_data.get("motor_idx")
    all_labels = worm_data.get("labels", [])

    if neurons == "motor" and motor_idx is not None:
        u_sel = u_all[:, motor_idx]
        sel_labels = [all_labels[i] if i < len(all_labels) else f"n{i}"
                      for i in motor_idx]
        neuron_label = "motor"
    else:
        u_sel = u_all
        sel_labels = all_labels
        neuron_label = "all"

    T, N = u_sel.shape
    Kw = b.shape[1]

    # ── Atlas indices ────────────────────────────────────────────────────
    atlas_idx = _get_atlas_indices(sel_labels)

    # ── Temporal split ───────────────────────────────────────────────────
    train_end = int(train_frac * T)
    n_test = T - train_end

    out_dir = base_out / worm_id / neuron_label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 65}")
    print(f"  WORM: {worm_id}   T={T}  N={N} ({neuron_label})  Kw={Kw}")
    print(f"  Split: train [0,{train_end})  test [{train_end},{T})")
    print(f"  SetTRF: d={cfg.d_model} h={cfg.n_heads} L={cfg.n_encoder_layers}")
    print(f"  Output → {out_dir}")
    print(f"{'═' * 65}")

    # ── Normalise (train stats) ──────────────────────────────────────────
    u_train, b_train = u_sel[:train_end], b[:train_end]
    mu_u  = u_train[K:].mean(0).astype(np.float32)
    sig_u = (u_train[K:].std(0) + 1e-8).astype(np.float32)
    mu_b  = b_train[K:].mean(0).astype(np.float32)
    sig_b = (b_train[K:].std(0) + 1e-8).astype(np.float32)

    u_norm = ((u_sel - mu_u) / sig_u).astype(np.float32)
    b_norm = ((b - mu_b) / sig_b).astype(np.float32)

    gt_neural_test = u_sel[train_end:]
    gt_beh_test    = b[train_end:]

    # ── Train ────────────────────────────────────────────────────────────
    print("\n  Training SetTransformerJoint ...")
    t0 = time.time()
    model = train_set_trf_joint(
        u_norm[:train_end], b_norm[:train_end], atlas_idx, cfg, device,
    )
    print(f"  Training done ({time.time() - t0:.1f}s)")

    # ── Seeds ────────────────────────────────────────────────────────────
    u_seed = u_norm[train_end - K : train_end].copy()
    b_seed = b_norm[train_end - K : train_end].copy()

    # ── Stochastic samples (T=1.0) ──────────────────────────────────────
    print(f"\n  Generating {n_samples} stochastic samples ...")
    samples_neural, samples_beh = [], []
    for i in range(n_samples):
        pn, pb = stochastic_joint_settrf(
            model, u_seed, b_seed, n_test, device, temperature=1.0)
        # De-normalise
        pn_raw = (pn * sig_u + mu_u).astype(np.float32)
        pb_raw = (pb * sig_b + mu_b).astype(np.float32)
        samples_neural.append(pn_raw)
        samples_beh.append(pb_raw)
        if (i + 1) % 5 == 0:
            print(f"    sample {i+1}/{n_samples}")

    # ── Distributional metrics ───────────────────────────────────────────
    met_list = []
    for s in samples_beh:
        m, _ = compute_distributional_metrics(gt_beh_test, s)
        met_list.append(m)
    med_metrics = ensemble_median_metrics(met_list)
    compute_distributional_metrics(gt_beh_test, samples_beh[0],
                                   label="SetTRF Joint")

    # ══════════════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Generating plots ...")

    # 1) Neural activity traces
    plot_neural_traces(gt_neural_test, samples_neural, out_dir, worm_id,
                       labels=sel_labels, dt=dt)

    # 2) Eigenworm traces
    plot_ew_traces(gt_beh_test, samples_beh, out_dir, worm_id, dt=dt)

    # 3) Ensemble traces (behaviour — reuse from utils)
    n_show = min(300, n_test)
    from scripts.free_run.utils import plot_ensemble_traces
    plot_ensemble_traces(gt_beh_test, samples_beh, out_dir,
                         "SetTRF_Joint", worm_id, n_show=n_show, dt=dt)

    # 4) PSD comparison
    fs = 1.0 / dt
    f_gt, psd_gt = compute_psd(gt_beh_test, fs=fs)
    psds_gen = [compute_psd(s, fs=fs)[1] for s in samples_beh]
    plot_psd_comparison(f_gt, psd_gt, [psds_gen], ["SetTRF Joint"],
                        out_dir, worm_id)

    # 5) Autocorrelation
    acf_gt = compute_autocorr(gt_beh_test)
    acfs_gen = [compute_autocorr(s) for s in samples_beh]
    plot_autocorr_comparison(acf_gt, [acfs_gen], ["SetTRF Joint"],
                             out_dir, worm_id, dt=dt)

    # 6) Marginals
    plot_marginals(gt_beh_test, [samples_beh], ["SetTRF Joint"],
                   out_dir, worm_id)

    # 7) Summary bars
    plot_summary_bars({"SetTRF Joint": med_metrics}, out_dir, worm_id)

    # ══════════════════════════════════════════════════════════════════════
    # POSTURE VIDEO  (20 seconds)
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Generating 20-second posture video ...")
    n_vid = min(video_frames, n_test)
    ew_gt  = gt_beh_test[:n_vid]
    ew_gen = samples_beh[0][:n_vid]  # use first sample
    video_path = str(out_dir / f"posture_SetTRF_Joint_{worm_id}.mp4")
    try:
        make_posture_compare_video(
            h5_path=h5_path,
            out_path=video_path,
            ew_stage1=ew_gt,
            ew_model_cv=ew_gen,
            max_frames=n_vid,
            fps=video_fps,
        )
        print(f"  ✓ Video: {video_path}")
    except Exception as e:
        print(f"  ✗ Video failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════════
    save_data = {
        "worm_id": worm_id,
        "neuron_label": neuron_label,
        "T": T, "N": N, "Kw": Kw, "K": K,
        "train_end": train_end, "n_test": n_test,
        "n_samples": n_samples,
        "architecture": "SetTransformerJoint",
        "config": {
            "d_model": cfg.d_model, "n_heads": cfg.n_heads,
            "n_encoder_layers": cfg.n_encoder_layers, "d_ff": cfg.d_ff,
            "neuron_embed_dim": cfg.neuron_embed_dim,
            "context_length": cfg.context_length,
        },
        "metrics_T1": med_metrics,
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)
    print(f"  Results → {json_path}")

    # Summary line
    m = med_metrics
    print(f"\n  SUMMARY  PSD={m['psd_log_distance']:.3f}  "
          f"ACF={m['autocorr_rmse']:.3f}  W1={m['wasserstein_1']:.3f}  "
          f"KS={m['ks_statistic']:.3f}  VarR={m['variance_ratio_mean']:.3f}")

    return save_data


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Set Transformer Joint Free-Run — batch all worms")
    ap.add_argument("--h5", default=None, help="Single H5 file")
    ap.add_argument("--h5_dir", default=None, help="Directory of H5 files")
    ap.add_argument("--out_dir", default="output_plots/free_run_settrf")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neurons", choices=["motor", "all"], default="motor")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--K", type=int, default=15, help="Context length")
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--video_frames", type=int, default=300,
                    help="Data frames for 20s video (300 @ 15fps)")
    ap.add_argument("--video_fps", type=int, default=15)
    args = ap.parse_args()

    if args.h5_dir:
        h5_files = sorted(glob.glob(str(Path(args.h5_dir) / "*.h5")))
    elif args.h5:
        h5_files = [args.h5]
    else:
        print("ERROR: provide --h5 or --h5_dir")
        sys.exit(1)

    cfg = SetTRFJointConfig(context_length=args.K)

    print(f"Set Transformer Joint Free-Run")
    print(f"  Files: {len(h5_files)}   Neurons: {args.neurons}")
    print(f"  Arch:  d={cfg.d_model}  h={cfg.n_heads}  "
          f"L={cfg.n_encoder_layers}  embed={cfg.neuron_embed_dim}")
    print(f"  K={cfg.context_length}  N_SAMPLES={args.n_samples}  "
          f"Device={args.device}")
    print(f"  Video: {args.video_frames} frames @ {args.video_fps} fps "
          f"= {args.video_frames / args.video_fps:.0f}s")

    all_results = []
    for i, h5 in enumerate(h5_files):
        print(f"\n[{i+1}/{len(h5_files)}] {Path(h5).stem}")
        try:
            res = process_one_worm(
                h5, cfg, args.device, Path(args.out_dir),
                neurons=args.neurons, n_samples=args.n_samples,
                train_frac=args.train_frac,
                video_frames=args.video_frames, video_fps=args.video_fps,
            )
            all_results.append(res)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── Batch summary ────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'═' * 65}")
        print(f"BATCH SUMMARY — {len(all_results)} worms")
        print(f"{'═' * 65}")
        print(f"{'Worm':<18} {'PSD↓':>8} {'ACF↓':>8} {'W1↓':>8} "
              f"{'KS↓':>8} {'VarR→1':>8}")
        print("-" * 60)
        for r in all_results:
            m = r["metrics_T1"]
            print(f"{r['worm_id']:<18} {m['psd_log_distance']:>8.3f} "
                  f"{m['autocorr_rmse']:>8.3f} {m['wasserstein_1']:>8.3f} "
                  f"{m['ks_statistic']:>8.3f} "
                  f"{m['variance_ratio_mean']:>8.3f}")

        # Save batch summary
        summary_path = Path(args.out_dir) / "batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        print(f"\nBatch summary → {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
