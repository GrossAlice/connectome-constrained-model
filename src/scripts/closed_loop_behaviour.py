#!/usr/bin/env python3
"""Closed-loop behaviour generation from a connectome-constrained neural model.

Pipeline
--------
1.  Train a Stage-2 neural dynamics model on a single worm.
2.  Fit a **forward decoder** (Ridge or MLP): motor-neuron activity → eigenworm
    mode coefficients (behaviour).
3.  Fit a **proprioceptive encoder** (linear): eigenworm modes → per-neuron
    additive drive (feedback from body posture to nervous system).
4.  Run the model in **closed loop** for T_gen steps:
        u(t) ─► prior_step ─► u(t+1)
                     ▲               │
                     │               ▼
              I_prop(t)        decoder ─► behaviour(t+1)
                     ▲               │
                     └───── encoder ──┘

The proprioceptive feedback is injected as an additive current to the tonic
drive  I0  (effectively making  I0(t) = I0_base + W_prop @ behaviour(t) ).

Usage
-----
    python -u -m scripts.closed_loop_behaviour \\
        --h5 data/used/.../2023-01-17-14.h5 \\
        --save_dir output_plots/closed_loop/run1 \\
        --device cuda

The script can also accept a pre-trained stage2 checkpoint via --stage2_dir
to skip the training phase.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── project imports ──────────────────────────────────────────────────
from stage2.config import Stage2PTConfig, make_config
from stage2.io_h5 import load_data_pt
from stage2.model import Stage2ModelPT
from stage2.init_from_data import init_lambda_u, init_all_from_data
from stage2.behavior_decoder_eval import (
    build_lagged_features_np,
    _log_ridge_grid,
    _ridge_cv_single_target,
)
from stage2.train import train_stage2, snapshot_model_state
from stage2.evaluate import compute_free_run

# ── helpers ──────────────────────────────────────────────────────────

def _motor_indices(neuron_labels: list[str], motor_list_path: str) -> list[int]:
    """Return indices of motor neurons within neuron_labels."""
    motor_names = set(
        Path(motor_list_path).read_text().strip().split()
    )
    return [i for i, lab in enumerate(neuron_labels) if lab in motor_names]


def _fit_forward_decoder(
    u: np.ndarray,
    behaviour: np.ndarray,
    motor_idx: list[int],
    n_lags: int = 8,
    K: int = 6,
) -> dict:
    """Fit a Ridge-CV decoder: motor neuron traces → eigenworm modes.

    Returns dict with 'coefs', 'intercepts', 'lambdas', 'r2_train'.
    """
    u_motor = u[:, motor_idx]
    X = build_lagged_features_np(u_motor, n_lags)
    b = behaviour[:, :K]
    T = X.shape[0]

    mu, sigma = X.mean(0), X.std(0) + 1e-8
    X_s = (X - mu) / sigma

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_folds = 5

    coefs, intercepts, lambdas, r2s = [], [], [], []
    for j in range(K):
        result = _ridge_cv_single_target(X_s, b[:, j],
                                          eval_idx=np.arange(T),
                                          ridge_grid=ridge_grid,
                                          n_folds=n_folds)
        coefs.append(result["coef"])
        intercepts.append(result["intercept"])
        lambdas.append(result["best_lambda"])
        # Training R²
        pred = X_s @ result["coef"] + result["intercept"]
        ss_res = np.sum((b[:, j] - pred) ** 2)
        ss_tot = np.sum((b[:, j] - b[:, j].mean()) ** 2)
        r2s.append(1 - ss_res / max(ss_tot, 1e-12))

    return {
        "coefs": np.array(coefs),          # (K, d_features)
        "intercepts": np.array(intercepts), # (K,)
        "lambdas": np.array(lambdas),       # (K,)
        "r2_train": np.array(r2s),          # (K,)
        "feature_mu": mu,
        "feature_sigma": sigma,
        "motor_idx": motor_idx,
        "n_lags": n_lags,
        "K": K,
    }


def _decode_behaviour(
    u_motor_history: np.ndarray,
    decoder: dict,
) -> np.ndarray:
    """Decode behaviour from a motor-neuron history buffer.

    u_motor_history: shape (n_lags+1, M) — most recent at index -1.
    Returns: (K,) eigenworm mode coefficients.
    """
    # Build lagged feature vector for the last time-step
    n_lags = decoder["n_lags"]
    M = u_motor_history.shape[1]
    # Stack: [u(t), u(t-1), ..., u(t-n_lags)]  → (1, M*(n_lags+1))
    feat = u_motor_history[::-1].reshape(1, -1)  # reverse so t is first
    feat_s = (feat - decoder["feature_mu"]) / decoder["feature_sigma"]
    pred = feat_s @ decoder["coefs"].T + decoder["intercepts"]  # (1, K)
    return pred[0]


def _fit_proprioceptive_encoder(
    behaviour: np.ndarray,
    u: np.ndarray,
    N: int,
    K: int = 6,
    ridge_alpha: float = 1.0,
) -> dict:
    """Fit a linear proprioceptive encoder: eigenworms → per-neuron additive drive.

    We learn  W_prop (N, K)  such that  I_prop(t) = W_prop @ behaviour(t)
    approximates the component of the neural drive that correlates with body posture.

    This is fit via Ridge regression:  u(t+1) - u_predicted(t+1) ≈ W_prop @ b(t)
    But since we don't have model predictions yet, we use a simpler approach:
    correlate the *derivative* of neural activity with behaviour.
    """
    b = behaviour[:, :K]
    T = min(len(b), len(u))

    # Target: du/dt proxy  (u[t+1] - u[t])
    du = u[1:T] - u[:T-1]
    b_aligned = b[:T-1]

    # Ridge regression:  du = b_aligned @ W.T + intercept
    # → W.T = (b.T @ b + alpha*I)^{-1} @ b.T @ du
    bTb = b_aligned.T @ b_aligned + ridge_alpha * np.eye(K)
    bTdu = b_aligned.T @ du
    W = np.linalg.solve(bTb, bTdu).T  # (N, K)

    # Also compute a bias (mean residual drive)
    bias = du.mean(0) - W @ b_aligned.mean(0)

    # R² of fit
    pred = b_aligned @ W.T + bias
    ss_res = np.sum((du - pred) ** 2, axis=0)
    ss_tot = np.sum((du - du.mean(0)) ** 2, axis=0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)

    return {
        "W": W,                    # (N, K)
        "bias": bias,              # (N,)
        "r2_per_neuron": r2,       # (N,)
        "K": K,
    }


def closed_loop_free_run(
    model: Stage2ModelPT,
    decoder: dict,
    encoder: dict,
    u_init: np.ndarray,
    T_gen: int,
    *,
    gating: np.ndarray | None = None,
    dt: float = 0.6,
    prop_gain: float = 1.0,
    noise_scale: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run the closed-loop simulation.

    Parameters
    ----------
    model : trained Stage2ModelPT
    decoder : forward decoder dict (motor activity → behaviour)
    encoder : proprioceptive encoder dict (behaviour → neural drive)
    u_init : (T_seed, N) initial neural state to seed from
    T_gen : number of time-steps to generate
    gating : (T_gen, N) or None — gating mask
    dt : time-step
    prop_gain : scaling factor for proprioceptive feedback strength
    noise_scale : scaling factor for process noise injection
    device : torch device

    Returns
    -------
    dict with:
        u_gen : (T_gen, N) generated neural traces
        b_gen : (T_gen, K) generated behaviour
        b_prop : (T_gen, N) proprioceptive drive injected
    """
    N = model.N
    K = decoder["K"]
    n_lags = decoder["n_lags"]
    motor_idx = decoder["motor_idx"]
    M = len(motor_idx)

    W_prop = torch.tensor(encoder["W"], dtype=torch.float32, device=device)  # (N,K)
    prop_bias = torch.tensor(encoder["bias"], dtype=torch.float32, device=device)

    # Seed: use last n_lags+1 steps from u_init
    T_seed = u_init.shape[0]
    seed_len = min(T_seed, n_lags + 1)

    u_gen = np.zeros((T_gen, N), dtype=np.float32)
    b_gen = np.zeros((T_gen, K), dtype=np.float32)
    I_prop_all = np.zeros((T_gen, N), dtype=np.float32)

    # Fill seed
    u_gen[:seed_len] = u_init[-seed_len:]

    # Motor-neuron history buffer for lagged features
    motor_buf = np.zeros((n_lags + 1, M), dtype=np.float32)
    for i in range(seed_len):
        motor_buf[i % (n_lags + 1)] = u_gen[i, motor_idx]

    # Decode initial behaviour from seed
    for t in range(seed_len):
        # Build proper history
        idx_range = np.arange(max(0, t - n_lags), t + 1)
        hist = u_gen[idx_range, :][:, motor_idx]
        if hist.shape[0] < n_lags + 1:
            pad = np.zeros((n_lags + 1 - hist.shape[0], M), dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)
        b_gen[t] = _decode_behaviour(hist, decoder)

    # State variables
    s_sv = torch.zeros(N, model.r_sv, device=device)
    s_dcv = torch.zeros(N, model.r_dcv, device=device)
    ones = torch.ones(N, device=device)

    # Warm up synaptic variables using seed
    with torch.no_grad():
        for t in range(1, seed_len):
            u_prev = torch.tensor(u_gen[t - 1], dtype=torch.float32, device=device)
            g = ones
            if gating is not None and t - 1 < gating.shape[0]:
                g = torch.tensor(gating[t - 1], dtype=torch.float32, device=device)
            phi_gated = model.phi(u_prev) * g
            # Advance synaptic state variables
            if model.r_sv > 0:
                gamma_sv = torch.exp(-dt / (model.tau_sv + 1e-12))
                s_sv = gamma_sv.view(1, -1) * s_sv + phi_gated.unsqueeze(1)
            if model.r_dcv > 0:
                gamma_dcv = torch.exp(-dt / (model.tau_dcv + 1e-12))
                s_dcv = gamma_dcv.view(1, -1) * s_dcv + phi_gated.unsqueeze(1)

    # ── Main generation loop ─────────────────────────────────────────
    with torch.no_grad():
        I0_base = model.I0.clone()

        for t in range(seed_len, T_gen):
            u_prev = torch.tensor(u_gen[t - 1], dtype=torch.float32, device=device)

            # Proprioceptive feedback from decoded behaviour
            b_t = torch.tensor(b_gen[t - 1], dtype=torch.float32, device=device)
            I_prop = prop_gain * (W_prop @ b_t)
            I_prop_all[t] = I_prop.cpu().numpy()

            # Inject proprioceptive feedback as additive tonic drive
            model.I0.data.copy_(I0_base + I_prop)

            g = ones
            if gating is not None and t - 1 < gating.shape[0]:
                g = torch.tensor(gating[t - 1], dtype=torch.float32, device=device)

            # One step of neural dynamics
            u_next, s_sv, s_dcv = model.prior_step(u_prev, s_sv, s_dcv, g)

            # Add process noise
            if noise_scale > 0:
                sigma = model.sigma_at(u_prev)
                eps = torch.randn_like(u_next)
                u_next = u_next + noise_scale * sigma * eps

            # Clip
            lo, hi = model.u_clip
            if lo is not None or hi is not None:
                u_next = u_next.clamp(min=lo, max=hi)

            u_gen[t] = u_next.cpu().numpy()

            # Decode behaviour from motor neurons
            idx_range = np.arange(max(0, t - n_lags), t + 1)
            hist = u_gen[idx_range, :][:, motor_idx]
            if hist.shape[0] < n_lags + 1:
                pad = np.zeros((n_lags + 1 - hist.shape[0], M), dtype=np.float32)
                hist = np.concatenate([pad, hist], axis=0)
            b_gen[t] = _decode_behaviour(hist, decoder)

        # Restore I0
        model.I0.data.copy_(I0_base)

    return {
        "u_gen": u_gen,
        "b_gen": b_gen,
        "I_prop": I_prop_all,
    }


# ── Plotting ─────────────────────────────────────────────────────────

def _plot_behaviour_traces(
    b_gt: np.ndarray,
    b_open: np.ndarray,
    b_closed: np.ndarray,
    dt: float,
    out_path: Path,
    K: int = 6,
):
    """Compare GT, open-loop free-run, and closed-loop behaviour."""
    T_gt = b_gt.shape[0]
    T_gen = b_closed.shape[0]
    t_gt = np.arange(T_gt) * dt
    t_gen = np.arange(T_gen) * dt

    fig, axes = plt.subplots(K, 1, figsize=(14, 2.5 * K), sharex=False)
    mode_names = [f"a{j+1}" for j in range(K)]

    for j in range(K):
        ax = axes[j]
        ax.plot(t_gt, b_gt[:, j], "k-", alpha=0.5, lw=1, label="Ground truth")
        if b_open is not None:
            T_op = min(len(b_open), T_gen)
            ax.plot(t_gen[:T_op], b_open[:T_op, j], "b-", alpha=0.6, lw=1,
                    label="Open-loop (no feedback)")
        ax.plot(t_gen, b_closed[:, j], "r-", alpha=0.7, lw=1,
                label="Closed-loop (proprioceptive)")
        ax.axvline(t_gt[-1], color="gray", ls="--", alpha=0.4)
        ax.set_ylabel(mode_names[j], fontsize=11)
        ax.set_xlim(0, max(t_gt[-1], t_gen[-1]))
        if j == 0:
            ax.legend(fontsize=9, loc="upper right")
    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.suptitle("Closed-loop behaviour generation", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved behaviour traces → {out_path}")


def _plot_neural_traces(
    u_gt: np.ndarray,
    u_gen: np.ndarray,
    dt: float,
    out_path: Path,
    n_show: int = 12,
):
    """Show a selection of neural traces: GT vs generated."""
    N = u_gt.shape[1]
    T_gt = u_gt.shape[0]
    T_gen = u_gen.shape[0]
    t_gt = np.arange(T_gt) * dt
    t_gen = np.arange(T_gen) * dt

    idx = np.linspace(0, N - 1, n_show, dtype=int)
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2 * n_show), sharex=False)
    for i, ni in enumerate(idx):
        ax = axes[i]
        ax.plot(t_gt, u_gt[:, ni], "k-", alpha=0.5, lw=0.8, label="GT")
        ax.plot(t_gen, u_gen[:, ni], "r-", alpha=0.6, lw=0.8, label="Generated")
        ax.set_ylabel(f"n{ni}", fontsize=9)
        ax.axvline(t_gt[-1], color="gray", ls="--", alpha=0.4)
        if i == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.suptitle("Neural traces: GT vs closed-loop", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved neural traces → {out_path}")


def _plot_phase_portrait(
    b_gt: np.ndarray,
    b_closed: np.ndarray,
    out_path: Path,
):
    """Phase portrait of a₁ vs a₂."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(b_gt[:, 0], b_gt[:, 1], "k-", alpha=0.3, lw=0.5)
    ax.set_title("Ground truth", fontsize=12)
    ax.set_xlabel("a₁"); ax.set_ylabel("a₂")
    ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(b_closed[:, 0], b_closed[:, 1], "r-", alpha=0.3, lw=0.5)
    ax.set_title("Closed-loop generated", fontsize=12)
    ax.set_xlabel("a₁"); ax.set_ylabel("a₂")
    ax.set_aspect("equal")

    # Match axis limits
    all_vals = np.concatenate([b_gt[:, :2], b_closed[:, :2]], axis=0)
    lim = np.abs(all_vals).max() * 1.1
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.suptitle("Phase portrait: a₁ vs a₂", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved phase portrait → {out_path}")


def _plot_power_spectrum(
    b_gt: np.ndarray,
    b_closed: np.ndarray,
    dt: float,
    out_path: Path,
    K: int = 6,
):
    """Compare power spectra of GT and generated behaviour."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.ravel()
    mode_names = [f"a{j+1}" for j in range(K)]

    for j in range(K):
        ax = axes[j]
        for label, data, color in [("GT", b_gt[:, j], "k"),
                                    ("Closed-loop", b_closed[:, j], "r")]:
            n = len(data)
            freq = np.fft.rfftfreq(n, d=dt)
            psd = np.abs(np.fft.rfft(data - data.mean())) ** 2 / n
            # Smooth
            win = min(5, len(psd) // 10)
            if win > 1:
                psd = np.convolve(psd, np.ones(win) / win, mode="same")
            ax.semilogy(freq[1:], psd[1:], color=color, alpha=0.7, lw=1, label=label)
        ax.set_title(mode_names[j], fontsize=11)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        if j == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Power spectra: GT vs closed-loop", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved power spectra → {out_path}")


def _compute_behaviour_stats(b: np.ndarray, dt: float, K: int = 6) -> dict:
    """Compute summary statistics for a behaviour sequence."""
    stats = {}
    for j in range(min(K, b.shape[1])):
        stats[f"a{j+1}_std"] = float(np.std(b[:, j]))
        stats[f"a{j+1}_mean"] = float(np.mean(b[:, j]))
        # Dominant frequency
        n = len(b[:, j])
        freq = np.fft.rfftfreq(n, d=dt)
        psd = np.abs(np.fft.rfft(b[:, j] - b[:, j].mean())) ** 2
        if len(freq) > 1:
            peak_idx = np.argmax(psd[1:]) + 1
            stats[f"a{j+1}_peak_freq_hz"] = float(freq[peak_idx])
            stats[f"a{j+1}_peak_period_s"] = float(1.0 / max(freq[peak_idx], 1e-6))
    return stats


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="Path to worm h5 file")
    ap.add_argument("--save_dir", default="output_plots/closed_loop/run1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage2_dir", default=None,
                    help="Path to pre-trained stage2 output directory (skip training)")
    ap.add_argument("--T_gen_mult", type=float, default=3.0,
                    help="Generate T_gen = T_data * T_gen_mult time-steps")
    ap.add_argument("--prop_gain", type=float, default=1.0,
                    help="Proprioceptive feedback gain")
    ap.add_argument("--noise_scale", type=float, default=1.0,
                    help="Process noise scale (0=deterministic, 1=learned noise)")
    ap.add_argument("--n_lags", type=int, default=8,
                    help="Number of temporal lags for behaviour decoder")
    ap.add_argument("--K", type=int, default=6,
                    help="Number of eigenworm modes")
    ap.add_argument("--num_epochs", type=int, default=100,
                    help="Stage2 training epochs")
    args = ap.parse_args()

    DEVICE = torch.device(args.device)
    out_dir = Path(args.save_dir)

    h5_path = args.h5
    K = args.K
    n_lags = args.n_lags

    # ── Step 1: Train or load Stage-2 model ──────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1: Stage-2 neural dynamics model")
    print("=" * 70)

    stage2_dir = args.stage2_dir
    if stage2_dir is None:
        stage2_dir = str(out_dir / "stage2")

    stage2_ckpt_h5 = Path(stage2_dir) / "stage2_results.h5"
    stage2_ckpt_pt = Path(stage2_dir) / "model_final.pt"

    # Load data (needed both for training and for decoder fitting)
    cfg = make_config(h5_path, device=args.device)
    data = load_data_pt(cfg)
    u_stage1 = data["u_stage1"]  # (T, N) tensor
    T_data, N = u_stage1.shape
    dt = float(data["dt"])
    behaviour_tensor = data["b"]  # (T, K_full) – key is "b" in load_data_pt
    if behaviour_tensor is None:
        raise RuntimeError(f"No behaviour data found in {h5_path}")
    behaviour = behaviour_tensor.cpu().numpy()
    neuron_labels = data.get("neuron_labels", [f"n{i}" for i in range(N)])
    if isinstance(neuron_labels, torch.Tensor):
        neuron_labels = [str(x) for x in neuron_labels]

    sign_t = data.get("sign_t")
    d_ell = data.get("d_ell", 0)

    if stage2_ckpt_h5.exists() or stage2_ckpt_pt.exists():
        print(f"  Found existing checkpoint in {stage2_dir}")
        print("  Rebuilding model and loading weights...")
    else:
        print(f"  Training stage2 on {h5_path} ...")
        print("  (Skipping slow baseline evaluation to speed up)")
        # Monkey-patch the slow baseline eval to return a dummy
        import stage2.train as _train_mod
        _orig_beh_all = _train_mod.behaviour_all_neurons_prediction
        _train_mod.behaviour_all_neurons_prediction = lambda data: None
        try:
            from stage2.run_stage2 import main as run_stage2_main
            run_stage2_main([
                "--h5", h5_path,
                "--save_dir", stage2_dir,
                "--device", args.device,
                "--set", "num_epochs", str(args.num_epochs),
            ])
        finally:
            _train_mod.behaviour_all_neurons_prediction = _orig_beh_all
        print(f"  Stage2 training complete → {stage2_dir}")

    # Rebuild model architecture
    device = torch.device(args.device)
    lambda_u_init = init_lambda_u(u_stage1, cfg)

    model = Stage2ModelPT(
        N, data["T_e"], data["T_sv"], data["T_dcv"], dt, cfg, device,
        d_ell=d_ell, lambda_u_init=lambda_u_init, sign_t=sign_t,
    ).to(device)

    init_all_from_data(model, u_stage1.to(device), cfg)

    # Load saved parameters
    if stage2_ckpt_pt.exists():
        # Prefer .pt checkpoint (full snapshot)
        saved = torch.load(str(stage2_ckpt_pt), map_location=device,
                           weights_only=False)
        # saved is a dict from snapshot_model_state: mix of raw params and
        # derived values.  Load the raw params into the model.
        sd = model.state_dict()
        for key in sd:
            if key in saved:
                sd[key] = saved[key].to(sd[key].dtype)
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded from {stage2_ckpt_pt}")
    elif stage2_ckpt_h5.exists():
        import h5py
        with h5py.File(str(stage2_ckpt_h5), "r") as f:
            grp = f["stage2_pt/params"]
            sd = model.state_dict()
            for key in sd:
                if key in grp:
                    sd[key] = torch.tensor(np.array(grp[key]),
                                           dtype=sd[key].dtype)
                elif key in grp.attrs:
                    val = float(grp.attrs[key])
                    sd[key] = torch.tensor(val, dtype=sd[key].dtype)
            model.load_state_dict(sd, strict=False)
        print(f"  Loaded from {stage2_ckpt_h5}")

    model.eval()
    u_stage1_np = u_stage1.cpu().numpy()

    print(f"\n  Model loaded: N={N}, T={T_data}, dt={dt:.3f}s")
    print(f"  Behaviour: {behaviour.shape[1]} modes, using K={K}")

    # ── Step 2: Forward decoder (motor → behaviour) ──────────────────
    print("\n" + "=" * 70)
    print("  STEP 2: Forward decoder (motor neurons → behaviour)")
    print("=" * 70)

    motor_names = set(cfg.motor_neurons) if cfg.motor_neurons else set()
    motor_idx = [i for i, lab in enumerate(neuron_labels) if lab in motor_names]
    print(f"  Motor neurons: {len(motor_idx)} / {N}")

    decoder = _fit_forward_decoder(
        u_stage1_np, behaviour, motor_idx,
        n_lags=n_lags, K=K,
    )
    print(f"  Forward decoder R² (training): "
          f"{' '.join(f'{v:.3f}' for v in decoder['r2_train'])}")

    # ── Step 3: Proprioceptive encoder (behaviour → neural drive) ────
    print("\n" + "=" * 70)
    print("  STEP 3: Proprioceptive encoder (behaviour → neural drive)")
    print("=" * 70)

    encoder = _fit_proprioceptive_encoder(
        behaviour, u_stage1_np, N, K=K, ridge_alpha=10.0,
    )
    r2_prop = encoder["r2_per_neuron"]
    print(f"  Proprioceptive encoder R² (du prediction):")
    print(f"    mean={np.mean(r2_prop):.4f}, median={np.median(r2_prop):.4f}, "
          f"max={np.max(r2_prop):.4f}")

    # ── Step 4: Open-loop free run (baseline) ────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 4: Open-loop free run (no proprioceptive feedback)")
    print("=" * 70)

    T_gen = int(T_data * args.T_gen_mult)
    print(f"  Generating {T_gen} steps ({T_gen * dt:.0f}s) ...")

    # Get gating tensor, convert to numpy if present
    gating_np = None
    if "gating" in data and data["gating"] is not None:
        g = data["gating"]
        gating_np = g.cpu().numpy() if isinstance(g, torch.Tensor) else np.asarray(g)

    # Open-loop: just run the model without feedback
    open_loop = closed_loop_free_run(
        model, decoder, encoder,
        u_init=u_stage1_np,
        T_gen=T_gen,
        gating=gating_np,
        dt=dt,
        prop_gain=0.0,  # No feedback
        noise_scale=args.noise_scale,
        device=DEVICE,
    )

    # ── Step 5: Closed-loop free run ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 5: Closed-loop free run (with proprioceptive feedback)")
    print("=" * 70)

    closed_loop = closed_loop_free_run(
        model, decoder, encoder,
        u_init=u_stage1_np,
        T_gen=T_gen,
        gating=gating_np,
        dt=dt,
        prop_gain=args.prop_gain,
        noise_scale=args.noise_scale,
        device=DEVICE,
    )

    # ── Step 6: Analysis & plots ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 6: Analysis & Plots")
    print("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    b_gt = behaviour[:, :K]
    b_open = open_loop["b_gen"]
    b_closed = closed_loop["b_gen"]

    # Compute stats
    stats_gt = _compute_behaviour_stats(b_gt, dt, K)
    stats_open = _compute_behaviour_stats(b_open, dt, K)
    stats_closed = _compute_behaviour_stats(b_closed, dt, K)

    # R² for the data-overlap region
    T_overlap = min(T_data, T_gen)
    r2_open, r2_closed = [], []
    for j in range(K):
        gt = b_gt[:T_overlap, j]
        ss_tot = np.sum((gt - gt.mean()) ** 2)
        r2_open.append(1 - np.sum((gt - b_open[:T_overlap, j]) ** 2) / max(ss_tot, 1e-12))
        r2_closed.append(1 - np.sum((gt - b_closed[:T_overlap, j]) ** 2) / max(ss_tot, 1e-12))

    print(f"\n  R² (data overlap, {T_overlap} steps):")
    print(f"    Open-loop:   {' '.join(f'{v:.3f}' for v in r2_open)}  mean={np.mean(r2_open):.3f}")
    print(f"    Closed-loop: {' '.join(f'{v:.3f}' for v in r2_closed)}  mean={np.mean(r2_closed):.3f}")

    print(f"\n  Behaviour amplitude (std):")
    for j in range(K):
        print(f"    a{j+1}: GT={stats_gt[f'a{j+1}_std']:.3f}  "
              f"Open={stats_open[f'a{j+1}_std']:.3f}  "
              f"Closed={stats_closed[f'a{j+1}_std']:.3f}")

    print(f"\n  Dominant period (s):")
    for j in range(K):
        p_gt = stats_gt.get(f"a{j+1}_peak_period_s", float("nan"))
        p_op = stats_open.get(f"a{j+1}_peak_period_s", float("nan"))
        p_cl = stats_closed.get(f"a{j+1}_peak_period_s", float("nan"))
        print(f"    a{j+1}: GT={p_gt:.1f}s  Open={p_op:.1f}s  Closed={p_cl:.1f}s")

    # ── Plots ────────────────────────────────────────────────────────
    _plot_behaviour_traces(b_gt, b_open, b_closed, dt,
                           out_dir / "behaviour_traces.png", K=K)
    _plot_neural_traces(u_stage1_np, closed_loop["u_gen"], dt,
                        out_dir / "neural_traces.png")
    _plot_phase_portrait(b_gt, b_closed, out_dir / "phase_portrait.png")
    _plot_power_spectrum(b_gt, b_closed, dt,
                         out_dir / "power_spectrum.png", K=K)

    # Save summary
    summary = {
        "h5_path": h5_path,
        "N": N, "T_data": T_data, "T_gen": T_gen, "dt": dt, "K": K,
        "n_motor": len(motor_idx),
        "prop_gain": args.prop_gain,
        "noise_scale": args.noise_scale,
        "decoder_r2_train": decoder["r2_train"].tolist(),
        "encoder_r2_mean": float(np.mean(encoder["r2_per_neuron"])),
        "r2_open_loop": r2_open,
        "r2_closed_loop": r2_closed,
        "stats_gt": stats_gt,
        "stats_open": stats_open,
        "stats_closed": stats_closed,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {out_dir / 'summary.json'}")

    # Save generated data
    np.savez_compressed(
        str(out_dir / "generated.npz"),
        u_gen=closed_loop["u_gen"],
        b_gen=closed_loop["b_gen"],
        b_open=open_loop["b_gen"],
        u_open=open_loop["u_gen"],
        b_gt=b_gt,
        u_gt=u_stage1_np,
        I_prop=closed_loop["I_prop"],
    )
    print(f"  Generated data → {out_dir / 'generated.npz'}")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
