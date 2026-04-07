#!/usr/bin/env python3
"""
Quick all-models comparison: 9 architectures + SetTRF-Small + Stage2 Connectome.

Trains all models, generates short (50-frame) free-run trajectories,
creates a side-by-side posture comparison video, and prints a
distributional-metrics summary.

Speed optimisations vs the full run.py:
  - Reduced TRF epochs: max_epochs=80, patience=15
  - Fewer samples: n_samples=3 (enough for median metrics)
  - Only 50 generation frames (30s of worm time at dt=0.6s)
  - No temperature sweep
  - SetTRF-Small only (smallest variant, ~2× faster than Default)

Models (11):
  1. TRF Joint       2. TRF AR+Dec       3. TRF Cascaded
  4. MLP Joint       5. MLP AR+Dec       6. MLP Cascaded
  7. Ridge Joint     8. Ridge AR+Dec     9. Ridge Cascaded
  10. SetTRF-Small   11. Stage2 Connectome

Usage:
  python -m scripts.free_run.run_all_models_quick \
      --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-06-14-13.h5" \
      --device cuda
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.dataset import load_worm_data
from baseline_transformer.train import train_single_worm_cv

# Reuse free-run loops from run.py
from scripts.free_run.run import (
    stochastic_joint_trf, stochastic_joint_mlp, stochastic_joint_ridge,
    stochastic_ar_dec_trf, stochastic_ar_dec_mlp, stochastic_ar_dec_ridge,
    stochastic_cascaded_trf, stochastic_cascaded_mlp, stochastic_cascaded_ridge,
    _train_trf_neural_only, _train_trf_decoder,
)

# Stage2 connectome model
from stage2.config import make_config as _s2_make_config
from stage2.io_h5 import load_data_pt as _s2_load_data
from stage2.init_from_data import init_lambda_u as _s2_init_lambda_u, init_all_from_data as _s2_init_all
from stage2.model import Stage2ModelPT
from stage2.train import compute_dynamics_loss as _s2_dynamics_loss, compute_rollout_loss as _s2_rollout_loss

# Reuse utils
from scripts.free_run.utils import (
    build_lagged, train_mlp, train_mlp_rollout, train_mlp_beh_rollout,
    train_ridge, estimate_residual_std,
    compute_distributional_metrics, ensemble_median_metrics,
    compute_psd, compute_autocorr,
    plot_ensemble_traces, plot_psd_comparison, plot_autocorr_comparison,
    plot_marginals, plot_summary_bars,
    NumpyEncoder,
)

# SetTRF
from scripts.free_run.run_set_transformer_joint import (
    SetTRFJointConfig, train_set_trf_joint, stochastic_joint_settrf,
    _get_atlas_indices,
)

# Video
from scripts.free_run.run_comparison import make_multi_model_video

import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Neural trace plot: GT vs generated, one neuron per model
# ─────────────────────────────────────────────────────────────────────────────

def _plot_neural_traces(
    gt_neural: np.ndarray,        # (n_steps, N)
    all_neural: dict,             # label → list of (n_steps, N) arrays
    model_names: list[str],
    sel_labels: list[str],        # neuron names
    out_dir: Path,
    worm_id: str,
    dt: float = 0.6,
):
    """One subplot per model. Each subplot shows GT (black) + 3 samples (colour)
    for a single neuron (cycling through neurons to give variety)."""
    n_models = len(model_names)
    if n_models == 0:
        return

    n_steps = gt_neural.shape[0]
    N = gt_neural.shape[1]
    t_sec = np.arange(n_steps) * dt

    # Pick a distinct neuron for each model (cycle if more models than neurons)
    # Sort neurons by variance so we pick the most interesting ones first
    var_order = np.argsort(-gt_neural.var(0))  # most variable first

    ncols = 2
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.8 * nrows),
                              sharex=True, squeeze=False)

    SAMPLE_COLORS = ["#2196F3", "#FF5722", "#4CAF50"]

    for i, label in enumerate(model_names):
        ax = axes[i // ncols, i % ncols]
        neuron_idx = var_order[i % N]
        neuron_name = sel_labels[neuron_idx] if neuron_idx < len(sel_labels) else f"n{neuron_idx}"

        # GT trace
        ax.plot(t_sec, gt_neural[:, neuron_idx], color="k", lw=1.5,
                label="GT", alpha=0.85)

        # Generated samples
        if label in all_neural:
            for j, samp in enumerate(all_neural[label]):
                c = SAMPLE_COLORS[j % len(SAMPLE_COLORS)]
                lbl = "gen" if j == 0 else None
                ax.plot(t_sec, samp[:n_steps, neuron_idx], color=c, lw=0.9,
                        alpha=0.7, label=lbl)

        ax.set_title(f"{label}  —  {neuron_name}", fontsize=10, fontweight="bold")
        ax.set_ylabel("ΔF/F", fontsize=8)
        if i >= n_models - ncols:
            ax.set_xlabel("time (s)", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

    # Hide unused axes
    for j in range(n_models, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(f"Neural traces — GT vs generated  ({worm_id})", fontsize=13, y=1.01)
    fig.tight_layout()
    fpath = out_dir / f"neural_traces_{worm_id}.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage2 connectome model: training + stochastic generation
# ─────────────────────────────────────────────────────────────────────────────

def _train_stage2_fast(
    h5_path: str,
    train_end: int,
    device: str,
    num_epochs: int = 50,
    rollout_weight: float = 0.3,
    rollout_steps: int = 15,
    rollout_starts: int = 4,
) -> tuple:
    """Train a Stage2 connectome-constrained model on the training half.

    Uses rollout NLL loss to calibrate the noise model for multi-step
    free-run generation (sweep-optimised: 50 epochs, rollout_weight=0.3).

    Returns (model, data_dict) where model is eval-mode and data_dict
    contains u_stage1 (T_full, N_all) on device plus connectivity info.
    """
    cfg = _s2_make_config(
        h5_path,
        device=device,
        num_epochs=num_epochs,
        learning_rate=0.001,
        learn_noise=True,
        noise_mode="heteroscedastic",
        coupling_gate=True,
        graph_poly_order=2,
        behavior_weight=0.0,   # decode behaviour separately
        rollout_weight=rollout_weight,
        rollout_steps=rollout_steps,
        rollout_starts=rollout_starts,
        loo_aux_weight=0.0,    # speed: no LOO aux
        grad_clip_norm=1.0,
    )

    data = _s2_load_data(cfg)
    data["_cfg"] = cfg
    u_full = data["u_stage1"]          # (T_full, N_all) on device
    T_full, N_all = u_full.shape
    dev = torch.device(device)

    u_train = u_full[:train_end]
    sigma_u = data["sigma_u"]
    g_train = data["gating"][:train_end] if data.get("gating") is not None else None
    s_train = data["stim"][:train_end] if data.get("stim") is not None else None

    # Build model
    lambda_u_init = _s2_init_lambda_u(u_full, cfg)
    model = Stage2ModelPT(
        N_all, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
        cfg, dev, d_ell=data.get("d_ell", 0),
        lambda_u_init=lambda_u_init,
        sign_t=data.get("sign_t"),
    ).to(dev)
    _s2_init_all(model, u_full, cfg)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)
    use_rollout = (rollout_weight > 0 and rollout_steps > 0)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        _p = model.precompute_params()
        prior_mu = model.forward_sequence(
            u_train, gating_data=g_train, stim_data=s_train, params=_p)

        if model._noise_mode == "heteroscedastic":
            model_sigma = model.sigma_at(u_train[:-1])
        else:
            model_sigma = model.sigma_at()

        loss = _s2_dynamics_loss(
            u_train[1:], prior_mu[1:], sigma_u,
            model_sigma=model_sigma,
        )

        # Rollout NLL loss: calibrates noise on multi-step residuals
        if use_rollout:
            r_loss = _s2_rollout_loss(
                model, u_train, sigma_u,
                rollout_steps=rollout_steps,
                rollout_starts=rollout_starts,
                gating_data=g_train,
                stim_data=s_train,
                use_nll=True,
            )
            loss = loss + rollout_weight * r_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"    Stage2 epoch {epoch+1}/{num_epochs}: "
                  f"loss={loss.item():.4f}")

    model.eval()
    return model, data


# Sweep-optimised temperature for Stage2 generation (from sweep_stage2.py)
_STAGE2_TEMPERATURE = 1.5

def _stage2_generate(
    model: Stage2ModelPT,
    data: dict,
    train_end: int,
    n_steps: int,
    n_samples: int,
    motor_idx: list[int],
    temperature: float = _STAGE2_TEMPERATURE,
) -> list[np.ndarray]:
    """Stochastic autonomous free-run from Stage2 connectome model.

    1. Teacher-forced burn-in through training data to warm synaptic states.
    2. Autonomous stochastic generation for *n_steps* frames.
    3. Extract motor-neuron subset.

    Temperature > 1 scales noise at inference to correct for under-dispersed
    learned noise (sweep-optimised: T=1.5).

    Returns list of *n_samples* arrays, each (n_steps, N_motor) in raw
    u_stage1 space.
    """
    u0 = data["u_stage1"]                    # (T_full, N_all) on device
    T_full, N_all = u0.shape
    device = next(model.parameters()).device
    gating = data.get("gating")
    stim = data.get("stim")
    lo, hi = model.u_clip
    ones = torch.ones(N_all, device=device)
    is_hetero = (model._noise_mode == "heteroscedastic")

    if not is_hetero:
        sigma_const = model.sigma_at().detach()

    motor_np = np.array(motor_idx, dtype=int)
    results: list[np.ndarray] = []

    with torch.no_grad():
        for k in range(n_samples):
            # ── Phase 1: teacher-forced burn-in ──
            s_sv  = torch.zeros(N_all, model.r_sv,  device=device)
            s_dcv = torch.zeros(N_all, model.r_dcv, device=device)
            for t in range(train_end):
                g = gating[t] if gating is not None else ones
                s = stim[t]   if stim   is not None else None
                _, s_sv, s_dcv = model.prior_step(u0[t], s_sv, s_dcv, g, s)

            # ── Phase 2: autonomous generation ──
            buf = np.zeros((n_steps, len(motor_idx)), dtype=np.float32)
            u_cur = u0[train_end].clone()
            buf[0] = u_cur.cpu().numpy()[motor_np]

            for t in range(n_steps - 1):
                t_global = train_end + t
                g = gating[t_global] if gating is not None and t_global < T_full else ones
                s = stim[t_global]   if stim   is not None and t_global < T_full else None
                mu_next, s_sv, s_dcv = model.prior_step(u_cur, s_sv, s_dcv, g, s)

                # Process noise (temperature-scaled)
                if is_hetero:
                    sigma_t = model.sigma_at(u_cur).detach() * temperature
                else:
                    sigma_t = sigma_const * temperature

                if getattr(model, 'noise_corr_rank', 0) > 0:
                    u_next = mu_next + model.sample_correlated_noise(sigma_t)
                else:
                    u_next = mu_next + sigma_t * torch.randn(N_all, device=device)

                u_next = u_next.clamp(min=lo, max=hi)
                u_cur = u_next
                buf[t + 1] = u_cur.cpu().numpy()[motor_np]

            results.append(buf)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Config: speed-optimised TRF
# ─────────────────────────────────────────────────────────────────────────────

def _make_quick_trf_config(context_length: int = 15,
                           max_epochs: int = 200,
                           patience: int = 25) -> TransformerBaselineConfig:
    """B_wide TRF config with configurable epochs/patience."""
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
    cfg.max_epochs = max_epochs
    cfg.patience = patience
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch _make_B_wide_config so _train_trf_neural_only / _train_trf_decoder
# use the quick config too.
# ─────────────────────────────────────────────────────────────────────────────

import scripts.free_run.run as _run_module
_orig_make_B_wide = _run_module._make_B_wide_config

def _quick_B_wide(context_length: int = 15) -> TransformerBaselineConfig:
    cfg = _orig_make_B_wide(context_length)
    # Use full training — no epoch reduction
    return cfg

_run_module._make_B_wide_config = _quick_B_wide


# ─────────────────────────────────────────────────────────────────────────────
# Incremental save helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_partial_results(
    out_dir: Path,
    worm_id: str,
    all_metrics: dict,
    all_samples_beh: dict,
    timings: dict,
    meta: dict,
):
    """Flush current metrics + sample arrays after each model completes."""
    # metrics JSON (always overwritten)
    partial = {
        **meta,
        "metrics_T1": all_metrics,
        "training_times": timings,
        "completed_models": list(all_metrics.keys()),
    }
    with open(out_dir / "results_partial.json", "w") as f:
        json.dump(partial, f, indent=2, cls=NumpyEncoder)

    # per-model .npz so samples survive a crash
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    for label, samples in all_samples_beh.items():
        safe = label.replace(" ", "_").replace("+", "_")
        np.savez_compressed(
            samples_dir / f"{safe}.npz",
            **{f"sample_{i}": s for i, s in enumerate(samples)},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Quick 11-model comparison on one worm")
    ap.add_argument("--h5", required=True, help="Path to worm .h5 file")
    ap.add_argument("--out_dir", default="output_plots/free_run/all_models_quick")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--K", type=int, default=3, help="Neural context length")
    ap.add_argument("--K_beh", type=int, default=5, help="Behaviour context length")
    ap.add_argument("--rollout_beh", type=int, default=10, help="Rollout steps for beh models")
    ap.add_argument("--neurons", choices=["motor", "all"], default="motor")
    ap.add_argument("--n_samples", type=int, default=3,
                    help="Ensemble size (3 = enough for median metrics + video)")
    ap.add_argument("--n_steps", type=int, default=50,
                    help="Number of free-run generation frames")
    ap.add_argument("--no_video", action="store_true", help="Skip video generation")
    ap.add_argument("--max_epochs", type=int, default=200,
                    help="Max TRF training epochs (default: 200)")
    ap.add_argument("--patience", type=int, default=25,
                    help="TRF early stopping patience (default: 25)")
    args = ap.parse_args()

    device = args.device
    dt = 0.6
    K = args.K
    K_beh = args.K_beh
    N_SAMPLES = args.n_samples
    N_STEPS = args.n_steps

    t_total = time.time()

    # ── Load data ────────────────────────────────────────────────────────
    worm_data = load_worm_data(args.h5, n_beh_modes=6)
    u, b, worm_id = worm_data["u"], worm_data["b"], worm_data["worm_id"]
    motor_idx = worm_data.get("motor_idx")
    all_labels = worm_data.get("labels", [])

    # Load raw body angle for realistic GT posture in videos
    import h5py as _h5
    _gt_body_angle_full = None
    with _h5.File(args.h5, "r") as _f:
        for _ba_key in ("behaviour/body_angle_dtarget",
                        "behavior/body_angle_all",
                        "behaviour/body_angle_all"):
            if _ba_key in _f:
                _gt_body_angle_full = np.asarray(_f[_ba_key][:], dtype=float)
                break

    if args.neurons == "motor" and motor_idx is not None:
        u_sel = u[:, motor_idx]
        sel_labels = [all_labels[i] if i < len(all_labels) else f"n{i}"
                      for i in motor_idx]
        neuron_label = "motor"
    else:
        u_sel = u
        sel_labels = all_labels
        neuron_label = "all"

    T, N, Kw = u_sel.shape[0], u_sel.shape[1], b.shape[1]
    D_joint = N + Kw

    # Strict temporal split: first half train, second half test
    train_end = T // 2
    n_test = T - train_end

    out_dir = Path(args.out_dir) / worm_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Worm: {worm_id}  T={T}  N={N} ({neuron_label})  Kw={Kw}")
    print(f"SPLIT: train=[0,{train_end})  test=[{train_end},{T})")
    print(f"  K={K}  K_beh={K_beh}  N_SAMPLES={N_SAMPLES}  N_STEPS={N_STEPS}")
    print(f"Output: {out_dir}\n")

    # ── Normalisation (train only) ───────────────────────────────────────
    u_train, b_train = u_sel[:train_end], b[:train_end]
    mu_u = u_train[K:].mean(0).astype(np.float32)
    sig_u = (u_train[K:].std(0) + 1e-8).astype(np.float32)
    mu_b = b_train[K:].mean(0).astype(np.float32)
    sig_b = (b_train[K:].std(0) + 1e-8).astype(np.float32)

    u_norm = ((u_sel - mu_u) / sig_u).astype(np.float32)
    b_norm = ((b - mu_b) / sig_b).astype(np.float32)

    gt_beh_test = b[train_end:T]
    gt_neural_test = u_sel[train_end:T]  # raw-scale neural GT for trace plots

    u_norm_train = u_norm[:train_end]
    b_norm_train = b_norm[:train_end]
    x_joint_train = np.concatenate([u_norm_train, b_norm_train], axis=1)

    # Lagged features
    X_u_lag = build_lagged(u_norm_train, K)
    X_j_lag = build_lagged(x_joint_train, K)
    tr_idx = np.arange(K, train_end)
    y_joint = x_joint_train[tr_idx]
    y_neural = u_norm_train[tr_idx]
    y_beh = b_norm_train[tr_idx]

    # Seeds
    u_seed = u_norm[train_end - K : train_end].copy()
    b_seed = b_norm[train_end - K_beh : train_end].copy()
    joint_seed = np.concatenate([u_seed, b_seed[-K:]], axis=1)

    # ═════════════════════════════════════════════════════════════════════
    # TRAIN ALL MODELS
    # ═════════════════════════════════════════════════════════════════════
    timings = {}

    # ─── TRF Joint ───────────────────────────────────────────────────────
    print("=" * 60); print("1/10  TRF Joint"); print("=" * 60)
    t0 = time.time()
    cfg = _make_quick_trf_config(context_length=K,
                                 max_epochs=args.max_epochs,
                                 patience=args.patience); cfg.device = device
    trf_joint_result = train_single_worm_cv(
        u=u_sel[:train_end], cfg=cfg, device=device, verbose=False,
        save_dir=None, worm_id=worm_id, b=b[:train_end],
    )
    trf_joint = trf_joint_result["best_model"]
    n_neural_trf = trf_joint_result["n_neural"]
    x_trf = trf_joint_result["x"]
    trf_seed = x_trf[train_end - K : train_end].copy()
    timings["TRF Joint"] = time.time() - t0
    print(f"  Done ({timings['TRF Joint']:.1f}s)")

    # ─── TRF Neural-only (shared by AR+Dec, Cascaded) ───────────────────
    print("\n2/10  TRF Neural-only (shared)")
    t0 = time.time()
    trf_neural, trf_neural_result = _train_trf_neural_only(u_sel[:train_end], K, device)
    trf_neural_seed = trf_neural_result["x"][train_end - K : train_end].copy()
    timings["TRF Neural-only"] = time.time() - t0
    print(f"  Done ({timings['TRF Neural-only']:.1f}s)")

    # ─── TRF Decoder (for AR+Dec) ───────────────────────────────────────
    print("\n3/10  TRF Decoder (neural→beh)")
    t0 = time.time()
    trf_decoder, dec_mu_u, dec_sig_u, dec_mu_b, dec_sig_b = _train_trf_decoder(
        u_sel[:train_end], b[:train_end], K, device)
    timings["TRF Decoder"] = time.time() - t0
    print(f"  Done ({timings['TRF Decoder']:.1f}s)")

    # TRF Cascaded Beh reuses TRF Joint weights (different generation loop)

    # ─── MLP Joint (rollout) ────────────────────────────────────────────
    _mlp_rollout = min(15, args.rollout_beh)
    print(f"\n4/10  MLP Joint (rollout={_mlp_rollout})")
    t0 = time.time()
    mlp_joint = train_mlp_rollout(x_joint_train, K, device, rollout_steps=_mlp_rollout)
    res_std_joint_mlp = estimate_residual_std(mlp_joint, X_j_lag[tr_idx], y_joint, device)
    timings["MLP Joint"] = time.time() - t0
    print(f"  Done ({timings['MLP Joint']:.1f}s)")

    # ─── MLP Neural AR (shared by AR+Dec, Cascaded) ─────────────────────
    print(f"\n5/10  MLP Neural AR (rollout={_mlp_rollout})")
    t0 = time.time()
    mlp_neural_ar = train_mlp_rollout(u_norm_train, K, device, rollout_steps=_mlp_rollout)
    res_std_u_mlp = estimate_residual_std(mlp_neural_ar, X_u_lag[tr_idx], y_neural, device)
    timings["MLP Neural AR"] = time.time() - t0
    print(f"  Done ({timings['MLP Neural AR']:.1f}s)")

    # ─── MLP Beh Decoder (for AR+Dec) ───────────────────────────────────
    print("\n6/10  MLP Beh Decoder (neural→beh)")
    t0 = time.time()
    mlp_beh_dec = train_mlp(X_u_lag[tr_idx], y_beh, device)
    res_std_b_mlp_dec = estimate_residual_std(mlp_beh_dec, X_u_lag[tr_idx], y_beh, device)
    timings["MLP Beh Decoder"] = time.time() - t0
    print(f"  Done ({timings['MLP Beh Decoder']:.1f}s)")

    # ─── MLP Cascaded Beh (rollout) ─────────────────────────────────────
    print(f"\n7/10  MLP Cascaded Beh (K_beh={K_beh}, rollout={args.rollout_beh})")
    t0 = time.time()
    mlp_beh_casc = train_mlp_beh_rollout(
        u_norm_train, b_norm_train, K_beh, device,
        rollout_steps=args.rollout_beh, input_mode="cascaded"
    )
    X_b_lag_ext = build_lagged(b_norm_train, K_beh)
    X_nb_casc = np.concatenate([u_norm_train[K_beh:], X_b_lag_ext[K_beh:]], axis=1)
    y_beh_ext = b_norm_train[K_beh:]
    res_std_b_mlp_casc = estimate_residual_std(mlp_beh_casc, X_nb_casc, y_beh_ext, device)
    timings["MLP Cascaded Beh"] = time.time() - t0
    print(f"  Done ({timings['MLP Cascaded Beh']:.1f}s)")

    # ─── Ridge Joint ────────────────────────────────────────────────────
    print("\n8/10  Ridge Joint")
    t0 = time.time()
    ridge_joint = train_ridge(X_j_lag[tr_idx], y_joint)
    res_std_joint_ridge = estimate_residual_std(ridge_joint, X_j_lag[tr_idx], y_joint, is_torch=False)
    timings["Ridge Joint"] = time.time() - t0
    print(f"  Done ({timings['Ridge Joint']:.1f}s)")

    # ─── Ridge Neural AR (shared) ───────────────────────────────────────
    print("\n   Ridge Neural AR (shared)")
    t0 = time.time()
    ridge_neural_ar = train_ridge(X_u_lag[tr_idx], y_neural)
    res_std_u_ridge = estimate_residual_std(ridge_neural_ar, X_u_lag[tr_idx], y_neural, is_torch=False)
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ─── Ridge Beh Decoder ──────────────────────────────────────────────
    print("\n   Ridge Beh Decoder")
    t0 = time.time()
    ridge_beh_dec = train_ridge(X_u_lag[tr_idx], y_beh)
    res_std_b_ridge_dec = estimate_residual_std(ridge_beh_dec, X_u_lag[tr_idx], y_beh, is_torch=False)
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ─── Ridge Cascaded Beh ─────────────────────────────────────────────
    print("\n   Ridge Cascaded Beh")
    t0 = time.time()
    ridge_beh_casc = train_ridge(X_nb_casc, y_beh_ext)
    res_std_b_ridge_casc = estimate_residual_std(ridge_beh_casc, X_nb_casc, y_beh_ext, is_torch=False)
    timings["Ridge (all)"] = time.time() - t0
    print(f"  Done ({timings['Ridge (all)']:.1f}s)")

    # ─── SetTRF-Small ───────────────────────────────────────────────────
    print("\n9/10  SetTRF-Small")
    t0 = time.time()
    settrf_cfg = SetTRFJointConfig(
        d_model=128, n_heads=4, n_encoder_layers=2, d_ff=256,
        neuron_embed_dim=32, context_length=K,
        dropout=0.15,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    atlas_idx = _get_atlas_indices(sel_labels)
    settrf_model = train_set_trf_joint(
        u_norm_train, b_norm_train, atlas_idx, settrf_cfg, device,
    )
    timings["SetTRF-Small"] = time.time() - t0
    print(f"  Done ({timings['SetTRF-Small']:.1f}s)")

    # SetTRF seeds
    settrf_u_seed = u_norm[train_end - K : train_end].copy()
    settrf_b_seed = b_norm[train_end - K : train_end].copy()

    # ─── Stage2 Connectome ──────────────────────────────────────────────
    print("\n10/11  Stage2 Connectome")
    t0 = time.time()
    try:
        stage2_model, stage2_data = _train_stage2_fast(
            args.h5, train_end, device)
        timings["Stage2 Connectome"] = time.time() - t0
        print(f"  Done ({timings['Stage2 Connectome']:.1f}s)")
        _stage2_ok = True
    except Exception as e:
        print(f"  ✗ Stage2 training FAILED: {e}")
        import traceback; traceback.print_exc()
        _stage2_ok = False
        timings["Stage2 Connectome"] = time.time() - t0

    train_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"All models trained in {train_time:.1f}s")
    print(f"{'='*60}")

    # ═════════════════════════════════════════════════════════════════════
    # CLAMP RANGES (±5σ of training data in normalised space)
    # ═════════════════════════════════════════════════════════════════════
    CLAMP_SIGMA = 5.0
    # Normalised data has μ≈0, σ≈1 → clamp to ±5
    clamp_norm_u = (-CLAMP_SIGMA * np.ones(N, dtype=np.float32),
                     CLAMP_SIGMA * np.ones(N, dtype=np.float32))
    clamp_norm_b = (-CLAMP_SIGMA * np.ones(Kw, dtype=np.float32),
                     CLAMP_SIGMA * np.ones(Kw, dtype=np.float32))
    # For TRF models that operate in raw (already z-scored internally):
    # use the TRF's own normalisation to compute raw-space clamp
    x_trf_train = x_trf[:train_end]
    trf_train_mu = x_trf_train.mean(0)
    trf_train_std = x_trf_train.std(0) + 1e-8
    clamp_trf = (trf_train_mu - CLAMP_SIGMA * trf_train_std,
                 trf_train_mu + CLAMP_SIGMA * trf_train_std)
    clamp_trf = (clamp_trf[0].astype(np.float32), clamp_trf[1].astype(np.float32))
    # For TRF decoder (separate normalisation)
    clamp_dec_b = (-CLAMP_SIGMA * np.ones(Kw, dtype=np.float32),
                    CLAMP_SIGMA * np.ones(Kw, dtype=np.float32))
    # For TRF neural-only
    x_trf_n_train = trf_neural_result["x"][:train_end]
    trf_n_mu = x_trf_n_train.mean(0)
    trf_n_std = x_trf_n_train.std(0) + 1e-8
    clamp_trf_n = (trf_n_mu - CLAMP_SIGMA * trf_n_std,
                   trf_n_mu + CLAMP_SIGMA * trf_n_std)
    clamp_trf_n = (clamp_trf_n[0].astype(np.float32), clamp_trf_n[1].astype(np.float32))

    # ═════════════════════════════════════════════════════════════════════
    # GENERATE STOCHASTIC SAMPLES
    # ═════════════════════════════════════════════════════════════════════

    # For joint normalised-space models (MLP/Ridge Joint): clamp covers [neural, beh]
    clamp_norm_joint = (np.concatenate([clamp_norm_u[0], clamp_norm_b[0]]),
                        np.concatenate([clamp_norm_u[1], clamp_norm_b[1]]))

    # Per-model temperature: AR+Dec and SetTRF get T=0.7, others T=1.0
    T_LOW = 0.7

    # Registry:  label → (gen_fn, gen_kwargs, denorm_u, denorm_b, temperature)
    configs = {}

    # Joint
    configs["TRF Joint"] = dict(
        fn=stochastic_joint_trf,
        kw=dict(model=trf_joint, seed=trf_seed, n_neural=n_neural_trf,
                n_steps=N_STEPS, device=device, clamp_range=clamp_trf),
        denorm_u=(1, 0), denorm_b=(1, 0), temperature=1.0,
    )
    configs["MLP Joint"] = dict(
        fn=stochastic_joint_mlp,
        kw=dict(mlp=mlp_joint, seed=joint_seed, n_steps=N_STEPS,
                n_neural=N, device=device, residual_std=res_std_joint_mlp,
                clamp_range=clamp_norm_joint),
        denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b), temperature=1.0,
    )
    configs["Ridge Joint"] = dict(
        fn=stochastic_joint_ridge,
        kw=dict(model=ridge_joint, seed=joint_seed, n_steps=N_STEPS,
                n_neural=N, residual_std=res_std_joint_ridge,
                clamp_range=clamp_norm_joint),
        denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b), temperature=1.0,
    )

    # AR+Dec  (temperature=0.7)
    dec_u_seed_norm = ((u_sel[train_end - K:train_end] - dec_mu_u) / dec_sig_u).astype(np.float32)
    configs["TRF AR+Dec"] = dict(
        fn=stochastic_ar_dec_trf,
        kw=dict(neural_model=trf_neural, beh_decoder_model=trf_decoder,
                u_seed=trf_neural_seed, b_seed=b_seed,
                n_steps=N_STEPS, n_neural=trf_neural_result["n_neural"],
                n_beh=Kw, device=device,
                clamp_range_u=clamp_trf_n, clamp_range_b=clamp_dec_b),
        denorm_u=(1, 0), denorm_b=(dec_sig_b, dec_mu_b), temperature=T_LOW,
    )
    configs["MLP AR+Dec"] = dict(
        fn=stochastic_ar_dec_mlp,
        kw=dict(neural_ar=mlp_neural_ar, beh_decoder=mlp_beh_dec,
                u_seed=u_seed, n_steps=N_STEPS,
                n_neural=N, n_beh=Kw, device=device,
                res_std_u=res_std_u_mlp, res_std_b=res_std_b_mlp_dec,
                clamp_range_u=clamp_norm_u, clamp_range_b=clamp_norm_b),
        denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b), temperature=T_LOW,
    )
    configs["Ridge AR+Dec"] = dict(
        fn=stochastic_ar_dec_ridge,
        kw=dict(neural_ar=ridge_neural_ar, beh_decoder=ridge_beh_dec,
                u_seed=u_seed, n_steps=N_STEPS,
                n_neural=N, n_beh=Kw,
                res_std_u=res_std_u_ridge, res_std_b=res_std_b_ridge_dec,
                clamp_range_u=clamp_norm_u, clamp_range_b=clamp_norm_b),
        denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b), temperature=T_LOW,
    )

    # TRF Cascaded beh clamp: beh portion of trf_joint scale
    clamp_trf_b = (clamp_trf[0][n_neural_trf:], clamp_trf[1][n_neural_trf:])

    # Cascaded
    configs["TRF Cascaded"] = dict(
        fn=stochastic_cascaded_trf,
        kw=dict(neural_model=trf_neural, beh_model=trf_joint,
                u_seed=trf_neural_seed,
                b_seed=trf_seed[:, n_neural_trf:],
                n_steps=N_STEPS, n_neural=trf_neural_result["n_neural"],
                n_beh=Kw, device=device,
                clamp_range_u=clamp_trf_n, clamp_range_b=clamp_trf_b),
        denorm_u=(1, 0), denorm_b=(1, 0), temperature=1.0,
    )
    configs["MLP Cascaded"] = dict(
        fn=stochastic_cascaded_mlp,
        kw=dict(neural_ar=mlp_neural_ar, beh_model=mlp_beh_casc,
                u_seed=u_seed, b_seed=b_seed, n_steps=N_STEPS,
                n_neural=N, n_beh=Kw, device=device,
                res_std_u=res_std_u_mlp, res_std_b=res_std_b_mlp_casc,
                K_beh=K_beh,
                clamp_range_u=clamp_norm_u, clamp_range_b=clamp_norm_b),
        denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b), temperature=1.0,
    )
    configs["Ridge Cascaded"] = dict(
        fn=stochastic_cascaded_ridge,
        kw=dict(neural_ar=ridge_neural_ar, beh_model=ridge_beh_casc,
                u_seed=u_seed, b_seed=b_seed, n_steps=N_STEPS,
                n_neural=N, n_beh=Kw,
                res_std_u=res_std_u_ridge, res_std_b=res_std_b_ridge_casc,
                clamp_range_u=clamp_norm_u, clamp_range_b=clamp_norm_b),
        denorm_u=(sig_u, mu_u), denorm_b=(sig_b, mu_b), temperature=1.0,
    )

    # SetTRF-Small (separate because it has its own generation fn)
    # We'll handle it after the loop.

    # ── Run all 9 standard configs ───────────────────────────────────────
    all_metrics = {}
    all_samples_beh = {}
    all_samples_neural = {}  # also collect neural traces for plotting
    failed_models = []

    # Metadata for partial saves
    _meta = dict(worm_id=worm_id, T=T, N=N, Kw=Kw, K=K, K_beh=K_beh,
                 train_end=train_end, n_samples=N_SAMPLES, n_steps=N_STEPS, dt=dt)

    print(f"\nGenerating {N_SAMPLES} × {N_STEPS}-step stochastic samples per model")
    print(f"  Default T=1.0, AR+Dec T={T_LOW}, SetTRF T={T_LOW}")
    print(f"  Clamping at ±{CLAMP_SIGMA}σ")
    print("=" * 60)

    for label, spec in configs.items():
        try:
            fn, kw = spec["fn"], spec["kw"]
            s_u, off_u = spec["denorm_u"]
            s_b, off_b = spec["denorm_b"]
            temp = spec.get("temperature", 1.0)

            samples_beh = []
            samples_neural = []
            for i in range(N_SAMPLES):
                pn, pb = fn(temperature=temp, **kw)
                pb_raw = np.asarray(pb, dtype=np.float64) * s_b + off_b
                pn_raw = np.asarray(pn, dtype=np.float64) * s_u + off_u
                samples_beh.append(pb_raw.astype(np.float32))
                samples_neural.append(pn_raw.astype(np.float32))

            all_samples_beh[label] = samples_beh
            all_samples_neural[label] = samples_neural

            met_list = [compute_distributional_metrics(gt_beh_test[:N_STEPS], s)[0]
                        for s in samples_beh]
            med = ensemble_median_metrics(met_list)
            all_metrics[label] = med
            compute_distributional_metrics(gt_beh_test[:N_STEPS], samples_beh[0], label=label)

            # ── incremental save after each model ──
            _save_partial_results(out_dir, worm_id, all_metrics,
                                  all_samples_beh, timings, _meta)
            print(f"    [saved {label}]")

        except Exception as e:
            print(f"  ✗ {label} FAILED: {e}")
            import traceback; traceback.print_exc()
            failed_models.append(label)

    # ── SetTRF-Small generation ──────────────────────────────────────────
    try:
        print(f"\n  SetTRF-Small (T={T_LOW}):")
        settrf_samples_beh = []
        settrf_samples_neural = []
        for i in range(N_SAMPLES):
            pn, pb = stochastic_joint_settrf(
                settrf_model, settrf_u_seed, settrf_b_seed,
                N_STEPS, device, temperature=T_LOW,
                clamp_range_u=clamp_norm_u, clamp_range_b=clamp_norm_b)
            pb_raw = (pb * sig_b + mu_b).astype(np.float32)
            pn_raw = (pn * sig_u + mu_u).astype(np.float32)
            settrf_samples_beh.append(pb_raw)
            settrf_samples_neural.append(pn_raw)

        all_samples_beh["SetTRF-Small"] = settrf_samples_beh
        all_samples_neural["SetTRF-Small"] = settrf_samples_neural
        met_list = [compute_distributional_metrics(gt_beh_test[:N_STEPS], s)[0]
                    for s in settrf_samples_beh]
        all_metrics["SetTRF-Small"] = ensemble_median_metrics(met_list)
        compute_distributional_metrics(gt_beh_test[:N_STEPS], settrf_samples_beh[0],
                                       label="SetTRF-Small")

        _save_partial_results(out_dir, worm_id, all_metrics,
                              all_samples_beh, timings, _meta)
        print(f"    [saved SetTRF-Small]")

    except Exception as e:
        print(f"  ✗ SetTRF-Small FAILED: {e}")
        import traceback; traceback.print_exc()
        failed_models.append("SetTRF-Small")

    # ── Stage2 Connectome generation ─────────────────────────────────────
    if _stage2_ok:
        try:
            print(f"\n  Stage2 Connectome:")
            s2_neural_samples = _stage2_generate(
                stage2_model, stage2_data, train_end, N_STEPS,
                N_SAMPLES, motor_idx)
            # Decode behaviour via Ridge decoder (reuse ridge_beh_dec)
            # Neural samples are in raw u_sel space.
            # Normalise → build lagged features → decode → denormalise.
            s2_beh_samples = []
            s2_neural_final = []
            # Seed context: last K frames of normalised training data for lag
            u_norm_seed_for_lag = u_norm[train_end - K : train_end].copy()
            for k_s in range(N_SAMPLES):
                pn_raw = s2_neural_samples[k_s]        # (n_steps, N_motor), raw
                pn_norm = ((pn_raw - mu_u) / sig_u).astype(np.float32)
                # Prepend K frames from training data so lagged features are valid
                pn_with_ctx = np.concatenate(
                    [u_norm_seed_for_lag, pn_norm], axis=0)  # (K+n_steps, N)
                X_lag_full = build_lagged(pn_with_ctx, K)    # (K+n_steps, K*N)
                # Predict beh from index K onward (= test portion)
                pb_norm = ridge_beh_dec.predict(X_lag_full[K:])  # (n_steps, Kw)
                pb_raw = (pb_norm * sig_b + mu_b).astype(np.float32)
                s2_beh_samples.append(pb_raw)
                s2_neural_final.append(pn_raw.astype(np.float32))

            all_samples_beh["Stage2 Connectome"] = s2_beh_samples
            all_samples_neural["Stage2 Connectome"] = s2_neural_final

            met_list = [compute_distributional_metrics(gt_beh_test[:N_STEPS], s)[0]
                        for s in s2_beh_samples]
            all_metrics["Stage2 Connectome"] = ensemble_median_metrics(met_list)
            compute_distributional_metrics(gt_beh_test[:N_STEPS], s2_beh_samples[0],
                                           label="Stage2 Connectome")

            _save_partial_results(out_dir, worm_id, all_metrics,
                                  all_samples_beh, timings, _meta)
            print(f"    [saved Stage2 Connectome]")

        except Exception as e:
            print(f"  ✗ Stage2 Connectome generation FAILED: {e}")
            import traceback; traceback.print_exc()
            failed_models.append("Stage2 Connectome")
    else:
        failed_models.append("Stage2 Connectome")

    gen_time = time.time() - t_total - train_time
    print(f"\nGeneration done in {gen_time:.1f}s")
    if failed_models:
        print(f"  ⚠ Failed models ({len(failed_models)}): {', '.join(failed_models)}")
    print(f"  ✓ Succeeded: {len(all_metrics)}/{len(configs)+2}")

    # ═════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Generating plots")
    print("=" * 60)

    model_names_ordered = list(all_samples_beh.keys())
    gt_short = gt_beh_test[:N_STEPS]
    fs = 1.0 / dt

    # Summary bars
    plot_summary_bars(all_metrics, out_dir, worm_id)

    # PSD comparison
    f_gt, psd_gt = compute_psd(gt_short, fs=fs)
    all_psd_gen = []
    for label in model_names_ordered:
        psds = [compute_psd(s, fs=fs)[1] for s in all_samples_beh[label]]
        all_psd_gen.append(psds)
    plot_psd_comparison(f_gt, psd_gt, all_psd_gen, model_names_ordered,
                        out_dir, worm_id)

    # Autocorrelation
    acf_gt = compute_autocorr(gt_short)
    all_acf_gen = []
    for label in model_names_ordered:
        acfs = [compute_autocorr(s) for s in all_samples_beh[label]]
        all_acf_gen.append(acfs)
    plot_autocorr_comparison(acf_gt, all_acf_gen, model_names_ordered,
                             out_dir, worm_id, dt=dt)

    # Marginals
    plot_marginals(gt_short,
                   [all_samples_beh[n] for n in model_names_ordered],
                   model_names_ordered, out_dir, worm_id)

    # Neural traces — GT vs generated, one neuron per model
    _plot_neural_traces(
        gt_neural=gt_neural_test[:N_STEPS],
        all_neural=all_samples_neural,
        model_names=model_names_ordered,
        sel_labels=sel_labels,
        out_dir=out_dir,
        worm_id=worm_id,
        dt=dt,
    )

    # ═════════════════════════════════════════════════════════════════════
    # POSTURE VIDEO — all 10 models side-by-side
    # ═════════════════════════════════════════════════════════════════════
    if not args.no_video:
        print("\n" + "=" * 60)
        print("Generating side-by-side posture comparison video")
        print("=" * 60)

        # Build dict: model_name → first sample (n_steps, Kw)
        video_samples = {name: samples[0][:N_STEPS]
                         for name, samples in all_samples_beh.items()}

        video_path = str(out_dir / f"all_models_{worm_id}.mp4")
        video_fps = round(1.0 / dt, 2)   # 1.67 fps → real-time at dt=0.6s

        # Pass raw body angle (heading-removed) for realistic GT posture
        _gt_ba_for_video = None
        if _gt_body_angle_full is not None:
            _ba_test = _gt_body_angle_full[train_end:T]
            # Remove heading (per-frame mean) — eigenworms don't encode heading
            _heading = np.nanmean(_ba_test, axis=1, keepdims=True)
            _gt_ba_for_video = _ba_test - _heading

        try:
            make_multi_model_video(
                h5_path=args.h5,
                out_path=video_path,
                gt_beh=gt_beh_test,
                model_samples=video_samples,
                max_frames=N_STEPS,
                fps=video_fps,
                dpi=120,
                model_metrics=all_metrics,
                gt_body_angle=_gt_ba_for_video,
            )
        except Exception as e:
            print(f"  ✗ Video failed: {e}")
            import traceback; traceback.print_exc()

    # ═════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═════════════════════════════════════════════════════════════════════
    save_data = {
        "worm_id": worm_id,
        "T": T, "N": N, "Kw": Kw, "K": K,
        "K_beh": K_beh,
        "train_end": train_end,
        "n_samples": N_SAMPLES,
        "n_steps": N_STEPS,
        "dt": dt,
        "metrics_T1": all_metrics,
        "training_times": timings,
        "total_time_s": time.time() - t_total,
        "failed_models": failed_models,
        "completed_models": list(all_metrics.keys()),
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {json_path}")
    # Clean up partial file now that final is written
    partial_path = out_dir / "results_partial.json"
    if partial_path.exists():
        partial_path.unlink()

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"SUMMARY — {worm_id}  ({N_STEPS} frames, {N_SAMPLES} samples, T=1.0)")
    print("=" * 70)
    print(f"{'Model':<22} {'PSD↓':>8} {'ACF↓':>8} {'W1↓':>8} {'KS↓':>8} {'VarR→1':>8}")
    print("-" * 66)
    for name in model_names_ordered:
        m = all_metrics[name]
        print(f"{name:<22} {m['psd_log_distance']:>8.3f} {m['autocorr_rmse']:>8.3f} "
              f"{m['wasserstein_1']:>8.3f} {m['ks_statistic']:>8.3f} "
              f"{m['variance_ratio_mean']:>8.3f}")

    wall = time.time() - t_total
    print(f"\nTotal wall time: {wall:.0f}s ({wall/60:.1f} min)")
    print(f"Plots + video in {out_dir}")


if __name__ == "__main__":
    main()
