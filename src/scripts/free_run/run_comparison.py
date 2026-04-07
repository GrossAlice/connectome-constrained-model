#!/usr/bin/env python3
"""
Free-Run Architecture Comparison — TRF Joint vs SetTRF Joint variants.

Trains the standard TRF Joint (baseline_transformer) AND several
SetTransformerJoint architecture variants on 3 worms, generates
stochastic free-run samples, distributional metrics, plots, and
side-by-side posture comparison videos.

Usage
-----
    python -m scripts.free_run.run_comparison \
        --h5_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
        --worms 2022-06-14-01 2022-06-14-07 2022-06-14-13 \
        --out_dir output_plots/free_run/comparison_videos \
        --device cuda
"""
from __future__ import annotations

import argparse, glob, json, sys, time, warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.dataset import load_worm_data
from baseline_transformer.train import train_single_worm_cv
from stage2.io_multi import _load_full_atlas
from stage2.io_h5 import _recover_labels_to_atlas

# Reuse the SetTRF building blocks from existing script
from scripts.free_run.run_set_transformer_joint import (
    SetTRFJointConfig, SetTransformerJoint,
    _get_atlas_indices, train_set_trf_joint, stochastic_joint_settrf,
)
from scripts.free_run.utils import (
    compute_distributional_metrics, ensemble_median_metrics,
    compute_psd, compute_autocorr,
    plot_ensemble_traces, plot_psd_comparison, plot_autocorr_comparison,
    plot_marginals, plot_summary_bars,
    NumpyEncoder,
)

# Import stochastic_joint_trf for the baseline TRF model
from scripts.free_run.run import stochastic_joint_trf, _make_B_wide_config

COLORS = ["#333333",   # GT
          "#E24A33",   # TRF Joint
          "#348ABD",   # SetTRF Default
          "#2CA02C",   # SetTRF Small
          "#9467BD",   # SetTRF Deep
          "#FF7F0E",   # SetTRF LongCtx
          "#D62728",   # SetTRF Wide
          "#1F77B4"]   # extra


# ══════════════════════════════════════════════════════════════════════════════
# Architecture Variants
# ══════════════════════════════════════════════════════════════════════════════

def get_settrf_variants() -> Dict[str, SetTRFJointConfig]:
    """Return named SetTRF architecture variants to test.

    Design rationale for each variant:
    - Default:  Current architecture (d=256, h=8, L=2) — known to explode
    - Small:    Fewer params → less capacity to memorise, may regularise better
    - Deep:     More layers but same width → richer representations
    - LongCtx: Longer context window (K=30) → more history for stability
    - WideFF:  Wider feed-forward (d_ff=1024) → more expressive transform
    """
    variants = {}

    # 1. Default (same as original batch run)
    variants["SetTRF-Default"] = SetTRFJointConfig(
        d_model=256, n_heads=8, n_encoder_layers=2, d_ff=512,
        neuron_embed_dim=64, context_length=15,
        dropout=0.1,
    )

    # 2. Small — fewer params, stronger implicit regularisation
    variants["SetTRF-Small"] = SetTRFJointConfig(
        d_model=128, n_heads=4, n_encoder_layers=2, d_ff=256,
        neuron_embed_dim=32, context_length=15,
        dropout=0.15,
    )

    # 3. Deep — more encoder layers, moderate width
    variants["SetTRF-Deep"] = SetTRFJointConfig(
        d_model=192, n_heads=8, n_encoder_layers=4, d_ff=384,
        neuron_embed_dim=48, context_length=15,
        dropout=0.15,
    )

    # 4. Long context — K=30 instead of 15 (~18s history)
    variants["SetTRF-LongCtx"] = SetTRFJointConfig(
        d_model=256, n_heads=8, n_encoder_layers=2, d_ff=512,
        neuron_embed_dim=64, context_length=30,
        dropout=0.1,
    )

    # 5. Wide feed-forward — wider MLP in attention blocks
    variants["SetTRF-WideFF"] = SetTRFJointConfig(
        d_model=256, n_heads=8, n_encoder_layers=2, d_ff=1024,
        neuron_embed_dim=64, context_length=15,
        dropout=0.15,
    )

    return variants


# ══════════════════════════════════════════════════════════════════════════════
# TRF Joint — train and generate
# ══════════════════════════════════════════════════════════════════════════════

def train_and_sample_trf_joint(
    h5_path: str,
    device: str,
    neurons: str,
    n_samples: int,
    K: int = 15,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], dict, dict]:
    """Train standard TRF Joint and generate stochastic samples.

    Returns: (gt_beh_test, samples_beh, samples_neural, metrics, info_dict)
    """
    print("  ── Training TRF Joint (baseline transformer) ──")
    worm_data = load_worm_data(h5_path, n_beh_modes=6)
    worm_id = worm_data["worm_id"]
    u_all = worm_data["u"]
    b_raw = worm_data["b"]
    dt = worm_data.get("dt", 0.6)
    motor_idx = worm_data.get("motor_idx")
    all_labels = worm_data.get("labels", [])

    # Neuron selection
    if neurons == "motor" and motor_idx is not None:
        u_raw = u_all[:, motor_idx]
        sel_labels = [all_labels[i] if i < len(all_labels) else f"n{i}"
                      for i in motor_idx]
    else:
        u_raw = u_all
        sel_labels = all_labels

    T, N = u_raw.shape
    Kw = b_raw.shape[1]

    # Temporal split
    train_end = T // 2
    n_test = T - train_end

    # Normalise (from training portion, after lag window)
    mu_u = u_raw[:train_end][K:].mean(0).astype(np.float32)
    sig_u = (u_raw[:train_end][K:].std(0) + 1e-8).astype(np.float32)
    mu_b = b_raw[:train_end][K:].mean(0).astype(np.float32)
    sig_b = (b_raw[:train_end][K:].std(0) + 1e-8).astype(np.float32)

    gt_beh_test = b_raw[train_end:]

    # Train TRF Joint (using the existing CV training pipeline)
    cfg = _make_B_wide_config(context_length=K)
    cfg.include_beh_input = True
    cfg.predict_beh = True
    cfg.n_cv_folds = 2  # Fewer folds for speed

    t0 = time.time()
    result = train_single_worm_cv(
        u_raw[:train_end], cfg, device,
        verbose=True, worm_id=worm_id,
        b=b_raw[:train_end],
    )
    elapsed = time.time() - t0
    print(f"  TRF Joint trained in {elapsed:.1f}s")

    trf_model = result["best_model"]
    n_neural_trf = result["n_neural"]

    # Seed from normalised data in the model's coordinate system
    x_all = result["x"]          # (T_train, D) already normalised by train_single_worm_cv
    trf_seed = x_all[train_end - K : train_end].copy()

    # Generate samples
    print(f"  Generating {n_samples} TRF Joint samples ...")
    samples_neural, samples_beh = [], []
    for i in range(n_samples):
        pn, pb = stochastic_joint_trf(
            trf_model, trf_seed, n_neural_trf, n_test, device,
            temperature=1.0)
        # TRF already works in its own normalised space — denorm is (1,0)
        samples_neural.append(pn.astype(np.float32))
        samples_beh.append(pb.astype(np.float32))
        if (i + 1) % 5 == 0:
            print(f"    sample {i+1}/{n_samples}")

    # Distributional metrics
    met_list = []
    for s in samples_beh:
        m, _ = compute_distributional_metrics(gt_beh_test, s)
        met_list.append(m)
    med_metrics = ensemble_median_metrics(met_list)
    compute_distributional_metrics(gt_beh_test, samples_beh[0], label="TRF Joint")

    info = dict(
        worm_id=worm_id, T=T, N=N, Kw=Kw, K=K, dt=dt,
        train_end=train_end, n_test=n_test,
        gt_beh_test=gt_beh_test,
        h5_path=h5_path,
        u_raw=u_raw, b_raw=b_raw,
        mu_u=mu_u, sig_u=sig_u, mu_b=mu_b, sig_b=sig_b,
        sel_labels=sel_labels, neurons=neurons,
        motor_idx=motor_idx, all_labels=all_labels,
    )

    return gt_beh_test, samples_beh, samples_neural, med_metrics, info


# ══════════════════════════════════════════════════════════════════════════════
# SetTRF Joint — train and generate  (one variant)
# ══════════════════════════════════════════════════════════════════════════════

def train_and_sample_settrf(
    h5_path: str,
    cfg: SetTRFJointConfig,
    variant_name: str,
    device: str,
    neurons: str,
    n_samples: int,
    info: dict,
) -> Tuple[List[np.ndarray], List[np.ndarray], dict]:
    """Train one SetTRF variant and generate samples.

    Returns: (samples_beh, samples_neural, metrics)
    """
    print(f"  ── Training {variant_name} ──")
    worm_id = info["worm_id"]
    u_raw, b_raw = info["u_raw"], info["b_raw"]
    mu_u, sig_u = info["mu_u"], info["sig_u"]
    mu_b, sig_b = info["mu_b"], info["sig_b"]
    train_end = info["train_end"]
    n_test = info["n_test"]
    gt_beh_test = info["gt_beh_test"]
    K = cfg.context_length

    # Normalise
    u_norm = ((u_raw - mu_u) / sig_u).astype(np.float32)
    b_norm = ((b_raw - mu_b) / sig_b).astype(np.float32)

    # Get neuron labels and atlas indices from info dict
    sel_labels = info.get("sel_labels", [f"n{i}" for i in range(u_raw.shape[1])])
    atlas_idx = _get_atlas_indices(sel_labels)

    # Train
    t0 = time.time()
    model = train_set_trf_joint(
        u_norm[:train_end], b_norm[:train_end], atlas_idx, cfg, device,
    )
    elapsed = time.time() - t0
    print(f"  {variant_name} trained in {elapsed:.1f}s")

    # Seeds
    u_seed = u_norm[train_end - K : train_end].copy()
    b_seed = b_norm[train_end - K : train_end].copy()

    # Generate samples
    print(f"  Generating {n_samples} {variant_name} samples ...")
    samples_neural, samples_beh = [], []
    for i in range(n_samples):
        pn, pb = stochastic_joint_settrf(
            model, u_seed, b_seed, n_test, device, temperature=1.0)
        pn_raw = (pn * sig_u + mu_u).astype(np.float32)
        pb_raw = (pb * sig_b + mu_b).astype(np.float32)
        samples_neural.append(pn_raw)
        samples_beh.append(pb_raw)
        if (i + 1) % 5 == 0:
            print(f"    sample {i+1}/{n_samples}")

    # Distributional metrics
    met_list = []
    for s in samples_beh:
        m, _ = compute_distributional_metrics(gt_beh_test, s)
        met_list.append(m)
    med_metrics = ensemble_median_metrics(met_list)
    compute_distributional_metrics(gt_beh_test, samples_beh[0], label=variant_name)

    return samples_beh, samples_neural, med_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Multi-model side-by-side posture video
# ══════════════════════════════════════════════════════════════════════════════

def _angles_to_xy(body_angles: np.ndarray):
    """Convert body (tangent) angles → (x, y) coordinates.

    The eigenworm reconstruction  ew @ E.T  already gives tangent angles θ(s),
    so we go straight to  x = Σ cos(θ),  y = Σ sin(θ)  — no extra cumsum.
    """
    x = np.cumsum(np.cos(body_angles))
    y = np.cumsum(np.sin(body_angles))
    x -= np.nanmean(x); y -= np.nanmean(y)
    return x, y


def _make_worm_lc(x, y, curvature, cmap, norm, lw_head=5.0, lw_tail=1.5):
    """Create a :class:`LineCollection` coloured by curvature with tapered width."""
    from matplotlib.collections import LineCollection
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    n = len(segments)
    # Tapered linewidth: thick at head (index 0), thin at tail
    lws = np.linspace(lw_head, lw_tail, n)
    c = 0.5 * (curvature[:-1] + curvature[1:])
    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        linewidths=lws, capstyle="round")
    lc.set_array(c)
    return lc


def make_multi_model_video(
    h5_path: str,
    out_path: str,
    gt_beh: np.ndarray,
    model_samples: Dict[str, np.ndarray],  # name → (n_test, Kw) first sample
    max_frames: int = 300,
    fps: int = 15,
    dpi: int = 100,
    model_metrics: Optional[Dict[str, dict]] = None,
    gt_body_angle: Optional[np.ndarray] = None,
):
    """Create a side-by-side posture comparison video with GT + N models.

    Panels: [GT | Model1 | Model2 | ... ]

    If *gt_body_angle* is provided (raw tangent angles, heading-removed,
    shape ``(T_test, n_seg)``), the GT panel uses it directly instead of
    the 6-mode eigenworm reconstruction — this preserves fine postural
    details and looks much more realistic.

    Model-predicted eigenworm coefficients are rescaled per-frame so that
    the spatial standard deviation of the reconstructed tangent angles
    matches the GT.  This prevents variance-exploding models from
    producing tight spirals (23+ body revolutions) while preserving the
    predicted *shape* (which modes dominate, S-curve vs C-curve, etc.).
    The original VarR is shown in each panel title for reference.
    """
    import h5py
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors

    # Load eigenvectors and body angle
    with h5py.File(h5_path, "r") as f:
        _ew_d_target = None
        if "behaviour/eigenworms_stephens" in f:
            _ew_d_target = f["behaviour/eigenworms_stephens"].attrs.get("d_target")
            if _ew_d_target is not None:
                _ew_d_target = int(_ew_d_target)

        for ew_key in ("behaviour/eigenworms_stephens",):
            if ew_key in f:
                break
        ew_raw = np.asarray(f[ew_key][:], dtype=float)

        sr = 1.0
        if "stage1/params" in f and "sample_rate_hz" in f["stage1/params"].attrs:
            sr = float(f["stage1/params"].attrs["sample_rate_hz"])
        elif "sample_rate_hz" in f.attrs:
            sr = float(f.attrs["sample_rate_hz"])

    dt = 1.0 / sr

    # Load eigenvectors
    from stage2.posture_videos import _load_eigenvectors
    eigvecs_all = _load_eigenvectors(eigvec_npy=None, h5_path=h5_path,
                                     d_target=_ew_d_target)

    n_modes = min(ew_raw.shape[1], gt_beh.shape[1], eigvecs_all.shape[1], 6)
    E = eigvecs_all[:, :n_modes]

    T = min(ew_raw.shape[0], gt_beh.shape[0], max_frames)
    for name, s in model_samples.items():
        T = min(T, s.shape[0])
    if gt_body_angle is not None:
        T = min(T, gt_body_angle.shape[0])

    d_recon = E.shape[0]

    # Reconstruct postures
    model_names = list(model_samples.keys())
    n_panels = 1 + len(model_names)  # GT + models

    # GT reconstruction — use raw body angle if available, else eigenworm recon
    gt_ew = gt_beh[:T, :n_modes]
    if gt_body_angle is not None:
        # Use real body angle (heading-removed) resampled to d_recon points
        ba = gt_body_angle[:T]
        n_seg_orig = ba.shape[1]
        if n_seg_orig != d_recon:
            xp = np.linspace(0, 1, n_seg_orig)
            xq = np.linspace(0, 1, d_recon)
            recon_gt = np.array([np.interp(xq, xp, ba[t]) for t in range(T)])
        else:
            recon_gt = ba.copy()
    else:
        recon_gt = gt_ew @ E.T  # (T, d_recon)

    # GT spatial std per frame (for angle normalisation of models)
    gt_spatial_std = np.array([np.std(recon_gt[t]) for t in range(T)])  # (T,)
    gt_spatial_std = np.maximum(gt_spatial_std, 1e-8)

    # Model reconstructions — normalise bending intensity to match GT
    recon_models = {}
    var_ratios = {}
    for name, s in model_samples.items():
        recon = s[:T, :n_modes] @ E.T  # (T, d_recon)
        gt_var = np.var(gt_ew, axis=0) + 1e-12
        mod_var = np.var(s[:T, :n_modes], axis=0) + 1e-12
        var_ratios[name] = float(np.mean(mod_var / gt_var))
        for t in range(T):
            sigma_m = np.std(recon[t])
            if sigma_m > 1e-8:
                recon[t] *= gt_spatial_std[t] / sigma_m
        recon_models[name] = recon

    # Collect all angle arrays (for curvature colouring)
    all_recon = [recon_gt] + [recon_models[n] for n in model_names]

    # Precompute curvature (local angular change) for colormap limits
    all_curv = []
    for recon in all_recon:
        curv = np.zeros_like(recon)
        curv[:, 1:] = np.diff(recon, axis=1)
        curv[:, 0] = curv[:, 1]
        all_curv.append(curv)
    curv_abs_max = np.percentile(np.abs(np.concatenate(
        [c.ravel() for c in all_curv])), 99)
    curv_norm = mcolors.Normalize(vmin=-curv_abs_max, vmax=curv_abs_max)
    curv_cmap = plt.get_cmap("coolwarm")

    # Convert to XY
    def to_xy(recon, T):
        xy = np.zeros((T, d_recon, 2), dtype=float)
        for t in range(T):
            x, y = _angles_to_xy(recon[t])
            xy[t, :, 0], xy[t, :, 1] = x, y
        return xy

    all_xy_list = [to_xy(r, T) for r in all_recon]

    # Shared axis limits
    all_px = np.concatenate([xy[:, :, 0].ravel() for xy in all_xy_list])
    all_py = np.concatenate([xy[:, :, 1].ravel() for xy in all_xy_list])
    m = np.isfinite(all_px) & np.isfinite(all_py)
    if m.any():
        xmin, xmax = np.nanmin(all_px[m]), np.nanmax(all_px[m])
        ymin, ymax = np.nanmin(all_py[m]), np.nanmax(all_py[m])
    else:
        xmin, xmax, ymin, ymax = -1, 1, -1, 1
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = 0.5 * max((xmax - xmin), (ymax - ymin)) + 2.0

    # Figure — 2-row grid for better visibility
    import math
    n_cols = math.ceil(n_panels / 2)
    n_rows = 2 if n_panels > 1 else 1
    fig_w = max(14, 4.5 * n_cols)
    fig_h = 4.5 * n_rows + 1.0
    fig, axes_2d = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                                facecolor="white")
    # Flatten and hide unused axes
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes_2d]
    else:
        axes_flat = list(axes_2d.ravel())
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    axes = axes_flat[:n_panels]

    # Build titles with VarR annotation
    titles = ["Ground Truth"]
    for name in model_names:
        vr = var_ratios.get(name)
        if vr is not None:
            titles.append(f"{name}\nVarR={vr:.2f}")
        else:
            titles.append(name)

    panel_colors = [COLORS[i % len(COLORS)] for i in range(n_panels)]

    heads, tails = [], []
    for i, (ax, ttl) in enumerate(zip(axes, titles)):
        ax.set_title(ttl, fontsize=11, fontweight="bold", color=panel_colors[i])
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f7f7")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for s in ax.spines.values():
            s.set_visible(False)
        head, = ax.plot([], [], "o", ms=7, alpha=0.95, color="black", zorder=10)
        tail, = ax.plot([], [], "s", ms=4, alpha=0.6, color="gray", zorder=10)
        heads.append(head)
        tails.append(tail)

    time_text = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=12)
    worm_id = Path(h5_path).stem
    fig.suptitle(f"Posture Comparison — {worm_id}", fontsize=14, fontweight="bold", y=0.99)
    fig.subplots_adjust(hspace=0.35, wspace=0.15)

    # Hold references to LineCollections so we can remove them each frame
    lc_holders: List[Optional[object]] = [None] * n_panels

    def _update(frame):
        for i in range(n_panels):
            # Remove previous LineCollection
            if lc_holders[i] is not None:
                lc_holders[i].remove()
                lc_holders[i] = None

            xy = all_xy_list[i]
            x = xy[frame, :, 0]
            y = xy[frame, :, 1]

            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                heads[i].set_data([], [])
                tails[i].set_data([], [])
                continue

            curv = all_curv[i][frame]
            lc = _make_worm_lc(x, y, curv, curv_cmap, curv_norm)
            axes[i].add_collection(lc)
            lc_holders[i] = lc

            heads[i].set_data([x[0]], [y[0]])
            tails[i].set_data([x[-1]], [y[-1]])

        time_text.set_text(f"t = {frame * dt:.2f}s   ({frame + 1}/{T})")
        return heads + tails + [time_text]

    anim = FuncAnimation(fig, _update, frames=T,
                         interval=max(1, 1000 // max(1, fps)), blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": "Multi-model posture comparison"},
                          bitrate=3000)
    print(f"  Rendering {T} frames at {fps} fps → {out_path}")
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Video: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Comparison plots (all models on one figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison_ensemble(
    gt_beh: np.ndarray,
    model_samples: Dict[str, List[np.ndarray]],
    out_dir: Path,
    worm_id: str,
    n_show: int = 300,
    dt: float = 0.6,
):
    """Overlay ensemble traces for all models on same axes."""
    Kw = gt_beh.shape[1]
    n_modes = min(Kw, 4)
    t_sec = np.arange(n_show) * dt
    model_names = list(model_samples.keys())

    fig, axes = plt.subplots(n_modes, 1, figsize=(16, 3 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    for row in range(n_modes):
        ax = axes[row]
        # GT
        ax.plot(t_sec, gt_beh[:n_show, row], color="#333", lw=2, alpha=0.9, label="GT")

        for mi, name in enumerate(model_names):
            c = COLORS[(mi + 1) % len(COLORS)]
            samples = model_samples[name]
            # Ensemble ribbon
            arr = np.array([s[:n_show, row] for s in samples[:10]])
            mean = arr.mean(0)
            q25, q75 = np.percentile(arr, [25, 75], axis=0)
            ax.fill_between(t_sec, q25, q75, color=c, alpha=0.15)
            ax.plot(t_sec, mean, color=c, lw=1.5, ls="--", alpha=0.8, label=name)

        ax.set_ylabel(f"EW{row+1}", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15)
        if row == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.suptitle(f"Free-Run Ensemble Comparison — {worm_id}\n"
                 f"(IQR ribbon + mean)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = f"comparison_ensemble_{worm_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


def plot_comparison_bars(
    all_metrics: Dict[str, dict],
    out_dir: Path,
    worm_id: str,
):
    """Side-by-side bar chart for all models."""
    model_names = list(all_metrics.keys())
    metric_keys = ["psd_log_distance", "autocorr_rmse", "wasserstein_1",
                   "ks_statistic", "variance_ratio_mean"]
    metric_labels = ["PSD ↓", "ACF ↓", "W1 ↓", "KS ↓", "VarR →1"]

    n_m = len(metric_keys)
    fig, axes = plt.subplots(1, n_m, figsize=(4 * n_m, 5))
    bar_colors = [COLORS[(i + 1) % len(COLORS)] for i in range(len(model_names))]

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[i]
        vals = [all_metrics[m].get(key, 0) for m in model_names]
        x = np.arange(len(model_names))
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=55, ha="right", fontsize=7)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if key == "variance_ratio_mean":
            ax.axhline(y=1.0, color="green", ls="--", alpha=0.5, lw=1)

    fig.suptitle(f"Architecture Comparison — {worm_id}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = f"comparison_bars_{worm_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


def plot_aggregate_summary(
    all_worm_metrics: Dict[str, Dict[str, dict]],
    out_dir: Path,
):
    """Aggregate bar chart: median metrics across worms for each model."""
    # all_worm_metrics[worm_id][model_name] = metrics_dict
    worm_ids = list(all_worm_metrics.keys())
    model_names = list(next(iter(all_worm_metrics.values())).keys())

    metric_keys = ["psd_log_distance", "autocorr_rmse", "wasserstein_1",
                   "ks_statistic", "variance_ratio_mean"]
    metric_labels = ["PSD ↓", "ACF ↓", "W1 ↓", "KS ↓", "VarR →1"]

    # Compute median ± IQR across worms for each model
    agg = {}
    for name in model_names:
        agg[name] = {}
        for key in metric_keys:
            vals = [all_worm_metrics[w][name][key] for w in worm_ids
                    if name in all_worm_metrics[w]]
            agg[name][key] = (np.median(vals), np.percentile(vals, 25),
                              np.percentile(vals, 75))

    n_m = len(metric_keys)
    fig, axes = plt.subplots(1, n_m, figsize=(4 * n_m, 5.5))
    bar_colors = [COLORS[(i + 1) % len(COLORS)] for i in range(len(model_names))]
    x = np.arange(len(model_names))

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[i]
        medians = [agg[m][key][0] for m in model_names]
        lo = [agg[m][key][0] - agg[m][key][1] for m in model_names]
        hi = [agg[m][key][2] - agg[m][key][0] for m in model_names]
        ax.bar(x, medians, color=bar_colors, edgecolor="black", linewidth=0.5,
               yerr=[lo, hi], capsize=4, error_kw=dict(lw=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=55, ha="right", fontsize=7)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if key == "variance_ratio_mean":
            ax.axhline(y=1.0, color="green", ls="--", alpha=0.5, lw=1)

    fig.suptitle(f"Aggregate Architecture Comparison — {len(worm_ids)} worms\n"
                 f"(median ± IQR)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = "aggregate_comparison.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline — process one worm
# ══════════════════════════════════════════════════════════════════════════════

def process_one_worm(
    h5_path: str,
    settrf_variants: Dict[str, SetTRFJointConfig],
    device: str,
    out_dir: Path,
    neurons: str = "motor",
    n_samples: int = 20,
    video_frames: int = 300,
    video_fps: int = 15,
) -> Dict[str, dict]:
    """Full pipeline for one worm: TRF Joint + all SetTRF variants.

    Returns: {model_name: metrics_dict}
    """
    worm_id = Path(h5_path).stem
    worm_dir = out_dir / worm_id
    worm_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"WORM: {worm_id}")
    print(f"{'='*70}")

    # ── 1. TRF Joint baseline ──
    gt_beh, trf_beh, trf_neural, trf_metrics, info = train_and_sample_trf_joint(
        h5_path, device, neurons, n_samples)

    all_metrics = {"TRF Joint": trf_metrics}
    all_samples_beh = {"TRF Joint": trf_beh}

    # ── 2. SetTRF variants ──
    for vname, vcfg in settrf_variants.items():
        try:
            sbeh, sneural, smet = train_and_sample_settrf(
                h5_path, vcfg, vname, device, neurons, n_samples, info)
            all_metrics[vname] = smet
            all_samples_beh[vname] = sbeh
        except Exception as e:
            print(f"  ✗ {vname} failed: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── 3. Summary table ──
    print(f"\n{'─'*65}")
    print(f"  {'Model':<20} {'PSD↓':>8} {'ACF↓':>8} {'W1↓':>8} "
          f"{'KS↓':>8} {'VarR→1':>8}")
    print(f"  {'─'*60}")
    for name in all_metrics:
        m = all_metrics[name]
        print(f"  {name:<20} {m['psd_log_distance']:>8.3f} "
              f"{m['autocorr_rmse']:>8.3f} {m['wasserstein_1']:>8.3f} "
              f"{m['ks_statistic']:>8.3f} {m['variance_ratio_mean']:>8.3f}")

    # ── 4. Plots ──
    print(f"\n  Generating comparison plots ...")
    dt = info.get("dt", 0.6)
    n_show = min(300, info["n_test"])

    # Ensemble traces per model
    for name, samples in all_samples_beh.items():
        plot_ensemble_traces(gt_beh, samples, worm_dir,
                             name.replace(" ", "_").replace("-", "_"),
                             worm_id, n_show=n_show, dt=dt)

    # Comparison overlay
    plot_comparison_ensemble(gt_beh, all_samples_beh, worm_dir, worm_id,
                             n_show=n_show, dt=dt)

    # PSD comparison
    fs = 1.0 / dt
    f_gt, psd_gt = compute_psd(gt_beh, fs=fs)
    model_names = list(all_samples_beh.keys())
    all_psd_gen = []
    for name in model_names:
        psds = [compute_psd(s, fs=fs)[1] for s in all_samples_beh[name]]
        all_psd_gen.append(psds)
    plot_psd_comparison(f_gt, psd_gt, all_psd_gen, model_names,
                        worm_dir, worm_id)

    # Autocorrelation
    acf_gt = compute_autocorr(gt_beh)
    all_acf_gen = []
    for name in model_names:
        acfs = [compute_autocorr(s) for s in all_samples_beh[name]]
        all_acf_gen.append(acfs)
    plot_autocorr_comparison(acf_gt, all_acf_gen, model_names,
                             worm_dir, worm_id, dt=dt)

    # Marginals
    plot_marginals(gt_beh,
                   [all_samples_beh[n] for n in model_names],
                   model_names, worm_dir, worm_id)

    # Summary bars
    plot_comparison_bars(all_metrics, worm_dir, worm_id)

    # ── 5. Multi-model posture video ──
    print(f"\n  Generating multi-model posture video ...")
    first_samples = {name: samples[0] for name, samples in all_samples_beh.items()}
    n_vid = min(video_frames, info["n_test"])
    video_path = str(worm_dir / f"{worm_id}_comparison.mp4")
    try:
        make_multi_model_video(
            h5_path, video_path, gt_beh,
            first_samples, max_frames=n_vid, fps=video_fps)
    except Exception as e:
        print(f"  ✗ Multi-model video failed: {e}")
        import traceback; traceback.print_exc()

    # ── 6. Save results JSON ──
    save_data = {
        "worm_id": worm_id,
        "n_samples": n_samples,
        "models": {},
    }
    for name in all_metrics:
        save_data["models"][name] = {
            "metrics": all_metrics[name],
        }
        # Add architecture config for SetTRF variants
        if name in settrf_variants:
            cfg = settrf_variants[name]
            save_data["models"][name]["config"] = {
                "d_model": cfg.d_model, "n_heads": cfg.n_heads,
                "n_encoder_layers": cfg.n_encoder_layers,
                "d_ff": cfg.d_ff, "neuron_embed_dim": cfg.neuron_embed_dim,
                "context_length": cfg.context_length, "dropout": cfg.dropout,
            }

    json_path = worm_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)
    print(f"  Results → {json_path}")

    return all_metrics


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Free-Run Architecture Comparison: TRF Joint vs SetTRF variants")
    ap.add_argument("--h5_dir",
                    default="data/used/behaviour+neuronal activity atanas (2023)/2",
                    help="Directory containing H5 files")
    ap.add_argument("--worms", nargs="+",
                    default=["2022-06-14-01", "2022-06-14-07", "2022-06-14-13"],
                    help="Worm IDs to process (stem of H5 filenames)")
    ap.add_argument("--out_dir",
                    default="output_plots/free_run/comparison_videos",
                    help="Output directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neurons", choices=["motor", "all"], default="motor")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--video_frames", type=int, default=300,
                    help="Frames for video (300 @ 15fps = 20s)")
    ap.add_argument("--video_fps", type=int, default=15)
    args = ap.parse_args()

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve H5 paths
    h5_files = []
    for worm_id in args.worms:
        h5_path = h5_dir / f"{worm_id}.h5"
        if h5_path.exists():
            h5_files.append(str(h5_path))
        else:
            print(f"WARNING: {h5_path} not found, skipping")

    if not h5_files:
        print("ERROR: No valid H5 files found")
        sys.exit(1)

    settrf_variants = get_settrf_variants()

    print("=" * 70)
    print("FREE-RUN ARCHITECTURE COMPARISON")
    print("=" * 70)
    print(f"  Worms: {len(h5_files)}")
    print(f"  Models: TRF Joint + {len(settrf_variants)} SetTRF variants")
    for name, cfg in settrf_variants.items():
        n_approx = cfg.d_model * cfg.d_model * cfg.n_encoder_layers * 4  # rough estimate
        print(f"    {name:20s}: d={cfg.d_model} h={cfg.n_heads} "
              f"L={cfg.n_encoder_layers} d_ff={cfg.d_ff} "
              f"embed={cfg.neuron_embed_dim} K={cfg.context_length} "
              f"drop={cfg.dropout}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Device: {args.device}")
    print(f"  Output: {out_dir}")

    all_worm_metrics = {}  # worm_id → {model_name: metrics}

    for i, h5_path in enumerate(h5_files):
        worm_id = Path(h5_path).stem
        print(f"\n[{i+1}/{len(h5_files)}] {worm_id}")
        try:
            metrics = process_one_worm(
                h5_path, settrf_variants, args.device, out_dir,
                neurons=args.neurons, n_samples=args.n_samples,
                video_frames=args.video_frames, video_fps=args.video_fps,
            )
            all_worm_metrics[worm_id] = metrics
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── Aggregate summary ──
    if len(all_worm_metrics) > 1:
        plot_aggregate_summary(all_worm_metrics, out_dir)

    # ── Print final summary table ──
    if all_worm_metrics:
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY — {len(all_worm_metrics)} worms")
        print(f"{'='*80}")

        # Collect all model names
        all_model_names = []
        for wm in all_worm_metrics.values():
            for name in wm:
                if name not in all_model_names:
                    all_model_names.append(name)

        print(f"\n  {'Model':<20} {'PSD↓':>8} {'ACF↓':>8} {'W1↓':>8} "
              f"{'KS↓':>8} {'VarR→1':>8}")
        print(f"  {'─'*60}")

        for name in all_model_names:
            vals = {k: [] for k in ["psd_log_distance", "autocorr_rmse",
                                     "wasserstein_1", "ks_statistic",
                                     "variance_ratio_mean"]}
            for worm_id, wm in all_worm_metrics.items():
                if name in wm:
                    for k in vals:
                        vals[k].append(wm[name][k])

            print(f"  {name:<20} "
                  f"{np.median(vals['psd_log_distance']):>8.3f} "
                  f"{np.median(vals['autocorr_rmse']):>8.3f} "
                  f"{np.median(vals['wasserstein_1']):>8.3f} "
                  f"{np.median(vals['ks_statistic']):>8.3f} "
                  f"{np.median(vals['variance_ratio_mean']):>8.3f}")

        # Save aggregate results
        agg_data = {
            "worms": list(all_worm_metrics.keys()),
            "per_worm": {},
            "aggregate": {},
        }
        for worm_id, wm in all_worm_metrics.items():
            agg_data["per_worm"][worm_id] = {
                name: m for name, m in wm.items()
            }
        for name in all_model_names:
            vals = {}
            for k in ["psd_log_distance", "autocorr_rmse", "wasserstein_1",
                       "ks_statistic", "variance_ratio_mean"]:
                v = [all_worm_metrics[w][name][k] for w in all_worm_metrics
                     if name in all_worm_metrics[w]]
                vals[k] = float(np.median(v))
            agg_data["aggregate"][name] = vals

        agg_path = out_dir / "aggregate_results.json"
        with open(agg_path, "w") as f:
            json.dump(agg_data, f, indent=2, cls=NumpyEncoder)
        print(f"\n  Aggregate results → {agg_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
