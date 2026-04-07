#!/usr/bin/env python
"""
Transformer architecture sweep for LOO evaluation.

Sweeps over model sizes, context lengths, loss functions, etc. to find the
best transformer configuration for one-step and LOO R².

Usage
─────
    TORCHDYNAMO_DISABLE=1 python -u -m scripts.sweep_transformer_arch \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
        --save_dir output_plots/stage2/transformer_arch_sweep \
        --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import itertools
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.model import build_model
from baseline_transformer.dataset import SlidingWindowDataset
from torch.utils.data import DataLoader


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


# ═══════════════════════════════════════════════════════════════════════
#  Config for one sweep trial
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrialConfig:
    name: str
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    K: int = 10
    lr: float = 1e-3
    dropout: float = 0.1
    loss: str = "nll"       # "nll" or "mse"
    max_epochs: int = 200
    patience: int = 20
    batch_size: int = 64
    weight_decay: float = 1e-4


# ═══════════════════════════════════════════════════════════════════════
#  MSE-aware training/eval epochs
# ═══════════════════════════════════════════════════════════════════════

def _train_epoch_flex(model, loader, optimizer, device, grad_clip=1.0,
                      loss_type="nll"):
    """Training epoch supporting both NLL and MSE losses."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_neural = model.n_neural

    for context, target, target_mask in loader:
        context = context.to(device)
        target = target.to(device)

        mu_u, sigma_u, mu_b, sigma_b = model(context, return_all_steps=False)
        target_u = target[:, :n_neural]

        if loss_type == "mse":
            loss = F.mse_loss(mu_u, target_u)
        else:  # nll
            from baseline_transformer.model import gaussian_nll_loss
            loss = gaussian_nll_loss(mu_u, sigma_u, target_u)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _eval_epoch_flex(model, loader, device, loss_type="nll"):
    """Validation epoch supporting both NLL and MSE losses."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    n_neural = model.n_neural

    for context, target, target_mask in loader:
        context = context.to(device)
        target = target.to(device)

        mu_u, sigma_u, mu_b, sigma_b = model(context, return_all_steps=False)
        target_u = target[:, :n_neural]

        if loss_type == "mse":
            loss = F.mse_loss(mu_u, target_u)
        else:
            from baseline_transformer.model import gaussian_nll_loss
            loss = gaussian_nll_loss(mu_u, sigma_u, target_u)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ═══════════════════════════════════════════════════════════════════════
#  Run one trial: train + evaluate (2-fold temporal CV)
# ═══════════════════════════════════════════════════════════════════════

def run_trial(
    tcfg: TrialConfig,
    u: np.ndarray,
    loo_neurons: List[int],
    device: str = "cuda",
    window_size: int = 50,
    verbose: bool = True,
) -> Dict:
    """Train transformer with given config, return one-step + LOO R²."""
    T, N = u.shape
    K = tcfg.K
    dev = torch.device(device)

    cfg = TransformerBaselineConfig(
        d_model=tcfg.d_model,
        n_heads=tcfg.n_heads,
        n_layers=tcfg.n_layers,
        d_ff=tcfg.d_ff,
        dropout=tcfg.dropout,
        context_length=K,
        lr=tcfg.lr,
        weight_decay=tcfg.weight_decay,
        batch_size=tcfg.batch_size,
        max_epochs=tcfg.max_epochs,
        patience=tcfg.patience,
        grad_clip=1.0,
        device=device,
        predict_beh=False,
        include_beh_input=False,
        w_beh=0.0,
    )

    mid = T // 2 + 1
    folds = [(mid, T, 0, mid), (0, mid, mid, T)]

    x = u.astype(np.float32)
    os_pred = np.full((T, N), np.nan, dtype=np.float32)
    loo_pred = np.full((T, N), np.nan, dtype=np.float32)

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        # ── Datasets ──
        tr_len = tr_e - tr_s
        val_size = max(1, int(tr_len * 0.15))
        val_s = tr_e - val_size
        train_ds = SlidingWindowDataset(x, n_neural=N, context_length=K,
                                        start=tr_s, end=val_s)
        val_ds = SlidingWindowDataset(x, n_neural=N, context_length=K,
                                      start=val_s, end=tr_e)
        if len(train_ds) == 0 or len(val_ds) == 0:
            continue

        train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size,
                                  shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size,
                                shuffle=False, drop_last=False)

        # ── Model ──
        model = build_model(N, cfg, device=device, n_beh=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                                      weight_decay=tcfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg.max_epochs, eta_min=tcfg.lr * 0.01)

        best_val, best_state, wait = float("inf"), None, 0
        for epoch in range(1, tcfg.max_epochs + 1):
            _train_epoch_flex(model, train_loader, optimizer, dev,
                              loss_type=tcfg.loss)
            vl = _eval_epoch_flex(model, val_loader, dev, loss_type=tcfg.loss)
            scheduler.step()
            if vl < best_val - 1e-6:
                best_val = vl
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= tcfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        if verbose:
            print(f"    fold {fi+1}/2  {epoch} ep  best_val={best_val:.4f}")

        # ── One-step predictions ──
        x_t = torch.tensor(x, dtype=torch.float32, device=dev)
        with torch.no_grad():
            first = max(K, te_s)
            for t in range(first, te_e):
                ctx = x_t[t - K : t].unsqueeze(0)
                mu_u, _ = model.predict_mean_split(ctx)
                os_pred[t] = mu_u.squeeze(0).cpu().numpy()

        # ── Windowed LOO (replace ALL K context frames for held-out) ──
        with torch.no_grad():
            for ni in loo_neurons:
                T_te = te_e - te_s
                pred = np.full(T_te, np.nan, dtype=np.float32)
                for s in range(0, T_te, window_size):
                    e = min(s + window_size, T_te)
                    pred[s] = x[te_s + s, ni]  # re-seed
                    for t_local in range(s, e - 1):
                        t_abs = te_s + t_local
                        if t_abs < K:
                            continue
                        ctx_np = x[t_abs - K : t_abs].copy()
                        # Replace ALL K frames of neuron ni with predictions
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

    return {
        "name": tcfg.name,
        "onestep_r2_all": float(np.nanmean(onestep_r2)),
        "onestep_r2_loo30": float(np.nanmean(onestep_r2[loo_neurons])),
        "onestep_r2_median": float(np.nanmedian(onestep_r2)),
        "loo_r2_mean": float(np.nanmean(loo_r2_w)),
        "loo_r2_median": float(np.nanmedian(loo_r2_w)),
        "onestep_r2_per_neuron": onestep_r2.tolist(),
        "loo_r2_per_neuron": loo_r2_w.tolist(),
        "config": asdict(tcfg),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Default sweep grid
# ═══════════════════════════════════════════════════════════════════════

def build_sweep_configs() -> List[TrialConfig]:
    """Build a selective sweep grid (not full Cartesian)."""
    trials = []

    # ── Baseline (current) ──
    trials.append(TrialConfig(
        name="d64_K5_L2_NLL",
        d_model=64, n_heads=4, n_layers=2, d_ff=128, K=5, loss="nll"))

    # ── Scale up model width ──
    trials.append(TrialConfig(
        name="d128_K5_L2_NLL",
        d_model=128, n_heads=4, n_layers=2, d_ff=256, K=5, loss="nll"))
    trials.append(TrialConfig(
        name="d256_K5_L2_NLL",
        d_model=256, n_heads=8, n_layers=2, d_ff=512, K=5, loss="nll"))

    # ── Scale up context length ──
    trials.append(TrialConfig(
        name="d128_K10_L2_NLL",
        d_model=128, n_heads=4, n_layers=2, d_ff=256, K=10, loss="nll"))
    trials.append(TrialConfig(
        name="d128_K20_L2_NLL",
        d_model=128, n_heads=4, n_layers=2, d_ff=256, K=20, loss="nll"))

    # ── Scale up depth ──
    trials.append(TrialConfig(
        name="d128_K10_L4_NLL",
        d_model=128, n_heads=4, n_layers=4, d_ff=256, K=10, loss="nll"))
    trials.append(TrialConfig(
        name="d128_K10_L6_NLL",
        d_model=128, n_heads=4, n_layers=6, d_ff=256, K=10, loss="nll"))

    # ── MSE loss variants (key comparison) ──
    trials.append(TrialConfig(
        name="d64_K5_L2_MSE",
        d_model=64, n_heads=4, n_layers=2, d_ff=128, K=5, loss="mse"))
    trials.append(TrialConfig(
        name="d128_K10_L2_MSE",
        d_model=128, n_heads=4, n_layers=2, d_ff=256, K=10, loss="mse"))
    trials.append(TrialConfig(
        name="d256_K10_L4_MSE",
        d_model=256, n_heads=8, n_layers=4, d_ff=512, K=10, loss="mse"))
    trials.append(TrialConfig(
        name="d128_K20_L4_MSE",
        d_model=128, n_heads=4, n_layers=4, d_ff=256, K=20, loss="mse"))

    # ── Large + long context ──
    trials.append(TrialConfig(
        name="d256_K20_L4_NLL",
        d_model=256, n_heads=8, n_layers=4, d_ff=512, K=20, loss="nll"))
    trials.append(TrialConfig(
        name="d256_K20_L4_MSE",
        d_model=256, n_heads=8, n_layers=4, d_ff=512, K=20, loss="mse"))

    # ── Lower LR for larger models ──
    trials.append(TrialConfig(
        name="d256_K10_L4_MSE_lr5e4",
        d_model=256, n_heads=8, n_layers=4, d_ff=512, K=10,
        loss="mse", lr=5e-4))
    trials.append(TrialConfig(
        name="d256_K20_L4_MSE_lr5e4",
        d_model=256, n_heads=8, n_layers=4, d_ff=512, K=20,
        loss="mse", lr=5e-4))

    # ── Higher dropout for larger models ──
    trials.append(TrialConfig(
        name="d256_K10_L4_MSE_dp02",
        d_model=256, n_heads=8, n_layers=4, d_ff=512, K=10,
        loss="mse", dropout=0.2))

    return trials


# ═══════════════════════════════════════════════════════════════════════
#  Summary plot
# ═══════════════════════════════════════════════════════════════════════

def plot_sweep_summary(results: List[Dict], save_path: Path, worm: str):
    """Bar chart comparing all trial configs."""
    names = [r["name"] for r in results]
    os_all = [r["onestep_r2_all"] for r in results]
    os_loo = [r["onestep_r2_loo30"] for r in results]
    loo_mean = [r["loo_r2_mean"] for r in results]
    loo_median = [r["loo_r2_median"] for r in results]

    n = len(names)
    fig, axes = plt.subplots(2, 1, figsize=(max(16, n * 0.9), 12))

    # ── Panel A: One-step R² ──
    ax = axes[0]
    x_pos = np.arange(n)
    w = 0.35
    bars1 = ax.bar(x_pos - w / 2, os_all, w, label="All neurons", alpha=0.7,
                   color="#1f77b4", edgecolor="white")
    bars2 = ax.bar(x_pos + w / 2, os_loo, w, label="LOO-30 neurons", alpha=0.7,
                   color="#ff7f0e", edgecolor="white")
    for bar, val in zip(bars1, os_all):
        ax.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", rotation=45)
    for bar, val in zip(bars2, os_loo):
        ax.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", rotation=45)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("One-step R²", fontsize=12)
    ax.set_title("A.  One-step R² across architectures", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel B: LOO R² ──
    ax = axes[1]
    bars1 = ax.bar(x_pos - w / 2, loo_mean, w, label="Mean LOO R²", alpha=0.7,
                   color="#2ca02c", edgecolor="white")
    bars2 = ax.bar(x_pos + w / 2, loo_median, w, label="Median LOO R²",
                   alpha=0.7, color="#d62728", edgecolor="white")
    for bar, val in zip(bars1, loo_mean):
        ax.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", rotation=45)
    for bar, val in zip(bars2, loo_median):
        ax.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", rotation=45)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("LOO R² (windowed, w=50)", fontsize=12)
    ax.set_title("B.  LOO R² across architectures", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Transformer Architecture Sweep — {worm}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_heatmap(results: List[Dict], save_path: Path, worm: str):
    """Heat-map of LOO R² vs architecture dimensions."""
    # Extract unique values for heatmap axes
    configs = [(r["name"], r["config"], r["loo_r2_mean"], r["onestep_r2_loo30"])
               for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(configs) * 0.7), 5))

    names = [c[0] for c in configs]
    loo_vals = [c[2] for c in configs]
    os_vals = [c[3] for c in configs]

    # Sorted by LOO R² descending
    order = np.argsort(loo_vals)[::-1]
    names_s = [names[i] for i in order]
    loo_s = [loo_vals[i] for i in order]
    os_s = [os_vals[i] for i in order]

    x = np.arange(len(names_s))
    ax.barh(x, loo_s, height=0.4, align="center", label="LOO R² (mean)",
            color="#2ca02c", alpha=0.7)
    ax.barh(x + 0.4, os_s, height=0.4, align="center",
            label="One-step R² (LOO 30)", color="#1f77b4", alpha=0.7)

    for i, (l, o) in enumerate(zip(loo_s, os_s)):
        ax.text(max(l, 0) + 0.01, i, f"{l:.3f}", va="center", fontsize=8)
        ax.text(max(o, 0) + 0.01, i + 0.4, f"{o:.3f}", va="center",
                fontsize=8, color="#1f77b4")

    ax.set_yticks(x + 0.2)
    ax.set_yticklabels(names_s, fontsize=9)
    ax.set_xlabel("R²", fontsize=12)
    ax.set_title(f"Architecture Ranking (sorted by LOO R²) — {worm}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axvline(0, color="gray", lw=0.6, ls="--")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Transformer architecture sweep")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loo_subset", type=int, default=30)
    ap.add_argument("--window_size", type=int, default=50)
    ap.add_argument("--max_epochs", type=int, default=200,
                    help="Override max_epochs for all trials")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--filter", default=None, type=str,
                    help="Only run trials whose name contains this substring")
    args = ap.parse_args()

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    worm = Path(args.h5).stem

    # ── Load data ──
    from stage2.config import make_config
    from stage2.io_h5 import load_data_pt

    cfg = make_config(args.h5, num_epochs=1, device=args.device,
                      skip_final_eval=True)
    data = load_data_pt(cfg)
    u_stage1 = data["u_stage1"]
    T, N = u_stage1.shape
    u_np = u_stage1.cpu().numpy()
    print(f"  {worm}  T={T}  N={N}")

    # ── LOO neurons (top by variance) ──
    var = np.var(u_np, axis=0)
    loo_neurons = sorted(np.argsort(var)[::-1][:args.loo_subset].tolist())
    print(f"  LOO neurons ({len(loo_neurons)}): {loo_neurons}")

    # ── Build sweep grid ──
    trials = build_sweep_configs()

    # Apply overrides
    for t in trials:
        t.max_epochs = args.max_epochs
        t.patience = args.patience

    # Filter if requested
    if args.filter:
        trials = [t for t in trials if args.filter in t.name]

    print(f"\n  Running {len(trials)} trials...\n")

    # ── Run sweep ──
    results = []
    for ti, tcfg in enumerate(trials):
        print(f"{'═' * 60}")
        print(f"  [{ti + 1}/{len(trials)}]  {tcfg.name}")
        print(f"    d_model={tcfg.d_model}  n_heads={tcfg.n_heads}  "
              f"n_layers={tcfg.n_layers}  d_ff={tcfg.d_ff}")
        print(f"    K={tcfg.K}  loss={tcfg.loss}  lr={tcfg.lr}  "
              f"dropout={tcfg.dropout}")
        print(f"{'═' * 60}")

        t0 = time.time()
        try:
            result = run_trial(tcfg, u_np, loo_neurons,
                               device=args.device,
                               window_size=args.window_size)
            elapsed = time.time() - t0
            result["time_s"] = elapsed
            results.append(result)

            print(f"  ✓ one-step(all)={result['onestep_r2_all']:.4f}  "
                  f"one-step(30)={result['onestep_r2_loo30']:.4f}  "
                  f"LOO={result['loo_r2_mean']:.4f}  "
                  f"LOO-med={result['loo_r2_median']:.4f}  "
                  f"({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ✗ FAILED: {e}  ({elapsed:.0f}s)")
            results.append({
                "name": tcfg.name, "error": str(e),
                "config": asdict(tcfg), "time_s": elapsed,
            })

        # Save incremental results
        with open(save / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # ── Print summary table ──
    print(f"\n{'═' * 80}")
    print(f"  SUMMARY — {worm}")
    print(f"{'═' * 80}")
    print(f"  {'Name':<30s}  {'OS(all)':>8s}  {'OS(30)':>8s}  "
          f"{'LOO-μ':>8s}  {'LOO-med':>8s}  {'time':>6s}")
    print(f"  {'─' * 30}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 6}")

    # Sort by LOO mean R²
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r["loo_r2_mean"], reverse=True)
    for r in valid:
        print(f"  {r['name']:<30s}  {r['onestep_r2_all']:>8.4f}  "
              f"{r['onestep_r2_loo30']:>8.4f}  "
              f"{r['loo_r2_mean']:>8.4f}  {r['loo_r2_median']:>8.4f}  "
              f"{r['time_s']:>5.0f}s")
    print(f"{'═' * 80}\n")

    # ── Plot ──
    if len(valid) > 0:
        plot_sweep_summary(valid, save / "sweep_summary.png", worm)
        plot_heatmap(valid, save / "sweep_ranking.png", worm)

    print("  Done!")


if __name__ == "__main__":
    main()
