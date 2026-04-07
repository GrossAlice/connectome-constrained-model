#!/usr/bin/env python3
"""Fair retrain-per-condition sweep: Transformer (TRF) architectures.

Adds TRF architecture sweep to the existing Ridge/MLP retrain-sweep results.
Each TRF is retrained per condition using **condaware masking** — the fairest
approach matching what was done for Ridge and MLP.

Architecture sweep
------------------
    (d_model, n_heads, n_layers, d_ff)  ∈  {
        ( 64,  2, 1,  128),   # tiny
        (128,  4, 2,  256),   # small
        (256,  8, 2,  512),   # medium  ← v4 default
        (256,  8, 4,  512),   # deep
        (512,  8, 2, 1024),   # wide
        (512,  8, 4, 1024),   # big
    }

Conditions
----------
    causal_self :  train TRF on full (B, K, N) windows  → predict all N
    self        :  condaware TRF — random neuron column kept per sample
    causal      :  condaware TRF — random neuron column zeroed per sample

Why no concurrent ("conc") conditions?
    TRF operates on lagged windows [t-K … t-1] with NO access to features
    at time t (concurrent).  Unlike Ridge/MLP which use (K+1)·N features
    including the concurrent block, TRF sees only K·N lagged features.
    So only the 3 lag-based conditions {causal_self, self, causal} apply.

Loads existing Ridge + MLP results from retrain_sweep/results.json for
combined comparison plots.

Usage
-----
    python -m scripts.diag_retrain_sweep_trf --device cuda --n_worms 3
"""
from __future__ import annotations

import argparse, json, sys, time, glob, warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.dataset import load_worm_data
from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.model import TemporalTransformerGaussian

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

K = 5
N_FOLDS = 5

CONDS = ["causal_self", "self", "causal"]
COND_LABELS = {
    "causal_self": "Causal + Self",
    "self":        "Self (AR)",
    "causal":      "Causal (Granger)",
}

TRF_ARCHS = [
    {"tag": "trf_64d1L",  "d_model": 64,  "n_heads": 2,
     "n_layers": 1, "d_ff": 128},
    {"tag": "trf_128d2L", "d_model": 128, "n_heads": 4,
     "n_layers": 2, "d_ff": 256},
    {"tag": "trf_256d2L", "d_model": 256, "n_heads": 8,
     "n_layers": 2, "d_ff": 512},          # v4 default
    {"tag": "trf_256d4L", "d_model": 256, "n_heads": 8,
     "n_layers": 4, "d_ff": 512},
    {"tag": "trf_512d2L", "d_model": 512, "n_heads": 8,
     "n_layers": 2, "d_ff": 1024},
    {"tag": "trf_512d4L", "d_model": 512, "n_heads": 8,
     "n_layers": 4, "d_ff": 1024},
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _r2(gt, pred):
    ss_res = np.nansum((gt - pred) ** 2)
    ss_tot = np.nansum((gt - np.nanmean(gt)) ** 2) + 1e-12
    return 1 - ss_res / ss_tot


def _per_neuron_r2(ho, gt):
    return np.array([_r2(gt[:, i], ho[:, i]) for i in range(ho.shape[1])])


def _make_folds(T, warmup, n_folds=N_FOLDS):
    fold_size = (T - warmup) // n_folds
    folds = []
    for i in range(n_folds):
        ts = warmup + i * fold_size
        te = warmup + (i + 1) * fold_size if i < n_folds - 1 else T
        folds.append((ts, te))
    return folds


def _get_train_idx(T, warmup, ts, te, buffer):
    before = np.arange(warmup, ts)
    after  = np.arange(min(te + buffer, T), T)
    return np.concatenate([before, after])


def _build_windows(u, indices, K):
    """Build (len(indices), K, N) lagged windows.

    Window[i] = [u[t-K], u[t-K+1], …, u[t-1]]  for t = indices[i].
    """
    T, N = u.shape
    out = np.zeros((len(indices), K, N), dtype=u.dtype)
    for wi, t in enumerate(indices):
        for k in range(K):
            src = t - K + k
            if 0 <= src < T:
                out[wi, k] = u[src]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# TRF CONFIG / TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _make_trf_config(arch, context_length):
    """Create TransformerBaselineConfig from an architecture dict."""
    return TransformerBaselineConfig(
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=0.1,
        context_length=context_length,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=0,
        max_epochs=200,
        patience=25,
    )


def _build_model(N, K, arch, device):
    """Instantiate a fresh TemporalTransformerGaussian."""
    cfg = _make_trf_config(arch, context_length=K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TemporalTransformerGaussian(
            n_neural=N, n_beh=0, cfg=cfg
        ).to(device)
    return model, cfg


def _train_trf_simple(X_tr, y_tr, X_val, y_val, N, K, arch, device):
    """Train standard multi-output TRF on full windows (causal_self)."""
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    model, cfg = _build_model(N, K, arch, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    bvl, bs, pat = float("inf"), None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        pred_mu, _, _, _ = model(Xt)
        loss = F.mse_loss(pred_mu, yt)
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pv, _, _, _ = model(Xv)
            vl = F.mse_loss(pv, yv).item()
        if vl < bvl - 1e-6:
            bvl = vl
            bs = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat > cfg.patience:
            break

    if bs:
        model.load_state_dict(bs)
    model.eval().cpu()
    return model


def _train_trf_condaware(X_tr, y_tr, X_val, y_val, N, K, arch, device,
                         cond):
    """Train TRF with random per-sample neuron masks (self / causal).

    Each training step:
        1. Pick a random neuron nᵢ per sample.
        2. 'self'   → keep only column nᵢ in the window (AR).
           'causal' → zero column nᵢ in the window   (Granger).
        3. MSE loss only on neuron nᵢ's output.

    This teaches the model to give correct predictions under any
    neuron's condition mask, analogous to the MLP condaware approach.
    """
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    model, cfg = _build_model(N, K, arch, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    bvl, bs, pat = float("inf"), None, 0
    for ep in range(cfg.max_epochs):
        model.train()

        # --- random neuron masking per sample ---
        ni_rand = torch.randint(0, N, (Xt.shape[0],), device=device)

        if cond == "self":
            # Keep only neuron nᵢ; zero everything else
            mask = torch.zeros(Xt.shape[0], 1, N, device=device)
            mask.scatter_(2, ni_rand.view(-1, 1, 1), 1.0)
        else:  # "causal"
            # Zero neuron nᵢ; keep everything else
            mask = torch.ones(Xt.shape[0], 1, N, device=device)
            mask.scatter_(2, ni_rand.view(-1, 1, 1), 0.0)

        pred_mu, _, _, _ = model(Xt * mask)          # (B, N)
        idx = torch.arange(Xt.shape[0], device=device)
        loss = F.mse_loss(pred_mu[idx, ni_rand], yt[idx, ni_rand])
        opt.zero_grad(); loss.backward(); opt.step()

        # --- validation on a few neurons ---
        model.eval()
        with torch.no_grad():
            vl, n_check = 0.0, min(5, N)
            for ni in range(n_check):
                if cond == "self":
                    m = torch.zeros(1, 1, N, device=device)
                    m[0, 0, ni] = 1.0
                else:
                    m = torch.ones(1, 1, N, device=device)
                    m[0, 0, ni] = 0.0
                pv, _, _, _ = model(Xv * m)
                vl += F.mse_loss(pv[:, ni], yv[:, ni]).item()
            vl /= n_check

        if vl < bvl - 1e-6:
            bvl = vl
            bs = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat > cfg.patience:
            break

    if bs:
        model.load_state_dict(bs)
    model.eval().cpu()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# TRF — RETRAIN PER CONDITION
# ══════════════════════════════════════════════════════════════════════════════

def _run_trf_retrain(u, N, arch, device):
    """Fair TRF: retrain per condition using condaware masking.

    causal_self : one TRF trained on full windows (B, K, N)
    self        : condaware TRF (random neuron column kept)
    causal      : condaware TRF (random neuron column zeroed)
    """
    T = u.shape[0]
    warmup = K
    ho = {c: np.full((T, N), np.nan) for c in CONDS}

    for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]

        W_tr  = _build_windows(u_n, tr_i,  K)
        W_val = _build_windows(u_n, val_i, K)
        test_idx = np.arange(ts, te)
        W_te = _build_windows(u_n, test_idx, K)
        W_te_t = torch.tensor(W_te, dtype=torch.float32)

        # ── causal_self: full windows ──
        trf = _train_trf_simple(
            W_tr, u_n[tr_i], W_val, u_n[val_i], N, K, arch, device)
        with torch.no_grad():
            pred, _, _, _ = trf(W_te_t)
            ho["causal_self"][ts:te] = pred.numpy() * sig + mu
        del trf

        # ── self & causal: condaware TRF ──
        for cond in ["self", "causal"]:
            trf = _train_trf_condaware(
                W_tr, u_n[tr_i], W_val, u_n[val_i],
                N, K, arch, device, cond)
            with torch.no_grad():
                for ni in range(N):
                    if cond == "self":
                        m = torch.zeros(1, 1, N)
                        m[0, 0, ni] = 1.0
                    else:
                        m = torch.ones(1, 1, N)
                        m[0, 0, ni] = 0.0
                    pred, _, _, _ = trf(W_te_t * m)
                    ho[cond][ts:te, ni] = \
                        pred[:, ni].numpy() * sig[ni] + mu[ni]
            del trf

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"      Fold {fi + 1}/{N_FOLDS}")

    return {c: _per_neuron_r2(ho[c], u) for c in CONDS}


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _agg(all_results, exp_name, cond):
    """Return (mean, sem) of R² across worms for one experiment/condition."""
    vals = [all_results[w][exp_name]["r2"][cond]["mean"]
            for w in all_results
            if exp_name in all_results[w]]
    if not vals:
        return np.nan, np.nan
    return np.mean(vals), np.std(vals) / max(1, np.sqrt(len(vals)))


def _plot_trf_arch_sweep(all_results, trf_exp_names, out):
    """Grouped bar chart: TRF architectures per condition, with best
    Ridge & MLP reference lines."""
    n_worms = len(all_results)
    arch_labels = [n.replace("_retrain", "") for n in trf_exp_names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ci, cond in enumerate(CONDS):
        ax = axes[ci]

        # --- Best Ridge retrain reference ---
        ridge_names = [
            k for k in next(iter(all_results.values()))
            if "ridge" in k and "analytical" not in k
        ]
        if ridge_names:
            ridge_means = []
            for rn in ridge_names:
                m, _ = _agg(all_results, rn, cond)
                if not np.isnan(m):
                    ridge_means.append(m)
            if ridge_means:
                best_ridge = max(ridge_means)
                ax.axhline(best_ridge, color="#2ca02c", ls="--", lw=2,
                           alpha=0.8,
                           label=f"Best Ridge = {best_ridge:.3f}")

        # --- Best MLP retrain reference ---
        mlp_names = [
            k for k in next(iter(all_results.values()))
            if "mlp" in k
        ]
        if mlp_names:
            mlp_means = []
            for mn in mlp_names:
                m, _ = _agg(all_results, mn, cond)
                if not np.isnan(m):
                    mlp_means.append(m)
            if mlp_means:
                best_mlp = max(mlp_means)
                ax.axhline(best_mlp, color="#ff7f0e", ls="--", lw=2,
                           alpha=0.8,
                           label=f"Best MLP = {best_mlp:.3f}")

        # --- TRF bars ---
        means, sems = [], []
        for name in trf_exp_names:
            m, s = _agg(all_results, name, cond)
            means.append(m)
            sems.append(s)

        x = np.arange(len(trf_exp_names))
        bars = ax.bar(x, means, 0.6, yerr=sems, capsize=3,
                      color="#9467bd", edgecolor="white", linewidth=0.5)
        for bar, y_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(y_val, 0) + 0.015, f"{y_val:.3f}",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(arch_labels, fontsize=7.5, rotation=30,
                           ha="right")
        if ci == 0:
            ax.set_ylabel("Mean R²  (± SEM)", fontsize=11)
        ax.set_title(COND_LABELS[cond], fontsize=12, fontweight="bold")
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(axis="y", alpha=0.2)
        ax.axhline(0, color="k", lw=0.4, alpha=0.3)

    fig.suptitle(
        f"TRF Architecture Sweep — Retrain per Condition  "
        f"(K={K}, {n_worms} worms)",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sweep_trf_arch.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ sweep_trf_arch.png")


def _plot_combined_all(all_results, out):
    """Per-condition horizontal bar charts ranking ALL experiments."""
    n_worms = len(all_results)
    # Collect all experiment names that appear in at least one worm
    all_exp_names = set()
    for w in all_results.values():
        all_exp_names.update(w.keys())
    all_exp_names = sorted(all_exp_names)

    fig, axes = plt.subplots(
        1, 3, figsize=(20, max(10, len(all_exp_names) * 0.5)))

    for ci, cond in enumerate(CONDS):
        ax = axes[ci]
        data = []
        for ename in all_exp_names:
            m, s = _agg(all_results, ename, cond)
            if not np.isnan(m):
                data.append({"name": ename, "mean": m, "sem": s})
        data.sort(key=lambda d: d["mean"], reverse=True)

        y = np.arange(len(data))
        colors = []
        for d in data:
            if d["name"] == "ridge_analytical":
                colors.append("#d62728")
            elif "ridge" in d["name"]:
                colors.append("#2ca02c")
            elif "mlp" in d["name"]:
                colors.append("#1f77b4")
            elif "trf" in d["name"]:
                colors.append("#9467bd")
            else:
                colors.append("gray")

        bars = ax.barh(
            y, [d["mean"] for d in data],
            xerr=[d["sem"] for d in data],
            color=colors, capsize=2,
            edgecolor="white", linewidth=0.5)
        for bar, d in zip(bars, data):
            xpos = max(d["mean"], 0) + 0.008
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{d['mean']:.3f}", va="center", fontsize=6.5,
                    fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels([d["name"] for d in data], fontsize=6.5)
        ax.set_xlabel("Mean R²", fontsize=10)
        ax.set_title(COND_LABELS[cond], fontsize=12, fontweight="bold")
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

    fig.suptitle(
        f"All Experiments Ranked  (K={K}, {n_worms} worms)\n"
        "🟥 Ridge analytical   🟩 Ridge retrain   "
        "🟦 MLP retrain   🟪 TRF retrain",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sweep_combined_all.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ sweep_combined_all.png")


def _plot_heatmap_all(all_results, out):
    """Heatmap: all experiments × conditions."""
    n_worms = len(all_results)
    all_exp_names = set()
    for w in all_results.values():
        all_exp_names.update(w.keys())
    all_exp_names = sorted(all_exp_names)

    mat = np.full((len(all_exp_names), len(CONDS)), np.nan)
    for ei, ename in enumerate(all_exp_names):
        for ci, cond in enumerate(CONDS):
            m, _ = _agg(all_results, ename, cond)
            mat[ei, ci] = m

    fig, ax = plt.subplots(
        figsize=(8, max(10, len(all_exp_names) * 0.45)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                   vmin=-0.1, vmax=0.8)

    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([COND_LABELS[c] for c in CONDS], fontsize=10)
    ax.set_yticks(range(len(all_exp_names)))
    ax.set_yticklabels(all_exp_names, fontsize=7)

    for ei in range(len(all_exp_names)):
        for ci in range(len(CONDS)):
            v = mat[ei, ci]
            if not np.isnan(v):
                ax.text(ci, ei, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if v < 0.15 or v > 0.6 else "black")

    fig.colorbar(im, ax=ax, shrink=0.6, label="Mean R²")
    ax.set_title(
        f"R² Heatmap — All Models  (K={K}, {n_worms} worms)",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sweep_heatmap_all.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ sweep_heatmap_all.png")


def _plot_model_comparison(all_results, out):
    """Best model per family: Ridge vs MLP vs TRF."""
    n_worms = len(all_results)
    all_exp_names = set()
    for w in all_results.values():
        all_exp_names.update(w.keys())

    families = OrderedDict([
        ("Ridge\n(analytical)", [n for n in all_exp_names
                                 if n == "ridge_analytical"]),
        ("Ridge\n(best retrain)", [n for n in all_exp_names
                                   if "ridge" in n
                                   and "analytical" not in n]),
        ("MLP\n(best retrain)", [n for n in all_exp_names
                                 if "mlp" in n]),
        ("TRF\n(best retrain)", [n for n in all_exp_names
                                 if "trf" in n]),
    ])

    fam_colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ci, cond in enumerate(CONDS):
        ax = axes[ci]
        labels, means, sems, cols = [], [], [], []

        for fi, (fname, members) in enumerate(families.items()):
            if not members:
                continue
            best_m, best_s, best_n = -999, 0, ""
            for m_name in members:
                m, s = _agg(all_results, m_name, cond)
                if not np.isnan(m) and m > best_m:
                    best_m, best_s, best_n = m, s, m_name
            if best_m > -999:
                labels.append(fname)
                means.append(best_m)
                sems.append(best_s)
                cols.append(fam_colors[fi])

        x = np.arange(len(labels))
        bars = ax.bar(x, means, 0.6, yerr=sems, capsize=4,
                      color=cols, edgecolor="white", linewidth=0.5)
        for bar, y_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(y_val, 0) + 0.02, f"{y_val:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        if ci == 0:
            ax.set_ylabel("Mean R²  (± SEM)", fontsize=11)
        ax.set_title(COND_LABELS[cond], fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.axhline(0, color="k", lw=0.4, alpha=0.3)

    fig.suptitle(
        f"Best Model per Family — Ridge vs MLP vs TRF  "
        f"(K={K}, {n_worms} worms)",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sweep_model_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ sweep_model_comparison.png")


def _plot_all(all_results, trf_exp_names, out):
    _plot_trf_arch_sweep(all_results, trf_exp_names, out)
    _plot_combined_all(all_results, out)
    _plot_heatmap_all(all_results, out)
    _plot_model_comparison(all_results, out)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(all_results):
    all_exp_names = set()
    for w in all_results.values():
        all_exp_names.update(w.keys())
    all_exp_names = sorted(all_exp_names)

    print("\n" + "=" * 90)
    print(f"{'Experiment':<28s}  {'causal_self':>12s}"
          f"  {'self':>12s}  {'causal':>12s}")
    print("=" * 90)

    for ename in all_exp_names:
        row = f"{ename:<28s}"
        for cond in CONDS:
            m, s = _agg(all_results, ename, cond)
            if np.isnan(m):
                row += f"  {'N/A':>12s}"
            else:
                row += f"  {m:>6.3f}±{s:.3f}"
        print(row)

    # Best per family
    print("-" * 90)
    for cond in CONDS:
        best_per_fam = {}
        for ename in all_exp_names:
            m, _ = _agg(all_results, ename, cond)
            if np.isnan(m):
                continue
            if "trf" in ename:
                fam = "TRF"
            elif "mlp" in ename:
                fam = "MLP"
            elif "ridge" in ename:
                fam = "Ridge"
            else:
                fam = "Other"
            if fam not in best_per_fam or m > best_per_fam[fam][1]:
                best_per_fam[fam] = (ename, m)
        for fam in ["Ridge", "MLP", "TRF"]:
            if fam in best_per_fam:
                name, val = best_per_fam[fam]
                print(f"  Best {fam:5s} for {cond:14s}: "
                      f"{name} = {val:.3f}")
    print("=" * 90)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="data/used/behaviour+neuronal activity "
                            "atanas (2023)/2")
    ap.add_argument("--n_worms", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out",
                    default="output_plots/neural_activity_decoder_v4/"
                            "retrain_sweep")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load existing Ridge + MLP results ──
    existing_json = out / "results.json"
    if existing_json.exists():
        with open(existing_json) as f:
            all_results = json.load(f)
        n_existing = len(next(iter(all_results.values())))
        print(f"Loaded existing results: {len(all_results)} worms, "
              f"{n_existing} experiments each")
    else:
        all_results = {}
        print("No existing results found; running TRF only.")

    h5_files = sorted(glob.glob(str(Path(args.data_dir) / "*.h5")))
    if len(h5_files) > args.n_worms:
        indices = np.linspace(0, len(h5_files) - 1,
                              args.n_worms, dtype=int)
        h5_files = [h5_files[i] for i in indices]

    # ── TRF experiments ──
    trf_experiments = OrderedDict()
    for arch in TRF_ARCHS:
        trf_experiments[arch["tag"] + "_retrain"] = arch
    trf_exp_names = list(trf_experiments.keys())

    print(f"\nTRF Retrain Sweep — K={K}, {len(h5_files)} worms, "
          f"{len(trf_experiments)} architectures")
    print(f"Device: {args.device}")
    for arch in TRF_ARCHS:
        print(f"  {arch['tag']:14s}  d={arch['d_model']:4d}  "
              f"h={arch['n_heads']}  L={arch['n_layers']}  "
              f"ff={arch['d_ff']}")
    print()

    for h5_path in h5_files:
        worm_data = load_worm_data(h5_path, n_beh_modes=6)
        u = worm_data["u"].astype(np.float32)
        worm_id = worm_data["worm_id"]
        T, N = u.shape

        print(f"  ═══ {worm_id}  T={T}  N={N} ═══")
        if worm_id not in all_results:
            all_results[worm_id] = {}

        for exp_name, arch in trf_experiments.items():
            print(f"    {exp_name} ...", end=" ", flush=True)
            t0 = time.time()

            r2s = _run_trf_retrain(u, N, arch, args.device)

            elapsed = time.time() - t0
            all_results[worm_id][exp_name] = {
                "r2": {c: {"mean": float(np.nanmean(v)),
                           "per_neuron": [float(x) for x in v]}
                       for c, v in r2s.items()},
                "time": elapsed,
            }
            summary = "  ".join(f"{c}={np.nanmean(v):.3f}"
                                for c, v in r2s.items())
            print(f"{summary}  [{elapsed:.1f}s]")

        # Save incrementally
        with open(out / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Summary & Plots ──
    _print_summary(all_results)
    print("\nGenerating plots...")
    _plot_all(all_results, trf_exp_names, out)
    print("\nDone!")


if __name__ == "__main__":
    main()
