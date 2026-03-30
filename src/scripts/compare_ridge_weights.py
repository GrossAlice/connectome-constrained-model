#!/usr/bin/env python
"""Compare Ridge decoder weights across worms.

For each worm, fit Ridge regression: motor neurons → eigenworms (no time lags,
just instantaneous mapping) and with lags. Then compare the weight vectors
across the top worms to see if the same neurons carry the same signal.

Also fits a "pooled" Ridge on the shared neurons across all top worms and
evaluates its transfer to held-out worms.

Usage:
    .venv/bin/python -m scripts.compare_ridge_weights \
        --h5_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
        --out_dir output_plots/behaviour_decoder/weight_comparison
"""
from __future__ import annotations
import argparse, glob, json, sys, warnings
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats as sp_stats
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.benchmark_ar_decoder_v2 import load_data, build_ar_features, r2_score
from scripts.unified_benchmark import _ridge_fit
from stage2.behavior_decoder_eval import _log_ridge_grid, build_lagged_features_np


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def get_neuron_labels(h5_path):
    """Return list of neuron label strings."""
    with h5py.File(h5_path, "r") as f:
        return [x.decode() if isinstance(x, bytes) else str(x)
                for x in f["gcamp/neuron_labels"][:]]


def get_motor_mask(all_labels, motor_list):
    """Return boolean mask and ordered names for motor neurons present."""
    mask = np.array([l in motor_list for l in all_labels])
    names = [l for l in all_labels if l in motor_list]
    return mask, names


def fit_ridge_per_worm(h5_path, motor_list, n_lags=0, K=6, n_folds=5):
    """Fit Ridge on motor neurons → eigenworms, return weights + R².

    Returns:
        coefs: (K, n_features) Ridge coefficients
        intercepts: (K,) intercepts
        r2s: (K,) held-out R²
        neuron_names: list of motor neuron names in order
        feature_names: list of feature names (neuron × lag)
    """
    u, b_full, dt = load_data(h5_path, all_neurons=True)
    K = min(K, b_full.shape[1])
    b = b_full[:, :K]
    T = b.shape[0]

    all_labels = get_neuron_labels(h5_path)
    motor_set = set(motor_list)
    mask, neuron_names = get_motor_mask(all_labels, motor_set)
    u_motor = u[:, mask]
    M = u_motor.shape[1]

    if n_lags > 0:
        X = build_lagged_features_np(u_motor, n_lags)
        feature_names = []
        for lag in range(n_lags + 1):
            for nm in neuron_names:
                feature_names.append(f"{nm}_t-{lag}" if lag > 0 else nm)
    else:
        X = u_motor.copy()
        feature_names = list(neuron_names)

    warmup = max(2, n_lags)
    valid_len = T - warmup
    fold_size = valid_len // n_folds
    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = n_folds - 1

    folds = []
    for i in range(n_folds):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < n_folds - 1 else T
        folds.append((s, e))

    ho_preds = np.full((T, K), np.nan)

    # Fit on ALL data to get "canonical" weight vector
    mu, sigma = X[warmup:].mean(0), X[warmup:].std(0) + 1e-8
    X_s = (X - mu) / sigma

    coefs = np.zeros((K, X.shape[1]))
    intercepts = np.zeros(K)
    for j in range(K):
        coef, intc, _ = _ridge_fit(X_s[warmup:], b[warmup:, j],
                                   ridge_grid, n_inner)
        coefs[j] = coef
        intercepts[j] = intc

    # CV R²
    for fi, (ts, te) in enumerate(folds):
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[ts:te] = False
        train_idx = np.where(train_mask)[0]

        mu_tr = X[train_idx].mean(0)
        sig_tr = X[train_idx].std(0) + 1e-8
        X_tr_s = (X[train_idx] - mu_tr) / sig_tr
        X_te_s = (X[ts:te] - mu_tr) / sig_tr

        for j in range(K):
            c, ic, _ = _ridge_fit(X_tr_s, b[train_idx, j],
                                  ridge_grid, n_inner)
            ho_preds[ts:te, j] = X_te_s @ c + ic

    valid = np.arange(warmup, T)
    ok = np.isfinite(ho_preds[valid, 0])
    idx = valid[ok]
    r2s = np.array([r2_score(b[idx, j], ho_preds[idx, j]) for j in range(K)])

    # Un-scale coefficients: coefs were fit on standardised X,
    # so the "importance" of neuron i is coefs[j,i] (already on std scale)
    # For comparison across worms, keep on standardised scale.
    return coefs, intercepts, r2s, neuron_names, feature_names, dt


def extract_instantaneous_weights(coefs, neuron_names, n_lags):
    """From (K, M*(lags+1)) coef matrix, extract per-neuron importance.

    Returns dict: neuron_name → (K,) weight vector (sum of absolute
    weights across lags for that neuron, preserving sign of lag-0).
    """
    M = len(neuron_names)
    n_feat_per_lag = M
    K = coefs.shape[0]

    weights = {}
    for ni, nm in enumerate(neuron_names):
        # Collect weights for this neuron across all lags
        w_across_lags = np.zeros((K, n_lags + 1))
        for lag in range(n_lags + 1):
            idx = lag * n_feat_per_lag + ni
            if idx < coefs.shape[1]:
                w_across_lags[:, lag] = coefs[:, idx]
        # Summarise: L2 norm across lags, preserving sign of lag-0
        l2 = np.sqrt(np.sum(w_across_lags ** 2, axis=1))
        sign = np.sign(w_across_lags[:, 0])
        sign[sign == 0] = 1
        weights[nm] = sign * l2
    return weights


# ═══════════════════════════════════════════════════════════════════════
#  Main analysis
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5_dir", required=True)
    ap.add_argument("--out_dir",
                    default="output_plots/behaviour_decoder/weight_comparison")
    ap.add_argument("--n_lags", type=int, default=8)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n_top", type=int, default=10,
                    help="Number of top worms to compare")
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    K = args.K
    n_lags = args.n_lags

    motor_list_full = [l.strip() for l in
        open("data/used/masks+motor neurons/motor_neurons_with_control.txt")
        if l.strip()]
    motor_set = set(motor_list_full)

    # ── Load existing R² to rank worms ────────────────────────────────
    results_path = Path("output_plots/behaviour_decoder/comparison_3models/results.json")
    if results_path.exists():
        all_r2 = json.load(open(results_path))
    else:
        all_r2 = {}

    h5_files = sorted(glob.glob(str(Path(args.h5_dir) / "*.h5")))
    worm_ids = [Path(f).stem for f in h5_files]

    # Rank by existing Ridge R²
    scores = []
    for wid in worm_ids:
        if wid in all_r2 and "Ridge" in all_r2[wid]:
            r2 = np.nanmean(all_r2[wid]["Ridge"][:K])
            scores.append((r2, wid))
    scores.sort(reverse=True)
    top_worms = [wid for _, wid in scores[:args.n_top]]
    print(f"Top {args.n_top} worms: {top_worms}")

    # If no existing results, use all worms
    if not scores:
        top_worms = worm_ids[:args.n_top]

    # ── Find shared neurons ───────────────────────────────────────────
    worm_h5 = {Path(f).stem: f for f in h5_files}
    neuron_sets = []
    for wid in top_worms:
        labels = get_neuron_labels(worm_h5[wid])
        present = set(l for l in labels if l in motor_set)
        neuron_sets.append(present)

    shared_all = neuron_sets[0]
    for s in neuron_sets[1:]:
        shared_all = shared_all & s
    shared_all = sorted(shared_all)

    # Also find neurons in >= 70% of top worms
    cnt = Counter()
    for s in neuron_sets:
        for n in s:
            cnt[n] += 1
    threshold = max(int(0.7 * len(top_worms)), 2)
    common_neurons = sorted([n for n, c in cnt.items() if c >= threshold])

    print(f"\nShared neurons (all {len(top_worms)}): {len(shared_all)}")
    print(f"  {shared_all}")
    print(f"Common neurons (>={threshold}/{len(top_worms)}): {len(common_neurons)}")
    print(f"  {common_neurons}")

    # ── Fit Ridge on each top worm ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Fitting Ridge (n_lags={n_lags}) on top {len(top_worms)} worms...")
    print(f"{'='*70}")

    worm_data = {}  # wid → {coefs, r2s, neuron_names, weights}
    for wid in top_worms:
        print(f"\n  {wid}...")
        coefs, intc, r2s, neuron_names, feat_names, dt = fit_ridge_per_worm(
            worm_h5[wid], motor_list_full, n_lags=n_lags, K=K,
            n_folds=args.n_folds)
        weights = extract_instantaneous_weights(coefs, neuron_names, n_lags)
        worm_data[wid] = {
            "coefs": coefs, "intercepts": intc, "r2s": r2s,
            "neuron_names": neuron_names, "weights": weights, "dt": dt,
        }
        print(f"    R² = {' '.join(f'{v:.3f}' for v in r2s[:K])}"
              f"  mean={np.mean(r2s[:K]):.3f}")

    # ══════════════════════════════════════════════════════════════════
    #  ALSO fit with NO lags (instantaneous) for cleaner comparison
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Fitting Ridge (NO lags, instantaneous) for cleaner weight comparison...")
    print(f"{'='*70}")

    worm_data_inst = {}
    for wid in top_worms:
        print(f"  {wid}...")
        coefs, intc, r2s, neuron_names, feat_names, dt = fit_ridge_per_worm(
            worm_h5[wid], motor_list_full, n_lags=0, K=K,
            n_folds=args.n_folds)
        worm_data_inst[wid] = {
            "coefs": coefs, "r2s": r2s,
            "neuron_names": neuron_names, "dt": dt,
        }
        print(f"    R²(inst) = {' '.join(f'{v:.3f}' for v in r2s[:K])}"
              f"  mean={np.mean(r2s[:K]):.3f}")

    # ══════════════════════════════════════════════════════════════════
    #  Analysis 1: Weight similarity on shared neurons
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Analysis 1: Weight correlation across worms (shared neurons)")
    print(f"{'='*70}")

    # Build weight matrix on shared neurons (instantaneous fit)
    # Shape: (n_worms, K, n_shared)
    n_shared = len(shared_all)
    W = np.zeros((len(top_worms), K, n_shared))
    for wi, wid in enumerate(top_worms):
        d = worm_data_inst[wid]
        name2idx = {n: i for i, n in enumerate(d["neuron_names"])}
        for si, sn in enumerate(shared_all):
            if sn in name2idx:
                W[wi, :, si] = d["coefs"][:, name2idx[sn]]

    # Pairwise correlation of weight vectors (per EW mode)
    n_w = len(top_worms)
    corr_per_mode = np.zeros((K, n_w, n_w))
    for j in range(K):
        for a in range(n_w):
            for b2 in range(n_w):
                r, _ = sp_stats.pearsonr(W[a, j], W[b2, j])
                corr_per_mode[j, a, b2] = r

    # Print summary
    for j in range(min(K, 4)):
        triu = corr_per_mode[j][np.triu_indices(n_w, k=1)]
        print(f"  EW{j+1}: mean pairwise r={np.mean(triu):.3f}, "
              f"median={np.median(triu):.3f}, "
              f"min={np.min(triu):.3f}, max={np.max(triu):.3f}")

    # ── Also with lagged weights ──────────────────────────────────────
    print(f"\n  With lagged features (L2 summary per neuron):")
    W_lag = np.zeros((len(top_worms), K, n_shared))
    for wi, wid in enumerate(top_worms):
        d = worm_data[wid]
        for si, sn in enumerate(shared_all):
            if sn in d["weights"]:
                W_lag[wi, :, si] = d["weights"][sn]

    corr_lag = np.zeros((K, n_w, n_w))
    for j in range(K):
        for a in range(n_w):
            for b2 in range(n_w):
                r, _ = sp_stats.pearsonr(W_lag[a, j], W_lag[b2, j])
                corr_lag[j, a, b2] = r

    for j in range(min(K, 4)):
        triu = corr_lag[j][np.triu_indices(n_w, k=1)]
        print(f"  EW{j+1}: mean pairwise r={np.mean(triu):.3f}, "
              f"median={np.median(triu):.3f}, "
              f"min={np.min(triu):.3f}, max={np.max(triu):.3f}")

    # ══════════════════════════════════════════════════════════════════
    #  Analysis 2: "Universal" Ridge — pool all top worms, shared neurons
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Analysis 2: Pooled Ridge on shared neurons → transfer test")
    print(f"{'='*70}")

    # Build pooled dataset from top worms on shared neurons
    X_pool_list, b_pool_list = [], []
    for wid in top_worms:
        u, b_full, dt = load_data(worm_h5[wid], all_neurons=True)
        labels = get_neuron_labels(worm_h5[wid])
        # Extract shared neurons in consistent order
        idx_map = {l: i for i, l in enumerate(labels)}
        shared_idx = [idx_map[sn] for sn in shared_all if sn in idx_map]
        u_shared = u[:, shared_idx]
        if n_lags > 0:
            X = build_lagged_features_np(u_shared, n_lags)
        else:
            X = u_shared.copy()
        warmup = max(2, n_lags)
        X_pool_list.append(X[warmup:])
        b_pool_list.append(b_full[warmup:, :K])

    X_pool = np.concatenate(X_pool_list, axis=0)
    b_pool = np.concatenate(b_pool_list, axis=0)
    print(f"  Pooled: {X_pool.shape[0]} samples, {X_pool.shape[1]} features")

    # Standardise
    mu_pool = X_pool.mean(0)
    sig_pool = X_pool.std(0) + 1e-8
    X_pool_s = (X_pool - mu_pool) / sig_pool

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = 4

    pooled_coefs = np.zeros((K, X_pool.shape[1]))
    pooled_intc = np.zeros(K)
    for j in range(K):
        c, ic, _ = _ridge_fit(X_pool_s, b_pool[:, j], ridge_grid, n_inner)
        pooled_coefs[j] = c
        pooled_intc[j] = ic

    # Evaluate pooled model on each worm (including non-top worms)
    print(f"\n  Evaluating pooled model on ALL worms...")
    transfer_results = {}
    for h5_path in h5_files:
        wid = Path(h5_path).stem
        u, b_full, dt = load_data(h5_path, all_neurons=True)
        labels = get_neuron_labels(h5_path)
        idx_map = {l: i for i, l in enumerate(labels)}

        # Check if all shared neurons are present
        missing = [sn for sn in shared_all if sn not in idx_map]
        if missing:
            transfer_results[wid] = {"r2": [np.nan] * K, "missing": missing}
            continue

        shared_idx = [idx_map[sn] for sn in shared_all]
        u_shared = u[:, shared_idx]
        if n_lags > 0:
            X = build_lagged_features_np(u_shared, n_lags)
        else:
            X = u_shared.copy()
        warmup = max(2, n_lags)
        X_s = (X - mu_pool) / sig_pool

        preds = X_s @ pooled_coefs.T + pooled_intc
        b = b_full[:, :K]
        r2s = [float(r2_score(b[warmup:, j], preds[warmup:, j]))
               for j in range(K)]
        transfer_results[wid] = {
            "r2": r2s,
            "in_training": wid in top_worms,
        }

    # Print transfer results
    print(f"\n  {'Worm':<20s}  {'Train?':>6s}  {'R² mean':>8s}  "
          + "  ".join(f"EW{j+1:d}" for j in range(min(K,4))))
    for wid in sorted(transfer_results.keys()):
        tr = transfer_results[wid]
        if "missing" in tr:
            continue
        r2 = tr["r2"]
        flag = "YES" if tr.get("in_training") else ""
        mean_r2 = np.nanmean(r2[:K])
        mode_str = "  ".join(f"{v:5.3f}" for v in r2[:min(K,4)])
        print(f"  {wid:<20s}  {flag:>6s}  {mean_r2:8.3f}  {mode_str}")

    # ══════════════════════════════════════════════════════════════════
    #  PLOTS
    # ══════════════════════════════════════════════════════════════════

    # ── Plot 1: Weight heatmaps per EW mode across worms ──────────────
    for j in range(min(K, 4)):
        fig, ax = plt.subplots(figsize=(max(8, n_shared * 0.5),
                                        max(4, len(top_worms) * 0.6)))
        mat = W[: , j, :]  # (n_worms, n_shared)
        vmax = np.max(np.abs(mat)) * 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(n_shared))
        ax.set_xticklabels(shared_all, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(top_worms)))
        short_labels = [f"{w} (R²={np.mean(worm_data_inst[w]['r2s'][:K]):.2f})"
                        for w in top_worms]
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(f"EW{j+1}: Ridge weights on {n_shared} shared neurons "
                     f"(instantaneous)", fontsize=11)
        plt.colorbar(im, ax=ax, label="Weight (standardised)")
        fig.tight_layout()
        fig.savefig(out_dir / f"01_weights_EW{j+1}_heatmap.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Plot 2: Pairwise correlation matrices ─────────────────────────
    fig, axes = plt.subplots(1, min(K, 4), figsize=(4 * min(K, 4), 4))
    if min(K, 4) == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        im = ax.imshow(corr_per_mode[j], vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n_w))
        ax.set_xticklabels([w[:10] for w in top_worms], rotation=45,
                           ha="right", fontsize=7)
        ax.set_yticks(range(n_w))
        ax.set_yticklabels([w[:10] for w in top_worms], fontsize=7)
        triu = corr_per_mode[j][np.triu_indices(n_w, k=1)]
        ax.set_title(f"EW{j+1}  (mean r={np.mean(triu):.2f})", fontsize=10)
        for a in range(n_w):
            for b2 in range(n_w):
                v = corr_per_mode[j, a, b2]
                ax.text(b2, a, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.5 else "black")
    fig.suptitle(f"Pairwise weight correlation (instantaneous, "
                 f"{n_shared} shared neurons)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "02_pairwise_weight_corr.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: Mean ± std weight profile across worms ────────────────
    fig, axes = plt.subplots(min(K, 4), 1,
                             figsize=(max(10, n_shared * 0.6), 3 * min(K, 4)),
                             sharex=True)
    if min(K, 4) == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        w_mean = W[:, j, :].mean(axis=0)
        w_std = W[:, j, :].std(axis=0)
        x = np.arange(n_shared)
        ax.bar(x, w_mean, yerr=w_std, capsize=3, alpha=0.7,
               color=["#d62728" if m > 0 else "#1f77b4" for m in w_mean],
               edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel(f"EW{j+1} weight", fontsize=10)
        ax.set_title(f"EW{j+1}: mean ± std weight across {n_w} worms",
                     fontsize=10)
    axes[-1].set_xticks(range(n_shared))
    axes[-1].set_xticklabels(shared_all, rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "03_mean_weight_profile.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: Transfer performance ──────────────────────────────────
    wids_ok = [w for w in sorted(transfer_results.keys())
               if "missing" not in transfer_results[w]]
    pooled_r2 = np.array([transfer_results[w]["r2"][:K] for w in wids_ok])
    is_train = np.array([transfer_results[w].get("in_training", False)
                         for w in wids_ok])

    # Compare pooled vs per-worm Ridge
    per_worm_r2 = []
    for w in wids_ok:
        if w in all_r2 and "Ridge" in all_r2[w]:
            per_worm_r2.append(all_r2[w]["Ridge"][:K])
        else:
            per_worm_r2.append([np.nan] * K)
    per_worm_r2 = np.array(per_worm_r2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart mean R² comparison
    ax = axes[0]
    pw_means = np.nanmean(per_worm_r2, axis=1)  # per worm
    pool_means = np.nanmean(pooled_r2, axis=1)
    x = np.arange(len(wids_ok))
    w_bar = 0.35
    bars1 = ax.bar(x - w_bar/2, pw_means, w_bar, label="Per-worm Ridge",
                   color="#1f77b4", alpha=0.7, edgecolor="white")
    bars2 = ax.bar(x + w_bar/2, pool_means, w_bar, label="Pooled Ridge",
                   color="#d62728", alpha=0.7, edgecolor="white")
    # Highlight training worms
    for i, tr in enumerate(is_train):
        if tr:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="green")
    ax.set_xticks(x)
    ax.set_xticklabels([w[:10] for w in wids_ok], rotation=60,
                       ha="right", fontsize=7)
    ax.set_ylabel("Mean R² (6 modes)")
    ax.set_title("Per-worm Ridge vs Pooled (shared neurons)")
    ax.legend(fontsize=9)
    ax.axhline(0, color="k", lw=0.5, ls="--")

    # Right: scatter per-worm vs pooled
    ax = axes[1]
    colors = ["green" if tr else "gray" for tr in is_train]
    ax.scatter(pw_means, pool_means, c=colors, s=40, alpha=0.7,
               edgecolors="white", linewidth=0.5, zorder=3)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0], -0.1)
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1], 0.6)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Per-worm Ridge R²")
    ax.set_ylabel("Pooled Ridge R² (shared neurons)")
    ax.set_title("Transfer: pooled vs per-worm")
    ax.set_aspect("equal")
    # Annotate
    train_mean = np.nanmean(pool_means[is_train])
    test_mean = np.nanmean(pool_means[~is_train])
    ax.text(0.05, 0.95, f"Train worms: {train_mean:.3f}\n"
            f"Test worms: {test_mean:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_dir / "04_transfer_performance.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Plot 5: Top neurons by mean |weight| across worms (with lags) ─
    # Use common_neurons (>= 70% of worms) with lagged weights
    all_neuron_importance = {}  # neuron → (K,) mean |weight|
    all_neuron_weights = {}    # neuron → (n_worms_with_it, K)
    for nm in common_neurons:
        ws = []
        for wi, wid in enumerate(top_worms):
            d = worm_data[wid]
            if nm in d["weights"]:
                ws.append(d["weights"][nm])
        if ws:
            ws = np.array(ws)  # (n_worms_with, K)
            all_neuron_weights[nm] = ws
            all_neuron_importance[nm] = np.mean(np.abs(ws), axis=0)

    # Rank by total importance (sum across EW1-4)
    ranked = sorted(all_neuron_importance.keys(),
                    key=lambda n: np.sum(all_neuron_importance[n][:4]),
                    reverse=True)

    fig, axes = plt.subplots(min(K, 4), 1,
                             figsize=(max(12, len(ranked) * 0.5),
                                      3 * min(K, 4)),
                             sharex=True)
    if min(K, 4) == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        means = [np.mean(all_neuron_weights[nm][:, j]) for nm in ranked]
        stds = [np.std(all_neuron_weights[nm][:, j]) for nm in ranked]
        x = np.arange(len(ranked))
        colors = ["#d62728" if m > 0 else "#1f77b4" for m in means]
        ax.bar(x, means, yerr=stds, capsize=2, color=colors,
               alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel(f"EW{j+1}", fontsize=10)
    axes[-1].set_xticks(range(len(ranked)))
    axes[-1].set_xticklabels(ranked, rotation=45, ha="right", fontsize=8)
    fig.suptitle(f"Neuron importance (mean ± std across worms, lagged Ridge)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "05_neuron_importance_ranked.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Plot 6: Consistency — coefficient of variation per neuron ──────
    fig, ax = plt.subplots(figsize=(max(10, len(common_neurons) * 0.5), 5))
    cv_per_neuron = {}
    for nm in common_neurons:
        if nm in all_neuron_weights:
            ws = all_neuron_weights[nm]  # (n_worms, K)
            # CV = std / |mean| for each mode, then average
            means = np.mean(ws, axis=0)
            stds = np.std(ws, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv = np.where(np.abs(means) > 1e-6, stds / np.abs(means), np.nan)
            cv_per_neuron[nm] = np.nanmean(cv[:4])

    neurons_sorted = sorted(cv_per_neuron.keys(), key=lambda n: cv_per_neuron[n])
    cvs = [cv_per_neuron[n] for n in neurons_sorted]
    colors = ["#2ca02c" if c < 1.0 else "#ff7f0e" if c < 2.0 else "#d62728"
              for c in cvs]
    ax.bar(range(len(neurons_sorted)), cvs, color=colors, alpha=0.7,
           edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5, label="CV=1")
    ax.set_xticks(range(len(neurons_sorted)))
    ax.set_xticklabels(neurons_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Coefficient of Variation (across worms)")
    ax.set_title("Weight consistency per neuron (green=CV<1, orange=1-2, red>2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_weight_cv_per_neuron.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════
    #  Summary text
    # ═══════════════════════════════════════════════════════════════════
    lines = []
    lines.append("=" * 70)
    lines.append("  WEIGHT COMPARISON SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  Top worms analysed: {len(top_worms)}")
    lines.append(f"  Shared neurons (all): {len(shared_all)}  {shared_all}")
    lines.append(f"  Common neurons (>={threshold}): {len(common_neurons)}")
    lines.append("")
    lines.append("  WEIGHT CORRELATION (instantaneous, shared neurons):")
    for j in range(min(K, 4)):
        triu = corr_per_mode[j][np.triu_indices(n_w, k=1)]
        lines.append(f"    EW{j+1}: mean r={np.mean(triu):.3f}, "
                     f"median={np.median(triu):.3f}")
    lines.append("")
    lines.append("  WEIGHT CORRELATION (lagged L2, shared neurons):")
    for j in range(min(K, 4)):
        triu = corr_lag[j][np.triu_indices(n_w, k=1)]
        lines.append(f"    EW{j+1}: mean r={np.mean(triu):.3f}, "
                     f"median={np.median(triu):.3f}")
    lines.append("")
    lines.append("  POOLED MODEL TRANSFER:")
    lines.append(f"    Train worms mean R²: "
                 f"{np.nanmean(pool_means[is_train]):.3f}")
    lines.append(f"    Test  worms mean R²: "
                 f"{np.nanmean(pool_means[~is_train]):.3f}")
    lines.append(f"    Per-worm Ridge mean R² (train): "
                 f"{np.nanmean(pw_means[is_train]):.3f}")
    lines.append(f"    Per-worm Ridge mean R² (test):  "
                 f"{np.nanmean(pw_means[~is_train]):.3f}")
    lines.append("")
    lines.append("  MOST CONSISTENT NEURONS (lowest CV):")
    for nm in neurons_sorted[:10]:
        cv = cv_per_neuron[nm]
        imp = np.sum(all_neuron_importance.get(nm, np.zeros(K))[:4])
        lines.append(f"    {nm:12s}  CV={cv:.2f}  total_importance={imp:.3f}")
    lines.append("")
    lines.append("  MOST IMPORTANT NEURONS (by mean |weight|):")
    for nm in ranked[:10]:
        imp = all_neuron_importance[nm]
        lines.append(f"    {nm:12s}  "
                     + "  ".join(f"EW{j+1}={imp[j]:.3f}" for j in range(4)))
    lines.append("=" * 70)

    summary = "\n".join(lines)
    print(f"\n{summary}")

    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary)

    # Save pooled weights for later use
    np.savez(out_dir / "pooled_weights.npz",
             coefs=pooled_coefs,
             intercepts=pooled_intc,
             mu=mu_pool,
             sigma=sig_pool,
             shared_neurons=shared_all,
             n_lags=n_lags,
             K=K)

    print(f"\n  All outputs → {out_dir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
