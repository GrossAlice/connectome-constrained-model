#!/usr/bin/env python
"""Neuron importance analysis for the Ridge decoder.

Two importance measures per neuron per eigenworm mode:
  1. **Coefficient magnitude** – sum of |wⱼ| across lags for each neuron.
  2. **Permutation importance** – Δ R² when that neuron's activity
     (across all lags) is time-shuffled.

Usage
-----
.venv/bin/python -m scripts.neuron_importance \
    --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2023-01-17-14.h5" \
    --out_dir output_plots/neuron_importance
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.behavior_decoder_eval import (
    _log_ridge_grid,
    _ridge_cv_single_target,
    build_lagged_features_np,
)

try:
    from scripts.benchmark_ar_decoder_v2 import load_data
except ModuleNotFoundError:
    from benchmark_ar_decoder_v2 import load_data


# ================================================================== #
#  Helpers
# ================================================================== #

def _load_neuron_names(h5_path: str) -> list[str]:
    """Read neuron names from h5 file."""
    with h5py.File(h5_path, "r") as f:
        for key in ("neuron_names", "gcamp/neuron_labels", "gcamp/neuron_names"):
            if key in f:
                return [n.decode() if isinstance(n, bytes) else n
                        for n in f[key][:]]
    raise KeyError("Cannot find neuron names in h5 file")


def _motor_neuron_indices(all_names: list[str]) -> tuple[list[int], list[str]]:
    """Return indices and names for the motor-neuron subset."""
    motor_file = (Path(__file__).resolve().parent.parent /
                  "data/used/masks+motor neurons/motor_neurons_with_control.txt")
    motor_names = [l.strip() for l in motor_file.read_text().splitlines()
                   if l.strip() and not l.startswith("#")]
    name2idx = {n: i for i, n in enumerate(all_names)}
    idx = [name2idx[n] for n in motor_names if n in name2idx]
    names = [all_names[i] for i in idx]
    return idx, names


def _ridge_fit(X_train, y_train, ridge_grid, n_inner):
    """Ridge CV on training data. Returns (coef, intercept, best_alpha)."""
    idx = np.arange(X_train.shape[0])
    fit = _ridge_cv_single_target(X_train, y_train, idx, ridge_grid, n_inner)
    return fit["coef"], fit["intercept"], fit["best_lambda"]


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ================================================================== #
#  Coefficient-based importance
# ================================================================== #

def coef_importance(coefs: np.ndarray, n_neurons: int, n_lags: int,
                    ) -> np.ndarray:
    """Aggregate |coefficient| per neuron across all lags.

    Parameters
    ----------
    coefs : (d_features, K)  – Ridge coefficients for K modes.
    n_neurons : M
    n_lags : L

    Returns
    -------
    importance : (M, K) – per-neuron, per-mode importance.

    Layout of build_lagged_features_np:
        lag-0: columns [0 : M]
        lag-1: columns [M : 2M]
        ...
        lag-L: columns [L*M : (L+1)*M]
    So neuron j's columns are at [j, j+M, j+2M, ..., j+L*M].
    """
    d_features, K = coefs.shape
    assert d_features == n_neurons * (n_lags + 1), \
        f"Feature dim mismatch: {d_features} != {n_neurons}×{n_lags+1}"
    importance = np.zeros((n_neurons, K))
    for lag in range(n_lags + 1):
        # columns for this lag: [lag*M : (lag+1)*M], one per neuron
        start = lag * n_neurons
        end = start + n_neurons
        importance += np.abs(coefs[start:end, :])
    return importance


# ================================================================== #
#  Permutation importance
# ================================================================== #

def permutation_importance(X: np.ndarray, b: np.ndarray,
                           coefs: np.ndarray, intercepts: np.ndarray,
                           n_neurons: int, n_lags: int,
                           n_repeats: int = 10,
                           rng_seed: int = 42) -> np.ndarray:
    """Permutation importance: shuffle each neuron's lag block, measure R² drop.

    Parameters
    ----------
    X : (T, d_features) – lagged neural features (test set).
    b : (T, K)          – ground truth eigenworm amplitudes (test set).
    coefs : (d_features, K)
    intercepts : (K,)
    n_neurons : M
    n_lags : L
    n_repeats : number of shuffle repetitions to average.

    Returns
    -------
    importance : (M, K) – mean R² drop per neuron per mode.
    """
    T_test, K = b.shape
    rng = np.random.default_rng(rng_seed)

    # Baseline R² per mode
    pred_base = X @ coefs + intercepts[None, :]
    r2_base = np.array([r2_score(b[:, j], pred_base[:, j]) for j in range(K)])

    importance = np.zeros((n_neurons, K))

    for ni in range(n_neurons):
        # Columns for this neuron across all lags
        cols = [ni + lag * n_neurons for lag in range(n_lags + 1)]
        r2_perm_sum = np.zeros(K)
        for _rep in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(T_test)
            for c in cols:
                X_perm[:, c] = X[perm_idx, c]
            pred_perm = X_perm @ coefs + intercepts[None, :]
            for j in range(K):
                r2_perm_sum[j] += r2_score(b[:, j], pred_perm[:, j])
        r2_perm_mean = r2_perm_sum / n_repeats
        importance[ni, :] = r2_base - r2_perm_mean  # Δ R²

    return importance


# ================================================================== #
#  Leave-one-out neuron importance
# ================================================================== #

def leave_one_out_importance(u: np.ndarray, b: np.ndarray,
                             n_lags: int, n_folds: int,
                             ridge_grid: np.ndarray,
                             warmup: int) -> np.ndarray:
    """Leave-one-neuron-out: drop each neuron, refit Ridge, measure R² drop.

    Returns
    -------
    importance : (M, K) – R² drop when neuron i is removed.
    """
    T, M = u.shape
    K = b.shape[1]
    n_inner = n_folds - 1

    # Full-model held-out R²
    X_full = build_lagged_features_np(u, n_lags)
    full_r2 = _cv_ridge_r2(X_full, b, n_folds, warmup, ridge_grid, n_inner)
    print(f"    Full-model R²: {[f'{v:.3f}' for v in full_r2]}")

    importance = np.zeros((M, K))
    for ni in range(M):
        # Drop neuron ni
        keep = np.concatenate([np.arange(ni), np.arange(ni + 1, M)])
        u_drop = u[:, keep]
        X_drop = build_lagged_features_np(u_drop, n_lags)
        drop_r2 = _cv_ridge_r2(X_drop, b, n_folds, warmup, ridge_grid, n_inner)
        importance[ni, :] = full_r2 - drop_r2
        if (ni + 1) % 5 == 0 or ni == M - 1:
            print(f"      LOO neuron {ni+1}/{M} done")
    return importance


def _cv_ridge_r2(X: np.ndarray, b: np.ndarray,
                 n_folds: int, warmup: int,
                 ridge_grid: np.ndarray, n_inner: int) -> np.ndarray:
    """Cross-validated R² per mode for Ridge."""
    T, K = b.shape
    valid_len = T - warmup
    fold_size = valid_len // n_folds

    ho_pred = np.full((T, K), np.nan)
    for fi in range(n_folds):
        ts = warmup + fi * fold_size
        te = warmup + (fi + 1) * fold_size if fi < n_folds - 1 else T
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[ts:te] = False
        train_idx = np.where(train_mask)[0]

        X_tr = X[train_idx]
        X_te = X[ts:te]
        b_tr = b[train_idx]

        for j in range(K):
            coef, intc, _ = _ridge_fit(X_tr, b_tr[:, j], ridge_grid, n_inner)
            ho_pred[ts:te, j] = X_te @ coef + intc

    valid = ~np.isnan(ho_pred[:, 0])
    return np.array([r2_score(b[valid, j], ho_pred[valid, j]) for j in range(K)])


# ================================================================== #
#  Plotting
# ================================================================== #

def plot_bar_importance(names: list[str], importance: np.ndarray,
                        title: str, out_path: Path,
                        top_n: int = 15):
    """Horizontal bar chart of top-N neurons per mode + overall."""
    M, K = importance.shape
    overall = np.mean(importance, axis=1)

    n_panels = K + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, max(6, top_n * 0.35)))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    for col in range(n_panels):
        ax = axes[col]
        if col < K:
            vals = importance[:, col]
            ax.set_title(f"a{col+1}", fontsize=11)
        else:
            vals = overall
            ax.set_title("Overall (mean)", fontsize=11)

        order = np.argsort(vals)[::-1][:top_n]
        top_names = [names[i] for i in order]
        top_vals = vals[order]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
        ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.set_xlabel("Importance")
        ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_heatmap(names: list[str], importance: np.ndarray,
                 title: str, out_path: Path, top_n: int = 20):
    """Heatmap of neuron × mode importance for top-N neurons."""
    M, K = importance.shape
    overall = np.mean(importance, axis=1)
    order = np.argsort(overall)[::-1][:top_n]

    data = importance[order]
    ylabels = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(5, K * 1.2), max(5, top_n * 0.35)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"a{j+1}" for j in range(K)])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Importance")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ================================================================== #
#  Main
# ================================================================== #

def main():
    ap = argparse.ArgumentParser(description="Neuron importance for Ridge decoder")
    ap.add_argument("--h5", required=True, help="Path to .h5 recording file")
    ap.add_argument("--n_modes", type=int, default=6, help="Number of eigenworm modes")
    ap.add_argument("--neural_lags", type=int, default=3, help="Number of neural lags")
    ap.add_argument("--n_folds", type=int, default=5, help="CV folds")
    ap.add_argument("--max_frames", type=int, default=0, help="Truncate at N frames")
    ap.add_argument("--all_neurons", action="store_true",
                    help="Use all neurons (not just motor)")
    ap.add_argument("--perm_repeats", type=int, default=20,
                    help="Number of permutation repeats")
    ap.add_argument("--skip_loo", action="store_true",
                    help="Skip leave-one-out (slow)")
    ap.add_argument("--top_n", type=int, default=15,
                    help="Show top-N neurons in plots")
    ap.add_argument("--out_dir", default="output_plots/neuron_importance")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    all_neuron_names = _load_neuron_names(args.h5)
    u, b_full, dt = load_data(args.h5, all_neurons=args.all_neurons)
    if args.max_frames > 0:
        u = u[:args.max_frames]
        b_full = b_full[:args.max_frames]
    K = min(args.n_modes, b_full.shape[1])
    b = b_full[:, :K]
    T, M = u.shape
    n_lags = args.neural_lags
    warmup = max(2, n_lags)

    # Resolve neuron names for the columns of u
    if args.all_neurons:
        neuron_names = all_neuron_names
    else:
        _, neuron_names = _motor_neuron_indices(all_neuron_names)
    assert len(neuron_names) == M, \
        f"Names mismatch: {len(neuron_names)} names vs {M} columns"

    print(f"\n  Data: T={T}, M={M} neurons, K={K} modes, "
          f"n_lags={n_lags}, dt={dt:.2f}s")
    print(f"  Neurons: {neuron_names}\n")

    # ── Build features ────────────────────────────────────────────────
    X = build_lagged_features_np(u, n_lags)
    d_in = X.shape[1]  # M * (n_lags + 1)

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = args.n_folds - 1

    # ── Fit Ridge on full training set (all folds as train) ──────────
    # We use all valid frames for the coefficient-based and permutation
    # analyses — fit on train, evaluate on held-out via CV.
    print("  ══ Fitting Ridge (full data, for coefficient analysis) ══")
    coefs_full = np.zeros((d_in, K))
    intercepts_full = np.zeros(K)
    valid_idx = np.arange(warmup, T)
    for j in range(K):
        fit = _ridge_cv_single_target(X[valid_idx], b[valid_idx, j],
                                       np.arange(valid_idx.size),
                                       ridge_grid, n_inner)
        coefs_full[:, j] = fit["coef"]
        intercepts_full[j] = fit["intercept"]
        print(f"    Mode a{j+1}: α={fit['best_lambda']:.2e}")

    # ── 1. Coefficient magnitude importance ───────────────────────────
    print("\n  ══ Coefficient-magnitude importance ══")
    imp_coef = coef_importance(coefs_full, M, n_lags)

    # Normalize per mode for display
    imp_coef_norm = imp_coef / (imp_coef.max(axis=0, keepdims=True) + 1e-12)

    plot_bar_importance(neuron_names, imp_coef,
                        "Ridge coefficient magnitude (|w| across lags)",
                        out_dir / "coef_magnitude_bar.png",
                        top_n=args.top_n)
    plot_heatmap(neuron_names, imp_coef_norm,
                 "Ridge |coef| (normalized per mode)",
                 out_dir / "coef_magnitude_heatmap.png",
                 top_n=args.top_n)

    # ── 2. Permutation importance (on held-out folds) ─────────────────
    print("\n  ══ Permutation importance (CV held-out) ══")
    valid_len = T - warmup
    fold_size = valid_len // args.n_folds
    folds = []
    for i in range(args.n_folds):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < args.n_folds - 1 else T
        folds.append((s, e))

    # Accumulate permutation importance across folds
    imp_perm_acc = np.zeros((M, K))
    for fi, (ts, te) in enumerate(folds):
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[ts:te] = False
        train_idx = np.where(train_mask)[0]

        X_tr = X[train_idx]
        X_te = X[ts:te]
        b_te = b[ts:te]
        b_tr = b[train_idx]

        # Fit Ridge on train fold
        coefs_fold = np.zeros((d_in, K))
        ints_fold = np.zeros(K)
        for j in range(K):
            c, ic, _ = _ridge_fit(X_tr, b_tr[:, j], ridge_grid, n_inner)
            coefs_fold[:, j] = c
            ints_fold[j] = ic

        # Permutation importance on held-out fold
        imp_fold = permutation_importance(
            X_te, b_te, coefs_fold, ints_fold,
            M, n_lags, n_repeats=args.perm_repeats)
        imp_perm_acc += imp_fold
        print(f"    Fold {fi+1}/{args.n_folds} done")

    imp_perm = imp_perm_acc / args.n_folds
    imp_perm_norm = imp_perm / (np.abs(imp_perm).max(axis=0, keepdims=True) + 1e-12)

    plot_bar_importance(neuron_names, imp_perm,
                        "Permutation importance (mean Δ R² across folds)",
                        out_dir / "permutation_bar.png",
                        top_n=args.top_n)
    plot_heatmap(neuron_names, imp_perm_norm,
                 "Permutation Δ R² (normalized per mode)",
                 out_dir / "permutation_heatmap.png",
                 top_n=args.top_n)

    # ── 3. Leave-one-out (optional) ──────────────────────────────────
    if not args.skip_loo:
        print("\n  ══ Leave-one-out neuron importance ══")
        imp_loo = leave_one_out_importance(
            u, b, n_lags, args.n_folds, ridge_grid, warmup)
        imp_loo_norm = imp_loo / (np.abs(imp_loo).max(axis=0, keepdims=True) + 1e-12)

        plot_bar_importance(neuron_names, imp_loo,
                            "Leave-one-neuron-out (Δ R²)",
                            out_dir / "loo_bar.png",
                            top_n=args.top_n)
        plot_heatmap(neuron_names, imp_loo_norm,
                     "LOO Δ R² (normalized per mode)",
                     out_dir / "loo_heatmap.png",
                     top_n=args.top_n)
    else:
        imp_loo = None

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print("  Neuron importance summary (top {})".format(args.top_n))
    print(f"{'═' * 80}")

    overall_coef = np.mean(imp_coef, axis=1)
    overall_perm = np.mean(imp_perm, axis=1)

    # Build ranking
    ranking = []
    for ni in range(M):
        entry = {
            "neuron": neuron_names[ni],
            "coef_overall": float(overall_coef[ni]),
            "perm_overall": float(overall_perm[ni]),
        }
        for j in range(K):
            entry[f"coef_a{j+1}"] = float(imp_coef[ni, j])
            entry[f"perm_a{j+1}"] = float(imp_perm[ni, j])
        if imp_loo is not None:
            entry["loo_overall"] = float(np.mean(imp_loo[ni]))
            for j in range(K):
                entry[f"loo_a{j+1}"] = float(imp_loo[ni, j])
        ranking.append(entry)

    # Print table sorted by permutation overall
    ranking.sort(key=lambda x: x["perm_overall"], reverse=True)
    header = f"  {'Rank':<5} {'Neuron':<10} {'Perm ΔR²':>9} {'|Coef|':>9}"
    if imp_loo is not None:
        header += f" {'LOO ΔR²':>9}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for rank, entry in enumerate(ranking[:args.top_n], 1):
        line = (f"  {rank:<5} {entry['neuron']:<10} "
                f"{entry['perm_overall']:>9.4f} "
                f"{entry['coef_overall']:>9.4f}")
        if imp_loo is not None:
            line += f" {entry['loo_overall']:>9.4f}"
        print(line)

    # Per-mode leaders
    print(f"\n  Per-mode top neuron (by permutation Δ R²):")
    for j in range(K):
        best_ni = np.argmax(imp_perm[:, j])
        print(f"    a{j+1}: {neuron_names[best_ni]:<10} "
              f"Δ R² = {imp_perm[best_ni, j]:.4f}")

    # Save JSON
    summary = {
        "h5": args.h5,
        "n_neurons": M,
        "n_modes": K,
        "n_lags": n_lags,
        "neuron_names": neuron_names,
        "ranking": ranking,
    }
    json_path = out_dir / "importance_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary to {json_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
