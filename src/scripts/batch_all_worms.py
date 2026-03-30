#!/usr/bin/env python
"""Run OLS, Ridge, and MLP (with weight-decay CV) on all 40 worms.

The Polar-Ridge model exploits the quadrature relationship between a₁
and a₂: brain neurons encode dφ/dt (phase velocity = crawling speed),
not φ itself.  Instead of predicting a₁, a₂ independently, we:
  1.  Ridge-regress  n(t) → dφ/dt   (easy: neurons directly encode this)
  2.  Ridge-regress  n(t) → r(t)    (amplitude is slowly varying)
  3.  Integrate:  φ̂(t) = φ(t₀) + Σ dφ̂/dt · Δt
  4.  Reconstruct: â₁ = r̂·cos(φ̂),  â₂ = r̂·sin(φ̂)

Modes a₃–a₆ are predicted the same way as standard Ridge.

Produces:
  • per-worm JSON results  (out_dir/results.json)
  • summary heatmap  (out_dir/heatmap_per_worm.png)
  • summary scatter  (out_dir/scatter_per_worm.png)
  • summary bar      (out_dir/bar_summary.png)
"""
from __future__ import annotations

import argparse, glob, json, sys, time, traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.behavior_decoder_eval import (
    _log_ridge_grid,
    _fit_ridge_regression,
    _ridge_cv_single_target,
    build_lagged_features_np,
)
try:
    from scripts.benchmark_ar_decoder_v2 import (
        load_data, build_ar_features, r2_score,
    )
except ModuleNotFoundError:
    from benchmark_ar_decoder_v2 import (
        load_data, build_ar_features, r2_score,
    )
from scripts.unified_benchmark import _ridge_fit, _ols_fit

import torch
import torch.nn as nn


WD_GRID = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]


def _make_mlp(d_in, K, hidden=32):
    """Create a fresh 2-layer MLP."""
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.LayerNorm(hidden),
        nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, K),
    )


def _train_mlp_one(X_t, y_t, d_in, K, hidden=32, lr=1e-3,
                    epochs=200, weight_decay=1e-3):
    """Train a single MLP with a fixed weight_decay.  Returns (model, best_loss)."""
    mlp = _make_mlp(d_in, K, hidden)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr,
                           weight_decay=weight_decay)
    best_l, best_s, pat = float("inf"), None, 0
    for ep in range(epochs):
        mlp.train()
        loss = nn.functional.mse_loss(mlp(X_t), y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best_l - 1e-6:
            best_l, pat = loss.item(), 0
            best_s = {k: v.clone() for k, v in mlp.state_dict().items()}
        else:
            pat += 1
            if pat > 20:
                break
    if best_s:
        mlp.load_state_dict(best_s)
    mlp.eval()
    return mlp, best_l


def _train_mlp_cv(X_train, y_train, K, n_inner=4, hidden=32,
                   lr=1e-3, epochs=200):
    """MLP with inner-CV over weight_decay grid (analogous to Ridge α-CV).

    1. Split training data into *n_inner* contiguous temporal folds.
    2. For each weight_decay candidate, train on (n_inner−1) folds,
       evaluate MSE on the held-out fold.  Average across folds.
    3. Retrain on full training data with the best weight_decay.
    Returns trained model (eval mode).
    """
    n_tr = X_train.shape[0]
    d_in = X_train.shape[1]
    fold_size = n_tr // n_inner

    # Build inner-fold indices (contiguous temporal blocks)
    inner_folds = []
    for i in range(n_inner):
        s = i * fold_size
        e = s + fold_size if i < n_inner - 1 else n_tr
        inner_folds.append(np.arange(s, e))

    X_t_all = torch.tensor(X_train, dtype=torch.float32)
    y_t_all = torch.tensor(y_train, dtype=torch.float32)

    best_wd, best_cv_mse = WD_GRID[0], float("inf")
    for wd in WD_GRID:
        cv_mse = 0.0
        for fi in range(n_inner):
            val_idx = inner_folds[fi]
            tr_idx = np.concatenate(
                [inner_folds[j] for j in range(n_inner) if j != fi])

            X_tr_f = X_t_all[tr_idx]
            y_tr_f = y_t_all[tr_idx]
            X_va_f = X_t_all[val_idx]
            y_va_f = y_t_all[val_idx]

            mlp, _ = _train_mlp_one(X_tr_f, y_tr_f, d_in, K,
                                     hidden, lr, epochs, wd)
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(
                    mlp(X_va_f), y_va_f).item()
            cv_mse += val_loss
        cv_mse /= n_inner
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_wd = wd

    # Retrain on full training data with best weight_decay
    mlp_final, _ = _train_mlp_one(X_t_all, y_t_all, d_in, K,
                                   hidden, lr, epochs, best_wd)
    return mlp_final, best_wd


def _predict_mlp(mlp, X):
    with torch.no_grad():
        return mlp(torch.tensor(X, dtype=torch.float32)).numpy()


# ──────────────────────────────────────────────────────────────────────
#  Polar coordinate helpers
# ──────────────────────────────────────────────────────────────────────
def _compute_polar_targets(a1, a2, dt):
    """From a₁(t), a₂(t) compute r(t) and dφ/dt.

    Uses the robust formula:  dφ/dt = (a₁·ȧ₂ − a₂·ȧ₁) / (a₁² + a₂²)
    which avoids atan2 unwrapping issues.
    """
    r = np.sqrt(a1 ** 2 + a2 ** 2)
    # Central difference for derivatives (forward at edges)
    da1 = np.gradient(a1, dt)
    da2 = np.gradient(a2, dt)
    r2 = a1 ** 2 + a2 ** 2
    r2 = np.maximum(r2, 1e-8)  # avoid division by zero
    dphi_dt = (a1 * da2 - a2 * da1) / r2
    return r, dphi_dt


# ──────────────────────────────────────────────────────────────────────
#  Single-worm benchmark
# ──────────────────────────────────────────────────────────────────────
def _run_one_worm(h5_path: str, *,
                  n_lags: int = 8,
                  n_folds: int = 5,
                  K: int = 6,
                  **kwargs) -> dict:
    """Return dict {model_name: [r2_a1, ..., r2_aK], 'T': int, ...}."""

    u, b_full, dt = load_data(h5_path, all_neurons=False)
    K = min(K, b_full.shape[1])
    b = b_full[:, :K]
    T = b.shape[0]
    warmup = max(2, n_lags)
    M_raw = u.shape[1]
    X_neural = build_lagged_features_np(u, n_lags)
    d_in = X_neural.shape[1]

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = n_folds - 1

    # ── folds (contiguous temporal blocks) ────────────────────────────
    valid_len = T - warmup
    fold_size = valid_len // n_folds
    folds = []
    for i in range(n_folds):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < n_folds - 1 else T
        folds.append((s, e))

    RIDGE  = "Ridge"
    OLS    = "OLS"
    MLP    = "MLP"
    models = [OLS, RIDGE, MLP]
    ho_preds = {m: np.full((T, K), np.nan) for m in models}

    for fi, (ts, te) in enumerate(folds):
        train_idx = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        X_tr = X_neural[train_idx]
        y_tr = b[train_idx]

        # Standardise features
        mu, sigma = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr_s = (X_tr - mu) / sigma
        X_te_s = (X_neural[ts:te] - mu) / sigma

        # ── OLS (per-timestep, all K modes) ──────────────────────────
        for j in range(K):
            coef, intc = _ols_fit(X_tr_s, y_tr[:, j])
            ho_preds[OLS][ts:te, j] = X_te_s @ coef + intc

        # ── Ridge (per-timestep, all K modes) ────────────────────────
        for j in range(K):
            coef, intc, _ = _ridge_fit(X_tr_s, y_tr[:, j],
                                       ridge_grid, n_inner)
            ho_preds[RIDGE][ts:te, j] = X_te_s @ coef + intc

        # ── MLP with inner-CV over weight_decay ─────────────────────
        mlp, best_wd = _train_mlp_cv(X_tr_s, y_tr, K, n_inner=n_inner,
                                      hidden=32, lr=1e-3, epochs=200)
        ho_preds[MLP][ts:te] = _predict_mlp(mlp, X_te_s)

    # ── R² ────────────────────────────────────────────────────────────
    valid = np.arange(warmup, T)
    results = {"T": int(T), "N": int(M_raw), "K": K, "dt": float(dt)}
    for name in models:
        preds = ho_preds[name]
        ok = np.isfinite(preds[valid, 0])
        idx = valid[ok]
        if idx.size < 10:
            results[name] = [float("nan")] * K
        else:
            results[name] = [
                float(r2_score(b[idx, j], preds[idx, j]))
                for j in range(K)]
    return results


# ──────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────
def _plot_heatmaps(all_results: dict, out_dir: Path, K: int = 6):
    """One heatmap per model: worms × modes."""
    models = ["OLS", "Ridge", "MLP"]
    worm_ids = sorted(all_results.keys())
    n_worms = len(worm_ids)
    mode_names = [f"a{j+1}" for j in range(K)]

    fig, axes = plt.subplots(1, 3, figsize=(18, max(8, n_worms * 0.35)),
                             sharey=True)
    for ax, mname in zip(axes, models):
        mat = np.array([all_results[w].get(mname, [np.nan]*K)[:K]
                        for w in worm_ids])
        im = ax.imshow(mat, aspect="auto", vmin=-0.2, vmax=1.0,
                       cmap="RdYlGn")
        ax.set_xticks(range(K))
        ax.set_xticklabels(mode_names, fontsize=10)
        ax.set_title(mname, fontsize=13, fontweight="bold")
        # Annotate each cell
        for i in range(n_worms):
            for j in range(K):
                v = mat[i, j]
                if np.isfinite(v):
                    color = "white" if v < 0.3 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color=color)
    axes[0].set_yticks(range(n_worms))
    axes[0].set_yticklabels(worm_ids, fontsize=7)
    axes[0].set_ylabel("Worm", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="R²")
    fig.suptitle("Per-worm decoder R² — 5-fold temporal CV, free-run",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    p = out_dir / "heatmap_per_worm.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap → {p}")


def _plot_scatter(all_results: dict, out_dir: Path, K: int = 6):
    """Scatter / strip plot: per-mode R² across worms, one panel per mode."""
    models = ["OLS", "Ridge", "MLP"]
    colors = {"OLS": "#d62728", "Ridge": "#1f77b4", "MLP": "#2ca02c"}
    worm_ids = sorted(all_results.keys())
    mode_names = [f"a{j+1}" for j in range(K)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    axes = axes.ravel()

    for j in range(K):
        ax = axes[j]
        for mi, mname in enumerate(models):
            vals = [all_results[w].get(mname, [np.nan]*K)[j]
                    for w in worm_ids]
            vals = np.array(vals)
            ok = np.isfinite(vals)
            x = np.full(ok.sum(), mi) + np.random.uniform(-0.15, 0.15,
                                                           size=ok.sum())
            ax.scatter(x, vals[ok], s=18, alpha=0.6, color=colors[mname],
                       edgecolors="white", linewidth=0.3)
            # Mean bar
            if ok.any():
                mn = np.nanmean(vals[ok])
                ax.plot([mi - 0.3, mi + 0.3], [mn, mn],
                        lw=2, color=colors[mname])
                ax.text(mi, mn + 0.03, f"{mn:.3f}", ha="center",
                        fontsize=8, fontweight="bold", color=colors[mname])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=9)
        ax.set_title(mode_names[j], fontsize=12)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
        ax.set_ylim(-0.3, 1.05)
        if j % 3 == 0:
            ax.set_ylabel("R²", fontsize=11)

    fig.suptitle(f"Per-mode R² across {len(worm_ids)} worms — "
                 f"OLS vs Ridge vs MLP",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    p = out_dir / "scatter_per_worm.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter → {p}")


def _plot_bar_summary(all_results: dict, out_dir: Path, K: int = 6):
    """Grouped bar chart: mean ± SEM across worms, per mode."""
    models = ["OLS", "Ridge", "MLP"]
    colors = {"OLS": "#d62728", "Ridge": "#1f77b4", "MLP": "#2ca02c"}
    worm_ids = sorted(all_results.keys())
    mode_names = [f"a{j+1}" for j in range(K)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(K)
    n = len(models)
    w = 0.8 / n

    for mi, mname in enumerate(models):
        mat = np.array([all_results[wid].get(mname, [np.nan]*K)[:K]
                        for wid in worm_ids])
        means = np.nanmean(mat, axis=0)
        sems  = np.nanstd(mat, axis=0) / np.sqrt(np.sum(np.isfinite(mat), axis=0).clip(1))
        ax.bar(x + mi * w, means, w, yerr=sems, capsize=3,
               label=mname, color=colors[mname], edgecolor="white",
               linewidth=0.5, alpha=0.85)
        for j in range(K):
            ax.text(x[j] + mi * w, means[j] + sems[j] + 0.02,
                    f"{means[j]:.3f}", ha="center", fontsize=7,
                    color=colors[mname])

    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(mode_names, fontsize=12)
    ax.set_ylabel("R²  (held-out, free-run)", fontsize=12)
    ax.set_title(f"Mean decoder R² across {len(worm_ids)} worms "
                 f"(± SEM) — 5-fold CV",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "bar_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar summary → {p}")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5_dir", required=True,
                    help="Directory containing .h5 worm files")
    ap.add_argument("--neural_lags", type=int, default=8)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_modes", type=int, default=6)
    ap.add_argument("--out_dir",
                    default="output_plots/behaviour_decoder/all_worms")
    ap.add_argument("--resume", action="store_true",
                    help="Skip worms already in results.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover h5 files ─────────────────────────────────────────────
    h5_files = sorted(glob.glob(str(Path(args.h5_dir) / "*.h5")))
    if not h5_files:
        print(f"  No .h5 files found in {args.h5_dir}")
        sys.exit(1)
    print(f"\n  Found {len(h5_files)} worms in {args.h5_dir}\n")

    # ── Resume support ────────────────────────────────────────────────
    json_path = out_dir / "results.json"
    if args.resume and json_path.exists():
        with open(json_path) as f:
            all_results = json.load(f)
        print(f"  Resuming: {len(all_results)} worms already done\n")
    else:
        all_results = {}

    t0 = time.time()

    for wi, h5_path in enumerate(h5_files):
        worm_id = Path(h5_path).stem          # e.g. "2023-01-17-14"
        if worm_id in all_results:
            print(f"  [{wi+1}/{len(h5_files)}] {worm_id}  ── SKIP (already done)")
            continue

        print(f"  [{wi+1}/{len(h5_files)}] {worm_id}  ──────────────────────")
        try:
            res = _run_one_worm(
                h5_path,
                n_lags=args.neural_lags,
                n_folds=args.n_folds,
                K=args.n_modes,
            )
            all_results[worm_id] = res
            # Print one-liner summary
            for mname in ["OLS", "Ridge", "MLP"]:
                vals = res.get(mname, [float("nan")]*6)
                print(f"    {mname:<8s}{' '.join(f'{v:7.3f}' for v in vals)}")
        except Exception as exc:
            print(f"    ✗ FAILED: {exc}")
            traceback.print_exc()
            all_results[worm_id] = {"error": str(exc)}

        # Checkpoint after every worm
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  All worms done in {elapsed:.0f}s ({elapsed/60:.1f} min)\n")

    # ── Filter out errors for plotting ────────────────────────────────
    plot_results = {k: v for k, v in all_results.items()
                    if "error" not in v}

    if len(plot_results) < 2:
        print("  Too few successful worms for plotting.")
        return

    K = args.n_modes

    # ── Generate plots ────────────────────────────────────────────────
    _plot_heatmaps(plot_results, out_dir, K=K)
    _plot_scatter(plot_results, out_dir, K=K)
    _plot_bar_summary(plot_results, out_dir, K=K)

    # ── Text summary ──────────────────────────────────────────────────
    txt_path = out_dir / "summary.txt"
    models = ["OLS", "Ridge", "MLP"]
    worm_ids = sorted(plot_results.keys())
    mode_names = [f"a{j+1}" for j in range(K)]
    with open(txt_path, "w") as f:
        f.write(f"Batch decoder benchmark — {len(worm_ids)} worms, "
                f"5-fold temporal CV\n")
        f.write(f"{'Worm':<20s}")
        for mname in models:
            for mn in mode_names:
                f.write(f"  {mname}_{mn:>3s}")
        f.write("\n")
        for wid in worm_ids:
            f.write(f"{wid:<20s}")
            for mname in models:
                vals = plot_results[wid].get(mname, [float("nan")]*K)
                for v in vals[:K]:
                    f.write(f"  {v:8.3f}")
            f.write("\n")
        # Means
        f.write(f"\n{'MEAN':<20s}")
        for mname in models:
            mat = np.array([plot_results[w].get(mname, [np.nan]*K)[:K]
                            for w in worm_ids])
            means = np.nanmean(mat, axis=0)
            for v in means:
                f.write(f"  {v:8.3f}")
        f.write("\n")
    print(f"  Summary → {txt_path}")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
