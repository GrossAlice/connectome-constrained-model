#!/usr/bin/env python
"""MLP architecture sweep: width × depth, reporting train & test R².

Answers: is the 2-layer H=32 MLP overfitting, underfitting, or just
right?  Ridge is included as a linear ceiling reference.

Usage:
    python -m scripts.mlp_arch_sweep \
        --h5 "data/used/.../2023-01-17-14.h5"
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.behavior_decoder_eval import (
    _log_ridge_grid, build_lagged_features_np,
)
try:
    from scripts.benchmark_ar_decoder_v2 import load_data, r2_score
except ModuleNotFoundError:
    from benchmark_ar_decoder_v2 import load_data, r2_score
from scripts.unified_benchmark import _ridge_fit


# ──────────────────────────────────────────────────────────────────────
def _build_mlp(d_in: int, K: int, hidden: int, n_layers: int,
               dropout: float = 0.1) -> nn.Sequential:
    """Build an MLP with *n_layers* hidden layers of width *hidden*."""
    layers = []
    in_dim = d_in
    for _ in range(n_layers):
        layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                   nn.ReLU(), nn.Dropout(dropout)]
        in_dim = hidden
    layers.append(nn.Linear(in_dim, K))
    return nn.Sequential(*layers)


def _train_mlp_flex(X_train, y_train, K, hidden=32, n_layers=2,
                    dropout=0.1, lr=1e-3, epochs=500, patience=30,
                    weight_decay=0.0):
    """Train MLP with flexible depth/width. Returns (model, train_loss)."""
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    d = X_train.shape[1]
    mlp = _build_mlp(d, K, hidden, n_layers, dropout)
    n_params = sum(p.numel() for p in mlp.parameters())

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
            if pat > patience:
                break
    if best_s:
        mlp.load_state_dict(best_s)
    mlp.eval()
    return mlp, n_params, best_l


def _predict(mlp, X):
    with torch.no_grad():
        return mlp(torch.tensor(X, dtype=torch.float32)).numpy()


def _r2_vec(y_true, y_pred, K):
    return np.array([r2_score(y_true[:, j], y_pred[:, j]) for j in range(K)])


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--neural_lags", type=int, default=8)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_modes", type=int, default=6)
    ap.add_argument("--out_dir",
                    default="output_plots/behaviour_decoder/mlp_sweep")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    u, b_full, dt = load_data(args.h5, all_neurons=False)
    K = min(args.n_modes, b_full.shape[1])
    b = b_full[:, :K]
    T = b.shape[0]
    n_lags = args.neural_lags
    warmup = max(2, n_lags)
    X = build_lagged_features_np(u, n_lags)
    d_in = X.shape[1]

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = args.n_folds - 1

    # Folds
    valid_len = T - warmup
    fold_size = valid_len // args.n_folds
    folds = []
    for i in range(args.n_folds):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < args.n_folds - 1 else T
        folds.append((s, e))

    # Architecture grid
    configs = [
        # (label, n_layers, hidden, dropout, weight_decay)
        ("Linear",          0,   0,   0.0, 0.0),     # = OLS via MLP
        ("1×16",            1,  16,   0.1, 0.0),
        ("1×32",            1,  32,   0.1, 0.0),
        ("1×64",            1,  64,   0.1, 0.0),
        ("1×128",           1, 128,   0.1, 0.0),
        ("2×16",            2,  16,   0.1, 0.0),
        ("2×32",            2,  32,   0.1, 0.0),      # current default
        ("2×64",            2,  64,   0.1, 0.0),
        ("2×128",           2, 128,   0.1, 0.0),
        ("3×32",            3,  32,   0.1, 0.0),
        ("3×64",            3,  64,   0.1, 0.0),
        # With weight decay (L2 regularization)
        ("2×32+wd1e-3",     2,  32,   0.1, 1e-3),
        ("2×64+wd1e-3",     2,  64,   0.1, 1e-3),
        ("2×128+wd1e-3",    2, 128,   0.1, 1e-3),
        ("2×32+wd1e-2",     2,  32,   0.1, 1e-2),
        # Higher dropout
        ("2×64+d0.3",       2,  64,   0.3, 0.0),
        ("2×128+d0.3",      2, 128,   0.3, 0.0),
    ]

    print(f"\n  Data: T={T}, K={K}, d_in={d_in}, {args.n_folds} folds")
    print(f"  Sweeping {len(configs)} architectures + Ridge\n")

    # Storage: {name: {'train': [...], 'test': [...]}}
    results = {}

    # Ridge baseline
    ridge_train_r2 = []
    ridge_test_r2 = []

    for fi, (ts, te) in enumerate(folds):
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[ts:te] = False
        train_idx = np.where(train_mask)[0]

        X_tr, X_te = X[train_idx], X[ts:te]
        b_tr, b_te = b[train_idx], b[ts:te]
        mu, sig = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr_s = (X_tr - mu) / sig
        X_te_s = (X_te - mu) / sig

        # Ridge
        preds_tr = np.zeros_like(b_tr)
        preds_te = np.zeros_like(b_te)
        for j in range(K):
            coef, intc, _ = _ridge_fit(X_tr_s, b_tr[:, j],
                                       ridge_grid, n_inner)
            preds_tr[:, j] = X_tr_s @ coef + intc
            preds_te[:, j] = X_te_s @ coef + intc
        ridge_train_r2.append(_r2_vec(b_tr, preds_tr, K))
        ridge_test_r2.append(_r2_vec(b_te, preds_te, K))

        # MLP sweep
        for label, n_layers, hidden, dropout, wd in configs:
            if label not in results:
                results[label] = {"train": [], "test": [], "params": 0}

            if n_layers == 0:
                # Pure linear (1 layer, no hidden)
                mlp = nn.Sequential(nn.Linear(d_in, K))
                X_t = torch.tensor(X_tr_s, dtype=torch.float32)
                y_t = torch.tensor(b_tr, dtype=torch.float32)
                opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
                for ep in range(200):
                    mlp.train()
                    loss = nn.functional.mse_loss(mlp(X_t), y_t)
                    opt.zero_grad(); loss.backward(); opt.step()
                mlp.eval()
                n_params = sum(p.numel() for p in mlp.parameters())
            else:
                mlp, n_params, _ = _train_mlp_flex(
                    X_tr_s, b_tr, K, hidden, n_layers, dropout,
                    weight_decay=wd)
            results[label]["params"] = n_params

            ptr = _predict(mlp, X_tr_s)
            pte = _predict(mlp, X_te_s)
            results[label]["train"].append(_r2_vec(b_tr, ptr, K))
            results[label]["test"].append(_r2_vec(b_te, pte, K))

        print(f"  Fold {fi+1}/{args.n_folds} done")

    # Aggregate
    ridge_train = np.mean(ridge_train_r2, axis=0)
    ridge_test = np.mean(ridge_test_r2, axis=0)

    # Print table
    mode_names = [f"a{j+1}" for j in range(K)]
    cw = 22
    print(f"\n{'═'*100}")
    print(f"  {'Model':<{cw}s} {'#par':>6s}"
          + "".join(f" {'tr_'+m:>7s}" for m in mode_names)
          + f" {'tr_mn':>7s}"
          + "".join(f" {'te_'+m:>7s}" for m in mode_names)
          + f" {'te_mn':>7s}"
          + f" {'gap':>7s}")
    print(f"  {'─'*96}")

    # Ridge first
    tr_mn = np.mean(ridge_train)
    te_mn = np.mean(ridge_test)
    gap = tr_mn - te_mn
    line = f"  {'Ridge':<{cw}s} {'—':>6s}"
    for v in ridge_train:
        line += f" {v:7.3f}"
    line += f" {tr_mn:7.3f}"
    for v in ridge_test:
        line += f" {v:7.3f}"
    line += f" {te_mn:7.3f}"
    line += f" {gap:+7.3f}"
    print(line)

    # Sort by test mean R²
    sorted_cfgs = sorted(results.items(),
                         key=lambda kv: np.mean(kv[1]["test"]),
                         reverse=True)

    for label, data in sorted_cfgs:
        tr = np.mean(data["train"], axis=0)
        te = np.mean(data["test"], axis=0)
        tr_mn = np.mean(tr)
        te_mn = np.mean(te)
        gap = tr_mn - te_mn
        n_par = data["params"]
        line = f"  {label:<{cw}s} {n_par:6d}"
        for v in tr:
            line += f" {v:7.3f}"
        line += f" {tr_mn:7.3f}"
        for v in te:
            line += f" {v:7.3f}"
        line += f" {te_mn:7.3f}"
        line += f" {gap:+7.3f}"
        print(line)
    print()

    # ── Plot: train-test gap vs #params ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: test R² mean vs #params
    ax = axes[0]
    for label, data in sorted_cfgs:
        te = np.mean(data["test"])
        n_par = data["params"]
        ax.scatter(n_par, te, s=50, zorder=3)
        ax.annotate(label, (n_par, te), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(np.mean(ridge_test), color="red", ls="--", lw=1.5,
               label=f"Ridge test={np.mean(ridge_test):.3f}")
    ax.set_xlabel("#parameters", fontsize=11)
    ax.set_ylabel("Test R² (mean)", fontsize=11)
    ax.set_title("Test R² vs model size", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xscale("log")

    # Right panel: train-test gap
    ax = axes[1]
    for label, data in sorted_cfgs:
        tr_mn = np.mean(data["train"])
        te_mn = np.mean(data["test"])
        n_par = data["params"]
        ax.scatter(n_par, tr_mn - te_mn, s=50, zorder=3,
                   color="tab:red" if (tr_mn - te_mn) > 0.1 else "tab:blue")
        ax.annotate(label, (n_par, tr_mn - te_mn), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(np.mean(ridge_train) - np.mean(ridge_test),
               color="red", ls="--", lw=1.5, label="Ridge gap")
    ax.set_xlabel("#parameters", fontsize=11)
    ax.set_ylabel("Train−Test R² gap (overfitting)", fontsize=11)
    ax.set_title("Overfitting diagnostic", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xscale("log")

    fig.tight_layout()
    fig_path = out_dir / "mlp_arch_sweep.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {fig_path}")

    # ── Per-mode plot: train vs test for each architecture ──────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for j, ax in enumerate(axes.flat):
        if j >= K:
            ax.set_visible(False); continue
        # Ridge reference
        ax.axhline(ridge_test[j], color="red", ls="--", lw=1.5,
                    label=f"Ridge={ridge_test[j]:.3f}")
        names, tr_vals, te_vals = [], [], []
        for label, data in sorted_cfgs:
            names.append(label)
            tr_vals.append(np.mean(data["train"], axis=0)[j])
            te_vals.append(np.mean(data["test"], axis=0)[j])
        x = np.arange(len(names))
        ax.bar(x - 0.15, tr_vals, 0.3, label="train", alpha=0.6,
               color="tab:blue")
        ax.bar(x + 0.15, te_vals, 0.3, label="test", alpha=0.6,
               color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6.5)
        ax.set_title(f"a{j+1}", fontsize=12)
        ax.set_ylabel("R²")
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Train vs Test R² per mode per architecture", fontsize=13)
    fig.tight_layout()
    fig_path2 = out_dir / "mlp_per_mode.png"
    fig.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {fig_path2}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
