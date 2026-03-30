#!/usr/bin/env python3
"""
Predict the *next-frame* activity of a single **masked neuron** from the
lagged activity of the remaining neurons.  Two model families:

  ridge-linear  – RidgeCV (L2-regularised linear regression)
  ridge-mlp     – 2-hidden-layer MLP backbone + RidgeCV readout

Two causal modes controlled by ``--causal``:

  inclusive  (default)
      The feature matrix includes activity of the *other* (non-masked)
      neurons at time *t+1* (the same frame as the prediction target)
      plus lags back to *t+1 − L + 1*.  The masked neuron is excluded
      from features at ALL lags.

  strict
      Only activity of the *other* (non-masked) neurons up to time *t*
      is used (one frame behind the target).  Lags go back to
      *t − L + 1*.  The masked neuron is excluded at ALL lags.

Evaluation: 5-fold contiguous temporal CV (every frame predicted
exactly once out-of-fold).

Usage
-----
    # predict AVAL; other neurons include time t
    python scripts/masked_neuron_prediction.py \
        --dataset_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
        --neuron AVAR --causal inclusive --device cuda

    # predict AVAR; fully causal (only t-1 and earlier)
    python scripts/masked_neuron_prediction.py \
        --neuron AVAR --causal strict

    # predict all neurons one at a time (leave-one-out scan)
    python scripts/masked_neuron_prediction.py --neuron ALL
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# ── defaults ─────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
_DEFAULT_DATASET_DIR = _ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2"
_DEFAULT_OUT_DIR = _ROOT / "output_plots/masked_neuron_prediction"
_RIDGE_ALPHAS = np.logspace(-2, 8, 60)

# ── data loading ─────────────────────────────────────────────────────────────


def _load_worm(h5_path: Path) -> Optional[dict]:
    """Return dict(u, labels) or None."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "stage1/u_mean" not in f or "gcamp/neuron_labels" not in f:
                return None
            u = np.array(f["stage1/u_mean"], dtype=np.float32)
            labels = f["gcamp/neuron_labels"][:]
            # try to read dt
            dt = 0.6
            if "timing/timestamp_confocal" in f:
                ts = f["timing/timestamp_confocal"][:]
                if len(ts) > 1:
                    dt = float(np.median(np.diff(ts)))
    except Exception as exc:
        print(f"  [warn] {h5_path.name}: {exc}")
        return None

    # ensure (T, N)
    if u.ndim != 2:
        return None
    if u.shape[0] < 400 and u.shape[1] >= 400:
        u = u.T

    labels = [lb.decode() if isinstance(lb, bytes) else str(lb) for lb in labels]
    T, N = u.shape
    if len(labels) != N:
        return None

    # drop all-NaN neurons
    keep = np.any(np.isfinite(u), axis=0)
    u = u[:, keep]
    labels = [l for l, k in zip(labels, keep) if k]

    # interpolate sporadic NaN
    for j in range(u.shape[1]):
        nans = ~np.isfinite(u[:, j])
        if nans.any() and not nans.all():
            good = np.where(~nans)[0]
            u[:, j] = np.interp(np.arange(u.shape[0]), good, u[good, j])

    return {"u": u, "labels": labels, "name": h5_path.stem, "dt": dt}


# ── feature engineering ──────────────────────────────────────────────────────


def _build_features(
    u: np.ndarray,           # (T, N)
    target_idx: int,         # column index of masked neuron
    n_lags: int,             # number of lag frames
    causal_mode: str,        # "inclusive" | "strict"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) pair.

    Target: u[t+1, target_idx]   (next-frame activity of masked neuron)

    Only *other* (non-masked) neurons are used as features in both modes.

    Features depend on causal_mode:

    inclusive:
        Other neurons at t+1, t, t-1, …, t+1 - n_lags + 1
        (i.e. includes the same frame as the prediction target and
        n_lags-1 frames of history).
        The masked neuron is excluded at ALL lags.

    strict:
        Other neurons at t, t-1, …, t - n_lags + 1
        (i.e. one step behind the target; nothing from frame t+1).
        The masked neuron is excluded at ALL lags.
    """
    T, N = u.shape

    # Mask out the target neuron – used in both modes
    feature_cols = np.ones(N, dtype=bool)
    feature_cols[target_idx] = False
    N_feat = N - 1

    if causal_mode == "inclusive":
        # For output row i the prediction target is frame  s = i + n_lags.
        # Features: other neurons at s, s-1, …, s - n_lags + 1
        #   lag=0  → frame s   = i + n_lags
        #   lag=k  → frame s-k = i + n_lags - k
        # We need s <= T-1  ⟹  i <= T - 1 - n_lags
        #   and  s - n_lags + 1 >= 0  ⟹  i >= 0
        T_out = T - n_lags
        if T_out <= 0:
            raise ValueError(f"n_lags={n_lags} too large for T={T}")

        blocks = []
        for lag in range(n_lags):
            start = n_lags - lag           # first original-u row for this lag
            block = u[start : start + T_out, :][:, feature_cols]
            blocks.append(block)
        X = np.concatenate(blocks, axis=1).astype(np.float32)  # (T_out, N_feat * n_lags)

        # Target: u[s, target_idx] = u[i + n_lags, target_idx]
        y = u[n_lags : n_lags + T_out, target_idx].astype(np.float32)

    elif causal_mode == "strict":
        # For output row i the prediction target is frame  s = i + n_lags.
        # Features: other neurons at s-1, s-2, …, s - n_lags
        #   lag=1  → frame s-1 = i + n_lags - 1
        #   lag=k  → frame s-k = i + n_lags - k    (k = 1 … n_lags)
        # We need s <= T-1  ⟹  i <= T - 1 - n_lags
        #   and  s - n_lags >= 0  ⟹  i >= 0
        T_out = T - n_lags
        if T_out <= 0:
            raise ValueError(f"n_lags={n_lags} too large for T={T}")

        blocks = []
        for lag in range(1, n_lags + 1):
            start = n_lags - lag
            block = u[start : start + T_out, :][:, feature_cols]
            blocks.append(block)
        X = np.concatenate(blocks, axis=1).astype(np.float32)  # (T_out, N_feat * n_lags)

        # Target: u[s, target_idx] = u[i + n_lags, target_idx]
        y = u[n_lags : n_lags + T_out, target_idx].astype(np.float32)

    else:
        raise ValueError(f"Unknown causal_mode={causal_mode!r}")

    return X, y


# ── fold generation ──────────────────────────────────────────────────────────


def _make_folds(T: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    sizes = np.full(n_folds, T // n_folds, dtype=int)
    sizes[: T % n_folds] += 1
    folds, cur = [], 0
    for s in sizes:
        te = np.arange(cur, cur + s)
        tr = np.concatenate([np.arange(0, cur), np.arange(cur + s, T)])
        folds.append((tr, te))
        cur += s
    return folds


def _inner_split(tr_idx: np.ndarray, frac: float = 0.2):
    n_va = max(1, int(len(tr_idx) * frac))
    return tr_idx[:-n_va], tr_idx[-n_va:]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _zscore(Xtr: np.ndarray):
    mu = Xtr.mean(0, keepdims=True)
    std = Xtr.std(0, keepdims=True).clip(1e-8)
    return mu, std


# ── Ridge-linear ─────────────────────────────────────────────────────────────


def _run_ridge_linear(X, y, n_folds, seed=0):
    from sklearn.linear_model import RidgeCV

    T = X.shape[0]
    folds = _make_folds(T, n_folds)
    pred = np.zeros(T, dtype=np.float32)

    for tr_idx, te_idx in folds:
        ridge = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
        ridge.fit(X[tr_idx], y[tr_idx])
        pred[te_idx] = ridge.predict(X[te_idx]).astype(np.float32)

    return pred, _r2(y, pred)


# ── Ridge-MLP ───────────────────────────────────────────────────────────────


def _make_mlp(in_dim, hidden, out_dim, dropout=0.0):
    import torch.nn as nn
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def _train_mlp(
    net, Xtr, Ytr, Xva, Yva, *,
    epochs, lr, batch_size, weight_decay, patience, device,
):
    import torch
    import torch.nn as nn
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    n = Xtr.shape[0]
    best_val, stale, best_state = float("inf"), 0, None

    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = crit(net(Xtr[idx]), Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            vloss = crit(net(Xva), Yva).item()

        if vloss < best_val:
            best_val, stale = vloss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1

        if patience and stale >= patience:
            break

    if best_state is not None:
        net.load_state_dict(best_state)


def _run_ridge_mlp(X, y, n_folds, args):
    import torch
    import torch.nn as nn
    from sklearn.linear_model import RidgeCV

    device = torch.device(args.device)
    T = X.shape[0]
    folds = _make_folds(T, n_folds)
    pred = np.zeros(T, dtype=np.float32)

    for fi, (tr_idx, te_idx) in enumerate(folds):
        tr_inner, va_inner = _inner_split(tr_idx)
        Xtr, Ytr = X[tr_inner], y[tr_inner]
        Xva, Yva = X[va_inner], y[va_inner]
        Xte = X[te_idx]

        mu_x, std_x = _zscore(Xtr)
        mu_y, std_y = _zscore(Ytr.reshape(-1, 1))

        to_t = lambda a: torch.from_numpy(a).to(device)
        Xtr_z = to_t(((Xtr - mu_x) / std_x).astype(np.float32))
        Ytr_z = to_t(((Ytr.reshape(-1, 1) - mu_y) / std_y).astype(np.float32))
        Xva_z = to_t(((Xva - mu_x) / std_x).astype(np.float32))
        Yva_z = to_t(((Yva.reshape(-1, 1) - mu_y) / std_y).astype(np.float32))

        torch.manual_seed(args.seed + fi)
        net = _make_mlp(X.shape[1], args.hidden, 1, args.dropout).to(device)
        _train_mlp(
            net, Xtr_z, Ytr_z, Xva_z, Yva_z,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            weight_decay=args.weight_decay, patience=args.patience,
            device=device,
        )

        # Extract backbone features (everything before last Linear)
        net.eval()
        backbone = nn.Sequential(*list(net.children())[:-1])

        X_ridge_z = to_t(((X[tr_idx] - mu_x) / std_x).astype(np.float32))
        Xte_z = to_t(((Xte - mu_x) / std_x).astype(np.float32))

        with torch.no_grad():
            feat_tr = backbone(X_ridge_z).cpu().numpy()
            feat_te = backbone(Xte_z).cpu().numpy()

        ridge = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
        ridge.fit(feat_tr, y[tr_idx])
        pred[te_idx] = ridge.predict(feat_te).astype(np.float32)

    return pred, _r2(y, pred)


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════


def _plot_single_neuron(
    out_dir: Path,
    neuron_name: str,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],   # model_name -> predictions
    r2s: dict[str, float],
    causal_mode: str,
    dt: float,
):
    """Plots for a single masked neuron (across both models)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Safety: ensure output directory exists (handles relative-path edge cases)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    T = len(y_true)
    t_sec = np.arange(T) * dt

    tag = f"causal={causal_mode}"
    models = list(preds.keys())

    # ── 1. Time-series overlay ───────────────────────────────────────────
    fig, axes = plt.subplots(len(models), 1,
                             figsize=(14, 3.5 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        ax.plot(t_sec, y_true, lw=0.8, label="ground truth", color="#2c3e50")
        ax.plot(t_sec, preds[m], lw=0.8, label=f"{m}  R²={r2s[m]:.3f}",
                alpha=0.85, color="#e74c3c" if "mlp" in m else "#3498db")
        ax.set_ylabel("Activity")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{neuron_name} — {m}  ({tag})")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{neuron_name}_timeseries.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── 2. Scatter pred vs actual ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(models),
                             figsize=(5.5 * len(models), 5), squeeze=False)
    for ai, m in enumerate(models):
        ax = axes[0, ai]
        ax.scatter(y_true, preds[m], s=2, alpha=0.25, rasterized=True)
        lo = min(y_true.min(), preds[m].min())
        hi = max(y_true.max(), preds[m].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{m}  R²={r2s[m]:.3f}")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"{neuron_name}  ({tag})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{neuron_name}_scatter.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── 3. Residual histogram ────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(models),
                             figsize=(5.5 * len(models), 4), squeeze=False)
    for ai, m in enumerate(models):
        ax = axes[0, ai]
        res = y_true - preds[m]
        ax.hist(res, bins=60, density=True, alpha=0.7, edgecolor="k", lw=0.3)
        ax.axvline(0, color="r", ls="--", lw=0.8)
        ax.set_title(f"{m}  μ={res.mean():.3f}  σ={res.std():.3f}")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Density")
    fig.suptitle(f"{neuron_name} residuals  ({tag})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{neuron_name}_residuals.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── 4. Zoomed time-series (first 200 frames) ────────────────────────
    zoom = min(200, T)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_sec[:zoom], y_true[:zoom], lw=1.2, label="ground truth",
            color="#2c3e50")
    colors = {"ridge-linear": "#3498db", "ridge-mlp": "#e74c3c"}
    for m in models:
        ax.plot(t_sec[:zoom], preds[m][:zoom], lw=1.0,
                label=f"{m}  R²={r2s[m]:.3f}", alpha=0.85,
                color=colors.get(m, "grey"))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity")
    ax.set_title(f"{neuron_name} — first {zoom} frames  ({tag})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{neuron_name}_zoom.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_summary(
    out_dir: Path,
    results: list[dict],       # per-neuron-per-worm records
    causal_mode: str,
    n_folds: int,
):
    """Summary plots across all neurons / worms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = f"causal={causal_mode}, {n_folds}-fold CV"
    models = sorted({m for r in results for m in r["r2"]})

    # ── per-neuron R² bar chart (averaged over worms) ────────────────────
    neuron_names = sorted({r["neuron"] for r in results})
    neuron_r2 = {m: [] for m in models}
    for nn_ in neuron_names:
        for m in models:
            vals = [r["r2"][m] for r in results if r["neuron"] == nn_ and m in r["r2"]]
            neuron_r2[m].append(float(np.mean(vals)) if vals else 0.0)

    # sort by ridge-linear R²
    sort_key = "ridge-linear" if "ridge-linear" in models else models[0]
    order = np.argsort(neuron_r2[sort_key])[::-1]

    fig, ax = plt.subplots(figsize=(max(10, len(neuron_names) * 0.5), 5))
    bar_w = 0.35
    for mi, m in enumerate(models):
        vals = np.array(neuron_r2[m])
        xs = np.arange(len(neuron_names)) + mi * bar_w - bar_w / 2
        color = "#3498db" if "linear" in m else "#e74c3c"
        ax.bar(xs, vals[order], width=bar_w, label=m, color=color,
               edgecolor="k", lw=0.2, alpha=0.85)
    ax.set_xticks(range(len(neuron_names)))
    ax.set_xticklabels([neuron_names[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel(f"R²  ({tag})")
    ax.set_title(f"Per-neuron mean R²  ({tag})")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "summary_r2_per_neuron.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── per-worm R² bar chart (averaged over neurons) ────────────────────
    worm_names = sorted({r["worm"] for r in results})
    worm_r2 = {m: [] for m in models}
    for wn in worm_names:
        for m in models:
            vals = [r["r2"][m] for r in results if r["worm"] == wn and m in r["r2"]]
            worm_r2[m].append(float(np.mean(vals)) if vals else 0.0)

    order_w = np.argsort(worm_r2[sort_key])[::-1]
    fig, ax = plt.subplots(figsize=(max(10, len(worm_names) * 0.5), 5))
    for mi, m in enumerate(models):
        vals = np.array(worm_r2[m])
        xs = np.arange(len(worm_names)) + mi * bar_w - bar_w / 2
        color = "#3498db" if "linear" in m else "#e74c3c"
        ax.bar(xs, vals[order_w], width=bar_w, label=m, color=color,
               edgecolor="k", lw=0.2, alpha=0.85)
    ax.set_xticks(range(len(worm_names)))
    ax.set_xticklabels([worm_names[i] for i in order_w], rotation=90, fontsize=7)
    ax.set_ylabel(f"R²  ({tag})")
    ax.set_title(f"Per-worm mean R²  ({tag})")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "summary_r2_per_worm.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── overall bar chart (mean + median across all) ─────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    all_r2 = {m: [r["r2"][m] for r in results if m in r["r2"]] for m in models}
    x = np.arange(len(models))
    means = [float(np.mean(all_r2[m])) for m in models]
    medians = [float(np.median(all_r2[m])) for m in models]
    colors = ["#3498db" if "linear" in m else "#e74c3c" for m in models]
    ax.bar(x - 0.15, means, 0.3, label="mean", color=colors, edgecolor="k", lw=0.4)
    ax.bar(x + 0.15, medians, 0.3, label="median", color=colors, edgecolor="k",
           lw=0.4, alpha=0.5)
    for i in range(len(models)):
        ax.text(i - 0.15, means[i] + 0.005, f"{means[i]:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.15, medians[i] + 0.005, f"{medians[i]:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel(f"R²  ({tag})")
    ax.set_title(f"Overall R²  ({tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "summary_overall_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── R² distribution histogram ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(models),
                             figsize=(5.5 * len(models), 4), squeeze=False)
    for ai, m in enumerate(models):
        ax = axes[0, ai]
        vals = np.array(all_r2[m])
        ax.hist(vals, bins=30, density=True, alpha=0.7, edgecolor="k", lw=0.3,
                color="#3498db" if "linear" in m else "#e74c3c")
        ax.axvline(np.mean(vals), color="navy", ls="--", lw=1,
                   label=f"mean={np.mean(vals):.3f}")
        ax.axvline(np.median(vals), color="orange", ls=":", lw=1,
                   label=f"median={np.median(vals):.3f}")
        ax.set_title(m)
        ax.set_xlabel("R²")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle(f"R² distribution  ({tag})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_r2_distribution.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Scatter: ridge-linear vs ridge-mlp ───────────────────────────────
    if "ridge-linear" in models and "ridge-mlp" in models:
        rl = np.array([r["r2"]["ridge-linear"] for r in results])
        rm = np.array([r["r2"]["ridge-mlp"] for r in results])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(rl, rm, s=15, alpha=0.6, edgecolors="k", lw=0.3, zorder=3)
        lo = min(rl.min(), rm.min()) - 0.05
        hi = max(rl.max(), rm.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("R²  ridge-linear")
        ax.set_ylabel("R²  ridge-mlp")
        ax.set_title(f"Ridge-linear vs Ridge-MLP  ({tag})")
        ax.set_aspect("equal", adjustable="datalim")
        frac_above = float((rm > rl).mean())
        ax.text(0.05, 0.95, f"MLP wins {frac_above:.0%} of cases",
                transform=ax.transAxes, fontsize=10, va="top")
        fig.tight_layout()
        fig.savefig(out_dir / "summary_linear_vs_mlp.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # ── Per-worm R² distribution (violin) ────────────────────────────────
    mlp_key = "ridge-mlp" if "ridge-mlp" in models else None
    if mlp_key and len(worm_names) >= 2:
        worm_r2_lists = {}
        for wn in worm_names:
            vals = [r["r2"][mlp_key] for r in results
                    if r["worm"] == wn and mlp_key in r["r2"]]
            if vals:
                worm_r2_lists[wn] = vals
        if worm_r2_lists:
            sorted_worms = sorted(worm_r2_lists,
                                  key=lambda w: np.median(worm_r2_lists[w]),
                                  reverse=True)
            fig, ax = plt.subplots(
                figsize=(max(10, len(sorted_worms) * 0.7), 6))
            parts = ax.violinplot(
                [worm_r2_lists[w] for w in sorted_worms],
                positions=range(len(sorted_worms)),
                showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("#e74c3c")
                pc.set_alpha(0.55)
            parts["cmedians"].set_color("black")
            # overlay individual points
            for i, w in enumerate(sorted_worms):
                xs = np.random.default_rng(0).normal(i, 0.06,
                                                     len(worm_r2_lists[w]))
                ax.scatter(xs, worm_r2_lists[w], s=10, alpha=0.5,
                           color="#2c3e50", zorder=3)
            ax.set_xticks(range(len(sorted_worms)))
            ax.set_xticklabels(sorted_worms, rotation=90, fontsize=7)
            ax.set_ylabel(f"R²  {mlp_key}")
            ax.set_title(f"Per-worm {mlp_key} R² distribution  ({tag})")
            ax.axhline(0, color="k", lw=0.5, ls="--")
            # annotate counts
            for i, w in enumerate(sorted_worms):
                ax.text(i, ax.get_ylim()[0] - 0.02,
                        f"n={len(worm_r2_lists[w])}",
                        ha="center", fontsize=6, color="grey")
            fig.tight_layout()
            fig.savefig(out_dir / "summary_r2_per_worm_violin.png", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)

    # ── Paired difference histogram ──────────────────────────────────────
    if "ridge-linear" in models and "ridge-mlp" in models:
        diff = rm - rl
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(diff, bins=40, density=True, alpha=0.7, edgecolor="k", lw=0.3,
                color="#9b59b6")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.axvline(diff.mean(), color="red", ls=":", lw=1.2,
                   label=f"mean Δ={diff.mean():.4f}")
        ax.set_xlabel("R²(MLP) − R²(linear)")
        ax.set_ylabel("Density")
        ax.set_title(f"MLP improvement over linear  ({tag})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "summary_mlp_improvement.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    pa = argparse.ArgumentParser(
        description="Predict masked neuron next-frame activity from lagged population data")
    pa.add_argument("--dataset_dir", type=Path, default=_DEFAULT_DATASET_DIR)
    pa.add_argument("--out_dir", type=Path, default=_DEFAULT_OUT_DIR)
    pa.add_argument("--neuron", type=str, nargs="+", default=["AVAR"],
                    help="Neuron name(s) to mask, or 'ALL' for full LOO scan")
    pa.add_argument("--causal", type=str, default="inclusive",
                    choices=["inclusive", "strict"],
                    help="inclusive = other neurons at time t; "
                         "strict = only t-1 and earlier")
    pa.add_argument("--lag_sec", type=float, default=10.0,
                    help="Lag window in seconds (default 10)")
    pa.add_argument("--n_folds", type=int, default=5)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--max_worms", type=int, default=0,
                    help="Max worms to load (0 = all)")
    # MLP hyper-parameters
    pa.add_argument("--hidden", type=int, default=128)
    pa.add_argument("--epochs", type=int, default=300)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--weight_decay", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=30)
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--skip_mlp", action="store_true",
                    help="Only run ridge-linear (faster)")
    pa.add_argument("--skip_neurons", type=str, nargs="+", default=[],
                    help="Neuron names to skip (for resuming partial runs)")
    args = pa.parse_args()

    np.random.seed(args.seed)

    # ── load worms ───────────────────────────────────────────────────────
    h5_files = sorted(args.dataset_dir.glob("*.h5"))
    print(f"H5 files found: {len(h5_files)}")

    worms: list[dict] = []
    for h5p in h5_files:
        w = _load_worm(h5p)
        if w is None:
            print(f"  [skip] {h5p.name}")
            continue
        worms.append(w)
        print(f"  {h5p.name}: T={w['u'].shape[0]}, N={w['u'].shape[1]}, "
              f"dt={w['dt']:.3f}s")
        if args.max_worms and len(worms) >= args.max_worms:
            break

    if not worms:
        sys.exit("No valid worms found.")

    # ── determine neurons to mask ────────────────────────────────────────
    if args.neuron == ["ALL"]:
        # Union of all labels across worms
        all_labels = set()
        for w in worms:
            all_labels.update(w["labels"])
        target_neurons = sorted(all_labels)
        print(f"\nLOO scan: {len(target_neurons)} unique neurons")
    elif args.neuron == ["MOTOR_CONTROL"]:
        mc_file = _ROOT / "data/used/masks+motor neurons/motor_neurons_with_control.txt"
        mc_names = {l.strip() for l in mc_file.read_text().splitlines() if l.strip()}
        # Intersect with neurons actually present in at least one worm
        all_labels = set()
        for w in worms:
            all_labels.update(w["labels"])
        target_neurons = sorted(mc_names & all_labels)
        print(f"\nMOTOR_CONTROL: {len(mc_names)} in list, "
              f"{len(target_neurons)} present in data")
    else:
        target_neurons = args.neuron
        print(f"\nTarget neurons: {target_neurons}")

    # Filter out skipped neurons
    if args.skip_neurons:
        before = len(target_neurons)
        target_neurons = [n for n in target_neurons if n not in args.skip_neurons]
        print(f"\nSkipping {before - len(target_neurons)} neurons, "
              f"{len(target_neurons)} remaining")

    print(f"Causal mode: {args.causal}")
    print(f"Lag window: {args.lag_sec:.1f} s")
    print(f"Models: ridge-linear" + ("" if args.skip_mlp else " + ridge-mlp"))
    print(f"Evaluation: {args.n_folds}-fold contiguous temporal CV")

    # ── run ──────────────────────────────────────────────────────────────
    args.out_dir = args.out_dir.resolve()  # ensure absolute path
    out_base = args.out_dir / args.causal
    out_base.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    t0_total = time.time()

    for ni, neuron_name in enumerate(target_neurons):
        print(f"\n{'─'*60}")
        print(f"  Neuron {ni+1}/{len(target_neurons)}: {neuron_name}")
        print(f"{'─'*60}")

        neuron_out = out_base / neuron_name
        neuron_out.mkdir(parents=True, exist_ok=True)

        for wi, w in enumerate(worms):
            labels = w["labels"]
            if neuron_name not in labels:
                continue

            target_idx = labels.index(neuron_name)
            u = w["u"]
            dt = w["dt"]
            n_lags = max(1, int(round(args.lag_sec / dt)))

            print(f"  {w['name']}: T={u.shape[0]}, N={u.shape[1]}, "
                  f"n_lags={n_lags} ({n_lags*dt:.1f}s), target_idx={target_idx}")

            X, y = _build_features(u, target_idx, n_lags, args.causal)
            print(f"    Features: X={X.shape}, y={y.shape}")

            preds, r2s = {}, {}

            # Ridge-linear
            t0 = time.time()
            pred_rl, r2_rl = _run_ridge_linear(X, y, args.n_folds)
            dt_rl = time.time() - t0
            preds["ridge-linear"] = pred_rl
            r2s["ridge-linear"] = r2_rl
            print(f"    ridge-linear  R²={r2_rl:.4f}  ({dt_rl:.1f}s)")

            # Ridge-MLP
            if not args.skip_mlp:
                t0 = time.time()
                pred_rm, r2_rm = _run_ridge_mlp(X, y, args.n_folds, args)
                dt_rm = time.time() - t0
                preds["ridge-mlp"] = pred_rm
                r2s["ridge-mlp"] = r2_rm
                print(f"    ridge-mlp     R²={r2_rm:.4f}  ({dt_rm:.1f}s)")

            all_results.append({
                "neuron": neuron_name, "worm": w["name"],
                "T": len(y), "N": u.shape[1], "n_lags": n_lags,
                "r2": r2s,
            })

            # Per-neuron-per-worm plots
            try:
                _plot_single_neuron(
                    neuron_out, f"{neuron_name}_{w['name']}",
                    y, preds, r2s, args.causal, dt,
                )
            except Exception as exc:
                print(f"    [WARN] plot failed: {exc}")

    elapsed = time.time() - t0_total
    print(f"\n{'═'*60}")
    print(f"  Done.  {len(all_results)} (neuron, worm) pairs  ({elapsed:.1f}s)")
    print(f"{'═'*60}")

    if not all_results:
        print("No results to summarize.")
        return

    # ── summary ──────────────────────────────────────────────────────────
    models = sorted({m for r in all_results for m in r["r2"]})
    for m in models:
        vals = [r["r2"][m] for r in all_results if m in r["r2"]]
        print(f"  {m:20s}  mean R²={np.mean(vals):.4f}  "
              f"median R²={np.median(vals):.4f}  (n={len(vals)})")

    # Save JSON
    summary = {
        "causal_mode": args.causal,
        "lag_sec": args.lag_sec,
        "n_folds": args.n_folds,
        "results": all_results,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (out_base / "results.json").write_text(json.dumps(summary, indent=2))

    # Summary plots
    _plot_summary(out_base, all_results, args.causal, args.n_folds)
    print(f"\n  Plots saved to {out_base}")


if __name__ == "__main__":
    main()
