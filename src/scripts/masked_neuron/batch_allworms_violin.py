#!/usr/bin/env python3
"""
Batch ridge vs MLP masked-neuron prediction across ALL worms.
Fixed lag = N frames (default 10), strict causal, 5-fold CV.
Reports both R² and Pearson correlation per neuron.
Produces per-worm + cross-worm summary violin plots.

Speed optimisations (same model, same CV, same predictions):
  1. Precompute full lagged matrix once per worm, slice columns per neuron.
  2. Ridge: joblib-parallel across neurons.
  3. MLP: full-batch training (dataset is ~1600 rows — mini-batching is pure
     overhead).  Reduced per-fold Python/CUDA launch cost.

Usage
-----
    python -m scripts.masked_neuron.batch_allworms_violin --n_lags 10 --device cuda
    python -m scripts.masked_neuron.batch_allworms_violin --n_lags 10 --device cpu --max_worms 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET = _ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2"
_DEFAULT_OUT = _ROOT / "output_plots/masked_neuron_prediction/batch_allworms_violin"

# ── reuse helpers ────────────────────────────────────────────────────────────
from scripts.masked_neuron.masked_neuron_prediction import (
    _load_worm,
    _make_folds,
    _inner_split,
    _r2,
    _zscore,
    _make_mlp,
)

_RIDGE_ALPHAS_FAST = np.logspace(-4, 6, 30)   # reduced grid (vs 60 original)


# ── metrics ──────────────────────────────────────────────────────────────────

def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# ══════════════════════════════════════════════════════════════════════════════
#  Precompute full lagged matrix — build once, slice per neuron
# ══════════════════════════════════════════════════════════════════════════════

def _precompute_lagged(u: np.ndarray, n_lags: int, causal: str):
    """Build the full (T_out, N*n_lags) lagged feature matrix including ALL
    neurons.  For target neuron *i* we drop n_lags columns via _slice_features."""
    T, N = u.shape
    T_out = T - n_lags
    blocks = []
    if causal == "strict":
        for lag in range(1, n_lags + 1):
            start = n_lags - lag
            blocks.append(u[start:start + T_out])
    elif causal == "inclusive":
        for lag in range(n_lags):
            start = n_lags - lag
            blocks.append(u[start:start + T_out])
    else:
        raise ValueError(causal)
    X_full = np.concatenate(blocks, axis=1).astype(np.float32)
    Y_all = u[n_lags:n_lags + T_out].astype(np.float32)
    return X_full, Y_all


def _slice_features(X_full: np.ndarray, target_idx: int, N: int,
                    n_lags: int) -> np.ndarray:
    """Drop the target neuron's own columns from the full lagged matrix."""
    cols_to_remove = [target_idx + lag * N for lag in range(n_lags)]
    mask = np.ones(X_full.shape[1], dtype=bool)
    mask[cols_to_remove] = False
    return X_full[:, mask]


# ══════════════════════════════════════════════════════════════════════════════
#  Full-batch MLP training (no mini-batch loop → 3-5× faster)
# ══════════════════════════════════════════════════════════════════════════════

def _train_mlp_fullbatch(net, Xtr, Ytr, Xva, Yva, *,
                         epochs, lr, weight_decay, patience):
    """Full-batch Adam with early stopping.  Dropout layers still work."""
    import torch
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    crit = torch.nn.MSELoss()
    best_val, stale, best_state = float("inf"), 0, None

    for _ in range(epochs):
        net.train()
        loss = crit(net(Xtr), Ytr)
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


# ══════════════════════════════════════════════════════════════════════════════
#  Process one worm — optimised
# ══════════════════════════════════════════════════════════════════════════════

def _process_worm(worm, n_lags, causal, n_folds, mlp_kwargs, n_jobs=4):
    """
    Return list of dicts, one per neuron.

    Optimisations vs v1:
      - Full lagged matrix precomputed once; per-neuron cost = column slice.
      - Ridge: joblib-parallel across all neurons.
      - MLP: full-batch training (1 fwd+bwd per epoch instead of ~5).
    """
    import torch
    from joblib import Parallel, delayed

    u = worm["u"]
    labels = worm["labels"]
    T, N = u.shape

    # ── 1) Precompute ────────────────────────────────────────────────────
    t_pre = time.time()
    X_full, Y_all = _precompute_lagged(u, n_lags, causal)
    T_out = X_full.shape[0]
    folds = _make_folds(T_out, n_folds)
    print(f"    precompute: X_full={X_full.shape}  ({time.time()-t_pre:.2f}s)")

    # ── 2) Ridge: parallel across neurons ────────────────────────────────
    t_ridge = time.time()

    def _ridge_one(ni):
        from sklearn.linear_model import RidgeCV
        X = _slice_features(X_full, ni, N, n_lags)
        y = Y_all[:, ni]
        pred = np.zeros(T_out, dtype=np.float32)
        for tr_idx, te_idx in folds:
            ridge = RidgeCV(alphas=_RIDGE_ALPHAS_FAST, fit_intercept=True)
            ridge.fit(X[tr_idx], y[tr_idx])
            pred[te_idx] = ridge.predict(X[te_idx]).astype(np.float32)
        return ni, pred, _r2(y, pred), _pearson_r(y, pred)

    ridge_raw = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_ridge_one)(ni) for ni in range(N))

    ridge_r2, ridge_corr = {}, {}
    for ni, pred, r2, corr in ridge_raw:
        ridge_r2[ni] = r2
        ridge_corr[ni] = corr
    dt_ridge = time.time() - t_ridge
    print(f"    ridge: {N} neurons in {dt_ridge:.1f}s "
          f"(med R²={np.median(list(ridge_r2.values())):.4f}, {n_jobs} jobs)")

    # ── 3) MLP: sequential, full-batch ───────────────────────────────────
    t_mlp = time.time()
    device = torch.device(mlp_kwargs["device_str"])
    to_t = lambda a: torch.from_numpy(a).to(device)

    mlp_r2, mlp_corr = {}, {}

    for ni in range(N):
        X = _slice_features(X_full, ni, N, n_lags)
        y = Y_all[:, ni]
        pred = np.zeros(T_out, dtype=np.float32)

        for fi, (tr_idx, te_idx) in enumerate(folds):
            tr_inner, va_inner = _inner_split(tr_idx)
            Xtr, Ytr = X[tr_inner], y[tr_inner]
            Xva, Yva = X[va_inner], y[va_inner]

            mu_x, std_x = _zscore(Xtr)
            mu_y = float(Ytr.mean())
            std_y = float(max(Ytr.std(), 1e-8))

            Xtr_z = to_t(((Xtr - mu_x) / std_x).astype(np.float32))
            Ytr_z = to_t(((Ytr.reshape(-1, 1) - mu_y) / std_y).astype(np.float32))
            Xva_z = to_t(((Xva - mu_x) / std_x).astype(np.float32))
            Yva_z = to_t(((Yva.reshape(-1, 1) - mu_y) / std_y).astype(np.float32))

            torch.manual_seed(mlp_kwargs["seed"] + fi)
            net = _make_mlp(X.shape[1], mlp_kwargs["hidden"], 1,
                            mlp_kwargs["dropout"]).to(device)
            _train_mlp_fullbatch(
                net, Xtr_z, Ytr_z, Xva_z, Yva_z,
                epochs=mlp_kwargs["epochs"], lr=mlp_kwargs["lr"],
                weight_decay=mlp_kwargs["weight_decay"],
                patience=mlp_kwargs["patience"],
            )

            net.eval()
            Xte_z = to_t(((X[te_idx] - mu_x) / std_x).astype(np.float32))
            with torch.no_grad():
                pred_z = net(Xte_z).cpu().numpy().ravel()
            pred[te_idx] = (pred_z * std_y + mu_y).astype(np.float32)

        mlp_r2[ni] = _r2(y, pred)
        mlp_corr[ni] = _pearson_r(y, pred)

        if (ni + 1) % 10 == 0 or ni == N - 1 or N <= 20:
            print(f"    [{ni+1:3d}/{N}] {labels[ni]:8s}  "
                  f"ridge R²={ridge_r2[ni]:+.4f} r={ridge_corr[ni]:.3f}  |  "
                  f"mlp R²={mlp_r2[ni]:+.4f} r={mlp_corr[ni]:.3f}")

    dt_mlp = time.time() - t_mlp
    print(f"    mlp: {N} neurons in {dt_mlp:.1f}s "
          f"(med R²={np.median(list(mlp_r2.values())):.4f})")

    # ── assemble records ─────────────────────────────────────────────────
    records = []
    for ni in range(N):
        records.append({
            "worm": worm["name"],
            "neuron": labels[ni],
            "neuron_idx": ni,
            "T": T_out,
            "N": N,
            "n_lags": n_lags,
            "r2_ridge": ridge_r2[ni],
            "r2_mlp": mlp_r2[ni],
            "corr_ridge": ridge_corr[ni],
            "corr_mlp": mlp_corr[ni],
        })
    return records


# ── plotting ─────────────────────────────────────────────────────────────────

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
    })
    return plt


_HIGHLIGHT_PREFIXES = ("AVA",)   # neuron prefixes to highlight with ★


def plot_per_worm_violin(records, worm_name, out_dir, causal, n_lags, n_folds,
                         highlight_prefixes=_HIGHLIGHT_PREFIXES):
    """Violin plot for a single worm: ridge vs MLP, both R² and corr.
    Neurons whose label starts with any of *highlight_prefixes* are
    overlaid with a star marker and labelled.
    """
    plt = _setup_mpl()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    neuron_names = [r["neuron"] for r in records]
    r2_ridge = np.array([r["r2_ridge"] for r in records])
    r2_mlp = np.array([r["r2_mlp"] for r in records])
    corr_ridge = np.array([r["corr_ridge"] for r in records])
    corr_mlp = np.array([r["corr_mlp"] for r in records])
    N = len(records)

    # find highlighted neuron indices
    hl_idx = [i for i, n in enumerate(neuron_names)
              if any(n.startswith(p) for p in highlight_prefixes)]

    colors = {"ridge": "#3498db", "mlp": "#e74c3c"}
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, metric, vals_r, vals_m, ylabel in [
        (axes[0], "R²", r2_ridge, r2_mlp, "R²"),
        (axes[1], "Correlation", corr_ridge, corr_mlp, "Pearson r"),
    ]:
        data = [vals_r, vals_m]
        labels = ["Ridge", "MLP 2×128"]
        cols = [colors["ridge"], colors["mlp"]]

        parts = ax.violinplot(data, positions=[0, 1], showmedians=False,
                              showextrema=False)
        for pc, c in zip(parts["bodies"], cols):
            pc.set_facecolor(c)
            pc.set_alpha(0.35)
            pc.set_edgecolor(c)

        # store jitters so we can reuse them for highlights
        all_jitter = []
        for i, (vals, c) in enumerate(zip(data, cols)):
            jitter = rng.normal(0, 0.04, size=len(vals))
            all_jitter.append(jitter)
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=12, alpha=0.45, color=c, edgecolors="0.4",
                       linewidths=0.2, zorder=3)
            med = float(np.median(vals))
            ax.plot([i - 0.22, i + 0.22], [med, med], color="k",
                    lw=2.5, zorder=4)
            ax.text(i + 0.28, med, f"{med:.3f}", va="center",
                    fontsize=10, fontweight="bold")

        # ★ highlight AVA / selected neurons ★
        for hi in hl_idx:
            for i, (vals, c) in enumerate(zip(data, cols)):
                xpos = i + all_jitter[i][hi]
                ypos = vals[hi]
                ax.scatter([xpos], [ypos], marker="*", s=120,
                           color="gold", edgecolors="k", linewidths=0.7,
                           zorder=6)
                # label only on the left violin to avoid clutter
                if i == 0:
                    ax.annotate(
                        neuron_names[hi],
                        xy=(xpos, ypos),
                        xytext=(-38, 8), textcoords="offset points",
                        fontsize=7, fontweight="bold", color="k",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"),
                    )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{metric}  (N={N} neurons)")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)

    tag = f"{causal}, {n_lags}-frame lag, {n_folds}-fold CV"
    fig.suptitle(f"{worm_name}  ({tag})", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_violin.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_summary_violin(all_records, out_dir, causal, n_lags, n_folds):
    """
    Cross-worm summary: one violin per worm, ridge + MLP side-by-side.
    Two rows: top = R², bottom = Pearson r.
    """
    plt = _setup_mpl()
    out_dir = Path(out_dir)

    import pandas as pd
    df = pd.DataFrame(all_records)
    worm_names = sorted(df["worm"].unique())
    n_worms = len(worm_names)

    colors = {"ridge": "#3498db", "mlp": "#e74c3c"}
    tag = f"{causal}, {n_lags}-frame lag, {n_folds}-fold CV"

    # ── 1) Side-by-side violin per worm (R² and corr) ────────────────────
    for metric, col_r, col_m, ylabel in [
        ("R²", "r2_ridge", "r2_mlp", "R² (cross-neuron)"),
        ("Pearson r", "corr_ridge", "corr_mlp", "Pearson r (cross-neuron)"),
    ]:
        fig, ax = plt.subplots(
            figsize=(max(8, n_worms * 1.4 + 2), 6))

        positions_r = np.arange(n_worms) * 2.5
        positions_m = positions_r + 0.8
        width = 0.7

        all_vals_r, all_vals_m = [], []

        for wi, wn in enumerate(worm_names):
            ws = df[df["worm"] == wn]
            vr = ws[col_r].values
            vm = ws[col_m].values
            all_vals_r.append(vr)
            all_vals_m.append(vm)

        # Ridge violins
        parts_r = ax.violinplot(
            all_vals_r, positions=positions_r, widths=width,
            showmedians=False, showextrema=False)
        for pc in parts_r["bodies"]:
            pc.set_facecolor(colors["ridge"])
            pc.set_alpha(0.45)
            pc.set_edgecolor(colors["ridge"])

        # MLP violins
        parts_m = ax.violinplot(
            all_vals_m, positions=positions_m, widths=width,
            showmedians=False, showextrema=False)
        for pc in parts_m["bodies"]:
            pc.set_facecolor(colors["mlp"])
            pc.set_alpha(0.45)
            pc.set_edgecolor(colors["mlp"])

        # Median bars
        for wi in range(n_worms):
            for vals, pos in [(all_vals_r[wi], positions_r[wi]),
                              (all_vals_m[wi], positions_m[wi])]:
                med = float(np.median(vals))
                ax.plot([pos - 0.25, pos + 0.25], [med, med],
                        color="k", lw=2, zorder=4)

        ax.set_xticks((positions_r + positions_m) / 2)
        ax.set_xticklabels(worm_names, rotation=70, fontsize=7, ha="right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)

        # Legend
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(facecolor=colors["ridge"], alpha=0.5, label="Ridge"),
                     Patch(facecolor=colors["mlp"], alpha=0.5, label="MLP 2×128")],
            loc="lower left", fontsize=10)

        ax.set_title(f"{metric} per worm — Ridge vs MLP  ({tag})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        safe = metric.replace("²", "2").replace(" ", "_")
        fig.savefig(out_dir / f"summary_{safe}_per_worm.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 2) Grand violin: all neurons pooled ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    rng = np.random.default_rng(42)

    for ax, metric, col_r, col_m, ylabel in [
        (axes[0], "R²", "r2_ridge", "r2_mlp", "R²"),
        (axes[1], "Pearson r", "corr_ridge", "corr_mlp", "Pearson r"),
    ]:
        vr = df[col_r].values
        vm = df[col_m].values

        parts = ax.violinplot([vr, vm], positions=[0, 1],
                              showmedians=False, showextrema=False)
        cols = [colors["ridge"], colors["mlp"]]
        for pc, c in zip(parts["bodies"], cols):
            pc.set_facecolor(c)
            pc.set_alpha(0.4)
            pc.set_edgecolor(c)

        for i, (vals, c) in enumerate(zip([vr, vm], cols)):
            jitter = rng.normal(0, 0.04, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=4, alpha=0.15, color=c, edgecolors="none",
                       zorder=3, rasterized=True)
            med = float(np.median(vals))
            mean = float(np.mean(vals))
            ax.plot([i - 0.22, i + 0.22], [med, med], color="k",
                    lw=2.5, zorder=4)
            ax.text(i + 0.28, med, f"med={med:.3f}", va="center",
                    fontsize=9, fontweight="bold")
            ax.text(i + 0.28, mean, f"μ={mean:.3f}", va="center",
                    fontsize=8, color="0.4")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ridge", "MLP 2×128"], fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)

        n_total = len(vr)
        frac = float((vm > vr).mean())
        ax.set_title(f"{metric}  (n={n_total}, MLP↑ {frac:.0%})")

    fig.suptitle(f"All worms pooled — Ridge vs MLP  ({tag})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_grand_violin.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── 3) Scatter: ridge R² vs MLP R²  &  ridge corr vs MLP corr ───────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    for ax, col_r, col_m, label in [
        (axes[0], "r2_ridge", "r2_mlp", "R²"),
        (axes[1], "corr_ridge", "corr_mlp", "Pearson r"),
    ]:
        vr = df[col_r].values
        vm = df[col_m].values
        ax.scatter(vr, vm, s=6, alpha=0.3, rasterized=True,
                   edgecolors="none", color="#2c3e50")
        lo = min(vr.min(), vm.min()) - 0.05
        hi = max(vr.max(), vm.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8, alpha=0.7)
        ax.set_xlabel(f"Ridge {label}")
        ax.set_ylabel(f"MLP {label}")
        ax.set_aspect("equal", adjustable="datalim")
        frac = float((vm > vr).mean())
        ax.text(0.05, 0.95, f"MLP wins {frac:.0%}",
                transform=ax.transAxes, fontsize=10, va="top")
        ax.set_title(f"{label}: Ridge vs MLP")

    fig.suptitle(f"Ridge vs MLP scatter  ({tag})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 4) Per-worm median bar chart (R² + corr) ────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(max(10, n_worms * 0.8), 9))

    for ax, metric, col_r, col_m, ylabel in [
        (axes[0], "R²", "r2_ridge", "r2_mlp", "Median R²"),
        (axes[1], "Pearson r", "corr_ridge", "corr_mlp", "Median Pearson r"),
    ]:
        med_r = df.groupby("worm")[col_r].median().reindex(worm_names).values
        med_m = df.groupby("worm")[col_m].median().reindex(worm_names).values
        x = np.arange(n_worms)
        w = 0.35
        ax.bar(x - w / 2, med_r, w, color=colors["ridge"], alpha=0.8,
               edgecolor="k", lw=0.3, label="Ridge")
        ax.bar(x + w / 2, med_m, w, color=colors["mlp"], alpha=0.8,
               edgecolor="k", lw=0.3, label="MLP 2×128")
        ax.set_xticks(x)
        ax.set_xticklabels(worm_names, rotation=70, fontsize=7, ha="right")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.legend(fontsize=9)
        ax.set_title(f"Per-worm median {metric}")

    fig.suptitle(f"Per-worm summary  ({tag})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_per_worm_bars.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── 5) Histogram of R² and corr for both models ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for row, (metric, col_r, col_m) in enumerate([
        ("R²", "r2_ridge", "r2_mlp"),
        ("Pearson r", "corr_ridge", "corr_mlp"),
    ]):
        for col_i, (model, col, c) in enumerate([
            ("Ridge", col_r, colors["ridge"]),
            ("MLP 2×128", col_m, colors["mlp"]),
        ]):
            ax = axes[row, col_i]
            vals = df[col].values
            ax.hist(vals, bins=50, density=True, alpha=0.7,
                    edgecolor="k", lw=0.3, color=c)
            ax.axvline(np.mean(vals), color="navy", ls="--", lw=1,
                       label=f"mean={np.mean(vals):.3f}")
            ax.axvline(np.median(vals), color="orange", ls=":", lw=1,
                       label=f"med={np.median(vals):.3f}")
            ax.set_xlabel(metric)
            ax.set_ylabel("Density")
            ax.set_title(f"{model} — {metric}")
            ax.legend(fontsize=8)

    fig.suptitle(f"Distributions  ({tag})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_histograms.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── 6) Linearity diagnostic: MLP − Ridge gap distribution ────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col_r, col_m, label in [
        (axes[0], "r2_ridge", "r2_mlp", "R²"),
        (axes[1], "corr_ridge", "corr_mlp", "Pearson r"),
    ]:
        gap = df[col_m].values - df[col_r].values
        ax.hist(gap, bins=60, density=True, alpha=0.7,
                edgecolor="k", lw=0.3, color="#9b59b6")
        med = float(np.median(gap))
        mu = float(np.mean(gap))
        frac_pos = float((gap > 0).mean())
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.axvline(med, color="orange", ls=":", lw=1.5,
                   label=f"med={med:+.4f}")
        ax.axvline(mu, color="navy", ls="--", lw=1,
                   label=f"μ={mu:+.4f}")
        ax.set_xlabel(f"MLP {label} − Ridge {label}")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} gap  (MLP > Ridge: {frac_pos:.0%})")
        ax.legend(fontsize=9)

    fig.suptitle(f"Linearity check: MLP − Ridge gap  ({tag})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_linearity_gap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── 7) Negative R² analysis: which neurons & why ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, model_name, c in [
        (axes[0], "r2_ridge", "Ridge", colors["ridge"]),
        (axes[1], "r2_mlp", "MLP", colors["mlp"]),
    ]:
        vals = df[col].values
        n_neg = int((vals < 0).sum())
        n_tot = len(vals)
        ax.hist(vals, bins=60, density=True, alpha=0.7,
                edgecolor="k", lw=0.3, color=c)
        ax.axvline(0, color="k", lw=1.2, ls="--")
        # shade the negative region
        ax.axvspan(vals.min() - 0.05, 0, alpha=0.10, color="red")
        ax.set_xlabel(f"{model_name} R²")
        ax.set_ylabel("Density")
        ax.set_title(f"{model_name}: {n_neg}/{n_tot} neurons with R²<0 "
                     f"({n_neg/n_tot:.0%})")

    fig.suptitle(f"Negative R² breakdown  ({tag})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_negative_r2.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── 8) AVA neurons: per-worm R² highlight ────────────────────────────
    ava_rows = df[df["neuron"].str.startswith("AVA")]
    if len(ava_rows) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        rng = np.random.default_rng(42)

        for ax, col_r, col_m, label in [
            (axes[0], "r2_ridge", "r2_mlp", "R²"),
            (axes[1], "corr_ridge", "corr_mlp", "Pearson r"),
        ]:
            # background: all neurons (small grey)
            ax.scatter(df[col_r].values, df[col_m].values,
                       s=5, alpha=0.15, color="0.6", edgecolors="none",
                       rasterized=True, label="all neurons")
            # AVA overlay
            for _, row in ava_rows.iterrows():
                ax.scatter([row[col_r]], [row[col_m]], marker="*",
                           s=120, color="gold", edgecolors="k",
                           linewidths=0.6, zorder=5)
                ax.annotate(f"{row['neuron']}\n({row['worm'][-5:]})",
                            xy=(row[col_r], row[col_m]),
                            xytext=(8, 5), textcoords="offset points",
                            fontsize=6, fontweight="bold")

            lo = min(df[col_r].min(), df[col_m].min()) - 0.05
            hi = max(df[col_r].max(), df[col_m].max()) + 0.05
            ax.plot([lo, hi], [lo, hi], "r--", lw=0.8, alpha=0.6)
            ax.set_xlabel(f"Ridge {label}")
            ax.set_ylabel(f"MLP {label}")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"AVA neurons — {label}")

        fig.suptitle(f"AVA neurons highlighted  ({tag})",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / "summary_AVA_highlight.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Batch ridge vs MLP on all worms, fixed frame lag, "
                    "R² + Pearson r, violin plots")
    pa.add_argument("--dataset_dir", type=Path, default=_DEFAULT_DATASET)
    pa.add_argument("--out_dir", type=Path, default=_DEFAULT_OUT)
    pa.add_argument("--n_lags", type=int, default=10,
                    help="Number of lag frames (default 10)")
    pa.add_argument("--causal", type=str, default="strict",
                    choices=["inclusive", "strict"])
    pa.add_argument("--n_folds", type=int, default=5)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--max_worms", type=int, default=0)
    pa.add_argument("--n_jobs", type=int, default=8,
                    help="Parallel workers for ridge (default 8)")
    # MLP hparams
    pa.add_argument("--hidden", type=int, default=128)
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--weight_decay", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=15)
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--replot", action="store_true",
                    help="Skip model fitting; regenerate all plots from "
                         "existing all_results.csv")
    args = pa.parse_args()

    np.random.seed(args.seed)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── replot mode ──────────────────────────────────────────────────────
    if args.replot:
        import pandas as pd
        csv_path = args.out_dir / "all_results.csv"
        if not csv_path.exists():
            sys.exit(f"No CSV found at {csv_path}")
        df = pd.read_csv(csv_path)
        all_records = df.to_dict("records")
        print(f"Replot mode: loaded {len(all_records)} records from {csv_path}")

        # per-worm violins
        worm_out = args.out_dir / "per_worm"
        for worm_name in sorted(df["worm"].unique()):
            recs = [r for r in all_records if r["worm"] == worm_name]
            try:
                plot_per_worm_violin(
                    recs, worm_name, worm_out,
                    args.causal, args.n_lags, args.n_folds)
                print(f"  → {worm_name} violin")
            except Exception as exc:
                print(f"  [WARN] {worm_name}: {exc}")

        # summary
        try:
            plot_summary_violin(all_records, args.out_dir,
                                args.causal, args.n_lags, args.n_folds)
            print(f"  Summary plots saved to {args.out_dir}")
        except Exception as exc:
            print(f"  [WARN] summary plots failed: {exc}")
            import traceback
            traceback.print_exc()

        print("Replot done.")
        return

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

    mlp_kwargs = dict(
        hidden=args.hidden, dropout=args.dropout, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, device_str=args.device, seed=args.seed,
    )

    print(f"\nSettings: n_lags={args.n_lags} frames, causal={args.causal}, "
          f"{args.n_folds}-fold CV, device={args.device}, n_jobs={args.n_jobs}")
    print(f"MLP: hidden={args.hidden}, epochs={args.epochs}, "
          f"patience={args.patience}, dropout={args.dropout}, lr={args.lr}")
    print(f"Output: {args.out_dir}\n")

    # ── run per worm ─────────────────────────────────────────────────────
    all_records: list[dict] = []
    t0_total = time.time()

    for wi, worm in enumerate(worms):
        print(f"\n{'═'*60}")
        print(f"  Worm {wi+1}/{len(worms)}: {worm['name']}  "
              f"T={worm['u'].shape[0]}  N={worm['u'].shape[1]}  "
              f"dt={worm['dt']:.3f}s")
        print(f"{'═'*60}")

        t0_w = time.time()
        records = _process_worm(
            worm, args.n_lags, args.causal, args.n_folds,
            mlp_kwargs, n_jobs=args.n_jobs)
        dt_w = time.time() - t0_w
        all_records.extend(records)

        # Per-worm violin
        worm_out = args.out_dir / "per_worm"
        try:
            plot_per_worm_violin(
                records, worm["name"], worm_out,
                args.causal, args.n_lags, args.n_folds)
            print(f"  → violin saved to {worm_out / (worm['name'] + '_violin.png')}")
        except Exception as exc:
            print(f"  [WARN] per-worm plot failed: {exc}")

        # Quick stats
        r2_r = np.array([r["r2_ridge"] for r in records])
        r2_m = np.array([r["r2_mlp"] for r in records])
        cr = np.array([r["corr_ridge"] for r in records])
        cm = np.array([r["corr_mlp"] for r in records])
        print(f"  Ridge:  med R²={np.median(r2_r):.4f}  med r={np.median(cr):.4f}")
        print(f"  MLP:    med R²={np.median(r2_m):.4f}  med r={np.median(cm):.4f}")
        print(f"  ({len(records)} neurons, {dt_w:.1f}s)")

    elapsed = time.time() - t0_total
    print(f"\n{'═'*60}")
    print(f"  TOTAL: {len(all_records)} (neuron, worm) pairs across "
          f"{len(worms)} worms  ({elapsed:.1f}s)")
    print(f"{'═'*60}")

    # ── save CSV ─────────────────────────────────────────────────────────
    csv_path = args.out_dir / "all_results.csv"
    keys = list(all_records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n  CSV saved: {csv_path}")

    # ── save JSON summary ────────────────────────────────────────────────
    r2_all_r = np.array([r["r2_ridge"] for r in all_records])
    r2_all_m = np.array([r["r2_mlp"] for r in all_records])
    c_all_r = np.array([r["corr_ridge"] for r in all_records])
    c_all_m = np.array([r["corr_mlp"] for r in all_records])

    summary = {
        "n_worms": len(worms),
        "n_records": len(all_records),
        "n_lags": args.n_lags,
        "causal": args.causal,
        "n_folds": args.n_folds,
        "elapsed_sec": round(elapsed, 1),
        "ridge": {
            "mean_r2": float(np.mean(r2_all_r)),
            "median_r2": float(np.median(r2_all_r)),
            "mean_corr": float(np.mean(c_all_r)),
            "median_corr": float(np.median(c_all_r)),
        },
        "mlp": {
            "mean_r2": float(np.mean(r2_all_m)),
            "median_r2": float(np.median(r2_all_m)),
            "mean_corr": float(np.mean(c_all_m)),
            "median_corr": float(np.median(c_all_m)),
        },
        "mlp_win_frac_r2": float((r2_all_m > r2_all_r).mean()),
        "mlp_win_frac_corr": float((c_all_m > c_all_r).mean()),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    json_path = args.out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"  JSON saved: {json_path}")

    # ── global stats ─────────────────────────────────────────────────────
    print(f"\n  {'─'*40}")
    print(f"  Ridge:  mean R²={summary['ridge']['mean_r2']:.4f}  "
          f"med R²={summary['ridge']['median_r2']:.4f}  "
          f"mean r={summary['ridge']['mean_corr']:.4f}  "
          f"med r={summary['ridge']['median_corr']:.4f}")
    print(f"  MLP:    mean R²={summary['mlp']['mean_r2']:.4f}  "
          f"med R²={summary['mlp']['median_r2']:.4f}  "
          f"mean r={summary['mlp']['mean_corr']:.4f}  "
          f"med r={summary['mlp']['median_corr']:.4f}")
    print(f"  MLP > Ridge: {summary['mlp_win_frac_r2']:.0%} (R²)  "
          f"{summary['mlp_win_frac_corr']:.0%} (corr)")
    print(f"  {'─'*40}")

    # ── summary plots ────────────────────────────────────────────────────
    print("\nGenerating summary plots …")
    try:
        plot_summary_violin(all_records, args.out_dir,
                            args.causal, args.n_lags, args.n_folds)
        print(f"  Summary plots saved to {args.out_dir}")
    except Exception as exc:
        print(f"  [WARN] summary plots failed: {exc}")
        import traceback
        traceback.print_exc()

    print(f"\nDone. All outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
