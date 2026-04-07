#!/usr/bin/env python3
"""
Sweep the lag window for masked-neuron prediction and visualise R² vs lag.

Models: ridge-linear (fast) + MLP (optional, GPU-accelerated).

Speed strategy
--------------
1. Precompute features for the maximum lag; smaller lags are column slices
   (same samples, fair comparison).
2. PCA capping (default 50 components) keeps effective dimensionality constant
   across all lag values — avoids the **curse-of-dimensionality collapse**
   that destroyed R² when n_features > n_samples (~5 s lag) in the original
   ridge-only sweep.
3. joblib parallelism across neurons (ridge).
4. Reduced RidgeCV alpha grid (30 values).

Usage
-----
    # single worm, ridge only (fast, ~2 min)
    python -m scripts.masked_neuron.lag_sweep \\
        --worm 2022-08-02-01 --causal inclusive --n_jobs 8

    # single worm, ridge + MLP, narrower lag range (fast enough, ~10 min)
    python -m scripts.masked_neuron.lag_sweep \\
        --worm 2022-08-02-01 --causal inclusive --n_jobs 8 \\
        --models ridge mlp --device cuda \\
        --lag_secs 0.6 1.2 2 3 5 8

    # custom PCA budget
    python -m scripts.masked_neuron.lag_sweep \\
        --worm 2022-08-02-01 --max_features 100 --n_jobs 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── paths & constants ────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET = _ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2"
_DEFAULT_OUT = _ROOT / "output_plots/masked_neuron_prediction/lag_sweep"
_ALPHAS = np.logspace(-4, 6, 30)  # reduced grid for speed

# ── reuse data-loading + utilities from the main prediction script ───────────

from scripts.masked_neuron.masked_neuron_prediction import (
    _load_worm,
    _make_folds,
    _inner_split,
    _zscore,
    _make_mlp,
    _train_mlp,
    _r2,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Core: precompute features & sweep
# ══════════════════════════════════════════════════════════════════════════════


def _build_sweep_features(
    u: np.ndarray,        # (T, N)
    target_idx: int,
    max_lag_frames: int,
    causal_mode: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build the feature matrix for ``max_lag_frames`` lags.

    For smaller lag values *k*, the first ``k * N_feat`` columns give the
    correct features (all evaluated on the same ``T_out = T − max_lag``
    time points, ensuring a fair comparison).

    Returns (X_full, y, N_feat).
    """
    T, N = u.shape
    feat_mask = np.ones(N, dtype=bool)
    feat_mask[target_idx] = False
    N_feat = int(feat_mask.sum())
    T_out = T - max_lag_frames
    if T_out < 10:
        raise ValueError(
            f"max_lag_frames={max_lag_frames} leaves only {T_out} samples "
            f"(T={T})"
        )

    if causal_mode == "inclusive":
        # lag 0 = same frame as target, lag k = k frames before target
        blocks = []
        for lag in range(max_lag_frames):
            start = max_lag_frames - lag
            blocks.append(u[start : start + T_out][:, feat_mask])
        X_full = np.concatenate(blocks, axis=1).astype(np.float32)
        y = u[max_lag_frames : max_lag_frames + T_out, target_idx].astype(
            np.float32
        )

    elif causal_mode == "strict":
        # lag 1 = one frame before target, lag k = k frames before target
        blocks = []
        for lag in range(1, max_lag_frames + 1):
            start = max_lag_frames - lag
            blocks.append(u[start : start + T_out][:, feat_mask])
        X_full = np.concatenate(blocks, axis=1).astype(np.float32)
        y = u[max_lag_frames : max_lag_frames + T_out, target_idx].astype(
            np.float32
        )
    else:
        raise ValueError(f"Unknown causal_mode={causal_mode!r}")

    return X_full, y, N_feat


def _sweep_one_neuron(
    u: np.ndarray,
    target_idx: int,
    lag_frame_values: list[int],
    max_lag_frames: int,
    causal_mode: str,
    n_folds: int,
    max_features: int = 0,
    models: list[str] | None = None,
    mlp_args: dict | None = None,
) -> dict[int, dict[str, float]]:
    """
    Return {lag_frames: {model: R²}} for one neuron, sweeping all lag values.

    If ``max_features > 0``, apply PCA (fit on train) to cap the number
    of features — this keeps effective dimensionality constant across
    all lag values and avoids the curse-of-dimensionality collapse that
    occurs when n_features ≫ n_samples.
    """
    from sklearn.linear_model import RidgeCV

    if models is None:
        models = ["ridge"]

    X_full, y, N_feat = _build_sweep_features(
        u, target_idx, max_lag_frames, causal_mode
    )
    folds = _make_folds(len(y), n_folds)
    results: dict[int, dict[str, float]] = {}

    need_pca = max_features > 0
    if need_pca:
        from sklearn.decomposition import PCA

    run_mlp = "mlp" in models and mlp_args is not None

    for lag_f in lag_frame_values:
        n_cols = lag_f * N_feat
        X_lag = X_full[:, :n_cols]
        use_pca = need_pca and n_cols > max_features

        lag_results: dict[str, float] = {}

        # ── Ridge ────────────────────────────────────────────────
        if "ridge" in models:
            pred_ridge = np.zeros(len(y), dtype=np.float32)
            for tr_idx, te_idx in folds:
                X_tr = X_lag[tr_idx]
                X_te = X_lag[te_idx]

                if use_pca:
                    pca = PCA(n_components=max_features, random_state=0)
                    X_tr = pca.fit_transform(X_tr)
                    X_te = pca.transform(X_te)

                ridge = RidgeCV(alphas=_ALPHAS, fit_intercept=True)
                ridge.fit(X_tr, y[tr_idx])
                pred_ridge[te_idx] = ridge.predict(X_te).astype(np.float32)
            lag_results["ridge"] = _r2(y, pred_ridge)

        # ── MLP ──────────────────────────────────────────────────
        if run_mlp:
            import torch

            device = torch.device(mlp_args.get("device", "cpu"))
            to_t = lambda a: torch.from_numpy(a).to(device)
            pred_mlp = np.zeros(len(y), dtype=np.float32)

            for fi, (tr_idx, te_idx) in enumerate(folds):
                X_tr_full = X_lag[tr_idx]
                X_te_full = X_lag[te_idx]

                # PCA fit on full outer train — same transform for
                # inner splits AND test set (single feature space).
                if use_pca:
                    pca = PCA(n_components=max_features, random_state=0)
                    X_tr_full = pca.fit_transform(X_tr_full)
                    X_te_full = pca.transform(X_te_full)

                # Inner split: slice the already-PCA-transformed outer
                # train data so everything stays in the same space.
                n_va = max(1, int(len(tr_idx) * 0.2))
                Xtr_np = X_tr_full[:-n_va]
                Xva_np = X_tr_full[-n_va:]
                tr_inner, va_inner = _inner_split(tr_idx)
                Ytr_np, Yva_np = y[tr_inner], y[va_inner]

                mu_x, std_x = _zscore(Xtr_np)
                mu_y = float(Ytr_np.mean())
                std_y = float(max(Ytr_np.std(), 1e-8))

                Xtr_z = to_t(((Xtr_np - mu_x) / std_x).astype(np.float32))
                Ytr_z = to_t(((Ytr_np.reshape(-1, 1) - mu_y) / std_y).astype(np.float32))
                Xva_z = to_t(((Xva_np - mu_x) / std_x).astype(np.float32))
                Yva_z = to_t(((Yva_np.reshape(-1, 1) - mu_y) / std_y).astype(np.float32))

                in_dim = Xtr_z.shape[1]
                torch.manual_seed(mlp_args.get("seed", 0) + fi)
                net = _make_mlp(
                    in_dim,
                    mlp_args.get("hidden", 128),
                    1,
                    mlp_args.get("dropout", 0.1),
                ).to(device)
                _train_mlp(
                    net, Xtr_z, Ytr_z, Xva_z, Yva_z,
                    epochs=mlp_args.get("epochs", 200),
                    lr=mlp_args.get("lr", 1e-3),
                    batch_size=mlp_args.get("batch_size", 64),
                    weight_decay=mlp_args.get("weight_decay", 1e-4),
                    patience=mlp_args.get("patience", 20),
                    device=device,
                )

                # Predict on test (same PCA space, same z-score stats)
                Xte_z = to_t(((X_te_full - mu_x) / std_x).astype(np.float32))
                net.eval()
                with torch.no_grad():
                    pred_z = net(Xte_z).cpu().numpy().ravel()
                pred_mlp[te_idx] = (pred_z * std_y + mu_y).astype(np.float32)

            lag_results["mlp"] = _r2(y, pred_mlp)

        results[lag_f] = lag_results

    return results


def sweep_worm(
    worm: dict,
    lag_secs: list[float],
    causal_mode: str,
    n_folds: int = 5,
    n_jobs: int = 1,
    max_features: int = 0,
    models: list[str] | None = None,
    mlp_args: dict | None = None,
    neuron_names: list[str] | None = None,
):
    """
    Run the full lag sweep for one worm.  Returns a list of dicts
    [{neuron, lag_sec, lag_frames, model, r2}, …].

    Parameters
    ----------
    max_features : int
        If > 0, apply PCA to cap the feature count per regression.
        Recommended: 50–100 to keep comparisons fair across lags.
    models : list[str]
        Which models to run. Default: ["ridge"].
        Options: "ridge", "mlp".
    mlp_args : dict
        MLP hyperparameters (hidden, epochs, lr, etc.).
    """
    import pandas as pd

    if models is None:
        models = ["ridge"]

    u = worm["u"]
    labels = worm["labels"]
    dt = worm["dt"]
    T, N = u.shape

    # Convert seconds → frames, deduplicate, sort
    lag_frame_set: dict[int, float] = {}
    for s in sorted(lag_secs):
        f = max(1, int(round(s / dt)))
        if f not in lag_frame_set:
            lag_frame_set[f] = s
    lag_frames_sorted = sorted(lag_frame_set.keys())
    max_lag = max(lag_frames_sorted)

    # Filter to requested neurons
    if neuron_names is not None:
        neuron_indices = [i for i, lb in enumerate(labels) if lb in neuron_names]
        missing = set(neuron_names) - set(labels)
        if missing:
            print(f"    [warn] neurons not in this worm: {sorted(missing)}")
    else:
        neuron_indices = list(range(N))

    pca_str = f", PCA→{max_features}" if max_features > 0 else ""
    print(
        f"  {worm['name']}: N={N}, T={T}, dt={dt:.3f}s, "
        f"max_lag={max_lag} frames ({max_lag*dt:.1f}s), "
        f"{len(lag_frames_sorted)} lag values, "
        f"models={models}, {len(neuron_indices)} neurons{pca_str}"
    )

    # For MLP we cannot use joblib easily (GPU contention), run sequentially
    use_parallel = n_jobs != 1 and "mlp" not in models

    if use_parallel:
        from joblib import Parallel, delayed

        jobs = [
            delayed(_sweep_one_neuron)(
                u, idx, lag_frames_sorted, max_lag, causal_mode, n_folds,
                max_features=max_features,
                models=models,
                mlp_args=mlp_args,
            )
            for idx in neuron_indices
        ]
        raw = Parallel(n_jobs=n_jobs, verbose=0)(jobs)
        neuron_results = dict(zip(neuron_indices, raw))
    else:
        neuron_results = {}
        for ni_count, idx in enumerate(neuron_indices):
            try:
                res = _sweep_one_neuron(
                    u, idx, lag_frames_sorted, max_lag, causal_mode, n_folds,
                    max_features=max_features,
                    models=models,
                    mlp_args=mlp_args,
                )
            except Exception as exc:
                import traceback
                print(f"    !! ERROR on neuron {labels[idx]}: {exc}")
                traceback.print_exc()
                continue
            neuron_results[idx] = res
            # Show progress for every neuron (small subset) or every 10th
            n_total = len(neuron_indices)
            if n_total <= 20 or (ni_count + 1) % 10 == 0 or ni_count == n_total - 1:
                sample_lag = lag_frames_sorted[0]
                sample_r2 = {m: res[sample_lag][m] for m in res[sample_lag]}
                r2_str = "  ".join(f"{m}={v:.3f}" for m, v in sample_r2.items())
                print(f"    {ni_count+1}/{n_total} {labels[idx]:<10s} lag0: {r2_str}")

    # Flatten to list of records — one row per (neuron, lag, model)
    records = []
    for idx, lag_dict in neuron_results.items():
        for lag_f, model_r2 in lag_dict.items():
            for model_name, r2 in model_r2.items():
                records.append(
                    {
                        "worm": worm["name"],
                        "neuron": labels[idx],
                        "neuron_idx": idx,
                        "lag_frames": lag_f,
                        "lag_sec": lag_f * dt,
                        "model": model_name,
                        "r2": r2,
                    }
                )
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════


def _setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
        }
    )
    return plt


def plot_lag_curve(df, out_dir: Path, title_extra: str = ""):
    """
    Main result: mean R² ± SEM vs lag, one line per model.
    If multiple worms, shows grand mean with shaded CI.
    """
    import pandas as pd
    plt = _setup_style()

    causal_modes = sorted(df["causal"].unique())
    model_names = sorted(df["model"].unique())
    worms = sorted(df["worm"].unique())
    multi_worm = len(worms) > 1

    model_palette = {"ridge": "#2980b9", "mlp": "#e74c3c"}
    model_markers = {"ridge": "o", "mlp": "s"}

    fig, axes = plt.subplots(
        1, len(causal_modes),
        figsize=(7 * len(causal_modes), 5),
        squeeze=False,
    )

    for ax, cmode in zip(axes[0], causal_modes):
        sub = df[df["causal"] == cmode]

        for model_name in model_names:
            msub = sub[sub["model"] == model_name]
            color = model_palette.get(model_name, "#555")
            marker = model_markers.get(model_name, "^")

            if multi_worm:
                # thin per-worm lines
                for worm in worms:
                    ws = msub[msub["worm"] == worm]
                    agg = ws.groupby("lag_sec")["r2"].mean().sort_index()
                    ax.plot(agg.index, agg.values, lw=0.4, alpha=0.2,
                            color=color)
                # grand mean ± SEM
                worm_means = (
                    msub.groupby(["worm", "lag_sec"])["r2"]
                    .mean().reset_index()
                )
                grand = worm_means.groupby("lag_sec")["r2"].agg(
                    ["mean", "std", "count"]
                )
                grand["sem"] = grand["std"] / np.sqrt(grand["count"])
                ax.fill_between(
                    grand.index,
                    grand["mean"] - grand["sem"],
                    grand["mean"] + grand["sem"],
                    alpha=0.2, color=color,
                )
                ax.plot(
                    grand.index, grand["mean"],
                    lw=2, color=color, marker=marker, ms=5,
                    label=f"{model_name}  (n={len(worms)} worms)",
                )
            else:
                agg = msub.groupby("lag_sec")["r2"].agg(
                    ["mean", "std", "count"]
                )
                agg["sem"] = agg["std"] / np.sqrt(agg["count"])
                ax.fill_between(
                    agg.index,
                    agg["mean"] - agg["sem"],
                    agg["mean"] + agg["sem"],
                    alpha=0.2, color=color,
                )
                ax.plot(
                    agg.index, agg["mean"],
                    lw=2, color=color, marker=marker, ms=5,
                    label=f"{model_name}  (n={agg['count'].iloc[0]} neurons)",
                )

        ax.set_xlabel("Lag window (s)")
        ax.set_ylabel("R² (5-fold CV)")
        ax.set_title(f"causal = {cmode}")
        ax.legend()
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_xlim(left=0)

    models_str = " + ".join(model_names)
    fig.suptitle(
        f"Masked-neuron R² vs lag window  ({models_str}){title_extra}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "lag_curve.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / 'lag_curve.png'}")


def plot_heatmap(df, out_dir: Path):
    """Neuron × lag heatmap, sorted by mean R² across lags. One per model×causal."""
    import pandas as pd
    plt = _setup_style()

    causal_modes = sorted(df["causal"].unique())
    model_names = sorted(df["model"].unique())

    for cmode in causal_modes:
        for model_name in model_names:
            sub = df[(df["causal"] == cmode) & (df["model"] == model_name)]
            if sub.empty:
                continue
            # average across worms
            piv = sub.pivot_table(
                index="neuron", columns="lag_sec", values="r2", aggfunc="mean"
            )
            # sort rows by mean R²
            row_order = piv.mean(axis=1).sort_values(ascending=False).index
            piv = piv.loc[row_order]

            n_neurons = len(piv)
            fig_h = max(4, n_neurons * 0.12)
            fig, ax = plt.subplots(figsize=(10, fig_h))

            im = ax.imshow(
                piv.values, aspect="auto", cmap="RdYlGn",
                vmin=max(piv.values.min(), -0.1),
                vmax=min(piv.values.max(), 1.0),
                interpolation="nearest",
            )
            ax.set_yticks(range(n_neurons))
            ax.set_yticklabels(piv.index, fontsize=max(4, min(7, 400 // n_neurons)))
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([f"{c:.1f}" for c in piv.columns], fontsize=8,
                               rotation=45, ha="right")
            ax.set_xlabel("Lag (s)")
            ax.set_ylabel("Neuron (sorted by mean R²)")
            ax.set_title(f"Per-neuron R² vs lag  ({model_name}, causal={cmode})")
            plt.colorbar(im, ax=ax, label="R²", shrink=0.7)
            fig.tight_layout()
            fname = f"heatmap_{model_name}_{cmode}.png"
            fig.savefig(out_dir / fname, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out_dir / fname}")


def plot_optimal_lag(df, out_dir: Path):
    """Histogram of the lag that maximises R² for each neuron, per model."""
    import pandas as pd
    plt = _setup_style()

    causal_modes = sorted(df["causal"].unique())
    model_names = sorted(df["model"].unique())
    model_palette = {"ridge": "#2980b9", "mlp": "#e74c3c"}

    fig, axes = plt.subplots(
        len(model_names), len(causal_modes),
        figsize=(6 * len(causal_modes), 4.5 * len(model_names)),
        squeeze=False,
    )

    for ri, model_name in enumerate(model_names):
        for ci, cmode in enumerate(causal_modes):
            ax = axes[ri, ci]
            sub = df[(df["causal"] == cmode) & (df["model"] == model_name)]
            if sub.empty:
                continue
            mean_r2 = sub.groupby(["neuron", "lag_sec"])["r2"].mean().reset_index()
            best = mean_r2.loc[mean_r2.groupby("neuron")["r2"].idxmax()]

            color = model_palette.get(model_name, "#27ae60")
            ax.hist(best["lag_sec"], bins=20, alpha=0.75,
                    edgecolor="k", lw=0.4, color=color)
            med_lag = best["lag_sec"].median()
            ax.axvline(med_lag, color="red", ls="--", lw=1.5,
                       label=f"median = {med_lag:.1f} s")
            ax.set_xlabel("Optimal lag (s)")
            ax.set_ylabel("# neurons")
            ax.set_title(f"{model_name}, causal = {cmode}")
            ax.legend()

    fig.suptitle("Optimal lag per neuron", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "optimal_lag.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / 'optimal_lag.png'}")


def plot_marginal_gain(df, out_dir: Path):
    """Marginal ΔR² per additional second of lag, per model."""
    import pandas as pd
    plt = _setup_style()

    causal_modes = sorted(df["causal"].unique())
    model_names = sorted(df["model"].unique())
    model_palette = {"ridge": "#2980b9", "mlp": "#e74c3c"}

    fig, axes = plt.subplots(
        1, len(causal_modes),
        figsize=(7 * len(causal_modes), 4.5), squeeze=False,
    )

    for ax, cmode in zip(axes[0], causal_modes):
        for model_name in model_names:
            sub = df[(df["causal"] == cmode) & (df["model"] == model_name)]
            if sub.empty:
                continue
            agg = sub.groupby("lag_sec")["r2"].mean().sort_index()
            lags = agg.index.values
            r2s = agg.values
            d_lag = np.diff(lags)
            d_r2 = np.diff(r2s)
            marginal = d_r2 / d_lag
            mid_lags = (lags[:-1] + lags[1:]) / 2

            color = model_palette.get(model_name, "#8e44ad")
            ax.plot(mid_lags, marginal, lw=1.5, marker="o", ms=4,
                    color=color, label=model_name, alpha=0.8)

        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_xlabel("Lag midpoint (s)")
        ax.set_ylabel("ΔR² / Δ lag (per second)")
        ax.set_title(f"Marginal gain  (causal={cmode})")
        ax.legend()

    fig.suptitle("Diminishing returns: R² gain per additional second of lag",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "marginal_gain.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / 'marginal_gain.png'}")


def plot_causal_comparison(df, out_dir: Path):
    """Overlay inclusive vs strict on the same axes (if both present), per model."""
    import pandas as pd
    plt = _setup_style()

    causal_modes = sorted(df["causal"].unique())
    model_names = sorted(df["model"].unique())
    if len(causal_modes) < 2:
        return

    causal_palette = {"inclusive": "#2980b9", "strict": "#e67e22"}
    model_ls = {"ridge": "-", "mlp": "--"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in model_names:
        for cmode in causal_modes:
            sub = df[(df["causal"] == cmode) & (df["model"] == model_name)]
            if sub.empty:
                continue
            agg = sub.groupby("lag_sec")["r2"].agg(["mean", "std", "count"])
            agg["sem"] = agg["std"] / np.sqrt(agg["count"])
            color = causal_palette.get(cmode, "#555")
            ls = model_ls.get(model_name, "-")

            ax.fill_between(
                agg.index,
                agg["mean"] - agg["sem"],
                agg["mean"] + agg["sem"],
                alpha=0.12, color=color,
            )
            ax.plot(
                agg.index, agg["mean"],
                lw=2, color=color, ls=ls, marker="o", ms=4,
                label=f"{cmode} {model_name}",
            )

    ax.set_xlabel("Lag window (s)")
    ax.set_ylabel("R² (5-fold CV)")
    ax.set_title("Inclusive vs Strict × model")
    ax.legend()
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out_dir / "causal_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / 'causal_comparison.png'}")


def plot_ridge_vs_mlp(df, out_dir: Path):
    """Scatter and Δ-histogram: ridge vs MLP at each lag, if both present."""
    import pandas as pd
    plt = _setup_style()

    model_names = sorted(df["model"].unique())
    if "ridge" not in model_names or "mlp" not in model_names:
        return

    causal_modes = sorted(df["causal"].unique())
    lag_secs = sorted(df["lag_sec"].unique())

    # ── 1.  Scatter per lag ──────────────────────────────────────────
    n_lags = len(lag_secs)
    ncols = min(4, n_lags)
    nrows = (n_lags + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.2 * nrows),
                             squeeze=False)

    for i, lag_s in enumerate(lag_secs):
        ax = axes[i // ncols][i % ncols]
        for cmode in causal_modes:
            ridge_sub = df[(df["causal"] == cmode) & (df["model"] == "ridge")
                           & (df["lag_sec"] == lag_s)]
            mlp_sub = df[(df["causal"] == cmode) & (df["model"] == "mlp")
                         & (df["lag_sec"] == lag_s)]
            if ridge_sub.empty or mlp_sub.empty:
                continue
            # align by neuron (and worm)
            merged = pd.merge(
                ridge_sub[["worm", "neuron", "r2"]].rename(columns={"r2": "ridge"}),
                mlp_sub[["worm", "neuron", "r2"]].rename(columns={"r2": "mlp"}),
                on=["worm", "neuron"],
            )
            ax.scatter(merged["ridge"], merged["mlp"], s=8, alpha=0.5,
                       edgecolors="none")

        lo = min(df["r2"].min(), -0.05)
        hi = max(df["r2"].max(), 1.0)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.5)
        frac = float((merged["mlp"] > merged["ridge"]).mean()) if len(merged) else 0
        ax.set_title(f"lag={lag_s:.1f}s  (MLP↑ {frac:.0%})", fontsize=9)
        ax.set_xlabel("Ridge R²", fontsize=8)
        ax.set_ylabel("MLP R²", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")

    # hide unused axes
    for j in range(n_lags, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Ridge vs MLP  per neuron (each panel = one lag)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "ridge_vs_mlp_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / 'ridge_vs_mlp_scatter.png'}")

    # ── 2.  Δ(MLP − Ridge) vs lag  ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for cmode in causal_modes:
        deltas_by_lag = []
        lag_vals = []
        for lag_s in lag_secs:
            ridge_sub = df[(df["causal"] == cmode) & (df["model"] == "ridge")
                           & (df["lag_sec"] == lag_s)]
            mlp_sub = df[(df["causal"] == cmode) & (df["model"] == "mlp")
                         & (df["lag_sec"] == lag_s)]
            merged = pd.merge(
                ridge_sub[["worm", "neuron", "r2"]].rename(columns={"r2": "ridge"}),
                mlp_sub[["worm", "neuron", "r2"]].rename(columns={"r2": "mlp"}),
                on=["worm", "neuron"],
            )
            if merged.empty:
                continue
            delta = (merged["mlp"] - merged["ridge"]).values
            deltas_by_lag.append(delta)
            lag_vals.append(lag_s)

        means = [d.mean() for d in deltas_by_lag]
        sems = [d.std() / np.sqrt(len(d)) for d in deltas_by_lag]
        color = "#2980b9" if cmode == "inclusive" else "#e67e22"
        ax.errorbar(lag_vals, means, yerr=sems, lw=2, marker="o", ms=5,
                    color=color, capsize=3, label=cmode)

    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("Lag window (s)")
    ax.set_ylabel("Δ R²  (MLP − Ridge)")
    ax.set_title("MLP advantage over Ridge vs lag")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mlp_advantage_vs_lag.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / 'mlp_advantage_vs_lag.png'}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    pa = argparse.ArgumentParser(
        description="Sweep lag window for masked-neuron prediction (ridge + MLP)"
    )
    pa.add_argument("--dataset_dir", type=Path, default=_DEFAULT_DATASET)
    pa.add_argument("--out_dir", type=Path, default=_DEFAULT_OUT)
    pa.add_argument(
        "--worm", type=str, default=None,
        help="Single worm stem (e.g. 2022-08-02-01). Default: all worms.",
    )
    pa.add_argument("--max_worms", type=int, default=0,
                    help="Limit number of worms (0 = all)")
    pa.add_argument(
        "--lag_secs", type=float, nargs="+",
        default=[0.6, 1.2, 2, 3, 5, 8],
        help="Lag values to test in seconds (default: 0.6–8 s)",
    )
    pa.add_argument(
        "--causal", type=str, nargs="+", default=["inclusive"],
        choices=["inclusive", "strict"],
        help="Causal mode(s) to evaluate",
    )
    pa.add_argument(
        "--models", type=str, nargs="+", default=["ridge"],
        choices=["ridge", "mlp"],
        help="Models to run.  'ridge' is fast; add 'mlp' for comparison.",
    )
    pa.add_argument(
        "--neurons", type=str, nargs="+", default=None,
        help="Subset of neuron names to evaluate. Default: all neurons.",
    )
    pa.add_argument("--n_folds", type=int, default=5)
    pa.add_argument("--n_jobs", type=int, default=1,
                    help="Parallel workers for neurons (−1 = all cores). "
                         "Only used for ridge-only runs; MLP always sequential.")
    pa.add_argument(
        "--max_features", type=int, default=50,
        help="PCA cap on feature count (0 = no PCA). Default 50 — prevents "
             "curse-of-dimensionality collapse at high lags.",
    )
    # MLP hyperparameters
    pa.add_argument("--hidden", type=int, default=128)
    pa.add_argument("--epochs", type=int, default=200,
                    help="MLP max epochs (lower than main script for speed)")
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--weight_decay", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=20,
                    help="MLP early-stopping patience (lower for speed)")
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--seed", type=int, default=0)
    args = pa.parse_args()

    np.random.seed(args.seed)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── load worms ───────────────────────────────────────────────────────
    if args.worm:
        h5_files = [args.dataset_dir / f"{args.worm}.h5"]
    else:
        h5_files = sorted(args.dataset_dir.glob("*.h5"))

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
    print(f"\nLoaded {len(worms)} worm(s)")
    print(f"Lag values (s): {args.lag_secs}")
    print(f"Causal modes:   {args.causal}")
    print(f"Models:         {args.models}")
    print(f"PCA cap:        {args.max_features if args.max_features > 0 else 'OFF'}")
    print(f"n_jobs:         {args.n_jobs}")

    # Build MLP args dict
    mlp_args = {
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "device": args.device,
        "seed": args.seed,
    } if "mlp" in args.models else None

    # ── sweep ────────────────────────────────────────────────────────────
    import pandas as pd

    all_records: list[dict] = []
    t0 = time.time()

    for cmode in args.causal:
        print(f"\n{'═' * 50}")
        print(f"  Causal mode: {cmode}")
        print(f"{'═' * 50}")

        for worm in worms:
            t_w = time.time()
            records = sweep_worm(
                worm, args.lag_secs, cmode,
                n_folds=args.n_folds, n_jobs=args.n_jobs,
                max_features=args.max_features,
                models=args.models,
                mlp_args=mlp_args,
                neuron_names=args.neurons,
            )
            for r in records:
                r["causal"] = cmode
            all_records.extend(records)
            print(f"    {worm['name']}: {len(records)} records  "
                  f"({time.time() - t_w:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nTotal sweep: {len(all_records)} records  ({elapsed:.1f}s)")

    # ── save results ─────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    csv_path = args.out_dir / "lag_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Quick summary
    print("\n── Summary (mean R² across neurons) ──")
    summary = (
        df.groupby(["causal", "model", "lag_sec"])["r2"]
        .agg(["mean", "median", "std"])
        .round(4)
    )
    print(summary.to_string())

    # ── plots ────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    worm_tag = f"  ({worms[0]['name']})" if len(worms) == 1 else ""
    plot_lag_curve(df, args.out_dir, title_extra=worm_tag)
    plot_heatmap(df, args.out_dir)
    plot_optimal_lag(df, args.out_dir)
    plot_marginal_gain(df, args.out_dir)
    plot_causal_comparison(df, args.out_dir)
    plot_ridge_vs_mlp(df, args.out_dir)

    # ── save args ────────────────────────────────────────────────────────
    meta = {
        "lag_secs": args.lag_secs,
        "causal_modes": args.causal,
        "models": args.models,
        "n_folds": args.n_folds,
        "n_jobs": args.n_jobs,
        "max_features": args.max_features,
        "n_worms": len(worms),
        "worms": [w["name"] for w in worms],
        "elapsed_sec": round(elapsed, 1),
        "n_records": len(all_records),
    }
    if mlp_args:
        meta["mlp_args"] = mlp_args
    (args.out_dir / "sweep_meta.json").write_text(
        json.dumps(meta, indent=2)
    )

    print(f"\nAll outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
