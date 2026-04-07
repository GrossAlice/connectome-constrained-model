#!/usr/bin/env python3
"""
Diagnostic plots for masked-neuron prediction on a single worm.

Investigates:
  1. Why Ridge R² goes negative (AR vs population, fold-by-fold breakdown)
  2. Whether the relationship is linear (MLP–Ridge gap distribution)
  3. AVA neurons highlighted everywhere

Usage:
  python -m scripts.masked_neuron.diagnose_one_worm \
      --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-07-15-12.h5"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.masked_neuron.masked_neuron_prediction import (
    _load_worm,
    _make_folds,
    _inner_split,
    _r2,
    _zscore,
    _make_mlp,
)

_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_OUT = _ROOT / "output_plots/masked_neuron_prediction/diagnostics"
_RIDGE_ALPHAS = np.logspace(-4, 6, 30)
_HIGHLIGHT = ("AVA",)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _pearson_r(y, yhat):
    c = np.corrcoef(y, yhat)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _precompute_lagged(u, n_lags, causal="strict"):
    T, N = u.shape
    T_out = T - n_lags
    blocks = []
    if causal == "strict":
        for lag in range(1, n_lags + 1):
            blocks.append(u[n_lags - lag:n_lags - lag + T_out])
    else:
        for lag in range(n_lags):
            blocks.append(u[n_lags - lag:n_lags - lag + T_out])
    return np.concatenate(blocks, axis=1).astype(np.float32), \
           u[n_lags:n_lags + T_out].astype(np.float32)


def _slice(X_full, ni, N, n_lags):
    cols = [ni + lag * N for lag in range(n_lags)]
    mask = np.ones(X_full.shape[1], dtype=bool); mask[cols] = False
    return X_full[:, mask]


def _ar_features(u, ni, n_lags, T_out):
    X_ar = np.empty((T_out, n_lags), dtype=np.float32)
    for lag in range(1, n_lags + 1):
        X_ar[:, lag - 1] = u[n_lags - lag:n_lags - lag + T_out, ni]
    return X_ar


def _train_mlp_fullbatch(net, Xtr, Ytr, Xva, Yva, *, epochs, lr, wd, patience):
    import torch
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    crit = torch.nn.MSELoss()
    best_val, stale, best_state = float("inf"), 0, None
    for _ in range(epochs):
        net.train()
        loss = crit(net(Xtr), Ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = crit(net(Xva), Yva).item()
        if vl < best_val:
            best_val, stale = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if patience and stale >= patience:
            break
    if best_state:
        net.load_state_dict(best_state)


# ═══════════════════════════════════════════════════════════════════════
#  Per-neuron diagnostics
# ═══════════════════════════════════════════════════════════════════════

def run_diagnostics(worm, n_lags, causal, n_folds, mlp_kwargs):
    """Return a list of dicts with comprehensive per-neuron metrics."""
    import torch
    from sklearn.linear_model import RidgeCV

    u = worm["u"]
    labs = worm["labels"]
    T, N = u.shape
    T_out = T - n_lags

    X_full, Y_all = _precompute_lagged(u, n_lags, causal)
    folds = _make_folds(T_out, n_folds)

    autocorr = []
    for j in range(N):
        c = np.corrcoef(u[:-n_lags, j], u[n_lags:, j])[0, 1]
        autocorr.append(c if np.isfinite(c) else 0.0)
    autocorr = np.array(autocorr)
    var = Y_all.var(axis=0)

    device = torch.device(mlp_kwargs["device_str"])
    to_t = lambda a: torch.from_numpy(a).to(device)

    records = []
    for ni in range(N):
        y = Y_all[:, ni]
        X_pop = _slice(X_full, ni, N, n_lags)
        X_ar = _ar_features(u, ni, n_lags, T_out)

        # ── fold-by-fold Ridge (pop) + Ridge (AR) ──────────────────
        pred_pop = np.zeros(T_out, np.float32)
        pred_ar = np.zeros(T_out, np.float32)
        fold_r2_pop, fold_r2_ar = [], []
        alphas_pop, alphas_ar = [], []

        for tr_idx, te_idx in folds:
            # pop
            rp = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
            rp.fit(X_pop[tr_idx], y[tr_idx])
            pred_pop[te_idx] = rp.predict(X_pop[te_idx]).astype(np.float32)
            fold_r2_pop.append(_r2(y[te_idx], pred_pop[te_idx]))
            alphas_pop.append(rp.alpha_)
            # ar
            ra = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
            ra.fit(X_ar[tr_idx], y[tr_idx])
            pred_ar[te_idx] = ra.predict(X_ar[te_idx]).astype(np.float32)
            fold_r2_ar.append(_r2(y[te_idx], pred_ar[te_idx]))
            alphas_ar.append(ra.alpha_)

        r2_pop = _r2(y, pred_pop)
        r2_ar = _r2(y, pred_ar)
        corr_pop = _pearson_r(y, pred_pop)

        # ── MLP (pop) ─────────────────────────────────────────────
        pred_mlp = np.zeros(T_out, np.float32)
        fold_r2_mlp = []
        for fi, (tr_idx, te_idx) in enumerate(folds):
            tr_in, va_in = _inner_split(tr_idx)
            Xtr, Ytr = X_pop[tr_in], y[tr_in]
            Xva, Yva = X_pop[va_in], y[va_in]
            mu_x, std_x = _zscore(Xtr)
            mu_y, std_y = float(Ytr.mean()), float(max(Ytr.std(), 1e-8))
            Xtr_z = to_t(((Xtr - mu_x) / std_x).astype(np.float32))
            Ytr_z = to_t(((Ytr.reshape(-1,1) - mu_y) / std_y).astype(np.float32))
            Xva_z = to_t(((Xva - mu_x) / std_x).astype(np.float32))
            Yva_z = to_t(((Yva.reshape(-1,1) - mu_y) / std_y).astype(np.float32))
            torch.manual_seed(mlp_kwargs["seed"] + fi)
            net = _make_mlp(X_pop.shape[1], mlp_kwargs["hidden"], 1,
                            mlp_kwargs["dropout"]).to(device)
            _train_mlp_fullbatch(net, Xtr_z, Ytr_z, Xva_z, Yva_z,
                                 epochs=mlp_kwargs["epochs"], lr=mlp_kwargs["lr"],
                                 wd=mlp_kwargs["weight_decay"],
                                 patience=mlp_kwargs["patience"])
            net.eval()
            Xte_z = to_t(((X_pop[te_idx] - mu_x) / std_x).astype(np.float32))
            with torch.no_grad():
                pz = net(Xte_z).cpu().numpy().ravel()
            pred_mlp[te_idx] = (pz * std_y + mu_y).astype(np.float32)
            fold_r2_mlp.append(_r2(y[te_idx], pred_mlp[te_idx]))

        r2_mlp = _r2(y, pred_mlp)
        corr_mlp = _pearson_r(y, pred_mlp)

        records.append({
            "neuron": labs[ni], "ni": ni,
            "var": float(var[ni]),
            "autocorr": float(autocorr[ni]),
            "r2_ar": r2_ar,
            "r2_ridge": r2_pop,
            "r2_mlp": r2_mlp,
            "corr_ridge": corr_pop,
            "corr_mlp": corr_mlp,
            "fold_r2_ridge": fold_r2_pop,
            "fold_r2_ar": fold_r2_ar,
            "fold_r2_mlp": fold_r2_mlp,
            "alphas_pop": alphas_pop,
            "is_ava": labs[ni].startswith("AVA"),
        })

        if (ni + 1) % 10 == 0 or ni == N - 1:
            print(f"  [{ni+1:3d}/{N}] {labs[ni]:8s}  "
                  f"AR={r2_ar:+.3f}  Ridge={r2_pop:+.4f}  "
                  f"MLP={r2_mlp:+.4f}")

    return records


# ═══════════════════════════════════════════════════════════════════════
#  Lag sweep: Ridge vs MLP at many lags to probe nonlinearity
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_LAGS = [1, 2, 5, 10, 20, 30, 50]


def run_lag_sweep(worm, lags, causal, n_folds, mlp_kwargs, pca_k=20):
    """For each lag, compute Ridge, PCA-Ridge, and MLP R² for every neuron.

    Returns
    -------
    lag_results : dict[int, list[dict]]
        Keyed by lag value; each value is a list of per-neuron dicts
        with keys 'neuron', 'is_ava', 'r2_ridge', 'r2_pca_ridge',
        'r2_mlp', 'corr_ridge', 'corr_pca_ridge', 'corr_mlp'.
    """
    import torch
    from sklearn.linear_model import RidgeCV
    from joblib import Parallel, delayed

    u = worm["u"]
    labs = worm["labels"]
    T, N = u.shape
    device = torch.device(mlp_kwargs["device_str"])
    to_t = lambda a: torch.from_numpy(a).to(device)

    lag_results = {}

    for n_lags in lags:
        if n_lags >= T - 50:
            print(f"  [skip lag={n_lags}: T_out too small]")
            continue
        T_out = T - n_lags
        X_full, Y_all = _precompute_lagged(u, n_lags, causal)
        folds = _make_folds(T_out, n_folds)

        # ── Ridge + PCA-Ridge: parallel across neurons ──────────
        def _ridge_pca_one(ni):
            from sklearn.linear_model import RidgeCV as _R
            from sklearn.decomposition import PCA
            X = _slice(X_full, ni, N, n_lags)
            y = Y_all[:, ni]
            pred_r = np.zeros(T_out, np.float32)
            pred_p = np.zeros(T_out, np.float32)
            k = min(pca_k, X.shape[1])
            for tr_idx, te_idx in folds:
                # Raw Ridge
                r = _R(alphas=_RIDGE_ALPHAS, fit_intercept=True)
                r.fit(X[tr_idx], y[tr_idx])
                pred_r[te_idx] = r.predict(X[te_idx]).astype(np.float32)
                # PCA-Ridge
                k_eff = min(k, len(tr_idx) - 1)
                pca = PCA(n_components=k_eff)
                Xtr_p = pca.fit_transform(X[tr_idx])
                Xte_p = pca.transform(X[te_idx])
                rp = _R(alphas=_RIDGE_ALPHAS, fit_intercept=True)
                rp.fit(Xtr_p, y[tr_idx])
                pred_p[te_idx] = rp.predict(Xte_p).astype(np.float32)
            return (_r2(y, pred_r), _pearson_r(y, pred_r),
                    _r2(y, pred_p), _pearson_r(y, pred_p))

        ridge_raw = Parallel(n_jobs=min(8, N), verbose=0)(
            delayed(_ridge_pca_one)(ni) for ni in range(N))

        # ── MLP: sequential, full-batch ─────────────────────────
        recs = []
        for ni in range(N):
            X = _slice(X_full, ni, N, n_lags)
            y = Y_all[:, ni]
            pred = np.zeros(T_out, np.float32)
            for fi, (tr_idx, te_idx) in enumerate(folds):
                tr_in, va_in = _inner_split(tr_idx)
                Xtr, Ytr = X[tr_in], y[tr_in]
                Xva, Yva = X[va_in], y[va_in]
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
                _train_mlp_fullbatch(net, Xtr_z, Ytr_z, Xva_z, Yva_z,
                                     epochs=mlp_kwargs["epochs"],
                                     lr=mlp_kwargs["lr"],
                                     wd=mlp_kwargs["weight_decay"],
                                     patience=mlp_kwargs["patience"])
                net.eval()
                Xte_z = to_t(((X[te_idx] - mu_x) / std_x).astype(np.float32))
                with torch.no_grad():
                    pz = net(Xte_z).cpu().numpy().ravel()
                pred[te_idx] = (pz * std_y + mu_y).astype(np.float32)

            r2_r, corr_r, r2_p, corr_p = ridge_raw[ni]
            r2_m = _r2(y, pred)
            corr_m = _pearson_r(y, pred)
            recs.append({
                "neuron": labs[ni], "is_ava": labs[ni].startswith("AVA"),
                "r2_ridge": r2_r, "r2_pca_ridge": r2_p, "r2_mlp": r2_m,
                "corr_ridge": corr_r, "corr_pca_ridge": corr_p, "corr_mlp": corr_m,
            })

        lag_results[n_lags] = recs
        med_rr = np.median([r["r2_ridge"] for r in recs])
        med_rp = np.median([r["r2_pca_ridge"] for r in recs])
        med_rm = np.median([r["r2_mlp"] for r in recs])
        gap_nonlin = med_rm - med_rp
        print(f"  lag={n_lags:3d}  Ridge={med_rr:+.4f}  "
              f"PCA-Ridge={med_rp:+.4f}  MLP={med_rm:+.4f}  "
              f"overfit={med_rp - med_rr:+.4f}  nonlin={gap_nonlin:+.4f}")

    return lag_results


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "figure.facecolor": "white",
    })
    return plt


def make_diagnostic_plots(records, worm_name, out_dir, n_lags, n_folds):
    plt = _setup_mpl()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    r2_ar    = np.array([r["r2_ar"] for r in records])
    r2_ridge = np.array([r["r2_ridge"] for r in records])
    r2_mlp   = np.array([r["r2_mlp"] for r in records])
    corr_r   = np.array([r["corr_ridge"] for r in records])
    corr_m   = np.array([r["corr_mlp"] for r in records])
    autocorr = np.array([r["autocorr"] for r in records])
    var      = np.array([r["var"] for r in records])
    names    = [r["neuron"] for r in records]
    ava_mask = np.array([r["is_ava"] for r in records])
    N = len(records)

    ava_c = "gold"
    reg_c = "#2c3e50"

    # ── 1) AR R² vs Ridge R² ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(r2_ar[~ava_mask], r2_ridge[~ava_mask],
               s=18, alpha=0.5, color=reg_c, edgecolors="none", label="other")
    if ava_mask.any():
        ax.scatter(r2_ar[ava_mask], r2_ridge[ava_mask],
                   marker="*", s=200, color=ava_c, edgecolors="k",
                   linewidths=0.8, zorder=6, label="AVA")
        for i in np.where(ava_mask)[0]:
            ax.annotate(names[i], (r2_ar[i], r2_ridge[i]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=7, fontweight="bold")
    ax.axhline(0, color="r", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0, color="r", lw=0.8, ls="--", alpha=0.6)
    ax.plot([0, 1], [0, 1], "k:", lw=0.7, alpha=0.5)
    ax.set_xlabel("AR-only Ridge R² (self-predictability)")
    ax.set_ylabel("Population Ridge R² (other neurons → target)")
    ax.set_title(f"{worm_name}:  AR R² vs Pop-Ridge R²  (N={N})")
    ax.legend(fontsize=9)
    n_neg = int((r2_ridge < 0).sum())
    ax.text(0.02, 0.02,
            f"{n_neg}/{N} neurons with R²<0\n"
            f"These have high AR R² but pop. features\n"
            f"add noise → worse than predicting mean",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(facecolor="lightyellow", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_ar_vs_pop_r2.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2) R² vs autocorrelation at lag-n ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, vals, label in [
        (axes[0], r2_ridge, "Ridge R²"),
        (axes[1], r2_mlp, "MLP R²"),
    ]:
        ax.scatter(autocorr[~ava_mask], vals[~ava_mask],
                   s=18, alpha=0.5, color=reg_c, edgecolors="none")
        if ava_mask.any():
            ax.scatter(autocorr[ava_mask], vals[ava_mask],
                       marker="*", s=200, color=ava_c, edgecolors="k",
                       linewidths=0.8, zorder=6)
            for i in np.where(ava_mask)[0]:
                ax.annotate(names[i], (autocorr[i], vals[i]),
                            xytext=(6, 4), textcoords="offset points",
                            fontsize=7, fontweight="bold")
        ax.axhline(0, color="r", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel(f"Autocorrelation at lag={n_lags}")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs self-autocorr  (N={N})")
    fig.suptitle(f"{worm_name}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_r2_vs_autocorr.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3) MLP − Ridge gap → linearity test ──────────────────────────
    gap_r2 = r2_mlp - r2_ridge
    gap_corr = corr_m - corr_r

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, gap, label in [
        (axes[0], gap_r2, "R²"),
        (axes[1], gap_corr, "Pearson r"),
    ]:
        ax.hist(gap, bins=40, density=True, alpha=0.6, color="#9b59b6",
                edgecolor="k", lw=0.3)
        ax.axvline(0, color="k", lw=1, ls="--")
        med = float(np.median(gap))
        mu = float(np.mean(gap))
        frac = float((gap > 0).mean())
        ax.axvline(med, color="orange", ls=":", lw=1.5,
                   label=f"median={med:+.4f}")
        ax.axvline(mu, color="navy", ls="--", lw=1,
                   label=f"mean={mu:+.4f}")
        ax.set_xlabel(f"MLP {label} − Ridge {label}")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} gap  (MLP > Ridge: {frac:.0%})")
        ax.legend(fontsize=9)
    fig.suptitle(f"{worm_name}: linearity check — MLP−Ridge gap",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_linearity_gap.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 4) Ridge R² vs MLP R² scatter + identity ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, vr, vm, label in [
        (axes[0], r2_ridge, r2_mlp, "R²"),
        (axes[1], corr_r, corr_m, "Pearson r"),
    ]:
        ax.scatter(vr[~ava_mask], vm[~ava_mask],
                   s=18, alpha=0.5, color=reg_c, edgecolors="none")
        if ava_mask.any():
            ax.scatter(vr[ava_mask], vm[ava_mask],
                       marker="*", s=200, color=ava_c, edgecolors="k",
                       linewidths=0.8, zorder=6)
            for i in np.where(ava_mask)[0]:
                ax.annotate(names[i], (vr[i], vm[i]),
                            xytext=(6, 4), textcoords="offset points",
                            fontsize=7, fontweight="bold")
        lo = min(vr.min(), vm.min()) - 0.05
        hi = max(vr.max(), vm.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8, alpha=0.6)
        ax.set_xlabel(f"Ridge {label}")
        ax.set_ylabel(f"MLP {label}")
        ax.set_aspect("equal", adjustable="datalim")
        frac = float((vm > vr).mean())
        ax.set_title(f"{label}: Ridge vs MLP  (MLP wins {frac:.0%})")
    fig.suptitle(f"{worm_name}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_ridge_vs_mlp_scatter.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 5) Fold-by-fold R² for worst, median, best & AVA neurons ────
    sorted_idx = np.argsort(r2_ridge)
    pick = list(sorted_idx[:3])                           # worst 3
    pick += list(sorted_idx[N//2-1:N//2+2])               # median 3
    pick += list(sorted_idx[-3:])                          # best 3
    pick += list(np.where(ava_mask)[0])                    # AVA
    pick = list(dict.fromkeys(pick))                       # dedupe

    n_pick = len(pick)
    fig, axes = plt.subplots(1, n_pick, figsize=(2.5 * n_pick, 4),
                             sharey=True)
    if n_pick == 1:
        axes = [axes]
    for ax, ni in zip(axes, pick):
        r = records[ni]
        folds_r = r["fold_r2_ridge"]
        folds_a = r["fold_r2_ar"]
        folds_m = r["fold_r2_mlp"]
        x = np.arange(n_folds) + 1
        ax.bar(x - 0.25, folds_a, 0.22, color="#2ecc71", alpha=0.8,
               label="AR-only")
        ax.bar(x, folds_r, 0.22, color="#3498db", alpha=0.8,
               label="Ridge(pop)")
        ax.bar(x + 0.25, folds_m, 0.22, color="#e74c3c", alpha=0.8,
               label="MLP(pop)")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Fold")
        if ax is axes[0]:
            ax.set_ylabel("R²")
            ax.legend(fontsize=6, loc="lower left")
        lbl = r["neuron"]
        star = " ★" if r["is_ava"] else ""
        ax.set_title(f"{lbl}{star}\nR²={r['r2_ridge']:+.3f}", fontsize=9)
        ax.set_xticks(x)
    fig.suptitle(f"{worm_name}: fold-by-fold R² (worst / median / best / AVA)",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_fold_breakdown.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 6) Violin with AVA stars (same style as batch) ───────────────
    rng = np.random.default_rng(42)
    hl_idx = list(np.where(ava_mask)[0])
    colors = {"ridge": "#3498db", "mlp": "#e74c3c"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, metric, vals_r, vals_m, ylabel in [
        (axes[0], "R²", r2_ridge, r2_mlp, "R²"),
        (axes[1], "Correlation", corr_r, corr_m, "Pearson r"),
    ]:
        data = [vals_r, vals_m]
        xlabels = ["Ridge", "MLP 2×128"]
        cols = [colors["ridge"], colors["mlp"]]

        parts = ax.violinplot(data, positions=[0, 1], showmedians=False,
                              showextrema=False)
        for pc, c in zip(parts["bodies"], cols):
            pc.set_facecolor(c); pc.set_alpha(0.35); pc.set_edgecolor(c)

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

        # ★ AVA overlay
        for hi in hl_idx:
            for i, (vals, c) in enumerate(zip(data, cols)):
                xpos = i + all_jitter[i][hi]
                ypos = vals[hi]
                ax.scatter([xpos], [ypos], marker="*", s=120,
                           color="gold", edgecolors="k", linewidths=0.7,
                           zorder=6)
                if i == 0:
                    ax.annotate(
                        names[hi], xy=(xpos, ypos),
                        xytext=(-38, 8), textcoords="offset points",
                        fontsize=7, fontweight="bold", color="k",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(xlabels, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{metric}  (N={N} neurons)")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)

    tag = f"strict, {n_lags}-frame lag, {n_folds}-fold CV"
    fig.suptitle(f"{worm_name}  ({tag})", fontsize=13, fontweight="bold",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_violin_ava.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 7) Signal variance vs R² ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, vals, label in [
        (axes[0], r2_ridge, "Ridge R²"),
        (axes[1], r2_mlp, "MLP R²"),
    ]:
        ax.scatter(var[~ava_mask], vals[~ava_mask],
                   s=18, alpha=0.5, color=reg_c, edgecolors="none")
        if ava_mask.any():
            ax.scatter(var[ava_mask], vals[ava_mask],
                       marker="*", s=200, color=ava_c, edgecolors="k",
                       linewidths=0.8, zorder=6)
            for i in np.where(ava_mask)[0]:
                ax.annotate(names[i], (var[i], vals[i]),
                            xytext=(6, 4), textcoords="offset points",
                            fontsize=7, fontweight="bold")
        ax.axhline(0, color="r", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("Signal variance")
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.set_title(f"{label} vs signal variance")
    fig.suptitle(f"{worm_name}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_r2_vs_variance.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFixed-lag diagnostic plots saved to {out_dir}/")


def make_lag_sweep_plots(lag_results, worm_name, out_dir):
    """Plots from the multi-lag sweep."""
    plt = _setup_mpl()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    lags = sorted(lag_results.keys())
    N = len(lag_results[lags[0]])
    neuron_names = [r["neuron"] for r in lag_results[lags[0]]]
    ava_mask = np.array([r["is_ava"] for r in lag_results[lags[0]]])

    # build arrays: shape (n_lags, N)
    r2_ridge = np.array([[r["r2_ridge"] for r in lag_results[l]] for l in lags])
    r2_pca   = np.array([[r["r2_pca_ridge"] for r in lag_results[l]] for l in lags])
    r2_mlp   = np.array([[r["r2_mlp"]   for r in lag_results[l]] for l in lags])
    cr_ridge = np.array([[r["corr_ridge"] for r in lag_results[l]] for l in lags])
    cr_pca   = np.array([[r["corr_pca_ridge"] for r in lag_results[l]] for l in lags])
    cr_mlp   = np.array([[r["corr_mlp"]  for r in lag_results[l]] for l in lags])

    gap_total_r2   = r2_mlp - r2_ridge
    gap_nonlin_r2  = r2_mlp - r2_pca       # pure nonlinearity
    gap_overfit_r2 = r2_pca - r2_ridge      # dimensionality benefit
    gap_total_corr   = cr_mlp - cr_ridge
    gap_nonlin_corr  = cr_mlp - cr_pca
    gap_overfit_corr = cr_pca - cr_ridge

    c_ridge   = "#3498db"
    c_pca     = "#27ae60"
    c_mlp     = "#e74c3c"
    c_gap     = "#9b59b6"
    c_nonlin  = "#e67e22"
    c_overfit = "#1abc9c"
    c_ava     = "gold"

    # ── 1) Median R² vs lag + IQR band ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, rr, rp, rm, label in [
        (axes[0], r2_ridge, r2_pca, r2_mlp, "R²"),
        (axes[1], cr_ridge, cr_pca, cr_mlp, "Pearson r"),
    ]:
        med_r = np.median(rr, axis=1)
        med_p = np.median(rp, axis=1)
        med_m = np.median(rm, axis=1)
        q25_r, q75_r = np.percentile(rr, [25, 75], axis=1)
        q25_p, q75_p = np.percentile(rp, [25, 75], axis=1)
        q25_m, q75_m = np.percentile(rm, [25, 75], axis=1)

        ax.fill_between(lags, q25_r, q75_r, alpha=0.12, color=c_ridge)
        ax.fill_between(lags, q25_p, q75_p, alpha=0.12, color=c_pca)
        ax.fill_between(lags, q25_m, q75_m, alpha=0.12, color=c_mlp)
        ax.plot(lags, med_r, "o-", color=c_ridge, lw=2, ms=6, label="Ridge")
        ax.plot(lags, med_p, "D-", color=c_pca,   lw=2, ms=5, label="PCA-Ridge")
        ax.plot(lags, med_m, "s-", color=c_mlp,   lw=2, ms=6, label="MLP")

        ax.set_xlabel("Lag (frames)")
        ax.set_ylabel(f"Median {label}")
        ax.set_title(f"{label} vs lag  (N={N})")
        ax.legend(fontsize=10)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)

    fig.suptitle(f"{worm_name}: Ridge vs PCA-Ridge vs MLP across lags",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_lag_sweep_median.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2) Gap decomposition vs lag ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, g_total, g_nonlin, g_overfit, label in [
        (axes[0], gap_total_r2, gap_nonlin_r2, gap_overfit_r2, "R²"),
        (axes[1], gap_total_corr, gap_nonlin_corr, gap_overfit_corr, "Pearson r"),
    ]:
        med_t = np.median(g_total, axis=1)
        med_n = np.median(g_nonlin, axis=1)
        med_o = np.median(g_overfit, axis=1)

        ax.plot(lags, med_t, "o-", color=c_gap, lw=1.5, ms=5,
                label="MLP − Ridge (total)", alpha=0.5)
        ax.plot(lags, med_o, "D-", color=c_overfit, lw=2, ms=5,
                label="PCA-Ridge − Ridge (overfit)")
        ax.plot(lags, med_n, "s-", color=c_nonlin, lw=2.5, ms=6,
                label="MLP − PCA-Ridge (nonlinearity)")
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set_xlabel("Lag (frames)")
        ax.set_ylabel(f"Median {label} gap")
        ax.set_title(f"{label} gap decomposition")

        # annotate nonlinearity fraction at each lag
        frac_nl = (g_nonlin > 0).mean(axis=1)
        for li, l in enumerate(lags):
            ax.annotate(f"{frac_nl[li]:.0%}",
                        (l, med_n[li]), xytext=(0, 10),
                        textcoords="offset points", fontsize=7,
                        ha="center", color=c_nonlin, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle(f"{worm_name}: gap decomposition vs lag  "
                 f"(orange ↑ = genuine nonlinearity)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_lag_sweep_gap.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3) Per-neuron lag curves: AVA neurons + 5 best + 5 worst ──
    # rank neurons by median-across-lags Ridge R²
    med_ridge_per_neuron = np.median(r2_ridge, axis=0)
    sorted_idx = np.argsort(med_ridge_per_neuron)
    pick = list(sorted_idx[:3]) + list(sorted_idx[-3:])
    pick += list(np.where(ava_mask)[0])
    pick = list(dict.fromkeys(pick))  # dedupe

    n_pick = len(pick)
    ncols = (n_pick + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()
    for ai, ni in enumerate(pick):
        ax = axes_flat[ai]
        ax.plot(lags, r2_ridge[:, ni], "o-", color=c_ridge, lw=1.5,
                ms=5, label="Ridge")
        ax.plot(lags, r2_pca[:, ni], "D-", color=c_pca, lw=1.5,
                ms=4, label="PCA-Ridge")
        ax.plot(lags, r2_mlp[:, ni], "s-", color=c_mlp, lw=1.5,
                ms=5, label="MLP")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.set_xlabel("Lag")
        star = " ★" if ava_mask[ni] else ""
        ax.set_title(f"{neuron_names[ni]}{star}", fontsize=10)
        if ai == 0:
            ax.legend(fontsize=7)
        if ai % ncols == 0:
            ax.set_ylabel("R²")
    # hide unused axes
    for ai in range(n_pick, len(axes_flat)):
        axes_flat[ai].set_visible(False)
    fig.suptitle(f"{worm_name}: per-neuron R² vs lag (worst / best / AVA)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_lag_sweep_per_neuron.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 4) Heatmaps: total gap + nonlinearity gap ─────────────────
    # sort neurons by mean nonlinearity gap (MLP − PCA-Ridge)
    mean_nonlin = gap_nonlin_r2.mean(axis=0)
    order = np.argsort(mean_nonlin)[::-1]
    ordered_names = [neuron_names[i] for i in order]
    ava_y = [yi for yi, n in enumerate(ordered_names) if n.startswith("AVA")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, N * 0.12)))
    for ax, data, title, cblabel in [
        (ax1, gap_total_r2, "MLP − Ridge (total)",
         "MLP R² − Ridge R²"),
        (ax2, gap_nonlin_r2, "MLP − PCA-Ridge (nonlinearity)",
         "MLP R² − PCA-Ridge R²"),
    ]:
        im = ax.imshow(data[:, order].T, aspect="auto",
                       cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                       interpolation="nearest")
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels([str(l) for l in lags])
        ax.set_xlabel("Lag (frames)")
        ax.set_ylabel("Neuron (sorted by nonlinearity)")
        for yy in ava_y:
            ax.annotate(ordered_names[yy], xy=(len(lags) - 0.5, yy),
                        fontsize=6, fontweight="bold", color="gold",
                        va="center")
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(cblabel)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"{worm_name}: R² gap heatmaps  (red = advantage)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{worm_name}_lag_sweep_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Lag-sweep plots saved to {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import time
    pa = argparse.ArgumentParser()
    pa.add_argument("--h5", type=Path, required=True)
    pa.add_argument("--out_dir", type=Path, default=_DEFAULT_OUT)
    pa.add_argument("--n_lags", type=int, default=10)
    pa.add_argument("--lags", type=int, nargs="+", default=None,
                    help="Lag values for sweep (default: 1 2 5 10 20 30 50)")
    pa.add_argument("--causal", default="strict")
    pa.add_argument("--n_folds", type=int, default=5)
    pa.add_argument("--hidden", type=int, default=128)
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--weight_decay", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=15)
    pa.add_argument("--device", default="cpu")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--pca_k", type=int, default=20,
                    help="Number of PCA components for PCA-Ridge (default: 20)")
    pa.add_argument("--skip_fixed", action="store_true",
                    help="Skip the fixed-lag diagnostics, run only lag sweep")
    args = pa.parse_args()

    worm = _load_worm(args.h5)
    if worm is None:
        raise ValueError(f"Cannot load {args.h5}")

    T, N = worm["u"].shape
    print(f"Worm: {worm['name']}  T={T}  N={N}  dt={worm['dt']:.3f}s")

    mlp_kwargs = dict(
        hidden=args.hidden, dropout=args.dropout, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, device_str=args.device, seed=args.seed,
    )

    # ── Fixed-lag diagnostics ─────────────────────────────────────
    if not args.skip_fixed:
        print(f"\n── Fixed-lag diagnostics (lag={args.n_lags}) ──")
        t0 = time.time()
        records = run_diagnostics(worm, args.n_lags, args.causal,
                                  args.n_folds, mlp_kwargs)
        make_diagnostic_plots(records, worm["name"], args.out_dir,
                              args.n_lags, args.n_folds)
        print(f"  ({time.time() - t0:.1f}s)")

    # ── Lag sweep ─────────────────────────────────────────────────
    sweep_lags = args.lags if args.lags else _DEFAULT_LAGS
    print(f"\n── Lag sweep: {sweep_lags} ──")
    t0 = time.time()
    lag_results = run_lag_sweep(worm, sweep_lags, args.causal,
                                args.n_folds, mlp_kwargs,
                                pca_k=args.pca_k)
    make_lag_sweep_plots(lag_results, worm["name"], args.out_dir)
    print(f"  ({time.time() - t0:.1f}s)")

    print(f"\nDone. All outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
