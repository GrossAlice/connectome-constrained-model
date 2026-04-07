#!/usr/bin/env python3
r"""
Connectome-constrained neural activity decoder.

═══════════════════════════════════════════════════════════════════════════
  DESIGN
═══════════════════════════════════════════════════════════════════════════

Like neural_activity_decoder.py, but each target neuron's predictor set
is restricted to its **connectome neighbours** (pre-synaptic partners)
from a single T_*.npy adjacency matrix.

For each T matrix (T_e, T_sv, T_dcv) we run independently:
  - For each target neuron i, find its presynaptic partners j where
    T[j, i] > 0  (column = post-synaptic = target).
  - Build lag features from ONLY those partners (+ optionally self-history).
  - Run Ridge, PCA-Ridge, MLP across K_VALUES lags and 2 conditions:
      • causal + self (AR):     lagged partners + own history
      • causal cross-neuron:    lagged partners only (no self)
  - Also run an "unconstrained" baseline using ALL neurons for comparison.

Plots:
  1. Lag-sweep:  mean R² vs K, constrained vs unconstrained
  2. Per-neuron violin:  R² distribution across neurons
  3. Scatter: constrained vs unconstrained R² per neuron
  4. Connectivity analysis:  R² vs number of presynaptic partners

═══════════════════════════════════════════════════════════════════════════
  USAGE
═══════════════════════════════════════════════════════════════════════════

  python scripts/connectome_neural_decoder.py \
      --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-07-15-12.h5" \
      --T_npy data/used/masks+motor\ neurons/T_e.npy \
      --device cuda

  # Or replot from cached results:
  python scripts/connectome_neural_decoder.py --h5 ... --T_npy ... --replot
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.dataset import load_worm_data

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

_RIDGE_ALPHAS = np.logspace(-4, 6, 30)
K_VALUES = [5]  # reduced from [1, 3, 5, 10, 15, 20] for faster batch runs
N_FOLDS = 5
PCA_K = 20

COND_NAME = {
    "conn+self":   "connectome + self (AR)",
    "conn_only":   "connectome cross-neuron",
    "all+self":    "unconstrained + self",
    "all_only":    "unconstrained cross-neuron",
}

COND_COLOR = {
    "conn+self":   "#e74c3c",
    "conn_only":   "#3498db",
    "all+self":    "#2ecc71",
    "all_only":    "#9b59b6",
}

MODEL_COLOR = {
    "Ridge":     "#3498db",
    "PCA-Ridge": "#27ae60",
    "MLP":       "#e74c3c",
}


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def _pearson_r(a, b):
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _make_folds(T, n_folds):
    sizes = np.full(n_folds, T // n_folds, dtype=int)
    sizes[:T % n_folds] += 1
    folds, cur = [], 0
    for s in sizes:
        te = np.arange(cur, cur + s)
        tr = np.concatenate([np.arange(0, cur), np.arange(cur + s, T)])
        folds.append((tr, te))
        cur += s
    return folds


def _inner_split(tr_idx, frac=0.2):
    n_va = max(1, int(len(tr_idx) * frac))
    return tr_idx[:-n_va], tr_idx[-n_va:]


# ═══════════════════════════════════════════════════════════════════════
#  Connectome mapping
# ═══════════════════════════════════════════════════════════════════════

def load_connectome_adjacency(T_npy_path, worm_labels):
    """Load T matrix and map to worm neuron indices.

    Returns
    -------
    adj : np.ndarray (N_worm, N_worm)
        Sub-matrix of the connectome for the worm's recorded neurons.
        adj[j, i] > 0 means j → i (j is presynaptic to i).
    presynaptic : dict[int, list[int]]
        For each target neuron i, list of presynaptic partner indices
        (in worm-local indexing).
    """
    atlas_names = np.load(
        ROOT / "data/used/masks+motor neurons/neuron_names.npy"
    )
    name_to_atlas = {n: idx for idx, n in enumerate(atlas_names)}
    T_full = np.load(T_npy_path)

    worm_atlas_idx = []
    for lab in worm_labels:
        if lab in name_to_atlas:
            worm_atlas_idx.append(name_to_atlas[lab])
        else:
            worm_atlas_idx.append(-1)

    N = len(worm_labels)
    adj = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            ai, aj = worm_atlas_idx[i], worm_atlas_idx[j]
            if ai >= 0 and aj >= 0:
                adj[j, i] = T_full[aj, ai]  # adj[source, target]

    presynaptic = {}
    for i in range(N):
        pre = [j for j in range(N) if j != i and adj[j, i] > 0]
        presynaptic[i] = pre

    return adj, presynaptic


# ═══════════════════════════════════════════════════════════════════════
#  Feature construction
# ═══════════════════════════════════════════════════════════════════════

def _precompute_full(u, n_lags):
    """Block 0 = concurrent (t), blocks 1..L = lags (t-1 .. t-L).

    Returns X_full (T_out, N*(L+1)),  Y_all (T_out, N).
    """
    T, N = u.shape
    T_out = T - n_lags
    blocks = [u[n_lags:n_lags + T_out]]
    for k in range(1, n_lags + 1):
        blocks.append(u[n_lags - k:n_lags - k + T_out])
    return (np.concatenate(blocks, axis=1).astype(np.float32),
            u[n_lags:n_lags + T_out].astype(np.float32))


def _connectome_feature_mask(N, n_lags, ni, partners, include_self):
    """Boolean mask over X_full columns for neuron ni using only connectome partners.

    X_full layout: [concurrent(N), lag1(N), lag2(N), ..., lagL(N)]
    We use NO concurrent block (causal only), only lags 1..L.
    Partners are used from lag blocks, self from lag blocks if include_self.
    """
    total = N * (n_lags + 1)
    mask = np.zeros(total, dtype=bool)
    partner_set = set(partners)
    for k in range(1, n_lags + 1):
        offset = k * N
        for j in partner_set:
            mask[offset + j] = True
        if include_self:
            mask[offset + ni] = True
    return mask


def _unconstrained_feature_mask(N, n_lags, ni, include_self):
    """Unconstrained baseline: all neurons from lag blocks (causal only)."""
    total = N * (n_lags + 1)
    mask = np.zeros(total, dtype=bool)
    for k in range(1, n_lags + 1):
        offset = k * N
        mask[offset:offset + N] = True
        if not include_self:
            mask[offset + ni] = False
    return mask


# ═══════════════════════════════════════════════════════════════════════
#  Ridge + PCA-Ridge  (per neuron)
# ═══════════════════════════════════════════════════════════════════════

def _eval_ridge_pca_neuron(X_full, Y_all, ni, mask, folds, pca_k):
    """Ridge + PCA-Ridge CV for one neuron given a feature mask.

    Returns (r2_ridge, corr_ridge, r2_pca, corr_pca).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.decomposition import PCA

    X, y = X_full[:, mask], Y_all[:, ni]
    if X.shape[1] == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    T_out = X.shape[0]
    k = min(pca_k, X.shape[1])

    pred_r = np.zeros(T_out, np.float32)
    pred_p = np.zeros(T_out, np.float32)

    for tr, te in folds:
        mu, sig = X[tr].mean(0), X[tr].std(0).clip(1e-8)
        Xtr_z, Xte_z = (X[tr] - mu) / sig, (X[te] - mu) / sig

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
            r.fit(Xtr_z, y[tr])
        pred_r[te] = r.predict(Xte_z).astype(np.float32)

        ke = min(k, len(tr) - 1, Xtr_z.shape[1])
        if ke < 1:
            pred_p[te] = pred_r[te]
            continue
        pca = PCA(n_components=ke)
        Ztr = pca.fit_transform(Xtr_z)
        Zte = pca.transform(Xte_z)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rp = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
            rp.fit(Ztr, y[tr])
        pred_p[te] = rp.predict(Zte).astype(np.float32)

    return (_r2(y, pred_r), _pearson_r(y, pred_r),
            _r2(y, pred_p), _pearson_r(y, pred_p))


# ═══════════════════════════════════════════════════════════════════════
#  MLP  (per neuron)
# ═══════════════════════════════════════════════════════════════════════

def _make_mlp(d_in, hidden=128, n_layers=2):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


def _eval_mlp_neuron(X_full, Y_all, ni, mask, folds, device,
                     epochs=150, lr=1e-3, wd=1e-3, patience=20):
    """MLP CV for one neuron given a feature mask. Returns (r2, corr)."""
    X, y = X_full[:, mask], Y_all[:, ni]
    if X.shape[1] == 0:
        return (np.nan, np.nan)
    T_out, d_in = X.shape
    pred = np.zeros(T_out, np.float32)

    for tr, te in folds:
        tri, vai = _inner_split(tr)
        mu, sig = X[tri].mean(0), X[tri].std(0).clip(1e-8)
        mu_y, sig_y = float(y[tri].mean()), float(max(y[tri].std(), 1e-8))

        Xtr = torch.tensor(((X[tri] - mu) / sig), dtype=torch.float32, device=device)
        Ytr = torch.tensor(((y[tri] - mu_y) / sig_y).reshape(-1, 1), dtype=torch.float32, device=device)
        Xva = torch.tensor(((X[vai] - mu) / sig), dtype=torch.float32, device=device)
        Yva = torch.tensor(((y[vai] - mu_y) / sig_y).reshape(-1, 1), dtype=torch.float32, device=device)

        net = _make_mlp(d_in).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        bvl, bs, pat = float("inf"), None, 0
        for _ in range(epochs):
            net.train()
            loss = F.mse_loss(net(Xtr), Ytr)
            opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = F.mse_loss(net(Xva), Yva).item()
            if vl < bvl - 1e-6:
                bvl, bs, pat = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                pat += 1
            if pat > patience:
                break
        if bs:
            net.load_state_dict(bs)
        net.eval()
        with torch.no_grad():
            pz = net(torch.tensor(((X[te] - mu) / sig), dtype=torch.float32, device=device))
            pred[te] = (pz.cpu().numpy().ravel() * sig_y + mu_y).astype(np.float32)

    return _r2(y, pred), _pearson_r(y, pred)


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def _plot_lag_sweep(results, out, worm_id, N, T_name):
    """Mean R² & Corr vs K, constrained vs unconstrained."""
    ks = sorted(int(k) for k in results.keys())
    models = ["Ridge", "PCA-Ridge", "MLP"]
    markers = {"Ridge": "o", "PCA-Ridge": "D", "MLP": "s"}
    cond_ls = {
        "conn+self": "-",   "conn_only": "--",
        "all+self":  "-.",  "all_only":  ":",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric, ylabel in [
        (axes[0], "r2", "Mean R²"),
        (axes[1], "corr", "Mean Pearson r"),
    ]:
        for model in models:
            mkey = model.lower().replace("-", "_")
            for ck in ["conn+self", "conn_only", "all+self", "all_only"]:
                field = f"{metric}_mean_{mkey}"
                vals = []
                for K in ks:
                    res_k = results.get(str(K), {})
                    vals.append(res_k.get(ck, {}).get(field, np.nan))
                if all(np.isnan(v) for v in vals):
                    continue
                label = f"{model} {COND_NAME[ck]}"
                ax.plot(ks, vals, marker=markers[model],
                        color=MODEL_COLOR[model], ls=cond_ls[ck],
                        lw=1.8, ms=6, label=label, alpha=0.85)
        ax.set_xlabel("Context length K (lags)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric == "corr":
            ax.legend(fontsize=6, ncol=2, loc="best")

    fig.suptitle(f"Connectome-Constrained Decoder — {T_name}\n"
                 f"{worm_id}  (N={N}, {N_FOLDS}-fold CV)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = out / f"lag_sweep_{worm_id}_{T_name}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


def _plot_condition_comparison(results, out, worm_id, N, T_name):
    """PCA-Ridge R²/Corr vs lag, one line per condition."""
    ks = sorted(int(k) for k in results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in [
        (axes[0], "r2", "Mean R² (PCA-Ridge)"),
        (axes[1], "corr", "Mean Pearson r (PCA-Ridge)"),
    ]:
        for ck in ["conn+self", "conn_only", "all+self", "all_only"]:
            field = f"{metric}_mean_pca_ridge"
            vals = [results.get(str(K), {}).get(ck, {}).get(field, np.nan) for K in ks]
            if all(np.isnan(v) for v in vals):
                continue
            ax.plot(ks, vals, "o-", color=COND_COLOR[ck], lw=2, ms=7,
                    label=COND_NAME[ck])
        ax.set_xlabel("Context length K (lags)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{worm_id}: Connectome vs Unconstrained — {T_name} (N={N})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = out / f"condition_comparison_{worm_id}_{T_name}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


def _plot_scatter_constrained_vs_unconstrained(results, out, worm_id, N, T_name):
    """Scatter: per-neuron R² (connectome) vs R² (unconstrained) for PCA-Ridge."""
    ks = sorted(int(k) for k in results.keys())
    best_K = None
    best_mean = -999
    for K in ks:
        r = results.get(str(K), {}).get("conn+self", {})
        v = r.get("r2_mean_pca_ridge", -999)
        if v > best_mean:
            best_mean = v
            best_K = K
    if best_K is None:
        return

    sk = str(best_K)
    for inc_self, ck_conn, ck_all in [
        (True, "conn+self", "all+self"),
        (False, "conn_only", "all_only"),
    ]:
        conn_d = results.get(sk, {}).get(ck_conn, {})
        all_d = results.get(sk, {}).get(ck_all, {})
        r2_conn = conn_d.get("r2_per_neuron_pca_ridge")
        r2_all = all_d.get("r2_per_neuron_pca_ridge")
        if r2_conn is None or r2_all is None:
            continue

        # Filter out None values
        r2_conn = np.array([v if v is not None else np.nan for v in r2_conn])
        r2_all = np.array([v if v is not None else np.nan for v in r2_all])
        
        # Skip if no valid values
        valid = np.isfinite(r2_conn) & np.isfinite(r2_all)
        if not np.any(valid):
            continue
            
        lbl = "with self" if inc_self else "no self"

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(r2_all[valid], r2_conn[valid], s=25, alpha=0.6, 
                   color=COND_COLOR[ck_conn],
                   edgecolors="white", linewidths=0.3)
        lim = [min(np.nanmin(r2_all), np.nanmin(r2_conn)) - 0.05,
               max(np.nanmax(r2_all), np.nanmax(r2_conn)) + 0.05]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"R² unconstrained ({lbl})", fontsize=11)
        ax.set_ylabel(f"R² connectome-constrained ({lbl})", fontsize=11)
        ax.set_title(f"{worm_id} — {T_name}, K={best_K} (PCA-Ridge)\n"
                     f"mean conn={np.nanmean(r2_conn):.3f}, all={np.nanmean(r2_all):.3f}",
                     fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        tag = "self" if inc_self else "cross"
        fname = out / f"scatter_{tag}_{worm_id}_{T_name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {fname}")


def _plot_r2_vs_n_partners(results, out, worm_id, presynaptic, N, T_name):
    """R² vs number of presynaptic partners (connectivity degree)."""
    ks = sorted(int(k) for k in results.keys())
    best_K = None
    best_mean = -999
    for K in ks:
        r = results.get(str(K), {}).get("conn+self", {})
        v = r.get("r2_mean_pca_ridge", -999)
        if v > best_mean:
            best_mean = v
            best_K = K
    if best_K is None:
        return

    sk = str(best_K)
    n_partners = np.array([len(presynaptic.get(i, [])) for i in range(N)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ck, title in [
        (axes[0], "conn+self", "connectome + self"),
        (axes[1], "conn_only", "connectome cross-neuron"),
    ]:
        r2_arr = results.get(sk, {}).get(ck, {}).get("r2_per_neuron_pca_ridge")
        if r2_arr is None:
            continue
        r2_arr = np.array([v if v is not None else np.nan for v in r2_arr],
                          dtype=np.float64)
        valid = np.isfinite(r2_arr)
        sc = ax.scatter(n_partners[valid], r2_arr[valid], s=30, alpha=0.6,
                        c=COND_COLOR[ck], edgecolors="white", linewidths=0.3)
        # Trend line
        if valid.sum() > 3:
            z = np.polyfit(n_partners[valid], r2_arr[valid], 1)
            xs = np.linspace(n_partners.min(), n_partners.max(), 100)
            ax.plot(xs, np.polyval(z, xs), "k--", lw=1.2, alpha=0.6,
                    label=f"slope={z[0]:.4f}")
            ax.legend(fontsize=9)
        ax.set_xlabel("# presynaptic partners", fontsize=11)
        ax.set_ylabel("R² (PCA-Ridge)", fontsize=11)
        ax.set_title(f"{title}, K={best_K}", fontsize=11, fontweight="bold")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{worm_id}: R² vs Connectivity Degree — {T_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = out / f"r2_vs_degree_{worm_id}_{T_name}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


def _plot_violin_per_neuron(results, out, worm_id, N, T_name, neuron_labels=None):
    """Violin plot: constrained vs unconstrained per neuron (PCA-Ridge at best K)."""
    ks = sorted(int(k) for k in results.keys())
    # Use K=5 if available, else best K
    use_K = 5 if "5" in results else ks[-1] if ks else None
    if use_K is None:
        return
    sk = str(use_K)

    for inc_self, ck_conn, ck_all, tag in [
        (True,  "conn+self", "all+self",  "self"),
        (False, "conn_only", "all_only", "cross"),
    ]:
        conn_r2 = results.get(sk, {}).get(ck_conn, {}).get("r2_per_neuron_pca_ridge")
        all_r2 = results.get(sk, {}).get(ck_all, {}).get("r2_per_neuron_pca_ridge")
        if conn_r2 is None or all_r2 is None:
            continue
        conn_r2 = np.array([v if v is not None else np.nan for v in conn_r2],
                           dtype=np.float64)
        all_r2 = np.array([v if v is not None else np.nan for v in all_r2],
                          dtype=np.float64)

        fig, ax = plt.subplots(figsize=(min(40, max(14, N * 0.4)), 6))
        x = np.arange(N)
        w = 0.35
        ax.bar(x - w/2, all_r2, w, color=COND_COLOR[ck_all], alpha=0.6,
               label=f"Unconstrained ({tag})")
        ax.bar(x + w/2, conn_r2, w, color=COND_COLOR[ck_conn], alpha=0.6,
               label=f"Connectome ({tag})")

        xlabels = neuron_labels if neuron_labels and len(neuron_labels) == N \
            else [f"n{i}" for i in range(N)]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels,
                           fontsize=max(5, min(7, 180 // N)),
                           rotation=45, ha="right")
        ax.set_ylabel("R² (PCA-Ridge)", fontsize=11)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{worm_id} — {T_name}, K={use_K}\n"
                     f"mean conn={conn_r2.mean():.3f}, all={all_r2.mean():.3f}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fname = out / f"bar_{tag}_{worm_id}_{T_name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {fname}")


def _plot_delta_r2_bar(results, out, worm_id, N, T_name, neuron_labels=None):
    """Bar chart of ΔR² = R²(connectome) − R²(unconstrained) per neuron."""
    ks = sorted(int(k) for k in results.keys())
    use_K = 5 if "5" in results else ks[-1] if ks else None
    if use_K is None:
        return
    sk = str(use_K)

    for tag, ck_conn, ck_all in [
        ("self", "conn+self", "all+self"),
        ("cross", "conn_only", "all_only"),
    ]:
        conn_r2 = results.get(sk, {}).get(ck_conn, {}).get("r2_per_neuron_pca_ridge")
        all_r2 = results.get(sk, {}).get(ck_all, {}).get("r2_per_neuron_pca_ridge")
        if conn_r2 is None or all_r2 is None:
            continue
        conn_r2 = np.array([v if v is not None else np.nan for v in conn_r2],
                           dtype=np.float64)
        all_r2 = np.array([v if v is not None else np.nan for v in all_r2],
                          dtype=np.float64)
        delta = conn_r2 - all_r2
        order = np.argsort(delta)

        fig, ax = plt.subplots(figsize=(min(40, max(14, N * 0.35)), 5))
        xlabels = neuron_labels if neuron_labels and len(neuron_labels) == N \
            else [f"n{i}" for i in range(N)]
        colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in delta[order]]
        ax.bar(range(N), delta[order], color=colors, alpha=0.7)
        ax.set_xticks(range(N))
        ax.set_xticklabels([xlabels[i] for i in order],
                           fontsize=max(5, min(7, 180 // N)),
                           rotation=45, ha="right")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("ΔR² (connectome − unconstrained)", fontsize=11)
        ax.set_title(f"{worm_id} — {T_name}, K={use_K} ({tag})\n"
                     f"median ΔR²={np.median(delta):.4f}",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fname = out / f"delta_r2_{tag}_{worm_id}_{T_name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {fname}")


def _plot_connectivity_heatmap(adj, out, worm_id, T_name, neuron_labels=None):
    """Heatmap of the connectome sub-matrix for this worm."""
    N = adj.shape[0]
    fig, ax = plt.subplots(figsize=(min(20, max(8, N * 0.15)),
                                    min(20, max(8, N * 0.15))))
    im = ax.imshow(adj, aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8)
    if neuron_labels and len(neuron_labels) == N and N <= 100:
        fs = max(4, min(6, 150 // N))
        ax.set_xticks(range(N))
        ax.set_xticklabels(neuron_labels, fontsize=fs, rotation=90)
        ax.set_yticks(range(N))
        ax.set_yticklabels(neuron_labels, fontsize=fs)
    ax.set_xlabel("Post-synaptic (target)")
    ax.set_ylabel("Pre-synaptic (source)")
    ax.set_title(f"{worm_id} — {T_name} adjacency\n"
                 f"(N={N}, nnz={np.count_nonzero(adj)})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = out / f"adjacency_{worm_id}_{T_name}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--T_npy", required=True,
                    help="Path to T_*.npy connectome adjacency matrix")
    ap.add_argument("--out_dir", default="output_plots/connectome_decoder")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--neurons", default="motor",
                    choices=["motor", "nonmotor", "all"])
    ap.add_argument("--K_values", nargs="+", type=int, default=None,
                    help="Context lengths to evaluate (default: [5])")
    args = ap.parse_args()

    # Override K_VALUES if specified
    global K_VALUES
    if args.K_values:
        K_VALUES = sorted(args.K_values)

    device = args.device
    T_npy_path = Path(args.T_npy)
    T_name = T_npy_path.stem  # e.g. "T_e", "T_sv", "T_dcv"

    # ── Load data ─────────────────────────────────────────────────
    worm_data = load_worm_data(args.h5, n_beh_modes=6)
    u_all = worm_data["u"]
    worm_id = worm_data["worm_id"]
    motor_idx = worm_data.get("motor_idx")
    labels = worm_data.get("labels", [])
    N_total = u_all.shape[1]

    if motor_idx is None or len(motor_idx) == 0:
        print(f"ERROR: no motor neurons found for {worm_id}")
        sys.exit(1)

    motor_set = set(motor_idx)
    if args.neurons == "motor":
        sel_idx = sorted(motor_idx)
        subset_tag = "motor"
    elif args.neurons == "nonmotor":
        sel_idx = sorted(i for i in range(N_total) if i not in motor_set)
        subset_tag = "nonmotor"
    else:
        sel_idx = list(range(N_total))
        subset_tag = "all"

    if not sel_idx:
        print(f"ERROR: no {args.neurons} neurons found for {worm_id}")
        sys.exit(1)

    sel_labels = [labels[i] if i < len(labels) else f"n{i}" for i in sel_idx]
    u_sel = u_all[:, sel_idx].astype(np.float32)
    T_total, N = u_sel.shape

    # ── Load connectome ───────────────────────────────────────────
    # We need the full worm's labels for atlas mapping, then sub-index
    adj_full, pre_full = load_connectome_adjacency(T_npy_path, labels)
    # Sub-matrix for selected neurons
    # Map sel_idx -> new local indices
    sel_set = set(sel_idx)
    old_to_new = {old: new for new, old in enumerate(sel_idx)}
    adj = adj_full[np.ix_(sel_idx, sel_idx)]
    presynaptic = {}
    for new_i, old_i in enumerate(sel_idx):
        pre = [old_to_new[j] for j in pre_full.get(old_i, [])
               if j in old_to_new]
        presynaptic[new_i] = pre

    n_partners = [len(presynaptic[i]) for i in range(N)]
    n_isolated = sum(1 for np_ in n_partners if np_ == 0)
    print(f"Worm: {worm_id}  subset={subset_tag}  T={T_total}  N={N}")
    print(f"Connectome: {T_name}  nnz(sub)={np.count_nonzero(adj)}  "
          f"density={np.count_nonzero(adj)/(N*N):.3f}")
    print(f"  partners: min={min(n_partners)}, max={max(n_partners)}, "
          f"mean={np.mean(n_partners):.1f}, isolated={n_isolated}")

    out = Path(args.out_dir) / f"{worm_id}_{subset_tag}" / T_name
    out.mkdir(parents=True, exist_ok=True)
    worm_id_full = f"{worm_id}_{subset_tag}"

    results_path = out / "results.json"

    # ── Plot connectome heatmap ───────────────────────────────────
    _plot_connectivity_heatmap(adj, out, worm_id_full, T_name, sel_labels)

    # ── Replot mode ───────────────────────────────────────────────
    if args.replot:
        if not results_path.exists():
            print(f"No results.json at {results_path}"); sys.exit(1)
        with open(results_path) as f:
            results = json.load(f)
        _plot_lag_sweep(results, out, worm_id_full, N, T_name)
        _plot_condition_comparison(results, out, worm_id_full, N, T_name)
        _plot_scatter_constrained_vs_unconstrained(results, out, worm_id_full, N, T_name)
        _plot_r2_vs_n_partners(results, out, worm_id_full, presynaptic, N, T_name)
        _plot_violin_per_neuron(results, out, worm_id_full, N, T_name, sel_labels)
        _plot_delta_r2_bar(results, out, worm_id_full, N, T_name, sel_labels)
        print("Replot done."); return

    # ── Load cached results ───────────────────────────────────────
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} previous K results")

    run_mlp = not args.no_mlp

    from joblib import Parallel, delayed

    t_total = time.time()

    for K in K_VALUES:
        sk = str(K)

        # Check if fully done
        existing = results.get(sk, {})
        all_conds = ["conn+self", "conn_only", "all+self", "all_only"]
        all_done = True
        for ck in all_conds:
            cd = existing.get(ck, {})
            if "r2_mean_ridge" not in cd:
                all_done = False; break
            if run_mlp and "r2_mean_mlp" not in cd:
                all_done = False; break
        if all_done:
            print(f"\n  K={K} already done — skipping"); continue

        print(f"\n{'='*60}\n  Context length K = {K}\n{'='*60}")

        T_out = T_total - K
        if T_out < 100:
            print(f"  [skip K={K}: T_out={T_out} too small]"); continue

        X_full, Y_all = _precompute_full(u_sel, K)
        folds = _make_folds(T_out, N_FOLDS)
        res_k = dict(existing)

        # ── For each condition ────────────────────────────────
        for ck, get_mask_fn in [
            ("conn+self",
             lambda ni: _connectome_feature_mask(N, K, ni, presynaptic[ni], True)),
            ("conn_only",
             lambda ni: _connectome_feature_mask(N, K, ni, presynaptic[ni], False)),
            ("all+self",
             lambda ni: _unconstrained_feature_mask(N, K, ni, True)),
            ("all_only",
             lambda ni: _unconstrained_feature_mask(N, K, ni, False)),
        ]:
            cn = COND_NAME[ck]
            cond_d = dict(res_k.get(ck, {}))

            # Compute feature dims
            dims = [int(get_mask_fn(ni).sum()) for ni in range(N)]
            d_min, d_max, d_mean = min(dims), max(dims), np.mean(dims)
            print(f"\n  K={K}  {cn:35s}  dim=[{d_min}..{d_max}] mean={d_mean:.0f}")

            # Ridge + PCA-Ridge
            if "r2_mean_ridge" not in cond_d:
                t0 = time.time()
                rp = Parallel(n_jobs=min(8, N), verbose=0)(
                    delayed(_eval_ridge_pca_neuron)(
                        X_full, Y_all, ni, get_mask_fn(ni), folds, PCA_K)
                    for ni in range(N))
                t_rp = time.time() - t0

                r2_ridge = [r[0] for r in rp]
                corr_ridge = [r[1] for r in rp]
                r2_pca = [r[2] for r in rp]
                corr_pca = [r[3] for r in rp]

                cond_d["r2_mean_ridge"] = float(np.nanmean(r2_ridge))
                cond_d["corr_mean_ridge"] = float(np.nanmean(corr_ridge))
                cond_d["r2_mean_pca_ridge"] = float(np.nanmean(r2_pca))
                cond_d["corr_mean_pca_ridge"] = float(np.nanmean(corr_pca))
                cond_d["r2_per_neuron_ridge"] = [float(v) if np.isfinite(v) else None for v in r2_ridge]
                cond_d["corr_per_neuron_ridge"] = [float(v) if np.isfinite(v) else None for v in corr_ridge]
                cond_d["r2_per_neuron_pca_ridge"] = [float(v) if np.isfinite(v) else None for v in r2_pca]
                cond_d["corr_per_neuron_pca_ridge"] = [float(v) if np.isfinite(v) else None for v in corr_pca]
                cond_d["n_partners"] = [len(presynaptic[i]) for i in range(N)]

                print(f"    Ridge  mean R²={np.nanmean(r2_ridge):+.4f}  "
                      f"corr={np.nanmean(corr_ridge):.4f}")
                print(f"    PCA    mean R²={np.nanmean(r2_pca):+.4f}  "
                      f"corr={np.nanmean(corr_pca):.4f}  [{t_rp:.0f}s]")
            else:
                print(f"    Ridge/PCA: cached  "
                      f"R²={cond_d['r2_mean_ridge']:+.4f} / "
                      f"{cond_d['r2_mean_pca_ridge']:+.4f}")

            # MLP
            if run_mlp and "r2_mean_mlp" not in cond_d:
                t0 = time.time()
                mlp_r2, mlp_corr = [], []
                for ni in range(N):
                    mask = get_mask_fn(ni)
                    if mask.sum() == 0:
                        mlp_r2.append(np.nan)
                        mlp_corr.append(np.nan)
                        continue
                    r2m, cm = _eval_mlp_neuron(
                        X_full, Y_all, ni, mask, folds, device)
                    mlp_r2.append(r2m)
                    mlp_corr.append(cm)
                    if (ni + 1) % 10 == 0:
                        print(f"      MLP: {ni+1}/{N}")
                t_mlp = time.time() - t0

                cond_d["r2_mean_mlp"] = float(np.nanmean(mlp_r2))
                cond_d["corr_mean_mlp"] = float(np.nanmean(mlp_corr))
                cond_d["r2_per_neuron_mlp"] = [float(v) if np.isfinite(v) else None for v in mlp_r2]
                cond_d["corr_per_neuron_mlp"] = [float(v) if np.isfinite(v) else None for v in mlp_corr]

                print(f"    MLP    mean R²={np.nanmean(mlp_r2):+.4f}  "
                      f"corr={np.nanmean(mlp_corr):.4f}  [{t_mlp:.0f}s]")
            elif run_mlp:
                print(f"    MLP: cached  R²={cond_d['r2_mean_mlp']:+.4f}")

            res_k[ck] = cond_d

        results[sk] = res_k

        # Checkpoint
        _js = json.dumps(results, indent=2,
                         default=lambda o: int(o) if isinstance(o, np.integer)
                         else float(o) if isinstance(o, np.floating) else o)
        with open(results_path, "w") as f:
            f.write(_js)
        print(f"\n  Checkpoint saved ({results_path})")

    total_s = time.time() - t_total
    print(f"\nTotal time: {total_s:.0f}s ({total_s/60:.1f}min)")

    # ── Generate all plots ────────────────────────────────────
    print("\nGenerating plots...")
    _plot_lag_sweep(results, out, worm_id_full, N, T_name)
    _plot_condition_comparison(results, out, worm_id_full, N, T_name)
    _plot_scatter_constrained_vs_unconstrained(results, out, worm_id_full, N, T_name)
    _plot_r2_vs_n_partners(results, out, worm_id_full, presynaptic, N, T_name)
    _plot_violin_per_neuron(results, out, worm_id_full, N, T_name, sel_labels)
    _plot_delta_r2_bar(results, out, worm_id_full, N, T_name, sel_labels)

    print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
