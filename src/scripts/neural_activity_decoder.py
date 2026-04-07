#!/usr/bin/env python3
r"""
Neural activity decoder — predict motor-neuron activity from other motor neurons.

═══════════════════════════════════════════════════════════════════════════
  DESIGN
═══════════════════════════════════════════════════════════════════════════

Analogous to *behaviour decoder models.py* but predicts **neural activity**
instead of behaviour.  No behaviour features are used anywhere.

2×2 factorial conditions (concurrent × include_self):

  ┌─────────────┬──────────────┬─────────────────────────────────────┐
  │ concurrent  │ include_self │ Interpretation                      │
  ├─────────────┼──────────────┼─────────────────────────────────────┤
  │ True        │ True         │ population snapshot + own history    │
  │ True        │ False        │ population snapshot only             │
  │ False       │ True         │ causal + autoregressive              │
  │ False       │ False        │ pure causal cross-neuron             │
  └─────────────┴──────────────┴─────────────────────────────────────┘

Models:
  1. Ridge   (RidgeCV, logspace boundary check)
  2. PCA-Ridge   (PCA → RidgeCV)
  3. MLP    (2×128, full-batch Adam, early-stop)
  4. Transformer (TemporalTransformerGaussian, causal conditions only)

═══════════════════════════════════════════════════════════════════════════
  USAGE
═══════════════════════════════════════════════════════════════════════════

  python scripts/neural_activity_decoder.py \
      --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-07-15-12.h5" \
      --device cuda

  python scripts/neural_activity_decoder.py --h5 "..." --replot
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

from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.dataset import load_worm_data
from baseline_transformer.model import TemporalTransformerGaussian

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

_RIDGE_ALPHAS = np.logspace(-4, 6, 30)
K_VALUES = [1, 3, 5, 10, 15, 20]
N_FOLDS = 5
PCA_K = 20

ALL_CONDITIONS = [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]

COND_NAME = {
    (True,  True):  "concurrent + self",
    (True,  False): "concurrent only",
    (False, True):  "causal + self (AR)",
    (False, False): "causal cross-neuron",
}

COND_KEY = {
    (True,  True):  "conc+self",
    (True,  False): "conc_only",
    (False, True):  "self_only",
    (False, False): "cross_only",
}

KEY_TO_COND = {v: k for k, v in COND_KEY.items()}

COND_COLOR = {
    "conc+self":  "#e74c3c",
    "conc_only":  "#3498db",
    "self_only":  "#2ecc71",
    "cross_only": "#9b59b6",
}

MODEL_COLOR = {
    "Ridge":     "#3498db",
    "PCA-Ridge": "#27ae60",
    "MLP":       "#e74c3c",
    "TRF":       "#ff7f0e",
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
    """Contiguous temporal folds over T_out samples."""
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
#  Feature construction  (same as masked_neuron_lag_sweep)
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


def _feature_mask(N, n_lags, ni, concurrent, include_self):
    """Boolean mask over X_full columns for neuron ni."""
    total = N * (n_lags + 1)
    mask = np.ones(total, dtype=bool)
    if not concurrent:
        mask[:N] = False
    else:
        mask[ni] = False                       # target never concurrent
    if not include_self:
        for k in range(1, n_lags + 1):
            mask[k * N + ni] = False
    return mask


def _feature_dim(N, n_lags, concurrent, include_self):
    if concurrent and include_self:
        return (N - 1) + N * n_lags
    elif concurrent and not include_self:
        return (N - 1) * (n_lags + 1)
    elif not concurrent and include_self:
        return N * n_lags
    else:
        return (N - 1) * n_lags


# ═══════════════════════════════════════════════════════════════════════
#  Ridge + PCA-Ridge  (vectorised per-neuron, parallel via joblib)
# ═══════════════════════════════════════════════════════════════════════

def _check_alpha_boundary(alpha, tag=""):
    """Warn if selected alpha is at the boundary of the log grid."""
    lo, hi = _RIDGE_ALPHAS[0], _RIDGE_ALPHAS[-1]
    if np.isclose(alpha, lo, rtol=1e-3):
        print(f"  ⚠ {tag}: alpha={alpha:.2e} hit LOWER boundary")
    if np.isclose(alpha, hi, rtol=1e-3):
        print(f"  ⚠ {tag}: alpha={alpha:.2e} hit UPPER boundary")


def _eval_ridge_pca_neuron(X_full, Y_all, ni, N, n_lags,
                           conc, incself, folds, pca_k):
    """Ridge + PCA-Ridge CV for one neuron.

    Returns (r2_ridge, corr_ridge, r2_pca, corr_pca,
             n_boundary_ridge, n_boundary_pca).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.decomposition import PCA

    mask = _feature_mask(N, n_lags, ni, conc, incself)
    X, y = X_full[:, mask], Y_all[:, ni]
    T_out = X.shape[0]
    k = min(pca_k, X.shape[1])

    pred_r = np.zeros(T_out, np.float32)
    pred_p = np.zeros(T_out, np.float32)
    bdry_r, bdry_p = 0, 0
    lo, hi = _RIDGE_ALPHAS[0], _RIDGE_ALPHAS[-1]

    for tr, te in folds:
        # z-score from train
        mu, sig = X[tr].mean(0), X[tr].std(0).clip(1e-8)
        Xtr_z, Xte_z = (X[tr] - mu) / sig, (X[te] - mu) / sig

        # Raw Ridge
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
            r.fit(Xtr_z, y[tr])
        pred_r[te] = r.predict(Xte_z).astype(np.float32)
        if np.isclose(r.alpha_, lo, rtol=1e-3) or np.isclose(r.alpha_, hi, rtol=1e-3):
            bdry_r += 1

        # PCA-Ridge
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
        if np.isclose(rp.alpha_, lo, rtol=1e-3) or np.isclose(rp.alpha_, hi, rtol=1e-3):
            bdry_p += 1

    return (_r2(y, pred_r), _pearson_r(y, pred_r),
            _r2(y, pred_p), _pearson_r(y, pred_p),
            bdry_r, bdry_p)


# ═══════════════════════════════════════════════════════════════════════
#  MLP  (per-neuron, sequential)
# ═══════════════════════════════════════════════════════════════════════

def _make_mlp(d_in, hidden=128, n_layers=2):
    """Create MLP with LayerNorm and Dropout (matches behaviour decoder)."""
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


def _eval_mlp_neuron(X_full, Y_all, ni, N, n_lags,
                     conc, incself, folds, device,
                     epochs=150, lr=1e-3, wd=1e-3, patience=20):
    """MLP CV for one neuron. Returns (r2, corr)."""
    mask = _feature_mask(N, n_lags, ni, conc, incself)
    X, y = X_full[:, mask], Y_all[:, ni]
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
#  Transformer  (one model per fold → all N neurons, causal only)
# ═══════════════════════════════════════════════════════════════════════

def _build_windows(u_n, indices, K):
    return np.stack([u_n[idx:idx + K] for idx in indices])


def _train_trf_fold(X_tr, Y_tr, X_va, Y_va, N, K, device,
                    d_model=256, n_layers=2, n_heads=8,
                    epochs=150, lr=1e-3, wd=1e-4, patience=20):
    """Train transformer with B_wide_256h8 config (matches behaviour decoder)."""
    cfg = TransformerBaselineConfig()
    cfg.context_length = K
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.n_heads = n_heads
    cfg.d_ff = 512
    cfg.dropout = 0.1

    model = TemporalTransformerGaussian(n_neural=N, n_beh=0, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=device)
    Yv = torch.tensor(Y_va, dtype=torch.float32, device=device)

    bvl, bs, pat = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        loss = F.mse_loss(model.predict_mean(Xt), Yt)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model.predict_mean(Xv), Yv).item()
        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
        if pat > patience:
            break
    if bs:
        model.load_state_dict(bs)
    model.eval()
    return model


def _eval_transformer_causal(u, n_lags, folds, device, trf_kw=None):
    """Transformer for both causal conditions (self_only + cross_only).

    One model per fold predicts all N neurons from K past frames.
    - self_only:  direct forward pass (context has all N incl. target history)
    - cross_only: zero-out target column (mean imputation in z-space)

    Returns dict {cond_key: (r2_arr(N,), corr_arr(N,))}
    """
    kw = trf_kw or {}
    T, N = u.shape
    T_out = T - n_lags
    pred_self  = np.zeros((T_out, N), dtype=np.float32)
    pred_cross = np.zeros((T_out, N), dtype=np.float32)

    for fi, (tr, te) in enumerate(folds):
        tri, vai = _inner_split(tr)
        mu = u[n_lags + tri].mean(0)
        sig = u[n_lags + tri].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)

        X_tri = _build_windows(u_n, tri, n_lags)
        Y_tri = u_n[n_lags + tri]
        X_vai = _build_windows(u_n, vai, n_lags)
        Y_vai = u_n[n_lags + vai]

        model = _train_trf_fold(X_tri, Y_tri, X_vai, Y_vai,
                                N, n_lags, device, **kw)

        # causal + self: direct forward pass
        X_te_np = _build_windows(u_n, te, n_lags)
        X_te = torch.tensor(X_te_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_z = model.predict_mean(X_te).cpu().numpy()
        pred_self[te] = pred_z * sig + mu

        # causal cross-neuron: zero-out each target column
        for ni in range(N):
            X_m = X_te_np.copy()
            X_m[:, :, ni] = 0.0          # mean imputation in z-space
            with torch.no_grad():
                p = model.predict_mean(
                    torch.tensor(X_m, dtype=torch.float32,
                                 device=device)).cpu().numpy()
            pred_cross[te, ni] = p[:, ni] * sig[ni] + mu[ni]

        model.cpu()
        print(f"    TRF fold {fi+1}/{len(folds)} done")

    Y_raw = u[n_lags:]
    r2_s  = np.array([_r2(Y_raw[:, i], pred_self[:, i])  for i in range(N)])
    co_s  = np.array([_pearson_r(Y_raw[:, i], pred_self[:, i])  for i in range(N)])
    r2_x  = np.array([_r2(Y_raw[:, i], pred_cross[:, i]) for i in range(N)])
    co_x  = np.array([_pearson_r(Y_raw[:, i], pred_cross[:, i]) for i in range(N)])

    return {
        "self_only":  (r2_s, co_s),
        "cross_only": (r2_x, co_x),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Plotting — behaviour-decoder style
# ═══════════════════════════════════════════════════════════════════════

def _plot_lag_sweep(results, out, worm_id, N_motor):
    """Lag-sweep line plot: mean R² & Corr vs K, one line per model×condition.

    Style matches behaviour decoder _plot_sweep.
    """
    ks = sorted(int(k) for k in results.keys())
    models = ["Ridge", "PCA-Ridge", "MLP", "TRF"]
    markers = {"Ridge": "o", "PCA-Ridge": "D", "MLP": "s", "TRF": "^"}
    cond_ls = {
        "conc+self": "-",
        "conc_only": "--",
        "self_only": "-.",
        "cross_only": ":",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, ylabel in [
        (axes[0], "r2_mean", "Mean R²"),
        (axes[1], "corr_mean", "Mean Pearson r"),
    ]:
        for model in models:
            mkey = f"{metric.split('_')[0]}_{model.lower().replace('-', '_')}"
            for ck in ["conc+self", "conc_only", "self_only", "cross_only"]:
                # TRF only has causal conditions
                if model == "TRF" and ck in ("conc+self", "conc_only"):
                    continue
                vals = []
                for K in ks:
                    res_k = results.get(str(K)) or results.get(K) or {}
                    cond_d = res_k.get(ck, {})
                    # metric stored as e.g. r2_mean_ridge, corr_mean_pca_ridge
                    m_field = metric.replace("mean", f"mean_{model.lower().replace('-', '_')}")
                    v = cond_d.get(m_field, np.nan)
                    vals.append(v)
                if all(np.isnan(v) for v in vals):
                    continue
                label = f"{model} {COND_NAME[KEY_TO_COND[ck]]}"
                ax.plot(ks, vals, marker=markers[model],
                        color=MODEL_COLOR[model], ls=cond_ls[ck],
                        lw=1.8, ms=6, label=label, alpha=0.85)

        ax.set_xlabel("Context length K (lags)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric == "corr_mean":
            ax.legend(fontsize=6.5, ncol=2, loc="best")

    fig.suptitle(f"Neural Activity Decoder — {worm_id}  "
                 f"(motor neurons, N={N_motor}, {N_FOLDS}-fold CV)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = out / f"lag_sweep_{worm_id}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


def _plot_condition_comparison(results, out, worm_id, N_motor):
    """PCA-Ridge R²/Corr vs lag, one line per condition (like masked_neuron cross_condition)."""
    ks = sorted(int(k) for k in results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in [
        (axes[0], "r2", "Mean R² (PCA-Ridge)"),
        (axes[1], "corr", "Mean Pearson r (PCA-Ridge)"),
    ]:
        for ck in ["conc+self", "conc_only", "self_only", "cross_only"]:
            field = f"{metric}_mean_pca_ridge"
            vals = []
            for K in ks:
                res_k = results.get(str(K)) or results.get(K) or {}
                vals.append(res_k.get(ck, {}).get(field, np.nan))
            if all(np.isnan(v) for v in vals):
                continue
            ax.plot(ks, vals, "o-", color=COND_COLOR[ck], lw=2, ms=7,
                    label=COND_NAME[KEY_TO_COND[ck]])

        ax.set_xlabel("Context length K (lags)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{worm_id}: Cross-condition comparison (motor N={N_motor})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = out / f"condition_comparison_{worm_id}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


def _plot_model_grid(results, out, worm_id, N_motor):
    """2×2 grid (one panel per condition), all models vs lag."""
    ks = sorted(int(k) for k in results.keys())
    models_info = [
        ("ridge",     "Ridge",     MODEL_COLOR["Ridge"],     "o"),
        ("pca_ridge", "PCA-Ridge", MODEL_COLOR["PCA-Ridge"], "D"),
        ("mlp",       "MLP",       MODEL_COLOR["MLP"],       "s"),
        ("trf",       "TRF",       MODEL_COLOR["TRF"],       "^"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

    for ax, (conc, incself) in zip(axes.flat, ALL_CONDITIONS):
        ck = COND_KEY[(conc, incself)]
        for mkey, mname, mc, mm in models_info:
            # TRF only for causal conditions
            if mkey == "trf" and conc:
                continue
            field = f"r2_mean_{mkey}"
            vals = []
            for K in ks:
                res_k = results.get(str(K)) or results.get(K) or {}
                vals.append(res_k.get(ck, {}).get(field, np.nan))
            if all(np.isnan(v) for v in vals):
                continue
            ax.plot(ks, vals, f"{mm}-", color=mc, lw=2, ms=6, label=mname)

        ax.set_title(COND_NAME[(conc, incself)], fontsize=11,
                     fontweight="bold", color=COND_COLOR[ck])
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1, 0].set_xlabel("Lag K"); axes[1, 1].set_xlabel("Lag K")
    axes[0, 0].set_ylabel("Mean R²"); axes[1, 0].set_ylabel("Mean R²")

    fig.suptitle(f"{worm_id}: Model comparison per condition (motor N={N_motor})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = out / f"model_grid_{worm_id}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


def _plot_neuron_traces(u_motor, pred_dict, n_lags, out, worm_id,
                        n_neurons_show=6, n_frames=200, dt=0.6,
                        condition_suffix=""):
    """Time-series traces for a few neurons (GT vs pred), behaviour-decoder style.
    
    condition_suffix: appended to filename, e.g. "_self_only" or "_cross_only"
    """
    N = u_motor.shape[1]
    T_out = u_motor.shape[0] - n_lags
    Y_gt = u_motor[n_lags:]

    # Pick neurons with highest R² spread
    show_idx = np.linspace(0, N - 1, min(n_neurons_show, N), dtype=int)

    start = min(50, T_out - n_frames)
    end = min(start + n_frames, T_out)
    t_sec = np.arange(start, end) * dt

    models = [(name, pred) for name, pred in pred_dict.items()
              if pred is not None and pred.shape[0] >= end]
    if not models:
        return

    n_models = len(models)
    n_show = len(show_idx)
    fig, axes = plt.subplots(n_show, n_models,
                             figsize=(4 * n_models, 2 * n_show),
                             sharex=True, squeeze=False)

    for col, (mname, pred) in enumerate(models):
        mc = MODEL_COLOR.get(mname, "#E24A33")
        for row, ni in enumerate(show_idx):
            ax = axes[row, col]
            gt = Y_gt[start:end, ni]
            pr = pred[start:end, ni]
            ax.plot(t_sec, gt, color="#333", lw=1.2, alpha=0.8, label="GT")
            ax.plot(t_sec, pr, color=mc, lw=1.0, alpha=0.9, ls="--", label="Pred")
            r2 = _r2(gt, pr)
            ax.text(0.98, 0.95, f"R²={r2:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            ax.set_ylabel(f"n{ni}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 0:
                ax.set_title(mname, fontsize=10, fontweight="bold")
            if row == n_show - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")

    cond_label = condition_suffix.replace("_", " ").strip() if condition_suffix else ""
    title_suffix = f" ({cond_label})" if cond_label else ""
    fig.suptitle(f"Neural Traces — {worm_id}  K={n_lags} (motor neurons){title_suffix}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = out / f"traces_K{n_lags}{condition_suffix}_{worm_id}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


def _plot_factor_effects(results, out, worm_id, N_motor):
    """Marginal effects of concurrent and self-history factors (PCA-Ridge)."""
    ks = sorted(int(k) for k in results.keys())
    all_ckeys = ["conc+self", "conc_only", "self_only", "cross_only"]

    available = []
    for K in ks:
        res_k = results.get(str(K)) or results.get(K) or {}
        if all(ck in res_k and f"r2_per_neuron_pca_ridge" in res_k.get(ck, {})
               for ck in all_ckeys):
            available.append(K)

    if not available:
        print("  [skip factor_effects: not all 4 conditions have per-neuron data]")
        return

    eff_conc, eff_self, eff_int = [], [], []
    for K in available:
        res_k = results.get(str(K)) or results.get(K) or {}
        cs = np.array(res_k["conc+self"]["r2_per_neuron_pca_ridge"])
        co = np.array(res_k["conc_only"]["r2_per_neuron_pca_ridge"])
        so = np.array(res_k["self_only"]["r2_per_neuron_pca_ridge"])
        xo = np.array(res_k["cross_only"]["r2_per_neuron_pca_ridge"])
        c_eff = 0.5 * ((cs - so) + (co - xo))
        s_eff = 0.5 * ((cs - co) + (so - xo))
        inter = (cs - co) - (so - xo)
        eff_conc.append(np.median(c_eff))
        eff_self.append(np.median(s_eff))
        eff_int.append(np.median(inter))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(available, eff_conc, "o-", color="#e74c3c", lw=2, ms=7,
                 label="Concurrent effect")
    axes[0].plot(available, eff_self, "s-", color="#2ecc71", lw=2, ms=7,
                 label="Self-history effect")
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].set_xlabel("Lag K"); axes[0].set_ylabel("Median R² gain (PCA-Ridge)")
    axes[0].set_title("Marginal factor effects"); axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.2)

    axes[1].plot(available, eff_int, "D-", color="#9b59b6", lw=2, ms=7)
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set_xlabel("Lag K"); axes[1].set_ylabel("Median interaction")
    axes[1].set_title("Interaction (conc × self)"); axes[1].grid(alpha=0.2)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{worm_id}: Factor effects (motor N={N_motor})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = out / f"factor_effects_{worm_id}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════
#  Violin plots — one violin per neuron, dots = K experiments
# ═══════════════════════════════════════════════════════════════════════

def _plot_violin_per_neuron(results, out, worm_id, N_motor, neuron_labels=None):
    """Violin plot per neuron comparing all models.

    For each condition a separate figure is produced.
    X-axis = neuron index, one violin per model per neuron.
    Each dot inside a violin is the R² from a different K value.
    """
    # Only use K = 1, 3, 5 for violin plots
    ks_all = sorted(int(k) for k in results.keys())
    ks = [k for k in ks_all if k in [1, 3, 5]]
    if not ks:
        print("  Skipping violin plots: no K in [1,3,5] available")
        return
    models_all = ["Ridge", "PCA-Ridge", "MLP", "TRF"]
    model_keys = {"Ridge": "ridge", "PCA-Ridge": "pca_ridge",
                  "MLP": "mlp", "TRF": "trf"}

    all_ckeys = ["conc+self", "conc_only", "self_only", "cross_only"]

    for ck in all_ckeys:
        # Determine which models have data for this condition
        models = []
        for m in models_all:
            mk = model_keys[m]
            field = f"r2_per_neuron_{mk}"
            has_data = any(
                field in (results.get(str(K)) or results.get(K, {})).get(ck, {})
                for K in ks
            )
            if has_data:
                models.append(m)
        if not models:
            continue

        n_models = len(models)
        # Collect per-neuron R² across K values: shape (n_models, N, len(ks))
        data = {m: [] for m in models}  # model -> list over K of array(N,)
        for K in ks:
            res_k = results.get(str(K)) or results.get(K) or {}
            cond_d = res_k.get(ck, {})
            for m in models:
                mk = model_keys[m]
                field = f"r2_per_neuron_{mk}"
                arr = cond_d.get(field)
                if arr is not None:
                    data[m].append(np.array(arr))
                else:
                    data[m].append(np.full(N_motor, np.nan))

        # data[m] is list of len(ks) arrays each of shape (N,)
        # For neuron ni: values across K = [data[m][ki][ni] for ki in range(len(ks))]

        # ── Figure ────────────────────────────────────────────
        width_per_neuron = max(0.6, 3.0 / n_models)
        fig_w = max(14, N_motor * n_models * width_per_neuron * 0.35 + 2)
        fig, ax = plt.subplots(figsize=(min(fig_w, 40), 6))

        group_width = 0.8  # total width allocated per neuron
        vw = group_width / n_models  # width of each violin

        neuron_positions = np.arange(N_motor)

        for mi, m in enumerate(models):
            mc = MODEL_COLOR.get(m, "#999")
            offset = (mi - (n_models - 1) / 2) * vw
            positions = neuron_positions + offset

            # Gather values for each neuron (across K experiments)
            neuron_vals = []
            for ni in range(N_motor):
                vals = [data[m][ki][ni] for ki in range(len(ks))
                        if not np.isnan(data[m][ki][ni])]
                neuron_vals.append(vals if vals else [np.nan])

            # Draw violins
            valid_pos = []
            valid_vals = []
            for ni in range(N_motor):
                if len(neuron_vals[ni]) >= 2 and not any(np.isnan(neuron_vals[ni])):
                    valid_pos.append(positions[ni])
                    valid_vals.append(neuron_vals[ni])

            if valid_vals:
                parts = ax.violinplot(
                    valid_vals, positions=valid_pos,
                    widths=vw * 0.85, showmeans=False,
                    showmedians=False, showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(mc)
                    pc.set_alpha(0.35)
                    pc.set_edgecolor(mc)
                    pc.set_linewidth(0.8)

            # Overlay individual dots (one per K)
            for ni in range(N_motor):
                nv = neuron_vals[ni]
                if nv and not any(np.isnan(nv)):
                    jitter = np.random.default_rng(ni + mi * 1000).uniform(
                        -vw * 0.18, vw * 0.18, size=len(nv)
                    )
                    ax.scatter(
                        np.full(len(nv), positions[ni]) + jitter,
                        nv, s=14, color=mc, alpha=0.75,
                        edgecolors="white", linewidths=0.3,
                        zorder=3,
                    )

            # Median line per neuron
            for ni in range(N_motor):
                nv = neuron_vals[ni]
                if nv and not any(np.isnan(nv)):
                    med = np.median(nv)
                    ax.plot(
                        [positions[ni] - vw * 0.25, positions[ni] + vw * 0.25],
                        [med, med], color=mc, lw=1.5, zorder=4,
                    )

            # Invisible scatter for legend
            ax.scatter([], [], s=40, color=mc, label=m)

        ax.set_xticks(neuron_positions)
        xlabels = neuron_labels if neuron_labels and len(neuron_labels) == N_motor \
            else [f"n{i}" for i in range(N_motor)]
        ax.set_xticklabels(xlabels,
                           fontsize=max(5, min(7, 180 // N_motor)),
                           rotation=45, ha="right")
        ax.set_xlabel("Neuron", fontsize=11)
        ax.set_ylabel("R² (each dot = one K value)", fontsize=11)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(bottom=-1)  # Don't show y-axis below -1
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        cond_label = COND_NAME[KEY_TO_COND[ck]]
        ax.set_title(
            f"{worm_id} — {cond_label}\n"
            f"(N={N_motor}, dots = K∈{{{','.join(str(k) for k in ks)}}}, "
            f"{N_FOLDS}-fold CV)",
            fontsize=12, fontweight="bold",
        )

        fig.tight_layout()
        fname = out / f"violin_{ck.replace('+', '_')}_{worm_id}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out_dir", default="output_plots/neural_activity_decoder")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--replot", action="store_true",
                    help="Regenerate plots from existing results.json")
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--no_trf", action="store_true")
    ap.add_argument("--neurons", default="motor",
                    choices=["motor", "nonmotor", "all"],
                    help="Which neuron subset to predict (default: motor)")
    args = ap.parse_args()

    device = args.device
    base_out = Path(args.out_dir)

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
    else:  # "all"
        sel_idx = list(range(N_total))
        subset_tag = "all"

    if not sel_idx:
        print(f"ERROR: no {args.neurons} neurons found for {worm_id}")
        sys.exit(1)

    sel_labels = [labels[i] if i < len(labels) else f"n{i}" for i in sel_idx]
    u_motor = u_all[:, sel_idx].astype(np.float32)
    T, N = u_motor.shape
    out = base_out / f"{worm_id}_{subset_tag}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Worm: {worm_id}  subset={subset_tag}  T={T}  N={N}")
    worm_id_full = f"{worm_id}_{subset_tag}"  # used in plot titles

    results_path = out / "results.json"

    # ── Replot mode ───────────────────────────────────────────────
    if args.replot:
        if not results_path.exists():
            print(f"No results.json at {results_path}"); sys.exit(1)
        with open(results_path) as f:
            results = json.load(f)
        _plot_lag_sweep(results, out, worm_id_full, N)
        _plot_condition_comparison(results, out, worm_id_full, N)
        _plot_model_grid(results, out, worm_id_full, N)
        _plot_factor_effects(results, out, worm_id_full, N)
        _plot_violin_per_neuron(results, out, worm_id_full, N, sel_labels)
        print("Replot done."); return

    # ── Load cached results ───────────────────────────────────────
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} previous K results")

    run_mlp = not args.no_mlp
    run_trf = not args.no_trf

    from joblib import Parallel, delayed

    t_total = time.time()

    for K in K_VALUES:
        sk = str(K)

        # Check if fully done
        existing = results.get(sk, {})
        all_done = True
        for conc, incself in ALL_CONDITIONS:
            ck = COND_KEY[(conc, incself)]
            cd = existing.get(ck, {})
            need_trf = run_trf and not conc and K >= 2
            if f"r2_mean_ridge" not in cd:
                all_done = False; break
            if run_mlp and f"r2_mean_mlp" not in cd:
                all_done = False; break
            if need_trf and f"r2_mean_trf" not in cd:
                all_done = False; break
        if all_done:
            print(f"\n  K={K} already done — skipping"); continue

        print(f"\n{'='*60}\n  Context length K = {K}\n{'='*60}")

        T_out = T - K
        if T_out < 100:
            print(f"  [skip K={K}: T_out={T_out} too small]"); continue

        X_full, Y_all = _precompute_full(u_motor, K)
        folds = _make_folds(T_out, N_FOLDS)
        res_k = dict(existing)  # preserve any cached sub-results

        # ── Transformer (shared for both causal conditions) ───
        trf_res = {}
        if run_trf and K >= 2:
            need_trf = ("self_only" not in res_k or
                        f"r2_mean_trf" not in res_k.get("self_only", {}))
            if need_trf:
                print(f"\n  TRF (d=128, L=2)...")
                t0 = time.time()
                trf_res = _eval_transformer_causal(u_motor, K, folds, device)
                print(f"    [{time.time()-t0:.0f}s]")
                for ck_t, (r2a, _) in trf_res.items():
                    print(f"    {ck_t:12s}  TRF mean R²={np.mean(r2a):+.4f}")
            else:
                print(f"  TRF: cached")

        # ── Per-condition: Ridge, PCA-Ridge, MLP ──────────────
        for conc, incself in ALL_CONDITIONS:
            ck = COND_KEY[(conc, incself)]
            cn = COND_NAME[(conc, incself)]
            d = _feature_dim(N, K, conc, incself)
            if d == 0:
                print(f"\n  K={K}  {cn:26s}  dim=0 — SKIPPED"); continue

            print(f"\n  K={K}  {cn:26s}  dim={d}")

            cond_d = dict(res_k.get(ck, {}))

            # Ridge + PCA-Ridge
            if f"r2_mean_ridge" not in cond_d:
                t0 = time.time()
                rp = Parallel(n_jobs=min(8, N), verbose=0)(
                    delayed(_eval_ridge_pca_neuron)(
                        X_full, Y_all, ni, N, K,
                        conc, incself, folds, PCA_K)
                    for ni in range(N))
                t_rp = time.time() - t0

                r2_ridge = [r[0] for r in rp]
                corr_ridge = [r[1] for r in rp]
                r2_pca = [r[2] for r in rp]
                corr_pca = [r[3] for r in rp]
                bdry_ridge = sum(r[4] for r in rp)
                bdry_pca = sum(r[5] for r in rp)

                cond_d["r2_mean_ridge"] = float(np.mean(r2_ridge))
                cond_d["corr_mean_ridge"] = float(np.mean(corr_ridge))
                cond_d["r2_mean_pca_ridge"] = float(np.mean(r2_pca))
                cond_d["corr_mean_pca_ridge"] = float(np.mean(corr_pca))
                cond_d["r2_per_neuron_ridge"] = [float(v) for v in r2_ridge]
                cond_d["corr_per_neuron_ridge"] = [float(v) for v in corr_ridge]
                cond_d["r2_per_neuron_pca_ridge"] = [float(v) for v in r2_pca]
                cond_d["corr_per_neuron_pca_ridge"] = [float(v) for v in corr_pca]
                cond_d["boundary_hits_ridge"] = bdry_ridge
                cond_d["boundary_hits_pca_ridge"] = bdry_pca

                print(f"    Ridge  mean R²={np.mean(r2_ridge):+.4f}  "
                      f"corr={np.mean(corr_ridge):.4f}  "
                      f"(bdry={bdry_ridge}/{N*N_FOLDS})")
                print(f"    PCA    mean R²={np.mean(r2_pca):+.4f}  "
                      f"corr={np.mean(corr_pca):.4f}  "
                      f"(bdry={bdry_pca}/{N*N_FOLDS})  [{t_rp:.0f}s]")
            else:
                print(f"    Ridge/PCA: cached  "
                      f"R²={cond_d['r2_mean_ridge']:+.4f} / "
                      f"{cond_d['r2_mean_pca_ridge']:+.4f}")

            # MLP
            if run_mlp and f"r2_mean_mlp" not in cond_d:
                t0 = time.time()
                mlp_r2, mlp_corr = [], []
                for ni in range(N):
                    r2m, cm = _eval_mlp_neuron(
                        X_full, Y_all, ni, N, K,
                        conc, incself, folds, device)
                    mlp_r2.append(r2m)
                    mlp_corr.append(cm)
                    if (ni + 1) % 10 == 0:
                        print(f"      MLP: {ni+1}/{N}")
                t_mlp = time.time() - t0

                cond_d["r2_mean_mlp"] = float(np.mean(mlp_r2))
                cond_d["corr_mean_mlp"] = float(np.mean(mlp_corr))
                cond_d["r2_per_neuron_mlp"] = [float(v) for v in mlp_r2]
                cond_d["corr_per_neuron_mlp"] = [float(v) for v in mlp_corr]

                print(f"    MLP    mean R²={np.mean(mlp_r2):+.4f}  "
                      f"corr={np.mean(mlp_corr):.4f}  [{t_mlp:.0f}s]")
            elif run_mlp:
                print(f"    MLP: cached  R²={cond_d['r2_mean_mlp']:+.4f}")

            # Transformer (only for causal conditions)
            if not conc and ck in trf_res:
                r2_t, corr_t = trf_res[ck]
                cond_d["r2_mean_trf"] = float(np.mean(r2_t))
                cond_d["corr_mean_trf"] = float(np.mean(corr_t))
                cond_d["r2_per_neuron_trf"] = [float(v) for v in r2_t]
                cond_d["corr_per_neuron_trf"] = [float(v) for v in corr_t]
                print(f"    TRF    mean R²={np.mean(r2_t):+.4f}  "
                      f"corr={np.mean(corr_t):.4f}")

            res_k[ck] = cond_d

        results[sk] = res_k

        # Checkpoint after each K
        _js = json.dumps(results, indent=2,
                         default=lambda o: int(o) if isinstance(o, np.integer)
                         else float(o) if isinstance(o, np.floating) else o)
        with open(results_path, "w") as f:
            f.write(_js)
        print(f"\n  Checkpoint saved ({results_path})")

    total_s = time.time() - t_total
    print(f"\nTotal time: {total_s:.0f}s ({total_s/60:.1f}min)")

    # ── Generate neuron trace plots for K=5 (causal+self) ─────
    trace_K = min(5, max(K_VALUES))
    if trace_K in K_VALUES:
        print(f"\nGenerating neuron trace plots for K={trace_K} (self_only)...")
        T_out = T - trace_K
        folds = _make_folds(T_out, N_FOLDS)
        X_full, Y_all = _precompute_full(u_motor, trace_K)
        Y_gt = u_motor[trace_K:]

        # Ridge + PCA-Ridge predictions (self_only condition)
        pred_ridge = np.zeros((T_out, N), np.float32)
        pred_pca = np.zeros((T_out, N), np.float32)
        from sklearn.linear_model import RidgeCV
        from sklearn.decomposition import PCA
        for ni in range(N):
            mask = _feature_mask(N, trace_K, ni, False, True)
            X, y = X_full[:, mask], Y_all[:, ni]
            for tr, te in folds:
                mu, sig = X[tr].mean(0), X[tr].std(0).clip(1e-8)
                Xtr_z, Xte_z = (X[tr] - mu) / sig, (X[te] - mu) / sig
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
                    r.fit(Xtr_z, y[tr])
                pred_ridge[te, ni] = r.predict(Xte_z)
                ke = min(PCA_K, len(tr) - 1, Xtr_z.shape[1])
                if ke >= 1:
                    pca = PCA(n_components=ke)
                    rp = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
                    rp.fit(pca.fit_transform(Xtr_z), y[tr])
                    pred_pca[te, ni] = rp.predict(pca.transform(Xte_z))
                else:
                    pred_pca[te, ni] = pred_ridge[te, ni]

        # MLP predictions (self_only condition)
        pred_mlp = None
        if run_mlp:
            print("  Generating MLP traces...")
            pred_mlp = np.zeros((T_out, N), np.float32)
            for ni in range(N):
                mask = _feature_mask(N, trace_K, ni, False, True)
                X, y = X_full[:, mask], Y_all[:, ni]
                d_in = X.shape[1]
                for tr, te in folds:
                    tri, vai = _inner_split(tr)
                    mu, sig = X[tri].mean(0), X[tri].std(0).clip(1e-8)
                    mu_y, sig_y = float(y[tri].mean()), float(max(y[tri].std(), 1e-8))
                    Xtr = torch.tensor(((X[tri] - mu) / sig), dtype=torch.float32, device=device)
                    Ytr = torch.tensor(((y[tri] - mu_y) / sig_y).reshape(-1, 1), dtype=torch.float32, device=device)
                    Xva = torch.tensor(((X[vai] - mu) / sig), dtype=torch.float32, device=device)
                    Yva = torch.tensor(((y[vai] - mu_y) / sig_y).reshape(-1, 1), dtype=torch.float32, device=device)
                    net = _make_mlp(d_in).to(device)
                    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
                    bvl, bs, pat = float("inf"), None, 0
                    for _ in range(150):
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
                        if pat > 20:
                            break
                    if bs:
                        net.load_state_dict(bs)
                    net.eval()
                    with torch.no_grad():
                        pz = net(torch.tensor(((X[te] - mu) / sig), dtype=torch.float32, device=device))
                        pred_mlp[te, ni] = (pz.cpu().numpy().ravel() * sig_y + mu_y).astype(np.float32)
                if (ni + 1) % 10 == 0:
                    print(f"    MLP traces: {ni+1}/{N}")

        # Transformer predictions (self_only condition)
        pred_trf = None
        if run_trf and trace_K >= 2:
            print("  Generating Transformer traces...")
            pred_trf = np.zeros((T_out, N), np.float32)
            for fi, (tr, te) in enumerate(folds):
                tri, vai = _inner_split(tr)
                mu = u_motor[trace_K + tri].mean(0)
                sig = u_motor[trace_K + tri].std(0) + 1e-8
                u_n = ((u_motor - mu) / sig).astype(np.float32)
                X_tri = _build_windows(u_n, tri, trace_K)
                Y_tri = u_n[trace_K + tri]
                X_vai = _build_windows(u_n, vai, trace_K)
                Y_vai = u_n[trace_K + vai]
                model = _train_trf_fold(X_tri, Y_tri, X_vai, Y_vai, N, trace_K, device)
                X_te_np = _build_windows(u_n, te, trace_K)
                X_te = torch.tensor(X_te_np, dtype=torch.float32, device=device)
                with torch.no_grad():
                    pred_z = model.predict_mean(X_te).cpu().numpy()
                pred_trf[te] = pred_z * sig + mu
                model.cpu()
                print(f"    TRF traces fold {fi+1}/{len(folds)}")

        # Plot all model traces (self_only condition)
        pred_dict = {"Ridge": pred_ridge, "PCA-Ridge": pred_pca}
        if pred_mlp is not None:
            pred_dict["MLP"] = pred_mlp
        if pred_trf is not None:
            pred_dict["TRF"] = pred_trf
        _plot_neuron_traces(u_motor, pred_dict, trace_K, out, worm_id_full,
                            condition_suffix="_self_only")

        # ── Also generate traces for cross_only (no self-history) ──
        print(f"\nGenerating cross_only traces for K={trace_K}...")
        pred_ridge_x = np.zeros((T_out, N), np.float32)
        for ni in range(N):
            mask = _feature_mask(N, trace_K, ni, False, False)  # cross_only
            X, y = X_full[:, mask], Y_all[:, ni]
            for tr, te in folds:
                mu, sig = X[tr].mean(0), X[tr].std(0).clip(1e-8)
                Xtr_z, Xte_z = (X[tr] - mu) / sig, (X[te] - mu) / sig
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
                    r.fit(Xtr_z, y[tr])
                pred_ridge_x[te, ni] = r.predict(Xte_z)

        # TRF cross_only (zero out self-column)
        pred_trf_x = None
        if run_trf and trace_K >= 2 and pred_trf is not None:
            print("  Generating TRF cross_only traces...")
            pred_trf_x = np.zeros((T_out, N), np.float32)
            for fi, (tr, te) in enumerate(folds):
                tri, vai = _inner_split(tr)
                mu = u_motor[trace_K + tri].mean(0)
                sig = u_motor[trace_K + tri].std(0) + 1e-8
                u_n = ((u_motor - mu) / sig).astype(np.float32)
                X_tri = _build_windows(u_n, tri, trace_K)
                Y_tri = u_n[trace_K + tri]
                X_vai = _build_windows(u_n, vai, trace_K)
                Y_vai = u_n[trace_K + vai]
                model = _train_trf_fold(X_tri, Y_tri, X_vai, Y_vai, N, trace_K, device)
                X_te_np = _build_windows(u_n, te, trace_K)
                # Zero out each target column for cross_only
                for ni in range(N):
                    X_m = X_te_np.copy()
                    X_m[:, :, ni] = 0.0
                    with torch.no_grad():
                        p = model.predict_mean(
                            torch.tensor(X_m, dtype=torch.float32, device=device)
                        ).cpu().numpy()
                    pred_trf_x[te, ni] = p[:, ni] * sig[ni] + mu[ni]
                model.cpu()

        pred_dict_x = {"Ridge (cross)": pred_ridge_x}
        if pred_trf_x is not None:
            pred_dict_x["TRF (cross)"] = pred_trf_x
        _plot_neuron_traces(u_motor, pred_dict_x, trace_K, out, worm_id_full,
                            condition_suffix="_cross_only")

    # ── Generate all summary plots ────────────────────────────
    print("\nGenerating summary plots...")
    _plot_lag_sweep(results, out, worm_id_full, N)
    _plot_condition_comparison(results, out, worm_id_full, N)
    _plot_model_grid(results, out, worm_id_full, N)
    _plot_factor_effects(results, out, worm_id_full, N)
    _plot_violin_per_neuron(results, out, worm_id_full, N, sel_labels)

    print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
