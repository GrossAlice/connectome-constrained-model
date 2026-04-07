#!/usr/bin/env python3
"""
Linearity test: are neuron→neuron relationships linear?

Mask each neuron in turn, predict from the rest using a ≈3 s strictly-causal
lag window.  Compare ridge (linear) with a 2-layer MLP (nonlinear) in four
setups:

  (A) Cross-neuron features only  →  one-step
  (B) Cross-neuron + self-history  →  one-step  AND  autoregressive (AR)

Conditions evaluated
--------------------
  self_ridge       target's own history only (autocorrelation baseline)
  xn_ridge         other neurons only, ridge        (linear cross-neuron)
  xn_mlp           other neurons only, MLP           (nonlinear cross-neuron)
  full_ridge       other + self, ridge, one-step
  full_mlp         other + self, MLP, one-step
  full_ridge_ar    same ridge, free-running AR on test folds
  full_mlp_ar      same MLP,   free-running AR on test folds

Cross-neuron features are PCA-compressed (default 50 PC) so the comparison
is not biased by the curse of dimensionality.

Plots
-----
  Panel 1  scatter  ridge vs MLP  (cross-neuron, one-step)
  Panel 2  violin   all 7 conditions
  Panel 3  scatter  one-step vs AR  (full_mlp)

Usage
-----
  python -m scripts.masked_neuron.linearity_test \
      --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
      --device cuda
"""
from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
_ALPHAS = np.logspace(-4, 6, 30)

# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_worm(h5_path: Path) -> dict | None:
    try:
        with h5py.File(h5_path, "r") as f:
            if "stage1/u_mean" not in f or "gcamp/neuron_labels" not in f:
                return None
            u = np.array(f["stage1/u_mean"], dtype=np.float32)
            labels = f["gcamp/neuron_labels"][:]
            dt = 0.6
            if "timing/timestamp_confocal" in f:
                ts = f["timing/timestamp_confocal"][:]
                if len(ts) > 1:
                    dt = float(np.median(np.diff(ts)))
    except Exception as exc:
        print(f"  [warn] {h5_path.name}: {exc}")
        return None
    if u.ndim != 2:
        return None
    if u.shape[0] < 400 and u.shape[1] >= 400:
        u = u.T
    labels = [lb.decode() if isinstance(lb, bytes) else str(lb) for lb in labels]
    T, N = u.shape
    if len(labels) != N:
        return None
    keep = np.any(np.isfinite(u), axis=0)
    u = u[:, keep]
    labels = [l for l, k in zip(labels, keep) if k]
    for j in range(u.shape[1]):
        nans = ~np.isfinite(u[:, j])
        if nans.any() and not nans.all():
            good = np.where(~nans)[0]
            u[:, j] = np.interp(np.arange(u.shape[0]), good, u[good, j])
    return {"u": u, "labels": labels, "name": h5_path.stem, "dt": dt}


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature construction (strict causal)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_features(u: np.ndarray, target_idx: int, n_lag: int):
    """
    Strict causal features.

    Returns
    -------
    X_cross : (T_out, n_lag * N_other)  other neurons' lagged values
    X_self  : (T_out, n_lag)            target's own lagged values
    y       : (T_out,)                  target at time t

    Feature ordering:  col-block 0 = lag 1 (t−1), ..., col-block n_lag−1 = lag n_lag (t−n_lag)
    """
    T, N = u.shape
    T_out = T - n_lag
    cross_mask = np.ones(N, dtype=bool)
    cross_mask[target_idx] = False
    N_cross = int(cross_mask.sum())

    X_cross = np.empty((T_out, n_lag * N_cross), dtype=np.float32)
    X_self  = np.empty((T_out, n_lag), dtype=np.float32)

    for lag_idx in range(n_lag):
        # lag_idx=0 → t-1 (most recent), lag_idx=n_lag-1 → t-n_lag (oldest)
        start = n_lag - 1 - lag_idx
        X_cross[:, lag_idx * N_cross : (lag_idx + 1) * N_cross] = \
            u[start : start + T_out, :][:, cross_mask]
        X_self[:, lag_idx] = u[start : start + T_out, target_idx]

    y = u[n_lag : n_lag + T_out, target_idx].astype(np.float32)
    return X_cross, X_self, y


# ═══════════════════════════════════════════════════════════════════════════════
#  CV + helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_folds(T, n_folds):
    sizes = np.full(n_folds, T // n_folds, dtype=int)
    sizes[: T % n_folds] += 1
    folds, cur = [], 0
    for s in sizes:
        te = np.arange(cur, cur + s)
        tr = np.concatenate([np.arange(0, cur), np.arange(cur + s, T)])
        folds.append((tr, te))
        cur += s
    return folds


def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _zs(X):
    """z-score stats: returns (mu, std) with std clipped."""
    mu  = X.mean(0)
    std = X.std(0).clip(1e-8)
    return mu, std


def _z(X, mu, std):
    """z-score X."""
    return ((X - mu) / std).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  MLP
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mlp(in_dim, hidden=128, dropout=0.1):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(hidden, 1),
    )


def _train_mlp(net, Xtr, Ytr, Xva, Yva, *,
               epochs=200, lr=1e-3, batch_size=64,
               weight_decay=1e-4, patience=20):
    import torch
    opt  = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    crit = torch.nn.MSELoss()
    n    = Xtr.shape[0]
    best_val, stale, best_state = float("inf"), 0, None
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = crit(net(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
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
    if best_state:
        net.load_state_dict(best_state)


# ═══════════════════════════════════════════════════════════════════════════════
#  AR evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ar_evaluate_ridge(ridge, Xcp_te, Xs_te, mu_f, std_f, mu_y, std_y):
    """Free-running AR for ridge on one test fold."""
    T = Xcp_te.shape[0]
    preds = np.zeros(T, np.float32)
    buf = Xs_te[0].copy()                     # (n_lag,), original scale
    for t in range(T):
        x = np.concatenate([Xcp_te[t], buf])  # same space as training
        x_z = _z(x.reshape(1, -1), mu_f, std_f)
        p_z = float(ridge.predict(x_z).ravel()[0])
        p = p_z * std_y + mu_y
        preds[t] = p
        buf = np.roll(buf, 1); buf[0] = p     # insert prediction as most recent
    return preds


def _ar_evaluate_mlp(net, Xcp_te, Xs_te, mu_f, std_f, mu_y, std_y, device):
    """Free-running AR for MLP on one test fold."""
    import torch
    T = Xcp_te.shape[0]
    preds = np.zeros(T, np.float32)
    buf = Xs_te[0].copy()
    net.eval()
    with torch.no_grad():
        for t in range(T):
            x = np.concatenate([Xcp_te[t], buf])
            x_z = torch.from_numpy(
                _z(x.reshape(1, -1), mu_f, std_f)
            ).to(device)
            p_z = net(x_z).item()
            p = p_z * std_y + mu_y
            preds[t] = p
            buf = np.roll(buf, 1); buf[0] = p
    return preds


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-neuron evaluation (all 7 conditions)
# ═══════════════════════════════════════════════════════════════════════════════

CONDS = [
    "self_ridge",
    "xn_ridge", "xn_mlp",
    "full_ridge", "full_mlp",
    "full_ridge_ar", "full_mlp_ar",
]


def _run_one_neuron(
    u: np.ndarray,
    target_idx: int,
    n_lag: int,
    n_folds: int     = 5,
    max_pca: int     = 50,
    device_str: str  = "cpu",
    seed: int        = 0,
    hidden: int      = 128,
    dropout: float   = 0.1,
    epochs: int      = 200,
    lr: float        = 1e-3,
    batch_size: int  = 64,
    weight_decay: float = 1e-4,
    patience: int    = 20,
) -> dict[str, float]:
    """
    Return {condition: R²} for one masked neuron across 7 conditions.
    """
    import torch
    from sklearn.linear_model import RidgeCV
    from sklearn.decomposition import PCA

    X_cross, X_self, y = _build_features(u, target_idx, n_lag)
    T_out = len(y)
    folds = _make_folds(T_out, n_folds)
    use_pca = X_cross.shape[1] > max_pca
    device  = torch.device(device_str)
    to_t    = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)

    pred = {k: np.zeros(T_out, np.float32) for k in CONDS}

    for fi, (tr_idx, te_idx) in enumerate(folds):
        # ── slice ────────────────────────────────────────────────────────────
        Xc_tr, Xc_te = X_cross[tr_idx], X_cross[te_idx]
        Xs_tr, Xs_te = X_self[tr_idx],  X_self[te_idx]
        y_tr         = y[tr_idx]

        # ── PCA on cross-neuron features ─────────────────────────────────────
        if use_pca:
            pca = PCA(n_components=max_pca, random_state=0)
            Xcp_tr = pca.fit_transform(Xc_tr).astype(np.float32)
            Xcp_te = pca.transform(Xc_te).astype(np.float32)
        else:
            Xcp_tr, Xcp_te = Xc_tr.copy(), Xc_te.copy()

        # ── build "full" = cross-PCA + self ──────────────────────────────────
        Xf_tr = np.hstack([Xcp_tr, Xs_tr])
        Xf_te = np.hstack([Xcp_te, Xs_te])

        # ── z-score stats (per feature-set) ──────────────────────────────────
        mu_s,  std_s  = _zs(Xs_tr)
        mu_xn, std_xn = _zs(Xcp_tr)
        mu_f,  std_f  = _zs(Xf_tr)
        mu_y  = float(y_tr.mean())
        std_y = float(max(y_tr.std(), 1e-8))
        y_tr_z = (y_tr - mu_y) / std_y

        # inner split (last 20 % of train) for MLP early stopping
        n_va = max(1, int(len(tr_idx) * 0.2))

        # ── SELF RIDGE ───────────────────────────────────────────────────────
        r = RidgeCV(alphas=_ALPHAS, fit_intercept=True)
        r.fit(_z(Xs_tr, mu_s, std_s), y_tr_z)
        pred["self_ridge"][te_idx] = \
            r.predict(_z(Xs_te, mu_s, std_s)) * std_y + mu_y

        # ── CROSS-NEURON RIDGE ───────────────────────────────────────────────
        r = RidgeCV(alphas=_ALPHAS, fit_intercept=True)
        r.fit(_z(Xcp_tr, mu_xn, std_xn), y_tr_z)
        pred["xn_ridge"][te_idx] = \
            r.predict(_z(Xcp_te, mu_xn, std_xn)) * std_y + mu_y

        # ── CROSS-NEURON MLP ────────────────────────────────────────────────
        Xtr_z = to_t(_z(Xcp_tr[:-n_va], mu_xn, std_xn))
        Xva_z = to_t(_z(Xcp_tr[-n_va:], mu_xn, std_xn))
        Ytr_z = to_t(y_tr_z[:-n_va].reshape(-1, 1))
        Yva_z = to_t(y_tr_z[-n_va:].reshape(-1, 1))
        torch.manual_seed(seed + fi)
        net_xn = _make_mlp(Xcp_tr.shape[1], hidden, dropout).to(device)
        _train_mlp(net_xn, Xtr_z, Ytr_z, Xva_z, Yva_z,
                   epochs=epochs, lr=lr, batch_size=batch_size,
                   weight_decay=weight_decay, patience=patience)
        net_xn.eval()
        with torch.no_grad():
            p = net_xn(to_t(_z(Xcp_te, mu_xn, std_xn))).cpu().numpy().ravel()
        pred["xn_mlp"][te_idx] = p * std_y + mu_y

        # ── FULL RIDGE (one-step) ───────────────────────────────────────────
        ridge_f = RidgeCV(alphas=_ALPHAS, fit_intercept=True)
        ridge_f.fit(_z(Xf_tr, mu_f, std_f), y_tr_z)
        pred["full_ridge"][te_idx] = \
            ridge_f.predict(_z(Xf_te, mu_f, std_f)) * std_y + mu_y

        # ── FULL MLP (one-step) ─────────────────────────────────────────────
        Xtr_f_z = to_t(_z(Xf_tr[:-n_va], mu_f, std_f))
        Xva_f_z = to_t(_z(Xf_tr[-n_va:], mu_f, std_f))
        torch.manual_seed(seed + fi + 1000)
        net_f = _make_mlp(Xf_tr.shape[1], hidden, dropout).to(device)
        _train_mlp(net_f, Xtr_f_z, Ytr_z, Xva_f_z, Yva_z,
                   epochs=epochs, lr=lr, batch_size=batch_size,
                   weight_decay=weight_decay, patience=patience)
        net_f.eval()
        with torch.no_grad():
            p = net_f(to_t(_z(Xf_te, mu_f, std_f))).cpu().numpy().ravel()
        pred["full_mlp"][te_idx] = p * std_y + mu_y

        # ── FULL RIDGE AR ───────────────────────────────────────────────────
        pred["full_ridge_ar"][te_idx] = _ar_evaluate_ridge(
            ridge_f, Xcp_te, Xs_te, mu_f, std_f, mu_y, std_y)

        # ── FULL MLP AR ─────────────────────────────────────────────────────
        pred["full_mlp_ar"][te_idx] = _ar_evaluate_mlp(
            net_f, Xcp_te, Xs_te, mu_f, std_f, mu_y, std_y, device)

    return {k: _r2(y, pred[k]) for k in CONDS}


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

_COND_LABELS = {
    "self_ridge":    "Self\n(ridge)",
    "xn_ridge":      "Cross\nridge",
    "xn_mlp":        "Cross\nMLP",
    "full_ridge":    "Full\nridge",
    "full_mlp":      "Full\nMLP",
    "full_ridge_ar": "Full ridge\nAR",
    "full_mlp_ar":   "Full MLP\nAR",
}

_COND_COLORS = {
    "self_ridge":    "#95a5a6",
    "xn_ridge":      "#3498db",
    "xn_mlp":        "#e74c3c",
    "full_ridge":    "#2980b9",
    "full_mlp":      "#c0392b",
    "full_ridge_ar": "#1abc9c",
    "full_mlp_ar":   "#e67e22",
}


def _plot_results(out_dir: Path, results: list[dict], worm_name: str):
    """Produce 3-panel summary figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    neurons = [r["neuron"] for r in results]
    N = len(results)

    # Collect per-condition arrays
    vals = {k: np.array([r[k] for r in results]) for k in CONDS}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── Panel A: Scatter  ridge vs MLP (cross-neuron only) ──────────────────
    ax = axes[0]
    x, y_ = vals["xn_ridge"], vals["xn_mlp"]
    ax.scatter(x, y_, s=18, alpha=0.6, c="#8e44ad", edgecolors="k", lw=0.3, zorder=3)
    lo = min(x.min(), y_.min()) - 0.05
    hi = max(x.max(), y_.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Ridge R²  (cross-neuron, one-step)", fontsize=10)
    ax.set_ylabel("MLP R²  (cross-neuron, one-step)", fontsize=10)
    frac = float((y_ > x).mean())
    delta = float(np.median(y_ - x))
    ax.set_title(f"A — Linearity test\n"
                 f"MLP wins {frac:.0%}  |  median Δ = {delta:+.3f}", fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")

    # ── Panel B: Violin / box  all 7 conditions ────────────────────────────
    ax = axes[1]
    data  = [vals[k] for k in CONDS]
    labs  = [_COND_LABELS[k] for k in CONDS]
    cols  = [_COND_COLORS[k] for k in CONDS]
    positions = np.arange(len(CONDS))
    parts = ax.violinplot(data, positions=positions,
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(cols[i])
        pc.set_alpha(0.5)
    parts["cmedians"].set_color("k")
    # swarm (jittered)
    rng = np.random.default_rng(42)
    for i, k in enumerate(CONDS):
        jitter = rng.normal(0, 0.07, N)
        ax.scatter(positions[i] + jitter, vals[k],
                   s=10, alpha=0.45, c=cols[i], edgecolors="none", zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("R²", fontsize=10)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    medians_str = "  ".join(f"{_COND_LABELS[k].replace(chr(10), ' ')}: "
                            f"{np.median(vals[k]):.3f}" for k in CONDS)
    ax.set_title(f"B — All conditions  (n={N} neurons)", fontsize=11)

    # ── Panel C: one-step vs AR  (full_mlp) ─────────────────────────────────
    ax = axes[2]
    os_ = vals["full_mlp"]
    ar_ = vals["full_mlp_ar"]
    # Paired lines
    for i in range(N):
        ax.plot([0, 1], [os_[i], ar_[i]], lw=0.5, alpha=0.25, c="#7f8c8d")
    ax.scatter(np.zeros(N), os_, s=16, alpha=0.6, c="#c0392b",
               edgecolors="k", lw=0.3, zorder=3, label="one-step")
    ax.scatter(np.ones(N),  ar_, s=16, alpha=0.6, c="#e67e22",
               edgecolors="k", lw=0.3, zorder=3, label="AR")
    # summary
    ax.plot([0, 1], [np.median(os_), np.median(ar_)], "k-o",
            lw=2.5, ms=10, zorder=5, label="median")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["One-step\n(teacher-forced)", "Autoregressive\n(free-running)"],
                       fontsize=10)
    ax.set_ylabel("R²  (full MLP: cross + self)", fontsize=10)
    drop = float(np.median(os_ - ar_))
    ax.set_title(f"C — AR degradation\n"
                 f"median drop = {drop:.3f}", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")

    fig.suptitle(f"Linearity & AR test — {worm_name}\n"
                 f"(strict causal, {results[0].get('n_lag', '?')}-frame lag, "
                 f"5-fold CV, PCA-50)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "linearity_test.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "linearity_test.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'linearity_test.png'}")

    # ── Supplementary: ridge vs MLP for full (with self-history) ────────────
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    x2, y2 = vals["full_ridge"], vals["full_mlp"]
    ax.scatter(x2, y2, s=18, alpha=0.6, c="#8e44ad", edgecolors="k", lw=0.3)
    lo2 = min(x2.min(), y2.min()) - 0.03
    hi2 = max(x2.max(), y2.max()) + 0.03
    ax.plot([lo2, hi2], [lo2, hi2], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Ridge R²  (full, one-step)"); ax.set_ylabel("MLP R²  (full, one-step)")
    frac2 = float((y2 > x2).mean())
    ax.set_title(f"Ridge vs MLP (with self-history)\nMLP wins {frac2:.0%}")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_dir / "linearity_full_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Table: median R² per condition ──────────────────────────────────────
    print("\n  Condition medians:")
    for k in CONDS:
        m = np.median(vals[k])
        q25, q75 = np.percentile(vals[k], [25, 75])
        print(f"    {_COND_LABELS[k].replace(chr(10), ' '):20s}  "
              f"median={m:.4f}  IQR=[{q25:.3f}, {q75:.3f}]")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(description="Linearity & AR test")
    pa.add_argument("--h5", type=Path, required=True)
    pa.add_argument("--out_dir", type=Path,
                    default=_ROOT / "output_plots/masked_neuron_prediction/linearity_test")
    pa.add_argument("--lag_sec", type=float, default=3.0)
    pa.add_argument("--n_folds", type=int, default=5)
    pa.add_argument("--max_pca", type=int, default=50)
    pa.add_argument("--hidden", type=int, default=128)
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--weight_decay", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=20)
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--neurons", type=str, nargs="+", default=None,
                    help="Subset of neuron names (default: all)")
    args = pa.parse_args()
    np.random.seed(args.seed)

    worm = _load_worm(args.h5)
    if worm is None:
        sys.exit(f"Cannot load {args.h5}")

    u, labels, dt = worm["u"], worm["labels"], worm["dt"]
    T, N = u.shape
    n_lag = max(1, int(round(args.lag_sec / dt)))
    print(f"Worm: {worm['name']}  T={T}  N={N}  dt={dt:.3f}s  n_lag={n_lag} "
          f"({n_lag * dt:.1f}s)")
    print(f"Cross-neuron raw features: {n_lag} × {N-1} = {n_lag * (N-1)}"
          f"  → PCA({args.max_pca})")
    print(f"Self features: {n_lag}")
    print(f"Full features: {args.max_pca} + {n_lag} = {args.max_pca + n_lag}")
    print(f"Device: {args.device}")

    if args.neurons:
        target_indices = [(i, lb) for i, lb in enumerate(labels)
                          if lb in args.neurons]
        missing = set(args.neurons) - {lb for _, lb in target_indices}
        if missing:
            print(f"  [warn] not found: {sorted(missing)}")
    else:
        target_indices = list(enumerate(labels))

    print(f"Neurons to test: {len(target_indices)}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    t0_total = time.time()

    for ni, (tidx, name) in enumerate(target_indices):
        t0 = time.time()
        r = _run_one_neuron(
            u, tidx, n_lag,
            n_folds=args.n_folds, max_pca=args.max_pca,
            device_str=args.device, seed=args.seed,
            hidden=args.hidden, dropout=args.dropout,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, weight_decay=args.weight_decay,
            patience=args.patience,
        )
        dt_sec = time.time() - t0
        r["neuron"] = name
        r["n_lag"]  = n_lag
        results.append(r)
        print(f"  [{ni+1:3d}/{len(target_indices)}] {name:10s}  "
              f"xn_ridge={r['xn_ridge']:.3f}  xn_mlp={r['xn_mlp']:.3f}  "
              f"full_mlp={r['full_mlp']:.3f}  ar={r['full_mlp_ar']:.3f}  "
              f"({dt_sec:.1f}s)")

    elapsed = time.time() - t0_total
    print(f"\nDone — {len(results)} neurons, {elapsed:.0f}s total\n")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = args.out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["neuron", "n_lag"] + CONDS)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in ["neuron", "n_lag"] + CONDS})
    print(f"  CSV: {csv_path}")

    # ── Save meta ────────────────────────────────────────────────────────────
    meta = {
        "worm": worm["name"], "T": T, "N": N, "dt": dt,
        "lag_sec": args.lag_sec, "n_lag": n_lag,
        "n_folds": args.n_folds, "max_pca": args.max_pca,
        "n_neurons": len(results),
        "device": args.device, "elapsed_sec": elapsed,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ── Plot ─────────────────────────────────────────────────────────────────
    _plot_results(args.out_dir, results, worm["name"])

    # ── Summary stats ────────────────────────────────────────────────────────
    print("\n  Summary:")
    for k in CONDS:
        v = [r[k] for r in results]
        print(f"    {k:20s}  median={np.median(v):.4f}  "
              f"mean={np.mean(v):.4f}")


if __name__ == "__main__":
    main()
