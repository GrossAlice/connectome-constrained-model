#!/usr/bin/env python3
"""
Two clean tests on one worm — **no PCA**, raw features only.

Test 1 — Architecture sweep  (fixed lag = 5 frames ≈ 3 s)
  Ridge vs 5 MLP architectures, cross-neuron features only.
  Architectures:
    mlp_1x64    1 hidden, 64 units
    mlp_1x256   1 hidden, 256 units
    mlp_2x128   2 hidden, 128 units  (previous default)
    mlp_2x256   2 hidden, 256 units
    mlp_3x256   3 hidden, 256 units

Test 2 — Lag sweep  (1, 2, 3, 5 frames)
  Ridge + MLP(2×128) + AR variants, cross-neuron + self features.
  One-step and free-running AR.

All conditions use strict-causal features, 5-fold temporal CV.
Cross-neuron features are NOT PCA-compressed.

Usage
-----
  python -m scripts.masked_neuron.linearity_test_v2 \
      --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
      --device cuda
"""
from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path
from dataclasses import dataclass

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
#  Feature construction  (strict causal, no PCA)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_features(u: np.ndarray, target_idx: int, n_lag: int, max_lag: int):
    """
    Strict causal: features use lags 1..n_lag  (t−1 to t−n_lag).
    All outputs are aligned to the same T_out = T − max_lag  so different
    lag values are evaluated on exactly the same time points.

    Returns  X_cross (T_out, n_lag*N_other),  X_self (T_out, n_lag),  y (T_out,)
    """
    T, N = u.shape
    T_out = T - max_lag
    cross_mask = np.ones(N, dtype=bool)
    cross_mask[target_idx] = False
    N_cross = int(cross_mask.sum())

    X_cross = np.empty((T_out, n_lag * N_cross), dtype=np.float32)
    X_self  = np.empty((T_out, n_lag), dtype=np.float32)

    for k in range(n_lag):
        # k=0 → lag 1 (t-1),  k=n_lag-1 → lag n_lag (t-n_lag)
        start = max_lag - 1 - k
        X_cross[:, k * N_cross : (k + 1) * N_cross] = \
            u[start : start + T_out][:, cross_mask]
        X_self[:, k] = u[start : start + T_out, target_idx]

    y = u[max_lag : max_lag + T_out, target_idx].astype(np.float32)
    return X_cross, X_self, y


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
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
    return X.mean(0), X.std(0).clip(1e-8)


def _z(X, mu, std):
    return ((X - mu) / std).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  MLP  (flexible architecture)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Arch:
    name: str
    n_layers: int
    width: int
    dropout: float = 0.1


ARCHS = [
    Arch("mlp_1x64",   1,  64),
    Arch("mlp_1x256",  1, 256),
    Arch("mlp_2x128",  2, 128),      # previous default
    Arch("mlp_2x256",  2, 256),
    Arch("mlp_3x256",  3, 256),
]


def _make_mlp(in_dim: int, arch: Arch):
    import torch.nn as nn
    layers = []
    prev = in_dim
    for i in range(arch.n_layers):
        layers.append(nn.Linear(prev, arch.width))
        layers.append(nn.ReLU())
        if arch.dropout > 0:
            layers.append(nn.Dropout(arch.dropout))
        prev = arch.width
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


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


def _run_mlp_cv(X, y, folds, arch, device, seed,
                epochs=200, lr=1e-3, batch_size=64,
                weight_decay=1e-4, patience=20):
    """Run one MLP architecture through CV folds, return R²."""
    import torch
    dev = torch.device(device)
    to_t = lambda a: torch.from_numpy(a.astype(np.float32)).to(dev)
    T_out = len(y)
    pred = np.zeros(T_out, np.float32)
    for fi, (tr_idx, te_idx) in enumerate(folds):
        Xtr, Xte = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]
        mu_x, std_x = _zs(Xtr)
        mu_y = float(y_tr.mean())
        std_y = float(max(y_tr.std(), 1e-8))
        n_va = max(1, int(len(tr_idx) * 0.2))
        Xtr_z = to_t(_z(Xtr[:-n_va], mu_x, std_x))
        Xva_z = to_t(_z(Xtr[-n_va:], mu_x, std_x))
        Ytr_z = to_t(((y_tr[:-n_va] - mu_y) / std_y).reshape(-1, 1))
        Yva_z = to_t(((y_tr[-n_va:] - mu_y) / std_y).reshape(-1, 1))
        torch.manual_seed(seed + fi)
        net = _make_mlp(X.shape[1], arch).to(dev)
        _train_mlp(net, Xtr_z, Ytr_z, Xva_z, Yva_z,
                   epochs=epochs, lr=lr, batch_size=batch_size,
                   weight_decay=weight_decay, patience=patience)
        net.eval()
        with torch.no_grad():
            p_z = net(to_t(_z(Xte, mu_x, std_x))).cpu().numpy().ravel()
        pred[te_idx] = p_z * std_y + mu_y
    return _r2(y, pred)


def _run_ridge_cv(X, y, folds):
    """Run ridge through CV folds, return R²."""
    from sklearn.linear_model import RidgeCV
    T_out = len(y)
    pred = np.zeros(T_out, np.float32)
    for tr_idx, te_idx in folds:
        Xtr = X[tr_idx]
        mu_x, std_x = _zs(Xtr)
        mu_y = float(y[tr_idx].mean())
        std_y = float(max(y[tr_idx].std(), 1e-8))
        r = RidgeCV(alphas=_ALPHAS, fit_intercept=True)
        r.fit(_z(Xtr, mu_x, std_x), (y[tr_idx] - mu_y) / std_y)
        p_z = r.predict(_z(X[te_idx], mu_x, std_x)).ravel()
        pred[te_idx] = (p_z * std_y + mu_y).astype(np.float32)
    return _r2(y, pred)


# ═══════════════════════════════════════════════════════════════════════════════
#  AR evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _ar_evaluate_mlp(net, Xc_te, Xs_te, mu_f, std_f, mu_y, std_y, device):
    """Free-running AR for MLP on one test fold."""
    import torch
    T = Xc_te.shape[0]
    preds = np.zeros(T, np.float32)
    buf = Xs_te[0].copy()
    net.eval()
    with torch.no_grad():
        for t in range(T):
            x = np.concatenate([Xc_te[t], buf])
            x_z = torch.from_numpy(_z(x.reshape(1, -1), mu_f, std_f)).to(device)
            p_z = net(x_z).item()
            p = p_z * std_y + mu_y
            preds[t] = p
            buf = np.roll(buf, 1); buf[0] = p
    return preds


def _ar_evaluate_ridge(ridge, Xc_te, Xs_te, mu_f, std_f, mu_y, std_y):
    """Free-running AR for ridge on one test fold."""
    T = Xc_te.shape[0]
    preds = np.zeros(T, np.float32)
    buf = Xs_te[0].copy()
    for t in range(T):
        x = np.concatenate([Xc_te[t], buf])
        x_z = _z(x.reshape(1, -1), mu_f, std_f)
        p_z = float(ridge.predict(x_z).ravel()[0])
        p = p_z * std_y + mu_y
        preds[t] = p
        buf = np.roll(buf, 1); buf[0] = p
    return preds


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — Architecture sweep (fixed lag, cross-neuron only)
# ═══════════════════════════════════════════════════════════════════════════════

def run_test1(u, labels, n_lag, max_lag, n_folds, device, seed,
              epochs, lr, batch_size, weight_decay, patience):
    """Return list[dict] with {neuron, ridge, mlp_1x64, ...}."""
    arch_names = [a.name for a in ARCHS]
    conds = ["ridge"] + arch_names
    results = []
    N = u.shape[1]

    for ni in range(N):
        X_cross, _, y = _build_features(u, ni, n_lag, max_lag)
        folds = _make_folds(len(y), n_folds)
        t0 = time.time()

        r = {"neuron": labels[ni]}
        r["ridge"] = _run_ridge_cv(X_cross, y, folds)

        for arch in ARCHS:
            r[arch.name] = _run_mlp_cv(
                X_cross, y, folds, arch, device, seed,
                epochs=epochs, lr=lr, batch_size=batch_size,
                weight_decay=weight_decay, patience=patience)

        dt = time.time() - t0
        results.append(r)
        best_mlp_name = max(arch_names, key=lambda a: r[a])
        print(f"  T1 [{ni+1:3d}/{N}] {labels[ni]:10s}  "
              f"ridge={r['ridge']:.3f}  best_mlp={best_mlp_name}({r[best_mlp_name]:.3f})  "
              f"({dt:.1f}s)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — Lag sweep (ridge + MLP + AR, cross + self)
# ═══════════════════════════════════════════════════════════════════════════════

def run_test2(u, labels, lag_frames, max_lag, n_folds, device, seed,
              epochs, lr, batch_size, weight_decay, patience):
    """Return list[dict] with {neuron, lag, xn_ridge, xn_mlp, full_ridge, full_mlp, full_ridge_ar, full_mlp_ar}."""
    import torch
    from sklearn.linear_model import RidgeCV

    arch = Arch("mlp_2x128", 2, 128)
    results = []
    N = u.shape[1]
    dev = torch.device(device)
    to_t = lambda a: torch.from_numpy(a.astype(np.float32)).to(dev)

    for lag in lag_frames:
        print(f"\n  === Test 2:  lag = {lag} frames ===")
        for ni in range(N):
            X_cross, X_self, y = _build_features(u, ni, lag, max_lag)
            T_out = len(y)
            folds = _make_folds(T_out, n_folds)
            X_full = np.hstack([X_cross, X_self])
            t0 = time.time()

            # -- cross-neuron ridge
            xn_ridge = _run_ridge_cv(X_cross, y, folds)

            # -- cross-neuron MLP
            xn_mlp = _run_mlp_cv(X_cross, y, folds, arch, device, seed,
                                 epochs, lr, batch_size, weight_decay, patience)

            # -- full ridge + MLP (one-step) and AR  — need per-fold models
            pred_fr = np.zeros(T_out, np.float32)
            pred_fm = np.zeros(T_out, np.float32)
            pred_fr_ar = np.zeros(T_out, np.float32)
            pred_fm_ar = np.zeros(T_out, np.float32)

            for fi, (tr_idx, te_idx) in enumerate(folds):
                Xc_tr, Xc_te = X_cross[tr_idx], X_cross[te_idx]
                Xs_tr, Xs_te = X_self[tr_idx],  X_self[te_idx]
                Xf_tr = np.hstack([Xc_tr, Xs_tr])
                Xf_te = np.hstack([Xc_te, Xs_te])
                y_tr = y[tr_idx]

                mu_f, std_f = _zs(Xf_tr)
                mu_y = float(y_tr.mean())
                std_y = float(max(y_tr.std(), 1e-8))
                y_tr_z = (y_tr - mu_y) / std_y
                n_va = max(1, int(len(tr_idx) * 0.2))

                # full ridge
                ridge_f = RidgeCV(alphas=_ALPHAS, fit_intercept=True)
                ridge_f.fit(_z(Xf_tr, mu_f, std_f), y_tr_z)
                p_z = ridge_f.predict(_z(Xf_te, mu_f, std_f)).ravel()
                pred_fr[te_idx] = p_z * std_y + mu_y
                # full ridge AR
                pred_fr_ar[te_idx] = _ar_evaluate_ridge(
                    ridge_f, Xc_te, Xs_te, mu_f, std_f, mu_y, std_y)

                # full MLP
                Xtr_z = to_t(_z(Xf_tr[:-n_va], mu_f, std_f))
                Xva_z = to_t(_z(Xf_tr[-n_va:], mu_f, std_f))
                Ytr_z = to_t(y_tr_z[:-n_va].reshape(-1, 1))
                Yva_z = to_t(y_tr_z[-n_va:].reshape(-1, 1))
                torch.manual_seed(seed + fi + 2000)
                net_f = _make_mlp(Xf_tr.shape[1], arch).to(dev)
                _train_mlp(net_f, Xtr_z, Ytr_z, Xva_z, Yva_z,
                           epochs=epochs, lr=lr, batch_size=batch_size,
                           weight_decay=weight_decay, patience=patience)
                net_f.eval()
                with torch.no_grad():
                    p_z = net_f(to_t(_z(Xf_te, mu_f, std_f))).cpu().numpy().ravel()
                pred_fm[te_idx] = p_z * std_y + mu_y
                # full MLP AR
                pred_fm_ar[te_idx] = _ar_evaluate_mlp(
                    net_f, Xc_te, Xs_te, mu_f, std_f, mu_y, std_y, dev)

            dt_s = time.time() - t0
            r = {
                "neuron": labels[ni], "lag": lag,
                "xn_ridge": xn_ridge, "xn_mlp": xn_mlp,
                "full_ridge": _r2(y, pred_fr), "full_mlp": _r2(y, pred_fm),
                "full_ridge_ar": _r2(y, pred_fr_ar),
                "full_mlp_ar": _r2(y, pred_fm_ar),
            }
            results.append(r)
            print(f"  T2 [{ni+1:3d}/{N}] lag={lag} {labels[ni]:10s}  "
                  f"xn_r={xn_ridge:.3f} xn_m={xn_mlp:.3f}  "
                  f"full_m={r['full_mlp']:.3f} ar={r['full_mlp_ar']:.3f}  "
                  f"({dt_s:.1f}s)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting — Test 1
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_test1(out_dir: Path, results: list[dict], worm_name: str, n_lag: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    arch_names = [a.name for a in ARCHS]
    conds = ["ridge"] + arch_names
    N = len(results)
    vals = {c: np.array([r[c] for r in results]) for c in conds}

    colors = {
        "ridge":     "#3498db",
        "mlp_1x64":  "#f39c12",
        "mlp_1x256": "#e67e22",
        "mlp_2x128": "#e74c3c",
        "mlp_2x256": "#c0392b",
        "mlp_3x256": "#8e44ad",
    }
    labels_nice = {
        "ridge":     "Ridge",
        "mlp_1x64":  "MLP 1×64",
        "mlp_1x256": "MLP 1×256",
        "mlp_2x128": "MLP 2×128",
        "mlp_2x256": "MLP 2×256",
        "mlp_3x256": "MLP 3×256",
    }

    # ── Panel A: violin of all conditions ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    positions = np.arange(len(conds))
    parts = ax.violinplot([vals[c] for c in conds], positions=positions,
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[conds[i]])
        pc.set_alpha(0.5)
    parts["cmedians"].set_color("k")
    rng = np.random.default_rng(42)
    for i, c in enumerate(conds):
        jitter = rng.normal(0, 0.07, N)
        ax.scatter(positions[i] + jitter, vals[c], s=8, alpha=0.4,
                   c=colors[c], edgecolors="none", zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels_nice[c] for c in conds], fontsize=9)
    ax.set_ylabel("R²  (cross-neuron, one-step)", fontsize=10)
    ax.axhline(0, color="k", lw=0.5, ls=":")

    # annotate medians
    for i, c in enumerate(conds):
        m = np.median(vals[c])
        ax.text(i, ax.get_ylim()[1] + 0.01, f"{m:.3f}", ha="center",
                fontsize=8, color=colors[c], fontweight="bold")
    ax.set_title(f"A — Architecture comparison  (lag={n_lag} frames, n={N})",
                 fontsize=11)

    # ── Panel B: scatter ridge vs best MLP per neuron ───────────────────────
    ax = axes[1]
    # for each neuron, pick the best MLP
    best_mlp = np.array([max(r[a] for a in arch_names) for r in results])
    best_names = [max(arch_names, key=lambda a: r[a]) for r in results]
    x, y_ = vals["ridge"], best_mlp
    for a in arch_names:
        mask = np.array([bn == a for bn in best_names])
        if mask.any():
            ax.scatter(x[mask], y_[mask], s=18, alpha=0.6,
                       c=colors[a], edgecolors="k", lw=0.3, zorder=3,
                       label=f"{labels_nice[a]} ({mask.sum()})")
    lo = min(x.min(), y_.min()) - 0.05
    hi = max(x.max(), y_.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Ridge R²", fontsize=10)
    ax.set_ylabel("Best MLP R²", fontsize=10)
    frac = float((y_ > x).mean())
    delta = float(np.median(y_ - x))
    ax.set_title(f"B — Ridge vs best MLP\n"
                 f"MLP wins {frac:.0%}  |  median Δ = {delta:+.3f}", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(f"Test 1 — Architecture sweep — {worm_name}\n"
                 f"(cross-neuron only, strict causal, no PCA, 5-fold CV)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "test1_arch_sweep.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "test1_arch_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'test1_arch_sweep.png'}")

    # summary
    print("\n  Test 1 medians:")
    for c in conds:
        m = np.median(vals[c])
        q25, q75 = np.percentile(vals[c], [25, 75])
        print(f"    {labels_nice[c]:14s}  median={m:.4f}  IQR=[{q25:.3f}, {q75:.3f}]")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting — Test 2
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_test2(out_dir: Path, results: list[dict], worm_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    lags = sorted(set(r["lag"] for r in results))
    conds = ["xn_ridge", "xn_mlp", "full_ridge", "full_mlp",
             "full_ridge_ar", "full_mlp_ar"]
    cond_labels = {
        "xn_ridge":      "Cross ridge",
        "xn_mlp":        "Cross MLP",
        "full_ridge":    "Full ridge",
        "full_mlp":      "Full MLP",
        "full_ridge_ar": "Full ridge AR",
        "full_mlp_ar":   "Full MLP AR",
    }
    cond_colors = {
        "xn_ridge":      "#3498db",
        "xn_mlp":        "#e74c3c",
        "full_ridge":    "#2980b9",
        "full_mlp":      "#c0392b",
        "full_ridge_ar": "#1abc9c",
        "full_mlp_ar":   "#e67e22",
    }

    # ── Panel A: median R² vs lag  (line plot) ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    for c in conds:
        medians = []
        for lag in lags:
            v = [r[c] for r in results if r["lag"] == lag]
            medians.append(float(np.median(v)))
        ax.plot(lags, medians, "o-", label=cond_labels[c],
                color=cond_colors[c], lw=2, ms=7)
    ax.set_xlabel("Lag (frames)", fontsize=10)
    ax.set_ylabel("Median R²", fontsize=10)
    ax.set_xticks(lags)
    ax.legend(fontsize=8, loc="best")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_title("A — Median R² vs lag (all neurons)", fontsize=11)

    # ── Panel B: violin per lag for xn_ridge vs xn_mlp ──────────────────────
    ax = axes[1]
    group_w = 0.35
    positions_r = np.arange(len(lags)) - group_w / 2
    positions_m = np.arange(len(lags)) + group_w / 2
    data_r = [[r["xn_ridge"] for r in results if r["lag"] == lag] for lag in lags]
    data_m = [[r["xn_mlp"]   for r in results if r["lag"] == lag] for lag in lags]
    vp1 = ax.violinplot(data_r, positions=positions_r, widths=group_w * 0.9,
                         showmedians=True, showextrema=False)
    for pc in vp1["bodies"]:
        pc.set_facecolor("#3498db"); pc.set_alpha(0.5)
    vp1["cmedians"].set_color("k")
    vp2 = ax.violinplot(data_m, positions=positions_m, widths=group_w * 0.9,
                         showmedians=True, showextrema=False)
    for pc in vp2["bodies"]:
        pc.set_facecolor("#e74c3c"); pc.set_alpha(0.5)
    vp2["cmedians"].set_color("k")
    ax.set_xticks(np.arange(len(lags)))
    ax.set_xticklabels([str(l) for l in lags])
    ax.set_xlabel("Lag (frames)", fontsize=10)
    ax.set_ylabel("R²  (cross-neuron only)", fontsize=10)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#3498db", alpha=0.5, label="Ridge"),
                       Patch(facecolor="#e74c3c", alpha=0.5, label="MLP 2×128")],
              fontsize=9, loc="best")
    ax.set_title("B — Cross-neuron: Ridge vs MLP per lag", fontsize=11)

    # ── Panel C: one-step vs AR drop by lag ─────────────────────────────────
    ax = axes[2]
    os_medians, ar_medians = [], []
    for lag in lags:
        os_v = [r["full_mlp"] for r in results if r["lag"] == lag]
        ar_v = [r["full_mlp_ar"] for r in results if r["lag"] == lag]
        os_medians.append(float(np.median(os_v)))
        ar_medians.append(float(np.median(ar_v)))
    ax.plot(lags, os_medians, "o-", label="One-step", color="#c0392b", lw=2, ms=7)
    ax.plot(lags, ar_medians, "s--", label="AR (free-run)", color="#e67e22", lw=2, ms=7)
    for i, lag in enumerate(lags):
        drop = os_medians[i] - ar_medians[i]
        ax.annotate(f"Δ={drop:.2f}", xy=(lag, ar_medians[i]),
                    xytext=(lag + 0.15, ar_medians[i] - 0.03),
                    fontsize=8, color="#7f8c8d")
    ax.set_xlabel("Lag (frames)", fontsize=10)
    ax.set_ylabel("Median R²  (full MLP: cross + self)", fontsize=10)
    ax.set_xticks(lags)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.legend(fontsize=9)
    ax.set_title("C — AR degradation by lag", fontsize=11)

    fig.suptitle(f"Test 2 — Lag sweep — {worm_name}\n"
                 f"(strict causal, no PCA, 5-fold CV, MLP 2×128)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "test2_lag_sweep.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "test2_lag_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'test2_lag_sweep.png'}")

    # summary table
    print("\n  Test 2 medians:")
    for lag in lags:
        vals_lag = {c: [r[c] for r in results if r["lag"] == lag] for c in conds}
        line = f"    lag={lag}:  "
        line += "  ".join(f"{cond_labels[c][:8]}={np.median(vals_lag[c]):.3f}"
                          for c in conds)
        print(line)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(description="Linearity & AR test v2 (no PCA)")
    pa.add_argument("--h5", type=Path, required=True)
    pa.add_argument("--out_dir", type=Path,
                    default=_ROOT / "output_plots/masked_neuron_prediction/linearity_v2")
    pa.add_argument("--n_folds", type=int, default=5)
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--weight_decay", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=20)
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--skip_test1", action="store_true")
    pa.add_argument("--skip_test2", action="store_true")
    args = pa.parse_args()
    np.random.seed(args.seed)

    worm = _load_worm(args.h5)
    if worm is None:
        sys.exit(f"Cannot load {args.h5}")

    u, labels, dt = worm["u"], worm["labels"], worm["dt"]
    T, N = u.shape

    LAG_FRAMES = [1, 2, 3, 5]
    MAX_LAG = max(LAG_FRAMES)         # = 5, all features aligned to same T_out
    ARCH_LAG = 5                       # test 1 uses 5 frames

    print(f"Worm: {worm['name']}  T={T}  N={N}  dt={dt:.3f}s")
    print(f"Lag frames tested: {LAG_FRAMES}  (max_lag={MAX_LAG})")
    print(f"T_out = {T - MAX_LAG}")
    print(f"Cross features @ lag=1: {N-1}   @ lag=5: {5*(N-1)}")
    print(f"Samples: {T - MAX_LAG}  →  features/samples @ lag=5: "
          f"{5*(N-1)/(T - MAX_LAG):.2f}")
    print(f"No PCA — raw features")
    print(f"Device: {args.device}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Test 1 ───────────────────────────────────────────────────────────────
    if not args.skip_test1:
        print("═" * 60)
        print("  TEST 1 — Architecture sweep (lag=5, cross-neuron, no PCA)")
        print("═" * 60)
        t0 = time.time()
        res1 = run_test1(u, labels, ARCH_LAG, MAX_LAG, args.n_folds,
                         args.device, args.seed,
                         args.epochs, args.lr, args.batch_size,
                         args.weight_decay, args.patience)
        dt1 = time.time() - t0
        print(f"\n  Test 1 done in {dt1:.0f}s")

        # save CSV
        arch_names = [a.name for a in ARCHS]
        csv1 = args.out_dir / "test1_results.csv"
        with open(csv1, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["neuron", "ridge"] + arch_names)
            w.writeheader()
            for r in res1:
                w.writerow({k: r[k] for k in ["neuron", "ridge"] + arch_names})
        print(f"  CSV: {csv1}")

        _plot_test1(args.out_dir, res1, worm["name"], ARCH_LAG)
    else:
        print("  Skipping Test 1")

    # ── Test 2 ───────────────────────────────────────────────────────────────
    if not args.skip_test2:
        print("\n" + "═" * 60)
        print("  TEST 2 — Lag sweep (1,2,3,5 frames, no PCA)")
        print("═" * 60)
        t0 = time.time()
        res2 = run_test2(u, labels, LAG_FRAMES, MAX_LAG, args.n_folds,
                         args.device, args.seed,
                         args.epochs, args.lr, args.batch_size,
                         args.weight_decay, args.patience)
        dt2 = time.time() - t0
        print(f"\n  Test 2 done in {dt2:.0f}s")

        # save CSV
        t2_conds = ["xn_ridge", "xn_mlp", "full_ridge", "full_mlp",
                     "full_ridge_ar", "full_mlp_ar"]
        csv2 = args.out_dir / "test2_results.csv"
        with open(csv2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["neuron", "lag"] + t2_conds)
            w.writeheader()
            for r in res2:
                w.writerow({k: r[k] for k in ["neuron", "lag"] + t2_conds})
        print(f"  CSV: {csv2}")

        _plot_test2(args.out_dir, res2, worm["name"])
    else:
        print("  Skipping Test 2")

    # ── Save meta ────────────────────────────────────────────────────────────
    meta = {
        "worm": worm["name"], "T": T, "N": N, "dt": dt,
        "lag_frames": LAG_FRAMES, "max_lag": MAX_LAG,
        "arch_lag": ARCH_LAG,
        "n_folds": args.n_folds, "no_pca": True,
        "device": args.device,
        "archs": [{"name": a.name, "layers": a.n_layers, "width": a.width}
                  for a in ARCHS],
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  All done.  Results in {args.out_dir}")


if __name__ == "__main__":
    main()
