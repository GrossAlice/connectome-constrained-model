#!/usr/bin/env python3
"""
Run ridge-linear + end-to-end MLP on ONE worm (ALL neurons, strict mode).
Track selected ridge alpha (for the linear model) to check grid boundary.
Produce a violin plot: one dot per neuron, ridge-linear vs MLP.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).parent.parent.parent
_RIDGE_ALPHAS = np.logspace(-6, 8, 60)
_ALPHA_LO, _ALPHA_HI = _RIDGE_ALPHAS[0], _RIDGE_ALPHAS[-1]


# ── reuse helpers from the main script ──────────────────────────────────────

def _load_worm(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        u = np.array(f["stage1/u_mean"], dtype=np.float32)
        labels = f["gcamp/neuron_labels"][:]
        dt = 0.6
        if "timing/timestamp_confocal" in f:
            ts = f["timing/timestamp_confocal"][:]
            if len(ts) > 1:
                dt = float(np.median(np.diff(ts)))
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


def _build_features_strict(u, target_idx, n_lags):
    T, N = u.shape
    feature_cols = np.ones(N, dtype=bool)
    feature_cols[target_idx] = False
    T_out = T - n_lags
    blocks = []
    for lag in range(1, n_lags + 1):
        start = n_lags - lag
        block = u[start : start + T_out, :][:, feature_cols]
        blocks.append(block)
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    y = u[n_lags : n_lags + T_out, target_idx].astype(np.float32)
    return X, y


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


def _inner_split(tr_idx, frac=0.2):
    n_va = max(1, int(len(tr_idx) * frac))
    return tr_idx[:-n_va], tr_idx[-n_va:]


def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _zscore(Xtr):
    mu = Xtr.mean(0, keepdims=True)
    std = Xtr.std(0, keepdims=True).clip(1e-8)
    return mu, std


# ── Ridge-linear (with alpha tracking) ──────────────────────────────────────

def _run_ridge_linear(X, y, n_folds):
    from sklearn.linear_model import RidgeCV
    T = X.shape[0]
    folds = _make_folds(T, n_folds)
    pred = np.zeros(T, dtype=np.float32)
    alphas_chosen = []
    for tr_idx, te_idx in folds:
        ridge = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
        ridge.fit(X[tr_idx], y[tr_idx])
        pred[te_idx] = ridge.predict(X[te_idx]).astype(np.float32)
        alphas_chosen.append(float(ridge.alpha_))
    return pred, _r2(y, pred), alphas_chosen


# ── MLP (end-to-end, no ridge readout) ───────────────────────────────────────

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


def _train_mlp(net, Xtr, Ytr, Xva, Yva, *, epochs, lr, batch_size,
               weight_decay, patience, device):
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
    if best_state is not None:
        net.load_state_dict(best_state)


def _run_mlp(X, y, n_folds, hidden, dropout, epochs, lr, batch_size,
             weight_decay, patience, device_str, seed):
    import torch
    import torch.nn as nn
    device = torch.device(device_str)
    T = X.shape[0]
    folds = _make_folds(T, n_folds)
    pred = np.zeros(T, dtype=np.float32)
    for fi, (tr_idx, te_idx) in enumerate(folds):
        tr_inner, va_inner = _inner_split(tr_idx)
        Xtr, Ytr = X[tr_inner], y[tr_inner]
        Xva, Yva = X[va_inner], y[va_inner]
        Xte = X[te_idx]
        mu_x, std_x = _zscore(Xtr)
        mu_y = float(Ytr.mean())
        std_y = float(max(Ytr.std(), 1e-8))
        to_t = lambda a: torch.from_numpy(a).to(device)
        Xtr_z = to_t(((Xtr - mu_x) / std_x).astype(np.float32))
        Ytr_z = to_t(((Ytr.reshape(-1, 1) - mu_y) / std_y).astype(np.float32))
        Xva_z = to_t(((Xva - mu_x) / std_x).astype(np.float32))
        Yva_z = to_t(((Yva.reshape(-1, 1) - mu_y) / std_y).astype(np.float32))
        torch.manual_seed(seed + fi)
        net = _make_mlp(X.shape[1], hidden, 1, dropout).to(device)
        _train_mlp(net, Xtr_z, Ytr_z, Xva_z, Yva_z,
                   epochs=epochs, lr=lr, batch_size=batch_size,
                   weight_decay=weight_decay, patience=patience, device=device)
        # Predict directly with full MLP
        net.eval()
        Xte_z = to_t(((Xte - mu_x) / std_x).astype(np.float32))
        with torch.no_grad():
            pred_z = net(Xte_z).cpu().numpy().ravel()
        pred[te_idx] = (pred_z * std_y + mu_y).astype(np.float32)
    return pred, _r2(y, pred)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--h5", type=Path, required=True, help="Path to single worm H5")
    pa.add_argument("--out_dir", type=Path,
                    default=_ROOT / "output_plots/masked_neuron_prediction/one_worm_violin")
    pa.add_argument("--lag_sec", type=float, default=10.0)
    pa.add_argument("--n_folds", type=int, default=5)
    pa.add_argument("--hidden", type=int, default=128)
    pa.add_argument("--epochs", type=int, default=300)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--weight_decay", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=30)
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--seed", type=int, default=0)
    args = pa.parse_args()
    np.random.seed(args.seed)

    worm = _load_worm(args.h5)
    if worm is None:
        sys.exit(f"Failed to load {args.h5}")
    u, labels = worm["u"], worm["labels"]
    dt = worm["dt"]
    T, N = u.shape
    n_lags = max(1, int(round(args.lag_sec / dt)))
    print(f"Worm: {worm['name']}  T={T}  N={N}  dt={dt:.3f}s  n_lags={n_lags}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    results = []
    all_alphas_linear = []

    for ni, neuron_name in enumerate(labels):
        target_idx = ni
        X, y = _build_features_strict(u, target_idx, n_lags)

        # Ridge-linear
        t0 = time.time()
        pred_rl, r2_rl, alphas_rl = _run_ridge_linear(X, y, args.n_folds)
        dt_rl = time.time() - t0
        all_alphas_linear.extend(alphas_rl)

        # MLP (end-to-end)
        t0 = time.time()
        pred_rm, r2_rm = _run_mlp(
            X, y, args.n_folds,
            hidden=args.hidden, dropout=args.dropout, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch_size,
            weight_decay=args.weight_decay, patience=args.patience,
            device_str=args.device, seed=args.seed,
        )
        dt_rm = time.time() - t0

        results.append({
            "neuron": neuron_name,
            "r2_linear": r2_rl,
            "r2_mlp": r2_rm,
            "alphas_linear": alphas_rl,
        })
        print(f"  [{ni+1:3d}/{N}] {neuron_name:8s}  "
              f"ridge-lin R²={r2_rl:+.4f}  mlp R²={r2_rm:+.4f}  "
              f"α_lin={np.median(alphas_rl):.1e}  "
              f"({dt_rl:.1f}s / {dt_rm:.1f}s)")

    # ── Alpha boundary analysis (ridge-linear only) ──────────────────────
    al = np.array(all_alphas_linear)
    n_lo_lin = int((al <= _ALPHA_LO * 1.01).sum())
    n_hi_lin = int((al >= _ALPHA_HI * 0.99).sum())
    n_total = len(al)
    print(f"\n{'═'*60}")
    print(f"  Alpha boundary check  (grid: {_ALPHA_LO:.0e} .. {_ALPHA_HI:.0e})")
    print(f"  Ridge-linear:  {n_lo_lin}/{n_total} at lower bound, "
          f"{n_hi_lin}/{n_total} at upper bound")
    print(f"  Median α:  linear={np.median(al):.2e}")
    print(f"{'═'*60}")

    # ── Save JSON ────────────────────────────────────────────────────────
    r2_lin = np.array([r["r2_linear"] for r in results])
    r2_mlp = np.array([r["r2_mlp"] for r in results])
    summary = {
        "worm": worm["name"], "T": T, "N": N, "n_lags": n_lags,
        "causal_mode": "strict", "n_folds": args.n_folds,
        "median_r2_linear": float(np.median(r2_lin)),
        "median_r2_mlp": float(np.median(r2_mlp)),
        "alpha_boundary": {
            "grid_lo": float(_ALPHA_LO), "grid_hi": float(_ALPHA_HI),
            "linear_at_lo": n_lo_lin, "linear_at_hi": n_hi_lin,
            "total_folds": n_total,
        },
        "results": results,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))

    # ── Violin plot ──────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))

    data = [r2_lin, r2_mlp]
    positions = [0, 1]
    model_labels = ["Ridge-Linear", "MLP"]
    colors = ["#3498db", "#e74c3c"]

    parts = ax.violinplot(data, positions=positions, showmedians=False,
                          showextrema=False)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.3)
        pc.set_edgecolor(c)

    # Individual dots (one per neuron) with jitter
    rng = np.random.default_rng(42)
    for i, (vals, c) in enumerate(zip(data, colors)):
        jitter = rng.normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=18, alpha=0.6, color=c, edgecolors="0.3",
                   linewidths=0.3, zorder=3)

    # Median bars
    for i, vals in enumerate(data):
        med = float(np.median(vals))
        ax.plot([i - 0.2, i + 0.2], [med, med], color="k", lw=2.5, zorder=4)
        ax.text(i + 0.25, med, f"{med:.3f}", va="center", fontsize=11,
                fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(model_labels, fontsize=13)
    ax.set_ylabel("R²  (strict, 5-fold CV)", fontsize=13)
    ax.set_title(f"Per-neuron R²: {worm['name']}  (N={N}, strict)",
                 fontsize=14, fontweight="bold")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)

    # Annotate MLP win fraction
    frac = float((r2_mlp > r2_lin).mean())
    ax.text(0.5, 0.02, f"MLP > Linear in {frac:.0%} of neurons",
            transform=ax.transAxes, ha="center", fontsize=11,
            style="italic", color="0.3")

    fig.tight_layout()
    fig.savefig(out / "violin_ridge_vs_mlp.png", dpi=180, bbox_inches="tight")
    print(f"\n  Violin saved to {out / 'violin_ridge_vs_mlp.png'}")
    plt.close(fig)

    # ── Alpha distribution histogram (ridge-linear only) ─────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    log_a = np.log10(np.array(all_alphas_linear))
    ax.hist(log_a, bins=30, color="#3498db", alpha=0.7, edgecolor="k", lw=0.3)
    ax.axvline(np.log10(_ALPHA_LO), color="red", ls=":", lw=1.5,
               label="grid boundary")
    ax.axvline(np.log10(_ALPHA_HI), color="red", ls=":", lw=1.5)
    ax.set_xlabel("log₁₀(α chosen)")
    ax.set_ylabel("Count (folds)")
    ax.set_title("Ridge-Linear: chosen α distribution")
    ax.legend(fontsize=9)
    fig.suptitle(f"Ridge α selection — {worm['name']}  (strict)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "alpha_distribution.png", dpi=150, bbox_inches="tight")
    print(f"  Alpha plot saved to {out / 'alpha_distribution.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
