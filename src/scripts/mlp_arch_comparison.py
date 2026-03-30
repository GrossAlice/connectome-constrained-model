#!/usr/bin/env python3
"""
Compare MLP architectures for masked-neuron prediction.

Runs the same masked-neuron cross-validated prediction pipeline from
``masked_neuron_prediction.py`` but sweeps over MLP architecture variants:
  - hidden width   (32, 64, 128, 256, 512)
  - depth          (1, 2, 3 hidden layers)
  - dropout        (0.0, 0.1, 0.2)
  - skip/residual  connections
  - ridge-linear   baseline (no MLP)

All models use MLP backbone → RidgeCV readout (same as the original script).

Usage
-----
    # Single best worm for AVAR, all architectures
    python scripts/mlp_arch_comparison.py \
        --worm "2023-01-09-15" --neuron AVAR --device cuda

    # Top-3 worms, multiple neurons
    python scripts/mlp_arch_comparison.py \
        --worm "2023-01-09-15" "2023-01-19-01" "2023-01-10-07" \
        --neuron AVAR NSML AVL --device cuda

    # Dry-run: list architectures without training
    python scripts/mlp_arch_comparison.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_DEFAULT_DATASET_DIR = _ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2"
_DEFAULT_OUT_DIR = _ROOT / "output_plots/masked_neuron_prediction/mlp_comparison"
_RIDGE_ALPHAS = np.logspace(-2, 8, 60)


# ── architecture definitions ────────────────────────────────────────────────

@dataclass
class ArchConfig:
    """One MLP architecture variant."""
    name: str
    hidden: int = 64
    n_layers: int = 2
    dropout: float = 0.0
    residual: bool = False
    batch_norm: bool = False
    activation: str = "relu"      # "relu" | "gelu" | "silu"


def _default_architectures() -> list[ArchConfig]:
    archs = []

    # ── Width sweep (2-layer, no dropout) ────────────────────────────────
    for h in [32, 64, 128, 256, 512]:
        archs.append(ArchConfig(name=f"mlp-{h}x2", hidden=h, n_layers=2))

    # ── Depth sweep (hidden=128, no dropout) ─────────────────────────────
    for d in [1, 3, 4]:
        archs.append(ArchConfig(name=f"mlp-128x{d}", hidden=128, n_layers=d))

    # ── Dropout sweep (hidden=128, 2 layers) ────────────────────────────
    for dp in [0.1, 0.2, 0.3]:
        archs.append(ArchConfig(name=f"mlp-128x2-dp{dp}", hidden=128, n_layers=2, dropout=dp))

    # ── Residual (hidden=128, 2 & 3 layers) ─────────────────────────────
    archs.append(ArchConfig(name="mlp-128x2-res", hidden=128, n_layers=2, residual=True))
    archs.append(ArchConfig(name="mlp-128x3-res", hidden=128, n_layers=3, residual=True))

    # ── Batch-norm ───────────────────────────────────────────────────────
    archs.append(ArchConfig(name="mlp-128x2-bn", hidden=128, n_layers=2, batch_norm=True))

    # ── Activation variants ──────────────────────────────────────────────
    archs.append(ArchConfig(name="mlp-128x2-gelu", hidden=128, n_layers=2, activation="gelu"))
    archs.append(ArchConfig(name="mlp-128x2-silu", hidden=128, n_layers=2, activation="silu"))

    return archs


# ── model building ───────────────────────────────────────────────────────────

def _get_activation(name: str):
    import torch.nn as nn
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]


class _ResidualBlock:
    """Wrapper to add residual connections around hidden layers."""
    pass


def _make_mlp(in_dim: int, cfg: ArchConfig, out_dim: int):
    import torch
    import torch.nn as nn

    Act = _get_activation(cfg.activation)

    if cfg.residual and cfg.n_layers >= 2:
        return _ResidualMLP(in_dim, cfg.hidden, out_dim, cfg.n_layers,
                            cfg.dropout, cfg.batch_norm, Act)

    layers = []
    prev = in_dim
    for i in range(cfg.n_layers):
        layers.append(nn.Linear(prev, cfg.hidden))
        if cfg.batch_norm:
            layers.append(nn.BatchNorm1d(cfg.hidden))
        layers.append(Act())
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        prev = cfg.hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class _ResidualMLP:
    """MLP with residual connections between hidden layers."""
    pass


import torch
import torch.nn as nn


class _ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers, dropout, batch_norm, Act):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        blocks = []
        for _ in range(n_layers):
            block = [nn.Linear(hidden, hidden)]
            if batch_norm:
                block.append(nn.BatchNorm1d(hidden))
            block.append(Act())
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output(h)


# ── data loading (reuse from masked_neuron_prediction) ──────────────────────

def _load_worm(h5_path: Path) -> Optional[dict]:
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
    except Exception:
        return None

    if u.ndim != 2:
        return None
    if u.shape[0] < 400 and u.shape[1] >= 400:
        u = u.T
    labels = [lb.decode() if isinstance(lb, bytes) else str(lb) for lb in labels]
    if len(labels) != u.shape[1]:
        return None

    keep = np.any(np.isfinite(u), axis=0)
    u, labels = u[:, keep], [l for l, k in zip(labels, keep) if k]
    for j in range(u.shape[1]):
        nans = ~np.isfinite(u[:, j])
        if nans.any() and not nans.all():
            good = np.where(~nans)[0]
            u[:, j] = np.interp(np.arange(u.shape[0]), good, u[good, j])
    return {"u": u, "labels": labels, "name": h5_path.stem, "dt": dt}


# ── feature building (reuse) ────────────────────────────────────────────────

def _build_features(u, target_idx, n_lags, causal_mode="inclusive"):
    T, N = u.shape
    feature_cols = np.ones(N, dtype=bool)
    feature_cols[target_idx] = False

    if causal_mode == "inclusive":
        T_out = T - n_lags
        blocks = []
        for lag in range(n_lags):
            start = n_lags - lag
            blocks.append(u[start:start + T_out][:, feature_cols])
        X = np.concatenate(blocks, axis=1).astype(np.float32)
        y = u[n_lags:n_lags + T_out, target_idx].astype(np.float32)
    else:
        T_out = T - n_lags
        blocks = []
        for lag in range(1, n_lags + 1):
            start = n_lags - lag
            blocks.append(u[start:start + T_out][:, feature_cols])
        X = np.concatenate(blocks, axis=1).astype(np.float32)
        y = u[n_lags:n_lags + T_out, target_idx].astype(np.float32)
    return X, y


# ── CV folds ─────────────────────────────────────────────────────────────────

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


def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# ── training ─────────────────────────────────────────────────────────────────

def _train_mlp_model(net, Xtr, Ytr, Xva, Yva, *, epochs, lr, batch_size,
                     weight_decay, patience, device):
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    n = Xtr.shape[0]
    best_val, stale, best_state = float("inf"), 0, None

    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
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


# ── Ridge-linear baseline ───────────────────────────────────────────────────

def _run_ridge_linear(X, y, n_folds):
    from sklearn.linear_model import RidgeCV
    T = X.shape[0]
    folds = _make_folds(T, n_folds)
    pred = np.zeros(T, dtype=np.float32)
    for tr_idx, te_idx in folds:
        ridge = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
        ridge.fit(X[tr_idx], y[tr_idx])
        pred[te_idx] = ridge.predict(X[te_idx]).astype(np.float32)
    return pred, _r2(y, pred)


# ── Ridge-MLP (one architecture) ────────────────────────────────────────────

def _run_ridge_mlp(X, y, n_folds, arch: ArchConfig, args):
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

        mu_x = Xtr.mean(0, keepdims=True)
        std_x = Xtr.std(0, keepdims=True).clip(1e-8)
        mu_y = Ytr.mean()
        std_y = max(Ytr.std(), 1e-8)

        to_t = lambda a: torch.from_numpy(a).to(device)
        Xtr_z = to_t(((Xtr - mu_x) / std_x).astype(np.float32))
        Ytr_z = to_t(((Ytr.reshape(-1, 1) - mu_y) / std_y).astype(np.float32))
        Xva_z = to_t(((Xva - mu_x) / std_x).astype(np.float32))
        Yva_z = to_t(((Yva.reshape(-1, 1) - mu_y) / std_y).astype(np.float32))

        torch.manual_seed(args.seed + fi)
        net = _make_mlp(X.shape[1], arch, 1).to(device)
        _train_mlp_model(
            net, Xtr_z, Ytr_z, Xva_z, Yva_z,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            weight_decay=args.weight_decay, patience=args.patience,
            device=device,
        )

        # Extract backbone features
        net.eval()
        if isinstance(net, _ResidualMLP):
            # For residual MLP, use everything up to output layer
            class _Backbone(nn.Module):
                def __init__(self, parent):
                    super().__init__()
                    self.parent = parent
                def forward(self, x):
                    h = self.parent.input_proj(x)
                    for block in self.parent.blocks:
                        h = h + block(h)
                    return h
            backbone = _Backbone(net)
        else:
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


# ── plotting ─────────────────────────────────────────────────────────────────

def _plot_comparison(out_dir: Path, records: list[dict], neuron: str, worm: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = [r["arch"] for r in records]
    r2s = [r["r2"] for r in records]
    times = [r["time_s"] for r in records]
    is_linear = [r.get("is_linear", False) for r in records]

    # Sort by R²
    order = np.argsort(r2s)[::-1]

    # ── Bar chart: R² by architecture ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(archs) * 0.6), 6))
    colors = []
    for i in order:
        if is_linear[i]:
            colors.append("#3498db")   # blue for linear
        elif records[i].get("residual"):
            colors.append("#2ecc71")   # green for residual
        elif records[i].get("batch_norm"):
            colors.append("#f39c12")   # orange for BN
        else:
            colors.append("#e74c3c")   # red for standard MLP
    xs = np.arange(len(archs))
    bars = ax.bar(xs, [r2s[i] for i in order], color=colors,
                  edgecolor="k", lw=0.4, alpha=0.85)

    # Add R² labels on bars
    for xi, i in enumerate(order):
        val = r2s[i]
        ax.text(xi, val + 0.005, f"{val:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([archs[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R² (5-fold temporal CV)")
    ax.set_title(f"MLP Architecture Comparison — {neuron} in {worm}")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    best_linear_r2 = max((r2s[i] for i in range(len(archs)) if is_linear[i]), default=0)
    if best_linear_r2 > -5:
        ax.axhline(best_linear_r2, color="#3498db", lw=1, ls=":", alpha=0.7,
                   label=f"ridge-linear = {best_linear_r2:.3f}")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{neuron}_{worm}_r2_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── R² vs model size (n_params) ──────────────────────────────────────
    n_params = [r.get("n_params", 0) for r in records]
    if any(p > 0 for p in n_params):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(len(archs)):
            if is_linear[i]:
                continue
            c = "#2ecc71" if records[i].get("residual") else "#e74c3c"
            ax.scatter(n_params[i], r2s[i], s=60, c=c, edgecolors="k", lw=0.5,
                       zorder=3)
            ax.annotate(archs[i], (n_params[i], r2s[i]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")
        if best_linear_r2 > -5:
            ax.axhline(best_linear_r2, color="#3498db", lw=1, ls=":",
                       label=f"ridge-linear = {best_linear_r2:.3f}")
        ax.set_xlabel("Number of parameters")
        ax.set_ylabel("R²")
        ax.set_title(f"R² vs Model Size — {neuron} in {worm}")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{neuron}_{worm}_r2_vs_params.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # ── R² vs training time ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(len(archs)):
        c = "#3498db" if is_linear[i] else "#e74c3c"
        ax.scatter(times[i], r2s[i], s=60, c=c, edgecolors="k", lw=0.5, zorder=3)
        ax.annotate(archs[i], (times[i], r2s[i]),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("R²")
    ax.set_title(f"R² vs Time — {neuron} in {worm}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{neuron}_{worm}_r2_vs_time.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_traces(out_dir: Path, y_true: np.ndarray, preds: dict,
                 r2s: dict, neuron: str, worm: str, dt: float):
    """Overlay traces for top-5 architectures vs ground truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(r2s.items(), key=lambda x: x[1], reverse=True)
    top = ranked[:5]
    T = len(y_true)
    t_sec = np.arange(T) * dt

    cmap = plt.cm.Set1
    zoom = min(300, T)

    # Full time-series
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t_sec, y_true, lw=0.8, color="k", label="ground truth", alpha=0.8)
    for i, (name, r2) in enumerate(top):
        ax.plot(t_sec, preds[name], lw=0.6, color=cmap(i),
                label=f"{name}  R²={r2:.3f}", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity")
    ax.set_title(f"{neuron} in {worm} — Top 5 architectures")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{neuron}_{worm}_traces_full.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # Zoomed
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t_sec[:zoom], y_true[:zoom], lw=1.2, color="k",
            label="ground truth", alpha=0.9)
    for i, (name, r2) in enumerate(top):
        ax.plot(t_sec[:zoom], preds[name][:zoom], lw=0.8, color=cmap(i),
                label=f"{name}  R²={r2:.3f}", alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity")
    ax.set_title(f"{neuron} in {worm} — first {zoom} frames")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{neuron}_{worm}_traces_zoom.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(description="Compare MLP architectures for masked-neuron prediction")
    pa.add_argument("--dataset_dir", type=Path, default=_DEFAULT_DATASET_DIR)
    pa.add_argument("--out_dir", type=Path, default=_DEFAULT_OUT_DIR)
    pa.add_argument("--worm", type=str, nargs="+",
                    default=["2023-01-09-15"],
                    help="Worm name(s) (stem of H5 file)")
    pa.add_argument("--neuron", type=str, nargs="+", default=["AVAR"],
                    help="Neuron(s) to mask")
    pa.add_argument("--causal", type=str, default="inclusive",
                    choices=["inclusive", "strict"])
    pa.add_argument("--lag_sec", type=float, default=10.0)
    pa.add_argument("--n_folds", type=int, default=5)
    pa.add_argument("--seed", type=int, default=0)
    # MLP training
    pa.add_argument("--epochs", type=int, default=300)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--weight_decay", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=30)
    pa.add_argument("--device", type=str, default="cpu")
    # Architecture selection
    pa.add_argument("--archs", type=str, nargs="*", default=None,
                    help="Architecture names to run (substring match). "
                         "Default: all. E.g. --archs 128 res")
    pa.add_argument("--dry-run", action="store_true")
    args = pa.parse_args()

    np.random.seed(args.seed)

    # ── architecture list ────────────────────────────────────────────────
    all_archs = _default_architectures()
    if args.archs:
        all_archs = [a for a in all_archs
                     if any(sub in a.name for sub in args.archs)]

    print(f"Architectures to compare ({len(all_archs)} MLP + ridge-linear):")
    for a in all_archs:
        print(f"  {a.name:30s}  hidden={a.hidden} layers={a.n_layers} "
              f"drop={a.dropout} res={a.residual} bn={a.batch_norm} "
              f"act={a.activation}")
    print(f"  {'ridge-linear':30s}  (baseline)")
    print()

    if args.dry_run:
        return

    # ── load requested worms ─────────────────────────────────────────────
    h5_files = sorted(args.dataset_dir.glob("*.h5"))
    worms = {}
    for h5p in h5_files:
        if h5p.stem in args.worm:
            w = _load_worm(h5p)
            if w:
                worms[h5p.stem] = w
                print(f"Loaded {h5p.stem}: T={w['u'].shape[0]}, N={w['u'].shape[1]}")
    missing = set(args.worm) - set(worms.keys())
    if missing:
        print(f"[warn] Worms not found: {missing}")
    if not worms:
        print("No worms loaded!")
        return

    # ── run all combinations ─────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for worm_name in args.worm:
        if worm_name not in worms:
            continue
        w = worms[worm_name]

        for neuron_name in args.neuron:
            if neuron_name not in w["labels"]:
                print(f"  [skip] {neuron_name} not in {worm_name}")
                continue

            target_idx = w["labels"].index(neuron_name)
            n_lags = max(1, int(round(args.lag_sec / w["dt"])))
            X, y = _build_features(w["u"], target_idx, n_lags, args.causal)
            print(f"\n{'='*60}")
            print(f"  {neuron_name} in {worm_name}: X={X.shape}, "
                  f"n_lags={n_lags} ({n_lags*w['dt']:.1f}s)")
            print(f"{'='*60}")

            records = []
            pred_store = {}

            # Ridge-linear baseline
            t0 = time.time()
            pred_rl, r2_rl = _run_ridge_linear(X, y, args.n_folds)
            dt_rl = time.time() - t0
            records.append({
                "arch": "ridge-linear", "r2": r2_rl, "time_s": dt_rl,
                "is_linear": True, "n_params": 0,
            })
            pred_store["ridge-linear"] = pred_rl
            print(f"  ridge-linear       R²={r2_rl:.4f}  ({dt_rl:.1f}s)")

            # MLP architectures
            for arch in all_archs:
                t0 = time.time()
                pred, r2 = _run_ridge_mlp(X, y, args.n_folds, arch, args)
                dt_arch = time.time() - t0

                # Count params
                net_tmp = _make_mlp(X.shape[1], arch, 1)
                n_params = sum(p.numel() for p in net_tmp.parameters())

                records.append({
                    "arch": arch.name, "r2": r2, "time_s": dt_arch,
                    "is_linear": False, "n_params": n_params,
                    "hidden": arch.hidden, "n_layers": arch.n_layers,
                    "dropout": arch.dropout, "residual": arch.residual,
                    "batch_norm": arch.batch_norm, "activation": arch.activation,
                })
                pred_store[arch.name] = pred
                print(f"  {arch.name:22s}  R²={r2:.4f}  "
                      f"({n_params:,d} params, {dt_arch:.1f}s)")

            # ── plots ────────────────────────────────────────────────────
            _plot_comparison(args.out_dir, records, neuron_name, worm_name)
            r2_dict = {r["arch"]: r["r2"] for r in records}
            _plot_traces(args.out_dir, y, pred_store, r2_dict,
                         neuron_name, worm_name, w["dt"])

            all_results.append({
                "neuron": neuron_name, "worm": worm_name,
                "n_lags": n_lags, "T": len(y), "N": w["u"].shape[1],
                "records": records,
            })

    # ── save ─────────────────────────────────────────────────────────────
    summary_path = args.out_dir / "comparison_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            "causal_mode": args.causal,
            "lag_sec": args.lag_sec,
            "n_folds": args.n_folds,
            "results": all_results,
            "args": {k: str(v) for k, v in vars(args).items()},
        }, f, indent=2)
    print(f"\nResults saved to {summary_path}")
    print(f"Plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
