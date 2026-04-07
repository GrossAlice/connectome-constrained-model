#!/usr/bin/env python
"""
Per-neuron R² distribution comparison.

Generates a two-panel figure (matching panels B & C of the Stage2 overview):
  B.  One-step R²  histogram  (all N neurons)
  C.  LOO R²       histogram  (top-30 neurons by variance, windowed w=50)

Models compared
───────────────
  Stage2        connectome-constrained ODE        (loaded from npz)
  Ridge         joint multi-output Ridge-CV       (all N inputs)
  EN            per-neuron ElasticNet(0.1, 0.1)   (all N inputs)
  MLP           joint MLP-256                     (all N inputs)
  Conn-Ridge    per-neuron Ridge-CV               (connectome neighbours + self)
  Conn-MLP      per-neuron MLP-64                 (connectome neighbours + self)

The "connectome-constrained" baselines restrict each neuron's input
features to its presynaptic partners (union of T_e + T_sv + T_dcv)
plus its own previous activity.

Usage
─────
    python -m scripts.model_distribution_comparison \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
        --stage2_npz output_plots/stage2/default_config_run_v2/cv_onestep.npz \
        --save_dir output_plots/stage2/model_distributions \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, ElasticNet, RidgeCV

ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m].astype(np.float64), y_pred[m].astype(np.float64)
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-15))


# ═══════════════════════════════════════════════════════════════════════
#  Connectome loading
# ═══════════════════════════════════════════════════════════════════════

def load_connectome(worm_labels: List[str]):
    """Load combined connectome (T_e + T_sv + T_dcv) mapped to worm neurons.

    Returns
    -------
    adj : (N, N) – adj[j, i] > 0 means j → i (j presynaptic to i)
    partners : dict[int, list[int]] – presynaptic indices per neuron
    """
    d = ROOT / "data/used/masks+motor neurons"
    atlas_names = np.load(d / "neuron_names.npy")
    n2a = {str(n): i for i, n in enumerate(atlas_names)}

    T_all = sum(
        np.abs(np.load(d / f"{t}.npy")) for t in ("T_e", "T_sv", "T_dcv")
    )

    N = len(worm_labels)
    wa = [n2a.get(lab, -1) for lab in worm_labels]

    adj = np.zeros((N, N), np.float64)
    for i in range(N):
        for j in range(N):
            if wa[i] >= 0 and wa[j] >= 0:
                adj[j, i] = T_all[wa[j], wa[i]]

    partners: Dict[int, List[int]] = {}
    for i in range(N):
        partners[i] = sorted(j for j in range(N) if j != i and adj[j, i] > 0)

    np_arr = [len(partners[i]) for i in range(N)]
    n_zero = sum(1 for v in np_arr if v == 0)
    print(f"  Connectome: partners per neuron — "
          f"min={min(np_arr)}, median={int(np.median(np_arr))}, "
          f"max={max(np_arr)}, isolated={n_zero}/{N}")
    return adj, partners


# ═══════════════════════════════════════════════════════════════════════
#  Model definitions
# ═══════════════════════════════════════════════════════════════════════

class JointMLP(nn.Module):
    """x(t) ∈ R^N → x(t+1) ∈ R^N  (all neurons jointly)."""

    def __init__(self, n: int, hidden: Tuple[int, ...] = (256,)):
        super().__init__()
        layers: list = []
        d = n
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyMLP(nn.Module):
    """Small per-neuron MLP: feat_dim → 1."""

    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
#  Training helpers
# ═══════════════════════════════════════════════════════════════════════

def _train_ridge(X: np.ndarray, Y: np.ndarray) -> Ridge:
    """Joint multi-output Ridge with 3-fold temporal CV for alpha."""
    alphas = np.logspace(-3, 6, 30)
    best_a, best_s = 1.0, -np.inf
    n = X.shape[0]
    fs = n // 3
    for a in alphas:
        sc = []
        for k in range(3):
            s = k * fs
            e = (k + 1) * fs if k < 2 else n
            Xtr = np.concatenate([X[:s], X[e:]])
            Ytr = np.concatenate([Y[:s], Y[e:]])
            sc.append(Ridge(alpha=a).fit(Xtr, Ytr).score(X[s:e], Y[s:e]))
        if np.mean(sc) > best_s:
            best_s, best_a = np.mean(sc), a
    return Ridge(alpha=best_a).fit(X, Y)


def _train_en(X: np.ndarray, Y: np.ndarray,
              alpha: float = 0.1, l1_ratio: float = 0.1):
    """Per-neuron ElasticNet → (W, b) weight matrix + bias."""
    N_out = Y.shape[1]
    W = np.zeros((N_out, X.shape[1]), np.float64)
    b = np.zeros(N_out, np.float64)
    for j in range(N_out):
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        en.fit(X, Y[:, j])
        W[j], b[j] = en.coef_, en.intercept_
    return W, b


def _train_mlp(X: np.ndarray, Y: np.ndarray, device: str,
               hidden: Tuple[int, ...] = (256,),
               max_ep: int = 500, patience: int = 30) -> nn.Module:
    """Joint MLP-256 with early stopping."""
    N = X.shape[1]
    nv = max(int(X.shape[0] * 0.15), 1)
    nf = X.shape[0] - nv
    Xf = torch.tensor(X[:nf], dtype=torch.float32, device=device)
    Xv = torch.tensor(X[nf:], dtype=torch.float32, device=device)
    Yf = torch.tensor(Y[:nf], dtype=torch.float32, device=device)
    Yv = torch.tensor(Y[nf:], dtype=torch.float32, device=device)

    m = JointMLP(N, hidden).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    bvl, bst, w = float("inf"), None, 0
    for ep in range(max_ep):
        m.train()
        perm = torch.randperm(nf, device=device)
        for bs in range(0, nf, 256):
            idx = perm[bs : bs + 256]
            loss = nn.functional.mse_loss(m(Xf[idx]), Yf[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(m(Xv), Yv).item()
        if vl < bvl - 1e-6:
            bvl = vl
            bst = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            w = 0
        else:
            w += 1
            if w >= patience:
                break
    if bst:
        m.load_state_dict({k: v.to(device) for k, v in bst.items()})
    m.eval()
    return m


def _train_conn_ridge(X: np.ndarray, Y: np.ndarray,
                      partners: Dict[int, List[int]], N: int):
    """Per-neuron Ridge-CV using only connectome neighbours + self."""
    models: Dict[int, Tuple] = {}
    for i in range(N):
        feats = sorted(partners[i] + [i])
        if not feats:
            feats = [i]
        m = RidgeCV(alphas=np.logspace(-3, 6, 20)).fit(X[:, feats], Y[:, i])
        models[i] = (m, feats)
    return models


def _train_conn_mlp(X: np.ndarray, Y: np.ndarray,
                    partners: Dict[int, List[int]], N: int,
                    device: str, hidden: int = 64,
                    max_ep: int = 200, patience: int = 15):
    """Per-neuron MLP using only connectome neighbours + self."""
    models: Dict[int, Tuple] = {}
    nv = max(int(X.shape[0] * 0.1), 1)
    nf = X.shape[0] - nv

    for i in range(N):
        feats = sorted(partners[i] + [i])
        if not feats:
            feats = [i]
        din = len(feats)

        Xf = torch.tensor(X[:nf, feats], dtype=torch.float32, device=device)
        Xv = torch.tensor(X[nf:, feats], dtype=torch.float32, device=device)
        Yf = torch.tensor(Y[:nf, i], dtype=torch.float32, device=device)
        Yv = torch.tensor(Y[nf:, i], dtype=torch.float32, device=device)

        m = TinyMLP(din, hidden).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        bvl, bst, w = float("inf"), None, 0
        for ep in range(max_ep):
            m.train()
            loss = nn.functional.mse_loss(m(Xf), Yf)
            opt.zero_grad()
            loss.backward()
            opt.step()
            m.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(m(Xv), Yv).item()
            if vl < bvl - 1e-6:
                bvl = vl
                bst = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                w = 0
            else:
                w += 1
                if w >= patience:
                    break
        if bst:
            m.load_state_dict({k: v.to(device) for k, v in bst.items()})
        m.eval()
        models[i] = (m, feats)

        if (i + 1) % 30 == 0 or i == N - 1:
            print(f"      Conn-MLP: {i + 1}/{N} neurons")

    return models


# ═══════════════════════════════════════════════════════════════════════
#  LOO evaluation (windowed, w=50)
# ═══════════════════════════════════════════════════════════════════════

def _loo_linear(W, b, u_te, ni, ws=50, lag=1):
    """Windowed LOO for joint linear model (Ridge or EN)."""
    T = u_te.shape[0]
    pred = np.full(T, np.nan)
    pred[0] = u_te[0, ni]
    for s in range(0, T, ws):
        e = min(s + ws, T)
        pred[s] = u_te[s, ni]
        for t in range(s, e - lag, lag):
            x = u_te[t].copy()
            x[ni] = pred[t]
            pred[t + lag] = x @ W[ni] + b[ni]
    return pred


def _loo_sklearn(model, u_te, ni, ws=50, lag=1):
    """Windowed LOO for sklearn Ridge (multi-output)."""
    return _loo_linear(model.coef_, model.intercept_, u_te, ni, ws, lag)


def _loo_mlp(model, u_te, ni, device, ws=50, lag=1):
    """Windowed LOO for joint MLP."""
    T = u_te.shape[0]
    pred = np.full(T, np.nan, dtype=np.float32)
    pred[0] = u_te[0, ni]
    model.eval()
    with torch.no_grad():
        for s in range(0, T, ws):
            e = min(s + ws, T)
            pred[s] = u_te[s, ni]
            for t in range(s, e - lag, lag):
                x = u_te[t].copy()
                x[ni] = pred[t]
                xt = torch.tensor(
                    x, dtype=torch.float32, device=device
                ).unsqueeze(0)
                pred[t + lag] = model(xt)[0, ni].cpu().item()
    return pred


def _loo_perneuron(model, feats, u_te, ni, is_sklearn, device, ws=50, lag=1):
    """Windowed LOO for per-neuron model (Conn-Ridge / Conn-MLP).

    `feats` includes neuron ni itself (self-feedback), so replacing
    ni's value with the model's own prediction automatically propagates.
    """
    T = u_te.shape[0]
    pred = np.full(T, np.nan)
    pred[0] = u_te[0, ni]

    # Position of ni in feats (for self-feedback)
    sp = feats.index(ni) if ni in feats else None

    if is_sklearn:
        for s in range(0, T, ws):
            e = min(s + ws, T)
            pred[s] = u_te[s, ni]
            for t in range(s, e - lag, lag):
                x = u_te[t, feats].copy()
                if sp is not None:
                    x[sp] = pred[t]
                pred[t + lag] = model.predict(x.reshape(1, -1))[0]
    else:
        model.eval()
        with torch.no_grad():
            for s in range(0, T, ws):
                e = min(s + ws, T)
                pred[s] = u_te[s, ni]
                for t in range(s, e - lag, lag):
                    x = u_te[t, feats].copy().astype(np.float32)
                    if sp is not None:
                        x[sp] = pred[t]
                    xt = torch.tensor(
                        x, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    pred[t + lag] = model(xt).cpu().item()
    return pred


# ═══════════════════════════════════════════════════════════════════════
#  Plot:  Panel B (one-step R²)  +  Panel C (LOO R²)
# ═══════════════════════════════════════════════════════════════════════

COLORS = {
    "Stage2":     "#d62728",
    "Ridge":      "#4682b4",
    "EN":         "#228b22",
    "MLP":        "#ff7f0e",
    "Conn-Ridge": "#1e90ff",
    "Conn-MLP":   "#9932cc",
}
ORDER = ["Stage2", "Ridge", "EN", "MLP", "Conn-Ridge", "Conn-MLP"]


def _strip_box(ax, ordered, results, key, ylabel):
    """Draw strip (jittered dots) + box for each model on one axis."""
    positions = np.arange(len(ordered))
    bp_data = []

    for xi, name in enumerate(ordered):
        vals = results[name][key]
        v = vals[np.isfinite(vals)]
        bp_data.append(v)
        if len(v) == 0:
            continue
        c = COLORS.get(name, "#888")
        # Jittered strip
        jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(v))
        ax.scatter(
            xi + jitter, v, s=18, alpha=0.55, color=c,
            edgecolors="white", linewidths=0.3, zorder=3,
        )

    # Box plot overlay
    bp = ax.boxplot(
        bp_data, positions=positions, widths=0.45,
        patch_artist=True, showfliers=False, zorder=4,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(color="gray", linewidth=1.0),
        capprops=dict(color="gray", linewidth=1.0),
    )
    for patch, name in zip(bp["boxes"], ordered):
        c = COLORS.get(name, "#888")
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)

    # Mean markers + annotations
    for xi, (name, v) in enumerate(zip(ordered, bp_data)):
        if len(v) == 0:
            continue
        mu = float(np.mean(v))
        ax.plot(xi, mu, marker="D", color=COLORS.get(name, "#888"),
                markersize=7, markeredgecolor="black", markeredgewidth=0.8,
                zorder=5)
        ax.annotate(
            f"{mu:.3f}", (xi, mu), textcoords="offset points",
            xytext=(0, 10), ha="center", fontsize=8, fontweight="bold",
            color=COLORS.get(name, "#888"),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(ordered, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def plot_bc(results: Dict, save_path: Path, worm_name: str, N: int,
            loo_neurons: List[int], lag: int = 1):
    """Two-panel strip+box plot: B (one-step R²) + C (LOO R²)."""

    ordered = [n for n in ORDER if n in results]

    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(15, 6))

    # ── Panel B:  One-step R² (all N neurons) ────────────────────────
    os_lbl = "One-step R²" if lag == 1 else f"Lag-{lag} R²"
    _strip_box(ax_b, ordered, results, "onestep_r2", os_lbl)
    ax_b.set_title(
        f"B.  {os_lbl}   (all {N} neurons)",
        fontsize=13, fontweight="bold",
    )
    ax_b.set_ylim(-0.3, 1.05)

    # ── Panel C:  LOO R² (30 neurons) ────────────────────────────────
    _strip_box(ax_c, ordered, results, "loo_r2_w", "LOO R² (windowed)")
    ax_c.set_title(
        f"C.  LOO R²   ({len(loo_neurons)} neurons, w=50)",
        fontsize=13, fontweight="bold",
    )
    ax_c.set_ylim(-0.8, 1.05)

    fig.suptitle(
        f"Per-neuron R² Distributions — {worm_name}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def run(args):
    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    worm = Path(args.h5).stem

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading data …")
    with h5py.File(args.h5, "r") as f:
        u = f["stage1/u_mean"][:]  # (T, N)
        labels = [
            l.decode() if isinstance(l, bytes) else str(l)
            for l in f["gcamp/neuron_labels"][:]
        ]
    T, N = u.shape
    print(f"  {worm}  T={T}  N={N}")

    # ── LOO neurons (top‑30 by variance) ──────────────────────────────
    var = np.var(u, axis=0)
    loo_neurons = sorted(np.argsort(var)[::-1][: args.loo_subset].tolist())
    print(f"  LOO neurons: {len(loo_neurons)}")

    # ── Connectome ────────────────────────────────────────────────────
    _, partners = load_connectome(labels)

    # ── 2‑fold temporal CV ────────────────────────────────────────────
    mid = T // 2 + 1
    folds = [(mid, T, 0, mid), (0, mid, mid, T)]
    ws = args.window_size

    MODEL_NAMES = ["Ridge", "EN", "MLP", "Conn-Ridge", "Conn-MLP"]
    os_pred = {m: np.full((T, N), np.nan) for m in MODEL_NAMES}
    loo_pred = {m: np.full((T, N), np.nan) for m in MODEL_NAMES}

    t_total = time.time()

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        print(f"\n{'─'*60}")
        print(f"  Fold {fi + 1}/2  train=[{tr_s},{tr_e})  test=[{te_s},{te_e})")
        print(f"{'─'*60}")
        u_tr = u[tr_s:tr_e]
        u_te = u[te_s:te_e]
        lag = args.lag
        X_tr, Y_tr = u_tr[:-lag], u_tr[lag:]
        X_te = u_te[:-lag]

        # ── Ridge (joint, multi-output) ───────────────────────────────
        t0 = time.time()
        ridge = _train_ridge(X_tr, Y_tr)
        os_pred["Ridge"][te_s + lag : te_e] = ridge.predict(X_te)
        for ni in loo_neurons:
            loo_pred["Ridge"][te_s:te_e, ni] = _loo_sklearn(
                ridge, u_te, ni, ws, lag
            )
        print(f"    Ridge        {time.time() - t0:6.1f}s")

        # ── EN(0.1, 0.1)  (per-neuron ElasticNet) ────────────────────
        t0 = time.time()
        W_en, b_en = _train_en(X_tr, Y_tr)
        os_pred["EN"][te_s + lag : te_e] = X_te @ W_en.T + b_en
        for ni in loo_neurons:
            loo_pred["EN"][te_s:te_e, ni] = _loo_linear(
                W_en, b_en, u_te, ni, ws, lag
            )
        print(f"    EN           {time.time() - t0:6.1f}s")

        # ── MLP-256 (joint) ───────────────────────────────────────────
        t0 = time.time()
        mlp = _train_mlp(X_tr, Y_tr, args.device, hidden=(256,))
        with torch.no_grad():
            Xt = torch.tensor(X_te, dtype=torch.float32, device=args.device)
            os_pred["MLP"][te_s + lag : te_e] = mlp(Xt).cpu().numpy()
        for ni in loo_neurons:
            loo_pred["MLP"][te_s:te_e, ni] = _loo_mlp(
                mlp, u_te, ni, args.device, ws, lag
            )
        del mlp
        torch.cuda.empty_cache()
        print(f"    MLP          {time.time() - t0:6.1f}s")

        # ── Conn-Ridge (per-neuron, connectome-restricted) ────────────
        t0 = time.time()
        cr_models = _train_conn_ridge(X_tr, Y_tr, partners, N)
        for i in range(N):
            m, feats = cr_models[i]
            os_pred["Conn-Ridge"][te_s + lag : te_e, i] = m.predict(
                X_te[:, feats]
            )
        for ni in loo_neurons:
            m, feats = cr_models[ni]
            loo_pred["Conn-Ridge"][te_s:te_e, ni] = _loo_perneuron(
                m, feats, u_te, ni, True, args.device, ws, lag
            )
        print(f"    Conn-Ridge   {time.time() - t0:6.1f}s")

        # ── Conn-MLP (per-neuron, connectome-restricted) ──────────────
        t0 = time.time()
        cm_models = _train_conn_mlp(
            X_tr, Y_tr, partners, N, args.device,
            hidden=args.conn_mlp_hidden,
        )
        for i in range(N):
            m, feats = cm_models[i]
            m.eval()
            with torch.no_grad():
                Xf = torch.tensor(
                    X_te[:, feats], dtype=torch.float32, device=args.device
                )
                os_pred["Conn-MLP"][te_s + lag : te_e, i] = (
                    m(Xf).cpu().numpy()
                )
        for ni in loo_neurons:
            m, feats = cm_models[ni]
            loo_pred["Conn-MLP"][te_s:te_e, ni] = _loo_perneuron(
                m, feats, u_te, ni, False, args.device, ws, lag
            )
        del cm_models
        torch.cuda.empty_cache()
        print(f"    Conn-MLP     {time.time() - t0:6.1f}s")

    elapsed = time.time() - t_total
    print(f"\n  Total training + LOO: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    # ══════════════════════════════════════════════════════════════════
    #  Compute per-neuron R²
    # ══════════════════════════════════════════════════════════════════
    results: Dict[str, Dict] = {}
    for mname in MODEL_NAMES:
        # One-step R² (all N neurons)
        os_r2 = np.full(N, np.nan)
        for i in range(N):
            m = np.isfinite(os_pred[mname][:, i])
            if m.sum() > 3:
                os_r2[i] = _r2(u[m, i], os_pred[mname][m, i])

        # LOO R² (LOO neurons only)
        loo_r2_full = np.full(N, np.nan)
        for i in loo_neurons:
            m = np.isfinite(loo_pred[mname][:, i])
            if m.sum() > 3:
                loo_r2_full[i] = _r2(u[m, i], loo_pred[mname][m, i])

        loo_vals = loo_r2_full[loo_neurons]
        results[mname] = {
            "onestep_r2": os_r2,
            "loo_r2_w": loo_vals,  # shape (n_loo,)
            "mean_onestep": float(np.nanmean(os_r2)),
            "median_onestep": float(np.nanmedian(os_r2)),
            "mean_loo_w": float(np.nanmean(loo_vals)),
            "median_loo_w": float(np.nanmedian(loo_vals)),
        }

    # ── Stage2 (loaded from npz) ─────────────────────────────────────
    if args.stage2_npz and Path(args.stage2_npz).exists():
        s2 = np.load(args.stage2_npz, allow_pickle=True)
        s2_os = s2["cv_r2"]                          # (N,)
        s2_loo_full = s2.get("cv_loo_r2_windowed",
                             np.full(N, np.nan))      # (N,) with NaN
        s2_loo_vals = s2_loo_full[loo_neurons]
        results["Stage2"] = {
            "onestep_r2": s2_os,
            "loo_r2_w": s2_loo_vals,
            "mean_onestep": float(np.nanmean(s2_os)),
            "median_onestep": float(np.nanmedian(s2_os)),
            "mean_loo_w": float(np.nanmean(s2_loo_vals)),
            "median_loo_w": float(np.nanmedian(s2_loo_vals)),
        }
        print("  Loaded Stage2 results from npz.")
    else:
        print("  ⚠  No stage2_npz provided — Stage2 omitted from plot.")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  {'Model':15s}  {'1step mean':>10s}  {'1step med':>10s}"
          f"  {'LOO_w mean':>10s}  {'LOO_w med':>10s}")
    print(f"  {'─'*60}")
    for mname in ORDER:
        if mname not in results:
            continue
        r = results[mname]
        print(f"  {mname:15s}  {r['mean_onestep']:10.4f}  "
              f"{r['median_onestep']:10.4f}  {r['mean_loo_w']:10.4f}  "
              f"{r['median_loo_w']:10.4f}")
    print(f"{'═'*65}")

    # ── Save JSON ─────────────────────────────────────────────────────
    json_out = {}
    for mname, r in results.items():
        json_out[mname] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in r.items()
        }
    json_out["_meta"] = {
        "worm": worm, "h5": args.h5, "N": N, "T": T,
        "loo_neurons": loo_neurons, "n_loo": len(loo_neurons),
        "window_size": ws, "time_s": elapsed,
    }
    with open(save / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  Saved {save / 'results.json'}")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_bc(results, save / "distribution_bc.png", worm, N, loo_neurons, lag=args.lag)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Per-neuron R² distribution: Stage2 vs baselines "
                    "vs connectome-constrained"
    )
    ap.add_argument("--h5", required=True,
                    help="Path to worm HDF5 file")
    ap.add_argument("--stage2_npz", default=None,
                    help="Path to Stage2 cv_onestep.npz for comparison")
    ap.add_argument("--save_dir", required=True,
                    help="Output directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loo_subset", type=int, default=30,
                    help="Number of LOO neurons (top by variance)")
    ap.add_argument("--window_size", type=int, default=50,
                    help="LOO re-seed window size")
    ap.add_argument("--lag", type=int, default=5,
                    help="Prediction lag in frames (predict u(t+lag) from u(t))")
    ap.add_argument("--conn_mlp_hidden", type=int, default=64,
                    help="Hidden size for per-neuron Conn-MLP")
    args = ap.parse_args()

    run(args)


if __name__ == "__main__":
    main()
