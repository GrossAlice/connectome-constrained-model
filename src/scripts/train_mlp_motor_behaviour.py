#!/usr/bin/env python3
"""
Predict eigenworm amplitudes (Stephens basis) from motor + control neuron
activity.  All models evaluated via **5-fold contiguous temporal CV** so
every frame is predicted exactly once (out-of-fold).

Models
------
  linear         – OLS, per-worm, per-amplitude  (no regularisation)
  ridge          – RidgeCV, per-worm, per-amplitude  (L2, log α grid + boundary check)
  mlp-per-worm   – per-worm 2-hidden-layer MLP
  mlp-residual   – per-worm MLP with skip connection  (ŷ = Linear(x) + MLP(x))
  mlp-pooled     – single pooled MLP  (140-dim zero-padded input)
  mlp-ridge      – per-worm MLP backbone + RidgeCV readout

Note: linear and ridge are *linear* models (no hidden layers).
Ridge is NOT an MLP.

RidgeCV's *internal* hyper-parameter search already uses log-spaced α
(np.logspace(-2, 6, 40)).  A boundary check warns if the chosen α lands
at the grid edge.

Evaluation
----------
5-fold contiguous temporal CV on each worm's recording.  For MLPs the
last 20 % of each fold's training portion is used as an inner validation
set for early-stopping only.

Usage
-----
    python scripts/train_mlp_motor_behaviour.py
    python scripts/train_mlp_motor_behaviour.py --n_folds 5 --patience 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# ── Default paths ────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
_DEFAULT_DATASET_DIR = _ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2"
_DEFAULT_MOTOR_LIST  = _ROOT / "data/used/masks+motor neurons/motor_neurons_with_control.txt"
_DEFAULT_OUT_DIR     = _ROOT / "output_plots/mlp_motor_behaviour"

_EW_LABELS = [f"EW{i}" for i in range(1, 7)]
_N_EW = 6
_RIDGE_ALPHAS = np.logspace(-2, 6, 40)

# ── Data loading ─────────────────────────────────────────────────────────────

def _load_motor_names(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _load_worm(
    h5_path: Path,
    motor_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (X_full, Y, obs_mask) or None."""
    required = ("stage1/u_mean", "behaviour/eigenworms_stephens", "gcamp/neuron_labels")
    try:
        with h5py.File(h5_path, "r") as f:
            if not all(k in f for k in required):
                return None
            u      = f["stage1/u_mean"][:]
            y      = f["behaviour/eigenworms_stephens"][:]
            labels = f["gcamp/neuron_labels"][:]
    except Exception as exc:
        print(f"  [warn] could not read {h5_path.name}: {exc}")
        return None

    labels = [lb.decode() if isinstance(lb, bytes) else lb for lb in labels]
    label_to_col = {name: i for i, name in enumerate(labels)}

    T, n_motor = u.shape[0], len(motor_names)
    X = np.zeros((T, n_motor), dtype=np.float32)
    obs_mask = np.zeros(n_motor, dtype=bool)
    for j, name in enumerate(motor_names):
        if name in label_to_col:
            X[:, j] = u[:, label_to_col[name]].astype(np.float32)
            obs_mask[j] = True

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    if valid.sum() < 20:
        return None
    return X[valid], y[valid].astype(np.float32), obs_mask


def _delay_embed(
    X: np.ndarray, Y: np.ndarray, obs_mask: np.ndarray, n_lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Delay-embed X with *n_lags* frames of history.

    Input  X : (T, N)
    Output X': (T - n_lags + 1,  N * n_lags)
           Y': trimmed to match
           obs_mask': tiled (N * n_lags,)

    Column layout per row t (0-indexed in the output):
        [u(t), u(t-1), …, u(t - n_lags + 1)]
    i.e. most-recent first.  For n_lags=8 this gives u(t) … u(t-7).
    """
    if n_lags <= 1:
        return X, Y, obs_mask
    T, N = X.shape
    T_out = T - n_lags + 1
    assert T_out > 0, f"n_lags={n_lags} >= T={T}"
    # build (T_out, N * n_lags)
    cols = [X[n_lags - 1 - lag : n_lags - 1 - lag + T_out, :]
            for lag in range(n_lags)]          # lag 0 = current
    X_emb = np.concatenate(cols, axis=1)       # (T_out, N*n_lags)
    Y_emb = Y[n_lags - 1:]                     # trim first n_lags-1 rows
    obs_emb = np.tile(obs_mask, n_lags)         # each lag copies mask
    return X_emb.astype(np.float32), Y_emb.astype(np.float32), obs_emb


# ── Fold generation ──────────────────────────────────────────────────────────

def _make_folds(T: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous temporal k-fold.  Returns list of (train_idx, test_idx)."""
    sizes = np.full(n_folds, T // n_folds, dtype=int)
    sizes[: T % n_folds] += 1
    folds, cur = [], 0
    for s in sizes:
        te = np.arange(cur, cur + s)
        tr = np.concatenate([np.arange(0, cur), np.arange(cur + s, T)])
        folds.append((tr, te))
        cur += s
    return folds


# ── Metrics ──────────────────────────────────────────────────────────────────

def _r2_vec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-column R², shape (D,)."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _r2_row(r2: np.ndarray, name: str, T: int, extra: dict | None = None):
    row = {"worm": name, "T": T,
           **{f"r2_ew{i+1}": float(r2[i]) for i in range(_N_EW)},
           "r2_mean": float(r2.mean())}
    if extra:
        row.update(extra)
    return row


def _check_ridge_boundary(alpha, alphas=_RIDGE_ALPHAS):
    """Return 'lower'/'upper'/None."""
    if alpha <= alphas[0]:
        return "lower"
    if alpha >= alphas[-1]:
        return "upper"
    return None


def _report_boundary(hits: dict, label: str):
    n_lo, n_hi = hits["lower"], hits["upper"]
    n_tot = hits["total"]
    if n_lo + n_hi > 0:
        print(f"\n  ⚠ {label}: RidgeCV α hit grid boundary "
              f"{n_lo + n_hi}/{n_tot} times  "
              f"(lower[≤{_RIDGE_ALPHAS[0]:.0e}]={n_lo}, "
              f"upper[≥{_RIDGE_ALPHAS[-1]:.0e}]={n_hi})")
    else:
        print(f"\n  ✓ {label}: RidgeCV α never hit grid boundary (0/{n_tot})")


# ── MLP helpers ──────────────────────────────────────────────────────────────

def _make_mlp(in_dim: int, hidden: int, out_dim: int, dropout: float):
    import torch.nn as nn
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class _ResidualMLP:
    """ŷ = Linear(x) + MLP(x).  The skip/linear path lets the network
    start from a good linear solution; the MLP only needs to learn the
    nonlinear residual."""

    @staticmethod
    def build(in_dim: int, hidden: int, out_dim: int, dropout: float):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.skip = nn.Linear(in_dim, out_dim)
                layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(hidden, out_dim))
                self.mlp = nn.Sequential(*layers)

            def forward(self, x):
                return self.skip(x) + self.mlp(x)

        return _Net()


def _train_mlp(
    net, X_tr, Y_tr, X_va, Y_va,
    *, epochs, lr, batch_size, weight_decay, patience, device, quiet=False,
):
    import torch, torch.nn as nn, torch.optim as optim
    opt  = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    n    = X_tr.shape[0]
    history = {"train_loss": [], "val_loss": []}
    best_val, stale, best_state = float("inf"), 0, None

    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, device=device)
        eloss, nseen = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = crit(net(X_tr[idx]), Y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item() * idx.shape[0]; nseen += idx.shape[0]
        eloss /= nseen

        net.eval()
        with torch.no_grad():
            vloss = crit(net(X_va), Y_va).item()
        history["train_loss"].append(eloss)
        history["val_loss"].append(vloss)

        if vloss < best_val:
            best_val, stale = vloss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1

        if (not quiet) and (ep % 50 == 0 or ep == 1):
            print(f"      ep {ep:4d}/{epochs}  tr={eloss:.4f}  va={vloss:.4f}"
                  f"  best={best_val:.4f}  stale={stale}")

        if patience and stale >= patience:
            if not quiet:
                print(f"      early stop at ep {ep}")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    history["stopped_epoch"] = ep
    return history


def _inner_split(tr_idx: np.ndarray, inner_val_frac: float = 0.2):
    """Split training indices into inner-train and inner-val (for MLP early-stopping)."""
    n_va = max(1, int(len(tr_idx) * inner_val_frac))
    return tr_idx[:-n_va], tr_idx[-n_va:]


def _zscore(Xtr):
    mu = Xtr.mean(0, keepdims=True)
    std = Xtr.std(0, keepdims=True).clip(1e-8)
    return mu, std


# ══════════════════════════════════════════════════════════════════════════════
#  Linear OLS  (per-worm, per-amplitude, 5-fold CV)
# ══════════════════════════════════════════════════════════════════════════════

def _run_linear(worms, n_folds):
    from sklearn.linear_model import LinearRegression

    r2_rows, worm_meta, Y_list, pred_list = [], [], [], []

    for w in worms:
        name, Xf, Y, obs = w["name"], w["X"], w["Y"], w["obs"]
        Xo = Xf[:, obs];  n_obs = int(obs.sum());  T = Xo.shape[0]
        folds = _make_folds(T, n_folds)
        pred_oof = np.zeros_like(Y)

        for tr_idx, te_idx in folds:
            Xtr, Ytr, Xte = Xo[tr_idx], Y[tr_idx], Xo[te_idx]
            for ew in range(_N_EW):
                lr = LinearRegression(fit_intercept=True)
                lr.fit(Xtr, Ytr[:, ew])
                pred_oof[te_idx, ew] = lr.predict(Xte)

        r2 = _r2_vec(Y, pred_oof)
        r2_rows.append(_r2_row(r2, name, T))
        worm_meta.append({"name": name, "T": T, "n_neurons_observed": n_obs})
        Y_list.append(Y);  pred_list.append(pred_oof.astype(np.float32))
        bar = " ".join(f"{v:+.2f}" for v in r2)
        print(f"  {name}  n={n_obs}  R²={r2.mean():.3f}  [{bar}]")

    return r2_rows, worm_meta, Y_list, pred_list


# ══════════════════════════════════════════════════════════════════════════════
#  Ridge-CV  (per-worm, per-amplitude, 5-fold CV, boundary check)
#  α grid: 40 values log-spaced from 10⁻² to 10⁶
# ══════════════════════════════════════════════════════════════════════════════

def _run_ridge(worms, n_folds):
    from sklearn.linear_model import RidgeCV

    r2_rows, worm_meta, Y_list, pred_list = [], [], [], []
    boundary = {"lower": 0, "upper": 0, "total": 0}

    for w in worms:
        name, Xf, Y, obs = w["name"], w["X"], w["Y"], w["obs"]
        Xo = Xf[:, obs];  n_obs = int(obs.sum());  T = Xo.shape[0]
        folds = _make_folds(T, n_folds)
        pred_oof = np.zeros_like(Y)
        all_alphas = []   # (n_folds, _N_EW)

        for tr_idx, te_idx in folds:
            Xtr, Ytr, Xte = Xo[tr_idx], Y[tr_idx], Xo[te_idx]
            fold_a = []
            for ew in range(_N_EW):
                ridge = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
                ridge.fit(Xtr, Ytr[:, ew])
                pred_oof[te_idx, ew] = ridge.predict(Xte)
                fold_a.append(float(ridge.alpha_))
                boundary["total"] += 1
                side = _check_ridge_boundary(ridge.alpha_)
                if side:
                    boundary[side] += 1
            all_alphas.append(fold_a)

        r2 = _r2_vec(Y, pred_oof)
        avg_a = np.mean(all_alphas, axis=0)
        a_str = " ".join(f"{a:.1f}" for a in avg_a)
        r2_rows.append(_r2_row(r2, name, T, extra={"alphas_avg": avg_a.tolist()}))
        worm_meta.append({"name": name, "T": T, "n_neurons_observed": n_obs})
        Y_list.append(Y);  pred_list.append(pred_oof.astype(np.float32))
        bar = " ".join(f"{v:+.2f}" for v in r2)
        print(f"  {name}  n={n_obs}  R²={r2.mean():.3f}  [{bar}]  ᾱ=[{a_str}]")

    _report_boundary(boundary, "ridge")
    return r2_rows, worm_meta, Y_list, pred_list


# ══════════════════════════════════════════════════════════════════════════════
#  MLP per-worm  (5-fold CV, inner val for early-stopping)
# ══════════════════════════════════════════════════════════════════════════════

def _run_mlp_per_worm(worms, args):
    import torch
    device = torch.device(args.device)
    r2_rows, worm_meta, Y_list, pred_list = [], [], [], []

    for wi, w in enumerate(worms):
        name, Xf, Y, obs = w["name"], w["X"], w["Y"], w["obs"]
        Xo = Xf[:, obs];  n_obs = int(obs.sum());  T = Xo.shape[0]
        folds = _make_folds(T, args.n_folds)
        pred_oof = np.zeros_like(Y)
        fold_epochs = []

        print(f"  {name}  n={n_obs}  T={T}", end="", flush=True)
        for fi, (tr_idx, te_idx) in enumerate(folds):
            tr_inner, va_inner = _inner_split(tr_idx)
            Xtr, Ytr = Xo[tr_inner], Y[tr_inner]
            Xva, Yva = Xo[va_inner], Y[va_inner]
            Xte       = Xo[te_idx]

            mu_x, std_x = _zscore(Xtr)
            mu_y, std_y = _zscore(Ytr)

            Xtr_z = torch.from_numpy((Xtr - mu_x) / std_x).to(device)
            Ytr_z = torch.from_numpy((Ytr - mu_y) / std_y).to(device)
            Xva_z = torch.from_numpy((Xva - mu_x) / std_x).to(device)
            Yva_z = torch.from_numpy((Yva - mu_y) / std_y).to(device)
            Xte_z = torch.from_numpy((Xte - mu_x) / std_x).to(device)

            torch.manual_seed(args.seed + wi * args.n_folds + fi)
            net = _make_mlp(n_obs, args.hidden, _N_EW, args.dropout).to(device)
            hist = _train_mlp(net, Xtr_z, Ytr_z, Xva_z, Yva_z,
                       epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                       weight_decay=args.weight_decay, patience=args.patience,
                       device=device, quiet=True)
            fold_epochs.append(hist["stopped_epoch"])

            net.eval()
            with torch.no_grad():
                pred_oof[te_idx] = net(Xte_z).cpu().numpy() * std_y + mu_y

        r2 = _r2_vec(Y, pred_oof)
        r2_rows.append(_r2_row(r2, name, T))
        worm_meta.append({"name": name, "T": T, "n_neurons_observed": n_obs})
        Y_list.append(Y);  pred_list.append(pred_oof.astype(np.float32))
        ep_str = ",".join(str(e) for e in fold_epochs[-args.n_folds:])
        bar = " ".join(f"{v:+.2f}" for v in r2)
        print(f"  R²={r2.mean():.3f}  [{bar}]  epochs=[{ep_str}]")

    return r2_rows, worm_meta, Y_list, pred_list


# ══════════════════════════════════════════════════════════════════════════════
#  MLP pooled  (5-fold CV, inner val for early-stopping)
# ══════════════════════════════════════════════════════════════════════════════

def _run_mlp_pooled(worms, args):
    import torch
    device = torch.device(args.device)
    n_input = worms[0]["X"].shape[1]
    n_folds = args.n_folds

    # Pre-compute per-worm folds
    worm_folds = [_make_folds(w["X"].shape[0], n_folds) for w in worms]
    pred_oofs  = [np.zeros_like(w["Y"]) for w in worms]

    for fi in range(n_folds):
        print(f"  Fold {fi+1}/{n_folds}")
        X_tr_l, Y_tr_l, X_va_l, Y_va_l = [], [], [], []
        X_te_per_worm, te_idx_per_worm = [], []

        for wi, w in enumerate(worms):
            tr_idx, te_idx = worm_folds[wi][fi]
            te_idx_per_worm.append(te_idx)
            X_te_per_worm.append(w["X"][te_idx])
            tr_inner, va_inner = _inner_split(tr_idx)
            X_tr_l.append(w["X"][tr_inner]);  Y_tr_l.append(w["Y"][tr_inner])
            X_va_l.append(w["X"][va_inner]);  Y_va_l.append(w["Y"][va_inner])

        X_tr = np.concatenate(X_tr_l);  Y_tr = np.concatenate(Y_tr_l)
        X_va = np.concatenate(X_va_l);  Y_va = np.concatenate(Y_va_l)

        mu_x, std_x = _zscore(X_tr)
        mu_y, std_y = _zscore(Y_tr)

        Xtr_z = torch.from_numpy((X_tr - mu_x) / std_x).to(device)
        Ytr_z = torch.from_numpy((Y_tr - mu_y) / std_y).to(device)
        Xva_z = torch.from_numpy((X_va - mu_x) / std_x).to(device)
        Yva_z = torch.from_numpy((Y_va - mu_y) / std_y).to(device)

        torch.manual_seed(args.seed + fi)
        net = _make_mlp(n_input, args.hidden, _N_EW, args.dropout).to(device)
        if fi == 0:
            n_params = sum(p.numel() for p in net.parameters())
            print(f"    Model: {n_input}→{args.hidden}→{args.hidden}→{_N_EW}  "
                  f"({n_params} params)")

        _train_mlp(net, Xtr_z, Ytr_z, Xva_z, Yva_z,
                   epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                   weight_decay=args.weight_decay, patience=args.patience,
                   device=device)

        net.eval()
        for wi in range(len(worms)):
            Xte_z = torch.from_numpy(
                (X_te_per_worm[wi] - mu_x) / std_x).to(device)
            with torch.no_grad():
                pred_oofs[wi][te_idx_per_worm[wi]] = \
                    net(Xte_z).cpu().numpy() * std_y + mu_y

    # Per-worm summary
    r2_rows, worm_meta, Y_list, pred_list = [], [], [], []
    for wi, w in enumerate(worms):
        r2 = _r2_vec(w["Y"], pred_oofs[wi])
        name, n_obs = w["name"], int(w["obs"].sum())
        r2_rows.append(_r2_row(r2, name, w["Y"].shape[0]))
        worm_meta.append({"name": name, "T": w["Y"].shape[0],
                          "n_neurons_observed": n_obs})
        Y_list.append(w["Y"])
        pred_list.append(pred_oofs[wi].astype(np.float32))
        bar = " ".join(f"{v:+.2f}" for v in r2)
        print(f"  {name}  R²={r2.mean():.3f}  [{bar}]")

    return r2_rows, worm_meta, Y_list, pred_list


# ══════════════════════════════════════════════════════════════════════════════
#  MLP with skip / residual connection  (ŷ = Linear(x) + MLP(x))
# ══════════════════════════════════════════════════════════════════════════════

def _run_mlp_residual(worms, args):
    import torch
    device = torch.device(args.device)
    r2_rows, worm_meta, Y_list, pred_list = [], [], [], []

    for wi, w in enumerate(worms):
        name, Xf, Y, obs = w["name"], w["X"], w["Y"], w["obs"]
        Xo = Xf[:, obs];  n_obs = int(obs.sum());  T = Xo.shape[0]
        folds = _make_folds(T, args.n_folds)
        pred_oof = np.zeros_like(Y)
        fold_epochs = []

        print(f"  {name}  n={n_obs}  T={T}", end="", flush=True)
        for fi, (tr_idx, te_idx) in enumerate(folds):
            tr_inner, va_inner = _inner_split(tr_idx)
            Xtr, Ytr = Xo[tr_inner], Y[tr_inner]
            Xva, Yva = Xo[va_inner], Y[va_inner]
            Xte       = Xo[te_idx]

            mu_x, std_x = _zscore(Xtr)
            mu_y, std_y = _zscore(Ytr)

            Xtr_z = torch.from_numpy((Xtr - mu_x) / std_x).to(device)
            Ytr_z = torch.from_numpy((Ytr - mu_y) / std_y).to(device)
            Xva_z = torch.from_numpy((Xva - mu_x) / std_x).to(device)
            Yva_z = torch.from_numpy((Yva - mu_y) / std_y).to(device)
            Xte_z = torch.from_numpy((Xte - mu_x) / std_x).to(device)

            torch.manual_seed(args.seed + wi * args.n_folds + fi)
            net = _ResidualMLP.build(n_obs, args.hidden, _N_EW, args.dropout).to(device)
            hist = _train_mlp(net, Xtr_z, Ytr_z, Xva_z, Yva_z,
                       epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                       weight_decay=args.weight_decay, patience=args.patience,
                       device=device, quiet=True)
            fold_epochs.append(hist["stopped_epoch"])

            net.eval()
            with torch.no_grad():
                pred_oof[te_idx] = net(Xte_z).cpu().numpy() * std_y + mu_y

        r2 = _r2_vec(Y, pred_oof)
        r2_rows.append(_r2_row(r2, name, T))
        worm_meta.append({"name": name, "T": T, "n_neurons_observed": n_obs})
        Y_list.append(Y);  pred_list.append(pred_oof.astype(np.float32))
        ep_str = ",".join(str(e) for e in fold_epochs[-args.n_folds:])
        bar = " ".join(f"{v:+.2f}" for v in r2)
        print(f"  R²={r2.mean():.3f}  [{bar}]  epochs=[{ep_str}]")

    return r2_rows, worm_meta, Y_list, pred_list


# ══════════════════════════════════════════════════════════════════════════════
#  MLP backbone + RidgeCV readout  (per-worm, 5-fold CV)
#  Train MLP end-to-end, freeze backbone, fit RidgeCV on hidden features
# ══════════════════════════════════════════════════════════════════════════════

def _run_mlp_ridge(worms, args):
    import torch, torch.nn as nn
    from sklearn.linear_model import RidgeCV

    device = torch.device(args.device)
    r2_rows, worm_meta, Y_list, pred_list = [], [], [], []
    boundary = {"lower": 0, "upper": 0, "total": 0}

    for wi, w in enumerate(worms):
        name, Xf, Y, obs = w["name"], w["X"], w["Y"], w["obs"]
        Xo = Xf[:, obs];  n_obs = int(obs.sum());  T = Xo.shape[0]
        folds = _make_folds(T, args.n_folds)
        pred_oof = np.zeros_like(Y)

        fold_epochs = []
        print(f"  {name}  n={n_obs}  T={T}", end="", flush=True)
        for fi, (tr_idx, te_idx) in enumerate(folds):
            tr_inner, va_inner = _inner_split(tr_idx)
            Xtr, Ytr = Xo[tr_inner], Y[tr_inner]
            Xva, Yva = Xo[va_inner], Y[va_inner]
            Xte       = Xo[te_idx]

            mu_x, std_x = _zscore(Xtr)
            mu_y, std_y = _zscore(Ytr)

            Xtr_z = torch.from_numpy((Xtr - mu_x) / std_x).to(device)
            Ytr_z = torch.from_numpy((Ytr - mu_y) / std_y).to(device)
            Xva_z = torch.from_numpy((Xva - mu_x) / std_x).to(device)
            Yva_z = torch.from_numpy((Yva - mu_y) / std_y).to(device)

            # 1) Train MLP backbone
            torch.manual_seed(args.seed + wi * args.n_folds + fi)
            net = _make_mlp(n_obs, args.hidden, _N_EW, args.dropout).to(device)
            hist = _train_mlp(net, Xtr_z, Ytr_z, Xva_z, Yva_z,
                       epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                       weight_decay=args.weight_decay, patience=args.patience,
                       device=device, quiet=True)
            fold_epochs.append(hist["stopped_epoch"])

            # 2) Extract hidden features (everything before last Linear)
            net.eval()
            backbone = nn.Sequential(*list(net.children())[:-1])

            # Use full fold training set for Ridge fitting
            X_ridge_tr = Xo[tr_idx];  Y_ridge_tr = Y[tr_idx]
            X_ridge_z = torch.from_numpy((X_ridge_tr - mu_x) / std_x).to(device)
            Xte_z     = torch.from_numpy((Xte - mu_x) / std_x).to(device)

            with torch.no_grad():
                feat_tr = backbone(X_ridge_z).cpu().numpy()
                feat_te = backbone(Xte_z).cpu().numpy()

            # 3) RidgeCV per amplitude on learned features
            for ew in range(_N_EW):
                ridge = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
                ridge.fit(feat_tr, Y_ridge_tr[:, ew])
                pred_oof[te_idx, ew] = ridge.predict(feat_te)
                boundary["total"] += 1
                side = _check_ridge_boundary(ridge.alpha_)
                if side:
                    boundary[side] += 1

        r2 = _r2_vec(Y, pred_oof)
        r2_rows.append(_r2_row(r2, name, T))
        worm_meta.append({"name": name, "T": T, "n_neurons_observed": n_obs})
        Y_list.append(Y);  pred_list.append(pred_oof.astype(np.float32))
        ep_str = ",".join(str(e) for e in fold_epochs)
        bar = " ".join(f"{v:+.2f}" for v in r2)
        print(f"  R²={r2.mean():.3f}  [{bar}]  epochs=[{ep_str}]")

    _report_boundary(boundary, "mlp-ridge")
    return r2_rows, worm_meta, Y_list, pred_list


# ══════════════════════════════════════════════════════════════════════════════
#  Per-model diagnostic plots
# ══════════════════════════════════════════════════════════════════════════════

def _generate_model_plots(
    *, out_dir, r2_rows, worm_meta, Y_list, pred_list, label, n_folds,
):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib: {exc}"); return

    cv_tag = f"{n_folds}-fold CV"
    worm_names = [r["worm"] for r in r2_rows]
    n_worms    = len(worm_names)
    r2_mat     = np.array([[r[f"r2_ew{i+1}"] for i in range(_N_EW)] for r in r2_rows])
    r2_means   = np.array([r["r2_mean"] for r in r2_rows])
    n_obs      = np.array([m["n_neurons_observed"] for m in worm_meta])
    Y_all = np.concatenate(Y_list); P_all = np.concatenate(pred_list)

    saved = []
    def _s(fig, name):
        fig.savefig(out_dir / name, dpi=140, bbox_inches="tight"); plt.close(fig)
        saved.append(name)

    # 1. Per-worm mean R² bar
    order = np.argsort(r2_means)[::-1]
    fig, ax = plt.subplots(figsize=(max(10, n_worms * 0.35), 5))
    ax.bar(range(n_worms), r2_means[order],
           color=["#2ecc71" if v > 0 else "#e74c3c" for v in r2_means[order]],
           edgecolor="k", lw=0.3)
    ax.set_xticks(range(n_worms))
    ax.set_xticklabels([worm_names[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel(f"Mean R² ({cv_tag})")
    ax.set_title(f"Per-worm mean R²  [{label}]")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axhline(np.median(r2_means), color="blue", lw=0.8, ls=":",
               label=f"median={np.median(r2_means):.3f}")
    ax.legend(); fig.tight_layout(); _s(fig, "r2_per_worm_bar.png")

    # 2. R² heatmap
    fig, ax = plt.subplots(figsize=(6, max(6, n_worms * 0.28)))
    im = ax.imshow(r2_mat[order], aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    ax.set_yticks(range(n_worms))
    ax.set_yticklabels([worm_names[i] for i in order], fontsize=7)
    ax.set_xticks(range(_N_EW)); ax.set_xticklabels(_EW_LABELS)
    ax.set_title(f"R² heatmap  [{label}]")
    fig.colorbar(im, ax=ax, shrink=0.7).set_label("R²")
    for yi, wi in enumerate(order):
        for xi in range(_N_EW):
            v = r2_mat[wi, xi]
            ax.text(xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > 0.35 else "black")
    fig.tight_layout(); _s(fig, "r2_heatmap.png")

    # 3. Per-eigenworm boxplot
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot([r2_mat[:, i] for i in range(_N_EW)], tick_labels=_EW_LABELS,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
    for p in bp["boxes"]: p.set_facecolor("#aec6cf")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel(f"R² ({cv_tag})")
    ax.set_title(f"R² per eigenworm  [{label}]")
    fig.tight_layout(); _s(fig, "r2_per_eigenworm_boxplot.png")

    # 4. Scatter pred vs actual
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for i, ax in enumerate(axes.flat):
        yt, yp = Y_all[:, i], P_all[:, i]
        ax.scatter(yt, yp, s=1, alpha=0.15, rasterized=True)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8)
        g = 1.0 - ((yt - yp)**2).sum() / ((yt - yt.mean())**2).sum()
        ax.set_title(f"{_EW_LABELS[i]}  R²={g:.3f}")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    fig.suptitle(f"Pred vs actual ({cv_tag})  [{label}]", y=1.01)
    fig.tight_layout(); _s(fig, "pred_vs_actual_scatter.png")

    # 5. R² vs coverage
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(n_obs, r2_means, s=40, edgecolors="k", lw=0.3, zorder=3)
    for i in range(n_worms):
        ax.annotate(worm_names[i], (n_obs[i], r2_means[i]),
                    fontsize=5, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    z = np.polyfit(n_obs, r2_means, 1)
    xs = np.linspace(n_obs.min(), n_obs.max(), 50)
    ax.plot(xs, np.polyval(z, xs), "r--", lw=1, label=f"slope={z[0]:.4f}")
    corr = np.corrcoef(n_obs, r2_means)[0, 1]
    ax.set_xlabel("# observed neurons")
    ax.set_ylabel(f"Mean R² ({cv_tag})")
    ax.set_title(f"R² vs coverage (r={corr:.3f})  [{label}]")
    ax.axhline(0, color="grey", lw=0.5, ls="--"); ax.legend()
    fig.tight_layout(); _s(fig, "r2_vs_neuron_coverage.png")

    # 6. Residual distributions
    res = Y_all - P_all
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, ax in enumerate(axes.flat):
        r = res[:, i]
        ax.hist(r, bins=80, density=True, alpha=0.7, edgecolor="k", lw=0.3)
        ax.axvline(0, color="r", ls="--", lw=0.8)
        ax.set_title(f"{_EW_LABELS[i]}  μ={r.mean():.3f}  σ={r.std():.3f}")
        ax.set_xlabel("Residual"); ax.set_ylabel("Density")
    fig.suptitle(f"Residual distributions ({cv_tag})  [{label}]", y=1.01)
    fig.tight_layout(); _s(fig, "residual_distributions.png")

    # 7. Time-series best / median / worst
    rank = np.argsort(r2_means)[::-1]
    picks = [rank[0], rank[len(rank)//2], rank[-1]]
    plbl  = ["best", "median", "worst"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    for ax, wi, lb in zip(axes, picks, plbl):
        yt = Y_list[wi][:, 0]; yp = pred_list[wi][:, 0]
        ax.plot(np.arange(len(yt)), yt, label="actual EW1", lw=0.9)
        ax.plot(np.arange(len(yt)), yp, label="predicted EW1", lw=0.9, alpha=0.8)
        ax.set_title(f"{worm_names[wi]} ({lb}, R²={r2_means[wi]:.3f})")
        ax.set_xlabel("Frame"); ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"EW1 time-series ({cv_tag})  [{label}]", y=1.01)
    fig.tight_layout(); _s(fig, "timeseries_best_median_worst.png")

    # 8. Global R² per eigenworm
    gr2 = np.array([1.0 - ((Y_all[:, i] - P_all[:, i])**2).sum()
                         / ((Y_all[:, i] - Y_all[:, i].mean())**2).sum()
                    for i in range(_N_EW)])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(_EW_LABELS, gr2,
           color=["#2ecc71" if v > 0 else "#e74c3c" for v in gr2],
           edgecolor="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    for i, v in enumerate(gr2):
        ax.text(i, v + 0.005*np.sign(v), f"{v:.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_ylabel(f"R² (pooled {cv_tag})")
    ax.set_title(f"Global R² per eigenworm  [{label}]")
    fig.tight_layout(); _s(fig, "r2_global_per_eigenworm.png")

    print(f"  {label}: {len(saved)} plots saved → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  Comparison plots across all models
# ══════════════════════════════════════════════════════════════════════════════

def _generate_comparison_plots(all_results: dict, out_dir: Path, n_folds: int):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return

    cv_tag = f"{n_folds}-fold CV"
    model_names  = list(all_results.keys())
    model_colors = {"linear": "#27ae60", "ridge": "#3498db",
                    "mlp-per-worm": "#e67e22", "mlp-pooled": "#9b59b6",
                    "mlp-ridge": "#e74c3c"}
    saved = []
    def _s(fig, name):
        fig.savefig(out_dir / name, dpi=140, bbox_inches="tight"); plt.close(fig)
        saved.append(name)

    worm_names = [r["worm"] for r in all_results[model_names[0]]["r2_rows"]]
    n_worms = len(worm_names)
    n_models = len(model_names)

    r2_means = {m: np.array([r["r2_mean"] for r in all_results[m]["r2_rows"]])
                for m in model_names}

    # Sort by ridge performance
    sort_key = "ridge" if "ridge" in r2_means else model_names[0]
    order = np.argsort(r2_means[sort_key])[::-1]

    # ── 1. Per-worm mean R² bar chart ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, n_worms * 0.6), 6))
    bar_w = 0.8 / n_models
    for mi, m in enumerate(model_names):
        xs = np.arange(n_worms) + mi * bar_w - 0.4 + bar_w / 2
        ax.bar(xs, r2_means[m][order], width=bar_w, label=m,
               color=model_colors.get(m, f"C{mi}"), edgecolor="k", lw=0.2,
               alpha=0.85)
    ax.set_xticks(range(n_worms))
    ax.set_xticklabels([worm_names[i] for i in order], rotation=90, fontsize=6)
    ax.set_ylabel(f"Mean R² ({cv_tag})")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title(f"Per-worm mean R² ({cv_tag})  –  all models"); ax.legend()
    fig.tight_layout(); _s(fig, "comparison_r2_per_worm.png")

    # ── 2. Per-eigenworm boxplot ─────────────────────────────────────────
    fig, axes = plt.subplots(1, _N_EW, figsize=(3.5 * _N_EW, 5), sharey=True)
    for ei, ax in enumerate(axes):
        data = [[r[f"r2_ew{ei+1}"] for r in all_results[m]["r2_rows"]]
                for m in model_names]
        short = [m.replace("mlp-", "mlp\n") for m in model_names]
        bp = ax.boxplot(data, tick_labels=short, patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red",
                                       markersize=4))
        for pi, p in enumerate(bp["boxes"]):
            p.set_facecolor(model_colors.get(model_names[pi], f"C{pi}"))
            p.set_alpha(0.6)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(_EW_LABELS[ei])
        if ei == 0:
            ax.set_ylabel(f"R² ({cv_tag})")
    fig.suptitle(f"R² per eigenworm ({cv_tag})  –  all models", y=1.02)
    fig.tight_layout(); _s(fig, "comparison_r2_per_eigenworm.png")

    # ── 3. Overall mean / median bar ─────────────────────────────────────
    overall = {m: float(np.mean(r2_means[m])) for m in model_names}
    median_ = {m: float(np.median(r2_means[m])) for m in model_names}
    fig, ax = plt.subplots(figsize=(max(7, n_models * 1.6), 4))
    x = np.arange(n_models)
    ax.bar(x - 0.15, [overall[m] for m in model_names], width=0.3,
           label="mean",
           color=[model_colors.get(m, "grey") for m in model_names],
           edgecolor="k", lw=0.4)
    ax.bar(x + 0.15, [median_[m] for m in model_names], width=0.3,
           label="median",
           color=[model_colors.get(m, "grey") for m in model_names],
           edgecolor="k", lw=0.4, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=9)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    for i, m in enumerate(model_names):
        ax.text(i - 0.15, overall[m] + 0.003, f"{overall[m]:.3f}",
                ha="center", fontsize=8)
        ax.text(i + 0.15, median_[m] + 0.003, f"{median_[m]:.3f}",
                ha="center", fontsize=8)
    ax.set_ylabel(f"R² ({cv_tag})")
    ax.set_title(f"Overall R² ({cv_tag})  –  all models")
    ax.legend(); fig.tight_layout(); _s(fig, "comparison_overall_r2.png")

    # ── 4. Global R² per eigenworm ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_w = 0.8 / n_models
    for mi, m in enumerate(model_names):
        Y_all = np.concatenate(all_results[m]["Y_list"])
        P_all = np.concatenate(all_results[m]["pred_list"])
        gr2 = [1.0 - ((Y_all[:, i] - P_all[:, i])**2).sum()
                    / ((Y_all[:, i] - Y_all[:, i].mean())**2).sum()
               for i in range(_N_EW)]
        xs = np.arange(_N_EW) + mi * bar_w - 0.4 + bar_w / 2
        ax.bar(xs, gr2, width=bar_w, label=m,
               color=model_colors.get(m, f"C{mi}"), edgecolor="k", lw=0.3,
               alpha=0.85)
    ax.set_xticks(range(_N_EW)); ax.set_xticklabels(_EW_LABELS)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel(f"R² (pooled {cv_tag})")
    ax.set_title(f"Global R² per eigenworm ({cv_tag})  –  all models")
    ax.legend(); fig.tight_layout(); _s(fig, "comparison_global_r2_per_ew.png")

    # ── 5. Side-by-side R² heatmap  (mlp-per-worm vs mlp-ridge) ──────────
    hm_models = [m for m in ("mlp-per-worm", "mlp-ridge") if m in all_results]
    if len(hm_models) == 2:
        r2_mats = {}
        for m in hm_models:
            r2_mats[m] = np.array(
                [[r[f"r2_ew{i+1}"] for i in range(_N_EW)]
                 for r in all_results[m]["r2_rows"]])

        fig, axes = plt.subplots(1, 2, figsize=(13, max(7, n_worms * 0.3)),
                                 sharey=True)
        for ai, m in enumerate(hm_models):
            ax = axes[ai]
            mat = r2_mats[m][order]
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                           vmin=-0.5, vmax=0.5)
            ax.set_yticks(range(n_worms))
            if ai == 0:
                ax.set_yticklabels([worm_names[i] for i in order], fontsize=7)
            ax.set_xticks(range(_N_EW))
            ax.set_xticklabels(_EW_LABELS)
            ax.set_title(f"{m}  (mean {r2_mats[m].mean():.3f})")
            for yi, wi in enumerate(order):
                for xi in range(_N_EW):
                    v = r2_mats[m][wi, xi]
                    ax.text(xi, yi, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5,
                            color="white" if abs(v) > 0.35 else "black")
        fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02).set_label("R²")
        fig.suptitle(f"R² heatmap ({cv_tag})  –  mlp-per-worm vs mlp-ridge",
                     y=1.01)
        fig.tight_layout(); _s(fig, "comparison_r2_heatmap_mlp.png")

    # ── 6. Per-worm per-mode bar chart (one subplot per worm) ────────────
    n_cols = min(n_worms, 4)
    n_rows = int(np.ceil(n_worms / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False, sharey=True,
    )
    bar_w = 0.8 / n_models
    for wi in range(n_worms):
        ax = axes[wi // n_cols][wi % n_cols]
        wname = worm_names[wi]
        for mi, m in enumerate(model_names):
            r = all_results[m]["r2_rows"][wi]
            vals = [r[f"r2_ew{i+1}"] for i in range(_N_EW)]
            xs = np.arange(_N_EW) + mi * bar_w - 0.4 + bar_w / 2
            ax.bar(xs, vals, width=bar_w, label=m if wi == 0 else "",
                   color=model_colors.get(m, f"C{mi}"), edgecolor="k",
                   lw=0.3, alpha=0.85)
        ax.set_xticks(range(_N_EW))
        ax.set_xticklabels(_EW_LABELS, fontsize=8)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(wname, fontsize=10)
        if wi % n_cols == 0:
            ax.set_ylabel(f"R² ({cv_tag})")
    # hide unused axes
    for wi in range(n_worms, n_rows * n_cols):
        axes[wi // n_cols][wi % n_cols].set_visible(False)
    fig.legend(model_names,
               loc="upper center", ncol=n_models, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Per-worm eigenworm R² ({cv_tag})", y=1.03, fontsize=13)
    fig.tight_layout(); _s(fig, "comparison_per_worm_per_mode.png")

    print(f"\n  Comparison: {len(saved)} plots → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    pa = argparse.ArgumentParser(
        description="Motor+control → eigenworm amplitudes  (5 models, k-fold CV)")
    pa.add_argument("--dataset_dir", type=Path, default=_DEFAULT_DATASET_DIR)
    pa.add_argument("--motor_list",  type=Path, default=_DEFAULT_MOTOR_LIST)
    pa.add_argument("--out_dir",     type=Path, default=_DEFAULT_OUT_DIR)
    pa.add_argument("--n_folds",     type=int,   default=5,
                    help="Number of folds for temporal CV (default 5)")
    pa.add_argument("--n_lags",      type=int,   default=1,
                    help="Number of time-lags (1 = no history; "
                         "8 = u(t)…u(t-7), i.e. 8 frames)")
    pa.add_argument("--seed",        type=int,   default=0)
    # MLP-specific
    pa.add_argument("--hidden",       type=int,   default=32)
    pa.add_argument("--epochs",       type=int,   default=300)
    pa.add_argument("--lr",           type=float, default=1e-3)
    pa.add_argument("--batch_size",   type=int,   default=64)
    pa.add_argument("--dropout",      type=float, default=0.0)
    pa.add_argument("--weight_decay", type=float, default=0.0)
    pa.add_argument("--patience",     type=int,   default=0,
                    help="Early-stopping patience for MLPs (0 = off)")
    pa.add_argument("--device",       type=str,   default="cpu")
    args = pa.parse_args()

    np.random.seed(args.seed)

    # ── load data ────────────────────────────────────────────────────────
    motor_names = _load_motor_names(args.motor_list)
    print(f"Motor + control neurons: {len(motor_names)}")

    h5_files = sorted(args.dataset_dir.glob("*.h5"))
    print(f"H5 files found: {len(h5_files)}")

    worms: list[dict] = []
    for h5p in h5_files:
        result = _load_worm(h5p, motor_names)
        if result is None:
            print(f"  [skip] {h5p.name}"); continue
        X, Y, obs = result
        X, Y, obs = _delay_embed(X, Y, obs, args.n_lags)
        worms.append({"name": h5p.stem, "X": X, "Y": Y, "obs": obs})
        print(f"  {h5p.name}: T={X.shape[0]}, "
              f"observed={int(obs.sum())}/{obs.shape[0]}")

    if not worms:
        sys.exit("No valid worms.")

    lag_str = (f", {args.n_lags}-frame history [u(t)…u(t-{args.n_lags-1})]"
               if args.n_lags > 1 else ", no time-lag (single frame)")
    print(f"\nInput dim per worm: {worms[0]['X'].shape[1]}  "
          f"({int(worms[0]['obs'].sum())} observed){lag_str}")
    print(f"Evaluation: {args.n_folds}-fold contiguous temporal CV "
          f"(inner 20% val for MLP early-stopping)")

    # ── run all five models ──────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    timings: dict[str, float] = {}

    # 1. Linear OLS
    print(f"\n{'='*60}\n  Model: linear  (OLS, per-worm, per-amplitude)\n{'='*60}\n")
    t0 = time.time()
    r2l, wml, yl, pl = _run_linear(worms, args.n_folds)
    timings["linear"] = time.time() - t0
    all_results["linear"] = {"r2_rows": r2l, "worm_meta": wml,
                             "Y_list": yl, "pred_list": pl}

    # 2. Ridge-CV
    print(f"\n{'='*60}\n  Model: ridge  (L2-CV, per-worm, per-amplitude, "
          f"α log-spaced 1e-2…1e6)\n{'='*60}\n")
    t0 = time.time()
    r2r, wmr, yr, pr = _run_ridge(worms, args.n_folds)
    timings["ridge"] = time.time() - t0
    all_results["ridge"] = {"r2_rows": r2r, "worm_meta": wmr,
                            "Y_list": yr, "pred_list": pr}

    # 3. MLP per-worm
    print(f"\n{'='*60}\n  Model: mlp-per-worm\n{'='*60}\n")
    import torch
    t0 = time.time()
    r2m, wmm, ym, pm = _run_mlp_per_worm(worms, args)
    timings["mlp-per-worm"] = time.time() - t0
    all_results["mlp-per-worm"] = {"r2_rows": r2m, "worm_meta": wmm,
                                   "Y_list": ym, "pred_list": pm}

    # 4. MLP with skip/residual connection
    print(f"\n{'='*60}\n  Model: mlp-residual  "
          f"(ŷ = Linear(x) + MLP(x))\n{'='*60}\n")
    t0 = time.time()
    r2res, wmres, yres, pres = _run_mlp_residual(worms, args)
    timings["mlp-residual"] = time.time() - t0
    all_results["mlp-residual"] = {"r2_rows": r2res, "worm_meta": wmres,
                                   "Y_list": yres, "pred_list": pres}

    # 5. MLP pooled  (disabled – zero-padding hurts; re-enable if needed)
    # print(f"\n{'='*60}\n  Model: mlp-pooled\n{'='*60}\n")
    # t0 = time.time()
    # r2p, wmp, yp, pp = _run_mlp_pooled(worms, args)
    # timings["mlp-pooled"] = time.time() - t0
    # all_results["mlp-pooled"] = {"r2_rows": r2p, "worm_meta": wmp,
    #                              "Y_list": yp, "pred_list": pp}

    # 6. MLP backbone + RidgeCV readout
    print(f"\n{'='*60}\n  Model: mlp-ridge  (MLP backbone + RidgeCV readout)\n"
          f"{'='*60}\n")
    t0 = time.time()
    r2mr, wmmr, ymr, pmr = _run_mlp_ridge(worms, args)
    timings["mlp-ridge"] = time.time() - t0
    all_results["mlp-ridge"] = {"r2_rows": r2mr, "worm_meta": wmmr,
                                "Y_list": ymr, "pred_list": pmr}

    # ── per-model summaries + plots ──────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Summary  ({args.n_folds}-fold CV)")
    print(f"{'='*60}")
    for m, res in all_results.items():
        mean_r2 = float(np.mean([r["r2_mean"] for r in res["r2_rows"]]))
        med_r2  = float(np.median([r["r2_mean"] for r in res["r2_rows"]]))
        dt = timings.get(m, 0)
        print(f"  {m:20s}  mean R²={mean_r2:+.3f}   "
              f"median R²={med_r2:+.3f}   ({dt:.1f}s)")

        sub = args.out_dir / m
        sub.mkdir(parents=True, exist_ok=True)

        summary = {
            "model": m,
            "evaluation": f"{args.n_folds}-fold temporal CV",
            "worms": res["worm_meta"],
            "r2_per_worm": res["r2_rows"],
            "r2_mean_all": mean_r2,
            "r2_median_all": med_r2,
            "time_s": dt,
            "args": {k: str(v) for k, v in vars(args).items()},
        }
        (sub / "summary.json").write_text(json.dumps(summary, indent=2))

        _generate_model_plots(
            out_dir=sub, r2_rows=res["r2_rows"], worm_meta=res["worm_meta"],
            Y_list=res["Y_list"], pred_list=res["pred_list"],
            label=m, n_folds=args.n_folds,
        )

    # ── comparison plots ─────────────────────────────────────────────────
    _generate_comparison_plots(all_results, args.out_dir, args.n_folds)

    total = sum(timings.values())
    print(f"\n  Total time: {total:.1f}s")


if __name__ == "__main__":
    main()
