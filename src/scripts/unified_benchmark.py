#!/usr/bin/env python
r"""Unified decoder benchmark — all models, 5-fold temporal CV.

Combines all decoder architectures from the three separate benchmark
scripts into a single evaluation with consistent cross-validation,
per-mode Ridge α selection, and boundary diagnostics.

Models (with formula labels):
  1. b̂ⱼ = wⱼᵀn + cⱼ                       (Ridge, CV α per mode)
  2. b̂ = MLP(n)                             (2-layer, H=32)
  3. b̂ = M₁b̂₋₁ + M₂b̂₋₂ + Wn + c          (VAR₂+Ridge, 2-step, free-run)
  4. b̂ = M₁b̂₋₁ + M₂b̂₋₂ + MLP(n) + c      (VAR₂+MLP, 2-step, free-run)
  5. b̂ = M₁b̂₋₁ + M₂b̂₋₂ + MLP(n) + c      (E2E BPTT, free-run)
  6. b̂ = Wh+c, h = GRU(h, [n, b̂₋₁])       (GRU+beh, free-run)
  7. b̂ = Wh+c, h = GRU(h, n)               (GRU, free-run)

Regularisation
──────────────
  • Ridge λ chosen independently per output mode via inner CV
    on the training folds (n_folds−1 inner folds).
  • Input features are standardised per neuron×lag before fitting
    → each neuron receives equal effective penalty weight.
  • Boundary hits (λ=0 or λ=grid-max) are flagged per mode.

Usage:
    python -m scripts.unified_benchmark \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2023-01-10-07.h5"
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.behavior_decoder_eval import (
    _log_ridge_grid,
    _fit_ridge_regression,
    _ridge_cv_single_target,
    build_lagged_features_np,
)
try:
    from scripts.benchmark_ar_decoder_v2 import (
        load_data, build_ar_features, r2_score, E2EOscillatorMLP,
        E2EUndampedMLP, make_comparison_video,
    )
except ModuleNotFoundError:
    from benchmark_ar_decoder_v2 import (
        load_data, build_ar_features, r2_score, E2EOscillatorMLP,
        E2EUndampedMLP, make_comparison_video,
    )

try:
    from scripts.benchmark_gru_decoder import GRUDecoder, train_gru, free_run_eval
except ModuleNotFoundError:
    try:
        from benchmark_gru_decoder import GRUDecoder, train_gru, free_run_eval
    except ModuleNotFoundError:
        GRUDecoder = train_gru = free_run_eval = None


# ================================================================== #
#  Ridge / OLS helpers
# ================================================================== #

def _ridge_fit(X_train, y_train, ridge_grid, n_inner):
    """Ridge CV on training data.  Returns (coef, intercept, info)."""
    idx = np.arange(X_train.shape[0])
    fit = _ridge_cv_single_target(X_train, y_train, idx, ridge_grid, n_inner)
    info = {"alpha": fit["best_lambda"],
            "at_zero": fit["at_zero"],
            "at_upper": fit["at_upper_boundary"]}
    return fit["coef"], fit["intercept"], info


def _ols_fit(X_train, y_train):
    """OLS (α=0).  Returns (coef, intercept)."""
    res = _fit_ridge_regression(X_train, y_train, 0.0)
    if res is None:
        return np.full(X_train.shape[1], np.nan), float("nan")
    return res[1], res[0]


# ================================================================== #
#  MLP helper
# ================================================================== #

def _train_mlp(X_train, y_train, K, hidden=128, n_layers=2, lr=1e-3,
               epochs=500, weight_decay=1e-3, patience=40, cv_folds=5):
    """Train MLP decoder (default: 2×128).  Returns trained model (eval mode).

    Architecture winner from mlp_arch_sweep.py: 2×128+wd1e-3.

    Epoch selection via inner *cv_folds*-fold CV (Ridge-style):
      Phase 1 — train *cv_folds* models, each holding out one fold as
                validation.  Record per-epoch val loss.  Each inner run
                uses patience-based early stopping for efficiency.
      Phase 2 — average val-loss curves across folds, pick the epoch
                with lowest mean val loss → *best_epoch*.
      Phase 3 — retrain a fresh MLP on ALL training data for exactly
                *best_epoch* epochs (no data held out).
    """
    n = X_train.shape[0]
    d = X_train.shape[1]
    X_all = torch.tensor(X_train, dtype=torch.float32)
    y_all = torch.tensor(y_train, dtype=torch.float32)

    def _make_mlp():
        layers = []
        in_dim = d
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                       nn.ReLU(), nn.Dropout(0.1)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, K))
        return nn.Sequential(*layers)

    # ── Phase 1: inner CV to find optimal epoch ──────────────────────
    fold_size = n // cv_folds
    val_curves = []                         # list of (max_ep,) arrays

    for fi in range(cv_folds):
        s = fi * fold_size
        e = (s + fold_size) if fi < cv_folds - 1 else n
        mask = np.ones(n, dtype=bool)
        mask[s:e] = False
        X_t, y_t = X_all[mask], y_all[mask]
        X_v, y_v = X_all[s:e], y_all[s:e]

        mlp_i = _make_mlp()
        opt_i = torch.optim.Adam(mlp_i.parameters(), lr=lr,
                                 weight_decay=weight_decay)
        fold_vl = []
        best_vl, pat = float("inf"), 0
        for ep in range(epochs):
            mlp_i.train()
            loss = nn.functional.mse_loss(mlp_i(X_t), y_t)
            opt_i.zero_grad(); loss.backward(); opt_i.step()
            mlp_i.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(mlp_i(X_v), y_v).item()
            fold_vl.append(vl)
            if vl < best_vl - 1e-6:
                best_vl, pat = vl, 0
            else:
                pat += 1
                if pat > patience:
                    break
        # Pad to full length with last value (plateau)
        last = fold_vl[-1]
        fold_vl.extend([last] * (epochs - len(fold_vl)))
        val_curves.append(fold_vl)

    # ── Phase 2: pick best epoch from averaged curve ─────────────────
    mean_curve = np.mean(val_curves, axis=0)
    best_epoch = int(np.argmin(mean_curve)) + 1     # 1-indexed
    best_epoch = max(best_epoch, 10)                 # floor at 10

    # ── Phase 3: retrain on ALL training data for best_epoch ─────────
    mlp = _make_mlp()
    opt = torch.optim.Adam(mlp.parameters(), lr=lr,
                           weight_decay=weight_decay)
    for ep in range(best_epoch):
        mlp.train()
        loss = nn.functional.mse_loss(mlp(X_all), y_all)
        opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    return mlp


def _predict_mlp(mlp, X):
    with torch.no_grad():
        return mlp(torch.tensor(X, dtype=torch.float32)).numpy()


# ================================================================== #
#  E2E training helper
# ================================================================== #

def _clamp_diag_spectral_radius(d1, d2, max_rho=0.98):
    """Project diagonal AR(2) coefficients so spectral radius ≤ max_rho.

    For each mode j the companion matrix is [[d1_j, d2_j],[1, 0]].
    Eigenvalues: λ = (d1 ± sqrt(d1² + 4·d2)) / 2.
    We compute |λ|_max per mode and rescale (d1_j, d2_j) jointly if needed.
    """
    with torch.no_grad():
        d1v = d1.data
        d2v = d2.data
        disc = d1v ** 2 + 4.0 * d2v
        # Real eigenvalues
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        lam_r1 = (d1v + sqrt_disc) / 2.0
        lam_r2 = (d1v - sqrt_disc) / 2.0
        rho_real = torch.max(lam_r1.abs(), lam_r2.abs())
        # Complex eigenvalues: |λ| = sqrt(-d2)  when disc < 0
        rho_complex = torch.sqrt(torch.clamp(-d2v, min=0.0))
        rho = torch.where(disc >= 0, rho_real, rho_complex)
        # Rescale modes that exceed max_rho
        scale = torch.where(rho > max_rho,
                            max_rho / (rho + 1e-8),
                            torch.ones_like(rho))
        d1.data.mul_(scale)
        d2.data.mul_(scale ** 2)   # d2 scales quadratically (λ² ~ d2)


def _clamp_full_spectral_radius(M1_mod, M2_mod, K, max_rho=0.98):
    """Project full AR(2) matrices so companion spectral radius ≤ max_rho.

    Companion matrix: [[M1, M2], [I, 0]].  If ρ > max_rho we jointly
    rescale M1 by s and M2 by s² so that eigenvalues scale by s.
    """
    with torch.no_grad():
        C = torch.zeros(2 * K, 2 * K)
        C[:K, :K] = M1_mod.weight.data
        C[:K, K:] = M2_mod.weight.data
        C[K:, :K] = torch.eye(K)
        eigs = torch.linalg.eigvals(C)
        rho = eigs.abs().max().item()
        if rho > max_rho:
            s = max_rho / (rho + 1e-8)
            M1_mod.weight.data.mul_(s)
            M2_mod.weight.data.mul_(s ** 2)


def _train_e2e(d_in, K, segs, b, X_neural, epochs, chunk,
               weight_decay=0.0, linear_drive=False,
               diagonal_ar=False, max_rho=0.98,
               w_phase=0.0, patience=40, tag="E2E"):
    """Train E2E VAR(2)+MLP via BPTT.  Returns (M1, M2, c, drive) as numpy.

    If *linear_drive* is True the MLP is replaced by a single
    ``nn.Linear(d_in, K)`` layer — effectively a ridge-penalised
    linear drive when combined with *weight_decay* > 0.

    If *diagonal_ar* is True the K×K AR matrices are replaced by
    diagonal vectors (K params each) and a spectral-radius clamp
    (≤ *max_rho*) is applied after every optimiser step.

    If *w_phase* > 0 an additional cosine-similarity loss on the
    (a₁, a₂) pair is added:  ``w_phase * (1 − cos_sim(pred[:2], gt[:2]))``.
    This encourages correct phase-plane angle tracking.
    """
    b_t = torch.tensor(b, dtype=torch.float32)
    X_t = torch.tensor(X_neural, dtype=torch.float32)

    e2e = E2EOscillatorMLP(d_in, K, hidden=128, n_layers=2)
    if linear_drive:
        e2e.mlp = nn.Linear(d_in, K).to('cpu')

    # --- diagonal AR override ------------------------------------------------
    if diagonal_ar:
        e2e._d1 = nn.Parameter(torch.full((K,), 0.8))   # diag of M1
        e2e._d2 = nn.Parameter(torch.zeros(K))           # diag of M2
        # Remove the old nn.Linear M1/M2 from param list;
        # we'll manually include _d1, _d2 instead.
        e2e._params = ([e2e._d1, e2e._d2, e2e.c] +
                       list(e2e.mlp.parameters()))
        # Replace M1/M2 with callable objects that support .train()/.eval()
        class _DiagMul(nn.Module):
            def __init__(self, diag):
                super().__init__()
                self.diag = diag
            def forward(self, x):
                return self.diag * x
        e2e.M1 = _DiagMul(e2e._d1)
        e2e.M2 = _DiagMul(e2e._d2)
    else:
        e2e._params = (list(e2e.M1.parameters()) +
                       list(e2e.M2.parameters()) +
                       [e2e.c] + list(e2e.mlp.parameters()))
        with torch.no_grad():
            e2e.M1.weight.copy_(0.8 * torch.eye(K))
            e2e.M2.weight.zero_()

    opt = torch.optim.Adam(e2e.parameters(), lr=1e-3,
                           weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_l, best_p, pat = float("inf"), None, 0

    for ep in range(epochs):
        e2e.train()
        ep_loss, ep_n = 0.0, 0
        for s0, s1 in segs:
            drv = e2e.mlp(X_t)
            p1e = b_t[s0 - 1].detach()
            p2e = b_t[s0 - 2].detach()
            for cs in range(s0, s1, chunk):
                ce = min(cs + chunk, s1)
                cl = torch.tensor(0.0)
                for tt in range(cs, ce):
                    pt = e2e.M1(p1e) + e2e.M2(p2e) + drv[tt] + e2e.c
                    cl = cl + nn.functional.mse_loss(pt, b_t[tt])
                    if w_phase > 0 and K >= 2:
                        cos = nn.functional.cosine_similarity(
                            pt[:2].unsqueeze(0),
                            b_t[tt, :2].unsqueeze(0))
                        cl = cl + w_phase * (1.0 - cos.squeeze())
                    p2e, p1e = p1e, pt
                cl = cl / (ce - cs)
                opt.zero_grad(); cl.backward()
                nn.utils.clip_grad_norm_(e2e.parameters(), 1.0)
                opt.step()
                # Spectral-radius clamp
                if diagonal_ar:
                    _clamp_diag_spectral_radius(e2e._d1, e2e._d2,
                                                max_rho)
                else:
                    _clamp_full_spectral_radius(e2e.M1, e2e.M2,
                                                K, max_rho)
                ep_loss += cl.item() * (ce - cs)
                ep_n += ce - cs
                p1e = p1e.detach(); p2e = p2e.detach()
                drv = e2e.mlp(X_t)
        sched.step()
        if ep_n > 0:
            avg = ep_loss / ep_n
            if avg < best_l - 1e-6:
                best_l, pat = avg, 0
                if diagonal_ar:
                    best_p = {
                        'mlp': {k: v.clone()
                                for k, v in e2e.mlp.state_dict().items()},
                        'd1': e2e._d1.detach().clone(),
                        'd2': e2e._d2.detach().clone(),
                        'c': e2e.c.detach().clone()}
                else:
                    best_p = {
                        'mlp': {k: v.clone()
                                for k, v in e2e.mlp.state_dict().items()},
                        'M1': e2e.M1.weight.detach().clone(),
                        'M2': e2e.M2.weight.detach().clone(),
                        'c': e2e.c.detach().clone()}
            else:
                pat += 1
                if pat > patience:
                    break
        if (ep + 1) % 100 == 0 and ep_n > 0:
            if diagonal_ar:
                d1s = e2e._d1.detach().cpu().numpy()
                d2s = e2e._d2.detach().cpu().numpy()
                print(f"        {tag} ep {ep+1}: loss={avg:.5f}  "
                      f"best={best_l:.5f}  "
                      f"d1={np.array2string(d1s, precision=3, suppress_small=True)}  "
                      f"d2={np.array2string(d2s, precision=3, suppress_small=True)}")
            else:
                print(f"        {tag} ep {ep+1}: loss={avg:.5f}  "
                      f"best={best_l:.5f}")

    # Restore best
    if best_p:
        e2e.mlp.load_state_dict(best_p['mlp'])
        with torch.no_grad():
            if diagonal_ar:
                e2e._d1.copy_(best_p['d1'])
                e2e._d2.copy_(best_p['d2'])
            else:
                e2e.M1.weight.copy_(best_p['M1'])
                e2e.M2.weight.copy_(best_p['M2'])
            e2e.c.copy_(best_p['c'])
    e2e.eval()

    with torch.no_grad():
        drv_np = e2e.mlp(X_t).numpy()
        if diagonal_ar:
            M1e = np.diag(e2e._d1.numpy())
            M2e = np.diag(e2e._d2.numpy())
        else:
            M1e = e2e.M1.weight.numpy()
            M2e = e2e.M2.weight.numpy()
        ce_np = e2e.c.numpy()
    return M1e, M2e, ce_np, drv_np


# ================================================================== #
#  E2E spectral oscillator training helper
# ================================================================== #

def _train_e2e_spectral(d_in, K, segs, b, X_neural, epochs, chunk,
                        weight_decay=1e-3, alpha_init=0.99,
                        w_phase=0.0, tag="E2E+Spec"):
    """Train E2EUndampedMLP (spectral oscillator) via BPTT.

    Uses per-block (α_j, θ_j) parameterisation with guaranteed
    |λ| ∈ [0.90, 1.0].  Returns (M1, M2, c, drive) as numpy arrays.
    """
    b_t = torch.tensor(b, dtype=torch.float32)
    X_t = torch.tensor(X_neural, dtype=torch.float32)

    model = E2EUndampedMLP(d_in, K, hidden=32, alpha_init=alpha_init)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                           weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_l, best_p, pat = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for s0, s1 in segs:
            drv = model.mlp(X_t)
            M1, M2 = model._get_M1_M2()
            p1e = b_t[s0 - 1].detach()
            p2e = b_t[s0 - 2].detach()
            for cs in range(s0, s1, chunk):
                ce = min(cs + chunk, s1)
                cl = torch.tensor(0.0)
                for tt in range(cs, ce):
                    pt = (M1 @ p1e + M2 @ p2e + drv[tt]
                          + model.c[:model.K])
                    mse = nn.functional.mse_loss(pt, b_t[tt])
                    if w_phase > 0 and K >= 2:
                        cos = nn.functional.cosine_similarity(
                            pt[:2].unsqueeze(0),
                            b_t[tt, :2].unsqueeze(0))
                        mse = mse + w_phase * (1.0 - cos.squeeze())
                    cl = cl + mse
                    p2e, p1e = p1e, pt
                cl = cl / (ce - cs)
                opt.zero_grad(); cl.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += cl.item() * (ce - cs)
                ep_n += ce - cs
                p1e = p1e.detach(); p2e = p2e.detach()
                drv = model.mlp(X_t)
                M1, M2 = model._get_M1_M2()
        sched.step()
        if ep_n > 0:
            avg = ep_loss / ep_n
            if avg < best_l - 1e-6:
                best_l, pat = avg, 0
                best_p = {
                    'mlp': {k: v.clone()
                            for k, v in model.mlp.state_dict().items()},
                    'alpha_raw': model._alpha_raw.detach().clone(),
                    'theta': model._theta.detach().clone(),
                    'basis': model._basis_params.detach().clone(),
                    'c': model.c.detach().clone()}
            else:
                pat += 1
                if pat > 40:  # spectral oscillator uses fixed patience
                    break
        if (ep + 1) % 100 == 0 and ep_n > 0:
            alpha = model._get_alpha().detach().cpu().numpy()
            theta = model._theta.detach().cpu().numpy()
            print(f"        {tag} ep {ep+1}: loss={avg:.5f}  "
                  f"best={best_l:.5f}  "
                  f"α={np.array2string(alpha, precision=4)}  "
                  f"θ={np.array2string(theta, precision=3)}")

    # Restore best
    if best_p:
        model.mlp.load_state_dict(best_p['mlp'])
        with torch.no_grad():
            model._alpha_raw.copy_(best_p['alpha_raw'])
            model._theta.copy_(best_p['theta'])
            model._basis_params.copy_(best_p['basis'])
            model.c.copy_(best_p['c'])
    model.eval()

    with torch.no_grad():
        drv_np = model.mlp(X_t).numpy()
        M1_np, M2_np = model._get_M1_M2()
        M1_np = M1_np.numpy()
        M2_np = M2_np.numpy()
        ce_np = model.c.numpy()
    return M1_np, M2_np, ce_np, drv_np


# ================================================================== #
#  Main
# ================================================================== #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--neural_lags", type=int, default=8)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_modes", type=int, default=6)
    ap.add_argument("--e2e_epochs", type=int, default=300)
    ap.add_argument("--gru_epochs", type=int, default=300)
    ap.add_argument("--gru_hidden", type=int, default=64)
    ap.add_argument("--tbptt_chunk", type=int, default=64)
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Truncate to first N frames (0=all)")
    ap.add_argument("--all_neurons", action="store_true")
    ap.add_argument("--no_gru", action="store_true",
                    help="Skip GRU models (much faster)")
    ap.add_argument("--e2e_only", action="store_true",
                    help="Run only E2E and E2E+L2 models")
    ap.add_argument("--e2e_ridge_only", action="store_true",
                    help="Run only E2E+Ridge (linear drive, L2 penalty)")
    ap.add_argument("--e2e_diag_only", action="store_true",
                    help="Run only E2E+Diag (diagonal AR, spectral clamp)")
    ap.add_argument("--e2e_spectral_only", action="store_true",
                    help="Run only E2E+Spectral (α,θ oscillator)")
    ap.add_argument("--max_rho", type=float, default=0.98,
                    help="Max spectral radius for diagonal AR clamp")
    ap.add_argument("--w_phase", type=float, default=0.0,
                    help="Weight for phase-angle cosine loss on (a1,a2). "
                         "0=off, try 0.5–2.0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="output_plots/eigenworms/unified")
    args = ap.parse_args()

    t0 = time.time()

    # ── Data ──────────────────────────────────────────────────────────
    u, b_full, dt = load_data(args.h5, all_neurons=args.all_neurons)
    if args.max_frames > 0:
        u = u[:args.max_frames]
        b_full = b_full[:args.max_frames]
    K = min(args.n_modes, b_full.shape[1])
    b = b_full[:, :K]
    T = b.shape[0]
    n_lags = args.neural_lags
    warmup = max(2, n_lags)
    M_raw = u.shape[1]                          # raw neurons (for GRU)
    X_neural = build_lagged_features_np(u, n_lags)
    d_in = X_neural.shape[1]                    # lagged features

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = args.n_folds - 1                  # inner folds for α

    print(f"\n  Data: T={T}, K={K}, M={M_raw}, "
          f"d_features={d_in} ({M_raw}×{n_lags+1}), dt={dt:.2f}s")
    print(f"  CV: {args.n_folds} outer folds, {n_inner} inner folds for Ridge α")
    print(f"  Ridge grid: {len(ridge_grid)} values "
          f"[{ridge_grid[1]:.1e} .. {ridge_grid[-1]:.1e}]")

    # ── Folds (contiguous temporal blocks) ────────────────────────────
    valid_len = T - warmup
    fold_size = valid_len // args.n_folds
    folds = []
    for i in range(args.n_folds):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < args.n_folds - 1 else T
        folds.append((s, e))

    # ── Model names ───────────────────────────────────────────────────
    M_OLS   = "b̂ⱼ = wⱼᵀn + cⱼ                (OLS)"
    M_RIDGE = "b̂ⱼ = wⱼᵀn + cⱼ                (Ridge)"
    M_MLP   = "b̂ = MLP(n)"
    M_VAR_O = "b̂ = M₁b̂₋₁+M₂b̂₋₂+Wn+c         (VAR₂+OLS)"
    M_VAR_R = "b̂ = M₁b̂₋₁+M₂b̂₋₂+Wn+c         (VAR₂+Ridge)"
    M_VAR_M = "b̂ = M₁b̂₋₁+M₂b̂₋₂+MLP(n)+c     (VAR₂+MLP)"
    M_E2E   = "b̂ = M₁b̂₋₁+M₂b̂₋₂+MLP(n)+c     (E2E)"
    M_E2EL2 = "b̂ = M₁b̂₋₁+M₂b̂₋₂+MLP(n)+c     (E2E+L2)"
    M_E2E_R = "b̂ = M₁b̂₋₁+M₂b̂₋₂+Wn+c         (E2E+Ridge)"
    M_E2E_D = "b̂ = d₁⊙b̂₋₁+d₂⊙b̂₋₂+MLP(n)+c   (E2E+Diag)"
    M_E2E_A = "b̂ = d₁⊙b̂₋₁+d₂⊙b̂₋₂+MLP(n)+c   (E2E★)"
    M_E2E_S = "b̂ = U(α,θ)b̂₋₁+U(α²)b̂₋₂+MLP(n)  (Spectral)"
    M_GRU_B = "b̂ = Wh+c, h=GRU(h,[n,b̂₋₁])    (free-run)"
    M_GRU_N = "b̂ = Wh+c, h=GRU(h,n)           (free-run)"

    # Short names for video panel titles
    short_names = {
        M_OLS:   "OLS",
        M_RIDGE: "Ridge",
        M_MLP:   "MLP",
        M_VAR_O: "VAR₂+OLS",
        M_VAR_R: "VAR₂+Ridge",
        M_VAR_M: "VAR₂+MLP",
        M_E2E:   "E2E",
        M_E2EL2: "E2E+L2",
        M_E2E_R: "E2E+Ridge",
        M_E2E_D: "E2E+Diag",
        M_E2E_A: "E2E★",
        M_E2E_S: "Spectral",
        M_GRU_B: "GRU+beh",
        M_GRU_N: "GRU",
    }

    if args.e2e_spectral_only:
        all_models = [M_E2E_S]
    elif args.e2e_diag_only:
        all_models = [M_E2E_D]
    elif args.e2e_ridge_only:
        all_models = [M_E2E_R]
    elif args.e2e_only:
        all_models = [M_E2E, M_E2EL2, M_E2E_D, M_E2E_A, M_E2E_S]
    elif args.no_gru:
        all_models = [M_OLS, M_RIDGE, M_MLP, M_VAR_O, M_VAR_R, M_VAR_M,
                      M_E2E, M_E2EL2, M_E2E_R, M_E2E_D, M_E2E_A, M_E2E_S]
    else:
        all_models = [M_OLS, M_RIDGE, M_MLP, M_VAR_O, M_VAR_R, M_VAR_M,
                      M_E2E, M_E2EL2, M_E2E_R, M_E2E_D, M_E2E_A,
                      M_E2E_S, M_GRU_B, M_GRU_N]
    n_models = len(all_models)
    ho_preds = {m: np.full((T, K), np.nan) for m in all_models}

    # α tracker:  tag → mode_j → [{fold, alpha, at_zero, at_upper}]
    alpha_log = defaultdict(lambda: defaultdict(list))

    print(f"\n{'═'*90}")
    print(f"  {n_models} models × {args.n_folds} folds")
    print(f"{'═'*90}\n")

    # ══════════════════════════════════════════════════════════════════
    #  Per-fold loop
    # ══════════════════════════════════════════════════════════════════
    for fi, (ts, te) in enumerate(folds):
        test_len = te - ts
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[ts:te] = False
        train_idx = np.where(train_mask)[0]

        X_tr = X_neural[train_idx]
        X_te = X_neural[ts:te]
        b_tr = b[train_idx]

        print(f"  ═══ Fold {fi+1}/{args.n_folds}: test=[{ts}:{te}) "
              f"({test_len} fr, {test_len*dt:.0f}s), "
              f"train={train_idx.size} fr ═══")

        # ── 1. OLS linear ────────────────────────────────────────────
        if M_OLS in all_models:
            print(f"    [1/{n_models}] OLS linear")
            for j in range(K):
                coef, intc = _ols_fit(X_tr, b_tr[:, j])
                ho_preds[M_OLS][ts:te, j] = X_te @ coef + intc

        # ── 2. Ridge linear ──────────────────────────────────────────
        if M_RIDGE in all_models:
            print(f"    [2/{n_models}] Ridge linear")
            for j in range(K):
                coef, intc, info = _ridge_fit(
                    X_tr, b_tr[:, j], ridge_grid, n_inner)
                ho_preds[M_RIDGE][ts:te, j] = X_te @ coef + intc
                alpha_log["Ridge linear"][j].append({"fold": fi, **info})

        # ── 3. MLP ───────────────────────────────────────────────────
        if M_MLP in all_models:
            print(f"    [3/{n_models}] MLP")
            mlp = _train_mlp(X_tr, b_tr, K)
            ho_preds[M_MLP][ts:te] = _predict_mlp(mlp, X_te)

        # ── 4. VAR(2)+OLS  (2-step, free-run) ───────────────────────
        if M_VAR_O in all_models:
            print(f"    [4/{n_models}] VAR₂+OLS")
            X_ar = build_ar_features(b, 2)
            X_ar_tr = X_ar[train_idx]

            M1_o = np.zeros((K, K)); M2_o = np.zeros((K, K)); c_o = np.zeros(K)
            for j in range(K):
                coef, intc = _ols_fit(X_ar_tr, b_tr[:, j])
                M1_o[j] = coef[:K]
                M2_o[j] = coef[K:2*K]
                c_o[j] = intc

            # Residual & neural drive (OLS)
            b_tm1 = np.zeros_like(b); b_tm1[1:] = b[:-1]
            b_tm2 = np.zeros_like(b); b_tm2[2:] = b[:-2]
            resid_o = b - (b_tm1 @ M1_o.T + b_tm2 @ M2_o.T + c_o)
            r_tr_o = resid_o[train_idx]

            coefs_do = np.zeros((d_in, K)); ints_do = np.zeros(K)
            for j in range(K):
                coefs_do[:, j], ints_do[j] = _ols_fit(X_tr, r_tr_o[:, j])

            # Free-run
            drive_o = X_neural @ coefs_do + ints_do[None, :]
            p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1_o @ p1 + M2_o @ p2 + drive_o[t] + c_o
                ho_preds[M_VAR_O][t] = p_new
                p2, p1 = p1, p_new

        # ── 5. VAR(2)+Ridge  (2-step, free-run) ─────────────────────
        need_var_ridge = (M_VAR_R in all_models or M_VAR_M in all_models)
        if need_var_ridge:
            print(f"    [5/{n_models}] VAR₂+Ridge")
            X_ar = build_ar_features(b, 2)
            X_ar_tr = X_ar[train_idx]

            M1 = np.zeros((K, K)); M2 = np.zeros((K, K)); c_v = np.zeros(K)
            for j in range(K):
                coef, intc, info = _ridge_fit(
                    X_ar_tr, b_tr[:, j], ridge_grid, n_inner)
                M1[j] = coef[:K]
                M2[j] = coef[K:2*K]
                c_v[j] = intc
                alpha_log["VAR₂ matrices"][j].append({"fold": fi, **info})

            # Residual & neural drive (Ridge)
            b_tm1 = np.zeros_like(b); b_tm1[1:] = b[:-1]
            b_tm2 = np.zeros_like(b); b_tm2[2:] = b[:-2]
            resid = b - (b_tm1 @ M1.T + b_tm2 @ M2.T + c_v)
            r_tr = resid[train_idx]

            coefs_d = np.zeros((d_in, K)); ints_d = np.zeros(K)
            for j in range(K):
                coefs_d[:, j], ints_d[j], info = _ridge_fit(
                    X_tr, r_tr[:, j], ridge_grid, n_inner)
                alpha_log["VAR₂ neural drive"][j].append(
                    {"fold": fi, **info})

            # Free-run
            drive_all = X_neural @ coefs_d + ints_d[None, :]
            p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1 @ p1 + M2 @ p2 + drive_all[t] + c_v
                ho_preds[M_VAR_R][t] = p_new
                p2, p1 = p1, p_new

            # ── 6. VAR(2)+MLP  (2-step, free-run) ────────────────────
            if M_VAR_M in all_models:
                print(f"    [6/{n_models}] VAR₂+MLP")
                mlp_r = _train_mlp(X_tr, r_tr, K)
                drive_mlp = _predict_mlp(mlp_r, X_neural)
                p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
                for t in range(ts, te):
                    p_new = M1 @ p1 + M2 @ p2 + drive_mlp[t] + c_v
                    ho_preds[M_VAR_M][t] = p_new
                    p2, p1 = p1, p_new

        # ── 7. E2E VAR(2)+MLP  (BPTT, free-run, no reg) ─────────────
        if M_E2E in all_models:
            print(f"    [7/{n_models}] E2E ({args.e2e_epochs} ep)")
            segs = []
            if ts > warmup + 2:
                segs.append((warmup, ts))
            if te + 2 < T:
                segs.append((te, T))

            M1e, M2e, ce_np, drv_np = _train_e2e(
                d_in, K, segs, b, X_neural, args.e2e_epochs,
                args.tbptt_chunk, weight_decay=0.0,
                w_phase=args.w_phase, tag="E2E")
            p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1e @ p1 + M2e @ p2 + drv_np[t] + ce_np
                ho_preds[M_E2E][t] = p_new
                p2, p1 = p1, p_new

        # ── 8. E2E VAR(2)+MLP  (BPTT, free-run, L2 reg) ───────────
        if M_E2EL2 in all_models:
            print(f"    [8/{n_models}] E2E+L2 ({args.e2e_epochs} ep)")
            segs = []
            if ts > warmup + 2:
                segs.append((warmup, ts))
            if te + 2 < T:
                segs.append((te, T))
            M1r, M2r, cr_np, drv_r = _train_e2e(
                d_in, K, segs, b, X_neural, args.e2e_epochs,
                args.tbptt_chunk, weight_decay=1e-3,
                w_phase=args.w_phase, tag="E2E+L2")
            p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1r @ p1 + M2r @ p2 + drv_r[t] + cr_np
                ho_preds[M_E2EL2][t] = p_new
                p2, p1 = p1, p_new

        # ── 9. E2E VAR(2)+Ridge (linear drive, L2) ────────────────
        if M_E2E_R in all_models:
            e2e_r_idx = all_models.index(M_E2E_R) + 1
            print(f"    [{e2e_r_idx}/{n_models}] E2E+Ridge "
                  f"({args.e2e_epochs} ep)")
            segs = []
            if ts > warmup + 2:
                segs.append((warmup, ts))
            if te + 2 < T:
                segs.append((te, T))
            M1lr, M2lr, clr_np, drv_lr = _train_e2e(
                d_in, K, segs, b, X_neural, args.e2e_epochs,
                args.tbptt_chunk, weight_decay=1e-3,
                linear_drive=True, w_phase=args.w_phase,
                tag="E2E+Ridge")
            p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1lr @ p1 + M2lr @ p2 + drv_lr[t] + clr_np
                ho_preds[M_E2E_R][t] = p_new
                p2, p1 = p1, p_new

        # ── 10. E2E VAR(2)+Diag (diagonal AR, spectral clamp) ──────
        if M_E2E_D in all_models:
            e2e_d_idx = all_models.index(M_E2E_D) + 1
            print(f"    [{e2e_d_idx}/{n_models}] E2E+Diag "
                  f"({args.e2e_epochs} ep, ρ≤{args.max_rho})")
            segs = []
            if ts > warmup + 2:
                segs.append((warmup, ts))
            if te + 2 < T:
                segs.append((te, T))
            M1d, M2d, cd_np, drv_d = _train_e2e(
                d_in, K, segs, b, X_neural, args.e2e_epochs,
                args.tbptt_chunk, weight_decay=1e-3,
                diagonal_ar=True, max_rho=args.max_rho,
                w_phase=args.w_phase, tag="E2E+Diag")
            p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1d @ p1 + M2d @ p2 + drv_d[t] + cd_np
                ho_preds[M_E2E_D][t] = p_new
                p2, p1 = p1, p_new

        # ── 11. E2E★ (Diag + L2 + phase — all combined) ─────────────
        if M_E2E_A in all_models:
            e2e_a_idx = all_models.index(M_E2E_A) + 1
            # Use w_phase=1.0 if user didn't set it, otherwise honour CLI
            wp = args.w_phase if args.w_phase > 0 else 1.0
            print(f"    [{e2e_a_idx}/{n_models}] E2E★ "
                  f"({args.e2e_epochs} ep, ρ≤{args.max_rho}, "
                  f"wd=1e-3, φ={wp})")
            segs = []
            if ts > warmup + 2:
                segs.append((warmup, ts))
            if te + 2 < T:
                segs.append((te, T))
            M1a, M2a, ca_np, drv_a = _train_e2e(
                d_in, K, segs, b, X_neural, args.e2e_epochs,
                args.tbptt_chunk, weight_decay=1e-3,
                diagonal_ar=True, max_rho=args.max_rho,
                w_phase=wp, tag="E2E★")
            p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1a @ p1 + M2a @ p2 + drv_a[t] + ca_np
                ho_preds[M_E2E_A][t] = p_new
                p2, p1 = p1, p_new

        # ── 12. E2E+Spectral (α,θ oscillator) ──────────────────────
        if M_E2E_S in all_models:
            e2e_s_idx = all_models.index(M_E2E_S) + 1
            wp = args.w_phase if args.w_phase > 0 else 1.0
            print(f"    [{e2e_s_idx}/{n_models}] Spectral "
                  f"({args.e2e_epochs} ep, wd=1e-3, φ={wp})")
            segs = []
            if ts > warmup + 2:
                segs.append((warmup, ts))
            if te + 2 < T:
                segs.append((te, T))
            M1s, M2s, cs_np, drv_s = _train_e2e_spectral(
                d_in, K, segs, b, X_neural, args.e2e_epochs,
                args.tbptt_chunk, weight_decay=1e-3,
                w_phase=wp, tag="Spectral")
            p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1s @ p1 + M2s @ p2 + drv_s[t] + cs_np
                ho_preds[M_E2E_S][t] = p_new
                p2, p1 = p1, p_new

        # ── 13–14. GRU models (skipped with --no_gru) ────────────────
        if not args.no_gru and M_GRU_B in all_models:
            print(f"    [9/{n_models}] GRU+beh ({args.gru_epochs} ep)")
            gru_warmup = 10
            segs_g = []
            if ts > warmup + gru_warmup + 2:
                segs_g.append((warmup, ts))
            if te + gru_warmup + 2 < T:
                segs_g.append((te, T))
            if not segs_g:
                segs_g.append(
                    (warmup, ts) if ts > warmup else (te, T))

            gru = GRUDecoder(M_raw, K, hidden=args.gru_hidden,
                             use_beh_input=True)
            gru = train_gru(gru, u, b, segs_g, warmup=gru_warmup,
                            epochs=args.gru_epochs, lr=1e-3,
                            device=args.device)
            ho_preds[M_GRU_B][ts:te] = free_run_eval(
                gru, u, b, ts, te, warmup=gru_warmup,
                device=args.device)

            print(f"    [10/{n_models}] GRU (no beh, {args.gru_epochs} ep)")
            gru_nb = GRUDecoder(M_raw, K, hidden=args.gru_hidden,
                                use_beh_input=False)
            gru_nb = train_gru(gru_nb, u, b, segs_g, warmup=gru_warmup,
                               epochs=args.gru_epochs, lr=1e-3,
                               device=args.device)
            ho_preds[M_GRU_N][ts:te] = free_run_eval(
                gru_nb, u, b, ts, te, warmup=gru_warmup,
                device=args.device)

        print()

    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════════════
    #  Compute R²
    # ══════════════════════════════════════════════════════════════════
    valid = np.arange(warmup, T)
    results = {}
    for name in all_models:
        preds = ho_preds[name]
        ok = np.isfinite(preds[valid, 0])
        idx = valid[ok]
        if idx.size < 10:
            results[name] = np.full(K, np.nan)
            continue
        results[name] = np.array([
            r2_score(b[idx, j], preds[idx, j]) for j in range(K)])

    # ══════════════════════════════════════════════════════════════════
    #  Report
    # ══════════════════════════════════════════════════════════════════
    mode_names = [f"a{j+1}" for j in range(K)]
    cw = 52

    print("\n" + "═" * 100)
    print("  UNIFIED DECODER BENCHMARK — 5-fold temporal CV, free-run R²")
    print("═" * 100)
    header = (f"  {'Model (formula)':<{cw}s}"
              + "".join(f"{m:>8s}" for m in mode_names))
    print(header)
    print("  " + "─" * (cw + 8 * K))

    for name in all_models:
        vals = results[name]
        parts = [f"  {name:<{cw}s}"]
        for v in vals[:K]:
            parts.append(f"{v:8.3f}" if np.isfinite(v) else f"{'---':>8s}")
        print("".join(parts))

    # ── Ridge α per mode ─────────────────────────────────────────────
    print(f"\n  Ridge α per mode (across {args.n_folds} folds):")
    print("  " + "─" * 85)
    boundary_warnings = []

    for tag in sorted(alpha_log.keys()):
        print(f"\n  {tag}:")
        for j in range(K):
            entries = alpha_log[tag][j]
            if not entries:
                continue
            alphas = np.array([e["alpha"] for e in entries])
            n_zero = sum(1 for e in entries if e["at_zero"])
            n_upper = sum(1 for e in entries if e["at_upper"])
            line = (f"    a{j+1}:  α median={np.median(alphas):10.1f}   "
                    f"range=[{alphas.min():.1f}, {alphas.max():.1f}]")
            if n_zero:
                line += f"   ⚠ α=0 in {n_zero}/{len(entries)} folds"
                boundary_warnings.append(
                    f"{tag} a{j+1}: α hit lower bound (0)")
            if n_upper:
                line += f"   ⚠ α=MAX in {n_upper}/{len(entries)} folds"
                boundary_warnings.append(
                    f"{tag} a{j+1}: α hit upper bound ({ridge_grid[-1]:.0e})")
            print(line)

    if boundary_warnings:
        print(f"\n  ⚠ BOUNDARY WARNINGS ({len(boundary_warnings)}):")
        for w in boundary_warnings:
            print(f"    • {w}")
    else:
        print(f"\n  ✓ No Ridge α boundary hits.")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ══════════════════════════════════════════════════════════════════
    #  Save: figure + text
    # ══════════════════════════════════════════════════════════════════
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Bar chart ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, K * 2.5), 6.5))
    x = np.arange(K)
    n = len(all_models)
    w = 0.8 / n
    cmap = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))

    for i, name in enumerate(all_models):
        vals = np.clip(results[name], -0.3, 1.0)
        ax.bar(x + i * w, vals, w, label=name.strip(),
               color=cmap[i], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(mode_names, fontsize=12)
    ax.set_ylabel("R²  (held-out, free-run)", fontsize=12)
    ax.set_title("Unified Decoder Benchmark — 5-fold temporal CV, free-run",
                 fontsize=13)
    ax.legend(fontsize=6, ncol=2, loc="upper right",
              framealpha=0.9, borderpad=0.8)
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "unified_benchmark.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure → {fig_path}")

    # ── Text summary ─────────────────────────────────────────────────
    txt_path = out_dir / "unified_benchmark.txt"
    with open(txt_path, "w") as f:
        f.write("Unified Decoder Benchmark — 5-fold temporal CV, "
                "free-run R²\n")
        f.write(f"h5: {args.h5}\n")
        f.write(f"neural_lags={n_lags}, K={K}, "
                f"n_folds={args.n_folds}, n_inner={n_inner}\n\n")
        for name in all_models:
            vals = results[name]
            f.write(f"{name:52s}: "
                    + " ".join(f"{v:7.3f}" for v in vals) + "\n")

        f.write(f"\nRidge α per mode:\n")
        for tag in sorted(alpha_log.keys()):
            f.write(f"  {tag}:\n")
            for j in range(K):
                entries = alpha_log[tag][j]
                if entries:
                    alphas = np.array([e["alpha"] for e in entries])
                    f.write(
                        f"    a{j+1}: median={np.median(alphas):.1f}  "
                        f"range=[{alphas.min():.1f}, "
                        f"{alphas.max():.1f}]\n")
        if boundary_warnings:
            f.write(f"\nBoundary warnings:\n")
            for w in boundary_warnings:
                f.write(f"  • {w}\n")

    print(f"  Summary → {txt_path}")

    # ── Comparison video ─────────────────────────────────────────────
    # Use the middle fold's test window for the video
    mid = args.n_folds // 2
    vid_start, vid_end = folds[mid]
    vid_len = vid_end - vid_start
    vid_preds = {}
    for name in all_models:
        r2s = results[name]
        mean_r2 = np.nanmean(r2s[:2])  # mean of a1, a2
        label = f"{short_names[name]} (R²={mean_r2:.2f})"
        vid_preds[label] = ho_preds[name][vid_start:vid_end]

    vid_path = str(out_dir / "unified_benchmark.mp4")
    try:
        make_comparison_video(
            h5_path=args.h5,
            out_path=vid_path,
            b_gt=b[vid_start:vid_end],
            predictions=vid_preds,
            dt=dt,
            fps=15,
            max_frames=min(vid_len, 200),
            body_angle_offset=vid_start,
        )
    except Exception as exc:
        print(f"  ⚠ Video failed: {exc}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
