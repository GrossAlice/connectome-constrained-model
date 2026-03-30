#!/usr/bin/env python
r"""Benchmark behaviour decoders: Ridge vs MLP × AR(0)/AR(1)/AR(2).

Compares six decoder configurations on held-out CV folds and generates a
posture comparison video for the best strategies.

## Formulas
─────────────────────────────────────────────────────────────────────────────

### Feature construction

Given neural activity  n_t ∈ ℝ^M  (M motor+command neurons) and behaviour
(eigenworm amplitudes)  b_t ∈ ℝ^K  (K=6 modes):

  Neural features (lag L=8):
      x^neural_t = [ n_t ; n_{t-1} ; … ; n_{t-L} ]  ∈ ℝ^{M·(L+1)}

  AR(p) features (lag p):
      x^ar_t = [ b_{t-1} ; b_{t-2} ; … ; b_{t-p} ]  ∈ ℝ^{K·p}

### Decoders

  A. Ridge-linear (per-mode j):
      b̂_t^(j) = w_j^⊤ x_t + c_j
      where x_t ∈ {x^neural, x^ar, [x^ar ; x^neural]}, and w_j, c_j
      are fit by ridge regression with λ chosen by 5-fold CV.

  B. MLP (2-layer, shared across modes):
      h = ReLU( LayerNorm( W₁ x_t + b₁ ) )
      b̂_t = W₂ h + b₂
      W₁ ∈ ℝ^{H×d}, W₂ ∈ ℝ^{K×H}, H=32 hidden units.
      Trained by Adam (lr=1e-3, 200 epochs, 5-fold CV, early stopping).

### AR orders

  AR(0):  x_t = x^neural_t             (current pipeline, no AR)
  AR(1):  x_t = [ b_{t-1} ; x^neural_t ]
  AR(2):  x_t = [ b_{t-1} ; b_{t-2} ; x^neural_t ]

### Evaluation modes

  Teacher-forced:  AR lags use ground-truth b
  Free-running:    AR lags use own predictions (seeded with GT for warmup)

### Free-running rollout loop

  b̂_0 … b̂_{w-1} = b_0 … b_{w-1}   (warmup = max(p, L))
  for t = w … T-1:
      x^ar_t  = [ b̂_{t-1} ; … ; b̂_{t-p} ]
      x^nn_t  = x^neural_t              (from GT neural traces)
      b̂_t    = decoder( [x^ar_t ; x^nn_t] )

Usage:
    python -m scripts.benchmark_ar_decoder_v2 \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-06-14-01.h5"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --------------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stage2.behavior_decoder_eval import (
    _log_ridge_grid,
    _ridge_cv_single_target,
    build_lagged_features_np,
)
from stage2.posture_videos import angles_to_xy, _load_eigenvectors

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def load_data(h5_path: str, all_neurons: bool = False):
    """Return u (T, M), b (T, K), dt.

    If *all_neurons* is True, return all N neurons; otherwise restrict to
    the 27 motor+command neurons.
    """
    motor_file = (Path(__file__).resolve().parent.parent /
                  "data/used/masks+motor neurons/motor_neurons_with_control.txt")
    motor_names = [l.strip() for l in motor_file.read_text().splitlines()
                   if l.strip() and not l.startswith("#")]

    with h5py.File(h5_path, "r") as f:
        u = f["stage1/u_mean"][:]
        for key in ("neuron_names", "gcamp/neuron_labels", "gcamp/neuron_names"):
            if key in f:
                neuron_names = [n.decode() if isinstance(n, bytes) else n
                                for n in f[key][:]]
                break
        else:
            raise KeyError("Cannot find neuron names in h5 file")
        b = f["behaviour/eigenworms_stephens"][:]
        dt = float(f.attrs.get("dt", 0.6))

    if all_neurons:
        u_out = u
        label = f"N_all={u.shape[1]}"
    else:
        name2idx = {n: i for i, n in enumerate(neuron_names)}
        motor_idx = [name2idx[n] for n in motor_names if n in name2idx]
        u_out = u[:, motor_idx]
        label = f"M_motor={len(motor_idx)}"
    print(f"  Loaded {h5_path}")
    print(f"    T={u.shape[0]}, N_total={u.shape[1]}, {label}, "
          f"K={b.shape[1]}, dt={dt:.3f}s")
    return u_out, b, dt


def build_ar_features(b: np.ndarray, ar_lags: int) -> np.ndarray:
    """(T, K*ar_lags): row t = [b_{t-1}, ..., b_{t-p}]."""
    if ar_lags <= 0:
        return np.zeros((b.shape[0], 0), dtype=b.dtype)
    parts = []
    for lag in range(1, ar_lags + 1):
        shifted = np.zeros_like(b)
        shifted[lag:] = b[:-lag]
        parts.append(shifted)
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------- #
#  Ridge decoder (per-mode)
# --------------------------------------------------------------------------- #

def ridge_cv_decode(X, b, n_folds, ridge_grid, valid_start):
    """Returns held-out predictions (T, K) and R² per mode."""
    T, K = b.shape
    mask = np.ones(T, dtype=bool)
    mask[:valid_start] = False
    idx = np.where(mask)[0]
    preds_ho = np.full_like(b, np.nan)
    preds_full = np.full_like(b, np.nan)
    r2s = np.full(K, np.nan)
    coefs_out = np.zeros((X.shape[1], K))
    intercepts_out = np.zeros(K)

    for j in range(K):
        if idx.size < 10:
            continue
        fit = _ridge_cv_single_target(X, b[:, j], idx, ridge_grid, n_folds)
        preds_ho[:, j] = fit["held_out"]
        preds_full[:, j] = fit["pred_full"]
        coefs_out[:, j] = fit["coef"]
        intercepts_out[j] = fit["intercept"]
        ok = mask & np.isfinite(fit["held_out"])
        if ok.sum() > 2:
            r2s[j] = r2_score(b[ok, j], fit["held_out"][ok])

    return r2s, preds_ho, preds_full, coefs_out, intercepts_out


# --------------------------------------------------------------------------- #
#  MLP decoder (all modes jointly, 5-fold CV)
# --------------------------------------------------------------------------- #

def mlp_cv_decode(X, b, n_folds, valid_start, hidden=32, lr=1e-3, epochs=200):
    """Train MLP with 5-fold CV, return held-out preds and R²."""
    import torch
    import torch.nn as nn

    T, K = b.shape
    d = X.shape[1]
    mask = np.ones(T, dtype=bool)
    mask[:valid_start] = False
    idx = np.where(mask)[0]

    # Split into folds
    fold_size = len(idx) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(idx)
        folds.append(idx[start:end])

    preds_ho = np.full_like(b, np.nan)
    device = "cpu"

    for fi, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fi])

        X_tr = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
        y_tr = torch.tensor(b[train_idx], dtype=torch.float32, device=device)
        X_va = torch.tensor(X[val_idx], dtype=torch.float32, device=device)

        model = nn.Sequential(
            nn.Linear(d, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, K),
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_loss = float("inf")
        patience_ctr = 0

        for ep in range(epochs):
            model.train()
            pred = model(X_tr)
            loss = nn.functional.mse_loss(pred, y_tr)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr > 20:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds_ho[val_idx] = model(X_va).cpu().numpy()

    r2s = np.full(K, np.nan)
    for j in range(K):
        ok = mask & np.isfinite(preds_ho[:, j])
        if ok.sum() > 2:
            r2s[j] = r2_score(b[ok, j], preds_ho[ok, j])

    return r2s, preds_ho


# --------------------------------------------------------------------------- #
#  Free-running rollout (works for both ridge and MLP)
# --------------------------------------------------------------------------- #

def free_run_ridge(b, X_neural, ar_lags, coefs, intercepts):
    """Free-running rollout with ridge model."""
    T, K = b.shape
    warmup = max(ar_lags, 1)
    b_hat = np.copy(b)  # seed everything, overwrite from warmup on

    for t in range(warmup, T):
        ar_feats = []
        for lag in range(1, ar_lags + 1):
            ar_feats.append(b_hat[t - lag])
        ar_part = np.concatenate(ar_feats) if ar_feats else np.array([])
        neural_part = X_neural[t]
        x = np.concatenate([ar_part, neural_part])
        for j in range(K):
            b_hat[t, j] = intercepts[j] + x @ coefs[:, j]

    return b_hat


# --------------------------------------------------------------------------- #
#  Oscillator + MLP neural drive
#
#  Model:  P_t = M1 · P_{t-1} + M2 · P_{t-2} + MLP(n_{t-L:t})
#
#  Fitting:
#    1. Fit VAR(2) on GT behaviour  →  M1, M2 ∈ ℝ^{K×K}
#    2. Compute residual  r_t = b_t − M1·b_{t-1} − M2·b_{t-2}
#    3. Fit MLP: neural features → r_t
#  Free-run:
#    b̂_t = M1 · b̂_{t-1} + M2 · b̂_{t-2} + MLP(n_{t-L:t})
# --------------------------------------------------------------------------- #

def fit_var2_matrices(b: np.ndarray, ridge_grid: np.ndarray,
                      n_folds: int, warmup: int):
    """Fit VAR(2): b_t = M1 b_{t-1} + M2 b_{t-2} + c.

    Returns M1 (K,K), M2 (K,K), c (K,), and the residual (T,K).
    """
    T, K = b.shape
    X_ar = build_ar_features(b, 2)  # (T, 2K): [b_{t-1}, b_{t-2}]
    mask = np.ones(T, dtype=bool)
    mask[:warmup] = False
    idx = np.where(mask)[0]

    M1 = np.zeros((K, K))
    M2 = np.zeros((K, K))
    c = np.zeros(K)

    for j in range(K):
        fit = _ridge_cv_single_target(X_ar, b[:, j], idx, ridge_grid, n_folds)
        # coef layout: [b_{t-1} weights (K), b_{t-2} weights (K)]
        M1[j, :] = fit["coef"][:K]
        M2[j, :] = fit["coef"][K:2*K]
        c[j] = fit["intercept"]

    # Compute residual: r_t = b_t - M1 b_{t-1} - M2 b_{t-2} - c
    b_tm1 = np.zeros_like(b); b_tm1[1:] = b[:-1]
    b_tm2 = np.zeros_like(b); b_tm2[2:] = b[:-2]
    residual = b - (b_tm1 @ M1.T + b_tm2 @ M2.T + c[None, :])
    residual[:warmup] = 0.0

    return M1, M2, c, residual


def fit_mlp_on_residual(X_neural, residual, n_folds, warmup,
                        hidden=32, lr=1e-3, epochs=200):
    """Fit MLP: neural features → VAR(2) residual. Returns trained models."""
    import torch
    import torch.nn as nn

    T, K = residual.shape
    d = X_neural.shape[1]
    mask = np.ones(T, dtype=bool)
    mask[:warmup] = False
    idx = np.where(mask)[0]

    fold_size = len(idx) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(idx)
        folds.append(idx[start:end])

    # Train on ALL data for the final model (for free-run)
    device = "cpu"
    X_all = torch.tensor(X_neural[idx], dtype=torch.float32, device=device)
    y_all = torch.tensor(residual[idx], dtype=torch.float32, device=device)

    model = nn.Sequential(
        nn.Linear(d, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, K),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        pred = model(X_all)
        loss = nn.functional.mse_loss(pred, y_all)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience > 20:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Also get the full-data prediction for diagnostics
    with torch.no_grad():
        X_full_t = torch.tensor(X_neural, dtype=torch.float32, device=device)
        drive_full = model(X_full_t).cpu().numpy()

    return model, drive_full


def free_run_oscillator_mlp(b, X_neural, M1, M2, c, mlp_model, warmup):
    """Free-run: P_t = M1·P_{t-1} + M2·P_{t-2} + MLP(n_t) + c."""
    import torch

    T, K = b.shape
    b_hat = np.copy(b)
    device = "cpu"

    mlp_model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_neural, dtype=torch.float32, device=device)
        drive = mlp_model(X_t).cpu().numpy()  # (T, K)

    for t in range(warmup, T):
        b_hat[t] = M1 @ b_hat[t-1] + M2 @ b_hat[t-2] + drive[t] + c

    return b_hat


def free_run_oscillator_ridge(b, X_neural, M1, M2, c, ridge_coefs,
                              ridge_intercepts, warmup):
    """Free-run: P_t = M1·P_{t-1} + M2·P_{t-2} + Ridge(n_t) + c."""
    T, K = b.shape
    b_hat = np.copy(b)

    # Pre-compute ridge neural drive for all t
    drive = X_neural @ ridge_coefs + ridge_intercepts[None, :]

    for t in range(warmup, T):
        b_hat[t] = M1 @ b_hat[t-1] + M2 @ b_hat[t-2] + drive[t] + c

    return b_hat


# --------------------------------------------------------------------------- #
#  End-to-end Oscillator + MLP  (jointly trained via BPTT)
#
#   P̂_t = M1·P̂_{t-1} + M2·P̂_{t-2} + MLP(n_t) + c
#
#   All parameters trained jointly by unrolling the recurrence and
#   backpropagating through time.  No GT posture is used at inference.
# --------------------------------------------------------------------------- #

class E2EOscillatorMLP:
    """End-to-end VAR(2) oscillator + MLP neural drive, trained via BPTT."""

    def __init__(self, d_neural: int, K: int, hidden: int = 128,
                 n_layers: int = 2):
        import torch
        import torch.nn as nn

        self.K = K
        self.device = "cpu"

        # Oscillator matrices  (K × K)
        self.M1 = nn.Linear(K, K, bias=False).to(self.device)
        self.M2 = nn.Linear(K, K, bias=False).to(self.device)
        self.c  = nn.Parameter(torch.zeros(K, device=self.device))

        # Neural drive MLP  (default: 2×128, matching standalone MLP)
        layers = []
        in_dim = d_neural
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                       nn.ReLU(), nn.Dropout(0.1)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, K))
        self.mlp = nn.Sequential(*layers).to(self.device)

        self._params = (list(self.M1.parameters()) +
                        list(self.M2.parameters()) +
                        [self.c] +
                        list(self.mlp.parameters()))

    def parameters(self):
        return self._params

    def train(self):
        self.M1.train(); self.M2.train(); self.mlp.train()

    def eval(self):
        self.M1.eval(); self.M2.eval(); self.mlp.eval()

    def free_run(self, b_gt_np, X_neural_np, warmup):
        """Unroll recurrence: P̂_t = M1·P̂_{t-1} + M2·P̂_{t-2} + MLP(n_t) + c.

        Returns (T, K) tensor.  b_gt is used only for warmup frames.
        """
        import torch

        T = X_neural_np.shape[0]
        b_gt = torch.tensor(b_gt_np, dtype=torch.float32, device=self.device)
        X_n  = torch.tensor(X_neural_np, dtype=torch.float32, device=self.device)

        drive = self.mlp(X_n)  # (T, K)

        preds = []
        p_prev1 = b_gt[warmup - 1]  # seed
        p_prev2 = b_gt[warmup - 2]

        for t in range(warmup, T):
            p_t = self.M1(p_prev1) + self.M2(p_prev2) + drive[t] + self.c
            preds.append(p_t)
            p_prev2 = p_prev1
            p_prev1 = p_t

        return torch.stack(preds, dim=0)  # (T - warmup, K)


# --------------------------------------------------------------------------- #
#  Polar phase-aware loss
#
#  The a1-a2 pair encodes a limit cycle.  Standard MSE misses slow phase
#  drift because |Δa| is small frame-to-frame even when φ drifts.  This
#  ADDS a phase penalty on top of MSE:
#
#      L = MSE(pred, target) + w_φ · (1 − cos(φ̂ − φ)) / K
# --------------------------------------------------------------------------- #

def _polar_loss_frame(pred, target, w_phase=1.0):
    """MSE + auxiliary phase penalty on a1-a2.

    The standard Cartesian MSE is the base loss.  An extra cosine-distance
    term on the phase of the (a1, a2) pair nudges the optimiser toward
    phase coherence without destabilising the rest.

    pred, target: (K,) tensors.
    """
    import torch
    K = pred.shape[0]

    # Base: standard MSE over ALL modes
    mse = torch.sum((pred - target) ** 2) / K

    # Auxiliary: phase penalty on modes 0, 1 (limit-cycle pair)
    phi_pred = torch.atan2(pred[1], pred[0])
    phi_gt   = torch.atan2(target[1], target[0])
    phase_pen = (1.0 - torch.cos(phi_pred - phi_gt)) / K

    return mse + w_phase * phase_pen


# --------------------------------------------------------------------------- #
#  Spectral oscillator  (VAR(2), companion eigenvalues controlled)
#
#   b̂_t = M1·b̂_{t-1} + M2·b̂_{t-2} + MLP(n_t) + c
#
#  M1, M2 are constructed so the companion matrix C = [[M1,M2],[I,0]]
#  has eigenvalues with prescribed |eigenvalue| near a target radius.
#
#  Parameterisation (for K even):
#    • K/2 oscillator blocks, each with learnable (αⱼ, θⱼ)
#      αⱼ = damping radius ∈ [0.90, 1.0]   (initialised near 0.99)
#      θⱼ = rotation angle                    (initialised from data)
#    • Block j has VAR(2) companion eigenvalues αⱼ·e^{±iθⱼ}
#      This gives: m1_j = 2αⱼ·cos(θⱼ),  m2_j = -αⱼ²
#    • Learnable orthogonal basis U mixes modes.
#
#  Result: each oscillatory mode has independently controlled
#  damping and frequency.  a1-a2 can be nearly undamped (α≈0.998)
#  while a5-a6 can be more damped (α≈0.95).
# --------------------------------------------------------------------------- #

class E2EUndampedMLP:
    """Spectral VAR(2) + MLP:  controlled companion eigenvalues.

    b̂_t = M1·b̂_{t-1} + M2·b̂_{t-2} + MLP(n_t) + c
    M1, M2 constructed from K/2 (radius, angle) pairs + orthogonal basis.
    """

    def __init__(self, d_neural: int, K: int, hidden: int = 32,
                 alpha_init: float = 0.99):
        import torch
        import torch.nn as nn

        self.K = K
        self.device = "cpu"
        n_blocks = K // 2
        self.n_blocks = n_blocks
        K_eff = 2 * n_blocks  # handle odd K by ignoring last mode
        self.K_eff = K_eff

        # Per-block learnable parameters
        # α_j ∈ [0.90, 1.0] via sigmoid: α = 0.90 + 0.10 * sigmoid(raw)
        init_raw = float(np.log(((alpha_init - 0.90) / 0.10)
                                / (1.0 - (alpha_init - 0.90) / 0.10)))
        self._alpha_raw = nn.Parameter(
            torch.full((n_blocks,), init_raw))

        # θ_j: rotation angles (initialised to typical oscillation freqs)
        # periods ≈ 9s, 6s, 20s → θ = 2π·dt/period
        dt_est = 0.6
        default_periods = [9.0, 6.0, 20.0, 12.0, 8.0, 15.0][:n_blocks]
        init_theta = [2.0 * np.pi * dt_est / p for p in default_periods]
        self._theta = nn.Parameter(
            torch.tensor(init_theta, dtype=torch.float32))

        # Orthogonal basis U = exp(B), B skew-symmetric
        n_basis = K_eff * (K_eff - 1) // 2
        self._basis_params = nn.Parameter(torch.randn(n_basis) * 0.01)

        self.c = nn.Parameter(torch.zeros(K))

        self.mlp = nn.Sequential(
            nn.Linear(d_neural, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, K),
        )

        self._params = ([self._alpha_raw, self._theta,
                         self._basis_params, self.c]
                        + list(self.mlp.parameters()))

    def _get_alpha(self):
        import torch
        return 0.90 + 0.10 * torch.sigmoid(self._alpha_raw)

    def _get_M1_M2(self):
        """Construct M1, M2 from spectral parameterisation.

        For each block j with eigenvalue pair α_j·e^{±iθ_j}:
            m1_jj = 2α_j cos(θ_j)    (coefficient of b_{t-1})
            m2_jj = -α_j²            (coefficient of b_{t-2})
        These are assembled in a diagonal matrix, then rotated by U.
        """
        import torch
        K = self.K_eff
        alpha = self._get_alpha()  # (n_blocks,)
        theta = self._theta        # (n_blocks,)

        # Build diagonal M1d, M2d in block space
        # Each 2x2 block has the same m1, m2 (scalar VAR(2) per block)
        m1_diag = 2.0 * alpha * torch.cos(theta)   # (n_blocks,)
        m2_diag = -(alpha ** 2)                     # (n_blocks,)

        # Expand to K_eff diagonal
        m1_full = torch.zeros(K, device=alpha.device)
        m2_full = torch.zeros(K, device=alpha.device)
        for j in range(self.n_blocks):
            m1_full[2*j]   = m1_diag[j]
            m1_full[2*j+1] = m1_diag[j]
            m2_full[2*j]   = m2_diag[j]
            m2_full[2*j+1] = m2_diag[j]

        M1d = torch.diag(m1_full)
        M2d = torch.diag(m2_full)

        # Orthogonal basis U = exp(B)
        B = torch.zeros(K, K, device=alpha.device)
        idx = torch.triu_indices(K, K, offset=1)
        B[idx[0], idx[1]] = self._basis_params
        B = B - B.T
        U = torch.matrix_exp(B)

        # M1 = U M1d U^T,  M2 = U M2d U^T
        M1 = U @ M1d @ U.T
        M2 = U @ M2d @ U.T

        # If K is odd, the last mode gets identity-like pass-through
        if self.K > self.K_eff:
            M1_full = torch.zeros(self.K, self.K, device=alpha.device)
            M2_full = torch.zeros(self.K, self.K, device=alpha.device)
            M1_full[:K, :K] = M1
            M2_full[:K, :K] = M2
            # Last mode: mild damping, no coupling
            M1_full[-1, -1] = 0.5
            return M1_full, M2_full
        return M1, M2

    def parameters(self):
        return self._params

    def train(self):
        self.mlp.train()

    def eval(self):
        self.mlp.eval()


def train_e2e_oscillator_mlp(b, X_neural, warmup, K,
                             hidden=32, lr=1e-3, epochs=400,
                             tbptt_chunk=64):
    """Train end-to-end oscillator+MLP via truncated BPTT.

    Returns the trained model and (T, K) free-run predictions.
    """
    import torch
    import torch.nn as nn

    T, d = X_neural.shape
    model = E2EOscillatorMLP(d, K, hidden=hidden)

    # Initialize M1 close to identity, M2 close to zero (warm start)
    with torch.no_grad():
        model.M1.weight.copy_(0.8 * torch.eye(K))
        model.M2.weight.zero_()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    b_gt = torch.tensor(b, dtype=torch.float32, device=model.device)
    X_n  = torch.tensor(X_neural, dtype=torch.float32, device=model.device)

    best_loss = float("inf")
    best_state = {}
    patience_ctr = 0
    seq_len = T - warmup

    for ep in range(epochs):
        model.train()

        # Pre-compute MLP drive for all frames
        drive = model.mlp(X_n)  # (T, K)

        total_loss = torch.tensor(0.0, device=model.device)

        # Truncated BPTT: process sequence in chunks
        p_prev1 = b_gt[warmup - 1].detach()
        p_prev2 = b_gt[warmup - 2].detach()

        for chunk_start in range(warmup, T, tbptt_chunk):
            chunk_end = min(chunk_start + tbptt_chunk, T)
            chunk_loss = torch.tensor(0.0, device=model.device)

            for t in range(chunk_start, chunk_end):
                p_t = model.M1(p_prev1) + model.M2(p_prev2) + drive[t] + model.c
                chunk_loss = chunk_loss + nn.functional.mse_loss(p_t, b_gt[t])
                p_prev2 = p_prev1
                p_prev1 = p_t

            # Backprop through this chunk
            chunk_loss = chunk_loss / (chunk_end - chunk_start)
            opt.zero_grad()
            chunk_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss = total_loss + chunk_loss.detach() * (chunk_end - chunk_start)

            # Detach hidden state at chunk boundary (truncated BPTT)
            p_prev1 = p_prev1.detach()
            p_prev2 = p_prev2.detach()

            # Recompute drive after parameter update
            drive = model.mlp(X_n)

        scheduler.step()
        epoch_loss = (total_loss / seq_len).item()

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in
                          zip([n for n, _ in
                               [(f"M1.{n}", p) for n, p in model.M1.named_parameters()] +
                                [(f"M2.{n}", p) for n, p in model.M2.named_parameters()] +
                                [("c", model.c)] +
                                [(f"mlp.{n}", p) for n, p in model.mlp.named_parameters()]],
                              model.parameters())}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr > 40:
                break

    # Restore best weights manually
    if best_state:
        with torch.no_grad():
            params = model.parameters()
            for p, (_, v) in zip(params, best_state.items()):
                p.copy_(v)

    # Final free-run prediction
    model.eval()
    with torch.no_grad():
        preds = model.free_run(b, X_neural, warmup)  # (T-warmup, K)

    b_hat = np.copy(b)
    b_hat[warmup:] = preds.cpu().numpy()

    # Extract learned eigenvalues
    M1_np = model.M1.weight.detach().cpu().numpy()
    M2_np = model.M2.weight.detach().cpu().numpy()
    companion = np.zeros((2*K, 2*K))
    companion[:K, :K] = M1_np
    companion[:K, K:] = M2_np
    companion[K:, :K] = np.eye(K)
    eigs = np.linalg.eigvals(companion)

    return model, b_hat, eigs


# --------------------------------------------------------------------------- #
#  Posture video
# --------------------------------------------------------------------------- #

def make_comparison_video(
    h5_path: str, out_path: str,
    b_gt: np.ndarray,  # (T, K)
    predictions: dict,  # name → (T, K)
    dt: float = 0.6,
    fps: int = 15, dpi: int = 100, max_frames: int = 200,
    body_angle_offset: int = 0,
):
    """4-panel posture video: GT + up to 3 predictions.

    Parameters
    ----------
    body_angle_offset : int
        Frame offset into the full body_angle array that corresponds to
        the start of *b_gt*.  Fixes time-alignment when b_gt comes from
        a test split that doesn't start at frame 0.
    """
    eigvecs = _load_eigenvectors(h5_path=h5_path)

    with h5py.File(h5_path, "r") as f:
        for ba_key in ("behaviour/body_angle_dtarget",
                       "behavior/body_angle_all",
                       "behaviour/body_angle_all"):
            if ba_key in f:
                break
        body_angle_full = np.asarray(f[ba_key][:], dtype=float)

    n_modes = min(b_gt.shape[1], eigvecs.shape[1], 6)
    E = eigvecs[:, :n_modes]
    d_recon = E.shape[0]
    d_seg = body_angle_full.shape[1]

    pred_names = list(predictions.keys())
    n_panels = 1 + len(pred_names)  # GT + predictions

    # Slice body_angle to the same window as b_gt (fix time alignment)
    ba_end = min(body_angle_full.shape[0], body_angle_offset + b_gt.shape[0])
    body_angle = body_angle_full[body_angle_offset:ba_end]

    T = min(body_angle.shape[0], b_gt.shape[0], max_frames)
    for v in predictions.values():
        T = min(T, v.shape[0])

    body_angle = body_angle[:T]
    b_gt_t = b_gt[:T, :n_modes]

    # Reconstruct XY for GT (100-dim eigenvector space)
    recon_gt = b_gt_t @ E.T
    xy_gt = np.zeros((T, d_recon, 2))
    for t in range(T):
        x, y = angles_to_xy(recon_gt[t])
        xy_gt[t, :, 0], xy_gt[t, :, 1] = x, y

    # Also raw body angle for GT panel.
    # Resample from d_seg → d_recon so both overlays have the same
    # number of segments (fixes visual scale mismatch, e.g. 49 vs 100).
    s_raw = np.linspace(0, 1, d_seg)
    s_rec = np.linspace(0, 1, d_recon)
    xy_raw = np.zeros((T, d_recon, 2))
    for t in range(T):
        shape = body_angle[t] - np.mean(body_angle[t])
        if np.all(np.isfinite(shape)):
            shape_resampled = np.interp(s_rec, s_raw, shape)
            x, y = angles_to_xy(shape_resampled)
        else:
            x, y = np.full(d_recon, np.nan), np.full(d_recon, np.nan)
        xy_raw[t, :, 0], xy_raw[t, :, 1] = x, y

    # Reconstruct XY for predictions
    xy_preds = {}
    for name, pred in predictions.items():
        recon = pred[:T, :n_modes] @ E.T
        xy = np.zeros((T, d_recon, 2))
        for t in range(T):
            x, y = angles_to_xy(recon[t])
            xy[t, :, 0], xy[t, :, 1] = x, y
        xy_preds[name] = xy

    # Axis limits
    all_xy = [xy_raw.reshape(-1, 2), xy_gt.reshape(-1, 2)]
    for xy in xy_preds.values():
        all_xy.append(xy.reshape(-1, 2))
    all_pts = np.concatenate(all_xy, axis=0)
    m = np.all(np.isfinite(all_pts), axis=1)
    xmin, xmax = all_pts[m, 0].min(), all_pts[m, 0].max()
    ymin, ymax = all_pts[m, 1].min(), all_pts[m, 1].max()
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    span = 0.5 * max(xmax - xmin, ymax - ymin) + 2.0

    # Figure
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 4.5),
                             facecolor="white")
    if n_panels == 1:
        axes = [axes]

    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]
    lines = []
    heads = []
    gt_overlay_line = None
    gt_overlay_head = None
    titles = ["Ground truth"] + pred_names

    for i, (ax, ttl) in enumerate(zip(axes, titles)):
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f7f7")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for s in ax.spines.values():
            s.set_visible(False)

        if i == 0:
            # GT panel: raw body angle in black, EW reconstruction in orange
            line, = ax.plot([], [], "k-", lw=2.5, alpha=0.9)
            head, = ax.plot([], [], "o", color="crimson", ms=5)
            gt_overlay_line, = ax.plot([], [], "-", color="darkorange", lw=1.8,
                                       alpha=0.7, label=f"EW recon ({n_modes}m)")
            gt_overlay_head, = ax.plot([], [], "o", color="darkorange", ms=4, alpha=0.7)
            ax.legend(loc="upper right", frameon=False, fontsize=8)
        else:
            c = colors[(i - 1) % len(colors)]
            line, = ax.plot([], [], "-", color=c, lw=2.5, alpha=0.9)
            head, = ax.plot([], [], "o", color="crimson", ms=5)
        lines.append(line)
        heads.append(head)

    time_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=11)

    pred_xy_list = [xy_preds[n] for n in pred_names]

    def _update(frame):
        # GT raw
        x, y = xy_raw[frame, :, 0], xy_raw[frame, :, 1]
        lines[0].set_data(x, y)
        heads[0].set_data([x[0]], [y[0]])
        # GT EW overlay
        xo, yo = xy_gt[frame, :, 0], xy_gt[frame, :, 1]
        gt_overlay_line.set_data(xo, yo)
        gt_overlay_head.set_data([xo[0]], [yo[0]])
        # Predictions
        for i, xy in enumerate(pred_xy_list):
            xp, yp = xy[frame, :, 0], xy[frame, :, 1]
            lines[i + 1].set_data(xp, yp)
            heads[i + 1].set_data([xp[0]], [yp[0]])
        time_text.set_text(f"t = {frame * dt:.2f} s   ({frame + 1}/{T})")
        return lines + heads + [gt_overlay_line, gt_overlay_head, time_text]

    anim = FuncAnimation(fig, _update, frames=T, interval=1000 // fps, blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": "AR decoder comparison"},
                          bitrate=2400)
    print(f"  Rendering {T} frames → {out_path}")
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  Video saved: {out_path}")


# --------------------------------------------------------------------------- #
#  Summary figure (bar chart)
# --------------------------------------------------------------------------- #

def make_summary_figure(results: dict, mode_names: list, out_path: str):
    """Grouped bar chart of R² per mode for each strategy."""
    strategies = list(results.keys())
    K = len(mode_names)
    n = len(strategies)

    fig, ax = plt.subplots(figsize=(max(10, K * 1.8), 5.5))
    x = np.arange(K)
    w = 0.8 / n
    colors = plt.cm.Set2(np.linspace(0, 1, max(n, 3)))

    for i, (name, vals) in enumerate(results.items()):
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals[:K], w,
                      label=name, color=colors[i], edgecolor="white", lw=0.5)
        for bar, v in zip(bars, vals[:K]):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(mode_names, fontsize=11)
    ax.set_ylabel("R² (held-out, free-run)", fontsize=12)
    ax.set_title("Behaviour Decoder Comparison  (5-fold temporal CV, free-run)",
                 fontsize=13)
    ax.legend(fontsize=7, ncol=1, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary figure saved: {out_path}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--h5", required=True)
    parser.add_argument("--neural_lags", type=int, default=8)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_modes", type=int, default=6)
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--no_mlp", action="store_true", help="skip MLP (faster)")
    parser.add_argument("--tbptt_chunk", type=int, default=64,
                        help="TBPTT chunk size; larger → longer gradient horizon")
    parser.add_argument("--w_rollout", type=float, default=0.0,
                        help="weight for multi-step rollout loss (0=off)")
    parser.add_argument("--rollout_steps", type=int, default=50,
                        help="rollout horizon for multi-step loss (frames)")
    parser.add_argument("--e2e_epochs", type=int, default=300,
                        help="training epochs for E2E oscillator+MLP")
    parser.add_argument("--w_phase", type=float, default=1.0,
                        help="weight for polar phase loss on a1-a2 (0=pure MSE)")
    parser.add_argument("--all_neurons", action="store_true",
                        help="use ALL neurons instead of just motor+command")
    args = parser.parse_args()

    u_motor, b_full, dt = load_data(args.h5, all_neurons=args.all_neurons)
    K = min(args.n_modes, b_full.shape[1])
    b = b_full[:, :K]
    T = b.shape[0]

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_lags = args.neural_lags
    n_folds = args.n_folds
    warmup = max(2, n_lags)  # max AR lag is 2

    print(f"\n  Config: neural_lags={n_lags}, K={K}, n_folds={n_folds}, "
          f"tbptt_chunk={args.tbptt_chunk}, w_rollout={args.w_rollout}, "
          f"rollout_steps={args.rollout_steps}, e2e_epochs={args.e2e_epochs}, "
          f"w_phase={args.w_phase}")
    print()

    X_neural = build_lagged_features_np(u_motor, n_lags)

    # ================================================================ #
    #  5-fold temporal CV  (contiguous held-out blocks)
    #
    #  For every model the held-out block is predicted WITHOUT any
    #  ground-truth posture:
    #   • Ridge / MLP:  predict each frame independently from n_t
    #   • Oscillator:   seed with 2 GT frames before the block,
    #                   then free-run through the block
    #   • E2E:          same seeding, free-run
    # ================================================================ #
    valid_len = T - warmup
    fold_size = valid_len // n_folds
    folds = []
    for i in range(n_folds):
        start = warmup + i * fold_size
        end = warmup + (i + 1) * fold_size if i < n_folds - 1 else T
        folds.append((start, end))

    # Model names (formula labels)
    MODEL_RIDGE   = "b̂=Wn+c"
    MODEL_MLP     = "b̂=MLP(n)"
    MODEL_OSC_R   = "b̂=M₁b̂₋₁+M₂b̂₋₂+Wn+c  (2-step)"
    MODEL_OSC_M   = "b̂=M₁b̂₋₁+M₂b̂₋₂+MLP(n)+c  (2-step)"
    MODEL_E2E     = "b̂=M₁b̂₋₁+M₂b̂₋₂+MLP(n)+c  (E2E)"
    MODEL_E2E_UD  = "b̂=M₁b̂₋₁+M₂b̂₋₂+MLP(n)+c  (spectral)"
    MODEL_E2E_POL = "b̂=M₁b̂₋₁+M₂b̂₋₂+MLP(n)+c  (polar)"

    # Accumulators: held-out predictions, shape (T, K), NaN outside test
    ho_preds = {}
    for m in [MODEL_RIDGE, MODEL_MLP, MODEL_OSC_R, MODEL_OSC_M, MODEL_E2E,
              MODEL_E2E_UD, MODEL_E2E_POL]:
        ho_preds[m] = np.full((T, K), np.nan)

    print("  ── 5-fold temporal CV (all models, free-run) ──\n")

    for fold_i, (ts, te) in enumerate(folds):
        # Train indices: all valid frames outside this fold
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[ts:te] = False
        train_idx = np.where(train_mask)[0]
        test_len = te - ts

        print(f"  Fold {fold_i+1}/{n_folds}:  test=[{ts}:{te})  "
              f"({test_len} frames, {test_len*dt:.0f}s)")

        # ── 1) Ridge linear: b̂ = W·n + c ──
        for j in range(K):
            fit = _ridge_cv_single_target(
                X_neural, b[:, j], train_idx, ridge_grid, n_folds)
            ho_preds[MODEL_RIDGE][ts:te, j] = (
                X_neural[ts:te] @ fit["coef"] + fit["intercept"])

        # ── 2) MLP: b̂ = MLP(n) ──
        if not args.no_mlp:
            import torch, torch.nn as nn
            d_in = X_neural.shape[1]
            X_tr = torch.tensor(X_neural[train_idx], dtype=torch.float32)
            y_tr = torch.tensor(b[train_idx], dtype=torch.float32)
            mlp = nn.Sequential(
                nn.Linear(d_in, 32), nn.LayerNorm(32), nn.ReLU(),
                nn.Dropout(0.1), nn.Linear(32, K))
            opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
            best_l, best_s, pat = float("inf"), None, 0
            for ep in range(200):
                mlp.train()
                loss = nn.functional.mse_loss(mlp(X_tr), y_tr)
                opt.zero_grad(); loss.backward(); opt.step()
                if loss.item() < best_l - 1e-6:
                    best_l = loss.item()
                    best_s = {k: v.clone() for k, v in mlp.state_dict().items()}
                    pat = 0
                else:
                    pat += 1
                    if pat > 20: break
            if best_s: mlp.load_state_dict(best_s)
            mlp.eval()
            with torch.no_grad():
                X_te = torch.tensor(X_neural[ts:te], dtype=torch.float32)
                ho_preds[MODEL_MLP][ts:te] = mlp(X_te).numpy()

        # ── 3) Osc + Ridge (2-step): M₁,M₂ on train b, ridge on residual ──
        M1f, M2f, cf, res_f = fit_var2_matrices(
            b, ridge_grid, n_folds, warmup=0,  # use train_idx for masking below
        )
        # Re-fit VAR(2) using only train indices
        X_ar_full = build_ar_features(b, 2)  # (T, 2K)
        M1_fold = np.zeros((K, K)); M2_fold = np.zeros((K, K)); c_fold = np.zeros(K)
        for j in range(K):
            fit = _ridge_cv_single_target(
                X_ar_full, b[:, j], train_idx, ridge_grid, n_folds)
            M1_fold[j] = fit["coef"][:K]
            M2_fold[j] = fit["coef"][K:2*K]
            c_fold[j] = fit["intercept"]
        # Residual on train
        b_tm1 = np.zeros_like(b); b_tm1[1:] = b[:-1]
        b_tm2 = np.zeros_like(b); b_tm2[2:] = b[:-2]
        residual_fold = b - (b_tm1 @ M1_fold.T + b_tm2 @ M2_fold.T + c_fold)
        # Ridge drive on residual (train only)
        coefs_dr = np.zeros((X_neural.shape[1], K))
        ints_dr = np.zeros(K)
        for j in range(K):
            fit = _ridge_cv_single_target(
                X_neural, residual_fold[:, j], train_idx, ridge_grid, n_folds)
            coefs_dr[:, j] = fit["coef"]
            ints_dr[j] = fit["intercept"]
        # Free-run on test block
        drive_all = X_neural @ coefs_dr + ints_dr[None, :]
        p1 = b[ts - 1].copy(); p2 = b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1_fold @ p1 + M2_fold @ p2 + drive_all[t] + c_fold
            ho_preds[MODEL_OSC_R][t] = p_new
            p2 = p1; p1 = p_new

        # ── 4) Osc + MLP (2-step): same VAR(2), MLP on residual ──
        if not args.no_mlp:
            r_tr = torch.tensor(residual_fold[train_idx], dtype=torch.float32)
            mlp_r = nn.Sequential(
                nn.Linear(d_in, 32), nn.LayerNorm(32), nn.ReLU(),
                nn.Dropout(0.1), nn.Linear(32, K))
            opt_r = torch.optim.Adam(mlp_r.parameters(), lr=1e-3)
            best_l, best_s, pat = float("inf"), None, 0
            for ep in range(200):
                mlp_r.train()
                loss = nn.functional.mse_loss(mlp_r(X_tr), r_tr)
                opt_r.zero_grad(); loss.backward(); opt_r.step()
                if loss.item() < best_l - 1e-6:
                    best_l = loss.item()
                    best_s = {k: v.clone() for k, v in mlp_r.state_dict().items()}
                    pat = 0
                else:
                    pat += 1
                    if pat > 20: break
            if best_s: mlp_r.load_state_dict(best_s)
            mlp_r.eval()
            with torch.no_grad():
                X_full_t = torch.tensor(X_neural, dtype=torch.float32)
                drive_mlp = mlp_r(X_full_t).numpy()
            p1 = b[ts - 1].copy(); p2 = b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1_fold @ p1 + M2_fold @ p2 + drive_mlp[t] + c_fold
                ho_preds[MODEL_OSC_M][t] = p_new
                p2 = p1; p1 = p_new

        # ── 5) E2E Osc+MLP (BPTT): train on train segments, test free-run ──
        if not args.no_mlp:
            import torch, torch.nn as nn
            # Build contiguous training segments (skip test fold)
            # Segment 1: [warmup, ts),  Segment 2: [te, T)
            segs = []
            if ts > warmup + 2:
                segs.append((warmup, ts))
            if te + 2 < T:
                segs.append((te, T))

            d_in = X_neural.shape[1]
            e2e = E2EOscillatorMLP(d_in, K, hidden=32)
            with torch.no_grad():
                e2e.M1.weight.copy_(0.8 * torch.eye(K))
                e2e.M2.weight.zero_()

            opt_e = torch.optim.Adam(e2e.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_e, T_max=args.e2e_epochs)
            b_gt_t = torch.tensor(b, dtype=torch.float32)
            X_n_t  = torch.tensor(X_neural, dtype=torch.float32)
            best_l, best_s, pat = float("inf"), None, 0
            chunk = args.tbptt_chunk
            w_roll = args.w_rollout
            roll_H = args.rollout_steps

            for ep in range(args.e2e_epochs):
                e2e.train()
                ep_loss = 0.0
                ep_count = 0

                # ── TBPTT pass over training segments ──
                for seg_s, seg_e in segs:
                    drive = e2e.mlp(X_n_t)
                    p1 = b_gt_t[seg_s - 1].detach()
                    p2 = b_gt_t[seg_s - 2].detach()
                    for cs in range(seg_s, seg_e, chunk):
                        ce = min(cs + chunk, seg_e)
                        closs = torch.tensor(0.0)
                        for t in range(cs, ce):
                            pt = e2e.M1(p1) + e2e.M2(p2) + drive[t] + e2e.c
                            closs = closs + nn.functional.mse_loss(pt, b_gt_t[t])
                            p2 = p1; p1 = pt
                        closs = closs / (ce - cs)
                        opt_e.zero_grad(); closs.backward()
                        torch.nn.utils.clip_grad_norm_(e2e.parameters(), 1.0)
                        opt_e.step()
                        ep_loss += closs.item() * (ce - cs)
                        ep_count += (ce - cs)
                        p1 = p1.detach(); p2 = p2.detach()
                        drive = e2e.mlp(X_n_t)

                # ── Multi-step rollout loss (long-horizon drift penalty) ──
                #  Sample random starting points in training data, free-run
                #  for roll_H frames, and penalise the accumulated MSE with
                #  weight w_roll.  This creates strong gradients against the
                #  slow phase drift that per-frame MSE misses.
                if w_roll > 0 and len(segs) > 0:
                    for _ in range(4):  # 4 random rollouts per epoch
                        si = np.random.randint(len(segs))
                        s0, s1 = segs[si]
                        max_st = s1 - roll_H - 2
                        if max_st <= s0:
                            continue
                        rs = np.random.randint(s0, max_st)
                        re = min(rs + roll_H, s1)
                        drive_r = e2e.mlp(X_n_t)
                        p1r = b_gt_t[rs - 1].detach()
                        p2r = b_gt_t[rs - 2].detach()
                        rloss = torch.tensor(0.0)
                        for t in range(rs, re):
                            pt = (e2e.M1(p1r) + e2e.M2(p2r)
                                  + drive_r[t] + e2e.c)
                            rloss = rloss + nn.functional.mse_loss(
                                pt, b_gt_t[t])
                            p2r = p1r; p1r = pt
                        rloss = w_roll * rloss / (re - rs)
                        opt_e.zero_grad()
                        rloss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            e2e.parameters(), 1.0)
                        opt_e.step()
                        ep_loss += rloss.item() * (re - rs)
                        ep_count += (re - rs)

                sched.step()
                if ep_count > 0:
                    avg = ep_loss / ep_count
                    if avg < best_l - 1e-6:
                        best_l = avg
                        best_s = {k: v.clone()
                                  for k, v in e2e.mlp.state_dict().items()}
                        best_M1 = e2e.M1.weight.detach().clone()
                        best_M2 = e2e.M2.weight.detach().clone()
                        best_c  = e2e.c.detach().clone()
                        pat = 0
                    else:
                        pat += 1
                        if pat > 40: break

            # Restore best
            if best_s is not None:
                e2e.mlp.load_state_dict(best_s)
                with torch.no_grad():
                    e2e.M1.weight.copy_(best_M1)
                    e2e.M2.weight.copy_(best_M2)
                    e2e.c.copy_(best_c)

            # Free-run on test fold
            e2e.eval()
            with torch.no_grad():
                drive = e2e.mlp(X_n_t).numpy()
                M1e = e2e.M1.weight.numpy()
                M2e = e2e.M2.weight.numpy()
                ce  = e2e.c.numpy()
            p1 = b[ts - 1].copy(); p2 = b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1e @ p1 + M2e @ p2 + drive[t] + ce
                ho_preds[MODEL_E2E][t] = p_new
                p2 = p1; p1 = p_new

        # ── 6) E2E Spectral (VAR(2), controlled companion eigenvalues) ──
        if not args.no_mlp:
            import torch, torch.nn as nn
            segs_u = []
            if ts > warmup + 2:
                segs_u.append((warmup, ts))
            if te + 2 < T:
                segs_u.append((te, T))

            d_in = X_neural.shape[1]
            ud = E2EUndampedMLP(d_in, K, hidden=32, alpha_init=0.99)
            opt_u = torch.optim.Adam(ud.parameters(), lr=1e-3)
            sched_u = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_u, T_max=args.e2e_epochs)
            b_gt_t = torch.tensor(b, dtype=torch.float32)
            X_n_t  = torch.tensor(X_neural, dtype=torch.float32)
            best_l, best_s_u, pat = float("inf"), None, 0
            chunk = args.tbptt_chunk

            for ep in range(args.e2e_epochs):
                ud.train()
                M1_u, M2_u = ud._get_M1_M2()  # (K, K) each
                drive = ud.mlp(X_n_t)
                ep_loss = 0.0; ep_count = 0

                for seg_s, seg_e in segs_u:
                    p1 = b_gt_t[seg_s - 1].detach()
                    p2 = b_gt_t[seg_s - 2].detach()
                    for cs in range(seg_s, seg_e, chunk):
                        ce_ = min(cs + chunk, seg_e)
                        closs = torch.tensor(0.0)
                        for t in range(cs, ce_):
                            pt = M1_u @ p1 + M2_u @ p2 + drive[t] + ud.c
                            closs = closs + nn.functional.mse_loss(
                                pt, b_gt_t[t])
                            p2 = p1; p1 = pt
                        closs = closs / (ce_ - cs)
                        opt_u.zero_grad(); closs.backward()
                        torch.nn.utils.clip_grad_norm_(ud.parameters(), 1.0)
                        opt_u.step()
                        ep_loss += closs.item() * (ce_ - cs)
                        ep_count += (ce_ - cs)
                        p1 = p1.detach(); p2 = p2.detach()
                        M1_u, M2_u = ud._get_M1_M2()
                        drive = ud.mlp(X_n_t)

                sched_u.step()
                if ep_count > 0:
                    avg = ep_loss / ep_count
                    if avg < best_l - 1e-6:
                        best_l = avg
                        best_s_u = {k: v.clone()
                                    for k, v in ud.mlp.state_dict().items()}
                        best_alpha = ud._alpha_raw.detach().clone()
                        best_theta = ud._theta.detach().clone()
                        best_basis = ud._basis_params.detach().clone()
                        best_cu    = ud.c.detach().clone()
                        pat = 0
                    else:
                        pat += 1
                        if pat > 40: break

            if best_s_u is not None:
                ud.mlp.load_state_dict(best_s_u)
                with torch.no_grad():
                    ud._alpha_raw.copy_(best_alpha)
                    ud._theta.copy_(best_theta)
                    ud._basis_params.copy_(best_basis)
                    ud.c.copy_(best_cu)

            # Free-run on test fold (VAR(2))
            ud.eval()
            with torch.no_grad():
                M1_np, M2_np = [m.numpy() for m in ud._get_M1_M2()]
                drive_np = ud.mlp(X_n_t).numpy()
                cu_np = ud.c.numpy()
            p1 = b[ts - 1].copy(); p2 = b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1_np @ p1 + M2_np @ p2 + drive_np[t] + cu_np
                ho_preds[MODEL_E2E_UD][t] = p_new
                p2 = p1; p1 = p_new

        # ── 7) E2E + polar phase loss on a1-a2 ──
        if not args.no_mlp:
            import torch, torch.nn as nn
            segs_p = []
            if ts > warmup + 2:
                segs_p.append((warmup, ts))
            if te + 2 < T:
                segs_p.append((te, T))

            d_in = X_neural.shape[1]
            e2e_p = E2EOscillatorMLP(d_in, K, hidden=32)
            with torch.no_grad():
                e2e_p.M1.weight.copy_(0.8 * torch.eye(K))
                e2e_p.M2.weight.zero_()

            opt_p = torch.optim.Adam(e2e_p.parameters(), lr=1e-3)
            sched_p = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_p, T_max=args.e2e_epochs)
            b_gt_t = torch.tensor(b, dtype=torch.float32)
            X_n_t  = torch.tensor(X_neural, dtype=torch.float32)
            best_l, best_s_p, pat = float("inf"), None, 0
            chunk = args.tbptt_chunk
            w_roll = args.w_rollout
            roll_H = args.rollout_steps
            w_ph = args.w_phase

            for ep in range(args.e2e_epochs):
                e2e_p.train()
                ep_loss = 0.0; ep_count = 0

                for seg_s, seg_e in segs_p:
                    drive = e2e_p.mlp(X_n_t)
                    p1 = b_gt_t[seg_s - 1].detach()
                    p2 = b_gt_t[seg_s - 2].detach()
                    for cs in range(seg_s, seg_e, chunk):
                        ce_ = min(cs + chunk, seg_e)
                        closs = torch.tensor(0.0)
                        for t in range(cs, ce_):
                            pt = (e2e_p.M1(p1) + e2e_p.M2(p2)
                                  + drive[t] + e2e_p.c)
                            closs = closs + _polar_loss_frame(
                                pt, b_gt_t[t], w_phase=w_ph)
                            p2 = p1; p1 = pt
                        closs = closs / (ce_ - cs)
                        opt_p.zero_grad(); closs.backward()
                        torch.nn.utils.clip_grad_norm_(
                            e2e_p.parameters(), 1.0)
                        opt_p.step()
                        ep_loss += closs.item() * (ce_ - cs)
                        ep_count += (ce_ - cs)
                        p1 = p1.detach(); p2 = p2.detach()
                        drive = e2e_p.mlp(X_n_t)

                # Polar-aware rollout loss
                if w_roll > 0 and len(segs_p) > 0:
                    for _ in range(4):
                        si = np.random.randint(len(segs_p))
                        s0, s1 = segs_p[si]
                        max_st = s1 - roll_H - 2
                        if max_st <= s0:
                            continue
                        rs = np.random.randint(s0, max_st)
                        re = min(rs + roll_H, s1)
                        drive_r = e2e_p.mlp(X_n_t)
                        p1r = b_gt_t[rs - 1].detach()
                        p2r = b_gt_t[rs - 2].detach()
                        rloss = torch.tensor(0.0)
                        for t in range(rs, re):
                            pt = (e2e_p.M1(p1r) + e2e_p.M2(p2r)
                                  + drive_r[t] + e2e_p.c)
                            rloss = rloss + _polar_loss_frame(
                                pt, b_gt_t[t], w_phase=w_ph)
                            p2r = p1r; p1r = pt
                        rloss = w_roll * rloss / (re - rs)
                        opt_p.zero_grad()
                        rloss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            e2e_p.parameters(), 1.0)
                        opt_p.step()
                        ep_loss += rloss.item() * (re - rs)
                        ep_count += (re - rs)

                sched_p.step()
                if ep_count > 0:
                    avg = ep_loss / ep_count
                    if avg < best_l - 1e-6:
                        best_l = avg
                        best_s_p = {k: v.clone()
                                    for k, v in e2e_p.mlp.state_dict().items()}
                        best_M1p = e2e_p.M1.weight.detach().clone()
                        best_M2p = e2e_p.M2.weight.detach().clone()
                        best_cp  = e2e_p.c.detach().clone()
                        pat = 0
                    else:
                        pat += 1
                        if pat > 40: break

            if best_s_p is not None:
                e2e_p.mlp.load_state_dict(best_s_p)
                with torch.no_grad():
                    e2e_p.M1.weight.copy_(best_M1p)
                    e2e_p.M2.weight.copy_(best_M2p)
                    e2e_p.c.copy_(best_cp)

            # Free-run on test fold
            e2e_p.eval()
            with torch.no_grad():
                drive = e2e_p.mlp(X_n_t).numpy()
                M1p = e2e_p.M1.weight.numpy()
                M2p = e2e_p.M2.weight.numpy()
                cp  = e2e_p.c.numpy()
            p1 = b[ts - 1].copy(); p2 = b[ts - 2].copy()
            for t in range(ts, te):
                p_new = M1p @ p1 + M2p @ p2 + drive[t] + cp
                ho_preds[MODEL_E2E_POL][t] = p_new
                p2 = p1; p1 = p_new

    # ================================================================ #
    #  Compute R² from held-out predictions
    # ================================================================ #
    results = {}
    best_preds = {}
    valid = np.arange(warmup, T)

    all_models = [MODEL_RIDGE]
    if not args.no_mlp:
        all_models += [MODEL_MLP, MODEL_OSC_M, MODEL_E2E,
                       MODEL_E2E_UD, MODEL_E2E_POL]
    all_models.insert(1 if not args.no_mlp else 1, MODEL_OSC_R)
    # Deduplicate while preserving order
    seen = set()
    ordered_models = []
    for m in all_models:
        if m not in seen:
            seen.add(m)
            ordered_models.append(m)
    all_models = ordered_models

    for name in all_models:
        preds = ho_preds[name]
        ok = np.isfinite(preds[valid, 0])
        if ok.sum() < 10:
            results[name] = np.full(K, np.nan)
            continue
        idx = valid[ok]
        r2 = np.array([r2_score(b[idx, j], preds[idx, j]) for j in range(K)])
        results[name] = r2
        # Build full prediction array for video
        full = np.copy(b)
        full[idx] = preds[idx]
        best_preds[name] = full

    # ================================================================ #
    #  Eigenvalue analysis (full-data fit for reporting)
    # ================================================================ #
    X_ar_full = build_ar_features(b, 2)
    mask_all = np.ones(T, dtype=bool); mask_all[:warmup] = False
    M1_all = np.zeros((K, K)); M2_all = np.zeros((K, K))
    for j in range(K):
        fit = _ridge_cv_single_target(
            X_ar_full, b[:, j], np.where(mask_all)[0], ridge_grid, n_folds)
        M1_all[j] = fit["coef"][:K]
        M2_all[j] = fit["coef"][K:2*K]
    companion = np.zeros((2*K, 2*K))
    companion[:K, :K] = M1_all
    companion[:K, K:] = M2_all
    companion[K:, :K] = np.eye(K)
    eigs_comp = np.linalg.eigvals(companion)

    # ================================================================ #
    #  Report
    # ================================================================ #
    mode_names = [f"a{j+1}" for j in range(K)]

    print()
    print("=" * 100)
    print("  DECODER COMPARISON  (5-fold temporal CV, free-run R²)")
    print("=" * 100)
    col_w = 55
    header = f"  {'Model (formula)':<{col_w}s}" + "".join(f"{m:>8s}" for m in mode_names)
    print(header)
    print("  " + "-" * (col_w + 8 * K))

    for name in all_models:
        vals = results.get(name, np.full(K, np.nan))
        parts = [f"  {name:<{col_w}s}"]
        for v in vals[:K]:
            parts.append(f"{v:8.3f}" if np.isfinite(v) else f"{'---':>8s}")
        print("".join(parts))

    print()
    print("  VAR(2) companion eigenvalues (full-data fit):")
    top_eigs = sorted(eigs_comp, key=lambda x: -abs(x))[:6]
    for ev in top_eigs:
        mod = abs(ev)
        if abs(ev.imag) > 1e-6:
            period = 2 * np.pi / abs(np.angle(ev)) * dt
            hl = -np.log(2) / np.log(max(mod, 1e-10)) * dt if mod < 1 else float('inf')
            print(f"    λ={ev:.4f}  |λ|={mod:.3f}  period={period:.1f}s  hl={hl:.1f}s")
        else:
            hl = (-np.log(2) / np.log(max(mod, 1e-10)) * dt
                  if 0 < mod < 1 else float('inf'))
            print(f"    λ={ev.real:.4f}  |λ|={mod:.3f}  hl={hl:.1f}s")
    print()

    # Print undamped model learned eigenvalues (if available)
    if MODEL_E2E_UD in results and not np.all(np.isnan(results[MODEL_E2E_UD])):
        try:
            import torch
            alpha_vals = ud._get_alpha().detach().numpy()
            theta_vals = ud._theta.detach().numpy()
            print("  Spectral model learned modes (last fold):")
            for j in range(ud.n_blocks):
                a = alpha_vals[j]
                th = theta_vals[j]
                period = 2 * np.pi / max(abs(th), 1e-8) * dt
                hl = -np.log(2) / np.log(max(a, 1e-10)) * dt
                print(f"    Block {j+1}:  α={a:.4f}  θ={th:.3f}  "
                      f"period={period:.1f}s  hl={hl:.1f}s")
            print()
        except Exception:
            pass

    # ================================================================ #
    #  Save outputs
    # ================================================================ #
    out_dir = Path("output_plots/eigenworms")
    out_dir.mkdir(parents=True, exist_ok=True)

    make_summary_figure(results, mode_names,
                        str(out_dir / "ar_decoder_comparison.png"))

    with open(out_dir / "ar_decoder_comparison.txt", "w") as f:
        f.write("Decoder Comparison: 5-fold temporal CV, free-run R²\n")
        f.write(f"h5: {args.h5}\n")
        f.write(f"neural_lags={n_lags}, K={K}, n_folds={n_folds}\n\n")
        for name in all_models:
            vals = results.get(name, np.full(K, np.nan))
            f.write(f"{name:55s}: " + " ".join(f"{v:.3f}" for v in vals) + "\n")

    if not args.no_video and best_preds:
        print("\n  ── Generating posture comparison video ──")
        make_comparison_video(
            args.h5,
            str(out_dir / "ar_decoder_posture.mp4"),
            b_gt=b,
            predictions=best_preds,
            dt=dt,
            fps=15,
            dpi=100,
            max_frames=args.max_frames,
        )

    print("\n  All done.")


if __name__ == "__main__":
    main()
