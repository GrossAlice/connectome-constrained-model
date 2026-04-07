#!/usr/bin/env python
"""Ridge regularization sweep for LOO stability.

The vanilla Ridge has great one-step R² (0.66) but diverges in LOO (−0.25)
because the weight matrix W has spectral radius > 1.  This script explores
7 regularization strategies to stabilize Ridge for free-running LOO:

  1. Ridge-HiAlpha    — much higher alpha than CV selects (sacrifices 1step for stability)
  2. Ridge-Spectral   — fit Ridge, clip spectral radius of W to ρ<1
  3. ElasticNet       — L1+L2 promotes sparsity in W
  4. Ridge-PCR        — reduce input dim via PCA first (fewer unstable modes)
  5. Ridge-Noise      — augment training with input noise (robustness)
  6. Ridge-Rollout    — train on multi-step rollout loss (penalises divergence directly)
  7. Ridge-DiagBias   — heavier penalty on off-diagonal W (encourage self-prediction)

Usage
-----
    python -m scripts.ridge_regularization_sweep \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
        --save_dir output_plots/stage2/ridge_reg_sweep \
        --stage2_npz output_plots/stage2/default_config_run_v2/cv_onestep.npz
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.decomposition import PCA


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m].astype(np.float64), y_pred[m].astype(np.float64)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot < 1e-15:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _spectral_radius(W: np.ndarray) -> float:
    """Spectral radius of matrix W."""
    return float(np.max(np.abs(np.linalg.eigvals(W))))


# ═══════════════════════════════════════════════════════════════════════
#  Linear model wrapper — everything returns (W, b) where
#      prediction = X @ W.T + b      (shapes: X=(T,N), W=(N,N), b=(N,))
#  This lets us share the LOO eval code across all variants.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LinearModel:
    """Thin wrapper so all variants share the same predict interface."""
    W: np.ndarray          # (N_out, N_in)  — sklearn convention
    b: np.ndarray          # (N_out,)
    pca: Optional[PCA] = None     # for PCR variants
    name: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (batch, N_in) → (batch, N_out)."""
        if self.pca is not None:
            X = self.pca.transform(X)
        return X @ self.W.T + self.b


# ═══════════════════════════════════════════════════════════════════════
#  Strategy 1: Ridge with high fixed alpha
# ═══════════════════════════════════════════════════════════════════════

def train_ridge_high_alpha(
    X: np.ndarray, Y: np.ndarray, alpha: float,
) -> LinearModel:
    m = Ridge(alpha=alpha).fit(X, Y)
    rho = _spectral_radius(m.coef_)
    print(f"    alpha={alpha:.1e}  ρ(W)={rho:.4f}")
    return LinearModel(W=m.coef_, b=m.intercept_, name=f"Ridge-α{alpha:.0e}",
                       meta={"alpha": alpha, "spectral_radius": rho})


# ═══════════════════════════════════════════════════════════════════════
#  Strategy 2: Ridge + spectral clipping
# ═══════════════════════════════════════════════════════════════════════

def _clip_spectral_radius(W: np.ndarray, rho_max: float) -> np.ndarray:
    """Project W so its spectral radius ≤ rho_max."""
    eigvals, P = np.linalg.eig(W)
    rho = np.max(np.abs(eigvals))
    if rho <= rho_max:
        return W
    # Scale eigenvalues: keep phase, clip magnitude
    scale = np.minimum(np.abs(eigvals), rho_max) / (np.abs(eigvals) + 1e-15)
    eigvals_clipped = eigvals * scale
    W_clipped = np.real(P @ np.diag(eigvals_clipped) @ np.linalg.inv(P))
    return W_clipped.astype(W.dtype)


def train_ridge_spectral(
    X: np.ndarray, Y: np.ndarray, rho_max: float, base_alpha: float = 1.0,
) -> LinearModel:
    """Fit Ridge, then clip W's spectral radius to ρ_max."""
    m = Ridge(alpha=base_alpha).fit(X, Y)
    rho_before = _spectral_radius(m.coef_)
    W_clipped = _clip_spectral_radius(m.coef_, rho_max)
    rho_after = _spectral_radius(W_clipped)
    # Refit intercept after clipping W
    residual = Y - X @ W_clipped.T
    b = residual.mean(axis=0)
    print(f"    ρ_max={rho_max}  ρ_before={rho_before:.4f}  ρ_after={rho_after:.4f}")
    return LinearModel(W=W_clipped, b=b, name=f"Ridge-Spec(ρ={rho_max})",
                       meta={"rho_max": rho_max, "rho_before": rho_before,
                              "rho_after": rho_after})


# ═══════════════════════════════════════════════════════════════════════
#  Strategy 3: Elastic Net (L1 + L2)
# ═══════════════════════════════════════════════════════════════════════

def train_elastic_net(
    X: np.ndarray, Y: np.ndarray, alpha: float = 0.01, l1_ratio: float = 0.5,
    max_iter: int = 5000,
) -> LinearModel:
    """Multi-output Elastic Net: fit per-neuron (sklearn limitation)."""
    N = Y.shape[1]
    W = np.zeros((N, X.shape[1]), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    for j in range(N):
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter,
                        warm_start=False)
        en.fit(X, Y[:, j])
        W[j] = en.coef_
        b[j] = en.intercept_
    nnz = np.count_nonzero(W)
    total = W.size
    rho = _spectral_radius(W)
    print(f"    alpha={alpha}  l1_ratio={l1_ratio}  "
          f"sparsity={1-nnz/total:.1%}  ρ(W)={rho:.4f}")
    return LinearModel(W=W, b=b,
                       name=f"ElasticNet(α={alpha},l1={l1_ratio})",
                       meta={"alpha": alpha, "l1_ratio": l1_ratio,
                              "sparsity": 1 - nnz / total,
                              "spectral_radius": rho})


# ═══════════════════════════════════════════════════════════════════════
#  Strategy 4: Reduced-rank regression (PCA + Ridge)
# ═══════════════════════════════════════════════════════════════════════

def train_ridge_pcr(
    X: np.ndarray, Y: np.ndarray, n_components: int, alpha: float = 1.0,
) -> LinearModel:
    """PCA on inputs → Ridge in reduced space → predict all N outputs."""
    pca = PCA(n_components=n_components).fit(X)
    X_pc = pca.transform(X)
    m = Ridge(alpha=alpha).fit(X_pc, Y)
    var_explained = pca.explained_variance_ratio_.sum()
    rho = _spectral_radius(m.coef_ @ pca.components_)  # effective W in original space
    print(f"    k={n_components}  var_expl={var_explained:.3f}  ρ(eff)={rho:.4f}")
    return LinearModel(W=m.coef_, b=m.intercept_, pca=pca,
                       name=f"Ridge-PCR(k={n_components})",
                       meta={"n_components": n_components,
                              "var_explained": var_explained,
                              "spectral_radius": rho})


# ═══════════════════════════════════════════════════════════════════════
#  Strategy 5: Input noise augmentation
# ═══════════════════════════════════════════════════════════════════════

def train_ridge_noise(
    X: np.ndarray, Y: np.ndarray, sigma: float, n_aug: int = 10,
    alpha: float = 1.0,
) -> LinearModel:
    """Augment training set with Gaussian noise on inputs."""
    rng = np.random.default_rng(42)
    X_aug = [X]
    Y_aug = [Y]
    for _ in range(n_aug):
        noise = rng.normal(0, sigma, size=X.shape).astype(X.dtype)
        X_aug.append(X + noise)
        Y_aug.append(Y)  # targets stay clean
    X_all = np.concatenate(X_aug, axis=0)
    Y_all = np.concatenate(Y_aug, axis=0)
    m = Ridge(alpha=alpha).fit(X_all, Y_all)
    rho = _spectral_radius(m.coef_)
    print(f"    σ={sigma}  n_aug={n_aug}  ρ(W)={rho:.4f}")
    return LinearModel(W=m.coef_, b=m.intercept_,
                       name=f"Ridge-Noise(σ={sigma})",
                       meta={"sigma": sigma, "n_aug": n_aug,
                              "spectral_radius": rho})


# ═══════════════════════════════════════════════════════════════════════
#  Strategy 6: Multi-step rollout training (iterative)
# ═══════════════════════════════════════════════════════════════════════

def train_ridge_rollout(
    X: np.ndarray, Y: np.ndarray, u_train: np.ndarray,
    rollout_steps: int = 5, alpha: float = 1.0,
    n_iter: int = 50, lr: float = 1e-4,
) -> LinearModel:
    """Initialize with Ridge, then fine-tune W, b via gradient descent
    on multi-step rollout MSE."""
    # Initialize with Ridge
    m = Ridge(alpha=alpha).fit(X, Y)
    W = m.coef_.copy().astype(np.float64)      # (N, N)
    b = m.intercept_.copy().astype(np.float64)  # (N,)
    N = W.shape[0]
    T = u_train.shape[0]
    rho_init = _spectral_radius(W)

    best_loss = float("inf")
    best_W, best_b = W.copy(), b.copy()

    for it in range(n_iter):
        # Accumulate gradient over random starting points
        grad_W = np.zeros_like(W)
        grad_b = np.zeros_like(b)
        total_loss = 0.0
        n_starts = min(50, T - rollout_steps - 1)
        rng = np.random.default_rng(it)
        starts = rng.choice(T - rollout_steps - 1, size=n_starts, replace=False)

        for t0 in starts:
            # Forward rollout
            states = [u_train[t0].copy()]  # list of (N,) arrays
            for k in range(rollout_steps):
                x_next = states[-1] @ W.T + b
                states.append(x_next)

            # Loss: sum of MSE at each step
            loss = 0.0
            for k in range(1, rollout_steps + 1):
                diff = states[k] - u_train[t0 + k]
                loss += np.sum(diff ** 2)
            total_loss += loss

            # Backprop through the linear rollout (manual)
            # d_loss/d_state[K] = 2*(state[K] - gt[K])
            # state[k] = state[k-1] @ W.T + b
            # d_loss/d_W += d_loss/d_state[k] outer state[k-1]
            # d_loss/d_b += d_loss/d_state[k]
            # Chain through steps:
            d_state = np.zeros(N)
            for k in range(rollout_steps, 0, -1):
                d_step = 2.0 * (states[k] - u_train[t0 + k]) / (N * n_starts)
                d_state_k = d_step + d_state
                grad_W += np.outer(d_state_k, states[k - 1])
                grad_b += d_state_k
                # Propagate: d_state[k-1] += d_state_k @ W
                d_state = d_state_k @ W

        # L2 penalty gradient
        grad_W += 2 * alpha * W / (N * N)

        # Update
        W -= lr * grad_W
        b -= lr * grad_b

        total_loss /= (n_starts * rollout_steps * N)
        if total_loss < best_loss:
            best_loss = total_loss
            best_W, best_b = W.copy(), b.copy()

    rho_final = _spectral_radius(best_W)
    print(f"    K={rollout_steps}  ρ_init={rho_init:.4f}  ρ_final={rho_final:.4f}  "
          f"rollout_loss={best_loss:.6f}")
    return LinearModel(W=best_W, b=best_b,
                       name=f"Ridge-Rollout(K={rollout_steps})",
                       meta={"rollout_steps": rollout_steps,
                              "rho_init": rho_init, "rho_final": rho_final,
                              "rollout_loss": best_loss})


# ═══════════════════════════════════════════════════════════════════════
#  Strategy 7: Diagonal bias (penalise off-diagonal more)
# ═══════════════════════════════════════════════════════════════════════

def train_ridge_diag_bias(
    X: np.ndarray, Y: np.ndarray,
    alpha_diag: float = 1.0, alpha_off: float = 100.0,
) -> LinearModel:
    """Tikhonov with different penalties for diagonal vs off-diagonal.

    Solve:  min_W  ||Y - XW^T||² + α_diag * ||diag(W)||² + α_off * ||offdiag(W)||²

    Implemented as iterative re-weighted Ridge.
    """
    N = X.shape[1]
    # Equivalent: for each output j, the penalty on W[j,:] is
    # alpha_diag for coef j, alpha_off for all others.
    # This is Tikhonov with a per-output diagonal penalty matrix.
    W = np.zeros((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)

    XtX = X.T @ X
    XtY = X.T @ Y

    for j in range(N):
        # Penalty matrix for neuron j's weights
        penalty = np.full(N, alpha_off)
        penalty[j] = alpha_diag
        A = XtX + np.diag(penalty)
        w_j = np.linalg.solve(A, XtY[:, j])
        W[j] = w_j
        b[j] = Y[:, j].mean() - X.mean(axis=0) @ w_j

    # Recompute with proper intercept
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_c = X - X_mean
    Y_c = Y - Y_mean
    XtX_c = X_c.T @ X_c
    XtY_c = X_c.T @ Y_c
    for j in range(N):
        penalty = np.full(N, alpha_off)
        penalty[j] = alpha_diag
        A = XtX_c + np.diag(penalty)
        W[j] = np.linalg.solve(A, XtY_c[:, j])
    b = Y_mean - X_mean @ W.T

    rho = _spectral_radius(W)
    print(f"    α_diag={alpha_diag}  α_off={alpha_off}  ρ(W)={rho:.4f}")
    return LinearModel(W=W, b=b,
                       name=f"Ridge-Diag(d={alpha_diag},o={alpha_off})",
                       meta={"alpha_diag": alpha_diag, "alpha_off": alpha_off,
                              "spectral_radius": rho})


# ═══════════════════════════════════════════════════════════════════════
#  LOO evaluation (shared for all linear models)
# ═══════════════════════════════════════════════════════════════════════

def loo_windowed(
    model: LinearModel, u_test: np.ndarray, neuron_idx: int,
    window_size: int = 50,
) -> np.ndarray:
    T, N = u_test.shape
    u_pred = np.zeros(T, dtype=np.float64)
    u_pred[0] = u_test[0, neuron_idx]

    for w_start in range(0, T, window_size):
        w_end = min(w_start + window_size, T)
        u_pred[w_start] = u_test[w_start, neuron_idx]
        for t in range(w_start, w_end - 1):
            x_t = u_test[t].copy()
            x_t[neuron_idx] = u_pred[t]
            all_pred = model.predict(x_t.reshape(1, -1))[0]
            u_pred[t + 1] = all_pred[neuron_idx]
    return u_pred


def loo_full(
    model: LinearModel, u_test: np.ndarray, neuron_idx: int,
) -> np.ndarray:
    T, N = u_test.shape
    u_pred = np.zeros(T, dtype=np.float64)
    u_pred[0] = u_test[0, neuron_idx]
    for t in range(T - 1):
        x_t = u_test[t].copy()
        x_t[neuron_idx] = u_pred[t]
        all_pred = model.predict(x_t.reshape(1, -1))[0]
        u_pred[t + 1] = all_pred[neuron_idx]
    return u_pred


# ═══════════════════════════════════════════════════════════════════════
#  Model registry — all variants to sweep
# ═══════════════════════════════════════════════════════════════════════

def build_model_configs() -> List[Tuple[str, Callable]]:
    """Return list of (name, train_fn) where train_fn(X, Y, u_train) → LinearModel."""
    configs = []

    # 0. Vanilla Ridge (CV-selected alpha, baseline)
    def _ridge_cv(X, Y, u):
        alphas = np.logspace(-3, 6, 30)
        best_alpha, best_score = None, -np.inf
        n = X.shape[0]; fold_size = n // 3
        for alpha in alphas:
            scores = []
            for k in range(3):
                vs = k * fold_size
                ve = vs + fold_size if k < 2 else n
                X_tr = np.concatenate([X[:vs], X[ve:]])
                Y_tr = np.concatenate([Y[:vs], Y[ve:]])
                m = Ridge(alpha=alpha).fit(X_tr, Y_tr)
                scores.append(m.score(X[vs:ve], Y[vs:ve]))
            ms = np.mean(scores)
            if ms > best_score: best_score, best_alpha = ms, alpha
        m = Ridge(alpha=best_alpha).fit(X, Y)
        rho = _spectral_radius(m.coef_)
        print(f"    CV-alpha={best_alpha:.4g}  CV-R²={best_score:.4f}  ρ(W)={rho:.4f}")
        return LinearModel(W=m.coef_, b=m.intercept_, name="Ridge-CV",
                           meta={"alpha": best_alpha, "spectral_radius": rho})
    configs.append(("Ridge-CV", _ridge_cv))

    # 1. High-alpha variants
    for alpha in [1e2, 1e3, 1e4, 1e5]:
        configs.append((f"Ridge-α{alpha:.0e}",
                        lambda X, Y, u, a=alpha: train_ridge_high_alpha(X, Y, a)))

    # 2. Spectral clipping
    for rho_max in [0.90, 0.95, 0.99, 1.00]:
        configs.append((f"Ridge-Spec(ρ={rho_max})",
                        lambda X, Y, u, r=rho_max: train_ridge_spectral(X, Y, r)))

    # 3. Elastic Net
    for alpha, l1 in [(0.001, 0.1), (0.001, 0.5), (0.01, 0.1), (0.01, 0.5),
                      (0.1, 0.1), (0.1, 0.5)]:
        configs.append((f"EN(α={alpha},l1={l1})",
                        lambda X, Y, u, a=alpha, l=l1:
                            train_elastic_net(X, Y, alpha=a, l1_ratio=l)))

    # 4. PCA + Ridge (reduced rank)
    for k in [10, 20, 40, 60, 80]:
        configs.append((f"Ridge-PCR(k={k})",
                        lambda X, Y, u, n=k: train_ridge_pcr(X, Y, n)))

    # 5. Input noise
    for sigma in [0.01, 0.05, 0.1, 0.2]:
        configs.append((f"Ridge-Noise(σ={sigma})",
                        lambda X, Y, u, s=sigma: train_ridge_noise(X, Y, s)))

    # 6. Rollout training
    for K in [3, 5, 10]:
        configs.append((f"Ridge-Rollout(K={K})",
                        lambda X, Y, u, k=K: train_ridge_rollout(X, Y, u, k)))

    # 7. Diagonal bias
    for d, o in [(1, 10), (1, 100), (1, 1000), (0.1, 100)]:
        configs.append((f"Ridge-Diag(d={d},o={o})",
                        lambda X, Y, u, dd=d, oo=o: train_ridge_diag_bias(X, Y, dd, oo)))

    return configs


# ═══════════════════════════════════════════════════════════════════════
#  Full pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_sweep(
    h5_path: str,
    save_dir: str,
    stage2_npz: Optional[str] = None,
    loo_subset_size: int = 30,
    window_size: int = 50,
):
    save = Path(save_dir)
    save.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading data …")
    with h5py.File(h5_path, "r") as f:
        u = f["stage1/u_mean"][:]
        labels = [
            (l.decode() if isinstance(l, bytes) else str(l))
            for l in f["gcamp/neuron_labels"][:]
        ]
    T, N = u.shape
    print(f"  T={T}, N={N}")

    mid = T // 2 + 1
    folds = [(mid, T, 0, mid), (0, mid, mid, T)]

    var = np.var(u, axis=0)
    loo_neurons = sorted(np.argsort(var)[::-1][:loo_subset_size].tolist())
    print(f"  LOO neurons ({len(loo_neurons)}): {loo_neurons}")

    # Stage2 reference
    stage2_loo_r2_w = None
    stage2_cv_r2 = None
    if stage2_npz and Path(stage2_npz).exists():
        s2 = np.load(stage2_npz, allow_pickle=True)
        stage2_loo_r2_w = s2.get("cv_loo_r2_windowed", None)
        stage2_cv_r2 = s2.get("cv_r2", None)
        print("  Loaded stage2 reference.")

    # ── Build model configs ───────────────────────────────────────────
    configs = build_model_configs()
    model_names = [name for name, _ in configs]
    print(f"\n  {len(configs)} Ridge variants to evaluate")

    # Per-fold accumulated predictions
    cv_preds_w = {m: np.full((T, N), np.nan) for m in model_names}
    cv_onestep = {m: np.full((T, N), np.nan) for m in model_names}

    t_start = time.time()

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        print(f"\n{'='*70}")
        print(f"  Fold {fold_idx+1}/2  train=[{tr_s},{tr_e})  test=[{te_s},{te_e})")
        print(f"{'='*70}")

        u_train = u[tr_s:tr_e]
        u_test  = u[te_s:te_e]
        X_train = u_train[:-1]
        Y_train = u_train[1:]
        X_test  = u_test[:-1]

        for cfg_name, train_fn in configs:
            print(f"\n  ── {cfg_name} ──")
            t0 = time.time()
            try:
                model = train_fn(X_train, Y_train, u_train)
            except Exception as e:
                print(f"    FAILED: {e}")
                continue
            dt = time.time() - t0
            print(f"    trained in {dt:.1f}s")

            # One-step
            pred_os = model.predict(X_test)
            cv_onestep[cfg_name][te_s + 1 : te_e] = pred_os

            # LOO windowed
            for cnt, i in enumerate(loo_neurons):
                pred_w = loo_windowed(model, u_test, i, window_size=window_size)
                cv_preds_w[cfg_name][te_s:te_e, i] = pred_w
                if cnt == 0 or cnt == len(loo_neurons) - 1:
                    r2_w = _r2(u_test[1:, i], pred_w[1:])
                    print(f"    [{cnt+1:2d}/{len(loo_neurons)}] neuron {i:3d} "
                          f"({labels[i]:8s})  LOO_w={r2_w:.4f}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ═══════════════════════════════════════════════════════════════════
    #  Aggregate
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"  {'Model':35s}  1step(all)  LOO_w_mean  LOO_w_med")
    print("=" * 80)

    summary = {}
    for mname in model_names:
        r2_w_arr = np.full(N, np.nan)
        for i in loo_neurons:
            pred = cv_preds_w[mname][:, i]
            m = np.isfinite(pred)
            if m.sum() > 3:
                r2_w_arr[i] = _r2(u[m, i], pred[m])

        os_r2 = np.full(N, np.nan)
        for i in range(N):
            pred = cv_onestep[mname][:, i]
            m = np.isfinite(pred)
            if m.sum() > 3:
                os_r2[i] = _r2(u[m, i], pred[m])

        loo_vals_w = r2_w_arr[loo_neurons]
        summary[mname] = {
            "loo_r2_windowed": r2_w_arr.tolist(),
            "onestep_r2": os_r2.tolist(),
            "mean_loo_r2_w": float(np.nanmean(loo_vals_w)),
            "median_loo_r2_w": float(np.nanmedian(loo_vals_w)),
            "mean_onestep_r2_all": float(np.nanmean(os_r2)),
        }
        print(f"  {mname:35s}  {np.nanmean(os_r2):9.4f}  "
              f"{np.nanmean(loo_vals_w):10.4f}  {np.nanmedian(loo_vals_w):9.4f}")

    if stage2_loo_r2_w is not None:
        s2_w = stage2_loo_r2_w[loo_neurons]
        s2_os = float(np.nanmean(stage2_cv_r2)) if stage2_cv_r2 is not None else float("nan")
        print(f"  {'Stage2 (connectome)':35s}  {s2_os:9.4f}  "
              f"{np.nanmean(s2_w):10.4f}  {np.nanmedian(s2_w):9.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    with open(save / "ridge_sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    np_data = {"loo_neurons": np.array(loo_neurons)}
    for mname in model_names:
        key = mname.replace(" ", "_").replace("-", "_").replace(".", "p").replace("(", "").replace(")", "").replace(",", "_").replace("=", "")
        np_data[f"{key}_loo_pred_w"] = cv_preds_w[mname][:, loo_neurons]
        np_data[f"{key}_onestep"] = cv_onestep[mname]
    np.savez(save / "ridge_sweep_predictions.npz", **np_data)

    # ═══════════════════════════════════════════════════════════════════
    #  Plots
    # ═══════════════════════════════════════════════════════════════════
    _plot_summary(summary, stage2_loo_r2_w, stage2_cv_r2, loo_neurons, save)
    _plot_strategy_groups(summary, stage2_loo_r2_w, loo_neurons, save)

    print(f"\n  All results saved to {save}")
    return summary


def _plot_summary(summary, stage2_loo_r2_w, stage2_cv_r2, loo_neurons, save):
    """Bar chart: LOO_w mean & median, sorted by LOO_w mean."""
    model_names = sorted(summary.keys(),
                         key=lambda m: summary[m]["mean_loo_r2_w"], reverse=True)

    means = [summary[m]["mean_loo_r2_w"] for m in model_names]
    medians = [summary[m]["median_loo_r2_w"] for m in model_names]
    os_means = [summary[m]["mean_onestep_r2_all"] for m in model_names]
    x_labels = model_names[:]

    # Add stage2
    if stage2_loo_r2_w is not None:
        s2_w = stage2_loo_r2_w[loo_neurons]
        means.append(float(np.nanmean(s2_w)))
        medians.append(float(np.nanmedian(s2_w)))
        os_means.append(float(np.nanmean(stage2_cv_r2)) if stage2_cv_r2 is not None else 0)
        x_labels.append("Stage2")

    n = len(means)

    fig, axes = plt.subplots(3, 1, figsize=(max(14, n * 0.5), 14))

    # Color by strategy
    def _color(name):
        if "Stage2" in name: return "black"
        if "CV" in name: return "#1f77b4"
        if "α" in name and "EN" not in name: return "#ff7f0e"
        if "Spec" in name: return "#2ca02c"
        if "EN" in name: return "#d62728"
        if "PCR" in name: return "#9467bd"
        if "Noise" in name: return "#8c564b"
        if "Rollout" in name: return "#e377c2"
        if "Diag" in name: return "#7f7f7f"
        return "#17becf"

    colors = [_color(n) for n in x_labels]

    for ax, vals, ylabel, title in [
        (axes[0], means, "Mean LOO R² (windowed)", "Mean windowed LOO R² — Ridge variants"),
        (axes[1], medians, "Median LOO R² (windowed)", "Median windowed LOO R² — Ridge variants"),
        (axes[2], os_means, "Mean one-step R²", "One-step R² — Ridge variants"),
    ]:
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=6)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    max(val + 0.005, 0.005),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=5, rotation=90)

    plt.tight_layout()
    fig.savefig(save / "ridge_sweep_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save / 'ridge_sweep_summary.png'}")


def _plot_strategy_groups(summary, stage2_loo_r2_w, loo_neurons, save):
    """One subplot per strategy group: LOO_w mean as a function of the param."""
    groups = {
        "High Alpha": [m for m in summary if m.startswith("Ridge-α")],
        "Spectral Clip": [m for m in summary if "Spec" in m],
        "Elastic Net": [m for m in summary if m.startswith("EN")],
        "PCA + Ridge": [m for m in summary if "PCR" in m],
        "Input Noise": [m for m in summary if "Noise" in m],
        "Rollout": [m for m in summary if "Rollout" in m],
        "Diagonal Bias": [m for m in summary if "Diag" in m],
    }

    s2_ref = None
    if stage2_loo_r2_w is not None:
        s2_ref = float(np.nanmean(stage2_loo_r2_w[loo_neurons]))

    ridge_cv_ref = summary.get("Ridge-CV", {}).get("mean_loo_r2_w", None)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for idx, (group_name, members) in enumerate(groups.items()):
        ax = axes[idx]
        if not members:
            ax.set_visible(False)
            continue
        vals = [summary[m]["mean_loo_r2_w"] for m in members]
        ax.bar(range(len(members)), vals, color=plt.cm.Set2(np.linspace(0, 1, len(members))),
               edgecolor="black", linewidth=0.3)
        ax.set_xticks(range(len(members)))
        ax.set_xticklabels([m.split("(")[-1].rstrip(")") if "(" in m else m
                            for m in members], rotation=45, ha="right", fontsize=7)
        ax.set_title(group_name, fontsize=10)
        ax.set_ylabel("Mean LOO_w R²")
        if s2_ref is not None:
            ax.axhline(s2_ref, color="red", ls="--", lw=1, label=f"Stage2={s2_ref:.3f}")
        if ridge_cv_ref is not None:
            ax.axhline(ridge_cv_ref, color="blue", ls=":", lw=1, label=f"Ridge-CV={ridge_cv_ref:.3f}")
        ax.axhline(0, color="gray", lw=0.3, ls="--")
        ax.legend(fontsize=6, loc="upper right")
        for i, v in enumerate(vals):
            ax.text(i, max(v + 0.005, 0.005), f"{v:.3f}", ha="center",
                    va="bottom", fontsize=6)

    axes[-1].set_visible(False)  # unused
    plt.suptitle("Ridge regularization strategies — LOO_w R² (mean, 30 neurons)", fontsize=12)
    plt.tight_layout()
    fig.savefig(save / "ridge_strategy_groups.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save / 'ridge_strategy_groups.png'}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ridge regularization sweep for LOO")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--stage2_npz", default=None)
    parser.add_argument("--loo_subset", type=int, default=30)
    parser.add_argument("--window_size", type=int, default=50)
    args = parser.parse_args()

    run_sweep(
        h5_path=args.h5,
        save_dir=args.save_dir,
        stage2_npz=args.stage2_npz,
        loo_subset_size=args.loo_subset,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
