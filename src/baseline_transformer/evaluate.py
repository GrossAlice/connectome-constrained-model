"""Evaluation matching stage-2 semantics exactly.

Metrics
-------
1. One-step R²   — teacher-forced, per neuron (from CV held-out predictions)
2. LOO R²        — leave-one-out forward simulation, per neuron
3. Free-run R²   — fully autoregressive rollout
4. Behaviour R²  — direct from joint model output (if predict_beh)
                    OR ridge-CV decoder on motor neurons → eigenworms

All R² values use the same ``_r2`` function from stage2._utils.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import TransformerBaselineConfig
from .model import TemporalTransformerGaussian

# Reuse stage2 utilities directly
try:
    from stage2._utils import _r2
    from stage2.behavior_decoder_eval import (
        build_lagged_features_np,
        _log_ridge_grid,
        _make_contiguous_folds,
        _ridge_cv_single_target,
        valid_lag_mask_np,
    )
except ImportError:
    # Fallback: add parent to path if running from a different working dir
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stage2._utils import _r2
    from stage2.behavior_decoder_eval import (
        build_lagged_features_np,
        _log_ridge_grid,
        _make_contiguous_folds,
        _ridge_cv_single_target,
        valid_lag_mask_np,
    )

__all__ = [
    "compute_onestep_r2",
    "compute_onestep_r2_from_preds",
    "compute_loo_r2",
    "loo_forward_simulate_single_windowed",
    "compute_free_run_r2",
    "compute_behaviour_r2",
    "compute_behaviour_r2_direct",
    "compute_behaviour_r2_mlp",
    "run_full_evaluation",
    "run_full_evaluation_cv",
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _per_neuron_r2(u_true: np.ndarray, u_pred: np.ndarray) -> np.ndarray:
    """Per-neuron R² array."""
    N = u_true.shape[1]
    return np.array([_r2(u_true[:, i], u_pred[:, i]) for i in range(N)])


# ── 1. One-step R² ──────────────────────────────────────────────────────────


@torch.no_grad()
def compute_onestep_r2(
    model: TemporalTransformerGaussian,
    x: np.ndarray,
    start: int = 0,
    end: Optional[int] = None,
    n_neural: Optional[int] = None,
) -> Dict[str, Any]:
    """Teacher-forced one-step predictions on [start, end).

    Works with joint state x = [u, b].  Only evaluates neural columns
    for R².

    Parameters
    ----------
    model : trained model
    x     : (T, D) joint state
    start, end : evaluation range
    n_neural : number of neural columns (default: model.n_neural)

    Returns dict with pred, gt, r2, r2_mean (all neural-only).
    """
    device = next(model.parameters()).device
    T, D = x.shape
    K = model.cfg.context_length
    nn_ = n_neural or model.n_neural
    if end is None:
        end = T

    first = max(K, start + K)
    if first >= end:
        return {"pred": np.zeros((0, nn_)), "r2": np.full(nn_, np.nan),
                "r2_mean": float("nan")}

    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    preds_u = []
    preds_b = []
    for t in range(first, end):
        ctx = x_t[t - K : t].unsqueeze(0)  # (1, K, D)
        mu_u, mu_b = model.predict_mean_split(ctx)
        preds_u.append(mu_u.squeeze(0).cpu().numpy())
        if mu_b is not None:
            preds_b.append(mu_b.squeeze(0).cpu().numpy())

    pred_u = np.stack(preds_u, axis=0)          # (T_eval, n_neural)
    gt_u = x[first:end, :nn_]                   # (T_eval, n_neural)
    r2 = _per_neuron_r2(gt_u, pred_u)

    result = {
        "pred": pred_u,
        "gt": gt_u,
        "r2": r2,
        "r2_mean": float(np.nanmean(r2)),
        "eval_range": (first, end),
    }

    if preds_b:
        result["pred_b"] = np.stack(preds_b, axis=0)
        result["gt_b"] = x[first:end, nn_:]

    return result


def compute_onestep_r2_from_preds(
    pred_u: np.ndarray,
    gt_u: np.ndarray,
    pred_b: Optional[np.ndarray] = None,
    gt_b: Optional[np.ndarray] = None,
    b_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute one-step R² from pre-computed (stitched) predictions.

    Used after 5-fold CV to evaluate the stitched held-out predictions.

    Parameters
    ----------
    pred_u, gt_u : (T, n_neural)
    pred_b, gt_b : (T, n_beh) or None
    b_mask       : (T, n_beh) or None

    Returns dict with r2, r2_mean (neural) and beh_r2, beh_r2_mean.
    """
    # Filter out NaN rows (unfilled from CV stitching)
    valid = np.all(np.isfinite(pred_u), axis=1)
    pu = pred_u[valid]
    gu = gt_u[valid]

    if pu.shape[0] == 0:
        n_neural = pred_u.shape[1]
        return {"r2": np.full(n_neural, np.nan), "r2_mean": float("nan")}

    r2 = _per_neuron_r2(gu, pu)
    result: Dict[str, Any] = {
        "r2": r2,
        "r2_mean": float(np.nanmean(r2)),
        "n_valid_samples": int(valid.sum()),
    }

    # Behaviour R² (direct from model output)
    if pred_b is not None and gt_b is not None:
        pb = pred_b[valid]
        gb = gt_b[valid]
        n_beh = gb.shape[1]
        beh_r2 = np.full(n_beh, np.nan)
        if b_mask is not None:
            bm = b_mask[valid]
        else:
            bm = np.ones_like(gb)

        for j in range(n_beh):
            mask_j = bm[:, j] > 0.5
            if mask_j.sum() > 2:
                beh_r2[j] = _r2(gb[mask_j, j], pb[mask_j, j])

        result["beh_r2"] = beh_r2
        result["beh_r2_mean"] = float(np.nanmean(beh_r2))

    return result


# ── 2. LOO R² ───────────────────────────────────────────────────────────────


@torch.no_grad()
def loo_forward_simulate_single(
    model: TemporalTransformerGaussian,
    x_gt: np.ndarray,
    neuron_idx: int,
) -> np.ndarray:
    """LOO forward simulation for a single neuron.

    Works with joint state x = [u, b].  Only neuron_idx (< n_neural)
    is replaced with model predictions; all other channels use GT.

    Returns
    -------
    pred : (T, D) — all-channel predictions; only column neuron_idx
           is the LOO prediction.
    """
    device = next(model.parameters()).device
    T, D = x_gt.shape
    K = model.cfg.context_length
    i = neuron_idx
    assert i < model.n_neural

    buf = x_gt.copy().astype(np.float32)
    x_t = torch.tensor(buf, dtype=torch.float32, device=device)
    pred_all = np.zeros((T, D), dtype=np.float32)
    pred_all[:K] = buf[:K]

    for t in range(K, T):
        ctx = x_t[t - K : t].unsqueeze(0)
        mu = model.one_step(ctx)  # (1, D)
        mu_np = mu.squeeze(0).cpu().numpy()
        pred_all[t] = mu_np
        x_t[t, i] = mu[0, i]

    return pred_all


@torch.no_grad()
def loo_forward_simulate_single_windowed(
    model: TemporalTransformerGaussian,
    x_gt: np.ndarray,
    neuron_idx: int,
    window_size: int = 50,
) -> np.ndarray:
    """LOO forward simulation with periodic re-seeding.

    Matches ``stage2.evaluate.loo_forward_simulate_windowed`` semantics.
    """
    device = next(model.parameters()).device
    T, D = x_gt.shape
    K = model.cfg.context_length
    i = neuron_idx
    assert i < model.n_neural

    buf = x_gt.copy().astype(np.float32)
    x_t = torch.tensor(buf, dtype=torch.float32, device=device)
    pred_all = np.zeros((T, D), dtype=np.float32)
    pred_all[:K] = buf[:K]

    x_gt_t = torch.tensor(x_gt, dtype=torch.float32, device=device)
    for t in range(K, T):
        if (t - K) % window_size == 0:
            x_t[t - 1, i] = x_gt_t[t - 1, i]
            start = max(K, t - K)
            for s in range(start, t):
                x_t[s, i] = x_gt_t[s, i]

        ctx = x_t[t - K : t].unsqueeze(0)
        mu = model.one_step(ctx)
        mu_np = mu.squeeze(0).cpu().numpy()
        pred_all[t] = mu_np
        x_t[t, i] = mu[0, i]

    return pred_all


def compute_loo_r2(
    model: TemporalTransformerGaussian,
    x: np.ndarray,
    subset: Optional[List[int]] = None,
    verbose: bool = False,
    window_size: int = 0,
    n_neural: Optional[int] = None,
) -> Dict[str, Any]:
    """LOO R² for a subset of neurons.

    Parameters
    ----------
    model : trained Transformer
    x     : (T, D) joint state (or neural-only)
    subset : neuron indices to evaluate (must be < n_neural)
    window_size : if > 0, use windowed LOO
    n_neural : number of neural columns (default: model.n_neural)
    """
    T, D = x.shape
    K = model.cfg.context_length
    nn_ = n_neural or model.n_neural
    if subset is None:
        subset = list(range(nn_))

    use_windowed = window_size > 0
    tag = f"LOO-w{window_size}" if use_windowed else "LOO"

    r2_arr = np.full(nn_, np.nan)
    preds = {}

    for cnt, i in enumerate(subset):
        if use_windowed:
            pred_i = loo_forward_simulate_single_windowed(model, x, i, window_size)
        else:
            pred_i = loo_forward_simulate_single(model, x, i)
        r2_i = _r2(x[K:, i], pred_i[K:, i])
        r2_arr[i] = r2_i
        preds[i] = pred_i[:, i]

        if verbose and (cnt == 0 or (cnt + 1) % 10 == 0 or cnt == len(subset) - 1):
            print(f"  {tag} {cnt+1}/{len(subset)}  neuron={i}  R²={r2_i:.4f}")

    finite = np.isfinite(r2_arr)
    r2_mean = float(np.nanmean(r2_arr)) if finite.any() else float("nan")

    return {
        "r2": r2_arr,
        "r2_mean": r2_mean,
        "pred": preds,
        "subset": subset,
        "window_size": window_size,
    }


# ── 3. Free-run R² ──────────────────────────────────────────────────────────


@torch.no_grad()
def compute_free_run_r2(
    model: TemporalTransformerGaussian,
    x: np.ndarray,
    motor_idx: Optional[List[int]] = None,
    n_neural: Optional[int] = None,
) -> Dict[str, Any]:
    """Autoregressive free-run from x[0:K].

    Two modes:
    * autonomous:         all channels predicted autoregressively
    * motor-conditioned:  motor neurons use GT, others predicted

    Works with joint state x = [u, b].
    """
    device = next(model.parameters()).device
    T, D = x.shape
    K = model.cfg.context_length
    nn_ = n_neural or model.n_neural

    conditioned = motor_idx is not None and 0 < len(motor_idx) < nn_
    motor_mask = np.zeros(D, dtype=bool)
    if motor_idx:
        for mi in motor_idx:
            if mi < D:
                motor_mask[mi] = True

    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    pred = np.zeros((T, D), dtype=np.float32)
    pred[:K] = x[:K]
    buf = x_t[:K].clone()

    for t in range(K, T):
        ctx = buf[-K:].unsqueeze(0)
        mu = model.predict_mean(ctx).squeeze(0)  # (D,)
        mu_np = mu.cpu().numpy()

        if conditioned:
            step = x[t].copy()
            step[motor_mask] = mu_np[motor_mask]
        else:
            step = mu_np

        pred[t] = step
        buf = torch.cat([buf, torch.tensor(step, dtype=torch.float32, device=device).unsqueeze(0)], dim=0)

    # Evaluate only neural columns
    eval_idx = list(np.where(motor_mask[:nn_])[0]) if conditioned else list(range(nn_))
    r2_arr = np.full(nn_, np.nan)
    for i in eval_idx:
        r2_arr[i] = _r2(x[K:, i], pred[K:, i])

    return {
        "pred": pred,
        "r2": r2_arr,
        "r2_mean": float(np.nanmean(r2_arr[eval_idx])) if eval_idx else float("nan"),
        "mode": "motor_conditioned" if conditioned else "autonomous",
        "motor_idx": motor_idx,
    }


# ── 4. Behaviour R² (direct from model) ─────────────────────────────────────


def compute_behaviour_r2_direct(
    pred_b: np.ndarray,
    gt_b: np.ndarray,
    b_mask: np.ndarray,
) -> Dict[str, Any]:
    """Behaviour R² directly from model's behaviour predictions.

    No ridge decoder needed — the model predicts behaviour directly.

    Parameters
    ----------
    pred_b : (T, n_beh) predicted behaviour
    gt_b   : (T, n_beh) ground truth behaviour
    b_mask : (T, n_beh) validity mask
    """
    n_beh = gt_b.shape[1]
    r2 = np.full(n_beh, np.nan)

    for j in range(n_beh):
        valid = b_mask[:, j] > 0.5
        if valid.sum() > 2:
            r2[j] = _r2(gt_b[valid, j], pred_b[valid, j])

    return {
        "r2_direct": r2,
        "r2_direct_mean": float(np.nanmean(r2)),
    }


# ── 5. Behaviour R² (ridge decoder — for comparison) ────────────────────────


def compute_behaviour_r2(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    b: np.ndarray,
    b_mask: np.ndarray,
    motor_idx: List[int],
    n_lags: int = 8,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Behaviour R² via ridge-CV decoder on motor neurons → eigenworms.

    Kept for backward compatibility and comparison with stage2.
    """
    n_modes = b.shape[1]
    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)

    X_gt = build_lagged_features_np(u_gt[:, motor_idx], n_lags)
    X_pred = build_lagged_features_np(u_pred[:, motor_idx], n_lags)

    r2_gt = np.full(n_modes, np.nan)
    r2_model = np.full(n_modes, np.nan)

    for j in range(n_modes):
        valid = valid_lag_mask_np(b.shape[0], n_lags, b_mask[:, j] > 0.5)
        idx_v = np.where(valid)[0]
        if len(idx_v) < 10:
            continue

        fit_gt = _ridge_cv_single_target(X_gt, b[:, j], idx_v, ridge_grid, n_folds)
        mask_gt = np.isfinite(fit_gt["held_out"]) & (b_mask[:, j] > 0.5)
        if mask_gt.sum() > 2:
            r2_gt[j] = _r2(b[mask_gt, j], fit_gt["held_out"][mask_gt])

        fit_model = _ridge_cv_single_target(X_pred, b[:, j], idx_v, ridge_grid, n_folds)
        mask_model = np.isfinite(fit_model["held_out"]) & (b_mask[:, j] > 0.5)
        if mask_model.sum() > 2:
            r2_model[j] = _r2(b[mask_model, j], fit_model["held_out"][mask_model])

    return {
        "r2_gt": r2_gt,
        "r2_model": r2_model,
        "r2_gt_mean": float(np.nanmean(r2_gt)),
        "r2_model_mean": float(np.nanmean(r2_model)),
        "n_motor": len(motor_idx),
        "n_lags": n_lags,
    }


# ── 5b. Behaviour R² (MLP decoder — stage2-style) ───────────────────────────


def _train_mlp_decoder_cv(
    X: np.ndarray,
    Y: np.ndarray,
    b_mask: np.ndarray,
    n_folds: int = 5,
    hidden: int = 128,
    n_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 300,
    patience: int = 30,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Train an MLP decoder with temporal k-fold CV, return held-out R².

    Uses the same MLPBehaviourDecoder architecture from stage2.
    Returns per-mode held-out R² and per-mode full-data R².
    """
    import torch
    import torch.nn as nn
    from stage2.behavior_decoder_eval import MLPBehaviourDecoder

    n_modes = Y.shape[1]
    T = X.shape[0]
    device = torch.device("cpu")

    # Build contiguous temporal folds
    valid_all = np.all(b_mask > 0.5, axis=1) & np.all(np.isfinite(X), axis=1)
    valid_idx = np.where(valid_all)[0]
    if valid_idx.size < 20:
        return {
            "r2_ho": np.full(n_modes, np.nan),
            "r2_ho_mean": float("nan"),
        }

    fold_sizes = np.full(n_folds, valid_idx.size // n_folds, dtype=int)
    fold_sizes[: valid_idx.size % n_folds] += 1
    folds, cur = [], 0
    for s in fold_sizes:
        folds.append(valid_idx[cur : cur + s])
        cur += s

    pred_ho = np.full_like(Y, np.nan)

    for fold_i, te_idx in enumerate(folds):
        te_set = set(te_idx.tolist())
        tr_idx = np.array([i for i in valid_idx if i not in te_set])

        # Inner val split (last 20% of training for early stopping)
        n_va = max(1, int(len(tr_idx) * 0.2))
        inner_tr, inner_va = tr_idx[:-n_va], tr_idx[-n_va:]

        # z-score using training stats
        mu = X[inner_tr].mean(axis=0)
        std = X[inner_tr].std(axis=0).clip(1e-8)
        X_tr_z = torch.tensor((X[inner_tr] - mu) / std, dtype=torch.float32, device=device)
        Y_tr = torch.tensor(Y[inner_tr], dtype=torch.float32, device=device)
        X_va_z = torch.tensor((X[inner_va] - mu) / std, dtype=torch.float32, device=device)
        Y_va = torch.tensor(Y[inner_va], dtype=torch.float32, device=device)
        X_te_z = torch.tensor((X[te_idx] - mu) / std, dtype=torch.float32, device=device)

        model = MLPBehaviourDecoder(X.shape[1], n_modes, hidden, n_layers, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.MSELoss()

        best_val, stale, best_state = float("inf"), 0, None
        n_tr = X_tr_z.shape[0]

        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(n_tr, device=device)
            for i in range(0, n_tr, batch_size):
                idx = perm[i : i + batch_size]
                loss = crit(model(X_tr_z[idx]), Y_tr[idx])
                opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                vloss = crit(model(X_va_z), Y_va).item()
            if vloss < best_val:
                best_val, stale = vloss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if patience and stale >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_ho[te_idx] = model(X_te_z).cpu().numpy()

    # Compute held-out R² per mode
    r2_ho = np.full(n_modes, np.nan)
    for j in range(n_modes):
        mask = valid_all & np.isfinite(pred_ho[:, j])
        if mask.sum() > 2:
            r2_ho[j] = _r2(Y[mask, j], pred_ho[mask, j])

    return {
        "r2_ho": r2_ho,
        "r2_ho_mean": float(np.nanmean(r2_ho)),
    }


def compute_behaviour_r2_mlp(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    b: np.ndarray,
    b_mask: np.ndarray,
    motor_idx: List[int],
    n_lags: int = 8,
    n_folds: int = 5,
    hidden: int = 128,
    n_layers: int = 2,
    dropout: float = 0.1,
) -> Dict[str, Any]:
    """Behaviour R² via MLP decoder on motor neurons → eigenworms (CV).

    Same structure as compute_behaviour_r2 but uses a per-fold MLP
    (stage2 MLPBehaviourDecoder) instead of ridge regression.
    """
    X_gt = build_lagged_features_np(u_gt[:, motor_idx], n_lags)
    X_pred = build_lagged_features_np(u_pred[:, motor_idx], n_lags)

    # Trim b_mask for lag validity
    bm = b_mask.copy()
    if n_lags > 0:
        bm[:n_lags] = 0.0

    gt_result = _train_mlp_decoder_cv(X_gt, b, bm, n_folds=n_folds,
                                       hidden=hidden, n_layers=n_layers,
                                       dropout=dropout)
    model_result = _train_mlp_decoder_cv(X_pred, b, bm, n_folds=n_folds,
                                          hidden=hidden, n_layers=n_layers,
                                          dropout=dropout)

    return {
        "r2_gt": gt_result["r2_ho"],
        "r2_model": model_result["r2_ho"],
        "r2_gt_mean": gt_result["r2_ho_mean"],
        "r2_model_mean": model_result["r2_ho_mean"],
        "n_motor": len(motor_idx),
        "n_lags": n_lags,
        "decoder": "mlp",
    }


# ── 6. All-neuron behaviour baseline ────────────────────────────────────────


def compute_behaviour_all_neurons(
    u: np.ndarray,
    b: np.ndarray,
    b_mask: np.ndarray,
    n_lags: int = 8,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """All-neuron ridge-CV behaviour decoder (upper bound baseline)."""
    n_modes = b.shape[1]
    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    X_all = build_lagged_features_np(u, n_lags)

    r2_all = np.full(n_modes, np.nan)
    for j in range(n_modes):
        valid = valid_lag_mask_np(b.shape[0], n_lags, b_mask[:, j] > 0.5)
        idx_v = np.where(valid)[0]
        if len(idx_v) < 10:
            continue
        fit_j = _ridge_cv_single_target(X_all, b[:, j], idx_v, ridge_grid, n_folds)
        mask = np.isfinite(fit_j["held_out"]) & (b_mask[:, j] > 0.5)
        if mask.sum() > 2:
            r2_all[j] = _r2(b[mask, j], fit_j["held_out"][mask])

    return {
        "r2_all_neurons": r2_all,
        "r2_all_mean": float(np.nanmean(r2_all)),
    }


# ── Full evaluation (CV version) ────────────────────────────────────────────


def run_full_evaluation_cv(
    train_result: Dict[str, Any],
    worm_data: Dict[str, Any],
    cfg: TransformerBaselineConfig,
    loo_subset_size: int = 20,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all evaluation metrics after 5-fold CV training.

    Uses:
    * Stitched held-out predictions for one-step R² (neural + behaviour)
    * Best-fold model for LOO / free-run

    Parameters
    ----------
    train_result : dict from train_single_worm_cv
    worm_data    : dict from load_worm_data
    cfg          : config
    loo_subset_size : number of neurons for LOO (0 = all)
    verbose      : print progress

    Returns dict with onestep, loo, free_run, behaviour results.
    """
    u = worm_data["u"]
    b = worm_data.get("b")
    b_mask = worm_data.get("b_mask")
    motor_idx = worm_data.get("motor_idx")
    worm_id = worm_data.get("worm_id", "unknown")

    n_neural = train_result["n_neural"]
    n_beh = train_result["n_beh"]
    pred_u_full = train_result["pred_u_full"]
    pred_b_full = train_result.get("pred_b_full")
    x = train_result["x"]       # (T, D) joint state
    x_mask = train_result.get("x_mask")

    T, N = u.shape
    K = cfg.context_length

    if verbose:
        print(f"\n[{worm_id}] Evaluating (CV, n_neural={n_neural}, n_beh={n_beh})")

    # ---- 1. One-step R² from stitched CV predictions ----
    if verbose:
        print(f"  Computing one-step R² from CV held-out predictions...")

    gt_b_for_eval = b if (b is not None and n_beh > 0) else None
    bm_for_eval = b_mask if (b_mask is not None and n_beh > 0) else None

    onestep = compute_onestep_r2_from_preds(
        pred_u=pred_u_full,
        gt_u=u,
        pred_b=pred_b_full,
        gt_b=gt_b_for_eval,
        b_mask=bm_for_eval,
    )

    if verbose:
        print(f"  One-step R² (neural) = {onestep['r2_mean']:.4f}")
        if "beh_r2_mean" in onestep:
            print(f"  One-step R² (behaviour) = {onestep['beh_r2_mean']:.4f}")

    # ---- 2. LOO R² (use best-fold model, full recording) ----
    best_model = train_result["best_model"]
    device = next(best_model.parameters()).device  # keep on original device (cuda)
    best_model.eval()

    if verbose:
        print(f"  Computing LOO R² (full-series, best fold model)...")

    if 0 < loo_subset_size < n_neural:
        variances = np.nanvar(u, axis=0)
        subset = list(np.argsort(variances)[::-1][:loo_subset_size])
    else:
        subset = list(range(n_neural))

    loo = compute_loo_r2(best_model, x, subset=subset, verbose=verbose,
                         window_size=0, n_neural=n_neural)
    if verbose:
        print(f"  LOO R² mean = {loo['r2_mean']:.4f}")

    # Windowed LOO
    loo_window_size = 50
    if verbose:
        print(f"  Computing LOO R² (windowed, w={loo_window_size})...")
    loo_windowed = compute_loo_r2(best_model, x, subset=subset, verbose=verbose,
                                  window_size=loo_window_size, n_neural=n_neural)
    if verbose:
        print(f"  LOO-windowed R² mean = {loo_windowed['r2_mean']:.4f}")

    # ---- 3. Free-run R² ----
    if verbose:
        print(f"  Computing free-run R²...")
    free_run = compute_free_run_r2(best_model, x, motor_idx=motor_idx,
                                   n_neural=n_neural)
    if verbose:
        print(f"  Free-run R² mean = {free_run['r2_mean']:.4f}  ({free_run['mode']})")

    # ---- 4. Behaviour R² ----
    beh_direct_result = None
    beh_ridge_result = None
    beh_mlp_result = None
    beh_all_result = None

    if b is not None and b_mask is not None:
        # 4a. Direct behaviour R² from model output (if available)
        if pred_b_full is not None and n_beh > 0:
            valid_rows = np.all(np.isfinite(pred_b_full), axis=1)
            if valid_rows.sum() > 10:
                beh_direct_result = compute_behaviour_r2_direct(
                    pred_b=pred_b_full[valid_rows],
                    gt_b=b[valid_rows],
                    b_mask=b_mask[valid_rows],
                )
                if verbose:
                    print(f"  Behaviour R² (direct from model) = {beh_direct_result['r2_direct_mean']:.4f}")

        # 4b. Ridge decoder (for comparison)
        if motor_idx is not None:
            if verbose:
                print(f"  Computing behaviour R² (ridge decoder)...")

            # Use the best-fold model to get one-step predictions on full recording
            onestep_full = compute_onestep_r2(best_model, x, n_neural=n_neural)

            beh_ridge_result = compute_behaviour_r2(
                u_pred=onestep_full["pred"],
                u_gt=u[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
                b=b[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
                b_mask=b_mask[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
                motor_idx=motor_idx,
                n_lags=cfg.behavior_lag_steps,
            )
            if verbose:
                print(f"  Behaviour R² (ridge, model) = {beh_ridge_result['r2_model_mean']:.4f}")
                print(f"  Behaviour R² (ridge, GT)    = {beh_ridge_result['r2_gt_mean']:.4f}")

            # 4c. MLP decoder
            if verbose:
                print(f"  Computing behaviour R² (MLP decoder)...")
            beh_mlp_result = compute_behaviour_r2_mlp(
                u_pred=onestep_full["pred"],
                u_gt=u[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
                b=b[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
                b_mask=b_mask[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
                motor_idx=motor_idx,
                n_lags=cfg.behavior_lag_steps,
            )
            if verbose:
                print(f"  Behaviour R² (MLP, model) = {beh_mlp_result['r2_model_mean']:.4f}")
                print(f"  Behaviour R² (MLP, GT)    = {beh_mlp_result['r2_gt_mean']:.4f}")

            beh_all_result = compute_behaviour_all_neurons(
                u=u, b=b, b_mask=b_mask, n_lags=cfg.behavior_lag_steps,
            )

    elif verbose:
        print(f"  Skipping behaviour R² (no behaviour data)")

    return {
        "worm_id": worm_id,
        "n_neural": n_neural,
        "n_beh": n_beh,
        "n_folds": cfg.n_cv_folds,
        "onestep": {
            "r2": onestep["r2"].tolist(),
            "r2_mean": onestep["r2_mean"],
            "n_valid_samples": onestep.get("n_valid_samples"),
        },
        "onestep_beh": {
            "r2": onestep["beh_r2"].tolist() if "beh_r2" in onestep else None,
            "r2_mean": onestep.get("beh_r2_mean"),
        },
        "loo": {
            "r2": loo["r2"].tolist(),
            "r2_mean": loo["r2_mean"],
            "subset": loo["subset"],
        },
        "loo_windowed": {
            "r2": loo_windowed["r2"].tolist(),
            "r2_mean": loo_windowed["r2_mean"],
            "subset": loo_windowed["subset"],
            "window_size": loo_windowed["window_size"],
        },
        "free_run": {
            "r2": free_run["r2"].tolist(),
            "r2_mean": free_run["r2_mean"],
            "mode": free_run["mode"],
        },
        "behaviour_direct": {
            "r2": beh_direct_result["r2_direct"].tolist() if beh_direct_result else None,
            "r2_mean": beh_direct_result["r2_direct_mean"] if beh_direct_result else None,
        },
        "behaviour_ridge": {
            "r2_model": beh_ridge_result["r2_model"].tolist() if beh_ridge_result else None,
            "r2_model_mean": beh_ridge_result["r2_model_mean"] if beh_ridge_result else None,
            "r2_gt": beh_ridge_result["r2_gt"].tolist() if beh_ridge_result else None,
            "r2_gt_mean": beh_ridge_result["r2_gt_mean"] if beh_ridge_result else None,
        },
        "behaviour_mlp": {
            "r2_model": beh_mlp_result["r2_model"].tolist() if beh_mlp_result else None,
            "r2_model_mean": beh_mlp_result["r2_model_mean"] if beh_mlp_result else None,
            "r2_gt": beh_mlp_result["r2_gt"].tolist() if beh_mlp_result else None,
            "r2_gt_mean": beh_mlp_result["r2_gt_mean"] if beh_mlp_result else None,
        },
        "behaviour_all_neurons": {
            "r2": beh_all_result["r2_all_neurons"].tolist() if beh_all_result else None,
            "r2_mean": beh_all_result["r2_all_mean"] if beh_all_result else None,
        },
        "fold_val_losses": [fr["best_val_loss"] for fr in train_result["fold_results"]],
    }


# ── Full evaluation (legacy single-split) ───────────────────────────────────


def run_full_evaluation(
    model: TemporalTransformerGaussian,
    worm_data: Dict[str, Any],
    split: Dict[str, Tuple[int, int]],
    cfg: TransformerBaselineConfig,
    loo_subset_size: int = 20,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all evaluation metrics for a single worm (legacy single-split).

    Kept for backward compatibility.
    """
    from .dataset import build_joint_state

    model.eval()
    device = next(model.parameters()).device
    u = worm_data["u"]
    b = worm_data.get("b")
    b_mask = worm_data.get("b_mask")
    motor_idx = worm_data.get("motor_idx")
    T, N = u.shape
    K = cfg.context_length
    worm_id = worm_data.get("worm_id", "unknown")

    # Build joint state for evaluation
    x, x_mask, n_neural, n_beh = build_joint_state(
        u, b, b_mask, include_beh=cfg.include_beh_input and cfg.predict_beh,
    )

    te_s, te_e = split["test"]

    if verbose:
        print(f"\n[{worm_id}] Evaluating on test region [{te_s}, {te_e})")

    # ---- 1. One-step R² (on test region) ----
    if verbose:
        print(f"  Computing one-step R²...")
    onestep = compute_onestep_r2(model, x, start=te_s, end=te_e, n_neural=n_neural)
    if verbose:
        print(f"  One-step R² mean = {onestep['r2_mean']:.4f}")

    onestep_full = compute_onestep_r2(model, x, n_neural=n_neural)

    # ---- 2. LOO R² ----
    if verbose:
        print(f"  Computing LOO R² (full-series)...")

    if 0 < loo_subset_size < n_neural:
        variances = np.nanvar(u, axis=0)
        subset = list(np.argsort(variances)[::-1][:loo_subset_size])
    else:
        subset = list(range(n_neural))

    loo = compute_loo_r2(model, x, subset=subset, verbose=verbose,
                         window_size=0, n_neural=n_neural)
    if verbose:
        print(f"  LOO R² mean = {loo['r2_mean']:.4f}")

    loo_window_size = 50
    if verbose:
        print(f"  Computing LOO R² (windowed, w={loo_window_size})...")
    loo_windowed = compute_loo_r2(model, x, subset=subset, verbose=verbose,
                                  window_size=loo_window_size, n_neural=n_neural)
    if verbose:
        print(f"  LOO-windowed R² mean = {loo_windowed['r2_mean']:.4f}")

    # ---- 3. Free-run R² ----
    if verbose:
        print(f"  Computing free-run R²...")
    free_run = compute_free_run_r2(model, x, motor_idx=motor_idx, n_neural=n_neural)
    if verbose:
        print(f"  Free-run R² mean = {free_run['r2_mean']:.4f}  ({free_run['mode']})")

    # ---- 4. Behaviour R² ----
    beh_result = None
    beh_mlp_result = None
    beh_all_result = None
    if b is not None and b_mask is not None and motor_idx is not None:
        if verbose:
            print(f"  Computing behaviour R²...")
        beh_result = compute_behaviour_r2(
            u_pred=onestep_full["pred"],
            u_gt=u[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
            b=b[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
            b_mask=b_mask[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
            motor_idx=motor_idx,
            n_lags=cfg.behavior_lag_steps,
        )
        if verbose:
            print(f"  Computing behaviour R² (MLP decoder)...")
        beh_mlp_result = compute_behaviour_r2_mlp(
            u_pred=onestep_full["pred"],
            u_gt=u[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
            b=b[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
            b_mask=b_mask[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
            motor_idx=motor_idx,
            n_lags=cfg.behavior_lag_steps,
        )
        beh_all_result = compute_behaviour_all_neurons(
            u=u, b=b, b_mask=b_mask, n_lags=cfg.behavior_lag_steps,
        )
        if verbose:
            print(f"  Behaviour R² (model): {beh_result['r2_model_mean']:.4f}")
            print(f"  Behaviour R² (GT):    {beh_result['r2_gt_mean']:.4f}")
            print(f"  Behaviour R² (MLP, model): {beh_mlp_result['r2_model_mean']:.4f}")
            print(f"  Behaviour R² (MLP, GT):    {beh_mlp_result['r2_gt_mean']:.4f}")
    elif verbose:
        print(f"  Skipping behaviour R² (no behaviour data or motor neurons)")

    return {
        "worm_id": worm_id,
        "onestep": {
            "r2": onestep["r2"].tolist(),
            "r2_mean": onestep["r2_mean"],
            "eval_range": onestep["eval_range"],
        },
        "onestep_full": {
            "r2": onestep_full["r2"].tolist(),
            "r2_mean": onestep_full["r2_mean"],
        },
        "loo": {
            "r2": loo["r2"].tolist(),
            "r2_mean": loo["r2_mean"],
            "subset": loo["subset"],
        },
        "loo_windowed": {
            "r2": loo_windowed["r2"].tolist(),
            "r2_mean": loo_windowed["r2_mean"],
            "subset": loo_windowed["subset"],
            "window_size": loo_windowed["window_size"],
        },
        "free_run": {
            "r2": free_run["r2"].tolist(),
            "r2_mean": free_run["r2_mean"],
            "mode": free_run["mode"],
        },
        "behaviour": {
            "r2_model": beh_result["r2_model"].tolist() if beh_result else None,
            "r2_model_mean": beh_result["r2_model_mean"] if beh_result else None,
            "r2_gt": beh_result["r2_gt"].tolist() if beh_result else None,
            "r2_gt_mean": beh_result["r2_gt_mean"] if beh_result else None,
        },
        "behaviour_mlp": {
            "r2_model": beh_mlp_result["r2_model"].tolist() if beh_mlp_result else None,
            "r2_model_mean": beh_mlp_result["r2_model_mean"] if beh_mlp_result else None,
            "r2_gt": beh_mlp_result["r2_gt"].tolist() if beh_mlp_result else None,
            "r2_gt_mean": beh_mlp_result["r2_gt_mean"] if beh_mlp_result else None,
        },
        "behaviour_all_neurons": {
            "r2": beh_all_result["r2_all_neurons"].tolist() if beh_all_result else None,
            "r2_mean": beh_all_result["r2_all_mean"] if beh_all_result else None,
        },
    }


def save_evaluation(results: Dict[str, Any], save_dir: str, worm_id: str) -> None:
    """Save evaluation results to JSON."""
    out = Path(save_dir) / worm_id
    out.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    clean = _sanitize(results)
    (out / "eval_results.json").write_text(json.dumps(clean, indent=2))
