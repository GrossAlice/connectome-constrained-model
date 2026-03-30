"""Evaluation matching stage-2 semantics exactly.

Metrics
-------
1. One-step R²   — teacher-forced, per neuron
2. LOO R²        — leave-one-out forward simulation, per neuron
3. Free-run R²   — fully autoregressive rollout
4. Behaviour R²  — ridge-CV decoder on motor neurons → eigenworms

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
    "compute_loo_r2",
    "compute_free_run_r2",
    "compute_behaviour_r2",
    "run_full_evaluation",
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
    u: np.ndarray,
    start: int = 0,
    end: Optional[int] = None,
) -> Dict[str, Any]:
    """Teacher-forced one-step predictions on [start, end).

    For each t in [K+start, end), predict u(t) from u[t-K:t].

    Returns dict with keys:
        pred   : (T_eval, N) predicted means
        r2     : (N,) per-neuron R²
        r2_mean: float  mean R²
    """
    device = next(model.parameters()).device
    T, N = u.shape
    K = model.cfg.context_length
    if end is None:
        end = T

    first = max(K, start + K)
    if first >= end:
        return {"pred": np.zeros((0, N)), "r2": np.full(N, np.nan), "r2_mean": float("nan")}

    u_t = torch.tensor(u, dtype=torch.float32, device=device)
    preds = []
    for t in range(first, end):
        ctx = u_t[t - K : t].unsqueeze(0)  # (1, K, N)
        mu = model.predict_mean(ctx)        # (1, N)
        preds.append(mu.squeeze(0).cpu().numpy())

    pred = np.stack(preds, axis=0)          # (T_eval, N)
    gt = u[first:end]                       # (T_eval, N)
    r2 = _per_neuron_r2(gt, pred)

    return {
        "pred": pred,
        "gt": gt,
        "r2": r2,
        "r2_mean": float(np.nanmean(r2)),
        "eval_range": (first, end),
    }


# ── 2. LOO R² ───────────────────────────────────────────────────────────────


@torch.no_grad()
def loo_forward_simulate_single(
    model: TemporalTransformerGaussian,
    u_gt: np.ndarray,
    neuron_idx: int,
) -> np.ndarray:
    """LOO forward simulation for a single neuron.

    Matches stage2.evaluate.loo_forward_simulate semantics:
    at each step t, the context window uses ground truth for all neurons
    except neuron_idx, which uses the model's own prediction from the
    previous step.

    Parameters
    ----------
    model : trained Transformer
    u_gt  : (T, N) ground truth
    neuron_idx : index of the held-out neuron

    Returns
    -------
    pred : (T, N) — all-neuron predictions; only column neuron_idx
           is the LOO prediction; others are teacher-forced.
    """
    device = next(model.parameters()).device
    T, N = u_gt.shape
    K = model.cfg.context_length
    i = neuron_idx

    # Create a buffer: starts as GT, but neuron i gets replaced with
    # model predictions as we go.
    buf = u_gt.copy().astype(np.float32)
    u_t = torch.tensor(buf, dtype=torch.float32, device=device)
    pred_all = np.zeros((T, N), dtype=np.float32)
    pred_all[:K] = buf[:K]

    for t in range(K, T):
        ctx = u_t[t - K : t].unsqueeze(0)  # (1, K, N)
        mu = model.predict_mean(ctx)        # (1, N)
        mu_np = mu.squeeze(0).cpu().numpy()
        pred_all[t] = mu_np

        # Replace neuron i's GT at time t with model prediction
        u_t[t, i] = mu[0, i]

    return pred_all


def compute_loo_r2(
    model: TemporalTransformerGaussian,
    u: np.ndarray,
    subset: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """LOO R² for a subset of neurons.

    Parameters
    ----------
    model : trained Transformer
    u     : (T, N) ground truth
    subset : neuron indices to evaluate (default: all)
    verbose : print progress

    Returns
    -------
    dict with r2 (N,), r2_mean, pred dict
    """
    T, N = u.shape
    K = model.cfg.context_length
    if subset is None:
        subset = list(range(N))

    r2_arr = np.full(N, np.nan)
    preds = {}

    for cnt, i in enumerate(subset):
        pred_i = loo_forward_simulate_single(model, u, i)
        r2_i = _r2(u[K:, i], pred_i[K:, i])
        r2_arr[i] = r2_i
        preds[i] = pred_i[:, i]

        if verbose and (cnt == 0 or (cnt + 1) % 10 == 0 or cnt == len(subset) - 1):
            print(f"  LOO {cnt+1}/{len(subset)}  neuron={i}  R²={r2_i:.4f}")

    finite = np.isfinite(r2_arr)
    r2_mean = float(np.nanmean(r2_arr)) if finite.any() else float("nan")

    return {
        "r2": r2_arr,
        "r2_mean": r2_mean,
        "pred": preds,
        "subset": subset,
    }


# ── 3. Free-run R² ──────────────────────────────────────────────────────────


@torch.no_grad()
def compute_free_run_r2(
    model: TemporalTransformerGaussian,
    u: np.ndarray,
    motor_idx: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Autoregressive free-run from u[0:K].

    Two modes:
    * autonomous:         all neurons are predicted autoregressively
    * motor-conditioned:  motor neurons use GT, others predicted

    Matches stage2.evaluate.compute_free_run semantics.
    """
    device = next(model.parameters()).device
    T, N = u.shape
    K = model.cfg.context_length

    conditioned = motor_idx is not None and 0 < len(motor_idx) < N
    motor_mask = np.zeros(N, dtype=bool)
    if motor_idx:
        motor_mask[np.array(motor_idx)] = True

    u_t = torch.tensor(u, dtype=torch.float32, device=device)

    # Initialize with GT context
    pred = np.zeros((T, N), dtype=np.float32)
    pred[:K] = u[:K]
    buf = u_t[:K].clone()  # running context buffer

    for t in range(K, T):
        ctx = buf[-K:].unsqueeze(0)  # (1, K, N)
        mu = model.predict_mean(ctx).squeeze(0)  # (N,)
        mu_np = mu.cpu().numpy()

        if conditioned:
            # Use GT for non-motor neurons, prediction for motor neurons
            step = u[t].copy()
            step[motor_mask] = mu_np[motor_mask]
        else:
            step = mu_np

        pred[t] = step
        buf = torch.cat([buf, torch.tensor(step, dtype=torch.float32, device=device).unsqueeze(0)], dim=0)

    # Evaluate
    eval_idx = list(np.where(motor_mask)[0]) if conditioned else list(range(N))
    r2_arr = np.full(N, np.nan)
    for i in eval_idx:
        r2_arr[i] = _r2(u[K:, i], pred[K:, i])

    return {
        "pred": pred,
        "r2": r2_arr,
        "r2_mean": float(np.nanmean(r2_arr[eval_idx])) if eval_idx else float("nan"),
        "mode": "motor_conditioned" if conditioned else "autonomous",
        "motor_idx": motor_idx,
    }


# ── 4. Behaviour R² ─────────────────────────────────────────────────────────


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

    Matches stage2.behavior_decoder_eval.behaviour_all_neurons_prediction
    semantics: fit ridge-CV decoder on motor neuron traces → eigenworm modes.

    We evaluate TWO decoders:
    * GT decoder:    fit on u_gt motor neurons    → behaviour
    * Model decoder: fit on u_pred motor neurons  → behaviour

    Parameters
    ----------
    u_pred   : (T, N) model-predicted neural activity
    u_gt     : (T, N) ground truth neural activity
    b        : (T, L_b) behaviour targets (eigenworms)
    b_mask   : (T, L_b) validity mask
    motor_idx: indices of motor neurons in N
    n_lags   : lag steps for lagged features
    n_folds  : number of CV folds

    Returns
    -------
    dict with r2_gt, r2_model, etc.
    """
    n_modes = b.shape[1]
    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)

    # Build lagged features from motor neurons
    X_gt = build_lagged_features_np(u_gt[:, motor_idx], n_lags)
    X_pred = build_lagged_features_np(u_pred[:, motor_idx], n_lags)

    r2_gt = np.full(n_modes, np.nan)
    r2_model = np.full(n_modes, np.nan)

    for j in range(n_modes):
        valid = valid_lag_mask_np(b.shape[0], n_lags, b_mask[:, j] > 0.5)
        idx_v = np.where(valid)[0]
        if len(idx_v) < 10:
            continue

        # GT decoder
        fit_gt = _ridge_cv_single_target(X_gt, b[:, j], idx_v, ridge_grid, n_folds)
        mask_gt = np.isfinite(fit_gt["held_out"]) & (b_mask[:, j] > 0.5)
        if mask_gt.sum() > 2:
            r2_gt[j] = _r2(b[mask_gt, j], fit_gt["held_out"][mask_gt])

        # Model decoder (transfer: fit on model predictions)
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


# ── 5. All-neuron behaviour baseline ────────────────────────────────────────


def compute_behaviour_all_neurons(
    u: np.ndarray,
    b: np.ndarray,
    b_mask: np.ndarray,
    n_lags: int = 8,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """All-neuron ridge-CV behaviour decoder (upper bound baseline).

    Same as stage2.behavior_decoder_eval.behaviour_all_neurons_prediction.
    """
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


# ── Full evaluation pipeline ────────────────────────────────────────────────


def run_full_evaluation(
    model: TemporalTransformerGaussian,
    worm_data: Dict[str, Any],
    split: Dict[str, Tuple[int, int]],
    cfg: TransformerBaselineConfig,
    loo_subset_size: int = 20,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all evaluation metrics for a single worm.

    Parameters
    ----------
    model      : trained TemporalTransformerGaussian (will be set to eval)
    worm_data  : dict from load_worm_data
    split      : temporal split dict
    cfg        : config
    loo_subset_size : number of neurons for LOO (0 = all)
    verbose    : print progress

    Returns
    -------
    dict with onestep, loo, free_run, behaviour results
    """
    model.eval()
    device = next(model.parameters()).device
    u = worm_data["u"]
    T, N = u.shape
    K = cfg.context_length
    worm_id = worm_data.get("worm_id", "unknown")

    te_s, te_e = split["test"]

    if verbose:
        print(f"\n[{worm_id}] Evaluating on test region [{te_s}, {te_e})")

    # ---- 1. One-step R² (on test region) ----
    if verbose:
        print(f"  Computing one-step R²...")
    onestep = compute_onestep_r2(model, u, start=te_s, end=te_e)
    if verbose:
        print(f"  One-step R² mean = {onestep['r2_mean']:.4f}")

    # Also compute on full recording for comparison
    onestep_full = compute_onestep_r2(model, u)

    # ---- 2. LOO R² (on full recording) ----
    if verbose:
        print(f"  Computing LOO R²...")

    # Select subset: top-variance neurons
    if 0 < loo_subset_size < N:
        variances = np.nanvar(u, axis=0)
        subset = list(np.argsort(variances)[::-1][:loo_subset_size])
    else:
        subset = list(range(N))

    loo = compute_loo_r2(model, u, subset=subset, verbose=verbose)
    if verbose:
        print(f"  LOO R² mean = {loo['r2_mean']:.4f}")

    # ---- 3. Free-run R² ----
    if verbose:
        print(f"  Computing free-run R²...")
    motor_idx = worm_data.get("motor_idx")
    free_run = compute_free_run_r2(model, u, motor_idx=motor_idx)
    if verbose:
        print(f"  Free-run R² mean = {free_run['r2_mean']:.4f}  ({free_run['mode']})")

    # ---- 4. Behaviour R² ----
    beh_result = None
    beh_all_result = None
    b = worm_data.get("b")
    b_mask = worm_data.get("b_mask")

    if b is not None and b_mask is not None and motor_idx is not None:
        if verbose:
            print(f"  Computing behaviour R²...")

        # Behaviour from one-step predicted traces
        beh_result = compute_behaviour_r2(
            u_pred=onestep_full["pred"],
            u_gt=u[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
            b=b[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
            b_mask=b_mask[onestep_full["eval_range"][0] : onestep_full["eval_range"][1]],
            motor_idx=motor_idx,
            n_lags=cfg.behavior_lag_steps,
        )

        # All-neuron baseline
        beh_all_result = compute_behaviour_all_neurons(
            u=u,
            b=b,
            b_mask=b_mask,
            n_lags=cfg.behavior_lag_steps,
        )

        if verbose:
            print(f"  Behaviour R² (model): {beh_result['r2_model_mean']:.4f}")
            print(f"  Behaviour R² (GT):    {beh_result['r2_gt_mean']:.4f}")
            if beh_all_result:
                print(f"  Behaviour R² (all):   {beh_all_result['r2_all_mean']:.4f}")
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
        "behaviour_all_neurons": {
            "r2": beh_all_result["r2_all_neurons"].tolist() if beh_all_result else None,
            "r2_mean": beh_all_result["r2_all_mean"] if beh_all_result else None,
        },
    }


def save_evaluation(results: Dict[str, Any], save_dir: str, worm_id: str) -> None:
    """Save evaluation results to JSON."""
    out = Path(save_dir) / worm_id
    out.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
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
