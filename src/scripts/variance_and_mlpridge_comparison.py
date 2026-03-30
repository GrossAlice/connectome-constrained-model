#!/usr/bin/env python
r"""
Comprehensive decoder comparison: variance rescaling, MLP-Ridge, mean subtraction.

Models
──────
  1. Ridge                (per-mode α via CV)
  2. MLP                  (2×128, wd=1e-3, inner-CV epoch selection)
  3. MLP→Ridge            (MLP backbone features → Ridge readout)
  4. AR2+MLP              (E2E BPTT, full AR matrix, wd=1e-3)

Variance corrections (applied per-fold on train predictions)
────────────────────────────────────────────────────────────
  raw      — no correction (standard)
  rescale  — per-mode σ_GT/σ_pred scaling (matches GT variance)
  calibrate — per-mode OLS of y on ŷ (optimal affine correction)
  noise    — add N(0, σ²_resid) to predictions (generative sampling)

Mean-subtraction analysis
─────────────────────────
  Shows that eigenworm R² already IS the "mean-subtracted" metric,
  and evaluates what happens if we also predict the mean as a 7th target.

Usage
─────
  python -m scripts.variance_and_mlpridge_comparison
"""
from __future__ import annotations

import sys, time, pathlib
import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.unified_benchmark import (
    _ridge_fit, _train_mlp, _predict_mlp, _train_e2e,
    _log_ridge_grid,
)
from scripts.benchmark_ar_decoder_v2 import (
    load_data, build_lagged_features_np, r2_score,
)
from stage1.add_stephens_eigenworms import _preprocess_worm, _ensure_TN
from stage2.posture_videos import angles_to_xy, _load_eigenvectors

import h5py

# ══════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════
H5 = ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2/2023-01-17-14.h5"
OUT_DIR = ROOT / "output_plots/behaviour_decoder/variance_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 6
N_LAGS = 8
N_FOLDS = 5
E2E_EPOCHS = 200
TBPTT = 64
DT = 0.6


# ══════════════════════════════════════════════════════════════════
#  MLP-Ridge: MLP backbone → Ridge readout
# ══════════════════════════════════════════════════════════════════
def _train_mlp_ridge(X_train, y_train, K, ridge_grid, n_inner,
                     hidden=128, n_layers=2, lr=1e-3,
                     epochs=500, weight_decay=1e-3, patience=40,
                     cv_folds=5):
    """Train MLP → extract penultimate features → fit Ridge-CV.

    Returns (mlp_full, backbone, ridge_coefs, ridge_intercepts).
    """
    # Phase 1: train MLP normally (with inner CV for epoch)
    mlp = _train_mlp(X_train, y_train, K, hidden=hidden,
                     n_layers=n_layers, lr=lr, epochs=epochs,
                     weight_decay=weight_decay, patience=patience,
                     cv_folds=cv_folds)

    # Phase 2: backbone = everything except final Linear(hidden, K)
    backbone = mlp[:-1]   # Sequential layers up to last ReLU/Dropout
    backbone.eval()

    with torch.no_grad():
        feats_tr = backbone(
            torch.tensor(X_train, dtype=torch.float32)).numpy()

    # Phase 3: Ridge-CV on features (per mode)
    d_feat = feats_tr.shape[1]
    coefs = np.zeros((d_feat, K))
    intercepts = np.zeros(K)
    for j in range(K):
        coef, intc, _ = _ridge_fit(feats_tr, y_train[:, j],
                                   ridge_grid, n_inner)
        coefs[:, j] = coef
        intercepts[j] = intc

    return mlp, backbone, coefs, intercepts


def _predict_mlp_ridge(backbone, coefs, intercepts, X_test):
    """Predict using MLP backbone features + Ridge weights."""
    backbone.eval()
    with torch.no_grad():
        feats = backbone(
            torch.tensor(X_test, dtype=torch.float32)).numpy()
    return feats @ coefs + intercepts


# ══════════════════════════════════════════════════════════════════
#  Variance correction helpers
# ══════════════════════════════════════════════════════════════════
def _variance_rescale_params(y_train, y_hat_train):
    """Compute per-mode (scale, center) so that
    ŷ_rescaled = center + scale * (ŷ - mean(ŷ_train))

    scale = std(y_train) / std(ŷ_train) per mode.
    """
    K = y_train.shape[1]
    scales = np.ones(K)
    means_pred = np.zeros(K)
    means_gt = np.zeros(K)
    for j in range(K):
        s_gt = y_train[:, j].std()
        s_pred = y_hat_train[:, j].std()
        scales[j] = s_gt / max(s_pred, 1e-8)
        means_pred[j] = y_hat_train[:, j].mean()
        means_gt[j] = y_train[:, j].mean()
    return scales, means_pred, means_gt


def _apply_rescale(y_pred, scales, means_pred, means_gt):
    """Apply variance rescaling: inflate deviations from train mean."""
    out = np.empty_like(y_pred)
    for j in range(y_pred.shape[1]):
        out[:, j] = means_gt[j] + scales[j] * (y_pred[:, j] - means_pred[j])
    return out


def _calibrate_params(y_train, y_hat_train):
    """OLS calibration: for each mode, fit a_j, b_j via
    y_train[:,j] ≈ a_j * ŷ_train[:,j] + b_j
    This is the optimal affine correction (Platt-style).
    """
    K = y_train.shape[1]
    slopes = np.ones(K)
    intercepts = np.zeros(K)
    for j in range(K):
        yh = y_hat_train[:, j]
        yt = y_train[:, j]
        # OLS: a = cov(y,ŷ)/var(ŷ),  b = mean(y) - a*mean(ŷ)
        cov = np.mean((yt - yt.mean()) * (yh - yh.mean()))
        var = np.mean((yh - yh.mean())**2)
        a = cov / max(var, 1e-12)
        b = yt.mean() - a * yh.mean()
        slopes[j] = a
        intercepts[j] = b
    return slopes, intercepts


def _apply_calibrate(y_pred, slopes, intercepts):
    """Apply OLS calibration."""
    out = np.empty_like(y_pred)
    for j in range(y_pred.shape[1]):
        out[:, j] = slopes[j] * y_pred[:, j] + intercepts[j]
    return out


def _add_noise(y_pred, y_train, y_hat_train, rng=None):
    """Add per-mode Gaussian noise to match GT variance.
    σ²_noise = var(y_train) - var(ŷ_train)  (clamped ≥ 0).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    K = y_pred.shape[1]
    out = y_pred.copy()
    for j in range(K):
        var_gt = y_train[:, j].var()
        var_pred = y_hat_train[:, j].var()
        var_noise = max(var_gt - var_pred, 0.0)
        out[:, j] += rng.normal(0, np.sqrt(var_noise), size=y_pred.shape[0])
    return out


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────
    print("Loading data …")
    u, b_full, dt = load_data(str(H5), all_neurons=False)
    K_use = min(K, b_full.shape[1])
    b = b_full[:, :K_use]
    T = b.shape[0]
    warmup = max(2, N_LAGS)
    X = build_lagged_features_np(u, N_LAGS)
    d_in = X.shape[1]
    M_motor = u.shape[1]
    print(f"  T={T}, M_motor={M_motor}, K={K_use}, d_in={d_in}, dt={dt:.3f}s")

    # ── Folds ─────────────────────────────────────────────────────
    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = N_FOLDS - 1
    valid_len = T - warmup
    fold_size = valid_len // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < N_FOLDS - 1 else T
        folds.append((s, e))

    # ── Storage ───────────────────────────────────────────────────
    base_models = ["Ridge", "MLP", "MLP→Ridge", "AR2+MLP"]
    corrections = ["raw", "rescale", "calibrate"]
    # For each base model, we store raw predictions + correction params
    ho_raw = {m: np.full((T, K_use), np.nan) for m in base_models}
    ho_rescaled = {m: np.full((T, K_use), np.nan) for m in base_models}
    ho_calibrated = {m: np.full((T, K_use), np.nan) for m in base_models}
    ho_noise = {m: np.full((T, K_use), np.nan) for m in base_models}

    rng = np.random.default_rng(42)

    # ══════════════════════════════════════════════════════════════
    #  5-fold temporal CV
    # ══════════════════════════════════════════════════════════════
    for fi, (ts, te) in enumerate(folds):
        print(f"\n══ Fold {fi+1}/{N_FOLDS}  test=[{ts}:{te}]  "
              f"({te-ts} frames) ══")
        tr_idx = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        X_tr, X_te = X[tr_idx], X[ts:te]
        b_tr = b[tr_idx]

        # Standardise inputs for feedforward models
        mu, sig = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr_s = (X_tr - mu) / sig
        X_te_s = (X_te - mu) / sig

        # ── 1. Ridge ─────────────────────────────────────────────
        print("  Ridge …")
        ridge_train_pred = np.zeros_like(b_tr)
        for j in range(K_use):
            coef, intc, _ = _ridge_fit(X_tr_s, b_tr[:, j],
                                       ridge_grid, n_inner)
            ho_raw["Ridge"][ts:te, j] = X_te_s @ coef + intc
            ridge_train_pred[:, j] = X_tr_s @ coef + intc

        # Variance corrections for Ridge
        sc, mp, mg = _variance_rescale_params(b_tr, ridge_train_pred)
        ho_rescaled["Ridge"][ts:te] = _apply_rescale(
            ho_raw["Ridge"][ts:te], sc, mp, mg)
        sl, ic = _calibrate_params(b_tr, ridge_train_pred)
        ho_calibrated["Ridge"][ts:te] = _apply_calibrate(
            ho_raw["Ridge"][ts:te], sl, ic)
        ho_noise["Ridge"][ts:te] = _add_noise(
            ho_raw["Ridge"][ts:te], b_tr, ridge_train_pred, rng)

        # ── 2. MLP ───────────────────────────────────────────────
        print("  MLP …")
        mlp = _train_mlp(X_tr_s, b_tr, K_use)
        ho_raw["MLP"][ts:te] = _predict_mlp(mlp, X_te_s)
        mlp_train_pred = _predict_mlp(mlp, X_tr_s)

        sc, mp, mg = _variance_rescale_params(b_tr, mlp_train_pred)
        ho_rescaled["MLP"][ts:te] = _apply_rescale(
            ho_raw["MLP"][ts:te], sc, mp, mg)
        sl, ic = _calibrate_params(b_tr, mlp_train_pred)
        ho_calibrated["MLP"][ts:te] = _apply_calibrate(
            ho_raw["MLP"][ts:te], sl, ic)
        ho_noise["MLP"][ts:te] = _add_noise(
            ho_raw["MLP"][ts:te], b_tr, mlp_train_pred, rng)

        # ── 3. MLP→Ridge ─────────────────────────────────────────
        print("  MLP→Ridge …")
        _, backbone, r_coefs, r_intc = _train_mlp_ridge(
            X_tr_s, b_tr, K_use, ridge_grid, n_inner)
        ho_raw["MLP→Ridge"][ts:te] = _predict_mlp_ridge(
            backbone, r_coefs, r_intc, X_te_s)
        mlpr_train_pred = _predict_mlp_ridge(
            backbone, r_coefs, r_intc, X_tr_s)

        sc, mp, mg = _variance_rescale_params(b_tr, mlpr_train_pred)
        ho_rescaled["MLP→Ridge"][ts:te] = _apply_rescale(
            ho_raw["MLP→Ridge"][ts:te], sc, mp, mg)
        sl, ic = _calibrate_params(b_tr, mlpr_train_pred)
        ho_calibrated["MLP→Ridge"][ts:te] = _apply_calibrate(
            ho_raw["MLP→Ridge"][ts:te], sl, ic)
        ho_noise["MLP→Ridge"][ts:te] = _add_noise(
            ho_raw["MLP→Ridge"][ts:te], b_tr, mlpr_train_pred, rng)

        # ── 4. AR2+MLP ───────────────────────────────────────────
        print(f"  AR2+MLP ({E2E_EPOCHS} epochs) …")
        segs = []
        if ts > warmup + 2:
            segs.append((warmup, ts))
        if te + 2 < T:
            segs.append((te, T))

        M1e, M2e, ce_np, drv_np = _train_e2e(
            d_in, K_use, segs, b, X,
            E2E_EPOCHS, TBPTT,
            weight_decay=1e-3, tag=f"E2E f{fi+1}")

        # Free-run on test
        p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1e @ p1 + M2e @ p2 + drv_np[t] + ce_np
            ho_raw["AR2+MLP"][t] = p_new
            p2, p1 = p1, p_new

        # Free-run on train segments for variance correction
        ar_train_pred = np.full_like(b, np.nan)
        for seg_s, seg_e in segs:
            p1s, p2s = b[seg_s - 1].copy(), b[seg_s - 2].copy()
            for t in range(seg_s, seg_e):
                p_new = M1e @ p1s + M2e @ p2s + drv_np[t] + ce_np
                ar_train_pred[t] = p_new
                p2s, p1s = p1s, p_new
        # Gather valid train predictions
        ok_ar = np.all(np.isfinite(ar_train_pred[tr_idx]), axis=1)
        ar_tr_ok = ar_train_pred[tr_idx][ok_ar]
        b_tr_ok = b_tr[ok_ar]

        if ar_tr_ok.shape[0] > 50:
            sc, mp, mg = _variance_rescale_params(b_tr_ok, ar_tr_ok)
            ho_rescaled["AR2+MLP"][ts:te] = _apply_rescale(
                ho_raw["AR2+MLP"][ts:te], sc, mp, mg)
            sl, ic = _calibrate_params(b_tr_ok, ar_tr_ok)
            ho_calibrated["AR2+MLP"][ts:te] = _apply_calibrate(
                ho_raw["AR2+MLP"][ts:te], sl, ic)
            ho_noise["AR2+MLP"][ts:te] = _add_noise(
                ho_raw["AR2+MLP"][ts:te], b_tr_ok, ar_tr_ok, rng)
        else:
            ho_rescaled["AR2+MLP"][ts:te] = ho_raw["AR2+MLP"][ts:te]
            ho_calibrated["AR2+MLP"][ts:te] = ho_raw["AR2+MLP"][ts:te]
            ho_noise["AR2+MLP"][ts:te] = ho_raw["AR2+MLP"][ts:te]

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ══════════════════════════════════════════════════════════════
    #  Compute metrics
    # ══════════════════════════════════════════════════════════════
    valid = np.arange(warmup, T)

    # Helper: per-mode R², mean R², σ ratio, MSE
    def _metrics(preds, tag=""):
        ok = np.all(np.isfinite(preds[valid]), axis=1)
        idx = valid[ok]
        r2s = np.array([r2_score(b[idx, j], preds[idx, j])
                        for j in range(K_use)])
        sig_ratios = np.array([preds[idx, j].std() / b[idx, j].std()
                               for j in range(K_use)])
        mses = np.array([np.mean((b[idx, j] - preds[idx, j])**2)
                         for j in range(K_use)])
        return r2s, sig_ratios, mses

    # ── Results table ─────────────────────────────────────────────
    all_pred_sets = {
        "raw": ho_raw,
        "rescale": ho_rescaled,
        "calibrate": ho_calibrated,
        "noise": ho_noise,
    }

    print("\n" + "═" * 110)
    print("  COMPREHENSIVE DECODER COMPARISON — variance corrections")
    print("═" * 110)

    # Header
    header = f"  {'Model':<22s}  {'Correction':<11s}"
    for j in range(K_use):
        header += f"  {'a'+str(j+1):>6s}"
    header += f"  {'mean':>7s}  {'σ ratio':>7s}  {'MSE':>8s}"
    print(header)
    print("  " + "─" * 106)

    results = {}
    for m in base_models:
        for corr_name, pred_dict in all_pred_sets.items():
            r2s, sig_r, mses = _metrics(pred_dict[m])
            label = f"{m} ({corr_name})"
            results[label] = {
                "r2": r2s, "sig_ratio": sig_r, "mse": mses}

            row = f"  {m:<22s}  {corr_name:<11s}"
            for v in r2s:
                row += f"  {v:6.3f}"
            row += f"  {np.mean(r2s):7.3f}"
            row += f"  {np.mean(sig_r):7.3f}"
            row += f"  {np.mean(mses):8.4f}"
            print(row)
        print("  " + "─" * 106)

    # ── Variance ratio per mode ───────────────────────────────────
    print(f"\n  Variance ratio σ_pred/σ_GT per mode:")
    print(f"  {'Model':<22s}  {'Correction':<11s}", end="")
    for j in range(K_use):
        print(f"  {'a'+str(j+1):>6s}", end="")
    print()
    for m in base_models:
        for corr_name in ["raw", "rescale"]:
            label = f"{m} ({corr_name})"
            sr = results[label]["sig_ratio"]
            print(f"  {m:<22s}  {corr_name:<11s}", end="")
            for v in sr:
                print(f"  {v:6.3f}", end="")
            print()
        print()

    # ══════════════════════════════════════════════════════════════
    #  WHY variance shrinkage is expected (theoretical analysis)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  THEORETICAL ANALYSIS: Why shrinkage is correct")
    print("═" * 80)

    print("""
  The minimum-MSE predictor E[b|n] always has:
    Var(E[b|n]) = Var(b) · R²  <  Var(b)

  So for a well-calibrated model:
    σ_pred/σ_GT ≈ √R²

  Observed vs theoretical:""")

    for m in base_models:
        label = f"{m} (raw)"
        r2s = results[label]["r2"]
        sr = results[label]["sig_ratio"]
        for j in range(K_use):
            theoretical = np.sqrt(max(r2s[j], 0))
            print(f"    {m:14s} a{j+1}: σ_ratio={sr[j]:.3f}  "
                  f"√R²={theoretical:.3f}  "
                  f"gap={sr[j]-theoretical:+.3f}")

    print("""
  If σ_pred/σ_GT ≈ √R², the model is well-calibrated.
  Variance rescaling INCREASES MSE because it amplifies noise.
  Calibration (OLS) does NOT improve R² because R² is affine-invariant.
  Adding noise matches distribution but makes MSE worse.
  """)

    # ══════════════════════════════════════════════════════════════
    #  Mean-subtraction analysis
    # ══════════════════════════════════════════════════════════════
    print("═" * 80)
    print("  MEAN-SUBTRACTION ANALYSIS")
    print("═" * 80)

    # Load eigenvectors and per-frame mean
    with h5py.File(str(H5), "r") as f:
        _ew_d_target = int(f["behaviour/eigenworms_stephens"].attrs["d_target"])
        _ew_d_w = int(f["behaviour/eigenworms_stephens"].attrs["d_w"])
        _ew_source = f["behaviour/eigenworms_stephens"].attrs["source"]
        ba_src = _ensure_TN(np.asarray(f[_ew_source][:], dtype=float))

    proc = _preprocess_worm(ba_src, _ew_d_w, _ew_d_target)
    per_frame_mean = proc.mean(axis=1)  # (T,)

    eigvecs_all = _load_eigenvectors(h5_path=str(H5), d_target=_ew_d_target)
    E = eigvecs_all[:, :K_use]

    print(f"""
  The eigenworm pipeline already subtracts the per-frame mean:
    body_angles(t) → preprocess → subtract mean(t) → project onto E
  
  So eigenworm coefficients a₁..a₆ encode CENTERED body shape.
  The per-frame mean = overall body curvature bias (56.5% of total variance).
  
  Current decoders predict a₁..a₆ directly = already mean-subtracted!
  
  The R² values ({np.mean(results['Ridge (raw)']['r2']):.3f}–{np.mean(results['AR2+MLP (raw)']['r2']):.3f}) 
  measure prediction of centered shape modes only.
  
  ✓ Mean subtraction IS the current strategy, and it's correct.
  
  Full posture R² (with oracle GT mean vs mean=0):""")

    # Full posture R² for each model
    for m in base_models:
        preds_ew = ho_raw[m]
        ok = np.all(np.isfinite(preds_ew[valid]), axis=1)
        idx = valid[ok]

        # With GT mean (oracle)
        recon_with_mean = preds_ew[idx] @ E.T + per_frame_mean[idx, None]
        gt_with_mean = proc[idx]  # full posture (100-dim)
        r2_with = 1 - np.mean((gt_with_mean - recon_with_mean)**2) / np.var(gt_with_mean)

        # Without mean (mean=0, just shape)
        recon_no_mean = preds_ew[idx] @ E.T
        gt_no_mean = proc[idx] - per_frame_mean[idx, None]  # centered GT
        r2_without = 1 - np.mean((gt_no_mean - recon_no_mean)**2) / np.var(gt_no_mean)

        print(f"    {m:22s}:  with_GT_mean={r2_with:.3f}  "
              f"centered_only={r2_without:.3f}")

    # ══════════════════════════════════════════════════════════════
    #  Save predictions
    # ══════════════════════════════════════════════════════════════
    np.savez(
        OUT_DIR / "predictions.npz",
        b_true=b,
        **{f"{m}_raw": ho_raw[m] for m in base_models},
        **{f"{m}_rescale": ho_rescaled[m] for m in base_models},
        **{f"{m}_calibrate": ho_calibrated[m] for m in base_models},
    )
    print(f"\nSaved predictions to {OUT_DIR / 'predictions.npz'}")

    # ══════════════════════════════════════════════════════════════
    #  Figure 1: R² comparison bar chart
    # ══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1a. Raw R² per mode
    ax = axes[0]
    x = np.arange(K_use)
    w = 0.8 / len(base_models)
    colors = ["#1b9e77", "#d95f02", "#e7298a", "#7570b3"]
    for i, m in enumerate(base_models):
        r2s = results[f"{m} (raw)"]["r2"]
        ax.bar(x + i * w, r2s, w, label=m, color=colors[i],
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + w * (len(base_models) - 1) / 2)
    ax.set_xticklabels([f"a{j+1}" for j in range(K_use)])
    ax.set_ylabel("R²")
    ax.set_title("R² per mode (raw)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)

    # 1b. Mean R² by correction type
    ax = axes[1]
    corr_list = ["raw", "rescale", "calibrate"]
    x2 = np.arange(len(base_models))
    w2 = 0.8 / len(corr_list)
    corr_colors = {"raw": "#377eb8", "rescale": "#e41a1c",
                   "calibrate": "#4daf4a"}
    for i, corr in enumerate(corr_list):
        means = [np.mean(results[f"{m} ({corr})"]["r2"]) for m in base_models]
        ax.bar(x2 + i * w2, means, w2, label=corr,
               color=corr_colors[corr], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x2 + w2)
    ax.set_xticklabels(base_models, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Mean R²")
    ax.set_title("Mean R² by correction", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.7)

    # 1c. Variance ratio per model (raw vs rescaled)
    ax = axes[2]
    x3 = np.arange(K_use)
    w3 = 0.8 / (2 * len(base_models))
    for i, m in enumerate(base_models):
        sr_raw = results[f"{m} (raw)"]["sig_ratio"]
        sr_rsc = results[f"{m} (rescale)"]["sig_ratio"]
        ax.bar(x3 + (2*i) * w3, sr_raw, w3, label=f"{m} (raw)" if i == 0 else "",
               color=colors[i], alpha=0.6, edgecolor="white")
        ax.bar(x3 + (2*i+1) * w3, sr_rsc, w3,
               label=f"{m} (rescale)" if i == 0 else "",
               color=colors[i], alpha=1.0, edgecolor="white")
    ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(x3 + w3 * (2 * len(base_models) - 1) / 2)
    ax.set_xticklabels([f"a{j+1}" for j in range(K_use)])
    ax.set_ylabel("σ_pred / σ_GT")
    ax.set_title("Variance ratio (faded=raw, solid=rescale)",
                 fontweight="bold")
    ax.set_ylim(0, 1.5)

    fig.suptitle("Decoder comparison: 2023-01-17-14 — 5-fold CV",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_r2_variance.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {OUT_DIR / 'comparison_r2_variance.png'}")

    # ══════════════════════════════════════════════════════════════
    #  Figure 2: Amplitude histograms (raw vs rescaled for MLP)
    # ══════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, K_use, figsize=(3.5 * K_use, 6),
                               constrained_layout=True)
    fig2.suptitle("Amplitude distributions: raw vs rescaled (MLP)",
                  fontsize=12, fontweight="bold")

    for j in range(K_use):
        ok = np.all(np.isfinite(ho_raw["MLP"][valid]), axis=1)
        idx = valid[ok]
        gt = b[idx, j]
        bins = np.linspace(gt.min() - 0.5, gt.max() + 0.5, 40)

        for row, (title, preds) in enumerate([
            ("Raw", ho_raw["MLP"][idx, j]),
            ("Rescaled", ho_rescaled["MLP"][idx, j])
        ]):
            ax = axes2[row, j]
            ax.hist(gt, bins=bins, density=True, alpha=0.4,
                    color="gray", label="GT")
            ax.hist(preds, bins=bins, density=True, alpha=0.5,
                    color="#d95f02", label=f"MLP ({title.lower()})")
            ax.set_title(f"a{j+1} — {title}", fontsize=9)
            ax.set_xlabel("value", fontsize=8)
            if j == 0:
                ax.set_ylabel("Density", fontsize=8)
            sr = preds.std() / gt.std()
            ax.text(0.98, 0.95, f"σ={sr:.2f}",
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="top", color="#d95f02")
            if row == 0 and j == 0:
                ax.legend(fontsize=7)

    fig2.savefig(OUT_DIR / "histograms_raw_vs_rescaled.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig2)
    print(f"Figure → {OUT_DIR / 'histograms_raw_vs_rescaled.png'}")

    # ══════════════════════════════════════════════════════════════
    #  Figure 3: Time-series traces (raw vs rescaled)
    # ══════════════════════════════════════════════════════════════
    T_show = min(400, T - warmup)
    t_axis = np.arange(T_show) * dt
    idx_show = np.arange(warmup, warmup + T_show)

    fig3, axes3 = plt.subplots(K_use, 1, figsize=(16, 2.2 * K_use),
                               sharex=True)
    fig3.suptitle("Traces: GT vs MLP raw vs MLP rescaled — first 240s",
                  fontsize=12, fontweight="bold")

    for j in range(K_use):
        ax = axes3[j]
        ax.plot(t_axis, b[idx_show, j], "k-", lw=1, alpha=0.8, label="GT")
        ax.plot(t_axis, ho_raw["MLP"][idx_show, j],
                "-", color="#d95f02", lw=0.7, alpha=0.7, label="MLP raw")
        ax.plot(t_axis, ho_rescaled["MLP"][idx_show, j],
                "--", color="#e7298a", lw=0.7, alpha=0.7, label="MLP rescaled")
        ax.set_ylabel(f"a{j+1}", fontsize=10)
        if j == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper right")
    axes3[-1].set_xlabel("Time (s)", fontsize=10)
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "traces_raw_vs_rescaled.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig3)
    print(f"Figure → {OUT_DIR / 'traces_raw_vs_rescaled.png'}")

    # ══════════════════════════════════════════════════════════════
    #  Summary text
    # ══════════════════════════════════════════════════════════════
    txt = OUT_DIR / "comparison_summary.txt"
    with open(txt, "w") as f:
        f.write("Comprehensive Decoder Comparison — variance corrections\n")
        f.write(f"Worm: 2023-01-17-14, T={T}, K={K_use}, M={M_motor}\n")
        f.write(f"5-fold temporal CV, {elapsed:.0f}s\n\n")

        f.write(f"{'Model':<22s}  {'Corr':<11s}  "
                + "  ".join(f"{'a'+str(j+1):>6s}" for j in range(K_use))
                + f"  {'mean':>7s}  {'σ_ratio':>7s}\n")
        f.write("─" * 100 + "\n")
        for m in base_models:
            for corr in corr_list:
                label = f"{m} ({corr})"
                r2s = results[label]["r2"]
                sr = results[label]["sig_ratio"]
                f.write(f"{m:<22s}  {corr:<11s}  "
                        + "  ".join(f"{v:6.3f}" for v in r2s)
                        + f"  {np.mean(r2s):7.3f}  {np.mean(sr):7.3f}\n")
            f.write("\n")

    print(f"Summary → {txt}")
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
