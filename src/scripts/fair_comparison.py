#!/usr/bin/env python
"""Fair decoder comparison: per-timestep vs free-run, with optional LP filter.

Models (all evaluated on a₁–a₆):
  1. Ridge    (per-timestep) — upper bound, no error compounding
  2. MLP      (per-timestep) — nonlinear upper bound
  3. VAR₂+R   (2-step fit, free-run) — fair free-run baseline
  4. E2E+L2   (BPTT, free-run) — end-to-end trained

Optional:  --lpf_hz <freq>  applies a Butterworth low-pass to the
posture targets *before* fitting.  Since calcium imaging at dt=0.6s
acts as a ~1s LPF, neural features cannot predict posture fluctuations
faster than ~0.5 Hz.  Filtering removes unpredictable HF variance,
giving a fairer R².

Usage:
    python -m scripts.fair_comparison \
        --h5 "data/used/.../2023-01-17-14.h5" \
        --lpf_hz 0.3
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
    build_lagged_features_np,
)
try:
    from scripts.benchmark_ar_decoder_v2 import (
        load_data, build_ar_features, r2_score,
    )
except ModuleNotFoundError:
    from benchmark_ar_decoder_v2 import (
        load_data, build_ar_features, r2_score,
    )
from scripts.unified_benchmark import (
    _ridge_fit, _train_mlp, _predict_mlp, _train_e2e,
)


# ──────────────────────────────────────────────────────────────────────
#  Per-neuron-group Ridge
# ──────────────────────────────────────────────────────────────────────
def _ridge_per_neuron(X_tr_s, y_tr, X_te_s, n_neurons, n_lags_p1,
                      ridge_grid, n_inner, K):
    """Fit K per-neuron Ridge models (one λ per neuron per mode) and
    stack predictions.

    Each neuron contributes *n_lags_p1* consecutive columns in X.
    For each mode j, for each neuron n:
      - fit Ridge on X_tr[:, cols_n] → y_tr[:, j] − Σ_{m≠n} pred_m
    This is expensive, so instead we use a simpler approach:
    fit separate Ridge per neuron-group and combine via stacking.

    Approach: 2-level stacking
      Level 1: for each neuron n, fit Ridge on X[:, cols_n] → y[:, j]
               get held-out predictions P_n (T_test, K)
      Level 2: Ridge on [P_1 ... P_N] → y  to combine
    """
    preds_te = np.zeros((X_te_s.shape[0], K))

    for j in range(K):
        # Level 1: per-neuron Ridge → get train predictions (via inner CV)
        # and test predictions
        l1_train = np.zeros((X_tr_s.shape[0], n_neurons))
        l1_test = np.zeros((X_te_s.shape[0], n_neurons))

        for n in range(n_neurons):
            cols = slice(n * n_lags_p1, (n + 1) * n_lags_p1)
            Xn_tr = X_tr_s[:, cols]
            Xn_te = X_te_s[:, cols]
            coef, intc, _ = _ridge_fit(Xn_tr, y_tr[:, j],
                                       ridge_grid, n_inner)
            l1_train[:, n] = Xn_tr @ coef + intc
            l1_test[:, n] = Xn_te @ coef + intc

        # Level 2: Ridge on stacked per-neuron predictions
        coef2, intc2, _ = _ridge_fit(l1_train, y_tr[:, j],
                                     ridge_grid, n_inner)
        preds_te[:, j] = l1_test @ coef2 + intc2

    return preds_te


# ──────────────────────────────────────────────────────────────────────
#  Low-pass filter
# ──────────────────────────────────────────────────────────────────────
def _butter_lowpass(data: np.ndarray, cutoff_hz: float, dt: float,
                    order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth LPF to each column of data (T×K)."""
    from scipy.signal import butter, sosfiltfilt
    fs = 1.0 / dt
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return data  # nothing to filter
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    out = np.empty_like(data)
    for j in range(data.shape[1]):
        out[:, j] = sosfiltfilt(sos, data[:, j])
    return out


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--neural_lags", type=int, default=8)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_modes", type=int, default=6)
    ap.add_argument("--e2e_epochs", type=int, default=300)
    ap.add_argument("--tbptt_chunk", type=int, default=64)
    ap.add_argument("--lpf_hz", type=float, default=0.0,
                    help="Butterworth LPF cutoff for posture targets "
                         "(Hz, 0=off). Try 0.2–0.5.")
    ap.add_argument("--out_dir",
                    default="output_plots/behaviour_decoder/fair_comparison")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────
    u, b_full, dt = load_data(args.h5, all_neurons=False)
    K = min(args.n_modes, b_full.shape[1])
    b_raw = b_full[:, :K].copy()
    T = b_raw.shape[0]
    n_lags = args.neural_lags
    warmup = max(2, n_lags)
    M_raw = u.shape[1]
    X_neural = build_lagged_features_np(u, n_lags)
    d_in = X_neural.shape[1]

    # ── Optional LP filter ────────────────────────────────────────────
    if args.lpf_hz > 0:
        b = _butter_lowpass(b_raw, args.lpf_hz, dt)
        # Show how much variance was removed
        var_raw = np.var(b_raw, axis=0)
        var_filt = np.var(b, axis=0)
        pct = 100.0 * (1.0 - var_filt / var_raw)
        print(f"\n  LPF @ {args.lpf_hz:.2f} Hz  (Nyquist={0.5/dt:.2f} Hz)")
        for j in range(K):
            print(f"    a{j+1}: var removed = {pct[j]:.1f}%")
    else:
        b = b_raw

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = args.n_folds - 1

    print(f"\n  Data: T={T}, K={K}, M={M_raw}, "
          f"d_features={d_in} ({M_raw}×{n_lags+1}), dt={dt:.2f}s")
    if args.lpf_hz > 0:
        print(f"  LPF cutoff: {args.lpf_hz:.2f} Hz")
    print(f"  CV: {args.n_folds} outer folds, {n_inner} inner folds")

    # ── Folds ─────────────────────────────────────────────────────────
    valid_len = T - warmup
    fold_size = valid_len // args.n_folds
    folds = []
    for i in range(args.n_folds):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < args.n_folds - 1 else T
        folds.append((s, e))

    # ── Models ────────────────────────────────────────────────────────
    M_RIDGE  = "Ridge (per-t)"
    M_RIDGN  = "Ridge/neuron (per-t)"
    M_MLP    = "MLP (per-t)"
    M_VAR2R  = "VAR₂+Ridge (free)"
    M_E2EL2  = "E2E+L2 (free)"
    M_RIDGE_RAW = "Ridge-raw (per-t)"   # trains on raw b even when LPF is on
    all_models = [M_RIDGE, M_RIDGN, M_MLP, M_VAR2R, M_E2EL2]
    if args.lpf_hz > 0:
        all_models.insert(1, M_RIDGE_RAW)  # right after Ridge
    ho_preds = {m: np.full((T, K), np.nan) for m in all_models}

    n_models = len(all_models)
    print(f"\n{'═'*80}")
    print(f"  {n_models} models × {args.n_folds} folds")
    print(f"{'═'*80}\n")

    for fi, (ts, te) in enumerate(folds):
        test_len = te - ts
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[ts:te] = False
        train_idx = np.where(train_mask)[0]

        X_tr = X_neural[train_idx]
        X_te = X_neural[ts:te]
        b_tr = b[train_idx]

        # Standardise
        mu, sigma = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr_s = (X_tr - mu) / sigma
        X_te_s = (X_te - mu) / sigma

        print(f"  ═══ Fold {fi+1}/{args.n_folds}: "
              f"test=[{ts}:{te}) ({test_len} fr) ═══")

        # ── 1. Ridge (per-timestep) ──────────────────────────────────
        print(f"    Ridge (per-t)")
        for j in range(K):
            coef, intc, _ = _ridge_fit(X_tr_s, b_tr[:, j],
                                       ridge_grid, n_inner)
            ho_preds[M_RIDGE][ts:te, j] = X_te_s @ coef + intc

        # ── 1b. Ridge-raw (per-timestep, trained on raw b) ────────────
        if args.lpf_hz > 0:
            print(f"    Ridge-raw (per-t, trained on raw b)")
            b_raw_tr = b_raw[train_idx]
            for j in range(K):
                coef, intc, _ = _ridge_fit(X_tr_s, b_raw_tr[:, j],
                                           ridge_grid, n_inner)
                ho_preds[M_RIDGE_RAW][ts:te, j] = X_te_s @ coef + intc

        # ── 1c. Ridge per-neuron (per-timestep, stacked) ─────────────
        print(f"    Ridge/neuron (per-t, 2-level stacking)")
        n_lags_p1 = n_lags + 1   # features per neuron
        ho_preds[M_RIDGN][ts:te] = _ridge_per_neuron(
            X_tr_s, b_tr, X_te_s, M_raw, n_lags_p1,
            ridge_grid, n_inner, K)

        # ── 2. MLP (per-timestep) ────────────────────────────────────
        print(f"    MLP (per-t)")
        mlp = _train_mlp(X_tr_s, b_tr, K)
        ho_preds[M_MLP][ts:te] = _predict_mlp(mlp, X_te_s)

        # ── 3. VAR₂+Ridge (2-step, free-run) ────────────────────────
        print(f"    VAR₂+Ridge (free-run)")
        X_ar = build_ar_features(b, 2)
        X_ar_tr = X_ar[train_idx]

        M1 = np.zeros((K, K)); M2 = np.zeros((K, K)); c_v = np.zeros(K)
        for j in range(K):
            coef, intc, _ = _ridge_fit(X_ar_tr, b_tr[:, j],
                                       ridge_grid, n_inner)
            M1[j] = coef[:K]
            M2[j] = coef[K:2*K]
            c_v[j] = intc

        # Residual drive (Ridge)
        b_tm1 = np.zeros_like(b); b_tm1[1:] = b[:-1]
        b_tm2 = np.zeros_like(b); b_tm2[2:] = b[:-2]
        resid = b - (b_tm1 @ M1.T + b_tm2 @ M2.T + c_v)
        r_tr = resid[train_idx]

        coefs_d = np.zeros((d_in, K)); ints_d = np.zeros(K)
        for j in range(K):
            coefs_d[:, j], ints_d[j], _ = _ridge_fit(
                X_tr_s, r_tr[:, j], ridge_grid, n_inner)

        drive_all = X_neural @ coefs_d + ints_d[None, :]
        # Standardize drive features were already from standardized X
        # Actually we need drive from standardized features
        drive_all_s = ((X_neural - mu) / sigma) @ coefs_d + ints_d[None, :]

        p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1 @ p1 + M2 @ p2 + drive_all_s[t] + c_v
            ho_preds[M_VAR2R][t] = p_new
            p2, p1 = p1, p_new

        # ── 4. E2E+L2 (BPTT, free-run) ──────────────────────────────
        print(f"    E2E+L2 ({args.e2e_epochs} ep)")
        segs = []
        if ts > warmup + 2:
            segs.append((warmup, ts))
        if te + 2 < T:
            segs.append((te, T))
        M1e, M2e, ce_np, drv_np = _train_e2e(
            d_in, K, segs, b, X_neural, args.e2e_epochs,
            args.tbptt_chunk, weight_decay=1e-3,
            tag=f"E2E+L2 f{fi+1}")
        p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1e @ p1 + M2e @ p2 + drv_np[t] + ce_np
            ho_preds[M_E2EL2][t] = p_new
            p2, p1 = p1, p_new

        print()

    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════════════
    #  R²  — always evaluate against the *raw* (unfiltered) posture
    #  so that LPF only affects training targets, not evaluation.
    # ══════════════════════════════════════════════════════════════════
    valid = np.arange(warmup, T)
    results = {}
    for name in all_models:
        preds = ho_preds[name]
        ok = np.isfinite(preds[valid, 0])
        idx = valid[ok]
        if idx.size < 10:
            results[name] = np.full(K, np.nan)
        else:
            results[name] = np.array([
                r2_score(b_raw[idx, j], preds[idx, j]) for j in range(K)])

    # If we filtered, also report R² against the *filtered* target
    if args.lpf_hz > 0:
        results_filt = {}
        for name in all_models:
            preds = ho_preds[name]
            ok = np.isfinite(preds[valid, 0])
            idx = valid[ok]
            if idx.size < 10:
                results_filt[name] = np.full(K, np.nan)
            else:
                results_filt[name] = np.array([
                    r2_score(b[idx, j], preds[idx, j]) for j in range(K)])

    # ══════════════════════════════════════════════════════════════════
    #  Report
    # ══════════════════════════════════════════════════════════════════
    mode_names = [f"a{j+1}" for j in range(K)]
    cw = 28

    print("\n" + "═" * 80)
    print("  R² vs RAW posture (unfiltered ground truth)")
    print("═" * 80)
    header = (f"  {'Model':<{cw}s}"
              + "".join(f"{m:>8s}" for m in mode_names)
              + f"{'mean':>8s}")
    print(header)
    print("  " + "─" * (cw + 8 * (K + 1)))
    for name in all_models:
        vals = results[name]
        mn = np.nanmean(vals)
        parts = [f"  {name:<{cw}s}"]
        for v in vals:
            parts.append(f"{v:8.3f}" if np.isfinite(v) else f"{'---':>8s}")
        parts.append(f"{mn:8.3f}")
        print("".join(parts))

    if args.lpf_hz > 0:
        print(f"\n  R² vs FILTERED posture (LPF @ {args.lpf_hz:.2f} Hz)")
        print("  " + "─" * (cw + 8 * (K + 1)))
        for name in all_models:
            vals = results_filt[name]
            mn = np.nanmean(vals)
            parts = [f"  {name:<{cw}s}"]
            for v in vals:
                parts.append(f"{v:8.3f}" if np.isfinite(v) else f"{'---':>8s}")
            parts.append(f"{mn:8.3f}")
            print("".join(parts))

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ══════════════════════════════════════════════════════════════════
    #  Plot: grouped bar chart
    # ══════════════════════════════════════════════════════════════════
    colors = {
        M_RIDGE: "#1f77b4", M_RIDGN: "#17becf",
        M_MLP: "#ff7f0e",
        M_VAR2R: "#2ca02c", M_E2EL2: "#d62728",
        M_RIDGE_RAW: "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(K)
    n = len(all_models)
    w = 0.8 / n
    for i, name in enumerate(all_models):
        vals = np.clip(results[name], -0.3, 1.0)
        ax.bar(x + i * w, vals, w, label=name,
               color=colors[name], edgecolor="white", linewidth=0.5)
        for j in range(K):
            ax.text(x[j] + i * w, vals[j] + 0.02, f"{vals[j]:.3f}",
                    ha="center", fontsize=6.5, color=colors[name],
                    fontweight="bold")

    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(mode_names, fontsize=12)
    ax.set_ylabel("R²  (vs raw posture)", fontsize=12)
    lpf_tag = f", LPF={args.lpf_hz:.1f}Hz" if args.lpf_hz > 0 else ""
    ax.set_title(f"Per-timestep vs Free-run — 5-fold CV{lpf_tag}",
                 fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    fig.tight_layout()

    tag = f"_lpf{args.lpf_hz:.1f}" if args.lpf_hz > 0 else ""
    fig_path = out_dir / f"fair_comparison{tag}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure → {fig_path}")

    # ── Text ──────────────────────────────────────────────────────────
    txt_path = out_dir / f"fair_comparison{tag}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Fair decoder comparison — 5-fold temporal CV\n")
        f.write(f"h5: {args.h5}\n")
        f.write(f"neural_lags={n_lags}, K={K}, n_folds={args.n_folds}\n")
        if args.lpf_hz > 0:
            f.write(f"LPF cutoff: {args.lpf_hz:.2f} Hz\n")
        f.write(f"\nR² vs raw posture:\n")
        for name in all_models:
            vals = results[name]
            f.write(f"  {name:<28s}: "
                    + " ".join(f"{v:7.3f}" for v in vals)
                    + f"  mean={np.nanmean(vals):.3f}\n")
        if args.lpf_hz > 0:
            f.write(f"\nR² vs filtered posture (LPF@{args.lpf_hz:.2f}Hz):\n")
            for name in all_models:
                vals = results_filt[name]
                f.write(f"  {name:<28s}: "
                        + " ".join(f"{v:7.3f}" for v in vals)
                        + f"  mean={np.nanmean(vals):.3f}\n")
    print(f"  Summary → {txt_path}")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
