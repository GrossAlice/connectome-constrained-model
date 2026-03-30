#!/usr/bin/env python
r"""
AR2+MLP weight-decay CV analysis (analogous to Ridge α-CV).

For each outer fold:
  1. Split training segments into inner-train (80%) / inner-val (20%)
  2. Train AR2+MLP with each wd in grid on inner-train
  3. Free-run on inner-val → compute MSE
  4. Select best wd
  5. Retrain on full training with best wd → free-run on test

Reports: selected wd per fold, boundary hits, R² comparison with fixed wd=1e-3.
"""
from __future__ import annotations

import argparse, random, sys, time, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.unified_benchmark import _train_e2e
from scripts.benchmark_ar_decoder_v2 import (
    load_data, build_lagged_features_np, r2_score,
)

# ══════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════
DATA_DIR = ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2"
OUT_BASE = ROOT / "output_plots/behaviour_decoder/ar2mlp_wd_cv"

K = 6
N_LAGS = 8
N_FOLDS = 5
SEARCH_EPOCHS = 150   # fewer epochs for grid search (speed)
FINAL_EPOCHS = 200    # full training after wd is chosen
TBPTT = 64

# Weight-decay grid (log-spaced, plus 0)
WD_GRID = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def _free_run_mse(M1, M2, c, drv, b, seg_start, seg_end,
                  mode_weights=None):
    """Free-run on [seg_start, seg_end) and return MSE vs ground truth.

    Returns np.inf if the recurrence diverges.
    *mode_weights*: optional (K,) array for variance-weighted MSE.
    """
    K = b.shape[1]
    if mode_weights is None:
        mode_weights = np.ones(K)
    p1, p2 = b[seg_start - 1].copy(), b[seg_start - 2].copy()
    mse_acc, n = 0.0, 0
    for t in range(seg_start, seg_end):
        p_new = M1 @ p1 + M2 @ p2 + drv[t] + c
        if not np.all(np.isfinite(p_new)) or np.max(np.abs(p_new)) > 1e6:
            return np.inf                       # divergence sentinel
        mse_acc += np.mean(mode_weights * (p_new - b[t]) ** 2)
        n += 1
        p2, p1 = p1, p_new
    return mse_acc / max(n, 1)


def _free_run_preds(M1, M2, c, drv, b, seg_start, seg_end):
    """Free-run on [seg_start, seg_end) and return predictions.

    Fills remaining frames with NaN if the recurrence diverges.
    """
    T_seg = seg_end - seg_start
    K = b.shape[1]
    out = np.full((T_seg, K), np.nan)
    p1, p2 = b[seg_start - 1].copy(), b[seg_start - 2].copy()
    for i, t in enumerate(range(seg_start, seg_end)):
        p_new = M1 @ p1 + M2 @ p2 + drv[t] + c
        if not np.all(np.isfinite(p_new)) or np.max(np.abs(p_new)) > 1e6:
            break                               # leave rest as NaN
        out[i] = p_new
        p2, p1 = p1, p_new
    return out


def main():
    parser = argparse.ArgumentParser(description="AR2+MLP weight-decay CV")
    parser.add_argument("--worm", default="2023-01-17-14",
                        help="Worm ID (h5 stem), default=2023-01-17-14")
    args = parser.parse_args()

    H5 = DATA_DIR / f"{args.worm}.h5"
    OUT_DIR = OUT_BASE / args.worm
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Reproducibility ───────────────────────────────────────────
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # ── Load data ─────────────────────────────────────────────────
    print("Loading data …")
    u, b_full, dt = load_data(str(H5), all_neurons=False)
    K_use = min(K, b_full.shape[1])
    b = b_full[:, :K_use]
    T = b.shape[0]
    warmup = max(2, N_LAGS)
    X = build_lagged_features_np(u, N_LAGS)
    d_in = X.shape[1]
    print(f"  T={T}, M_motor={u.shape[1]}, K={K_use}, d_in={d_in}, "
          f"dt={dt:.3f}s")
    print(f"  WD grid ({len(WD_GRID)} values): {WD_GRID}")
    print(f"  Search epochs: {SEARCH_EPOCHS}, final epochs: {FINAL_EPOCHS}")

    # ── Folds ─────────────────────────────────────────────────────
    valid_len = T - warmup
    fold_size = valid_len // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < N_FOLDS - 1 else T
        folds.append((s, e))

    # ── Storage ───────────────────────────────────────────────────
    ho_cv = np.full((T, K_use), np.nan)      # AR2+MLP with wd-CV
    ho_fixed = np.full((T, K_use), np.nan)   # AR2+MLP with wd=1e-3 (baseline)

    fold_results = []

    # ══════════════════════════════════════════════════════════════
    #  5-fold outer CV
    # ══════════════════════════════════════════════════════════════
    for fi, (ts, te) in enumerate(folds):
        print(f"\n{'═'*70}")
        print(f"  Fold {fi+1}/{N_FOLDS}  test=[{ts}:{te})  "
              f"({te-ts} frames)")
        print(f"{'═'*70}")

        # Training segments (contiguous blocks outside test)
        # Post-test segment starts +2 so AR(2) seeds don't touch test data
        train_segs = []
        if ts > warmup:
            train_segs.append((warmup, ts))
        post_start = te + 2          # skip 2 buffer frames (AR seed leakage)
        if post_start + 4 < T:       # need ≥ 4 usable frames
            train_segs.append((post_start, T))

        # ── Inner train/val split (80/20 of each training segment) ──
        inner_train_segs = []
        inner_val_segs = []
        total_train_frames = sum(e - s for s, e in train_segs)

        for seg_s, seg_e in train_segs:
            seg_len = seg_e - seg_s
            split_pt = seg_s + int(0.8 * seg_len)
            # Need at least 3 frames in each part for AR(2)
            if split_pt - seg_s > 4 and seg_e - split_pt > 4:
                inner_train_segs.append((seg_s, split_pt))
                inner_val_segs.append((split_pt, seg_e))
            else:
                inner_train_segs.append((seg_s, seg_e))

        assert len(inner_val_segs) > 0, (
            f"Fold {fi}: all training segments too short for inner val "
            f"split (train_segs={train_segs})")
        inner_train_frames = sum(e - s for s, e in inner_train_segs)
        inner_val_frames = sum(e - s for s, e in inner_val_segs)
        print(f"  Inner split: train={inner_train_frames} frames, "
              f"val={inner_val_frames} frames")
        print(f"  Train segs: {train_segs}")
        print(f"  Inner train: {inner_train_segs}")
        print(f"  Inner val:   {inner_val_segs}")

        # Per-mode inverse variance for weighted MSE (#7)
        train_idx = np.concatenate([np.arange(s, e)
                                    for s, e in inner_train_segs])
        train_var = np.array([b[train_idx, j].var()
                              for j in range(K_use)])
        inv_train_var = 1.0 / np.maximum(train_var, 1e-8)

        # ── Grid search over weight_decay ──────────────────────────
        wd_mses = {}
        print(f"\n  wd search ({len(WD_GRID)} values × "
              f"{SEARCH_EPOCHS} epochs):")

        for wd in WD_GRID:
            M1, M2, c, drv = _train_e2e(
                d_in, K_use, inner_train_segs, b, X,
                SEARCH_EPOCHS, TBPTT,
                weight_decay=wd, patience=SEARCH_EPOCHS + 1,
                tag=f"  wd={wd:.0e} f{fi+1}")

            # Free-run on inner val segments (variance-weighted MSE)
            val_mse_total, val_n = 0.0, 0
            for vs, ve in inner_val_segs:
                mse_seg = _free_run_mse(M1, M2, c, drv, b, vs, ve,
                                        mode_weights=inv_train_var)
                val_mse_total += mse_seg * (ve - vs)
                val_n += ve - vs
            val_mse = val_mse_total / max(val_n, 1)
            wd_mses[wd] = val_mse
            print(f"    wd={wd:10.1e}  val_MSE={val_mse:.5f}")

        # ── Select best wd ─────────────────────────────────────────
        best_wd = min(wd_mses, key=wd_mses.get)
        best_idx = WD_GRID.index(best_wd)
        at_lower = (best_idx == 0)
        at_upper = (best_idx == len(WD_GRID) - 1)

        print(f"\n  ★ Best wd = {best_wd:.1e}  "
              f"(idx={best_idx}/{len(WD_GRID)-1}  "
              f"val_MSE={wd_mses[best_wd]:.5f})")
        if at_lower:
            print(f"  ⚠ HIT LOWER BOUNDARY (wd=0)")
        if at_upper:
            print(f"  ⚠ HIT UPPER BOUNDARY (wd={WD_GRID[-1]:.0e})")
        if not at_lower and not at_upper:
            print(f"  ✓ Not at boundary")

        fold_results.append({
            "fold": fi, "best_wd": best_wd, "best_idx": best_idx,
            "at_lower": at_lower, "at_upper": at_upper,
            "wd_mses": dict(wd_mses),
        })

        # ── Retrain with best wd on full training segments ─────────
        print(f"\n  Retrain with best wd={best_wd:.1e} "
              f"({FINAL_EPOCHS} epochs) …")
        M1_cv, M2_cv, c_cv, drv_cv = _train_e2e(
            d_in, K_use, train_segs, b, X,
            FINAL_EPOCHS, TBPTT,
            weight_decay=best_wd, tag=f"CV-best f{fi+1}")

        # Free-run on test
        preds_cv = _free_run_preds(M1_cv, M2_cv, c_cv, drv_cv, b, ts, te)
        ho_cv[ts:te] = preds_cv

        # ── Also train with fixed wd=1e-3 for comparison ──────────
        print(f"  Retrain with fixed wd=1e-3 ({FINAL_EPOCHS} epochs) …")
        M1_f, M2_f, c_f, drv_f = _train_e2e(
            d_in, K_use, train_segs, b, X,
            FINAL_EPOCHS, TBPTT,
            weight_decay=1e-3, tag=f"Fixed f{fi+1}")

        preds_f = _free_run_preds(M1_f, M2_f, c_f, drv_f, b, ts, te)
        ho_fixed[ts:te] = preds_f

        # Per-fold R² for confidence estimates (#9)
        ok_cv = np.all(np.isfinite(preds_cv), axis=1)
        fold_r2_cv = (np.array([r2_score(b[ts:te, j][ok_cv], preds_cv[ok_cv, j])
                                for j in range(K_use)])
                      if ok_cv.any() else np.full(K_use, np.nan))
        ok_fx = np.all(np.isfinite(preds_f), axis=1)
        fold_r2_fixed = (np.array([r2_score(b[ts:te, j][ok_fx], preds_f[ok_fx, j])
                                   for j in range(K_use)])
                         if ok_fx.any() else np.full(K_use, np.nan))
        fold_results[-1]["r2_cv_fold"] = fold_r2_cv.tolist()
        fold_results[-1]["r2_fixed_fold"] = fold_r2_fixed.tolist()

    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════════
    #  Compute R²
    # ══════════════════════════════════════════════════════════════
    valid = np.arange(warmup, T)

    def _r2_per_mode(preds):
        ok = np.all(np.isfinite(preds[valid]), axis=1)
        idx = valid[ok]
        return np.array([r2_score(b[idx, j], preds[idx, j])
                         for j in range(K_use)])

    r2_cv = _r2_per_mode(ho_cv)
    r2_fixed = _r2_per_mode(ho_fixed)

    # ══════════════════════════════════════════════════════════════
    #  Report
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  AR2+MLP WEIGHT-DECAY CV ANALYSIS")
    print(f"{'═'*80}")

    print(f"\n  WD grid: {WD_GRID}")

    print(f"\n  Per-fold best weight_decay:")
    n_lower, n_upper = 0, 0
    for fr in fold_results:
        flag = ""
        if fr["at_lower"]:
            flag = "  ⚠ LOWER"
            n_lower += 1
        if fr["at_upper"]:
            flag = "  ⚠ UPPER"
            n_upper += 1
        print(f"    Fold {fr['fold']+1}:  wd = {fr['best_wd']:.1e}{flag}")

    best_wds = [fr["best_wd"] for fr in fold_results]
    print(f"\n  wd median: {np.median(best_wds):.1e}  "
          f"range: [{min(best_wds):.1e}, {max(best_wds):.1e}]")

    if n_lower > 0:
        print(f"  ⚠ {n_lower}/{N_FOLDS} folds hit LOWER boundary (wd=0)")
    if n_upper > 0:
        print(f"  ⚠ {n_upper}/{N_FOLDS} folds hit UPPER boundary "
              f"(wd={WD_GRID[-1]:.0e})")
    if n_lower == 0 and n_upper == 0:
        print(f"  ✓ No boundary hits across {N_FOLDS} folds")

    print(f"\n  R² comparison (per mode):")
    header = f"    {'Model':<25s}" + "".join(f"  {'a'+str(j+1):>6s}"
                                             for j in range(K_use))
    header += f"  {'mean':>7s}"
    print(header)
    print(f"    {'─'*60}")
    for name, r2s in [("AR2+MLP (wd-CV)", r2_cv),
                      ("AR2+MLP (wd=1e-3 fixed)", r2_fixed)]:
        row = f"    {name:<25s}"
        for v in r2s:
            row += f"  {v:6.3f}"
        row += f"  {np.mean(r2s):7.3f}"
        print(row)

    # ── MSE landscape per fold ─────────────────────────────────────
    print(f"\n  Inner-val MSE landscape per fold (× 1000):")
    header = f"    {'wd':>10s}"
    for fi in range(N_FOLDS):
        header += f"  {'f'+str(fi+1):>8s}"
    header += f"  {'mean':>8s}"
    print(header)
    print(f"    {'─'*65}")

    for wd in WD_GRID:
        row = f"    {wd:10.1e}"
        vals = [fold_results[fi]["wd_mses"][wd] * 1000
                for fi in range(N_FOLDS)]
        for v in vals:
            row += f"  {v:8.2f}"
        row += f"  {np.mean(vals):8.2f}"
        star = " ←" if wd == np.median(best_wds) else ""
        print(row + star)

    # ── Per-fold R² with confidence (#9) ─────────────────────────
    fold_r2s_cv = np.array([fr["r2_cv_fold"] for fr in fold_results])
    fold_r2s_fixed = np.array([fr["r2_fixed_fold"] for fr in fold_results])
    diff = fold_r2s_cv.mean(axis=1) - fold_r2s_fixed.mean(axis=1)
    se_diff = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"\n  Per-fold mean R\u00b2 (CV \u2212 fixed): "
          f"{diff.mean():.4f} \u00b1 {se_diff:.4f} (SE, n={N_FOLDS})")
    for fi_r, fr in enumerate(fold_results):
        cv_m = np.mean(fr["r2_cv_fold"])
        fx_m = np.mean(fr["r2_fixed_fold"])
        print(f"    Fold {fi_r+1}: CV={cv_m:.3f}  "
              f"fixed={fx_m:.3f}  \u0394={cv_m - fx_m:+.3f}")

    # ── Variance ratio ─────────────────────────────────────────────
    ok = np.all(np.isfinite(ho_cv[valid]), axis=1)
    idx = valid[ok]
    print(f"\n  Variance ratio σ_pred/σ_GT:")
    for name, preds in [("wd-CV", ho_cv), ("wd=1e-3", ho_fixed)]:
        ok2 = np.all(np.isfinite(preds[valid]), axis=1)
        idx2 = valid[ok2]
        srs = [preds[idx2, j].std() / b[idx2, j].std()
               for j in range(K_use)]
        print(f"    {name:<15s}: "
              + "  ".join(f"a{j+1}={sr:.3f}" for j, sr in enumerate(srs))
              + f"  mean={np.mean(srs):.3f}")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Save ──────────────────────────────────────────────────────
    np.savez(OUT_DIR / "predictions.npz",
             b_true=b, ho_cv=ho_cv, ho_fixed=ho_fixed,
             wd_grid=np.array(WD_GRID),
             best_wds=np.array(best_wds))

    import json
    summary = {
        "seed": SEED,
        "wd_grid": WD_GRID,
        "fold_results": [
            {"fold": fr["fold"],
             "best_wd": fr["best_wd"],
             "best_idx": fr["best_idx"],
             "at_lower": fr["at_lower"],
             "at_upper": fr["at_upper"],
             "wd_mses": {str(k): v for k, v in fr["wd_mses"].items()},
             "r2_cv_fold": fr["r2_cv_fold"],
             "r2_fixed_fold": fr["r2_fixed_fold"]}
            for fr in fold_results
        ],
        "r2_cv": r2_cv.tolist(),
        "r2_fixed": r2_fixed.tolist(),
        "r2_cv_mean": float(np.mean(r2_cv)),
        "r2_fixed_mean": float(np.mean(r2_fixed)),
    }
    with open(OUT_DIR / "wd_cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {OUT_DIR / 'wd_cv_summary.json'}")


if __name__ == "__main__":
    main()
