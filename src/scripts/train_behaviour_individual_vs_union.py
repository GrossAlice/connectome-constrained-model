#!/usr/bin/env python
"""Train behaviour decoders: individual per-worm vs union+zero-fill.

Two models are compared:
  A) *Individual*  – one ridge-CV decoder per worm (local motor neurons only).
     Reports R² distribution across worms.
  B) *Union+zero-fill* – one decoder trained on concatenated data from all
     worms.  Motor-neuron features are aligned to the union set; missing
     neurons are zero-filled.  The decoder is evaluated per-worm (unseen
     time-points within each worm via blocked CV).

Both models use raw GCaMP motor-neuron traces (no stage-1 deconvolution
needed) and lagged features.

Usage
-----
    python scripts/train_behaviour_individual_vs_union.py \
        --worms 2022-08-02-01 2023-01-23-21 2023-01-10-07 \
               2023-01-19-08 2023-01-19-15 \
        --lag 8 --n_folds 5
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# --------------------------------------------------------------------------- #
#  Utilities                                                                    #
# --------------------------------------------------------------------------- #

def _load_motor_names(path: str) -> List[str]:
    with open(path) as f:
        return sorted(set(l.strip() for l in f if l.strip() and not l.strip().startswith("#")))


def _match_label(s: str, atlas: set) -> Optional[str]:
    """Fuzzy match a recording label to the atlas set."""
    s = s.strip()
    if s in atlas:
        return s
    base = re.sub(r"-alt\d*$", "", s)
    if base in atlas:
        return base
    if not base.endswith(("L", "R")):
        for suffix in ("L", "R"):
            c = base + suffix
            if c in atlas:
                return c
    return None


def _load_worm(
    h5_path: str,
    motor_set: set,
) -> Optional[Dict]:
    """Load raw GCaMP + behaviour for one worm.

    Returns dict with keys: name, traces (T, N_local), beh (T, L_b),
    beh_mask, labels (list[str]), motor_mask (bool array into local indices),
    motor_labels (list[str]).
    """
    with h5py.File(h5_path, "r") as f:
        if "gcamp/trace_array_original" not in f or "behaviour/eigenworms_calc_6" not in f:
            return None
        traces = np.array(f["gcamp/trace_array_original"], dtype=np.float64)
        beh = np.array(f["behaviour/eigenworms_calc_6"], dtype=np.float64)
        if "gcamp/neuron_labels" not in f:
            return None
        labels = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in f["gcamp/neuron_labels"][:]
        ]

    T, N = traces.shape
    beh_mask = np.isfinite(beh).astype(float)
    beh = np.nan_to_num(beh, nan=0.0)

    motor_mask = np.zeros(N, dtype=bool)
    motor_labels: List[str] = []
    for i, lab in enumerate(labels):
        if lab in motor_set:
            motor_mask[i] = True
            motor_labels.append(lab)

    if motor_mask.sum() == 0:
        return None

    name = Path(h5_path).stem
    return dict(
        name=name,
        traces=traces,          # (T, N)
        beh=beh,                # (T, L_b)
        beh_mask=beh_mask,      # (T, L_b)
        labels=labels,
        motor_mask=motor_mask,  # (N,) bool
        motor_labels=motor_labels,
        T=T,
    )


# --------------------------------------------------------------------------- #
#  Lagged features                                                              #
# --------------------------------------------------------------------------- #

def _build_lagged(X: np.ndarray, n_lags: int) -> np.ndarray:
    """Build [X_t, X_{t-1}, ..., X_{t-n_lags}] feature matrix."""
    if n_lags <= 0:
        return X
    T, M = X.shape
    parts = [X]
    for lag in range(1, n_lags + 1):
        lagged = np.zeros_like(X)
        if lag < T:
            lagged[lag:] = X[:-lag]
        parts.append(lagged)
    return np.column_stack(parts)


# --------------------------------------------------------------------------- #
#  Ridge-CV (blocked time-series CV)                                            #
# --------------------------------------------------------------------------- #

def _contiguous_folds(idx: np.ndarray, n_folds: int) -> List[np.ndarray]:
    idx = np.asarray(idx, dtype=int)
    n_folds = max(1, min(n_folds, idx.size))
    sizes = np.full(n_folds, idx.size // n_folds, dtype=int)
    sizes[: idx.size % n_folds] += 1
    folds, start = [], 0
    for s in sizes:
        folds.append(idx[start : start + s])
        start += s
    return [f for f in folds if f.size > 0]


def _fit_ridge(X: np.ndarray, y: np.ndarray, lam: float):
    x_mean = X.mean(0)
    x_std = X.std(0)
    x_std = np.where(x_std > 1e-12, x_std, 1.0)
    Xs = (X - x_mean) / x_std
    y_mean = y.mean()
    yc = y - y_mean
    G = Xs.T @ Xs
    if lam > 0:
        G += lam * np.eye(G.shape[0])
    try:
        c = np.linalg.solve(G, Xs.T @ yc)
    except np.linalg.LinAlgError:
        c, *_ = np.linalg.lstsq(G, Xs.T @ yc, rcond=None)
    coef = c / x_std
    intercept = y_mean - x_mean @ coef
    return intercept, coef


def _ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    n_folds: int = 5,
    log_lam_min: float = -2.0,
    log_lam_max: float = 6.0,
    n_grid: int = 40,
) -> Dict:
    """Ridge-CV for a single target column."""
    grid = np.concatenate([[0.0], np.logspace(log_lam_min, log_lam_max, n_grid)])
    idx = np.where(valid)[0]
    folds = _contiguous_folds(idx, n_folds)
    cv_mse = np.full(len(grid), np.inf)

    for li, lam in enumerate(grid):
        errs = []
        for fold in folds:
            train = idx[~np.isin(idx, fold)]
            if train.size < 3 or fold.size == 0:
                continue
            intercept, coef = _fit_ridge(X[train], y[train], lam)
            pred = intercept + X[fold] @ coef
            mse = np.mean((y[fold] - pred) ** 2)
            if np.isfinite(mse):
                errs.append(mse)
        if errs:
            cv_mse[li] = np.mean(errs)

    best = int(np.argmin(cv_mse))
    best_lam = grid[best]
    intercept, coef = _fit_ridge(X[idx], y[idx], best_lam)
    return dict(intercept=intercept, coef=coef, best_lam=best_lam)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# --------------------------------------------------------------------------- #
#  Individual decoder (per-worm)                                                #
# --------------------------------------------------------------------------- #

def train_individual(
    worm: Dict,
    n_lags: int,
    n_folds: int,
    test_frac: float = 0.2,
) -> Dict:
    """Train & test a decoder using local motor neurons for one worm."""
    X_motor = worm["traces"][:, worm["motor_mask"]]
    X = _build_lagged(X_motor, n_lags)
    beh = worm["beh"]
    bm = worm["beh_mask"]
    T, L_b = beh.shape

    # Train / test split (last test_frac as test)
    split = int(T * (1 - test_frac))
    r2_train = np.zeros(L_b)
    r2_test = np.zeros(L_b)

    for j in range(L_b):
        valid_all = (bm[:, j] > 0.5) & np.all(np.isfinite(X), axis=1)
        valid_train = valid_all.copy()
        valid_train[split:] = False

        if valid_train.sum() < 10:
            r2_train[j] = r2_test[j] = np.nan
            continue

        fit = _ridge_cv(X, beh[:, j], valid_train, n_folds=n_folds)
        pred = fit["intercept"] + X @ fit["coef"]

        # Train R²
        m_tr = valid_all & (np.arange(T) < split)
        if m_tr.sum() > 2:
            r2_train[j] = _r2(beh[m_tr, j], pred[m_tr])

        # Test R²
        m_te = valid_all & (np.arange(T) >= split)
        if m_te.sum() > 2:
            r2_test[j] = _r2(beh[m_te, j], pred[m_te])

    return dict(
        name=worm["name"],
        n_motor=int(worm["motor_mask"].sum()),
        motor_labels=worm["motor_labels"],
        r2_train=r2_train,
        r2_test=r2_test,
    )


# --------------------------------------------------------------------------- #
#  Union decoder (zero-fill)                                                    #
# --------------------------------------------------------------------------- #

def train_union(
    worms: List[Dict],
    union_labels: List[str],
    n_lags: int,
    n_folds: int,
    test_frac: float = 0.2,
) -> Dict:
    """Train a single decoder on concatenated worm data.

    Motor neurons are aligned to *union_labels*.  Missing neurons are zero-filled.
    """
    M_union = len(union_labels)
    label_to_idx = {lab: i for i, lab in enumerate(union_labels)}

    # Build concatenated feature matrix + targets
    X_parts, y_parts, mask_parts = [], [], []
    worm_boundaries = [0]  # cumulative time boundaries

    for worm in worms:
        T = worm["T"]
        # Scatter local motor traces to union-aligned array
        X_union = np.zeros((T, M_union), dtype=np.float64)
        for local_i, lab in enumerate(worm["motor_labels"]):
            union_j = label_to_idx.get(lab)
            if union_j is not None:
                local_idx = np.where(worm["motor_mask"])[0][
                    list(worm["motor_labels"]).index(lab)
                ]
                X_union[:, union_j] = worm["traces"][:, local_idx]

        X_parts.append(X_union)
        y_parts.append(worm["beh"])
        mask_parts.append(worm["beh_mask"])
        worm_boundaries.append(worm_boundaries[-1] + T)

    X_cat = np.vstack(X_parts)       # (T_total, M_union)
    y_cat = np.vstack(y_parts)       # (T_total, L_b)
    mask_cat = np.vstack(mask_parts) # (T_total, L_b)
    T_total = X_cat.shape[0]
    L_b = y_cat.shape[1]

    # Build lagged features on the full concatenated array
    # BUT: zero out cross-worm lag contamination
    X_lagged_parts = []
    for wi, worm in enumerate(worms):
        t0, t1 = worm_boundaries[wi], worm_boundaries[wi + 1]
        X_lagged_parts.append(_build_lagged(X_cat[t0:t1], n_lags))
    X_lagged = np.vstack(X_lagged_parts)  # (T_total, M_union * (1 + n_lags))

    # Train / test split: per-worm, last test_frac as test
    is_test = np.zeros(T_total, dtype=bool)
    for wi, worm in enumerate(worms):
        t0, t1 = worm_boundaries[wi], worm_boundaries[wi + 1]
        T_worm = t1 - t0
        split = int(T_worm * (1 - test_frac))
        is_test[t0 + split : t1] = True

    r2_train = np.zeros(L_b)
    r2_test = np.zeros(L_b)
    r2_per_worm_test = np.zeros((len(worms), L_b))
    fits = []

    for j in range(L_b):
        valid_all = (mask_cat[:, j] > 0.5) & np.all(np.isfinite(X_lagged), axis=1)
        valid_train = valid_all & (~is_test)

        if valid_train.sum() < 10:
            r2_train[j] = r2_test[j] = np.nan
            fits.append(None)
            continue

        fit = _ridge_cv(X_lagged, y_cat[:, j], valid_train, n_folds=n_folds)
        fits.append(fit)
        pred = fit["intercept"] + X_lagged @ fit["coef"]

        m_tr = valid_all & (~is_test)
        if m_tr.sum() > 2:
            r2_train[j] = _r2(y_cat[m_tr, j], pred[m_tr])
        m_te = valid_all & is_test
        if m_te.sum() > 2:
            r2_test[j] = _r2(y_cat[m_te, j], pred[m_te])

        # Per-worm test R²
        for wi, worm in enumerate(worms):
            t0, t1 = worm_boundaries[wi], worm_boundaries[wi + 1]
            m_w = valid_all[t0:t1] & is_test[t0:t1]
            if m_w.sum() > 2:
                r2_per_worm_test[wi, j] = _r2(
                    y_cat[t0:t1][m_w, j], pred[t0:t1][m_w]
                )

    return dict(
        n_motor_union=M_union,
        union_labels=union_labels,
        r2_train=r2_train,
        r2_test=r2_test,
        r2_per_worm_test=r2_per_worm_test,
        worm_names=[w["name"] for w in worms],
        T_total=T_total,
    )


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare individual vs union+zero-fill behaviour decoders"
    )
    p.add_argument(
        "--worms", nargs="+", required=True,
        help="Worm names (H5 stems without .h5)",
    )
    p.add_argument(
        "--data_dir", default="data/used/behaviour+neuronal activity atanas (2023)",
        help="Directory containing H5 files",
    )
    p.add_argument("--lag", type=int, default=8, help="Number of lag steps")
    p.add_argument("--n_folds", type=int, default=5, help="CV folds")
    p.add_argument("--test_frac", type=float, default=0.2, help="Hold-out fraction (last 20%%)")
    p.add_argument("--motor_file", default="data/used/masks+motor neurons/motor_neurons_with_control.txt")
    p.add_argument("--out", default=None, help="Output summary text file")
    args = p.parse_args()

    t0 = time.time()
    motor_set = set(_load_motor_names(args.motor_file))
    print(f"Motor neuron list: {len(motor_set)} names")

    # Load worms
    worms = []
    for name in args.worms:
        h5 = os.path.join(args.data_dir, name + ".h5")
        if not os.path.exists(h5):
            print(f"  [SKIP] {name}: file not found")
            continue
        w = _load_worm(h5, motor_set)
        if w is None:
            print(f"  [SKIP] {name}: missing data")
            continue
        print(f"  Loaded {name}: T={w['T']}, N_motor={w['motor_mask'].sum()}, labels={w['motor_labels'][:5]}...")
        worms.append(w)

    if len(worms) < 2:
        raise RuntimeError(f"Need at least 2 worms, got {len(worms)}")

    L_b = worms[0]["beh"].shape[1]
    ew_names = [f"EW{i+1}" for i in range(L_b)]

    # =======================================================================
    # A) Individual decoders
    # =======================================================================
    print(f"\n{'='*70}")
    print(f"  MODEL A: Individual per-worm decoders (lag={args.lag})")
    print(f"{'='*70}")

    indiv_results = []
    for w in worms:
        t1 = time.time()
        res = train_individual(w, args.lag, args.n_folds, args.test_frac)
        dt = time.time() - t1
        indiv_results.append(res)
        print(
            f"  {res['name']:20s}  N_motor={res['n_motor']:2d}  "
            f"test R²={np.nanmedian(res['r2_test']):.4f} (median)  "
            f"[{', '.join(f'{v:.3f}' for v in res['r2_test'])}]  ({dt:.1f}s)"
        )

    # Summary table: individual
    r2_test_all = np.array([r["r2_test"] for r in indiv_results])  # (n_worms, L_b)
    print(f"\n  Individual test R² summary (across {len(worms)} worms):")
    print(f"  {'Mode':<8} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'median':>8}")
    for j in range(L_b):
        col = r2_test_all[:, j]
        print(
            f"  {ew_names[j]:<8} {np.nanmean(col):8.4f} {np.nanstd(col):8.4f} "
            f"{np.nanmin(col):8.4f} {np.nanmax(col):8.4f} {np.nanmedian(col):8.4f}"
        )
    med_per_worm = np.nanmedian(r2_test_all, axis=1)
    print(
        f"  {'All':<8} {np.nanmean(med_per_worm):8.4f} {np.nanstd(med_per_worm):8.4f} "
        f"{np.nanmin(med_per_worm):8.4f} {np.nanmax(med_per_worm):8.4f} {np.nanmedian(med_per_worm):8.4f}"
    )

    # =======================================================================
    # B) Union + zero-fill decoder
    # =======================================================================
    # Build union of motor labels across worms
    union_labels_set: set = set()
    for w in worms:
        union_labels_set.update(w["motor_labels"])
    union_labels = sorted(union_labels_set)

    print(f"\n{'='*70}")
    print(f"  MODEL B: Union + zero-fill decoder (lag={args.lag})")
    print(f"  Union motor neurons: {len(union_labels)}")
    print(f"{'='*70}")

    t1 = time.time()
    union_res = train_union(worms, union_labels, args.lag, args.n_folds, args.test_frac)
    dt = time.time() - t1
    print(f"  Trained in {dt:.1f}s | T_total={union_res['T_total']}")

    print(f"\n  Union test R² (pooled across worms):")
    print(f"  {'Mode':<8} {'R² pool':>8}")
    for j in range(L_b):
        print(f"  {ew_names[j]:<8} {union_res['r2_test'][j]:8.4f}")
    print(f"  {'median':<8} {np.nanmedian(union_res['r2_test']):8.4f}")

    print(f"\n  Union test R² per worm:")
    print(f"  {'Worm':<22} " + " ".join(f"{e:>8}" for e in ew_names) + f" {'median':>8}")
    for wi, wname in enumerate(union_res["worm_names"]):
        row = union_res["r2_per_worm_test"][wi]
        vals = " ".join(f"{v:8.4f}" for v in row)
        print(f"  {wname:<22} {vals} {np.nanmedian(row):8.4f}")

    # =======================================================================
    # Comparison
    # =======================================================================
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Individual vs Union+zero-fill")
    print(f"{'='*70}")
    print(f"  {'Worm':<22} {'Indiv(med)':>12} {'Union(med)':>12} {'Δ':>8}")
    for wi, w in enumerate(worms):
        indiv_med = np.nanmedian(indiv_results[wi]["r2_test"])
        union_med = np.nanmedian(union_res["r2_per_worm_test"][wi])
        delta = union_med - indiv_med
        print(f"  {w['name']:<22} {indiv_med:12.4f} {union_med:12.4f} {delta:+8.4f}")

    overall_indiv = np.nanmedian(r2_test_all)
    overall_union = np.nanmedian(union_res["r2_per_worm_test"])
    print(f"  {'OVERALL':<22} {overall_indiv:12.4f} {overall_union:12.4f} {overall_union - overall_indiv:+8.4f}")

    print(f"\n  Total wall time: {time.time() - t0:.1f}s")

    # Save if requested
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"lag={args.lag} n_folds={args.n_folds} test_frac={args.test_frac}\n")
            f.write(f"worms: {[w['name'] for w in worms]}\n")
            f.write(f"union_motor_neurons: {union_labels}\n\n")

            f.write("Individual test R² (worm × mode):\n")
            f.write(f"{'worm':<22} " + " ".join(f"{e:>8}" for e in ew_names) + "\n")
            for wi, r in enumerate(indiv_results):
                vals = " ".join(f"{v:8.4f}" for v in r["r2_test"])
                f.write(f"{r['name']:<22} {vals}\n")

            f.write(f"\nUnion test R² (worm × mode):\n")
            f.write(f"{'worm':<22} " + " ".join(f"{e:>8}" for e in ew_names) + "\n")
            for wi, wname in enumerate(union_res["worm_names"]):
                vals = " ".join(f"{v:8.4f}" for v in union_res["r2_per_worm_test"][wi])
                f.write(f"{wname:<22} {vals}\n")
        print(f"  Summary saved to {args.out}")


if __name__ == "__main__":
    main()
