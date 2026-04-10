#!/usr/bin/env python
"""
Union-Ridge ODE
===============
Stripped-down connectome-constrained model that adopts Conn-Ridge's
structural advantages:

  1. Union mask : T_union = (|T_e| + |T_sv| + |T_dcv|) > 0
  2. Free sign  : unconstrained β_ij (no softplus / sigmoid)
  3. Per-neuron α: sklearn RidgeCV picks optimal regularisation per neuron

ODE interpretation
------------------
  u_i(t+1) = β_{i,self} · u_i(t)
            + Σ_{j∈N(i)} β_ij · u_j(t)
            + β_{i,0}                          (intercept)
            [+ Σ_{k=1}^K  β_{ik}^lag · u_i(t-k)]        (self-lag)
            [+ Σ_{k=1}^K Σ_j β_{ijk}^nbr · u_j(t-k)]   (neighbor-lag)

where N(i) is the union connectome neighborhood of neuron i.

Evaluation: Stage2-style 3-fold temporal CV + windowed LOO (50 steps).

Conditions
----------
  U0_base        current-step features only  (= Conn-Ridge)
  U1_self_lag5   + self-lag K=5
  U2_self_lag10  + self-lag K=10
  U3_full_lag5   + self + neighbor lag K=5
  U4_full_lag10  + self + neighbor lag K=10
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

# ── paths ───────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
os.chdir(SRC)

SAVE_ROOT = SRC / "output_plots" / "stage2" / "union_ridge_ode"
H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"

N_FOLDS = 3
WINDOW_SIZE = 50
LOO_SUBSET_SIZE = 30
RIDGE_ALPHAS = np.logspace(-3, 6, 20)


# ═══════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """Load u_stage1 and connectome matrices via the Stage2 I/O layer."""
    from stage2.config import make_config
    from stage2.io_h5 import load_data_pt

    cfg = make_config(h5_path=H5)
    data = load_data_pt(cfg)
    u = data["u_stage1"].cpu().numpy().astype(np.float64)  # (T, N)
    T_e = data["T_e"].cpu().numpy()
    T_sv = data["T_sv"].cpu().numpy()
    T_dcv = data["T_dcv"].cpu().numpy()
    labels = data.get("neuron_labels", [])
    return u, T_e, T_sv, T_dcv, labels


def build_union_input_mask(T_e, T_sv, T_dcv):
    """Boolean mask where mask[i, j] = True  ⟺  j → i in the connectome.

    Uses the convention T_x[pre, post], so we transpose to get
    the "input" mask indexed by [post, pre].
    """
    adj_pre_post = np.abs(T_e) + np.abs(T_sv) + np.abs(T_dcv)
    mask = adj_pre_post.T > 0       # mask[post, pre]
    np.fill_diagonal(mask, False)    # no self-loops
    return mask


# ═══════════════════════════════════════════════════════════════════════
#  Feature construction
# ═══════════════════════════════════════════════════════════════════════

def build_features(u, mask, lag_order=0, nbr_lag=False):
    """Build per-neuron feature matrices for RidgeCV.

    For neuron *i* at time *t* (predicting u(t+1)):
      [u_j(t) for j in sorted(N(i) ∪ {i})]                  (base)
      [u_i(t-1), u_i(t-2), …, u_i(t-K)]                     (self-lag)
      [u_j(t-1), …, u_j(t-K) for j in N(i)]                 (neighbour-lag)

    Returns
    -------
    feats : dict[int, ndarray (T_eff, n_feat_i)]
    Y     : ndarray (T_eff, N)          – prediction targets
    start : int                          – first valid timeline index
    meta  : dict[int, dict]              – per-neuron auxiliary info
    """
    T, N = u.shape
    start = lag_order + 1          # need K steps of history + 1 for target
    T_eff = T - start

    # "current" features:  u(start-1) … u(T-2)
    X_cur = u[start - 1: -1]      # (T_eff, N)
    # targets:             u(start)  … u(T-1)
    Y = u[start:]                  # (T_eff, N)

    feats: dict[int, np.ndarray] = {}
    meta: dict[int, dict] = {}

    for i in range(N):
        nbr_idx = np.where(mask[i])[0]
        feat_idx = np.array(sorted(set(nbr_idx.tolist()) | {i}))
        self_pos = int(np.searchsorted(feat_idx, i))

        parts = [X_cur[:, feat_idx]]                      # (T_eff, |feat_idx|)

        if lag_order > 0:
            # self-lag: u_i(t-1) … u_i(t-K)
            for k in range(1, lag_order + 1):
                parts.append(u[start - 1 - k: -1 - k, i: i + 1])

            # neighbour-lag: u_j(t-k) for j in N(i), k=1..K
            if nbr_lag and len(nbr_idx) > 0:
                for k in range(1, lag_order + 1):
                    parts.append(u[start - 1 - k: -1 - k, nbr_idx])

        feats[i] = np.hstack(parts)
        meta[i] = dict(feat_idx=feat_idx, nbr_idx=nbr_idx, self_pos=self_pos)

    return feats, Y, start, meta


# ═══════════════════════════════════════════════════════════════════════
#  Temporal folds (matches Stage2 _make_temporal_folds exactly)
# ═══════════════════════════════════════════════════════════════════════

def make_folds(T: int, n_folds: int):
    n = T - 1
    fold_size = n // n_folds
    rem = n - fold_size * n_folds
    folds, cursor = [], 1
    for fi in range(n_folds):
        size = fold_size + (1 if fi < rem else 0)
        folds.append((cursor, cursor + size))
        cursor += size
    return folds


# ═══════════════════════════════════════════════════════════════════════
#  R² (matches stage2/_utils.py _r2)
# ═══════════════════════════════════════════════════════════════════════

def _r2(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m].astype(np.float64), y_pred[m].astype(np.float64)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - np.sum((yt - yp) ** 2) / max(ss_tot, 1e-12))


# ═══════════════════════════════════════════════════════════════════════
#  Per-neuron Ridge training
# ═══════════════════════════════════════════════════════════════════════

def train_ridge(feats, Y, train_mask, N):
    models = {}
    for i in range(N):
        X_tr, y_tr = feats[i][train_mask], Y[train_mask, i]
        m = RidgeCV(alphas=RIDGE_ALPHAS)
        m.fit(X_tr, y_tr)
        models[i] = m
    return models


def predict_onestep(models, feats, N):
    T_eff = feats[0].shape[0]
    pred = np.empty((T_eff, N), dtype=np.float64)
    for i in range(N):
        pred[:, i] = models[i].predict(feats[i])
    return pred


# ═══════════════════════════════════════════════════════════════════════
#  LOO evaluation (windowed, matching Stage2)
# ═══════════════════════════════════════════════════════════════════════

def loo_evaluate(u, models, meta, mask, subset,
                 lag_order=0, nbr_lag=False, window_size=50):
    """Windowed LOO: in each window the held-out neuron feeds back its
    own predictions while all others use ground truth.

    Matches Stage2 ``loo_forward_simulate_batched`` semantics:
      • Each window initialises u_pred_i from GT at w_start
      • Lag buffer initialised from GT at w_start - 1 … w_start - K
      • Every step in the window produces a prediction
      • Lag buffer accumulates predicted values for the LOO neuron
    """
    T, N = u.shape
    preds: dict[int, np.ndarray] = {}

    for sub_idx, i in enumerate(subset):
        pred_i = np.full(T, np.nan, dtype=np.float64)

        m_i = models[i]
        coef = m_i.coef_                # (n_feat,)
        intercept = m_i.intercept_      # scalar

        feat_idx = meta[i]["feat_idx"]
        nbr_idx = meta[i]["nbr_idx"]
        self_pos = meta[i]["self_pos"]
        n_nbr = len(nbr_idx)

        # pre-allocate feature vector for speed
        n_feat = len(feat_idx) + lag_order + (lag_order * n_nbr if nbr_lag else 0)
        fv = np.empty(n_feat, dtype=np.float64)

        for w_start in range(0, T, window_size):
            w_end = min(w_start + window_size, T)

            # ── initialise from GT ────────────────────────────────
            u_pred_val = u[w_start, i]
            pred_i[w_start] = u_pred_val

            # lag buffer (self): [u(w_start-1), u(w_start-2), …]
            lag_self = np.zeros(lag_order, dtype=np.float64)
            for k in range(lag_order):
                t_k = w_start - 1 - k
                if 0 <= t_k < T:
                    lag_self[k] = u[t_k, i]

            # ── step through window ───────────────────────────────
            for t in range(w_start, w_end - 1):
                off = 0

                # current-step features (LOO neuron replaced)
                fv[:len(feat_idx)] = u[t, feat_idx]
                fv[self_pos] = u_pred_val
                off += len(feat_idx)

                if lag_order > 0:
                    # self-lag
                    fv[off: off + lag_order] = lag_self
                    off += lag_order

                    # neighbour-lag
                    if nbr_lag and n_nbr > 0:
                        for k in range(1, lag_order + 1):
                            t_k = t - k
                            if 0 <= t_k < T:
                                fv[off: off + n_nbr] = u[t_k, nbr_idx]
                            else:
                                fv[off: off + n_nbr] = 0.0
                            off += n_nbr

                # fast dot-product prediction
                next_pred = np.dot(coef, fv) + intercept

                # update lag buffer: shift right, insert current value
                if lag_order > 0:
                    lag_self[1:] = lag_self[:-1]
                    lag_self[0] = u_pred_val

                u_pred_val = next_pred
                pred_i[t + 1] = u_pred_val

        preds[i] = pred_i

        if (sub_idx + 1) % 10 == 0 or sub_idx == 0 or (sub_idx + 1) == len(subset):
            v = ~np.isnan(pred_i)
            r2i = _r2(u[v, i], pred_i[v]) if v.sum() > 1 else float("nan")
            print(f"    LOO neuron {sub_idx + 1}/{len(subset)} "
                  f"(i={i}): R²={r2i:.4f}")

    return preds


# ═══════════════════════════════════════════════════════════════════════
#  Condition definitions
# ═══════════════════════════════════════════════════════════════════════

CONDITIONS = {
    "U0_base":       dict(lag_order=0,  nbr_lag=False),
    "U1_self_lag5":  dict(lag_order=5,  nbr_lag=False),
    "U2_self_lag10": dict(lag_order=10, nbr_lag=False),
    "U3_full_lag5":  dict(lag_order=5,  nbr_lag=True),
    "U4_full_lag10": dict(lag_order=10, nbr_lag=True),
}


# ═══════════════════════════════════════════════════════════════════════
#  Run one condition
# ═══════════════════════════════════════════════════════════════════════

def run_condition(name, u, mask, *, lag_order=0, nbr_lag=False):
    T, N = u.shape
    save_dir = SAVE_ROOT / name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"[{name}]  lag_order={lag_order}  nbr_lag={nbr_lag}")
    print(f"{'=' * 60}")

    # ── features ──────────────────────────────────────────────────
    feats, Y, start, meta = build_features(u, mask, lag_order, nbr_lag)
    T_eff = Y.shape[0]
    sample = min(10, N - 1)
    print(f"  features[{sample}]: {feats[sample].shape}  "
          f"(T_eff={T_eff}, start={start})")

    # ── temporal CV ───────────────────────────────────────────────
    folds = make_folds(T, N_FOLDS)
    pred_u_full = np.full((T, N), np.nan, dtype=np.float64)
    fold_models_list: list[dict] = []
    fold_test_mse: list[float] = []

    for fi, (te_s, te_e) in enumerate(folds):
        print(f"\n  [fold {fi}] test=[{te_s},{te_e})  "
              f"({te_e - te_s} frames)")

        # train mask in feature-space
        train_mask = np.ones(T_eff, dtype=bool)
        fs = max(0, te_s - start)
        fe = min(T_eff, te_e - start)
        train_mask[fs: fe] = False
        n_train = int(train_mask.sum())

        t0 = time.time()
        models = train_ridge(feats, Y, train_mask, N)
        pred = predict_onestep(models, feats, N)
        elapsed = time.time() - t0
        print(f"    RidgeCV fit: {elapsed:.1f}s  train={n_train}/{T_eff}")

        # store test-window predictions
        for t_feat in range(fs, fe):
            pred_u_full[t_feat + start] = pred[t_feat]

        # fold MSE on test window
        test_mse = float(np.mean(
            (Y[fs:fe] - pred[fs:fe]) ** 2))
        fold_test_mse.append(test_mse)

        fold_models_list.append(models)

        fold_r2 = np.array([
            _r2(Y[fs:fe, i], pred[fs:fe, i]) for i in range(N)])
        print(f"    held-out R²: mean={np.nanmean(fold_r2):.4f}  "
              f"median={np.nanmedian(fold_r2):.4f}  "
              f"MSE={test_mse:.6f}")

    # ── stitched one-step R² ──────────────────────────────────────
    valid = ~np.isnan(pred_u_full[:, 0])
    cv_r2 = np.array([
        _r2(u[valid, i], pred_u_full[valid, i]) for i in range(N)])
    print(f"\n  CV one-step R²: mean={np.nanmean(cv_r2):.4f}  "
          f"median={np.nanmedian(cv_r2):.4f}  "
          f"#neg={int((cv_r2 < 0).sum())}/{N}")

    # ── LOO subset (top-variance, same as Stage2) ────────────────
    var = np.var(u, axis=0)
    subset = [int(i) for i in np.argsort(var)[::-1][:LOO_SUBSET_SIZE]]

    # ── LOO evaluation per fold ──────────────────────────────────
    loo_pred_full: dict[int, np.ndarray] = {
        i: np.full(T, np.nan, dtype=np.float64) for i in subset}

    for fi, (te_s, te_e) in enumerate(folds):
        print(f"\n  [fold {fi}] LOO eval ({len(subset)} neurons, "
              f"window={WINDOW_SIZE})…")
        t0 = time.time()
        loo_preds = loo_evaluate(
            u, fold_models_list[fi], meta, mask, subset,
            lag_order=lag_order, nbr_lag=nbr_lag,
            window_size=WINDOW_SIZE)
        for i in subset:
            loo_pred_full[i][te_s:te_e] = loo_preds[i][te_s:te_e]
        elapsed = time.time() - t0
        print(f"    LOO eval done in {elapsed:.1f}s")

    # ── LOO R² ───────────────────────────────────────────────────
    cv_loo_r2 = np.full(N, np.nan, dtype=np.float64)
    for i in subset:
        v = ~np.isnan(loo_pred_full[i])
        if v.sum() > 1:
            cv_loo_r2[i] = _r2(u[v, i], loo_pred_full[i][v])

    loo_valid = cv_loo_r2[np.isfinite(cv_loo_r2)]
    loo_mean = float(np.nanmean(loo_valid)) if len(loo_valid) > 0 else None
    loo_med = float(np.nanmedian(loo_valid)) if len(loo_valid) > 0 else None
    n_neg = int((loo_valid < 0).sum()) if len(loo_valid) > 0 else 0
    print(f"\n  LOO R²: mean={loo_mean:.4f}  median={loo_med:.4f}  "
          f"#neg={n_neg}/{len(loo_valid)}")

    # ── diagnostics ──────────────────────────────────────────────
    best_fi = int(np.argmin(fold_test_mse))
    best_models = fold_models_list[best_fi]

    # α distribution
    alphas = np.array([best_models[i].alpha_ for i in range(N)])
    print(f"  α (fold {best_fi}): min={alphas.min():.2e}  "
          f"max={alphas.max():.2e}  median={np.median(alphas):.2e}")

    # effective λ from self-weight
    self_w = np.array([best_models[i].coef_[meta[i]["self_pos"]]
                       for i in range(N)])
    print(f"  β_self: mean={self_w.mean():.4f}  "
          f"min={self_w.min():.4f}  max={self_w.max():.4f}")
    eff_lam = 1.0 - self_w
    print(f"  eff λ = 1 − β_self: mean={eff_lam.mean():.4f}  "
          f"min={eff_lam.min():.4f}  max={eff_lam.max():.4f}")

    # neighbour weight magnitude
    nbr_w_all: list[float] = []
    for i in range(N):
        feat_idx = meta[i]["feat_idx"]
        nbr_mask = np.ones(len(feat_idx), dtype=bool)
        nbr_mask[meta[i]["self_pos"]] = False
        nbr_w_all.extend(best_models[i].coef_[:len(feat_idx)][nbr_mask].tolist())
    nw = np.array(nbr_w_all)
    print(f"  β_nbr:  mean={nw.mean():.4f}  std={nw.std():.4f}  "
          f"min={nw.min():.4f}  max={nw.max():.4f}  "
          f"#neg={int((nw < 0).sum())}/{len(nw)}")

    # ── save results ─────────────────────────────────────────────
    result = {
        "condition": name,
        "cv_onestep_r2_mean": float(np.nanmean(cv_r2)),
        "cv_onestep_r2_median": float(np.nanmedian(cv_r2)),
        "cv_loo_r2_mean": loo_mean,
        "cv_loo_r2_median": loo_med,
        "n_neg_loo": n_neg,
        "n_loo_neurons": len(loo_valid),
        "lag_order": lag_order,
        "nbr_lag": nbr_lag,
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(save_dir / "cv_onestep.npz",
             cv_r2=cv_r2, cv_loo_r2=cv_loo_r2,
             pred_u_full=pred_u_full.astype(np.float32))
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("[Union-Ridge ODE] Loading data …")
    u, T_e, T_sv, T_dcv, labels = load_data()
    mask = build_union_input_mask(T_e, T_sv, T_dcv)
    T, N = u.shape
    n_edges = int(mask.sum())
    print(f"  T={T}  N={N}  union_edges={n_edges}  "
          f"density={n_edges / N / N:.1%}")
    print(f"  mean neighbours per neuron: {n_edges / N:.1f}")

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for name, kwargs in CONDITIONS.items():
        # skip if already done
        summary_path = SAVE_ROOT / name / "summary.json"
        if summary_path.exists():
            prev = json.loads(summary_path.read_text())
            all_results[name] = prev
            print(f"\n[SKIP] {name} already done: LOO="
                  f"{prev.get('cv_loo_r2_mean', '?')}")
            continue

        try:
            result = run_condition(name, u, mask, **kwargs)
            all_results[name] = result
        except Exception:
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": "failed"}

    # ── summary table ─────────────────────────────────────────────
    print(f"\n\n{'=' * 60}")
    print(f"[SUMMARY]  Union-Ridge ODE")
    print(f"{'=' * 60}")
    print(f"{'Condition':<20s}  {'1-step':>8s}  {'LOO':>8s}  "
          f"{'LOO med':>8s}  {'#neg':>5s}")
    print(f"{'-' * 20}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 5}")

    for name, r in all_results.items():
        if "error" in r:
            print(f"{name:<20s}  {'ERROR':>8s}")
            continue
        s1 = f"{r['cv_onestep_r2_mean']:.4f}"
        loo = (f"{r['cv_loo_r2_mean']:.4f}"
               if r.get("cv_loo_r2_mean") is not None else "N/A")
        lm = (f"{r['cv_loo_r2_median']:.4f}"
              if r.get("cv_loo_r2_median") is not None else "N/A")
        nn = str(r.get("n_neg_loo", "?"))
        print(f"{name:<20s}  {s1:>8s}  {loo:>8s}  {lm:>8s}  {nn:>5s}")

    # reference baselines
    print(f"\n  Reference baselines:")
    print(f"    Conn-Ridge (model_dist_comp): LOO ≈ 0.475")
    print(f"    D02 (Stage2 ODE):             LOO = 0.420")
    print(f"    F03_lag10 (no L2):            LOO = 0.461")

    with open(SAVE_ROOT / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[Union-Ridge ODE] Done!")


if __name__ == "__main__":
    main()
