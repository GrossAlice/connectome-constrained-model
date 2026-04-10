#!/usr/bin/env python
"""
Hybrid: Stage2 biophysical features + per-neuron RidgeCV
========================================================

Idea
----
Stage2 computes rich biophysical currents (I_gap, I_sv, I_dcv with IIR,
sigmoid, Laplacian), but uses shared L2 + SGD to fit weights. This costs
LOO compared to per-neuron RidgeCV.

This script:
  1. Loads a trained S2_equiv_lag5 model (or trains one fresh)
  2. Extracts per-timestep current components as fixed features
  3. Re-fits linear weights with per-neuron RidgeCV
  4. Evaluates with the same windowed LOO (50 steps)

Conditions tested
-----------------
  H0_raw_ridge     : U3 features (raw u_j, raw u_lag) + RidgeCV  ← existing U3
  H1_s2_currents   : [u_i(t), I_gap, I_sv, I_dcv] as aggregate features + raw lags + RidgeCV
  H2_s2_per_edge   : full per-edge biophysical features + raw lags + RidgeCV
  H3_iir_features  : use IIR-filtered s_j(t) instead of raw u_j(t) + RidgeCV

This disentangles:
  • Are biophysical features (IIR, sigmoid) better or worse than raw features?
  • Does per-neuron α close the gap?
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

# ── paths ───────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
os.chdir(SRC)

SAVE_ROOT = SRC / "output_plots" / "stage2" / "union_ridge_ode"
H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
S2_MODEL_DIR = SAVE_ROOT / "S2_equiv_lag5"

N_FOLDS = 3
WINDOW_SIZE = 50
LOO_SUBSET_SIZE = 30
RIDGE_ALPHAS = np.logspace(-3, 6, 20)
LAG_ORDER = 5


# ═══════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    from stage2.config import make_config
    from stage2.io_h5 import load_data_pt
    cfg = make_config(h5_path=H5)
    data = load_data_pt(cfg)
    u = data["u_stage1"].cpu().numpy().astype(np.float64)
    T_e = data["T_e"].cpu().numpy()
    T_sv = data["T_sv"].cpu().numpy()
    T_dcv = data["T_dcv"].cpu().numpy()
    return u, T_e, T_sv, T_dcv, data


def build_union_mask(T_e, T_sv, T_dcv):
    adj = np.abs(T_e) + np.abs(T_sv) + np.abs(T_dcv)
    mask = adj.T > 0
    np.fill_diagonal(mask, False)
    return mask


# ═══════════════════════════════════════════════════════════════════════
#  Extract biophysical features from trained Stage2 model
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_s2_features(data, model_state_path):
    """Run forward pass of trained Stage2 model, returning intermediate currents."""
    from stage2.config import make_config
    from stage2.model import Stage2ModelPT
    from stage2.init_from_data import init_lambda_u

    cfg = make_config(
        H5,
        lag_order=5, lag_neighbor=True, lag_connectome_mask="all",
        lag_neighbor_per_type=False, lag_neighbor_activation="none",
        use_gap_junctions=True, use_sv_synapses=True, use_dcv_synapses=True,
        chemical_synapse_mode="iir", chemical_synapse_activation="sigmoid",
        edge_specific_G=True, per_neuron_amplitudes=True,
        graph_poly_order=1, learn_reversals=False, synapse_lag_taps=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u_stage1 = data["u_stage1"].to(device)
    N = u_stage1.shape[1]
    lambda_u_init = init_lambda_u(u_stage1, cfg)
    sign_t = data.get("sign_t")

    model = Stage2ModelPT(
        N, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
        cfg, device, d_ell=0,
        lambda_u_init=lambda_u_init,
        sign_t=sign_t,
    ).to(device)

    state = torch.load(model_state_path, map_location=device, weights_only=False)
    # The state dict may be saved as full state_dict or just raw params
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        # snapshot_model_state saves individual param tensors by name
        for name, val in state.items():
            if hasattr(model, name):
                param = getattr(model, name)
                if isinstance(param, torch.Tensor) and param.shape == val.shape:
                    param.copy_(val.to(device))
    model.eval()

    u_pt = data["u_stage1"].to(device)
    T, N = u_pt.shape

    params = model.precompute_params()
    lambda_u = params["lambda_u"].cpu().numpy()
    I0 = params["I0"].cpu().numpy()
    L = params["L"]

    # ── I_gap: L @ u.T (Laplacian) ──
    I_gap_all = (L @ u_pt.T).T.cpu().numpy()  # (T, N)

    # ── Chemical synapse IIR currents ──
    phi_all = model.phi(u_pt)
    s_sv = torch.zeros(N, model.r_sv, device=device)
    s_dcv = torch.zeros(N, model.r_dcv, device=device)
    T_sv_eff = params.get("T_sv_eff")
    T_dcv_eff = params.get("T_dcv_eff")

    I_sv_all = np.zeros((T, N), dtype=np.float64)
    I_dcv_all = np.zeros((T, N), dtype=np.float64)
    s_sv_traces = np.zeros((T, N, model.r_sv), dtype=np.float64) if model.r_sv > 0 else None

    for t in range(1, T):
        u_prev = u_pt[t - 1]
        phi_gated = phi_all[t - 1]

        if model.r_sv > 0 and T_sv_eff is not None:
            I_sv_t, s_sv = model._synaptic_current(
                u_prev, phi_gated, s_sv,
                T_sv_eff, params["a_sv"], params["tau_sv"], params["E_sv"])
            I_sv_all[t] = I_sv_t.cpu().numpy()
            if s_sv_traces is not None:
                s_sv_traces[t] = s_sv.cpu().numpy()

        if model.r_dcv > 0 and T_dcv_eff is not None:
            I_dcv_t, s_dcv = model._synaptic_current(
                u_prev, phi_gated, s_dcv,
                T_dcv_eff, params["a_dcv"], params["tau_dcv"], params["E_dcv"])
            I_dcv_all[t] = I_dcv_t.cpu().numpy()

    # ── Lag features (same as U3: raw u values) ──
    u_np = u_pt.cpu().numpy().astype(np.float64)

    # ── Also extract phi(u) for the "IIR features" condition ──
    phi_np = phi_all.cpu().numpy().astype(np.float64)

    return {
        "u": u_np,
        "phi_u": phi_np,  # sigmoid(u)
        "I_gap": I_gap_all,
        "I_sv": I_sv_all,
        "I_dcv": I_dcv_all,
        "lambda_u": lambda_u,
        "I0": I0,
        "s_sv_traces": s_sv_traces,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Feature builders per condition
# ═══════════════════════════════════════════════════════════════════════

def build_features_H0(u, mask, K=5):
    """H0: exact U3 features (raw u_j(t) + raw lags)."""
    T, N = u.shape
    start = K + 1
    T_eff = T - start
    X_cur = u[start - 1: -1]
    Y = u[start:]

    feats, meta = {}, {}
    for i in range(N):
        nbr_idx = np.where(mask[i])[0]
        feat_idx = np.array(sorted(set(nbr_idx.tolist()) | {i}))
        self_pos = int(np.searchsorted(feat_idx, i))
        parts = [X_cur[:, feat_idx]]
        for k in range(1, K + 1):
            parts.append(u[start - 1 - k: -1 - k, i:i+1])
        if len(nbr_idx) > 0:
            for k in range(1, K + 1):
                parts.append(u[start - 1 - k: -1 - k, nbr_idx])
        feats[i] = np.hstack(parts)
        meta[i] = dict(feat_idx=feat_idx, nbr_idx=nbr_idx, self_pos=self_pos)
    return feats, Y, start, meta


def build_features_H1(u, mask, s2_feats, K=5):
    """H1: aggregate biophysical currents + raw lags.

    Features per neuron i:
      [u_i(t), I_gap_i(t), I_sv_i(t), I_dcv_i(t),
       u_i(t-1)..u_i(t-K), u_j(t-1)..u_j(t-K) for j in N(i)]
    """
    T, N = u.shape
    start = K + 1
    T_eff = T - start

    I_gap = s2_feats["I_gap"]
    I_sv = s2_feats["I_sv"]
    I_dcv = s2_feats["I_dcv"]

    X_cur = u[start - 1: -1]
    Y = u[start:]

    feats, meta = {}, {}
    for i in range(N):
        nbr_idx = np.where(mask[i])[0]
        parts = [
            X_cur[:, i:i+1],                    # u_i(t)
            I_gap[start - 1: -1, i:i+1],        # I_gap_i(t)
            I_sv[start - 1: -1, i:i+1],         # I_sv_i(t)
            I_dcv[start - 1: -1, i:i+1],        # I_dcv_i(t)
        ]
        # self-lags
        for k in range(1, K + 1):
            parts.append(u[start - 1 - k: -1 - k, i:i+1])
        # neighbor-lags (raw)
        if len(nbr_idx) > 0:
            for k in range(1, K + 1):
                parts.append(u[start - 1 - k: -1 - k, nbr_idx])
        feats[i] = np.hstack(parts)
        meta[i] = dict(nbr_idx=nbr_idx,
                       self_pos=0,  # u_i(t) is first feature
                       feat_idx=np.array([i]))  # for LOO replacement
    return feats, Y, start, meta


def build_features_H3(u, phi_u, mask, K=5):
    """H3: use phi(u_j(t)) = sigmoid(u_j(t)) instead of raw u_j(t) for neighbors.

    Features per neuron i:
      [u_i(t), phi(u_j(t)) for j in N(i),
       u_i(t-1)..u_i(t-K),
       phi(u_j(t-1))..phi(u_j(t-K)) for j in N(i)]
    """
    T, N = u.shape
    start = K + 1
    T_eff = T - start
    X_cur_u = u[start - 1: -1]
    X_cur_phi = phi_u[start - 1: -1]
    Y = u[start:]

    feats, meta = {}, {}
    for i in range(N):
        nbr_idx = np.where(mask[i])[0]
        feat_idx = np.array(sorted(set(nbr_idx.tolist()) | {i}))
        self_pos = int(np.searchsorted(feat_idx, i))

        # current step: self uses raw u, neighbors use phi(u)
        cur_parts = []
        for j in feat_idx:
            if j == i:
                cur_parts.append(X_cur_u[:, j:j+1])
            else:
                cur_parts.append(X_cur_phi[:, j:j+1])
        parts = [np.hstack(cur_parts)]

        # self-lags (raw)
        for k in range(1, K + 1):
            parts.append(u[start - 1 - k: -1 - k, i:i+1])
        # neighbor-lags (phi)
        if len(nbr_idx) > 0:
            for k in range(1, K + 1):
                parts.append(phi_u[start - 1 - k: -1 - k, nbr_idx])
        feats[i] = np.hstack(parts)
        meta[i] = dict(feat_idx=feat_idx, nbr_idx=nbr_idx, self_pos=self_pos)
    return feats, Y, start, meta


# ═══════════════════════════════════════════════════════════════════════
#  Temporal folds, R², training, LOO — same as run_union_ridge_ode.py
# ═══════════════════════════════════════════════════════════════════════

def make_folds(T, n_folds):
    n = T - 1
    fold_size = n // n_folds
    rem = n - fold_size * n_folds
    folds, cursor = [], 1
    for fi in range(n_folds):
        size = fold_size + (1 if fi < rem else 0)
        folds.append((cursor, cursor + size))
        cursor += size
    return folds


def _r2(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m].astype(np.float64), y_pred[m].astype(np.float64)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - np.sum((yt - yp) ** 2) / max(ss_tot, 1e-12))


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
#  LOO evaluation for H0 / H3 (per-edge features with self_pos)
# ═══════════════════════════════════════════════════════════════════════

def loo_evaluate_peredge(u, models, meta, mask, subset,
                          lag_order=5, window_size=50,
                          use_phi=False, phi_u=None):
    """Windowed LOO for conditions with per-edge features (H0, H3)."""
    T, N = u.shape
    preds = {}

    for sub_idx, i in enumerate(subset):
        pred_i = np.full(T, np.nan, dtype=np.float64)
        m_i = models[i]
        coef = m_i.coef_
        intercept = m_i.intercept_

        feat_idx = meta[i]["feat_idx"]
        nbr_idx = meta[i]["nbr_idx"]
        self_pos = meta[i]["self_pos"]
        n_nbr = len(nbr_idx)

        n_feat = len(feat_idx) + lag_order + lag_order * n_nbr
        fv = np.empty(n_feat, dtype=np.float64)

        # source arrays for neighbor features
        u_src = phi_u if use_phi else u  # phi for neighbors in H3

        for w_start in range(0, T, window_size):
            w_end = min(w_start + window_size, T)
            u_pred_val = u[w_start, i]
            pred_i[w_start] = u_pred_val

            lag_self = np.zeros(lag_order, dtype=np.float64)
            for k in range(lag_order):
                t_k = w_start - 1 - k
                if 0 <= t_k < T:
                    lag_self[k] = u[t_k, i]

            for t in range(w_start, w_end - 1):
                off = 0
                # current-step: self uses raw u_pred, neighbors use u_src
                for fi, j in enumerate(feat_idx):
                    if j == i:
                        fv[fi] = u_pred_val
                    else:
                        fv[fi] = u_src[t, j]
                off = len(feat_idx)

                # self-lag
                fv[off: off + lag_order] = lag_self
                off += lag_order

                # neighbor-lag
                if n_nbr > 0:
                    for k in range(1, lag_order + 1):
                        t_k = t - k
                        if 0 <= t_k < T:
                            fv[off: off + n_nbr] = u_src[t_k, nbr_idx]
                        else:
                            fv[off: off + n_nbr] = 0.0
                        off += n_nbr

                next_pred = np.dot(coef, fv) + intercept
                if lag_order > 0:
                    lag_self[1:] = lag_self[:-1]
                    lag_self[0] = u_pred_val
                u_pred_val = next_pred
                pred_i[t + 1] = u_pred_val

        preds[i] = pred_i
        if (sub_idx + 1) % 10 == 0 or sub_idx == 0 or sub_idx + 1 == len(subset):
            v = ~np.isnan(pred_i)
            r2i = _r2(u[v, i], pred_i[v]) if v.sum() > 1 else float("nan")
            print(f"    LOO neuron {sub_idx+1}/{len(subset)} (i={i}): R²={r2i:.4f}")
    return preds


def loo_evaluate_aggregate(u, models, meta, mask, subset, s2_feats,
                            lag_order=5, window_size=50):
    """Windowed LOO for H1 (aggregate biophysical currents).

    Key difference: I_gap, I_sv, I_dcv are pre-computed from GT data.
    During LOO, we can't cheaply recompute them with the LOO neuron replaced,
    so we use the GT-based currents as-is. The LOO effect comes from:
    (a) replacing u_i(t) with pred in the self-feature
    (b) using predicted self-lag values
    """
    T, N = u.shape
    I_gap = s2_feats["I_gap"]
    I_sv = s2_feats["I_sv"]
    I_dcv = s2_feats["I_dcv"]
    preds = {}

    for sub_idx, i in enumerate(subset):
        pred_i = np.full(T, np.nan, dtype=np.float64)
        m_i = models[i]
        coef = m_i.coef_
        intercept = m_i.intercept_
        nbr_idx = meta[i]["nbr_idx"]
        n_nbr = len(nbr_idx)

        # Feature layout: [u_i, I_gap_i, I_sv_i, I_dcv_i, lag_self*K, lag_nbr*K]
        n_feat = 4 + lag_order + lag_order * n_nbr
        fv = np.empty(n_feat, dtype=np.float64)

        for w_start in range(0, T, window_size):
            w_end = min(w_start + window_size, T)
            u_pred_val = u[w_start, i]
            pred_i[w_start] = u_pred_val

            lag_self = np.zeros(lag_order, dtype=np.float64)
            for k in range(lag_order):
                t_k = w_start - 1 - k
                if 0 <= t_k < T:
                    lag_self[k] = u[t_k, i]

            for t in range(w_start, w_end - 1):
                fv[0] = u_pred_val           # u_i(t) — LOO-replaced
                fv[1] = I_gap[t, i]          # I_gap_i(t) — from GT
                fv[2] = I_sv[t, i]           # I_sv_i(t) — from GT
                fv[3] = I_dcv[t, i]          # I_dcv_i(t) — from GT
                off = 4

                fv[off: off + lag_order] = lag_self
                off += lag_order

                if n_nbr > 0:
                    for k in range(1, lag_order + 1):
                        t_k = t - k
                        if 0 <= t_k < T:
                            fv[off: off + n_nbr] = u[t_k, nbr_idx]
                        else:
                            fv[off: off + n_nbr] = 0.0
                        off += n_nbr

                next_pred = np.dot(coef, fv) + intercept
                if lag_order > 0:
                    lag_self[1:] = lag_self[:-1]
                    lag_self[0] = u_pred_val
                u_pred_val = next_pred
                pred_i[t + 1] = u_pred_val

        preds[i] = pred_i
        if (sub_idx + 1) % 10 == 0 or sub_idx == 0 or sub_idx + 1 == len(subset):
            v = ~np.isnan(pred_i)
            r2i = _r2(u[v, i], pred_i[v]) if v.sum() > 1 else float("nan")
            print(f"    LOO neuron {sub_idx+1}/{len(subset)} (i={i}): R²={r2i:.4f}")
    return preds


# ═══════════════════════════════════════════════════════════════════════
#  Run one condition
# ═══════════════════════════════════════════════════════════════════════

def run_condition(name, u, mask, feats, Y, start, meta, *,
                  loo_fn, loo_kwargs):
    T, N = u.shape
    save_dir = SAVE_ROOT / name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{name}]")
    print(f"{'='*60}")

    T_eff = Y.shape[0]
    sample = min(10, N - 1)
    print(f"  features[{sample}]: {feats[sample].shape}  (T_eff={T_eff})")

    folds = make_folds(T, N_FOLDS)
    pred_u_full = np.full((T, N), np.nan, dtype=np.float64)
    fold_models_list = []
    fold_test_mse = []

    for fi, (te_s, te_e) in enumerate(folds):
        print(f"\n  [fold {fi}] test=[{te_s},{te_e})")
        train_mask = np.ones(T_eff, dtype=bool)
        fs = max(0, te_s - start)
        fe = min(T_eff, te_e - start)
        train_mask[fs:fe] = False

        t0 = time.time()
        models = train_ridge(feats, Y, train_mask, N)
        pred = predict_onestep(models, feats, N)
        elapsed = time.time() - t0
        print(f"    RidgeCV fit: {elapsed:.1f}s")

        for t_feat in range(fs, fe):
            pred_u_full[t_feat + start] = pred[t_feat]

        test_mse = float(np.mean((Y[fs:fe] - pred[fs:fe])**2))
        fold_test_mse.append(test_mse)
        fold_models_list.append(models)

        fold_r2 = np.array([_r2(Y[fs:fe, i], pred[fs:fe, i]) for i in range(N)])
        print(f"    held-out R²: mean={np.nanmean(fold_r2):.4f}  "
              f"median={np.nanmedian(fold_r2):.4f}")

    # stitched 1-step
    valid = ~np.isnan(pred_u_full[:, 0])
    cv_r2 = np.array([_r2(u[valid, i], pred_u_full[valid, i]) for i in range(N)])
    print(f"\n  CV 1-step R²: mean={np.nanmean(cv_r2):.4f}  "
          f"median={np.nanmedian(cv_r2):.4f}")

    # LOO
    var = np.var(u, axis=0)
    subset = [int(i) for i in np.argsort(var)[::-1][:LOO_SUBSET_SIZE]]

    loo_pred_full = {i: np.full(T, np.nan, dtype=np.float64) for i in subset}
    for fi, (te_s, te_e) in enumerate(folds):
        print(f"\n  [fold {fi}] LOO eval…")
        t0 = time.time()
        loo_preds = loo_fn(
            u, fold_models_list[fi], meta, mask, subset,
            **loo_kwargs)
        for i in subset:
            loo_pred_full[i][te_s:te_e] = loo_preds[i][te_s:te_e]
        print(f"    LOO done in {time.time()-t0:.1f}s")

    cv_loo_r2 = np.full(N, np.nan, dtype=np.float64)
    for i in subset:
        v = ~np.isnan(loo_pred_full[i])
        if v.sum() > 1:
            cv_loo_r2[i] = _r2(u[v, i], loo_pred_full[i][v])

    loo_valid = cv_loo_r2[np.isfinite(cv_loo_r2)]
    loo_mean = float(np.nanmean(loo_valid)) if len(loo_valid) > 0 else None
    loo_med = float(np.nanmedian(loo_valid)) if len(loo_valid) > 0 else None
    n_neg = int((loo_valid < 0).sum()) if len(loo_valid) > 0 else 0
    print(f"\n  LOO R²: mean={loo_mean:.4f}  median={loo_med:.4f}  #neg={n_neg}/{len(loo_valid)}")

    # α distribution
    best_fi = int(np.argmin(fold_test_mse))
    best_models = fold_models_list[best_fi]
    alphas = np.array([best_models[i].alpha_ for i in range(N)])
    print(f"  α distribution (fold {best_fi}): "
          f"min={alphas.min():.2e}  max={alphas.max():.2e}  "
          f"median={np.median(alphas):.2e}")

    result = {
        "condition": name,
        "cv_onestep_r2_mean": float(np.nanmean(cv_r2)),
        "cv_onestep_r2_median": float(np.nanmedian(cv_r2)),
        "cv_loo_r2_mean": loo_mean,
        "cv_loo_r2_median": loo_med,
        "n_neg_loo": n_neg,
        "alpha_median": float(np.median(alphas)),
        "alpha_min": float(alphas.min()),
        "alpha_max": float(alphas.max()),
    }
    (save_dir / "summary.json").write_text(json.dumps(result, indent=2))
    np.savez(save_dir / "cv_onestep.npz",
             cv_r2=cv_r2, cv_loo_r2=cv_loo_r2,
             pred_u_full=pred_u_full.astype(np.float32))
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("[Hybrid Ridge] Loading data…")
    u, T_e, T_sv, T_dcv, data = load_data()
    mask = build_union_mask(T_e, T_sv, T_dcv)
    T, N = u.shape
    print(f"  T={T}  N={N}")

    # ── Find best fold model from S2_equiv_lag5 ──
    best_fold_path = None
    summary_path = S2_MODEL_DIR / "summary.json"
    if summary_path.exists():
        s2_summary = json.loads(summary_path.read_text())
        best_fi = s2_summary.get("best_fold_idx", 0)
        p = S2_MODEL_DIR / f"fold_{best_fi}_state.pt"
        if p.exists():
            best_fold_path = p
            print(f"  Using trained S2 model: fold {best_fi}")

    if best_fold_path is None:
        print("  [WARNING] No trained S2 model found, conditions H1/H3 will be skipped")
        s2_feats = None
    else:
        print("  Extracting biophysical features from trained S2 model…")
        t0 = time.time()
        s2_feats = extract_s2_features(data, best_fold_path)
        print(f"  Feature extraction: {time.time()-t0:.1f}s")

    all_results = {}

    # ── H0: exact U3 features + RidgeCV (= U3_full_lag5 replica) ──
    print("\n\n" + "━"*60)
    print("Building H0 features (raw u, same as U3)…")
    feats_h0, Y_h0, start_h0, meta_h0 = build_features_H0(u, mask, K=LAG_ORDER)
    all_results["H0_raw_ridge"] = run_condition(
        "H0_raw_ridge", u, mask, feats_h0, Y_h0, start_h0, meta_h0,
        loo_fn=loo_evaluate_peredge,
        loo_kwargs=dict(lag_order=LAG_ORDER, window_size=WINDOW_SIZE))

    # ── H1: aggregate biophysical currents + raw lags ──
    if s2_feats is not None:
        print("\n\n" + "━"*60)
        print("Building H1 features (I_gap, I_sv, I_dcv aggregates + raw lags)…")
        feats_h1, Y_h1, start_h1, meta_h1 = build_features_H1(
            u, mask, s2_feats, K=LAG_ORDER)
        all_results["H1_s2_currents"] = run_condition(
            "H1_s2_currents", u, mask, feats_h1, Y_h1, start_h1, meta_h1,
            loo_fn=loo_evaluate_aggregate,
            loo_kwargs=dict(s2_feats=s2_feats, lag_order=LAG_ORDER,
                           window_size=WINDOW_SIZE))

    # ── H3: phi(u_j) features + RidgeCV ──
    if s2_feats is not None:
        print("\n\n" + "━"*60)
        print("Building H3 features (sigmoid(u_j) for neighbors + raw lags)…")
        feats_h3, Y_h3, start_h3, meta_h3 = build_features_H3(
            u, s2_feats["phi_u"], mask, K=LAG_ORDER)
        all_results["H3_sigmoid_features"] = run_condition(
            "H3_sigmoid_features", u, mask, feats_h3, Y_h3, start_h3, meta_h3,
            loo_fn=loo_evaluate_peredge,
            loo_kwargs=dict(lag_order=LAG_ORDER, window_size=WINDOW_SIZE,
                           use_phi=True, phi_u=s2_feats["phi_u"]))

    # ═══════════════════════════════════════════════════════════════
    #  Summary table
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print(f"[SUMMARY]  Hybrid Ridge: S2 features + per-neuron RidgeCV")
    print(f"{'='*70}")
    print(f"{'Condition':<24s}  {'1step':>8s}  {'LOO':>8s}  {'LOO med':>8s}  "
          f"{'#neg':>5s}  {'α med':>10s}")
    print(f"{'-'*24}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*10}")

    for name, r in all_results.items():
        if "error" in r:
            print(f"{name:<24s}  ERROR")
            continue
        print(f"{name:<24s}  "
              f"{r['cv_onestep_r2_mean']:.4f}  "
              f"{r.get('cv_loo_r2_mean',0):.4f}  "
              f"{r.get('cv_loo_r2_median',0):.4f}  "
              f"{r.get('n_neg_loo','?'):>5}  "
              f"{r.get('alpha_median',0):.2e}")

    print(f"\nReference baselines:")
    print(f"  U3_full_lag5 (original):   LOO = 0.4802")
    print(f"  S2_equiv_lag5 (60ep SGD):  LOO = 0.3996")
    print(f"  Conn-Ridge (retrain_loo):  LOO = 0.4750")
    print(f"{'='*70}")

    (SAVE_ROOT / "hybrid_results.json").write_text(
        json.dumps(all_results, indent=2, default=str))
    print("\n[Hybrid Ridge] Done!")


if __name__ == "__main__":
    main()
