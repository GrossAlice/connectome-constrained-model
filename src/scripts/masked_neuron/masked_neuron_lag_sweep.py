#!/usr/bin/env python3
r"""
Masked-neuron prediction lag sweep — 2×2 factorial design.

═══════════════════════════════════════════════════════════════════════════
  FORMULATION
═══════════════════════════════════════════════════════════════════════════

Given neural population u ∈ ℝ^{T×N}, predict u_t^{(i)} for each neuron i.

Two binary options control the feature vector x_t^{(i)}:

  CONCURRENT (include other neurons at time t?)
  ─────────────────────────────────────────────
    True:  include u_t^{╲i}  (other neurons at the SAME time as target)
    False: only use past frames u_{t-1}, u_{t-2}, ... (strictly causal)

  INCLUDE_SELF (include masked neuron's own lagged history?)
  ─────────────────────────────────────────────
    True:  lagged blocks contain ALL N neurons (including neuron i's past)
    False: lagged blocks contain only N−1 neurons (neuron i removed)

This gives 4 conditions:

  ┌─────────────┬──────────────┬─────────────────────────────────────┬────────────┐
  │ concurrent  │ include_self │ Features at time t                  │ Dimension  │
  ├─────────────┼──────────────┼─────────────────────────────────────┼────────────┤
  │ True        │ True         │ u_t^{╲i}, u_{t-1..t-L}^{all}       │ (N−1) + NL │
  │ True        │ False        │ u_t^{╲i}, u_{t-1..t-L}^{╲i}        │ (N−1)(L+1) │
  │ False       │ True         │ u_{t-1..t-L}^{all}                  │ NL         │
  │ False       │ False        │ u_{t-1..t-L}^{╲i}                   │ (N−1)L     │
  └─────────────┴──────────────┴─────────────────────────────────────┴────────────┘

  ╲i  =  all neurons except i  (target always excluded from concurrent block)

NOTE: even in "concurrent + self", neuron i at time t is NEVER a feature —
      that would be trivial.  Only its PAST values (t−1 … t−L) appear.

INTERPRETATION:
  • "concurrent + self"    — maximum information, prediction ceiling
  • "concurrent only"      — population snapshot alone (no self-history)
  • "causal + self (AR)"   — includes AR component, strictly causal
  • "causal cross-neuron"  — pure cross-neuron causal influence (hardest)

KEY COMPARISONS (what each factor buys you):
  • Effect of concurrent   = R²(conc) − R²(no conc)   [instantaneous coupling]
  • Effect of self-history = R²(self) − R²(no self)    [AR contribution]
  • The "causal cross-neuron" condition tests whether other neurons'
    PAST activity alone predicts neuron i's PRESENT state.

CONNECTION TO STAGE-2 MODEL:
  • I_gap  ↔ concurrent coupling (gap junctions are ~instantaneous)
  • I_sv, I_dcv  ↔ causal cross-neuron (synaptic currents are lagged)
  • λ_u, I0  ↔ self-history (leak/persistence)

═══════════════════════════════════════════════════════════════════════════
  MODELS & EVALUATION
═══════════════════════════════════════════════════════════════════════════

  1. Ridge   (RidgeCV, 30 alphas logspace(−4, 6))
  2. PCA-Ridge  (PCA to k dims, then RidgeCV) — controls overfitting
  3. MLP  (2×128 hidden, ReLU, dropout, full-batch Adam)

  5-fold contiguous temporal CV.

═══════════════════════════════════════════════════════════════════════════
  USAGE
═══════════════════════════════════════════════════════════════════════════

  python -m scripts.masked_neuron.masked_neuron_lag_sweep \
      --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-07-15-12.h5" \
      --device cuda

  python -m scripts.masked_neuron.masked_neuron_lag_sweep \
      --h5 "..." --lags 1 5 10 20 --no_mlp        # fast: Ridge + PCA only

  python -m scripts.masked_neuron.masked_neuron_lag_sweep \
      --h5 "..." --replot                           # regenerate plots from JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.masked_neuron.masked_neuron_prediction import (
    _load_worm,
    _make_folds,
    _inner_split,
    _r2,
    _zscore,
    _make_mlp,
)


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

_RIDGE_ALPHAS = np.logspace(-4, 6, 30)
_DEFAULT_LAGS = [1, 3, 5, 10, 20]
_DEFAULT_OUT = _ROOT / "output_plots/masked_neuron_prediction/lag_sweep_2x2"

# (concurrent, include_self) tuples
ALL_CONDITIONS = [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]

COND_NAME = {
    (True,  True):  "concurrent + self",
    (True,  False): "concurrent only",
    (False, True):  "causal + self (AR)",
    (False, False): "causal cross-neuron",
}

COND_KEY = {
    (True,  True):  "conc+self",
    (True,  False): "conc_only",
    (False, True):  "self_only",
    (False, False): "cross_only",
}

KEY_TO_COND = {v: k for k, v in COND_KEY.items()}

COND_COLOR = {
    "conc+self":  "#e74c3c",
    "conc_only":  "#3498db",
    "self_only":  "#2ecc71",
    "cross_only": "#9b59b6",
}


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _pearson_r(y, yhat):
    c = np.corrcoef(y, yhat)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _precompute_full(u, n_lags):
    """Build the full feature matrix with concurrent + L lagged blocks.

    Block layout (each block has N columns):
      Block 0 (concurrent): u[L .. L+T_out-1, :]   — neurons at time t
      Block k (lag k):       u[L-k .. L-k+T_out-1, :] — neurons at time t−k

    Returns
    -------
    X_full : (T_out, N*(L+1))  float32
    Y_all  : (T_out, N)        float32  — targets (u at time t)
    """
    T, N = u.shape
    T_out = T - n_lags
    blocks = [u[n_lags:n_lags + T_out]]               # concurrent block
    for k in range(1, n_lags + 1):
        blocks.append(u[n_lags - k:n_lags - k + T_out])
    return (np.concatenate(blocks, axis=1).astype(np.float32),
            u[n_lags:n_lags + T_out].astype(np.float32))


def _feature_mask(N, n_lags, ni, concurrent, include_self):
    """Boolean mask over X_full columns for target neuron ni and condition.

    - concurrent=True:  keep block 0, but remove col ni (the target)
    - concurrent=False: drop entire block 0
    - include_self=True:  keep col ni in lagged blocks
    - include_self=False: remove col ni from lagged blocks 1..L
    """
    total = N * (n_lags + 1)
    mask = np.ones(total, dtype=bool)

    if not concurrent:
        mask[:N] = False                       # drop entire concurrent block
    else:
        mask[ni] = False                       # remove target from concurrent

    if not include_self:
        for k in range(1, n_lags + 1):         # remove target from lagged blocks
            mask[k * N + ni] = False

    return mask


def _feature_dim(N, n_lags, concurrent, include_self):
    """Expected feature dimensionality (for display/validation)."""
    if concurrent and include_self:
        return (N - 1) + N * n_lags
    elif concurrent and not include_self:
        return (N - 1) * (n_lags + 1)
    elif not concurrent and include_self:
        return N * n_lags
    else:
        return (N - 1) * n_lags


def _train_mlp_fb(net, Xtr, Ytr, Xva, Yva, *, epochs, lr, wd, patience):
    """Full-batch MLP training with early stopping."""
    import torch
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    crit = torch.nn.MSELoss()
    bv, st, bs = float("inf"), 0, None
    for _ in range(epochs):
        net.train()
        loss = crit(net(Xtr), Ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = crit(net(Xva), Yva).item()
        if vl < bv:
            bv, st, bs = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            st += 1
        if patience and st >= patience:
            break
    if bs:
        net.load_state_dict(bs)


# ═══════════════════════════════════════════════════════════════════════
#  Per-neuron model evaluation
# ═══════════════════════════════════════════════════════════════════════

def _eval_ridge_pca(X_full, Y_all, ni, N, n_lags,
                    conc, incself, folds, pca_k):
    """Ridge + PCA-Ridge CV for one neuron.

    Returns (r2_ridge, corr_ridge, r2_pca, corr_pca).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.decomposition import PCA

    mask = _feature_mask(N, n_lags, ni, conc, incself)
    X, y = X_full[:, mask], Y_all[:, ni]
    T_out = X.shape[0]
    k = min(pca_k, X.shape[1])

    pred_r = np.zeros(T_out, np.float32)
    pred_p = np.zeros(T_out, np.float32)

    for tr, te in folds:
        # Raw Ridge
        r = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
        r.fit(X[tr], y[tr])
        pred_r[te] = r.predict(X[te]).astype(np.float32)

        # PCA-Ridge
        ke = min(k, len(tr) - 1)
        pca = PCA(n_components=ke)
        rp = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
        rp.fit(pca.fit_transform(X[tr]), y[tr])
        pred_p[te] = rp.predict(pca.transform(X[te])).astype(np.float32)

    return (_r2(y, pred_r), _pearson_r(y, pred_r),
            _r2(y, pred_p), _pearson_r(y, pred_p))


def _eval_mlp(X_full, Y_all, ni, N, n_lags,
              conc, incself, folds, kw, device, to_t):
    """MLP CV for one neuron. Returns (r2_mlp, corr_mlp)."""
    import torch

    mask = _feature_mask(N, n_lags, ni, conc, incself)
    X, y = X_full[:, mask], Y_all[:, ni]
    T_out, d_in = X.shape
    pred = np.zeros(T_out, np.float32)

    for fi, (tr, te) in enumerate(folds):
        tri, vai = _inner_split(tr)
        mu_x, std_x = _zscore(X[tri])
        mu_y = float(y[tri].mean())
        std_y = float(max(y[tri].std(), 1e-8))

        Xtr_z = to_t(((X[tri] - mu_x) / std_x).astype(np.float32))
        Ytr_z = to_t(((y[tri].reshape(-1, 1) - mu_y) / std_y).astype(np.float32))
        Xva_z = to_t(((X[vai] - mu_x) / std_x).astype(np.float32))
        Yva_z = to_t(((y[vai].reshape(-1, 1) - mu_y) / std_y).astype(np.float32))

        torch.manual_seed(kw["seed"] + fi)
        net = _make_mlp(d_in, kw["hidden"], 1, kw["dropout"]).to(device)
        _train_mlp_fb(net, Xtr_z, Ytr_z, Xva_z, Yva_z,
                      epochs=kw["epochs"], lr=kw["lr"],
                      wd=kw["weight_decay"], patience=kw["patience"])
        net.eval()
        with torch.no_grad():
            pz = net(to_t(((X[te] - mu_x) / std_x).astype(np.float32)))
            pz = pz.cpu().numpy().ravel()
        pred[te] = (pz * std_y + mu_y).astype(np.float32)

    return _r2(y, pred), _pearson_r(y, pred)


# ═══════════════════════════════════════════════════════════════════════
#  Transformer
# ═══════════════════════════════════════════════════════════════════════

def _build_windows(u_n, indices, K):
    """Context windows [idx : idx+K] for each index → (len, K, N)."""
    return np.stack([u_n[idx:idx + K] for idx in indices])


def _train_trf_fold(X_tr, Y_tr, X_va, Y_va, N, K, device,
                    d_model=128, n_layers=2, n_heads=4,
                    epochs=150, lr=1e-3, wd=1e-4, patience=20):
    """Train TemporalTransformerGaussian on one CV fold (MSE on predict_mean)."""
    import torch
    import torch.nn.functional as F
    from baseline_transformer.model import TemporalTransformerGaussian
    from baseline_transformer.config import TransformerBaselineConfig

    cfg = TransformerBaselineConfig()
    cfg.context_length = K
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.n_heads = n_heads

    model = TemporalTransformerGaussian(n_neural=N, n_beh=0, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=device)
    Yv = torch.tensor(Y_va, dtype=torch.float32, device=device)

    bvl, bs, pat = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        pred = model.predict_mean(Xt)
        loss = F.mse_loss(pred, Yt)
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model.predict_mean(Xv), Yv).item()
        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
        if pat > patience:
            break
    if bs:
        model.load_state_dict(bs)
    model.eval()
    return model


def _eval_transformer_causal(u, n_lags, folds, device, trf_kw):
    """Train transformer and evaluate for both causal conditions.

    One model per fold predicts ALL N neurons simultaneously from K past
    frames of the full population.

    *causal + self* (``self_only``):
        Direct forward pass — context contains all N neurons including the
        target's own history.

    *causal cross-neuron* (``cross_only``):
        At test time column *i* is zeroed in the context (= mean imputation
        in z-score space).  The model must predict neuron *i* from the
        remaining neurons only.

    Returns
    -------
    dict : {cond_key: (r2_array(N,), corr_array(N,))}
    """
    import torch

    T, N = u.shape
    T_out = T - n_lags
    pred_self  = np.zeros((T_out, N), dtype=np.float32)
    pred_cross = np.zeros((T_out, N), dtype=np.float32)

    for fi, (tr, te) in enumerate(folds):
        tri, vai = _inner_split(tr)

        # Z-score from inner-train statistics
        mu = u[n_lags + tri].mean(0)
        sig = u[n_lags + tri].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)

        # Build sliding-window datasets
        X_tri = _build_windows(u_n, tri, n_lags)
        Y_tri = u_n[n_lags + tri]
        X_vai = _build_windows(u_n, vai, n_lags)
        Y_vai = u_n[n_lags + vai]

        # Train
        model = _train_trf_fold(X_tri, Y_tri, X_vai, Y_vai,
                                N, n_lags, device, **trf_kw)
        model = model.to(device)

        # ── causal + self: direct forward pass ──
        X_te_np = _build_windows(u_n, te, n_lags)
        X_te = torch.tensor(X_te_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_z = model.predict_mean(X_te).cpu().numpy()
        pred_self[te] = pred_z * sig + mu

        # ── causal cross-neuron: zero-out each neuron's column ──
        for ni in range(N):
            X_m = X_te_np.copy()
            X_m[:, :, ni] = 0.0
            with torch.no_grad():
                p = model.predict_mean(
                    torch.tensor(X_m, dtype=torch.float32,
                                 device=device)).cpu().numpy()
            pred_cross[te, ni] = p[:, ni] * sig[ni] + mu[ni]

        model.cpu()
        print(f"    TRF fold {fi+1}/{len(folds)} done")

    Y_raw = u[n_lags:]
    r2_s  = np.array([_r2(Y_raw[:, i], pred_self[:, i])  for i in range(N)])
    co_s  = np.array([_pearson_r(Y_raw[:, i], pred_self[:, i])  for i in range(N)])
    r2_x  = np.array([_r2(Y_raw[:, i], pred_cross[:, i]) for i in range(N)])
    co_x  = np.array([_pearson_r(Y_raw[:, i], pred_cross[:, i]) for i in range(N)])

    return {
        "self_only":  (r2_s, co_s),
        "cross_only": (r2_x, co_x),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main sweep engine
# ═══════════════════════════════════════════════════════════════════════

def run_sweep(worm, lags, n_folds, mlp_kw, pca_k=20,
              run_mlp=True, run_trf=True, trf_kw=None):
    """Run the full 2×2 × lag sweep.

    Returns
    -------
    results : dict[int, dict[str, list[dict]]]
        results[lag][cond_key] = list of per-neuron dicts.
    """
    import torch
    from joblib import Parallel, delayed

    u, labs = worm["u"], worm["labels"]
    T, N = u.shape
    device = torch.device(mlp_kw["device_str"])
    to_t = lambda a: torch.from_numpy(a).to(device)

    results = {}

    for n_lags in lags:
        T_out = T - n_lags
        if T_out < 100:
            print(f"  [skip lag={n_lags}: T_out={T_out} too small]")
            continue

        X_full, Y_all = _precompute_full(u, n_lags)
        folds = _make_folds(T_out, n_folds)
        lag_r = {}

        # ── Transformer (shared across causal conditions) ─────────
        trf_res = {}
        t_trf = 0
        if run_trf and n_lags >= 2:
            kw = trf_kw or {}
            dm = kw.get("d_model", 128)
            nl = kw.get("n_layers", 2)
            print(f"\n  lag={n_lags:3d}  Transformer (d={dm}, L={nl})")
            t0 = time.time()
            trf_res = _eval_transformer_causal(
                u, n_lags, folds, device, kw)
            t_trf = time.time() - t0
            for ck_t, (r2a, _) in trf_res.items():
                print(f"    {ck_t:12s}  TRF med R²={np.median(r2a):+.4f}")
            print(f"    [{t_trf:.0f}s]")

        for conc, incself in ALL_CONDITIONS:
            ck = COND_KEY[(conc, incself)]
            cn = COND_NAME[(conc, incself)]
            d = _feature_dim(N, n_lags, conc, incself)

            if d == 0:
                print(f"\n  lag={n_lags:3d}  {cn:26s}  dim=0 — SKIPPED")
                continue

            print(f"\n  lag={n_lags:3d}  {cn:26s}  dim={d}")

            # ── Ridge + PCA-Ridge (parallel across neurons) ───────
            t0 = time.time()
            rp = Parallel(n_jobs=min(8, N), verbose=0)(
                delayed(_eval_ridge_pca)(
                    X_full, Y_all, ni, N, n_lags,
                    conc, incself, folds, pca_k)
                for ni in range(N))
            t_rp = time.time() - t0

            # ── MLP (sequential, full-batch) ──────────────────────
            mlp_r = []
            t_mlp = 0
            if run_mlp:
                t0 = time.time()
                for ni in range(N):
                    mlp_r.append(_eval_mlp(
                        X_full, Y_all, ni, N, n_lags,
                        conc, incself, folds, mlp_kw, device, to_t))
                    if (ni + 1) % 25 == 0:
                        print(f"    MLP: {ni+1}/{N}")
                t_mlp = time.time() - t0

            # ── Assemble per-neuron records ────────────────────────
            recs = []
            for ni in range(N):
                r2r, corrr, r2p, corrp = rp[ni]
                rec = dict(
                    neuron=labs[ni], ni=ni,
                    is_ava=labs[ni].startswith("AVA"),
                    r2_ridge=r2r, corr_ridge=corrr,
                    r2_pca=r2p, corr_pca=corrp,
                )
                if run_mlp:
                    rec["r2_mlp"], rec["corr_mlp"] = mlp_r[ni]
                # Transformer (causal conditions only)
                if not conc and ck in trf_res:
                    r2_t, corr_t = trf_res[ck]
                    rec["r2_trf"]   = float(r2_t[ni])
                    rec["corr_trf"] = float(corr_t[ni])
                recs.append(rec)

            lag_r[ck] = recs

            # print summary
            med_r = np.median([r["r2_ridge"] for r in recs])
            med_p = np.median([r["r2_pca"] for r in recs])
            s = f"    Ridge={med_r:+.4f}  PCA={med_p:+.4f}"
            if run_mlp:
                med_m = np.median([r["r2_mlp"] for r in recs])
                s += f"  MLP={med_m:+.4f}"
            if not conc and ck in trf_res:
                med_t = np.median([r["r2_trf"] for r in recs])
                s += f"  TRF={med_t:+.4f}"
            s += f"  [{t_rp:.0f}s + {t_mlp:.0f}s]"
            print(s)

        results[n_lags] = lag_r

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Save / Load
# ═══════════════════════════════════════════════════════════════════════

def _save_results(results, worm, lags, out_dir, has_mlp, has_trf=False):
    """Flatten results into JSON-serializable dict and save."""
    u, labs = worm["u"], worm["labels"]
    T, N = u.shape

    data = {}
    for lag, lag_r in results.items():
        ld = {}
        for ck, recs in lag_r.items():
            entry = {
                "r2_ridge":   [r["r2_ridge"]   for r in recs],
                "r2_pca":     [r["r2_pca"]     for r in recs],
                "corr_ridge": [r["corr_ridge"] for r in recs],
                "corr_pca":   [r["corr_pca"]   for r in recs],
            }
            if has_mlp:
                entry["r2_mlp"]   = [r["r2_mlp"]   for r in recs]
                entry["corr_mlp"] = [r["corr_mlp"] for r in recs]
            if has_trf and any("r2_trf" in r for r in recs):
                entry["r2_trf"]   = [r.get("r2_trf", float("nan")) for r in recs]
                entry["corr_trf"] = [r.get("corr_trf", float("nan")) for r in recs]
            ld[ck] = entry
        data[str(lag)] = ld

    first_lag = list(results.keys())[0]
    first_cond = list(results[first_lag].keys())[0]

    meta = {
        "worm": worm["name"],
        "N": N, "T": T,
        "lags": lags,
        "conditions": [COND_KEY[c] for c in ALL_CONDITIONS],
        "neuron_names": [r["neuron"] for r in results[first_lag][first_cond]],
        "ava_mask": [r["is_ava"] for r in results[first_lag][first_cond]],
        "has_mlp": has_mlp,
        "has_trf": has_trf,
        "data": data,
    }
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Results saved to {out_dir / 'results.json'}")
    return meta


def _load_results(json_path):
    with open(json_path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "figure.facecolor": "white",
    })
    return plt


def plot_cross_condition(meta, out_dir):
    """Plot 1: PCA-Ridge median R² & corr vs lag, one line per condition."""
    plt = _setup_mpl()
    out_dir = Path(out_dir)
    lags = meta["lags"]
    data = meta["data"]
    conds = [ck for ck in meta["conditions"]
             if ck in data.get(str(lags[0]), {})]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, metric, ylabel in [
        (axes[0], "r2_pca",   "Median R² (PCA-Ridge)"),
        (axes[1], "corr_pca", "Median Pearson r (PCA-Ridge)"),
    ]:
        for ck in conds:
            available = [l for l in lags if ck in data.get(str(l), {})]
            vals = [np.median(data[str(l)][ck][metric]) for l in available]
            q25 = [np.percentile(data[str(l)][ck][metric], 25) for l in available]
            q75 = [np.percentile(data[str(l)][ck][metric], 75) for l in available]
            ct = KEY_TO_COND[ck]
            ax.plot(available, vals, "o-", color=COND_COLOR[ck], lw=2, ms=7,
                    label=COND_NAME[ct])
            ax.fill_between(available, q25, q75, alpha=0.1, color=COND_COLOR[ck])

        ax.set_xlabel("Lag (frames)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{meta['worm']}: Cross-condition comparison (N={meta['N']})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = out_dir / f"{meta['worm']}_cross_condition.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_model_grid(meta, out_dir):
    """Plot 2: 2×2 grid (one panel per condition), Ridge/PCA/MLP vs lag."""
    plt = _setup_mpl()
    out_dir = Path(out_dir)
    lags = meta["lags"]
    data = meta["data"]
    has_mlp = meta.get("has_mlp", False)

    c_ridge, c_pca, c_mlp, c_trf = "#3498db", "#27ae60", "#e74c3c", "#ff7f0e"
    has_trf = meta.get("has_trf", False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

    for ax, (conc, incself) in zip(axes.flat, ALL_CONDITIONS):
        ck = COND_KEY[(conc, incself)]
        available = [l for l in lags if ck in data.get(str(l), {})]
        if not available:
            ax.set_visible(False); continue

        med_r = [np.median(data[str(l)][ck]["r2_ridge"]) for l in available]
        med_p = [np.median(data[str(l)][ck]["r2_pca"]) for l in available]
        ax.plot(available, med_r, "o-", color=c_ridge, lw=2, ms=6, label="Ridge")
        ax.plot(available, med_p, "D-", color=c_pca, lw=2, ms=5, label="PCA-Ridge")

        if has_mlp and "r2_mlp" in data[str(available[0])].get(ck, {}):
            med_m = [np.median(data[str(l)][ck]["r2_mlp"]) for l in available]
            ax.plot(available, med_m, "s-", color=c_mlp, lw=2, ms=6, label="MLP")

        if has_trf and "r2_trf" in data[str(available[0])].get(ck, {}):
            med_t = [np.median(data[str(l)][ck]["r2_trf"]) for l in available]
            ax.plot(available, med_t, "^-", color=c_trf, lw=2, ms=6, label="TRF")

        # IQR bands
        models = [("r2_ridge", c_ridge), ("r2_pca", c_pca)]
        if has_mlp:
            models.append(("r2_mlp", c_mlp))
        if has_trf:
            models.append(("r2_trf", c_trf))
        for mk, mc in models:
            if mk not in data[str(available[0])].get(ck, {}):
                continue
            q25 = [np.percentile(data[str(l)][ck][mk], 25) for l in available]
            q75 = [np.percentile(data[str(l)][ck][mk], 75) for l in available]
            ax.fill_between(available, q25, q75, alpha=0.08, color=mc)

        ax.set_title(COND_NAME[(conc, incself)], fontsize=11,
                     fontweight="bold", color=COND_COLOR[ck])
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1, 0].set_xlabel("Lag (frames)")
    axes[1, 1].set_xlabel("Lag (frames)")
    axes[0, 0].set_ylabel("Median R²")
    axes[1, 0].set_ylabel("Median R²")

    fig.suptitle(f"{meta['worm']}: Model comparison per condition (N={meta['N']})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = out_dir / f"{meta['worm']}_model_grid.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_factor_effects(meta, out_dir):
    """Plot 3: Marginal effects of concurrent and self-history factors.

    Uses PCA-Ridge R² to avoid conflating overfitting with factor effects.
    Decomposition (per neuron, then take median):
      concurrent_effect = 0.5 * [(conc+self − self_only) + (conc_only − cross_only)]
      self_effect       = 0.5 * [(conc+self − conc_only) + (self_only − cross_only)]
      interaction       = (conc+self − conc_only) − (self_only − cross_only)
    """
    plt = _setup_mpl()
    out_dir = Path(out_dir)
    lags = meta["lags"]
    data = meta["data"]
    N = meta["N"]

    # Check all 4 conditions are present
    all_ckeys = ["conc+self", "conc_only", "self_only", "cross_only"]
    available_lags = [l for l in lags
                      if all(ck in data.get(str(l), {}) for ck in all_ckeys)]
    if not available_lags:
        print("  [skip factor_effects: not all 4 conditions available]")
        return

    eff_conc, eff_self, eff_int = [], [], []
    eff_conc_iqr, eff_self_iqr = [[], []], [[], []]

    for l in available_lags:
        ld = data[str(l)]
        cs = np.array(ld["conc+self"]["r2_pca"])
        co = np.array(ld["conc_only"]["r2_pca"])
        so = np.array(ld["self_only"]["r2_pca"])
        xo = np.array(ld["cross_only"]["r2_pca"])

        c_eff = 0.5 * ((cs - so) + (co - xo))   # marginal effect of concurrent
        s_eff = 0.5 * ((cs - co) + (so - xo))   # marginal effect of self
        inter = (cs - co) - (so - xo)             # interaction

        eff_conc.append(np.median(c_eff))
        eff_self.append(np.median(s_eff))
        eff_int.append(np.median(inter))

        eff_conc_iqr[0].append(np.percentile(c_eff, 25))
        eff_conc_iqr[1].append(np.percentile(c_eff, 75))
        eff_self_iqr[0].append(np.percentile(s_eff, 25))
        eff_self_iqr[1].append(np.percentile(s_eff, 75))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Both marginal effects
    axes[0].plot(available_lags, eff_conc, "o-", color="#e74c3c", lw=2, ms=7,
                 label="Concurrent effect")
    axes[0].fill_between(available_lags, eff_conc_iqr[0], eff_conc_iqr[1],
                         alpha=0.15, color="#e74c3c")
    axes[0].plot(available_lags, eff_self, "s-", color="#2ecc71", lw=2, ms=7,
                 label="Self-history effect")
    axes[0].fill_between(available_lags, eff_self_iqr[0], eff_self_iqr[1],
                         alpha=0.15, color="#2ecc71")
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].set_xlabel("Lag (frames)")
    axes[0].set_ylabel("Median R² gain (PCA-Ridge)")
    axes[0].set_title("Marginal factor effects")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.2)

    # Panel 2: Interaction
    axes[1].plot(available_lags, eff_int, "D-", color="#9b59b6", lw=2, ms=7)
    axes[1].axhline(0, color="k", lw=1, ls="--")
    axes[1].set_xlabel("Lag (frames)")
    axes[1].set_ylabel("Median R² interaction")
    axes[1].set_title("Interaction (concurrent × self)")
    axes[1].grid(alpha=0.2)
    axes[1].text(0.02, 0.02,
                 "> 0 → synergistic (combined > sum of parts)\n"
                 "< 0 → redundant (overlap in information)",
                 transform=axes[1].transAxes, fontsize=9, va="bottom",
                 bbox=dict(facecolor="lightyellow", alpha=0.8))

    # Panel 3: Bar chart at mid-lag
    mid_lag = available_lags[len(available_lags) // 2]
    ld = data[str(mid_lag)]
    x = np.arange(4)
    vals = [np.median(ld[ck]["r2_pca"]) for ck in all_ckeys]
    colors = [COND_COLOR[ck] for ck in all_ckeys]
    labels = [COND_NAME[KEY_TO_COND[ck]] for ck in all_ckeys]
    axes[2].bar(x, vals, color=colors, edgecolor="k", lw=0.5, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[2].set_ylabel("Median R² (PCA-Ridge)")
    axes[2].set_title(f"Conditions at lag={mid_lag}")
    for xi, v in enumerate(vals):
        axes[2].text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=9,
                     fontweight="bold")
    axes[2].axhline(0, color="k", lw=0.5, ls="--")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{meta['worm']}: Factor decomposition — PCA-Ridge (N={N})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = out_dir / f"{meta['worm']}_factor_effects.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_heatmaps(meta, out_dir):
    """Plot 4: 2×2 heatmaps (neurons × lags) for PCA-Ridge R²."""
    plt = _setup_mpl()
    out_dir = Path(out_dir)
    lags = meta["lags"]
    data = meta["data"]
    N = meta["N"]
    names = meta["neuron_names"]
    ava_mask = np.array(meta["ava_mask"])

    # Build (n_lags, N) arrays per condition
    cond_arrays = {}
    for ck in meta["conditions"]:
        available = [l for l in lags if ck in data.get(str(l), {})]
        if available:
            cond_arrays[ck] = (available,
                               np.array([data[str(l)][ck]["r2_pca"]
                                         for l in available]))

    if not cond_arrays:
        return

    # Sort neurons by mean R² in "conc+self" (or first available)
    ref_ck = "conc+self" if "conc+self" in cond_arrays else list(cond_arrays)[0]
    mean_r2 = cond_arrays[ref_ck][1].mean(axis=0)
    order = np.argsort(mean_r2)[::-1]
    ordered_names = [names[i] for i in order]
    ava_y = [yi for yi, n in enumerate(ordered_names)
             if n.startswith("AVA")]

    fig, axes = plt.subplots(2, 2, figsize=(14, max(6, N * 0.12)))

    for ax, (conc, incself) in zip(axes.flat, ALL_CONDITIONS):
        ck = COND_KEY[(conc, incself)]
        if ck not in cond_arrays:
            ax.set_visible(False)
            continue
        avail, arr = cond_arrays[ck]
        arr_sorted = arr[:, order]
        vmax = max(0.3, np.percentile(np.abs(arr_sorted), 95))

        im = ax.imshow(arr_sorted.T, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(len(avail)))
        ax.set_xticklabels([str(l) for l in avail])
        ax.set_xlabel("Lag (frames)")
        ax.set_ylabel("Neuron")
        ax.set_title(COND_NAME[(conc, incself)], fontsize=10,
                     fontweight="bold", color=COND_COLOR[ck])

        for yy in ava_y:
            ax.annotate(ordered_names[yy], xy=(len(avail) - 0.5, yy),
                        fontsize=6, fontweight="bold", color="gold",
                        va="center")

        fig.colorbar(im, ax=ax, shrink=0.8, label="PCA-Ridge R²")

    fig.suptitle(f"{meta['worm']}: PCA-Ridge R² heatmaps (N={N})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = out_dir / f"{meta['worm']}_heatmaps.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_violin_conditions(meta, out_dir):
    """Plot 5: Violin plot at mid-lag comparing all 4 conditions (PCA-Ridge)."""
    plt = _setup_mpl()
    out_dir = Path(out_dir)
    lags = meta["lags"]
    data = meta["data"]
    N = meta["N"]
    names = meta["neuron_names"]
    ava_mask = np.array(meta["ava_mask"])

    all_ckeys = ["conc+self", "conc_only", "self_only", "cross_only"]
    mid_lag = lags[len(lags) // 2]
    ld = data.get(str(mid_lag), {})
    avail = [ck for ck in all_ckeys if ck in ld]
    if len(avail) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 3 * len(avail)), 6))
    rng = np.random.default_rng(42)

    for ax, metric, ylabel in [
        (axes[0], "r2_pca", "PCA-Ridge R²"),
        (axes[1], "corr_pca", "PCA-Ridge Pearson r"),
    ]:
        datasets = [np.array(ld[ck][metric]) for ck in avail]
        positions = np.arange(len(avail))
        colors = [COND_COLOR[ck] for ck in avail]
        labels = [COND_NAME[KEY_TO_COND[ck]] for ck in avail]

        parts = ax.violinplot(datasets, positions=positions,
                              showmedians=False, showextrema=False)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.35)
            pc.set_edgecolor(c)

        all_jitter = []
        for i, (vals, c) in enumerate(zip(datasets, colors)):
            jitter = rng.normal(0, 0.05, size=len(vals))
            all_jitter.append(jitter)
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=12, alpha=0.45, color=c, edgecolors="0.4",
                       linewidths=0.2, zorder=3)
            med = float(np.median(vals))
            ax.plot([i - 0.22, i + 0.22], [med, med], color="k",
                    lw=2.5, zorder=4)
            ax.text(i + 0.28, med, f"{med:.3f}", va="center",
                    fontsize=10, fontweight="bold")

        # AVA highlights
        ava_idx = np.where(ava_mask)[0]
        for hi in ava_idx:
            for i, (vals, c) in enumerate(zip(datasets, colors)):
                xpos = i + all_jitter[i][hi]
                ypos = vals[hi]
                ax.scatter([xpos], [ypos], marker="*", s=120,
                           color="gold", edgecolors="k", linewidths=0.7,
                           zorder=6)
                if i == 0:
                    ax.annotate(names[hi], xy=(xpos, ypos),
                                xytext=(-38, 8), textcoords="offset points",
                                fontsize=7, fontweight="bold",
                                arrowprops=dict(arrowstyle="-", lw=0.5,
                                                color="0.4"))

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)

    fig.suptitle(f"{meta['worm']}: Condition comparison at lag={mid_lag}  "
                 f"(N={N}, PCA-Ridge, ★=AVA)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = out_dir / f"{meta['worm']}_violin_lag{mid_lag}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════
#  Summary table
# ═══════════════════════════════════════════════════════════════════════

def _print_summary_table(meta):
    """Print tables of median R² across conditions × lags."""
    lags = meta["lags"]
    data = meta["data"]
    all_ckeys = ["conc+self", "conc_only", "self_only", "cross_only"]
    has_trf = meta.get("has_trf", False)

    # ── PCA-Ridge table ──
    print("\n" + "═" * 72)
    print("  RESULTS SUMMARY — Median PCA-Ridge R²")
    print("═" * 72)

    header = f"  {'Lag':>5s}"
    for ck in all_ckeys:
        ct = KEY_TO_COND[ck]
        header += f"  {COND_NAME[ct]:>22s}"
    print(header)
    print("  " + "─" * 5 + ("  " + "─" * 22) * 4)

    for l in lags:
        ld = data.get(str(l), {})
        row = f"  {l:5d}"
        for ck in all_ckeys:
            if ck in ld:
                med = np.median(ld[ck]["r2_pca"])
                row += f"  {med:+22.4f}"
            else:
                row += f"  {'—':>22s}"
        print(row)

    print("═" * 72)

    # ── Transformer table (causal conditions only) ──
    causal_ckeys = ["self_only", "cross_only"]
    has_any_trf = any(
        "r2_trf" in data.get(str(l), {}).get(ck, {})
        for l in lags for ck in causal_ckeys)
    if has_trf and has_any_trf:
        print("\n" + "═" * 52)
        print("  TRANSFORMER — Median R² (causal conditions)")
        print("═" * 52)
        header = f"  {'Lag':>5s}"
        for ck in causal_ckeys:
            ct = KEY_TO_COND[ck]
            header += f"  {COND_NAME[ct]:>22s}"
        print(header)
        print("  " + "─" * 5 + ("  " + "─" * 22) * 2)
        for l in lags:
            ld = data.get(str(l), {})
            row = f"  {l:5d}"
            for ck in causal_ckeys:
                if ck in ld and "r2_trf" in ld[ck]:
                    med = np.nanmedian(ld[ck]["r2_trf"])
                    row += f"  {med:+22.4f}"
                else:
                    row += f"  {'—':>22s}"
            print(row)
        print("═" * 52)


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(
        description="Masked-neuron prediction: 2×2 factorial lag sweep")
    pa.add_argument("--h5", type=Path, required=True,
                    help="Path to worm HDF5 file")
    pa.add_argument("--out_dir", type=Path, default=None,
                    help="Output directory (default: auto from worm name)")
    pa.add_argument("--lags", type=int, nargs="+", default=None,
                    help=f"Lag values for sweep (default: {_DEFAULT_LAGS})")
    pa.add_argument("--n_folds", type=int, default=5)
    pa.add_argument("--pca_k", type=int, default=20,
                    help="Number of PCA components for PCA-Ridge (default: 20)")
    pa.add_argument("--no_mlp", action="store_true",
                    help="Skip MLP (much faster, Ridge + PCA-Ridge only)")
    pa.add_argument("--no_trf", action="store_true",
                    help="Skip Transformer (saves time on GPU)")
    pa.add_argument("--hidden", type=int, default=128)
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--weight_decay", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=15)
    pa.add_argument("--device", default="cpu")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--replot", action="store_true",
                    help="Just regenerate plots from existing results.json")
    args = pa.parse_args()

    worm = _load_worm(args.h5)
    if worm is None:
        raise ValueError(f"Cannot load {args.h5}")

    T, N = worm["u"].shape
    worm_name = worm["name"]

    out_dir = args.out_dir or (_DEFAULT_OUT / worm_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Worm: {worm_name}  T={T}  N={N}  dt={worm['dt']:.3f}s")

    if args.replot:
        meta = _load_results(out_dir / "results.json")
        print(f"  Loaded results from {out_dir / 'results.json'}")
    else:
        sweep_lags = args.lags or _DEFAULT_LAGS
        mlp_kw = dict(
            hidden=args.hidden, dropout=args.dropout, epochs=args.epochs,
            lr=args.lr, weight_decay=args.weight_decay,
            patience=args.patience, device_str=args.device, seed=args.seed,
        )

        print(f"\n{'═' * 60}")
        print(f"  2×2 factorial sweep:  lags = {sweep_lags}")
        print(f"  Models: Ridge, PCA-Ridge (k={args.pca_k})"
              + ("" if args.no_mlp else ", MLP 2×128")
              + ("" if args.no_trf else ", Transformer"))
        print(f"{'═' * 60}")

        print(f"\n  Feature dimensions at each lag:")
        for n_lags in sweep_lags:
            dims = {COND_NAME[(c, s)]: _feature_dim(N, n_lags, c, s)
                    for c, s in ALL_CONDITIONS}
            print(f"    lag={n_lags:3d}:  "
                  + "  |  ".join(f"{k}: {v}" for k, v in dims.items()))

        trf_kw = dict(d_model=128, n_layers=2, n_heads=4,
                      epochs=150, lr=1e-3, wd=1e-4, patience=20)

        t0 = time.time()
        results = run_sweep(worm, sweep_lags, args.n_folds, mlp_kw,
                            pca_k=args.pca_k, run_mlp=not args.no_mlp,
                            run_trf=not args.no_trf, trf_kw=trf_kw)
        print(f"\nSweep done in {time.time() - t0:.0f}s")

        meta = _save_results(results, worm, sweep_lags, out_dir,
                             has_mlp=not args.no_mlp,
                             has_trf=not args.no_trf)

    # Print summary table
    _print_summary_table(meta)

    # Generate all plots
    print("\nGenerating plots...")
    plot_cross_condition(meta, out_dir)
    plot_model_grid(meta, out_dir)
    plot_factor_effects(meta, out_dir)
    plot_heatmaps(meta, out_dir)
    plot_violin_conditions(meta, out_dir)

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
