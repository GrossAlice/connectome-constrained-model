#!/usr/bin/env python3
"""Diagnostic tests: *why* does the Transformer beat the MLP?

Quick single-worm tests that isolate specific architectural advantages.
Run:
    python -m scripts.test_transformer_advantages \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2023-01-09-15.h5"
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.dataset import load_worm_data, build_joint_state
from baseline_transformer.model  import TemporalTransformerGaussian
from baseline_transformer.train  import train_single_worm_cv

# ── helpers ──────────────────────────────────────────────────────────────────

def _r2(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m].astype(np.float64), y_pred[m].astype(np.float64)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - np.sum((yt - yp) ** 2) / max(ss_tot, 1e-12))

def pearson_corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return (a * b).sum() / d if d > 1e-10 else np.nan

def _build_lagged(x, n_lags):
    T, D = x.shape
    parts = []
    for lag in range(1, n_lags + 1):
        s = np.zeros((T, D), dtype=x.dtype)
        if lag < T:
            s[lag:] = x[:-lag]
        parts.append(s)
    return np.concatenate(parts, axis=1)

def _make_folds(T, warmup, n_folds=5):
    fs = (T - warmup) // n_folds
    return [(warmup + i * fs,
             warmup + (i + 1) * fs if i < n_folds - 1 else T)
            for i in range(n_folds)]

def beh_metrics(pred_b, gt_b, warmup):
    n_modes = gt_b.shape[1]
    idx = np.arange(warmup, gt_b.shape[0])
    r2   = np.array([_r2(gt_b[idx, j], pred_b[idx, j]) for j in range(n_modes)])
    corr = np.array([pearson_corr(gt_b[idx, j], pred_b[idx, j]) for j in range(n_modes)])
    return r2, corr

def _make_mlp(d_in, K_out, hidden=128, n_layers=2):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(d, K_out))
    return nn.Sequential(*layers)


# ── MLP training helpers ─────────────────────────────────────────────────────

def _train_mlp_rollout_single(X_norm, b, Kw, device, *,
                               epochs=150, lr=1e-3, wd=1e-3, patience=20,
                               rollout_prob=0.5, rollout_len=20,
                               n_windows=50, beh_noise_std=0.1):
    """Train MLP with rollout on the provided (already-split) training set."""
    T  = b.shape[0]
    nv = max(10, int(T * 0.15))
    tr_end = T - nv
    d_in = (X_norm.shape[1] if X_norm is not None else 0) + Kw
    mlp = _make_mlp(d_in, Kw, hidden=128).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)

    b_t = torch.tensor(b, dtype=torch.float32, device=device)
    mu_b, sig_b = b_t[:tr_end].mean(0), b_t[:tr_end].std(0) + 1e-8
    b_n = (b_t - mu_b) / sig_b

    Xn = (torch.tensor(X_norm, dtype=torch.float32, device=device)
           if X_norm is not None else None)

    def _cat(idx, beh):
        return (torch.cat([Xn[idx], beh], dim=1) if Xn is not None
                else beh)

    bvl, bs, pat = float("inf"), None, 0
    for epoch in range(epochs):
        mlp.train()
        starts = np.random.randint(1, max(2, tr_end - rollout_len), size=n_windows)
        rp      = rollout_prob * min(1.0, epoch / 30)
        use_ro  = np.random.random(n_windows)
        tf_s, ro_s = starts[use_ro >= rp], starts[use_ro < rp]
        loss, nt = torch.tensor(0.0, device=device), 0

        # teacher-forced windows
        if len(tf_s) > 0:
            ta = np.concatenate([np.arange(s, min(s + rollout_len, tr_end))
                                 for s in tf_s])
            if len(ta) > 0:
                bi = b_n[ta - 1]
                if beh_noise_std > 0:
                    bi = bi + beh_noise_std * torch.randn_like(bi)
                loss = loss + nn.functional.mse_loss(mlp(_cat(ta, bi)), b_n[ta])
                nt += 1

        # rollout windows
        if len(ro_s) > 0:
            bh = b_n[ro_s - 1]
            preds, tgts = [], []
            for step in range(rollout_len):
                ti = ro_s + step
                ok = ti < tr_end
                if not ok.any():
                    break
                bhat = mlp(_cat(ti, bh))
                preds.append(bhat[ok])
                tgts.append(b_n[ti[ok]])
                bh = bhat.detach()
            if preds:
                loss = loss + nn.functional.mse_loss(torch.cat(preds),
                                                     torch.cat(tgts))
                nt += 1

        if nt > 0:
            opt.zero_grad()
            (loss / nt).backward()
            opt.step()

        # validation (full rollout on held-out tail)
        mlp.eval()
        with torch.no_grad():
            bh, vp = b_n[tr_end - 1:tr_end], []
            Xn_val = Xn[tr_end:T] if Xn is not None else None
            for i in range(nv):
                xt = (torch.cat([Xn_val[i:i+1], bh], dim=1)
                      if Xn is not None else bh)
                bh = mlp(xt)
                vp.append(bh)
            vl = nn.functional.mse_loss(torch.cat(vp), b_n[tr_end:T]).item()
        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
        else:
            pat += 1
        if pat > patience:
            break

    if bs:
        mlp.load_state_dict(bs)
    mlp.eval().cpu()
    return mlp, mu_b.cpu(), sig_b.cpu()


def _cv_mlp_rollout(X_base, b, warmup, device, rollout_k=15):
    """5-fold CV MLP free-run."""
    T, Kw = b.shape
    ho = np.full((T, Kw), np.nan)
    b_t = torch.tensor(b, dtype=torch.float32)
    for ts, te in _make_folds(T, warmup):
        tr = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        if X_base is not None:
            mu, sig = X_base[tr].mean(0), X_base[tr].std(0) + 1e-8
            Xn = (X_base - mu) / sig
            Xtr = np.concatenate([Xn[warmup:ts], Xn[te:T]])
        else:
            Xn, Xtr = None, None
        btr = np.concatenate([b[warmup:ts], b[te:T]])
        seg_a = ts - warmup
        sb = seg_a if 0 < seg_a < len(btr) else None
        mlp, mu_b, sig_b = _train_mlp_rollout_single(
            Xtr, btr, Kw, device, rollout_len=rollout_k)
        with torch.no_grad():
            Xte = (torch.tensor(Xn[ts:te], dtype=torch.float32)
                   if Xn is not None else None)
            bh = (b_t[ts - 1:ts] - mu_b) / sig_b
            for i in range(te - ts):
                xt = (torch.cat([Xte[i:i+1], bh], dim=1)
                      if Xte is not None else bh)
                bh = mlp(xt)
                ho[ts + i] = (bh[0] * sig_b + mu_b).numpy()
    return ho


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1 — MLP predicts neural+beh jointly vs beh only  (multi-task output)
# ═══════════════════════════════════════════════════════════════════════════

def _train_mlp_joint_rollout(X_norm, u_target, b, N_neural, Kw, device, *,
                              epochs=150, lr=1e-3, wd=1e-3, patience=20,
                              rollout_prob=0.5, rollout_len=20,
                              n_windows=50, beh_noise_std=0.1, w_neural=1.0):
    """Train MLP that predicts BOTH neural(t) and beh(t) from
    [neural_lags(t), prev_beh(t-1)].

    Output dimension = N_neural + Kw.  Loss = MSE(neural) + w_neural * MSE(beh).
    Free-run: only beh is fed back autoregressively; neural input is always GT.
    """
    D_out = N_neural + Kw
    T  = b.shape[0]
    nv = max(10, int(T * 0.15))
    tr_end = T - nv
    d_in = (X_norm.shape[1] if X_norm is not None else 0) + Kw
    mlp = _make_mlp(d_in, D_out, hidden=128).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)

    # joint target: [neural, beh]
    y_full = np.concatenate([u_target, b], axis=1).astype(np.float32)
    y_t = torch.tensor(y_full, dtype=torch.float32, device=device)
    mu_y, sig_y = y_t[:tr_end].mean(0), y_t[:tr_end].std(0) + 1e-8
    y_n = (y_t - mu_y) / sig_y

    # For beh input normalisation (separate, since we only feed back beh)
    b_t = torch.tensor(b, dtype=torch.float32, device=device)
    mu_b, sig_b = b_t[:tr_end].mean(0), b_t[:tr_end].std(0) + 1e-8
    b_n = (b_t - mu_b) / sig_b

    Xn = (torch.tensor(X_norm, dtype=torch.float32, device=device)
           if X_norm is not None else None)

    def _cat(idx, beh):
        return (torch.cat([Xn[idx], beh], dim=1) if Xn is not None
                else beh)

    bvl, bs, pat = float("inf"), None, 0
    for epoch in range(epochs):
        mlp.train()
        starts = np.random.randint(1, max(2, tr_end - rollout_len), size=n_windows)
        rp      = rollout_prob * min(1.0, epoch / 30)
        use_ro  = np.random.random(n_windows)
        tf_s, ro_s = starts[use_ro >= rp], starts[use_ro < rp]
        loss, nt = torch.tensor(0.0, device=device), 0

        # teacher-forced
        if len(tf_s) > 0:
            ta = np.concatenate([np.arange(s, min(s + rollout_len, tr_end))
                                 for s in tf_s])
            if len(ta) > 0:
                bi = b_n[ta - 1]
                if beh_noise_std > 0:
                    bi = bi + beh_noise_std * torch.randn_like(bi)
                pred = mlp(_cat(ta, bi))
                # joint loss: neural + beh
                loss_n = nn.functional.mse_loss(pred[:, :N_neural], y_n[ta, :N_neural])
                loss_b = nn.functional.mse_loss(pred[:, N_neural:], y_n[ta, N_neural:])
                loss = loss + loss_n + w_neural * loss_b
                nt += 1

        # rollout (beh is autoregressive, neural input is always GT)
        if len(ro_s) > 0:
            bh = b_n[ro_s - 1]
            preds, tgts = [], []
            for step in range(rollout_len):
                ti = ro_s + step
                ok = ti < tr_end
                if not ok.any():
                    break
                pred = mlp(_cat(ti, bh))
                preds.append(pred[ok])
                tgts.append(y_n[ti[ok]])
                # feed back only the beh part
                bh = pred[:, N_neural:].detach()
            if preds:
                all_pred = torch.cat(preds)
                all_tgt  = torch.cat(tgts)
                loss_n = nn.functional.mse_loss(all_pred[:, :N_neural], all_tgt[:, :N_neural])
                loss_b = nn.functional.mse_loss(all_pred[:, N_neural:], all_tgt[:, N_neural:])
                loss = loss + loss_n + w_neural * loss_b
                nt += 1

        if nt > 0:
            opt.zero_grad()
            (loss / nt).backward()
            opt.step()

        # validation (free-run beh on held-out tail)
        mlp.eval()
        with torch.no_grad():
            bh, vp = b_n[tr_end - 1:tr_end], []
            Xn_val = Xn[tr_end:T] if Xn is not None else None
            for i in range(nv):
                xt = (torch.cat([Xn_val[i:i+1], bh], dim=1)
                      if Xn is not None else bh)
                pred = mlp(xt)
                bh = pred[:, N_neural:]  # feed back beh
                vp.append(pred)
            # validate on beh portion only
            all_vp = torch.cat(vp)
            vl = nn.functional.mse_loss(all_vp[:, N_neural:],
                                        y_n[tr_end:T, N_neural:]).item()
        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
        else:
            pat += 1
        if pat > patience:
            break

    if bs:
        mlp.load_state_dict(bs)
    mlp.eval().cpu()
    return mlp, mu_y.cpu(), sig_y.cpu(), mu_b.cpu(), sig_b.cpu()


def _cv_mlp_joint_rollout(X_base, u_neural, b, warmup, device, rollout_k=15):
    """5-fold CV MLP free-run — joint (neural+beh) output.
    Returns beh-only hold-out predictions for fair comparison."""
    T, Kw = b.shape
    N_neural = u_neural.shape[1]
    ho = np.full((T, Kw), np.nan)
    b_t = torch.tensor(b, dtype=torch.float32)
    for ts, te in _make_folds(T, warmup):
        tr = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        if X_base is not None:
            mu, sig = X_base[tr].mean(0), X_base[tr].std(0) + 1e-8
            Xn = (X_base - mu) / sig
            Xtr = np.concatenate([Xn[warmup:ts], Xn[te:T]])
        else:
            Xn, Xtr = None, None
        btr = np.concatenate([b[warmup:ts], b[te:T]])
        utr = np.concatenate([u_neural[warmup:ts], u_neural[te:T]])
        mlp, mu_y, sig_y, mu_b, sig_b = _train_mlp_joint_rollout(
            Xtr, utr, btr, N_neural, Kw, device, rollout_len=rollout_k)
        with torch.no_grad():
            Xte = (torch.tensor(Xn[ts:te], dtype=torch.float32)
                   if Xn is not None else None)
            bh = (b_t[ts - 1:ts] - mu_b) / sig_b
            for i in range(te - ts):
                xt = (torch.cat([Xte[i:i+1], bh], dim=1)
                      if Xte is not None else bh)
                pred = mlp(xt)
                # extract beh from joint output, denormalise
                beh_pred_normed = pred[:, N_neural:]
                bh = beh_pred_normed  # feed back
                beh_denorm = beh_pred_normed[0] * sig_y[N_neural:] + mu_y[N_neural:]
                ho[ts + i] = beh_denorm.numpy()
    return ho


def test1_mlp_joint_vs_behonly(u_neural, b, K, device, rollout_k=15):
    """Does training the MLP to predict BOTH neural + beh (multi-task)
    improve behaviour prediction vs predicting beh only?

    This mirrors the Transformer's joint-loss design.  Both arms use
    the same input (neural lags + previous beh) and free-run evaluation.

    Compares:
      (a) MLP → [neural, beh]  (joint output, multi-task loss)
      (b) MLP → beh            (beh-only output)
    """
    warmup = K
    X_n = _build_lagged(u_neural, K)

    print("\n  [Test 1a] MLP joint output (→ neural+beh, multi-task loss)...")
    t0 = time.time()
    ho_joint = _cv_mlp_joint_rollout(X_n, u_neural, b, warmup, device,
                                      rollout_k=rollout_k)
    r2_j, corr_j = beh_metrics(ho_joint, b, warmup)
    print(f"    Beh R²={np.nanmean(r2_j):.4f}  corr={np.nanmean(corr_j):.4f}  "
          f"({time.time()-t0:.1f}s)")

    print("  [Test 1b] MLP beh-only output (→ beh only)...")
    t0 = time.time()
    ho_beh = _cv_mlp_rollout(X_n, b, warmup, device, rollout_k=rollout_k)
    r2_b, corr_b = beh_metrics(ho_beh, b, warmup)
    print(f"    Beh R²={np.nanmean(r2_b):.4f}  corr={np.nanmean(corr_b):.4f}  "
          f"({time.time()-t0:.1f}s)")

    delta_r2   = np.nanmean(r2_j) - np.nanmean(r2_b)
    delta_corr = np.nanmean(corr_j) - np.nanmean(corr_b)
    print(f"\n  ΔR²(joint_output − beh_only_output) = {delta_r2:+.4f}")
    print(f"  Δcorr                               = {delta_corr:+.4f}")
    if delta_r2 > 0.01:
        print("  → Joint multi-task training helps: predicting neural regularises beh")
    elif delta_r2 < -0.01:
        print("  → Beh-only is better: joint prediction hurts (capacity split)")
    else:
        print("  → Comparable: multi-task makes ≤0.01 R² difference")

    return {
        "mlp_joint_output_fr":  {"r2": r2_j.tolist(),  "r2_mean": float(np.nanmean(r2_j)),
                                 "corr": corr_j.tolist(), "corr_mean": float(np.nanmean(corr_j))},
        "mlp_behonly_output_fr": {"r2": r2_b.tolist(),  "r2_mean": float(np.nanmean(r2_b)),
                                  "corr": corr_b.tolist(), "corr_mean": float(np.nanmean(corr_b))},
        "delta_r2": float(delta_r2),
        "delta_corr": float(delta_corr),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2 — Attention-based lag selection  (TRF vs MLP, fixed-window 1-step)
# ═══════════════════════════════════════════════════════════════════════════

def _cv_mlp_fast(X, b, warmup, device, epochs=150, patience=20):
    """Simple teacher-forced 1-step MLP for comparison."""
    T, Kw = b.shape
    ho = np.full((T, Kw), np.nan)
    for ts, te in _make_folds(T, warmup):
        tr = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        mu, sig = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xn_tr, Xn_te = (X[tr] - mu) / sig, (X[ts:te] - mu) / sig
        # train
        nv = max(10, int(Xn_tr.shape[0] * 0.15))
        Xt = torch.tensor(Xn_tr[:-nv], dtype=torch.float32, device=device)
        yt = torch.tensor(b[tr][:-nv],  dtype=torch.float32, device=device)
        Xv = torch.tensor(Xn_tr[-nv:],  dtype=torch.float32, device=device)
        yv = torch.tensor(b[tr][-nv:],  dtype=torch.float32, device=device)
        mlp = _make_mlp(Xt.shape[1], Kw).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
        bvl, bs, pat = float("inf"), None, 0
        for _ in range(epochs):
            mlp.train()
            loss = nn.functional.mse_loss(mlp(Xt), yt)
            opt.zero_grad(); loss.backward(); opt.step()
            mlp.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(mlp(Xv), yv).item()
            if vl < bvl - 1e-6:
                bvl, bs, pat = vl, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
            else:
                pat += 1
            if pat > patience:
                break
        if bs:
            mlp.load_state_dict(bs)
        mlp.eval().cpu()
        with torch.no_grad():
            ho[ts:te] = mlp(torch.tensor(Xn_te, dtype=torch.float32)).numpy()
    return ho


def _cv_trf_1step(x_ctx, b, K, n_neural, n_beh, device, epochs=150, patience=20):
    """Teacher-forced 1-step Transformer for comparison."""
    T, Kw = b.shape
    ho = np.full((T, Kw), np.nan)
    for ts, te in _make_folds(T, K):
        tr = np.concatenate([np.arange(K, ts), np.arange(te, T)])
        # normalise
        mu_x, sig_x = x_ctx[tr].mean(0), x_ctx[tr].std(0) + 1e-8
        x_n = ((x_ctx - mu_x) / sig_x).astype(np.float32)
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        X_win = np.stack([x_n[t - K:t] for t in tr])
        y_tr  = b_n[tr]

        cfg = TransformerBaselineConfig()
        cfg.context_length = K
        model = TemporalTransformerGaussian(n_neural=n_neural, n_beh=n_beh,
                                            cfg=cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

        nv = max(10, int(len(tr) * 0.15))
        Xt = torch.tensor(X_win[:-nv], dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr[:-nv],  dtype=torch.float32, device=device)
        Xv = torch.tensor(X_win[-nv:], dtype=torch.float32, device=device)
        yv = torch.tensor(y_tr[-nv:],  dtype=torch.float32, device=device)

        bvl, bs, pat = float("inf"), None, 0
        for ep in range(epochs):
            model.train()
            out = model.predict_mean(Xt)
            loss = nn.functional.mse_loss(out[:, n_neural:], yt)
            opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(
                    model.predict_mean(Xv)[:, n_neural:], yv).item()
            if vl < bvl - 1e-6:
                bvl, bs, pat = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                pat += 1
            if pat > patience:
                break
        if bs:
            model.load_state_dict(bs)
        model.eval()

        X_test = torch.tensor(
            np.stack([x_n[t - K:t] for t in range(ts, te)]),
            dtype=torch.float32, device=device)
        with torch.no_grad():
            ho[ts:te] = (model.predict_mean(X_test)[:, n_neural:].cpu().numpy()
                         * sig_b + mu_b)
        model.cpu()
    return ho


def test2_attention_vs_flat(u_neural, b, K, device):
    """Does attention over time steps help even in 1-step (teacher-forced) mode?

    Both models receive *same* K neural-only context → predict beh.
    If TRF > MLP here, attention-based lag weighting is the key advantage.
    """
    warmup = K
    N, Kw = u_neural.shape[1], b.shape[1]

    # MLP: flattened lagged neural input
    X_flat = _build_lagged(u_neural, K)
    print("\n  [Test 2a] MLP 1-step (flat K neural lags → beh)...")
    t0 = time.time()
    ho_mlp = _cv_mlp_fast(X_flat, b, warmup, device)
    r2_mlp, corr_mlp = beh_metrics(ho_mlp, b, warmup)
    print(f"    R²={np.nanmean(r2_mlp):.4f}  corr={np.nanmean(corr_mlp):.4f}  "
          f"({time.time()-t0:.1f}s)")

    # TRF: windowed neural context, predict beh only
    x_ctx = np.concatenate([u_neural,
                            np.zeros_like(b)], axis=1).astype(np.float32)
    print("  [Test 2b] TRF 1-step (K neural windows → beh)...")
    t0 = time.time()
    ho_trf = _cv_trf_1step(x_ctx, b, K, N, Kw, device)
    r2_trf, corr_trf = beh_metrics(ho_trf, b, warmup)
    print(f"    R²={np.nanmean(r2_trf):.4f}  corr={np.nanmean(corr_trf):.4f}  "
          f"({time.time()-t0:.1f}s)")

    delta = np.nanmean(r2_trf) - np.nanmean(r2_mlp)
    print(f"\n  ΔR²(TRF − MLP, 1-step) = {delta:+.4f}")
    if delta > 0.01:
        print("  → Attention over lags helps: TRF selects relevant lags better")
    elif delta < -0.01:
        print("  → MLP does better: flat concatenation is sufficient here")
    else:
        print("  → Comparable: attention is not the primary advantage")

    return {
        "mlp_1step": {"r2_mean": float(np.nanmean(r2_mlp)),
                      "corr_mean": float(np.nanmean(corr_mlp))},
        "trf_1step": {"r2_mean": float(np.nanmean(r2_trf)),
                      "corr_mean": float(np.nanmean(corr_trf))},
        "delta_r2": float(delta),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 3 — Free-run stability (error accumulation over horizon)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _free_run_trf(model, x_gt, n_neural):
    """Transformer autoregressive free-run."""
    device = next(model.parameters()).device
    T, D = x_gt.shape
    K = model.cfg.context_length
    pred = np.zeros((T, D), dtype=np.float32)
    pred[:K] = x_gt[:K]
    x_gt_t = torch.tensor(x_gt, dtype=torch.float32, device=device)
    buf = x_gt_t[:K].unsqueeze(0).clone()
    for t in range(K, T):
        mu = model.predict_mean(buf)
        pred[t] = mu[0].cpu().numpy()
        next_frame = x_gt_t[t].clone()
        next_frame[n_neural:] = mu[0, n_neural:]
        buf = torch.cat([buf[:, 1:, :],
                         next_frame.unsqueeze(0).unsqueeze(0)], dim=1)
    return pred


def test3_freerun_stability(u_neural, b, K, device):
    """How fast does prediction error grow over time in free-run?

    Computes R² in sliding windows of 50 steps at increasing horizons
    for both TRF and MLP.  Plots error-vs-horizon curves.
    """
    warmup = K
    N, Kw = u_neural.shape[1], b.shape[1]
    X_n = _build_lagged(u_neural, K)

    # --- TRF free-run (use train_single_worm_cv for full joint model) ---
    print("\n  [Test 3a] Training TRF n+b for free-run stability test...")
    t0 = time.time()
    cfg = TransformerBaselineConfig()
    cfg.context_length, cfg.device = K, device
    train_result = train_single_worm_cv(
        u=u_neural, cfg=cfg, device=device, verbose=False,
        save_dir=None, worm_id="test3", b=b, b_mask=None)
    best_model = train_result["best_model"].to(device).eval()
    x_joint, n_neural = train_result["x"], train_result["n_neural"]
    pred_trf_full = _free_run_trf(best_model, x_joint, n_neural)
    pred_trf = pred_trf_full[:, n_neural:]
    best_model.cpu()
    print(f"    TRF trained ({time.time()-t0:.1f}s)")

    # --- MLP free-run ---
    print("  [Test 3b] MLP n+b free-run...")
    t0 = time.time()
    pred_mlp = _cv_mlp_rollout(X_n, b, warmup, device, rollout_k=15)
    print(f"    MLP done ({time.time()-t0:.1f}s)")

    # --- compute R² in sliding windows from warmup onward ---
    win = 50
    T = b.shape[0]
    horizons, r2_trf_h, r2_mlp_h = [], [], []
    for start in range(warmup, T - win, win):
        idx = np.arange(start, start + win)
        h = start - warmup  # steps into free-run
        r2_t = np.nanmean([_r2(b[idx, j], pred_trf[idx, j])
                           for j in range(Kw)])
        r2_m = np.nanmean([_r2(b[idx, j], pred_mlp[idx, j])
                           for j in range(Kw)])
        horizons.append(h)
        r2_trf_h.append(r2_t)
        r2_mlp_h.append(r2_m)

    # --- summary ---
    early = [i for i, h in enumerate(horizons) if h < 200]
    late  = [i for i, h in enumerate(horizons) if h >= 200]
    r2_trf_early = np.nanmean([r2_trf_h[i] for i in early]) if early else np.nan
    r2_mlp_early = np.nanmean([r2_mlp_h[i] for i in early]) if early else np.nan
    r2_trf_late  = np.nanmean([r2_trf_h[i] for i in late])  if late  else np.nan
    r2_mlp_late  = np.nanmean([r2_mlp_h[i] for i in late])  if late  else np.nan
    print(f"\n  Free-run R² (early <200 steps): TRF={r2_trf_early:.3f}  MLP={r2_mlp_early:.3f}")
    print(f"  Free-run R² (late ≥200 steps):  TRF={r2_trf_late:.3f}  MLP={r2_mlp_late:.3f}")
    drop_trf = r2_trf_early - r2_trf_late
    drop_mlp = r2_mlp_early - r2_mlp_late
    print(f"  R² drop (early→late): TRF={drop_trf:.3f}  MLP={drop_mlp:.3f}")
    if drop_trf < drop_mlp - 0.02:
        print("  → TRF is more stable: slower error accumulation")
    else:
        print("  → Similar stability (or MLP more stable)")

    return {
        "horizons": horizons,
        "r2_trf": r2_trf_h,
        "r2_mlp": r2_mlp_h,
        "trf_early": float(r2_trf_early), "trf_late": float(r2_trf_late),
        "mlp_early": float(r2_mlp_early), "mlp_late": float(r2_mlp_late),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 4 — Parameter scaling with K  (curse of dimensionality for MLP)
# ═══════════════════════════════════════════════════════════════════════════

def test4_param_scaling(N_neurons, Kw, K_values=(3, 7, 12, 15, 20),
                         d_model=128, n_heads=4, n_layers=2, d_ff=256,
                         mlp_hidden=128, mlp_layers=2):
    """Print parameter counts for TRF vs MLP at each K.

    TRF params are *constant* w.r.t. K (shared weights across positions).
    MLP params grow as O(K * N) because the input is a flat concatenation.
    """
    print("\n  [Test 4] Parameter count comparison:")
    print(f"  {'K':>4}  {'MLP params':>12}  {'TRF params':>12}  {'MLP/TRF':>8}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*8}")
    results = {}
    for K in K_values:
        # MLP: input = K * N_neurons (lagged neural) + Kw (prev beh)
        d_in_mlp = K * N_neurons + Kw
        mlp = _make_mlp(d_in_mlp, Kw, hidden=mlp_hidden, n_layers=mlp_layers)
        n_mlp = sum(p.numel() for p in mlp.parameters())

        # TRF: input proj + transformer + heads
        D = N_neurons + Kw
        cfg = TransformerBaselineConfig()
        cfg.context_length = K
        cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff = d_model, n_heads, n_layers, d_ff
        trf = TemporalTransformerGaussian(n_neural=N_neurons, n_beh=Kw, cfg=cfg)
        n_trf = sum(p.numel() for p in trf.parameters())

        ratio = n_mlp / n_trf
        print(f"  {K:>4}  {n_mlp:>12,}  {n_trf:>12,}  {ratio:>8.2f}x")
        results[K] = {"mlp": n_mlp, "trf": n_trf, "ratio": round(ratio, 2)}
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results, out_dir, worm_id):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ── Panel 1: Test 1 bar chart ──
    ax = axes[0]
    t1 = results.get("test1", {})
    if t1:
        labels = ["→ n+b\n(joint)", "→ beh\nonly"]
        r2_vals = [t1["mlp_joint_output_fr"]["r2_mean"],
                   t1["mlp_behonly_output_fr"]["r2_mean"]]
        corr_vals = [t1["mlp_joint_output_fr"]["corr_mean"],
                     t1["mlp_behonly_output_fr"]["corr_mean"]]
        x_pos = np.arange(len(labels))
        w = 0.35
        bars1 = ax.bar(x_pos - w/2, r2_vals, w, color=["#348ABD", "#999999"], label="R²")
        bars2 = ax.bar(x_pos + w/2, corr_vals, w, color=["#6BAED6", "#BDBDBD"], label="Corr")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean metric (free-run beh)")
        ax.set_title("Test 1: MLP output joint vs beh-only")
        all_vals = r2_vals + corr_vals
        ax.set_ylim(0, max(0.3, max(v for v in all_vals if np.isfinite(v)) * 1.3))
        for bar, v in zip(list(bars1) + list(bars2), r2_vals + corr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    # ── Panel 2: Test 2 bar chart ──
    ax = axes[1]
    t2 = results.get("test2", {})
    if t2:
        labels = ["MLP\nflat", "TRF\nattn"]
        r2_vals = [t2["mlp_1step"]["r2_mean"], t2["trf_1step"]["r2_mean"]]
        bars = ax.bar(labels, r2_vals, color=["#348ABD", "#E24A33"], width=0.5)
        ax.set_ylabel("Mean R² (1-step)")
        ax.set_title("Test 2: Attention vs flat lags")
        ax.set_ylim(0, max(0.3, max(r2_vals) * 1.3))
        for bar, v in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # ── Panel 3: Test 3 stability curves ──
    ax = axes[2]
    t3 = results.get("test3", {})
    if t3 and t3["horizons"]:
        ax.plot(t3["horizons"], t3["r2_trf"], "o-", color="#E24A33",
                ms=4, lw=1.5, label="TRF n+b FR")
        ax.plot(t3["horizons"], t3["r2_mlp"], "s-", color="#348ABD",
                ms=4, lw=1.5, label="MLP n+b FR")
        ax.set_xlabel("Horizon (steps from warmup)")
        ax.set_ylabel("Windowed R²")
        ax.set_title("Test 3: Free-run stability")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Transformer advantage diagnostics — {worm_id}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = out_dir / f"trf_advantage_diagnostics_{worm_id}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Transformer advantage diagnostics")
    ap.add_argument("--h5", required=True,
                    help="Path to single worm .h5 file")
    ap.add_argument("--out_dir",
                    default="output_plots/behaviour_decoder/trf_diagnostics")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--K", type=int, default=12,
                    help="Context length to use for tests")
    ap.add_argument("--neurons", choices=["motor", "all"], default="all")
    ap.add_argument("--tests", nargs="*", default=["1", "2", "3", "4"],
                    help="Which tests to run (1 2 3 4)")
    args = ap.parse_args()

    device = args.device
    K = args.K
    out_dir = Path(args.out_dir)

    worm_data = load_worm_data(args.h5, n_beh_modes=6)
    u, b = worm_data["u"], worm_data["b"]
    worm_id = worm_data["worm_id"]
    motor_idx = worm_data.get("motor_idx")

    if args.neurons == "motor" and motor_idx:
        u_sel = u[:, motor_idx]
        neuron_label = "motor"
    else:
        u_sel = u
        neuron_label = "all"

    N, Kw = u_sel.shape[1], b.shape[1]
    print(f"Worm: {worm_id}  T={u.shape[0]}  N={N} ({neuron_label})  Kw={Kw}  K={K}")
    print(f"Running tests: {args.tests}")

    results = {"worm_id": worm_id, "K": K, "N": N, "neuron_label": neuron_label}

    if "1" in args.tests:
        print(f"\n{'='*60}")
        print(f"TEST 1: MLP predict neural+beh (joint output) vs beh only")
        print(f"{'='*60}")
        results["test1"] = test1_mlp_joint_vs_behonly(u_sel, b, K, device)

    if "2" in args.tests:
        print(f"\n{'='*60}")
        print(f"TEST 2: Attention vs flat concatenation (1-step, neural→beh)")
        print(f"{'='*60}")
        results["test2"] = test2_attention_vs_flat(u_sel, b, K, device)

    if "3" in args.tests:
        print(f"\n{'='*60}")
        print(f"TEST 3: Free-run error accumulation (TRF vs MLP)")
        print(f"{'='*60}")
        results["test3"] = test3_freerun_stability(u_sel, b, K, device)

    if "4" in args.tests:
        print(f"\n{'='*60}")
        print(f"TEST 4: Parameter count scaling with K")
        print(f"{'='*60}")
        results["test4"] = test4_param_scaling(N, Kw)

    # ── save & plot ──
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"diagnostics_{worm_id}.json"
    json.dump(results, open(json_path, "w"), indent=2,
              default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else None)
    print(f"\n  JSON: {json_path}")

    plot_results(results, out_dir, worm_id)
    print("\nDone.")


if __name__ == "__main__":
    main()
