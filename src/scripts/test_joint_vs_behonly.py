#!/usr/bin/env python3
"""Compare joint (neural+beh) vs beh-only prediction for BOTH Transformer and MLP.

All 4 models:
  1. TRF → n+b   (joint Gaussian NLL loss on neural+beh outputs)
  2. TRF → b      (MSE loss only on beh output head, neural output ignored)
  3. MLP → n+b   (joint MSE loss on neural+beh outputs)
  4. MLP → b      (MSE loss on beh output only)

Evaluation:
  - 5-fold contiguous temporal CV
  - Free-run with neuronal activity CLAMPED to ground truth
  - Only BEHAVIOUR is autoregressive
  - Metrics: R² and Pearson correlation on behaviour eigenworms

Run:
    python -m scripts.test_joint_vs_behonly \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2023-01-09-15.h5" \
        --device cuda
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
from baseline_transformer.model import TemporalTransformerGaussian
from baseline_transformer.train import train_single_worm_cv


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    r2 = np.array([_r2(gt_b[idx, j], pred_b[idx, j]) for j in range(n_modes)])
    corr = np.array([pearson_corr(gt_b[idx, j], pred_b[idx, j]) for j in range(n_modes)])
    return r2, corr


def _make_mlp(d_in, K_out, hidden=128, n_layers=2):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden),
                   nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(d, K_out))
    return nn.Sequential(*layers)


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMER: free-run with neural clamped
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def free_run_autonomous(model, x_gt, n_neural):
    """Free-run: neural activity clamped to GT, only beh is autoregressive."""
    device = next(model.parameters()).device
    T, D = x_gt.shape
    K = model.cfg.context_length
    pred = np.zeros((T, D), dtype=np.float32)
    pred[:K] = x_gt[:K]
    x_gt_t = torch.tensor(x_gt, dtype=torch.float32, device=device)
    buf = x_gt_t[:K].unsqueeze(0).clone()
    for t in range(K, T):
        mu = model.predict_mean(buf)
        mu_np = mu[0].cpu().numpy()
        pred[t] = mu_np
        next_frame = x_gt_t[t].clone()       # neural = GT
        next_frame[n_neural:] = mu[0, n_neural:]  # beh = predicted
        buf = torch.cat([buf[:, 1:, :],
                         next_frame.unsqueeze(0).unsqueeze(0)], dim=1)
    return pred


# ── Model 1: TRF → n+b (joint loss) ─────────────────────────────────────────

def cv_trf_joint(u, b, b_mask, K, device):
    """Transformer trained with joint Gaussian NLL on neural+beh.
    Per-fold free-run with neural clamped → evaluate beh on held-out only."""
    cfg = TransformerBaselineConfig()
    cfg.context_length = K
    cfg.device = device
    cfg.predict_beh = True
    cfg.include_beh_input = True

    train_result = train_single_worm_cv(
        u=u, cfg=cfg, device=device, verbose=True,
        save_dir=None, worm_id="trf_joint", b=b, b_mask=b_mask,
    )
    x = train_result["x"]
    n_neural = train_result["n_neural"]
    n_beh = train_result["n_beh"]
    T = x.shape[0]
    Kw = b.shape[1]

    # Per-fold free-run: each fold's model only predicts its own test region
    ho = np.full((T, Kw), np.nan)
    fold_models = train_result["models"]
    folds = train_result["folds"]

    for fi, (fold_model, fold_spec) in enumerate(zip(fold_models, folds)):
        te_s, te_e = fold_spec["test"]
        fold_model = fold_model.to(device).eval()
        pred_fold = _trf_freerun_fold(fold_model, x, n_neural=n_neural,
                                       ts=te_s, te=te_e, K=K, device=device)
        # x is already in normalised joint-state space; the model predicts in
        # that same space, so we need to denormalise beh.  But build_joint_state
        # does NOT normalise — it concatenates raw u and b.  So pred_fold is
        # already in raw beh space.
        ho[te_s:te_e] = pred_fold
        fold_model.cpu()
        print(f"    Fold {fi+1}: free-run test=[{te_s},{te_e}) done")

    return ho


# ── Model 2: TRF → b only (beh-only loss, but neural in context) ────────────

def _cv_trf_behonly(u, b, K, device):
    """Transformer with neural+beh state, but loss ONLY on beh output.
    5-fold CV, free-run with neural clamped → evaluate beh."""
    T, N = u.shape
    Kw = b.shape[1]
    warmup = K
    ho = np.full((T, Kw), np.nan)

    for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
        tr = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        # normalise neural
        mu_u, sig_u = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu_u) / sig_u).astype(np.float32)
        # normalise beh
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        # joint state: [neural, beh]
        x_full = np.concatenate([u_n, b_n], axis=1)

        # build context windows for training indices
        X_win = np.stack([x_full[t - K:t] for t in tr])
        y_beh = b_n[tr]  # target is beh only

        cfg = TransformerBaselineConfig()
        cfg.context_length = K
        model = TemporalTransformerGaussian(
            n_neural=N, n_beh=Kw, cfg=cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

        nv = max(10, int(len(tr) * 0.15))
        Xt = torch.tensor(X_win[:-nv], dtype=torch.float32, device=device)
        yt = torch.tensor(y_beh[:-nv], dtype=torch.float32, device=device)
        Xv = torch.tensor(X_win[-nv:], dtype=torch.float32, device=device)
        yv = torch.tensor(y_beh[-nv:], dtype=torch.float32, device=device)

        bvl, bs, pat = float('inf'), None, 0
        for ep in range(200):
            model.train()
            out = model.predict_mean(Xt)
            loss = nn.functional.mse_loss(out[:, N:], yt)  # beh-only loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(
                    model.predict_mean(Xv)[:, N:], yv).item()
            if vl < bvl - 1e-6:
                bvl = vl
                bs = {k: v.clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
            if pat > 20:
                break
        if bs:
            model.load_state_dict(bs)
        model.eval()

        # Free-run on test fold with neural clamped
        pred_fold = _trf_freerun_fold(model, x_full, n_neural=N, ts=ts,
                                       te=te, K=K, device=device)
        ho[ts:te] = pred_fold * sig_b + mu_b
        model.cpu()
        print(f"    Fold {fi+1}: test=[{ts},{te}) done "
              f"(best_val_loss={bvl:.4f}, epochs={ep+1})")
    return ho


@torch.no_grad()
def _trf_freerun_fold(model, x_full, n_neural, ts, te, K, device):
    """Free-run on one fold's test region with neural clamped to GT."""
    x_gt_t = torch.tensor(x_full, dtype=torch.float32, device=device)
    Kw = x_full.shape[1] - n_neural
    ho = np.zeros((te - ts, Kw), dtype=np.float32)

    # initialise buffer from warmup before test start
    buf_start = max(0, ts - K)
    buf = x_gt_t[buf_start:ts].unsqueeze(0).clone()
    # pad if needed
    if buf.shape[1] < K:
        pad = torch.zeros(1, K - buf.shape[1], x_full.shape[1],
                          dtype=torch.float32, device=device)
        buf = torch.cat([pad, buf], dim=1)

    for i in range(te - ts):
        t = ts + i
        mu = model.predict_mean(buf)
        beh_pred = mu[0, n_neural:]
        ho[i] = beh_pred.cpu().numpy()

        # build next frame: neural from GT, beh from prediction
        next_frame = x_gt_t[t].clone()
        next_frame[n_neural:] = beh_pred
        buf = torch.cat([buf[:, 1:, :],
                         next_frame.unsqueeze(0).unsqueeze(0)], dim=1)
    return ho


# ══════════════════════════════════════════════════════════════════════════════
#  MLP: free-run with neural clamped
# ══════════════════════════════════════════════════════════════════════════════

def _train_mlp_rollout(X_norm, b, Kw, device, *,
                        epochs=150, lr=1e-3, wd=1e-3, patience=20,
                        rollout_prob=0.5, rollout_len=20,
                        n_windows=50, beh_noise_std=0.1):
    """Train MLP predicting beh only, with rollout training."""
    T = b.shape[0]
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
        return (torch.cat([Xn[idx], beh], dim=1)
                if Xn is not None else beh)

    bvl, bs, pat = float("inf"), None, 0
    for epoch in range(epochs):
        mlp.train()
        starts = np.random.randint(1, max(2, tr_end - rollout_len), size=n_windows)
        rp = rollout_prob * min(1.0, epoch / 30)
        use_ro = np.random.random(n_windows)
        tf_s, ro_s = starts[use_ro >= rp], starts[use_ro < rp]
        loss, nt = torch.tensor(0.0, device=device), 0

        if len(tf_s) > 0:
            ta = np.concatenate([np.arange(s, min(s + rollout_len, tr_end))
                                 for s in tf_s])
            if len(ta) > 0:
                bi = b_n[ta - 1]
                if beh_noise_std > 0:
                    bi = bi + beh_noise_std * torch.randn_like(bi)
                loss = loss + nn.functional.mse_loss(mlp(_cat(ta, bi)), b_n[ta])
                nt += 1

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
                loss = loss + nn.functional.mse_loss(
                    torch.cat(preds), torch.cat(tgts))
                nt += 1

        if nt > 0:
            opt.zero_grad()
            (loss / nt).backward()
            opt.step()

        mlp.eval()
        with torch.no_grad():
            bh, vp = b_n[tr_end - 1:tr_end], []
            Xn_val = Xn[tr_end:T] if Xn is not None else None
            for i in range(nv):
                xt = (torch.cat([Xn_val[i:i + 1], bh], dim=1)
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


# ── Model 4: MLP → b only ───────────────────────────────────────────────────

def cv_mlp_behonly(X_base, b, warmup, device, rollout_k=15):
    """5-fold CV MLP → beh only, free-run with neural lags as input."""
    T, Kw = b.shape
    ho = np.full((T, Kw), np.nan)
    b_t = torch.tensor(b, dtype=torch.float32)
    for ts, te in _make_folds(T, warmup):
        tr = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        mu, sig = X_base[tr].mean(0), X_base[tr].std(0) + 1e-8
        Xn = (X_base - mu) / sig
        Xtr = np.concatenate([Xn[warmup:ts], Xn[te:T]])
        btr = np.concatenate([b[warmup:ts], b[te:T]])
        mlp, mu_b, sig_b = _train_mlp_rollout(
            Xtr, btr, Kw, device, rollout_len=rollout_k)
        with torch.no_grad():
            Xte = torch.tensor(Xn[ts:te], dtype=torch.float32)
            bh = (b_t[ts - 1:ts] - mu_b) / sig_b
            for i in range(te - ts):
                xt = torch.cat([Xte[i:i + 1], bh], dim=1)
                bh = mlp(xt)
                ho[ts + i] = (bh[0] * sig_b + mu_b).numpy()
    return ho


# ── Model 3: MLP → n+b (joint output) ───────────────────────────────────────

def _train_mlp_joint_rollout(X_norm, u_target, b, N_neural, Kw, device, *,
                              epochs=150, lr=1e-3, wd=1e-3, patience=20,
                              rollout_prob=0.5, rollout_len=20,
                              n_windows=50, beh_noise_std=0.1):
    """MLP predicting [neural, beh] jointly. Free-run: only beh fed back."""
    D_out = N_neural + Kw
    T = b.shape[0]
    nv = max(10, int(T * 0.15))
    tr_end = T - nv
    d_in = (X_norm.shape[1] if X_norm is not None else 0) + Kw
    mlp = _make_mlp(d_in, D_out, hidden=128).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)

    y_full = np.concatenate([u_target, b], axis=1).astype(np.float32)
    y_t = torch.tensor(y_full, dtype=torch.float32, device=device)
    mu_y, sig_y = y_t[:tr_end].mean(0), y_t[:tr_end].std(0) + 1e-8
    y_n = (y_t - mu_y) / sig_y

    b_t = torch.tensor(b, dtype=torch.float32, device=device)
    mu_b, sig_b = b_t[:tr_end].mean(0), b_t[:tr_end].std(0) + 1e-8
    b_n = (b_t - mu_b) / sig_b

    Xn = (torch.tensor(X_norm, dtype=torch.float32, device=device)
           if X_norm is not None else None)

    def _cat(idx, beh):
        return (torch.cat([Xn[idx], beh], dim=1)
                if Xn is not None else beh)

    bvl, bs, pat = float("inf"), None, 0
    for epoch in range(epochs):
        mlp.train()
        starts = np.random.randint(1, max(2, tr_end - rollout_len), size=n_windows)
        rp = rollout_prob * min(1.0, epoch / 30)
        use_ro = np.random.random(n_windows)
        tf_s, ro_s = starts[use_ro >= rp], starts[use_ro < rp]
        loss, nt = torch.tensor(0.0, device=device), 0

        if len(tf_s) > 0:
            ta = np.concatenate([np.arange(s, min(s + rollout_len, tr_end))
                                 for s in tf_s])
            if len(ta) > 0:
                bi = b_n[ta - 1]
                if beh_noise_std > 0:
                    bi = bi + beh_noise_std * torch.randn_like(bi)
                pred = mlp(_cat(ta, bi))
                loss_n = nn.functional.mse_loss(pred[:, :N_neural],
                                                y_n[ta, :N_neural])
                loss_b = nn.functional.mse_loss(pred[:, N_neural:],
                                                y_n[ta, N_neural:])
                loss = loss + loss_n + loss_b
                nt += 1

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
                bh = pred[:, N_neural:].detach()  # feed back beh only
            if preds:
                all_pred = torch.cat(preds)
                all_tgt = torch.cat(tgts)
                loss_n = nn.functional.mse_loss(all_pred[:, :N_neural],
                                                all_tgt[:, :N_neural])
                loss_b = nn.functional.mse_loss(all_pred[:, N_neural:],
                                                all_tgt[:, N_neural:])
                loss = loss + loss_n + loss_b
                nt += 1

        if nt > 0:
            opt.zero_grad()
            (loss / nt).backward()
            opt.step()

        mlp.eval()
        with torch.no_grad():
            bh, vp = b_n[tr_end - 1:tr_end], []
            Xn_val = Xn[tr_end:T] if Xn is not None else None
            for i in range(nv):
                xt = (torch.cat([Xn_val[i:i + 1], bh], dim=1)
                      if Xn is not None else bh)
                pred = mlp(xt)
                bh = pred[:, N_neural:]  # feed back beh
                vp.append(pred)
            all_vp = torch.cat(vp)
            vl = nn.functional.mse_loss(
                all_vp[:, N_neural:], y_n[tr_end:T, N_neural:]).item()
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


def cv_mlp_joint(X_base, u_neural, b, warmup, device, rollout_k=15):
    """5-fold CV MLP → joint [neural, beh], free-run with neural clamped."""
    T, Kw = b.shape
    N_neural = u_neural.shape[1]
    ho = np.full((T, Kw), np.nan)
    b_t = torch.tensor(b, dtype=torch.float32)
    for ts, te in _make_folds(T, warmup):
        tr = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        mu, sig = X_base[tr].mean(0), X_base[tr].std(0) + 1e-8
        Xn = (X_base - mu) / sig
        Xtr = np.concatenate([Xn[warmup:ts], Xn[te:T]])
        btr = np.concatenate([b[warmup:ts], b[te:T]])
        utr = np.concatenate([u_neural[warmup:ts], u_neural[te:T]])
        mlp, mu_y, sig_y, mu_b, sig_b = _train_mlp_joint_rollout(
            Xtr, utr, btr, N_neural, Kw, device, rollout_len=rollout_k)
        with torch.no_grad():
            Xte = torch.tensor(Xn[ts:te], dtype=torch.float32)
            bh = (b_t[ts - 1:ts] - mu_b) / sig_b
            for i in range(te - ts):
                xt = torch.cat([Xte[i:i + 1], bh], dim=1)
                pred = mlp(xt)
                beh_normed = pred[:, N_neural:]
                bh = beh_normed  # feed back beh
                ho[ts + i] = (beh_normed[0] * sig_y[N_neural:]
                               + mu_y[N_neural:]).numpy()
    return ho


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_bar_comparison(results, out_dir, worm_id):
    """Bar chart comparing R² and correlation for all 4 models."""
    models = ["TRF → n+b", "TRF → b", "MLP → n+b", "MLP → b"]
    keys = ["trf_joint", "trf_behonly", "mlp_joint", "mlp_behonly"]
    colors = ["#E24A33", "#FF9966", "#348ABD", "#7CB9E8"]

    r2_means = [results[k]["r2_mean"] for k in keys]
    corr_means = [results[k]["corr_mean"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.6

    # R²
    ax = axes[0]
    bars = ax.bar(x, r2_means, w, color=colors, edgecolor="black", lw=0.5)
    for bar, v in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Mean R² (beh)")
    ax.set_title("Behaviour R² — Free-run (neural clamped)")
    ax.set_ylim(0, max(max(r2_means) * 1.3, 0.3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Correlation
    ax = axes[1]
    bars = ax.bar(x, corr_means, w, color=colors, edgecolor="black", lw=0.5)
    for bar, v in zip(bars, corr_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Mean Pearson corr (beh)")
    ax.set_title("Behaviour Correlation — Free-run (neural clamped)")
    ax.set_ylim(0, max(max(corr_means) * 1.3, 0.3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Joint (n+b) vs Beh-only prediction — {worm_id}\n"
                 f"5-fold CV, free-run with neural clamped",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = out_dir / f"joint_vs_behonly_{worm_id}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_per_mode(results, n_modes, out_dir, worm_id):
    """Per-eigenworm R² comparison."""
    models = ["TRF → n+b", "TRF → b", "MLP → n+b", "MLP → b"]
    keys = ["trf_joint", "trf_behonly", "mlp_joint", "mlp_behonly"]
    colors = ["#E24A33", "#FF9966", "#348ABD", "#7CB9E8"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_modes)
    w = 0.18
    for i, (key, name, color) in enumerate(zip(keys, models, colors)):
        r2_vals = results[key]["r2"]
        ax.bar(x + i * w, r2_vals, w, label=name, color=color,
               edgecolor="black", lw=0.3)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([f"EW{j+1}" for j in range(n_modes)], fontsize=10)
    ax.set_ylabel("R²")
    ax.set_title(f"Per-Eigenworm R² — {worm_id}\n"
                 f"Free-run (neural clamped, 5-fold CV)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / f"per_mode_r2_{worm_id}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_ew_traces(predictions, gt_b, warmup, out_dir, worm_id,
                   n_frames=200, dt=0.6):
    """Eigenworm trace comparison for all 4 models."""
    n_modes = gt_b.shape[1]
    T = gt_b.shape[0]
    start = warmup + 50
    end = min(start + n_frames, T)
    t_idx = np.arange(start, end)
    t_sec = t_idx * dt

    models = list(predictions.keys())
    n_models = len(models)
    colors = {"TRF → n+b": "#E24A33", "TRF → b": "#FF9966",
              "MLP → n+b": "#348ABD", "MLP → b": "#7CB9E8"}

    fig, axes = plt.subplots(n_modes, n_models, figsize=(4 * n_models, 2 * n_modes),
                             sharex=True, squeeze=False)
    for col, name in enumerate(models):
        pred = predictions[name]
        c = colors.get(name, "#E24A33")
        for row in range(n_modes):
            ax = axes[row, col]
            gt = gt_b[t_idx, row]
            pr = pred[t_idx, row]
            ax.plot(t_sec, gt, color="#333", lw=1.2, alpha=0.8, label="GT")
            ax.plot(t_sec, pr, color=c, lw=1.0, alpha=0.9, ls="--", label="Pred")
            valid = np.isfinite(gt) & np.isfinite(pr)
            if valid.sum() > 2:
                r2 = _r2(gt[valid], pr[valid])
                ax.text(0.98, 0.95, f"R²={r2:.2f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            ax.set_ylabel(f"EW{row+1}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 0:
                ax.set_title(name, fontsize=10, fontweight="bold")
            if row == n_modes - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Eigenworm Traces — {worm_id}  "
                 f"(frames {start}–{end}, neural clamped)", fontsize=12)
    fig.tight_layout()
    fname = out_dir / f"ew_traces_{worm_id}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True,
                    help="Path to worm h5 file")
    ap.add_argument("--out_dir",
                    default="output_plots/behaviour_decoder/joint_vs_behonly")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--K", type=int, default=15,
                    help="Context length / number of lags")
    ap.add_argument("--neurons", choices=["motor", "all"], default="all")
    args = ap.parse_args()

    device = args.device
    K = args.K
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    worm_data = load_worm_data(args.h5, n_beh_modes=6)
    u = worm_data["u"]
    b = worm_data["b"]
    b_mask = worm_data["b_mask"]
    worm_id = worm_data["worm_id"]
    motor_idx = worm_data.get("motor_idx")

    if args.neurons == "motor" and motor_idx is not None:
        u = u[:, motor_idx]
        neuron_label = "motor"
    else:
        neuron_label = "all"

    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]
    warmup = K
    print(f"Worm: {worm_id}  T={T}  N={N} ({neuron_label})  Kw={Kw}  K={K}")
    print(f"Output: {out_dir}")
    print(f"Device: {device}")
    print(f"Evaluation: free-run with neural clamped, 5-fold CV\n")

    results = {}
    X_n = _build_lagged(u, K)   # lagged neural features for MLP

    # ── 1. TRF → n+b (joint loss) ──
    print("=" * 60)
    print("  Model 1/4: TRF → n+b (joint Gaussian NLL loss)")
    print("=" * 60)
    t0 = time.time()
    pred_trf_joint = cv_trf_joint(u, b, b_mask, K, device)
    r2_tj, corr_tj = beh_metrics(pred_trf_joint, b, warmup)
    dt1 = time.time() - t0
    results["trf_joint"] = {
        "r2": r2_tj.tolist(), "r2_mean": float(np.nanmean(r2_tj)),
        "corr": corr_tj.tolist(), "corr_mean": float(np.nanmean(corr_tj)),
        "time_s": round(dt1, 1),
    }
    print(f"  TRF→n+b: R²={np.nanmean(r2_tj):.4f}  "
          f"corr={np.nanmean(corr_tj):.4f}  ({dt1:.1f}s)\n")

    # ── 2. TRF → b only (beh-only loss) ──
    print("=" * 60)
    print("  Model 2/4: TRF → b only (MSE beh-only loss)")
    print("=" * 60)
    t0 = time.time()
    pred_trf_beh = _cv_trf_behonly(u, b, K, device)
    r2_tb, corr_tb = beh_metrics(pred_trf_beh, b, warmup)
    dt2 = time.time() - t0
    results["trf_behonly"] = {
        "r2": r2_tb.tolist(), "r2_mean": float(np.nanmean(r2_tb)),
        "corr": corr_tb.tolist(), "corr_mean": float(np.nanmean(corr_tb)),
        "time_s": round(dt2, 1),
    }
    print(f"  TRF→b:   R²={np.nanmean(r2_tb):.4f}  "
          f"corr={np.nanmean(corr_tb):.4f}  ({dt2:.1f}s)\n")

    # ── 3. MLP → n+b (joint output) ──
    print("=" * 60)
    print("  Model 3/4: MLP → n+b (joint MSE loss)")
    print("=" * 60)
    t0 = time.time()
    pred_mlp_joint = cv_mlp_joint(X_n, u, b, warmup, device, rollout_k=K)
    r2_mj, corr_mj = beh_metrics(pred_mlp_joint, b, warmup)
    dt3 = time.time() - t0
    results["mlp_joint"] = {
        "r2": r2_mj.tolist(), "r2_mean": float(np.nanmean(r2_mj)),
        "corr": corr_mj.tolist(), "corr_mean": float(np.nanmean(corr_mj)),
        "time_s": round(dt3, 1),
    }
    print(f"  MLP→n+b: R²={np.nanmean(r2_mj):.4f}  "
          f"corr={np.nanmean(corr_mj):.4f}  ({dt3:.1f}s)\n")

    # ── 4. MLP → b only ──
    print("=" * 60)
    print("  Model 4/4: MLP → b only (beh-only MSE loss)")
    print("=" * 60)
    t0 = time.time()
    pred_mlp_beh = cv_mlp_behonly(X_n, b, warmup, device, rollout_k=K)
    r2_mb, corr_mb = beh_metrics(pred_mlp_beh, b, warmup)
    dt4 = time.time() - t0
    results["mlp_behonly"] = {
        "r2": r2_mb.tolist(), "r2_mean": float(np.nanmean(r2_mb)),
        "corr": corr_mb.tolist(), "corr_mean": float(np.nanmean(corr_mb)),
        "time_s": round(dt4, 1),
    }
    print(f"  MLP→b:   R²={np.nanmean(r2_mb):.4f}  "
          f"corr={np.nanmean(corr_mb):.4f}  ({dt4:.1f}s)\n")

    # ── Summary ──
    print("=" * 60)
    print("  SUMMARY — Behaviour R² (free-run, neural clamped, 5-fold CV)")
    print("=" * 60)
    for label, key in [("TRF → n+b", "trf_joint"), ("TRF → b", "trf_behonly"),
                       ("MLP → n+b", "mlp_joint"), ("MLP → b", "mlp_behonly")]:
        r = results[key]
        print(f"  {label:12s}  R²={r['r2_mean']:.4f}  corr={r['corr_mean']:.4f}  "
              f"({r['time_s']:.0f}s)")
    print()
    delta_trf = results["trf_joint"]["r2_mean"] - results["trf_behonly"]["r2_mean"]
    delta_mlp = results["mlp_joint"]["r2_mean"] - results["mlp_behonly"]["r2_mean"]
    print(f"  ΔR² TRF (joint−behonly) = {delta_trf:+.4f}")
    print(f"  ΔR² MLP (joint−behonly) = {delta_mlp:+.4f}")

    # Save results
    results["worm_id"] = worm_id
    results["K"] = K
    results["N"] = N
    results["T"] = T
    results["neurons"] = neuron_label
    json_path = out_dir / f"results_{worm_id}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_bar_comparison(results, out_dir, worm_id)
    plot_per_mode(results, Kw, out_dir, worm_id)

    predictions = {
        "TRF → n+b": pred_trf_joint,
        "TRF → b": pred_trf_beh,
        "MLP → n+b": pred_mlp_joint,
        "MLP → b": pred_mlp_beh,
    }
    plot_ew_traces(predictions, b, warmup, out_dir, worm_id)

    print(f"\nDone! Total time: {dt1 + dt2 + dt3 + dt4:.0f}s")


if __name__ == "__main__":
    main()
