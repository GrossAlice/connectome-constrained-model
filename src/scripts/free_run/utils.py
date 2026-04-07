"""
Shared utilities for stochastic free-run experiments.

Contains:
  - Data helpers: _build_lagged
  - Model training: train_mlp, train_ridge, _make_mlp
  - Residual noise estimation: estimate_residual_std
  - Distributional metrics: PSD, autocorrelation, Wasserstein, KS, variance ratio
  - Plotting: ensemble traces, PSD, autocorrelation, marginals, summary bars, temperature sweep
  - JSON encoder for numpy types
"""
from __future__ import annotations

import json, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_lagged(x: np.ndarray, n_lags: int) -> np.ndarray:
    """Stack *n_lags* time-shifted copies → (T, D*n_lags)."""
    T, D = x.shape
    parts = []
    for lag in range(1, n_lags + 1):
        s = np.zeros((T, D), dtype=x.dtype)
        if lag < T:
            s[lag:] = x[:-lag]
        parts.append(s)
    return np.concatenate(parts, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────

def _make_mlp(d_in: int, d_out: int, hidden: int = 64, n_layers: int = 2) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def train_mlp(X: np.ndarray, y: np.ndarray, device: str,
              epochs: int = 200, lr: float = 1e-3, wd: float = 1e-3,
              patience: int = 25) -> nn.Sequential:
    """Train MLP with early stopping and return best-val model."""
    nv = max(10, int(X.shape[0] * 0.15))
    Xt = torch.tensor(X[:-nv], dtype=torch.float32, device=device)
    yt = torch.tensor(y[:-nv], dtype=torch.float32, device=device)
    Xv = torch.tensor(X[-nv:], dtype=torch.float32, device=device)
    yv = torch.tensor(y[-nv:], dtype=torch.float32, device=device)

    mlp = _make_mlp(X.shape[1], y.shape[1]).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)

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
    return mlp


def train_mlp_rollout(x_full: np.ndarray, K: int, device: str,
                      rollout_steps: int = 15, epochs: int = 200,
                      lr: float = 1e-3, wd: float = 1e-3,
                      patience: int = 25, batch_size: int = 64) -> nn.Sequential:
    """Train MLP with multi-step rollout loss for free-run robustness.
    
    Uses mini-batched rollout: processes `batch_size` start indices in parallel
    for massive GPU speedup vs the old sequential-per-start loop.
    
    Parameters
    ----------
    x_full : (T, D) normalised time series (joint or single modality)
    K : context length (lag window)
    rollout_steps : number of steps to unroll during training
    batch_size : number of start indices to process in parallel
    """
    T, D = x_full.shape
    d_in = K * D
    
    # Build training segments: need at least K + rollout_steps contiguous
    valid_starts = T - K - rollout_steps
    if valid_starts < 10:
        print(f"  [rollout] Not enough data for rollout={rollout_steps}, falling back to 1-step")
        X_lag = build_lagged(x_full, K)
        return train_mlp(X_lag[K:], x_full[K:], device, epochs, lr, wd, patience)
    
    # Pre-build all segments as a single tensor: (n_starts, K+R, D)
    all_starts = np.arange(valid_starts)
    idx = np.arange(K + rollout_steps)[None, :] + all_starts[:, None]  # (n_starts, K+R)
    segments = torch.tensor(x_full[idx], dtype=torch.float32, device=device)  # (n_starts, K+R, D)
    
    # Train/val split (temporal)
    nv = max(10, int(valid_starts * 0.15))
    n_train = valid_starts - nv
    train_segs = segments[:n_train]  # (n_train, K+R, D)
    val_segs = segments[n_train:]    # (nv, K+R, D)
    
    mlp = _make_mlp(d_in, D).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
    
    bvl, bs, pat = float("inf"), None, 0
    
    for ep in range(epochs):
        mlp.train()
        perm = torch.randperm(n_train, device=device)
        
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i : i + batch_size]
            batch = train_segs[batch_idx]  # (B, K+R, D)
            B = batch.shape[0]
            
            # Initial context from true data
            context = batch[:, :K, :].reshape(B, -1)  # (B, K*D)
            
            loss = torch.tensor(0.0, device=device)
            for r in range(rollout_steps):
                pred = mlp(context)            # (B, D)
                target = batch[:, K + r, :]    # (B, D)
                loss = loss + nn.functional.mse_loss(pred, target)
                # Shift context: drop oldest, append prediction
                context = torch.cat([context[:, D:], pred], dim=1)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        # Validation (1-step, batched)
        mlp.eval()
        with torch.no_grad():
            ctx_v = val_segs[:, :K, :].reshape(nv, -1)   # (nv, K*D)
            pred_v = mlp(ctx_v)                            # (nv, D)
            tgt_v = val_segs[:, K, :]                      # (nv, D)
            val_loss = nn.functional.mse_loss(pred_v, tgt_v).item()
        
        if val_loss < bvl - 1e-6:
            bvl, bs, pat = val_loss, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
        else:
            pat += 1
        if pat > patience:
            break
    
    if bs:
        mlp.load_state_dict(bs)
    mlp.eval().cpu()
    return mlp


def train_mlp_beh_rollout(u_neural: np.ndarray, b_beh: np.ndarray, K_beh: int, 
                          device: str, rollout_steps: int = 30,
                          input_mode: str = "cascaded",
                          epochs: int = 300, lr: float = 1e-3, wd: float = 1e-3,
                          patience: int = 30, batch_size: int = 64) -> nn.Sequential:
    """Train MLP behavior decoder with mini-batched multi-step rollout.
    
    Processes `batch_size` start indices in parallel for massive GPU speedup.
    
    Parameters
    ----------
    u_neural : (T, N) neural activity (normalised)
    b_beh : (T, Kw) behavior/eigenworms (normalised)
    K_beh : behavior context length (lags)
    input_mode : "cascaded" (neural_current + beh_lags) or "decoder" (neural_lags only)
    rollout_steps : number of steps to unroll (should cover ~1-2 locomotion cycles)
    batch_size : number of start indices to process in parallel
    """
    T, N = u_neural.shape
    Kw = b_beh.shape[1]
    
    if input_mode == "cascaded":
        d_in = N + K_beh * Kw
    else:
        d_in = K_beh * N
    d_out = Kw
    
    valid_starts = T - K_beh - rollout_steps
    if valid_starts < 20:
        print(f"  [beh_rollout] Not enough data for rollout={rollout_steps}, reducing")
        rollout_steps = max(5, (T - K_beh) // 2)
        valid_starts = T - K_beh - rollout_steps
    
    # Pre-build indexed tensors on GPU
    u_t = torch.tensor(u_neural, dtype=torch.float32, device=device)
    b_t = torch.tensor(b_beh, dtype=torch.float32, device=device)
    
    # Pre-build behaviour segments: (n_starts, K_beh + rollout, Kw)
    all_starts = np.arange(valid_starts)
    b_idx = np.arange(K_beh + rollout_steps)[None, :] + all_starts[:, None]
    b_segs = b_t[b_idx]  # (n_starts, K_beh+R, Kw)
    
    # Pre-build neural indices for cascaded mode
    if input_mode == "cascaded":
        # For each start, neural_current at times [start+K_beh .. start+K_beh+R-1]
        u_idx = np.arange(rollout_steps)[None, :] + (all_starts[:, None] + K_beh)
        u_for_rollout = u_t[u_idx]  # (n_starts, R, N)
    else:
        # For decoder mode: neural lags at each rollout step
        u_lag_idx = np.stack([
            np.arange(K_beh)[None, :] + (all_starts[:, None] + r)
            for r in range(rollout_steps)
        ], axis=1)  # (n_starts, R, K_beh)
        u_lag_segs = u_t[u_lag_idx]  # (n_starts, R, K_beh, N)
    
    # Train/val split
    nv = max(10, int(valid_starts * 0.15))
    n_train = valid_starts - nv
    
    mlp = _make_mlp(d_in, d_out, hidden=64, n_layers=2).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
    
    bvl, bs, pat = float("inf"), None, 0
    
    for ep in range(epochs):
        mlp.train()
        perm = torch.randperm(n_train, device=device)
        
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i : i + batch_size]
            B = batch_idx.shape[0]
            
            # True behaviour segments for this batch
            b_batch = b_segs[batch_idx]  # (B, K_beh+R, Kw)
            
            # Initialize with true behaviour history
            beh_history = b_batch[:, :K_beh, :].clone()  # (B, K_beh, Kw)
            
            loss = torch.tensor(0.0, device=device)
            for r in range(rollout_steps):
                if input_mode == "cascaded":
                    neural_curr = u_for_rollout[batch_idx, r, :]   # (B, N)
                    beh_lags = beh_history.reshape(B, -1)           # (B, K_beh*Kw)
                    inp = torch.cat([neural_curr, beh_lags], dim=1) # (B, d_in)
                else:
                    neural_lags = u_lag_segs[batch_idx, r, :, :].reshape(B, -1)  # (B, K_beh*N)
                    inp = neural_lags
                
                pred = mlp(inp)                     # (B, Kw)
                target = b_batch[:, K_beh + r, :]   # (B, Kw)
                loss = loss + nn.functional.mse_loss(pred, target)
                
                # Shift behaviour history: drop oldest, append prediction
                beh_history = torch.cat([beh_history[:, 1:, :], pred.unsqueeze(1)], dim=1)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        # Validation (1-step, batched)
        mlp.eval()
        with torch.no_grad():
            val_idx = torch.arange(n_train, valid_starts, device=device)
            b_val = b_segs[val_idx]  # (nv, K_beh+R, Kw)
            
            if input_mode == "cascaded":
                neural_v = u_for_rollout[val_idx, 0, :]              # (nv, N)
                beh_v = b_val[:, :K_beh, :].reshape(nv, -1)          # (nv, K_beh*Kw)
                inp_v = torch.cat([neural_v, beh_v], dim=1)
            else:
                neural_v = u_lag_segs[val_idx, 0, :, :].reshape(nv, -1)
                inp_v = neural_v
            
            pred_v = mlp(inp_v)
            tgt_v = b_val[:, K_beh, :]
            val_loss = nn.functional.mse_loss(pred_v, tgt_v).item()
        
        if val_loss < bvl - 1e-6:
            bvl, bs, pat = val_loss, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
        else:
            pat += 1
        if pat > patience:
            break
    
    if bs:
        mlp.load_state_dict(bs)
    mlp.eval().cpu()
    return mlp


def train_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """Train Ridge regression."""
    from sklearn.linear_model import Ridge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Ridge(alpha=alpha).fit(X, y)


def estimate_residual_std(model, X: np.ndarray, y: np.ndarray,
                          device: str = "cpu", is_torch: bool = True) -> np.ndarray:
    """Per-dimension std of training residuals for MLP/Ridge noise injection.

    Returns
    -------
    residual_std : (D,)
    """
    if is_torch:
        model = model.to(device)
        with torch.no_grad():
            y_hat = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
        model = model.cpu()
    else:
        y_hat = model.predict(X)
    return (y - y_hat).std(axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Distributional metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_psd(x: np.ndarray, fs: float = 1 / 0.6, nperseg: int = 128):
    """Welch PSD per dimension → (freqs, psd_matrix(D, n_freqs))."""
    D = x.shape[1]
    psds = []
    for d in range(D):
        f, p = signal.welch(x[:, d], fs=fs,
                            nperseg=min(nperseg, len(x) // 2),
                            detrend="constant")
        psds.append(p)
    return f, np.array(psds)


def psd_log_distance(psd_gt: np.ndarray, psd_gen: np.ndarray) -> float:
    eps = 1e-10
    log_gt = np.log10(psd_gt + eps)
    log_gen = np.log10(psd_gen + eps)
    return float(np.mean(np.sqrt(np.mean((log_gt - log_gen) ** 2, axis=1))))


def compute_autocorr(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    T, D = x.shape
    max_lag = min(max_lag, T // 2)
    acfs = np.zeros((D, max_lag))
    for d in range(D):
        xd = x[:, d] - x[:, d].mean()
        var = np.sum(xd ** 2)
        if var < 1e-12:
            continue
        for lag in range(max_lag):
            acfs[d, lag] = 1.0 if lag == 0 else np.sum(xd[lag:] * xd[:-lag]) / var
    return acfs


def autocorr_rmse(acf_gt: np.ndarray, acf_gen: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((acf_gt - acf_gen) ** 2, axis=1))))


def marginal_wasserstein(gt: np.ndarray, gen: np.ndarray) -> float:
    D = gt.shape[1]
    dists = []
    for d in range(D):
        g = gt[:, d][np.isfinite(gt[:, d])]
        s = gen[:, d][np.isfinite(gen[:, d])]
        if len(g) > 2 and len(s) > 2:
            dists.append(wasserstein_distance(g, s))
    return float(np.mean(dists)) if dists else float("nan")


def marginal_ks(gt: np.ndarray, gen: np.ndarray) -> float:
    D = gt.shape[1]
    stats = []
    for d in range(D):
        g = gt[:, d][np.isfinite(gt[:, d])]
        s = gen[:, d][np.isfinite(gen[:, d])]
        if len(g) > 2 and len(s) > 2:
            stats.append(ks_2samp(g, s).statistic)
    return float(np.mean(stats)) if stats else float("nan")


def variance_ratio(gt: np.ndarray, gen: np.ndarray):
    var_gt = np.var(gt, axis=0)
    var_gen = np.var(gen, axis=0)
    ratios = var_gen / np.maximum(var_gt, 1e-12)
    return float(np.mean(ratios)), ratios


def compute_distributional_metrics(gt: np.ndarray, gen: np.ndarray,
                                   label: str = "", fs: float = 1 / 0.6):
    """Full suite of distributional metrics between gt and gen (T, D)."""
    f_gt, psd_gt = compute_psd(gt, fs=fs)
    f_gen, psd_gen = compute_psd(gen, fs=fs)
    psd_dist = psd_log_distance(psd_gt, psd_gen)

    acf_gt = compute_autocorr(gt)
    acf_gen = compute_autocorr(gen)
    acf_err = autocorr_rmse(acf_gt, acf_gen)

    w1 = marginal_wasserstein(gt, gen)
    ks = marginal_ks(gt, gen)
    vr_mean, vr_per_dim = variance_ratio(gt, gen)

    mean_err = float(np.mean(np.abs(gt.mean(0) - gen.mean(0))))
    std_err = float(np.mean(np.abs(gt.std(0) - gen.std(0))))

    metrics = {
        "psd_log_distance": psd_dist,
        "autocorr_rmse": acf_err,
        "wasserstein_1": w1,
        "ks_statistic": ks,
        "variance_ratio_mean": vr_mean,
        "variance_ratio_per_dim": vr_per_dim.tolist(),
        "mean_abs_error": mean_err,
        "std_abs_error": std_err,
    }
    if label:
        print(f"  {label}:  PSD={psd_dist:.3f}  ACF={acf_err:.3f}  "
              f"W1={w1:.3f}  KS={ks:.3f}  VarR={vr_mean:.3f}")
    return metrics, (f_gt, psd_gt, psd_gen, acf_gt, acf_gen)


def ensemble_median_metrics(metrics_list: list[dict]) -> dict:
    """Median of metrics across ensemble samples."""
    out = {k: float(np.median([m[k] for m in metrics_list]))
           for k in metrics_list[0] if k != "variance_ratio_per_dim"}
    out["variance_ratio_per_dim"] = np.median(
        [m["variance_ratio_per_dim"] for m in metrics_list], axis=0).tolist()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ["#E24A33", "#348ABD", "#2CA02C", "#9467BD", "#FF7F0E",
          "#D62728", "#1F77B4", "#8C564B", "#E377C2"]


def plot_ensemble_traces(gt_beh: np.ndarray, all_samples_beh: list[np.ndarray],
                         out_dir: Path, tag: str, worm_id: str,
                         n_show: int = 300, dt: float = 0.6, n_traces: int = 10):
    Kw = gt_beh.shape[1]
    n_modes = min(Kw, 4)
    t_sec = np.arange(n_show) * dt

    fig, axes = plt.subplots(n_modes, 1, figsize=(14, 2.5 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    for row in range(n_modes):
        ax = axes[row]
        for i in range(min(n_traces, len(all_samples_beh))):
            ax.plot(t_sec, all_samples_beh[i][:n_show, row],
                    color="#E24A33", alpha=0.15, lw=0.7)
        ens_mean = np.mean([s[:n_show, row] for s in all_samples_beh], axis=0)
        ax.plot(t_sec, ens_mean, color="#E24A33", alpha=0.8, lw=1.5,
                ls="--", label=f"{tag} mean")
        ax.plot(t_sec, gt_beh[:n_show, row], color="#333333", lw=1.5,
                alpha=0.9, label="GT")
        ax.set_ylabel(f"EW{row+1}", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)
        if row == 0:
            ax.legend(fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.suptitle(f"Stochastic {tag} — {worm_id}\n"
                 f"{len(all_samples_beh)} trajectories vs GT",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"ensemble_{tag}_{worm_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


def plot_psd_comparison(f_gt, psd_gt, all_psd_gen, model_names,
                        out_dir: Path, worm_id: str, n_modes_show: int = 4):
    n_modes = min(psd_gt.shape[0], n_modes_show)
    fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 4), sharey=True)
    if n_modes == 1:
        axes = [axes]

    for d in range(n_modes):
        ax = axes[d]
        ax.semilogy(f_gt, psd_gt[d], color="#333333", lw=2, label="GT")
        for i, (name, psds_list) in enumerate(zip(model_names, all_psd_gen)):
            stacked = np.array([p[d] for p in psds_list])
            median = np.median(stacked, axis=0)
            q25 = np.percentile(stacked, 25, axis=0)
            q75 = np.percentile(stacked, 75, axis=0)
            c = COLORS[i % len(COLORS)]
            ax.semilogy(f_gt, median, color=c, lw=1.5, label=name)
            ax.fill_between(f_gt, q25, q75, color=c, alpha=0.15)
        ax.set_xlabel("Freq (Hz)", fontsize=10)
        ax.set_title(f"EW{d+1}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if d == 0:
            ax.set_ylabel("PSD", fontsize=10)
            ax.legend(fontsize=7)

    fig.suptitle(f"PSD — {worm_id}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"psd_{worm_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


def plot_autocorr_comparison(acf_gt, all_acf_gen, model_names,
                             out_dir: Path, worm_id: str,
                             n_modes_show: int = 4, dt: float = 0.6):
    n_modes = min(acf_gt.shape[0], n_modes_show)
    lags = np.arange(acf_gt.shape[1]) * dt
    fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 4), sharey=True)
    if n_modes == 1:
        axes = [axes]

    for d in range(n_modes):
        ax = axes[d]
        ax.plot(lags, acf_gt[d], color="#333333", lw=2, label="GT")
        for i, (name, acfs_list) in enumerate(zip(model_names, all_acf_gen)):
            stacked = np.array([a[d] for a in acfs_list])
            median = np.median(stacked, axis=0)
            q25 = np.percentile(stacked, 25, axis=0)
            q75 = np.percentile(stacked, 75, axis=0)
            c = COLORS[i % len(COLORS)]
            ax.plot(lags, median, color=c, lw=1.5, label=name)
            ax.fill_between(lags, q25, q75, color=c, alpha=0.15)
        ax.set_xlabel("Lag (s)", fontsize=10)
        ax.set_title(f"EW{d+1}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if d == 0:
            ax.set_ylabel("Autocorrelation", fontsize=10)
            ax.legend(fontsize=7)

    fig.suptitle(f"Autocorrelation — {worm_id}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"autocorr_{worm_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


def plot_marginals(gt_beh: np.ndarray, all_samples: list[list[np.ndarray]],
                   model_names: list[str], out_dir: Path, worm_id: str,
                   n_modes_show: int = 4):
    n_modes = min(gt_beh.shape[1], n_modes_show)
    fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 4))
    if n_modes == 1:
        axes = [axes]

    for d in range(n_modes):
        ax = axes[d]
        gt_d = gt_beh[:, d][np.isfinite(gt_beh[:, d])]
        ax.hist(gt_d, bins=40, density=True, alpha=0.5, color="#333333",
                edgecolor="none", label="GT")
        for i, (name, samples_list) in enumerate(zip(model_names, all_samples)):
            pooled = np.concatenate([s[:, d] for s in samples_list])
            pooled = pooled[np.isfinite(pooled)]
            ax.hist(pooled, bins=40, density=True, alpha=0.35,
                    color=COLORS[i % len(COLORS)], edgecolor="none", label=name)
        ax.set_xlabel(f"EW{d+1}", fontsize=10)
        ax.set_title(f"EW{d+1}", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if d == 0:
            ax.set_ylabel("Density", fontsize=10)
            ax.legend(fontsize=7)

    fig.suptitle(f"Marginals — {worm_id}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"marginals_{worm_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


def plot_summary_bars(all_metrics: dict, out_dir: Path, worm_id: str):
    """Bar chart: one bar group per config."""
    model_names = list(all_metrics.keys())
    metric_keys = ["psd_log_distance", "autocorr_rmse", "wasserstein_1",
                   "ks_statistic", "variance_ratio_mean"]
    metric_labels = ["PSD log-dist ↓", "ACF RMSE ↓", "Wasserstein-1 ↓",
                     "KS statistic ↓", "Var ratio (→1)"]

    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[i]
        vals = [all_metrics[m].get(key, 0) for m in model_names]
        x = np.arange(len(model_names))
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=60, ha="right", fontsize=7)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if key == "variance_ratio_mean":
            ax.axhline(y=1.0, color="green", ls="--", alpha=0.5, lw=1)
            ax.set_ylim(0, max(2, max(vals) * 1.1))

    fig.suptitle(f"Distributional Metrics — {worm_id}\n(↓ better, VarRatio→1)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"metrics_{worm_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


def plot_temperature_sweep(temp_results: dict, out_dir: Path,
                           worm_id: str, model_name: str):
    temps = sorted(temp_results.keys())
    metric_keys = ["psd_log_distance", "autocorr_rmse",
                   "variance_ratio_mean", "wasserstein_1"]
    labels = ["PSD log-dist", "ACF RMSE", "Var ratio", "Wasserstein-1"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, (key, label) in enumerate(zip(metric_keys, labels)):
        ax = axes[i]
        vals = [temp_results[t][key] for t in temps]
        ax.plot(temps, vals, "o-", color="#E24A33", lw=2, ms=6)
        ax.set_xlabel("Temperature", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if key == "variance_ratio_mean":
            ax.axhline(y=1.0, color="green", ls="--", alpha=0.5)

    fig.suptitle(f"Temperature Sweep — {model_name} — {worm_id}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"temp_sweep_{model_name}_{worm_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / fname}")


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
