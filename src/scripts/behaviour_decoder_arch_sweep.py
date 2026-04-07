#!/usr/bin/env python
"""Architecture sweep for Transformer and MLP behaviour decoders.

Tests 5 configurations per model type at K=10 to find optimal architecture.

Author: Copilot
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.dataset import load_worm_data as _load_worm_data_baseline
from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.model import TemporalTransformerGaussian


def load_worm_data(h5_path, n_beh_modes=6):
    data = _load_worm_data_baseline(h5_path, n_beh_modes=n_beh_modes)
    return {
        "u": data["u"],
        "b": data["b"],
        "worm_id": data["worm_id"],
        "motor_idx": data.get("motor_idx"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

# 5 Transformer configs: small → large
TRF_CONFIGS = {
    "trf_tiny":   {"d_model": 32,  "n_heads": 2, "n_layers": 1, "d_ff": 64},
    "trf_small":  {"d_model": 64,  "n_heads": 4, "n_layers": 1, "d_ff": 128},
    "trf_medium": {"d_model": 128, "n_heads": 4, "n_layers": 2, "d_ff": 256},
    "trf_large":  {"d_model": 256, "n_heads": 8, "n_layers": 2, "d_ff": 512},
    "trf_xlarge": {"d_model": 512, "n_heads": 8, "n_layers": 3, "d_ff": 1024},
}

# 5 MLP configs: small → large
MLP_CONFIGS = {
    "mlp_tiny":   {"hidden": 32,  "n_layers": 1},
    "mlp_small":  {"hidden": 64,  "n_layers": 2},
    "mlp_medium": {"hidden": 128, "n_layers": 2},
    "mlp_large":  {"hidden": 256, "n_layers": 3},
    "mlp_xlarge": {"hidden": 512, "n_layers": 4},
}

K = 10  # Fixed context length
RECLAMP_FRAMES = 40  # 10 seconds at 4Hz


def _r2(gt, pred):
    ss_res = np.nansum((gt - pred) ** 2)
    ss_tot = np.nansum((gt - np.nanmean(gt)) ** 2) + 1e-12
    return 1 - ss_res / ss_tot


def _corr(gt, pred):
    valid = np.isfinite(gt) & np.isfinite(pred)
    if valid.sum() < 3:
        return np.nan
    return np.corrcoef(gt[valid], pred[valid])[0, 1]


def beh_metrics(pred, gt):
    n_modes = pred.shape[1]
    r2_per = np.array([_r2(gt[:, i], pred[:, i]) for i in range(n_modes)])
    corr_per = np.array([_corr(gt[:, i], pred[:, i]) for i in range(n_modes)])
    return np.nanmean(r2_per), np.nanmean(corr_per)


def _make_folds(T, warmup, n_folds=5):
    fold_size = (T - warmup) // n_folds
    folds = []
    for i in range(n_folds):
        ts = warmup + i * fold_size
        te = warmup + (i + 1) * fold_size if i < n_folds - 1 else T
        folds.append((ts, te))
    return folds


def _get_train_indices(T, warmup, ts, te, buffer):
    before = np.arange(warmup, ts)
    after_start = min(te + buffer, T)
    after = np.arange(after_start, T)
    return np.concatenate([before, after])


def _build_lagged(x, K):
    T, D = x.shape
    out = np.zeros((T, K * D), dtype=x.dtype)
    for lag in range(1, K + 1):
        out[lag:, (lag-1)*D:lag*D] = x[:-lag]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════════════════════

def _make_mlp(d_in, K_out, hidden=128, n_layers=2):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(d, K_out))
    return nn.Sequential(*layers)


def _train_mlp(X, y, K_out, device, hidden=128, n_layers=2, epochs=150, lr=1e-3, wd=1e-3, patience=20):
    nv = max(10, int(X.shape[0] * 0.15))
    Xt = torch.tensor(X[:-nv], dtype=torch.float32, device=device)
    yt = torch.tensor(y[:-nv], dtype=torch.float32, device=device)
    Xv = torch.tensor(X[-nv:], dtype=torch.float32, device=device)
    yv = torch.tensor(y[-nv:], dtype=torch.float32, device=device)
    
    mlp = _make_mlp(X.shape[1], K_out, hidden=hidden, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in mlp.parameters())
    
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
    bvl, bs, pat = float('inf'), None, 0
    
    for _ in range(epochs):
        mlp.train()
        loss = nn.functional.mse_loss(mlp(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(mlp(Xv), yv).item()
        if vl < bvl - 1e-6: bvl, bs, pat = vl, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
        else: pat += 1
        if pat > patience: break
    
    if bs: mlp.load_state_dict(bs)
    mlp.eval().cpu()
    mlp.n_params = n_params
    return mlp


def cv_mlp(u, b, device, hidden=128, n_layers=2):
    """MLP: n+b context → behavior, 1-step (teacher-forced for speed)."""
    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]
    warmup = K
    ho = np.full((T, Kw), np.nan)
    n_params = 0

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K)
        
        mu_u, sig_u = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu_u) / sig_u).astype(np.float32)
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        X_u = _build_lagged(u_n, K)
        X_b = _build_lagged(b_n, K)
        X = np.concatenate([X_u, X_b], axis=1)
        
        mlp = _train_mlp(X[tr], b_n[tr], Kw, device, hidden=hidden, n_layers=n_layers)
        n_params = mlp.n_params
        
        with torch.no_grad():
            ho[ts:te] = mlp(torch.tensor(X[ts:te], dtype=torch.float32)).numpy() * sig_b + mu_b

    return ho, n_params


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

def _train_transformer(X_win, y_target, n_neural, n_beh, device, 
                       d_model=256, n_heads=8, n_layers=2, d_ff=512,
                       epochs=200, lr=1e-3, wd=1e-4, patience=25):
    nv = max(10, int(len(X_win) * 0.15))
    Xt = torch.tensor(X_win[:-nv], dtype=torch.float32, device=device)
    yt = torch.tensor(y_target[:-nv], dtype=torch.float32, device=device)
    Xv = torch.tensor(X_win[-nv:], dtype=torch.float32, device=device)
    yv = torch.tensor(y_target[-nv:], dtype=torch.float32, device=device)

    cfg = TransformerBaselineConfig()
    cfg.d_model = d_model
    cfg.n_heads = n_heads
    cfg.n_layers = n_layers
    cfg.d_ff = d_ff
    cfg.dropout = 0.1
    cfg.context_length = K
    cfg.lr = lr
    cfg.weight_decay = wd
    cfg.sigma_min = 1e-4
    cfg.sigma_max = 10.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TemporalTransformerGaussian(n_neural=n_neural, n_beh=n_beh, cfg=cfg).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    bvl, bs, pat = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        _, _, pred_b_mu, _ = model.forward(Xt, return_all_steps=False)
        loss = nn.functional.mse_loss(pred_b_mu, yt)
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            _, _, pred_b_mu_v, _ = model.forward(Xv, return_all_steps=False)
            vl = nn.functional.mse_loss(pred_b_mu_v, yv).item()
                
        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
        if pat > patience: break

    if bs: model.load_state_dict(bs)
    model.eval().cpu()
    model.n_params = n_params
    return model


def cv_transformer(u, b, device, d_model=256, n_heads=8, n_layers=2, d_ff=512):
    """Transformer: n+b context → behavior, 1-step (teacher-forced for speed)."""
    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]
    warmup = K
    ho = np.full((T, Kw), np.nan)
    n_params = 0

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K)
        
        mu_u, sig_u = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu_u) / sig_u).astype(np.float32)
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        X_win = []
        for t in tr:
            ctx_u = u_n[t - K:t]
            ctx_b = b_n[t - K:t]
            ctx = np.concatenate([ctx_u, ctx_b], axis=1)
            X_win.append(ctx)
        X_win = np.stack(X_win)
        y_tr_beh = b_n[tr]

        model = _train_transformer(X_win, y_tr_beh, n_neural=N, n_beh=Kw, device=device,
                                   d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
        n_params = model.n_params

        # Test
        X_test = []
        for t in range(ts, te):
            ctx_u = u_n[t - K:t]
            ctx_b = b_n[t - K:t]
            ctx = np.concatenate([ctx_u, ctx_b], axis=1)
            X_test.append(ctx)
        X_test = np.stack(X_test)
        
        with torch.no_grad():
            ctx_t = torch.tensor(X_test, dtype=torch.float32)
            _, _, pred_b_mu, _ = model.forward(ctx_t, return_all_steps=False)
            ho[ts:te] = pred_b_mu.numpy() * sig_b + mu_b

    return ho, n_params


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def process_worm(h5_path, device, out_dir):
    """Run architecture sweep on one worm."""
    worm_data = load_worm_data(h5_path, n_beh_modes=6)
    u, b, worm_id = worm_data["u"], worm_data["b"], worm_data["worm_id"]
    
    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]
    
    print(f"\n{'='*70}")
    print(f"  WORM: {worm_id}  T={T}  N={N}")
    print(f"{'='*70}")
    
    results = {"worm_id": worm_id, "T": T, "N": N, "K": K}
    
    # Test all MLP configs
    print(f"\n  ── MLP Architectures ──")
    mlp_results = {}
    for name, cfg in MLP_CONFIGS.items():
        t0 = time.time()
        ho, n_params = cv_mlp(u, b, device, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
        r2, corr = beh_metrics(ho, b)
        elapsed = time.time() - t0
        mlp_results[name] = {
            "r2": float(r2), "corr": float(corr), 
            "n_params": n_params, "time_s": round(elapsed, 1),
            **cfg
        }
        print(f"    {name:12s}  h={cfg['hidden']:3d}  L={cfg['n_layers']}  params={n_params:>7,}  "
              f"R²={r2:.3f}  corr={corr:.3f}  ({elapsed:.1f}s)")
    results["mlp"] = mlp_results
    
    # Test all Transformer configs
    print(f"\n  ── Transformer Architectures ──")
    trf_results = {}
    for name, cfg in TRF_CONFIGS.items():
        t0 = time.time()
        ho, n_params = cv_transformer(u, b, device, 
                                       d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                                       n_layers=cfg["n_layers"], d_ff=cfg["d_ff"])
        r2, corr = beh_metrics(ho, b)
        elapsed = time.time() - t0
        trf_results[name] = {
            "r2": float(r2), "corr": float(corr),
            "n_params": n_params, "time_s": round(elapsed, 1),
            **cfg
        }
        print(f"    {name:12s}  d={cfg['d_model']:3d}  h={cfg['n_heads']}  L={cfg['n_layers']}  "
              f"params={n_params:>7,}  R²={r2:.3f}  corr={corr:.3f}  ({elapsed:.1f}s)")
    results["trf"] = trf_results
    
    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"arch_sweep_{worm_id}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    _plot_arch_comparison(results, out_dir)
    
    return results


def _plot_arch_comparison(results, out_dir):
    """Bar plot comparing architectures."""
    worm_id = results["worm_id"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MLP
    ax = axes[0]
    mlp_names = list(results["mlp"].keys())
    mlp_r2 = [results["mlp"][n]["r2"] for n in mlp_names]
    mlp_params = [results["mlp"][n]["n_params"] for n in mlp_names]
    
    x = np.arange(len(mlp_names))
    bars = ax.bar(x, mlp_r2, color='#1f77b4', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({p//1000}K)" for n, p in zip(mlp_names, mlp_params)], fontsize=9)
    ax.set_ylabel("R²")
    ax.set_ylim(0, max(mlp_r2) * 1.2 if max(mlp_r2) > 0 else 1)
    ax.set_title("MLP Architectures")
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, mlp_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Transformer
    ax = axes[1]
    trf_names = list(results["trf"].keys())
    trf_r2 = [results["trf"][n]["r2"] for n in trf_names]
    trf_params = [results["trf"][n]["n_params"] for n in trf_names]
    
    x = np.arange(len(trf_names))
    bars = ax.bar(x, trf_r2, color='#d62728', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({p//1000}K)" for n, p in zip(trf_names, trf_params)], fontsize=9)
    ax.set_ylabel("R²")
    ax.set_ylim(0, max(trf_r2) * 1.2 if max(trf_r2) > 0 else 1)
    ax.set_title("Transformer Architectures")
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, trf_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(f"Architecture Sweep — {worm_id}  (K={K}, N={results['N']}, 1-step teacher-forced)",
                 fontsize=12, fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(out_dir / f"arch_sweep_{worm_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / f'arch_sweep_{worm_id}.png'}")


def plot_summary(all_results, out_dir):
    """Summary plot across worms."""
    if len(all_results) < 2:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MLP summary
    ax = axes[0]
    mlp_names = list(MLP_CONFIGS.keys())
    for i, name in enumerate(mlp_names):
        r2_vals = [r["mlp"][name]["r2"] for r in all_results.values()]
        ax.bar(i, np.mean(r2_vals), yerr=np.std(r2_vals), color='#1f77b4', alpha=0.8, capsize=5)
    ax.set_xticks(range(len(mlp_names)))
    ax.set_xticklabels(mlp_names, rotation=45, ha='right')
    ax.set_ylabel("Mean R² ± std")
    ax.set_title("MLP Summary")
    ax.grid(axis='y', alpha=0.3)
    
    # Transformer summary
    ax = axes[1]
    trf_names = list(TRF_CONFIGS.keys())
    for i, name in enumerate(trf_names):
        r2_vals = [r["trf"][name]["r2"] for r in all_results.values()]
        ax.bar(i, np.mean(r2_vals), yerr=np.std(r2_vals), color='#d62728', alpha=0.8, capsize=5)
    ax.set_xticks(range(len(trf_names)))
    ax.set_xticklabels(trf_names, rotation=45, ha='right')
    ax.set_ylabel("Mean R² ± std")
    ax.set_title("Transformer Summary")
    ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle(f"Architecture Sweep Summary — {len(all_results)} worms (K={K})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "arch_sweep_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'arch_sweep_summary.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_dir", default="data/used/behaviour+neuronal activity atanas (2023)/2")
    ap.add_argument("--out_dir", default="output_plots/behaviour_decoder/arch_sweep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_worms", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get worm files
    h5_dir = Path(args.h5_dir)
    h5_files = sorted(h5_dir.glob("*.h5"))
    
    # Random selection
    np.random.seed(args.seed)
    selected = np.random.choice(len(h5_files), size=min(args.n_worms, len(h5_files)), replace=False)
    selected_files = [h5_files[i] for i in selected]
    
    print(f"Architecture sweep: K={K}, testing 5 MLP + 5 Transformer configs")
    print(f"Selected {len(selected_files)} worms: {[f.stem for f in selected_files]}")
    
    t_total = time.time()
    all_results = {}
    
    for h5_file in selected_files:
        try:
            results = process_worm(str(h5_file), device, out_dir)
            all_results[results["worm_id"]] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    plot_summary(all_results, out_dir)
    
    # Print best configs
    print(f"\n{'='*70}")
    print(f"  BEST ARCHITECTURES")
    print(f"{'='*70}")
    
    for worm_id, res in all_results.items():
        best_mlp = max(res["mlp"].items(), key=lambda x: x[1]["r2"])
        best_trf = max(res["trf"].items(), key=lambda x: x[1]["r2"])
        print(f"\n  {worm_id}:")
        print(f"    MLP:  {best_mlp[0]} (R²={best_mlp[1]['r2']:.3f}, {best_mlp[1]['n_params']:,} params)")
        print(f"    TRF:  {best_trf[0]} (R²={best_trf[1]['r2']:.3f}, {best_trf[1]['n_params']:,} params)")
    
    total_time = time.time() - t_total
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")


if __name__ == "__main__":
    main()
