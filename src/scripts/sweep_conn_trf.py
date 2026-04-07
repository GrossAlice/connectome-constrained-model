#!/usr/bin/env python
"""
Conn-Transformer architecture sweep.

Tests different architectures for the per-neuron connectome-constrained
transformer to find the best LOO configuration.

Sweep axes:
  1. Bugfix baseline  (d_model=32, 1 layer, dropout=0.1)   — original but with LOO fix
  2. d_model          {8, 16, 32, 64}
  3. n_layers         {1, 2}
  4. dropout          {0.1, 0.3, 0.5}
  5. weight_decay     {1e-4, 1e-3, 1e-2}
  6. ConnLinear        per-neuron linear model  (no attention, just lag-K Ridge)
  7. top_k_partners   {10, 20, all}  — cap neighbours

Usage
─────
    TORCHDYNAMO_DISABLE=1 python -u -m scripts.sweep_conn_trf \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
        --save_dir output_plots/stage2/conn_trf_sweep \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _r2(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m].astype(np.float64), y_pred[m].astype(np.float64)
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-15))


def load_connectome(worm_labels: List[str]) -> Dict[int, List[int]]:
    """Load combined connectome → partners dict with connection weights."""
    d = ROOT / "data/used/masks+motor neurons"
    atlas_names = np.load(d / "neuron_names.npy")
    n2a = {str(n): i for i, n in enumerate(atlas_names)}
    T_all = sum(
        np.abs(np.load(d / f"{t}.npy")) for t in ("T_e", "T_sv", "T_dcv")
    )
    N = len(worm_labels)
    wa = [n2a.get(lab, -1) for lab in worm_labels]
    adj = np.zeros((N, N), np.float64)
    for i in range(N):
        for j in range(N):
            if wa[i] >= 0 and wa[j] >= 0:
                adj[j, i] = T_all[wa[j], wa[i]]

    # Return dict: neuron → [(partner_idx, weight), ...]
    partners_w: Dict[int, list] = {}
    for i in range(N):
        pw = [(j, adj[j, i]) for j in range(N) if j != i and adj[j, i] > 0]
        pw.sort(key=lambda x: -x[1])  # strongest first
        partners_w[i] = pw
    return partners_w


def get_partner_indices(partners_w: Dict[int, list],
                        top_k: Optional[int] = None) -> Dict[int, List[int]]:
    """Get partner indices, optionally capped to top-K by weight."""
    partners: Dict[int, List[int]] = {}
    for i, pw in partners_w.items():
        if top_k is not None and len(pw) > top_k:
            partners[i] = sorted([j for j, _ in pw[:top_k]])
        else:
            partners[i] = sorted([j for j, _ in pw])
    return partners


# ═══════════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════════

class ConnTransformer(nn.Module):
    """Per-neuron causal transformer: (K, d_in) → scalar."""

    def __init__(self, d_in: int, d_model: int = 32, n_heads: int = 2,
                 n_layers: int = 1, K: int = 5, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, K, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.full((K, K), float("-inf")), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K_in = x.size(1)
        h = self.proj(x) + self.pos_emb[:, :K_in, :]
        mask = self._causal_mask[:K_in, :K_in]
        h = self.encoder(h, mask=mask)
        return self.head(h[:, -1, :]).squeeze(-1)


class ConnLinearK(nn.Module):
    """Per-neuron linear model using K lags of connectome neighbours.
    Flattens (K, d_in) → K*d_in, then linear → scalar.
    Much simpler than transformer — effectively a lag-K Ridge regression."""

    def __init__(self, d_in: int, K: int = 5):
        super().__init__()
        self.linear = nn.Linear(d_in * K, 1)
        self.K = K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, d_in)
        B = x.size(0)
        return self.linear(x.reshape(B, -1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
#  Sweep config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SweepConfig:
    name: str
    model_type: str = "transformer"  # "transformer" or "linear"
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 1
    K: int = 5
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    top_k_partners: Optional[int] = None  # None = all


def make_sweep_configs() -> List[SweepConfig]:
    """Generate the list of sweep configurations."""
    configs = []

    # --- Baseline (bugfix only) ---
    configs.append(SweepConfig(name="baseline_d32"))

    # --- d_model sweep ---
    for dm in [8, 16, 64]:
        nh = max(1, dm // 8)  # sensible head count
        configs.append(SweepConfig(
            name=f"d{dm}", d_model=dm, n_heads=nh))

    # --- n_layers sweep ---
    configs.append(SweepConfig(name="layers2", n_layers=2))

    # --- dropout sweep ---
    for do in [0.3, 0.5]:
        configs.append(SweepConfig(name=f"drop{do}", dropout=do))

    # --- weight_decay sweep ---
    for wd in [1e-3, 1e-2]:
        configs.append(SweepConfig(
            name=f"wd{wd:.0e}", weight_decay=wd))

    # --- top-K partners ---
    for tk in [10, 20]:
        configs.append(SweepConfig(
            name=f"topk{tk}", top_k_partners=tk))

    # --- ConnLinear (no attention) ---
    configs.append(SweepConfig(
        name="linear_K5", model_type="linear"))
    configs.append(SweepConfig(
        name="linear_K5_wd1e-2", model_type="linear",
        weight_decay=1e-2))

    return configs


# ═══════════════════════════════════════════════════════════════════════
#  Training + evaluation for one config
# ═══════════════════════════════════════════════════════════════════════

def run_one_config(
    cfg: SweepConfig,
    u: np.ndarray,
    partners: Dict[int, List[int]],
    loo_neurons: List[int],
    device: str = "cuda",
    window_size: int = 50,
    verbose: bool = True,
):
    """Train + evaluate one Conn-Trf config. Returns (onestep_r2, loo_r2_w)."""
    T, N = u.shape
    K = cfg.K
    dev = torch.device(device)
    mid = T // 2 + 1
    folds = [(mid, T, 0, mid), (0, mid, mid, T)]
    x = u.astype(np.float32)

    os_pred = np.full((T, N), np.nan, dtype=np.float32)
    loo_pred = np.full((T, N), np.nan, dtype=np.float32)
    loo_set = set(loo_neurons)

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        if verbose:
            print(f"    [fold {fi+1}/2]  train=[{tr_s},{tr_e})  "
                  f"test=[{te_s},{te_e})")
        u_tr = x[tr_s:tr_e]
        T_te = te_e - te_s

        for i in range(N):
            feats = sorted(partners[i] + [i])
            if not feats:
                feats = [i]
            d_in = len(feats)

            # Build training data: sliding window
            indices = np.arange(K, len(u_tr))
            X_np = np.stack([u_tr[t - K : t, feats] for t in indices])
            Y_np = u_tr[indices, i]
            X_t = torch.tensor(X_np, dtype=torch.float32, device=dev)
            Y_t = torch.tensor(Y_np, dtype=torch.float32, device=dev)

            nv = max(int(len(X_t) * 0.1), 1)
            nf = len(X_t) - nv

            # Create model
            if cfg.model_type == "linear":
                model = ConnLinearK(d_in, K=K).to(dev)
            else:
                model = ConnTransformer(
                    d_in, d_model=cfg.d_model, n_heads=cfg.n_heads,
                    n_layers=cfg.n_layers, K=K, dropout=cfg.dropout,
                ).to(dev)

            opt = torch.optim.Adam(
                model.parameters(), lr=cfg.lr,
                weight_decay=cfg.weight_decay)
            bvl, bst, w = float("inf"), None, 0

            for ep in range(cfg.max_epochs):
                model.train()
                perm = torch.randperm(nf, device=dev)
                for bs in range(0, nf, 128):
                    idx = perm[bs : bs + 128]
                    loss = nn.functional.mse_loss(model(X_t[idx]), Y_t[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vl = nn.functional.mse_loss(
                        model(X_t[nf:]), Y_t[nf:]).item()
                if vl < bvl - 1e-6:
                    bvl = vl
                    bst = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
                    w = 0
                else:
                    w += 1
                    if w >= cfg.patience:
                        break

            if bst:
                model.load_state_dict(
                    {k: v.to(dev) for k, v in bst.items()})
            model.eval()

            # --- One-step predictions on test fold ---
            with torch.no_grad():
                first = max(K, te_s)
                n_te = te_e - first
                if n_te > 0:
                    ctx_all = np.stack(
                        [x[t - K : t, feats]
                         for t in range(first, te_e)])
                    ctx_t = torch.tensor(
                        ctx_all, dtype=torch.float32, device=dev)
                    os_pred[first:te_e, i] = model(ctx_t).cpu().numpy()

            # --- Windowed LOO (only for LOO neurons) ---
            if i in loo_set:
                sp = feats.index(i) if i in feats else None
                u_te = x[te_s:te_e]
                pred_arr = np.full(T_te, np.nan, dtype=np.float32)
                with torch.no_grad():
                    for s in range(0, T_te, window_size):
                        e = min(s + window_size, T_te)
                        pred_arr[s] = u_te[s, i]
                        for t_loc in range(s, e - 1):
                            t_abs = te_s + t_loc
                            # Context [t_abs-K+1 .. t_abs] → predict t_abs+1
                            ctx_start = t_abs - K + 1
                            if ctx_start < 0:
                                continue
                            ctx = x[ctx_start : t_abs + 1, feats].copy()
                            if sp is not None:
                                for k in range(K):
                                    pa_idx = t_loc - (K - 1) + k
                                    if pa_idx >= s and pa_idx <= t_loc:
                                        ctx[k, sp] = pred_arr[pa_idx]
                            ctx_t = torch.tensor(
                                ctx, dtype=torch.float32, device=dev
                            ).unsqueeze(0)
                            pred_arr[t_loc + 1] = (
                                model(ctx_t).cpu().item())
                loo_pred[te_s:te_e, i] = pred_arr

            del model, opt
            if (i + 1) % 30 == 0 or i == N - 1:
                if verbose:
                    print(f"      {i + 1}/{N} neurons")

        torch.cuda.empty_cache()

    # R²
    onestep_r2 = np.full(N, np.nan)
    for i in range(N):
        m = np.isfinite(os_pred[:, i])
        if m.sum() > 3:
            onestep_r2[i] = _r2(u[m, i], os_pred[m, i])

    loo_r2_w = np.full(len(loo_neurons), np.nan)
    for ki, ni in enumerate(loo_neurons):
        m = np.isfinite(loo_pred[:, ni])
        if m.sum() > 3:
            loo_r2_w[ki] = _r2(u[m, ni], loo_pred[m, ni])

    return onestep_r2, loo_r2_w


# ═══════════════════════════════════════════════════════════════════════
#  Plot
# ═══════════════════════════════════════════════════════════════════════

def plot_sweep(results: Dict[str, dict], save_dir: Path):
    """Bar chart: LOO R² mean + std for each config, sorted by LOO mean."""
    names = sorted(results.keys(),
                   key=lambda n: results[n]["loo_mean"], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Panel A: One-step R²
    ax = axes[0]
    xs = np.arange(len(names))
    means_os = [results[n]["onestep_mean"] for n in names]
    ax.bar(xs, means_os, color="#4682b4", alpha=0.7, edgecolor="black",
           linewidth=0.5)
    for xi, (n, m) in enumerate(zip(names, means_os)):
        ax.text(xi, m + 0.01, f"{m:.3f}", ha="center", fontsize=7,
                fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("One-step R² (mean over N neurons)")
    ax.set_title("Panel A: One-step R²")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    # Panel B: LOO R²
    ax = axes[1]
    means_loo = [results[n]["loo_mean"] for n in names]
    stds_loo = [results[n]["loo_std"] for n in names]
    colors = ["#228b22" if m > 0 else "#d62728" for m in means_loo]
    ax.bar(xs, means_loo, yerr=stds_loo, color=colors, alpha=0.7,
           edgecolor="black", linewidth=0.5, capsize=3)
    for xi, (n, m) in enumerate(zip(names, means_loo)):
        offset = 0.02 if m >= 0 else -0.06
        ax.text(xi, m + offset, f"{m:.3f}", ha="center", fontsize=7,
                fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("LOO R² (mean over 30 neurons)")
    ax.set_title("Panel B: LOO R² (windowed)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    # Add reference lines for Conn-Ridge LOO
    for ax in axes:
        ax.axhline(0.196, color="blue", linestyle=":", linewidth=1,
                   alpha=0.5, label="Conn-Ridge LOO=0.196")
    axes[1].legend(fontsize=7, loc="lower right")

    fig.suptitle("Conn-Transformer Architecture Sweep", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "conn_trf_sweep.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_dir / 'conn_trf_sweep.png'}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--save_dir", default="output_plots/stage2/conn_trf_sweep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loo_subset", type=int, default=30)
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    h5_path = ROOT / args.h5
    from stage2.config import make_config
    from stage2.io_h5 import load_data_pt
    cfg0 = make_config(str(h5_path), device=args.device)
    data = load_data_pt(cfg0)
    u = data["u_stage1"].cpu().numpy()

    # Labels from h5
    with h5py.File(str(h5_path), "r") as f:
        labels = [
            l.decode() if isinstance(l, bytes) else str(l)
            for l in f["gcamp/neuron_labels"][:]
        ]

    T, N = u.shape
    worm_name = Path(args.h5).stem
    print(f"  {worm_name}  T={T}  N={N}")

    # LOO neurons (top variance)
    var = np.var(u, axis=0)
    loo_neurons = sorted(np.argsort(var)[-args.loo_subset:].tolist())
    print(f"  LOO neurons ({len(loo_neurons)}): {loo_neurons}")

    # Load connectome with weights
    partners_w = load_connectome(labels)
    np_arr = [len(partners_w[i]) for i in range(N)]
    print(f"  Connectome: partners/neuron — min={min(np_arr)}, "
          f"median={int(np.median(np_arr))}, max={max(np_arr)}")

    # Sweep configs
    configs = make_sweep_configs()
    print(f"\n  Running {len(configs)} configurations:\n")
    for c in configs:
        print(f"    • {c.name}: type={c.model_type} d={c.d_model} "
              f"L={c.n_layers} drop={c.dropout} wd={c.weight_decay} "
              f"topk={c.top_k_partners}")
    print()

    results = {}
    patience = 15  # used in training loop

    for ci, cfg in enumerate(configs):
        print(f"{'═'*60}")
        print(f"  [{ci+1}/{len(configs)}] {cfg.name}")
        print(f"{'═'*60}")

        t0 = time.time()

        # Get partners for this config (possibly top-K filtered)
        partners = get_partner_indices(partners_w, cfg.top_k_partners)

        onestep_r2, loo_r2_w = run_one_config(
            cfg, u, partners, loo_neurons,
            device=args.device, verbose=True,
        )

        dt = time.time() - t0
        os_mean = float(np.nanmean(onestep_r2))
        loo_mean = float(np.nanmean(loo_r2_w))
        loo_std = float(np.nanstd(loo_r2_w))
        loo_med = float(np.nanmedian(loo_r2_w))

        print(f"  → {cfg.name}: 1step={os_mean:.4f}  "
              f"LOO={loo_mean:.4f} ± {loo_std:.4f}  "
              f"(median={loo_med:.4f})  [{dt:.0f}s]")
        print()

        results[cfg.name] = {
            "config": asdict(cfg),
            "onestep_r2": onestep_r2.tolist(),
            "onestep_mean": os_mean,
            "loo_r2_w": loo_r2_w.tolist(),
            "loo_mean": loo_mean,
            "loo_std": loo_std,
            "loo_median": loo_med,
            "time_s": dt,
        }

        # Save incrementally
        with open(save_dir / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    print(f"\n{'═'*60}")
    print("  SWEEP SUMMARY (sorted by LOO R²)")
    print(f"{'═'*60}")
    sorted_names = sorted(results.keys(),
                          key=lambda n: results[n]["loo_mean"],
                          reverse=True)
    print(f"  {'Config':<25} {'1step':>8} {'LOO mean':>10} {'LOO med':>10} {'Time':>6}")
    print(f"  {'-'*62}")
    for n in sorted_names:
        r = results[n]
        print(f"  {n:<25} {r['onestep_mean']:>8.4f} {r['loo_mean']:>10.4f} "
              f"{r['loo_median']:>10.4f} {r['time_s']:>5.0f}s")

    # Plot
    plot_sweep(results, save_dir)
    print(f"\n  All results saved to {save_dir}")


if __name__ == "__main__":
    main()
