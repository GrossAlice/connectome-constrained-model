#!/usr/bin/env python3
"""
Test whether neural coupling is truly linear, using POOLED multi-worm data.

Motivation: with ~1600 timepoints per worm and N=27-83 neurons, a 2-layer MLP
overfits and Ridge wins.  This doesn't mean the dynamics ARE linear — it just
means we lack data.  To settle this, we pool across all 40 worms (and neurons),
giving us ~64,000 × N_neurons samples — enough for an MLP to express
nonlinearity if it exists.

Conditions tested (all neuron/worm-agnostic after normalisation):
  1. self:   x = [u_i(t-1), ..., u_i(t-K)] → predict u_i(t)
             Features are K-dimensional regardless of N → can pool.
  2. conc_rank:  x = rank-order statistics of u_j(t), j≠i
             Map the (N-1) concurrent values to a fixed set of
             percentile features [p5, p25, p50, p75, p95, mean, std, max, min, kurtosis]
             → 10 features regardless of N → can pool.
  3. combined:  [self K lags] + [10 conc_rank stats] → K+10 features

For each condition we train Ridge, MLP (2-layer, 256 hidden), and a bigger MLP
(3-layer, 512 hidden) to give nonlinearity the best shot.

Usage:
  python -m scripts.test_nonlinearity_pooled \
      --data_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
      --neurons motor --K 5 --device cuda
"""
from __future__ import annotations

import argparse, glob, json, sys, time, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kurtosis as _kurtosis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.dataset import load_worm_data
from sklearn.linear_model import RidgeCV

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

K_VALUES   = [1, 3, 5, 10]
_RIDGE_ALPHAS = np.logspace(-4, 6, 30)
N_FOLDS = 5          # temporal folds WITHIN each worm (avoid data leakage)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS  (per-neuron, worm-agnostic)
# ══════════════════════════════════════════════════════════════════════════════

def _build_self_features(u_n, K, ni):
    """(T, K) lagged self-history for neuron ni."""
    T = u_n.shape[0]
    out = np.zeros((T, K), dtype=np.float32)
    col = u_n[:, ni]
    for lag in range(1, K + 1):
        out[lag:, lag - 1] = col[:-lag]
    return out


def _build_conc_rank_features(u_n, ni):
    """(T, 10) rank-order summary statistics of u_j(t), j≠i.

    Fixed-dim representation of concurrent coupling regardless of N.
    """
    others = np.delete(u_n, ni, axis=1)   # (T, N-1)
    T = others.shape[0]
    feats = np.zeros((T, 10), dtype=np.float32)
    feats[:, 0] = np.percentile(others, 5,  axis=1)
    feats[:, 1] = np.percentile(others, 25, axis=1)
    feats[:, 2] = np.percentile(others, 50, axis=1)   # median
    feats[:, 3] = np.percentile(others, 75, axis=1)
    feats[:, 4] = np.percentile(others, 95, axis=1)
    feats[:, 5] = others.mean(axis=1)
    feats[:, 6] = others.std(axis=1) + 1e-8
    feats[:, 7] = others.max(axis=1)
    feats[:, 8] = others.min(axis=1)
    feats[:, 9] = _kurtosis(others, axis=1, fisher=True, nan_policy='omit')
    return feats


def _build_features(u_n, K, ni, condition):
    """Return (T, d_feat) feature matrix and valid start index."""
    if condition == "self":
        X = _build_self_features(u_n, K, ni)
        warmup = K
    elif condition == "conc_rank":
        X = _build_conc_rank_features(u_n, ni)
        warmup = 1
    elif condition == "combined":
        X_s = _build_self_features(u_n, K, ni)
        X_c = _build_conc_rank_features(u_n, ni)
        X = np.concatenate([X_s, X_c], axis=1)
        warmup = K
    else:
        raise ValueError(condition)
    return X, warmup


# ══════════════════════════════════════════════════════════════════════════════
# DATA POOLING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_worms(data_dir, neurons_tag):
    """Load and return list of (u_selected, worm_id, N)."""
    h5_files = sorted(glob.glob(str(Path(data_dir) / "*.h5")))
    worms = []
    for h5 in h5_files:
        data = load_worm_data(h5, n_beh_modes=6)
        u_all = data["u"]
        motor_idx = data.get("motor_idx", []) or []
        motor_set = set(motor_idx)
        N_total = u_all.shape[1]

        if neurons_tag == "motor":
            sel = sorted(motor_idx)
        elif neurons_tag == "nonmotor":
            sel = sorted(i for i in range(N_total) if i not in motor_set)
        else:
            sel = list(range(N_total))

        if not sel:
            continue
        u = u_all[:, sel].astype(np.float32)
        worms.append({"u": u, "worm_id": data["worm_id"], "N": u.shape[1],
                       "T": u.shape[0]})
    return worms


def pool_data(worms, K, condition):
    """Pool (X, y) pairs across all worms and neurons.

    Each worm is z-scored independently (train portion).  Temporal CV folds
    are respected: we return per-worm-neuron blocks so the caller can
    build leave-one-worm-out splits.

    Returns
    -------
    blocks : list of dict with keys 'X', 'y', 'worm_id', 'ni', 'T', 'warmup'
    """
    blocks = []
    for w in worms:
        u = w["u"]
        T, N = u.shape
        # z-score the whole worm
        mu, sig = u.mean(0), u.std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)

        for ni in range(N):
            X, warmup = _build_features(u_n, K, ni, condition)
            y = u_n[warmup:, ni].copy()       # target = u_i(t) after warmup
            X = X[warmup:]                     # align
            blocks.append({
                "X": X, "y": y,
                "worm_id": w["worm_id"], "ni": ni,
                "T": len(y), "warmup": warmup,
            })
    return blocks


def make_pooled_train_test(blocks, test_frac=0.2):
    """Temporal split within each block: last test_frac is test."""
    X_train_parts, y_train_parts = [], []
    X_test_parts, y_test_parts = [], []
    test_info = []   # (worm_id, ni, n_test)

    for blk in blocks:
        T = blk["T"]
        n_test = max(10, int(T * test_frac))
        n_train = T - n_test

        X_train_parts.append(blk["X"][:n_train])
        y_train_parts.append(blk["y"][:n_train])
        X_test_parts.append(blk["X"][n_train:])
        y_test_parts.append(blk["y"][n_train:])
        test_info.append({
            "worm_id": blk["worm_id"], "ni": blk["ni"], "n_test": T - n_train,
        })

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    X_test  = np.concatenate(X_test_parts, axis=0)
    y_test  = np.concatenate(y_test_parts, axis=0)

    return X_train, y_train, X_test, y_test, test_info


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def _make_mlp(d_in, hidden, n_layers):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(),
                   nn.Dropout(0.05)]
        d = hidden
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def train_mlp(X_train, y_train, X_test, y_test, hidden, n_layers, device,
              epochs=300, lr=1e-3, wd=1e-4, patience=30, batch_size=4096,
              label="MLP"):
    """Train MLP on pooled data with mini-batch SGD."""
    d_in = X_train.shape[1]
    N_train = X_train.shape[0]

    # Validation: carve 15% from training
    n_val = max(256, int(N_train * 0.15))
    perm = np.random.permutation(N_train)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xt = torch.tensor(X_train[tr_idx], dtype=torch.float32, device=device)
    yt = torch.tensor(y_train[tr_idx], dtype=torch.float32, device=device).unsqueeze(1)
    Xv = torch.tensor(X_train[val_idx], dtype=torch.float32, device=device)
    yv = torch.tensor(y_train[val_idx], dtype=torch.float32, device=device).unsqueeze(1)

    model = _make_mlp(d_in, hidden, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10)

    bvl, bs, pat = float("inf"), None, 0
    n_batches = max(1, len(Xt) // batch_size)

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt), device=device)
        epoch_loss = 0.0
        for bi in range(n_batches):
            s, e = bi * batch_size, min((bi + 1) * batch_size, len(Xt))
            batch_idx = idx[s:e]
            pred = model(Xt[batch_idx])
            loss = F.mse_loss(pred, yt[batch_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * (e - s)
        epoch_loss /= len(Xt)

        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(Xv), yv).item()
        scheduler.step(vl)

        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
        if pat > patience:
            break

    if bs:
        model.load_state_dict(bs)
    model.eval()

    # Predict on test
    with torch.no_grad():
        Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
        pred_test = model(Xte).squeeze(1).cpu().numpy()

    model.cpu()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    {label:20s}  val_loss={bvl:.5f}  epochs={epoch+1:3d}  "
          f"params={n_params:,}")
    return pred_test, bvl


def train_ridge(X_train, y_train, X_test):
    """Train RidgeCV on pooled data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = RidgeCV(alphas=_RIDGE_ALPHAS).fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"    {'Ridge':20s}  alpha={model.alpha_:.2e}  "
          f"coefs_norm={np.linalg.norm(model.coef_):.3f}")
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _r2(gt, pred):
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - gt.mean()) ** 2) + 1e-12
    return 1 - ss_res / ss_tot

def _corr(gt, pred):
    if len(gt) < 3:
        return np.nan
    return float(np.corrcoef(gt, pred)[0, 1])


def compute_metrics(y_test, pred, test_info):
    """Compute overall and per-worm R² / Corr."""
    overall_r2   = _r2(y_test, pred)
    overall_corr = _corr(y_test, pred)

    # Per worm-neuron
    offset = 0
    per_worm = {}
    for info in test_info:
        n = info["n_test"]
        gt_chunk  = y_test[offset:offset+n]
        pr_chunk  = pred[offset:offset+n]
        r2  = _r2(gt_chunk, pr_chunk)
        co  = _corr(gt_chunk, pr_chunk)
        wid = info["worm_id"]
        if wid not in per_worm:
            per_worm[wid] = {"r2": [], "corr": []}
        per_worm[wid]["r2"].append(r2)
        per_worm[wid]["corr"].append(co)
        offset += n

    # Average per worm
    worm_r2   = {w: np.mean(v["r2"]) for w, v in per_worm.items()}
    worm_corr = {w: np.mean(v["corr"]) for w, v in per_worm.items()}

    return {
        "overall_r2": float(overall_r2),
        "overall_corr": float(overall_corr),
        "mean_worm_r2": float(np.mean(list(worm_r2.values()))),
        "mean_worm_corr": float(np.mean(list(worm_corr.values()))),
        "per_worm_r2": {k: float(v) for k, v in worm_r2.items()},
        "per_worm_corr": {k: float(v) for k, v in worm_corr.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(all_results, out_dir, neurons_tag):
    """Bar chart: R² per model per condition per K."""
    conditions = sorted(set(r["condition"] for r in all_results))
    models     = sorted(set(r["model"] for r in all_results))
    K_vals     = sorted(set(r["K"] for r in all_results))

    model_colors = {
        "Ridge": "#2ca02c", "MLP_256x2": "#1f77b4",
        "MLP_512x3": "#d62728",
    }

    # ── Lag sweep ──
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5),
                             sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ci, cond in enumerate(conditions):
        ax = axes[ci]
        for m in models:
            xs, ys = [], []
            for K in K_vals:
                match = [r for r in all_results
                         if r["condition"] == cond and r["model"] == m and r["K"] == K]
                if match:
                    xs.append(K)
                    ys.append(match[0]["overall_r2"])
            if xs:
                c = model_colors.get(m, "#999")
                ax.plot(xs, ys, "-o", label=m, color=c, lw=2, ms=7)
        ax.set_xlabel("K (lags)", fontsize=11)
        ax.set_title(cond, fontsize=12, fontweight="bold")
        ax.set_xticks(K_vals)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.4)
        ax.set_ylim(-0.1, 1)
        if ci == 0:
            ax.set_ylabel("Pooled R²", fontsize=11)
            ax.legend(fontsize=9)

    fig.suptitle(f"Nonlinearity Test — Pooled {neurons_tag} neurons across all worms",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"lag_sweep_{neurons_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / f'lag_sweep_{neurons_tag}.png'}")

    # ── Per-worm scatter: Ridge R² vs MLP R² ──
    for cond in conditions:
        for K in [5, 10]:
            ridge_data = [r for r in all_results
                          if r["condition"] == cond and r["model"] == "Ridge" and r["K"] == K]
            mlp_data   = [r for r in all_results
                          if r["condition"] == cond and r["model"] == "MLP_512x3" and r["K"] == K]
            if not ridge_data or not mlp_data:
                continue
            rd, md = ridge_data[0], mlp_data[0]
            worms = sorted(set(rd["per_worm_r2"].keys()) & set(md["per_worm_r2"].keys()))
            r_vals = np.array([rd["per_worm_r2"][w] for w in worms])
            m_vals = np.array([md["per_worm_r2"][w] for w in worms])

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(r_vals, m_vals, s=40, alpha=0.7, edgecolors="white", lw=0.5)
            lo, hi = min(r_vals.min(), m_vals.min()) - 0.05, max(r_vals.max(), m_vals.max()) + 0.05
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
            ax.set_xlabel("Ridge R² (per worm)", fontsize=11)
            ax.set_ylabel("MLP_512x3 R² (per worm)", fontsize=11)
            ax.set_title(f"{cond}  K={K}  ({neurons_tag})\n"
                         f"Ridge>{len(worms)-np.sum(m_vals > r_vals + 0.01):.0f}  "
                         f"MLP>{np.sum(m_vals > r_vals + 0.01):.0f}",
                         fontsize=11, fontweight="bold")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fname = f"scatter_{cond}_K{K}_{neurons_tag}.png"
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out_dir / fname}")

    # ── Delta bar chart: MLP - Ridge per K ──
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5),
                             sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ci, cond in enumerate(conditions):
        ax = axes[ci]
        for m in ["MLP_256x2", "MLP_512x3"]:
            deltas, ks_plot = [], []
            for K in K_vals:
                ridge_match = [r for r in all_results
                               if r["condition"] == cond and r["model"] == "Ridge" and r["K"] == K]
                mlp_match = [r for r in all_results
                             if r["condition"] == cond and r["model"] == m and r["K"] == K]
                if ridge_match and mlp_match:
                    delta = mlp_match[0]["overall_r2"] - ridge_match[0]["overall_r2"]
                    deltas.append(delta)
                    ks_plot.append(K)
            if ks_plot:
                c = model_colors.get(m, "#999")
                ax.bar([k + (0.3 if m == "MLP_512x3" else -0.3) for k in ks_plot],
                       deltas, width=0.5, color=c, alpha=0.7, label=f"Δ({m}−Ridge)")
        ax.axhline(0, color="k", ls="-", lw=1)
        ax.set_xlabel("K (lags)", fontsize=11)
        ax.set_title(cond, fontsize=12, fontweight="bold")
        ax.set_xticks(K_vals)
        ax.grid(axis="y", alpha=0.3)
        if ci == 0:
            ax.set_ylabel("Δ R² (MLP − Ridge)", fontsize=11)
            ax.legend(fontsize=9)

    fig.suptitle(f"MLP − Ridge gap (pooled {neurons_tag})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"delta_bar_{neurons_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / f'delta_bar_{neurons_tag}.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Pooled multi-worm nonlinearity test")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir",
                    default="output_plots/nonlinearity_pooled")
    ap.add_argument("--neurons", default="motor",
                    choices=["motor", "nonmotor", "all"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    neurons_tag = args.neurons

    print(f"Loading all worms ({neurons_tag})...")
    t0 = time.time()
    worms = load_all_worms(args.data_dir, neurons_tag)
    total_T = sum(w["T"] for w in worms)
    total_neurons = sum(w["N"] for w in worms)
    print(f"  {len(worms)} worms, {total_neurons} total neurons, "
          f"{total_T:,} total timepoints  ({time.time()-t0:.1f}s)")

    all_results = []

    for condition in ["self", "conc_rank", "combined"]:
        for K in K_VALUES:
            print(f"\n{'═'*60}")
            print(f"  condition={condition}  K={K}  neurons={neurons_tag}")
            print(f"{'═'*60}")

            t0 = time.time()
            blocks = pool_data(worms, K, condition)
            X_train, y_train, X_test, y_test, test_info = make_pooled_train_test(
                blocks, test_frac=0.2)
            print(f"  Pooled: train={X_train.shape[0]:,}  test={X_test.shape[0]:,}  "
                  f"d_feat={X_train.shape[1]}  ({time.time()-t0:.1f}s)")

            # ── Ridge ──
            t0 = time.time()
            pred_ridge = train_ridge(X_train, y_train, X_test)
            metrics_ridge = compute_metrics(y_test, pred_ridge, test_info)
            dt_ridge = time.time() - t0
            print(f"    Ridge   R²={metrics_ridge['overall_r2']:.4f}  "
                  f"corr={metrics_ridge['overall_corr']:.4f}  ({dt_ridge:.0f}s)")
            res_r = {"condition": condition, "K": K, "model": "Ridge", **metrics_ridge}
            all_results.append(res_r)

            # ── MLP 256×2 ──
            t0 = time.time()
            pred_mlp_s, _ = train_mlp(
                X_train, y_train, X_test, y_test,
                hidden=256, n_layers=2, device=device, label="MLP_256x2")
            metrics_mlp_s = compute_metrics(y_test, pred_mlp_s, test_info)
            dt = time.time() - t0
            print(f"    MLP256  R²={metrics_mlp_s['overall_r2']:.4f}  "
                  f"corr={metrics_mlp_s['overall_corr']:.4f}  ({dt:.0f}s)")
            all_results.append({"condition": condition, "K": K,
                                "model": "MLP_256x2", **metrics_mlp_s})

            # ── MLP 512×3 ──
            t0 = time.time()
            pred_mlp_l, _ = train_mlp(
                X_train, y_train, X_test, y_test,
                hidden=512, n_layers=3, device=device, label="MLP_512x3")
            metrics_mlp_l = compute_metrics(y_test, pred_mlp_l, test_info)
            dt = time.time() - t0
            print(f"    MLP512  R²={metrics_mlp_l['overall_r2']:.4f}  "
                  f"corr={metrics_mlp_l['overall_corr']:.4f}  ({dt:.0f}s)")
            all_results.append({"condition": condition, "K": K,
                                "model": "MLP_512x3", **metrics_mlp_l})

            # Delta summary
            d_s = metrics_mlp_s["overall_r2"] - metrics_ridge["overall_r2"]
            d_l = metrics_mlp_l["overall_r2"] - metrics_ridge["overall_r2"]
            print(f"    Δ(MLP256−Ridge)={d_s:+.4f}  Δ(MLP512−Ridge)={d_l:+.4f}")

            # Checkpoint
            with open(out_dir / f"results_{neurons_tag}.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # ── Final plots ──
    print("\nGenerating plots...")
    plot_summary(all_results, out_dir, neurons_tag)

    # ── Final verdict ──
    print("\n" + "═" * 60)
    print("  VERDICT: Is there evidence for nonlinearity?")
    print("═" * 60)
    for cond in ["self", "conc_rank", "combined"]:
        print(f"\n  {cond}:")
        for K in K_VALUES:
            ridge_r2 = [r["overall_r2"] for r in all_results
                        if r["condition"] == cond and r["model"] == "Ridge" and r["K"] == K]
            mlp_r2 = [r["overall_r2"] for r in all_results
                      if r["condition"] == cond and r["model"] == "MLP_512x3" and r["K"] == K]
            if ridge_r2 and mlp_r2:
                d = mlp_r2[0] - ridge_r2[0]
                verdict = "MLP WINS ✓" if d > 0.005 else ("TIE ~" if d > -0.005 else "RIDGE WINS")
                print(f"    K={K:2d}  Ridge={ridge_r2[0]:.4f}  MLP512={mlp_r2[0]:.4f}  "
                      f"Δ={d:+.4f}  → {verdict}")

    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
