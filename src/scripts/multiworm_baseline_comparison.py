#!/usr/bin/env python
"""Multi-worm baseline comparison: Stage2 vs MLP vs ElasticNet vs Ridge-Diag.

Runs joint-model baselines (Ridge-CV, EN(0.1,0.1), Ridge-Diag(1,1000),
MLP-256, MLP-512-256) for multiple worms and generates a combined
comparison plot.

Also trains stage2 for each worm if no pre-existing results are found.

Usage
-----
    python -m scripts.multiworm_baseline_comparison \
        --worms 2022-08-02-01 2023-01-09-15 2022-06-14-13 \
        --save_dir output_plots/stage2/multiworm_baselines \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, ElasticNet


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m].astype(np.float64), y_pred[m].astype(np.float64)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot < 1e-15:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _spectral_radius(W: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(W))))


# ═══════════════════════════════════════════════════════════════════════
#  MLP model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MLPSpec:
    name: str
    hidden: Tuple[int, ...]
    dropout: float = 0.0


class JointMLP(nn.Module):
    def __init__(self, n_neurons: int, spec: MLPSpec):
        super().__init__()
        layers = []
        in_dim = n_neurons
        for h in spec.hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if spec.dropout > 0:
                layers.append(nn.Dropout(spec.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_neurons))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Subset of architectures (best performers from prior run)
MLP_SPECS = [
    MLPSpec("MLP-256", (256,)),
    MLPSpec("MLP-512-256", (512, 256)),
]


# ═══════════════════════════════════════════════════════════════════════
#  Training functions
# ═══════════════════════════════════════════════════════════════════════

def train_ridge_cv(X, Y):
    alphas = np.logspace(-3, 6, 30)
    best_alpha, best_score = None, -np.inf
    n = X.shape[0]; fold_size = n // 3
    for alpha in alphas:
        scores = []
        for k in range(3):
            vs = k * fold_size
            ve = vs + fold_size if k < 2 else n
            Xtr = np.concatenate([X[:vs], X[ve:]])
            Ytr = np.concatenate([Y[:vs], Y[ve:]])
            m = Ridge(alpha=alpha).fit(Xtr, Ytr)
            scores.append(m.score(X[vs:ve], Y[vs:ve]))
        ms = np.mean(scores)
        if ms > best_score: best_score, best_alpha = ms, alpha
    model = Ridge(alpha=best_alpha).fit(X, Y)
    print(f"    Ridge-CV: alpha={best_alpha:.4g}  CV-R²={best_score:.4f}")
    return model


def train_elastic_net(X, Y, alpha=0.1, l1_ratio=0.1):
    N = Y.shape[1]
    W = np.zeros((N, X.shape[1]), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    for j in range(N):
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        en.fit(X, Y[:, j])
        W[j] = en.coef_
        b[j] = en.intercept_
    print(f"    EN(α={alpha},l1={l1_ratio}): sparsity={1-np.count_nonzero(W)/W.size:.1%}")
    return W, b


def train_ridge_diag(X, Y, alpha_diag=1.0, alpha_off=1000.0):
    N = X.shape[1]
    X_mean, Y_mean = X.mean(0), Y.mean(0)
    Xc, Yc = X - X_mean, Y - Y_mean
    XtX = Xc.T @ Xc
    XtY = Xc.T @ Yc
    W = np.zeros((N, N), dtype=np.float64)
    for j in range(N):
        penalty = np.full(N, alpha_off)
        penalty[j] = alpha_diag
        W[j] = np.linalg.solve(XtX + np.diag(penalty), XtY[:, j])
    b = Y_mean - X_mean @ W.T
    print(f"    Ridge-Diag(d={alpha_diag},o={alpha_off}): ρ(W)={_spectral_radius(W):.4f}")
    return W, b


def train_joint_mlp(X, Y, spec, device="cuda", max_epochs=500, patience=30):
    N = X.shape[1]
    n_total = X.shape[0]
    n_val = max(int(n_total * 0.15), 1)
    n_fit = n_total - n_val

    X_fit = torch.tensor(X[:n_fit], dtype=torch.float32, device=device)
    X_val = torch.tensor(X[n_fit:], dtype=torch.float32, device=device)
    Y_fit = torch.tensor(Y[:n_fit], dtype=torch.float32, device=device)
    Y_val = torch.tensor(Y[n_fit:], dtype=torch.float32, device=device)

    model = JointMLP(N, spec).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_fit, device=device)
        for bs in range(0, n_fit, 256):
            idx = perm[bs:bs+256]
            loss = nn.functional.mse_loss(model(X_fit[idx]), Y_fit[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(X_val), Y_val).item()
        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    print(f"    {spec.name}: stopped epoch {epoch+1}, val_loss={best_val:.6f}")
    return model


# ═══════════════════════════════════════════════════════════════════════
#  LOO evaluation
# ═══════════════════════════════════════════════════════════════════════

def loo_windowed_linear(W, b, u_test, neuron_idx, window_size=50):
    T, N = u_test.shape
    u_pred = np.zeros(T, dtype=np.float64)
    u_pred[0] = u_test[0, neuron_idx]
    for ws in range(0, T, window_size):
        we = min(ws + window_size, T)
        u_pred[ws] = u_test[ws, neuron_idx]
        for t in range(ws, we - 1):
            x_t = u_test[t].copy()
            x_t[neuron_idx] = u_pred[t]
            pred = x_t @ W.T + b
            u_pred[t+1] = pred[neuron_idx]
    return u_pred


def loo_windowed_sklearn(model, u_test, neuron_idx, window_size=50):
    """For sklearn Ridge objects."""
    return loo_windowed_linear(model.coef_, model.intercept_, u_test, neuron_idx, window_size)


def loo_windowed_mlp(model, u_test, neuron_idx, device="cuda", window_size=50):
    T, N = u_test.shape
    u_pred = np.zeros(T, dtype=np.float32)
    u_pred[0] = u_test[0, neuron_idx]
    model.eval()
    with torch.no_grad():
        for ws in range(0, T, window_size):
            we = min(ws + window_size, T)
            u_pred[ws] = u_test[ws, neuron_idx]
            for t in range(ws, we - 1):
                x_t = u_test[t].copy()
                x_t[neuron_idx] = u_pred[t]
                xt = torch.tensor(x_t, dtype=torch.float32, device=device).unsqueeze(0)
                pred = model(xt)
                u_pred[t+1] = pred[0, neuron_idx].cpu().item()
    return u_pred


# ═══════════════════════════════════════════════════════════════════════
#  Run stage2 for a worm (if needed)
# ═══════════════════════════════════════════════════════════════════════

def run_stage2_for_worm(worm_name: str, h5_path: str, save_dir: Path,
                        device: str = "cuda") -> Optional[np.lib.npyio.NpzFile]:
    """Try to load existing stage2 results, or train from scratch."""
    npz_path = save_dir / worm_name / "cv_onestep.npz"
    if npz_path.exists():
        print(f"  Found existing stage2 results: {npz_path}")
        return np.load(npz_path, allow_pickle=True)

    print(f"  Training stage2 for {worm_name} ...")
    import subprocess, sys
    out_dir = save_dir / worm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "stage2.run_stage2",
        "--h5", h5_path,
        "--save_dir", str(out_dir),
        "--device", device,
        "--set", "epochs", "150",
        "--set", "cv_folds", "2",
        "--set", "loo_subset", "30",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(save_dir.parent.parent))
    if result.returncode != 0:
        print(f"  ⚠ stage2 training failed for {worm_name}:")
        print(result.stderr[-500:] if result.stderr else "no stderr")
        return None

    if npz_path.exists():
        return np.load(npz_path, allow_pickle=True)
    # Check if it saved elsewhere
    for p in out_dir.glob("**/cv_onestep.npz"):
        return np.load(p, allow_pickle=True)
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Per-worm pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_one_worm(
    worm_name: str,
    h5_path: str,
    device: str = "cuda",
    loo_subset: int = 30,
    window_size: int = 50,
    mlp_epochs: int = 500,
) -> Dict:
    """Run all baselines for one worm, return per-model summary dict."""
    print(f"\n{'═'*70}")
    print(f"  WORM: {worm_name}")
    print(f"{'═'*70}")

    with h5py.File(h5_path, "r") as f:
        u = f["stage1/u_mean"][:]
        labels = [(l.decode() if isinstance(l, bytes) else str(l))
                  for l in f["gcamp/neuron_labels"][:]]
    T, N = u.shape
    print(f"  T={T}, N={N}")

    mid = T // 2 + 1
    folds = [(mid, T, 0, mid), (0, mid, mid, T)]

    var = np.var(u, axis=0)
    loo_neurons = sorted(np.argsort(var)[::-1][:loo_subset].tolist())

    # Model definitions: name → (train_fn, predict_fn, loo_fn)
    # We'll handle each model type explicitly for clarity
    MODEL_NAMES = ["Ridge-CV", "EN(0.1,0.1)", "Ridge-Diag(1,1000)"] + [s.name for s in MLP_SPECS]

    cv_preds_w = {m: np.full((T, N), np.nan) for m in MODEL_NAMES}
    cv_onestep = {m: np.full((T, N), np.nan) for m in MODEL_NAMES}

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        print(f"\n  Fold {fold_idx+1}/2  train=[{tr_s},{tr_e})  test=[{te_s},{te_e})")
        u_train, u_test = u[tr_s:tr_e], u[te_s:te_e]
        X_train, Y_train = u_train[:-1], u_train[1:]
        X_test = u_test[:-1]

        # ── Ridge-CV ──
        print("  Training Ridge-CV ...")
        ridge = train_ridge_cv(X_train, Y_train)
        cv_onestep["Ridge-CV"][te_s+1:te_e] = ridge.predict(X_test)
        for j, i in enumerate(loo_neurons):
            cv_preds_w["Ridge-CV"][te_s:te_e, i] = loo_windowed_sklearn(
                ridge, u_test, i, window_size)
        del ridge

        # ── EN(0.1, 0.1) ──
        print("  Training EN(0.1,0.1) ...")
        W_en, b_en = train_elastic_net(X_train, Y_train, alpha=0.1, l1_ratio=0.1)
        cv_onestep["EN(0.1,0.1)"][te_s+1:te_e] = X_test @ W_en.T + b_en
        for j, i in enumerate(loo_neurons):
            cv_preds_w["EN(0.1,0.1)"][te_s:te_e, i] = loo_windowed_linear(
                W_en, b_en, u_test, i, window_size)
        del W_en, b_en

        # ── Ridge-Diag(1,1000) ──
        print("  Training Ridge-Diag(1,1000) ...")
        W_diag, b_diag = train_ridge_diag(X_train, Y_train, 1.0, 1000.0)
        cv_onestep["Ridge-Diag(1,1000)"][te_s+1:te_e] = X_test @ W_diag.T + b_diag
        for j, i in enumerate(loo_neurons):
            cv_preds_w["Ridge-Diag(1,1000)"][te_s:te_e, i] = loo_windowed_linear(
                W_diag, b_diag, u_test, i, window_size)
        del W_diag, b_diag

        # ── MLPs ──
        for spec in MLP_SPECS:
            print(f"  Training {spec.name} ...")
            mlp = train_joint_mlp(X_train, Y_train, spec, device=device,
                                  max_epochs=mlp_epochs)
            with torch.no_grad():
                Xt = torch.tensor(X_test, dtype=torch.float32, device=device)
                cv_onestep[spec.name][te_s+1:te_e] = mlp(Xt).cpu().numpy()
            for j, i in enumerate(loo_neurons):
                cv_preds_w[spec.name][te_s:te_e, i] = loo_windowed_mlp(
                    mlp, u_test, i, device=device, window_size=window_size)
            del mlp
            torch.cuda.empty_cache()

    # ── Aggregate ──
    summary = {}
    for mname in MODEL_NAMES:
        r2_w = np.full(N, np.nan)
        for i in loo_neurons:
            pred = cv_preds_w[mname][:, i]
            m = np.isfinite(pred)
            if m.sum() > 3:
                r2_w[i] = _r2(u[m, i], pred[m])

        os_r2 = np.full(N, np.nan)
        for i in range(N):
            pred = cv_onestep[mname][:, i]
            m = np.isfinite(pred)
            if m.sum() > 3:
                os_r2[i] = _r2(u[m, i], pred[m])

        loo_vals = r2_w[loo_neurons]
        summary[mname] = {
            "mean_onestep": float(np.nanmean(os_r2)),
            "mean_loo_w": float(np.nanmean(loo_vals)),
            "median_loo_w": float(np.nanmedian(loo_vals)),
            "q25_loo_w": float(np.nanpercentile(loo_vals, 25)),
            "q75_loo_w": float(np.nanpercentile(loo_vals, 75)),
            "per_neuron_loo_w": {str(loo_neurons[j]): float(loo_vals[j])
                                 for j in range(len(loo_neurons))},
        }
        print(f"  {mname:25s}  1step={np.nanmean(os_r2):.4f}  "
              f"LOO_w={np.nanmean(loo_vals):.4f}  med={np.nanmedian(loo_vals):.4f}")

    return summary


# ═══════════════════════════════════════════════════════════════════════
#  Combined multi-worm plot
# ═══════════════════════════════════════════════════════════════════════

def plot_multiworm(
    all_results: Dict[str, Dict],
    save_dir: Path,
):
    """Generate combined comparison plots across worms."""
    worm_names = list(all_results.keys())
    n_worms = len(worm_names)

    # Collect all model names (union across worms)
    all_models = []
    for worm in worm_names:
        for m in all_results[worm]:
            if m not in all_models:
                all_models.append(m)

    # Color map
    cmap = {
        "Ridge-CV": "#1f77b4",
        "EN(0.1,0.1)": "#2ca02c",
        "Ridge-Diag(1,1000)": "#7f7f7f",
        "MLP-256": "#ff7f0e",
        "MLP-512-256": "#d62728",
        "Stage2": "black",
    }

    # ═══ Figure 1: grouped bars — one-step and LOO_w per worm ═══
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 3 * n_worms), 10))

    n_models = len(all_models)
    width = 0.8 / n_models
    x = np.arange(n_worms)

    for metric_idx, (metric_key, ylabel, title) in enumerate([
        ("mean_onestep", "Mean one-step R²",
         "One-step R² per worm: baselines vs Stage2"),
        ("mean_loo_w", "Mean LOO R² (windowed)",
         "LOO R² (windowed) per worm: baselines vs Stage2"),
    ]):
        ax = axes[metric_idx]
        for m_idx, mname in enumerate(all_models):
            vals = []
            for worm in worm_names:
                if mname in all_results[worm]:
                    vals.append(all_results[worm][mname][metric_key])
                else:
                    vals.append(np.nan)
            color = cmap.get(mname, plt.cm.tab10(m_idx / n_models))
            bars = ax.bar(x + m_idx * width, vals, width,
                         label=mname, color=color, edgecolor="black", linewidth=0.3,
                         alpha=0.85)
            for xi, v in zip(x + m_idx * width, vals):
                if np.isfinite(v):
                    ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                            fontsize=6, rotation=90)

        ax.set_xticks(x + (n_models - 1) * width / 2)
        ax.set_xticklabels(worm_names, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.axhline(0, color="gray", lw=0.5, ls="--")

    plt.tight_layout()
    fig.savefig(save_dir / "multiworm_bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_dir / 'multiworm_bar_comparison.png'}")

    # ═══ Figure 2: box/violin per model across worms ═══
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # One-step
    ax = axes[0]
    data_os = []
    for mname in all_models:
        vals = [all_results[w][mname]["mean_onestep"]
                for w in worm_names if mname in all_results[w]]
        data_os.append(vals)
    bp = ax.boxplot(data_os, labels=all_models, patch_artist=True, widths=0.6)
    for patch, mname in zip(bp["boxes"], all_models):
        patch.set_facecolor(cmap.get(mname, "#aaaaaa"))
        patch.set_alpha(0.7)
    # Overlay individual points
    for i, (mname, vals) in enumerate(zip(all_models, data_os)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals, s=30, zorder=3,
                  edgecolors="black", linewidth=0.5,
                  color=cmap.get(mname, "#aaaaaa"))
    ax.set_ylabel("Mean one-step R²")
    ax.set_title("One-step R² across worms", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    # LOO_w
    ax = axes[1]
    data_loo = []
    for mname in all_models:
        vals = [all_results[w][mname]["mean_loo_w"]
                for w in worm_names if mname in all_results[w]]
        data_loo.append(vals)
    bp = ax.boxplot(data_loo, labels=all_models, patch_artist=True, widths=0.6)
    for patch, mname in zip(bp["boxes"], all_models):
        patch.set_facecolor(cmap.get(mname, "#aaaaaa"))
        patch.set_alpha(0.7)
    for i, (mname, vals) in enumerate(zip(all_models, data_loo)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals, s=30, zorder=3,
                  edgecolors="black", linewidth=0.5,
                  color=cmap.get(mname, "#aaaaaa"))
    ax.set_ylabel("Mean LOO R² (windowed)")
    ax.set_title("LOO R² (windowed) across worms", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(save_dir / "multiworm_box_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_dir / 'multiworm_box_comparison.png'}")

    # ═══ Figure 3: per-neuron LOO scatter (each worm: baselines vs best linear) ═══
    fig, axes = plt.subplots(1, n_worms, figsize=(6 * n_worms, 5), squeeze=False)
    for wi, worm in enumerate(worm_names):
        ax = axes[0, wi]
        res = all_results[worm]

        # If Stage2 present, scatter vs Stage2; else scatter vs Ridge-Diag
        ref_model = "Stage2" if "Stage2" in res else "Ridge-Diag(1,1000)"
        compare_models = [m for m in all_models if m != ref_model and m in res]

        if ref_model in res and "per_neuron_loo_w" in res[ref_model]:
            ref_vals = res[ref_model]["per_neuron_loo_w"]
            for cm in compare_models:
                if "per_neuron_loo_w" not in res[cm]:
                    continue
                cm_vals = res[cm]["per_neuron_loo_w"]
                common_neurons = sorted(set(ref_vals.keys()) & set(cm_vals.keys()))
                rv = np.array([ref_vals[n] for n in common_neurons])
                cv = np.array([cm_vals[n] for n in common_neurons])
                ax.scatter(rv, cv, s=15, alpha=0.5, label=cm,
                          color=cmap.get(cm, None))

            lims = [-0.3, 1.0]
            ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel(f"{ref_model} LOO R²")
            ax.set_ylabel("Baseline LOO R²")
            ax.legend(fontsize=6, loc="upper left")

        ax.set_title(worm, fontweight="bold")

    plt.suptitle("Per-neuron LOO R²: baselines vs reference", fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "multiworm_scatter_perneuron.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_dir / 'multiworm_scatter_perneuron.png'}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-worm baseline comparison")
    parser.add_argument("--worms", nargs="+", required=True,
                        help="Worm IDs (e.g. 2022-08-02-01)")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loo_subset", type=int, default=30)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--mlp_epochs", type=int, default=500)
    parser.add_argument("--stage2_results_dir", default=None,
                        help="Dir with existing stage2 cv_onestep.npz per worm")
    args = parser.parse_args()

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data/used/behaviour+neuronal activity atanas (2023)/2")

    all_results = {}
    t_total = time.time()

    for worm in args.worms:
        h5_path = str(data_dir / f"{worm}.h5")
        if not Path(h5_path).exists():
            print(f"⚠ Skipping {worm}: {h5_path} not found")
            continue

        t0 = time.time()
        summary = run_one_worm(
            worm_name=worm,
            h5_path=h5_path,
            device=args.device,
            loo_subset=args.loo_subset,
            window_size=args.window_size,
            mlp_epochs=args.mlp_epochs,
        )

        # Try loading stage2 results
        if args.stage2_results_dir:
            for candidate in [
                Path(args.stage2_results_dir) / "cv_onestep.npz",
                Path(args.stage2_results_dir) / worm / "cv_onestep.npz",
            ]:
                if candidate.exists():
                    s2 = np.load(candidate, allow_pickle=True)
                    # Load neuron list from h5
                    with h5py.File(h5_path, "r") as f:
                        u = f["stage1/u_mean"][:]
                    N = u.shape[1]
                    var = np.var(u, axis=0)
                    loo_neurons = sorted(np.argsort(var)[::-1][:args.loo_subset].tolist())

                    s2_loo_w = s2.get("cv_loo_r2_windowed", None)
                    s2_r2 = s2.get("cv_r2", None)
                    if s2_loo_w is not None:
                        loo_vals = s2_loo_w[loo_neurons]
                        summary["Stage2"] = {
                            "mean_onestep": float(np.nanmean(s2_r2)) if s2_r2 is not None else np.nan,
                            "mean_loo_w": float(np.nanmean(loo_vals)),
                            "median_loo_w": float(np.nanmedian(loo_vals)),
                            "q25_loo_w": float(np.nanpercentile(loo_vals, 25)),
                            "q75_loo_w": float(np.nanpercentile(loo_vals, 75)),
                            "per_neuron_loo_w": {str(loo_neurons[j]): float(loo_vals[j])
                                                 for j in range(len(loo_neurons))},
                        }
                        print(f"  Stage2 loaded: 1step={summary['Stage2']['mean_onestep']:.4f}  "
                              f"LOO_w={summary['Stage2']['mean_loo_w']:.4f}")
                    break

        all_results[worm] = summary
        print(f"\n  {worm} done in {time.time()-t0:.0f}s")

    print(f"\n{'═'*70}")
    print(f"  Total time: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")
    print(f"{'═'*70}")

    # Save JSON
    with open(save / "multiworm_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved {save / 'multiworm_results.json'}")

    # Generate plots
    plot_multiworm(all_results, save)


if __name__ == "__main__":
    main()
