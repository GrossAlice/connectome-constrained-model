#!/usr/bin/env python
"""LOO baseline: Ridge & MLP models for comparison with stage2.

Trains ONE model for ALL neurons (same as stage2), then evaluates with
the same LOO masking protocol: at eval time, one neuron is replaced
with the model's own prediction while all others are clamped to GT.

This matches stage2 exactly:
  - Stage2: train joint connectome ODE for all 123 neurons, mask at eval
  - Here:   train joint Ridge/MLP for all 123 neurons, mask at eval

Evaluation uses the stage2 windowed LOO protocol:
  - window_size = 50, warmup = 40 (re-seed every 50 frames)
  - Within each window, neuron i free-runs using the model's own
    prediction; all other neurons clamped to ground truth.

Usage
-----
    python -m scripts.loo_baseline_ridge_mlp \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
        --save_dir output_plots/stage2/loo_baselines \
        --stage2_npz output_plots/stage2/default_config_run_v2/cv_onestep.npz \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


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


# ═══════════════════════════════════════════════════════════════════════
#  MLP model: ONE model predicts ALL N neurons
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MLPSpec:
    name: str
    hidden: Tuple[int, ...]
    dropout: float = 0.0
    batch_norm: bool = False


# Architectures to sweep
MLP_SPECS: List[MLPSpec] = [
    MLPSpec("MLP-128",          (128,)),
    MLPSpec("MLP-256",          (256,)),
    MLPSpec("MLP-128-128",      (128, 128)),
    MLPSpec("MLP-256-128",      (256, 128)),
    MLPSpec("MLP-256-256",      (256, 256)),
    MLPSpec("MLP-512-256",      (512, 256)),
    MLPSpec("MLP-256-256-128",  (256, 256, 128)),
    MLPSpec("MLP-256-128-d0.1", (256, 128), dropout=0.1),
]


class JointMLP(nn.Module):
    """MLP: x(t) ∈ R^N → x(t+1) ∈ R^N  (one model for all neurons)."""

    def __init__(self, n_neurons: int, spec: MLPSpec):
        super().__init__()
        layers = []
        in_dim = n_neurons
        for h in spec.hidden:
            layers.append(nn.Linear(in_dim, h))
            if spec.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if spec.dropout > 0:
                layers.append(nn.Dropout(spec.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_neurons))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (batch, N) → (batch, N)


# ═══════════════════════════════════════════════════════════════════════
#  Training: ONE model for ALL neurons
# ═══════════════════════════════════════════════════════════════════════

def train_joint_ridge(
    X_train: np.ndarray, Y_train: np.ndarray,
) -> Ridge:
    """Train ONE multi-output Ridge:  x(t) → x(t+1) for all neurons.

    Uses CV to pick best alpha.
    """
    alphas = np.logspace(-3, 6, 30)
    best_alpha, best_score = None, -np.inf
    # Simple manual CV (3-fold temporal)
    n = X_train.shape[0]
    fold_size = n // 3
    for alpha in alphas:
        scores = []
        for k in range(3):
            val_s = k * fold_size
            val_e = val_s + fold_size if k < 2 else n
            X_tr = np.concatenate([X_train[:val_s], X_train[val_e:]], axis=0)
            Y_tr = np.concatenate([Y_train[:val_s], Y_train[val_e:]], axis=0)
            X_va = X_train[val_s:val_e]
            Y_va = Y_train[val_s:val_e]
            m = Ridge(alpha=alpha)
            m.fit(X_tr, Y_tr)
            scores.append(m.score(X_va, Y_va))
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    print(f"    Ridge best alpha={best_alpha:.4g}  CV-R²={best_score:.4f}")
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, Y_train)
    return model


def train_joint_mlp(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    spec: MLPSpec,
    device: str = "cuda",
    max_epochs: int = 500,
    patience: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_frac: float = 0.15,
) -> nn.Module:
    """Train ONE MLP that maps x(t) → x(t+1) for all N neurons."""
    N = X_train.shape[1]
    n_total = X_train.shape[0]
    n_val = max(int(n_total * val_frac), 1)
    n_fit = n_total - n_val

    # Temporal split: last chunk for validation
    X_fit = torch.tensor(X_train[:n_fit], dtype=torch.float32, device=device)
    X_val = torch.tensor(X_train[n_fit:], dtype=torch.float32, device=device)
    Y_fit = torch.tensor(Y_train[:n_fit], dtype=torch.float32, device=device)
    Y_val = torch.tensor(Y_train[n_fit:], dtype=torch.float32, device=device)

    model = JointMLP(N, spec).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_fit, device=device)
        for b_start in range(0, n_fit, batch_size):
            b_idx = perm[b_start : b_start + batch_size]
            pred = model(X_fit[b_idx])
            loss = nn.functional.mse_loss(pred, Y_fit[b_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.functional.mse_loss(val_pred, Y_val).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    print(f"    stopped at epoch {epoch+1}, val_loss={best_val_loss:.6f}")
    return model


# ═══════════════════════════════════════════════════════════════════════
#  LOO evaluation: mask ONE neuron, clamp all others to GT
#  (exactly matching stage2's protocol)
# ═══════════════════════════════════════════════════════════════════════

def loo_windowed_ridge(
    ridge_model: Ridge,
    u_test: np.ndarray,
    neuron_idx: int,
    window_size: int = 50,
) -> np.ndarray:
    """Windowed LOO: re-seed every window_size frames."""
    T, N = u_test.shape
    u_pred = np.zeros(T)
    u_pred[0] = u_test[0, neuron_idx]

    for w_start in range(0, T, window_size):
        w_end = min(w_start + window_size, T)
        u_pred[w_start] = u_test[w_start, neuron_idx]

        for t in range(w_start, w_end - 1):
            x_t = u_test[t].copy()
            x_t[neuron_idx] = u_pred[t]               # mask: replace with own prediction
            all_pred = ridge_model.predict(x_t.reshape(1, -1))[0]  # predict all N
            u_pred[t + 1] = all_pred[neuron_idx]       # extract neuron i

    return u_pred


def loo_windowed_mlp(
    mlp_model: nn.Module,
    u_test: np.ndarray,
    neuron_idx: int,
    device: str = "cuda",
    window_size: int = 50,
) -> np.ndarray:
    """Windowed LOO: re-seed every window_size frames."""
    T, N = u_test.shape
    u_pred = np.zeros(T, dtype=np.float32)
    u_pred[0] = u_test[0, neuron_idx]

    mlp_model.eval()
    with torch.no_grad():
        for w_start in range(0, T, window_size):
            w_end = min(w_start + window_size, T)
            u_pred[w_start] = u_test[w_start, neuron_idx]

            for t in range(w_start, w_end - 1):
                x_t = u_test[t].copy()
                x_t[neuron_idx] = u_pred[t]
                x_t_tensor = torch.tensor(x_t, dtype=torch.float32, device=device).unsqueeze(0)
                all_pred = mlp_model(x_t_tensor)       # (1, N)
                u_pred[t + 1] = all_pred[0, neuron_idx].cpu().item()

    return u_pred


def loo_full_ridge(
    ridge_model: Ridge,
    u_test: np.ndarray,
    neuron_idx: int,
) -> np.ndarray:
    """Full (non-windowed) LOO."""
    T, N = u_test.shape
    u_pred = np.zeros(T)
    u_pred[0] = u_test[0, neuron_idx]
    for t in range(T - 1):
        x_t = u_test[t].copy()
        x_t[neuron_idx] = u_pred[t]
        all_pred = ridge_model.predict(x_t.reshape(1, -1))[0]
        u_pred[t + 1] = all_pred[neuron_idx]
    return u_pred


def loo_full_mlp(
    mlp_model: nn.Module,
    u_test: np.ndarray,
    neuron_idx: int,
    device: str = "cuda",
) -> np.ndarray:
    """Full (non-windowed) LOO."""
    T, N = u_test.shape
    u_pred = np.zeros(T, dtype=np.float32)
    u_pred[0] = u_test[0, neuron_idx]
    mlp_model.eval()
    with torch.no_grad():
        for t in range(T - 1):
            x_t = u_test[t].copy()
            x_t[neuron_idx] = u_pred[t]
            x_t_tensor = torch.tensor(x_t, dtype=torch.float32, device=device).unsqueeze(0)
            all_pred = mlp_model(x_t_tensor)
            u_pred[t + 1] = all_pred[0, neuron_idx].cpu().item()
    return u_pred


# ═══════════════════════════════════════════════════════════════════════
#  Full pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_baseline_comparison(
    h5_path: str,
    save_dir: str,
    stage2_npz: Optional[str] = None,
    device: str = "cuda",
    loo_subset_size: int = 30,
    window_size: int = 50,
    warmup_steps: int = 40,
    mlp_epochs: int = 500,
):
    save = Path(save_dir)
    save.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading data …")
    with h5py.File(h5_path, "r") as f:
        u = f["stage1/u_mean"][:]                         # (T, N) float32
        labels = [
            (l.decode() if isinstance(l, bytes) else str(l))
            for l in f["gcamp/neuron_labels"][:]
        ]
    T, N = u.shape
    print(f"  T={T}, N={N}")

    # ── Same 2-fold temporal CV as stage2 ─────────────────────────────
    mid = T // 2 + 1          # = 801
    folds = [(mid, T, 0, mid), (0, mid, mid, T)]
    print(f"  Folds: train/test = [{mid},{T})/[0,{mid}), [0,{mid})/[{mid},{T})")

    # ── Select LOO neurons (top-variance, same as stage2) ─────────────
    var = np.var(u, axis=0)
    loo_neurons = sorted(np.argsort(var)[::-1][:loo_subset_size].tolist())
    print(f"  LOO neurons ({len(loo_neurons)}): {loo_neurons}")

    # ── Load stage2 results for comparison ────────────────────────────
    stage2_loo_r2 = None
    stage2_loo_r2_w = None
    stage2_cv_r2 = None
    if stage2_npz and Path(stage2_npz).exists():
        s2 = np.load(stage2_npz, allow_pickle=True)
        stage2_loo_r2 = s2["cv_loo_r2"]
        stage2_loo_r2_w = s2.get("cv_loo_r2_windowed", None)
        stage2_cv_r2 = s2.get("cv_r2", None)
        print("  Loaded stage2 reference results.")

    # ── Model registry ────────────────────────────────────────────────
    model_names = ["Ridge"] + [s.name for s in MLP_SPECS]

    # Per-fold accumulated predictions
    cv_preds: Dict[str, np.ndarray] = {m: np.full((T, N), np.nan) for m in model_names}
    cv_preds_w: Dict[str, np.ndarray] = {m: np.full((T, N), np.nan) for m in model_names}
    cv_onestep_preds: Dict[str, np.ndarray] = {m: np.full((T, N), np.nan) for m in model_names}

    t_start = time.time()

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx+1}/2  train=[{tr_s},{tr_e})  test=[{te_s},{te_e})")
        print(f"{'='*60}")

        u_train = u[tr_s:tr_e]
        u_test  = u[te_s:te_e]

        # One-step data: X=u(t), Y=u(t+1)
        X_train = u_train[:-1]
        Y_train = u_train[1:]
        X_test = u_test[:-1]

        # ── Ridge (one model, all neurons) ────────────────────────────
        print("\n  Training Ridge (joint, all neurons) …")
        t0 = time.time()
        ridge_model = train_joint_ridge(X_train, Y_train)
        print(f"    done in {time.time()-t0:.1f}s")

        # One-step prediction (all neurons)
        pred_os_all = ridge_model.predict(X_test)  # (T_test-1, N)
        cv_onestep_preds["Ridge"][te_s + 1 : te_e] = pred_os_all

        # LOO eval
        print("  LOO eval (Ridge) …")
        for cnt, i in enumerate(loo_neurons):
            pred = loo_full_ridge(ridge_model, u_test, i)
            pred_w = loo_windowed_ridge(ridge_model, u_test, i,
                                        window_size=window_size)
            r2_full = _r2(u_test[1:, i], pred[1:])
            r2_w = _r2(u_test[1:, i], pred_w[1:])
            cv_preds["Ridge"][te_s:te_e, i] = pred
            cv_preds_w["Ridge"][te_s:te_e, i] = pred_w
            if cnt == 0 or (cnt + 1) % 10 == 0 or cnt == len(loo_neurons) - 1:
                print(f"    [{cnt+1:2d}/{len(loo_neurons)}] neuron {i:3d} "
                      f"({labels[i]:8s})  LOO={r2_full:.4f}  LOO_w={r2_w:.4f}")

        del ridge_model

        # ── MLPs (one model each, all neurons) ────────────────────────
        for spec in MLP_SPECS:
            print(f"\n  Training {spec.name} (joint, all neurons) …")
            t0 = time.time()
            mlp_model = train_joint_mlp(
                X_train, Y_train, spec,
                device=device, max_epochs=mlp_epochs,
            )
            print(f"    done in {time.time()-t0:.1f}s")

            # One-step prediction (all neurons)
            with torch.no_grad():
                X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
                pred_os_all = mlp_model(X_test_t).cpu().numpy()
                cv_onestep_preds[spec.name][te_s + 1 : te_e] = pred_os_all

            # LOO eval
            print(f"  LOO eval ({spec.name}) …")
            for cnt, i in enumerate(loo_neurons):
                pred = loo_full_mlp(mlp_model, u_test, i, device=device)
                pred_w = loo_windowed_mlp(mlp_model, u_test, i,
                                          device=device,
                                          window_size=window_size)
                r2_full = _r2(u_test[1:, i], pred[1:])
                r2_w = _r2(u_test[1:, i], pred_w[1:])
                cv_preds[spec.name][te_s:te_e, i] = pred
                cv_preds_w[spec.name][te_s:te_e, i] = pred_w
                if cnt == 0 or (cnt + 1) % 10 == 0 or cnt == len(loo_neurons) - 1:
                    print(f"    [{cnt+1:2d}/{len(loo_neurons)}] neuron {i:3d} "
                          f"({labels[i]:8s})  LOO={r2_full:.4f}  LOO_w={r2_w:.4f}")

            del mlp_model
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ═══════════════════════════════════════════════════════════════════
    #  Aggregate CV results
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Cross-validated results summary")
    print("=" * 70)

    summary: Dict[str, Dict] = {}
    u_gt = u

    for mname in model_names:
        # LOO R² per neuron
        r2_arr = np.full(N, np.nan)
        r2_w_arr = np.full(N, np.nan)
        for i in loo_neurons:
            pred = cv_preds[mname][:, i]
            pred_w = cv_preds_w[mname][:, i]
            m = np.isfinite(pred)
            r2_arr[i] = _r2(u_gt[m, i], pred[m])
            m_w = np.isfinite(pred_w)
            r2_w_arr[i] = _r2(u_gt[m_w, i], pred_w[m_w])

        # One-step R² per neuron (all N neurons)
        os_r2 = np.full(N, np.nan)
        for i in range(N):
            os_pred = cv_onestep_preds[mname][:, i]
            m = np.isfinite(os_pred)
            if m.sum() > 3:
                os_r2[i] = _r2(u_gt[m, i], os_pred[m])

        loo_vals = r2_arr[loo_neurons]
        loo_vals_w = r2_w_arr[loo_neurons]
        summary[mname] = {
            "loo_r2": r2_arr.tolist(),
            "loo_r2_windowed": r2_w_arr.tolist(),
            "onestep_r2": os_r2.tolist(),
            "mean_loo_r2": float(np.nanmean(loo_vals)),
            "mean_loo_r2_w": float(np.nanmean(loo_vals_w)),
            "median_loo_r2": float(np.nanmedian(loo_vals)),
            "median_loo_r2_w": float(np.nanmedian(loo_vals_w)),
            "mean_onestep_r2_all": float(np.nanmean(os_r2)),
            "mean_onestep_r2_loo30": float(np.nanmean(os_r2[loo_neurons])),
        }
        print(f"\n  {mname:24s}  1step(all)={np.nanmean(os_r2):.4f}  "
              f"LOO={np.nanmean(loo_vals):.4f}  LOO_w={np.nanmean(loo_vals_w):.4f}  "
              f"med_w={np.nanmedian(loo_vals_w):.4f}")

    # Stage2 reference
    if stage2_loo_r2 is not None:
        s2_loo = stage2_loo_r2[loo_neurons]
        s2_loo_w = stage2_loo_r2_w[loo_neurons] if stage2_loo_r2_w is not None else s2_loo
        s2_os = float(np.nanmean(stage2_cv_r2)) if stage2_cv_r2 is not None else float("nan")
        print(f"\n  {'Stage2 (connectome)':24s}  1step(all)={s2_os:.4f}  "
              f"LOO={np.nanmean(s2_loo):.4f}  LOO_w={np.nanmean(s2_loo_w):.4f}  "
              f"med_w={np.nanmedian(s2_loo_w):.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    results_path = save / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    np_data = {"loo_neurons": np.array(loo_neurons)}
    for mname in model_names:
        key = mname.replace(" ", "_").replace("-", "_").replace(".", "p")
        np_data[f"{key}_loo_pred"] = cv_preds[mname][:, loo_neurons]
        np_data[f"{key}_loo_pred_w"] = cv_preds_w[mname][:, loo_neurons]
        np_data[f"{key}_onestep"] = cv_onestep_preds[mname]
    np.savez(save / "baseline_predictions.npz", **np_data)

    # ═══════════════════════════════════════════════════════════════════
    #  Plots
    # ═══════════════════════════════════════════════════════════════════
    _plot_comparison(summary, loo_neurons, labels,
                     stage2_loo_r2, stage2_loo_r2_w, stage2_cv_r2, save)

    best_neurons = sorted(loo_neurons,
                          key=lambda i: stage2_loo_r2_w[i] if stage2_loo_r2_w is not None else 0,
                          reverse=True)[:6]
    _plot_traces(u_gt, cv_preds_w, best_neurons, labels,
                 stage2_npz, save)

    return summary


def _plot_comparison(
    summary: Dict,
    loo_neurons: List[int],
    labels: List[str],
    stage2_loo_r2: Optional[np.ndarray],
    stage2_loo_r2_w: Optional[np.ndarray],
    stage2_cv_r2: Optional[np.ndarray],
    save: Path,
):
    model_names = list(summary.keys())

    # ── 1) Bar chart: mean & median LOO R² ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    means = [summary[m]["mean_loo_r2_w"] for m in model_names]
    medians = [summary[m]["median_loo_r2_w"] for m in model_names]
    x_labels = model_names[:]
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    if stage2_loo_r2_w is not None:
        means.append(float(np.nanmean(stage2_loo_r2_w[loo_neurons])))
        medians.append(float(np.nanmedian(stage2_loo_r2_w[loo_neurons])))
        x_labels.append("Stage2\n(connectome)")
        colors = np.vstack([colors, [[0.2, 0.2, 0.2, 1.0]]])

    ax = axes[0]
    bars = ax.bar(range(len(means)), means, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean LOO R² (windowed)")
    ax.set_title("Mean windowed LOO R² (30 neurons)\nTrain ONE model, mask at eval")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax = axes[1]
    bars = ax.bar(range(len(medians)), medians, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Median LOO R² (windowed)")
    ax.set_title("Median windowed LOO R² (30 neurons)\nTrain ONE model, mask at eval")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(save / "loo_bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save / 'loo_bar_comparison.png'}")

    # ── 2) Per-neuron scatter: each baseline vs stage2 ────────────────
    if stage2_loo_r2_w is None:
        return

    n_models = len(model_names)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                              squeeze=False)

    s2_vals = stage2_loo_r2_w[loo_neurons]
    for idx, mname in enumerate(model_names):
        ax = axes[idx // n_cols, idx % n_cols]
        m_r2_w = np.array(summary[mname]["loo_r2_windowed"])[loo_neurons]
        ax.scatter(s2_vals, m_r2_w, s=25, alpha=0.7, edgecolors="black", linewidth=0.3)
        all_vals = np.concatenate([s2_vals, m_r2_w])
        lims = [min(all_vals.min(), -0.1) - 0.05,
                max(all_vals.max(), 0.7) + 0.05]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Stage2 LOO R² (windowed)")
        ax.set_ylabel(f"{mname} LOO R²")
        ax.set_title(mname, fontsize=10)
        n_above = np.sum(m_r2_w > s2_vals)
        n_below = np.sum(m_r2_w < s2_vals)
        ax.text(0.05, 0.95, f"{mname} wins: {n_above}\nStage2 wins: {n_below}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    plt.tight_layout()
    fig.savefig(save / "loo_scatter_vs_stage2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save / 'loo_scatter_vs_stage2.png'}")

    # ── 3) Per-neuron grouped bar chart ───────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 6))
    n_groups = len(loo_neurons)
    n_bars = n_models + 1
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    for idx, mname in enumerate(model_names):
        vals = np.array(summary[mname]["loo_r2_windowed"])[loo_neurons]
        ax.bar(x + idx * width, vals, width, label=mname, alpha=0.85)

    ax.bar(x + n_models * width, s2_vals, width,
           label="Stage2 (connectome)", color="black", alpha=0.7)

    ax.set_xticks(x + (n_bars - 1) * width / 2)
    ax.set_xticklabels([f"{labels[i]}\n({i})" for i in loo_neurons],
                        rotation=90, fontsize=6)
    ax.set_ylabel("LOO R² (windowed)")
    ax.set_title("Per-neuron LOO R²: baselines vs Stage2 (all trained jointly, masked at eval)")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    plt.tight_layout()
    fig.savefig(save / "loo_per_neuron_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save / 'loo_per_neuron_bars.png'}")

    # ── 4) One-step R² comparison (all neurons) ──────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    os_means = [summary[m]["mean_onestep_r2_all"] for m in model_names]
    x_labels_os = model_names[:]
    colors_os = list(plt.cm.tab10(np.linspace(0, 1, len(model_names))))

    if stage2_cv_r2 is not None:
        os_means.append(float(np.nanmean(stage2_cv_r2)))
        x_labels_os.append("Stage2\n(connectome)")
        colors_os.append((0.2, 0.2, 0.2, 1.0))

    bars = ax.bar(range(len(os_means)), os_means, color=colors_os,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(x_labels_os)))
    ax.set_xticklabels(x_labels_os, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean one-step R² (all 123 neurons)")
    ax.set_title("One-step prediction: joint baselines vs Stage2")
    for bar, val in zip(bars, os_means):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(save / "onestep_bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save / 'onestep_bar_comparison.png'}")


def _plot_traces(
    u_gt: np.ndarray,
    cv_preds_w: Dict[str, np.ndarray],
    neurons: List[int],
    labels: List[str],
    stage2_npz: Optional[str],
    save: Path,
):
    s2_preds = {}
    if stage2_npz and Path(stage2_npz).exists():
        s2 = np.load(stage2_npz, allow_pickle=True)
        for k in s2.keys():
            if k.startswith("loo_pred_"):
                idx = int(k.split("_")[-1])
                s2_preds[idx] = s2[k]

    model_names = list(cv_preds_w.keys())
    show_models = ["Ridge", "MLP-256-128", "MLP-256-256"]
    show_models = [m for m in show_models if m in model_names]

    n_neurons = len(neurons)
    fig, axes = plt.subplots(n_neurons, 1, figsize=(16, 3 * n_neurons), sharex=True)
    if n_neurons == 1:
        axes = [axes]

    t = np.arange(u_gt.shape[0])
    for ax, nidx in zip(axes, neurons):
        ax.plot(t, u_gt[:, nidx], "k-", lw=1.2, alpha=0.8, label="Ground truth")
        if nidx in s2_preds:
            ax.plot(t, s2_preds[nidx], "--", lw=1.0, alpha=0.7, label="Stage2")
        for mname in show_models:
            pred = cv_preds_w[mname][:, nidx]
            m = np.isfinite(pred)
            ax.plot(t[m], pred[m], "-", lw=0.8, alpha=0.6, label=mname)
        ax.set_ylabel(f"{labels[nidx]} ({nidx})")
        ax.legend(fontsize=7, loc="upper right", ncol=4)
        ax.axvline(801, color="red", lw=0.5, ls=":", alpha=0.5)

    axes[-1].set_xlabel("Time (frames)")
    fig.suptitle("LOO traces: joint baselines vs Stage2 (windowed)", fontsize=12)
    plt.tight_layout()
    fig.savefig(save / "loo_traces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save / 'loo_traces.png'}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ridge & MLP LOO baselines (joint models) for stage2 comparison")
    parser.add_argument("--h5", required=True, help="Path to HDF5 file")
    parser.add_argument("--save_dir", required=True, help="Output directory")
    parser.add_argument("--stage2_npz", default=None,
                        help="Path to stage2 cv_onestep.npz for comparison")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loo_subset", type=int, default=30)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--mlp_epochs", type=int, default=500)
    args = parser.parse_args()

    run_baseline_comparison(
        h5_path=args.h5,
        save_dir=args.save_dir,
        stage2_npz=args.stage2_npz,
        device=args.device,
        loo_subset_size=args.loo_subset,
        window_size=args.window_size,
        warmup_steps=args.warmup_steps,
        mlp_epochs=args.mlp_epochs,
    )


if __name__ == "__main__":
    main()
