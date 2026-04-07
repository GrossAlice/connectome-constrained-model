#!/usr/bin/env python
"""Stage2 with integrated baseline comparison.

Trains EN(0.1,0.1) and MLP-256 baselines FIRST, then runs Stage2 in two phases:
  Phase 1: one-step teacher-forced training (150 epochs)
  Phase 2: rollout + noise injection + curriculum scheduling (50 epochs)

Generates a bar-chart comparison plot of one-step R² and LOO R² (windowed).

Usage
-----
    python -m scripts.run_stage2_with_baselines \
        --h5 data/used/behaviour+neuronal\ activity\ atanas\ \(2023\)/2/2022-06-14-13.h5 \
        --save_dir output_plots/stage2/integrated_run \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, ElasticNet

# ─── stage2 imports ──────────────────────────────────────────────────
from stage2.config import make_config
from stage2.train import train_stage2_cv


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════
#  Baseline models
# ═════════════════════════════════════════════════════════════════════

class JointMLP(nn.Module):
    def __init__(self, n_neurons: int, hidden: Tuple[int, ...] = (256,)):
        super().__init__()
        layers = []
        d = n_neurons
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_neurons))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_ridge_cv(X, Y):
    alphas = np.logspace(-3, 6, 30)
    best_a, best_s = None, -np.inf
    n = X.shape[0]; fs = n // 3
    for a in alphas:
        sc = []
        for k in range(3):
            s, e = k * fs, (k + 1) * fs if k < 2 else n
            m = Ridge(alpha=a).fit(np.concatenate([X[:s], X[e:]]),
                                   np.concatenate([Y[:s], Y[e:]]))
            sc.append(m.score(X[s:e], Y[s:e]))
        ms = np.mean(sc)
        if ms > best_s:
            best_s, best_a = ms, a
    model = Ridge(alpha=best_a).fit(X, Y)
    return model


def _train_en(X, Y, alpha=0.1, l1_ratio=0.1):
    N = Y.shape[1]
    W = np.zeros((N, X.shape[1]), np.float64)
    b = np.zeros(N, np.float64)
    for j in range(N):
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        en.fit(X, Y[:, j])
        W[j] = en.coef_
        b[j] = en.intercept_
    return W, b


def _train_mlp(X, Y, device="cuda", hidden=(256,),
               max_epochs=500, patience=30):
    N = X.shape[1]
    nv = max(int(X.shape[0] * 0.15), 1)
    nf = X.shape[0] - nv
    Xf = torch.tensor(X[:nf], dtype=torch.float32, device=device)
    Xv = torch.tensor(X[nf:], dtype=torch.float32, device=device)
    Yf = torch.tensor(Y[:nf], dtype=torch.float32, device=device)
    Yv = torch.tensor(Y[nf:], dtype=torch.float32, device=device)

    m = JointMLP(N, hidden).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    best_vl, best_st, wait = float("inf"), None, 0
    for ep in range(max_epochs):
        m.train()
        perm = torch.randperm(nf, device=device)
        for bs in range(0, nf, 256):
            idx = perm[bs:bs + 256]
            loss = nn.functional.mse_loss(m(Xf[idx]), Yf[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(m(Xv), Yv).item()
        if vl < best_vl - 1e-6:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_st:
        m.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    m.eval()
    return m


# ═════════════════════════════════════════════════════════════════════
#  Windowed LOO
# ═════════════════════════════════════════════════════════════════════

def _loo_w_linear(W, b, u_test, ni, ws=50):
    T = u_test.shape[0]
    pred = np.zeros(T, np.float64)
    pred[0] = u_test[0, ni]
    for s in range(0, T, ws):
        e = min(s + ws, T)
        pred[s] = u_test[s, ni]
        for t in range(s, e - 1):
            x = u_test[t].copy(); x[ni] = pred[t]
            pred[t + 1] = x @ W[ni] + b[ni]  # row ni of W = weights for output ni
    return pred


def _loo_w_sklearn(model, u_test, ni, ws=50):
    return _loo_w_linear(model.coef_, model.intercept_, u_test, ni, ws)


def _loo_w_mlp(model, u_test, ni, device="cuda", ws=50):
    T = u_test.shape[0]
    pred = np.zeros(T, np.float32)
    pred[0] = u_test[0, ni]
    model.eval()
    with torch.no_grad():
        for s in range(0, T, ws):
            e = min(s + ws, T)
            pred[s] = u_test[s, ni]
            for t in range(s, e - 1):
                x = u_test[t].copy(); x[ni] = pred[t]
                xt = torch.tensor(x, dtype=torch.float32,
                                  device=device).unsqueeze(0)
                pred[t + 1] = model(xt)[0, ni].cpu().item()
    return pred


# ═════════════════════════════════════════════════════════════════════
#  Run all baselines for one worm
# ═════════════════════════════════════════════════════════════════════

def run_baselines(
    u: np.ndarray,
    loo_neurons: list[int],
    device: str = "cuda",
    window_size: int = 50,
) -> Dict[str, Dict]:
    """Train Ridge-CV, EN(0.1,0.1), MLP-256 with 2-fold CV + LOO."""
    T, N = u.shape
    mid = T // 2 + 1
    folds = [(mid, T, 0, mid), (0, mid, mid, T)]  # (tr_s, tr_e, te_s, te_e)

    MODEL_NAMES = ["Ridge-CV", "EN(0.1,0.1)", "MLP-256"]
    cv_os = {m: np.full((T, N), np.nan) for m in MODEL_NAMES}
    cv_loo = {m: np.full((T, N), np.nan) for m in MODEL_NAMES}

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        print(f"  Fold {fi + 1}/2  train=[{tr_s},{tr_e})  test=[{te_s},{te_e})")
        u_tr, u_te = u[tr_s:tr_e], u[te_s:te_e]
        X_tr, Y_tr = u_tr[:-1], u_tr[1:]
        X_te = u_te[:-1]

        # ── Ridge-CV ──
        t0 = time.time()
        ridge = _train_ridge_cv(X_tr, Y_tr)
        cv_os["Ridge-CV"][te_s + 1:te_e] = ridge.predict(X_te)
        for i in loo_neurons:
            cv_loo["Ridge-CV"][te_s:te_e, i] = _loo_w_sklearn(
                ridge, u_te, i, window_size)
        print(f"    Ridge-CV done ({time.time() - t0:.1f}s)")

        # ── EN(0.1, 0.1) ──
        t0 = time.time()
        W_en, b_en = _train_en(X_tr, Y_tr)
        cv_os["EN(0.1,0.1)"][te_s + 1:te_e] = X_te @ W_en.T + b_en
        for i in loo_neurons:
            cv_loo["EN(0.1,0.1)"][te_s:te_e, i] = _loo_w_linear(
                W_en, b_en, u_te, i, window_size)
        print(f"    EN done ({time.time() - t0:.1f}s)")

        # ── MLP-256 ──
        t0 = time.time()
        mlp = _train_mlp(X_tr, Y_tr, device=device, hidden=(256,))
        with torch.no_grad():
            Xt = torch.tensor(X_te, dtype=torch.float32, device=device)
            cv_os["MLP-256"][te_s + 1:te_e] = mlp(Xt).cpu().numpy()
        for i in loo_neurons:
            cv_loo["MLP-256"][te_s:te_e, i] = _loo_w_mlp(
                mlp, u_te, i, device, window_size)
        del mlp; torch.cuda.empty_cache()
        print(f"    MLP-256 done ({time.time() - t0:.1f}s)")

    # Aggregate
    results = {}
    for mname in MODEL_NAMES:
        os_r2 = np.array([_r2(u[:, i],
                               cv_os[mname][np.isfinite(cv_os[mname][:, i]), i]
                               if False else cv_os[mname][:, i])
                          for i in range(N)])
        # Fix: compute properly
        os_r2 = np.full(N, np.nan)
        for i in range(N):
            m = np.isfinite(cv_os[mname][:, i])
            if m.sum() > 3:
                os_r2[i] = _r2(u[m, i], cv_os[mname][m, i])

        loo_r2 = np.full(N, np.nan)
        for i in loo_neurons:
            m = np.isfinite(cv_loo[mname][:, i])
            if m.sum() > 3:
                loo_r2[i] = _r2(u[m, i], cv_loo[mname][m, i])

        loo_vals = loo_r2[loo_neurons]
        results[mname] = {
            "mean_onestep": float(np.nanmean(os_r2)),
            "mean_loo_w": float(np.nanmean(loo_vals)),
            "median_loo_w": float(np.nanmedian(loo_vals)),
            "per_neuron_onestep": os_r2,
            "per_neuron_loo_w": loo_r2,
        }
        print(f"  {mname:20s}  1step={results[mname]['mean_onestep']:.4f}  "
              f"LOO_w={results[mname]['mean_loo_w']:.4f}")

    return results


# ═════════════════════════════════════════════════════════════════════
#  Comparison plot
# ═════════════════════════════════════════════════════════════════════

def plot_comparison(results: Dict[str, Dict], worm_name: str,
                    save_path: Path, loo_neurons: list[int]):
    """Bar chart: one-step R² and LOO_w R² per model."""

    models = list(results.keys())
    n = len(models)

    colors = {
        "Ridge-CV":      "#1f77b4",
        "EN(0.1,0.1)":   "#2ca02c",
        "MLP-256":       "#ff7f0e",
        "Stage2 (P1)":   "#7f7f7f",
        "Stage2 (P1+P2)":"#d62728",
    }

    fig, axes = plt.subplots(2, 1, figsize=(max(8, 1.8 * n), 9))
    x = np.arange(n)
    width = 0.6

    for ax_i, (key, ylabel, title) in enumerate([
        ("mean_onestep", "Mean one-step R²",
         f"One-step R² — {worm_name}"),
        ("mean_loo_w", "Mean LOO R² (windowed, w=50)",
         f"LOO R² (windowed) — {worm_name}"),
    ]):
        ax = axes[ax_i]
        vals = [results[m][key] for m in models]
        bars = ax.bar(x, vals, width,
                      color=[colors.get(m, "#999") for m in models],
                      edgecolor="black", linewidth=0.5, alpha=0.85)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                va = "bottom" if v >= 0 else "top"
                offset = 0.01 if v >= 0 else -0.01
                ax.text(xi, v + offset, f"{v:.3f}", ha="center", va=va,
                        fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_ylim(min(0, min(v for v in vals if np.isfinite(v)) - 0.1),
                    max(v for v in vals if np.isfinite(v)) + 0.12)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")

    # ── Per-neuron scatter: Stage2 (P1+P2) vs baselines ──────────────
    ref = "Stage2 (P1+P2)" if "Stage2 (P1+P2)" in results else "Stage2 (P1)"
    compare = [m for m in models if m != ref and m != "Stage2 (P1)"]
    if ref in results and compare:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        ref_os = results[ref]["per_neuron_onestep"]
        ref_loo = results[ref]["per_neuron_loo_w"]

        for ax, metric_arr, title in [
            (axes2[0], "per_neuron_onestep", "One-step R²"),
            (axes2[1], "per_neuron_loo_w", "LOO R² (windowed)"),
        ]:
            ref_v = results[ref][metric_arr]
            neurons = (list(range(len(ref_v)))
                       if metric_arr == "per_neuron_onestep"
                       else loo_neurons)
            for cm in compare:
                cm_v = results[cm][metric_arr]
                r = np.array([ref_v[i] for i in neurons])
                c = np.array([cm_v[i] for i in neurons])
                ok = np.isfinite(r) & np.isfinite(c)
                ax.scatter(r[ok], c[ok], s=12, alpha=0.5, label=cm,
                           color=colors.get(cm))
            lims = [-0.3, 1.0]
            ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel(f"{ref} R²")
            ax.set_ylabel("Baseline R²")
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=8)
        plt.suptitle(f"Per-neuron comparison — {worm_name}", fontweight="bold")
        plt.tight_layout()
        scatter_path = save_path.parent / "perneuron_scatter.png"
        fig2.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved {scatter_path}")


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage2 + baselines integrated comparison")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--device", default="cuda")
    # Phase 1
    parser.add_argument("--phase1_epochs", type=int, default=150)
    # Phase 2
    parser.add_argument("--phase2_epochs", type=int, default=50)
    parser.add_argument("--rollout_weight", type=float, default=0.3)
    parser.add_argument("--noise_sigma", type=float, default=0.05)
    parser.add_argument("--K_start", type=int, default=5)
    parser.add_argument("--K_end", type=int, default=20)
    parser.add_argument("--rollout_starts", type=int, default=4)
    # Common
    parser.add_argument("--cv_folds", type=int, default=2)
    parser.add_argument("--loo_subset", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    worm_name = Path(args.h5).stem

    # Load data for baselines
    with h5py.File(args.h5, "r") as f:
        u = f["stage1/u_mean"][:]
    T, N = u.shape
    print(f"\n{'═' * 70}")
    print(f"  {worm_name}  T={T}  N={N}")
    print(f"{'═' * 70}")

    var = np.var(u, axis=0)
    loo_neurons = sorted(np.argsort(var)[::-1][:args.loo_subset].tolist())

    t_total = time.time()

    # ══════════════════════════════════════════════════════════════════
    #  Step 1: Baselines (fast — ~3 min)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  BASELINES: Ridge-CV, EN(0.1,0.1), MLP-256")
    print(f"{'─' * 70}")
    t0 = time.time()
    baseline_results = run_baselines(
        u, loo_neurons, device=args.device, window_size=50)
    t_base = time.time() - t0
    print(f"  Baselines done in {t_base:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    #  Step 2: Stage2 Phase 1 — one-step training
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  STAGE2 PHASE 1: one-step training ({args.phase1_epochs} epochs)")
    print(f"{'─' * 70}")
    t0 = time.time()

    cfg1 = make_config(
        args.h5,
        device=args.device,
        num_epochs=args.phase1_epochs,
        learning_rate=args.lr,
        cv_folds=args.cv_folds,
        rollout_weight=0.0,
        input_noise_sigma=0.0,
        skip_final_eval=True,
        eval_loo_subset_neurons=tuple(loo_neurons),
        eval_loo_subset_size=len(loo_neurons),
    )
    p1_dir = str(save / "phase1")
    result1 = train_stage2_cv(cfg1, save_dir=p1_dir, show=False)
    t_p1 = time.time() - t0

    p1_os = result1.get("cv_onestep_r2_mean", float("nan"))
    p1_loo = result1.get("cv_loo_r2_windowed_mean", float("nan"))
    print(f"  Phase 1: 1step={p1_os:.4f}  LOO_w={p1_loo:.4f}  ({t_p1:.0f}s)")

    # ══════════════════════════════════════════════════════════════════
    #  Step 3: Stage2 Phase 2 — rollout + noise + curriculum
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  STAGE2 PHASE 2: rollout+noise ({args.phase2_epochs} epochs)")
    print(f"  noise_σ={args.noise_sigma}  rollout_w={args.rollout_weight}  "
          f"K={args.K_start}→{args.K_end}")
    print(f"{'─' * 70}")
    t0 = time.time()

    lr2 = args.lr / 3.0
    cfg2 = make_config(
        args.h5,
        device=args.device,
        num_epochs=args.phase2_epochs,
        learning_rate=lr2,
        cv_folds=args.cv_folds,
        # Rollout ON
        rollout_weight=args.rollout_weight,
        rollout_steps=args.K_end,
        rollout_starts=args.rollout_starts,
        warmstart_rollout=True,
        # Noise ON
        input_noise_sigma=args.noise_sigma,
        # Curriculum ON
        rollout_curriculum=True,
        rollout_K_start=args.K_start,
        rollout_K_end=args.K_end,
        rollout_curriculum_start_epoch=0,
        rollout_curriculum_end_epoch=args.phase2_epochs,
        # LOO on same neurons as baselines, skip expensive final eval
        skip_final_eval=True,
        eval_loo_subset_neurons=tuple(loo_neurons),
        eval_loo_subset_size=len(loo_neurons),
    )
    p2_dir = str(save / "phase2")
    result2 = train_stage2_cv(cfg2, save_dir=p2_dir, show=False,
                              warm_start_dir=p1_dir)
    t_p2 = time.time() - t0

    p2_os = result2.get("cv_onestep_r2_mean", float("nan"))
    p2_loo = result2.get("cv_loo_r2_windowed_mean", float("nan"))
    print(f"  Phase 2: 1step={p2_os:.4f}  LOO_w={p2_loo:.4f}  ({t_p2:.0f}s)")

    # ══════════════════════════════════════════════════════════════════
    #  Collect results & plot
    # ══════════════════════════════════════════════════════════════════
    cv_r2_p1 = result1.get("cv_onestep_r2", np.full(N, np.nan))
    cv_loo_w_p1 = result1.get("cv_loo_r2_windowed", np.full(N, np.nan))
    cv_r2_p2 = result2.get("cv_onestep_r2", np.full(N, np.nan))
    cv_loo_w_p2 = result2.get("cv_loo_r2_windowed", np.full(N, np.nan))

    all_results = {}
    # Baselines
    for mname, res in baseline_results.items():
        all_results[mname] = res

    # Stage2 Phase 1
    loo_p1 = cv_loo_w_p1[loo_neurons] if cv_loo_w_p1 is not None else np.full(len(loo_neurons), np.nan)
    all_results["Stage2 (P1)"] = {
        "mean_onestep": float(np.nanmean(cv_r2_p1)),
        "mean_loo_w": float(np.nanmean(loo_p1)),
        "median_loo_w": float(np.nanmedian(loo_p1)),
        "per_neuron_onestep": cv_r2_p1,
        "per_neuron_loo_w": cv_loo_w_p1 if cv_loo_w_p1 is not None else np.full(N, np.nan),
    }

    # Stage2 Phase 1+2
    loo_p2 = cv_loo_w_p2[loo_neurons] if cv_loo_w_p2 is not None else np.full(len(loo_neurons), np.nan)
    all_results["Stage2 (P1+P2)"] = {
        "mean_onestep": float(np.nanmean(cv_r2_p2)),
        "mean_loo_w": float(np.nanmean(loo_p2)),
        "median_loo_w": float(np.nanmedian(loo_p2)),
        "per_neuron_onestep": cv_r2_p2,
        "per_neuron_loo_w": cv_loo_w_p2 if cv_loo_w_p2 is not None else np.full(N, np.nan),
    }

    # ── Summary ──
    t_all = time.time() - t_total
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY — {worm_name}  (total {t_all:.0f}s = {t_all/60:.1f}min)")
    print(f"{'═' * 70}")
    print(f"  {'Model':<20s}  {'1step':>8s}  {'LOO_w':>8s}  {'LOO_w med':>10s}")
    print(f"  {'─' * 50}")
    for mname, res in all_results.items():
        print(f"  {mname:<20s}  {res['mean_onestep']:8.4f}  "
              f"{res['mean_loo_w']:8.4f}  {res['median_loo_w']:10.4f}")

    print(f"\n  Phase 2 vs Phase 1:")
    print(f"    one-step Δ = {p2_os - p1_os:+.4f}")
    print(f"    LOO_w Δ    = {p2_loo - p1_loo:+.4f}")
    print(f"{'═' * 70}")

    # Save JSON (convert numpy arrays to lists)
    json_results = {}
    for mname, res in all_results.items():
        jr = {}
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                jr[k] = v.tolist()
            else:
                jr[k] = v
        json_results[mname] = jr
    json_results["_meta"] = {
        "worm": worm_name, "h5": args.h5,
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "rollout_weight": args.rollout_weight,
        "noise_sigma": args.noise_sigma,
        "K_start": args.K_start, "K_end": args.K_end,
        "total_time_s": t_all,
        "baseline_time_s": t_base,
        "phase1_time_s": t_p1,
        "phase2_time_s": t_p2,
    }
    with open(save / "comparison_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Plot
    plot_comparison(all_results, worm_name, save / "comparison_bars.png",
                    loo_neurons)


if __name__ == "__main__":
    main()
