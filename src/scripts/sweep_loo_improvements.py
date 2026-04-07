#!/usr/bin/env python
"""
Sweep Stage2 config improvements targeting LOO R².

Trains Stage2 with multiple config variants, combines with baseline
results from model_distribution_comparison.py, and generates a
strip+box comparison plot.

Variants tested:
  1. Stage2 (baseline)           – existing npz, no retraining
  2. Stage2 + loo_aux            – LOO auxiliary loss during training
  3. Stage2 + lowrank            – low-rank dense coupling (non-connectome)
  4. Stage2 + loo_aux + lowrank  – both combined
  5. Stage2 + graph_poly=2       – multi-hop gap-junction propagation

Usage:
    python -m scripts.sweep_loo_improvements \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
        --baseline_json output_plots/stage2/model_distributions/results.json \
        --stage2_npz output_plots/stage2/default_config_run_v2/cv_onestep.npz \
        --save_dir output_plots/stage2/loo_improvement_sweep \
        --epochs 50 --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage2.config import make_config
from stage2.train import train_stage2_cv


# ═══════════════════════════════════════════════════════════════════════
#  Config variants to sweep
# ═══════════════════════════════════════════════════════════════════════

def _make_variants(args) -> Dict[str, Dict]:
    """Return {label: config_overrides} for each variant."""
    common = dict(
        device=args.device,
        num_epochs=args.epochs,
        learning_rate=1e-3,
        cv_folds=2,
        eval_loo_subset_size=30,
        eval_loo_subset_mode="variance",
        skip_final_eval=True,
    )

    variants = {}

    # 2. LOO auxiliary loss
    variants["+ loo_aux"] = {
        **common,
        "loo_aux_weight": 0.5,
        "loo_aux_steps": 5,
        "loo_aux_neurons": 3,
        "loo_aux_starts": 1,
        "warmstart_rollout": True,
    }

    # 3. Low-rank dense coupling
    variants["+ lowrank"] = {
        **common,
        "lowrank_rank": 4,
    }

    # 4. Both combined
    variants["+ loo+lr"] = {
        **common,
        "loo_aux_weight": 0.5,
        "loo_aux_steps": 5,
        "loo_aux_neurons": 3,
        "loo_aux_starts": 1,
        "warmstart_rollout": True,
        "lowrank_rank": 4,
    }

    # 5. Graph polynomial order 2
    variants["+ poly2"] = {
        **common,
        "graph_poly_order": 2,
    }

    return variants


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


# ═══════════════════════════════════════════════════════════════════════
#  Plot
# ═══════════════════════════════════════════════════════════════════════

# Colors for models
BASELINE_COLORS = {
    "EN":         "#228b22",
    "MLP":        "#ff7f0e",
    "Conn-Ridge": "#1e90ff",
}

STAGE2_COLORS = [
    "#d62728",   # Stage2 (baseline) – red
    "#e377c2",   # + loo_aux – pink
    "#17becf",   # + lowrank – teal
    "#9467bd",   # + loo+lr – purple
    "#8c564b",   # + poly2 – brown
]


def plot_comparison(all_results: Dict, save_path: Path, worm_name: str,
                    n_loo: int, stage2_order: List[str],
                    baseline_order: List[str]):
    """Strip + box plot: baselines + Stage2 variants."""

    # Order: baselines first, then Stage2 variants
    ordered = baseline_order + stage2_order
    n = len(ordered)

    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(max(16, 1.6 * n), 6.5))

    for ax, key, ylabel, title, ylims in [
        (ax_b, "onestep_r2", "One-step R²",
         f"B.  One-step R²  (all neurons)", (-0.3, 1.05)),
        (ax_c, "loo_r2_w", "LOO R² (windowed, w=50)",
         f"C.  LOO R²  ({n_loo} neurons, w=50)", (-0.8, 1.05)),
    ]:
        positions = np.arange(len(ordered))
        bp_data = []

        for xi, name in enumerate(ordered):
            vals = all_results[name][key]
            v = vals[np.isfinite(vals)]
            bp_data.append(v)
            if len(v) == 0:
                continue

            # Pick color
            if name in BASELINE_COLORS:
                c = BASELINE_COLORS[name]
            elif name in stage2_order:
                idx = stage2_order.index(name)
                c = STAGE2_COLORS[idx % len(STAGE2_COLORS)]
            else:
                c = "#888"

            jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(v))
            ax.scatter(
                xi + jitter, v, s=18, alpha=0.55, color=c,
                edgecolors="white", linewidths=0.3, zorder=3,
            )

        # Box plot
        bp = ax.boxplot(
            bp_data, positions=positions, widths=0.45,
            patch_artist=True, showfliers=False, zorder=4,
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(color="gray", linewidth=1.0),
            capprops=dict(color="gray", linewidth=1.0),
        )
        for patch, name in zip(bp["boxes"], ordered):
            if name in BASELINE_COLORS:
                c = BASELINE_COLORS[name]
            elif name in stage2_order:
                idx = stage2_order.index(name)
                c = STAGE2_COLORS[idx % len(STAGE2_COLORS)]
            else:
                c = "#888"
            patch.set_facecolor(c)
            patch.set_alpha(0.25)
            patch.set_edgecolor(c)
            patch.set_linewidth(1.2)

        # Mean markers
        for xi, (name, v) in enumerate(zip(ordered, bp_data)):
            if len(v) == 0:
                continue
            mu = float(np.mean(v))
            if name in BASELINE_COLORS:
                c = BASELINE_COLORS[name]
            elif name in stage2_order:
                idx = stage2_order.index(name)
                c = STAGE2_COLORS[idx % len(STAGE2_COLORS)]
            else:
                c = "#888"
            ax.plot(xi, mu, marker="D", color=c,
                    markersize=7, markeredgecolor="black",
                    markeredgewidth=0.8, zorder=5)
            ax.annotate(
                f"{mu:.3f}", (xi, mu), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8, fontweight="bold",
                color=c,
            )

        # Separator line between baselines and Stage2 variants
        if baseline_order and stage2_order:
            sep_x = len(baseline_order) - 0.5
            ax.axvline(sep_x, color="gray", lw=1.0, ls=":", alpha=0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(ordered, fontsize=9, fontweight="bold", rotation=15,
                           ha="right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.set_ylim(ylims)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.suptitle(
        f"LOO Improvement Sweep — {worm_name}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Sweep Stage2 config variants for LOO R² improvement")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--baseline_json", default=None,
                    help="Path to results.json from model_distribution_comparison.py")
    ap.add_argument("--stage2_npz", default=None,
                    help="Path to existing Stage2 cv_onestep.npz (baseline)")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    worm = Path(args.h5).stem

    # ── Load data for neuron count + LOO selection ────────────────────
    with h5py.File(args.h5, "r") as f:
        u = f["stage1/u_mean"][:]
    T, N = u.shape
    print(f"  {worm}  T={T}  N={N}")

    var = np.var(u, axis=0)
    loo_neurons = sorted(np.argsort(var)[::-1][:30].tolist())

    # ── Collect results ───────────────────────────────────────────────
    all_results: Dict[str, Dict] = {}
    baseline_order = []
    stage2_order = []

    # ── Load baselines from previous run ──────────────────────────────
    if args.baseline_json and Path(args.baseline_json).exists():
        with open(args.baseline_json) as f:
            prev = json.load(f)
        # Only keep the 3 most relevant baselines
        for bname in ["EN", "MLP", "Conn-Ridge"]:
            if bname in prev:
                all_results[bname] = {
                    "onestep_r2": np.array(prev[bname]["onestep_r2"]),
                    "loo_r2_w": np.array(prev[bname]["loo_r2_w"]),
                }
                baseline_order.append(bname)
        print(f"  Loaded baselines: {baseline_order}")
    else:
        print("  No baseline_json provided — baselines omitted.")

    # ── Stage2 baseline (from existing npz) ───────────────────────────
    if args.stage2_npz and Path(args.stage2_npz).exists():
        s2 = np.load(args.stage2_npz, allow_pickle=True)
        s2_os = s2["cv_r2"]
        s2_loo = s2.get("cv_loo_r2_windowed", np.full(N, np.nan))
        all_results["Stage2"] = {
            "onestep_r2": s2_os,
            "loo_r2_w": s2_loo[loo_neurons],
        }
        stage2_order.append("Stage2")
        print(f"  Stage2 baseline: 1step={np.nanmean(s2_os):.4f}  "
              f"LOO_w={np.nanmean(s2_loo[loo_neurons]):.4f}")
    else:
        print("  No stage2_npz — training baseline from scratch.")
        common = dict(
            device=args.device, num_epochs=args.epochs, learning_rate=1e-3,
            cv_folds=2, eval_loo_subset_size=30, eval_loo_subset_mode="variance",
            skip_final_eval=True,
        )
        cfg0 = make_config(args.h5, **common)
        res0 = train_stage2_cv(cfg0, save_dir=str(save / "baseline"), show=False)
        os0 = res0.get("cv_onestep_r2", np.full(N, np.nan))
        loo0 = res0.get("cv_loo_r2_windowed", np.full(N, np.nan))
        all_results["Stage2"] = {
            "onestep_r2": os0,
            "loo_r2_w": loo0[loo_neurons],
        }
        stage2_order.append("Stage2")
        print(f"  Stage2 baseline: 1step={np.nanmean(os0):.4f}  "
              f"LOO_w={np.nanmean(loo0[loo_neurons]):.4f}")

    # ── Train Stage2 variants ─────────────────────────────────────────
    variants = _make_variants(args)
    t_total = time.time()

    for label, overrides in variants.items():
        full_label = f"Stage2 {label}"
        vdir = str(save / label.replace(" ", "_").replace("+", "").strip("_"))
        print(f"\n{'─'*60}")
        print(f"  Training: {full_label}  ({args.epochs} epochs)")
        print(f"  Config:   {overrides}")
        print(f"{'─'*60}")

        t0 = time.time()
        cfg = make_config(args.h5, **overrides)
        res = train_stage2_cv(cfg, save_dir=vdir, show=False)
        elapsed = time.time() - t0

        os_r2 = res.get("cv_onestep_r2", np.full(N, np.nan))
        loo_r2 = res.get("cv_loo_r2_windowed", np.full(N, np.nan))

        all_results[full_label] = {
            "onestep_r2": os_r2,
            "loo_r2_w": loo_r2[loo_neurons],
        }
        stage2_order.append(full_label)

        os_m = float(np.nanmean(os_r2))
        loo_m = float(np.nanmean(loo_r2[loo_neurons]))
        print(f"  → {full_label}  1step={os_m:.4f}  LOO_w={loo_m:.4f}  "
              f"({elapsed:.0f}s)")

    t_all = time.time() - t_total
    print(f"\n  Total sweep: {t_all:.0f}s ({t_all/60:.1f}min)")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  {'Model':25s}  {'1step mean':>10s}  {'LOO_w mean':>10s}  "
          f"{'LOO_w med':>10s}")
    print(f"  {'─'*60}")
    for name in baseline_order + stage2_order:
        r = all_results[name]
        os_v = r["onestep_r2"]
        loo_v = r["loo_r2_w"]
        print(f"  {name:25s}  {np.nanmean(os_v):10.4f}  "
              f"{np.nanmean(loo_v):10.4f}  {np.nanmedian(loo_v):10.4f}")
    print(f"{'═'*65}")

    # ── Save ──────────────────────────────────────────────────────────
    json_out = {}
    for name, r in all_results.items():
        json_out[name] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in r.items()
        }
    json_out["_meta"] = {
        "worm": worm, "h5": args.h5, "epochs": args.epochs,
        "loo_neurons": loo_neurons, "n_loo": len(loo_neurons),
        "time_s": t_all,
        "baseline_order": baseline_order, "stage2_order": stage2_order,
    }
    with open(save / "sweep_results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  Saved {save / 'sweep_results.json'}")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_comparison(
        all_results, save / "loo_sweep_comparison.png", worm,
        len(loo_neurons), stage2_order, baseline_order,
    )


if __name__ == "__main__":
    main()
