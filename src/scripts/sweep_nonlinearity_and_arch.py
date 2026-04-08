#!/usr/bin/env python
"""
Sweep: nonlinearities + architectural enhancements to improve Stage2 above MLP.

Bottleneck diagnosis
────────────────────
Stage2 LOO R² ≈ 0.14  vs  MLP LOO R² ≈ 0.25+.
The tau kernel sweep showed temporal memory isn't the bottleneck.
The gap is likely: (1) rigid activation function, (2) no capacity for
non-connectome interactions, (3) missing neurons (~170/302 unobserved).

This sweep tests:
  A. Nonlinearities — φ(u) in the synaptic current equation
     sigmoid (default), tanh, softplus, relu, elu, swish, shifted_sigmoid, identity
  C. Low-rank coupling — V @ tanh(U @ u) (parameterised non-connectome interactions)
  D. Learned reversals — per-neuron E_sv instead of fixed
  E. Noise correlation rank — low-rank correlated noise (shared unmodeled inputs)
  F. Input noise — regularisation to prevent overfitting
  G. Combos — best nonlinearity + residual MLP + low-rank

Usage
─────
    python -u -m scripts.sweep_nonlinearity_and_arch \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
        --out output_plots/stage2/nonlinearity_arch_sweep \
        --epochs 30 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════
#  Sweep conditions
# ═══════════════════════════════════════════════════════════════════════

def build_conditions(epochs: int = 30) -> OrderedDict:
    """Build sweep conditions: name → config overrides dict."""
    C = OrderedDict()

    # ─────────── GROUP A: Nonlinearities ───────────
    # Baseline: current default sigmoid
    C["A0_sigmoid"] = dict(
        chemical_synapse_activation="sigmoid",
    )
    C["A1_tanh"] = dict(
        chemical_synapse_activation="tanh",
    )
    C["A2_softplus"] = dict(
        chemical_synapse_activation="softplus",
    )
    C["A3_relu"] = dict(
        chemical_synapse_activation="relu",
    )
    C["A4_elu"] = dict(
        chemical_synapse_activation="elu",
    )
    C["A5_swish"] = dict(
        chemical_synapse_activation="swish",
    )
    C["A6_shifted_sigmoid"] = dict(
        chemical_synapse_activation="shifted_sigmoid",
    )
    C["A7_identity"] = dict(
        chemical_synapse_activation="identity",
    )

    # ─────────── GROUP C: Low-rank coupling ───────────
    C["C1_lowrank_r5"] = dict(
        lowrank_rank=5,
    )
    C["C2_lowrank_r10"] = dict(
        lowrank_rank=10,
    )
    C["C3_lowrank_r20"] = dict(
        lowrank_rank=20,
    )

    # ─────────── GROUP D: Learned reversals ───────────
    C["D1_learn_rev"] = dict(
        learn_reversals=True,
        reversal_mode="per_neuron",
    )

    # ─────────── GROUP E: Low-rank correlated noise ───────────
    C["E1_noise_rank5"] = dict(
        noise_corr_rank=5,
    )
    C["E2_noise_rank10"] = dict(
        noise_corr_rank=10,
    )

    # ─────────── GROUP F: Input noise regularisation ───────────
    C["F1_input_noise_0.05"] = dict(
        input_noise_sigma=0.05,
    )
    C["F2_input_noise_0.1"] = dict(
        input_noise_sigma=0.1,
    )

    # ─────────── GROUP G: Rollout training ───────────
    C["G1_rollout"] = dict(
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
    )

    # ─────────── GROUP H: Best combos ───────────
    # Combo: residual MLP + low-rank
    C["H3_mlp64_lr10"] = dict(
        lowrank_rank=10,
    )
    # Combo: residual MLP + rollout
    C["H4_mlp64_rollout"] = dict(
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
    )
    # Combo: tanh + mlp + low-rank + rollout (kitchen sink)
    C["H5_tanh_mlp64_lr10_ro"] = dict(
        chemical_synapse_activation="tanh",
        lowrank_rank=10,
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
    )
    # Combo: noise rank + residual MLP
    C["H7_noise5_mlp64"] = dict(
        noise_corr_rank=5,
    )

    return C


# ═══════════════════════════════════════════════════════════════════════
#  Run one condition
# ═══════════════════════════════════════════════════════════════════════

def run_one(
    h5_path: str,
    cond_name: str,
    overrides: dict,
    *,
    save_dir: str,
    epochs: int,
    device: str,
    cv_folds: int,
    loo_subset: int,
) -> dict:
    """Train Stage2 with given overrides, return summary dict."""
    from stage2.config import make_config
    from stage2.train import train_stage2_cv

    kw = dict(overrides)
    kw.update(
        num_epochs=epochs,
        device=device,
        cv_folds=cv_folds,
        # Standard dynamics defaults (match current best)
        learn_lambda_u=True,
        learn_I0=True,
        edge_specific_G=True,
        learn_W_sv=True,
        learn_W_dcv=True,
        learn_noise=True,
        noise_mode="heteroscedastic",
        coupling_gate=True,
        graph_poly_order=2,
        per_neuron_amplitudes=True,
        # LOO evaluation
        eval_loo_subset_size=loo_subset,
        eval_loo_subset_mode="variance",
        eval_loo_window_size=50,
        eval_loo_warmup_steps=40,
        # Speed: skip final eval plots
        skip_final_eval=True,
    )

    cfg = make_config(h5_path, **kw)

    t0 = time.time()
    result = train_stage2_cv(cfg, save_dir=save_dir, show=False)
    elapsed = time.time() - t0

    summary: dict = {
        "name": cond_name,
        "overrides": {k: str(v) for k, v in overrides.items()},
        "time_s": round(elapsed, 1),
    }

    # Extract metrics from npz
    cv_path = Path(save_dir) / "cv_onestep.npz"
    if cv_path.exists():
        d = np.load(cv_path, allow_pickle=True)
        if "cv_r2" in d:
            cv_r2 = d["cv_r2"]
            summary["os_mean"] = float(np.nanmean(cv_r2))
            summary["os_median"] = float(np.nanmedian(cv_r2))
        if "cv_loo_r2" in d:
            loo = d["cv_loo_r2"]
            summary["loo_mean"] = float(np.nanmean(loo))
            summary["loo_median"] = float(np.nanmedian(loo))
            summary["loo_n_pos"] = int((loo[np.isfinite(loo)] > 0).sum())
        if "cv_loo_r2_windowed" in d:
            loo_w = d["cv_loo_r2_windowed"]
            summary["loo_w_mean"] = float(np.nanmean(loo_w))
            summary["loo_w_median"] = float(np.nanmedian(loo_w))
            summary["loo_w_n_pos"] = int((loo_w[np.isfinite(loo_w)] > 0).sum())

    return summary


# ═══════════════════════════════════════════════════════════════════════
#  Summary + plotting
# ═══════════════════════════════════════════════════════════════════════

def print_summary(results: List[dict]):
    """Print formatted summary table sorted by LOO windowed R²."""
    print(f"\n{'='*110}")
    print(f"  NONLINEARITY + ARCHITECTURE SWEEP SUMMARY")
    print(f"{'='*110}")

    hdr = (f"{'Condition':<30s} {'OS_mean':>7s} {'LOO_mean':>8s} "
           f"{'LOOw_mean':>9s} {'LOOw_med':>9s} "
           f"{'#pos':>5s} {'Time':>6s}  Overrides")
    print(hdr)
    print("-" * 110)

    sorted_r = sorted(
        results,
        key=lambda r: r.get("loo_w_mean", float("-inf")),
        reverse=True,
    )

    for r in sorted_r:
        nm = r.get("name", "?")[:29]
        os_m = r.get("os_mean", float("nan"))
        loo_m = r.get("loo_mean", float("nan"))
        loo_w_m = r.get("loo_w_mean", float("nan"))
        loo_w_med = r.get("loo_w_median", float("nan"))
        n_pos = r.get("loo_w_n_pos", -1)
        t_s = r.get("time_s", 0)
        ov = ", ".join(f"{k}={v}" for k, v in r.get("overrides", {}).items())
        if len(ov) > 45:
            ov = ov[:42] + "..."

        print(f"{nm:<30s} {os_m:>7.4f} {loo_m:>8.4f} "
              f"{loo_w_m:>9.4f} {loo_w_med:>9.4f} "
              f"{n_pos:>5d} {t_s:>5.0f}s  {ov}")

    if sorted_r:
        best = sorted_r[0]
        base = next((r for r in sorted_r if r["name"] == "A0_sigmoid"), sorted_r[-1])
        delta = best.get("loo_w_mean", 0) - base.get("loo_w_mean", 0)
        print(f"\n  ★ BEST: {best['name']}  LOO_w={best.get('loo_w_mean', 0):.4f}"
              f"  (Δ={delta:+.4f} vs sigmoid baseline)")
        print(f"    Overrides: {best.get('overrides', {})}")
    print()


def plot_sweep_results(results: List[dict], save_dir: Path):
    """Create comparison bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_r = sorted(
        results,
        key=lambda r: r.get("loo_w_mean", float("-inf")),
        reverse=True,
    )

    names = [r["name"] for r in sorted_r]
    loo_w = [r.get("loo_w_mean", 0) for r in sorted_r]
    os_m = [r.get("os_mean", 0) for r in sorted_r]

    # Color by group
    def group_color(name: str) -> str:
        if name.startswith("A"): return "#4682b4"  # blue  - nonlinearities
        if name.startswith("B"): return "#d62728"  # red   - residual MLP
        if name.startswith("C"): return "#ff7f0e"  # orange - low-rank
        if name.startswith("D"): return "#9467bd"  # purple - reversals
        if name.startswith("E"): return "#8c564b"  # brown  - noise rank
        if name.startswith("F"): return "#e377c2"  # pink   - input noise
        if name.startswith("G"): return "#7f7f7f"  # grey   - rollout
        if name.startswith("H"): return "#2ca02c"  # green  - combos
        return "#17becf"

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Panel 1: LOO windowed R²
    ax = axes[0]
    colors = [group_color(n) for n in names]
    # Highlight the best
    best_val = max(loo_w)
    colors_hl = ["#FFD700" if v == best_val else c for v, c in zip(loo_w, colors)]
    ax.barh(range(len(names)), loo_w, color=colors_hl, alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("LOO R² (windowed, mean)", fontsize=11)
    ax.set_title("LOO Windowed R² — Nonlinearity & Architecture Sweep", fontsize=13, fontweight="bold")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.invert_yaxis()
    for i, v in enumerate(loo_w):
        ax.text(max(v + 0.003, 0.003), i, f"{v:.4f}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.3)

    # Legend for groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4682b4", label="A: Nonlinearity"),
        Patch(facecolor="#d62728", label="B: Residual MLP"),
        Patch(facecolor="#ff7f0e", label="C: Low-rank"),
        Patch(facecolor="#9467bd", label="D: Reversals"),
        Patch(facecolor="#8c564b", label="E: Noise rank"),
        Patch(facecolor="#e377c2", label="F: Input noise"),
        Patch(facecolor="#7f7f7f", label="G: Rollout"),
        Patch(facecolor="#2ca02c", label="H: Combos"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right", ncol=2)

    # Panel 2: One-step R²
    ax = axes[1]
    ax.barh(range(len(names)), os_m, color=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("One-step R² (mean)", fontsize=11)
    ax.set_title("One-step R² — Nonlinearity & Architecture Sweep", fontsize=13, fontweight="bold")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.invert_yaxis()
    for i, v in enumerate(os_m):
        ax.text(max(v + 0.003, 0.003), i, f"{v:.4f}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "nonlinearity_arch_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_dir / 'nonlinearity_arch_sweep.png'}")

    # ── Nonlinearity-only comparison ──
    nonlin_r = [r for r in sorted_r if r["name"].startswith("A")]
    if len(nonlin_r) > 1:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        nl_names = [r["name"].replace("A", "").lstrip("0123456789_") for r in nonlin_r]
        nl_loo = [r.get("loo_w_mean", 0) for r in nonlin_r]
        nl_os = [r.get("os_mean", 0) for r in nonlin_r]
        x = np.arange(len(nl_names))
        w = 0.35
        ax2.bar(x - w/2, nl_loo, w, label="LOO R² (windowed)", color="#4682b4", alpha=0.8)
        ax2.bar(x + w/2, nl_os, w, label="One-step R²", color="#ff7f0e", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(nl_names, rotation=30, ha="right", fontsize=10)
        ax2.set_ylabel("R²", fontsize=11)
        ax2.set_title("Chemical Synapse Activation: φ(u) Comparison", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)
        for i, (v1, v2) in enumerate(zip(nl_loo, nl_os)):
            ax2.text(i - w/2, v1 + 0.003, f"{v1:.4f}", ha="center", fontsize=7)
            ax2.text(i + w/2, v2 + 0.003, f"{v2:.4f}", ha="center", fontsize=7)
        fig2.tight_layout()
        fig2.savefig(save_dir / "nonlinearity_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved {save_dir / 'nonlinearity_comparison.png'}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Nonlinearity + Architecture sweep for Stage2")
    ap.add_argument("--h5", required=True, help="Path to worm .h5 file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cv_folds", type=int, default=2)
    ap.add_argument("--loo_subset", type=int, default=30)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Run only specific conditions (e.g. A0_sigmoid B1_mlp_h32)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip conditions that already have results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    worm = Path(args.h5).stem

    conditions = build_conditions(args.epochs)
    if args.conditions:
        conditions = OrderedDict(
            (k, v) for k, v in conditions.items() if k in args.conditions
        )

    # Resume support
    results_path = out / "sweep_results.json"
    all_results: List[dict] = []
    done_names: set = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        done_names = {r["name"] for r in all_results if "error" not in r}
        print(f"[resume] {len(done_names)} conditions already done — skipping them.")

    # Filter out done
    todo = OrderedDict(
        (k, v) for k, v in conditions.items() if k not in done_names
    )

    print(f"\n{'='*70}")
    print(f"  NONLINEARITY + ARCHITECTURE SWEEP")
    print(f"  Worm: {worm}")
    print(f"  Conditions: {len(conditions)} total, {len(todo)} remaining")
    print(f"  Epochs: {args.epochs}")
    print(f"  LOO subset: {args.loo_subset}")
    print(f"{'='*70}\n")

    for i, (cname, overrides) in enumerate(todo.items(), 1):
        print(f"\n{'═'*70}")
        print(f"  [{i}/{len(todo)}] {cname}")
        ov_str = ", ".join(f"{k}={v}" for k, v in overrides.items())
        print(f"  Overrides: {ov_str}")
        print(f"{'═'*70}")

        cfg_dir = str(out / cname)
        try:
            summary = run_one(
                args.h5, cname, overrides,
                save_dir=cfg_dir,
                epochs=args.epochs,
                device=args.device,
                cv_folds=args.cv_folds,
                loo_subset=args.loo_subset,
            )
            all_results.append(summary)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({"name": cname, "error": str(e), "time_s": 0})

        # Save after each condition (crash-safe)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final summary
    valid = [r for r in all_results if "error" not in r]
    if valid:
        print_summary(valid)
        plot_sweep_results(valid, out)

    total = sum(r.get("time_s", 0) for r in all_results)
    print(f"\n  Total time: {total:.0f}s ({total/60:.1f}min)")
    print(f"  Results: {results_path}")


if __name__ == "__main__":
    main()
