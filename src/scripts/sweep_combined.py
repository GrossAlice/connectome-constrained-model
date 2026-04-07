#!/usr/bin/env python3
"""
Combined Sweep — rollout + softplus + coupling-dropout (now fixed)
===================================================================

Motivation
──────────
From the followup sweep the top-2 conditions were:
  1. F6_kitchen_sink_100ep  LOO_w = 0.1519
  2. E2_rollout_60ep        LOO_w = 0.1462

Both used rollout training.  However **coupling_dropout was silently
ignored** in all prior sweeps because the field was never registered
in DynamicsConfig.  D1/D2/D3 all produced bitwise-identical
LOO_w = 0.1402 — confirming the bug.

This sweep re-tests the best combinations with the **fixed**
coupling_dropout, and also explores:
  - Rollout + softplus (the two individually best features)
  - Fixed coupling_dropout at 0.1 / 0.2 / 0.3
  - Kitchen-sink with working dropout
  - Shorter epochs (30) for cost-effective screening

Usage
─────
    python -u -m scripts.sweep_combined \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
        --out output_plots/stage2/combined_sweep \
        --device cuda --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════
#  Sweep conditions
# ═══════════════════════════════════════════════════════════════════════

def build_conditions() -> OrderedDict:
    """Return OrderedDict[name → config overrides dict].

    ``num_epochs`` inside the overrides dict overrides the CLI default.
    """
    C = OrderedDict()

    # ─────────── GROUP A: Coupling dropout alone (now actually works) ───
    # Prior D1/D2/D3 were no-ops.  Re-run with the fix.
    C["A1_cdrop_0.1_30ep"] = dict(
        coupling_dropout=0.1,
    )
    C["A2_cdrop_0.2_30ep"] = dict(
        coupling_dropout=0.2,
    )
    C["A3_cdrop_0.3_30ep"] = dict(
        coupling_dropout=0.3,
    )
    C["A4_cdrop_0.1_60ep"] = dict(
        coupling_dropout=0.1,
        num_epochs=60,
    )
    C["A5_cdrop_0.2_60ep"] = dict(
        coupling_dropout=0.2,
        num_epochs=60,
    )

    # ─────────── GROUP B: Rollout + softplus (best individual features) ─
    # E2 rollout 60ep was #2 in followup sweep; softplus was best activation.
    C["B1_sp_rollout_30ep"] = dict(
        chemical_synapse_activation="softplus",
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
    )
    C["B2_sp_rollout_60ep"] = dict(
        chemical_synapse_activation="softplus",
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        num_epochs=60,
    )
    C["B3_sp_rollout_100ep"] = dict(
        chemical_synapse_activation="softplus",
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        num_epochs=100,
    )

    # ─────────── GROUP C: Rollout + softplus + coupling dropout ─────────
    # The big question: does regularising the connectome path via dropout
    # actually help when combined with rollout training?
    C["C1_sp_roll_cdrop01_60ep"] = dict(
        chemical_synapse_activation="softplus",
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        coupling_dropout=0.1,
        num_epochs=60,
    )
    C["C2_sp_roll_cdrop02_60ep"] = dict(
        chemical_synapse_activation="softplus",
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        coupling_dropout=0.2,
        num_epochs=60,
    )
    C["C3_sp_roll_cdrop03_60ep"] = dict(
        chemical_synapse_activation="softplus",
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        coupling_dropout=0.3,
        num_epochs=60,
    )
    C["C4_sp_roll_cdrop01_100ep"] = dict(
        chemical_synapse_activation="softplus",
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        coupling_dropout=0.1,
        num_epochs=100,
    )

    # ─────────── GROUP D: Rollout curriculum + softplus + cdrop ─────────
    # E3 curriculum was ~= E2 rollout in followup.  Does cdrop help here?
    C["D1_sp_curric_cdrop01_60ep"] = dict(
        chemical_synapse_activation="softplus",
        rollout_weight=0.3,
        rollout_curriculum=True,
        rollout_K_start=5,
        rollout_K_end=30,
        rollout_curriculum_start_epoch=5,
        rollout_curriculum_end_epoch=50,
        rollout_starts=4,
        coupling_dropout=0.1,
        num_epochs=60,
    )

    # ─────────── GROUP E: Kitchen sink (now with working cdrop) ─────────
    # F6 was the best overall (LOO_w=0.1519), but its cdrop=0.1 was a no-op.
    # Re-run with the fix.
    C["E1_kitchen_60ep"] = dict(
        chemical_synapse_activation="softplus",
        graph_poly_order=3,
        input_noise_sigma=0.05,
        coupling_dropout=0.1,
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        num_epochs=60,
    )
    C["E2_kitchen_100ep"] = dict(
        chemical_synapse_activation="softplus",
        graph_poly_order=3,
        input_noise_sigma=0.05,
        coupling_dropout=0.1,
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        num_epochs=100,
    )
    C["E3_kitchen_cdrop02_100ep"] = dict(
        chemical_synapse_activation="softplus",
        graph_poly_order=3,
        input_noise_sigma=0.05,
        coupling_dropout=0.2,
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        num_epochs=100,
    )

    # ─────────── GROUP F: Rollout weight / steps sensitivity ────────────
    # Only one rollout config (w=0.3, K=15) has been tested.  Try variants.
    C["F1_sp_roll_w01_60ep"] = dict(
        chemical_synapse_activation="softplus",
        rollout_weight=0.1,
        rollout_steps=15,
        rollout_starts=4,
        num_epochs=60,
    )
    C["F2_sp_roll_w05_60ep"] = dict(
        chemical_synapse_activation="softplus",
        rollout_weight=0.5,
        rollout_steps=15,
        rollout_starts=4,
        num_epochs=60,
    )
    C["F3_sp_roll_K30_60ep"] = dict(
        chemical_synapse_activation="softplus",
        rollout_weight=0.3,
        rollout_steps=30,
        rollout_starts=4,
        num_epochs=60,
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
    default_epochs: int,
    device: str,
    cv_folds: int,
    loo_subset: int,
) -> dict:
    """Train Stage2 with given overrides, return summary dict."""
    from stage2.config import make_config
    from stage2.train import train_stage2_cv

    # Per-condition epochs: pop from overrides if present, else use default
    cond_epochs = overrides.pop("num_epochs", default_epochs)

    kw = dict(overrides)
    kw.update(
        num_epochs=cond_epochs,
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
        per_neuron_amplitudes=True,
        # LOO evaluation
        eval_loo_subset_size=loo_subset,
        eval_loo_subset_mode="variance",
        eval_loo_window_size=50,
        eval_loo_warmup_steps=40,
        # Speed: skip final eval plots
        skip_final_eval=True,
    )
    # Defaults for coupling_gate and graph_poly_order unless overridden
    if "coupling_gate" not in overrides:
        kw["coupling_gate"] = True
    if "graph_poly_order" not in overrides:
        kw["graph_poly_order"] = 2

    cfg = make_config(h5_path, **kw)

    t0 = time.time()
    result = train_stage2_cv(cfg, save_dir=save_dir, show=False)
    elapsed = time.time() - t0

    summary: dict = {
        "name": cond_name,
        "overrides": {k: str(v) for k, v in overrides.items()},
        "epochs": cond_epochs,
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
    print(f"\n{'='*120}")
    print(f"  COMBINED SWEEP SUMMARY")
    print(f"{'='*120}")

    hdr = (f"{'Condition':<35s} {'Ep':>3s} {'OS_mean':>7s} {'LOO_mean':>8s} "
           f"{'LOOw_mean':>9s} {'LOOw_med':>9s} "
           f"{'#pos':>5s} {'Time':>6s}  Overrides")
    print(hdr)
    print("-" * 120)

    sorted_r = sorted(
        results,
        key=lambda r: r.get("loo_w_mean", float("-inf")),
        reverse=True,
    )

    for r in sorted_r:
        nm = r.get("name", "?")[:34]
        ep = r.get("epochs", "?")
        os_m = r.get("os_mean", float("nan"))
        loo_m = r.get("loo_mean", float("nan"))
        loo_w_m = r.get("loo_w_mean", float("nan"))
        loo_w_med = r.get("loo_w_median", float("nan"))
        n_pos = r.get("loo_w_n_pos", -1)
        t_s = r.get("time_s", 0)
        ov = ", ".join(f"{k}={v}" for k, v in r.get("overrides", {}).items())
        if len(ov) > 40:
            ov = ov[:37] + "..."

        print(f"{nm:<35s} {ep:>3} {os_m:>7.4f} {loo_m:>8.4f} "
              f"{loo_w_m:>9.4f} {loo_w_med:>9.4f} "
              f"{n_pos:>5d} {t_s:>5.0f}s  {ov}")

    if sorted_r:
        best = sorted_r[0]
        print(f"\n  ★ BEST: {best['name']}  LOO_w={best.get('loo_w_mean', 0):.4f}")
        print(f"    Overrides: {best.get('overrides', {})}")
    print()


def plot_sweep_results(results: List[dict], save_dir: Path):
    """Create comparison bar charts."""
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
    eps = [r.get("epochs", 30) for r in sorted_r]

    # Color by group
    def group_color(name: str) -> str:
        if name.startswith("A"): return "#9467bd"   # purple - cdrop alone
        if name.startswith("B"): return "#4682b4"   # blue   - rollout+softplus
        if name.startswith("C"): return "#2ca02c"   # green  - rollout+sp+cdrop
        if name.startswith("D"): return "#8c564b"   # brown  - curriculum
        if name.startswith("E"): return "#d62728"   # red    - kitchen sink
        if name.startswith("F"): return "#ff7f0e"   # orange - rollout sensitivity
        return "#17becf"

    fig, axes = plt.subplots(2, 1, figsize=(16, 14))

    # Panel 1: LOO windowed R²
    ax = axes[0]
    colors = [group_color(n) for n in names]
    best_val = max(loo_w) if loo_w else 0
    colors_hl = ["#FFD700" if v == best_val else c for v, c in zip(loo_w, colors)]
    ax.barh(range(len(names)), loo_w, color=colors_hl, alpha=0.85,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    labels = [f"{n}  ({e}ep)" for n, e in zip(names, eps)]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("LOO R² (windowed, mean)", fontsize=11)
    ax.set_title("Combined Sweep — LOO Windowed R²", fontsize=13, fontweight="bold")
    # Reference lines from prior sweeps
    ax.axvline(0.1402, color="gray", lw=1, ls="--", alpha=0.7,
               label="30ep baseline (0.140)")
    ax.axvline(0.1519, color="red", lw=1, ls=":", alpha=0.7,
               label="F6 kitchen sink prior (0.152, cdrop was no-op)")
    ax.axvline(0.1462, color="blue", lw=1, ls=":", alpha=0.7,
               label="E2 rollout 60ep (0.146)")
    ax.invert_yaxis()
    for i, v in enumerate(loo_w):
        ax.text(max(v + 0.002, 0.002), i, f"{v:.4f}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#9467bd", label="A: Coupling dropout alone"),
        Patch(facecolor="#4682b4", label="B: Rollout + softplus"),
        Patch(facecolor="#2ca02c", label="C: Rollout + softplus + cdrop"),
        Patch(facecolor="#8c564b", label="D: Curriculum + cdrop"),
        Patch(facecolor="#d62728", label="E: Kitchen sink (fixed cdrop)"),
        Patch(facecolor="#ff7f0e", label="F: Rollout sensitivity"),
        plt.Line2D([0], [0], color="gray", ls="--",
                   label="Prior baseline 0.140"),
        plt.Line2D([0], [0], color="red", ls=":",
                   label="Prior F6 kitchen 0.152 (cdrop no-op)"),
        plt.Line2D([0], [0], color="blue", ls=":",
                   label="Prior E2 rollout 0.146"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right", ncol=2)

    # Panel 2: One-step R²
    ax = axes[1]
    ax.barh(range(len(names)), os_m, color=colors, alpha=0.8,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("One-step R² (mean)", fontsize=11)
    ax.set_title("Combined Sweep — One-step R²", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for i, v in enumerate(os_m):
        ax.text(max(v + 0.002, 0.002), i, f"{v:.4f}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "combined_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_dir / 'combined_sweep.png'}")

    # ── Coupling dropout dose-response ──
    # Group A (cdrop alone) + Group C (with rollout+sp)
    cdrop_alone = [(r.get("overrides", {}).get("coupling_dropout", "0"),
                    r.get("loo_w_mean", 0), r.get("epochs", 30))
                   for r in sorted_r if r["name"].startswith("A")]
    cdrop_combo = [(r.get("overrides", {}).get("coupling_dropout", "0"),
                    r.get("loo_w_mean", 0), r.get("epochs", 30))
                   for r in sorted_r if r["name"].startswith("C")]

    if cdrop_alone or cdrop_combo:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        if cdrop_alone:
            # Sort by dropout rate
            cdrop_alone.sort(key=lambda x: (x[2], float(x[0])))
            for ep in sorted(set(x[2] for x in cdrop_alone)):
                pts = [(float(x[0]), x[1]) for x in cdrop_alone if x[2] == ep]
                pts.sort()
                ax2.plot(*zip(*pts), "o-", label=f"cdrop alone ({ep}ep)",
                         alpha=0.8, lw=2)
        if cdrop_combo:
            cdrop_combo.sort(key=lambda x: (x[2], float(x[0])))
            for ep in sorted(set(x[2] for x in cdrop_combo)):
                pts = [(float(x[0]), x[1]) for x in cdrop_combo if x[2] == ep]
                pts.sort()
                ax2.plot(*zip(*pts), "s--", label=f"rollout+sp+cdrop ({ep}ep)",
                         alpha=0.8, lw=2)
        ax2.axhline(0.1402, color="gray", ls="--", alpha=0.5, label="baseline")
        ax2.set_xlabel("Coupling Dropout Rate", fontsize=11)
        ax2.set_ylabel("LOO R² (windowed, mean)", fontsize=11)
        ax2.set_title("Coupling Dropout Dose-Response", fontsize=13,
                       fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(save_dir / "cdrop_dose_response.png", dpi=200,
                     bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved {save_dir / 'cdrop_dose_response.png'}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Combined sweep: rollout + softplus + coupling dropout (fixed)"
    )
    ap.add_argument("--h5", required=True, help="Path to worm .h5 file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--epochs", type=int, default=30,
                    help="Default epochs (overridden by per-condition num_epochs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cv_folds", type=int, default=2)
    ap.add_argument("--loo_subset", type=int, default=30)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Run only specific conditions")
    ap.add_argument("--resume", action="store_true",
                    help="Skip conditions that already have results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    worm = Path(args.h5).stem

    conditions = build_conditions()
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

    # Estimate total time
    est_times = []
    for cname, ov in todo.items():
        ep = ov.get("num_epochs", args.epochs)
        est_times.append(ep * 6.5)  # ~6.5s per epoch
    est_total_min = sum(est_times) / 60

    print(f"\n{'='*70}")
    print(f"  COMBINED SWEEP  (rollout + softplus + coupling dropout)")
    print(f"  Worm: {worm}")
    print(f"  Conditions: {len(conditions)} total, {len(todo)} remaining")
    print(f"  Default epochs: {args.epochs}")
    print(f"  LOO subset: {args.loo_subset}")
    print(f"  Estimated time: ~{est_total_min:.0f} min")
    print(f"{'='*70}\n")

    for i, (cname, overrides) in enumerate(todo.items(), 1):
        ov = dict(overrides)
        cond_epochs = ov.get("num_epochs", args.epochs)
        ov_display = {k: v for k, v in ov.items() if k != "num_epochs"}
        ov_str = ", ".join(f"{k}={v}" for k, v in ov_display.items()) or "(defaults)"

        print(f"\n{'═'*70}")
        print(f"  [{i}/{len(todo)}] {cname}  ({cond_epochs} epochs)")
        print(f"  Overrides: {ov_str}")
        print(f"{'═'*70}")

        cfg_dir = str(out / cname)
        try:
            summary = run_one(
                args.h5, cname, ov,
                save_dir=cfg_dir,
                default_epochs=args.epochs,
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

        # Print progress
        valid = [r for r in all_results if "error" not in r]
        if valid:
            print_summary(valid)

    # Final summary + plot
    valid = [r for r in all_results if "error" not in r]
    if valid:
        print_summary(valid)
        plot_sweep_results(valid, out)

    total = sum(r.get("time_s", 0) for r in all_results)
    print(f"\n  Total time: {total:.0f}s ({total/60:.1f}min)")
    print(f"  Results: {results_path}")


if __name__ == "__main__":
    main()
