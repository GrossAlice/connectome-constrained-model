#!/usr/bin/env python
"""
MLP Context Sweep: test multi-step history in residual MLP.
═══════════════════════════════════════════════════════════════════

Motivation
──────────
Stage2 with softplus+MLP64 gave LOO_w=0.182 — best ever — but the
MLP only sees u_t (K=1).  Ridge baseline (LOO_w=0.363) uses K=5 context.
Closing this structural gap should push LOO further.

This sweep tests:
  A.  Context depth K={1, 3, 5, 7}  with h=64 (current default)
  B.  Same K values with h=128      (more capacity for wider input)
  C.  Best K + MLP off baseline     (isolate MLP contribution)

All conditions use softplus activation (current best), 30 epochs,
coupling_gate=True, graph_poly_order=2 (defaults).

Usage
─────
    python -u -m scripts.sweep_mlp_context \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
        --out output_plots/stage2/mlp_context_sweep \
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
    """Return OrderedDict[name → config overrides dict]."""
    C = OrderedDict()

    # ─────────── GROUP A: Context depth with h=64 ───────────
    C["A1_K1_h64"] = dict(
        residual_mlp_hidden=64,
        residual_mlp_context=1,
    )
    C["A2_K3_h64"] = dict(
        residual_mlp_hidden=64,
        residual_mlp_context=3,
    )
    C["A3_K5_h64"] = dict(
        residual_mlp_hidden=64,
        residual_mlp_context=5,
    )
    C["A4_K7_h64"] = dict(
        residual_mlp_hidden=64,
        residual_mlp_context=7,
    )

    # ─────────── GROUP B: Context depth with h=128 ───────────
    C["B1_K1_h128"] = dict(
        residual_mlp_hidden=128,
        residual_mlp_context=1,
    )
    C["B2_K3_h128"] = dict(
        residual_mlp_hidden=128,
        residual_mlp_context=3,
    )
    C["B3_K5_h128"] = dict(
        residual_mlp_hidden=128,
        residual_mlp_context=5,
    )
    C["B4_K7_h128"] = dict(
        residual_mlp_hidden=128,
        residual_mlp_context=7,
    )

    # ─────────── GROUP C: MLP off baseline + larger h ───────────
    C["C1_noMLP"] = dict(
        residual_mlp_hidden=0,
        residual_mlp_context=1,
    )
    C["C2_K5_h256"] = dict(
        residual_mlp_hidden=256,
        residual_mlp_context=5,
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

    cond_epochs = overrides.pop("num_epochs", default_epochs)

    kw = dict(overrides)
    kw.update(
        num_epochs=cond_epochs,
        device=device,
        cv_folds=cv_folds,
        # Standard dynamics defaults
        chemical_synapse_activation="softplus",
        learn_lambda_u=True,
        learn_I0=True,
        edge_specific_G=True,
        learn_W_sv=True,
        learn_W_dcv=True,
        learn_noise=True,
        noise_mode="heteroscedastic",
        per_neuron_amplitudes=True,
        coupling_gate=True,
        graph_poly_order=2,
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
    print(f"  MLP CONTEXT SWEEP SUMMARY")
    print(f"{'='*120}")

    hdr = (f"{'Condition':<25s} {'Ep':>3s} {'OS_mean':>7s} {'LOO_mean':>8s} "
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
        nm = r.get("name", "?")[:24]
        ep = r.get("epochs", "?")
        os_m = r.get("os_mean", float("nan"))
        loo_m = r.get("loo_mean", float("nan"))
        loo_w_m = r.get("loo_w_mean", float("nan"))
        loo_w_med = r.get("loo_w_median", float("nan"))
        n_pos = r.get("loo_w_n_pos", -1)
        t_s = r.get("time_s", 0)
        ov = ", ".join(f"{k}={v}" for k, v in r.get("overrides", {}).items())

        print(f"{nm:<25s} {ep:>3} {os_m:>7.4f} {loo_m:>8.4f} "
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

    # Color by group
    def group_color(name: str) -> str:
        if name.startswith("A"): return "#4682b4"   # blue   - h=64
        if name.startswith("B"): return "#d62728"   # red    - h=128
        if name.startswith("C"): return "#2ca02c"   # green  - controls
        return "#17becf"

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: LOO windowed R²
    ax = axes[0]
    colors = [group_color(n) for n in names]
    best_val = max(loo_w) if loo_w else 0
    colors_hl = ["#FFD700" if v == best_val else c for v, c in zip(loo_w, colors)]
    ax.barh(range(len(names)), loo_w, color=colors_hl, alpha=0.85,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("LOO R² (windowed, mean)", fontsize=11)
    ax.set_title("MLP Context Sweep — LOO Windowed R²", fontsize=13, fontweight="bold")
    # Reference lines
    ax.axvline(0.145, color="gray", lw=1, ls="--", alpha=0.7, label="softplus K=1 (0.145)")
    ax.axvline(0.182, color="blue", lw=1, ls=":", alpha=0.7, label="softplus+MLP64 K=1 (0.182)")
    ax.axvline(0.363, color="red", lw=1, ls="-.", alpha=0.7, label="Ridge K=5 (0.363)")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()

    # Panel 2: 1-step R²
    ax = axes[1]
    ax.barh(range(len(names)), os_m, color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("1-step R² (mean)", fontsize=11)
    ax.set_title("MLP Context Sweep — 1-step R²", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_dir / "sweep_comparison.png", dpi=150)
    plt.close(fig)
    print(f"[sweep] Plot saved to {save_dir / 'sweep_comparison.png'}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="MLP context sweep")
    ap.add_argument("--h5", required=True, help="HDF5 file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cv_folds", type=int, default=2)
    ap.add_argument("--loo_subset", type=int, default=30)
    ap.add_argument("--resume", action="store_true",
                    help="Skip conditions whose output already exists")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sweep.log"
    results_path = out_dir / "sweep_results.json"

    # Load existing results for resume
    existing_results: List[dict] = []
    done_names: set = set()
    if args.resume and results_path.exists():
        try:
            existing_results = json.loads(results_path.read_text())
            done_names = {r["name"] for r in existing_results}
            print(f"[sweep] Resuming — {len(done_names)} conditions already done")
        except Exception:
            pass

    conditions = build_conditions()
    results = list(existing_results)

    # Redirect stdout/stderr to log file (keep console too)
    import io

    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s)
                st.flush()
        def flush(self):
            for st in self.streams:
                st.flush()

    log_f = open(log_path, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)

    total = len(conditions)
    for idx, (name, overrides) in enumerate(conditions.items(), 1):
        if name in done_names:
            print(f"[sweep] [{idx}/{total}] {name} — SKIPPED (already done)")
            continue

        cond_dir = out_dir / name
        cond_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#'*80}")
        print(f"[sweep] [{idx}/{total}] {name}")
        print(f"  overrides = {overrides}")
        print(f"{'#'*80}\n")

        try:
            summary = run_one(
                h5_path=args.h5,
                cond_name=name,
                overrides=dict(overrides),  # copy so pop doesn't mutate
                save_dir=str(cond_dir),
                default_epochs=args.epochs,
                device=args.device,
                cv_folds=args.cv_folds,
                loo_subset=args.loo_subset,
            )
            results.append(summary)
            # Crash-safe: save after each condition
            results_path.write_text(json.dumps(results, indent=2))
            print(f"[sweep] {name} done — "
                  f"LOO_w_mean={summary.get('loo_w_mean', '?'):.4f}  "
                  f"OS_mean={summary.get('os_mean', '?'):.4f}")
        except Exception as e:
            import traceback
            print(f"[sweep] {name} FAILED: {e}")
            traceback.print_exc()
            results.append({"name": name, "error": str(e)})
            results_path.write_text(json.dumps(results, indent=2))

    # Final summary
    valid = [r for r in results if "loo_w_mean" in r]
    if valid:
        print_summary(valid)
        try:
            plot_sweep_results(valid, out_dir)
        except Exception as e:
            print(f"[sweep] Plotting failed: {e}")

    print(f"\n[sweep] ALL DONE — {len(valid)}/{total} conditions completed")
    print(f"[sweep] Results saved to {results_path}")
    log_f.close()


if __name__ == "__main__":
    main()
