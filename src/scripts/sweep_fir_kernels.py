#!/usr/bin/env python
"""Sweep: FIR vs IIR synaptic kernels  +  lag-neighbor enhancements.

Conditions (each tackles one or more of the three bottlenecks):

  T00  IIR baseline           – current default (sigmoid, shared τ/a)
  T01  FIR K=5 identity       – removes sigmoid squash + free kernel shape
  T02  FIR K=5 sigmoid        – free kernel shape, keeps sigmoid
  T03  FIR K=3 identity       – shorter kernel
  T04  FIR K=8 identity       – longer kernel
  T05  FIR K=5 identity+rev   – adds (E-u) reversal driving force back
  T06  FIR K=5 + lag_nbr all  – FIR + lag-neighbor on combined connectome
  T07  IIR + lag_nbr all      – IIR + lag-neighbor on combined connectome

All share: edge_specific_G=True, per_neuron_amplitudes=True,
           num_epochs=30, cv_folds=3, seed=42.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from stage2.config import make_config
from stage2.train import train_stage2

# ── paths ──────────────────────────────────────────────────────────────
H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/sweep_fir_kernels")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── common hyper-parameters ────────────────────────────────────────────
COMMON: dict = dict(
    num_epochs=30,
    cv_folds=3,
    seed=42,
    edge_specific_G=True,
    per_neuron_amplitudes=True,
)

# ── conditions ─────────────────────────────────────────────────────────
CONDITIONS: dict[str, dict] = {
    # Baseline – IIR with sigmoid, globally-shared tau/a
    "T00_iir_baseline": dict(
        chemical_synapse_mode="iir",
    ),

    # FIR K=5, identity activation  (tackles: squash + kernel shape + per-edge)
    "T01_fir_K5_identity": dict(
        chemical_synapse_mode="fir",
        fir_kernel_len=5,
        fir_activation="identity",
    ),

    # FIR K=5, sigmoid  (tackles: kernel shape + per-edge; keeps squash)
    "T02_fir_K5_sigmoid": dict(
        chemical_synapse_mode="fir",
        fir_kernel_len=5,
        fir_activation="sigmoid",
    ),

    # FIR K=3, identity (shorter horizon)
    "T03_fir_K3_identity": dict(
        chemical_synapse_mode="fir",
        fir_kernel_len=3,
        fir_activation="identity",
    ),

    # FIR K=8, identity (longer horizon)
    "T04_fir_K8_identity": dict(
        chemical_synapse_mode="fir",
        fir_kernel_len=8,
        fir_activation="identity",
    ),

    # FIR K=5, identity + reversal driving force (E - u_i)
    "T05_fir_K5_identity_rev": dict(
        chemical_synapse_mode="fir",
        fir_kernel_len=5,
        fir_activation="identity",
        fir_include_reversal=True,
    ),

    # FIR K=5, identity  +  lag-neighbor on full connectome
    "T06_fir_K5_lag_nbr": dict(
        chemical_synapse_mode="fir",
        fir_kernel_len=5,
        fir_activation="identity",
        lag_order=5,
        lag_neighbor=True,
        lag_connectome_mask="all",
    ),

    # IIR  +  lag-neighbor on full connectome (boosted lag only)
    "T07_iir_lag_nbr": dict(
        chemical_synapse_mode="iir",
        lag_order=5,
        lag_neighbor=True,
        lag_connectome_mask="all",
    ),
}

# ── helpers ────────────────────────────────────────────────────────────
log_path = OUT_ROOT / "sweep.log"


def tee(msg: str) -> None:
    print(msg, flush=True)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


# ── main loop ──────────────────────────────────────────────────────────
def main() -> None:
    tee(f"=== FIR-kernel sweep  ({len(CONDITIONS)} conditions) ===")
    tee(f"Common: {COMMON}")
    tee(f"Output: {OUT_ROOT.resolve()}\n")

    results: dict[str, dict] = {}

    for name, overrides in CONDITIONS.items():
        save_dir = OUT_ROOT / name
        summary_path = save_dir / "summary.json"

        if summary_path.exists():
            tee(f">>> SKIP {name}  (summary.json exists)")
            with open(summary_path) as f:
                results[name] = json.load(f)
            continue

        tee(f"\n{'='*60}")
        tee(f">>> START  {name}")
        tee(f"    overrides: {overrides}")
        tee(f"{'='*60}")

        kw = {**COMMON, **overrides}
        cfg = make_config(H5, **kw)
        t0 = time.time()

        try:
            result = train_stage2(cfg, save_dir=str(save_dir), show=False)
            elapsed = time.time() - t0
            tee(f">>> {name} DONE in {elapsed/60:.1f} min")

            summary: dict = {
                "condition": name,
                "overrides": overrides,
                "elapsed_min": round(elapsed / 60, 1),
            }
            if result and isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, (int, float, str, bool)):
                        summary[k] = v
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            tee(f">>> Saved {summary_path}")
            results[name] = summary

        except Exception as e:
            elapsed = time.time() - t0
            tee(f">>> {name} FAILED after {elapsed/60:.1f} min: {e}")
            tee(traceback.format_exc())
            results[name] = {"condition": name, "error": str(e)}

    # ── summary table ──────────────────────────────────────────────────
    tee(f"\n{'='*60}")
    tee("SWEEP SUMMARY")
    tee(f"{'='*60}")
    header = f"{'Condition':<28} {'elapsed':>8}  {'metric':>10}"
    tee(header)
    tee("-" * len(header))
    for name, s in results.items():
        mins = s.get("elapsed_min", "?")
        # try common metric names
        metric = s.get("best_val_corr", s.get("mean_corr", s.get("val_loss", "?")))
        tee(f"{name:<28} {str(mins):>8}  {str(metric):>10}")
    tee(f"{'='*60}")


if __name__ == "__main__":
    main()
