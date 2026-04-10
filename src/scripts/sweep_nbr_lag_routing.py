#!/usr/bin/env python
"""Sweep: move I_lag_nbr into biophysical currents (I_gap / I_sv / I_dcv).

After self-lag was moved outside the λ·g gate (AR(K) intrinsic dynamics),
this sweep tests strategies for routing the *neighbor* temporal information
through biophysical channels instead of a separate phenomenological I_lag_nbr.

Strategies tested:
  S0  selfout_baseline   – new default: self-lag outside, nbr lag separate
  S1  no_nbr             – ablation: remove neighbor lag entirely (lower bound)
  S2  synlag10           – synapse_lag_taps=10, no nbr lag (FIR through T_e/T_sv/T_dcv)
  S3  synlag5            – synapse_lag_taps=5, no nbr lag (shorter FIR)
  S4  synlag15           – synapse_lag_taps=15, no nbr lag (longer FIR)
  S5  chemlag10          – chemical_synapse_mode="lag", no nbr lag (per-edge FIR replaces IIR)
  S6  chemlag10_synlag10 – chem FIR + synapse FIR for gap temporal taps
  S7  synlag10_both      – synapse FIR + keep nbr lag (redundancy / upper bound)
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from stage2.config import make_config
from stage2.train import train_stage2

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT = "output_plots/stage2/sweep_nbr_lag_routing"

# ── Base config: self-lag outside λ·g is now in model.py code ──────────
BASE = dict(
    num_epochs=60,
    seed=42,
)

CONDITIONS = {
    # S0: new baseline — self-lag outside gate, neighbor lag still separate inside λ·g
    "S0_selfout_baseline": dict(
        **BASE,
    ),
    # S1: ablation — no neighbor temporal info at all
    "S1_no_nbr": dict(
        **BASE,
        lag_neighbor=False,
    ),
    # S2: route neighbor temporal via synapse_lag_taps (per-neuron FIR through T_e/T_sv/T_dcv)
    "S2_synlag10": dict(
        **BASE,
        synapse_lag_taps=10,
        lag_neighbor=False,
    ),
    # S3: shorter synapse FIR
    "S3_synlag5": dict(
        **BASE,
        synapse_lag_taps=5,
        lag_neighbor=False,
    ),
    # S4: longer synapse FIR
    "S4_synlag15": dict(
        **BASE,
        synapse_lag_taps=15,
        lag_neighbor=False,
    ),
    # S5: replace IIR with per-edge FIR for chemical synapses (gap stays instantaneous)
    "S5_chemlag10": dict(
        **BASE,
        chemical_synapse_mode="lag",
        chem_lag_kernel_len=10,
        lag_neighbor=False,
    ),
    # S6: chem FIR + synapse FIR (gives gap junctions temporal taps via synapse_lag)
    "S6_chemlag10_synlag10": dict(
        **BASE,
        chemical_synapse_mode="lag",
        chem_lag_kernel_len=10,
        synapse_lag_taps=10,
        lag_neighbor=False,
    ),
    # S7: synapse FIR + keep neighbor lag (redundancy test / upper bound)
    "S7_synlag10_both": dict(
        **BASE,
        synapse_lag_taps=10,
        lag_neighbor=True,
    ),
}


def run_one(name: str, overrides: dict) -> dict:
    save_dir = str(Path(OUT) / name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  overrides: {overrides}")
    print(f"{'='*60}\n", flush=True)
    t0 = time.time()
    cfg = make_config(H5, **overrides)
    result = train_stage2(cfg, save_dir=save_dir, show=False)
    elapsed = time.time() - t0
    summary_path = Path(save_dir) / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            s = json.load(f)
    else:
        s = {}
    s["elapsed_s"] = elapsed
    s["condition"] = name
    with open(summary_path, "w") as f:
        json.dump(s, f, indent=2)
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    all_results = {}
    for name, overrides in CONDITIONS.items():
        save_dir = Path(OUT) / name / "summary.json"
        if save_dir.exists():
            print(f"SKIP {name} (already done)")
            with open(save_dir) as f:
                all_results[name] = json.load(f)
            continue
        try:
            s = run_one(name, overrides)
            all_results[name] = s
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  SWEEP RESULTS: neighbor lag routing strategies")
    print(f"{'='*80}")
    print(f"{'Condition':<30s} {'1step':>8s} {'LOO':>8s} {'med_LOO':>8s}")
    print("-" * 60)
    for name, s in all_results.items():
        onestep = s.get("cv_onestep_r2_mean", "?")
        loo = s.get("cv_loo_r2_mean", "?")
        med_loo = s.get("cv_loo_r2_median", "?")
        if isinstance(onestep, (int, float)):
            print(f"{name:<30s} {onestep:>8.4f} {loo:>8.4f} {med_loo:>8.4f}")
        else:
            err = s.get("error", "?")
            print(f"{name:<30s}  ERROR: {err[:40]}")

    with open(Path(OUT) / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT}/all_results.json")


if __name__ == "__main__":
    main()
