#!/usr/bin/env python
"""Sweep v2: move I_lag,nbr inside biophysical channels (I_gap / I_sv / I_dcv).

This sweep tests ALL routing strategies for neighbor temporal information.
Previous sweeps that FAILED (DO NOT REPEAT):
  - synapse_lag_taps (LOO~0.21): per-neuron FIR through T_e/T_sv/T_dcv, too few params
  - chemical_synapse_mode="lag" (LOO~0.19): replaces IIR entirely, loses IIR dynamics
  - conductance_lags with identity init: LOO goes negative (numerical instability)

Reference results:
  - Ridge baseline:       LOO = 0.4802
  - A0_baseline additive: LOO = 0.4705  (best S2 so far)
  - No neighbor lag:      LOO = 0.23    (catastrophic)

STRATEGIES (grouped by approach):

Group A: Baselines
  R00 additive baseline (reference)

Group B: Coded routing modes (replace I_lag_nbr with biophysical routing)
  R01 gap_laplacian         — Σ_k α_k · (L @ u(t-k)), K scalars
  R02 conductance_modulated — G modulated by tanh(Σ_k α_k · u(t-k))
  R03 iir_augmented         — inject Σ_k w_k · φ(u(t-k)) into IIR state
  R04 per_current           — scalar gap FIR + per-edge chem FIR (with φ)
  R05 per_current_linear    — same as R04 but skip φ activation on chem lag
  R06 weighted_topology     — combined weighted (w_e·L + w_sv·T_sv + w_dcv·T_dcv) @ (α_k⊙u)
  R07 gap_only              — additive lag restricted to gap topology
  R08 chem_only             — additive lag restricted to chemical topology

Group C: Hybrid (biophysical routing + keep additive lag_nbr)
  R09 per_current + additive         — per_current routing AND additive lag
  R10 per_current_linear + additive  — per_current (no φ) AND additive lag
  R11 gap_laplacian + additive       — gap FIR through L AND additive lag
  R12 iir_augmented + additive       — IIR augmentation AND additive lag
  R13 conductance_mod + additive     — conductance modulation AND additive lag

Group D: IIR delay (keep additive lag, add biophysical delays)
  R14 delay_sv2_dcv3     — iir_delay_sv=2, iir_delay_dcv=3
  R15 delay_sv3_dcv5     — iir_delay_sv=3, iir_delay_dcv=5
  R16 delay_sv5_dcv8     — iir_delay_sv=5, iir_delay_dcv=8

Group E: Multi-timescale IIR (more IIR ranks, keep additive lag)
  R17 multi_3x3          — r_sv=3, r_dcv=3
  R18 multi_4x4          — r_sv=4, r_dcv=4

Group F: Activation variants on additive routing
  R19 additive_sigmoid   — sigmoid activation on neighbor lag
  R20 additive_softplus  — softplus activation on neighbor lag
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from stage2.config import make_config
from stage2.train import train_stage2

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT = "output_plots/stage2/sweep_lag_routing_v2"

# ── Base config: matches A0_baseline (best S2 so far: LOO=0.4705) ────
BASE = dict(
    num_epochs=60,
    seed=42,
    lag_order=10,
    lag_neighbor=True,
    lag_neighbor_per_type=True,
    lag_connectome_mask="all",
    l2_penalty=1e-3,
    rollout_steps=10,
    rollout_weight=0.3,
    warmstart_rollout=True,
)

CONDITIONS = {
    # ═════════════════════════════════════════════════════════════════
    # Group A: Baselines
    # ═════════════════════════════════════════════════════════════════
    "R00_additive_baseline": dict(
        **BASE,
        lag_nbr_routing="additive",
    ),

    # ═════════════════════════════════════════════════════════════════
    # Group B: Biophysical routing modes (replace I_lag_nbr)
    # ═════════════════════════════════════════════════════════════════
    "R01_gap_laplacian": dict(
        **BASE,
        lag_nbr_routing="gap_laplacian",
    ),
    "R02_conductance_mod": dict(
        **BASE,
        lag_nbr_routing="conductance_modulated",
    ),
    "R03_iir_augmented": dict(
        **BASE,
        lag_nbr_routing="iir_augmented",
    ),
    "R04_per_current": dict(
        **BASE,
        lag_nbr_routing="per_current",
    ),
    "R05_per_current_lin": dict(
        **BASE,
        lag_nbr_routing="per_current",
        lag_nbr_routing_linear=True,
    ),
    "R06_weighted_topo": dict(
        **BASE,
        lag_nbr_routing="weighted_topology",
    ),
    "R07_gap_only": dict(
        **BASE,
        lag_nbr_routing="gap_only",
    ),
    "R08_chem_only": dict(
        **BASE,
        lag_nbr_routing="chem_only",
    ),

    # ═════════════════════════════════════════════════════════════════
    # Group C: Hybrid (biophysical + keep additive)
    # ═════════════════════════════════════════════════════════════════
    "R09_pc_plus_add": dict(
        **BASE,
        lag_nbr_routing="per_current",
        lag_nbr_routing_keep_additive=True,
    ),
    "R10_pclin_plus_add": dict(
        **BASE,
        lag_nbr_routing="per_current",
        lag_nbr_routing_linear=True,
        lag_nbr_routing_keep_additive=True,
    ),
    "R11_gaplap_plus_add": dict(
        **BASE,
        lag_nbr_routing="gap_laplacian",
        lag_nbr_routing_keep_additive=True,
    ),
    "R12_iiraug_plus_add": dict(
        **BASE,
        lag_nbr_routing="iir_augmented",
        lag_nbr_routing_keep_additive=True,
    ),
    "R13_condmod_plus_add": dict(
        **BASE,
        lag_nbr_routing="conductance_modulated",
        lag_nbr_routing_keep_additive=True,
    ),

    # ═════════════════════════════════════════════════════════════════
    # Group D: IIR delay + additive lag (biophysical interpretation)
    # ═════════════════════════════════════════════════════════════════
    "R14_delay_sv2_dcv3": dict(
        **BASE,
        lag_nbr_routing="additive",
        iir_delay_sv=2,
        iir_delay_dcv=3,
    ),
    "R15_delay_sv3_dcv5": dict(
        **BASE,
        lag_nbr_routing="additive",
        iir_delay_sv=3,
        iir_delay_dcv=5,
    ),
    "R16_delay_sv5_dcv8": dict(
        **BASE,
        lag_nbr_routing="additive",
        iir_delay_sv=5,
        iir_delay_dcv=8,
    ),

    # ═════════════════════════════════════════════════════════════════
    # Group E: Multi-timescale IIR (more IIR ranks to absorb lag)
    # ═════════════════════════════════════════════════════════════════
    "R17_multi_3x3": dict(
        **BASE,
        lag_nbr_routing="additive",
        tau_sv_init=(1.0, 3.0, 7.0),
        a_sv_init=(1.0, 0.8, 0.5),
        tau_dcv_init=(2.0, 5.0, 12.0),
        a_dcv_init=(0.8, 0.6, 0.3),
    ),
    "R18_multi_4x4": dict(
        **BASE,
        lag_nbr_routing="additive",
        tau_sv_init=(0.5, 1.5, 4.0, 8.0),
        a_sv_init=(1.0, 0.8, 0.5, 0.3),
        tau_dcv_init=(1.0, 3.0, 7.0, 15.0),
        a_dcv_init=(0.8, 0.6, 0.4, 0.2),
    ),

    # ═════════════════════════════════════════════════════════════════
    # Group F: Activation variants on additive routing
    # ═════════════════════════════════════════════════════════════════
    "R19_add_sigmoid": dict(
        **BASE,
        lag_nbr_routing="additive",
        lag_neighbor_activation="sigmoid",
    ),
    "R20_add_softplus": dict(
        **BASE,
        lag_nbr_routing="additive",
        lag_neighbor_activation="softplus",
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
    print(f"  SWEEP RESULTS: lag routing v2")
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
