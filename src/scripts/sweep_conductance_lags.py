#!/usr/bin/env python
"""Sweep: Conductance-based I_sv + I_dcv + I_gap, ALL with temporal lags in parallel.

Goal: replicate Conn-Ridge LOO R² ≥ 0.475 using ONLY biophysical channels,
      no I_lag (lag_order=0, lag_neighbor=False).

Strategy:
  - I_sv / I_dcv keep their conductance-based IIR formulation (g*(E-u))
    PLUS temporal enrichment via synapse_lag_taps (linear FIR through T_sv/T_dcv)
    and/or iir_delay (pure delay before IIR filter).
  - I_gap gets temporal enrichment via synapse_lag_taps through T_e,
    and spatial enrichment via graph_poly_order (multi-hop Laplacian).
  - chemical_synapse_activation = "identity" (Conn-Ridge is linear).

All settings match D02 exactly (60 epochs, 3-fold CV, dynamics_l2=1e-3, etc.).
Conditions that duplicate the running sweep_synaptic_vs_lag are EXCLUDED.

Already covered by sweep_synaptic_vs_lag (do NOT repeat):
  N2: IIR + syn_lag=10 + identity
  N4: IIR + syn_lag=10 + poly3 + identity
  N6: chem_lag(10) + identity  (no syn_lag for T_e)
  N7: chem_lag(10) + poly3 + identity  (no syn_lag for T_e)
  S11: chem_lag(10) + syn_lag=10 (sigmoid)
  S12: IIR + syn_lag=10 + delay(2,3) (sigmoid)

Conditions here — Group C: Conductance + Temporal Lags
  ---------- More syn_lag taps (all 3 channels with longer FIR) ----------
  C0  IIR + syn_lag=15 + identity
  C1  IIR + syn_lag=20 + identity
  C2  IIR + syn_lag=25 + identity
  ---------- Spatial enrichment on larger temporal kernels -----------
  C3  IIR + syn_lag=15 + poly3 + identity
  C4  IIR + syn_lag=20 + poly3 + identity
  ---------- IIR delay + syn_lag (delay IIR input, add FIR) ----------
  C5  IIR + syn_lag=10 + delay(2,3) + identity
  C6  IIR + syn_lag=15 + delay(2,3) + identity
  C7  IIR + syn_lag=15 + delay(2,3) + poly3 + identity  (kitchen sink)
  ---------- chem_lag (per-edge FIR) + syn_lag for T_e gap lags ----------
  C8  chem_lag(10) + syn_lag=10 + identity  (per-edge chemical + factored gap FIR)
  C9  chem_lag(10) + syn_lag=10 + poly3 + identity
  C10 chem_lag(15) + syn_lag=15 + identity  (bigger kernels everywhere)
  C11 chem_lag(15) + syn_lag=15 + poly3 + identity  (full kitchen sink)
"""
import json, os, sys, time, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.config import make_config
from stage2.train import train_stage2

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/sweep_conductance_lags")

# ── D02-matching baseline settings ──────────────────────────────────
_COMMON = dict(
    num_epochs=60,
    cv_folds=3,
    parallel_folds=True,
    rollout_steps=10,
    rollout_weight=0.3,
    rollout_starts=8,
    input_noise_sigma=0.0,
    dynamics_l2=1e-3,
    coupling_dropout=0.0,
    edge_specific_G=True,
    per_neuron_amplitudes=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    seed=42,
    # All conditions: NO I_lag, identity activation
    lag_order=0,
    lag_neighbor=False,
    chemical_synapse_activation="identity",
)

CONDITIONS = {
    # ═══════════════════════════════════════════════════════════════════
    #  More synapse_lag_taps: linear FIR through T_e + T_sv + T_dcv
    #  (longer temporal memory on ALL three connectome channels)
    # ═══════════════════════════════════════════════════════════════════
    "C0_iir_synlag15_ident": {
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 15,
        "graph_poly_order": 1,
    },
    "C1_iir_synlag20_ident": {
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 20,
        "graph_poly_order": 1,
    },
    "C2_iir_synlag25_ident": {
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 25,
        "graph_poly_order": 1,
    },

    # ═══════════════════════════════════════════════════════════════════
    #  Spatial enrichment + larger temporal kernels
    #  (multi-hop Laplacian on gap junctions + FIR on all channels)
    # ═══════════════════════════════════════════════════════════════════
    "C3_iir_synlag15_poly3_ident": {
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 15,
        "graph_poly_order": 3,
    },
    "C4_iir_synlag20_poly3_ident": {
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 20,
        "graph_poly_order": 3,
    },

    # ═══════════════════════════════════════════════════════════════════
    #  IIR delay + synapse_lag_taps + identity
    #  Delay shifts the IIR input (conductance I_sv/I_dcv see older state)
    #  + FIR adds explicit temporal filters on all channels
    # ═══════════════════════════════════════════════════════════════════
    "C5_iir_synlag10_delay_ident": {
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 10,
        "iir_delay_sv": 2,
        "iir_delay_dcv": 3,
        "graph_poly_order": 1,
    },
    "C6_iir_synlag15_delay_ident": {
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 15,
        "iir_delay_sv": 2,
        "iir_delay_dcv": 3,
        "graph_poly_order": 1,
    },
    "C7_iir_synlag15_delay_poly3_ident": {
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 15,
        "iir_delay_sv": 2,
        "iir_delay_dcv": 3,
        "graph_poly_order": 3,
    },

    # ═══════════════════════════════════════════════════════════════════
    #  chem_lag (per-edge FIR on T_sv/T_dcv) + syn_lag (factored FIR on T_e)
    #  This gives ALL three channels explicit temporal lags:
    #    I_sv, I_dcv: per-edge FIR with N×N×K params (via chem_lag)
    #    I_gap:       per-neuron FIR with K×N params routed through T_e
    # ═══════════════════════════════════════════════════════════════════
    "C8_chemlag10_synlag10_ident": {
        "chemical_synapse_mode": "lag",
        "chem_lag_kernel_len": 10,
        "synapse_lag_taps": 10,
        "graph_poly_order": 1,
    },
    "C9_chemlag10_synlag10_poly3_ident": {
        "chemical_synapse_mode": "lag",
        "chem_lag_kernel_len": 10,
        "synapse_lag_taps": 10,
        "graph_poly_order": 3,
    },
    "C10_chemlag15_synlag15_ident": {
        "chemical_synapse_mode": "lag",
        "chem_lag_kernel_len": 15,
        "synapse_lag_taps": 15,
        "graph_poly_order": 1,
    },
    "C11_chemlag15_synlag15_poly3_ident": {
        "chemical_synapse_mode": "lag",
        "chem_lag_kernel_len": 15,
        "synapse_lag_taps": 15,
        "graph_poly_order": 3,
    },
}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}

    for label, overrides in CONDITIONS.items():
        out = OUT_ROOT / label
        summary_f = out / "summary.json"
        if summary_f.exists():
            print(f"\n{'='*60}\n  SKIP {label} (already done)\n{'='*60}")
            results[label] = json.loads(summary_f.read_text())
            continue

        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        kw = {**_COMMON, **overrides}
        cfg = make_config(H5, **kw)
        cfg.output.out_u_mean = None

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  chem_mode={cfg.chemical_synapse_mode}, "
              f"syn_lag_taps={cfg.synapse_lag_taps}, "
              f"poly_order={cfg.graph_poly_order}, "
              f"activation={cfg.chemical_synapse_activation}")
        if hasattr(cfg, 'iir_delay_sv') and cfg.iir_delay_sv > 0:
            print(f"  iir_delay_sv={cfg.iir_delay_sv}, iir_delay_dcv={cfg.iir_delay_dcv}")
        if cfg.chemical_synapse_mode == 'lag':
            print(f"  chem_lag_kernel_len={cfg.chem_lag_kernel_len}")
        print(f"  lag_order={cfg.lag_order}, lag_neighbor={cfg.lag_neighbor}")
        print(f"{'='*60}\n")

        t0 = time.time()
        summary = train_stage2(cfg, save_dir=str(out))
        elapsed = time.time() - t0

        summary["elapsed_s"] = round(elapsed, 1)
        results[label] = summary
        summary_f.write_text(json.dumps(summary, indent=2, default=str))
        print(f"  ✓ {label} done in {elapsed:.0f}s")

    # ── Comparison table ────────────────────────────────────────────
    CONN_RIDGE_LOO = 0.475
    D02_LOO = 0.420
    print("\n" + "=" * 120)
    print(f"{'Condition':<40} {'1step_mean':>10} {'1step_med':>10} "
          f"{'LOO_mean':>10} {'LOO_med':>10} {'time_s':>8}  {'vs targets':>14}")
    print("-" * 120)
    for label, d in results.items():
        r1   = d.get("cv_onestep_r2_mean", "?")
        r1m  = d.get("cv_onestep_r2_median", "?")
        loo  = d.get("cv_loo_r2_mean", "?")
        loom = d.get("cv_loo_r2_median", "?")
        t    = d.get("elapsed_s", "?")
        fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        flag = ""
        if isinstance(loo, (int, float)):
            if loo >= CONN_RIDGE_LOO:
                flag = "★ ≥ ConnRidge"
            elif loo >= D02_LOO:
                flag = "✓ ≥ D02"
        print(f"{label:<40} {fmt(r1):>10} {fmt(r1m):>10} "
              f"{fmt(loo):>10} {fmt(loom):>10} {fmt(t):>8}  {flag:>14}")
    print("=" * 120)
    print(f"\nTargets: D02 LOO R² = {D02_LOO:.3f} | Conn-Ridge LOO R² = {CONN_RIDGE_LOO:.3f}")
    print(f"★ = beat Conn-Ridge  |  ✓ = beat D02 (but not Conn-Ridge)")

    (OUT_ROOT / "all_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {OUT_ROOT / 'all_results.json'}")


if __name__ == "__main__":
    main()
