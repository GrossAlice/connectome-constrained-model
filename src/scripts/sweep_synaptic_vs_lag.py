#!/usr/bin/env python
"""Sweep: Can enriched synaptic channels (I_sv+I_dcv+I_gap) replace I_lag?

Goal: achieve LOO R² ≥ 0.475 (Conn-Ridge level) WITHOUT using I_lag
(i.e. lag_order=0, lag_neighbor=False).

ALL settings match D02 exactly (60 epochs, 3 folds, dynamics_l2=1e-3,
coupling_dropout=0.0, input_noise_sigma=0.0, seed=42).

Four axes of enrichment:
  1. synapse_lag_taps: linear FIR on raw u routed through T_e + T_sv + T_dcv
  2. chemical_synapse_mode: "lag" or "fir" replace IIR with richer temporal kernels
  3. graph_poly_order: multi-hop gap-junction spatial mixing
  4. chemical_synapse_activation: sigmoid vs identity vs softplus

Key targets:
  D02 reference:       LOO R² = 0.420  (has I_lag)
  Conn-Ridge baseline: LOO R² = 0.475  (linear, connectome-masked)

Conditions — Group R: References
  R0  D02 reference (full I_lag, IIR, sigmoid)

Conditions — Group S: Structural enrichment (sigmoid activation)
  S0  Pure IIR, no lag at all              → lower bound
  S1  Self-lag only (lag_order=5, no nbr)  → how much from self-AR?
  S2  syn_lag_taps=5                       → FIR through all 3 connectome types
  S3  syn_lag_taps=10                      → more taps
  S4  syn_lag_taps=15                      → even more
  S5  syn_lag_taps=5 + poly_order=3        → temporal + spatial hops
  S6  syn_lag_taps=10 + poly_order=3       → richer combo
  S7  chem_mode=lag (per-edge FIR, K=10)   → replace IIR with lag-FIR
  S8  chem_mode=lag + poly_order=3         → lag-FIR + spatial
  S9  chem_mode=fir, kernel=10             → per-edge learnable FIR kernels
  S10 chem_mode=fir, kernel=10 + poly=3    → FIR + spatial
  S11 syn_lag_taps=10 + chem_mode=lag      → factored FIR + per-edge FIR
  S12 syn_lag_taps=10 + iir_delay=2/3      → FIR + delayed IIR

Conditions — Group N: Nonlinearity (no-lag + best structural configs)
  Conn-Ridge is LINEAR — sigmoid may be hurting generalization.
  N0  No lag, IIR, identity (no φ)         → closest to Conn-Ridge linearity
  N1  No lag, IIR, softplus                → smooth nonlinearity
  N2  syn_lag_10, IIR, identity            → temporal FIR + linear
  N3  syn_lag_10, IIR, softplus            → temporal FIR + smooth φ
  N4  syn_lag_10 + poly3, identity         → full enrichment + linear ★→Conn-Ridge target
  N5  syn_lag_10 + poly3, softplus         → full enrichment + smooth φ
  N6  chem_mode=lag K=10, identity         → per-edge FIR + linear ★→Conn-Ridge target
  N7  chem_mode=lag K=10 + poly3, identity → kitchen sink linear ★→Conn-Ridge target
"""
import json, os, sys, time, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.config import make_config
from stage2.train import train_stage2

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/sweep_synaptic_vs_lag")

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
)

# ── No-lag common: all "S" conditions share these ──────────────────
_NO_LAG = dict(
    lag_order=0,
    lag_neighbor=False,
)

# ── With-lag reference: R0 shares these ────────────────────────────
_WITH_LAG = dict(
    lag_order=5,
    lag_neighbor=True,
    lag_neighbor_per_type=True,
    lag_connectome_mask="all",
    lag_neighbor_activation="none",
)

CONDITIONS = {
    # ── Reference ──
    "R0_d02_reference": {
        **_WITH_LAG,
        "chemical_synapse_mode": "iir",
        "graph_poly_order": 1,
    },

    # ── S0: pure IIR, no lag at all (lower bound) ──
    "S0_no_lag_iir": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "graph_poly_order": 1,
    },

    # ── S1: self-lag only (AR terms but no neighbor lags) ──
    "S1_self_lag_only": {
        "lag_order": 5,
        "lag_neighbor": False,
        "chemical_synapse_mode": "iir",
        "graph_poly_order": 1,
    },

    # ── synapse_lag_taps sweep (FIR through T_e+T_sv+T_dcv) ──
    "S2_syn_lag_5": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 5,
        "graph_poly_order": 1,
    },
    "S3_syn_lag_10": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 10,
        "graph_poly_order": 1,
    },
    "S4_syn_lag_15": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 15,
        "graph_poly_order": 1,
    },

    # ── synapse_lag_taps + spatial hops ──
    "S5_syn_lag_5_poly3": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 5,
        "graph_poly_order": 3,
    },
    "S6_syn_lag_10_poly3": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 10,
        "graph_poly_order": 3,
    },

    # ── chem_mode=lag (per-edge FIR through T_sv/T_dcv) ──
    "S7_chem_lag": {
        **_NO_LAG,
        "chemical_synapse_mode": "lag",
        "chem_lag_kernel_len": 10,
        "graph_poly_order": 1,
    },
    "S8_chem_lag_poly3": {
        **_NO_LAG,
        "chemical_synapse_mode": "lag",
        "chem_lag_kernel_len": 10,
        "graph_poly_order": 3,
    },

    # ── chem_mode=fir (per-edge learnable FIR kernels) ──
    "S9_chem_fir_k10": {
        **_NO_LAG,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 10,
        "graph_poly_order": 1,
    },
    "S10_chem_fir_k10_poly3": {
        **_NO_LAG,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 10,
        "graph_poly_order": 3,
    },

    # ── Kitchen-sink combinations ──
    "S11_syn_lag_10_chem_lag": {
        **_NO_LAG,
        "chemical_synapse_mode": "lag",
        "chem_lag_kernel_len": 10,
        "synapse_lag_taps": 10,
        "graph_poly_order": 1,
    },
    "S12_syn_lag_10_iir_delay": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "synapse_lag_taps": 10,
        "iir_delay_sv": 2,
        "iir_delay_dcv": 3,
        "graph_poly_order": 1,
    },

    # ═══════════════════════════════════════════════════════════════════
    #  Group N: Nonlinearity axis (no I_lag)
    #  Conn-Ridge is LINEAR → sigmoid may hurt generalization.
    #  These test if removing / changing φ(u) closes the gap.
    # ═══════════════════════════════════════════════════════════════════

    # N0: pure IIR + identity activation (linearise chemical synapses)
    "N0_no_lag_identity": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "chemical_synapse_activation": "identity",
        "graph_poly_order": 1,
    },
    # N1: pure IIR + softplus (smooth unbounded nonlinearity)
    "N1_no_lag_softplus": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "chemical_synapse_activation": "softplus",
        "graph_poly_order": 1,
    },

    # N2–N3: syn_lag_taps=10 + identity / softplus
    "N2_syn_lag_10_identity": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "chemical_synapse_activation": "identity",
        "synapse_lag_taps": 10,
        "graph_poly_order": 1,
    },
    "N3_syn_lag_10_softplus": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "chemical_synapse_activation": "softplus",
        "synapse_lag_taps": 10,
        "graph_poly_order": 1,
    },

    # N4–N5: syn_lag_10 + poly3 + identity / softplus ★ Conn-Ridge targets
    "N4_syn_lag_10_poly3_identity": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "chemical_synapse_activation": "identity",
        "synapse_lag_taps": 10,
        "graph_poly_order": 3,
    },
    "N5_syn_lag_10_poly3_softplus": {
        **_NO_LAG,
        "chemical_synapse_mode": "iir",
        "chemical_synapse_activation": "softplus",
        "synapse_lag_taps": 10,
        "graph_poly_order": 3,
    },

    # N6–N7: chem_mode=lag + identity ★ Conn-Ridge targets (per-edge FIR, linear)
    "N6_chem_lag_identity": {
        **_NO_LAG,
        "chemical_synapse_mode": "lag",
        "chem_lag_kernel_len": 10,
        "chemical_synapse_activation": "identity",
        "graph_poly_order": 1,
    },
    "N7_chem_lag_poly3_identity": {
        **_NO_LAG,
        "chemical_synapse_mode": "lag",
        "chem_lag_kernel_len": 10,
        "chemical_synapse_activation": "identity",
        "graph_poly_order": 3,
    },
}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}

    for label, overrides in CONDITIONS.items():
        out = OUT_ROOT / label
        # Skip if already completed
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
        cfg.output.out_u_mean = None  # don't write u_mean to HDF5

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  lag_order={cfg.lag_order}, lag_neighbor={cfg.lag_neighbor}")
        print(f"  chem_mode={cfg.chemical_synapse_mode}, "
              f"syn_lag_taps={cfg.synapse_lag_taps}, "
              f"poly_order={cfg.graph_poly_order}")
        if hasattr(cfg, 'iir_delay_sv'):
            print(f"  iir_delay_sv={cfg.iir_delay_sv}, iir_delay_dcv={cfg.iir_delay_dcv}")
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
    print("\n" + "=" * 110)
    print(f"{'Condition':<35} {'1step_mean':>10} {'1step_med':>10} "
          f"{'LOO_mean':>10} {'LOO_med':>10} {'time_s':>8}  {'vs targets':>12}")
    print("-" * 110)
    for label, d in results.items():
        r1  = d.get("cv_onestep_r2_mean", "?")
        r1m = d.get("cv_onestep_r2_median", "?")
        loo  = d.get("cv_loo_r2_mean", "?")
        loom = d.get("cv_loo_r2_median", "?")
        t    = d.get("elapsed_s", "?")
        fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        # Flag conditions that beat D02 or Conn-Ridge
        flag = ""
        if isinstance(loo, (int, float)):
            if loo >= CONN_RIDGE_LOO:
                flag = "★ ≥ ConnRidge"
            elif loo >= D02_LOO:
                flag = "✓ ≥ D02"
        print(f"{label:<35} {fmt(r1):>10} {fmt(r1m):>10} "
              f"{fmt(loo):>10} {fmt(loom):>10} {fmt(t):>8}  {flag:>12}")
    print("=" * 110)
    print(f"\nTargets: D02 LOO R² = {D02_LOO:.3f} | Conn-Ridge LOO R² = {CONN_RIDGE_LOO:.3f}")
    print(f"★ = beat Conn-Ridge  |  ✓ = beat D02 (but not Conn-Ridge)")

    # Save all
    (OUT_ROOT / "all_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"Results saved to {OUT_ROOT / 'all_results.json'}")


if __name__ == "__main__":
    main()
