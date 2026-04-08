#!/usr/bin/env python
"""
Sweep: tau, lag order, FIR kernel length, activations, reversal, lag_neighbor_activation
========================================================================================

Starting from T07_iir_lag_nbr (LOO=0.296), explore:

  A) tau_sv / tau_dcv  — different time-scale combinations
  B) lag_order         — 3, 5, 8, 10, 15
  C) fir_kernel_len    — 3, 5, 8, 12  (FIR mode)
  D) chemical_synapse_activation — sigmoid, identity, softplus, tanh
  E) fir_include_reversal  — True (FIR mode only)
  F) lag_neighbor_activation — none vs sigmoid vs softplus vs tanh
     (NEW: applies φ(u_j) before neighbor-lag weighting, making it
      chemical-synapse-like instead of gap-junction-like)

Key insight: lag_neighbor currently operates on raw u_j (linear, like gap
junctions). With lag_neighbor_activation="sigmoid", the neighbor lags become
φ(u_j)-weighted — only active neighbours contribute, matching the
biophysics of chemical synaptic transmission.

All conditions: seed=42, cv_folds=3, skip_free_run=True.
"""
import sys, os, time, json
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/sweep_tau_lag_act"

# ── Shared base (T07-like) ─────────────────────────────────────────────
_BASE = dict(
    seed=42,
    cv_folds=3,
    skip_free_run=True,
    num_epochs=30,
    # Dynamics
    edge_specific_G=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    per_neuron_amplitudes=True,
    learn_noise=True,
    noise_mode="heteroscedastic",
    graph_poly_order=1,
    # IIR defaults
    chemical_synapse_mode="iir",
    chemical_synapse_activation="sigmoid",
    # Lag defaults (T07)
    lag_order=5,
    lag_neighbor=True,
    lag_connectome_mask="all",
    lag_neighbor_activation="none",
    lag_neighbor_per_type=False,
)

CONDITIONS = {
    # ═══════════════════════════════════════════════════════════════════
    #  A: tau time-scales
    # ═══════════════════════════════════════════════════════════════════
    "A0_tau_sv_1_5__dcv_3_7": {
        **_BASE,
        "tau_sv_init": (1.0, 5.0),
        "tau_dcv_init": (3.0, 7.0),
    },
    "A1_tau_sv_05_3__dcv_2_5": {
        **_BASE,
        "tau_sv_init": (0.5, 3.0),
        "tau_dcv_init": (2.0, 5.0),
    },
    "A2_tau_sv_02_1__dcv_1_3": {
        **_BASE,
        "tau_sv_init": (0.2, 1.0),
        "tau_dcv_init": (1.0, 3.0),
    },
    "A3_tau_sv_1_10__dcv_5_15": {
        **_BASE,
        "tau_sv_init": (1.0, 10.0),
        "tau_dcv_init": (5.0, 15.0),
    },
    "A4_tau_sv_05_5_10__dcv_3_7_15": {  # 3 ranks each
        **_BASE,
        "tau_sv_init": (0.5, 5.0, 10.0),
        "a_sv_init": (2.0, 0.8, 0.4),
        "tau_dcv_init": (3.0, 7.0, 15.0),
        "a_dcv_init": (0.8, 0.6, 0.3),
    },

    # ═══════════════════════════════════════════════════════════════════
    #  B: lag_order
    # ═══════════════════════════════════════════════════════════════════
    "B0_lag3": {
        **_BASE,
        "lag_order": 3,
    },
    "B1_lag8": {
        **_BASE,
        "lag_order": 8,
    },
    "B2_lag10": {
        **_BASE,
        "lag_order": 10,
    },
    "B3_lag15": {
        **_BASE,
        "lag_order": 15,
    },

    # ═══════════════════════════════════════════════════════════════════
    #  C: fir_kernel_len (FIR mode)
    # ═══════════════════════════════════════════════════════════════════
    "C0_fir_K3": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 3,
        "fir_activation": "softplus",
    },
    "C1_fir_K5": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 5,
        "fir_activation": "softplus",
    },
    "C2_fir_K8": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 8,
        "fir_activation": "softplus",
    },
    "C3_fir_K12": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 12,
        "fir_activation": "softplus",
    },

    # ═══════════════════════════════════════════════════════════════════
    #  D: chemical_synapse_activation (IIR mode)
    # ═══════════════════════════════════════════════════════════════════
    "D0_act_identity": {
        **_BASE,
        "chemical_synapse_activation": "identity",
    },
    "D1_act_softplus": {
        **_BASE,
        "chemical_synapse_activation": "softplus",
    },
    "D2_act_tanh": {
        **_BASE,
        "chemical_synapse_activation": "tanh",
    },
    "D3_act_relu": {
        **_BASE,
        "chemical_synapse_activation": "relu",
    },
    "D4_act_swish": {
        **_BASE,
        "chemical_synapse_activation": "swish",
    },

    # ═══════════════════════════════════════════════════════════════════
    #  E: fir_include_reversal (FIR mode)
    # ═══════════════════════════════════════════════════════════════════
    "E0_fir_K5_reversal": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 5,
        "fir_activation": "softplus",
        "fir_include_reversal": True,
        "learn_reversals": True,
    },
    "E1_fir_K8_reversal": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 8,
        "fir_activation": "softplus",
        "fir_include_reversal": True,
        "learn_reversals": True,
    },
    "E2_fir_K5_reversal_sigmoid": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 5,
        "fir_activation": "sigmoid",
        "fir_include_reversal": True,
        "learn_reversals": True,
    },

    # ═══════════════════════════════════════════════════════════════════
    #  F: lag_neighbor_activation  (NEW — the key experiment)
    #
    #  "none" = current behaviour (linear, gap-junction-like)
    #  "sigmoid"/"softplus"/"tanh" = apply φ(u_j) before neighbor-lag
    #    weighting → only active pre-synaptic neurons contribute,
    #    matching chemical synapse biophysics
    # ═══════════════════════════════════════════════════════════════════
    "F0_lagnbr_sigmoid": {
        **_BASE,
        "lag_neighbor_activation": "sigmoid",
    },
    "F1_lagnbr_softplus": {
        **_BASE,
        "lag_neighbor_activation": "softplus",
    },
    "F2_lagnbr_tanh": {
        **_BASE,
        "lag_neighbor_activation": "tanh",
    },
    "F3_lagnbr_relu": {
        **_BASE,
        "lag_neighbor_activation": "relu",
    },
    # lag_neighbor_activation + longer lag
    "F4_lagnbr_sigmoid_lag10": {
        **_BASE,
        "lag_neighbor_activation": "sigmoid",
        "lag_order": 10,
    },
    "F5_lagnbr_softplus_lag10": {
        **_BASE,
        "lag_neighbor_activation": "softplus",
        "lag_order": 10,
    },

    # ═══════════════════════════════════════════════════════════════════
    #  G: Combos — best guesses
    # ═══════════════════════════════════════════════════════════════════
    "G0_fast_tau_lag10_sigmoid_nbr": {
        **_BASE,
        "tau_sv_init": (0.2, 1.0),
        "tau_dcv_init": (1.0, 3.0),
        "lag_order": 10,
        "lag_neighbor_activation": "sigmoid",
    },
    "G1_slow_tau_lag8_softplus_nbr": {
        **_BASE,
        "tau_sv_init": (1.0, 10.0),
        "tau_dcv_init": (5.0, 15.0),
        "lag_order": 8,
        "lag_neighbor_activation": "softplus",
    },
    "G2_3rank_lag8_sigmoid_nbr": {
        **_BASE,
        "tau_sv_init": (0.5, 5.0, 10.0),
        "a_sv_init": (2.0, 0.8, 0.4),
        "tau_dcv_init": (3.0, 7.0, 15.0),
        "a_dcv_init": (0.8, 0.6, 0.3),
        "lag_order": 8,
        "lag_neighbor_activation": "sigmoid",
    },
    "G3_fir_K8_sigmoid_nbr_reversal": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 8,
        "fir_activation": "sigmoid",
        "fir_include_reversal": True,
        "learn_reversals": True,
        "lag_order": 8,
        "lag_neighbor_activation": "sigmoid",
    },
    "G4_identity_act_sigmoid_nbr_lag10": {
        **_BASE,
        "chemical_synapse_activation": "identity",
        "lag_order": 10,
        "lag_neighbor_activation": "sigmoid",
    },

    # ═════════════════════════════════════════════════════════════════
    #  H: lag_neighbor_per_type  (NEW — separate I_lag per synapse type)
    #
    #  Instead of a single _lag_G with a union mask, create separate
    #  per-type weights: _lag_G_e (gap junc), _lag_G_sv (SV), _lag_G_dcv (DCV).
    #  Gap junction lags stay linear; SV/DCV lags get the activation.
    #  This lets the model learn different temporal lag profiles per
    #  synapse type — fast electrical vs slow chemical.
    # ═════════════════════════════════════════════════════════════════
    "H0_pertype_none": {
        **_BASE,
        "lag_neighbor_per_type": True,
        "lag_neighbor_activation": "none",  # all linear
    },
    "H1_pertype_sigmoid": {
        **_BASE,
        "lag_neighbor_per_type": True,
        "lag_neighbor_activation": "sigmoid",  # σ(u_j) on SV/DCV, linear on gap
    },
    "H2_pertype_softplus": {
        **_BASE,
        "lag_neighbor_per_type": True,
        "lag_neighbor_activation": "softplus",
    },
    "H3_pertype_sigmoid_lag10": {
        **_BASE,
        "lag_neighbor_per_type": True,
        "lag_neighbor_activation": "sigmoid",
        "lag_order": 10,
    },
    "H4_pertype_sigmoid_lag8_fast_tau": {
        **_BASE,
        "lag_neighbor_per_type": True,
        "lag_neighbor_activation": "sigmoid",
        "lag_order": 8,
        "tau_sv_init": (0.2, 1.0),
        "tau_dcv_init": (1.0, 3.0),
    },
    "H5_pertype_sigmoid_lag8_3rank": {
        **_BASE,
        "lag_neighbor_per_type": True,
        "lag_neighbor_activation": "sigmoid",
        "lag_order": 8,
        "tau_sv_init": (0.5, 5.0, 10.0),
        "a_sv_init": (2.0, 0.8, 0.4),
        "tau_dcv_init": (3.0, 7.0, 15.0),
        "a_dcv_init": (0.8, 0.6, 0.3),
    },
}


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    all_results = {}

    log_path = os.path.join(SAVE_ROOT, "run.log")
    import builtins
    _print = builtins.print
    def tee(*args, **kwargs):
        _print(*args, **kwargs)
        with open(log_path, "a") as flog:
            _print(*args, file=flog, **kwargs)
    builtins.print = tee

    print(f"{'='*70}")
    print(f"Sweep: tau / lag / activation / reversal / lag_neighbor_activation")
    print(f"  {len(CONDITIONS)} conditions")
    print(f"  Baseline: T07_iir_lag_nbr  1step=0.747  LOO=0.296")
    print(f"  Target:   Conn-Ridge       LOO ≈ 0.365")
    print(f"{'='*70}\n")

    for tag, overrides in CONDITIONS.items():
        save_dir = os.path.join(SAVE_ROOT, tag)
        summary_path = os.path.join(save_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                prev = json.load(f)
            loo = prev.get("cv_loo_r2_mean", "?")
            print(f"[SKIP] {tag}  (LOO={loo})")
            all_results[tag] = prev
            continue

        os.makedirs(save_dir, exist_ok=True)
        print(f"\n[START] {tag}")
        t0 = time.time()

        cfg = make_config(H5, **overrides)

        try:
            results = train_stage2_cv(cfg, save_dir=save_dir)
            elapsed = (time.time() - t0) / 60

            # Build summary with the overrides that differ from base
            diff_keys = {k: v for k, v in overrides.items()
                         if k not in _BASE or _BASE[k] != v}
            summary = {
                "condition": tag,
                "overrides": diff_keys,
                "elapsed_min": round(elapsed, 1),
            }
            for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
                       "cv_fr_r2_mean", "cv_fr_r2_median", "best_fold_idx",
                       "cv_loo_r2_mean", "cv_loo_r2_median"]:
                if k in results:
                    summary[k] = results[k]

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            loo_mean = summary.get("cv_loo_r2_mean", "?")
            loo_med = summary.get("cv_loo_r2_median", "?")
            os_mean = summary.get("cv_onestep_r2_mean", "?")
            print(f"[DONE] {tag}  ({elapsed:.1f} min)")
            print(f"       1step={os_mean:.4f}  LOO mean={loo_mean:.4f}  median={loo_med:.4f}")
            all_results[tag] = summary

        except Exception as e:
            elapsed = (time.time() - t0) / 60
            print(f"[FAIL] {tag}  ({elapsed:.1f} min): {e}")
            import traceback; traceback.print_exc()
            all_results[tag] = {"condition": tag, "error": str(e)}

    # ── Final summary table ─────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"{'Condition':42s} {'1step':>8s} {'LOO mean':>10s} {'LOO med':>10s}")
    print(f"{'-'*78}")

    # Group by section letter
    sections = {}
    for tag in CONDITIONS:
        sec = tag[0]
        sections.setdefault(sec, []).append(tag)

    sec_names = {
        "A": "tau time-scales",
        "B": "lag_order",
        "C": "fir_kernel_len",
        "D": "chemical_synapse_activation",
        "E": "fir_include_reversal",
        "F": "lag_neighbor_activation",
        "G": "combos",
        "H": "lag_neighbor_per_type",
    }

    for sec in sorted(sections):
        print(f"  ── {sec_names.get(sec, sec)} {'─'*50}")
        for tag in sections[sec]:
            r = all_results.get(tag, {})
            os_v = r.get("cv_onestep_r2_mean", None)
            loo_m = r.get("cv_loo_r2_mean", None)
            loo_d = r.get("cv_loo_r2_median", None)
            os_s = f"{os_v:.4f}" if os_v else "ERROR"
            lm_s = f"{loo_m:.4f}" if loo_m else "ERROR"
            ld_s = f"{loo_d:.4f}" if loo_d else "ERROR"
            marker = " ***" if loo_m and loo_m > 0.35 else (
                     " **"  if loo_m and loo_m > 0.30 else (
                     " *"   if loo_m and loo_m > 0.296 else ""))
            print(f"  {tag:40s} {os_s:>8s} {lm_s:>10s} {ld_s:>10s}{marker}")

    print(f"{'-'*78}")
    print(f"  {'Conn-Ridge reference':40s} {'---':>8s} {'0.3650':>10s} {'---':>10s}")
    print(f"  {'T07_iir_lag_nbr (baseline)':40s} {'0.7472':>8s} {'0.2958':>10s} {'0.2235':>10s}")
    print(f"{'='*78}")
    print(f"  * = beats T07 baseline (>0.296)")
    print(f"  ** = strong (>0.30)    *** = target hit (>0.35)")

    with open(os.path.join(SAVE_ROOT, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
