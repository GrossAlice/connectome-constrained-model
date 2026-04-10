#!/usr/bin/env python
"""
Sweep: I_sv (IIR) × I_lag_sv (per-type neighbor lag) ablation
==============================================================

Runs the H-section (per-type neighbor lag) conditions together with a new
I-section that ablates the interaction between:

  • I_sv  — IIR exponential chemical synapse current  (always present in base)
  • I_lag_sv — per-type neighbor lag on SV/DCV edges  (lag_neighbor_per_type)

Key experimental question:
  Does I_lag_sv add value ON TOP of I_sv, or is it redundant?
  Can I_lag_sv REPLACE I_sv if the IIR path is linearised/removed?

I-section conditions:
  I0  pure IIR, no lag at all          — lag_order=0, lag_neighbor=False
  I1  IIR + self-lag only              — lag_neighbor=False
  I2  IIR(identity) + per-type sigmoid — linearise I_sv, keep nonlinear I_lag_sv
  I3  IIR(identity) + per-type sigmoid + lag10  — same but more lag capacity
  I4  IIR + per-type sigmoid + lag10   — full model, more capacity
  I5  IIR(identity) + per-type sigmoid + lag8 + 3-rank  — rich lag, weak IIR

All save to the same root as sweep_tau_lag_act for unified result collection.
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
    # ═════════════════════════════════════════════════════════════════
    #  H: lag_neighbor_per_type  (separate I_lag per synapse type)
    # ═════════════════════════════════════════════════════════════════
    "H0_pertype_none": {
        **_BASE,
        "lag_neighbor_per_type": True,
        "lag_neighbor_activation": "none",
    },
    "H1_pertype_sigmoid": {
        **_BASE,
        "lag_neighbor_per_type": True,
        "lag_neighbor_activation": "sigmoid",
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

    # ═════════════════════════════════════════════════════════════════
    #  I: I_sv × I_lag_sv ablation
    #
    #  Key question: does per-type neighbor lag (I_lag_sv) complement
    #  the IIR chemical synapse current (I_sv), or is it redundant?
    #
    #  Ablation axes:
    #    • IIR φ : sigmoid (default) vs identity (linearises I_sv)
    #    • Lag   : off / self-only / unified-nbr / per-type-nbr
    # ═════════════════════════════════════════════════════════════════

    # --- I0: pure IIR, NO lag at all ---
    # Removes autoregressive + neighbor lag entirely → isolates I_sv
    "I0_iir_only_no_lag": {
        **_BASE,
        "lag_order": 0,
        "lag_neighbor": False,
    },

    # --- I1: IIR + self-lag only (no neighbor lags) ---
    # Keeps autoregressive self but strips all neighbor-lag coupling
    "I1_iir_selflag_only": {
        **_BASE,
        "lag_neighbor": False,
    },

    # --- I2: identity IIR + per-type sigmoid ---
    # Linearises I_sv (identity act), but I_lag_sv keeps σ(u_j)
    # → tests if per-type lag can carry the nonlinear chemical signal
    "I2_identity_iir_pertype_sigmoid": {
        **_BASE,
        "chemical_synapse_activation": "identity",
        "lag_neighbor_per_type": True,
        "lag_neighbor_activation": "sigmoid",
    },

    # --- I3: identity IIR + per-type sigmoid + lag10 ---
    # Same as I2 but with more lag capacity
    "I3_identity_iir_pertype_sigmoid_lag10": {
        **_BASE,
        "chemical_synapse_activation": "identity",
        "lag_neighbor_per_type": True,
        "lag_neighbor_activation": "sigmoid",
        "lag_order": 10,
    },

    # --- I4: full IIR + per-type sigmoid + lag10 ---
    # Both I_sv (sigmoid IIR) and I_lag_sv (per-type sigmoid) active
    # → tests if they are complementary with extra lag capacity
    "I4_iir_pertype_sigmoid_lag10": {
        **_BASE,
        "lag_neighbor_per_type": True,
        "lag_neighbor_activation": "sigmoid",
        "lag_order": 10,
    },

    # --- I5: identity IIR + per-type sigmoid + lag8 + 3-rank ---
    # Rich per-type lag (8 lags, 3 tau ranks) with linearised IIR
    # → maximal lag capacity to compensate for weak IIR
    "I5_identity_iir_pertype_sig_lag8_3rank": {
        **_BASE,
        "chemical_synapse_activation": "identity",
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

    log_path = os.path.join(SAVE_ROOT, "run_HI.log")
    import builtins
    _print = builtins.print
    def tee(*args, **kwargs):
        kwargs_log = {k: v for k, v in kwargs.items() if k != 'file'}
        _print(*args, **kwargs)
        with open(log_path, "a") as flog:
            _print(*args, file=flog, **kwargs_log)
    builtins.print = tee

    print(f"{'='*70}")
    print(f"Sweep: I_sv × I_lag_sv ablation  (sections H + I)")
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
    print(f"{'Condition':48s} {'1step':>8s} {'LOO mean':>10s} {'LOO med':>10s}")
    print(f"{'-'*78}")

    sections = {}
    for tag in CONDITIONS:
        sec = tag[0]
        sections.setdefault(sec, []).append(tag)

    sec_names = {
        "H": "lag_neighbor_per_type",
        "I": "I_sv × I_lag_sv ablation",
    }

    for sec in sorted(sections):
        print(f"  ── {sec_names.get(sec, sec)} {'─'*40}")
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
            print(f"  {tag:46s} {os_s:>8s} {lm_s:>10s} {ld_s:>10s}{marker}")

    print(f"{'-'*78}")
    print(f"  {'Conn-Ridge reference':46s} {'---':>8s} {'0.3650':>10s} {'---':>10s}")
    print(f"  {'T07_iir_lag_nbr (baseline)':46s} {'0.7472':>8s} {'0.2958':>10s} {'0.2235':>10s}")
    print(f"{'='*78}")
    print(f"  * = beats T07 baseline (>0.296)")
    print(f"  ** = strong (>0.30)    *** = target hit (>0.35)")

    with open(os.path.join(SAVE_ROOT, "HI_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
