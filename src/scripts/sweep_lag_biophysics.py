#!/usr/bin/env python
"""
Lag biophysics sweep
====================

Tests the effect of:
  1. Fixing lag mask direction (T_sv/T_dcv transposed — now default)
  2. Restricting neighbor lag to chemical-only (exclude T_e — instant)
  3. Restricting neighbor lag to T_e-only (test if electrical lag helps)
  4. Per-type separate lag weights
  5. IIR pure delay before SV / DCV exponential filters

Conditions:
  L00  Baseline (fixed direction, mask=all, delay=0)
  L01  mask=chem          (exclude T_e from neighbor lag)
  L02  mask=T_e           (only electrical neighbor lag)
  L03  per_type           (separate lag weights per connectome type)
  L04  per_type+chem_only (per_type but skip T_e lag entirely)
  L05  iir_delay_sv=1     (1-step pure delay before SV IIR)
  L06  iir_delay_sv=2     (2-step pure delay before SV IIR)
  L07  iir_delay_dcv=2    (2-step pure delay before DCV IIR — ~1.2s)
  L08  iir_delay_dcv=5    (5-step pure delay before DCV IIR — ~3.0s)
  L09  delay_both          (sv=1, dcv=3 — fast SV, slow DCV)
  L10  chem+delay_both    (mask=chem + sv=1, dcv=3)
  L11  no_lag_nbr         (lag_neighbor=False, self-lag only)
"""
import sys, os, time, json
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/sweep_lag_biophysics"

# ── B1 baseline settings ───────────────────────────────────────────────
_BASE = dict(
    seed=42,
    cv_folds=3,
    skip_free_run=True,
    skip_cv_loo=False,
    # Dynamics (B1 defaults)
    chemical_synapse_mode="iir",
    chemical_synapse_activation="sigmoid",
    edge_specific_G=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    per_neuron_amplitudes=True,
    graph_poly_order=2,
    tau_sv_init=(1, 5),
    tau_dcv_init=(3.0, 7.0),
    learn_noise=True,
    noise_mode="heteroscedastic",
    # Lag
    lag_order=5,
    lag_neighbor=True,
    lag_connectome_mask="all",
    lag_neighbor_activation="none",
    # Training (B1)
    num_epochs=60,
    rollout_steps=10,
    rollout_weight=0.3,
    rollout_starts=8,
    warmstart_rollout=True,
)

CONDITIONS = {
    # ── L00: Baseline (fixed mask direction, all types) ─────────────
    "L00_baseline": {
        **_BASE,
    },

    # ── Mask type tests ─────────────────────────────────────────────
    "L01_mask_chem": {
        **_BASE,
        "lag_connectome_mask": "chem",   # exclude T_e from neighbor lag
    },
    "L02_mask_T_e": {
        **_BASE,
        "lag_connectome_mask": "T_e",    # only electrical neighbor lag
    },

    # ── Per-type lag ────────────────────────────────────────────────
    "L03_per_type": {
        **_BASE,
        "lag_neighbor_per_type": True,
    },
    "L04_per_type_no_gap_lag": {
        # Per-type weights, but T_e edges excluded from lag mask
        # (forces model to use gap lag=0 effectively)
        **_BASE,
        "lag_connectome_mask": "chem",
        "lag_neighbor_per_type": True,
    },

    # ── IIR pure delay tests ────────────────────────────────────────
    "L05_delay_sv1": {
        **_BASE,
        "iir_delay_sv": 1,              # ~0.6s SV delay
    },
    "L06_delay_sv2": {
        **_BASE,
        "iir_delay_sv": 2,              # ~1.2s SV delay
    },
    "L07_delay_dcv2": {
        **_BASE,
        "iir_delay_dcv": 2,             # ~1.2s DCV delay
    },
    "L08_delay_dcv5": {
        **_BASE,
        "iir_delay_dcv": 5,             # ~3.0s DCV delay (neuropeptide timescale)
    },
    "L09_delay_both": {
        **_BASE,
        "iir_delay_sv": 1,              # fast SV: 1 step
        "iir_delay_dcv": 3,             # slow DCV: 3 steps (~1.8s)
    },
    "L10_chem_delay_both": {
        **_BASE,
        "lag_connectome_mask": "chem",
        "iir_delay_sv": 1,
        "iir_delay_dcv": 3,
    },

    # ── Control: no neighbor lag ────────────────────────────────────
    "L11_no_lag_nbr": {
        **_BASE,
        "lag_neighbor": False,           # self-lag only
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
    print(f"Lag biophysics sweep: {len(CONDITIONS)} conditions")
    print(f"{'='*70}\n")

    for tag, overrides in CONDITIONS.items():
        save_dir = os.path.join(SAVE_ROOT, tag)
        summary_path = os.path.join(save_dir, "summary.json")

        # Skip if already done
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

            # What changed vs baseline
            diff = {k: v for k, v in overrides.items()
                    if k not in _BASE or _BASE[k] != v}

            summary = {
                "condition": tag,
                "overrides": diff,
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
    print(f"\n{'='*70}")
    print(f"{'Condition':40s} {'1step':>8s} {'LOO mean':>10s} {'LOO med':>10s} {'Δ LOO':>8s}")
    print(f"{'-'*70}")

    baseline_loo = None
    for tag in CONDITIONS:
        r = all_results.get(tag, {})
        os_v = r.get("cv_onestep_r2_mean", None)
        loo_m = r.get("cv_loo_r2_mean", None)
        loo_d = r.get("cv_loo_r2_median", None)
        os_s = f"{os_v:.4f}" if os_v else "ERROR"
        lm_s = f"{loo_m:.4f}" if loo_m else "ERROR"
        ld_s = f"{loo_d:.4f}" if loo_d else "ERROR"

        if tag == "L00_baseline" and loo_m is not None:
            baseline_loo = loo_m
        delta = ""
        if baseline_loo is not None and loo_m is not None:
            d = loo_m - baseline_loo
            delta = f"{d:+.4f}"

        print(f"  {tag:38s} {os_s:>8s} {lm_s:>10s} {ld_s:>10s} {delta:>8s}")
    print(f"{'='*70}")

    with open(os.path.join(SAVE_ROOT, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {SAVE_ROOT}/all_results.json")


if __name__ == "__main__":
    main()
