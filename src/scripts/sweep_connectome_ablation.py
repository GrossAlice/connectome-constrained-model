#!/usr/bin/env python
"""
Connectome-type ablation sweep
==============================

Tests which connectome types (T_e / T_sv / T_dcv) matter by switching
each current pathway on/off.  Also tests synapse_lag_taps — linear FIR
through the chemical connectome.

Conditions:
  C00  Baseline (all on)
  C01  No gap junctions          (use_gap_junctions=False)
  C02  No SV synapses            (use_sv_synapses=False)
  C03  No DCV synapses           (use_dcv_synapses=False)
  C04  No chemical (SV+DCV off)  (use_sv=False, use_dcv=False)
  C05  Gap only                  (sv=False, dcv=False)
  C06  SV only                   (gap=False, dcv=False)
  C07  DCV only                  (gap=False, sv=False)
  C08  No connectome currents    (gap=False, sv=False, dcv=False)
  C09  syn_lag_taps=3            (synapse FIR through T_sv/T_dcv, K=3)
  C10  syn_lag_taps=5            (K=5)
  C11  syn_lag_taps=5, no lag    (replace I_lag with synapse-routed lag)
  C12  syn_lag_taps=5, no IIR    (SV/DCV off, replaced by synapse lag)
"""
import sys, os, time, json
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/connectome_ablation"

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
    # Training (B1)
    num_epochs=60,
    rollout_steps=10,
    rollout_weight=0.3,
    rollout_starts=8,
    warmstart_rollout=True,
)

CONDITIONS = {
    # ── C00: Baseline (all on) ──────────────────────────────────────
    "C00_baseline": {
        **_BASE,
    },

    # ── Single knockouts ────────────────────────────────────────────
    "C01_no_gap": {
        **_BASE,
        "use_gap_junctions": False,
    },
    "C02_no_sv": {
        **_BASE,
        "use_sv_synapses": False,
    },
    "C03_no_dcv": {
        **_BASE,
        "use_dcv_synapses": False,
    },

    # ── Combined knockouts ──────────────────────────────────────────
    "C04_no_chem": {
        **_BASE,
        "use_sv_synapses": False,
        "use_dcv_synapses": False,
    },
    "C05_gap_only": {
        **_BASE,
        "use_sv_synapses": False,
        "use_dcv_synapses": False,
    },
    "C06_sv_only": {
        **_BASE,
        "use_gap_junctions": False,
        "use_dcv_synapses": False,
    },
    "C07_dcv_only": {
        **_BASE,
        "use_gap_junctions": False,
        "use_sv_synapses": False,
    },
    "C08_no_connectome": {
        **_BASE,
        "use_gap_junctions": False,
        "use_sv_synapses": False,
        "use_dcv_synapses": False,
    },

    # ── Synapse-routed linear lag (I_lag,sv) ─────────────────────────
    "C09_syn_lag_3": {
        **_BASE,
        "synapse_lag_taps": 3,
    },
    "C10_syn_lag_5": {
        **_BASE,
        "synapse_lag_taps": 5,
    },
    "C11_syn_lag_5_no_lag": {
        **_BASE,
        "synapse_lag_taps": 5,
        "lag_order": 0,
        "lag_neighbor": False,
    },
    "C12_syn_lag_5_no_iir": {
        **_BASE,
        "synapse_lag_taps": 5,
        "use_sv_synapses": False,
        "use_dcv_synapses": False,
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
    print(f"Connectome ablation sweep: {len(CONDITIONS)} conditions")
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

        if tag == "C00_baseline" and loo_m is not None:
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
