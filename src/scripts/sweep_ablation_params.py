#!/usr/bin/env python
"""
Ablation sweep: one parameter at a time vs B1 baseline
=======================================================

Baseline = B1_iir_lag_60ep_rollout (current best: LOO=0.322, 1step=0.782)

Tests (each changes exactly ONE thing from baseline):
  T00  Baseline (reproduce B1)
  T01  graph_poly_order=1  (disable 2nd-order Laplacian hops)
  T02  graph_poly_order=3  (add 3rd-order hops)
  T03  per_neuron_amplitudes=False  (shared amplitudes instead of per-neuron)
  T04  learn_reversals=True, reversal_mode="per_neuron"
  T05  learn_reversals=True, reversal_mode="per_edge"
  T06  fir_kernel_len=1   (FIR=1 for reference, only used if mode=fir)
  T07  fir_kernel_len=5
  T08  fir_kernel_len=10
  T09  fir_include_reversal=True
  T10  fix_tau_sv=False  (learn SV time-constants)
"""
import sys, os, time, json
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/ablation_params"

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
    # ── T00: Baseline (reproduce B1) ────────────────────────────────
    "T00_baseline": {
        **_BASE,
    },

    # ── graph_poly_order ────────────────────────────────────────────
    "T01_poly_order_1": {
        **_BASE,
        "graph_poly_order": 1,
    },
    "T02_poly_order_3": {
        **_BASE,
        "graph_poly_order": 3,
    },

    # ── per_neuron_amplitudes ───────────────────────────────────────
    "T03_shared_amplitudes": {
        **_BASE,
        "per_neuron_amplitudes": False,
    },

    # ── learn_reversals ─────────────────────────────────────────────
    "T04_learn_rev_per_neuron": {
        **_BASE,
        "learn_reversals": True,
        "reversal_mode": "per_neuron",
    },
    "T05_learn_rev_per_edge": {
        **_BASE,
        "learn_reversals": True,
        "reversal_mode": "per_edge",
    },

    # ── fix_tau_sv ──────────────────────────────────────────────────
    "T06_learn_tau_sv": {
        **_BASE,
        "fix_tau_sv": False,
    },

    # ── fir_kernel_len (switch to FIR mode) ─────────────────────────
    "T07_fir_K1": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 1,
        "fir_activation": "softplus",
    },
    "T08_fir_K5": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 5,
        "fir_activation": "softplus",
    },
    "T09_fir_K10": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 10,
        "fir_activation": "softplus",
    },

    # ── fir_include_reversal (in FIR mode) ──────────────────────────
    "T10_fir_K5_reversal": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 5,
        "fir_activation": "softplus",
        "fir_include_reversal": True,
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
    print(f"Ablation sweep: {len(CONDITIONS)} conditions")
    print(f"Baseline: B1_iir_lag_60ep_rollout  LOO=0.322  1step=0.782")
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

            # Compute what changed vs baseline
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

        if tag == "T00_baseline" and loo_m is not None:
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
