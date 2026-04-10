#!/usr/bin/env python
"""
Regularisation & generalisation sweep
======================================

All conditions start from the CURRENT BEST config:
  - graph_poly_order=1  (best LOO from ablation_params)
  - lag_neighbor_per_type=True  (best from sweep_lag_biophysics)
  - IIR chemical synapses, corr_weighted init, 60 epochs

Block A: Coupling dropout  (cheap regulariser on synaptic coupling)
Block B: Input noise  (data augmentation — σ injected into u during training)
Block C: Dynamics L2  (weight decay on coupling parameters)
Block D: Rollout horizon / weight  (multi-step regularisation)
Block E: Combined best regulars
Block F: Lag order  (is K=5 optimal?)
Block G: More CV folds  (80% train instead of 67%)
Block H: LOO aux loss  (directly optimise LOO metric)
"""
import sys, os, time, json
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/sweep_regularization"

# ── Best config from sweeps so far ─────────────────────────────────────
_BASE = dict(
    seed=42,
    cv_folds=3,
    skip_free_run=True,
    skip_cv_loo=False,
    # Dynamics
    chemical_synapse_mode="iir",
    chemical_synapse_activation="sigmoid",
    edge_specific_G=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    per_neuron_amplitudes=True,
    graph_poly_order=1,            # ← best from ablation_params
    lag_neighbor_per_type=True,    # ← best from sweep_lag_biophysics
    tau_sv_init=(1, 5),
    tau_dcv_init=(3.0, 7.0),
    learn_noise=True,
    noise_mode="heteroscedastic",
    # Lag
    lag_order=5,
    lag_neighbor=True,
    lag_connectome_mask="all",
    lag_neighbor_activation="none",
    # Training
    num_epochs=60,
    rollout_steps=10,
    rollout_weight=0.3,
    rollout_starts=8,
    warmstart_rollout=True,
    learning_rate=0.001,
    synaptic_lr_multiplier=5.0,
)

CONDITIONS = {
    # ── A: Baseline with combined winners ───────────────────────────
    "A00_baseline": {
        **_BASE,
    },

    # ── B: Coupling dropout ─────────────────────────────────────────
    "B01_dropout_005": {
        **_BASE,
        "coupling_dropout": 0.05,
    },
    "B02_dropout_01": {
        **_BASE,
        "coupling_dropout": 0.1,
    },
    "B03_dropout_02": {
        **_BASE,
        "coupling_dropout": 0.2,
    },

    # ── C: Input noise ──────────────────────────────────────────────
    "C01_noise_005": {
        **_BASE,
        "input_noise_sigma": 0.05,
    },
    "C02_noise_01": {
        **_BASE,
        "input_noise_sigma": 0.1,
    },
    "C03_noise_02": {
        **_BASE,
        "input_noise_sigma": 0.2,
    },

    # ── D: Dynamics L2 ──────────────────────────────────────────────
    "D01_l2_1e-4": {
        **_BASE,
        "dynamics_l2": 1e-4,
    },
    "D02_l2_1e-3": {
        **_BASE,
        "dynamics_l2": 1e-3,
    },
    "D03_l2_1e-2": {
        **_BASE,
        "dynamics_l2": 1e-2,
    },

    # ── E: Rollout horizon / weight ─────────────────────────────────
    "E01_rollout_K20_w03": {
        **_BASE,
        "rollout_steps": 20,
        "rollout_weight": 0.3,
    },
    "E02_rollout_K20_w05": {
        **_BASE,
        "rollout_steps": 20,
        "rollout_weight": 0.5,
    },
    "E03_rollout_K30_w03": {
        **_BASE,
        "rollout_steps": 30,
        "rollout_weight": 0.3,
    },
    "E04_rollout_K30_w10": {
        **_BASE,
        "rollout_steps": 30,
        "rollout_weight": 1.0,
    },

    # ── F: Lag order ────────────────────────────────────────────────
    "F01_lag_3": {
        **_BASE,
        "lag_order": 3,
    },
    "F02_lag_7": {
        **_BASE,
        "lag_order": 7,
    },
    "F03_lag_10": {
        **_BASE,
        "lag_order": 10,
    },

    # ── G: More CV folds (more training data) ───────────────────────
    "G01_5fold": {
        **_BASE,
        "cv_folds": 5,
    },

    # ── H: LOO auxiliary loss ───────────────────────────────────────
    "H01_loo_aux_01": {
        **_BASE,
        "loo_aux_weight": 0.1,
        "loo_aux_steps": 10,
        "loo_aux_neurons": 10,
    },

    # ── I: Reversal potentials (per-neuron) ─────────────────────────
    "I01_reversals": {
        **_BASE,
        "learn_reversals": True,
        "reversal_mode": "per_neuron",
    },

    # ── J: Combined best (will refine after seeing B–F) ────────────
    "J01_dropout+rollout20": {
        **_BASE,
        "coupling_dropout": 0.1,
        "rollout_steps": 20,
        "rollout_weight": 0.3,
    },
    "J02_dropout+noise": {
        **_BASE,
        "coupling_dropout": 0.1,
        "input_noise_sigma": 0.1,
    },
    "J03_kitchen_sink": {
        **_BASE,
        "coupling_dropout": 0.1,
        "input_noise_sigma": 0.05,
        "rollout_steps": 20,
        "rollout_weight": 0.5,
        "learn_reversals": True,
        "reversal_mode": "per_neuron",
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
    print(f"Regularisation sweep: {len(CONDITIONS)} conditions")
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

    # ── Final summary ───────────────────────────────────────────────
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

        if tag == "A00_baseline" and loo_m is not None:
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
