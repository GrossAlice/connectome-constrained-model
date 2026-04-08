#!/usr/bin/env python
"""
Rerun T06_fir_K5_lag_nbr with the FIR/lag gradient-flow fix.

Key change vs the original sweep: rollout_weight=0.5 (was 0.0) so the
differentiable FIR/lag history buffers actually get exercised during
multi-step rollout training.
"""
import sys, os, time, json, shutil
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/T06_gradfix_rerun"

# ── T06 parameters (from sweep_fir_kernels) + rollout enabled ──────────
OVERRIDES = dict(
    # FIR kernel
    chemical_synapse_mode="fir",
    fir_kernel_len=5,
    fir_activation="identity",
    fir_include_reversal=False,
    # Lag
    lag_order=5,
    lag_neighbor=True,
    lag_connectome_mask="all",
    # Dynamics (same as original)
    edge_specific_G=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    per_neuron_amplitudes=True,
    learn_noise=True,
    noise_mode="heteroscedastic",
    # Training — the key difference: rollout is ON
    num_epochs=30,
    learning_rate=0.001,
    seed=42,
    cv_folds=3,
    rollout_steps=10,
    rollout_weight=0.5,
    rollout_starts=8,
    warmstart_rollout=True,
    # Eval
    skip_free_run=False,
    skip_cv_loo=False,
    eval_loo_subset_size=30,
    eval_loo_subset_mode="variance",
)


def main():
    save_dir = SAVE_ROOT
    os.makedirs(save_dir, exist_ok=True)

    cfg = make_config(H5, **OVERRIDES)

    print(f"=== T06 FIR K5 + lag_nbr  (gradient-flow fix, rollout_weight=0.5) ===")
    print(f"  save_dir: {save_dir}")
    t0 = time.time()

    results = train_stage2_cv(cfg, save_dir=save_dir)

    elapsed = (time.time() - t0) / 60
    summary = {
        "condition": "T06_fir_K5_lag_nbr_gradfix",
        "overrides": OVERRIDES,
        "elapsed_min": round(elapsed, 1),
    }
    for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
              "cv_fr_r2_mean", "cv_fr_r2_median", "best_fold_idx",
              "cv_loo_r2_mean", "cv_loo_r2_median"]:
        if k in results:
            summary[k] = results[k]

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f} min")
    print(f"  1-step R² mean:   {summary.get('cv_onestep_r2_mean', '?'):.4f}")
    print(f"  1-step R² median: {summary.get('cv_onestep_r2_median', '?'):.4f}")
    print(f"  LOO R² mean:      {summary.get('cv_loo_r2_mean', '?'):.4f}")
    print(f"  LOO R² median:    {summary.get('cv_loo_r2_median', '?'):.4f}")
    print(f"  (prev LOO mean was 0.2709, median 0.2068 without rollout)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
