#!/usr/bin/env python
"""
Targeted sweep: close the gap to Conn-Ridge LOO ≈ 0.365
========================================================

Current best: T07_iir_lag_nbr  LOO=0.296 (no rollout, no gradfix)
Target:       Conn-Ridge       LOO=0.365
              Conn-MLP         LOO=0.256  (already beaten by T07)

Strategy:  Start from T07 (IIR + lag_neighbor) — the current best — and
systematically vary the levers that should most affect LOO:

  A) Rollout (now with gradient-flow fix for FIR/lag buffers)
  B) More epochs (the IIR models converge fast, may need longer)
  C) Input noise (regularisation that helps LOO generalisation)
  D) Lag order (longer memory = more conn-ridge-like)
  E) Identity activation (remove softplus/sigmoid nonlinearity)
  F) FIR mode with rollout (fully differentiable temporal kernels)
  G) Dynamics L2 (ridge-like regularisation on coupling weights)

All conditions use: seed=42, cv_folds=3, skip_free_run=True (speed).
"""
import sys, os, time, json, shutil
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/close_gap_sweep"

# ── Shared base (T07-like) ─────────────────────────────────────────────
_BASE = dict(
    seed=42,
    cv_folds=3,
    skip_free_run=True,
    # Dynamics
    edge_specific_G=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    per_neuron_amplitudes=True,
    learn_noise=True,
    noise_mode="heteroscedastic",
    # Lag
    lag_order=5,
    lag_neighbor=True,
    lag_connectome_mask="all",
)

CONDITIONS = {
    # ── A: IIR baseline + rollout (gradient fix active) ─────────────
    "A0_iir_lag_rollout_w03": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "num_epochs": 30,
        "rollout_steps": 10,
        "rollout_weight": 0.3,
        "rollout_starts": 8,
        "warmstart_rollout": True,
    },
    "A1_iir_lag_rollout_w05": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "num_epochs": 30,
        "rollout_steps": 10,
        "rollout_weight": 0.5,
        "rollout_starts": 8,
        "warmstart_rollout": True,
    },

    # ── B: More epochs ──────────────────────────────────────────────
    "B0_iir_lag_60ep": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "num_epochs": 60,
    },
    "B1_iir_lag_60ep_rollout": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "num_epochs": 60,
        "rollout_steps": 10,
        "rollout_weight": 0.3,
        "rollout_starts": 8,
        "warmstart_rollout": True,
    },

    # ── C: Input noise (regularisation) ─────────────────────────────
    "C0_iir_lag_inoise_01": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "num_epochs": 30,
        "input_noise_sigma": 0.1,
    },
    "C1_iir_lag_inoise_02_rollout": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "num_epochs": 30,
        "input_noise_sigma": 0.2,
        "rollout_steps": 10,
        "rollout_weight": 0.3,
        "rollout_starts": 8,
        "warmstart_rollout": True,
    },

    # ── D: Longer lag (more temporal memory) ────────────────────────
    "D0_iir_lag10": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "lag_order": 10,
        "num_epochs": 30,
    },
    "D1_iir_lag10_rollout": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "lag_order": 10,
        "num_epochs": 30,
        "rollout_steps": 10,
        "rollout_weight": 0.3,
        "rollout_starts": 8,
        "warmstart_rollout": True,
    },

    # ── E: Identity activation (linear, conn-ridge-like) ───────────
    "E0_iir_identity_lag": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "chemical_synapse_activation": "identity",
        "num_epochs": 30,
    },
    "E1_iir_identity_lag_rollout": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "chemical_synapse_activation": "identity",
        "num_epochs": 30,
        "rollout_steps": 10,
        "rollout_weight": 0.3,
        "rollout_starts": 8,
        "warmstart_rollout": True,
    },

    # ── F: FIR + rollout (now with gradient flow!) ──────────────────
    "F0_fir_K5_lag_rollout": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 5,
        "fir_activation": "identity",
        "num_epochs": 30,
        "rollout_steps": 10,
        "rollout_weight": 0.3,
        "rollout_starts": 8,
        "warmstart_rollout": True,
    },
    "F1_fir_K8_lag_rollout": {
        **_BASE,
        "chemical_synapse_mode": "fir",
        "fir_kernel_len": 8,
        "fir_activation": "identity",
        "num_epochs": 30,
        "rollout_steps": 10,
        "rollout_weight": 0.3,
        "rollout_starts": 8,
        "warmstart_rollout": True,
    },

    # ── G: Dynamics L2 (ridge-like regularisation) ──────────────────
    "G0_iir_lag_l2_1e-3": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "num_epochs": 30,
        "dynamics_l2": 1e-3,
    },
    "G1_iir_lag_l2_1e-2_rollout": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "num_epochs": 30,
        "dynamics_l2": 1e-2,
        "rollout_steps": 10,
        "rollout_weight": 0.3,
        "rollout_starts": 8,
        "warmstart_rollout": True,
    },

    # ── H: Kitchen sink (best guesses combined) ─────────────────────
    "H0_iir_lag10_inoise_rollout_60ep": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "lag_order": 10,
        "num_epochs": 60,
        "input_noise_sigma": 0.1,
        "rollout_steps": 10,
        "rollout_weight": 0.3,
        "rollout_starts": 8,
        "warmstart_rollout": True,
        "dynamics_l2": 1e-3,
    },
    "H1_iir_identity_lag10_rollout_60ep": {
        **_BASE,
        "chemical_synapse_mode": "iir",
        "chemical_synapse_activation": "identity",
        "lag_order": 10,
        "num_epochs": 60,
        "rollout_steps": 10,
        "rollout_weight": 0.3,
        "rollout_starts": 8,
        "warmstart_rollout": True,
        "dynamics_l2": 1e-3,
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
    print(f"Close-the-gap sweep: {len(CONDITIONS)} conditions")
    print(f"Target: Conn-Ridge LOO ≈ 0.365")
    print(f"Current best: T07_iir_lag_nbr LOO = 0.296")
    print(f"{'='*70}\n")

    for tag, overrides in CONDITIONS.items():
        save_dir = os.path.join(SAVE_ROOT, tag)
        # Skip if already done
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

        # Extract epochs override
        n_ep = overrides.pop("_epochs", None)
        cfg = make_config(H5, **overrides)
        if n_ep is not None:
            cfg.num_epochs = n_ep

        try:
            results = train_stage2_cv(cfg, save_dir=save_dir)
            elapsed = (time.time() - t0) / 60

            summary = {
                "condition": tag,
                "overrides": {k: v for k, v in overrides.items()
                              if k not in _BASE or _BASE[k] != v},
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
    print(f"{'Condition':40s} {'1step':>8s} {'LOO mean':>10s} {'LOO med':>10s}")
    print(f"{'-'*70}")
    for tag in CONDITIONS:
        r = all_results.get(tag, {})
        os_v = r.get("cv_onestep_r2_mean", None)
        loo_m = r.get("cv_loo_r2_mean", None)
        loo_d = r.get("cv_loo_r2_median", None)
        os_s = f"{os_v:.4f}" if os_v else "ERROR"
        lm_s = f"{loo_m:.4f}" if loo_m else "ERROR"
        ld_s = f"{loo_d:.4f}" if loo_d else "ERROR"
        marker = " ***" if loo_m and loo_m > 0.35 else (" **" if loo_m and loo_m > 0.30 else "")
        print(f"  {tag:38s} {os_s:>8s} {lm_s:>10s} {ld_s:>10s}{marker}")
    print(f"{'-'*70}")
    print(f"  {'Conn-Ridge reference':38s} {'---':>8s} {'0.3650':>10s} {'---':>10s}")
    print(f"  {'T07_iir_lag_nbr (prev best)':38s} {'0.7472':>8s} {'0.2958':>10s} {'0.2235':>10s}")
    print(f"{'='*70}")

    with open(os.path.join(SAVE_ROOT, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
