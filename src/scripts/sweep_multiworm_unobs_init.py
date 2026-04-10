#!/usr/bin/env python
"""
Multi-worm sweep: unobserved neuron initialisation strategies
=============================================================

Tests whether SV/DCV-informed initialisation of unobserved neuron traces
improves multi-worm training compared to gap-junction-only init or no init.

Uses the 2 worms from "the same neurons" folder (97 shared neurons each).
With atlas_min_worm_count=0 → full 302-neuron atlas → 205 unobserved/worm.

Best single-worm config (T01_poly_order_1, LOO=0.332) is used as base.

Conditions
----------
  M00  no_init            infer_unobserved=True, connectome_init=False, zeros start
  M01  gap_init           infer_unobserved=True, connectome_init=True, mode="gap"
  M02  svdcv_init         infer_unobserved=True, connectome_init=True, mode="svdcv"
  M03  svdcv_lowrank      infer_unobserved=True, svdcv init + low_rank mode
  M04  no_unobs           infer_unobserved=False (only shared 97 neurons matter)
  M05  svdcv_warmup40     svdcv init + 40 warmup epochs (vs default 20)
"""
import sys, os, time, json, gc

os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

from stage2.config import make_config
from stage2.train_multi import train_multi_worm

DATA_DIR = "data/used/behaviour+neuronal activity atanas (2023)/the same neurons"
H5_PATHS = tuple(sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR) if f.endswith(".h5")
]))
SAVE_ROOT = "output_plots/stage2/multiworm_unobs_init"

# ── Best single-worm config (T01_poly_order_1) as base ─────────────────
_BASE = dict(
    seed=42,
    # Multi-worm settings
    multi_worm=True,
    h5_paths=H5_PATHS,
    atlas_min_worm_count=0,       # full 302-neuron atlas
    common_dt=0.6,
    require_stage1=True,
    val_frac=0.2,
    worm_weight_mode="equal",
    # Unobserved neuron defaults
    infer_unobserved=True,
    u_unobs_lr=0.01,
    u_unobs_inner_steps=10,
    u_unobs_smoothness=0.01,
    sigma_u_unobs_scale=2.0,
    u_unobs_connectome_init=True,
    u_unobs_init_mode="gap",
    u_unobs_init_alpha=1.0,
    u_unobs_warmup_epochs=20,
    u_unobs_warmup_lr=0.01,
    u_unobs_low_rank=False,
    # Dynamics — best from T01_poly_order_1
    chemical_synapse_mode="iir",
    chemical_synapse_activation="sigmoid",
    edge_specific_G=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    per_neuron_amplitudes=True,
    graph_poly_order=1,           # T01 best: disable 2nd-order hops
    tau_sv_init=(1, 5),
    tau_dcv_init=(3.0, 7.0),
    learn_noise=True,
    noise_mode="heteroscedastic",
    # Lag neighbour
    lag_order=5,
    lag_neighbor=True,
    lag_connectome_mask="all",
    # Training
    device="cuda",
    num_epochs=60,
    learning_rate=1e-3,
    rollout_steps=10,
    rollout_weight=0.3,
    rollout_starts=8,
    warmstart_rollout=True,
    grad_clip_norm=1.0,
    print_every=5,
    # Output
    skip_free_run=True,
    plot_every=0,                 # no intermediate plots (speed)
)

CONDITIONS = {
    # ── M00: No connectome init (zeros) ──────────────────────────────
    "M00_no_init": {
        **_BASE,
        "u_unobs_connectome_init": False,
    },

    # ── M01: Gap-junction only init (existing) ──────────────────────
    "M01_gap_init": {
        **_BASE,
        "u_unobs_connectome_init": True,
        "u_unobs_init_mode": "gap",
    },

    # ── M02: SV/DCV-informed init ───────────────────────────────────
    "M02_svdcv_init": {
        **_BASE,
        "u_unobs_connectome_init": True,
        "u_unobs_init_mode": "svdcv",
        "u_unobs_init_weight_e": 1.0,
        "u_unobs_init_weight_sv": 1.0,
        "u_unobs_init_weight_dcv": 0.5,
    },

    # ── M03: SV/DCV init + low-rank parameterisation ────────────────
    "M03_svdcv_lowrank": {
        **_BASE,
        "u_unobs_connectome_init": True,
        "u_unobs_init_mode": "svdcv",
        "u_unobs_init_weight_e": 1.0,
        "u_unobs_init_weight_sv": 1.0,
        "u_unobs_init_weight_dcv": 0.5,
        "u_unobs_low_rank": True,
    },

    # ── M04: Disable unobserved inference entirely ──────────────────
    "M04_no_unobs": {
        **_BASE,
        "infer_unobserved": False,
        "u_unobs_connectome_init": False,
        "u_unobs_warmup_epochs": 0,
    },

    # ── M05: SV/DCV init + extended warmup ──────────────────────────
    "M05_svdcv_warmup40": {
        **_BASE,
        "u_unobs_connectome_init": True,
        "u_unobs_init_mode": "svdcv",
        "u_unobs_init_weight_e": 1.0,
        "u_unobs_init_weight_sv": 1.0,
        "u_unobs_init_weight_dcv": 0.5,
        "u_unobs_warmup_epochs": 40,
    },
}


def main():
    import torch

    os.makedirs(SAVE_ROOT, exist_ok=True)
    all_results = {}

    print(f"H5 files: {H5_PATHS}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Save root: {SAVE_ROOT}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    for tag, overrides in CONDITIONS.items():
        save_dir = os.path.join(SAVE_ROOT, tag)
        summary_path = os.path.join(save_dir, "summary.json")

        # Skip if already done
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                prev = json.load(f)
            val = prev.get("best_val", "?")
            print(f"[SKIP] {tag}  (best_val={val})")
            all_results[tag] = prev
            continue

        os.makedirs(save_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"[START] {tag}")
        print(f"{'='*60}")
        t0 = time.time()

        cfg = make_config(H5_PATHS[0], **overrides)

        try:
            results = train_multi_worm(cfg, save_dir=save_dir)
            elapsed = (time.time() - t0) / 60

            # Extract key metrics
            best_val = results.get("best_val", float("nan"))
            worm_res = results.get("worm_results", {})

            # Per-worm val losses
            per_worm_val = {k: v.get("val_loss", float("nan"))
                           for k, v in worm_res.items()}

            # Compute what changed vs baseline
            diff = {k: v for k, v in overrides.items()
                    if k not in _BASE or _BASE[k] != v}

            summary = {
                "condition": tag,
                "overrides": diff,
                "elapsed_min": round(elapsed, 1),
                "best_val": float(best_val),
                "per_worm_val": per_worm_val,
                "n_worms": len(worm_res),
            }

            # Try to extract eval results if available
            eval_res = results.get("eval", {})
            if eval_res:
                for metric in ["onestep_r2_mean", "loo_r2_mean",
                               "onestep_r2_per_worm", "loo_r2_per_worm"]:
                    if metric in eval_res:
                        val = eval_res[metric]
                        if isinstance(val, dict):
                            summary[metric] = {k: float(v) for k, v in val.items()}
                        else:
                            summary[metric] = float(val)

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\n[DONE] {tag}  ({elapsed:.1f} min)")
            print(f"       best_val={best_val:.5f}")
            for wid, vl in per_worm_val.items():
                print(f"       {wid}: val_loss={vl:.5f}")
            all_results[tag] = summary

        except Exception as e:
            elapsed = (time.time() - t0) / 60
            print(f"\n[FAIL] {tag}  ({elapsed:.1f} min): {e}")
            import traceback; traceback.print_exc()
            all_results[tag] = {"condition": tag, "error": str(e)}

        # Free GPU memory between conditions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final summary table ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Condition':35s} {'best_val':>10s} {'time(min)':>10s}")
    print(f"{'-'*70}")

    for tag in CONDITIONS:
        r = all_results.get(tag, {})
        bv = r.get("best_val", None)
        tm = r.get("elapsed_min", None)
        bv_s = f"{bv:.5f}" if bv is not None else "ERROR"
        tm_s = f"{tm:.1f}" if tm is not None else "?"
        print(f"  {tag:33s} {bv_s:>10s} {tm_s:>10s}")
    print(f"{'='*70}")

    with open(os.path.join(SAVE_ROOT, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {SAVE_ROOT}/all_results.json")


if __name__ == "__main__":
    main()
