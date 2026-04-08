#!/usr/bin/env python3
"""
Stage-2 Constraint-Feature Sweep
=================================

Tests the config parameters that were motivated by constraint-test comments
but never actually swept.  Each condition toggles ONE feature from the
current defaults so we can measure its isolated effect.

Features tested (from config.py comments):
  Test 5  — graph_poly_order > 1      (multi-hop gap propagation)
  Test 7  — behavior_weight_cap > 0   (L∞ cap on behaviour weights)
  Test 9  — noise_corr_rank > 0       (low-rank correlated noise)
  Extra   — per_neuron_tau_scale      (per-neuron tau multiplier)
  Extra   — lowrank_rank > 0          (non-connectome low-rank coupling)

Usage:
    python -u -m scripts.sweep_constraint_features \
        --worms 2022-08-02-01 2022-06-14-01 \
        --out output_plots/stage2/constraint_feature_sweep \
        --epochs 60 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────── #
#  Condition definitions                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def _make_conditions() -> OrderedDict:
    """Return OrderedDict[name → dict of config overrides].

    The 'baseline' condition uses the *current* config defaults so results
    are directly comparable.  Every other condition changes exactly one
    feature (or a small, tightly-coupled group).
    """
    C = OrderedDict()

    # ── 0. Baseline — current defaults ────────────────────────────────────
    C["baseline"] = {}

    # ── Test 5: multi-hop gap junction propagation ────────────────────────
    # Config comment: "lag plateau at K≈5-10 → multi-hop helps"
    C["T5_poly2"] = dict(graph_poly_order=2)
    C["T5_poly3"] = dict(graph_poly_order=3)

    # ── Test 7: behavior weight cap ───────────────────────────────────────
    # Config comment: "coupling gain ≈0.005-0.01 → cap stimulus weights"
    C["T7_beh_cap_0.01"] = dict(behavior_weight_cap=0.01)
    C["T7_beh_cap_0.1"]  = dict(behavior_weight_cap=0.1)

    # ── Test 9: low-rank noise correlations ───────────────────────────────
    # Config comment: "residual noise corr ≈0.05-0.10; low-rank Σ"
    C["T9_noise_rank5"]  = dict(noise_corr_rank=5)
    C["T9_noise_rank10"] = dict(noise_corr_rank=10)

    # ── Coupling dropout (regularise the connectome path) ─────────────────
    # Randomly zeros I_coupling per neuron during training.  Forces the model
    # to be robust to missing connectome inputs.
    C["cdrop_0.1"]  = dict(coupling_dropout=0.1)
    C["cdrop_0.2"]  = dict(coupling_dropout=0.2)
    C["cdrop_0.3"]  = dict(coupling_dropout=0.3)

    # ── Low-rank dense coupling ───────────────────────────────────────────
    # Config comment: "captures non-connectome interactions"
    C["lowrank_r5"]  = dict(lowrank_rank=5)
    C["lowrank_r10"] = dict(lowrank_rank=10)

    # ── Combo: linear + poly2 + noise rank 5 ─────────────────────────────
    C["combo_lin_poly2_noise5"] = dict(
        chemical_synapse_activation="identity",
        graph_poly_order=2,
        noise_corr_rank=5,
    )

    return C


# ─────────────────────────────────────────────────────────────────────────── #
#  Single-run wrapper                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def run_one(
    worm_h5: str,
    cond_name: str,
    overrides: dict,
    *,
    base_out: Path,
    epochs: int,
    device: str,
    cv_folds: int,
    loo_subset: int,
) -> dict:
    """Train + evaluate a single (worm, condition) and return summary."""
    from stage2.config import make_config
    from stage2.train import train_stage2

    tag = f"{Path(worm_h5).stem}_{cond_name}"
    save_dir = base_out / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    # Start from current defaults, then apply condition overrides
    kw = dict(overrides)
    kw["num_epochs"] = epochs
    kw["device"] = device
    kw["cv_folds"] = cv_folds
    kw["eval_loo_subset_size"] = loo_subset
    kw["eval_loo_subset_mode"] = "variance"
    kw["eval_loo_warmup_steps"] = 40
    kw["eval_loo_window_size"] = 50
    # Speed-up: skip posture video / stochastic plots
    kw["make_posture_video"] = False
    kw["n_freerun_samples"] = 0
    # Speed-up: disable LOO aux and rollout during training
    kw["loo_aux_neurons"] = 0
    kw["loo_aux_weight"] = 0
    kw["rollout_weight"] = 0
    kw["rollout_starts"] = 0
    # Speed-up: skip redundant final-eval pass (CV metrics suffice for sweep)
    kw["skip_final_eval"] = True
    # Speed-up: skip cached synaptic-state pass (saves ~15% per epoch)
    kw["warmstart_rollout"] = False

    cfg = make_config(worm_h5, **kw)

    t0 = time.time()
    try:
        result = train_stage2(cfg, save_dir=str(save_dir), show=False)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"tag": tag, "error": str(e)}
    elapsed = time.time() - t0

    summary: dict = {
        "tag": tag,
        "worm": Path(worm_h5).stem,
        "condition": cond_name,
        "overrides": {k: str(v) for k, v in overrides.items()},
        "epochs": epochs,
        "elapsed_s": round(elapsed, 1),
    }

    if result is not None:
        for k in (
            "cv_onestep_r2_mean", "cv_onestep_r2_median",
            "cv_loo_r2_mean", "cv_loo_r2_median",
            "cv_loo_r2_windowed_mean", "cv_loo_r2_windowed_median",
        ):
            v = result.get(k)
            summary[k] = (
                round(float(v), 4)
                if v is not None and np.isfinite(v)
                else None
            )

        cv_loo = result.get("cv_loo_r2")
        if cv_loo is not None:
            valid = cv_loo[np.isfinite(cv_loo)]
            summary["loo_n_neurons"]  = int(len(valid))
            summary["loo_n_positive"] = int((valid > 0).sum())
            summary["loo_q25"] = (
                round(float(np.percentile(valid, 25)), 4) if len(valid) else None
            )
            summary["loo_q75"] = (
                round(float(np.percentile(valid, 75)), 4) if len(valid) else None
            )

        # Behaviour-decoder R² if available
        beh = result.get("beh")
        if beh is not None:
            for bk in ("beh_r2_train", "beh_r2_val"):
                v = beh.get(bk) if isinstance(beh, dict) else None
                summary[bk] = round(float(v), 4) if v is not None else None

    return summary


# ─────────────────────────────────────────────────────────────────────────── #
#  Module-level worker for ProcessPoolExecutor (must be picklable)              #
# ─────────────────────────────────────────────────────────────────────────── #

def _run_worker(args_tuple):
    """Top-level function so ProcessPoolExecutor can pickle it."""
    h5, cname, overrides, base_out, epochs, device, cv_folds, loo_subset = args_tuple
    return run_one(
        h5, cname, overrides,
        base_out=Path(base_out),
        epochs=epochs,
        device=device,
        cv_folds=cv_folds,
        loo_subset=loo_subset,
    )


# ─────────────────────────────────────────────────────────────────────────── #
#  Pretty-print                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def _print_summary_table(results: list[dict]) -> None:
    header = (
        f"{'Tag':<40} {'1-step':>7} {'LOO-med':>8} {'LOO-w':>7} "
        f"{'#pos':>5} {'time':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['tag']:<40} ERROR: {r['error'][:35]}")
            continue
        onestep = r.get("cv_onestep_r2_median")
        loo_med = r.get("cv_loo_r2_median")
        loo_w   = r.get("cv_loo_r2_windowed_median")
        n_pos   = r.get("loo_n_positive", "?")
        elapsed = r.get("elapsed_s", 0)
        cols = []
        cols.append(f"{r['tag']:<40}")
        cols.append(f"{onestep:>7.4f}" if onestep is not None else f"{'N/A':>7}")
        cols.append(f"{loo_med:>8.4f}" if loo_med is not None else f"{'N/A':>8}")
        cols.append(f"{loo_w:>7.4f}" if loo_w is not None else f"{'N/A':>7}")
        cols.append(f"{n_pos:>5}")
        cols.append(f"{elapsed:>6.0f}s")
        print(" ".join(cols))


# ─────────────────────────────────────────────────────────────────────────── #
#  Main                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Sweep constraint-test features from stage2 config comments"
    )
    parser.add_argument(
        "--worms", nargs="+", default=["2022-08-02-01"],
    )
    parser.add_argument(
        "--out", default="output_plots/stage2/constraint_feature_sweep",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cv_folds", type=int, default=2)
    parser.add_argument(
        "--loo_subset", type=int, default=30,
        help="Neurons for LOO eval (0=all, 30=top-variance)",
    )
    parser.add_argument(
        "--conditions", nargs="*", default=None,
        help="Run only these conditions (default: all)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip conditions that already have results in sweep_results.json",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Run N conditions in parallel (each in a subprocess). "
             "Use N=2-4 on a single GPU, or more with multi-GPU.",
    )
    args = parser.parse_args()

    h5_dir = Path("data/used/behaviour+neuronal activity atanas (2023)/2")
    worm_paths = [str(h5_dir / f"{w}.h5") for w in args.worms]
    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)

    conditions = _make_conditions()
    if args.conditions:
        conditions = OrderedDict(
            (k, v) for k, v in conditions.items() if k in args.conditions
        )

    # ── Resume: load existing results and skip completed conditions ──
    all_results: list[dict] = []
    done_tags: set[str] = set()
    results_path = base_out / "sweep_results.json"
    if args.resume and results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        done_tags = {
            r["tag"] for r in all_results if "error" not in r
        }
        print(f"[resume] Loaded {len(all_results)} existing results, "
              f"{len(done_tags)} successful — will skip those.")

    # Build work list, skipping already-done items
    work: list[tuple[int, str, str, str, dict]] = []
    for wi, h5 in enumerate(worm_paths):
        for cname, overrides in conditions.items():
            tag = f"{Path(h5).stem}_{cname}"
            if tag in done_tags:
                continue
            work.append((wi, args.worms[wi], h5, cname, overrides))

    n_total_all = len(worm_paths) * len(conditions)
    n_todo = len(work)
    print(f"\n{'='*70}")
    print(f"Stage-2 Constraint-Feature Sweep")
    print(f"  Worms:      {args.worms}")
    print(f"  Conditions: {list(conditions.keys())}")
    print(f"  Total:      {n_total_all}  (skipping {n_total_all - n_todo} done)")
    print(f"  Remaining:  {n_todo}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  CV folds:   {args.cv_folds}")
    print(f"  LOO subset: {args.loo_subset}")
    print(f"  Parallel:   {args.parallel}")
    print(f"  Output:     {base_out}")
    print(f"{'='*70}\n")

    if n_todo == 0:
        print("Nothing to do — all conditions already completed.")
        _print_summary_table(all_results)
        return

    # Build picklable argument tuples for each work item
    worker_args = [
        (h5, cname, overrides, str(base_out), args.epochs,
         args.device, args.cv_folds, args.loo_subset)
        for (wi, worm_name, h5, cname, overrides) in work
    ]

    if args.parallel <= 1:
        # ── Sequential execution ──
        for idx, (item, wargs) in enumerate(zip(work, worker_args), 1):
            wi, worm_name, h5, cname, overrides = item
            print(f"\n{'#'*70}")
            print(f"  [{idx}/{n_todo}] worm={worm_name}  cond={cname}")
            print(f"{'#'*70}\n")
            sys.stdout.flush()

            summary = _run_worker(wargs)
            all_results.append(summary)
            _print_summary_table(all_results)

            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
    else:
        # ── Parallel execution via ProcessPoolExecutor ──
        # Use "fork" context: the main process hasn't done any CUDA work yet
        # (just arg-parsing & dir creation), so forking is safe and avoids
        # the pickling issues that "spawn" has with __main__-defined functions.
        import multiprocessing as mp
        print(f"Launching {min(args.parallel, n_todo)} workers ...")
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.parallel,
                                 mp_context=ctx) as pool:
            future_to_item = {
                pool.submit(_run_worker, wargs): item
                for item, wargs in zip(work, worker_args)
            }
            for fut in as_completed(future_to_item):
                item = future_to_item[fut]
                _, worm_name, _, cname, _ = item
                try:
                    summary = fut.result()
                except Exception as exc:
                    summary = {"tag": f"{worm_name}_{cname}",
                               "error": str(exc)}
                all_results.append(summary)
                _print_summary_table(all_results)

                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

    # ── Final ──
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    _print_summary_table(all_results)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
