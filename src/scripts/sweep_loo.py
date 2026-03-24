#!/usr/bin/env python
"""Quick 10-epoch hyperparameter sweep for single-worm LOO R².

Uses current config.py defaults as baseline.  Each run trains for 10 epochs,
then evaluates one-step + LOO (on a small subset) + free-run.
Results are collected into a comparison table at the end.

Run:  .venv/bin/python -u scripts/sweep_loo.py
"""
from __future__ import annotations

import sys, os, json, time, traceback, gc
from pathlib import Path
import numpy as np
import torch

# Force unbuffered output so nohup / tee work properly
os.environ["PYTHONUNBUFFERED"] = "1"

# Ensure the src directory is on sys.path so `stage2` is importable
_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

# ── sweep configs ────────────────────────────────────────────────────────
H5 = "data/used/behaviour+neuronal activity atanas (2023)/the same neurons/2023-01-23-21.h5"
BASE_DIR = "output_plots/stage2/sweep_loo"

# Common overrides applied to EVERY run:
#  - 10 epochs for speed
#  - no intermediate plots
#  - no posture video
#  - small LOO subset (10 neurons) for fast eval
#  - fewer diagnostics
COMMON = dict(
    num_epochs=10,
    plot_every=0,
    make_posture_video=False,
    eval_loo_subset_size=10,
    diagnostics_every=0,
)

# Current config.py defaults (for reference):
#   num_epochs=100, learning_rate=0.001, rollout_steps=5, rollout_weight=0.1,
#   rollout_starts=8, warmstart_rollout=False, synaptic_lr_multiplier=5.0,
#   per_neuron_amplitudes=True, learn_sigma_u=True, fix_a_sv=False, fix_a_dcv=False,
#   learn_W_sv=False, learn_W_dcv=False, network_init_mode="global"

RUNS = [
    # ── 0. Baseline: current defaults, 10 ep ──
    ("00_baseline", {}),

    # ── 1-2. Break a×W degeneracy ──
    ("01_fix_a", dict(fix_a_sv=True, fix_a_dcv=True)),

    ("02_fix_a_noperneuron", dict(fix_a_sv=True, fix_a_dcv=True,
                                   per_neuron_amplitudes=False)),

    # ── 3-5. Rollout strength (with fix_a) ──
    ("03_fix_a_roll03", dict(fix_a_sv=True, fix_a_dcv=True,
                             rollout_weight=0.3)),

    ("04_fix_a_roll05", dict(fix_a_sv=True, fix_a_dcv=True,
                             rollout_weight=0.5)),

    ("05_fix_a_roll10", dict(fix_a_sv=True, fix_a_dcv=True,
                             rollout_weight=1.0)),

    # ── 6-7. Rollout horizon ──
    ("06_fix_a_steps10", dict(fix_a_sv=True, fix_a_dcv=True,
                              rollout_steps=10)),

    ("07_fix_a_steps20", dict(fix_a_sv=True, fix_a_dcv=True,
                              rollout_steps=20, rollout_weight=0.3)),

    # ── 8. Warmstart rollout ──
    ("08_fix_a_warmstart", dict(fix_a_sv=True, fix_a_dcv=True,
                                warmstart_rollout=True,
                                rollout_weight=0.3, rollout_steps=10)),

    # ── 9-10. Learning rate ──
    ("09_fix_a_lr3e3", dict(fix_a_sv=True, fix_a_dcv=True,
                            learning_rate=0.003)),

    ("10_fix_a_lr3e4", dict(fix_a_sv=True, fix_a_dcv=True,
                            learning_rate=0.0003)),

    # ── 11-12. Synaptic LR multiplier ──
    ("11_fix_a_synlr10", dict(fix_a_sv=True, fix_a_dcv=True,
                              synaptic_lr_multiplier=10.0)),

    ("12_fix_a_synlr20", dict(fix_a_sv=True, fix_a_dcv=True,
                              synaptic_lr_multiplier=20.0)),

    # ── 13. Learn W (edge weights) ──
    ("13_fix_a_learnW", dict(fix_a_sv=True, fix_a_dcv=True,
                             learn_W_sv=True, learn_W_dcv=True)),

    # ── 14. LOO auxiliary loss ──
    ("14_fix_a_looaux", dict(fix_a_sv=True, fix_a_dcv=True,
                             loo_aux_weight=0.1, loo_aux_neurons=4)),

    # ── 15. No sigma_u learning (pure MSE) ──
    ("15_fix_a_nosigma", dict(fix_a_sv=True, fix_a_dcv=True,
                              learn_sigma_u=False)),

    # ── 16. Per-neuron ridge init ──
    ("16_fix_a_pnridge", dict(fix_a_sv=True, fix_a_dcv=True,
                              network_init_mode="per_neuron")),

    # ── 17. Kitchen sink: best guesses combined ──
    ("17_combo", dict(fix_a_sv=True, fix_a_dcv=True,
                      rollout_weight=0.3, rollout_steps=10,
                      warmstart_rollout=True,
                      synaptic_lr_multiplier=10.0)),
]


def run_one(name: str, overrides: dict) -> dict:
    """Train single-worm with given overrides, return summary metrics."""
    from stage2.config import make_config
    from stage2.train import train_stage2

    save_dir = str(Path(BASE_DIR) / name)
    kw = {**COMMON, **overrides}
    cfg = make_config(H5, **kw)

    print(f"\n{'='*72}", flush=True)
    print(f"  RUN: {name}", flush=True)
    print(f"  Overrides: {overrides}", flush=True)
    print(f"  Save dir:  {save_dir}", flush=True)
    print(f"{'='*72}\n", flush=True)

    t0 = time.time()
    try:
        eval_result = train_stage2(cfg, save_dir=save_dir, show=False)
    except Exception as exc:
        print(f"[SWEEP] train_stage2 failed for {name}: {exc}", flush=True)
        traceback.print_exc()
        return {"name": name, "error": str(exc), "time_s": round(time.time() - t0, 1)}
    elapsed = time.time() - t0

    summary = {"name": name, "time_s": round(elapsed, 1)}

    # Extract metrics from the eval_result (returned by generate_eval_loo_plots)
    # and from the run.log
    try:
        run_log = Path(save_dir) / "run.log"
        if run_log.exists():
            lines = run_log.read_text().splitlines()
            # Get last epoch line for loss
            for line in reversed(lines):
                if "ep 10/" in line or "ep 10 " in line:
                    summary["last_epoch_line"] = line.strip()
                    break
    except Exception:
        pass

    # Read metrics from the eval that was already run by train_stage2
    try:
        if eval_result is not None:
            r2_os = eval_result.get("onestep", {}).get("r2")
            r2_loo = eval_result.get("loo", {}).get("r2")
            r2_fr = eval_result.get("free_run", {}).get("r2")

            if r2_os is not None:
                summary["os_med"] = round(float(np.nanmedian(r2_os)), 4)
                summary["os_mean"] = round(float(np.nanmean(r2_os)), 4)
            if r2_loo is not None:
                fin = r2_loo[np.isfinite(r2_loo)]
                if fin.size > 0:
                    summary["loo_med"] = round(float(np.nanmedian(fin)), 4)
                    summary["loo_mean"] = round(float(np.nanmean(fin)), 4)
                    summary["loo_pos"] = round(float(np.mean(fin > 0)), 4)
                    summary["loo_max"] = round(float(np.nanmax(fin)), 4)
            if r2_fr is not None:
                fin = r2_fr[np.isfinite(r2_fr)]
                if fin.size > 0:
                    summary["fr_med"] = round(float(np.nanmedian(fin)), 4)
    except Exception as exc:
        summary["eval_parse_error"] = str(exc)

    # If eval_result didn't have the R² arrays, fall back to loading model
    if "loo_med" not in summary:
        try:
            from stage2.config import make_config as _mc
            from stage2.io_h5 import load_data_pt
            from stage2.evaluate import (compute_onestep, run_loo_all,
                                         compute_free_run, choose_loo_subset)

            cfg2 = _mc(H5, eval_loo_subset_size=10, **overrides)
            data = load_data_pt(cfg2)
            data["_cfg"] = cfg2

            from stage2.model import Stage2ModelPT
            sd = Path(save_dir)
            state = torch.load(sd / "model_final.pt", map_location="cpu",
                               weights_only=True)

            N = data["u_stage1"].shape[1]
            d_ell = data.get("d_ell", 0)
            model = Stage2ModelPT(
                N=N, dt=float(cfg2.dt or 0.6), d_ell=d_ell,
                T_e=data["T_e"], T_sv=data["T_sv"], T_dcv=data["T_dcv"],
                cfg=cfg2, logger=None,
            )
            model.load_state_dict(state, strict=False)
            model.eval()
            device = torch.device("cuda")
            model.to(device)
            for k in ("u_stage1", "sigma_u"):
                if isinstance(data.get(k), torch.Tensor):
                    data[k] = data[k].to(device)
            if data.get("gating") is not None:
                data["gating"] = data["gating"].to(device)
            if data.get("stim") is not None:
                data["stim"] = data["stim"].to(device)

            onestep = compute_onestep(model, data)
            subset = choose_loo_subset(data, onestep, subset_size=10,
                                       subset_mode="motor_best_onestep")
            loo = run_loo_all(model, data, subset=subset)
            free_run = compute_free_run(model, data)

            r2_os = onestep["r2"]
            r2_loo = loo["r2"]
            r2_fr = free_run["r2"]

            summary["os_med"] = round(float(np.nanmedian(r2_os)), 4)
            summary["os_mean"] = round(float(np.nanmean(r2_os)), 4)
            fin = r2_loo[np.isfinite(r2_loo)]
            if fin.size > 0:
                summary["loo_med"] = round(float(np.nanmedian(fin)), 4)
                summary["loo_mean"] = round(float(np.nanmean(fin)), 4)
                summary["loo_pos"] = round(float(np.mean(fin > 0)), 4)
                summary["loo_max"] = round(float(np.nanmax(fin)), 4)
            fin_fr = r2_fr[np.isfinite(r2_fr)]
            if fin_fr.size > 0:
                summary["fr_med"] = round(float(np.nanmedian(fin_fr)), 4)
            summary["G"] = round(float(model.G.mean()), 6)

            del model, data, onestep, loo, free_run, state
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as exc:
            summary["eval_error"] = str(exc)
            traceback.print_exc(file=sys.stdout)

    # Get G from run.log if not already set
    if "G" not in summary:
        try:
            run_log = Path(save_dir) / "run.log"
            if run_log.exists():
                for line in reversed(run_log.read_text().splitlines()):
                    if "G=" in line:
                        import re
                        m = re.search(r'G=([0-9.e+-]+)', line)
                        if m:
                            summary["G"] = round(float(m.group(1)), 6)
                            break
        except Exception:
            pass

    return summary


def print_table(results: list) -> None:
    """Print a comparison table of all completed runs."""
    hdr = (f"{'Run':<25s} {'OS med':>7s} {'LOO med':>8s} {'LOO mean':>9s} "
           f"{'LOO>0%':>7s} {'LOO max':>8s} {'FR med':>7s} {'G':>10s} {'time':>6s}")
    print(f"\n{'─'*95}", flush=True)
    print(hdr, flush=True)
    print(f"{'─'*95}", flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<25s}  ** ERROR: {r['error'][:50]}", flush=True)
            continue
        print(f"{r['name']:<25s} "
              f"{r.get('os_med','?'):>7} "
              f"{r.get('loo_med','?'):>8} "
              f"{r.get('loo_mean','?'):>9} "
              f"{r.get('loo_pos','?'):>7} "
              f"{r.get('loo_max','?'):>8} "
              f"{r.get('fr_med','?'):>7} "
              f"{r.get('G','?'):>10} "
              f"{r.get('time_s','?'):>6}", flush=True)
    print(f"{'─'*95}", flush=True)


def main():
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    results = []

    for name, overrides in RUNS:
        try:
            s = run_one(name, overrides)
            results.append(s)
            print_table(results)
            sys.stdout.flush()
        except Exception as exc:
            print(f"\n*** RUN {name} CRASHED: {exc}", flush=True)
            traceback.print_exc()
            results.append({"name": name, "error": str(exc)})

    # Save final table
    out_path = Path(BASE_DIR) / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Print sorted by LOO median
    valid = [r for r in results if "loo_med" in r]
    if valid:
        valid.sort(key=lambda r: r["loo_med"], reverse=True)
        print("\n\n=== RANKED BY LOO MEDIAN R² ===", flush=True)
        print_table(valid)


if __name__ == "__main__":
    main()
