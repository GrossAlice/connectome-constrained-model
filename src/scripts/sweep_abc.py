#!/usr/bin/env python
"""Tests A, B, C — resolve the dynamics-CV confound.

  A  Winner combo (edgG, shA, global) + dynamics_cv_every=5, inject_G=False
  B  Same as A but pure gradient (dynamics_cv_every=0)
  C  Old defaults (scalar G, per-neuron a) + dynamics_cv_every=5, inject_G=False

All 50 epochs, 20 LOO neurons.

Run:
  nohup .venv/bin/python -u scripts/sweep_abc.py \
        > output_plots/stage2/sweep_abc/sweep.log 2>&1 &
"""
from __future__ import annotations

import sys, os, json, time, traceback, gc, re
from pathlib import Path
import numpy as np
import torch

os.environ["PYTHONUNBUFFERED"] = "1"

_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

# ── constants ────────────────────────────────────────────────────────────
H5 = "data/used/behaviour+neuronal activity atanas (2023)/the same neurons/2023-01-23-21.h5"
BASE_DIR = "output_plots/stage2/sweep_abc"

# Common overrides applied to every run
COMMON = dict(
    num_epochs=50,
    eval_loo_subset_size=20,
    diagnostics_every=10,
    make_posture_video=False,
)

# ── runs ─────────────────────────────────────────────────────────────────
RUNS = [
    # A: Winner combo + dynamics-CV every 5 (inject_G=False is code default)
    ("A_winner_cv5", dict(
        edge_specific_G=True,
        per_neuron_amplitudes=False,
        learn_lambda_u=False,
        dynamics_cv_every=5,
    )),

    # B: Winner combo, pure gradient (no dynamics-CV)
    ("B_winner_nocv", dict(
        edge_specific_G=True,
        per_neuron_amplitudes=False,
        learn_lambda_u=False,
        dynamics_cv_every=0,
    )),

    # C: Old defaults (scalar G, per-neuron a) + dynamics-CV every 5
    ("C_olddefaults_cv5", dict(
        edge_specific_G=False,
        per_neuron_amplitudes=True,
        learn_lambda_u=True,
        dynamics_cv_every=5,
    )),
]


# ═════════════════════════════════════════════════════════════════════════
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
        return {"name": name, **overrides,
                "error": str(exc), "time_s": round(time.time() - t0, 1)}
    elapsed = time.time() - t0

    summary: dict = {"name": name, **overrides, "time_s": round(elapsed, 1)}

    # ── extract metrics from eval_result ──
    try:
        if eval_result is not None:
            r2_os  = eval_result.get("onestep", {}).get("r2")
            r2_loo = eval_result.get("loo", {}).get("r2")
            r2_fr  = eval_result.get("free_run", {}).get("r2")

            if r2_os is not None:
                summary["os_med"]  = round(float(np.nanmedian(r2_os)), 4)
                summary["os_mean"] = round(float(np.nanmean(r2_os)), 4)
            if r2_loo is not None:
                fin = r2_loo[np.isfinite(r2_loo)]
                if fin.size > 0:
                    summary["loo_med"]  = round(float(np.nanmedian(fin)), 4)
                    summary["loo_mean"] = round(float(np.nanmean(fin)), 4)
                    summary["loo_pos"]  = round(float(np.mean(fin > 0)), 4)
                    summary["loo_max"]  = round(float(np.nanmax(fin)), 4)
            if r2_fr is not None:
                fin = r2_fr[np.isfinite(r2_fr)]
                if fin.size > 0:
                    summary["fr_med"] = round(float(np.nanmedian(fin)), 4)
    except Exception as exc:
        summary["eval_parse_error"] = str(exc)

    # ── fallback: reload model if eval_result didn't carry R² ──
    if "loo_med" not in summary:
        try:
            from stage2.config import make_config as _mc
            from stage2.io_h5 import load_data_pt
            from stage2.evaluate import (compute_onestep, run_loo_all,
                                         compute_free_run, choose_loo_subset)
            from stage2.model import Stage2ModelPT

            cfg2 = _mc(H5, eval_loo_subset_size=20, **overrides)
            data = load_data_pt(cfg2)
            data["_cfg"] = cfg2

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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            for k in ("u_stage1", "sigma_u"):
                if isinstance(data.get(k), torch.Tensor):
                    data[k] = data[k].to(device)
            if data.get("gating") is not None:
                data["gating"] = data["gating"].to(device)
            if data.get("stim") is not None:
                data["stim"] = data["stim"].to(device)

            onestep  = compute_onestep(model, data)
            subset   = choose_loo_subset(data, onestep, subset_size=20,
                                         subset_mode="motor_best_onestep")
            loo      = run_loo_all(model, data, subset=subset)
            free_run = compute_free_run(model, data)

            r2_os  = onestep["r2"]
            r2_loo = loo["r2"]
            r2_fr  = free_run["r2"]

            summary["os_med"]  = round(float(np.nanmedian(r2_os)), 4)
            summary["os_mean"] = round(float(np.nanmean(r2_os)), 4)
            fin = r2_loo[np.isfinite(r2_loo)]
            if fin.size > 0:
                summary["loo_med"]  = round(float(np.nanmedian(fin)), 4)
                summary["loo_mean"] = round(float(np.nanmean(fin)), 4)
                summary["loo_pos"]  = round(float(np.mean(fin > 0)), 4)
                summary["loo_max"]  = round(float(np.nanmax(fin)), 4)
            fin_fr = r2_fr[np.isfinite(r2_fr)]
            if fin_fr.size > 0:
                summary["fr_med"] = round(float(np.nanmedian(fin_fr)), 4)

            summary["G"] = round(float(model.G.mean()), 6)

            del model, data, onestep, loo, free_run, state
            torch.cuda.empty_cache(); gc.collect()

        except Exception as exc:
            summary["eval_error"] = str(exc)
            traceback.print_exc(file=sys.stdout)

    # ── G from run.log if not yet set ──
    if "G" not in summary:
        try:
            run_log = Path(save_dir) / "run.log"
            if run_log.exists():
                for line in reversed(run_log.read_text().splitlines()):
                    if "G=" in line:
                        m = re.search(r'G=([0-9.e+-]+)', line)
                        if m:
                            summary["G"] = round(float(m.group(1)), 6)
                            break
        except Exception:
            pass

    return summary


def print_table(results: list) -> None:
    """Print a comparison table of all completed runs."""
    hdr = (f"{'Run':<25s} │ "
           f"{'OS med':>7s} {'LOO med':>8s} {'LOO mean':>9s} "
           f"{'LOO>0%':>7s} {'LOO max':>8s} {'FR med':>7s} "
           f"{'G':>10s} {'time':>6s}")
    print(f"\n{'─'*100}", flush=True)
    print(hdr, flush=True)
    print(f"{'─'*100}", flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<25s} │  ** ERROR: {r['error'][:50]}", flush=True)
            continue
        print(f"{r['name']:<25s} │ "
              f"{r.get('os_med','?'):>7} "
              f"{r.get('loo_med','?'):>8} "
              f"{r.get('loo_mean','?'):>9} "
              f"{r.get('loo_pos','?'):>7} "
              f"{r.get('loo_max','?'):>8} "
              f"{r.get('fr_med','?'):>7} "
              f"{r.get('G','?'):>10} "
              f"{r.get('time_s','?'):>6}", flush=True)
    print(f"{'─'*100}", flush=True)


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
            results.append({"name": name, **overrides, "error": str(exc)})

        gc.collect()
        torch.cuda.empty_cache()

    # ── save final results ──
    out_path = Path(BASE_DIR) / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # ── ranked by LOO median ──
    valid = [r for r in results if "loo_med" in r]
    if valid:
        valid.sort(key=lambda r: r["loo_med"], reverse=True)
        print("\n\n=== RANKED BY LOO MEDIAN R² ===", flush=True)
        print_table(valid)


if __name__ == "__main__":
    main()
