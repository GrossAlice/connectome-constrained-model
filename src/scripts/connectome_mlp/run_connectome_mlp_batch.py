#!/usr/bin/env python3
"""
Batch runner for connectome-constrained MLP vs unconstrained MLP.

This script runs connectome_neural_decoder.py on all worms in the dataset,
comparing MLP predictions with and without connectome constraints.

The key comparison:
  - conn+self:  MLP using only connectome neighbours + self history
  - all+self:   MLP using all neurons + self history (unconstrained)

This tells us whether the connectome provides useful structural priors
for predicting neural activity, or if the MLP can learn the same
representations from data alone.

Usage:
    python -m scripts.connectome_mlp.run_connectome_mlp_batch --device cuda

Output:
    output_plots/connectome_mlp_batch/
        {worm_id}_{neurons}/
            T_e/
            T_sv/
            T_dcv/
        summary/
            summary_all_worms.json
            summary_plots/
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    ap = argparse.ArgumentParser(
        description="Batch run connectome-constrained MLP on all worms"
    )
    ap.add_argument("--data_dir", 
                    default="data/used/behaviour+neuronal activity atanas (2023)/2",
                    help="Directory containing worm h5 files")
    ap.add_argument("--out_dir", 
                    default="output_plots/connectome_mlp_batch",
                    help="Output directory for results")
    ap.add_argument("--device", default="cuda",
                    help="Device for MLP training (cuda/cpu)")
    ap.add_argument("--neurons", default="all",
                    choices=["motor", "nonmotor", "all"],
                    help="Which neurons to analyze")
    ap.add_argument("--T_matrices", nargs="+", default=["T_e"],
                    choices=["T_e", "T_sv", "T_dcv"],
                    help="Which connectome matrices to use")
    ap.add_argument("--no_mlp", action="store_true",
                    help="Skip MLP (Ridge/PCA-Ridge only)")
    ap.add_argument("--worm_ids", nargs="+", default=None,
                    help="Specific worm IDs to process (default: all)")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all h5 files
    h5_files = sorted(glob.glob(str(data_dir / "*.h5")))
    print(f"Found {len(h5_files)} worm files in {data_dir}")

    # Filter if specific worms requested
    if args.worm_ids:
        h5_files = [f for f in h5_files 
                    if any(wid in Path(f).stem for wid in args.worm_ids)]
        print(f"  Filtered to {len(h5_files)} worms: {args.worm_ids}")

    T_matrices = [ROOT / f"data/used/masks+motor neurons/{t}.npy" 
                  for t in args.T_matrices]
    
    script_path = ROOT / "scripts" / "connectome_neural_decoder.py"

    results_log = []
    t_total = time.time()

    for i, h5_path in enumerate(h5_files):
        worm_id = Path(h5_path).stem
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(h5_files)}] Processing: {worm_id}")
        print(f"{'='*70}")

        for T_npy in T_matrices:
            T_name = T_npy.stem
            print(f"\n  Connectome: {T_name}")

            cmd = [
                sys.executable, str(script_path),
                "--h5", h5_path,
                "--T_npy", str(T_npy),
                "--out_dir", str(out_dir),
                "--device", args.device,
                "--neurons", args.neurons,
            ]
            if args.no_mlp:
                cmd.append("--no_mlp")

            t0 = time.time()
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=3600  # 1 hour timeout per worm
                )
                elapsed = time.time() - t0
                
                if result.returncode == 0:
                    print(f"    ✓ Done in {elapsed:.0f}s")
                    results_log.append({
                        "worm_id": worm_id,
                        "T_matrix": T_name,
                        "status": "success",
                        "time_s": elapsed
                    })
                else:
                    print(f"    ✗ Failed (exit code {result.returncode})")
                    print(f"      stderr: {result.stderr[:500]}")
                    results_log.append({
                        "worm_id": worm_id,
                        "T_matrix": T_name,
                        "status": "failed",
                        "error": result.stderr[:500]
                    })
            except subprocess.TimeoutExpired:
                print(f"    ✗ Timeout (>1h)")
                results_log.append({
                    "worm_id": worm_id,
                    "T_matrix": T_name,
                    "status": "timeout"
                })
            except Exception as e:
                print(f"    ✗ Error: {e}")
                results_log.append({
                    "worm_id": worm_id,
                    "T_matrix": T_name,
                    "status": "error",
                    "error": str(e)
                })

    total_time = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'='*70}")

    # Save batch log
    log_path = out_dir / "batch_log.json"
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nBatch log saved to: {log_path}")

    # Count successes
    n_success = sum(1 for r in results_log if r["status"] == "success")
    n_total = len(results_log)
    print(f"Completed: {n_success}/{n_total} ({100*n_success/n_total:.0f}%)")


if __name__ == "__main__":
    main()
