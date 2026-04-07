#!/usr/bin/env python
"""Batch Stage 2 rerun — repro_run14 parameters, all worms.

Reads the run_config.json from a reference run and trains Stage 2
for every worm in the data directory with the same hyperparameters.

Usage:
    python -m scripts.batch_stage2_repro \
        --ref output_plots/stage2/repro_run14/run_config.json \
        --h5_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
        --out_dir output_plots/stage2/batch_repro14 \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from stage2.config import make_config
from stage2.train import train_stage2


def _flatten_config(cfg_dict: dict) -> dict:
    """Flatten nested run_config.json into keyword arguments for make_config."""
    flat = {}
    for section, sub in cfg_dict.items():
        if isinstance(sub, dict):
            for k, v in sub.items():
                flat[k] = v
        else:
            flat[section] = v
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(SRC / "output_plots/stage2/repro_run14/run_config.json"),
                    help="Path to reference run_config.json")
    ap.add_argument("--h5_dir", default=str(SRC / "data/used/behaviour+neuronal activity atanas (2023)/2"),
                    help="Directory containing worm .h5 files")
    ap.add_argument("--out_dir", default=str(SRC / "output_plots/stage2/batch_repro14"),
                    help="Output directory (subfolders per worm)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--worms", nargs="*", default=None,
                    help="Specific worm stems to run (default: all)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip worms that already have a run.log")
    args = ap.parse_args()

    # ── Load reference config ──────────────────────────────────────────
    with open(args.ref) as f:
        ref_cfg = json.load(f)
    flat = _flatten_config(ref_cfg)

    # Remove keys that must be per-worm or are absolute paths from the ref run
    for key in ("h5_path",):
        flat.pop(key, None)

    # Override device
    flat["device"] = args.device

    # ── Discover worms ─────────────────────────────────────────────────
    h5_dir = Path(args.h5_dir)
    if args.worms:
        h5_files = [h5_dir / f"{w}.h5" for w in args.worms]
    else:
        h5_files = sorted(h5_dir.glob("*.h5"))

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Reference config: {args.ref}")
    print(f"Worms to process: {len(h5_files)}")
    print(f"Output base:      {out_base}")
    print()

    results = {}
    for hi, h5 in enumerate(h5_files):
        worm = h5.stem
        save_dir = out_base / worm
        if args.resume and (save_dir / "run.log").exists():
            print(f"[{hi+1}/{len(h5_files)}] SKIP {worm} (already has run.log)")
            continue

        print(f"[{hi+1}/{len(h5_files)}] START {worm}")
        t0 = time.time()
        try:
            cfg = make_config(str(h5), **flat)
            train_stage2(cfg, save_dir=str(save_dir), show=False)
            elapsed = time.time() - t0
            print(f"[{hi+1}/{len(h5_files)}] DONE  {worm}  ({elapsed:.0f}s)")
            results[worm] = "ok"
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{hi+1}/{len(h5_files)}] FAIL  {worm}  ({elapsed:.0f}s): {e}")
            traceback.print_exc()
            results[worm] = f"FAIL: {e}"

        # Reset stdout in case train_stage2's TeeWriter replaced it
        sys.stdout = sys.__stdout__

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    n_ok = sum(1 for v in results.values() if v == "ok")
    n_fail = sum(1 for v in results.values() if v != "ok")
    for w, r in results.items():
        tag = "✓" if r == "ok" else "✗"
        print(f"  {tag} {w}: {r}")
    print(f"\nTotal: {n_ok} ok, {n_fail} failed out of {len(h5_files)} worms")

    # Save summary JSON
    summary_path = out_base / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
