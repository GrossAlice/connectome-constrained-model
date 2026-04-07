#!/usr/bin/env python
"""Launch multi-worm Stage 2 training with shared atlas-space dynamics.

Usage
-----
    python -m scripts.run_multi_worm_stage2 --device cuda
    python -m scripts.run_multi_worm_stage2 --device cuda --epochs 50 --n_folds 2

This trains shared G/W_sv/W_dcv across all Atanas worms (with stage1)
while keeping lambda_u / I0 per-worm.  LOO is evaluated on AVAL/AVAR
per worm.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))


def _discover_h5_paths(data_dir: str | None = None) -> list[str]:
    """Find all Atanas worm H5 files with stage1 data."""
    if data_dir is None:
        data_dir = str(
            _SRC / "data" / "used"
            / "behaviour+neuronal activity atanas (2023)" / "2"
        )
    pattern = str(Path(data_dir) / "*.h5")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No H5 files found at {pattern}")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Multi-worm Stage 2 training with shared atlas dynamics",
    )
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing .h5 files")
    parser.add_argument("--save_dir", type=str,
                        default="output_plots/stage2/multi_worm_shared",
                        help="Output directory")
    parser.add_argument("--max_worms", type=int, default=0,
                        help="Limit number of worms (0=all)")
    parser.add_argument("--dry", action="store_true",
                        help="Dry run: load data and print summary, no training")
    args = parser.parse_args()

    from stage2.config import make_config

    h5_paths = _discover_h5_paths(args.data_dir)
    if args.max_worms > 0:
        h5_paths = h5_paths[:args.max_worms]
    print(f"[run] Found {len(h5_paths)} H5 files")

    # Create config — use first worm as the "primary" h5_path for DataConfig,
    # put all paths in multi.h5_paths
    cfg = make_config(
        h5_paths[0],
        # Multi-worm settings
        multi_worm=True,
        h5_paths=tuple(h5_paths),
        require_stage1=True,
        atlas_min_worm_count=0,   # full 302-neuron atlas
        # Training
        device=args.device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        cv_folds=2,
        # Dynamics
        learn_lambda_u=True,
        learn_I0=True,
        edge_specific_G=True,
        learn_W_sv=True,
        learn_W_dcv=True,
        # Regularisation
        network_strength_floor=1.0,
        network_strength_target=0.8,
        G_reg=0.0,
        dynamics_l2=0.0,
        # LOO eval
        eval_loo_subset_mode="named",
        eval_loo_subset_names=("AVAL", "AVAR"),
        eval_loo_window_size=50,
        eval_loo_warmup_steps=40,
    )

    if args.dry:
        from stage2.io_multi import load_multi_worm_data
        data = load_multi_worm_data(cfg)
        print(f"\n[dry] Would train on {len(data['worms'])} worms, "
              f"{data['atlas_size']}-neuron atlas")
        for w in data["worms"][:5]:
            print(f"  {w['worm_id']}: T={w['T']}, N_obs={w['N_obs']}")
        if len(data["worms"]) > 5:
            print(f"  ... and {len(data['worms'])-5} more")
        return

    from stage2.train_multi_worm import train_multi_worm
    train_multi_worm(cfg, save_dir=args.save_dir, show=False)


if __name__ == "__main__":
    main()
