#!/usr/bin/env python3
"""Compute cross-validated LOO R² averaged across folds for first 3 worms.

Writes JSON to the project's folder `avg_loo_cv_first3.json`.
"""
from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import List

import numpy as np
import torch

from atlas_transformer.config import AtlasTransformerConfig
from atlas_transformer.dataset import (
    load_atlas_worm_data,
    build_joint_state_atlas,
)
from atlas_transformer.model import build_atlas_model
from atlas_transformer.evaluate import compute_loo_r2_atlas

# stage2 helper to get full atlas labels
from stage2.io_multi import _load_full_atlas


PROJECT_DIR = Path("output_plots/atlas_transformer/bg_20260331_multiply_project")
OUT_NAME = "avg_loo_cv_first3.json"


def _parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", type=str, default=str(PROJECT_DIR))
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--subset_size", type=int, default=20,
                   help="LOO subset size (0=all observed)")
    p.add_argument("--n_worms", type=int, default=3)
    return p.parse_args()


def _find_h5_for_worm(worm_id: str) -> str | None:
    # search data/used for a file whose stem equals worm_id
    matches = glob.glob(str(Path("data/used") / "**" / f"{worm_id}.h5"), recursive=True)
    return matches[0] if matches else None


def _apply_config_overrides(cfg: AtlasTransformerConfig, cfg_dict: dict):
    # apply any keys present in saved config
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass


def main(project_dir: Path = PROJECT_DIR, n_worms: int = 3, device: str = "cuda", subset_size: int = 20):
    project_dir = Path(project_dir)
    assert project_dir.exists(), f"Project dir not found: {project_dir}"

    ds_path = project_dir / "data_summary.json"
    if not ds_path.exists():
        raise FileNotFoundError(f"data_summary.json not found in {project_dir}")

    ds = json.loads(ds_path.read_text())
    worms = ds.get("worms", [])[:n_worms]
    cfg_saved = ds.get("config", {})

    full_labels = _load_full_atlas()
    n_atlas = len(full_labels)
    atlas_idx = np.arange(n_atlas, dtype=np.int64)

    cfg = AtlasTransformerConfig()
    _apply_config_overrides(cfg, cfg_saved)
    include_beh = cfg.predict_beh and cfg.include_beh_input
    n_beh = cfg.n_beh_modes if include_beh else 0

    results = []

    for w in worms:
        worm_id = w["worm_id"]
        print(f"Processing worm {worm_id}")

        # find h5
        h5_path = _find_h5_for_worm(worm_id)
        if h5_path is None:
            print(f"  H5 not found for {worm_id}, skipping")
            continue

        worm = load_atlas_worm_data(h5_path, full_labels, atlas_idx, n_atlas,
                                    n_beh_modes=cfg.n_beh_modes)
        if worm is None:
            print(f"  load_atlas_worm_data failed for {worm_id}, skipping")
            continue

        x, b_mask_out, _ = build_joint_state_atlas(
            worm["u_atlas"], worm["obs_mask"], worm.get("b"), worm.get("b_mask"), include_beh=include_beh
        )

        # load saved fold models
        worm_dir = project_dir / worm_id
        if not worm_dir.exists():
            print(f"  run dir not found: {worm_dir}, skipping")
            continue

        fold_files = sorted(worm_dir.glob("atlas_model_fold*.pt"))
        if not fold_files:
            print(f"  no fold models for {worm_id} in {worm_dir}, skipping")
            continue

        per_fold_r2 = []
        for fi, mf in enumerate(fold_files):
            print(f"  fold {fi}: loading {mf}")
            model = build_atlas_model(cfg, device=device, n_beh=n_beh)
            state = torch.load(mf, map_location=("cpu" if device == "cpu" else "cuda"))
            # tolerate full dict or state_dict
            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                # strip module prefix if present
                new = {k.replace("module.", ""): v for k, v in state.items()}
                state = new
            try:
                model.load_state_dict(state)
            except RuntimeError:
                # maybe state has extra keys (meta wrapper)
                if "state_dict" in state:
                    model.load_state_dict(state["state_dict"])
                else:
                    raise
            model = model.to(device)
            model.eval()

            res = compute_loo_r2_atlas(model, x, worm["obs_mask"], subset_size=subset_size, verbose=False)
            per_fold_r2.append(res["r2"].tolist())

        # stack and average across folds (ignore NaNs)
        arr = np.array(per_fold_r2, dtype=np.float32)  # (n_folds, N_obs)
        avg = np.nanmean(arr, axis=0)

        obs_idx = np.where(worm["obs_mask"])[0].tolist()
        mean_avg = float(np.nanmean(avg)) if np.isfinite(avg).any() else float("nan")

        results.append({
            "worm_id": worm_id,
            "n_folds": int(arr.shape[0]),
            "obs_idx": obs_idx,
            "per_fold_r2": [r for r in per_fold_r2],
            "r2_avg": avg.tolist(),
            "r2_mean_avg": mean_avg,
        })

    out = project_dir / OUT_NAME
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    args = _parse_args()
    main(project_dir=Path(args.project_dir), n_worms=args.n_worms, device=args.device, subset_size=args.subset_size)
