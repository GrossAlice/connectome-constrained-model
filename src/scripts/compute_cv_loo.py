"""Compute cross-validated LOO R² by evaluating LOO with each fold's model.

Usage: run from repo root (src/)
    python -m scripts.compute_cv_loo --project output_plots/atlas_transformer/bg_20260331_multiply_project \
        --out results/avg_loo_cv.json

This script:
 - locates per-worm folders under the project dir
 - finds the corresponding .h5 in data/used by worm_id
 - loads atlas full labels and embeds the worm into atlas coords
 - for each saved fold model (atlas_model_fold{i}.pt) loads the model and
   computes `compute_loo_r2_atlas` (subset_size=0 => all observed neurons)
 - averages per-neuron R² across folds and writes JSON results.
"""
from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

import atlas_transformer.dataset as at_ds
import atlas_transformer.evaluate as at_eval
from atlas_transformer.config import AtlasTransformerConfig
from atlas_transformer.model import build_atlas_model

# stage2 helper to load full atlas
from stage2.io_multi import _load_full_atlas


def find_h5_for_worm(worm_id: str, search_root: Path) -> Path | None:
    # search common data/used locations under repo
    patterns = [str(search_root / "**" / f"{worm_id}.h5")]
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return Path(matches[0])
    return None


def process_worm(worm_dir: Path, data_root: Path, n_atlas: int) -> Dict[str, Any]:
    worm_id = worm_dir.name
    out: Dict[str, Any] = {"worm_id": worm_id, "folds": []}

    h5_path = find_h5_for_worm(worm_id, data_root)
    if h5_path is None:
        out["error"] = f"h5 not found for {worm_id}"
        return out

    full_labels = _load_full_atlas()
    atlas_idx = np.arange(len(full_labels), dtype=np.int64)

    worm = at_ds.load_atlas_worm_data(str(h5_path), full_labels, atlas_idx, n_atlas)
    if worm is None:
        out["error"] = "failed to load worm data"
        return out

    x, b_mask_out, _ = at_ds.build_joint_state_atlas(
        worm["u_atlas"], worm["obs_mask"], worm.get("b"), worm.get("b_mask"), include_beh=True
    )

    obs_mask = worm["obs_mask"].astype(bool)
    n_beh = worm.get("b").shape[1] if worm.get("b") is not None else 0

    # find models
    model_files = sorted(worm_dir.glob("atlas_model_fold*.pt"))
    if not model_files:
        out["error"] = "no fold models found"
        return out

    per_fold_r2: List[np.ndarray] = []

    cfg = AtlasTransformerConfig()
    cfg.n_atlas = n_atlas

    for mf in model_files:
        try:
            model = build_atlas_model(cfg, device="cpu", n_beh=n_beh)
            state = torch.load(str(mf), map_location="cpu")
            model.load_state_dict(state)
            model.eval()

            res = at_eval.compute_loo_r2_atlas(model, x, obs_mask, subset_size=0)
            r2_arr = np.array(res["r2"])
            per_fold_r2.append(r2_arr)
            out["folds"].append({"model": mf.name, "r2_mean": float(res["r2_mean"] if "r2_mean" in res else np.nan)})
        except Exception as e:
            out.setdefault("errors", []).append({"model": mf.name, "err": str(e)})

    if per_fold_r2:
        # stack and average ignoring NaNs
        stacked = np.stack([r for r in per_fold_r2], axis=0)  # (n_folds, N_obs)
        mean_per_neuron = np.nanmean(stacked, axis=0)
        out["avg_r2_per_neuron"] = mean_per_neuron.tolist()
        out["avg_r2_mean"] = float(np.nanmean(mean_per_neuron))

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="project folder with per-worm subfolders")
    p.add_argument("--data_root", default="data/used", help="root to search for worm .h5 files")
    p.add_argument("--out", default="results/avg_loo_cv.json")
    p.add_argument("--n_atlas", type=int, default=302)
    args = p.parse_args()

    project = Path(args.project)
    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for worm_dir in sorted(project.iterdir()):
        if not worm_dir.is_dir():
            continue
        # skip aggregate files like json
        if worm_dir.suffix == ".json":
            continue
        print(f"Processing {worm_dir.name}")
        res = process_worm(worm_dir, data_root, args.n_atlas)
        results.append(res)

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
