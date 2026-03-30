#!/usr/bin/env python3
"""
Compute eigenworms from the Stephens (2016) shapes.csv basis and write
``behaviour/eigenworms_stephens`` (6 amplitudes) to every worm H5.

Usage:
    python scripts/add_stephens_eigenworms.py
    python scripts/add_stephens_eigenworms.py --dataset_dir path/to/worms --n_modes 6
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

# ── Helpers ──────────────────────────────────────────────────────────────

def _ensure_TN(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D, got {arr.shape}")
    if arr.shape[0] < 400 and arr.shape[1] >= 400 and arr.shape[1] / max(arr.shape[0], 1) > 2:
        return arr.T
    return arr


def _find_body_angle_key(f: h5py.File) -> str | None:
    for k in ("behavior/body_angle_all", "behavior/body_angle",
              "behaviour/body_angle_all", "behaviour/body_angle"):
        if k in f:
            return k
    return None


def _resample_1d(row: np.ndarray, d: int) -> np.ndarray:
    if row.size == d:
        return row.astype(float)
    return np.interp(np.linspace(0, 1, d), np.linspace(0, 1, row.size), row)


def _prefix_len_from_head(row: np.ndarray) -> int:
    m = np.isfinite(row)
    if not m.any() or not bool(m[0]):
        return 0
    bad = np.where(~m)[0]
    return int(bad[0]) if bad.size else int(len(row))


def _preprocess_worm(arr: np.ndarray, d_w: int, d_target: int) -> np.ndarray:
    """(T×D) body-angle array → (T×d_target), carrying last good frame forward."""
    T = arr.shape[0]
    proc = np.zeros((T, d_target), dtype=float)
    fallback = np.zeros(d_w, dtype=float)
    for t in range(T):
        row = np.asarray(arr[t], dtype=float)
        l = _prefix_len_from_head(row)  # row[:l] is all-finite by construction
        if l >= 2:
            obs = row[:l]
            fallback = _resample_1d(obs[:d_w] if l > d_w else obs, d_w)
        proc[t] = _resample_1d(fallback, d_target)
    return proc


def _get_d_w(arr: np.ndarray) -> int:
    T = arr.shape[0]
    pref = np.array([_prefix_len_from_head(arr[t]) for t in range(T)])
    pv = pref[pref >= 2]
    if pv.size == 0:
        return max(2, arr.shape[1])
    c = Counter(pv.tolist())
    return sorted(c.items(), key=lambda x: (-x[1], x[0]))[0][0]


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="data/used/behaviour+neuronal activity atanas (2023)")
    ap.add_argument("--shapes_csv", default="data/used/stephens (2016)/shapes.csv")
    ap.add_argument("--d_target", type=int, default=None,
                    help="Body-angle dimensionality for projection. "
                         "Default: native column count of shapes.csv.")
    ap.add_argument("--n_modes", type=int, default=6)
    args = ap.parse_args()

    dataset_dir  = Path(args.dataset_dir)
    shapes_path  = Path(args.shapes_csv)
    stephens_dir = shapes_path.parent       # data/used/stephens (2016)/
    n_modes = args.n_modes

    # 1. Build Stephens eigenbasis from shapes.csv
    shapes_raw = np.loadtxt(str(shapes_path), delimiter=",")   # (6655, 100)
    d_native = shapes_raw.shape[1]
    d_target = args.d_target if args.d_target is not None else d_native
    print(f"Stephens shapes: {shapes_raw.shape}  d_target={d_target}")

    shapes = shapes_raw if d_target == d_native else np.array([_resample_1d(r, d_target) for r in shapes_raw])
    Xc = shapes - shapes.mean(axis=0, keepdims=True)
    eigvals_s, eigvecs_s = np.linalg.eigh((Xc.T @ Xc) / len(Xc))
    order     = np.argsort(eigvals_s)[::-1]
    eigvals_s = eigvals_s[order]
    eigvecs_s = eigvecs_s[:, order]   # (d_target, d_target)
    evr = eigvals_s / eigvals_s.sum()
    print(f"First {n_modes} EVR: {evr[:n_modes].round(4)}  sum={evr[:n_modes].sum():.4f}")

    np.save(stephens_dir / "eigenvectors_stephens.npy", eigvecs_s)
    np.save(stephens_dir / "eigenvalues_stephens.npy",  eigvals_s)
    print(f"Saved eigenvectors → {stephens_dir}")

    # 2. Project every worm → behaviour/eigenworms_stephens
    worms = sorted(dataset_dir.glob("*.h5"))
    print(f"\nProcessing {len(worms)} H5 files …")

    for fp in worms:
        with h5py.File(fp, "r") as f:
            key = _find_body_angle_key(f)
            if key is None:
                print(f"  SKIP {fp.name}: no body-angle dataset")
                continue
            arr = _ensure_TN(np.asarray(f[key], dtype=float))

        d_w  = _get_d_w(arr)
        proc = _preprocess_worm(arr, d_w, d_target)
        amps = ((proc - proc.mean(axis=1, keepdims=True)) @ eigvecs_s)[:, :n_modes]  # (T, n_modes)

        with h5py.File(fp, "a") as f:
            grp = f.require_group("behaviour")
            if "eigenworms_stephens" in grp:
                del grp["eigenworms_stephens"]
            ds = grp.create_dataset("eigenworms_stephens", data=amps,
                                    compression="gzip", compression_opts=4)
            ds.attrs.update({"source": key, "d_w": d_w, "d_target": d_target,
                             "n_modes": n_modes, "basis": "stephens_2016"})

        print(f"  ✓ {fp.name}  shape={amps.shape}")

    print("DONE ✓")


if __name__ == "__main__":
    main()
