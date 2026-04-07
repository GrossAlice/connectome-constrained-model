#!/usr/bin/env bash
# Batch run: Joint strategy (TRF, MLP, Ridge) on first 10 worms, motor neurons, with videos.
# MLP architecture matches v8: hidden=64, n_layers=2
set -euo pipefail

cd "/home/agross/Downloads/connectome-constrained model/src"
source .venv/bin/activate

DATA_DIR="data/used/behaviour+neuronal activity atanas (2023)/2"
OUT_DIR="output_plots/free_run"
DEVICE="cuda"
N_SAMPLES=20

# First 10 worms
WORMS=(
  "2022-06-14-01.h5"
  "2022-06-14-07.h5"
  "2022-06-14-13.h5"
  "2022-06-28-01.h5"
  "2022-06-28-07.h5"
  "2022-07-15-06.h5"
  "2022-07-15-12.h5"
  "2022-07-20-01.h5"
  "2022-07-26-01.h5"
  "2022-08-02-01.h5"
)

TOTAL=${#WORMS[@]}
echo "=========================================="
echo " Joint strategy batch: ${TOTAL} worms"
echo " MLP: hidden=64, n_layers=2 (v8 arch)"
echo " TRF: d_model=256, n_heads=8, n_layers=2"
echo "=========================================="

for i in "${!WORMS[@]}"; do
  idx=$((i + 1))
  worm="${WORMS[$i]}"
  echo ""
  echo "[$idx/$TOTAL] Processing $worm"
  echo "-------------------------------------------"

  python -u -m scripts.free_run.run \
    --h5 "$DATA_DIR/$worm" \
    --out_dir "$OUT_DIR" \
    --device "$DEVICE" \
    --neurons motor \
    --strategy joint \
    --n_samples "$N_SAMPLES" \
    --video \
    --video_frames 500 \
    2>&1

  echo "[$idx/$TOTAL] Done: $worm"
done

echo ""
echo "=========================================="
echo " All ${TOTAL} worms completed!"
echo "=========================================="
