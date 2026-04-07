#!/usr/bin/env bash
# Launch neural_activity_decoder_v4 on 20 worms using 4 parallel workers.
# Each worker processes 5 worms sequentially on CUDA.
set -euo pipefail

ROOT="/home/agross/Downloads/connectome-constrained model/src"
PYTHON="$ROOT/.venv/bin/python"
SCRIPT="scripts/neural_activity_decoder_v4.py"
DATA="$ROOT/data/used/behaviour+neuronal activity atanas (2023)/2"
LOGDIR="/tmp/v4_logs"

mkdir -p "$LOGDIR"

# --- 20 worm basenames (first 20 of 40) ---
WORMS=(
  2022-06-14-01
  2022-06-14-07
  2022-06-14-13
  2022-06-28-01
  2022-06-28-07
  2022-07-15-06
  2022-07-15-12
  2022-07-20-01
  2022-07-26-01
  2022-08-02-01
  2022-12-21-06
  2023-01-05-01
  2023-01-05-18
  2023-01-06-01
  2023-01-06-08
  2023-01-06-15
  2023-01-09-08
  2023-01-09-15
  2023-01-09-22
  2023-01-09-28
)

# Worker function: runs a slice of worms sequentially
worker() {
  local worker_id=$1
  shift
  local worms_slice=("$@")
  local log="$LOGDIR/w${worker_id}.log"
  echo "[Worker $worker_id] Starting ${#worms_slice[@]} worms: ${worms_slice[*]}" > "$log"
  for worm in "${worms_slice[@]}"; do
    local h5="$DATA/${worm}.h5"
    echo "========== [Worker $worker_id] $(date '+%H:%M:%S') Starting $worm ==========" >> "$log"
    "$PYTHON" -u -m scripts.neural_activity_decoder_v4 --h5 "$h5" --device cuda >> "$log" 2>&1 || \
      echo "[Worker $worker_id] ERROR on $worm (exit $?)" >> "$log"
  done
  echo "[Worker $worker_id] $(date '+%H:%M:%S') ALL DONE" >> "$log"
}

cd "$ROOT"

echo "Launching 4 workers (5 worms each) …"
echo "Logs: $LOGDIR/w{1,2,3,4}.log"

worker 1 "${WORMS[@]:0:5}"  &
worker 2 "${WORMS[@]:5:5}"  &
worker 3 "${WORMS[@]:10:5}" &
worker 4 "${WORMS[@]:15:5}" &

echo "PIDs: $(jobs -p | tr '\n' ' ')"
wait
echo "All workers finished."
