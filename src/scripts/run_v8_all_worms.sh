#!/bin/bash
# Run behaviour_decoder_models_v8.py on all worms with Kb=Kn ONLY
# Motor and all neurons run simultaneously per worm, worms run sequentially

set -e
cd "/home/agross/Downloads/connectome-constrained model/src"
source .venv/bin/activate

DATA_DIR="data/used/behaviour+neuronal activity atanas (2023)/2"
OUT_DIR="output_plots/behaviour_decoder/v8_KbKn"
DEVICE="cuda"

# Create single output directory for all plots
mkdir -p "$OUT_DIR"

# Get all h5 files (use mapfile to handle spaces in paths)
mapfile -t H5_FILES < <(find "$DATA_DIR" -maxdepth 1 -name "*.h5" | sort)
TOTAL=${#H5_FILES[@]}

echo "========================================"
echo "Running v8 on $TOTAL worms (Kb=Kn ONLY)"
echo "Output: $OUT_DIR"
echo "========================================"

for i in "${!H5_FILES[@]}"; do
    H5="${H5_FILES[$i]}"
    WORM_ID=$(basename "$H5" .h5)
    IDX=$((i + 1))
    
    echo ""
    echo "========================================"
    echo "[$IDX/$TOTAL] Processing $WORM_ID"
    echo "========================================"
    
    # Run all neurons and motor neurons in parallel
    # Both save to same root directory (filenames include worm_id and neuron_label)
    echo "  Starting all neurons..."
    python -u -m scripts.behaviour_decoder_models_v8 \
        --h5 "$H5" \
        --out_dir "$OUT_DIR" \
        --device "$DEVICE" \
        2>&1 | tee "$OUT_DIR/log_${WORM_ID}_all.txt" &
    PID_ALL=$!
    
    echo "  Starting motor neurons..."
    python -u -m scripts.behaviour_decoder_models_v8 \
        --h5 "$H5" \
        --out_dir "$OUT_DIR" \
        --motor_only \
        --device "$DEVICE" \
        2>&1 | tee "$OUT_DIR/log_${WORM_ID}_motor.txt" &
    PID_MOTOR=$!
    
    # Wait for both to complete
    echo "  Waiting for both runs (PIDs: $PID_ALL, $PID_MOTOR)..."
    wait $PID_ALL
    STATUS_ALL=$?
    wait $PID_MOTOR
    STATUS_MOTOR=$?
    
    if [ $STATUS_ALL -eq 0 ] && [ $STATUS_MOTOR -eq 0 ]; then
        echo "  ✓ $WORM_ID completed successfully"
    else
        echo "  ✗ $WORM_ID failed (all=$STATUS_ALL, motor=$STATUS_MOTOR)"
    fi
done

echo ""
echo "========================================"
echo "All $TOTAL worms completed!"
echo "========================================"
