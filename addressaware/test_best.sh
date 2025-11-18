#!/bin/bash

# Test the best Address-Aware BERT model on CFG and DFG data
# This script evaluates the trained model and reports metrics

echo "=========================================="
echo "Testing Address-Aware BERT Best Model"
echo "=========================================="

# Configuration
CHECKPOINT="output_addressaware/best_model.pt"
CFG_DATA="../data/test/cfg/all_cfg_combined.txt"
DFG_DATA="../data/test/dfg/all_dfg_combined.txt"
VOCAB="../pre-trained_model/palmtree/vocab"

# Test settings
BATCH_SIZE=128
NUM_WORKERS=4

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Check if data files exist
if [ ! -f "$CFG_DATA" ]; then
    echo "Error: CFG data not found at $CFG_DATA"
    exit 1
fi

if [ ! -f "$DFG_DATA" ]; then
    echo "Error: DFG data not found at $DFG_DATA"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  CFG Data:   $CFG_DATA"
echo "  DFG Data:   $DFG_DATA"
echo "  Vocab:      $VOCAB"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Run test on validation split (what the model hasn't seen during training)
echo "Running test on TEST SET (separate test data in /data/test/)..."
python3 test_best_model.py \
    --checkpoint "$CHECKPOINT" \
    --cfg_data "$CFG_DATA" \
    --dfg_data "$DFG_DATA" \
    --vocab "$VOCAB" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --cuda

echo ""
echo "=========================================="
echo "Testing complete!"
echo "Results saved to output_addressaware/test_results.json"
echo "=========================================="
