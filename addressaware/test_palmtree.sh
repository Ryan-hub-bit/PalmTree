#!/bin/bash

# Test the original PalmTree model on CFG and DFG data
# This script evaluates the pre-trained PalmTree model

echo "=========================================="
echo "Testing Original PalmTree Model"
echo "=========================================="

# Configuration
CHECKPOINT="../pre-trained_model/palmtree/transformer.ep19"
CFG_DATA="../data/test/cfg/all_cfg_palmtree.txt"
DFG_DATA="../data/test/dfg/all_dfg_palmtree.txt"
VOCAB="../pre-trained_model/palmtree/vocab"

# Check if converted files exist, if not, convert them
if [ ! -f "$CFG_DATA" ] || [ ! -f "$DFG_DATA" ]; then
    echo "Converted PalmTree format files not found."
    echo "Running conversion script..."
    bash convert_test_data.sh
    echo ""
fi

# Test settings
BATCH_SIZE=256
NUM_WORKERS=4
SEQ_LEN=100  # Match training seq_len

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: PalmTree checkpoint not found at $CHECKPOINT"
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
echo "  Seq length: $SEQ_LEN"
echo ""

# Run test
echo "Running test on TEST SET..."
python3 test_palmtree_model.py \
    --checkpoint "$CHECKPOINT" \
    --cfg_data "$CFG_DATA" \
    --dfg_data "$DFG_DATA" \
    --vocab "$VOCAB" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --seq_len $SEQ_LEN \
    --cuda

echo ""
echo "=========================================="
echo "Testing complete!"
echo "Results saved to palmtree_test_results.json"
echo "=========================================="
