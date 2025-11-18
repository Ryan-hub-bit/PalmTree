#!/bin/bash

#########################################################################
# Fair Comparison: Address-Aware vs Baseline
#
# Compares two models trained on SAME data:
# 1. Address-Aware: With 3-level address embeddings
# 2. Baseline: With sequential sinusoidal positions
#
# Only difference: Position encoding strategy
#########################################################################

# Test data paths
CFG_TEST="../data/test/cfg/all_cfg_combined.txt"
DFG_TEST="../data/test/dfg/all_dfg_combined.txt"
VOCAB_FILE="../pre-trained_model/palmtree/vocab"

# Model checkpoints
ADDRESSAWARE_CHECKPOINT="output_addressaware/best_model.pt"
BASELINE_CHECKPOINT="output_baseline/best_model.pt"

# Model configuration (MUST match training config)
HIDDEN=128
N_LAYERS=12
ATTN_HEADS=8
SEQ_LEN=20
DROPOUT=0.1

# Evaluation configuration
BATCH_SIZE=512
OUTPUT_FILE="fair_comparison_results.json"

# Hardware
USE_CUDA="--cuda"

echo "========================================================================"
echo "Fair Comparison: Address-Aware vs Baseline"
echo "========================================================================"
echo ""
echo "Test Data:"
echo "  CFG: ${CFG_TEST}"
echo "  DFG: ${DFG_TEST}"
echo "  Vocabulary: ${VOCAB_FILE}"
echo ""
echo "Models:"

if [ -f "${ADDRESSAWARE_CHECKPOINT}" ]; then
    echo "  ✓ Address-Aware: ${ADDRESSAWARE_CHECKPOINT}"
else
    echo "  ✗ Address-Aware: NOT FOUND - ${ADDRESSAWARE_CHECKPOINT}"
fi

if [ -f "${BASELINE_CHECKPOINT}" ]; then
    echo "  ✓ Baseline: ${BASELINE_CHECKPOINT}"
else
    echo "  ✗ Baseline: NOT FOUND - ${BASELINE_CHECKPOINT}"
fi

echo ""
echo "Configuration:"
echo "  Hidden: ${HIDDEN}, Layers: ${N_LAYERS}, Heads: ${ATTN_HEADS}"
echo "  Seq Length: ${SEQ_LEN}, Batch Size: ${BATCH_SIZE}"
echo ""
echo "Output: ${OUTPUT_FILE}"
echo "========================================================================"
echo ""

# Check if at least one model exists
if [ ! -f "${ADDRESSAWARE_CHECKPOINT}" ] && [ ! -f "${BASELINE_CHECKPOINT}" ]; then
    echo "Error: No model checkpoints found!"
    echo "Please train at least one model first:"
    echo "  ./train_addressaware.sh  # For address-aware model"
    echo "  ./train_baseline.sh      # For baseline model"
    exit 1
fi

# Build command
CMD="python3 test_comparison.py \
    --cfg_test \"${CFG_TEST}\" \
    --dfg_test \"${DFG_TEST}\" \
    --vocab \"${VOCAB_FILE}\" \
    --hidden ${HIDDEN} \
    --n_layers ${N_LAYERS} \
    --attn_heads ${ATTN_HEADS} \
    --seq_len ${SEQ_LEN} \
    --dropout ${DROPOUT} \
    --batch_size ${BATCH_SIZE} \
    --output_file \"${OUTPUT_FILE}\" \
    ${USE_CUDA}"

# Add model checkpoints if they exist
if [ -f "${ADDRESSAWARE_CHECKPOINT}" ]; then
    CMD="${CMD} --addressaware_checkpoint \"${ADDRESSAWARE_CHECKPOINT}\""
fi

if [ -f "${BASELINE_CHECKPOINT}" ]; then
    CMD="${CMD} --baseline_checkpoint \"${BASELINE_CHECKPOINT}\""
fi

# Run comparison
eval ${CMD}

echo ""
echo "========================================================================"
echo "Comparison complete!"
echo "Results saved to: ${OUTPUT_FILE}"
echo "========================================================================"
