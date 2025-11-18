#!/bin/bash

#########################################################################
# Test Script for MLM and NSP Evaluation
# 
# Evaluates both:
# 1. Address-Aware BERT (trained model)
# 2. Baseline PalmTree (pre-trained model)
#
# On tasks:
# - MLM (Masked Language Model) - CFG only
# - NSP_CFG (Next Sentence Prediction for CFG)
# - NSP_DFG (Next Sentence Prediction for DFG)
#########################################################################

# Test data paths
CFG_TEST="../data/test/cfg/all_cfg_combined.txt"
DFG_TEST="../data/test/dfg/all_dfg_combined.txt"
VOCAB="../pre-trained_model/palmtree/vocab"

# Model checkpoints
# Address-Aware model (adjust epoch as needed)
ADDRESSAWARE_CHECKPOINT="output_addressaware/best_model.pt"

# PalmTree baseline (pre-trained)
PALMTREE_CHECKPOINT="../pre-trained_model/palmtree/transformer.ep19"

# Model configuration (must match training)
HIDDEN=128
N_LAYERS=12
ATTN_HEADS=8
DROPOUT=0.1
SEQ_LEN=20  # Match training seq_len

# Evaluation config
BATCH_SIZE=512
OUTPUT_DIR="test_results_mlm_nsp"

# Hardware
USE_CUDA="--cuda"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "========================================================================"
echo "MLM and NSP Evaluation"
echo "========================================================================"
echo "Test Data:"
echo "  CFG: ${CFG_TEST}"
echo "  DFG: ${DFG_TEST}"
echo "  Vocabulary: ${VOCAB}"
echo ""
echo "Models:"
echo "  Address-Aware: ${ADDRESSAWARE_CHECKPOINT}"
echo "  PalmTree Baseline: ${PALMTREE_CHECKPOINT}"
echo ""
echo "Configuration:"
echo "  Hidden: ${HIDDEN}, Layers: ${N_LAYERS}, Heads: ${ATTN_HEADS}"
echo "  Sequence Length: ${SEQ_LEN}"
echo "  Batch Size: ${BATCH_SIZE}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "========================================================================"
echo ""

# Validate test data exists
if [ ! -f "${CFG_TEST}" ]; then
    echo "ERROR: CFG test data not found: ${CFG_TEST}"
    echo "Please create test data first."
    exit 1
fi

if [ ! -f "${DFG_TEST}" ]; then
    echo "ERROR: DFG test data not found: ${DFG_TEST}"
    echo "Please create test data first."
    exit 1
fi

if [ ! -f "${VOCAB}" ]; then
    echo "ERROR: Vocabulary not found: ${VOCAB}"
    exit 1
fi

# Check model checkpoints
MODELS_FOUND=0

if [ -f "${ADDRESSAWARE_CHECKPOINT}" ]; then
    echo "✓ Address-Aware checkpoint found"
    ADDRESSAWARE_ARG="--addressaware_checkpoint ${ADDRESSAWARE_CHECKPOINT}"
    MODELS_FOUND=$((MODELS_FOUND + 1))
else
    echo "⚠ Address-Aware checkpoint not found: ${ADDRESSAWARE_CHECKPOINT}"
    echo "  Will skip Address-Aware evaluation"
    ADDRESSAWARE_ARG=""
fi

if [ -f "${PALMTREE_CHECKPOINT}" ]; then
    echo "✓ PalmTree checkpoint found"
    PALMTREE_ARG="--palmtree_checkpoint ${PALMTREE_CHECKPOINT}"
    MODELS_FOUND=$((MODELS_FOUND + 1))
else
    echo "⚠ PalmTree checkpoint not found: ${PALMTREE_CHECKPOINT}"
    echo "  Will skip PalmTree evaluation"
    PALMTREE_ARG=""
fi

if [ ${MODELS_FOUND} -eq 0 ]; then
    echo ""
    echo "ERROR: No model checkpoints found. Please train models first."
    exit 1
fi

echo ""
echo "Starting evaluation..."
echo ""

# Run evaluation
python3 test_mlm_nsp.py \
    --cfg_test "${CFG_TEST}" \
    --dfg_test "${DFG_TEST}" \
    --vocab "${VOCAB}" \
    ${ADDRESSAWARE_ARG} \
    ${PALMTREE_ARG} \
    --hidden ${HIDDEN} \
    --n_layers ${N_LAYERS} \
    --attn_heads ${ATTN_HEADS} \
    --dropout ${DROPOUT} \
    --seq_len ${SEQ_LEN} \
    --batch_size ${BATCH_SIZE} \
    --output_dir "${OUTPUT_DIR}" \
    ${USE_CUDA}

echo ""
echo "========================================================================"
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}/comparison_results.json"
echo "========================================================================"
