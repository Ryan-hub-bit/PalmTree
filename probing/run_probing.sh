#!/bin/bash

# Run Position Probing Experiment
# This script probes the learned embeddings to evaluate how well they encode
# positional information at binary, function, and basic block levels.

echo "========================================================================="
echo "Position Embedding Probing Experiment"
echo "========================================================================="
echo ""

# Check if test data exists, if not generate it
if [ ! -f "${TEST_CFG}" ] || [ ! -f "${TEST_DFG}" ]; then
    echo "Test data not found. Generating from binaries in /home/kun/testbinary/..."
    echo ""
    ./generate_probing_data.sh
    echo ""
    if [ ! -f "${TEST_CFG}" ] || [ ! -f "${TEST_DFG}" ]; then
        echo "ERROR: Failed to generate test data!"
        exit 1
    fi
fi
echo ""

# Configuration
ADDRESSAWARE_MODEL="../addressaware/output_addressaware_new/best_model.pt"
BASELINE_MODEL="../addressaware/output_baseline_new/best_model.pt"
TEST_CFG="./test_data/all_cfg_probing.txt"  # Generated from testbinary
TEST_DFG="./test_data/all_dfg_probing.txt"  # Generated from testbinary
VOCAB="../pre-trained_model/palmtree/vocab"
OUTPUT_DIR="./results"

# Model configuration
HIDDEN_SIZE=128
SEQ_LEN=20
BATCH_SIZE=128

# Probe training configuration
PROBE_EPOCHS=10
PROBE_LR=0.001
SEED=42

echo "Models:"
echo "  Address-Aware: ${ADDRESSAWARE_MODEL}"
echo "  Baseline:      ${BASELINE_MODEL}"
echo ""
echo "Test Data:"
echo "  CFG: ${TEST_CFG}"
echo "  DFG: ${TEST_DFG}"
echo ""
echo "Configuration:"
echo "  Hidden Size:  ${HIDDEN_SIZE}"
echo "  Seq Length:   ${SEQ_LEN}"
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  Probe Epochs: ${PROBE_EPOCHS}"
echo "  Probe LR:     ${PROBE_LR}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "========================================================================="
echo ""

# Run probing
python probe_positions.py \
  --addressaware_model "${ADDRESSAWARE_MODEL}" \
  --baseline_model "${BASELINE_MODEL}" \
  --test_cfg "${TEST_CFG}" \
  --test_dfg "${TEST_DFG}" \
  --vocab "${VOCAB}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size ${BATCH_SIZE} \
  --hidden_size ${HIDDEN_SIZE} \
  --seq_len ${SEQ_LEN} \
  --probe_epochs ${PROBE_EPOCHS} \
  --probe_lr ${PROBE_LR} \
  --seed ${SEED}

echo ""
echo "========================================================================="
echo "Probing complete! Results saved to: ${OUTPUT_DIR}/probing_results.json"
echo "========================================================================="
