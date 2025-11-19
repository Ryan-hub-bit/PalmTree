#!/bin/bash

# Run Linear Relationship Probing Experiment
# Tests for linear relationships between position values and embeddings
# at binary, function, and BB levels.

echo "========================================================================="
echo "Linear Relationship Probing Experiment"
echo "========================================================================="
echo ""

# Configuration
ADDRESSAWARE_MODEL="../addressaware/output_addressaware_new/best_model.pt"
BASELINE_MODEL="../addressaware/output_baseline_new/best_model.pt"
BINARY_TEST="./test_data/binary_level_test.txt"
FUNCTION_TEST="./test_data/function_level_test.txt"
BB_TEST="./test_data/bb_level_test.txt"
VOCAB="../pre-trained_model/palmtree/vocab"
OUTPUT_DIR="./results"

# Check if test data exists, if not generate it
if [ ! -f "${BINARY_TEST}" ] || [ ! -f "${FUNCTION_TEST}" ] || [ ! -f "${BB_TEST}" ]; then
    echo "Test data not found. Generating from binaries in /home/kun/testbinary/..."
    echo ""
    ./generate_probing_data_linear.sh
    echo ""
    if [ ! -f "${BINARY_TEST}" ] || [ ! -f "${FUNCTION_TEST}" ] || [ ! -f "${BB_TEST}" ]; then
        echo "ERROR: Failed to generate test data!"
        exit 1
    fi
fi

echo "Models:"
echo "  Address-Aware: ${ADDRESSAWARE_MODEL}"
echo "  Baseline:      ${BASELINE_MODEL}"
echo ""
echo "Test Data:"
echo "  Binary Level:   ${BINARY_TEST}"
echo "  Function Level: ${FUNCTION_TEST}"
echo "  BB Level:       ${BB_TEST}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "========================================================================="
echo ""

# Run probing
python probe_linear_relationship.py \
  --addressaware_model "${ADDRESSAWARE_MODEL}" \
  --baseline_model "${BASELINE_MODEL}" \
  --binary_test "${BINARY_TEST}" \
  --function_test "${FUNCTION_TEST}" \
  --bb_test "${BB_TEST}" \
  --vocab "${VOCAB}" \
  --output_dir "${OUTPUT_DIR}" \
  --hidden_size 128 \
  --max_samples 5000 \
  --seed 42

echo ""
echo "========================================================================="
echo "Probing complete! Results saved to: ${OUTPUT_DIR}/linear_relationship_results.json"
echo "========================================================================="
