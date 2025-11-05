#!/bin/bash

# Test and compare PalmTree vs AddressAware models
# This script evaluates MLM and NSP performance on both models

set -e

# Configuration
CFG_DATA="../data/cfg/all_cfg_combined.txt"
DFG_DATA="../data/dfg/all_dfg_combined.txt"
VOCAB="../../pre-trained_model/palmtree/vocab"
PALMTREE_CHECKPOINT="../../pre-trained_model/palmtree/transformer.ep19"

# AddressAware checkpoint (if trained)
# Leave empty to test with freshly initialized model
ADDRESSAWARE_CHECKPOINT=""  # e.g., "../output_addressaware/checkpoint_epoch_5.pth"

# Test parameters
BATCH_SIZE=512
SEQ_LEN=20
NUM_WORKERS=4
TEST_SAMPLES=10000  # Number of samples to test (subset for speed)

# Output
OUTPUT_DIR="./results"
OUTPUT_FILE="${OUTPUT_DIR}/comparison_$(date +%Y%m%d_%H%M%S).json"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "Testing PalmTree vs AddressAware Models"
echo "========================================================================"
echo "Data:"
echo "  CFG: ${CFG_DATA}"
echo "  DFG: ${DFG_DATA}"
echo "  Vocabulary: ${VOCAB}"
echo ""
echo "Models:"
echo "  PalmTree: ${PALMTREE_CHECKPOINT}"
if [ -n "${ADDRESSAWARE_CHECKPOINT}" ] && [ -f "${ADDRESSAWARE_CHECKPOINT}" ]; then
    echo "  AddressAware: ${ADDRESSAWARE_CHECKPOINT} (trained)"
else
    echo "  AddressAware: [Fresh initialization - no training]"
fi
echo ""
echo "Test Config:"
echo "  Test Samples: ${TEST_SAMPLES}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Sequence Length: ${SEQ_LEN}"
echo "  Workers: ${NUM_WORKERS}"
echo ""
echo "Output: ${OUTPUT_FILE}"
echo "========================================================================"
echo ""

# Build command
CMD="python test_models.py \
    --cfg_data ${CFG_DATA} \
    --dfg_data ${DFG_DATA} \
    --vocab ${VOCAB} \
    --palmtree_checkpoint ${PALMTREE_CHECKPOINT} \
    --batch_size ${BATCH_SIZE} \
    --seq_len ${SEQ_LEN} \
    --num_workers ${NUM_WORKERS} \
    --test_samples ${TEST_SAMPLES} \
    --output ${OUTPUT_FILE}"

# Add addressaware checkpoint if provided
if [ -n "${ADDRESSAWARE_CHECKPOINT}" ] && [ -f "${ADDRESSAWARE_CHECKPOINT}" ]; then
    CMD="${CMD} --addressaware_checkpoint ${ADDRESSAWARE_CHECKPOINT}"
fi

# Run test
eval ${CMD}

echo ""
echo "========================================================================"
echo "Testing complete! Results saved to: ${OUTPUT_FILE}"
echo "========================================================================"
