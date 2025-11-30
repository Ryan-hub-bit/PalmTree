#!/bin/bash

#########################################################################
# Comprehensive Bucket Prediction Evaluation
#
# Evaluates BOTH baseline and address-aware models on all three bucket tasks:
# - BB-level bucket prediction
# - Function-level bucket prediction
# - Binary-level bucket prediction
#
# Uses JSON label files as ground truth
#########################################################################

# Paths
JSON_DIR="data/json_labels"
VOCAB="../strupos/vocab.pkl"
BASELINE_CHECKPOINT="../output/mlm/best_bert.pt"
ADDRESSAWARE_CHECKPOINT="../output/mlm_address/best_bert.pt"
OUTPUT="results_new/bucket_comparison.json"

# Model config
HIDDEN=768
N_LAYERS=12
ATTN_HEADS=12

# Evaluation config
SAMPLES_PER_BUCKET=20000 # Increased from 500 for better training
BINARY_LIMIT=60          # Increased from 1 to test on 5 binaries

# Device
DEVICE="cuda" # or "cpu"

echo "========================================================================"
echo "Comprehensive Bucket Prediction Evaluation"
echo "========================================================================"
echo "Baseline Model:       ${BASELINE_CHECKPOINT}"
echo "Address-Aware Model:  ${ADDRESSAWARE_CHECKPOINT}"
echo "JSON Labels:          ${JSON_DIR}"
echo "Samples per bucket:   ${SAMPLES_PER_BUCKET}"
echo "Binaries to test:     ${BINARY_LIMIT}"
echo "Device:               ${DEVICE}"
echo "========================================================================"
echo ""

python3 comprehensive_bucket_evaluation.py \
  --json_dir "${JSON_DIR}" \
  --vocab "${VOCAB}" \
  --baseline_checkpoint "${BASELINE_CHECKPOINT}" \
  --addressaware_checkpoint "${ADDRESSAWARE_CHECKPOINT}" \
  --hidden ${HIDDEN} \
  --n_layers ${N_LAYERS} \
  --attn_heads ${ATTN_HEADS} \
  --samples_per_bucket ${SAMPLES_PER_BUCKET} \
  --binary_limit ${BINARY_LIMIT} \
  --device ${DEVICE} \
  --output "${OUTPUT}"

echo ""
echo "========================================================================"
echo "Evaluation complete! Results saved to: ${OUTPUT}"
echo "========================================================================"
