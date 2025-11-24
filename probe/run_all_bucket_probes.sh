#!/bin/bash

# ==============================================================================
# Comprehensive Bucket Probe Testing Script
# Tests all three hierarchical levels: Binary, Function, and BB
# ==============================================================================

set -e # Exit on error

# Configuration
DATA_DIR="data_5bucket"
VOCAB="../pre-trained_model/palmtree/vocab"
ADDRESSAWARE_MODEL="../addressaware/output_addressaware_scope/best_bert.pt"
BASELINE_MODEL="../pre-trained_model/palmtree/transformer.ep19"
NUM_BUCKETS=5

echo "================================================================================"
echo "Bucket Probe Testing Suite"
echo "================================================================================"
echo "Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Vocabulary: $VOCAB"
echo "  Address-Aware Model: $ADDRESSAWARE_MODEL"
echo "  Baseline Model: $BASELINE_MODEL"
echo "  Number of Buckets: $NUM_BUCKETS"
echo "================================================================================"
echo ""

# ==============================================================================
# 1. BB-Level Bucket Probe
# ==============================================================================
echo "################################################################################"
echo "# 1. BB-Level Bucket Probe (Instruction Position within Basic Block)"
echo "################################################################################"
echo ""

OUTPUT_DIR="probe_results_bb_${NUM_BUCKETS}bucket"
echo "Running BB bucket probe..."
echo "Output directory: $OUTPUT_DIR"
echo ""

python bb_bucket_probe.py \
  --data_dir "$DATA_DIR" \
  --vocab "$VOCAB" \
  --addressaware_model "$ADDRESSAWARE_MODEL" \
  --baseline_model "$BASELINE_MODEL" \
  --output "$OUTPUT_DIR"

echo ""
echo "BB-Level Results:"
cat "$OUTPUT_DIR/comparison.json" | python -m json.tool | grep -A 10 "improvement"
echo ""
echo "================================================================================"
echo ""

# ==============================================================================
# 2. Function-Level Bucket Probe
# ==============================================================================
echo "################################################################################"
echo "# 2. Function-Level Bucket Probe (Instruction Position within Function)"
echo "################################################################################"
echo ""

OUTPUT_DIR="probe_results_function_${NUM_BUCKETS}bucket"
echo "Running function bucket probe..."
echo "Output directory: $OUTPUT_DIR"
echo ""

python function_bucket_probe.py \
  --data_dir "$DATA_DIR" \
  --vocab "$VOCAB" \
  --addressaware_model "$ADDRESSAWARE_MODEL" \
  --baseline_model "$BASELINE_MODEL" \
  --output "$OUTPUT_DIR"

echo ""
echo "Function-Level Results:"
cat "$OUTPUT_DIR/comparison.json" | python -m json.tool | grep -A 10 "improvement"
echo ""
echo "================================================================================"
echo ""

# ==============================================================================
# 3. Binary-Level Bucket Probe
# ==============================================================================
echo "################################################################################"
echo "# 3. Binary-Level Bucket Probe (Instruction Position within Binary)"
echo "################################################################################"
echo ""

OUTPUT_DIR="probe_results_binary_${NUM_BUCKETS}bucket"
echo "Running binary bucket probe..."
echo "Output directory: $OUTPUT_DIR"
echo ""

python binary_bucket_probe.py \
  --data_dir "$DATA_DIR" \
  --vocab "$VOCAB" \
  --addressaware_model "$ADDRESSAWARE_MODEL" \
  --baseline_model "$BASELINE_MODEL" \
  --output "$OUTPUT_DIR"

echo ""
echo "Binary-Level Results:"
cat "$OUTPUT_DIR/comparison.json" | python -m json.tool | grep -A 10 "improvement"
echo ""
echo "================================================================================"
echo ""

# ==============================================================================
# Summary Report
# ==============================================================================
echo "################################################################################"
echo "# SUMMARY REPORT - All Bucket Probe Results"
echo "################################################################################"
echo ""

echo "Extracting results from all probes..."
echo ""

# Function to extract test accuracy
extract_accuracy() {
  local file=$1
  local model=$2
  python -c "import json; data=json.load(open('$file')); print(f\"{data['$model']['test_accuracy']:.4f}\")"
}

# Function to extract improvement
extract_improvement() {
  local file=$1
  python -c "import json; data=json.load(open('$file')); print(f\"{data['improvement']['test_accuracy']:.4f}\")"
}

# Extract all results
BB_AA=$(extract_accuracy "probe_results_bb_${NUM_BUCKETS}bucket/comparison.json" "addressaware")
BB_BL=$(extract_accuracy "probe_results_bb_${NUM_BUCKETS}bucket/comparison.json" "baseline")
BB_IMP=$(extract_improvement "probe_results_bb_${NUM_BUCKETS}bucket/comparison.json")

FUNC_AA=$(extract_accuracy "probe_results_function_${NUM_BUCKETS}bucket/comparison.json" "addressaware")
FUNC_BL=$(extract_accuracy "probe_results_function_${NUM_BUCKETS}bucket/comparison.json" "baseline")
FUNC_IMP=$(extract_improvement "probe_results_function_${NUM_BUCKETS}bucket/comparison.json")

BIN_AA=$(extract_accuracy "probe_results_binary_${NUM_BUCKETS}bucket/comparison.json" "addressaware")
BIN_BL=$(extract_accuracy "probe_results_binary_${NUM_BUCKETS}bucket/comparison.json" "baseline")
BIN_IMP=$(extract_improvement "probe_results_binary_${NUM_BUCKETS}bucket/comparison.json")

# Calculate relative improvements
BB_REL=$(python -c "print(f\"{($BB_IMP / $BB_BL * 100):.1f}%\")")
FUNC_REL=$(python -c "print(f\"{($FUNC_IMP / $FUNC_BL * 100):.1f}%\")")
BIN_REL=$(python -c "print(f\"{($BIN_IMP / $BIN_BL * 100):.1f}%\")")

echo "┌──────────────────────┬─────────────────┬─────────────────┬────────────────┬──────────────────┐"
echo "│ Probe Type           │ Address-Aware   │ PalmTree        │ Absolute Δ     │ Relative Δ       │"
echo "├──────────────────────┼─────────────────┼─────────────────┼────────────────┼──────────────────┤"
printf "│ %-20s │ %15s │ %15s │ %14s │ %16s │\n" "BB-Level" "$BB_AA" "$BB_BL" "+$BB_IMP" "+$BB_REL"
printf "│ %-20s │ %15s │ %15s │ %14s │ %16s │\n" "Function-Level" "$FUNC_AA" "$FUNC_BL" "+$FUNC_IMP" "+$FUNC_REL"
printf "│ %-20s │ %15s │ %15s │ %14s │ %16s │\n" "Binary-Level" "$BIN_AA" "$BIN_BL" "+$BIN_IMP" "+$BIN_REL"
echo "└──────────────────────┴─────────────────┴─────────────────┴────────────────┴──────────────────┘"
echo ""
echo "Random Baseline: $(python -c "print(f'{100.0/$NUM_BUCKETS:.2f}%')")"
echo ""

echo "================================================================================"
echo "✓ All bucket probes completed successfully!"
echo "================================================================================"
echo ""
echo "Result directories:"
echo "  - probe_results_bb_${NUM_BUCKETS}bucket/"
echo "  - probe_results_function_${NUM_BUCKETS}bucket/"
echo "  - probe_results_binary_${NUM_BUCKETS}bucket/"
echo ""
