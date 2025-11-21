#!/bin/bash

# Test all bucket probes with data_with_binary directory
# Usage: ./test_all_probes_binary_data.sh

echo "Running all bucket probes with data_with_binary..."
echo ""

# BB-level probe
echo "1/3 Running BB-level bucket probe..."
python bb_bucket_probe.py \
    --data_dir data_with_binary \
    --vocab ../pre-trained_model/palmtree/vocab \
    --addressaware_model ../addressaware/output_addressaware_new/best_bert.pt \
    --baseline_model ../pre-trained_model/palmtree/transformer.ep19 \
    --output probe_results_bb_with_binary

echo ""
echo "---"
echo ""

# Function-level probe
echo "2/3 Running Function-level bucket probe..."
python function_bucket_probe.py \
    --data_dir data_with_binary \
    --vocab ../pre-trained_model/palmtree/vocab \
    --addressaware_model ../addressaware/output_addressaware_new/best_bert.pt \
    --baseline_model ../pre-trained_model/palmtree/transformer.ep19 \
    --output probe_results_function_with_binary

echo ""
echo "---"
echo ""

# Binary-level probe
echo "3/3 Running Binary-level bucket probe..."
python binary_bucket_probe.py \
    --data_dir data_with_binary \
    --vocab ../pre-trained_model/palmtree/vocab \
    --addressaware_model ../addressaware/output_addressaware_new/best_bert.pt \
    --baseline_model ../pre-trained_model/palmtree/transformer.ep19 \
    --output probe_results_binary_with_binary

echo ""
echo "================================================================================"
echo "✓ All probes with data_with_binary completed!"
echo "================================================================================"
echo ""
echo "Quick Summary:"
echo "BB-level results in: probe_results_bb_with_binary/comparison.json"
echo "Function-level results in: probe_results_function_with_binary/comparison.json"
echo "Binary-level results in: probe_results_binary_with_binary/comparison.json"
echo ""
