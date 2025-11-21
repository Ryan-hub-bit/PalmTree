#!/bin/bash

# Simple script to run all three bucket probes
# Usage: ./test_all_probes.sh

echo "Running all bucket probes (BB, Function, Binary)..."
echo ""

# BB-level probe
echo "1/3 Running BB-level bucket probe..."
python bb_bucket_probe.py \
    --data_dir data_5bucket \
    --vocab ../pre-trained_model/palmtree/vocab \
    --addressaware_model ../addressaware/output_addressaware_new/best_bert.pt \
    --baseline_model ../pre-trained_model/palmtree/transformer.ep19 \
    --output probe_results_bb_5bucket

echo ""
echo "---"
echo ""

# Function-level probe
echo "2/3 Running Function-level bucket probe..."
python function_bucket_probe.py \
    --data_dir data_5bucket \
    --vocab ../pre-trained_model/palmtree/vocab \
    --addressaware_model ../addressaware/output_addressaware_new/best_bert.pt \
    --baseline_model ../pre-trained_model/palmtree/transformer.ep19 \
    --output probe_results_function_5bucket

echo ""
echo "---"
echo ""

# Binary-level probe
echo "3/3 Running Binary-level bucket probe..."
python binary_bucket_probe.py \
    --data_dir data_5bucket \
    --vocab ../pre-trained_model/palmtree/vocab \
    --addressaware_model ../addressaware/output_addressaware_new/best_bert.pt \
    --baseline_model ../pre-trained_model/palmtree/transformer.ep19 \
    --output probe_results_binary_5bucket

echo ""
echo "================================================================================"
echo "✓ All probes completed!"
echo "================================================================================"
echo ""
echo "Quick Summary:"
echo "BB-level results in: probe_results_bb_5bucket/comparison.json"
echo "Function-level results in: probe_results_function_5bucket/comparison.json"
echo "Binary-level results in: probe_results_binary_5bucket/comparison.json"
echo ""
