#!/bin/bash

# Run Jump Understanding Probe
# Tests whether models understand control flow

set -e

echo "=========================================="
echo "Jump Understanding Probe"
echo "=========================================="

# Configuration
BASELINE_MODEL="./output/baseline_pretrain/best_model"
ADDRESSAWARE_MODEL="./output/addressaware_pretrain/best_model"
BASELINE_TOKENIZER="./output/baseline_pretrain/best_model"  # Baseline vocab
ADDRESSAWARE_TOKENIZER="./output/addressaware_pretrain/best_model"  # Address-aware vocab
OUTPUT_DIR="./probe/results/jump_understanding"
BASELINE_DATA_FILE="./probe/data/jump_probe_baseline.json"
ADDRESSAWARE_DATA_FILE="./probe/data/jump_probe_addressaware.json"
BASELINE_FUNC_BLOCKS="./output/funcsim/func_blocks_baseline.json"
ADDRESSAWARE_FUNC_BLOCKS="./output/funcsim/func_blocks_addressaware.json"
CFG_DIR="./data_generator/cfg_output"

# Step 1: Prepare probe data (if not exists)
echo ""
echo "Step 1: Preparing probe data..."
echo "----------------------------------------"

mkdir -p $(dirname "$BASELINE_DATA_FILE")

# Prepare baseline data
if [ ! -f "$BASELINE_DATA_FILE" ]; then
    echo "  Preparing baseline probe data..."
    python probe/prepare_jump_probe_data.py \
        --func_blocks "$BASELINE_FUNC_BLOCKS" \
        --cfg_dir "$CFG_DIR" \
        --output "$BASELINE_DATA_FILE" \
        --model_type baseline \
        --max_funcs 1000
    
    echo "  ✓ Baseline probe data prepared"
else
    echo "  Baseline data already exists: $BASELINE_DATA_FILE"
fi

# Prepare address-aware data
if [ ! -f "$ADDRESSAWARE_DATA_FILE" ]; then
    echo "  Preparing address-aware probe data..."
    python probe/prepare_jump_probe_data.py \
        --func_blocks "$ADDRESSAWARE_FUNC_BLOCKS" \
        --cfg_dir "$CFG_DIR" \
        --output "$ADDRESSAWARE_DATA_FILE" \
        --model_type addressaware \
        --max_funcs 1000
    
    echo "  ✓ Address-aware probe data prepared"
else
    echo "  Address-aware data already exists: $ADDRESSAWARE_DATA_FILE"
fi

# Step 2: Run probe evaluation
echo ""
echo "Step 2: Evaluating jump understanding..."
echo "----------------------------------------"

mkdir -p "$OUTPUT_DIR"

python probe/jump_probe.py \
    --baseline_model "$BASELINE_MODEL" \
    --addressaware_model "$ADDRESSAWARE_MODEL" \
    --baseline_tokenizer "$BASELINE_TOKENIZER" \
    --addressaware_tokenizer "$ADDRESSAWARE_TOKENIZER" \
    --baseline_data_file "$BASELINE_DATA_FILE" \
    --addressaware_data_file "$ADDRESSAWARE_DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --visualize_samples 10

echo ""
echo "=========================================="
echo "✓ Probe completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - results.json: Quantitative results"
echo "  - jump_prediction_comparison.png: Accuracy comparison"
echo "  - rank_distribution.png: Target rank distribution"
echo "  - baseline_func_*.png: Baseline attention visualizations"
echo "  - addressaware_func_*.png: Address-aware attention visualizations"
echo ""
