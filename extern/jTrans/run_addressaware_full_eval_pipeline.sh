#!/bin/bash
# Quick start script for addressaware evaluation pipeline
# Run all steps with one command

set -e

echo "========================================================================"
echo "        ADDRESSAWARE EVALUATION PIPELINE - QUICK START"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Generate addressaware evaluation data from small_test"
echo "  2. Create filtered evaluation pools"
echo "  3. Run evaluation on finetuned model"
echo ""
echo "IMPORTANT: Update MODEL_CHECKPOINT if needed!"
echo "========================================================================"
echo ""

# Configuration
MODEL_CHECKPOINT="/home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_5"
VOCAB_PATH="/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware"

# Check if model exists
if [ ! -d "$MODEL_CHECKPOINT" ]; then
    echo "ERROR: Model checkpoint not found: $MODEL_CHECKPOINT"
    echo ""
    echo "Please update MODEL_CHECKPOINT in this script to point to your finetuned model."
    echo "Available checkpoints:"
    ls -d /home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_* 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "Using model: $MODEL_CHECKPOINT"
echo "Using vocab: $VOCAB_PATH"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Generate evaluation data
echo ""
echo "========================================================================"
echo "STEP 1/3: Generating AddressAware Evaluation Data"
echo "========================================================================"
cd /home/kun/Document/AAE/extern/jTrans/datautils_addraware

if [ -f "/data/kun/jtrans/addressaware/eval/func_blocks_addr.json" ] && \
   [ -f "/data/kun/jtrans/addressaware/eval/ground_truth_addr.json" ]; then
    echo ""
    echo "Evaluation data already exists. Skip generation? (y/n)"
    read -p "> " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping Step 1..."
    else
        ./generate_addressaware_eval_data.sh
    fi
else
    ./generate_addressaware_eval_data.sh
fi

# Step 2: Create filtered pools
echo ""
echo "========================================================================"
echo "STEP 2/3: Creating Filtered Evaluation Pools"
echo "========================================================================"
cd /home/kun/Document/AAE/extern/jTrans

if [ -d "/data/kun/jtrans/addressaware/eval/pools_filtered" ] && \
   [ "$(ls -A /data/kun/jtrans/addressaware/eval/pools_filtered/*.json 2>/dev/null)" ]; then
    echo ""
    echo "Filtered pools already exist. Regenerate? (y/n)"
    read -p "> " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./run_create_addressaware_filtered_pools.sh
    else
        echo "Using existing pools..."
    fi
else
    ./run_create_addressaware_filtered_pools.sh
fi

# Step 3: Run evaluation
echo ""
echo "========================================================================"
echo "STEP 3/3: Running Evaluation"
echo "========================================================================"
cd /home/kun/Document/AAE/extern/jTrans

./run_addressaware_pool_evaluation.sh "$MODEL_CHECKPOINT" "$VOCAB_PATH"

echo ""
echo "========================================================================"
echo "                    EVALUATION COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  /data/kun/jtrans/addressaware/eval/pools_filtered/evaluation_results.json"
echo ""
echo "To compare with baseline, check:"
echo "  /data/kun/jtrans/baseline/eval/pools_filtered/evaluation_results.json"
echo ""
echo "========================================================================"
