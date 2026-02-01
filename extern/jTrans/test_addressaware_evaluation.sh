#!/bin/bash
# Quick test: Create small evaluation pools and test evaluation pipeline

set -e

echo "=================================="
echo "Address-Aware Evaluation Test"
echo "=================================="
echo ""

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Test configuration (small data for quick test)
FUNC_BLOCKS="/data/kun/jtrans/addressaware/func_blocks_addr.json"
GROUND_TRUTH="/data/kun/jtrans/addressaware/ground_truth_addr.json"
POOL_DIR="/home/kun/Document/AAE/extern/jTrans/test_pools"
VOCAB_PATH="/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware"
MODEL_CHECKPOINT="/home/kun/Document/AAE/output/jtrans/addressaware_finetune_sincos/finetune_epoch_2"

# Step 1: Create small test pools
echo "Step 1: Creating test pools (size 100 only)..."
echo ""
python create_filtered_addressaware_pools.py \
  --func_blocks "$FUNC_BLOCKS" \
  --ground_truth "$GROUND_TRUTH" \
  --output_dir "$POOL_DIR" \
  --pool_sizes 100 \
  --query_ratio 0.2 \
  --min_instructions 11

echo ""
echo "Test pools created:"
ls -lh "$POOL_DIR"/pool_*.json
echo ""

# Step 2: Test evaluation on one pool
echo "=================================="
echo "Step 2: Testing evaluation on O0_vs_O3 pool..."
echo ""

export CUDA_VISIBLE_DEVICES=0

python evaluate_addressaware_pools.py \
  --checkpoint "$MODEL_CHECKPOINT" \
  --vocab_path "$VOCAB_PATH" \
  --pool_dir "$POOL_DIR" \
  --func_blocks "$FUNC_BLOCKS" \
  --max_len 512 \
  --batch_size 32 \
  --device cuda \
  --pool_size 100 \
  --opt_pair O0_vs_O3

echo ""
echo "=================================="
echo "Test Completed!"
echo "=================================="
echo ""
echo "If this works, you can run the full evaluation with:"
echo "  bash run_addressaware_pool_evaluation.sh \\"
echo "    /home/kun/Document/AAE/output/jtrans/addressaware_finetune_sincos/finetune_epoch_2 \\"
echo "    /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware"
