#!/bin/bash

# Baseline jTrans Pretraining Script
# Pure MLM + JTP (Jump-Target Prediction) without address-aware features
# PRETRAINING: Uses single combined file (no train/val/test split)
#
# Usage: ./run_baseline_pretrain.sh [data_ratio]
# Example: ./run_baseline_pretrain.sh 0.1  # Use 10% of data
#          ./run_baseline_pretrain.sh 1.0  # Use 100% of data (default)

# Data ratio (default: 1.0 = 100% of data)
DATA_RATIO="${1:-1.0}"

# Use all available GPUs (default: 0,1,2,3 for 4 GPUs)
# SLURM will set CUDA_VISIBLE_DEVICES automatically, but if running locally, set it here
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
# Uncomment for debugging (makes CUDA synchronous, slower)
# export CUDA_LAUNCH_BLOCKING=1


echo "=========================================="
echo "Starting Baseline Pretraining"
echo "Data ratio: $DATA_RATIO ($(echo "$DATA_RATIO * 100" | bc)% of training data)"
echo "=========================================="

# For pretraining, we use single file for all data
python3 train_baseline.py \
    --train_path /work/kliu14/jtransdata/baseline_pretrain.txt \
    --vocab_path ./vocab_baseline.txt \
    --output_dir /work/kliu14/jtransoutput/baseline_pretrain \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --warmup_steps 10000 \
    --max_len 512 \
    --token_mask_prob 0.15 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --save_every 1 \
    --num_workers 16 \
    --data_ratio "$DATA_RATIO"

