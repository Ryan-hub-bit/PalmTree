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

export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate palmtree

echo "=========================================="
echo "Starting Baseline Pretraining"
echo "Data ratio: $DATA_RATIO ($(echo "$DATA_RATIO * 100" | bc)% of training data)"
echo "=========================================="

# For pretraining, we use single file for all data
python3 train_baseline.py \
    --train_data /data/kun/jtransdata/baseline_pretrain.pkl \
    --vocab /data/kun/jtransdata/vocab.pkl \
    --output_dir /home/kun/Document/AAE/output/jtrans/baseline_pretrain \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --mask_prob 0.15 \
    --save_best \
    --data_ratio "$DATA_RATIO"

