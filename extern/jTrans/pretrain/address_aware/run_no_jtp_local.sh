#!/bin/bash

# LOCAL: No JTP (MLM only, with binary_pos)
# Usage: ./run_no_jtp_local.sh [data_ratio]
# Example: ./run_no_jtp_local.sh 0.1

# Data ratio (default: 0.1 for local testing)
DATA_RATIO="${1:-0.000001}"

# Use both available GPUs
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=0,1

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

echo "=========================================="
echo "LOCAL: No JTP Pretraining (MLM only)"
echo "Data ratio: $DATA_RATIO"
echo "=========================================="

python3 train_addressaware.py \
    --train_path /data/kun/jtrans/addressaware/addr_pretrain.txt \
    --vocab_path ./vocab_addr.pkl \
    --output_dir /home/kun/Document/AAE/output/jtrans/addressaware_no_jtp \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_epochs 2 \
    --warmup_steps 10000 \
    --max_len 512 \
    --token_mask_prob 0.15 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --save_every 1 \
    --num_workers 4 \
    --data_ratio "$DATA_RATIO" \
    --no_jtp
