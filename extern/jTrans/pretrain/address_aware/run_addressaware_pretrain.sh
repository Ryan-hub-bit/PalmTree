#!/bin/bash

# Address-Aware jTrans Pretraining Script (Improved)
# Uses hierarchical address embeddings with instruction-level masking,
# mixed precision, validation split, and loss weighting.
#
# Usage: ./run_addressaware_pretrain.sh [data_ratio] [num_epochs]
# Example: ./run_addressaware_pretrain.sh 0.1 5    # 10% data, 5 epochs (testing)
#          ./run_addressaware_pretrain.sh 1.0 20   # Full data, 20 epochs (production)

# Data ratio (default: 1.0 = 100% of data)
DATA_RATIO="${1:-1.0}"
NUM_EPOCHS="${2:-10}"

# Use both available GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

echo "=========================================="
echo "Starting Address-Aware Pretraining (Improved)"
echo "Data ratio: $DATA_RATIO"
echo "Epochs: $NUM_EPOCHS"
echo "Masking strategy: mixed (token + instruction)"
echo "Mixed precision: enabled"
echo "=========================================="


python3 train_addressaware.py \
    --train_path /data/kun/jtrans/addressaware/addr_pretrain.txt \
    --vocab_path ./vocab_addr.pkl \
    --output_dir /home/kun/Document/AAE/output/jtrans/addressaware_pretrain \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_epochs "$NUM_EPOCHS" \
    --warmup_ratio 0.06 \
    --max_len 512 \
    --token_mask_prob 0.15 \
    --instruction_mask_prob 0.15 \
    --masking_strategy mixed \
    --jtp_weight 1.0 \
    --val_split 0.05 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --save_every 1 \
    --num_workers 4 \
    --data_ratio "$DATA_RATIO"

