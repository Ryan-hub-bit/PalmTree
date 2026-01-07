#!/bin/bash

# Address-Aware jTrans Pretraining Script
# Uses hierarchical address embeddings instead of position=word trick
# PRETRAINING: Uses single combined file (no train/val/test split)

export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate palmtree

# Data from address-aware function export with hierarchical positions
# Format: opcode(0xADDR:func_pos:bb_pos:inst_pos) operand1 operand2 ...
# For pretraining, we use the same file for train and test (no validation needed during MLM)

python3 train_addressaware.py \
    --train_path /data/kun/jtransdata/addr_pretrain.txt \
    --test_path /data/kun/jtransdata/addr_pretrain.txt \
    --vocab_path /data/kun/jtransdata/vocab_addr.pkl \
    --output_dir /home/kun/Document/AAE/output/jtrans/addressaware_pretrain \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --warmup_steps 10000 \
    --max_len 512 \
    --token_mask_prob 0.15 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --save_every 1 \
    --num_workers 4

