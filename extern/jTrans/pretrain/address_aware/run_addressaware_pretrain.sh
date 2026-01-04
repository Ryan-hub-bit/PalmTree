#!/bin/bash

# Address-Aware jTrans Pretraining Script
# Uses hierarchical address embeddings instead of position=word trick

export CUDA_VISIBLE_DEVICES=1

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Data from jTrans datautils with address annotations
# Format: opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 operand2 ...

python train_addressaware.py \
    --train_path /data/kun/jtransdata/addressaware_train.txt \
    --test_path /data/kun/jtransdata/addressaware_test.txt \
    --vocab_path vocab_addr.pkl \
    --output_dir /home/kun/Document/AAE/output/addressaware_pretrain \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 6 \
    --warmup_steps 10000 \
    --max_len 512 \
    --token_mask_prob 0.15 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --save_every 1 \
    --num_workers 4
