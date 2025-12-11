#!/bin/bash
# Train Address-Aware BERT (tokens + address positions + var offsets)

python train_palmtree_addressaware.py \
    --mode addressaware \
    --train_cfg data/training/addressaware/cfg_addressaware_train.txt \
    --train_dfg data/training/addressaware/dfg_addressaware_train.txt \
    --output_path output/addressaware_bert \
    --vocab_path vocab/addressaware \
    --hidden 128 \
    --n_layers 12 \
    --attn_heads 8 \
    --dropout 0.0 \
    --seq_len 20 \
    --address_hidden 64 \
    --var_size 256 \
    --batch_size 256 \
    --epochs 20 \
    --lr 1e-5 \
    --num_workers 10 \
    --cuda_devices 0
