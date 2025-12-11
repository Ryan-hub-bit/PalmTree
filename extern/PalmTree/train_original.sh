#!/bin/bash
# Train original BERT (tokens only, ignore address information)

python train_palmtree_addressaware.py \
    --mode original \
    --train_cfg data/training/cdfg_bert_1/cfg_train.txt \
    --train_dfg data/training/cdfg_bert_1/dfg_train.txt \
    --output_path output/original_bert \
    --vocab_path vocab/original \
    --hidden 128 \
    --n_layers 12 \
    --attn_heads 8 \
    --dropout 0.0 \
    --seq_len 20 \
    --batch_size 256 \
    --epochs 20 \
    --lr 1e-5 \
    --num_workers 10 \
    --cuda_devices 0
