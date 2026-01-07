#!/bin/bash

# Baseline jTrans Pretraining Script
# Pure MLM + JTP (Jump-Target Prediction) without address-aware features
# PRETRAINING: Uses single combined file (no train/val/test split)

export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate palmtree

# For pretraining, we use single file for all data
python3 train_baseline.py \
    --train_data /data/kun/jtransdata/baseline_pretrain.pkl \
    --vocab /data/kun/jtransdata/vocab.pkl \
    --output_dir /home/kun/Document/AAE/output/jtrans/baseline_pretrain \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --mask_prob 0.15 \
    --save_best

