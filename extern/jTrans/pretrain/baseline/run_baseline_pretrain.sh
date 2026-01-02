#!/bin/bash

# Baseline jTrans Pretraining Script
# Pure MLM + JTP (Jump-Target Prediction) without address-aware features

# export CUDA_VISIBLE_DEVICES=1

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

python train_baseline.py \
    --train_path /work/kliu14/jtransdata/pretrain_train.txt \
    --test_path /work/kliu14/jtransdata/pretrain_test.txt \
    --tokenizer_path /home/kliu14/PalmTree/extern/jTrans/pretrain/baseline \
    --output_dir /work/kliu14/jtransoutput/baseline_pretrain \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --num_epochs 6 \
    --warmup_steps 10000 \
    --max_len 512 \
    --mlm_probability 0.15 \
    --jtp_probability 0.20 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --intermediate_size 3072 \
    --save_every 1 \
    --num_workers 4
