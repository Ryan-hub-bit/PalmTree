#!/bin/bash
# Fine-tune baseline model on function similarity task
# Optimized for Recall@1 performance

export CUDA_VISIBLE_DEVICES=0
# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

python finetune.py \
  --data_type json \
  --func_blocks /data/kun/jtrans/baseline/func_blocks_baseline.json \
  --ground_truth /data/kun/jtrans/baseline/ground_truth_baseline.json \
  --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/baseline \
  --model_path /home/kun/Document/AAE/output/jtrans/baseline_pretrain/checkpoint_epoch_10 \
  --output_path /home/kun/Document/AAE/output/jtrans/baseline_finetune \
  --batch_size 16 \
  --eval_batch_size 64 \
  --lr 2e-5 \
  --epoch 10 \
  --weight_decay 0.01 \
  --warmup 500 \
  --triplet_margin 0.5 \
  --max_grad_norm 1.0 \
  --freeze_cnt 10 \
  --data_ratio 1.0
