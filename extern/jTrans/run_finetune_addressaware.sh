#!/bin/bash
# Fine-tune address-aware model on function similarity task
# Optimized for Recall@1 performance

# Set GPUs (use GPU 0 and 1 for multi-GPU training)
export CUDA_VISIBLE_DEVICES=0,1 

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Configuration for optimal Recall@1
# Key optimizations:
# - InfoNCE needs large batches: batch_size 64 → 63 in-batch negatives (was only 15)
# - gradient_accumulation_steps 2 → effective batch 128 for even more negatives
# - Temperature 0.1 (0.07 was too aggressive for small batches)
# - lr 1e-5 (slightly lower for stability with larger effective batch)
# - Target opt O3 (train specifically for Ox→O3 retrieval task)

python finetune.py \
  --model_type addressaware \
  --data_type json \
  --func_blocks /data/kun/jtrans/addressaware/func_blocks_addr.json \
  --ground_truth /data/kun/jtrans/addressaware/ground_truth_addr.json \
  --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
  --model_path /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_10 \
  --output_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune_infonce_b64 \
  --batch_size 64 \
  --eval_batch_size 64 \
  --lr 1e-5 \
  --epoch 15 \
  --weight_decay 0.01 \
  --warmup 500 \
  --loss_type infonce \
  --temperature 0.1 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1.0 \
  --data_ratio 1.0 \
  --use_projection \
  --embedding_dim 512 \
  --target_opt O3 \
