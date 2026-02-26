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
# - Higher learning rate (2e-5 → faster convergence)
# - Larger triplet margin (0.5 → stricter positive/negative separation)
# - Gradient clipping (1.0 → prevent gradient explosion)
# - Warmup steps (500 → stable training start)
# - Batch size 16 (good balance for address-aware model complexity)
# - Target opt O3 (train specifically for Ox→O3 retrieval task)

python finetune.py \
  --model_type addressaware \
  --data_type json \
  --func_blocks /data/kun/jtrans/addressaware/func_blocks_addr.json \
  --ground_truth /data/kun/jtrans/addressaware/ground_truth_addr.json \
  --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
  --model_path /home/kun/Document/AAE/output/jtrans/addressaware_pretrain_jtp/checkpoint_epoch_13 \
  --output_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune_jtp \
  --batch_size 16 \
  --eval_batch_size 32 \
  --lr 2e-5 \
  --epoch 15 \
  --weight_decay 0.01 \
  --warmup 500 \
  --triplet_margin 0.2 \
  --max_grad_norm 1.0 \
  --data_ratio 1.0 \
  --use_projection \
  --embedding_dim 512 \
  --target_opt O3 \
