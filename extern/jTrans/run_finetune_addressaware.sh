#!/bin/bash
# Fine-tune address-aware model on function similarity task
# Optimized for Recall@1 performance

# Set GPUs (single GPU to avoid OOM on shared GPU 1)
export CUDA_VISIBLE_DEVICES=0 

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Configuration for optimal Recall@1
# Key optimizations:
# - InfoNCE with cross-accumulation: batch_size 32, accum 4 → 128-sample similarity matrix
# - Temperature 0.07 (sharp with large effective batch)
# - lr 2e-5 with cosine schedule + 500 warmup steps
# - freeze_cnt 4: only freeze bottom 4 layers (layers 4-11 + projection trainable)
# - Mean pooling: better than CLS for similarity (Sentence-BERT finding)
# - Target opt O3 (train specifically for Ox→O3 retrieval task)

python finetune.py \
  --model_type addressaware \
  --data_type json \
  --func_blocks /data/kun/jtrans/addressaware/func_blocks_addr.json \
  --ground_truth /data/kun/jtrans/addressaware/ground_truth_addr.json \
  --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
  --model_path /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_9 \
  --output_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune_v2 \
  --batch_size 32 \
  --eval_batch_size 32 \
  --lr 2e-5 \
  --epoch 10 \
  --weight_decay 0.01 \
  --warmup 500 \
  --loss_type infonce \
  --temperature 0.07 \
  --gradient_accumulation_steps 4 \
  --max_grad_norm 1.0 \
  --data_ratio 1.0 \
  --use_projection \
  --embedding_dim 512 \
  --target_opt O3 \
  --freeze_cnt 4 \
  --pooling_type mean \
