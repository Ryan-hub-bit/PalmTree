#!/bin/bash
# Evaluate finetuned address-aware model on function similarity pools
# Metrics: MRR, Recall@1, Recall@5, Recall@10

export CUDA_VISIBLE_DEVICES=0

source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# --- Paths ---
CHECKPOINT=/home/kun/Document/AAE/output/jtrans/addressaware_finetune_infonce_b64/finetune_epoch_9
VOCAB=/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
POOL_DIR=/data/kun/jtrans/addressaware/eval/pools_filtered
FUNC_BLOCKS=/data/kun/jtrans/addressaware/eval/func_blocks_addr.json
GROUND_TRUTH=/data/kun/jtrans/addressaware/eval/ground_truth_addr.json
OUTPUT=/home/kun/Document/AAE/output/jtrans/addressaware_finetune_infonce_b64/eval_epoch_9.json

python evaluate_addressaware_pools.py \
  --checkpoint "$CHECKPOINT" \
  --vocab_path "$VOCAB" \
  --pool_dir   "$POOL_DIR" \
  --func_blocks "$FUNC_BLOCKS" \
  --ground_truth "$GROUND_TRUTH" \
  --batch_size 64 \
  --output "$OUTPUT"
