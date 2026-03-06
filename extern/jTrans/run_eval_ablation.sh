#!/bin/bash
# Ablation study: eval-time enhancements for addressaware finetune v2
# Run all combinations and compare results

set -e
cd /home/kun/Document/AAE/extern/jTrans

export CUDA_VISIBLE_DEVICES=0

CHECKPOINT="/home/kun/Document/AAE/output/addressaware_finetune_v2/finetune_epoch_9"
VOCAB="pretrain/address_aware"
POOL_DIR="/data/kun/jtrans/addressaware/dedup/pools_dedup"
FUNC_BLOCKS="/data/kun/jtrans/addressaware/eval/func_blocks_addr.json"
POOL_SIZE=10000

BASE_CMD="python evaluate_addressaware_pools.py \
  --checkpoint $CHECKPOINT \
  --vocab_path $VOCAB \
  --pool_dir $POOL_DIR \
  --func_blocks $FUNC_BLOCKS \
  --pool_size $POOL_SIZE \
  --batch_size 32"

echo "============================================"
echo "  Eval Ablation Study - finetune_v2 epoch 9"
echo "============================================"

echo ""
echo ">>> [1/5] Baseline (no enhancements)"
echo "============================================"
$BASE_CMD 2>&1 | tee /tmp/eval_baseline.log

echo ""
echo ">>> [2/5] Mask binary_pos"
echo "============================================"
$BASE_CMD --mask_binary_pos 2>&1 | tee /tmp/eval_mask_bp.log

echo ""
echo ">>> [3/5] Multi-layer pool (last 4 layers)"
echo "============================================"
$BASE_CMD --multi_layer_pool 4 2>&1 | tee /tmp/eval_multilayer.log

echo ""
echo ">>> [4/5] PCA Whitening"
echo "============================================"
$BASE_CMD --whiten 2>&1 | tee /tmp/eval_whiten.log

echo ""
echo ">>> [5/5] Mask binary_pos + Whitening"
echo "============================================"
$BASE_CMD --mask_binary_pos --whiten 2>&1 | tee /tmp/eval_mask_whiten.log

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "============================================"
echo ""
echo "Logs saved to /tmp/eval_*.log"
echo "To extract results:"
echo "  grep -A4 'SUMMARY' /tmp/eval_*.log"
