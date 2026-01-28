#!/bin/bash

# Evaluate baseline pools with finetuned model
#
# Usage:
#   ./run_baseline_pool_evaluation.sh <model_checkpoint> <vocab_path>
#
# Example:
#   ./run_baseline_pool_evaluation.sh \
#       /home/kun/Document/AAE/output/jtrans/baseline_finetune/finetune_epoch_10 \
#       /home/kun/Document/AAE/extern/jTrans/pretrain/baseline

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model_checkpoint> <vocab_path> [pool_size] [opt_pair]"
    echo ""
    echo "Arguments:"
    echo "  model_checkpoint  Path to finetuned model checkpoint (required)"
    echo "  vocab_path        Path to vocabulary file (required)"
    echo "  pool_size         Optional: 100, 1000, or 10000 (evaluate specific size only)"
    echo "  opt_pair          Optional: O0_vs_O3, O1_vs_O3, or O2_vs_O3"
    echo ""
    echo "Examples:"
    echo "  # Evaluate all pools"
    echo "  # Evaluate all pools with finetuned model"
    echo "  $0 output/jtrans/baseline_finetune/finetune_epoch_3 extern/jTrans/pretrain/baseline"
    echo ""
    echo "  # Evaluate only 1000-size pools"
    echo "  $0 output/jtrans/baseline_finetune/finetune_epoch_3 extern/jTrans/pretrain/baseline 1000"
    echo ""
    echo "  # Evaluate only O0->O3 pairs with 10000 pool size"
    echo "  $0 output/jtrans/baseline_finetune/finetune_epoch_3 extern/jTrans/pretrain/baseline 10000 O0_vs_O3"
    exit 1
fi

MODEL_CHECKPOINT=$1
VOCAB_PATH=$2
POOL_SIZE=${3:-""}
OPT_PAIR=${4:-""}

# Configuration
POOL_DIR="/data/kun/jtrans/baseline/eval/pools_filtered"
FUNC_BLOCKS="/data/kun/jtrans/baseline/eval/func_blocks_baseline.json"
MAX_LEN=512  # Same as finetune
BATCH_SIZE=64
DEVICE="cuda"

# Build command
CMD="python3 evaluate_baseline_pools.py \
    --model $MODEL_CHECKPOINT \
    --vocab $VOCAB_PATH \
    --pool-dir $POOL_DIR \
    --func-blocks $FUNC_BLOCKS \
    --max-len $MAX_LEN \
    --batch-size $BATCH_SIZE \
    --device $DEVICE"

# Add optional filters
if [ -n "$POOL_SIZE" ]; then
    CMD="$CMD --pool-size $POOL_SIZE"
fi

if [ -n "$OPT_PAIR" ]; then
    CMD="$CMD --opt-pair $OPT_PAIR"
fi

echo "=========================================="
echo "Baseline Pool Evaluation"
echo "=========================================="
echo "Model:        $MODEL_CHECKPOINT"
echo "Vocab:        $VOCAB_PATH"
echo "Pool Dir:     $POOL_DIR"
echo "Func Blocks:  $FUNC_BLOCKS"
echo "Max Length:   $MAX_LEN"
echo "Batch Size:   $BATCH_SIZE"
echo "Device:       $DEVICE"
if [ -n "$POOL_SIZE" ]; then
    echo "Pool Size:    $POOL_SIZE"
fi
if [ -n "$OPT_PAIR" ]; then
    echo "Opt Pair:     $OPT_PAIR"
fi
echo "=========================================="
echo ""

# Run evaluation
eval $CMD

echo ""
echo "Evaluation complete!"
echo "Results saved to: $POOL_DIR/evaluation_results.json"
