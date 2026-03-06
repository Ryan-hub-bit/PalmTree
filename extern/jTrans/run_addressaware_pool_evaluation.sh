#!/bin/bash
# Evaluate address-aware pools with finetuned model
#
# Usage:
#   ./run_addressaware_pool_evaluation.sh <model_checkpoint> <vocab_path>
#
# Example:
#   ./run_addressaware_pool_evaluation.sh \
#       /home/kun/Document/AAE/output/jtrans/addressaware_finetune_targetO3/finetune_epoch_3 \
#       /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware

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
    echo "  $0 output/jtrans/addressaware_finetune/finetune_epoch_2 extern/jTrans/pretrain/address_aware"
    echo ""
    echo "  # Evaluate only 1000-size pools"
    echo "  $0 output/jtrans/addressaware_finetune/finetune_epoch_2 extern/jTrans/pretrain/address_aware 1000"
    echo ""
    echo "  # Evaluate only O0->O3 pairs with 10000 pool size"
    echo "  $0 output/jtrans/addressaware_finetune/finetune_epoch_2 extern/jTrans/pretrain/address_aware 10000 O0_vs_O3"
    exit 1
fi

MODEL_CHECKPOINT=$1
VOCAB_PATH=$2
POOL_SIZE=${3:-""}
OPT_PAIR=${4:-""}

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Configuration (deduped: cross-binary duplicate functions removed)
POOL_DIR="/data/kun/jtrans/addressaware/dedup/pools_dedup"
# Use EVAL data (pools reference eval function IDs, not training IDs)
FUNC_BLOCKS="/data/kun/jtrans/addressaware/eval/func_blocks_addr.json"
MAX_LEN=512  # Same as finetune
BATCH_SIZE=32
DEVICE="cuda"

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Build command
CMD="python evaluate_addressaware_pools.py \
  --checkpoint $MODEL_CHECKPOINT \
  --vocab_path $VOCAB_PATH \
  --pool_dir $POOL_DIR \
  --func_blocks $FUNC_BLOCKS \
  --max_len $MAX_LEN \
  --batch_size $BATCH_SIZE \
  --device $DEVICE"

# Add optional filters
if [ -n "$POOL_SIZE" ]; then
    CMD="$CMD --pool_size $POOL_SIZE"
fi

if [ -n "$OPT_PAIR" ]; then
    CMD="$CMD --opt_pair $OPT_PAIR"
fi

# Add output file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="/home/kun/Document/AAE/output/jtrans/addressaware_eval_${TIMESTAMP}.json"
CMD="$CMD --output $OUTPUT_FILE"

echo "=================================="
echo "Address-Aware Pool Evaluation"
echo "=================================="
echo ""
echo "Model: $MODEL_CHECKPOINT"
echo "Vocab: $VOCAB_PATH"
echo "Pool dir: $POOL_DIR"
echo "Func blocks: $FUNC_BLOCKS"
echo ""
if [ -n "$POOL_SIZE" ]; then
    echo "Filter: Pool size = $POOL_SIZE"
fi
if [ -n "$OPT_PAIR" ]; then
    echo "Filter: Opt pair = $OPT_PAIR"
fi
echo ""
echo "Output: $OUTPUT_FILE"
echo ""
echo "=================================="
echo ""

# Run evaluation
$CMD

echo ""
echo "=================================="
echo "Evaluation Completed!"
echo "=================================="
echo "Results saved to: $OUTPUT_FILE"
