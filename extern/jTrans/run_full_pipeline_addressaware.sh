#!/bin/bash
# Complete pipeline: Fine-tune address-aware model then evaluate with different pool sizes

set -e  # Exit on error

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# GPU Configuration
GPU_IDS=${1:-"1"}  # Default to GPU 0 if not specified
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Configuration
MODEL_PATH="/home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_10"
TOKENIZER="/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware"
FUNC_BLOCKS="/data/kun/jtransdata/func_blocks_addr.json"
GROUND_TRUTH="/data/kun/jtransdata/ground_truth_addr.json"
OUTPUT_DIR="/home/kun/Document/AAE/output/jtrans/addressaware_finetune"
EMBEDDING_CACHE="${OUTPUT_DIR}/embeddings_cache"

# Training parameters
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
LR=1e-5
EPOCH=20
WEIGHT_DECAY=0.01
FREEZE_CNT=10
DATA_RATIO=1  # Use 10% of data for faster training

# Evaluation parameters
POOL_SIZES="100 1000 10000"
QUERY_RATIO=0.1  # Use 10% of queries for evaluation

echo "========================================"
echo "Address-Aware Model Pipeline"
echo "========================================"
echo "Training epochs: ${EPOCH}"
echo "Data ratio: ${DATA_RATIO}"
echo "Output: ${OUTPUT_DIR}"
echo ""

echo "========================================"
echo "Step 1: Fine-tuning address-aware model"
echo "========================================"

python finetune.py \
    --model_type addressaware \
    --data_type json \
    --func_blocks ${FUNC_BLOCKS} \
    --ground_truth ${GROUND_TRUTH} \
    --tokenizer ${TOKENIZER} \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --lr ${LR} \
    --epoch ${EPOCH} \
    --weight_decay ${WEIGHT_DECAY} \
    --freeze_cnt ${FREEZE_CNT} \
    --data_ratio ${DATA_RATIO}

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
echo "Model saved to: ${OUTPUT_DIR}/finetune_epoch_${EPOCH}"
echo "Run separate evaluation script for testing."
echo ""
