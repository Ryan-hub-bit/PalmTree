#!/bin/bash
# Complete pipeline: Fine-tune model then evaluate with different pool sizes

set -e  # Exit on error

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# GPU Configuration
GPU_IDS=${1:-"1"}  # Default to GPU 0 if not specified
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Configuration
MODEL_PATH="/home/kun/Document/AAE/output/jtrans/baseline_pretrain/checkpoint_epoch_10"
TOKENIZER="/home/kun/Document/AAE/extern/jTrans/pretrain/baseline"
FUNC_BLOCKS="/data/kun/jtransdata/func_blocks_baseline.json"
GROUND_TRUTH="/data/kun/jtransdata/ground_truth_baseline.json"
OUTPUT_DIR="/home/kun/Document/AAE/output/jtrans/baseline_finetune"
EMBEDDING_CACHE="${OUTPUT_DIR}/embeddings_cache"

# Training parameters
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
LR=1e-5
EPOCH=5
WEIGHT_DECAY=0.01
FREEZE_CNT=10
DATA_RATIO=1.0  # Use 10% of data for faster training

# Evaluation parameters
POOL_SIZES="100 1000 10000"
QUERY_RATIO=0.1  # Use 10% of queries for evaluation

echo "========================================"
echo "Step 1: Fine-tuning model"
echo "========================================"
echo "Using GPU(s): ${GPU_IDS}"
echo "Training epochs: ${EPOCH}"
echo "Data ratio: ${DATA_RATIO}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python finetune.py \
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
echo "Step 2: Evaluating with different pool sizes"
echo "========================================"
echo "Pool sizes: ${POOL_SIZES}"
echo "Query ratio: ${QUERY_RATIO} (10% of queries)"
echo ""

# Get the final fine-tuned model
FINETUNED_MODEL="${OUTPUT_DIR}/finetune_epoch_${EPOCH}"

# Create embeddings directory
mkdir -p ${EMBEDDING_CACHE}

# Run evaluation with different pool sizes
python evaluate_with_pools.py \
    --model_path ${FINETUNED_MODEL} \
    --tokenizer ${TOKENIZER} \
    --func_blocks ${FUNC_BLOCKS} \
    --ground_truth ${GROUND_TRUTH} \
    --model_type baseline \
    --embedding_cache ${EMBEDDING_CACHE}/embeddings_all.pkl \
    --pool_sizes ${POOL_SIZES} \
    --query_ratio ${QUERY_RATIO} \
    --data_ratio ${DATA_RATIO} \
    --output_file ${OUTPUT_DIR}/pool_evaluation_results.txt

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "Fine-tuned model: ${FINETUNED_MODEL}"
echo "Evaluation results: ${OUTPUT_DIR}/pool_evaluation_results.txt"
echo ""
