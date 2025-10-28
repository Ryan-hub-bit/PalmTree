#!/bin/bash

# Script to evaluate windows8 model on test data

# Activate palmtree conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate palmtree

MODEL_DIR="/home/louie/PalmTree/result/windows8"
TEST_DATA="/home/louie/PalmTree/datalong/test"
VOCAB_PATH="/home/louie/PalmTree/result/windows8/vocab"
OUTPUT_DIR="/home/louie/PalmTree/evaluation_results/windows8"
BATCH_SIZE=32
SEQ_LEN=100

echo "=========================================="
echo "Starting Model Evaluation"
echo "=========================================="
echo "Model directory: $MODEL_DIR"
echo "Test data: $TEST_DATA"
echo "Vocabulary: $VOCAB_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Sequence length: $SEQ_LEN"
echo "=========================================="
echo ""

python test_address_model.py \
    --model_dir "$MODEL_DIR" \
    --test_data "$TEST_DATA" \
    --vocab_path "$VOCAB_PATH" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="
