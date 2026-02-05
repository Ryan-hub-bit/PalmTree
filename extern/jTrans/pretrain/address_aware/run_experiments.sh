#!/bin/bash
#
# Run 3 experimental variations of address-aware pretraining:
# 1. MLM-only (no JTP task)
# 2. No binary_pos (only function_pos and bb_pos)
# 3. Baseline (both MLM+JTP, all 3 position levels)
#

set -e

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Common parameters
VOCAB_PATH="/home/kun/Document/AAE/strupos/vocab.py"
TRAIN_DATA="/path/to/corpus_train.txt"  # UPDATE THIS PATH
BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_EPOCHS=40
SAVE_EVERY=5

# Base output directory
BASE_OUTPUT="/home/kun/Document/AAE/output/addressaware_experiments"
mkdir -p "$BASE_OUTPUT"

echo "=========================================="
echo "Address-Aware jTrans Experimental Runs"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Vocab: $VOCAB_PATH"
echo "Train: $TRAIN_DATA"
echo "Base output: $BASE_OUTPUT"
echo ""

# ===========================================
# Experiment 1: MLM-only (no JTP)
# ===========================================
echo "=========================================="
echo "Experiment 1: MLM-only (no JTP)"
echo "=========================================="
OUTPUT_DIR="$BASE_OUTPUT/exp1_mlm_only"
python train_addressaware.py \
    --train_path "$TRAIN_DATA" \
    --vocab_path "$VOCAB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --save_every $SAVE_EVERY \
    --no_jtp \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Exp1 completed: $OUTPUT_DIR"
echo ""

# ===========================================
# Experiment 2: No binary_pos (only fn + bb)
# ===========================================
echo "=========================================="
echo "Experiment 2: No binary_pos"
echo "=========================================="
OUTPUT_DIR="$BASE_OUTPUT/exp2_no_binary_pos"
python train_addressaware.py \
    --train_path "$TRAIN_DATA" \
    --vocab_path "$VOCAB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --save_every $SAVE_EVERY \
    --no_binary_pos \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Exp2 completed: $OUTPUT_DIR"
echo ""

# ===========================================
# Experiment 3: Baseline (MLM+JTP, all 3 pos)
# ===========================================
echo "=========================================="
echo "Experiment 3: Baseline (MLM+JTP, all pos)"
echo "=========================================="
OUTPUT_DIR="$BASE_OUTPUT/exp3_baseline"
python train_addressaware.py \
    --train_path "$TRAIN_DATA" \
    --vocab_path "$VOCAB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --save_every $SAVE_EVERY \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Exp3 completed: $OUTPUT_DIR"
echo ""

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved to: $BASE_OUTPUT"
echo ""
echo "Next steps:"
echo "1. Finetune each pretrained model on function similarity"
echo "2. Evaluate on pool datasets"
echo "3. Compare MRR and Recall@K metrics"
