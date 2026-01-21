#!/bin/bash

# jTrans_instr Pretraining Script
# MLM + JTP with instruction-level embeddings

# Set GPU (change this to your GPU ID)
export CUDA_VISIBLE_DEVICES=1

# Paths (modify these to match your setup)
TRAIN_PATH="/data/kun/jtrans_instr/instr_pretrain.txt"
# TEST_PATH="/data/kun/jtrans_instr/instr_test.txt"  # Optional - comment out to skip validation
TOKENIZER_PATH="/home/kun/Document/AAE/extern/jTrans_instr/jtrans_tokenizer"
OUTPUT_DIR="/home/kun/Document/AAE/output/jtrans_instr/pretrain_run1"

# Model architecture
HIDDEN_SIZE=768
NUM_LAYERS=12
NUM_HEADS=12
INTERMEDIATE_SIZE=3072
MAX_INSTRUCTIONS=201

# Training hyperparameters
BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_EPOCHS=10
WARMUP_STEPS=10000
MAX_LEN=512
MLM_PROB=0.15
JTP_PROB=0.20
NUM_WORKERS=4

# Data sampling (for quick testing)
DATA_RATIO=0.0001  # 0.0001 = 0.01%, 0.01 = 1%, 1.0 = 100%

# Create output directory
mkdir -p $OUTPUT_DIR

# Log configuration
echo "=========================================="
echo "jTrans_instr Pretraining"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Train data: $TRAIN_PATH"
# echo "Test data: $TEST_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Architecture:"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Layers: $NUM_LAYERS"
echo "  Attention heads: $NUM_HEADS"
echo "  Max instructions: $MAX_INSTRUCTIONS"
echo ""
echo "Hyperparameters:"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Max length: $MAX_LEN"
echo "  MLM probability: $MLM_PROB"
echo "  JTP probability: $JTP_PROB"
echo "  Data ratio: $DATA_RATIO ($(echo "$DATA_RATIO * 100" | bc -l)%)"
echo "=========================================="

# Run training
python train_instr.py \
    --train_path $TRAIN_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --output_dir $OUTPUT_DIR \
    --hidden_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_LAYERS \
    --num_attention_heads $NUM_HEADS \
    --intermediate_size $INTERMEDIATE_SIZE \
    --max_instructions $MAX_INSTRUCTIONS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --max_len $MAX_LEN \
    --mlm_probability $MLM_PROB \
    --jtp_probability $JTP_PROB \
    --num_workers $NUM_WORKERS \
    --data_ratio $DATA_RATIO \
    --save_every 1
    # Add --test_path $TEST_PATH to enable validation

echo ""
echo "=========================================="
echo "Training completed!"
echo "Check results in: $OUTPUT_DIR"
echo "=========================================="
