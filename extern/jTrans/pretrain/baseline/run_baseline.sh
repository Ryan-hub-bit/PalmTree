#!/bin/bash

# Baseline jTrans Pretraining Script (MLM + JTP)
# Pure baseline without address-aware features

# Configuration
TRAIN_PATH="/path/to/your/train_data.txt"
TEST_PATH="/path/to/your/test_data.txt"
TOKENIZER_PATH="/path/to/your/tokenizer"
OUTPUT_DIR="./output_baseline"

# Model Architecture
HIDDEN_SIZE=768
NUM_LAYERS=12
NUM_HEADS=12
INTERMEDIATE_SIZE=3072
DROPOUT=0.1

# Training Hyperparameters
BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_EPOCHS=10
WARMUP_STEPS=10000
MAX_LEN=512

# Masking Probabilities
MLM_PROB=0.15  # Standard MLM masking
JTP_PROB=0.20  # jTrans paper uses 20% for jump tokens

# Data Loading
NUM_WORKERS=4

# Run training
python train_baseline.py \
    --train_path ${TRAIN_PATH} \
    --test_path ${TEST_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --hidden_size ${HIDDEN_SIZE} \
    --num_hidden_layers ${NUM_LAYERS} \
    --num_attention_heads ${NUM_HEADS} \
    --intermediate_size ${INTERMEDIATE_SIZE} \
    --hidden_dropout_prob ${DROPOUT} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --warmup_steps ${WARMUP_STEPS} \
    --max_len ${MAX_LEN} \
    --mlm_probability ${MLM_PROB} \
    --jtp_probability ${JTP_PROB} \
    --num_workers ${NUM_WORKERS} \
    --save_every 1

echo "Baseline training completed!"
echo "Check results in: ${OUTPUT_DIR}"
