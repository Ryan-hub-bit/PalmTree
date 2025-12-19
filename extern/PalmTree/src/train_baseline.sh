#!/bin/bash
#
# Train BASELINE model (Original BERT without address features)
#
# This trains a standard BERT model where:
# - Position numbers in address() are MASKED (set to 0.0)
# - NO address embeddings are used
# - Standard BERT architecture
#

set -e

echo "========================================"
echo "Training BASELINE Model"
echo "========================================"
echo "This is the original BERT without address features"
echo "Position numbers are masked for fair comparison"
echo "========================================"
echo ""

# Data paths
TRAIN_CFG="/data/kun/palmtreedata/cfg_train_2.txt"
TRAIN_DFG="/data/kun/palmtreedata/dfg_train_2.txt"
TEST_CFG="/data/kun/palmtreedata/cfg_test_2.txt"
TEST_DFG="/data/kun/palmtreedata/dfg_test_2.txt"
VOCAB="./vocab_base.pkl"

# Model hyperparameters
HIDDEN_SIZE=768
N_LAYERS=12
ATTN_HEADS=12
DROPOUT=0.1

# Training hyperparameters
BATCH_SIZE=16
LEARNING_RATE=1e-4
NUM_EPOCHS=20
WARMUP_STEPS=10000
SEQ_LEN=512

# Task configuration
INSTRUCTION_MASK_PROB=0.25
TOKEN_MASK_PROB=0.15

# Output
OUTPUT_DIR="./output_comparison/baseline"
mkdir -p ${OUTPUT_DIR}

# CUDA settings
export CUDA_VISIBLE_DEVICES=0
CUDA_DEVICES=(0)

# Data percentage (use 1.0 for full data, smaller for testing)
DATA_PERCENTAGE=1.0

echo "Configuration:"
echo "  Mode: BASELINE (no address features)"
echo "  Train CFG: ${TRAIN_CFG}"
echo "  Train DFG: ${TRAIN_DFG}"
echo "  Vocab: ${VOCAB}"
echo "  Hidden size: ${HIDDEN_SIZE}"
echo "  Layers: ${N_LAYERS}"
echo "  Attention heads: ${ATTN_HEADS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python3 train_comparison.py \
  --mode baseline \
  --train_cfg ${TRAIN_CFG} \
  --train_dfg ${TRAIN_DFG} \
  --test_cfg ${TEST_CFG} \
  --test_dfg ${TEST_DFG} \
  --vocab_path ${VOCAB} \
  --hidden_size ${HIDDEN_SIZE} \
  --n_layers ${N_LAYERS} \
  --attn_heads ${ATTN_HEADS} \
  --dropout ${DROPOUT} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --warmup_steps ${WARMUP_STEPS} \
  --seq_len ${SEQ_LEN} \
  --instruction_mask_prob ${INSTRUCTION_MASK_PROB} \
  --token_mask_prob ${TOKEN_MASK_PROB} \
  --enable_imc \
  --enable_mlm \
  --output_dir ${OUTPUT_DIR} \
  --cuda_devices ${CUDA_DEVICES[@]} \
  --data_percentage ${DATA_PERCENTAGE} \
  --num_workers 4 \
  --log_freq 100

echo ""
echo "========================================"
echo "BASELINE Training Complete!"
echo "========================================"
echo "Model saved to: ${OUTPUT_DIR}"
echo "========================================"
