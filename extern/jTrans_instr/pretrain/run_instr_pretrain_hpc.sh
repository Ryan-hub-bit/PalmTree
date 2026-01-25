#!/bin/bash

# jTrans_instr Pretraining Script for HPC
# MLM + JTP with instruction-level embeddings
# PRETRAINING: Uses single combined file (no train/val/test split)
#
# Usage: ./run_instr_pretrain_hpc.sh [data_ratio]
# Example: ./run_instr_pretrain_hpc.sh 0.1    # Use 10% of data
#          ./run_instr_pretrain_hpc.sh 1.0    # Use 100% of data (default)

# Data ratio (default: 1.0 = 100% of data)
DATA_RATIO="${1:-1.0}"  # Changed from 0.0001 to 1.0 for full training

# Use all available GPUs (default: 0,1,2,3 for 4 GPUs)
# SLURM will set CUDA_VISIBLE_DEVICES automatically, but if running locally, set it here
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
# Uncomment for debugging (makes CUDA synchronous, slower)
# export CUDA_LAUNCH_BLOCKING=1

# Paths (modify these to match your HPC setup)
TRAIN_PATH="/work/kliu14/jtransdata/instr_pretrain.txt"
TOKENIZER_PATH="/home/kliu14/AAE/extern/jTrans_instr/jtrans_tokenizer"
OUTPUT_DIR="/work/kliu14/jtransdata/instr_output"

# Model architecture
HIDDEN_SIZE=768
NUM_LAYERS=12
NUM_HEADS=12
INTERMEDIATE_SIZE=3072
MAX_INSTRUCTIONS=201

# Training hyperparameters
BATCH_SIZE=128  # Reduced from 256 to avoid OOM. Increase gradually if you have more GPU memory
LEARNING_RATE=1e-4
NUM_EPOCHS=10
WARMUP_STEPS=10000
MAX_LEN=512
MLM_PROB=0.15
JTP_PROB=0.20
NUM_WORKERS=16

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Starting jTrans_instr Pretraining"
echo "Data ratio: $DATA_RATIO ($(echo "$DATA_RATIO * 100" | bc)% of training data)"
echo "=========================================="

# Debug: Check vocabulary size before training
echo "Checking vocabulary..."
python3 <<'VOCABCHECK'
from transformers import BertTokenizer
import sys

tokenizer_path = "/work/kliu14/jtrans_instr/jtrans_tokenizer"
try:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: [PAD]={tokenizer.pad_token_id}, [UNK]={tokenizer.unk_token_id}, [MASK]={tokenizer.mask_token_id}")
    # Check for instr_addr tokens
    if 'instr_addr_0' in tokenizer.vocab:
        print(f"instr_addr_0 ID: {tokenizer.vocab['instr_addr_0']}")
        print(f"instr_addr_200 ID: {tokenizer.vocab.get('instr_addr_200', 'NOT FOUND')}")
    else:
        print("WARNING: instr_addr tokens not found in vocabulary!")
except Exception as e:
    print(f"ERROR loading tokenizer: {e}")
    sys.exit(1)
VOCABCHECK

# Log configuration
echo ""
echo "Configuration:"
echo "  Train data: $TRAIN_PATH"
echo "  Tokenizer: $TOKENIZER_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Architecture: ${HIDDEN_SIZE}d, ${NUM_LAYERS} layers, ${NUM_HEADS} heads"
echo "  Max instructions: $MAX_INSTRUCTIONS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Data ratio: $DATA_RATIO"
echo "=========================================="

# Run training
python3 train_instr.py \
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
    --data_ratio "$DATA_RATIO" \
    --save_every 1

echo ""
echo "=========================================="
echo "Training completed!"
echo "Check results in: $OUTPUT_DIR"
echo "=========================================="
