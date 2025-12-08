#!/bin/bash

# Run MLM + IMC + IMD + Scope + Address Training
# Full instruction masking (CFG + DFG) with scope prediction and address embeddings

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "MLM + IMC + IMD + Scope + Address Training"
echo "========================================"
echo ""

# Experiment configuration
EXPERIMENT="mlm_imc_scope_imd_address"
MODEL_NAME="${EXPERIMENT}"
OUTPUT_DIR="/work/kliu14/haapr/output/${MODEL_NAME}"
LOG_DIR="/work/kliu14/haapr/log/${MODEL_NAME}"

# Task flags
TASKS="--enable_mlm --enable_imc --enable_imd --enable_scope"  # MLM + IMC + IMD + Scope
ADDRESS_FLAG=""  # Use address embeddings

# Model parameters
HIDDEN=768
LAYERS=12
ATTN_HEADS=12
SEQ_LEN=60

# Training parameters
BATCH_SIZE=256
LR=1e-4
EPOCHS=10
WARMUP=1000
NUM_WORKERS=4
EARLY_STOPPING=3

# Masking parameters
TOKEN_MASK_PROB=0.15          # Standard MLM masking rate
INSTRUCTION_MASK_PROB=0.125    # Instruction-level masking rate

# Data parameters
TRAIN_PERCENTAGE=1.0  # Use 100% of training data
VAL_PERCENTAGE=1.0    # Use 100% of validation data

# Data paths
VOCAB_PATH="./strupos/vocab.pkl"
CFG_TRAIN="/work/kliu14/vardataset/train_cfg.txt"
DFG_TRAIN="/work/kliu14/vardataset/train_dfg.txt"
CFG_VAL="/work/kliu14/vardataset/val_cfg.txt"
DFG_VAL="/work/kliu14/vardataset/val_dfg.txt"
SCOPE_TRAIN="/work/kliu14/vardataset/train_scope.txt"
SCOPE_VAL="/work/kliu14/vardataset/val_scope.txt"

# Multi-GPU
CUDA="--cuda"
MULTI_GPU=""
# MULTI_GPU="--multi_gpu"

echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Tasks: MLM + IMC (CFG instruction masking) + IMD (DFG instruction masking) + Scope"
echo "  Address Embedding: Enabled"
echo "  Token mask rate: ${TOKEN_MASK_PROB}"
echo "  Instruction mask rate: ${INSTRUCTION_MASK_PROB}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LR}"
echo "  Epochs: ${EPOCHS}"
echo "  Train data: ${TRAIN_PERCENTAGE}"
echo "  Val data: ${VAL_PERCENTAGE}"
echo ""

# Create output and log directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Run training
echo "Starting training..."
echo ""

python strupos/train_with_instruction_mask.py \
  --cfg_train "${CFG_TRAIN}" \
  --dfg_train "${DFG_TRAIN}" \
  --cfg_val "${CFG_VAL}" \
  --dfg_val "${DFG_VAL}" \
  --scope_train "${SCOPE_TRAIN}" \
  --scope_val "${SCOPE_VAL}" \
  --vocab "${VOCAB_PATH}" \
  --hidden ${HIDDEN} \
  --layers ${LAYERS} \
  --attn_heads ${ATTN_HEADS} \
  --seq_len ${SEQ_LEN} \
  --dropout 0.1 \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --warmup_steps ${WARMUP} \
  --num_workers ${NUM_WORKERS} \
  --early_stopping_patience ${EARLY_STOPPING} \
  --token_mask_prob ${TOKEN_MASK_PROB} \
  --instruction_mask_prob ${INSTRUCTION_MASK_PROB} \
  --data_percentage ${TRAIN_PERCENTAGE} \
  --val_percentage ${VAL_PERCENTAGE} \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --resume \
  ${TASKS} \
  ${ADDRESS_FLAG} \
  ${CUDA} ${MULTI_GPU}

if [ $? -eq 0 ]; then
  echo ""
  echo -e "${GREEN}========================================"
  echo "Training Completed Successfully!"
  echo "========================================${NC}"
  echo ""
  echo "Model saved to: ${OUTPUT_DIR}"
  echo "  - best_model.pt"
  echo "  - best_bert.pt"
  echo "  - checkpoint_latest.pt"
  echo ""
  echo "Logs saved to: ${LOG_DIR}"
else
  echo -e "${YELLOW}Training failed or was interrupted.${NC}"
  exit 1
fi
