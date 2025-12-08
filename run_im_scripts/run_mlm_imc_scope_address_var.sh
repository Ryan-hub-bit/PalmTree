#!/bin/bash

# Run Instruction Masking Training
# This script trains a model with instruction-level masking

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "Instruction Masking Training"
echo "========================================"
echo ""

# =============================================================================
# Experiment Selection
# =============================================================================
# Uncomment ONE of the following experiment blocks:
# Experiment name is auto-generated from TASKS + ADDRESS + VAR settings

# -----------------------------------------------------------------------------
# Experiment 1: IMC + MLM + Address + Var (Full features)
# -> Auto-generates: imc_mlm_addr_var
# -----------------------------------------------------------------------------
TASKS="--enable_imc --enable_mlm --enable_scope"
USE_ADDRESS_EMBEDDING=true
USE_VAR_EMBEDDING=true

# -----------------------------------------------------------------------------
# Experiment 2: IMC + MLM + Address (No Var embedding)
# -> Auto-generates: imc_mlm_addr
# -----------------------------------------------------------------------------
# TASKS="--enable_imc --enable_mlm"
# USE_ADDRESS_EMBEDDING=true
# USE_VAR_EMBEDDING=false

# -----------------------------------------------------------------------------
# Experiment 3: IMC + MLM + Var (No Address embedding)
# -> Auto-generates: imc_mlm_var
# -----------------------------------------------------------------------------
# TASKS="--enable_imc --enable_mlm"
# USE_ADDRESS_EMBEDDING=false
# USE_VAR_EMBEDDING=true

# -----------------------------------------------------------------------------
# Experiment 4: IMC + MLM (Baseline - No Address, No Var)
# -> Auto-generates: imc_mlm
# -----------------------------------------------------------------------------
# TASKS="--enable_imc --enable_mlm"
# USE_ADDRESS_EMBEDDING=false
# USE_VAR_EMBEDDING=false

# -----------------------------------------------------------------------------
# Experiment 5: IMC + IMD + MLM + Address + Var (Full with DFG)
# -> Auto-generates: imc_imd_mlm_addr_var
# -----------------------------------------------------------------------------
# TASKS="--enable_imc --enable_imd --enable_mlm"
# USE_ADDRESS_EMBEDDING=true
# USE_VAR_EMBEDDING=true

# -----------------------------------------------------------------------------
# Experiment 6: IMC + MLM + Scope + Address + Var (Full with Scope)
# -> Auto-generates: imc_mlm_scope_addr_var
# -----------------------------------------------------------------------------
# TASKS="--enable_imc --enable_mlm --enable_scope"
# USE_ADDRESS_EMBEDDING=true
# USE_VAR_EMBEDDING=true

# =============================================================================
# Convert flags to command line arguments
# =============================================================================
if [ "$USE_ADDRESS_EMBEDDING" = true ]; then
  ADDRESS_FLAG=""
  ADDR_SUFFIX="_addr"
else
  ADDRESS_FLAG="--disable_address_embedding"
  ADDR_SUFFIX=""
fi

if [ "$USE_VAR_EMBEDDING" = true ]; then
  VAR_FLAG=""
  VAR_SUFFIX="_var"
else
  VAR_FLAG="--disable_var_embedding"
  VAR_SUFFIX=""
fi

# Auto-generate experiment name from TASKS and embeddings
# Extract task names from TASKS string
TASK_NAME=""
if [[ "$TASKS" == *"--enable_imc"* ]]; then TASK_NAME="${TASK_NAME}_imc"; fi
if [[ "$TASKS" == *"--enable_imd"* ]]; then TASK_NAME="${TASK_NAME}_imd"; fi
if [[ "$TASKS" == *"--enable_mlm"* ]]; then TASK_NAME="${TASK_NAME}_mlm"; fi
if [[ "$TASKS" == *"--enable_scope"* ]]; then TASK_NAME="${TASK_NAME}_scope"; fi
# Remove leading underscore
TASK_NAME="${TASK_NAME#_}"

# Generate full experiment name
EXPERIMENT="${TASK_NAME}${ADDR_SUFFIX}${VAR_SUFFIX}"

# Configuration
MODEL_NAME="${EXPERIMENT}"
OUTPUT_DIR="../output/${MODEL_NAME}"
LOG_DIR="../log/${MODEL_NAME}"

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
TOKEN_MASK_PROB=0.15      # Standard MLM masking rate
INSTRUCTION_MASK_PROB=0.125  # Instruction-level masking rate

# Data parameters
TRAIN_PERCENTAGE=1  # Use 20% of training data
VAL_PERCENTAGE=1    # Use 20% of validation data

# Data paths
VOCAB_PATH="./vocab.pkl"
CFG_TRAIN="/work/kliu14/vardataset/train_cfg.txt"
DFG_TRAIN="/work/kliu14/vardataset/train_dfg.txt"
CFG_VAL="/work/kliu14/vardataset/val_cfg.txt"
DFG_VAL="/work/kliu14/vardataset/val_dfg.txt"
SCOPE_TRAIN="/work/kliu14/vardataset/train_scope.txt"
SCOPE_VAL="/work/kliu14/vardataset/val_scope.txt"
# Test data (only used for vocab generation, not training)
CFG_TEST="/work/kliu14/vardataset/test_cfg.txt"
DFG_TEST="/work/kliu14/vardataset/test_dfg.txt"

# Multi-GPU
CUDA="--cuda"
MULTI_GPU=""
# MULTI_GPU="--multi_gpu"

echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Tasks: IMC (instruction masking for CFG)"
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
# Note: vocab.pkl will be automatically created if it doesn't exist
echo "Starting training..."
echo ""

python train.py \
  --cfg_train "${CFG_TRAIN}" \
  --dfg_train "${DFG_TRAIN}" \
  --cfg_val "${CFG_VAL}" \
  --dfg_val "${DFG_VAL}" \
  --scope_train "${SCOPE_TRAIN}" \
  --scope_val "${SCOPE_VAL}" \
  --cfg_test "${CFG_TEST}" \
  --dfg_test "${DFG_TEST}" \
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
  ${VAR_FLAG} \
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