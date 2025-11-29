#!/bin/bash

# Automated Ablation Study Runner
# Runs all experiments in a specific order to evaluate different task combinations
# 
# Order of experiments:
# 1. MLM + Address
# 2. MLM + Scope (no address)
# 3. MLM + Scope + Address
# 4. MLM + NSP-CFG
# 5. MLM + NSP-CFG + Address
# 6. MLM + NSP-DFG
# 7. MLM + NSP-DFG + Address
# 8. MLM + NSP-CFG + NSP-DFG + Scope
# 9. MLM + NSP-CFG + NSP-DFG + Scope + Address

set -e # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "========================================"
echo "AUTOMATED ABLATION STUDY"
echo "========================================"
echo ""
echo "This script will run 9 experiments:"
echo "  1. MLM + Address"
echo "  2. MLM + Scope"
echo "  3. MLM + Scope + Address"
echo "  4. MLM + NSP-CFG"
echo "  5. MLM + NSP-CFG + Address"
echo "  6. MLM + NSP-DFG"
echo "  7. MLM + NSP-DFG + Address"
echo "  8. MLM + NSP-CFG + NSP-DFG + Scope"
echo "  9. MLM + NSP-CFG + NSP-DFG + Scope + Address"
echo ""
echo "========================================"
echo ""

# Common configuration
CFG_TRAIN="/data/kun/dataset/train_cfg.txt"
DFG_TRAIN="/data/kun/dataset/train_dfg.txt"
CFG_VAL="/data/kun/dataset/val_cfg.txt"
DFG_VAL="/data/kun/dataset/val_dfg.txt"
CFG_TEST="/data/kun/dataset/test_cfg.txt"
DFG_TEST="/data/kun/dataset/test_dfg.txt"
SCOPE_TRAIN="/data/kun/dataset/scope_train.txt"
SCOPE_VAL="/data/kun/dataset/scope_val.txt"
VOCAB_PATH="./vocab.pkl"

# Model architecture
HIDDEN=768
LAYERS=12
ATTN_HEADS=12
SEQ_LEN=60
NSP_CONTENT_MAX=20
DROPOUT=0.1

# Training hyperparameters
EPOCHS=10
BATCH_SIZE=1024
LR=1e-4
WARMUP_STEPS=10000
NUM_WORKERS=4
EARLY_STOPPING_PATIENCE=5

# Data processing
MASK_PROB=0.15
NSP_PROB=0.5
TRAIN_PERCENTAGE=0.2
VAL_PERCENTAGE=0.2

# Device
CUDA="--cuda"
MULTI_GPU="--multi_gpu"

# Check if vocab exists
if [ ! -f "$VOCAB_PATH" ]; then
  echo -e "${YELLOW}⚠ Vocabulary file not found. Creating it first...${NC}"
  python create_vocab.py
  if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to create vocabulary${NC}"
    exit 1
  fi
  echo -e "${GREEN}✓ Vocabulary created${NC}"
  echo ""
fi

# Function to run a single experiment
run_experiment() {
  local exp_num=$1
  local exp_name=$2
  local mlm=$3
  local nsp_cfg=$4
  local nsp_dfg=$5
  local scope=$6
  local address=$7
  
  echo ""
  echo -e "${CYAN}========================================"
  echo "EXPERIMENT ${exp_num}/9: ${exp_name}"
  echo "========================================${NC}"
  echo ""
  
  # Build task flags
  TASK_FLAGS=""
  if [ "$mlm" = true ]; then
    TASK_FLAGS="${TASK_FLAGS} --enable_mlm"
  else
    TASK_FLAGS="${TASK_FLAGS} --disable_mlm"
  fi
  if [ "$nsp_cfg" = true ]; then
    TASK_FLAGS="${TASK_FLAGS} --enable_nsp_cfg"
  fi
  if [ "$nsp_dfg" = true ]; then
    TASK_FLAGS="${TASK_FLAGS} --enable_nsp_dfg"
  fi
  if [ "$scope" = true ]; then
    TASK_FLAGS="${TASK_FLAGS} --enable_scope"
  fi
  if [ "$address" = true ]; then
    TASK_FLAGS="${TASK_FLAGS} --use_address_embedding"
  fi
  
  # Generate model name
  MODEL_NAME=""
  if [ "$mlm" = true ]; then
    MODEL_NAME="${MODEL_NAME}_mlm"
  fi
  if [ "$nsp_cfg" = true ]; then
    MODEL_NAME="${MODEL_NAME}_nsp_cfg"
  fi
  if [ "$nsp_dfg" = true ]; then
    MODEL_NAME="${MODEL_NAME}_nsp_dfg"
  fi
  if [ "$scope" = true ]; then
    MODEL_NAME="${MODEL_NAME}_scope"
  fi
  if [ "$address" = true ]; then
    MODEL_NAME="${MODEL_NAME}_address"
  fi
  MODEL_NAME="${MODEL_NAME#_}"
  
  OUTPUT_DIR="../output/${MODEL_NAME}"
  LOG_DIR="../log/${MODEL_NAME}"
  
  echo "Configuration:"
  echo "  Model name: ${MODEL_NAME}"
  echo "  Tasks:"
  echo "    - MLM: ${mlm}"
  echo "    - NSP-CFG: ${nsp_cfg}"
  echo "    - NSP-DFG: ${nsp_dfg}"
  echo "    - SCOPE: ${scope}"
  echo "    - Address Embedding: ${address}"
  echo ""
  
  # Create output directories
  mkdir -p "${OUTPUT_DIR}"
  mkdir -p "${LOG_DIR}"
  
  # Run training
  python train_from_scratch.py \
    --cfg_train "${CFG_TRAIN}" \
    --dfg_train "${DFG_TRAIN}" \
    --cfg_val "${CFG_VAL}" \
    --dfg_val "${DFG_VAL}" \
    --cfg_test "${CFG_TEST}" \
    --dfg_test "${DFG_TEST}" \
    --scope_train "${SCOPE_TRAIN}" \
    --scope_val "${SCOPE_VAL}" \
    --vocab "${VOCAB_PATH}" \
    --hidden ${HIDDEN} \
    --layers ${LAYERS} \
    --attn_heads ${ATTN_HEADS} \
    --seq_len ${SEQ_LEN} \
    --nsp_content_max ${NSP_CONTENT_MAX} \
    --dropout ${DROPOUT} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --warmup_steps ${WARMUP_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --mask_prob ${MASK_PROB} \
    --nsp_prob ${NSP_PROB} \
    --data_percentage ${TRAIN_PERCENTAGE} \
    --val_percentage ${VAL_PERCENTAGE} \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}" \
    ${TASK_FLAGS} \
    ${CUDA} ${MULTI_GPU} \
    2>&1 | tee "${LOG_DIR}/training.log"
  
  if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Experiment ${exp_num} completed successfully!${NC}"
    echo ""
  else
    echo ""
    echo -e "${RED}✗ Experiment ${exp_num} failed!${NC}"
    echo ""
    exit 1
  fi
}

# Record start time
START_TIME=$(date +%s)
echo "Start time: $(date)"
echo ""

# Run all experiments in order
run_experiment 1 "MLM + Address" true false false false true
run_experiment 2 "MLM + Scope" true false false true false
run_experiment 3 "MLM + Scope + Address" true false false true true
run_experiment 4 "MLM + NSP-CFG" true true false false false
run_experiment 5 "MLM + NSP-CFG + Address" true true false false true
run_experiment 6 "MLM + NSP-DFG" true false true false false
run_experiment 7 "MLM + NSP-DFG + Address" true false true false true
run_experiment 8 "MLM + NSP-CFG + NSP-DFG + Scope" true true true true false
run_experiment 9 "MLM + NSP-CFG + NSP-DFG + Scope + Address" true true true true true

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${GREEN}========================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================${NC}"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "End time: $(date)"
echo ""
echo "Results saved in:"
echo "  - Models: ../output/"
echo "  - Logs: ../log/"
echo ""
echo "Experiments completed:"
echo "  ✓ 1. MLM + Address"
echo "  ✓ 2. MLM + Scope"
echo "  ✓ 3. MLM + Scope + Address"
echo "  ✓ 4. MLM + NSP-CFG"
echo "  ✓ 5. MLM + NSP-CFG + Address"
echo "  ✓ 6. MLM + NSP-DFG"
echo "  ✓ 7. MLM + NSP-DFG + Address"
echo "  ✓ 8. MLM + NSP-CFG + NSP-DFG + Scope"
echo "  ✓ 9. MLM + NSP-CFG + NSP-DFG + Scope + Address"
echo ""
