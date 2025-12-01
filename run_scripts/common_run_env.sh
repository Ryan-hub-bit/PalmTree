#!/bin/bash
# Common environment for experiment run scripts
# Defines colors, dataset paths, model hyperparams and the run_experiment function.

# Determine repo root (parent of this run_scripts directory)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null && pwd)"

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Common configuration (dataset paths remain absolute where provided)
CFG_TRAIN="/work/kliu14/txtdataset/train_cfg.txt"
DFG_TRAIN="/work/kliu14/txtdataset/train_dfg.txt"
CFG_VAL="/work/kliu14/txtdataset/val_cfg.txt"
DFG_VAL="/work/kliu14/txtdataset/val_dfg.txt"
CFG_TEST="/work/kliu14/txtdataset/test_cfg.txt"
DFG_TEST="/work/kliu14/txtdataset/test_dfg.txt"
SCOPE_TRAIN="/work/kliu14/txtdataset/train_scope.txt"
SCOPE_VAL="/work/kliu14/txtdataset/val_scope.txt"
VOCAB_PATH="${REPO_ROOT}/strupos/vocab.pkl"

# Model architecture
HIDDEN=768
LAYERS=12
ATTN_HEADS=12
SEQ_LEN=60
NSP_CONTENT_MAX=20
DROPOUT=0.1

# Training hyperparameters
EPOCHS=10
BATCH_SIZE=256
LR=1e-4
WARMUP_STEPS=10000
NUM_WORKERS=4
EARLY_STOPPING_PATIENCE=3

# Data processing
MASK_PROB=0.15
NSP_PROB=0.5
TRAIN_PERCENTAGE=0.2
VAL_PERCENTAGE=0.2

# Device
CUDA="--cuda"
# MULTI_GPU="--multi_gpu"
MULTI_GPU=""

# Check if vocab exists (create if missing)
if [ ! -f "${VOCAB_PATH}" ]; then
  echo -e "${YELLOW}⚠ Vocabulary file not found. Creating it first...${NC}"
  python "${REPO_ROOT}/strupos/create_vocab.py"
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
  echo "EXPERIMENT ${exp_num}: ${exp_name}"
  echo "========================================${NC}"
  echo ""

  # Build task flags
  TASK_FLAGS=""
  if [ "${mlm}" = true ]; then
    TASK_FLAGS="${TASK_FLAGS} --enable_mlm"
  else
    TASK_FLAGS="${TASK_FLAGS} --disable_mlm"
  fi
  if [ "${nsp_cfg}" = true ]; then
    TASK_FLAGS="${TASK_FLAGS} --enable_nsp_cfg"
  fi
  if [ "${nsp_dfg}" = true ]; then
    TASK_FLAGS="${TASK_FLAGS} --enable_nsp_dfg"
  fi
  if [ "${scope}" = true ]; then
    TASK_FLAGS="${TASK_FLAGS} --enable_scope"
  fi
  if [ "${address}" = true ]; then
    TASK_FLAGS="${TASK_FLAGS} --use_address_embedding"
  fi

  # Generate model name
  MODEL_NAME=""
  if [ "${mlm}" = true ]; then
    MODEL_NAME="${MODEL_NAME}_mlm"
  fi
  if [ "${nsp_cfg}" = true ]; then
    MODEL_NAME="${MODEL_NAME}_nsp_cfg"
  fi
  if [ "${nsp_dfg}" = true ]; then
    MODEL_NAME="${MODEL_NAME}_nsp_dfg"
  fi
  if [ "${scope}" = true ]; then
    MODEL_NAME="${MODEL_NAME}_scope"
  fi
  if [ "${address}" = true ]; then
    MODEL_NAME="${MODEL_NAME}_address"
  fi
  MODEL_NAME="${MODEL_NAME#_}"

  OUTPUT_DIR="/work/kliu14/haapr/output/${MODEL_NAME}"
  LOG_DIR="/work/kliu14/haapr/log/${MODEL_NAME}"

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
  python "${REPO_ROOT}/train_from_scratch.py" \
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
    --resume \
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
