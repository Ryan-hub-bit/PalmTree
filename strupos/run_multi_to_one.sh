#!/bin/bash

# Run consecutive pairs training
# This script will:
# 1. Check if vocab.txt exists
# 2. If not, create it from train/val/test data
# 3. Train the model

set -e # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "STRUPOS Training - Consecutive Pairs"
echo "========================================"
echo ""

# Check if vocab.txt exists
VOCAB_FILE="./vocab.pkl"

if [ -f "$VOCAB_FILE" ]; then
  echo -e "${GREEN}✓ Vocabulary file found: $VOCAB_FILE${NC}"
  # Get vocab size from pickle file using Python (with error handling)
  VOCAB_SIZE=$(python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from vocab import WordVocab
    vocab = WordVocab.load_vocab('$VOCAB_FILE')
    print(len(vocab))
except Exception as e:
    print('0')
    sys.exit(1)
" 2>/dev/null)
  
  if [ "$VOCAB_SIZE" = "0" ] || [ -z "$VOCAB_SIZE" ]; then
    echo -e "${YELLOW}⚠ Vocabulary file exists but cannot be loaded (old format)${NC}"
    echo "  Deleting old vocabulary and creating new one..."
    rm -f "$VOCAB_FILE"
    python3 create_vocab.py
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}✓ Vocabulary created successfully${NC}"
      VOCAB_SIZE=$(python3 -c "import sys; sys.path.insert(0, '.'); from vocab import WordVocab; vocab = WordVocab.load_vocab('$VOCAB_FILE'); print(len(vocab))")
      echo "  Vocabulary size: $VOCAB_SIZE tokens"
    else
      echo -e "${RED}✗ Failed to create vocabulary${NC}"
      exit 1
    fi
  else
    echo "  Vocabulary size: $VOCAB_SIZE tokens"
  fi
else
  echo -e "${YELLOW}⚠ Vocabulary file not found: $VOCAB_FILE${NC}"
  echo "  Creating vocabulary from train/val/test data..."
  echo ""

  # Check if data files exist
  DATA_FILES=(
    "/work/kliu14/txtdataset/train_cfg.txt"
    "/work/kliu14/txtdataset/train_dfg.txt"
    "/work/kliu14/txtdataset/val_cfg.txt"
    "/work/kliu14/txtdataset/val_dfg.txt"
    "/work/kliu14/txtdataset/test_cfg.txt"
    "/work/kliu14/txtdataset/test_dfg.txt"
  )

  MISSING_FILES=0
  for FILE in "${DATA_FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
      echo -e "${RED}  ✗ Missing: $FILE${NC}"
      MISSING_FILES=$((MISSING_FILES + 1))
    else
      echo -e "${GREEN}  ✓ Found: $FILE${NC}"
    fi
  done

  if [ $MISSING_FILES -gt 0 ]; then
    echo -e "${RED}Error: Some data files are missing. Cannot create vocabulary.${NC}"
    exit 1
  fi

  echo ""
  echo "Creating vocabulary..."
  python3 create_vocab.py

  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Vocabulary created successfully${NC}"
    # Get vocab size from pickle file using Python with local import
    VOCAB_SIZE=$(python3 -c "import sys; sys.path.insert(0, '.'); from vocab import WordVocab; vocab = WordVocab.load_vocab('$VOCAB_FILE'); print(len(vocab))")
    echo "  Vocabulary size: $VOCAB_SIZE tokens"
  else
    echo -e "${RED}✗ Failed to create vocabulary${NC}"
    exit 1
  fi
fi

echo ""
echo "========================================"
echo "Starting Training (Multi-to-One NSP)"
echo "========================================"
echo ""

# Training configuration (command-line arguments)
# Training configuration (command-line arguments)
CFG_TRAIN="/work/kliu14/txtdataset/train_cfg.txt"
DFG_TRAIN="/work/kliu14/txtdataset/train_dfg.txt"
CFG_VAL="/work/kliu14/txtdataset/val_cfg.txt"
DFG_VAL="/work/kliu14/txtdataset/val_dfg.txt"
CFG_TEST="/work/kliu14/txtdataset/test_cfg.txt"
DFG_TEST="/work/kliu14/txtdataset/test_dfg.txt"
# SCOPE_TRAIN="./scope_train.txt"
# SCOPE_VAL="./scope_val.txt"
# SCOPE_TEST="./scope_test.txt"
VOCAB_PATH="./vocab.pkl"

# ==================== Task Selection (Ablation Study) ====================
# Enable/disable each pretraining task
ENABLE_MLM=true             # Masked Language Modeling (CFG only)
ENABLE_NSP_CFG=true        # Next Sentence Prediction for CFG
ENABLE_NSP_DFG=false        # Next Sentence Prediction for DFG
ENABLE_SCOPE=false          # Scope Prediction (3-class)
USE_ADDRESS_EMBEDDING=false # Use 3-level address-aware embeddings
INSTRUCTION_LEVEL_SEGMENT=false # Use instruction-level segment IDs (each instruction gets unique segment)

# Optional mode argument to quickly pick common configs:
#   MLM_NSP_CFG                -> MLM + NSP-CFG
#   MLM_NSP_CFG_ADDRESS        -> MLM + NSP-CFG + Address
#   MLM_NSP_CFG_ADDRESS_INSTR  -> MLM + NSP-CFG + Address + Instruction-level segments
# If no argument is provided, the script uses the boolean flags defined above.
MODE="${1:-}"
if [ -n "$MODE" ]; then
  echo "Selected mode: $MODE"
  case "$MODE" in
    MLM_NSP_CFG)
      ENABLE_MLM=true
      ENABLE_NSP_CFG=true
      ENABLE_NSP_DFG=false
      ENABLE_SCOPE=false
      USE_ADDRESS_EMBEDDING=false
      INSTRUCTION_LEVEL_SEGMENT=false
      ;;
    MLM_NSP_CFG_ADDRESS)
      ENABLE_MLM=true
      ENABLE_NSP_CFG=true
      ENABLE_NSP_DFG=false
      ENABLE_SCOPE=false
      USE_ADDRESS_EMBEDDING=true
      INSTRUCTION_LEVEL_SEGMENT=false
      ;;
    MLM_NSP_CFG_ADDRESS_INSTR)
      ENABLE_MLM=true
      ENABLE_NSP_CFG=true
      ENABLE_NSP_DFG=false
      ENABLE_SCOPE=false
      USE_ADDRESS_EMBEDDING=true
      INSTRUCTION_LEVEL_SEGMENT=true
      ;;
    *)
      echo "Warning: unknown mode '$MODE' — continuing with default flags" >&2
      ;;
  esac
fi

# Build task flags for command line
TASK_FLAGS=""
if [ "$ENABLE_MLM" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --enable_mlm"
else
  TASK_FLAGS="${TASK_FLAGS} --disable_mlm"
fi
if [ "$ENABLE_NSP_CFG" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --enable_nsp_cfg"
fi
if [ "$ENABLE_NSP_DFG" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --enable_nsp_dfg"
fi
if [ "$ENABLE_SCOPE" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --enable_scope"
fi
if [ "$USE_ADDRESS_EMBEDDING" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --use_address_embedding"
fi
if [ "$INSTRUCTION_LEVEL_SEGMENT" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --instruction_level_segment"
fi

# Auto-generate model name based on enabled tasks
MODEL_NAME=""
if [ "$ENABLE_MLM" = true ]; then
  MODEL_NAME="${MODEL_NAME}_mlm"
fi
if [ "$ENABLE_NSP_CFG" = true ]; then
  MODEL_NAME="${MODEL_NAME}_nsp_cfg"
fi
if [ "$ENABLE_NSP_DFG" = true ]; then
  MODEL_NAME="${MODEL_NAME}_nsp_dfg"
fi
if [ "$ENABLE_SCOPE" = true ]; then
  MODEL_NAME="${MODEL_NAME}_scope"
fi
if [ "$USE_ADDRESS_EMBEDDING" = true ]; then
  MODEL_NAME="${MODEL_NAME}_address"
fi
if [ "$INSTRUCTION_LEVEL_SEGMENT" = true ]; then
  MODEL_NAME="${MODEL_NAME}_ins"
fi

# Remove leading underscore and set default if empty
MODEL_NAME="${MODEL_NAME#_}"
if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME="baseline"
fi

OUTPUT_DIR="../output/${MODEL_NAME}_multi_to_one"
LOG_DIR="../log/${MODEL_NAME}_multi_to_one"

# Model architecture
HIDDEN=768
LAYERS=12
ATTN_HEADS=12
SEQ_LEN=60
NSP_CONTENT_MAX=20 # Maximum content length for CFG/DFG NSP pairs
DROPOUT=0.1

# Training hyperparameters
EPOCHS=10
BATCH_SIZE=1024 # Further reduced to avoid OOM (was 256, original 1024)
LR=1e-4
WARMUP_STEPS=10000
NUM_WORKERS=4
EARLY_STOPPING_PATIENCE=3

# Data processing
MASK_PROB=0.15
NSP_PROB=0.5
TRAIN_PERCENTAGE=0.001 # Training data percentage (0.01 = 1%)
VAL_PERCENTAGE=0.001   # Validation data percentage (0.01 = 1%)

# Device
CUDA="--cuda"
MULTI_GPU=""

# Show configuration
echo "Configuration:"
echo "  Model name: ${MODEL_NAME}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Tasks enabled:"
echo "    - MLM: ${ENABLE_MLM}"
echo "    - NSP-CFG: ${ENABLE_NSP_CFG} (multi-to-one)"
echo "    - NSP-DFG: ${ENABLE_NSP_DFG}"
echo "    - SCOPE: ${ENABLE_SCOPE}"
echo "    - Address Embedding: ${USE_ADDRESS_EMBEDDING}"
echo "    - Instruction-Level Segments: ${INSTRUCTION_LEVEL_SEGMENT}"
echo "  Model architecture:"
echo "    - Seq length: ${SEQ_LEN}"
echo "    - NSP pair max: ${NSP_CONTENT_MAX}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LR}"
echo "  Epochs: ${EPOCHS}"
echo "  Vocab: ${VOCAB_PATH}"
echo "  Train: ${CFG_TRAIN} / ${DFG_TRAIN}"
echo "  Val: ${CFG_VAL} / ${DFG_VAL}"
echo "  Test: ${CFG_TEST} / ${DFG_TEST}"
if [ "$ENABLE_SCOPE" = true ]; then
  echo "  Scope: ${SCOPE_TRAIN} / ${SCOPE_VAL} / ${SCOPE_TEST}"
fi
echo ""
echo "Starting training script..."
echo ""

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES=1
# Run training with command-line arguments (Multi-to-One NSP)
python train_multi_to_one.py \
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
  2>&1 | tee train_strupos.log

if [ $? -eq 0 ]; then
  echo ""
  echo -e "${GREEN}========================================"
  echo "Training Completed Successfully!"
  echo "========================================${NC}"
  echo ""
  echo "Model saved to: ${OUTPUT_DIR}"
  if [ -f "${OUTPUT_DIR}/best_model.pt" ]; then
    echo "  - best_model.pt"
  fi
  if [ -f "${OUTPUT_DIR}/best_bert.pt" ]; then
    echo "  - best_bert.pt"
  fi
  if [ -f "${OUTPUT_DIR}/args.json" ]; then
    echo "  - args.json"
  fi
  echo "Logs saved to: ${LOG_DIR}"
else
  echo ""
  echo -e "${RED}========================================"
  echo "Training Failed!"
  echo "========================================${NC}"
  exit 1
fi
