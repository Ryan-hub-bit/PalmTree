#!/bin/bash

# Run consecutive pairs training
# This script will:
# 1. Check if vocab.txt exists
# 2. If not, create it from train/val/test data
# 3. Train the model

set -e # Exit on error

# ==================== GPU Selection ====================
# Specify which GPU(s) to use (comma-separated for multiple GPUs)
# Examples:
#   GPU_ID="0"        # Use GPU 0
#   GPU_ID="1"        # Use GPU 1
#   GPU_ID="0,1"      # Use GPU 0 and 1
#   GPU_ID="2,3"      # Use GPU 2 and 3
GPU_ID="1"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo "Using GPU(s): ${GPU_ID}"
echo ""

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

# if [ -f "$VOCAB_FILE" ]; then
#   echo -e "${GREEN}✓ Vocabulary file found: $VOCAB_FILE${NC}"
#   # Get vocab size from pickle file using Python (with error handling)
#   VOCAB_SIZE=$(python3 -c "
# import sys
# sys.path.insert(0, '.')
# try:
#     from vocab import WordVocab
#     vocab = WordVocab.load_vocab('$VOCAB_FILE')
#     print(len(vocab))
# except Exception as e:
#     print('0')
#     sys.exit(1)
# " 2>/dev/null)
  
#   if [ "$VOCAB_SIZE" = "0" ] || [ -z "$VOCAB_SIZE" ]; then
#     echo -e "${YELLOW}⚠ Vocabulary file exists but cannot be loaded (old format)${NC}"
#     echo "  Deleting old vocabulary and creating new one..."
#     rm -f "$VOCAB_FILE"
#     python3 create_vocab.py
#     if [ $? -eq 0 ]; then
#       echo -e "${GREEN}✓ Vocabulary created successfully${NC}"
#       VOCAB_SIZE=$(python3 -c "import sys; sys.path.insert(0, '.'); from vocab import WordVocab; vocab = WordVocab.load_vocab('$VOCAB_FILE'); print(len(vocab))")
#       echo "  Vocabulary size: $VOCAB_SIZE tokens"
#     else
#       echo -e "${RED}✗ Failed to create vocabulary${NC}"
#       exit 1
#     fi
#   else
#     echo "  Vocabulary size: $VOCAB_SIZE tokens"
#   fi
# else
#   echo -e "${YELLOW}⚠ Vocabulary file not found: $VOCAB_FILE${NC}"
#   echo "  Creating vocabulary from train/val/test data..."
#   echo ""

#   # Check if data files exist
#   DATA_FILES=(
#     "/data/kun/dataset/train_cfg.txt"
#     "/data/kun/dataset/train_dfg.txt"
#     "/data/kun/dataset/val_cfg.txt"
#     "/data/kun/dataset/val_dfg.txt"
#     "/data/kun/dataset/test_cfg.txt"
#     "/data/kun/dataset/test_dfg.txt"
#   )

#   MISSING_FILES=0
#   for FILE in "${DATA_FILES[@]}"; do
#     if [ ! -f "$FILE" ]; then
#       echo -e "${RED}  ✗ Missing: $FILE${NC}"
#       MISSING_FILES=$((MISSING_FILES + 1))
#     else
#       echo -e "${GREEN}  ✓ Found: $FILE${NC}"
#     fi
#   done

#   if [ $MISSING_FILES -gt 0 ]; then
#     echo -e "${RED}Error: Some data files are missing. Cannot create vocabulary.${NC}"
#     exit 1
#   fi

#   echo ""
#   echo "Creating vocabulary..."
#   python3 create_vocab.py

#   if [ $? -eq 0 ]; then
#     echo -e "${GREEN}✓ Vocabulary created successfully${NC}"
#     # Get vocab size from pickle file using Python with local import
#     VOCAB_SIZE=$(python3 -c "import sys; sys.path.insert(0, '.'); from vocab import WordVocab; vocab = WordVocab.load_vocab('$VOCAB_FILE'); print(len(vocab))")
#     echo "  Vocabulary size: $VOCAB_SIZE tokens"
#   else
#     echo -e "${RED}✗ Failed to create vocabulary${NC}"
#     exit 1
#   fi
# fi

echo ""
echo "========================================"
echo "Starting Training"
echo "========================================"
echo ""

# Training configuration (command-line arguments)
# Training configuration (command-line arguments)
CFG_TRAIN="/data/kun/dataset/train_cfg.txt"
DFG_TRAIN="/data/kun/dataset/train_dfg.txt"
CFG_VAL="/data/kun/dataset/val_cfg.txt"
DFG_VAL="/data/kun/dataset/val_dfg.txt"
CFG_TEST="/data/kun/dataset/test_cfg.txt"
DFG_TEST="/data/kun/dataset/test_dfg.txt"
SCOPE_TRAIN="/data/kun/dataset/train_scope.txt"
SCOPE_VAL="/data/kun/dataset/val_scope.txt"
SCOPE_TEST="/data/kun/dataset/test_scope.txt"
VOCAB_PATH="./vocab.pkl"

# ==================== Task Selection (Ablation Study) ====================
# Enable/disable each pretraining task
ENABLE_MLM=true             # Masked Language Modeling (CFG only)
ENABLE_NSP_CFG=true        # Next Sentence Prediction for CFG
ENABLE_NSP_DFG=false        # Next Sentence Prediction for DFG
ENABLE_SCOPE=false          # Scope Prediction (2-class)
USE_ADDRESS_EMBEDDING=false # Use 3-level address-aware embeddings
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

# Remove leading underscore and set default if empty
MODEL_NAME="${MODEL_NAME#_}"
if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME="baseline"
fi

OUTPUT_DIR="../output/${MODEL_NAME}_test"
LOG_DIR="../log/${MODEL_NAME}_test"

# Model architecture
HIDDEN=768
LAYERS=12
ATTN_HEADS=12
SEQ_LEN=60
NSP_CONTENT_MAX=20 # Maximum content length for CFG/DFG NSP pairs
DROPOUT=0.1

# Training hyperparameters
EPOCHS=10
BATCH_SIZE=256 # Further reduced to avoid OOM (was 256, original 1024)
LR=1e-4
WARMUP_STEPS=10000
NUM_WORKERS=4
EARLY_STOPPING_PATIENCE=3

# Data processing
MASK_PROB=0.15
NSP_PROB=0.5
TRAIN_PERCENTAGE=1 # Training data percentage (0.01 = 1%)
VAL_PERCENTAGE=1   # Validation data percentage (0.01 = 1%)

# Device
CUDA="--cuda"
MULTI_GPU=" "

# Show configuration
echo "Configuration:"
echo "  Model name: ${MODEL_NAME}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Tasks enabled:"
echo "    - MLM: ${ENABLE_MLM}"
echo "    - NSP-CFG: ${ENABLE_NSP_CFG}"
echo "    - NSP-DFG: ${ENABLE_NSP_DFG}"
echo "    - SCOPE: ${ENABLE_SCOPE}"
echo "    - Address Embedding: ${USE_ADDRESS_EMBEDDING}"
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

# Run training with command-line arguments
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
