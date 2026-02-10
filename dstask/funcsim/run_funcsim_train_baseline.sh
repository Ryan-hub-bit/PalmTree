#!/bin/bash
#
# Fine-tune AddressAwareBERT for Function Similarity
#
# Usage: ./run_funcsim_train.sh

set -e
export CUDA_VISIBLE_DEVICES=1

# Path
#
FUNCTION_BLOCKS="/data/kun/funcsim_match/function_blocks.json"
FUNCSIM_PAIRS="/data/kun/funcsim_match/funcsim_pairs.json"
VOCAB="/home/kun/Document/AAE/extern/PalmTree/src/vocab_base"

# Pre-trained BERT model
# The script will auto-detect if this model has address/var embeddings:
# - If YES (e.g., mlm_address/best_bert.pt): Use address/var info from data
# - If NO  (e.g., mlm/best_bert.pt):         Set all positions/offsets to 0
#PRETRAINED_BERT="../../output/mlm/best_bert.pt"
PRETRAINED_BERT="/home/kun/Document/AAE/extern/PalmTree/src/output/baseline/best_model_bert.pt"

# Output
OUTPUT_DIR="../../output/palmtreefuncsim/PalmtreeBaseline"
LOG_DIR="../../log/palmtreefuncsim/PalmtreeBaseline"
EXPERIMENT_NAME="funcsim_PalmtreeBaseline"
TASK_NAME="PalmtreeBaseline"  # Task identifier for embedding cache (e.g., 'mlm', 'mlm_addr_var')
# Model config (must match pre-trained BERT)
HIDDEN=128
N_LAYERS=12
ATTN_HEADS=8
EMBEDDING_DIM=128

# Training config
BATCH_SIZE=256
EPOCHS=5
LR=1e-4
SEQ_LEN=20
NEGATIVE_SAMPLES=3
MARGIN=1.0

# Data usage configuration
# DATASET_FRACTION: Fraction of entire dataset to use (for faster experiments)
# Set to 1.0 to use full dataset, 0.2 for 20%, 0.1 for 10%, etc.
DATASET_FRACTION=0.2  # Use only 20% of entire dataset

# Data split configuration (applied to the fraction above)
# TRAIN_SPLIT: Percentage for train+val (rest goes to test)
# VAL_SPLIT: Percentage of train+val that goes to validation
#
# With DATASET_FRACTION=0.2, TRAIN_SPLIT=0.8, VAL_SPLIT=0.125:
# → Use 20% of data, then split: 14% train, 2% val, 4% test (of total dataset)
#
TRAIN_SPLIT=0.8  # 80% of selected data for training+validation, 20% for testing
VAL_SPLIT=0.125  # 12.5% of train+val for validation
# Result within 20% selected: 70% train, 10% val, 20% test

# Device
DEVICE="cuda"
NUM_WORKERS=4

echo "========================================"
echo "Function Similarity Fine-tuning"
echo "========================================"
echo "Pre-trained BERT: ${PRETRAINED_BERT}"
echo "Function blocks:  ${FUNCTION_BLOCKS}"
echo "Funcsim pairs:    ${FUNCSIM_PAIRS}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "========================================"
echo ""

# Check if data files exist
if [ ! -f "$FUNCTION_BLOCKS" ]; then
    echo "[ERROR] Function blocks file not found: $FUNCTION_BLOCKS"
    echo "[INFO] Please run: cd ../../src/data_generator && ./generate_pairs.sh"
    exit 1
fi

if [ ! -f "$FUNCSIM_PAIRS" ]; then
    echo "[ERROR] Funcsim pairs file not found: $FUNCSIM_PAIRS"
    echo "[INFO] Please run: cd ../../src/data_generator && ./generate_pairs.sh"
    exit 1
fi

if [ ! -f "$PRETRAINED_BERT" ]; then
    echo "[ERROR] Pre-trained BERT not found: $PRETRAINED_BERT"
    echo "[INFO] Please train the base model first"
    exit 1
fi

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Run training
python3 train.py \
  --function_blocks "${FUNCTION_BLOCKS}" \
  --funcsim_pairs "${FUNCSIM_PAIRS}" \
  --vocab "${VOCAB}" \
  --pretrained_bert "${PRETRAINED_BERT}" \
  --hidden ${HIDDEN} \
  --n_layers ${N_LAYERS} \
  --attn_heads ${ATTN_HEADS} \
  --embedding_dim ${EMBEDDING_DIM} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --seq_len ${SEQ_LEN} \
  --negative_samples ${NEGATIVE_SAMPLES} \
  --margin ${MARGIN} \
  --dataset_fraction ${DATASET_FRACTION} \
  --train_split ${TRAIN_SPLIT} \
  --val_split ${VAL_SPLIT} \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --task_name "${TASK_NAME}" \
  --device "${DEVICE}" \
  --num_workers ${NUM_WORKERS}

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo "Model saved to: ${OUTPUT_DIR}/best_model.pt"
echo "Logs saved to:  ${LOG_DIR}/"
echo "========================================"
