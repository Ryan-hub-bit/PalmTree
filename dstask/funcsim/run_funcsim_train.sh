#!/bin/bash
#
# Fine-tune AddressAwareBERT for Function Similarity
#
# Usage: ./run_funcsim_train.sh

set -e

# Paths
FUNCTION_BLOCKS="/data/kun/funcsim_match/function_blocks.json"
FUNCSIM_PAIRS="/data/kun/funcsim_match/funcsim_pairs.json"
VOCAB="../../strupos/vocab.pkl"
PRETRAINED_BERT="../../output/mlm/best_bert.pt"

# Output
OUTPUT_DIR="../../output/funcsim"
LOG_DIR="../../log/funcsim"
EXPERIMENT_NAME="funcsim_mlm"

# Model config (must match pre-trained BERT)
HIDDEN=768
N_LAYERS=12
ATTN_HEADS=12
EMBEDDING_DIM=256

# Training config
BATCH_SIZE=32
EPOCHS=20
LR=1e-4
SEQ_LEN=512
NEGATIVE_SAMPLES=3
MARGIN=1.0
TRAIN_SPLIT=0.9  # 80% for training, 20% for testing
VAL_SPLIT=0.1    # 10% of training set for validation

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
  --train_split ${TRAIN_SPLIT} \
  --val_split ${VAL_SPLIT} \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --device "${DEVICE}" \
  --num_workers ${NUM_WORKERS}

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo "Model saved to: ${OUTPUT_DIR}/best_model.pt"
echo "Logs saved to:  ${LOG_DIR}/"
echo "========================================"
