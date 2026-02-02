#!/bin/bash
# Fine-tune address-aware model on function similarity task (strupos-style)
# Uses self-contained BERT implementation without HuggingFace dependencies

set -e

# Set GPUs
export CUDA_VISIBLE_DEVICES=0,1 

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Paths
FUNC_BLOCKS="/data/kun/jtrans/addressaware/func_blocks_addr.json"
GROUND_TRUTH="/data/kun/jtrans/addressaware/ground_truth_addr.json"
VOCAB_PATH="./pretrain/address_aware/vocab.txt"
MODEL_PATH="./output/jtrans/addressaware_pretrain_sincos/checkpoint_epoch_13"
OUTPUT_PATH="./output/jtrans/addressaware_finetune_v2"

# Model config (must match pretrain)
HIDDEN=768
N_LAYERS=12
ATTN_HEADS=12
EMBEDDING_DIM=256
MAX_LEN=512

# Training config
BATCH_SIZE=16
EPOCHS=15
LR=2e-5
WEIGHT_DECAY=0.01
TRIPLET_MARGIN=0.2
DATA_RATIO=1.0
TRAIN_SPLIT=0.8
VAL_SPLIT=0.125

# Freezing config
FREEZE_CNT=10  # Freeze first 10 layers, or -1 for no freezing

echo "========================================"
echo "jTrans Function Similarity Fine-tuning"
echo "========================================"
echo "Model:   ${MODEL_PATH}"
echo "Output:  ${OUTPUT_PATH}"
echo "========================================"
echo ""

# Check if files exist
if [ ! -d "$MODEL_PATH" ]; then
  echo "[ERROR] Pre-trained model not found: $MODEL_PATH"
  echo "[INFO] Please train the model first"
  exit 1
fi

if [ ! -f "$FUNC_BLOCKS" ]; then
  echo "[ERROR] Function blocks not found: $FUNC_BLOCKS"
  exit 1
fi

if [ ! -f "$GROUND_TRUTH" ]; then
  echo "[ERROR] Ground truth not found: $GROUND_TRUTH"
  exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_PATH}"

# Run training
python finetune_v2.py \
  --func_blocks "${FUNC_BLOCKS}" \
  --ground_truth "${GROUND_TRUTH}" \
  --vocab_path "${VOCAB_PATH}" \
  --model_path "${MODEL_PATH}/pytorch_model.bin" \
  --output_path "${OUTPUT_PATH}" \
  --experiment_name "jtrans_funcsim_v2" \
  --hidden ${HIDDEN} \
  --n_layers ${N_LAYERS} \
  --attn_heads ${ATTN_HEADS} \
  --embedding_dim ${EMBEDDING_DIM} \
  --max_len ${MAX_LEN} \
  --batch_size ${BATCH_SIZE} \
  --epoch ${EPOCHS} \
  --lr ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --triplet_margin ${TRIPLET_MARGIN} \
  --data_ratio ${DATA_RATIO} \
  --train_split ${TRAIN_SPLIT} \
  --val_split ${VAL_SPLIT} \
  --freeze_cnt ${FREEZE_CNT} \
  --device cuda \
  --num_workers 4

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo "Model saved to: ${OUTPUT_PATH}/best_model.pt"
echo "========================================"
