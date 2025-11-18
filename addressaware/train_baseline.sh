#!/bin/bash

#########################################################################
# Training Baseline BERT (NO address embeddings)
# - Same data as address-aware model
# - Same architecture but uses sequential positions
# - Fair comparison to measure impact of address embeddings
#########################################################################

# Data paths (SAME as address-aware model)
CFG_DATA="../data/cfg/all_cfg_combined.txt"
DFG_DATA="../data/dfg/all_dfg_combined.txt"
VOCAB_FILE="../pre-trained_model/palmtree/vocab"
DATA_PERCENTAGE=0.5  # Use same percentage as address-aware (0.0-1.0)
TRAIN_SPLIT=0.9      # 90% train, 10% validation (0.0-1.0)

# Output
OUTPUT_DIR="output_baseline"

# Model configuration (SAME as address-aware model)
HIDDEN=128           # Must match address-aware model
N_LAYERS=12          # Must match address-aware model
ATTN_HEADS=8         # Must match address-aware model
MAX_LEN=20           # Must match address-aware model

# Training configuration (SAME as address-aware model)
EPOCHS=20
BATCH_SIZE=512       # Same batch size for fair comparison
LEARNING_RATE=0.001  # Same learning rate
DROPOUT=0.1
MASK_PROB=0.15
NSP_PROB=0.5

# Hardware
USE_CUDA="--cuda"
USE_MULTI_GPU="--multi_gpu"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Validate inputs
if [ ! -f "${CFG_DATA}" ]; then
    echo "Error: CFG data file not found: ${CFG_DATA}"
    exit 1
fi

if [ ! -f "${DFG_DATA}" ]; then
    echo "Error: DFG data file not found: ${DFG_DATA}"
    exit 1
fi

if [ ! -f "${VOCAB_FILE}" ]; then
    echo "Error: Vocabulary file not found: ${VOCAB_FILE}"
    exit 1
fi

echo "========================================================================"
echo "Training Baseline BERT (NO address embeddings)"
echo "========================================================================"
echo "Purpose: Fair comparison with address-aware model"
echo ""
echo "Data:"
echo "  CFG: ${CFG_DATA}"
echo "  DFG: ${DFG_DATA}"
echo "  Vocabulary: ${VOCAB_FILE}"
echo "  Data Percentage: ${DATA_PERCENTAGE} ($(echo "$DATA_PERCENTAGE * 100" | bc)%)"
echo "  Train/Val Split: ${TRAIN_SPLIT} ($(echo "$TRAIN_SPLIT * 100" | bc)% train)"
echo ""
echo "Model Architecture (SAME as address-aware):"
echo "  Hidden: ${HIDDEN}, Layers: ${N_LAYERS}, Heads: ${ATTN_HEADS}"
echo "  Max Length: ${MAX_LEN}"
echo "  Position Embeddings: SEQUENTIAL (no address info)"
echo ""
echo "Training (SAME as address-aware):"
echo "  Epochs: ${EPOCHS}, Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Mask Prob: ${MASK_PROB}, NSP Prob: ${NSP_PROB}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "========================================================================"
echo ""

# Train
python3 train_baseline.py \
    --cfg_train "${CFG_DATA}" \
    --dfg_train "${DFG_DATA}" \
    --vocab "${VOCAB_FILE}" \
    --data_percentage ${DATA_PERCENTAGE} \
    --train_split ${TRAIN_SPLIT} \
    --hidden ${HIDDEN} \
    --layers ${N_LAYERS} \
    --attn_heads ${ATTN_HEADS} \
    --seq_len ${MAX_LEN} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --dropout ${DROPOUT} \
    --mask_prob ${MASK_PROB} \
    --nsp_prob ${NSP_PROB} \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 4 \
    --log_freq 50 \
    ${USE_CUDA} \
    ${USE_MULTI_GPU}

echo ""
echo "========================================================================"
echo "Training complete! Checkpoints saved to: ${OUTPUT_DIR}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Compare with address-aware model using test_mlm_nsp.py"
echo "  2. Check if address embeddings improve MLM/NSP performance"
echo "========================================================================"
