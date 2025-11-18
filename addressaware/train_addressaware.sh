#!/bin/bash

#########################################################################
# Training Address-Aware BERT
# - Loads pre-trained PalmTree (FROZEN)
# - Only trains NEW address-aware positional embeddings
#########################################################################

# Data paths
CFG_DATA="../data/cfg/all_cfg_combined.txt"
DFG_DATA="../data/dfg/all_dfg_combined.txt"
VOCAB_FILE="../pre-trained_model/palmtree/vocab"
DATA_PERCENTAGE=0.5  # Use 10% of dataset (0.0-1.0), set 1.0 for full dataset
TRAIN_SPLIT=0.9      # 90% train, 10% validation (0.0-1.0), set 1.0 for no validation

# Pre-trained PalmTree model (will be FROZEN)
PALMTREE_CHECKPOINT="../pre-trained_model/palmtree/transformer.ep19"

# Output
OUTPUT_DIR="output_addressaware"

# Model configuration (must match PalmTree checkpoint)
HIDDEN=128          # PalmTree's hidden size (must match checkpoint!)
N_LAYERS=12          # PalmTree's number of layers
ATTN_HEADS=8        # PalmTree's attention heads
MAX_LEN=20

# Training configuration
EPOCHS=20
BATCH_SIZE=512       # Increased to 512 (GPU 1 has ~24GB free)
LEARNING_RATE=0.001  # Higher LR since only training new components
DROPOUT=0.1
MASK_PROB=0.15
NSP_PROB=0.5

# Hardware
USE_CUDA="--cuda"
USE_MULTI_GPU="--multi_gpu"  # Enabled - GPU 1 has plenty of memory

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Check if data files exist
if [ ! -f "${CFG_DATA}" ]; then
    echo "Error: CFG data file not found: ${CFG_DATA}"
    exit 1
fi

if [ ! -f "${DFG_DATA}" ]; then
    echo "Error: DFG data file not found: ${DFG_DATA}"
    exit 1
fi

if [ ! -f "${PALMTREE_CHECKPOINT}" ]; then
    echo "Error: PalmTree checkpoint not found: ${PALMTREE_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${VOCAB_FILE}" ]; then
    echo "Error: Vocabulary file not found: ${VOCAB_FILE}"
    exit 1
fi

echo "========================================================================"
echo "Training Address-Aware BERT"
echo "========================================================================"
echo "Data:"
echo "  CFG: ${CFG_DATA}"
echo "  DFG: ${DFG_DATA}"
echo "  Vocabulary: ${VOCAB_FILE}"
echo "  Data Percentage: ${DATA_PERCENTAGE} ($(echo "$DATA_PERCENTAGE * 100" | bc)%)"
echo "  Train/Val Split: ${TRAIN_SPLIT} ($(echo "$TRAIN_SPLIT * 100" | bc)% train, $(echo "(1-$TRAIN_SPLIT) * 100" | bc)% val)"
echo ""
echo "Pre-trained Model:"
echo "  PalmTree Checkpoint: ${PALMTREE_CHECKPOINT}"
echo "  >>> ALL PalmTree components will be FROZEN <<<"
echo ""
echo "Model Architecture:"
echo "  Hidden: ${HIDDEN}, Layers: ${N_LAYERS}, Heads: ${ATTN_HEADS}"
echo "  Max Length: ${MAX_LEN}"
echo ""
echo "Training:"
echo "  Epochs: ${EPOCHS}, Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Mask Prob: ${MASK_PROB}, NSP Prob: ${NSP_PROB}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "========================================================================"
echo ""

# Train
python3 train.py \
    --cfg_train "${CFG_DATA}" \
    --dfg_train "${DFG_DATA}" \
    --vocab "${VOCAB_FILE}" \
    --data_percentage ${DATA_PERCENTAGE} \
    --train_split ${TRAIN_SPLIT} \
    --palmtree_checkpoint "${PALMTREE_CHECKPOINT}" \
    --freeze_token_emb \
    --freeze_position_emb \
    --freeze_segment_emb \
    --freeze_transformer \
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
