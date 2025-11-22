#!/bin/bash

#########################################################################
# Training Address-Aware BERT with Scope Prediction
# - Initializes from pre-trained PalmTree (SAME as baseline)
# - Trains ALL components (no freezing) for fair comparison
# - Address embeddings provide additional positional information
# - NEW: Scope prediction task for instruction relationship classification
#########################################################################

# Data paths
CFG_DATA="../data/ncfg/all_cfg_combined.txt"
DFG_DATA="../data/ndfg/all_dfg_combined.txt"
SCOPE_DATA="../data/scope/all_scope.txt"
VOCAB_FILE="../pre-trained_model/palmtree/vocab"
DATA_PERCENTAGE=1.0  # Use 100% of dataset (0.0-1.0)
TRAIN_SPLIT=0.9      # 90% train, 10% validation

# Pre-trained PalmTree model (for initialization)
PALMTREE_CHECKPOINT="../pre-trained_model/palmtree/transformer.ep19"

# Output
OUTPUT_DIR="output_addressaware_scope_0.3"
LOG_DIR="log"

# Model configuration (must match PalmTree checkpoint)
HIDDEN=128          # PalmTree's hidden size (must match checkpoint!)
N_LAYERS=12         # PalmTree's number of layers
ATTN_HEADS=8        # PalmTree's attention heads
MAX_LEN=20          # 8 instructions * ~10 tokens/instruction

# Training configuration
EPOCHS=20
BATCH_SIZE=1024     # Reduced from 512 due to longer sequences (4x length → 4x memory)
LEARNING_RATE=0.00001
DROPOUT=0.3
MASK_PROB=0.15
NSP_PROB=0.5

# Hardware
USE_CUDA="--cuda"
#USE_MULTI_GPU="--multi_gpu"

# Set environment variables for NCCL (fix multi-GPU errors)
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1  # Disable InfiniBand
# export NCCL_P2P_DISABLE=1  # Disable P2P
# export CUDA_VISIBLE_DEVICES=0,1  # Use GPU 0 and 1

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

if [ ! -f "${CFG_DATA}" ]; then
    echo "Error: CFG data file not found: ${CFG_DATA}"
    exit 1
fi

if [ ! -f "${DFG_DATA}" ]; then
    echo "Error: DFG data file not found: ${DFG_DATA}"
    exit 1
fi

if [ ! -f "${SCOPE_DATA}" ]; then
    echo "Error: Scope data file not found: ${SCOPE_DATA}"
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
echo "Training Address-Aware BERT with Scope Prediction"
echo "========================================================================"
echo "Purpose: Train with address embeddings + scope prediction task"
echo ""
echo "Data:"
echo "  CFG: ${CFG_DATA}"
echo "  DFG: ${DFG_DATA}"
echo "  SCOPE: ${SCOPE_DATA}"
echo "  Vocabulary: ${VOCAB_FILE}"
echo "  Data Percentage: ${DATA_PERCENTAGE} ($(echo "$DATA_PERCENTAGE * 100" | bc)%)"
echo "  Train/Val Split: ${TRAIN_SPLIT} ($(echo "$TRAIN_SPLIT * 100" | bc)% train, $(echo "(1-$TRAIN_SPLIT) * 100" | bc)% val)"
echo ""
echo "Pre-trained Model:"
echo "  PalmTree Checkpoint: ${PALMTREE_CHECKPOINT}"
echo "  >>> Used for INITIALIZATION (same as baseline) <<<"
echo ""
echo "Model Architecture:"
echo "  Hidden: ${HIDDEN}, Layers: ${N_LAYERS}, Heads: ${ATTN_HEADS}"
echo "  Max Length: ${MAX_LEN}"
echo "  Position Embeddings: ADDRESS-AWARE (3-level: binary, function, bb)"
echo "  Task Heads: MLM + NSP(CFG) + NSP(DFG) + SCOPE(3-class)"
echo ""
echo "Training:"
echo "  Epochs: ${EPOCHS}, Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Mask Prob: ${MASK_PROB}, NSP Prob: ${NSP_PROB}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"
echo "========================================================================"
echo ""

# Train
python3 train.py \
    --cfg_train "${CFG_DATA}" \
    --dfg_train "${DFG_DATA}" \
    --scope_train "${SCOPE_DATA}" \
    --vocab "${VOCAB_FILE}" \
    --data_percentage ${DATA_PERCENTAGE} \
    --train_split ${TRAIN_SPLIT} \
    --palmtree_checkpoint "${PALMTREE_CHECKPOINT}" \
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
    --log_dir "${LOG_DIR}" \
    --num_workers 4 \
    --log_freq 50 \
    ${USE_CUDA} \
    ${USE_MULTI_GPU}

echo ""
echo "========================================================================"
echo "Training complete! Checkpoints saved to: ${OUTPUT_DIR}"
echo "========================================================================"
