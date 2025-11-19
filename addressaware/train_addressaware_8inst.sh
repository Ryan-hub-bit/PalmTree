#!/bin/bash

#########################################################################
# Training Address-Aware BERT (8-Instruction Window with 7:1 Split)
# - Window: 8 instructions per line
# - CFG Split: 7:1 (C1:C2), Positive=forward, Negative=reversed
# - DFG Split: 7:1 (D1:D2), Positive=same line, Negative=random
# - Initializes from pre-trained PalmTree
# - Trains ALL components (no freezing) for fair comparison
#########################################################################

# Data paths (8-instruction format)
CFG_DATA="../data/cfg/all_cfg_combined.txt"
DFG_DATA="../data/dfg/all_dfg_combined.txt"
VOCAB_FILE="../pre-trained_model/palmtree/vocab"
DATA_PERCENTAGE=1.0  # Use 100% of dataset (0.0-1.0)
TRAIN_SPLIT=0.9      # 90% train, 10% validation

# Pre-trained PalmTree model (for initialization)
PALMTREE_CHECKPOINT="../pre-trained_model/palmtree/transformer.ep19"

# Output
OUTPUT_DIR="output_addressaware_8inst"

# Model configuration (must match PalmTree checkpoint)
HIDDEN=128          # PalmTree's hidden size
N_LAYERS=12         # PalmTree's number of layers
ATTN_HEADS=8        # PalmTree's attention heads
MAX_LEN=80          # ~8 instructions * 10 tokens/instruction

# Training configuration
EPOCHS=20
BATCH_SIZE=128      # Reduced due to longer sequences
LEARNING_RATE=0.001
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
    echo "Please create combined file first:"
    echo "  cd ../data/cfg && cat *_cfg_8_inline.txt > all_cfg_8inst_combined.txt"
    exit 1
fi

if [ ! -f "${DFG_DATA}" ]; then
    echo "Error: DFG data file not found: ${DFG_DATA}"
    echo "Please create combined file first:"
    echo "  cd ../data/dfg && cat *_dfg_8_inline.txt > all_dfg_8inst_combined.txt"
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
echo "Training Address-Aware BERT (8-Instruction Window, 7:1 Split)"
echo "========================================================================"
echo ""
echo "NSP Strategy:"
echo "  CFG: 7:1 split, POSITIVE=C1→C2, NEGATIVE=C2→C1 (reversed)"
echo "  DFG: 7:1 split, POSITIVE=D1→D2, NEGATIVE=D1→D_random"
echo ""
echo "Data:"
echo "  CFG: ${CFG_DATA}"
echo "  DFG: ${DFG_DATA}"
echo "  Vocabulary: ${VOCAB_FILE}"
echo "  Data Percentage: ${DATA_PERCENTAGE} (100%)"
echo "  Train/Val Split: ${TRAIN_SPLIT} (90% train, 10% val)"
echo ""
echo "Pre-trained Model:"
echo "  PalmTree Checkpoint: ${PALMTREE_CHECKPOINT}"
echo ""
echo "Model Architecture:"
echo "  Hidden: ${HIDDEN}, Layers: ${N_LAYERS}, Heads: ${ATTN_HEADS}"
echo "  Max Length: ${MAX_LEN}"
echo "  Position Embeddings: ADDRESS-AWARE (3-level: binary, function, bb)"
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
python3 train_8inst.py \
    --cfg_train "${CFG_DATA}" \
    --dfg_train "${DFG_DATA}" \
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
    --num_workers 4 \
    --log_freq 50 \
    ${USE_CUDA} \
    ${USE_MULTI_GPU}

echo ""
echo "========================================================================"
echo "Training complete! Checkpoints saved to: ${OUTPUT_DIR}"
echo "========================================================================"
