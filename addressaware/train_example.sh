#!/bin/bash
#
# Example training script for Address-Aware PalmTree
#

# Configuration
CFG_DATA="../data/cfg/all_cfg_combined.txt"
DFG_DATA="../data/dfg/all_dfg_combined.txt"
VOCAB_FILE="./vocab.pkl"
OUTPUT_DIR="./output"

# Model configuration
HIDDEN=768
LAYERS=12
ATTN_HEADS=12
SEQ_LEN=512

# Training configuration
EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=0.0001
DROPOUT=0.1
MASK_PROB=0.15
NSP_PROB=0.5

# Hardware
USE_CUDA="--cuda"
# USE_MULTI_GPU="--multi_gpu"  # Uncomment for multi-GPU

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Check if data files exist
if [ ! -f "${CFG_DATA}" ]; then
    echo "Error: CFG data file not found: ${CFG_DATA}"
    echo "Please run generate_data.sh first"
    exit 1
fi

if [ ! -f "${DFG_DATA}" ]; then
    echo "Error: DFG data file not found: ${DFG_DATA}"
    echo "Please run generate_data.sh first"
    exit 1
fi

# Build vocabulary if it doesn't exist
if [ ! -f "${VOCAB_FILE}" ]; then
    echo "Building vocabulary..."
    python3 << EOF
import sys
sys.path.insert(0, '../src')
from palmtree.dataset.vocab import WordVocab

print("Loading corpus...")
with open("${CFG_DATA}", "r") as f:
    vocab = WordVocab([f], max_size=50000, min_freq=3)

print(f"Vocabulary size: {len(vocab)}")
vocab.save_vocab("${VOCAB_FILE}")
print(f"Vocabulary saved to ${VOCAB_FILE}")
EOF
fi

echo "=================================================="
echo "Starting Address-Aware PalmTree Training"
echo "=================================================="
echo "Configuration:"
echo "  CFG Data: ${CFG_DATA}"
echo "  DFG Data: ${DFG_DATA}"
echo "  Vocab: ${VOCAB_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo ""
echo "Model:"
echo "  Hidden: ${HIDDEN}"
echo "  Layers: ${LAYERS}"
echo "  Heads: ${ATTN_HEADS}"
echo "  Seq Len: ${SEQ_LEN}"
echo ""
echo "Training:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "=================================================="

# Run training
python3 train.py \
    --cfg_train ${CFG_DATA} \
    --dfg_train ${DFG_DATA} \
    --vocab ${VOCAB_FILE} \
    --hidden ${HIDDEN} \
    --layers ${LAYERS} \
    --attn_heads ${ATTN_HEADS} \
    --seq_len ${SEQ_LEN} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --dropout ${DROPOUT} \
    --mask_prob ${MASK_PROB} \
    --nsp_prob ${NSP_PROB} \
    --output_dir ${OUTPUT_DIR} \
    --log_freq 100 \
    --save_freq 1 \
    --num_workers 4 \
    ${USE_CUDA} \
    ${USE_MULTI_GPU}

echo ""
echo "=================================================="
echo "Training completed!"
echo "Models saved to: ${OUTPUT_DIR}"
echo "=================================================="
