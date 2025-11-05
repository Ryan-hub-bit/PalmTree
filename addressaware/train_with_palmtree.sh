#!/bin/bash

#########################################################################
# Training Address-Aware BERT with Pre-trained PalmTree Initialization
#########################################################################

# Data paths
CFG_DATA="../data/cfg_output/combined_output.txt"
DFG_DATA="../data/dfg_output/combined_output.txt"
VOCAB_FILE="vocab.pkl"
OUTPUT_DIR="output_pretrained"

# Pre-trained PalmTree model
PALMTREE_CHECKPOINT="../pre-trained_model/palmtree/transformer.ep19"

# Model configuration (must match PalmTree)
HIDDEN=768          # Must match PalmTree checkpoint
N_LAYERS=12         # Must match PalmTree checkpoint
ATTN_HEADS=12       # Must match PalmTree checkpoint
MAX_LEN=512

# Training configuration
EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=0.0001
DROPOUT=0.1
MASK_PROB=0.15
NSP_PROB=0.5

# Freezing strategy
# When using --palmtree_checkpoint, ALL PalmTree components are automatically FROZEN:
#   - Token embeddings (FROZEN)
#   - Sequence positional encoding (FROZEN)
#   - Segment embeddings (FROZEN)
#   - Transformer blocks (FROZEN)
# ONLY these NEW components are TRAINABLE:
#   - Address positional embeddings (binary, function, basic block)
#   - MLM head
#   - NSP head

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

# Check if PalmTree checkpoint exists
if [ ! -f "${PALMTREE_CHECKPOINT}" ]; then
    echo "Error: PalmTree checkpoint not found: ${PALMTREE_CHECKPOINT}"
    echo "Please provide a valid PalmTree checkpoint"
    exit 1
fi

# Build vocabulary if it doesn't exist
if [ ! -f "${VOCAB_FILE}" ]; then
    echo "Building vocabulary from CFG data..."
    python3 << EOF
import sys
sys.path.insert(0, '../src')
from palmtree.dataset.vocab import WordVocab

print("Reading CFG data...")
with open("${CFG_DATA}", 'r') as f:
    lines = f.readlines()

print(f"Building vocabulary from {len(lines)} lines...")
vocab = WordVocab(lines, max_size=50000, min_freq=2)

print(f"Vocabulary size: {len(vocab)}")
vocab.save_vocab("${VOCAB_FILE}")
print(f"Vocabulary saved to ${VOCAB_FILE}")
EOF
fi

echo "========================================================================"
echo "Training Address-Aware BERT with Pre-trained PalmTree"
echo "========================================================================"
echo "Configuration:"
echo "  CFG Data: ${CFG_DATA}"
echo "  DFG Data: ${DFG_DATA}"
echo "  Vocabulary: ${VOCAB_FILE}"
echo "  PalmTree Checkpoint: ${PALMTREE_CHECKPOINT}"
echo "  Output: ${OUTPUT_DIR}"
echo ""
echo "Model:"
echo "  Hidden: ${HIDDEN}, Layers: ${N_LAYERS}, Heads: ${ATTN_HEADS}"
echo "  Max Length: ${MAX_LEN}"
echo ""
echo "Training:"
echo "  Epochs: ${EPOCHS}, Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Mask Probability: ${MASK_PROB}, NSP Probability: ${NSP_PROB}"
echo ""
echo "Freezing Strategy:"
echo "  ✗ Token Embeddings: FROZEN (from PalmTree)"
echo "  ✗ Sequence Position: FROZEN (from PalmTree)"
echo "  ✗ Segment Embeddings: FROZEN (from PalmTree)"
echo "  ✗ Transformer Blocks: FROZEN (from PalmTree)"
echo "  ✓ Address Position: TRAINABLE (NEW component)"
echo "  ✓ MLM Head: TRAINABLE"
echo "  ✓ NSP Head: TRAINABLE"
echo "========================================================================"

# Train
python3 train.py \
    --cfg_train "${CFG_DATA}" \
    --dfg_train "${DFG_DATA}" \
    --vocab "${VOCAB_FILE}" \
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
    ${USE_CUDA} \
    ${USE_MULTI_GPU}

echo ""
echo "========================================================================"
echo "Training complete! Checkpoints saved to: ${OUTPUT_DIR}"
echo "========================================================================"
