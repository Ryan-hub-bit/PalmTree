#!/bin/bash
#
# Evaluate Pre-trained BERT (without fine-tuning) on Function Similarity
#
# Usage: ./run_eval_pretrained.sh

set -e

# Paths
POOL_IDS="/data/kun/funcsim_match/pool_ids_10k.json"
POOL_FUNCTION_BLOCKS="/data/kun/funcsim_match/pool_function_blocks_10k.json"
FUNCSIM_PAIRS="/data/kun/funcsim_match/funcsim_pairs.json"
VOCAB="../../strupos/vocab.pkl"
CHECKPOINT="../../output/mlm_addr_var/best_bert.pt"  # Use pretrained BERT checkpoint

# Output
OUTPUT="../../output/funcsim/pretrained_results_addr.json"
LOG_DIR="../../log/funcsim"
TASK_NAME="mlm"  # Task identifier for caching

# Model config (must match pretrained BERT)
HIDDEN=768
N_LAYERS=12
ATTN_HEADS=12

# Evaluation config
SEQ_LEN=60
# POOL_SIZE=10000  # Deprecated: Use EVAL_POOL instead for fair comparison

# Device
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda"

echo "======================================="
echo "Pretrained BERT Evaluation (No Fine-tuning)"
echo "======================================="
echo "Checkpoint:      ${CHECKPOINT}"
echo "Pool IDs:        ${POOL_IDS}"
echo "Pool Blocks:     ${POOL_FUNCTION_BLOCKS}"
echo "Output:          ${OUTPUT}"
echo "======================================="
echo ""

# Check if files exist
if [ ! -f "$CHECKPOINT" ]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT"
  exit 1
fi

if [ ! -f "$POOL_IDS" ]; then
  echo "[ERROR] Pool IDs not found: $POOL_IDS"
  echo "Please run: ./generate_pool_data.sh"
  exit 1
fi

if [ ! -f "$POOL_FUNCTION_BLOCKS" ]; then
  echo "[ERROR] Pool function blocks not found: $POOL_FUNCTION_BLOCKS"
  echo "Please run: ./generate_pool_data.sh"
  exit 1
fi

python3 evaluate_pretrained.py \
  --function_blocks "${POOL_FUNCTION_BLOCKS}" \
  --funcsim_pairs "${FUNCSIM_PAIRS}" \
  --vocab "${VOCAB}" \
  --checkpoint "${CHECKPOINT}" \
  --hidden ${HIDDEN} \
  --n_layers ${N_LAYERS} \
  --attn_heads ${ATTN_HEADS} \
  --seq_len ${SEQ_LEN} \
  --eval_pool "${POOL_IDS}" \
  --output "${OUTPUT}" \
  --log_dir "${LOG_DIR}" \
  --task_name "${TASK_NAME}" \
  --device "${DEVICE}"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
echo "Results saved to: ${OUTPUT}"
echo "========================================"
