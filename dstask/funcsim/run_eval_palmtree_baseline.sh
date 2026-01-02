#!/bin/bash
#
# Evaluate PalmTree BASELINE Model on Function Similarity (10k pool)
#
# This tests the baseline PalmTree model (without address features)
# on function similarity retrieval task
#

set -e

# Paths
POOL_IDS="/data/kun/funcsim_match/pool_ids_10k.json"
POOL_FUNCTION_BLOCKS="/data/kun/funcsim_match/pool_function_blocks_10k.json"
FUNCSIM_PAIRS="/data/kun/funcsim_match/funcsim_pairs.json"
VOCAB="../../extern/PalmTree/src/vocab_base"
CHECKPOINT="../../extern/PalmTree/src/output/baseline/best_model.pt"

# Output
OUTPUT="../../output/funcsim/palmtree_baseline_results_10k.json"
LOG_DIR="../../log/funcsim"
TASK_NAME="palmtree_baseline"  # Task identifier for caching

# Model config (must match PalmTree baseline training)
HIDDEN=128
N_LAYERS=12
ATTN_HEADS=8

# Evaluation config
SEQ_LEN=60
BATCH_SIZE=32
NUM_WORKERS=4

# Device
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda"

echo "========================================"
echo "PalmTree BASELINE Evaluation (10k pool)"
echo "========================================"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Pool IDs:        ${POOL_IDS}"
echo "Pool Blocks:     ${POOL_FUNCTION_BLOCKS}"
echo "Output:          ${OUTPUT}"
echo "========================================"
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

python3 evaluate_palmtree.py \
  --function_blocks "${POOL_FUNCTION_BLOCKS}" \
  --funcsim_pairs "${FUNCSIM_PAIRS}" \
  --vocab "${VOCAB}" \
  --checkpoint "${CHECKPOINT}" \
  --hidden ${HIDDEN} \
  --n_layers ${N_LAYERS} \
  --attn_heads ${ATTN_HEADS} \
  --seq_len ${SEQ_LEN} \
  --batch_size ${BATCH_SIZE} \
  --eval_pool "${POOL_IDS}" \
  --output "${OUTPUT}" \
  --log_dir "${LOG_DIR}" \
  --task_name "${TASK_NAME}" \
  --device "${DEVICE}" \
  --num_workers ${NUM_WORKERS} \
  --model_type baseline

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
echo "Results saved to: ${OUTPUT}"
echo "========================================"
