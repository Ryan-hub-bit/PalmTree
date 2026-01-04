#!/bin/bash
#
# Evaluate Function Similarity Model on Test Set
#
# Usage: ./run_eval.sh

set -e

# Paths
POOL_IDS="/data/kun/funcsim_match/pool_ids_100k.json"
POOL_FUNCTION_BLOCKS="/data/kun/funcsim_match/pool_function_blocks_100k.json"
FUNCSIM_PAIRS="/data/kun/funcsim_match/funcsim_pairs.json"
VOCAB="/home/kun/Document/AAE/extern/PalmTree/src/vocab_addr"
CHECKPOINT="../../output/funcsim/mlm_addr_var/best_model.pt"
TEST_INDICES="../../output/funcsim/mlm_addr_var/test_indices.json"

# Output
OUTPUT="../../output/funcsim/mlm_addr_var/test_results_100k.json"
LOG_DIR="../../log/funcsim/mlm_addr_var"

# Model config (must match training)
HIDDEN=768
N_LAYERS=12
ATTN_HEADS=12
EMBEDDING_DIM=256
TASK_NAME="mlm_addr_var"  # Task identifier for embedding cache (should match training)

# Evaluation config
BATCH_SIZE=32
SEQ_LEN=60
NEGATIVE_SAMPLES=3
# POOL_SIZE=100  # Deprecated: Use EVAL_POOL instead for fair comparison
DATA_FRACTION=1.0 # Use only 20% of test data (set to 1.0 to use all)

# Device
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda"
NUM_WORKERS=4

echo "========================================"
echo "Function Similarity Evaluation"
echo "========================================"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Test indices:    ${TEST_INDICES}"
echo "Pool IDs:        ${POOL_IDS}"
echo "Pool Blocks:     ${POOL_FUNCTION_BLOCKS}"
echo "Output:          ${OUTPUT}"
echo "========================================"
echo ""

# Check if files exist
if [ ! -f "$CHECKPOINT" ]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT"
  echo "[INFO] Please train the model first: ./run_funcsim_train.sh"
  exit 1
fi

if [ ! -f "$TEST_INDICES" ]; then
  echo "[ERROR] Test indices not found: $TEST_INDICES"
  echo "[INFO] Test indices are created during training"
  exit 1
fi

if [ ! -f "$POOL_IDS" ]; then
  echo "[ERROR] Pool IDs not found: $POOL_IDS"
  echo "[INFO] Please run: ./generate_pool_data.sh"
  exit 1
fi

if [ ! -f "$POOL_FUNCTION_BLOCKS" ]; then
  echo "[ERROR] Pool function blocks not found: $POOL_FUNCTION_BLOCKS"
  echo "[INFO] Please run: ./generate_pool_data.sh"
  exit 1
fi

# Run evaluation
if [ -n "${DATA_FRACTION}" ] && (($(echo "${DATA_FRACTION} < 1.0" | bc -l))); then
  DATA_FRACTION_ARG="--data_fraction ${DATA_FRACTION}"
else
  DATA_FRACTION_ARG=""
fi

python3 evaluate.py \
  --function_blocks "${POOL_FUNCTION_BLOCKS}" \
  --funcsim_pairs "${FUNCSIM_PAIRS}" \
  --vocab "${VOCAB}" \
  --test_indices "${TEST_INDICES}" \
  --checkpoint "${CHECKPOINT}" \
  --hidden ${HIDDEN} \
  --n_layers ${N_LAYERS} \
  --attn_heads ${ATTN_HEADS} \
  --embedding_dim ${EMBEDDING_DIM} \
  --batch_size ${BATCH_SIZE} \
  --seq_len ${SEQ_LEN} \
  --negative_samples ${NEGATIVE_SAMPLES} \
  --eval_pool "${POOL_IDS}" \
  ${DATA_FRACTION_ARG} \
  --task_name "${TASK_NAME}" \
  --output "${OUTPUT}" \
  --log_dir "${LOG_DIR}" \
  --device "${DEVICE}" \
  --num_workers ${NUM_WORKERS}

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
echo "Results saved to: ${OUTPUT}"
echo "========================================"
