#!/bin/bash
#
# Evaluate Function Similarity Model on Test Set
#
# Usage: ./run_eval.sh

set -e

# Paths
FUNCTION_BLOCKS="/data/kun/funcsim_match/function_blocks.json"
FUNCSIM_PAIRS="/data/kun/funcsim_match/funcsim_pairs.json"
VOCAB="../../strupos/vocab.pkl"
CHECKPOINT="../../output/funcsim/best_model.pt"
TEST_INDICES="../../output/funcsim/test_indices.json"

# Output
OUTPUT="../../output/funcsim/test_results.json"
LOG_DIR="../../log/funcsim"

# Model config (must match training)
HIDDEN=768
N_LAYERS=12
ATTN_HEADS=12
EMBEDDING_DIM=256

# Evaluation config
BATCH_SIZE=32
SEQ_LEN=60
NEGATIVE_SAMPLES=3
POOL_SIZE=10000  # Limit retrieval pool size (comment out or set to 0 to use all)

# Device
DEVICE="cuda"
NUM_WORKERS=4

echo "========================================"
echo "Function Similarity Evaluation"
echo "========================================"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Test indices:    ${TEST_INDICES}"
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

# Run evaluation
if [ -n "${POOL_SIZE}" ] && [ "${POOL_SIZE}" -gt 0 ]; then
  POOL_SIZE_ARG="--pool_size ${POOL_SIZE}"
else
  POOL_SIZE_ARG=""
fi

python3 evaluate.py \
  --function_blocks "${FUNCTION_BLOCKS}" \
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
  ${POOL_SIZE_ARG} \
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
