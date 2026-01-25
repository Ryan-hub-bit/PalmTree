#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Configuration
DATA_DIR="/data/kun/jtransdata"

# Address-aware model configuration
ADDRESSAWARE_MODEL="/home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_1"
ADDRESSAWARE_TOKENIZER="/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware"
FUNC_BLOCKS_ADDRESSAWARE="${DATA_DIR}/func_blocks_addr.json"

POOL_DIR="${DATA_DIR}/fair_pools"
OUTPUT_DIR="${DATA_DIR}/fair_pool_addr_results"

mkdir -p "${OUTPUT_DIR}"

# Evaluate each pool configuration
for POOL_SIZE in 100 1000 10000; do
  for OPT_PAIR in "O0_vs_O3" "O1_vs_O3" "O2_vs_O3"; do
    echo "=================================================="
    echo "Evaluating: Pool Size=${POOL_SIZE}, Opt Pair=${OPT_PAIR}"
    echo "=================================================="

    POOL_FILE="${POOL_DIR}/pool_${POOL_SIZE}_${OPT_PAIR}.json"
    QUERY_FILE="${POOL_DIR}/query_pool_${POOL_SIZE}_${OPT_PAIR}.json"

    if [ ! -f "${POOL_FILE}" ]; then
      echo "Pool file not found: ${POOL_FILE}"
      continue
    fi

    if [ ! -f "${QUERY_FILE}" ]; then
      echo "Query file not found: ${QUERY_FILE}"
      continue
    fi

    # Evaluate address-aware model
    echo ""
    echo "Evaluating ADDRESS-AWARE model..."
    ADDRESSAWARE_OUTPUT="${OUTPUT_DIR}/addressaware_${POOL_SIZE}_${OPT_PAIR}.txt"

    python evaluate_addressaware_with_pools.py \
      --model_path "${ADDRESSAWARE_MODEL}" \
      --tokenizer "${ADDRESSAWARE_TOKENIZER}" \
      --func_blocks "${FUNC_BLOCKS_ADDRESSAWARE}" \
      --pool_file "${POOL_FILE}" \
      --query_file "${QUERY_FILE}" \
      --max_length 512 \
      --output_file "${ADDRESSAWARE_OUTPUT}"
  done
done

echo "=================================================="
echo "All evaluations complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=================================================="
