#!/bin/bash
#
# Generate pool data files (IDs + function blocks)
#
# Usage: ./generate_pool_data.sh

set -e

# Input
FUNCTION_BLOCKS="/data/kun/funcsim_match/function_blocks.json"
FUNCSIM_PAIRS="/data/kun/funcsim_match/funcsim_pairs.json"

# Output directory (same as funcsim_pairs)
OUTPUT_DIR="/data/kun/funcsim_match"

# Pool configuration
POOL_SIZE=10000
SEED=42

echo "========================================"
echo "Generating Pool Data Files"
echo "========================================"
echo "Pool size:       ${POOL_SIZE}"
echo "Random seed:     ${SEED}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "========================================"
echo ""

python3 generate_pool_data.py \
  --function_blocks "${FUNCTION_BLOCKS}" \
  --funcsim_pairs "${FUNCSIM_PAIRS}" \
  --pool_size ${POOL_SIZE} \
  --output_dir "${OUTPUT_DIR}" \
  --seed ${SEED}

echo ""
echo "========================================"
echo "Done! Files created:"
echo "  ${OUTPUT_DIR}/pool_ids_10k.json"
echo "  ${OUTPUT_DIR}/pool_function_blocks_10k.json"
echo "========================================"
