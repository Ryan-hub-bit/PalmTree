#!/bin/bash
# Create address-aware evaluation pools

set -e

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Configuration
FUNC_BLOCKS="/data/kun/jtrans/addressaware/eval/func_blocks_addr.json"
GROUND_TRUTH="/data/kun/jtrans/addressaware/eval/ground_truth_addr.json"
OUTPUT_DIR="/data/kun/jtrans/addressaware/eval/pools_filtered"

echo "=================================="
echo "Creating Address-Aware Eval Pools"
echo "=================================="
echo ""
echo "Input:"
echo "  func_blocks: $FUNC_BLOCKS"
echo "  ground_truth: $GROUND_TRUTH"
echo ""
echo "Output:"
echo "  pool_dir: $OUTPUT_DIR"
echo ""

python create_filtered_addressaware_pools.py \
  --func_blocks "$FUNC_BLOCKS" \
  --ground_truth "$GROUND_TRUTH" \
  --output_dir "$OUTPUT_DIR" \
  --pool_sizes 100 1000 10000 \
  --query_ratio 0.2 \
  --min_instructions 11

echo ""
echo "=================================="
echo "Pool Creation Completed!"
echo "=================================="
echo ""
echo "Generated pools:"
ls -lh "$OUTPUT_DIR"/pool_*.json
