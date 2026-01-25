#!/bin/bash
# Generate fair pool and query JSON files

# This script creates pool and query definitions that match entries
# across baseline and address-aware datasets by (binary, func_name, opt)

set -e

# Configuration
FUNC_BLOCKS_BASELINE="/data/kun/jtransdata/func_blocks_baseline.json"
FUNC_BLOCKS_ADDRESSAWARE="/data/kun/jtransdata/func_blocks_addr.json"
GROUND_TRUTH_BASELINE="/data/kun/jtransdata/ground_truth_baseline.json"
GROUND_TRUTH_ADDRESSAWARE="/data/kun/jtransdata/ground_truth_addr.json"

OUTPUT_DIR="/data/kun/jtransdata/fair_pools"
POOL_SIZES="100 1000 10000"
SEED=42

echo "Generating fair pool and query JSON files..."
echo "Output directory: $OUTPUT_DIR"

python generate_fair_pools.py \
    --func_blocks_baseline "$FUNC_BLOCKS_BASELINE" \
    --func_blocks_addressaware "$FUNC_BLOCKS_ADDRESSAWARE" \
    --ground_truth_baseline "$GROUND_TRUTH_BASELINE" \
    --ground_truth_addressaware "$GROUND_TRUTH_ADDRESSAWARE" \
    --output_dir "$OUTPUT_DIR" \
    --pool_sizes $POOL_SIZES \
    --seed $SEED

echo ""
echo "Done! Generated files in: $OUTPUT_DIR"
echo ""
echo "Files created (separate pools for each opt pair):"
echo "  - pool_100_O0_vs_O1.json (contains O0 queries + O1 ground truth)"
echo "  - pool_100_O0_vs_O2.json (contains O0 queries + O2 ground truth)"
echo "  - pool_100_O0_vs_O3.json (contains O0 queries + O3 ground truth)"
echo "  - query_pool_100_O0_vs_O1.json"
echo "  - query_pool_100_O0_vs_O2.json"
echo "  - query_pool_100_O0_vs_O3.json"
echo "  - pool_1000_O0_vs_O1.json, pool_1000_O0_vs_O2.json, pool_1000_O0_vs_O3.json"
echo "  - query_pool_1000_*.json"
echo "  - pool_10000_O0_vs_O1.json, pool_10000_O0_vs_O2.json, pool_10000_O0_vs_O3.json"
echo "  - query_pool_10000_*.json"
echo "  - pool_statistics.json"
