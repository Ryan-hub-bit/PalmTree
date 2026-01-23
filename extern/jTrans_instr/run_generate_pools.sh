#!/bin/bash
# Generate fair pools for baseline, addressaware, and jTrans_instr

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Paths
FUNC_BLOCKS_BASELINE="/data/kun/jtransdata/func_blocks_baseline.json"
FUNC_BLOCKS_ADDRESSAWARE="/data/kun/jtransdata/func_blocks_addressaware.json"
FUNC_BLOCKS_INSTR="/data/kun/jtrans_instr/func_blocks_instr.json"

GROUND_TRUTH_BASELINE="/data/kun/jtransdata/ground_truth_baseline.json"
GROUND_TRUTH_ADDRESSAWARE="/data/kun/jtransdata/ground_truth_addressaware.json"
GROUND_TRUTH_INSTR="/data/kun/jtrans_instr/ground_truth_instr.json"

OUTPUT_DIR="/data/kun/jtransdata/fair_pools"

echo "=========================================="
echo "Generating Fair Pools for All Three Models"
echo "=========================================="
echo "Baseline:       $FUNC_BLOCKS_BASELINE"
echo "AddressAware:   $FUNC_BLOCKS_ADDRESSAWARE"
echo "jTrans_instr:   $FUNC_BLOCKS_INSTR"
echo "Output:         $OUTPUT_DIR"
echo "=========================================="

python generate_fair_pools_all.py \
    --func_blocks_baseline "$FUNC_BLOCKS_BASELINE" \
    --func_blocks_addressaware "$FUNC_BLOCKS_ADDRESSAWARE" \
    --func_blocks_instr "$FUNC_BLOCKS_INSTR" \
    --ground_truth_baseline "$GROUND_TRUTH_BASELINE" \
    --ground_truth_addressaware "$GROUND_TRUTH_ADDRESSAWARE" \
    --ground_truth_instr "$GROUND_TRUTH_INSTR" \
    --output_dir "$OUTPUT_DIR" \
    --pool_sizes 100 1000 10000 \
    --min_token_length 30 \
    --seed 42

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo "Pool files saved in: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.json | head -20
echo ""
