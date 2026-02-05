#!/bin/bash
# Regenerate address-aware evaluation pools with ground truth included
# This ensures each pool has queries with ground truth properly embedded

set -e

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Configuration
FUNC_BLOCKS="/data/kun/jtrans/addressaware/eval/func_blocks_addr.json"
GROUND_TRUTH="/data/kun/jtrans/addressaware/eval/ground_truth_addr.json"
OUTPUT_DIR="/data/kun/jtrans/addressaware/eval/pools_filtered"

# Check if input files exist
if [ ! -f "$FUNC_BLOCKS" ]; then
    echo "Error: func_blocks not found at $FUNC_BLOCKS"
    exit 1
fi

if [ ! -f "$GROUND_TRUTH" ]; then
    echo "Error: ground_truth not found at $GROUND_TRUTH"
    exit 1
fi

echo "========================================"
echo "Regenerating Address-Aware Eval Pools"
echo "========================================"
echo ""
echo "Input files:"
echo "  func_blocks: $FUNC_BLOCKS"
echo "  ground_truth: $GROUND_TRUTH"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Pool sizes: 100, 1000, 10000"
echo "Optimization pairs: O0_vs_O3, O1_vs_O3, O2_vs_O3"
echo "Query ratio: 20%"
echo "Min instructions: 11 (filter trivial functions)"
echo ""
echo "========================================"
echo ""

# Backup old pools if they exist
if [ -d "$OUTPUT_DIR" ]; then
    BACKUP_DIR="${OUTPUT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing pools to: $BACKUP_DIR"
    cp -r "$OUTPUT_DIR" "$BACKUP_DIR"
    echo ""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run pool generation
python create_filtered_addressaware_pools.py \
    --func_blocks "$FUNC_BLOCKS" \
    --ground_truth "$GROUND_TRUTH" \
    --output_dir "$OUTPUT_DIR" \
    --pool_sizes 100 1000 10000 \
    --query_ratio 0.2 \
    --min_instructions 11

echo ""
echo "========================================"
echo "Pool Generation Complete!"
echo "========================================"
echo ""
echo "Generated pools:"
ls -lh "$OUTPUT_DIR"/pool_*.json

echo ""
echo "Verifying pool format..."
echo ""

# Verify each pool has queries and ground truth
for pool_file in "$OUTPUT_DIR"/pool_*.json; do
    if [ -f "$pool_file" ]; then
        filename=$(basename "$pool_file")
        echo "Checking $filename:"
        
        # Extract key information using Python
        python3 -c "
import json
with open('$pool_file', 'r') as f:
    data = json.load(f)
    
print(f\"  Pool size: {data.get('actual_size', 'N/A')}\")
print(f\"  Query count: {data.get('query_count', 'N/A')}\")
print(f\"  Opt pair: {data.get('opt_pair', 'N/A')}\")

if 'queries' in data and data['queries']:
    q = data['queries'][0]
    has_gt = 'gt_id' in q
    print(f\"  Has ground truth: {'✓' if has_gt else '✗'}\")
    if has_gt:
        print(f\"  Sample query: query_id={q.get('query_id', 'N/A')}, gt_id={q.get('gt_id', 'N/A')}\")
else:
    print(f\"  Has ground truth: ✗ (no queries)\")
"
        echo ""
    fi
done

echo "========================================"
echo "All pools regenerated successfully!"
echo "========================================"
