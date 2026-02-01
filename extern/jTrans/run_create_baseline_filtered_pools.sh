#!/bin/bash
# Create filtered evaluation pools
# Guarantees:
# 1. Query and Pool instructions differ (non-memorization test)
# 2. Each instruction in pool is unique (deduplicated)
# 3. Each query has only one unique GT in pool (unambiguous)

python3 create_filtered_baseline_pools.py \
    --func-blocks /data/kun/jtrans/baseline/eval/func_blocks_baseline.json \
    --ground-truth /data/kun/jtrans/baseline/eval/ground_truth_baseline.json \
    --output-dir /data/kun/jtrans/baseline/eval/pools_filtered \
    --min-instructions 11

echo ""
echo "✅ Creation complete!"
echo ""
echo "📊 Next evaluation strategy:"
echo "  1. Filtered pools → Real semantic understanding (no memorization)"
echo "  2. Focus on O0→O3 → Hardest task (maximum optimization difference)"
echo "  3. Pool deduplicated → Accurate MRR/Recall@K calculation"
echo ""
echo "🎯 Finetune vs Evaluation difference:"
echo "  - Finetune: Can have duplicate functions (helps learning)"
echo "  - Evaluation: Must be completely different (tests generalization)"
