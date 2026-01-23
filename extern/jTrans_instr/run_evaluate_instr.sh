#!/bin/bash
# Evaluate jTrans_instr model using fair pools (same pools as baseline/addressaware)

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Set GPU
export CUDA_VISIBLE_DEVICES=1

# Paths
MODEL_PATH="/home/kun/Document/AAE/output/jtrans_instr/finetune/finetune_epoch_5"
TOKENIZER="/home/kun/Document/AAE/extern/jTrans_instr/pretrain"
FUNC_BLOCKS="/data/kun/jtrans_instr/func_blocks_instr.json"

# Fair pool files (SAME as baseline and addressaware for fair comparison)
POOL_DIR="/data/kun/jtransdata/fair_pools"
OUTPUT_DIR="/home/kun/Document/AAE/extern/jTrans_instr/eval_results"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "jTrans_instr Evaluation with Fair Pools"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Pool directory: $POOL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Evaluate on different pool sizes and opt level pairs
for POOL_SIZE in 100 1000 10000; do
    for OPT_PAIR in "O0_vs_O1" "O0_vs_O2" "O0_vs_O3" "O1_vs_O2" "O1_vs_O3" "O2_vs_O3"; do
        POOL_FILE="${POOL_DIR}/pool_${POOL_SIZE}_${OPT_PAIR}.json"
        QUERY_FILE="${POOL_DIR}/query_pool_${POOL_SIZE}_${OPT_PAIR}.json"
        OUTPUT_FILE="${OUTPUT_DIR}/results_${POOL_SIZE}_${OPT_PAIR}.txt"
        
        if [ -f "$POOL_FILE" ] && [ -f "$QUERY_FILE" ]; then
            echo "[INFO] Evaluating pool_${POOL_SIZE}_${OPT_PAIR}..."
            
            python evaluate_instr_with_pools.py \
                --model_path "$MODEL_PATH" \
                --tokenizer "$TOKENIZER" \
                --func_blocks "$FUNC_BLOCKS" \
                --pool_file "$POOL_FILE" \
                --query_file "$QUERY_FILE" \
                --output_file "$OUTPUT_FILE"
            
            echo ""
        else
            echo "[SKIP] Pool files not found for pool_${POOL_SIZE}_${OPT_PAIR}"
        fi
    done
done

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# Summary of all results
echo ""
echo "SUMMARY OF ALL RESULTS:"
echo "=========================================="

for result_file in "$OUTPUT_DIR"/*.txt; do
    if [ -f "$result_file" ]; then
        echo ""
        echo "File: $(basename $result_file)"
        grep -E "MRR|Recall@1" "$result_file" | head -4
    fi
done

echo ""
echo "=========================================="
