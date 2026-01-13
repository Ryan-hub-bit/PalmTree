#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# Generate comparisons for all evaluation results
DATA_DIR="/data/kun/jtransdata"
OUTPUT_DIR="${DATA_DIR}/fair_pool_results"

echo "Generating comparison reports..."

for POOL_SIZE in 100 1000 10000; do
    for OPT_PAIR in "O0_vs_O3" "O1_vs_O3" "O2_vs_O3"; do
        BASELINE_RESULT="${OUTPUT_DIR}/baseline_${POOL_SIZE}_${OPT_PAIR}.txt"
        ADDRESSAWARE_RESULT="${OUTPUT_DIR}/addressaware_${POOL_SIZE}_${OPT_PAIR}.txt"
        COMPARISON_OUTPUT="${OUTPUT_DIR}/comparison_${POOL_SIZE}_${OPT_PAIR}.txt"
        
        if [ -f "${BASELINE_RESULT}" ] && [ -f "${ADDRESSAWARE_RESULT}" ]; then
            echo "Comparing: Pool Size=${POOL_SIZE}, Opt Pair=${OPT_PAIR}"
            
            python compare_results.py \
                --baseline_result "${BASELINE_RESULT}" \
                --addressaware_result "${ADDRESSAWARE_RESULT}" \
                --output_file "${COMPARISON_OUTPUT}"
        else
            echo "Skipping ${POOL_SIZE}_${OPT_PAIR}: Missing results"
        fi
    done
done

echo ""
echo "=================================================="
echo "All comparisons complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=================================================="
