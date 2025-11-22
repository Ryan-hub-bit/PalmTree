#!/bin/bash

#########################################################################
# Generate JSON files with all bucket labels for each instruction
#
# Creates one JSON file per binary containing:
# - All instructions with their text
# - Hierarchical positions (binary, function, BB)
# - All three bucket labels
# - Bucket distribution statistics
#########################################################################

CFG_DIR="../data/cfg_2"
OUTPUT_DIR="data/json_labels"
BB_BUCKETS=5
FUNC_BUCKETS=5
BINARY_BUCKETS=5

echo "========================================================================"
echo "Generating JSON Label Files for All Instructions"
echo "========================================================================"
echo "Input: ${CFG_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Buckets: BB=${BB_BUCKETS}, Func=${FUNC_BUCKETS}, Binary=${BINARY_BUCKETS}"
echo "========================================================================"
echo ""

python3 generate_json_labels.py \
    --cfg_dir "${CFG_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --bb_buckets ${BB_BUCKETS} \
    --func_buckets ${FUNC_BUCKETS} \
    --binary_buckets ${BINARY_BUCKETS} \
    --pattern "*_cfg_*_inline.txt"

echo ""
echo "========================================================================"
echo "Done! JSON files generated at: ${OUTPUT_DIR}"
echo "========================================================================"
