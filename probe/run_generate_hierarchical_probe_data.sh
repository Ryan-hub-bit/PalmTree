#!/bin/bash

#########################################################################
# Generate Hierarchical Probe Data
# 
# Uses cfg_hierarchical_icfg.py output format with address-based positions
# Creates balanced datasets for BB, function, and binary bucket probes
#########################################################################

CFG_DIR="../data/cfg_2"
OUTPUT_DIR="data/hierarchical"
NUM_BINARIES=5
BB_BUCKETS=5
FUNC_BUCKETS=5
BINARY_BUCKETS=5
SAMPLES_PER_BUCKET=1000

echo "========================================================================"
echo "Generating Hierarchical Probe Data"
echo "========================================================================"
echo "Input: ${CFG_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Binaries: ${NUM_BINARIES}"
echo "Buckets: BB=${BB_BUCKETS}, Func=${FUNC_BUCKETS}, Binary=${BINARY_BUCKETS}"
echo "Samples per bucket: ${SAMPLES_PER_BUCKET}"
echo "========================================================================"
echo ""

python3 generate_hierarchical_probe_data.py \
    --cfg_dir "${CFG_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_binaries ${NUM_BINARIES} \
    --bb_buckets ${BB_BUCKETS} \
    --func_buckets ${FUNC_BUCKETS} \
    --binary_buckets ${BINARY_BUCKETS} \
    --samples_per_bucket ${SAMPLES_PER_BUCKET} \
    --pattern "*_cfg_*_inline.txt"

echo ""
echo "========================================================================"
echo "Done! Probe data generated at: ${OUTPUT_DIR}"
echo "========================================================================"
