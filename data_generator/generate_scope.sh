#!/bin/bash

# Generate scope prediction training data from CFG files (binary-by-binary)
# This processes binaries from train_cdfg, val_cdfg, test_cdfg directories
# Note: These are CFG files only, not combined CFG+DFG

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="/data/kun/dataset"

# CDFG directory paths
TRAIN_CDFG="${DATA_DIR}/train_cdfg"
VAL_CDFG="${DATA_DIR}/val_cdfg"
TEST_CDFG="${DATA_DIR}/test_cdfg"

# Output paths
TRAIN_OUTPUT="${DATA_DIR}/train_scope.txt"
VAL_OUTPUT="${DATA_DIR}/val_scope.txt"
TEST_OUTPUT="${DATA_DIR}/test_scope.txt"

# Samples per type PER BINARY
# Total samples per binary = SAMPLES_PER_BIN (balanced across 3 labels)
# Each label gets SAMPLES_PER_BIN / 3, with easy/medium/hard difficulty levels
TRAIN_SAMPLES_PER_BIN=150   # 300 total per binary (100 per label)
VAL_SAMPLES_PER_BIN=150     # 150 total per binary (50 per label)
TEST_SAMPLES_PER_BIN=150    # 150 total per binary (50 per label)

echo "========================================================================"
echo "SCOPE DATA GENERATION - Binary-by-Binary Mode"
echo "========================================================================"
echo ""
echo "Data directories:"
echo "  Train CDFG: ${TRAIN_CDFG}"
echo "  Val CDFG:   ${VAL_CDFG}"
echo "  Test CDFG:  ${TEST_CDFG}"
echo ""
echo "Sample configuration (per binary):"
echo "  Train: ${TRAIN_SAMPLES_PER_BIN} total per binary (balanced: ~$((TRAIN_SAMPLES_PER_BIN/3)) per label)"
echo "  Val:   ${VAL_SAMPLES_PER_BIN} total per binary (balanced: ~$((VAL_SAMPLES_PER_BIN/3)) per label)"
echo "  Test:  ${TEST_SAMPLES_PER_BIN} total per binary (balanced: ~$((TEST_SAMPLES_PER_BIN/3)) per label)"
echo ""
echo "Features:"
echo "  ✓ Balanced labels (equal samples of same BB, cross BB, cross func)"
echo "  ✓ Difficulty levels (easy/medium/hard for each type)"
echo "  ✓ All pairs within same binary (CFG only)"
echo "  ✓ Exactly 8 instructions per line (sliding window format)"
echo ""
echo "Total samples = samples_per_binary * num_binaries"
echo ""
echo "========================================================================"
echo ""

# Function to process all CDFG files in a directory
process_cdfg_dir() {
    local input_dir=$1
    local output_file=$2
    local samples_per_bin=$3
    local split_name=$4
    
    echo "### Processing ${split_name} ###"
    echo "Input directory: ${input_dir}"
    echo "Output file: ${output_file}"
    echo ""
    
    # Create temporary directory for intermediate files
    local temp_dir="${DATA_DIR}/scope"
    mkdir -p "${temp_dir}"
    
    # Find all CFG files only (skip DFG files)
    local cdfg_files=($(find "${input_dir}" -name "*_cfg_*.txt" -type f))
    local num_files=${#cdfg_files[@]}
    
    echo "Found ${num_files} CFG files"
    echo ""
    
    # Process each CDFG file
    local count=0
    for cdfg_file in "${cdfg_files[@]}"; do
        count=$((count + 1))
        local basename=$(basename "${cdfg_file}" .txt)
        local temp_output="${temp_dir}/${basename}_scope.txt"
        
        echo "[${count}/${num_files}] Processing ${basename}..."
        
        python generate_scope_data.py \
            --cdfg "${cdfg_file}" \
            --output "${temp_output}" \
            --samples ${samples_per_bin}
        
        echo ""
    done
    
    # Combine all temporary files
    echo "Combining all scope files..."
    cat "${temp_dir}"/*_scope.txt > "${output_file}"
    
    # Count total pairs
    local total_pairs=$(wc -l < "${output_file}")
    echo "Total pairs generated: ${total_pairs}"
    
    # Clean up
    rm -rf "${temp_dir}"
    echo "Cleaned up temporary files"
    echo ""
}

# Generate training data
if [ -f "${TRAIN_OUTPUT}" ]; then
    echo "⚠️  ${TRAIN_OUTPUT} already exists, skipping..."
    echo ""
else
    process_cdfg_dir "${TRAIN_CDFG}" "${TRAIN_OUTPUT}" ${TRAIN_SAMPLES_PER_BIN} "TRAIN"
fi

# # Generate validation data
if [ -f "${VAL_OUTPUT}" ]; then
    echo "⚠️  ${VAL_OUTPUT} already exists, skipping..."
    echo ""
else
    process_cdfg_dir "${VAL_CDFG}" "${VAL_OUTPUT}" ${VAL_SAMPLES_PER_BIN} "VAL"
fi

# Generate test data
if [ -f "${TEST_OUTPUT}" ]; then
    echo "⚠️  ${TEST_OUTPUT} already exists, skipping..."
    echo ""
else
    process_cdfg_dir "${TEST_CDFG}" "${TEST_OUTPUT}" ${TEST_SAMPLES_PER_BIN} "TEST"
fi

echo "========================================================================"
echo "ALL DONE!"
echo "========================================================================"
echo ""
echo "Generated files:"
if [ -f "${TRAIN_OUTPUT}" ]; then
    echo "  Train: ${TRAIN_OUTPUT} ($(wc -l < ${TRAIN_OUTPUT}) pairs)"
fi
if [ -f "${VAL_OUTPUT}" ]; then
    echo "  Val:   ${VAL_OUTPUT} ($(wc -l < ${VAL_OUTPUT}) pairs)"
fi
if [ -f "${TEST_OUTPUT}" ]; then
    echo "  Test:  ${TEST_OUTPUT} ($(wc -l < ${TEST_OUTPUT}) pairs)"
fi
echo ""
echo "All pairs are generated binary-by-binary to ensure meaningful scope relationships!"
echo "========================================================================"
