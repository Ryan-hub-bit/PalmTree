#!/bin/bash

# Generate Test Data for Probing Experiment
# This script processes binaries from /home/kun/testbinary/ to create
# test data with inline address format for probing position embeddings.

echo "========================================================================="
echo "Generate Test Data for Probing Experiment"
echo "========================================================================="
echo ""

# Paths
BINARY_DIR="/home/kun/testbinary"
OUTPUT_DIR="/home/kun/Document/PalmTree/probing/test_data"
DATA_GENERATOR_DIR="/home/kun/Document/PalmTree/src/data_generator"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Configuration:"
echo "  Binary Directory: ${BINARY_DIR}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo ""

# List available binaries
echo "Available binaries in ${BINARY_DIR}:"
ls -1 "${BINARY_DIR}/"
echo ""

# Count binaries
BINARY_COUNT=$(ls "${BINARY_DIR}/" | wc -l)
echo "Found ${BINARY_COUNT} binaries"
echo ""

# Ask user to select a binary or use all
echo "========================================================================="
echo "Select binary for testing:"
echo "========================================================================="
echo ""
echo "Options:"
echo "  1) 389-ds-base__libderef-plugin.so"
echo "  2) 6tunnel__6tunnel"
echo "  3) a2jmidid__a2jmidi_bridge"
echo "  4) a52dec__liba52.so.0.0.0"
echo "  5) aalib__aasavefont"
echo "  6) Process ALL binaries (concatenate outputs)"
echo ""

read -p "Enter selection [1-6] (default: 6): " selection
selection=${selection:-6}

echo ""
echo "========================================================================="
echo "Processing..."
echo "========================================================================="
echo ""

# Function to process a single binary
process_binary() {
    local binary_path=$1
    local binary_name=$(basename "${binary_path}")
    local output_prefix="${OUTPUT_DIR}/${binary_name}"
    
    echo "Processing: ${binary_name}"
    echo "  Binary: ${binary_path}"
    echo "  Output prefix: ${output_prefix}"
    
    # Check if data generator exists
    if [ ! -d "${DATA_GENERATOR_DIR}" ]; then
        echo "  ERROR: Data generator not found at ${DATA_GENERATOR_DIR}"
        return 1
    fi
    
    # Run data generation (assuming you have a script to generate inline format)
    # This is a placeholder - you'll need to adapt this to your actual data generation process
    
    # For now, we'll create a simple command that you can customize
    echo "  Note: Please implement data generation for your specific format"
    echo "  Expected outputs:"
    echo "    - ${output_prefix}_cfg_inline.txt"
    echo "    - ${output_prefix}_dfg_inline.txt"
    echo ""
    
    return 0
}

# Process based on selection
case $selection in
    1)
        process_binary "${BINARY_DIR}/389-ds-base__libderef-plugin.so"
        ;;
    2)
        process_binary "${BINARY_DIR}/6tunnel__6tunnel"
        ;;
    3)
        process_binary "${BINARY_DIR}/a2jmidid__a2jmidi_bridge"
        ;;
    4)
        process_binary "${BINARY_DIR}/a52dec__liba52.so.0.0.0"
        ;;
    5)
        process_binary "${BINARY_DIR}/aalib__aasavefont"
        ;;
    6)
        echo "Processing all binaries..."
        for binary in "${BINARY_DIR}"/*; do
            if [ -f "$binary" ]; then
                process_binary "$binary"
            fi
        done
        
        # Concatenate all outputs
        echo ""
        echo "Concatenating all outputs..."
        
        if ls "${OUTPUT_DIR}"/*_cfg_inline.txt 1> /dev/null 2>&1; then
            cat "${OUTPUT_DIR}"/*_cfg_inline.txt > "${OUTPUT_DIR}/all_cfg_test.txt"
            echo "  ✓ Created: ${OUTPUT_DIR}/all_cfg_test.txt"
        fi
        
        if ls "${OUTPUT_DIR}"/*_dfg_inline.txt 1> /dev/null 2>&1; then
            cat "${OUTPUT_DIR}"/*_dfg_inline.txt > "${OUTPUT_DIR}/all_dfg_test.txt"
            echo "  ✓ Created: ${OUTPUT_DIR}/all_dfg_test.txt"
        fi
        ;;
    *)
        echo "Invalid selection!"
        exit 1
        ;;
esac

echo ""
echo "========================================================================="
echo "Generation complete!"
echo "========================================================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "To use this data for probing:"
echo "  ./run_probing.sh --test_cfg ${OUTPUT_DIR}/all_cfg_test.txt \\"
echo "                   --test_dfg ${OUTPUT_DIR}/all_dfg_test.txt"
echo ""
