#!/bin/bash
# Generate instr_pretrain.txt from pickle files

set -e

echo "========================================================================"
echo "Generate jTrans_instr Pretraining Text File"
echo "========================================================================"

# Configuration
EXTRACT_DIR="/data/kun/jtrans_instr/extract"
OUTPUT_FILE="/data/kun/jtrans_instr/instr_pretrain.txt"

echo ""
echo "Input:  $EXTRACT_DIR/*.pkl"
echo "Output: $OUTPUT_FILE"
echo ""

# Check if extract directory exists
if [ ! -d "$EXTRACT_DIR" ]; then
    echo "Error: Extract directory not found: $EXTRACT_DIR"
    echo "Please run generate_baseline.sh first to create pickle files."
    exit 1
fi

# Count pickle files
pkl_count=$(find "$EXTRACT_DIR" -name "*_extract.pkl" -type f | wc -l)
if [ "$pkl_count" -eq 0 ]; then
    echo "Error: No pickle files found in $EXTRACT_DIR"
    echo "Please run generate_baseline.sh first."
    exit 1
fi

echo "Found $pkl_count pickle files"
echo ""

# Run conversion
cd "$(dirname "$0")"

python3 convert_pkl_to_text.py \
    "$EXTRACT_DIR" \
    "$OUTPUT_FILE" \
    --min-instructions 5 \
    --max-instructions 512

echo ""
echo "========================================================================"
echo "DONE!"
echo "========================================================================"
echo ""
echo "Output: $OUTPUT_FILE"
echo ""
echo "Sample:"
head -2 "$OUTPUT_FILE"
echo ""
echo "Next step: Use this file for jTrans_instr pretraining"
echo ""
