#!/bin/bash
# Script to regenerate complete baseline dataset
# Ensures baseline and address-aware have identical ground truth

set -e

echo "========================================================================"
echo "Regenerating Complete Baseline Dataset"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Delete old incomplete pickle files"
echo "  2. Run IDA Pro on ALL binaries to generate complete extractions"
echo "  3. Convert pickles to pretrain text format (baseline_pretrain.txt)"
echo "  4. Generate baseline function dataset with grouped ground truth"
echo ""
echo "WARNING: This will take several hours/days depending on dataset size!"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# ============================================================================
# Configuration - all paths and settings
# ============================================================================
BINARY_DIR="/data/kun/jtrans/small_train"
EXTRACT_DIR="/data/kun/jtrans/baseline/extract"
STRIP_DIR="/data/kun/jtrans/small_train_strip"  # Stripped binaries with same names
OUTPUT_DIR="/data/kun/jtrans/baseline"
IDA_PATH="./ida-pro-9.0/idat"
PROCESS_SCRIPT="$(pwd)/process.py"

echo ""
echo "========================================================================"
echo "Step 1/3: Deleting old incomplete pickle files"
echo "========================================================================"

if [ -d "$EXTRACT_DIR" ]; then
    echo "Removing old pickle files from $EXTRACT_DIR..."
    rm -rf "$EXTRACT_DIR"
    echo "✓ Old files deleted"
else
    echo "No existing extract directory found, nothing to delete"
fi

# Create extract directory
mkdir -p "$EXTRACT_DIR"
mkdir -p "$STRIP_DIR"

echo ""
echo "========================================================================"
echo "Step 2/3: Running IDA Pro extraction on ALL binaries"
echo "========================================================================"
echo "Binary directory: $BINARY_DIR"
echo "Output directory: $EXTRACT_DIR"
echo ""
echo "NOTE: This step will take a LONG time!"
echo "You can monitor progress in: $EXTRACT_DIR/"
echo ""

# Change to datautils directory
cd "$(dirname "$0")"

# Switch IDA to system Python3
echo "Switching IDA to system Python3..."
if [ -f "./ida-pro-9.0/idapyswitch" ]; then
    ./ida-pro-9.0/idapyswitch
    echo "✓ IDA switched to system Python3"
else
    echo "WARNING: ./ida/idapyswitch not found, skipping..."
fi
echo ""

# Count total binaries (excluding IDA-generated files)
total_binaries=$(find "$BINARY_DIR" -maxdepth 1 -type f \
    ! -name "*.i64" ! -name "*.idb" ! -name "*.id0" ! -name "*.id1" ! -name "*.id2" \
    ! -name "*.nam" ! -name "*.til" ! -name "*.txt" ! -name "*.log" ! -name "*.strip" | wc -l)
echo "Found $total_binaries binaries to process"
echo ""

echo "Configuration:"
echo "  BINARY_DIR=$BINARY_DIR"
echo "  EXTRACT_DIR=$EXTRACT_DIR"
echo "  STRIP_DIR=$STRIP_DIR"
echo "  IDA_PATH=$IDA_PATH"
echo ""

# Run IDA extraction with arguments
python3 run.py \
    --binary-dir "$BINARY_DIR" \
    --extract-dir "$EXTRACT_DIR" \
    --strip-path "$STRIP_DIR" \
    --ida-path "$IDA_PATH" \
    --process-script "$PROCESS_SCRIPT"

echo ""
echo "✓ IDA extraction complete"

# Count generated pickle files
pkl_count=$(find "$EXTRACT_DIR" -name "*_extract.pkl" -type f | wc -l)
echo "Generated $pkl_count pickle files"

echo ""
echo "========================================================================"
echo "Step 3/4: Converting pickles to pretrain text format"
echo "========================================================================"
echo "Input: $EXTRACT_DIR/*_extract.pkl"
echo "Output: $OUTPUT_DIR/baseline_pretrain.txt"
echo ""

python3 convert_pkl_to_text.py \
    "$EXTRACT_DIR" \
    "$OUTPUT_DIR/baseline_pretrain.txt" \
    --min-instructions 5 \
    --max-instructions 512

echo ""
echo "✓ Pretrain text file generated"

# Count lines in pretrain file
if [ -f "$OUTPUT_DIR/baseline_pretrain.txt" ]; then
    line_count=$(wc -l < "$OUTPUT_DIR/baseline_pretrain.txt")
    echo "Generated $line_count functions in baseline_pretrain.txt"
fi

echo ""
echo "========================================================================"
echo "Step 4/4: Generating baseline function dataset"
echo "========================================================================"

python3 create_baseline_dataset.py \
    "$EXTRACT_DIR" \
    "$OUTPUT_DIR" \
    --binary-dir "$BINARY_DIR"

echo ""
echo "========================================================================"
echo "COMPLETE!"
echo "========================================================================"
echo ""
echo "Baseline dataset files:"
echo "  - $OUTPUT_DIR/baseline_pretrain.txt"
echo "  - $OUTPUT_DIR/func_blocks_baseline.json"
echo "  - $OUTPUT_DIR/ground_truth_baseline.json"
echo ""
echo "Next steps:"
echo "  1. Build vocabulary: python create_vocab.py --input_file baseline_pretrain.txt"
echo "  2. Run baseline pretraining"
echo "  3. Run baseline finetuning"
echo ""
