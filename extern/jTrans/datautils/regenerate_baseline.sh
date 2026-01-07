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
echo "  3. Generate baseline function dataset with grouped ground truth"
echo ""
echo "WARNING: This will take several hours/days depending on dataset size!"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Configuration
BINARY_DIR="/data/kun/jtransdata/small_train"
EXTRACT_DIR="/data/kun/jtransdata/extract"
OUTPUT_DIR="/data/kun/jtransdata"

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

# Run IDA extraction
python3 run.py

echo ""
echo "✓ IDA extraction complete"

# Count generated pickle files
pkl_count=$(find "$EXTRACT_DIR" -name "*_extract.pkl" -type f | wc -l)
echo "Generated $pkl_count pickle files"

echo ""
echo "========================================================================"
echo "Step 3/3: Generating baseline function dataset"
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
echo "  - $OUTPUT_DIR/func_blocks_baseline.json"
echo "  - $OUTPUT_DIR/ground_truth_baseline.json"
echo ""
echo "Next steps:"
echo "  1. Verify the new ground truth has same coverage as address-aware"
echo "  2. Run baseline pretraining"
echo "  3. Run baseline finetuning"
echo ""
