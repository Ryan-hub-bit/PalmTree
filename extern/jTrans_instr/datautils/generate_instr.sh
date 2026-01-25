#!/bin/bash
# Script to generate complete instruction-level dataset for jTrans_instr
# Uses same workflow as baseline but with instruction-level addressing

set -e

echo "========================================================================"
echo "Generating Complete Instruction-Level Dataset"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Delete old incomplete pickle files"
echo "  2. Run IDA Pro on ALL binaries to generate instruction-level extractions"
echo "  3. Generate pretraining text file (instr_pretrain.txt)"
echo "  4. Generate instruction-level function dataset with ground truth"
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
BINARY_DIR="/data/kun/jtrans/small_train"
STRIP_DIR="/data/kun/jtrans/small_train_strip"
EXTRACT_DIR="/data/kun/jtrans/instr/extract"
OUTPUT_DIR="/data/kun/jtrans/instr"
IDA_PATH="./ida-pro-9.0/idat"
PROCESS_SCRIPT="$(pwd)/process_instr.py"

echo ""
echo "========================================================================"
echo "Step 1/3: Checking for existing pickle files"
echo "========================================================================"

# Check if pickle files already exist
if [ -d "$EXTRACT_DIR" ]; then
    pkl_count=$(find "$EXTRACT_DIR" -name "*_extract.pkl" -type f 2>/dev/null | wc -l)
    if [ "$pkl_count" -gt 0 ]; then
        echo "Found $pkl_count existing pickle files in $EXTRACT_DIR"
        read -p "Do you want to reuse existing pickle files? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "✓ Reusing existing pickle files, skipping IDA extraction"
            SKIP_IDA=true
        else
            echo "Removing old pickle files from $EXTRACT_DIR..."
            rm -rf "$EXTRACT_DIR"
            mkdir -p "$EXTRACT_DIR"
            echo "✓ Old files deleted"
            SKIP_IDA=false
        fi
    else
        echo "Extract directory exists but no pickle files found"
        SKIP_IDA=false
    fi
else
    echo "No existing extract directory found"
    mkdir -p "$EXTRACT_DIR"
    SKIP_IDA=false
fi

echo ""
echo "========================================================================"
echo "Step 2/3: Running IDA Pro extraction on ALL binaries"
echo "========================================================================"

if [ "$SKIP_IDA" = true ]; then
    echo "Skipping IDA extraction (reusing existing pickle files)"
    pkl_count=$(find "$EXTRACT_DIR" -name "*_extract.pkl" -type f | wc -l)
    echo "Using $pkl_count existing pickle files"
else
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
    if [ -f "../jTrans/datautils/ida-pro-9.0/idapyswitch" ]; then
        ../jTrans/datautils/ida-pro-9.0/idapyswitch
        echo "✓ IDA switched to system Python3"
    else
        echo "WARNING: IDA Python switcher not found, skipping..."
    fi
    echo ""

    # Count total binaries (excluding IDA-generated files)
    total_binaries=$(find "$BINARY_DIR" -maxdepth 1 -type f \
        ! -name "*.i64" ! -name "*.idb" ! -name "*.id0" ! -name "*.id1" ! -name "*.id2" \
        ! -name "*.nam" ! -name "*.til" ! -name "*.txt" ! -name "*.log" ! -name "*.strip" | wc -l)
    echo "Found $total_binaries binaries to process"
    echo ""

    # Run IDA extraction with arguments
    python3 run_instr.py \
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
fi

echo ""
echo "========================================================================"
echo "Step 3/4: Generating pretraining text file (instr_pretrain.txt)"
echo "========================================================================"

PRETRAIN_FILE="$OUTPUT_DIR/instr_pretrain.txt"

echo "Converting pickle files to pretraining format..."
echo "Output: $PRETRAIN_FILE"
echo ""

python3 convert_pkl_to_text.py \
    "$EXTRACT_DIR" \
    "$PRETRAIN_FILE" \
    --min-instructions 5 \
    --max-instructions 512

if [ -f "$PRETRAIN_FILE" ]; then
    line_count=$(wc -l < "$PRETRAIN_FILE")
    echo "✓ Generated $PRETRAIN_FILE ($line_count functions)"
else
    echo "ERROR: Failed to generate pretrain file"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Step 4/4: Generating instruction-level function dataset (JSON)"
echo "========================================================================"

echo "Creating finetuning dataset with ground truth..."
echo ""

python3 create_instr_dataset.py \
    "$EXTRACT_DIR" \
    "$OUTPUT_DIR" \
    --binary-dir "$BINARY_DIR"

if [ -f "$OUTPUT_DIR/func_blocks_instr.json" ]; then
    func_count=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_DIR/func_blocks_instr.json'))))" 2>/dev/null || echo "?")
    echo "✓ Generated func_blocks_instr.json ($func_count functions)"
fi

if [ -f "$OUTPUT_DIR/ground_truth_instr.json" ]; then
    pair_count=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/ground_truth_instr.json'))['total_pairs'])" 2>/dev/null || echo "?")
    echo "✓ Generated ground_truth_instr.json ($pair_count groups)"
fi

echo ""
echo "========================================================================"
echo "COMPLETE!"
echo "========================================================================"
echo ""
echo "Generated files in: $OUTPUT_DIR"
echo ""
echo "Pretraining:"
echo "  - instr_pretrain.txt"
echo ""
echo "Finetuning/Evaluation:"
echo "  - func_blocks_instr.json"
echo "  - ground_truth_instr.json"
echo ""
echo "Next steps:"
echo "  1. Pretrain: Use instr_pretrain.txt for pretraining"
echo "  2. Finetune: Use func_blocks_instr.json for finetuning"
echo "  3. Evaluate: Use ground_truth_instr.json for evaluation"
echo ""
