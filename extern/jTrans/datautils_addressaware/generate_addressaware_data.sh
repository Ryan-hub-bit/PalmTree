#!/bin/bash
#
# One-command script to generate address-aware training data from pickles
# This is the pickle conversion approach (Approach 2)
#
# Usage:
#   bash generate_addressaware_data.sh
#

set -e

# Configuration
INPUT_PKL_DIR="${1:-/data/kun/jtransdata/extract}"
OUTPUT_DIR="${2:-./data}"
BINARY_ADDR_RANGE="0x400000:0x600000"

echo "================================"
echo "Address-Aware Data Generation"
echo "================================"
echo "Input (pickles): $INPUT_PKL_DIR"
echo "Output: $OUTPUT_DIR"
echo "Binary address range: $BINARY_ADDR_RANGE"
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_PKL_DIR" ]; then
    echo "[ERROR] Input directory does not exist: $INPUT_PKL_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count pickle files
pkl_count=$(find "$INPUT_PKL_DIR" -name "*.pkl" | wc -l)
echo "[INFO] Found $pkl_count pickle files"
echo ""

# Step 1: Convert pickles to address-aware format
echo "[STEP 1] Converting pickle files to address-aware format..."
python convert_pkl_to_addressaware.py \
    --input "$INPUT_PKL_DIR" \
    --output "$OUTPUT_DIR/addressaware_raw" \
    --binary-addr-range "$BINARY_ADDR_RANGE"

echo ""
echo "[STEP 1] Complete!"
echo ""

# Step 2: Combine all files
echo "[STEP 2] Combining address-aware files..."
COMBINED_FILE="$OUTPUT_DIR/combined_all.txt"
rm -f "$COMBINED_FILE"

file_count=0
for file in "$OUTPUT_DIR/addressaware_raw"/*.txt; do
    if [ -f "$file" ]; then
        cat "$file" >> "$COMBINED_FILE"
        file_count=$((file_count + 1))
    fi
done

echo "[INFO] Combined $file_count files"
total_lines=$(wc -l < "$COMBINED_FILE")
echo "[INFO] Total lines: $total_lines"
echo ""

# Step 3: Split into train and test
echo "[STEP 3] Splitting into train/test (90%/10%)..."
train_lines=$((total_lines * 9 / 10))
test_lines=$((total_lines - train_lines))

TRAIN_FILE="$OUTPUT_DIR/addressaware_train.txt"
TEST_FILE="$OUTPUT_DIR/addressaware_test.txt"

head -n "$train_lines" "$COMBINED_FILE" > "$TRAIN_FILE"
tail -n "$test_lines" "$COMBINED_FILE" > "$TEST_FILE"

echo "[INFO] Train: $train_lines lines"
echo "[INFO] Test: $test_lines lines"
echo ""

# Step 4: Summary
echo "================================"
echo "Generation Complete!"
echo "================================"
echo "Train file: $TRAIN_FILE"
echo "Test file: $TEST_FILE"
echo ""
echo "Sample from train file:"
echo "--------------------------------"
head -n 2 "$TRAIN_FILE"
echo "--------------------------------"
echo ""
echo "Token format examples:"
echo "  Opcode:    push(0x401000:0.12345678:0.23456789:0.34567890)"
echo "  Address:   address(0x401050:0.12345678:0.23456789:0.34567890)"
echo "  Data addr: daddr(0x600000:0.50000000:0.25000000:0.00000000)"
echo "  Stack var: var(0x10)"
echo "  Immediate: imm"
echo ""
echo "Next steps:"
echo "1. Create vocabulary:"
echo "   cd ../pretrain/address_aware"
echo "   python -c 'from vocab import WordVocab; WordVocab.create_vocab(\"$TRAIN_FILE\", \"vocab.pkl\")'"
echo ""
echo "2. Train model:"
echo "   bash run_addressaware_pretrain.sh"
echo "================================"
