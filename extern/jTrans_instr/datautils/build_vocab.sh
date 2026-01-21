#!/bin/bash
# Build vocabulary from jTrans_instr pretraining data

set -e

echo "========================================================================"
echo "Build jTrans_instr Vocabulary"
echo "========================================================================"

# Configuration
INPUT_FILE="/data/kun/jtrans_instr/instr_pretrain.txt"
OUTPUT_FILE="/home/kun/Document/AAE/extern/jTrans_instr/jtrans_tokenizer/vocab.txt"

echo ""
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo "Please run generate_text.sh first to create the pretraining text file."
    exit 1
fi

# Run vocabulary builder
cd "$(dirname "$0")"

python3 build_vocab.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --min-freq 1

echo ""
echo "========================================================================"
echo "DONE!"
echo "========================================================================"
echo ""
echo "Vocabulary: $OUTPUT_FILE"
echo ""
echo "Sample (first 20 lines):"
head -20 "$OUTPUT_FILE"
echo "..."
echo ""
echo "Sample (last 10 lines):"
tail -10 "$OUTPUT_FILE"
echo ""
echo "Next step: Use this vocabulary for jTrans_instr pretraining"
echo ""
