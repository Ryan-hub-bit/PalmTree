#!/bin/bash

# Combine function export files for PRETRAINING (address-aware version)
# Combines ALL functions into a single file without train/val/test split

# Usage: ./run_combine_pretrain.sh <input_dir> [output_dir] [sample_ratio]
# Example: ./run_combine_pretrain.sh /data/kun/jtransdata/function_exports /data/kun/jtransdata 0.1

echo "=========================================="
echo "Pretraining Data Combiner (Address-Aware)"
echo "=========================================="

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_dir> [output_dir] [sample_ratio]"
    echo ""
    echo "Arguments:"
    echo "  input_dir    : Directory containing *_functions.txt files"
    echo "  output_dir   : (Optional) Output directory (default: /data/kun/jtransdata)"
    echo "  sample_ratio : (Optional) Ratio of data to use (0.0-1.0, default: 1.0)"
    echo ""
    echo "Examples:"
    echo "  $0 /data/kun/jtransdata/function_exports /data/kun/jtransdata 1.0    # Use 100% of data"
    echo "  $0 /data/kun/jtransdata/function_exports /data/kun/jtransdata 0.1    # Use 10% of data"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-/data/kun/jtransdata}"
SAMPLE_RATIO="${3:-1.0}"

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Sample ratio: $SAMPLE_RATIO ($(echo "$SAMPLE_RATIO * 100" | bc)% of data)"
echo "  Output file: addr_pretrain.txt"
echo "  Symbol file: top_symbols.txt"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

echo "Running combine script..."
echo "=========================================="

# Run the Python script
python3 combine_function_files_pretrain.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --top_n 100 \
    --sample_ratio "$SAMPLE_RATIO" \
    --enable_smart_merge \
    --output_name addr_pretrain.txt

echo ""
echo "=========================================="
echo "DONE!"
echo ""
echo "Output file: $OUTPUT_DIR/addr_pretrain.txt"
echo "Symbol file: $OUTPUT_DIR/top_symbols.txt"
echo ""
echo "Next step: Create vocabulary from addr_pretrain.txt"
echo "  cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware"
echo "  python3 create_vocab.py \\
    --input $OUTPUT_DIR/addr_pretrain.txt \\
    --output $OUTPUT_DIR/vocab_addr.txt \\
    --min_freq 50"
echo "=========================================="
