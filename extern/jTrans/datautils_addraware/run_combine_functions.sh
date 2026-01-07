#!/bin/bash

# Combine function export files from train/val/test directories
# This script merges all *_functions.txt files from separate directories into single train/val/test files
# with proper 0.8:0.1:0.1 ratio

# Usage: ./run_combine_functions.sh <input_dir> [output_dir]
# Example: ./run_combine_functions.sh /data/kun/jtransdata/function_exports /data/kun/jtransdata

echo "=========================================="
echo "Function File Combiner"
echo "=========================================="

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_dir> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  input_dir  : Directory containing *_functions.txt files"
    echo "  output_dir : (Optional) Output directory (default: /data/kun/jtransdata)"
    echo ""
    echo "Example:"
    echo "  $0 /data/kun/jtransdata/function_exports /data/kun/jtransdata"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-/data/kun/jtransdata}"

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
echo "  Symbol file: $INPUT_DIR/top_symbols.txt"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if required packages are installed
python3 -c "import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required package: tqdm"
    pip install tqdm
fi

# Run the combine script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMBINE_SCRIPT="$SCRIPT_DIR/combine_function_files.py"

if [ ! -f "$COMBINE_SCRIPT" ]; then
    echo "Error: Combine script not found: $COMBINE_SCRIPT"
    exit 1
fi

echo "Running combine script..."
python3 "$COMBINE_SCRIPT" "$INPUT_DIR" "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Combination complete!"
    echo "=========================================="
    echo "Output files:"
    echo "  - $OUTPUT_DIR/train.txt"
    echo "  - $OUTPUT_DIR/val.txt"
    echo "  - $OUTPUT_DIR/test.txt"
    echo "  - $INPUT_DIR/top_symbols.txt"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Error: Combination failed!"
    echo "=========================================="
    exit 1
fi
