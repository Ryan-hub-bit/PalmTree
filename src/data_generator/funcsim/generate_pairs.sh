#!/bin/bash
#
# Generate function similarity pairs from deduplicated funcsim data
#
# Usage: ./generate_pairs.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate_funcsim_pairs.py"
INPUT_FILE="/data/kun/funcsim_match/combined_deduplicated.json"

echo "========================================"
echo "Generate Function Similarity Pairs"
echo "========================================"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "[ERROR] Input file not found: $INPUT_FILE"
    echo "[INFO] Please run combine_funcsim.sh first to generate the deduplicated data"
    exit 1
fi

# Run the Python script
python3 "$PYTHON_SCRIPT" "$INPUT_FILE"

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
echo ""
echo "Output files created in: /data/kun/funcsim_match/"
echo "  - function_blocks.json"
echo "  - funcsim_pairs.json"
echo ""
