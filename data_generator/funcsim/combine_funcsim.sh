#!/bin/bash
#
# Wrapper script to combine and deduplicate funcsim_match data
#
# Usage: ./combine_funcsim.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/combine_and_deduplicate_funcsim.py"
FUNCSIM_DIR="/data/kun/funcsim_match"

echo "========================================"
echo "Combine and Deduplicate FuncSim Data"
echo "========================================"
echo ""

# Check if funcsim directory exists
if [ ! -d "$FUNCSIM_DIR" ]; then
    echo "[ERROR] FuncSim directory not found: $FUNCSIM_DIR"
    exit 1
fi

# Run the Python script
python3 "$PYTHON_SCRIPT" "$FUNCSIM_DIR"

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
echo ""
echo "Output file: ${FUNCSIM_DIR}/combined_deduplicated.json"
echo ""
