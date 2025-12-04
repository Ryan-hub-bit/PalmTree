#!/bin/bash
#
# Generate Function Similarity JSON using IDA Pro
# 
# This script processes the same binary at different optimization levels
# and generates a JSON file with function instructions.
#
# Usage: ./generate_funcsim_json.sh <binary_name> <dataset_path> <output_file>
# Example: ./generate_funcsim_json.sh openssl /data/kun/funcsim_dataset /tmp/openssl_funcsim.json
#

set -e

# Configuration
IDA_PATH="/home/kun/ida-pro-9.0/idat64"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IDA_SCRIPT="${SCRIPT_DIR}/generate_funcsim_json_ida.py"

# Parse arguments
BINARY_NAME="${1:-openssl}"
DATASET_PATH="${2:-/data/kun/funcsim_dataset}"
OUTPUT_FILE="${3:-/tmp/${BINARY_NAME}_funcsim.json}"

# Optimization levels to process
OPT_LEVELS=("O0" "O1" "O2" "O3")

echo "================================================"
echo "Generate Function Similarity JSON"
echo "================================================"
echo "Binary: ${BINARY_NAME}"
echo "Dataset path: ${DATASET_PATH}"
echo "Output file: ${OUTPUT_FILE}"
echo "IDA script: ${IDA_SCRIPT}"
echo "================================================"

# Remove existing output file to start fresh
if [ -f "${OUTPUT_FILE}" ]; then
    echo "[INFO] Removing existing output file: ${OUTPUT_FILE}"
    rm "${OUTPUT_FILE}"
fi

# Process each optimization level
for opt in "${OPT_LEVELS[@]}"; do
    BINARY_PATH="${DATASET_PATH}/${BINARY_NAME}_${opt}/${BINARY_NAME}"
    
    if [ ! -f "${BINARY_PATH}" ]; then
        echo "[WARNING] Binary not found: ${BINARY_PATH}"
        continue
    fi
    
    echo ""
    echo "================================================"
    echo "[INFO] Processing ${opt}: ${BINARY_PATH}"
    echo "================================================"
    
    # Set environment variables for the IDA script
    export OPT_LEVEL="${opt}"
    export OUTPUT_FILE="${OUTPUT_FILE}"
    
    # Run IDA in batch mode
    # -A: Autonomous mode (no dialogs)
    # -S: Run script
    # -L: Log file
    LOG_FILE="/tmp/ida_funcsim_${BINARY_NAME}_${opt}.log"
    
    "${IDA_PATH}" -A -S"${IDA_SCRIPT}" -L"${LOG_FILE}" "${BINARY_PATH}"
    
    echo "[INFO] Finished processing ${opt}"
    echo "[INFO] Log file: ${LOG_FILE}"
done

echo ""
echo "================================================"
echo "[DONE] All optimization levels processed"
echo "Output file: ${OUTPUT_FILE}"
echo "================================================"

# Show summary
if [ -f "${OUTPUT_FILE}" ]; then
    echo ""
    echo "[INFO] JSON file summary:"
    python3 -c "
import json
with open('${OUTPUT_FILE}', 'r') as f:
    data = json.load(f)
    
total_funcs = len(data)
funcs_with_all = 0
opt_counts = {'O0': 0, 'O1': 0, 'O2': 0, 'O3': 0}

for func_name, opt_data in data.items():
    has_all = True
    for opt in ['O0', 'O1', 'O2', 'O3']:
        if opt in opt_data:
            opt_counts[opt] += 1
        else:
            has_all = False
    if has_all:
        funcs_with_all += 1

print(f'  Total functions: {total_funcs}')
print(f'  Functions with all 4 opt levels: {funcs_with_all}')
for opt, count in opt_counts.items():
    print(f'  Functions in {opt}: {count}')
"
fi
