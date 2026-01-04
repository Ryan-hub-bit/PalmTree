#!/bin/bash
#
# Batch process binaries with IDA Pro to generate address-aware format
# Saves both pickle files (for metadata) and text files (for training)
#
# Usage:
#   bash run_ida_addressaware.sh /path/to/binaries/dir /path/to/output/dir [/path/to/unstripped/dir]
#

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir> [dataroot_dir]"
    echo ""
    echo "Arguments:"
    echo "  input_dir    : Directory containing binaries to process"
    echo "  output_dir   : Directory where pickle and text files will be saved (SAVEROOT)"
    echo "  dataroot_dir : (Optional) Directory with unstripped binaries for symbol info (DATAROOT)"
    echo ""
    echo "Example:"
    echo "  $0 /data/kun/jtransdata/dataset /data/kun/jtransdata/addr_extract"
    echo "  $0 /data/kun/jtransdata/dataset /data/kun/jtransdata/addr_extract /data/kun/jtransdata/dataset"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
DATAROOT_DIR="${3:-$INPUT_DIR}"  # Default to INPUT_DIR if not provided
IDA_PATH="/home/kun/ida-pro-9.0/idat"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/process.py"

# Verify paths
if [ ! -d "$INPUT_DIR" ]; then
    echo "[ERROR] Input directory does not exist: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$IDA_PATH" ]; then
    echo "[ERROR] IDA Pro not found: $IDA_PATH"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[ERROR] process.py not found: $SCRIPT_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Input directory (binaries): $INPUT_DIR"
echo "[INFO] Output directory (SAVEROOT): $OUTPUT_DIR"
echo "[INFO] Data directory (DATAROOT): $DATAROOT_DIR"
echo "[INFO] IDA Pro: $IDA_PATH"
echo "[INFO] Script: $SCRIPT_PATH"
echo ""

# Export environment variables for process.py
export SAVEROOT="$OUTPUT_DIR"
export DATAROOT="$DATAROOT_DIR"
export LOGROOT="${OUTPUT_DIR}/logs"  # Separate directory for IDA logs
export TVHEADLESS=1  # Required for IDA Pro to work with redirected output

# Create logs directory
mkdir -p "$LOGROOT"

# Process binaries
binary_count=0
success_count=0
fail_count=0

for binary in "$INPUT_DIR"/*; do
    # Skip if not a file
    if [ ! -f "$binary" ]; then
        continue
    fi
    
    binary_name=$(basename "$binary")
    
    # Skip IDA database files and other non-binary files
    case "$binary_name" in
        *.i64|*.i32|*.idb|*.id0|*.id1|*.id2|*.nam|*.til|*.pkl|*.txt|*.json|*.log)
            continue
            ;;
    esac
    
    binary_count=$((binary_count + 1))
    
    echo "[$binary_count] Processing: $binary_name"
    
    # Check if output files already exist
    pickle_file="$OUTPUT_DIR/${binary_name}_extract.pkl"
    text_file="$OUTPUT_DIR/${binary_name}_addressaware.txt"
    
    if [ -f "$pickle_file" ] && [ -f "$text_file" ]; then
        echo "    [SKIP] Both outputs already exist"
        success_count=$((success_count + 1))
        continue
    fi

    
    # Run IDA Pro (process.py reads SAVEROOT and DATAROOT from env vars)
    # Use -c (console mode) and -L (log file) flags following jTrans/datautils/run.py
    # -OIDAPython:AnalysisFlags:~AF_USEDBG disables debug symbol names for local variables
    LOG_FILE="$OUTPUT_DIR/${binary_name}_ida.log"
    "$IDA_PATH" -L"$LOG_FILE" -c -A -OIDAPython:AnalysisFlags:~AF_USEDBG -S"$SCRIPT_PATH" "$binary" >/dev/null 2>&1
    
    # Check if outputs were generated
    if [ -f "$pickle_file" ] && [ -f "$text_file" ]; then
        echo "    [SUCCESS] Generated pickle and text files"
        success_count=$((success_count + 1))
    else
        echo "    [FAILED] Missing output files"
        fail_count=$((fail_count + 1))
    fi

done

echo ""
echo "================================"
echo "Processing complete"
echo "Total binaries: $binary_count"
echo "Success: $success_count"
echo "Failed: $fail_count"
echo "================================"
