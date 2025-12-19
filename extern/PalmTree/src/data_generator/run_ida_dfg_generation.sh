#!/bin/bash
# Wrapper script to process binaries with IDA Pro batch mode for DFG generation

IDA_PATH="/home/kun/ida-pro-9.0"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/dfg_hierarchical_idfg_ida.py"

# Default parameters
SEG_LEN=${1:-2}
BIN_FOLDER=${2:-"/data/kun/samplebinary"}
OUTPUT_DIR=${3:-"/data/kun/palmtreedata"}

export OUTPUT_DIR
export SEG_LEN

echo "=========================================="
echo "IDA Pro DFG Generation"
echo "=========================================="
echo "IDA Path: $IDA_PATH"
echo "Script: $SCRIPT_PATH"
echo "SEG_LEN: $SEG_LEN"
echo "Binary Folder: $BIN_FOLDER"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each binary file
count=0
total=$(find "$BIN_FOLDER" -type f ! -name "*.txt" ! -name "*.idb" ! -name "*.i64" ! -name "*.id0" ! -name "*.id1" ! -name "*.id2" ! -name "*.nam" ! -name "*.til" | wc -l)

for binary in "$BIN_FOLDER"/*; do
    # Skip non-files and IDA database files
    if [ ! -f "$binary" ]; then
        continue
    fi
    
    basename=$(basename "$binary")
    
    # Skip text files and IDA database files
    if [[ "$basename" == *.txt ]] || \
       [[ "$basename" == *.idb ]] || \
       [[ "$basename" == *.i64 ]] || \
       [[ "$basename" == *.id0 ]] || \
       [[ "$basename" == *.id1 ]] || \
       [[ "$basename" == *.id2 ]] || \
       [[ "$basename" == *.nam ]] || \
       [[ "$basename" == *.til ]]; then
        continue
    fi
    
    count=$((count + 1))
    echo ""
    echo "======================================================================"
    echo "Processing $count/$total: $basename"
    echo "======================================================================"
    
    # Run IDA in batch mode
    # -A: autonomous mode (no user interaction)
    # -S: run script
    "$IDA_PATH/idat" -A -S"$SCRIPT_PATH" "$binary"
    
    # Clean up IDA database files if you don't want to keep them
    # Uncomment the following lines to remove .idb files after processing
    # rm -f "${binary}.idb" "${binary}.i64" "${binary}.id0" "${binary}.id1" "${binary}.id2" "${binary}.nam" "${binary}.til"
    
    echo "[INFO] Completed: $basename"
done

echo ""
echo "======================================================================"
echo "All binaries processed!"
echo "Output location: $OUTPUT_DIR"
echo "======================================================================"
