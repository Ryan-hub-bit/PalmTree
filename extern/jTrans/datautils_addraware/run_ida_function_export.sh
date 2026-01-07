#!/bin/bash

# Run IDA Pro to export entire functions with hierarchical position encoding
# This script processes binary file(s) or a directory of binaries
# Usage: ./run_ida_function_export.sh <binary_file_or_directory> [output_dir]

# IDA Pro installation path
IDA_PATH="/home/kun/ida-pro-9.0/idat"

# Python script to run inside IDA
SCRIPT_PATH="/home/kun/Document/AAE/extern/jTrans/datautils_addraware/function_export_ida.py"

# Default output directory
DEFAULT_OUTPUT_DIR="/data/kun/jtransdata/function_exports"

# Check if input is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <binary_file_or_directory> [output_dir]"
    echo "Example: $0 /path/to/binary"
    echo "         $0 /path/to/binary_directory"
    echo "         $0 /path/to/binary /custom/output/dir"
    exit 1
fi

INPUT="$1"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

# Check if input exists
if [ ! -e "$INPUT" ]; then
    echo "Error: Input not found: $INPUT"
    exit 1
fi

# Check if input exists
if [ ! -e "$INPUT" ]; then
    echo "Error: Input not found: $INPUT"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Export environment variables for the IDA script
export OUTPUT_DIR="$OUTPUT_DIR"

# Function to process a single binary
process_binary() {
    local binary="$1"
    echo "=========================================="
    echo "IDA Pro Function Export"
    echo "=========================================="
    echo "Binary:  $binary"
    echo "Output:  $OUTPUT_DIR"
    echo "Script:  $SCRIPT_PATH"
    echo "=========================================="

    # Run IDA Pro in batch mode
    # -A: autonomous mode (no dialogs)
    # -S: run script on startup
    "$IDA_PATH" -A -S"$SCRIPT_PATH" "$binary"

    return $?
}

# Check if input is a directory or a file
if [ -d "$INPUT" ]; then
    # Process all files in directory
    echo "=========================================="
    echo "Processing directory: $INPUT"
    echo "=========================================="
    
    TOTAL_FILES=0
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    # Find all executable files (binaries) in the directory
    while IFS= read -r -d '' binary; do
        ((TOTAL_FILES++))
        echo ""
        echo "Processing file $TOTAL_FILES: $(basename "$binary")"
        
        if process_binary "$binary"; then
            ((SUCCESS_COUNT++))
            echo "✓ Success: $(basename "$binary")"
        else
            ((FAIL_COUNT++))
            echo "✗ Failed: $(basename "$binary")"
        fi
    done < <(find "$INPUT" -maxdepth 1 -type f -executable -print0)
    
    # Summary
    echo ""
    echo "=========================================="
    echo "Batch Processing Summary"
    echo "=========================================="
    echo "Total files: $TOTAL_FILES"
    echo "Successful:  $SUCCESS_COUNT"
    echo "Failed:      $FAIL_COUNT"
    echo "Output dir:  $OUTPUT_DIR"
    echo "=========================================="
    
    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
    
elif [ -f "$INPUT" ]; then
    # Process single file
    if process_binary "$INPUT"; then
        echo "=========================================="
        echo "Processing completed successfully!"
        echo "Output files are in: $OUTPUT_DIR"
        echo "=========================================="
    else
        echo "=========================================="
        echo "Error: Processing failed!"
        echo "Check the log file in: $OUTPUT_DIR/ida_processing.log"
        echo "=========================================="
        exit 1
    fi
else
    echo "Error: Input is neither a file nor a directory: $INPUT"
    exit 1
fi
