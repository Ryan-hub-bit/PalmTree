#!/bin/bash

# Script to generate BB pairs from all binaries in ~/smallbinary/
# and combine them into a single file

BINARY_DIR=~/smallbinary
OUTPUT_DIR=~/PalmTree/bb_pairs_output
COMBINED_FILE=~/PalmTree/all_bb_pairs.txt
SCRIPT_PATH=~/PalmTree/scripts/bb_flow.py

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Remove old combined file if exists
rm -f "$COMBINED_FILE"

# Counter for statistics
total_binaries=0
total_pairs=0
failed_binaries=0

echo "========================================="
echo "Generating BB pairs from all binaries..."
echo "========================================="

# Process each binary in the directory
for binary in "$BINARY_DIR"/*; do
    # Skip if not a file
    if [ ! -f "$binary" ]; then
        continue
    fi
    
    binary_name=$(basename "$binary")
    output_file="$OUTPUT_DIR/${binary_name}_bb_pairs.txt"
    
    echo ""
    echo "Processing: $binary_name"
    
    # Run bb_flow.py on this binary
    if conda run -n palmtree python "$SCRIPT_PATH" "$binary" "$output_file" 2>&1; then
        # Count pairs in this file
        pair_count=$(wc -l < "$output_file")
        total_pairs=$((total_pairs + pair_count))
        total_binaries=$((total_binaries + 1))
        
        echo "  ✓ Generated $pair_count pairs"
        
        # Append to combined file with a header comment
        echo "# Binary: $binary_name ($pair_count pairs)" >> "$COMBINED_FILE"
        cat "$output_file" >> "$COMBINED_FILE"
        
    else
        echo "  ✗ Failed to process $binary_name"
        failed_binaries=$((failed_binaries + 1))
    fi
done

echo ""
echo "========================================="
echo "Summary:"
echo "========================================="
echo "Total binaries processed: $total_binaries"
echo "Failed binaries: $failed_binaries"
echo "Total BB pairs: $total_pairs"
echo ""
echo "Individual outputs saved in: $OUTPUT_DIR"
echo "Combined output saved in: $COMBINED_FILE"
echo "========================================="
