#!/bin/bash

# Convert address-format test data to original PalmTree format
# This creates new files suitable for testing the original PalmTree model

echo "=========================================="
echo "Converting Test Data to PalmTree Format"
echo "=========================================="

# Input files (with addresses)
CFG_INPUT="../data/test/cfg/all_cfg_combined.txt"
DFG_INPUT="../data/test/dfg/all_dfg_combined.txt"

# Output files (without addresses, for PalmTree)
CFG_OUTPUT="../data/test/cfg/all_cfg_palmtree.txt"
DFG_OUTPUT="../data/test/dfg/all_dfg_palmtree.txt"

# Check if input files exist
if [ ! -f "$CFG_INPUT" ]; then
    echo "Error: CFG input file not found: $CFG_INPUT"
    exit 1
fi

if [ ! -f "$DFG_INPUT" ]; then
    echo "Error: DFG input file not found: $DFG_INPUT"
    exit 1
fi

echo ""
echo "Converting CFG data..."
python3 convert_to_palmtree_format.py \
    --input "$CFG_INPUT" \
    --output "$CFG_OUTPUT"

echo ""
echo "Converting DFG data..."
python3 convert_to_palmtree_format.py \
    --input "$DFG_INPUT" \
    --output "$DFG_OUTPUT"

echo ""
echo "=========================================="
echo "Conversion Complete!"
echo "=========================================="
echo ""
echo "Output files created:"
echo "  CFG: $CFG_OUTPUT"
echo "  DFG: $DFG_OUTPUT"
echo ""
echo "These files can be used with the original PalmTree model."
echo "Addresses have been stripped from all instructions."
echo "=========================================="
