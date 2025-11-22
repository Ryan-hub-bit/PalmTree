#!/bin/bash

# Generate scope prediction dataset - process each binary separately
# This ensures all pairs are from the same binary

cd "$(dirname "$0")"

python generate_scope_per_binary.py \
    --cfg_dir ../../data/ncfg \
    --output ../../data/scope/all_scope.txt \
    --target_per_class 50000

echo ""
echo "Verifying output..."
echo "First 5 pairs:"
head -5 ../../data/scope/all_scope.txt

echo ""
echo "Label distribution:"
cut -f3 ../../data/scope/all_scope.txt | sort | uniq -c
