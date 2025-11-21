#!/bin/bash
# Generate BB bucket probe data from all binaries in /home/kun/onebinary/

echo "Generating BB Bucket Probe Data"
echo "================================"
echo ""
echo "Binary directory: /home/kun/onebinary/"
echo "Output directory: data/"
echo ""

python generate_bb_bucket_data.py \
    --binary_dir /home/kun/onebinary/ \
    --output_dir data \
    --num_buckets 10

echo ""
echo "Done! Check the 'data/' directory for results."
