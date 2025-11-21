#!/bin/bash

# Test the BB bucket probe with the generated data

echo "=================================================================="
echo "Testing BB Bucket Probe"
echo "=================================================================="

cd /home/kun/Document/PalmTree/probe

python bb_bucket_probe.py \
    --data_dir data \
    --vocab ../pre-trained_model/palmtree/vocab \
    --addressaware_model ../addressaware/output_addressaware_all/best_bert.pt \
    --baseline_model ../addressaware/output_baseline_new/best_bert.pt \
    --output probe_results \
    --batch_size 16 \
    --seq_len 10 \
    --test_size 0.1

echo ""
echo "=================================================================="
echo "Test complete! Check probe_results/ for output."
echo "=================================================================="
