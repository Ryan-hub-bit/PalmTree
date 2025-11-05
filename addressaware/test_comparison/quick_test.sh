#!/bin/bash

# Quick test with minimal samples to verify the testing framework works

set -e

echo "Running quick test with 1000 samples to verify setup..."
echo ""

python test_models.py \
    --cfg_data ../data/cfg/all_cfg_combined.txt \
    --dfg_data ../data/dfg/all_dfg_combined.txt \
    --vocab ../../pre-trained_model/palmtree/vocab \
    --palmtree_checkpoint ../../pre-trained_model/palmtree/transformer.ep19 \
    --batch_size 128 \
    --seq_len 20 \
    --test_samples 1000 \
    --num_workers 2 \
    --output results/quick_test.json

echo ""
echo "✓ Quick test completed successfully!"
echo "Check results/quick_test.json for output"
