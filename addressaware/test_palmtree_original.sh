#!/bin/bash

# Generate dummy address files and test original PalmTree model

echo "=========================================="
echo "Step 1: Generate dummy address files"
echo "=========================================="
python generate_dummy_addresses.py

echo ""
echo "=========================================="
echo "Step 2: Test original PalmTree model"
echo "=========================================="

python test_palmtree_original.py \
    --checkpoint ../pre-trained_model/palmtree/transformer.ep19 \
    --vocab ../pre-trained_model/palmtree/vocab \
    --cfg_corpus ../data/test/cfg/all_cfg_palmtree.txt \
    --dfg_corpus ../data/test/dfg/all_dfg_palmtree.txt \
    --cfg_src ../data/test/cfg/all_cfg_src_addr.txt \
    --dfg_src ../data/test/dfg/all_dfg_src_addr.txt \
    --cfg_tgt ../data/test/cfg/all_cfg_tgt_addr.txt \
    --dfg_tgt ../data/test/dfg/all_dfg_tgt_addr.txt \
    --seq_len 100 \
    --batch_size 128 \
    --num_workers 4

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
