#!/usr/bin/env python3
"""
Test model forward pass on CPU to get clearer error message.
"""

import torch
from transformers import BertTokenizer
from finetune import BinBertModel
from data_json import FunctionDataset_CL_Load_JSON

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('/home/kun/Document/AAE/extern/jTrans/pretrain/baseline')

print("Loading model...")
model = BinBertModel.from_pretrained('/home/kun/Document/AAE/output/jtrans/baseline_pretrain/checkpoint_epoch_10')
print("Model loaded on CPU")

print(f"word_embeddings shape: {model.embeddings.word_embeddings.weight.shape}")
print(f"position_embeddings shape: {model.embeddings.position_embeddings.weight.shape}")
print(f"Are they same? {model.embeddings.position_embeddings.weight is model.embeddings.word_embeddings.weight}")

print("\nLoading dataset...")
dataset = FunctionDataset_CL_Load_JSON(
    tokenizer,
    '/data/kun/jtransdata/func_blocks_baseline.json',
    '/data/kun/jtransdata/ground_truth_baseline.json',
    opt=['O0','O1','O2','O3'],
    add_ebd=True,
    data_ratio=0.001
)

print("Getting first batch...")
item = dataset[0]
input_ids1, input_ids2, input_ids3, mask1, mask2, mask3, seg1, seg2, seg3 = item

print(f"input_ids1 shape: {input_ids1.shape}, min: {input_ids1.min()}, max: {input_ids1.max()}")
print(f"mask1 shape: {mask1.shape}")
print(f"seg1 shape: {seg1.shape}")

# Add batch dimension
input_ids1 = input_ids1.unsqueeze(0)
mask1 = mask1.unsqueeze(0)
seg1 = seg1.unsqueeze(0)

print("\nTrying forward pass...")
try:
    output = model(input_ids=input_ids1, attention_mask=mask1, token_type_ids=seg1)
    print("✓ Forward pass successful!")
    print(f"Output shape: {output.pooler_output.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
