#!/usr/bin/env python3
"""
Test that baseline model can handle the data correctly.
"""

import torch
from transformers import BertTokenizer
from data_json import FunctionDataset_CL_Load_JSON

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('/home/kun/Document/AAE/extern/jTrans/pretrain/baseline')

print("Loading dataset...")
dataset = FunctionDataset_CL_Load_JSON(
    tokenizer,
    '/data/kun/jtransdata/func_blocks_baseline.json',
    '/data/kun/jtransdata/ground_truth_baseline.json',
    opt=['O0','O1','O2','O3'],
    add_ebd=True,
    data_ratio=0.001
)

print(f"Dataset size: {len(dataset)}")

print(f"\nGetting first item...")
item = dataset[0]

print(f"Number of elements in item: {len(item)}")

# Check each element individually
for i, elem in enumerate(item):
    print(f"Element {i}: shape={elem.shape}, min={elem.min().item()}, max={elem.max().item()}")

# Now unpack to see which is which
input_ids1, input_ids2, input_ids3, mask1, mask2, mask3, seg1, seg2, seg3 = item

print(f"\nAnchor (input_ids1):")
print(f"  Shape: {input_ids1.shape}")
print(f"  Min: {input_ids1.min().item()}, Max: {input_ids1.max().item()}")
print(f"  Values: {input_ids1[:20]}")

print(f"\nPositive (input_ids2):")
print(f"  Shape: {input_ids2.shape}")
print(f"  Min: {input_ids2.min().item()}, Max: {input_ids2.max().item()}")

print(f"\nNegative (input_ids3):")
print(f"  Shape: {input_ids3.shape}")
print(f"  Min: {input_ids3.min().item()}, Max: {input_ids3.max().item()}")

print(f"\nToken type IDs 1:")
print(f"  Shape: {token_type_ids1.shape}")
print(f"  Min: {token_type_ids1.min().item()}, Max: {token_type_ids1.max().item()}")
print(f"  Unique values: {torch.unique(token_type_ids1).tolist()}")

print("\n✓ Dataset access works correctly!")
