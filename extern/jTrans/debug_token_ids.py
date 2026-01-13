#!/usr/bin/env python3
"""
Debug script to check if token IDs are valid for the vocabulary.
"""

import json
from transformers import BertTokenizer

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('/home/kun/Document/AAE/extern/jTrans/pretrain/baseline')
vocab_size = len(tokenizer)
print(f"Vocab size: {vocab_size}")

print("\nLoading ground truth...")
with open('/data/kun/jtransdata/ground_truth_baseline.json', 'r') as f:
    gt = json.load(f)

print("Loading function blocks...")
with open('/data/kun/jtransdata/func_blocks_baseline.json', 'r') as f:
    fb = json.load(f)

# Check first few functions
print("\nChecking first 10 functions for token ID validity...")
pairs = gt['pairs'][:10]

for i, pair in enumerate(pairs):
    func_id = str(pair['O0'])
    if func_id in fb:
        func_data = fb[func_id]
        tokens_str = func_data['tokens']
        
        # Tokenize
        encoded = tokenizer.encode_plus(
            tokens_str,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        token_ids = encoded['input_ids'][0]
        max_id = token_ids.max().item()
        min_id = token_ids.min().item()
        
        print(f"\nFunction {i} (ID {func_id}):")
        print(f"  Token text: {tokens_str[:100]}...")
        print(f"  Token ID range: [{min_id}, {max_id}]")
        
        if max_id >= vocab_size:
            print(f"  ⚠️  ERROR: Max token ID {max_id} >= vocab_size {vocab_size}")
            # Find which tokens are out of range
            invalid_mask = token_ids >= vocab_size
            invalid_positions = invalid_mask.nonzero(as_tuple=True)[0]
            print(f"  Invalid positions: {invalid_positions.tolist()}")
            print(f"  Invalid token IDs: {token_ids[invalid_positions].tolist()}")
        else:
            print(f"  ✓ All token IDs valid")

print("\n" + "="*60)
print("Checking special tokens...")
print(f"PAD token: '{tokenizer.pad_token}' = {tokenizer.pad_token_id}")
print(f"UNK token: '{tokenizer.unk_token}' = {tokenizer.unk_token_id}")
print(f"CLS token: '{tokenizer.cls_token}' = {tokenizer.cls_token_id}")
print(f"SEP token: '{tokenizer.sep_token}' = {tokenizer.sep_token_id}")
print(f"MASK token: '{tokenizer.mask_token}' = {tokenizer.mask_token_id}")
print("="*60)
