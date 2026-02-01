#!/usr/bin/env python3
"""
Diagnose addressaware evaluation issues by comparing:
1. Pretrain data format
2. Finetune data format  
3. Evaluation data format
"""

import json
import torch
from transformers import BertTokenizer
from pathlib import Path

print("="*80)
print("DIAGNOSING ADDRESSAWARE EVALUATION")
print("="*80)

# Load tokenizer
vocab_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware'
tokenizer = BertTokenizer.from_pretrained(vocab_path)
print(f"\n1. Tokenizer Info:")
print(f"   Vocab size: {len(tokenizer)}")
print(f"   Pad: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

# Load a sample function from eval data
func_blocks_path = '/data/kun/jtrans/addressaware/eval/func_blocks_addr.json'
print(f"\n2. Loading sample from: {func_blocks_path}")
print("   (This may take a moment...)")

# Use a smaller sample by reading only first few lines
import ijson
sample_func = None
with open(func_blocks_path, 'rb') as f:
    parser = ijson.items(f, 'item')
    for idx, (func_id, func_data) in enumerate(parser):
        if idx == 0:
            sample_func = func_data
            sample_id = func_id
            break

if sample_func:
    print(f"\n3. Sample Function (ID: {sample_id}):")
    print(f"   Fields: {list(sample_func.keys())}")
    print(f"   'tokens' length: {len(sample_func.get('tokens', ''))}")
    print(f"   'binary_pos' length: {len(sample_func.get('binary_pos', []))}")
    print(f"   'function_pos' length: {len(sample_func.get('function_pos', []))}")
    
    # Check if 'tokens' is pre-tokenized
    tokens_str = sample_func.get('tokens', '')
    tokens_list = tokens_str.split()
    print(f"\n4. Tokens Format Analysis:")
    print(f"   First 100 chars: {tokens_str[:100]}")
    print(f"   First 10 tokens after split: {tokens_list[:10]}")
    print(f"   Number of space-separated tokens: {len(tokens_list)}")
    
    # Key question: Does 'tokens' match the position arrays?
    pos_len = len(sample_func.get('binary_pos', []))
    tok_len = len(tokens_list)
    print(f"\n5. Length Matching:")
    print(f"   Tokens (space-split): {tok_len}")
    print(f"   Position arrays: {pos_len}")
    print(f"   Match: {tok_len == pos_len}")
    
    # Try tokenizing with BertTokenizer
    print(f"\n6. BertTokenizer Encoding:")
    print(f"   Input: '{tokens_str[:50]}...'")
    encoded = tokenizer.encode_plus(
        tokens_str,
        max_length=512,
        padding='max_length',
        truncation=True
    )
    non_pad = sum(1 for x in encoded['input_ids'] if x != tokenizer.pad_token_id)
    print(f"   Output token IDs (first 20): {encoded['input_ids'][:20]}")
    print(f"   Non-padding tokens: {non_pad}")
    
    # The problem: BertTokenizer will RETOKENIZE the string!
    print(f"\n7. THE PROBLEM:")
    print(f"   'tokens' field has {tok_len} space-separated tokens")
    print(f"   BertTokenizer produces {non_pad} tokens (with [CLS], [SEP])")
    print(f"   Position arrays have {pos_len} entries")
    print(f"   ")
    if tok_len == pos_len:
        print(f"   ✓ 'tokens' is PRE-TOKENIZED and matches position arrays")
        print(f"   ✗ BUT BertTokenizer.encode_plus() will RE-TOKENIZE it!")
        print(f"   ✗ This breaks the alignment with position arrays!")
        print(f"\n   SOLUTION: Don't use encode_plus()!")
        print(f"   Instead: tokenizer.convert_tokens_to_ids(tokens_list)")
    else:
        print(f"   ✗ Mismatch between tokens and positions!")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
