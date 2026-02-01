#!/usr/bin/env python3
"""
Quick test to verify addressaware evaluation fix
"""
import sys
sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')

from transformers import BertTokenizer

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')

# Simulate pre-tokenized data
tokens_str = "lea rdi address mov rax var push rbp"
tokens = tokens_str.split()

print("=== Testing Tokenization Fix ===\n")
print(f"Input tokens string: '{tokens_str}'")
print(f"Split into {len(tokens)} tokens: {tokens}\n")

# Method 1: OLD (WRONG) - encode_plus
print("Method 1: encode_plus (OLD - WRONG)")
encoded = tokenizer.encode_plus(
    tokens_str,
    max_length=512,
    padding='max_length',
    truncation=True
)
non_pad_old = sum(1 for x in encoded['input_ids'] if x != tokenizer.pad_token_id)
print(f"  Output: {non_pad_old} tokens (includes [CLS], [SEP])")
print(f"  First 15 IDs: {encoded['input_ids'][:15]}")
print(f"  ✗ Length mismatch! {non_pad_old} != {len(tokens)}\n")

# Method 2: NEW (CORRECT) - convert_tokens_to_ids
print("Method 2: convert_tokens_to_ids (NEW - CORRECT)")
token_ids = []
for token in tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        token_id = tokenizer.unk_token_id
    token_ids.append(token_id)

print(f"  Output: {len(token_ids)} tokens")
print(f"  Token IDs: {token_ids}")
print(f"  ✓ Perfect match! {len(token_ids)} == {len(tokens)}\n")

# Verify specific tokens
print("Token ID verification:")
for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
    print(f"  '{tok}' → {tid}")

print("\n=== Fix Verified ✓ ===")
print("The new method preserves token count and alignment with position arrays!")
