#!/usr/bin/env python3
"""
Test the extended vocabulary with address tokens
"""

import sys
import os
import pickle

# Check if extended vocab exists
vocab_path = "vocab_extended"

if not os.path.exists(vocab_path):
    print(f"Error: Extended vocabulary not found: {vocab_path}")
    print("Run: python extend_vocab.py ../pre-trained_model/palmtree/vocab")
    sys.exit(1)

# Load vocab
print("Loading extended vocabulary...")
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

print(f"✓ Vocabulary loaded: {len(vocab)} tokens")
print(f"  Type: {type(vocab).__name__}")

# Test address tokens
print("\n" + "="*70)
print("ADDRESS TOKENS TEST")
print("="*70)

addr_tokens = ['addr_start', 'addr_end', 'addr_code', 'addr_data']
all_found = True

for token in addr_tokens:
    if token in vocab.stoi:
        token_id = vocab.stoi[token]
        # Verify reverse lookup
        reverse_token = vocab.itos[token_id]
        status = "✓" if reverse_token == token else "✗"
        print(f"{status} {token:12s} -> ID: {token_id:5d} -> reverse: {reverse_token}")
    else:
        print(f"✗ {token:12s} -> NOT FOUND")
        all_found = False

if all_found:
    print("\n✓ All address tokens found and verified!")
else:
    print("\n✗ Some address tokens missing!")
    sys.exit(1)

# Test with sample BB text
print("\n" + "="*70)
print("TOKENIZATION TEST")
print("="*70)

sample_bb = "<addr_start:0x4020f0:0.267785:0.000000> mov rax [ rel <addr_data:0x405150:0.661077:-3.000000> ] <addr_end:0x402103:0.268388:0.558824>"

print(f"Sample BB:\n{sample_bb}\n")

# Simple tokenization (split by spaces)
tokens = sample_bb.split()
print("Tokens and their IDs:")

for token in tokens:
    # Try exact match first
    if token in vocab.stoi:
        token_id = vocab.stoi[token]
        print(f"  {token:30s} -> {token_id}")
    else:
        # Check if it's an address tag
        if token.startswith('<addr_'):
            addr_type = token.split(':')[0][1:]  # Extract addr_start, addr_data, etc.
            if addr_type in vocab.stoi:
                token_id = vocab.stoi[addr_type]
                print(f"  {token:30s} -> {addr_type:12s} -> {token_id}")
            else:
                token_id = vocab.stoi.get('<unk>', 1)
                print(f"  {token:30s} -> <unk> -> {token_id}")
        else:
            token_id = vocab.stoi.get(token, vocab.stoi.get('<unk>', 1))
            if token not in vocab.stoi:
                print(f"  {token:30s} -> <unk> -> {token_id}")
            else:
                print(f"  {token:30s} -> {token_id}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Extended vocabulary ready with {len(vocab)} tokens")
print(f"✓ Original PalmTree tokens: {len(vocab) - 4}")
print(f"✓ New address tokens: 4")
print(f"✓ vocab_extended file can be used with data_loader.py")
print()
print("Next steps:")
print("  1. Update config.py to use 'vocab_extended' as VOCAB_FILE")
print("  2. Run test_dataloader.py to verify everything works")
print("  3. Start training with address-focused tasks!")
