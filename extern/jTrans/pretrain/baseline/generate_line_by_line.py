#!/usr/bin/env python3
"""
Generate line-by-line token verification in PalmTree test format
"""

import sys
import os
import torch
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import BertTokenizer
from dataloader_baseline import BaselinePretrainingDataset

# Set seed
random.seed(42)
torch.manual_seed(42)

# Load tokenizer
tokenizer_path = os.path.dirname(os.path.abspath(__file__))
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Load vocab
with open(os.path.join(tokenizer_path, 'vocab.txt'), 'r') as f:
    vocab = [line.strip() for line in f]

# Test data
test_data = "/tmp/test_recursive.txt"

# Get first line
with open(test_data, 'r') as f:
    first_line = f.readline().strip()

tokens_raw = first_line.split()

# Truncate and add special tokens
max_len = 512
tokens_truncated = tokens_raw[:max_len - 2]
tokens_with_special = ['[CLS]'] + tokens_truncated + ['[SEP]']

# Convert to IDs
input_ids_original = tokenizer.convert_tokens_to_ids(tokens_with_special)

# Create dataset and get sample
dataset = BaselinePretrainingDataset(
    data_path=test_data,
    tokenizer=tokenizer,
    max_len=512,
    mlm_probability=0.15,
    jtp_probability=0.20,
    on_memory=True
)

random.seed(42)
torch.manual_seed(42)
sample = dataset[0]

input_ids_masked = sample['input_ids'].tolist()
mlm_labels = sample['mlm_labels'].tolist()
jtp_labels = sample['jtp_labels'].tolist()

# Generate output file
output_file = 'token_line_by_line.txt'

with open(output_file, 'w') as f:
    f.write("="*120 + "\n")
    f.write("TOKEN PROCESSING LINE BY LINE (PalmTree Test Format)\n")
    f.write("="*120 + "\n\n")
    
    f.write("Dataset: /tmp/test_recursive.txt (first function)\n")
    f.write(f"Total tokens after truncation: {len(tokens_with_special)}\n\n")
    
    # Process each token
    for idx in range(len(tokens_with_special)):
        f.write(f"Position {idx}\n")
        f.write("-" * 120 + "\n")
        
        # Original token
        orig_token = tokens_with_special[idx]
        orig_id = input_ids_original[idx]
        vocab_check = vocab[orig_id] if orig_id < len(vocab) else "ERROR"
        
        f.write(f"  Original Token:  {orig_token}\n")
        f.write(f"  Token ID:        {orig_id}\n")
        f.write(f"  Vocab[{orig_id}]:      {vocab_check}\n")
        
        # Check if vocab matches
        if vocab_check == orig_token:
            f.write(f"  Vocab Match:     ✓ CORRECT\n")
        else:
            f.write(f"  Vocab Match:     ✗ MISMATCH!\n")
        
        # After masking
        masked_id = input_ids_masked[idx]
        masked_token = vocab[masked_id] if masked_id < len(vocab) else "ERROR"
        
        f.write(f"  After Masking:   {masked_token} (ID {masked_id})\n")
        
        # Check if masked
        is_mlm = mlm_labels[idx] != -100
        is_jtp = jtp_labels[idx] != -100
        
        if is_mlm:
            label_token = vocab[mlm_labels[idx]]
            f.write(f"  MLM:             MASKED (label: {label_token}, ID {mlm_labels[idx]})\n")
            if orig_id != masked_id:
                if masked_id == 4:
                    f.write(f"                   → Replaced with [MASK]\n")
                else:
                    f.write(f"                   → Replaced with random token: {masked_token}\n")
            else:
                f.write(f"                   → Kept original (10% strategy)\n")
        
        if is_jtp:
            target_pos = jtp_labels[idx]
            target_token = tokens_with_special[target_pos] if target_pos < len(tokens_with_special) else "OUT_OF_RANGE"
            f.write(f"  JTP:             MASKED (target position: {target_pos})\n")
            f.write(f"                   → Jump target: {target_token} at position {target_pos}\n")
            if masked_id == 4:
                f.write(f"                   → Replaced with [MASK]\n")
            elif orig_id != masked_id:
                f.write(f"                   → Replaced with random token: {masked_token}\n")
            else:
                f.write(f"                   → Kept original\n")
        
        if not is_mlm and not is_jtp:
            f.write(f"  Status:          NOT MASKED\n")
        
        f.write("\n")
        
        # Stop after first 50 for preview, but write full file
        if idx >= 49 and idx < 50:
            f.write("... (output continues for all 512 tokens) ...\n\n")

print(f"✓ Line-by-line verification written to: {output_file}")
print(f"  Total lines: ~{len(tokens_with_special) * 10}")
print()
print("File format:")
print("  - Each token on separate lines")
print("  - Shows original token and ID")
print("  - Verifies against vocab.txt")
print("  - Shows masked token and ID")
print("  - Indicates MLM/JTP status")
print()
print(f"View with: less {output_file}")
print(f"Or: head -100 {output_file}")
