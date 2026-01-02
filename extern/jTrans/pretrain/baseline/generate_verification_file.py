#!/usr/bin/env python3
"""
Generate comprehensive token verification files
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

# Write comprehensive output
output_file = 'token_verification_complete.txt'

with open(output_file, 'w') as f:
    f.write("="*100 + "\n")
    f.write("COMPLETE TOKEN VERIFICATION FOR DATALOADER\n")
    f.write("="*100 + "\n\n")
    
    f.write("FILE: /tmp/test_recursive.txt (first function)\n")
    f.write(f"Original token count: {len(tokens_raw)}\n")
    f.write(f"After truncation + special tokens: {len(tokens_with_special)}\n\n")
    
    f.write("="*100 + "\n")
    f.write("COMPLETE TOKEN TABLE\n")
    f.write("="*100 + "\n")
    f.write(f"{'Idx':<5} {'Original Token':<25} {'Orig ID':<8} {'Vocab[ID]':<25} {'Masked ID':<10} {'Masked Token':<25} {'MLM Label':<10} {'JTP Label':<10}\n")
    f.write("-"*100 + "\n")
    
    for i in range(len(tokens_with_special)):
        orig_token = tokens_with_special[i]
        orig_id = input_ids_original[i]
        vocab_token = vocab[orig_id] if orig_id < len(vocab) else "OUT_OF_RANGE"
        masked_id = input_ids_masked[i]
        masked_token = vocab[masked_id] if masked_id < len(vocab) else "OUT_OF_RANGE"
        mlm_label = mlm_labels[i]
        jtp_label = jtp_labels[i]
        
        # Format labels
        mlm_str = str(mlm_label) if mlm_label != -100 else "-"
        jtp_str = str(jtp_label) if jtp_label != -100 else "-"
        
        # Mark masked tokens
        marker = ""
        if mlm_label != -100:
            marker = " [MLM]"
        if jtp_label != -100:
            marker = " [JTP]"
        
        f.write(f"{i:<5} {orig_token:<25} {orig_id:<8} {vocab_token:<25} {masked_id:<10} {masked_token:<25} {mlm_str:<10} {jtp_str:<10}{marker}\n")
    
    # Add summary section
    f.write("\n" + "="*100 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("="*100 + "\n")
    
    mlm_masked = [i for i, label in enumerate(mlm_labels) if label != -100]
    jtp_masked = [i for i, label in enumerate(jtp_labels) if label != -100]
    
    f.write(f"Total tokens: {len(tokens_with_special)}\n")
    f.write(f"MLM masked tokens: {len(mlm_masked)} ({100*len(mlm_masked)/len(tokens_with_special):.2f}%)\n")
    f.write(f"JTP masked tokens: {len(jtp_masked)}\n")
    f.write(f"\n")
    
    f.write("MLM Masked Positions:\n")
    for idx in mlm_masked:
        f.write(f"  Position {idx}: {tokens_with_special[idx]} (ID {input_ids_original[idx]}) -> {vocab[input_ids_masked[idx]]} (ID {input_ids_masked[idx]}), Label: {vocab[mlm_labels[idx]]}\n")
    
    f.write("\nJTP Masked Positions:\n")
    for idx in jtp_masked:
        target_pos = jtp_labels[idx]
        target_token = tokens_with_special[target_pos] if target_pos < len(tokens_with_special) else "OUT_OF_RANGE"
        f.write(f"  Position {idx}: {tokens_with_special[idx]} (ID {input_ids_original[idx]}) -> {vocab[input_ids_masked[idx]]} (ID {input_ids_masked[idx]}), Target Position: {target_pos} ({target_token})\n")
    
    # Add vocab mapping section
    f.write("\n" + "="*100 + "\n")
    f.write("SPECIAL TOKEN IDs\n")
    f.write("="*100 + "\n")
    f.write(f"[PAD]  = {tokenizer.pad_token_id} (vocab[{tokenizer.pad_token_id}] = {vocab[tokenizer.pad_token_id]})\n")
    f.write(f"[CLS]  = {tokenizer.cls_token_id} (vocab[{tokenizer.cls_token_id}] = {vocab[tokenizer.cls_token_id]})\n")
    f.write(f"[SEP]  = {tokenizer.sep_token_id} (vocab[{tokenizer.sep_token_id}] = {vocab[tokenizer.sep_token_id]})\n")
    f.write(f"[MASK] = {tokenizer.mask_token_id} (vocab[{tokenizer.mask_token_id}] = {vocab[tokenizer.mask_token_id]})\n")
    f.write(f"[UNK]  = {tokenizer.unk_token_id} (vocab[{tokenizer.unk_token_id}] = {vocab[tokenizer.unk_token_id]})\n")
    
    # Add JUMP_ADDR tokens section
    f.write("\n" + "="*100 + "\n")
    f.write("JUMP_ADDR TOKENS IN THIS FUNCTION\n")
    f.write("="*100 + "\n")
    import re
    jump_pattern = re.compile(r'JUMP_ADDR_(\d+)')
    for i, token in enumerate(tokens_with_special):
        if jump_pattern.match(token):
            orig_id = input_ids_original[i]
            target = int(token.split('_')[2])
            target_token = tokens_with_special[target] if target < len(tokens_with_special) else "OUT_OF_RANGE"
            f.write(f"Position {i}: {token} (ID {orig_id}) -> jumps to position {target} ({target_token})\n")

print(f"✓ Complete verification written to: {output_file}")
print(f"  File size: {os.path.getsize(output_file)} bytes")
print()
print("This file contains:")
print("  - Complete token-by-token breakdown")
print("  - Original tokens and their IDs")
print("  - Masked tokens and their IDs")
print("  - MLM and JTP labels")
print("  - Verification against vocab.txt")
print()
print(f"View with: cat {output_file}")
print(f"Or open in editor to check each token")
