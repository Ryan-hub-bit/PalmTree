#!/usr/bin/env python3
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('/home/kun/Document/AAE/extern/jTrans/pretrain/baseline')

func_str = "endbr64 xor ebp ebp mov r9 rdx pop rsi mov"
max_length = 512

# Manually call the function
def _create_segment_labels(func_str, tokenizer, max_length):
    """Test version"""
    import torch
    segment_labels = [0] * max_length
    return torch.tensor(segment_labels, dtype=torch.long)

result = _create_segment_labels(func_str, tokenizer, max_length)
print(f"Result shape: {result.shape}")
print(f"Result min: {result.min()}, max: {result.max()}")
print(f"First 20: {result[:20].tolist()}")
print(f"Unique: {torch.unique(result).tolist()}")
