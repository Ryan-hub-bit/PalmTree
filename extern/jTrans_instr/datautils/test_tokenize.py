#!/usr/bin/env python3
import pickle
import sys
import os
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import readidadata

# Import tokenization functions from create_baseline_dataset
exec(open('create_baseline_dataset.py').read().split('def create_function_blocks_baseline')[0])

# Test on one pkl file
pkl_file = '/data/kun/jtransdata/extract/libnfc-nfc-mfultralight/libnfc-nfc-mfultralight-O0-49324b20dc16d7ad6180ee796e60b134_extract.pkl'

with open(pkl_file, 'rb') as pf:
    data = pickle.load(pf)

# Find a function with jumps
for func_name, func_data in list(data.items()):
    asm_list = func_data.get('asm', [])
    if len(asm_list) > 20:
        tokens = tokenize_function_baseline(func_data)
        
        if tokens:
            print(f"Function: {func_name}")
            print(f"Tokens preview: {tokens[:250]}...")
            
            # Count token types
            jump_addr_count = len(re.findall(r'JUMP_ADDR_\d+', tokens))
            unk_jump_count = tokens.count('UNK_JUMP_ADDR')
            exceeded_count = tokens.count('JUMP_ADDR_EXCEEDED')
            
            print(f"\nJump token stats:")
            print(f"  JUMP_ADDR_X: {jump_addr_count}")
            print(f"  UNK_JUMP_ADDR: {unk_jump_count}")
            print(f"  JUMP_ADDR_EXCEEDED: {exceeded_count}")
        break
