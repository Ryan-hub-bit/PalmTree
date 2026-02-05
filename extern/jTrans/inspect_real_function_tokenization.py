#!/usr/bin/env python3
"""
Inspect a real function's tokenization to see all embedding layer information.
"""

import json
import sys
import os
from transformers import BertTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data_json import FunctionDataset_CL_AddressAware_JSON

def main():
    # Load tokenizer
    vocab_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware'
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    
    # Load function blocks
    func_blocks_path = '/data/kun/jtrans/addressaware/eval/func_blocks_addr.json'
    with open(func_blocks_path, 'r') as f:
        func_blocks = json.load(f)
    
    print(f"Total functions available: {len(func_blocks)}")
    print()
    
    # Find a function with interesting features (addresses, multiple instructions, vars)
    print("Searching for a complex function with addresses, vars, and multiple instructions...")
    
    interesting_func = None
    for func_id, func_block in list(func_blocks.items())[:1000]:  # Check first 1000
        tokens_str = func_block['tokens']
        
        # Look for functions with:
        # 1. Multiple instructions (tabs)
        # 2. Address information (contains address() or daddr())
        # 3. Variable information (contains var())
        has_tabs = '\t' in tokens_str
        has_address = 'address(' in tokens_str or 'daddr(' in tokens_str
        has_var = 'var(' in tokens_str
        num_instructions = tokens_str.count('\t') + 1 if has_tabs else 1
        
        if has_tabs and has_address and num_instructions >= 3:
            interesting_func = (func_id, func_block)
            print(f"Found function ID: {func_id}")
            print(f"  Instructions: {num_instructions}")
            print(f"  Has addresses: {has_address}")
            print(f"  Has vars: {has_var}")
            print(f"  Total length: {len(tokens_str)} chars")
            break
    
    if not interesting_func:
        print("Using first function as fallback...")
        func_id = list(func_blocks.keys())[0]
        interesting_func = (func_id, func_blocks[func_id])
    
    func_id, func_block = interesting_func
    func_str = func_block['tokens']
    
    print()
    print("=" * 100)
    print("FUNCTION RAW DATA:")
    print("=" * 100)
    print(f"ID: {func_id}")
    print(f"Length: {len(func_str)} chars")
    print()
    
    # Show first 500 chars with instruction boundaries marked
    instructions = func_str.split('\t')
    print(f"Number of instructions: {len(instructions)}")
    print()
    print("First 3 instructions:")
    for i, inst in enumerate(instructions[:3]):
        print(f"  Instruction {i+1}: {inst[:150]}{'...' if len(inst) > 150 else ''}")
    print()
    
    # Create dataset to tokenize
    import tempfile
    temp_gt = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_gt.write(json.dumps({
        'pairs': [{
            'binary': 'test',
            'func_name': 'test_func',
            'opt1': 'O0',
            'opt2': 'O3',
            'func_id1': func_id,
            'func_id2': func_id
        }]
    }))
    temp_gt.close()
    
    try:
        # Create dataset
        print("Tokenizing with FunctionDataset_CL_AddressAware_JSON...")
        dataset = FunctionDataset_CL_AddressAware_JSON(
            tokenizer,
            func_blocks_path,
            temp_gt.name,
            opt=['O0', 'O3'],
            add_ebd=False,
            max_length=512,
            data_ratio=1.0
        )
        
        # Get processed data
        processed = dataset.processed_datas[0][0]
        
        # Convert to lists
        token_ids = processed['input_ids'].tolist()
        segments = processed['token_type_ids'].tolist()
        binary_pos = processed['binary_pos'].tolist()
        function_pos = processed['function_pos'].tolist()
        bb_pos = processed['bb_pos'].tolist()
        var_offsets = processed['var_offsets'].tolist()
        attention_mask = processed['attention_mask'].tolist()
        
        print()
        print("=" * 100)
        print("TOKENIZATION RESULT (First 30 tokens):")
        print("=" * 100)
        print()
        
        # Build reverse vocab for token names
        vocab_file = os.path.join(vocab_path, 'vocab.txt')
        vocab_itos = []
        with open(vocab_file, 'r') as f:
            for line in f:
                vocab_itos.append(line.strip())
        
        # Print header
        print(f"{'Pos':<4} {'Token':<20} {'ID':<6} {'Seg':<4} {'BinPos':<8} {'FuncPos':<8} {'BBPos':<8} {'VarOff':<10} {'Mask':<4}")
        print("-" * 100)
        
        # Print first 30 tokens
        for i in range(min(30, len(token_ids))):
            if token_ids[i] >= len(vocab_itos):
                token_name = f"<OOV:{token_ids[i]}>"
            else:
                token_name = vocab_itos[token_ids[i]]
            
            print(f"{i:<4} {token_name:<20} {token_ids[i]:<6} {segments[i]:<4} "
                  f"{binary_pos[i]:<8.4f} {function_pos[i]:<8.4f} {bb_pos[i]:<8.4f} "
                  f"{var_offsets[i]:<10} {attention_mask[i]:<4}")
        
        print()
        print("=" * 100)
        print("SUMMARY STATISTICS:")
        print("=" * 100)
        
        # Find non-padding tokens
        non_padding = [i for i, m in enumerate(attention_mask) if m == 1]
        print(f"Total tokens: {len(token_ids)}")
        print(f"Non-padding tokens: {len(non_padding)}")
        print()
        
        # Count tokens with position information
        tokens_with_binpos = sum(1 for p in binary_pos if p != -1.0)
        tokens_with_funcpos = sum(1 for p in function_pos if p != -1.0)
        tokens_with_bbpos = sum(1 for p in bb_pos if p != -1.0)
        tokens_with_var = sum(1 for v in var_offsets if v != -1)
        
        print(f"Tokens with binary position: {tokens_with_binpos}")
        print(f"Tokens with function position: {tokens_with_funcpos}")
        print(f"Tokens with BB position: {tokens_with_bbpos}")
        print(f"Tokens with var offsets: {tokens_with_var}")
        print()
        
        # Show unique segment values
        unique_segments = sorted(set(s for s in segments if s > 0))
        print(f"Unique segment values (non-padding): {unique_segments[:20]}")
        print(f"Number of unique segments: {len(unique_segments)}")
        print()
        
        # Show example tokens with actual position information
        print("=" * 100)
        print("TOKENS WITH POSITION INFORMATION:")
        print("=" * 100)
        
        tokens_with_pos_info = []
        for i in range(len(token_ids)):
            if binary_pos[i] != -1.0 or function_pos[i] != -1.0 or bb_pos[i] != -1.0 or var_offsets[i] != -1:
                token_name = vocab_itos[token_ids[i]] if token_ids[i] < len(vocab_itos) else f"<OOV:{token_ids[i]}>"
                tokens_with_pos_info.append((i, token_name, binary_pos[i], function_pos[i], bb_pos[i], var_offsets[i]))
        
        if tokens_with_pos_info:
            print(f"Found {len(tokens_with_pos_info)} tokens with position information")
            print()
            print(f"{'Pos':<4} {'Token':<20} {'BinPos':<8} {'FuncPos':<8} {'BBPos':<8} {'VarOff':<10}")
            print("-" * 80)
            for pos, token, binp, funcp, bbp, varo in tokens_with_pos_info[:20]:
                print(f"{pos:<4} {token:<20} {binp:<8.4f} {funcp:<8.4f} {bbp:<8.4f} {varo:<10}")
        else:
            print("⚠️  NO TOKENS WITH POSITION INFORMATION FOUND!")
            print("This might indicate the function doesn't have address-aware annotations.")
        
    finally:
        os.unlink(temp_gt.name)

if __name__ == '__main__':
    main()
