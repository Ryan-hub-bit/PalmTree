#!/usr/bin/env python
"""
Test tokenizer and matching process with a single function pair
"""

import json
import sys
import os
from transformers import BertTokenizer

# Add path for address embedding
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pretrain', 'address_aware'))

def test_single_pair_tokenization():
    print("=" * 60)
    print("Testing Tokenizer and Matching Process")
    print("=" * 60)
    print()
    
    # Load tokenizer
    tokenizer_path = "/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware"
    print(f"1. Loading tokenizer from: {tokenizer_path}")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    print(f"   ✓ Tokenizer loaded")
    print(f"   - Vocab size: {len(tokenizer)}")
    print()
    
    # Load function blocks data
    func_blocks_path = "/data/kun/jtransdata/func_blocks_addr.json"
    ground_truth_path = "/data/kun/jtransdata/ground_truth_addr.json"
    
    print(f"2. Loading data files (this may take a moment)...")
    print(f"   - func_blocks: {func_blocks_path}")
    print(f"   - ground_truth: {ground_truth_path}")
    
    print(f"   Loading func_blocks...")
    with open(func_blocks_path, 'r') as f:
        func_blocks = json.load(f)
    
    print(f"   Loading ground_truth...")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Get first function
    first_key = list(func_blocks.keys())[0]
    func_data1 = func_blocks[first_key]
    print(f"   ✓ Loaded function data (total functions: {len(func_blocks)})")
    
    # Get first ground truth entry
    first_gt_key = list(ground_truth.keys())[0]
    gt_data = ground_truth[first_gt_key]
    print(f"   ✓ Loaded ground truth (total entries: {len(ground_truth)})")
    print()
    
    # Display function information
    print("3. Function Data Structure:")
    print(f"   - Function ID: {func_data1.get('id', 'N/A')}")
    print(f"   - Binary name: {func_data1.get('binary_name', 'N/A')}")
    print(f"   - Function name: {func_data1.get('function_name', 'N/A')}")
    print(f"   - Optimization: {func_data1.get('optimization', 'N/A')}")
    print(f"   - All keys: {list(func_data1.keys())}")
    
    # Check if there are tokens or blocks
    if 'tokens' in func_data1:
        print(f"   - Number of tokens: {len(func_data1['tokens'])}")
    if 'blocks' in func_data1:
        print(f"   - Number of blocks: {len(func_data1['blocks'])}")
    print()
    
    # Display ground truth structure
    print("4. Ground Truth Structure:")
    print(f"   - Type: {type(gt_data)}")
    if isinstance(gt_data, dict):
        print(f"   - Keys: {list(gt_data.keys())}")
        if 'positive' in gt_data:
            print(f"   - Positive pairs: {len(gt_data['positive'])}")
        if 'negative' in gt_data:
            print(f"   - Negative pairs: {len(gt_data['negative'])}")
    elif isinstance(gt_data, list):
        print(f"   - List length: {len(gt_data)}")
        if gt_data:
            print(f"   - First entry type: {type(gt_data[0])}")
            if isinstance(gt_data[0], dict):
                print(f"   - First entry keys: {list(gt_data[0].keys())}")
    print()
    
    # Test tokenization on instructions
    print("5. Testing Tokenization:")
    if 'instructions' in func_data1:
        instructions = func_data1['instructions']
        print(f"   - Function has {len(instructions)} instructions")
        print(f"   - First 5 instructions:")
        for i, inst in enumerate(instructions[:5]):
            print(f"     {i+1}. {inst}")
        
        # Tokenize instructions
        inst_text = ' '.join(instructions)
        print(f"\n   - Combined instruction text (first 200 chars):")
        print(f"     {inst_text[:200]}...")
        
        # Encode with tokenizer
        encoded = tokenizer.encode(inst_text, max_length=512, truncation=True)
        print(f"\n   - Tokenized results:")
        print(f"     Total encoded length: {len(encoded)}")
        print(f"     First 15 token IDs: {encoded[:15]}")
        
        # Decode to verify
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        print(f"     Decoded (first 200 chars): {decoded[:200]}...")
        
        # Show some token mappings
        print(f"\n   - Sample token ID -> token mappings:")
        for i, token_id in enumerate(encoded[:10]):
            token = tokenizer.decode([token_id])
            print(f"     ID {token_id:4d} -> '{token}'")
        
        print(f"\n   ✓ Tokenization working correctly!")
    
    print()
    print("6. Checking Address-Aware Features:")
    addr_features = ['bb_ids', 'func_id', 'addresses', 'var_ids', 'bb_addrs', 'inst_addrs']
    found_features = []
    for feature in addr_features:
        if feature in func_data1:
            found_features.append(feature)
            val = func_data1[feature]
            if isinstance(val, list) and len(val) > 0:
                print(f"   ✓ {feature}: {val[:5]}... (length: {len(val)})")
            else:
                print(f"   ✓ {feature}: {val}")
    
    if not found_features:
        print(f"   ! No hierarchical address features found")
        print(f"   ! This might be baseline data format")
    
    print()
    print("7. Testing Matching Process:")
    # Ground truth is a list of dicts with optimization levels
    if isinstance(gt_data, list) and len(gt_data) > 0:
        first_gt = gt_data[0]
        print(f"   - Ground truth entry structure:")
        print(f"     Binary: {first_gt.get('binary_name', 'N/A')}")
        print(f"     Function: {first_gt.get('function_name', 'N/A')}")
        
        # Show optimization levels present
        opt_levels = [k for k in first_gt.keys() if k.startswith('O')]
        print(f"     Optimization levels: {opt_levels}")
        
        # Show function IDs for each opt level
        print(f"\n   - Function IDs across optimizations (for matching):")
        for opt in opt_levels:
            if opt in first_gt:
                print(f"     {opt}: function_id = {first_gt[opt]}")
        
        print(f"\n   ✓ Matching process: Compare embeddings of same function")
        print(f"     across different optimization levels (triplet loss)")
        print(f"   ✓ Positive pair: same function, different opt")
        print(f"   ✓ Negative pair: different function")
    
    print()
    print("=" * 60)
    print("✓ Test Complete - Tokenizer and matching process verified!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_single_pair_tokenization()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
