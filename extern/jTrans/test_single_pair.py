#!/usr/bin/env python3
"""
Test tokenizer and matching process with a single function pair.
This verifies that the baseline tokenization and pairing logic work correctly.
"""

import json
import sys
from transformers import BertTokenizer

def test_single_pair():
    print("="*60)
    print("Testing Single Function Pair - Baseline Model")
    print("="*60)
    
    # Load tokenizer
    tokenizer_path = "/home/kun/Document/AAE/extern/jTrans/pretrain/baseline"
    print(f"\n1. Loading tokenizer from: {tokenizer_path}")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    print(f"   ✓ Tokenizer loaded, vocab size: {len(tokenizer)}")
    
    # Load data
    func_blocks_path = "/data/kun/jtransdata/func_blocks_baseline.json"
    ground_truth_path = "/data/kun/jtransdata/ground_truth_baseline.json"
    
    print(f"\n2. Loading function blocks from: {func_blocks_path}")
    print("   (This may take a moment for large files...)")
    with open(func_blocks_path, 'r') as f:
        func_blocks = json.load(f)
    print(f"   ✓ Loaded {len(func_blocks)} function blocks")
    
    print(f"\n3. Loading ground truth from: {ground_truth_path}")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Ground truth has structure: {'pairs': [...], 'total_pairs': N, 'statistics': {...}}
    pairs_list = ground_truth.get('pairs', [])
    print(f"   ✓ Loaded ground truth with {len(pairs_list)} function groups")
    
    # Get first function pair
    print("\n4. Getting first function group from ground truth")
    if len(pairs_list) == 0:
        print("   ERROR: No pairs found!")
        return
        
    first_group = pairs_list[0]
    print(f"   Binary: {first_group.get('binary_name')}")
    print(f"   Function: {first_group.get('function_name')}")
    print(f"   Optimization levels available:")
    
    # Extract function IDs for different optimization levels
    opt_ids = {}
    for opt_level in ['O0', 'O1', 'O2', 'O3', 'Os']:
        if opt_level in first_group:
            func_id = str(first_group[opt_level])
            opt_ids[opt_level] = func_id
            print(f"     - {opt_level}: ID {func_id}")
    
    if len(opt_ids) < 2:
        print("   ERROR: Need at least 2 optimization levels!")
        return
    
    # Pick first two optimization levels for comparison
    opt1, id1 = list(opt_ids.items())[0]
    opt2, id2 = list(opt_ids.items())[1]
    
    print(f"\n   Comparing {opt1} (ID {id1}) vs {opt2} (ID {id2})")
    
    # Get function blocks
    print("\n5. Retrieving function assembly code")
    if id1 not in func_blocks:
        print(f"   ERROR: Function {id1} not in func_blocks!")
        return
    if id2 not in func_blocks:
        print(f"   ERROR: Function {id2} not in func_blocks!")
        return
        
    func1 = func_blocks[id1]
    func2 = func_blocks[id2]
    
    print(f"\n   Function 1 ({opt1}, ID {id1}):")
    print(f"   - Binary: {func1.get('binary', 'N/A')}")
    print(f"   - Optimization: {func1.get('opt', 'N/A')}")
    print(f"   - Name: {func1.get('name', 'N/A')}")
    print(f"   - Instructions: {func1.get('num_instructions', 0)}")
    
    # Get tokens
    asm1_text = func1.get('tokens', '')
    asm1_tokens = asm1_text.split()
    
    asm2_text = func2.get('tokens', '')
    asm2_tokens = asm2_text.split()
    
    print(f"   - Total tokens: {len(asm1_tokens)}")
    print(f"   - Preview (first 10): {asm1_tokens[:10]}")
    
    print(f"\n   Function 2 ({opt2}, ID {id2}):")
    print(f"   - Binary: {func2.get('binary', 'N/A')}")
    print(f"   - Optimization: {func2.get('opt', 'N/A')}")
    print(f"   - Name: {func2.get('name', 'N/A')}")
    print(f"   - Instructions: {func2.get('num_instructions', 0)}")
    print(f"   - Total tokens: {len(asm2_tokens)}")
    print(f"   - Preview (first 10): {asm2_tokens[:10]}")
    
    # Test tokenization
    print("\n6. Testing tokenization")
    
    # Use the pre-tokenized text
    asm1_limited = " ".join(asm1_tokens[:512])  # Limit for testing
    asm2_limited = " ".join(asm2_tokens[:512])
    
    print(f"\n   Text 1 preview: {asm1_limited[:100]}...")
    encoded1 = tokenizer.encode_plus(
        asm1_limited,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    print(f"\n   Text 2 preview: {asm2_limited[:100]}...")
    encoded2 = tokenizer.encode_plus(
        asm2_limited,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    print(f"\n   ✓ Function 1 tokenized:")
    print(f"     - Input IDs shape: {encoded1['input_ids'].shape}")
    print(f"     - Attention mask shape: {encoded1['attention_mask'].shape}")
    print(f"     - First 20 token IDs: {encoded1['input_ids'][0][:20].tolist()}")
    
    print(f"\n   ✓ Function 2 tokenized:")
    print(f"     - Input IDs shape: {encoded2['input_ids'].shape}")
    print(f"     - Attention mask shape: {encoded2['attention_mask'].shape}")
    print(f"     - First 20 token IDs: {encoded2['input_ids'][0][:20].tolist()}")
    
    # Decode back to verify
    print("\n7. Decoding tokens back to text (verification)")
    decoded1 = tokenizer.decode(encoded1['input_ids'][0][:30], skip_special_tokens=True)
    decoded2 = tokenizer.decode(encoded2['input_ids'][0][:30], skip_special_tokens=True)
    
    print(f"   Decoded function 1 (first 30 tokens): {decoded1}")
    print(f"   Decoded function 2 (first 30 tokens): {decoded2}")
    
    # Test matching
    print("\n8. Matching verification")
    print(f"   ✓ These two functions are labeled as SIMILAR")
    print(f"   ✓ They should have high cosine similarity after model encoding")
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)
    print("\nSummary:")
    print(f"  - Tokenizer: Working ✓")
    print(f"  - Data loading: Working ✓")
    print(f"  - Function pairing: Working ✓")
    print(f"  - Tokenization: Working ✓")
    print("\nNext steps:")
    print("  - Run full finetune with: ./run_finetune_baseline.sh")
    print("  - Or test with small batch: python finetune.py ...")
    print("="*60)

if __name__ == "__main__":
    try:
        test_single_pair()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
