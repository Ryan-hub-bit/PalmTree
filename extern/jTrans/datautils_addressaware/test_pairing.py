#!/usr/bin/env python3
"""
Quick test to verify paired data generation and loading.
"""

import pickle
from data_addressaware import DatasetBase, load_addressaware_paired_data

def test_dataset_base():
    """Test DatasetBase iterator."""
    print("="*60)
    print("Test 1: DatasetBase.get_paired_data_iter()")
    print("="*60)
    
    dataset = DatasetBase(
        path='/data/kun/jtransdata/addr_extract/',
        prefixfilter=None,
        all_data=True,
        opt=['O0', 'O1', 'O2', 'O3', 'Os']
    )
    
    count = 0
    for project, func_name, opt_data in dataset.get_paired_data_iter():
        count += 1
        if count <= 3:
            print(f"\nFunction group {count}:")
            print(f"  Project: {project}")
            print(f"  Function: {func_name}")
            print(f"  Opts: {list(opt_data.keys())}")
            
            # Show first variant
            opt = list(opt_data.keys())[0]
            func_addr, asm_list, rawbytes, cfg, bai = opt_data[opt]
            print(f"  {opt}: address={hex(func_addr)}, {len(asm_list)} instructions")
            print(f"       First 3 instructions: {asm_list[:3]}")
    
    print(f"\n✓ Total function groups: {count}")
    return count

def test_paired_data():
    """Test pre-generated paired data."""
    print("\n" + "="*60)
    print("Test 2: load_addressaware_paired_data()")
    print("="*60)
    
    pkl_path = '/tmp/test_paired_data.pkl'
    
    functions, metadata = load_addressaware_paired_data(pkl_path)
    
    print(f"\nLoaded from: {pkl_path}")
    print(f"Function groups: {len(functions)}")
    print(f"Metadata entries: {len(metadata)}")
    
    # Show sample
    sample_idx = 5
    sample_group = functions[sample_idx]
    sample_meta = metadata[sample_idx]
    
    print(f"\nSample function group:")
    print(f"  Project: {sample_meta['project']}")
    print(f"  Function: {sample_meta['function']}")
    print(f"  Opts: {list(sample_meta['opts'].keys())}")
    print(f"  Variants: {len(sample_group)}")
    
    # Show each variant
    for opt, idx in sample_meta['opts'].items():
        func_addr, asm_list, rawbytes, cfg, bai = sample_group[idx]
        print(f"    {opt}: address={hex(func_addr)}, {len(asm_list)} instructions")
    
    print(f"\n✓ Paired data loaded successfully")
    return len(functions)

def simulate_triplet_creation():
    """Simulate how triplet loss would create anchor/positive/negative."""
    print("\n" + "="*60)
    print("Test 3: Simulate Triplet Creation")
    print("="*60)
    
    functions, metadata = load_addressaware_paired_data('/tmp/test_paired_data.pkl')
    
    import random
    
    # Pick a random function group
    idx = random.randint(0, len(functions) - 1)
    pairs = functions[idx]
    meta = metadata[idx]
    
    print(f"\nSelected function group {idx}:")
    print(f"  Project: {meta['project']}")
    print(f"  Function: {meta['function']}")
    print(f"  Available variants: {list(meta['opts'].keys())}")
    
    # Create anchor and positive (same function, different opts)
    pos1 = random.randint(0, len(pairs) - 1)
    pos2 = random.randint(0, len(pairs) - 1)
    while pos2 == pos1:
        pos2 = random.randint(0, len(pairs) - 1)
    
    anchor = pairs[pos1]
    positive = pairs[pos2]
    
    # Create negative (different function)
    neg_idx = random.randint(0, len(functions) - 1)
    while neg_idx == idx:
        neg_idx = random.randint(0, len(functions) - 1)
    
    neg_pairs = functions[neg_idx]
    neg_meta = metadata[neg_idx]
    neg_pos = random.randint(0, len(neg_pairs) - 1)
    negative = neg_pairs[neg_pos]
    
    print(f"\nTriplet creation:")
    print(f"  Anchor:   {meta['project']}/{meta['function']} variant {pos1}")
    print(f"            {len(anchor[1])} instructions")
    print(f"  Positive: {meta['project']}/{meta['function']} variant {pos2}")
    print(f"            {len(positive[1])} instructions")
    print(f"  Negative: {neg_meta['project']}/{neg_meta['function']} variant {neg_pos}")
    print(f"            {len(negative[1])} instructions")
    
    print(f"\n✓ Triplet loss simulation successful")

if __name__ == '__main__':
    try:
        count1 = test_dataset_base()
        count2 = test_paired_data()
        simulate_triplet_creation()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print(f"\nYour address-aware data has {count1} function groups ready for fine-tuning!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
