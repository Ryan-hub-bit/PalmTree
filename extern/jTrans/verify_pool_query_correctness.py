#!/usr/bin/env python3
"""
Verify pool and query correctness
Checks:
1. Whether query and GT are from same function's different optimization versions
2. Whether query and GT instructions are actually different
3. Whether pool contains duplicate instructions
"""

import json
import sys
from pathlib import Path
import hashlib


def compute_hash(tokens_str):
    """Compute normalized hash"""
    if not tokens_str:
        return hashlib.md5(b'').hexdigest()
    normalized = ' '.join(tokens_str.split())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def verify_pools(func_blocks_path, pool_dir, num_samples=5):
    """Verify pool and query correctness"""
    
    print("=" * 80)
    print("Verifying Pool and Query Correctness")
    print("=" * 80)
    
    # Load func_blocks
    print(f"\nLoading func_blocks: {func_blocks_path}")
    with open(func_blocks_path, 'r') as f:
        func_blocks = json.load(f)
    
    pool_path = Path(pool_dir)
    
    # Find all pool and query files
    pool_files = sorted(pool_path.glob('pool_*_filtered.json'))
    query_files = sorted(pool_path.glob('query_*_filtered.json'))
    
    if not pool_files:
        print(f"\n⚠️  No pool files found in {pool_dir}")
        return
    
    print(f"\nFound {len(pool_files)} pool files")
    
    for pool_file in pool_files:
        # Find corresponding query file
        # pool_10000_O2_vs_O3_filtered.json -> query_*_from_pool_10000_O2_vs_O3_filtered.json
        pool_stem = pool_file.stem  # pool_10000_O2_vs_O3_filtered
        
        # Extract key information
        # Format: pool_{size}_{opt_low}_vs_{opt_high}_filtered
        parts = pool_stem.split('_')
        if len(parts) < 5:
            continue
        
        pool_size = parts[1]  # 10000
        opt_pair = '_'.join(parts[2:])  # O2_vs_O3_filtered
        
        # Find matching query file
        # Format: query_{query_size}_from_pool_{pool_size}_{opt_low}_vs_{opt_high}_filtered.json
        matching_queries = list(pool_path.glob(f'query_*_from_pool_{pool_size}_{opt_pair}.json'))
        
        if not matching_queries:
            print(f"\n⚠️  No matching query file found: {pool_file.name}")
            print(f"    Expected: query_*_from_pool_{pool_size}_{opt_pair}.json")
            continue
        
        query_file = matching_queries[0]
        
        print(f"\n{'='*80}")
        print(f"Checking: {pool_file.name}")
        print(f"          {query_file.name}")
        print(f"{'='*80}")
        
        # Load pool and query
        with open(pool_file, 'r') as f:
            pool_data = json.load(f)
        
        with open(query_file, 'r') as f:
            query_data = json.load(f)
        
        pool_ids = pool_data['pool']
        query_ids = query_data['queries']
        gt_indices = query_data['ground_truth']
        
        opt_low = query_data['opt_level']
        opt_high = pool_data['opt_level']
        
        print(f"\nPool Info:")
        print(f"  Optimization level: {opt_high}")
        print(f"  Size: {len(pool_ids)}")
        
        print(f"\nQuery Info:")
        print(f"  Optimization level: {opt_low}")
        print(f"  Size: {len(query_ids)}")
        print(f"  GT indices range: [{min(gt_indices)}, {max(gt_indices)}]")
        
        # Check for duplicates in pool
        pool_hashes = []
        for pid in pool_ids:
            if str(pid) in func_blocks:
                tokens = func_blocks[str(pid)].get('tokens', '')
                pool_hashes.append(compute_hash(tokens))
        
        unique_hashes = len(set(pool_hashes))
        print(f"\nPool Uniqueness Check:")
        print(f"  Total: {len(pool_hashes)}")
        print(f"  Unique: {unique_hashes}")
        print(f"  Duplicates: {len(pool_hashes) - unique_hashes}")
        
        if unique_hashes < len(pool_hashes):
            print(f"  ⚠️  Pool contains duplicate instructions!")
        else:
            print(f"  ✓ All instructions in pool are unique")
        
        # Verify query-GT correspondence
        print(f"\nVerifying Query-GT Correspondence (first {num_samples} samples):")
        print(f"-" * 80)
        
        all_correct = True
        query_gt_same_count = 0
        
        for i in range(min(num_samples, len(query_ids))):
            query_id = str(query_ids[i])
            gt_idx = gt_indices[i]
            
            # Check if GT index is within pool range
            if gt_idx >= len(pool_ids):
                print(f"\n  Sample {i+1}: ⚠️  GT index {gt_idx} exceeds pool size {len(pool_ids)}!")
                all_correct = False
                continue
            
            gt_id = str(pool_ids[gt_idx])
            
            if query_id not in func_blocks or gt_id not in func_blocks:
                print(f"\n  Sample {i+1}: ⚠️  ID does not exist")
                all_correct = False
                continue
            
            query_func = func_blocks[query_id]
            gt_func = func_blocks[gt_id]
            
            # Check if from same function
            query_binary = query_func.get('binary', '')
            query_name = query_func.get('name', '')
            query_opt = query_func.get('opt', '')
            query_tokens = query_func.get('tokens', '')
            
            gt_binary = gt_func.get('binary', '')
            gt_name = gt_func.get('name', '')
            gt_opt = gt_func.get('opt', '')
            gt_tokens = gt_func.get('tokens', '')
            
            same_function = (query_binary == gt_binary and query_name == gt_name)
            same_instruction = (compute_hash(query_tokens) == compute_hash(gt_tokens))
            
            print(f"\n  Sample {i+1}:")
            print(f"    Query ID: {query_id}")
            print(f"      Binary: {query_binary}")
            print(f"      Function: {query_name}")
            print(f"      Opt: {query_opt}")
            print(f"      Instructions: {query_func.get('num_instructions', 'N/A')}")
            print(f"    GT ID: {gt_id} (pool index: {gt_idx})")
            print(f"      Binary: {gt_binary}")
            print(f"      Function: {gt_name}")
            print(f"      Opt: {gt_opt}")
            print(f"      Instructions: {gt_func.get('num_instructions', 'N/A')}")
            
            if same_function:
                print(f"    ✓ Same function's different optimization versions")
            else:
                print(f"    ✗ Not the same function! (binary or name different)")
                all_correct = False
            
            if same_instruction:
                print(f"    ✗ Query and GT instructions are identical!")
                query_gt_same_count += 1
            else:
                print(f"    ✓ Query and GT instructions are different")
            
            # Check if GT is unique in pool
            gt_hash = compute_hash(gt_tokens)
            gt_count_in_pool = pool_hashes.count(gt_hash)
            if gt_count_in_pool > 1:
                print(f"    ✗ GT's instruction appears {gt_count_in_pool} times in pool!")
                all_correct = False
            else:
                print(f"    ✓ GT is unique in pool")
        
        print(f"\n{'='*80}")
        if all_correct and query_gt_same_count == 0:
            print(f"✅ All checks passed!")
        else:
            print(f"⚠️  Issues found:")
            if not all_correct:
                print(f"  - query-GT correspondence errors exist")
            if query_gt_same_count > 0:
                print(f"  - {query_gt_same_count} queries have identical instructions as GT")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify pool and query correctness')
    parser.add_argument('--func-blocks', type=str,
                        default='/data/kun/jtrans/baseline/eval/func_blocks_baseline.json',
                        help='Path to func_blocks JSON')
    parser.add_argument('--pool-dir', type=str,
                        default='/data/kun/jtrans/baseline/eval/pools_filtered',
                        help='Directory containing pool files')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of samples to check per pool')
    
    args = parser.parse_args()
    
    verify_pools(args.func_blocks, args.pool_dir, args.samples)
