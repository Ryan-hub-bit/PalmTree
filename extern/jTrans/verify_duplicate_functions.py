#!/usr/bin/env python3
"""
Verify the number of functions with identical instructions across different optimization levels in the baseline dataset

Check how many functions have completely identical instruction sequences across O0, O1, O2, O3 optimization levels.
This helps identify data quality issues and trivial cases.
"""

import json
import hashlib
from collections import defaultdict, Counter
from tqdm import tqdm


def compute_instruction_hash(tokens_str):
    """
    Compute hash value of instruction sequence
    
    Args:
        tokens_str: Function's tokens string (tab-separated instructions)
    
    Returns:
        str: MD5 hash
    """
    # Use MD5 to compute hash (fast, no need for cryptographic-level security)
    return hashlib.md5(tokens_str.encode('utf-8')).hexdigest()


def analyze_duplicate_functions(func_blocks_path, ground_truth_path):
    """
    Analyze duplicate functions within each optimization level
    
    Args:
        func_blocks_path: Path to func_blocks_baseline.json
        ground_truth_path: Path to ground_truth_baseline.json
    """
    print("="*80)
    print("Baseline Dataset Function Duplication Analysis")
    print("="*80)
    
    # Load data
    print(f"\nLoading ground truth: {ground_truth_path}")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"Loading function blocks: {func_blocks_path}")
    with open(func_blocks_path, 'r') as f:
        func_blocks = json.load(f)
    
    pairs = ground_truth['pairs']
    print(f"Total pairs: {len(pairs)}")
    
    # Collect hashes for each optimization level
    opt_levels = ['O0', 'O1', 'O2', 'O3', 'Os']
    opt_hashes = {opt: [] for opt in opt_levels}
    opt_hash_to_funcs = {opt: defaultdict(list) for opt in opt_levels}
    
    print("\nComputing instruction hash for each function...")
    for pair in tqdm(pairs, desc="Processing pairs"):
        binary = pair['binary_name']
        func_name = pair['function_name']
        
        for opt in opt_levels:
            if opt in pair:
                func_id = str(pair[opt])
                
                if func_id in func_blocks:
                    tokens_str = func_blocks[func_id].get('tokens', '')
                    
                    # Compute hash
                    instr_hash = compute_instruction_hash(tokens_str)
                    opt_hashes[opt].append(instr_hash)
                    opt_hash_to_funcs[opt][instr_hash].append({
                        'binary': binary,
                        'function': func_name,
                        'func_id': func_id,
                        'num_instructions': len(tokens_str.split('\t')) if tokens_str else 0
                    })
    
    # Statistics results
    print("\n" + "="*80)
    print("Statistics Results")
    print("="*80)
    
    for opt in opt_levels:
        total_funcs = len(opt_hashes[opt])
        unique_hashes = len(set(opt_hashes[opt]))
        duplicate_count = total_funcs - unique_hashes
        
        print(f"\n【{opt}】")
        print(f"  Total functions:         {total_funcs:,}")
        print(f"  Unique instruction seqs: {unique_hashes:,}")
        print(f"  Duplicate functions:     {duplicate_count:,} ({100*duplicate_count/total_funcs:.2f}%)")
        
        # Find most common duplicates
        hash_counts = Counter(opt_hashes[opt])
        most_common = hash_counts.most_common(5)
        
        if len(most_common) > 0 and most_common[0][1] > 1:
            print(f"\n  Top 5 most common duplicate instruction sequences:")
            for i, (hash_val, count) in enumerate(most_common, 1):
                if count > 1:
                    sample_funcs = opt_hash_to_funcs[opt][hash_val][:3]
                    print(f"    {i}. Appears {count} times (hash: {hash_val[:12]}...)")
                    print(f"       Example function: {sample_funcs[0]['function']} ({sample_funcs[0]['num_instructions']} instructions)")
                    if len(sample_funcs) > 1:
                        print(f"                {sample_funcs[1]['function']} ({sample_funcs[1]['num_instructions']} instructions)")
    
    # Cross-optimization level analysis
    print("\n" + "="*80)
    print("Cross-Optimization Level Analysis")
    print("="*80)
    
    print("\nChecking functions with identical instructions across different optimization levels...")
    
    # For each pair, check if different opt levels have identical instructions
    same_across_opts = defaultdict(int)
    
    for pair in tqdm(pairs, desc="Comparing across opts"):
        # Get hashes for all opt levels
        pair_hashes = {}
        for opt in opt_levels:
            if opt in pair:
                func_id = str(pair[opt])
                if func_id in func_blocks:
                    tokens_str = func_blocks[func_id].get('tokens', '')
                    pair_hashes[opt] = compute_instruction_hash(tokens_str)
        
        # Compare between different opt levels
        for i, opt1 in enumerate(opt_levels):
            if opt1 not in pair_hashes:
                continue
            for opt2 in opt_levels[i+1:]:
                if opt2 not in pair_hashes:
                    continue
                
                if pair_hashes[opt1] == pair_hashes[opt2]:
                    same_across_opts[f"{opt1}={opt2}"] += 1
    
    print("\nFunction pairs with identical instructions:")
    for pair_key in sorted(same_across_opts.keys()):
        count = same_across_opts[pair_key]
        percentage = 100 * count / len(pairs)
        print(f"  {pair_key}: {count:,} pairs ({percentage:.2f}%)")
    
    # Find functions that are identical across all optimization levels
    all_same_count = 0
    for pair in pairs:
        pair_hashes = {}
        for opt in opt_levels:
            if opt in pair:
                func_id = str(pair[opt])
                if func_id in func_blocks:
                    tokens_str = func_blocks[func_id].get('tokens', '')
                    pair_hashes[opt] = compute_instruction_hash(tokens_str)
        
        if len(pair_hashes) >= 2:
            if len(set(pair_hashes.values())) == 1:
                all_same_count += 1
    
    print(f"\nAll optimization levels have identical instructions: {all_same_count:,} functions ({100*all_same_count/len(pairs):.2f}%)")
    
    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze duplicate functions in baseline dataset')
    parser.add_argument('--func-blocks', type=str,
                        default='/data/kun/jtrans/baseline/func_blocks_baseline.json',
                        help='Path to func_blocks_baseline.json')
    parser.add_argument('--ground-truth', type=str,
                        default='/data/kun/jtrans/baseline/ground_truth_baseline.json',
                        help='Path to ground_truth_baseline.json')
    
    args = parser.parse_args()
    
    analyze_duplicate_functions(args.func_blocks, args.ground_truth)
