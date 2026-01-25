#!/usr/bin/env python3
"""
Simplified version: Only analyze duplicate functions within a single optimization level

Only requires func_blocks_baseline.json, no need for ground_truth.
Suitable for quick data quality checks at each optimization level.
"""

import json
import hashlib
from collections import defaultdict, Counter
from tqdm import tqdm


def compute_instruction_hash(tokens_str):
    """Compute hash value of instruction sequence"""
    return hashlib.md5(tokens_str.encode('utf-8')).hexdigest()


def analyze_single_opt_duplicates(func_blocks_path):
    """
    Analyze duplicate functions within each optimization level (no ground_truth needed)
    """
    print("="*80)
    print("Single Optimization Level Function Duplication Analysis (Simplified)")
    print("="*80)
    
    print(f"\nLoading function blocks: {func_blocks_path}")
    with open(func_blocks_path, 'r') as f:
        func_blocks = json.load(f)
    
    print(f"Total functions: {len(func_blocks):,}")
    
    # Group by optimization level
    opt_levels = ['O0', 'O1', 'O2', 'O3', 'Os']
    opt_funcs = {opt: [] for opt in opt_levels}
    opt_hash_to_funcs = {opt: defaultdict(list) for opt in opt_levels}
    
    print("\nGrouping by optimization level and computing hashes...")
    for func_id, func_data in tqdm(func_blocks.items(), desc="Processing functions"):
        opt = func_data.get('opt', 'unknown')
        
        if opt in opt_levels:
            tokens_str = func_data.get('tokens', '')
            instr_hash = compute_instruction_hash(tokens_str)
            
            opt_funcs[opt].append(instr_hash)
            opt_hash_to_funcs[opt][instr_hash].append({
                'func_id': func_id,
                'binary': func_data.get('binary', 'unknown'),
                'name': func_data.get('name', 'unknown'),
                'num_instructions': len(tokens_str.split('\t')) if tokens_str else 0
            })
    
    # Statistics results
    print("\n" + "="*80)
    print("Statistics Results")
    print("="*80)
    
    for opt in opt_levels:
        if len(opt_funcs[opt]) == 0:
            print(f"\n【{opt}】: No data")
            continue
            
        total_funcs = len(opt_funcs[opt])
        unique_hashes = len(set(opt_funcs[opt]))
        duplicate_count = total_funcs - unique_hashes
        
        print(f"\n【{opt}】")
        print(f"  Total functions:         {total_funcs:,}")
        print(f"  Unique instruction seqs: {unique_hashes:,}")
        print(f"  Duplicate functions:     {duplicate_count:,} ({100*duplicate_count/total_funcs:.2f}%)")
        
        # Find most common duplicates
        hash_counts = Counter(opt_funcs[opt])
        most_common = hash_counts.most_common(10)
        
        duplicates = [(h, c) for h, c in most_common if c > 1]
        
        if duplicates:
            print(f"\n  Top {min(5, len(duplicates))} most common duplicate instruction sequences:")
            for i, (hash_val, count) in enumerate(duplicates[:5], 1):
                sample_funcs = opt_hash_to_funcs[opt][hash_val][:3]
                print(f"    {i}. Appears {count} times (hash: {hash_val[:12]}...)")
                for j, func_info in enumerate(sample_funcs):
                    prefix = "       " if j == 0 else "       "
                    print(f"{prefix}• {func_info['name']} - {func_info['binary'][:30]} ({func_info['num_instructions']} instr)")
    
    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)
    print("\nNote: This version only analyzes duplicates within a single optimization level")
    print("To analyze changes of the same function across different optimization levels, use the full version script")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze duplicate functions within a single optimization level (simplified version)')
    parser.add_argument('--func-blocks', type=str,
                        default='/data/kun/jtrans/baseline/func_blocks_baseline.json',
                        help='Path to func_blocks_baseline.json')
    
    args = parser.parse_args()
    
    analyze_single_opt_duplicates(args.func_blocks)
