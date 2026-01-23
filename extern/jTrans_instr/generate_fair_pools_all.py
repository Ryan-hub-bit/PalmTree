#!/usr/bin/env python
"""
Generate fair pool and query JSON files for baseline, addressaware, and jTrans_instr.

This script creates pool and query files that work with ALL THREE models:
- pool_100_{opt_pair}.json, pool_1000_{opt_pair}.json, pool_10000_{opt_pair}.json  
- query_pool_100_{opt_pair}.json, query_pool_1000_{opt_pair}.json, query_pool_10000_{opt_pair}.json

Each entry contains IDs for all three models for fair comparison.
"""

import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm


def load_json_with_metadata(func_blocks_path, ground_truth_path, model_name):
    """
    Load function blocks with metadata from JSON format.
    
    Returns:
        dict: {(binary, func_name, opt) -> func_id}
    """
    print(f"[{model_name}] Loading {ground_truth_path}...")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    all_pairs = ground_truth['pairs']
    print(f"[{model_name}]   Loaded {len(all_pairs)} pairs")
    
    # Build mapping: (binary, func_name, opt) -> func_id
    metadata_to_id = {}
    
    for pair in tqdm(all_pairs, desc=f"[{model_name}]   Processing"):
        binary = pair.get('binary_name', pair.get('binary', 'unknown'))
        func_name = pair.get('function_name', pair.get('func_name', 'unknown'))
        
        # Handle both baseline and address-aware formats
        if 'opt1' in pair and 'opt2' in pair:
            # Address-aware or jTrans_instr format
            opt1, opt2 = pair['opt1'], pair['opt2']
            func_id1, func_id2 = str(pair['func_id1']), str(pair['func_id2'])
            
            key1 = (binary, func_name, opt1)
            key2 = (binary, func_name, opt2)
            
            if key1 not in metadata_to_id:
                metadata_to_id[key1] = func_id1
            if key2 not in metadata_to_id:
                metadata_to_id[key2] = func_id2
        else:
            # Baseline format
            for opt_level in ['O0', 'O1', 'O2', 'O3', 'Os']:
                if opt_level in pair:
                    func_id = str(pair[opt_level])
                    key = (binary, func_name, opt_level)
                    if key not in metadata_to_id:
                        metadata_to_id[key] = func_id
    
    print(f"[{model_name}]   Found {len(metadata_to_id)} unique entries")
    return metadata_to_id


def generate_pools_and_queries(baseline_meta, addressaware_meta, instr_meta,
                               baseline_func_blocks,
                               pool_sizes, output_dir, 
                               min_token_length=30,
                               query_opt_pairs=[('O0', 'O1'), ('O0', 'O2'), ('O0', 'O3')]):
    """
    Generate pool and query JSON files for all three models.
    """
    print("\nFinding common entries across all three models...")
    
    # Find common (binary, func_name, opt) entries
    common_keys = set(baseline_meta.keys()) & set(addressaware_meta.keys()) & set(instr_meta.keys())
    print(f"Found {len(common_keys)} common entries")
    
    if not common_keys:
        print("ERROR: No common entries found!")
        return
    
    # Filter by baseline token length
    print(f"\nFiltering functions with baseline token size >= {min_token_length}...")
    filtered_keys = []
    for meta_key in tqdm(common_keys, desc="  Checking token lengths"):
        baseline_func_id = baseline_meta[meta_key]
        func_data = baseline_func_blocks.get(str(baseline_func_id), {})
        tokens = func_data.get('tokens', func_data.get('instructions', ''))
        
        token_count = len(tokens.split())
        
        if token_count >= min_token_length:
            filtered_keys.append(meta_key)
    
    print(f"  Kept {len(filtered_keys)} / {len(common_keys)} entries")
    
    if not filtered_keys:
        print("ERROR: No entries passed token length filter!")
        return
    
    # Build matched pool entries GROUPED BY FUNCTION
    print("\nGrouping entries by (binary, function_name)...")
    function_groups = {}
    
    for meta_key in tqdm(sorted(filtered_keys)):
        binary, func_name, opt = meta_key
        func_key = (binary, func_name)
        
        if func_key not in function_groups:
            function_groups[func_key] = []
        
        # Get function IDs from all three models
        baseline_func_id = baseline_meta[meta_key]
        addressaware_func_id = addressaware_meta[meta_key]
        instr_func_id = instr_meta[meta_key]
        
        function_groups[func_key].append({
            'binary': binary,
            'function_name': func_name,
            'opt': opt,
            'baseline_func_id': baseline_func_id,
            'addressaware_func_id': addressaware_func_id,
            'instr_func_id': instr_func_id
        })
    
    print(f"Found {len(function_groups)} unique functions")
    
    # Generate pools and queries for each pool size
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pool_size in pool_sizes:
        print(f"\n{'='*60}")
        print(f"Generating pools for pool_size={pool_size}")
        print(f"{'='*60}")
        
        for anchor_opt, target_opt in query_opt_pairs:
            pair_name = f"{anchor_opt}_vs_{target_opt}"
            print(f"\n  Processing {pair_name}...")
            
            # Sample functions that have BOTH anchor_opt AND target_opt
            eligible_functions = []
            for func_key, opt_variants in function_groups.items():
                opts_available = set(e['opt'] for e in opt_variants)
                if anchor_opt in opts_available and target_opt in opts_available:
                    eligible_functions.append(func_key)
            
            print(f"    Eligible functions: {len(eligible_functions)}")
            
            # 20% of pool should be queries
            num_query_functions = max(1, int(pool_size * 0.2))
            num_distractor_functions = max(0, (pool_size - num_query_functions) // 2)
            
            if len(eligible_functions) < num_query_functions:
                num_query_functions = len(eligible_functions)
            
            # Sample query functions
            query_functions = random.sample(eligible_functions, num_query_functions)
            
            # Sample additional functions for pool
            remaining_functions = [f for f in eligible_functions if f not in query_functions]
            if num_distractor_functions > 0 and remaining_functions:
                additional_functions = random.sample(remaining_functions, 
                                                    min(num_distractor_functions, len(remaining_functions)))
            else:
                additional_functions = []
            
            # Build pool and query lists
            pool_entries = []
            query_entries = []
            
            # Add query functions
            for func_key in query_functions:
                for entry in function_groups[func_key]:
                    if entry['opt'] == anchor_opt:
                        query_entry = entry.copy()
                        query_entry['query_id'] = len(query_entries)
                        query_entries.append(query_entry)
                    elif entry['opt'] == target_opt:
                        pool_entry = entry.copy()
                        pool_entry['pool_id'] = len(pool_entries)
                        pool_entries.append(pool_entry)
            
            # Add additional functions
            all_opts = ['O0', 'O1', 'O2', 'O3']
            for func_key in additional_functions:
                for entry in function_groups[func_key]:
                    if entry['opt'] in all_opts:
                        pool_entry = entry.copy()
                        pool_entry['pool_id'] = len(pool_entries)
                        pool_entries.append(pool_entry)
                        
                        if len(pool_entries) >= pool_size:
                            break
                if len(pool_entries) >= pool_size:
                    break
            
            print(f"    Pool: {len(pool_entries)} entries, Queries: {len(query_entries)}")
            
            # Save files
            pool_file = output_dir / f"pool_{pool_size}_{pair_name}.json"
            with open(pool_file, 'w') as f:
                json.dump({'pool': [e['baseline_func_id'] for e in pool_entries]}, f, indent=2)
            
            query_file = output_dir / f"query_pool_{pool_size}_{pair_name}.json"
            with open(query_file, 'w') as f:
                json.dump({'queries': [e['baseline_func_id'] for e in query_entries]}, f, indent=2)
            
            # Save metadata with all three model IDs
            meta_file = output_dir / f"pool_{pool_size}_{pair_name}_meta.json"
            with open(meta_file, 'w') as f:
                json.dump({
                    'pool_entries': pool_entries,
                    'query_entries': query_entries
                }, f, indent=2)
            
            print(f"    Saved: {pool_file.name}, {query_file.name}, {meta_file.name}")
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--func_blocks_baseline', required=True)
    parser.add_argument('--func_blocks_addressaware', required=True)
    parser.add_argument('--func_blocks_instr', required=True)
    parser.add_argument('--ground_truth_baseline', required=True)
    parser.add_argument('--ground_truth_addressaware', required=True)
    parser.add_argument('--ground_truth_instr', required=True)
    parser.add_argument('--output_dir', default='./fair_pools')
    parser.add_argument('--pool_sizes', type=int, nargs='+', default=[100, 1000, 10000])
    parser.add_argument('--min_token_length', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load all three datasets
    print("="*60)
    print("Loading datasets")
    print("="*60)
    
    baseline_meta = load_json_with_metadata(
        args.func_blocks_baseline,
        args.ground_truth_baseline,
        'baseline'
    )
    
    addressaware_meta = load_json_with_metadata(
        args.func_blocks_addressaware,
        args.ground_truth_addressaware,
        'addressaware'
    )
    
    instr_meta = load_json_with_metadata(
        args.func_blocks_instr,
        args.ground_truth_instr,
        'instr'
    )
    
    # Load baseline func_blocks for filtering
    with open(args.func_blocks_baseline, 'r') as f:
        baseline_func_blocks = json.load(f)
    
    # Generate pools
    generate_pools_and_queries(
        baseline_meta,
        addressaware_meta,
        instr_meta,
        baseline_func_blocks,
        pool_sizes=args.pool_sizes,
        output_dir=args.output_dir,
        min_token_length=args.min_token_length,
        query_opt_pairs=[('O0', 'O3'), ('O1', 'O3'), ('O2', 'O3')]
    )


if __name__ == '__main__':
    main()
