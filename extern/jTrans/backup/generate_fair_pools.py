#!/usr/bin/env python
"""
Generate pool and query JSON files with matched IDs for fair comparison.

This script creates:
- pool_100.json, pool_1000.json, pool_10000.json
- query_pool_100.json, query_pool_1000.json, query_pool_10000.json

Each entry contains:
- id: unique identifier
- binary: binary name
- function_name: function name
- opt: optimization level
- baseline_id: index in func_blocks_baseline.json
- addressaware_id: index in func_blocks_addressaware.json

This ensures both models use EXACTLY the same pool and query definitions.
"""

import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm


def load_json_with_metadata(func_blocks_path, ground_truth_path):
    """
    Load function blocks with metadata from JSON format.
    
    Returns:
        dict: {(binary, func_name, opt) -> func_id}
    """
    print(f"Loading {ground_truth_path}...")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    all_pairs = ground_truth['pairs']
    print(f"  Loaded {len(all_pairs)} pairs")
    
    # Build mapping: (binary, func_name, opt) -> func_id
    metadata_to_id = {}
    
    for pair in tqdm(all_pairs, desc="  Processing pairs"):
        # Handle both baseline format (binary_name, function_name) and address-aware format (binary, func_name)
        binary = pair.get('binary_name', pair.get('binary', 'unknown'))
        func_name = pair.get('function_name', pair.get('func_name', 'unknown'))
        
        # Baseline format: has O0, O1, O2, O3, Os fields directly
        # Address-aware format: has opt1, opt2, func_id1, func_id2
        if 'opt1' in pair and 'opt2' in pair:
            # Address-aware format
            opt1, opt2 = pair['opt1'], pair['opt2']
            func_id1, func_id2 = str(pair['func_id1']), str(pair['func_id2'])
            
            key1 = (binary, func_name, opt1)
            key2 = (binary, func_name, opt2)
            
            # Use first occurrence if duplicate
            if key1 not in metadata_to_id:
                metadata_to_id[key1] = func_id1
            if key2 not in metadata_to_id:
                metadata_to_id[key2] = func_id2
        else:
            # Baseline format - all optimization levels in one entry
            for opt_level in ['O0', 'O1', 'O2', 'O3', 'Os']:
                if opt_level in pair:
                    func_id = str(pair[opt_level])
                    key = (binary, func_name, opt_level)
                    if key not in metadata_to_id:
                        metadata_to_id[key] = func_id
    
    print(f"  Found {len(metadata_to_id)} unique (binary, func_name, opt) entries")
    return metadata_to_id


def build_metadata_index(metadata_to_id):
    """
    Already have the right structure, just return it.
    
    Returns:
        dict: {(binary, func_name, opt) -> func_id}
    """
    return metadata_to_id


def generate_pools_and_queries(baseline_meta, addressaware_meta, 
                               baseline_func_blocks,
                               pool_sizes, output_dir, 
                               min_token_length=30,
                               query_opt_pairs=[('O0', 'O1'), ('O0', 'O2'), ('O0', 'O3')]):
    """
    Generate pool and query JSON files.
    
    Args:
        baseline_meta: {(binary, func_name, opt) -> func_id}
        addressaware_meta: {(binary, func_name, opt) -> func_id}
        baseline_func_blocks: dict with function token data
        min_token_length: minimum number of tokens for baseline functions
    """
    print("\nFinding common entries...")
    
    # Find common (binary, func_name, opt) entries
    common_keys = set(baseline_meta.keys()) & set(addressaware_meta.keys())
    print(f"Found {len(common_keys)} common (binary, func_name, opt) entries")
    
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
        
        # Count tokens (split by whitespace)
        token_count = len(tokens.split())
        
        if token_count >= min_token_length:
            filtered_keys.append(meta_key)
    
    print(f"  Kept {len(filtered_keys)} / {len(common_keys)} entries (filtered out {len(common_keys) - len(filtered_keys)} short functions)")
    
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
        
        # Get function IDs from both models
        baseline_func_id = baseline_meta[meta_key]
        addressaware_func_id = addressaware_meta[meta_key]
        
        function_groups[func_key].append({
            'binary': binary,
            'function_name': func_name,
            'opt': opt,
            'baseline_func_id': baseline_func_id,
            'addressaware_func_id': addressaware_func_id
        })
    
    print(f"Found {len(function_groups)} unique functions")
    print(f"Total entries: {sum(len(v) for v in function_groups.values())}")
    
    # Convert to flat list with IDs (keep function groups together)
    matched_entries = []
    entry_id = 0
    for func_key, opt_variants in function_groups.items():
        for entry in opt_variants:
            entry['id'] = entry_id
            matched_entries.append(entry)
            entry_id += 1
    
    print(f"Total matched entries: {len(matched_entries)}")
    
    # Generate pools and queries for each pool size
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pool_size in pool_sizes:
        print(f"\n{'='*60}")
        print(f"Generating pools and queries for pool_size={pool_size}")
        print(f"{'='*60}")
        
        # For each opt pair, create a separate pool
        for anchor_opt, target_opt in query_opt_pairs:
            pair_name = f"{anchor_opt}_vs_{target_opt}"
            print(f"\n  Processing {pair_name}...")
            
            # Sample FUNCTION GROUPS that have BOTH anchor_opt AND target_opt
            eligible_functions = []
            for func_key, opt_variants in function_groups.items():
                opts_available = set(e['opt'] for e in opt_variants)
                if anchor_opt in opts_available and target_opt in opts_available:
                    eligible_functions.append(func_key)
            
            print(f"    Eligible functions (have both {anchor_opt} and {target_opt}): {len(eligible_functions)}")
            
            # 20% of pool should be queries
            num_queries_target = int(pool_size * 0.2)
            num_queries_target = max(1, num_queries_target)
            
            # Each query function contributes: 1 ground truth (target) to pool
            # Queries themselves are NOT in the pool
            num_query_functions = num_queries_target
            
            # Remaining pool entries will be filled with both opts (mixed negatives/distractors)
            num_remaining_entries = pool_size - num_query_functions
            # Each distractor function can contribute up to 2 entries (anchor + target)
            num_distractor_functions = max(0, num_remaining_entries // 2)
            
            if len(eligible_functions) < num_query_functions:
                print(f"    WARNING: Only {len(eligible_functions)} eligible functions available, need {num_query_functions}")
                num_query_functions = len(eligible_functions)
                num_distractor_functions = 0
            
            # Sample query functions (will have both anchor and target in pool)
            query_functions = random.sample(eligible_functions, min(num_query_functions, len(eligible_functions)))
            
            # Sample additional functions for pool (only target opt as distractors)
            remaining_functions = [f for f in eligible_functions if f not in query_functions]
            if num_distractor_functions > 0 and remaining_functions:
                additional_functions = random.sample(remaining_functions, min(num_distractor_functions, len(remaining_functions)))
            else:
                additional_functions = []
            
            # Build pool and query lists
            pool_entries = []
            query_entries = []
            
            # Add query functions
            for func_key in query_functions:
                for entry in function_groups[func_key]:
                    # Add queries (anchor opt) - NOT in pool, only in query list
                    if entry['opt'] == anchor_opt:
                        query_entry = entry.copy()
                        query_entry['query_id'] = len(query_entries)
                        query_entries.append(query_entry)
                    
                    # Add ground truth (target opt) - IN pool
                    elif entry['opt'] == target_opt:
                        pool_entry = entry.copy()
                        pool_entry['pool_id'] = len(pool_entries)
                        pool_entries.append(pool_entry)
            
            # Add additional functions as harder negatives
            # Strategy: Include functions from same binaries as queries for harder challenge
            query_binaries = set(function_groups[func_key][0]['binary'] for func_key in query_functions)
            
            # Partition additional functions into same-binary and different-binary
            same_binary_funcs = [f for f in additional_functions 
                                if function_groups[f][0]['binary'] in query_binaries]
            diff_binary_funcs = [f for f in additional_functions 
                                if function_groups[f][0]['binary'] not in query_binaries]
            
            # Prefer 50% same-binary negatives (harder) and 50% different-binary
            target_same_binary = min(len(same_binary_funcs), num_distractor_functions // 2)
            target_diff_binary = num_distractor_functions - target_same_binary
            
            selected_additional = []
            if same_binary_funcs:
                selected_additional.extend(random.sample(same_binary_funcs, min(target_same_binary, len(same_binary_funcs))))
            if diff_binary_funcs:
                needed = num_distractor_functions - len(selected_additional)
                selected_additional.extend(random.sample(diff_binary_funcs, min(needed, len(diff_binary_funcs))))
            
            # Add all optimization levels (O0, O1, O2, O3) as distractors for maximum hardness
            all_opts = ['O0', 'O1', 'O2', 'O3']
            for func_key in selected_additional:
                for entry in function_groups[func_key]:
                    # Include ALL optimization levels in pool (not just anchor/target)
                    if entry['opt'] in all_opts:
                        pool_entry = entry.copy()
                        pool_entry['pool_id'] = len(pool_entries)
                        pool_entries.append(pool_entry)
                        
                        # Stop if we've reached pool size
                        if len(pool_entries) >= pool_size:
                            break
                if len(pool_entries) >= pool_size:
                    break
            
            query_percentage = (len(query_entries) / len(pool_entries) * 100) if pool_entries else 0
            print(f"    Pool entries: {len(pool_entries)} ({len(query_entries)} queries = {query_percentage:.1f}%, {len(pool_entries) - len(query_entries)} candidates)")
            
            # Save pool for this opt pair
            pool_file = output_dir / f"pool_{pool_size}_{pair_name}.json"
            with open(pool_file, 'w') as f:
                json.dump(pool_entries, f, indent=2)
            print(f"    Saved pool: {pool_file}")
            
            # Save query file for this opt pair
            query_file = output_dir / f"query_pool_{pool_size}_{pair_name}.json"
            with open(query_file, 'w') as f:
                json.dump(query_entries, f, indent=2)
            print(f"    Saved queries: {query_file}")
    
    # Generate statistics
    stats = {
        'total_common_entries': len(matched_entries),
        'pool_sizes': pool_sizes,
        'opt_pairs': [f"{a}_vs_{t}" for a, t in query_opt_pairs],
        'binaries': sorted(list(set(e['binary'] for e in matched_entries))),
        'num_binaries': len(set(e['binary'] for e in matched_entries)),
        'opts': sorted(list(set(e['opt'] for e in matched_entries))),
        'num_functions': len(set((e['binary'], e['function_name']) for e in matched_entries))
    }
    
    stats_file = output_dir / 'pool_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to: {stats_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total matched entries: {stats['total_common_entries']}")
    print(f"Unique binaries: {stats['num_binaries']}")
    print(f"Unique functions: {stats['num_functions']}")
    print(f"Optimization levels: {stats['opts']}")
    print(f"\nGenerated pools: {pool_sizes}")
    for pool_size in pool_sizes:
        actual_size = min(pool_size, len(matched_entries))
        print(f"  pool_{pool_size}.json: {actual_size} entries")
        for anchor_opt, target_opt in query_opt_pairs:
            num_queries = len([e for e in (matched_entries if len(matched_entries) <= pool_size else random.sample(matched_entries, pool_size)) if e['opt'] == anchor_opt])
            print(f"    query_pool_{pool_size}_{anchor_opt}_vs_{target_opt}.json: ~{num_queries} queries")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate fair pool and query JSON files')
    parser.add_argument('--func_blocks_baseline', type=str, required=True,
                        help='Path to baseline func_blocks.json')
    parser.add_argument('--func_blocks_addressaware', type=str, required=True,
                        help='Path to address-aware func_blocks.json')
    parser.add_argument('--ground_truth_baseline', type=str, required=True,
                        help='Path to baseline ground_truth.json')
    parser.add_argument('--ground_truth_addressaware', type=str, required=True,
                        help='Path to address-aware ground_truth.json')
    parser.add_argument('--output_dir', type=str, default='./fair_pools',
                        help='Directory to save pool and query JSON files')
    parser.add_argument('--pool_sizes', type=int, nargs='+', 
                        default=[100, 1000, 10000],
                        help='Pool sizes to generate')
    parser.add_argument('--min_token_length', type=int, default=30,
                        help='Minimum baseline token count to include function')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load datasets
    print("="*60)
    print("Loading baseline dataset...")
    print("="*60)
    baseline_meta = load_json_with_metadata(
        args.func_blocks_baseline,
        args.ground_truth_baseline
    )
    
    # Load baseline func_blocks for token filtering
    print("\nLoading baseline func_blocks for token filtering...")
    with open(args.func_blocks_baseline, 'r') as f:
        baseline_func_blocks = json.load(f)
    print(f"  Loaded {len(baseline_func_blocks)} baseline functions")
    
    print("\n" + "="*60)
    print("Loading address-aware dataset...")
    print("="*60)
    addressaware_meta = load_json_with_metadata(
        args.func_blocks_addressaware,
        args.ground_truth_addressaware
    )
    
    # Generate pools and queries
    generate_pools_and_queries(
        baseline_meta,
        addressaware_meta,
        baseline_func_blocks,
        pool_sizes=args.pool_sizes,
        output_dir=args.output_dir,
        min_token_length=args.min_token_length,
        query_opt_pairs=[('O0', 'O3'), ('O1', 'O3'), ('O2', 'O3')]
    )


if __name__ == '__main__':
    main()
