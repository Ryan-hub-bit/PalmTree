#!/usr/bin/env python3
"""
Create filtered baseline evaluation pools

Filtering strategy:
1. Exclude trivial functions with instruction count <= 10
2. Exclude pairs where query and GT have identical instructions
   - Example: Function A's O2 version and O3 version have identical instructions
   - Such pairs are too simple, model only needs exact match, not semantic understanding
3. Pool deduplication: Ensure each instruction sequence appears only once
4. Maintain target pool size: Continue adding after deduplication until target is reached
"""

import json
import random
import hashlib
from pathlib import Path
from tqdm import tqdm


def compute_instruction_hash(tokens_str):
    """
    Compute hash of instruction sequence
    Normalize: Remove extra spaces to ensure same instructions get same hash
    """
    if not tokens_str:
        return hashlib.md5(b'').hexdigest()
    
    # Normalize: split and rejoin to remove extra spaces
    normalized = ' '.join(tokens_str.split())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def create_filtered_pools(func_blocks_path, ground_truth_path, output_dir,
                         pool_sizes=[100, 1000, 10000], query_ratio=0.2,
                         min_instructions=11, filter_trivial_pairs=True):
    """
    Create filtered evaluation pools
    
    Args:
        func_blocks_path: Path to func_blocks
        ground_truth_path: Path to ground_truth
        output_dir: Output directory
        pool_sizes: List of pool sizes
        query_ratio: Ratio of queries to pool size
        min_instructions: Minimum instruction count (filter trivial functions)
        filter_trivial_pairs: Whether to filter O2=O3 identical pairs
    """
    print("="*80)
    print("Creating Filtered Baseline Evaluation Pools")
    print("="*80)
    
    # Load data
    print(f"\nLoading ground truth: {ground_truth_path}")
    with open(ground_truth_path, 'r') as f:
        gt = json.load(f)
    
    print(f"Loading func blocks: {func_blocks_path}")
    with open(func_blocks_path, 'r') as f:
        func_blocks = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Optimization level pairs
    opt_pairs = [('O0', 'O3'), ('O1', 'O3'), ('O2', 'O3')]
    
    for opt_low, opt_high in opt_pairs:
        print(f"\n{'='*80}")
        print(f"Processing {opt_low} vs {opt_high}")
        print(f"{'='*80}")
        
        # Collect valid pairs
        valid_pairs = []
        trivial_count = 0
        same_instruction_count = 0
        
        print(f"Scanning valid pairs (min_instructions={min_instructions})...")
        for pair in tqdm(gt['pairs']):
            if opt_low not in pair or opt_high not in pair:
                continue
            
            low_id = str(pair[opt_low])
            high_id = str(pair[opt_high])
            
            if low_id not in func_blocks or high_id not in func_blocks:
                continue
            
            low_func = func_blocks[low_id]
            high_func = func_blocks[high_id]
            
            # Filter 1: Too few instructions
            # Use stored num_instructions field (already computed by create_baseline_dataset.py)
            low_num_instr = low_func.get('num_instructions', 0)
            high_num_instr = high_func.get('num_instructions', 0)
            
            if low_num_instr < min_instructions or high_num_instr < min_instructions:
                trivial_count += 1
                continue
            
            # Get tokens for subsequent hash computation
            low_tokens = low_func.get('tokens', '')
            high_tokens = high_func.get('tokens', '')
            
            # Check if tokens are empty (for debugging)
            if not low_tokens or not high_tokens:
                trivial_count += 1
                continue
            
            # Filter 2: Query and GT have identical instructions
            # Example: Function A's O2 version and O3 version have identical instructions
            # Such pairs are too simple, model only needs exact match, no semantic understanding needed
            low_hash = compute_instruction_hash(low_tokens)
            high_hash = compute_instruction_hash(high_tokens)
            
            if low_hash == high_hash:
                same_instruction_count += 1
                continue
            
            valid_pairs.append({
                'low_id': low_id,
                'high_id': high_id,
                'low_hash': low_hash,
                'high_hash': high_hash,
                'binary': pair.get('binary_name', 'unknown'),
                'function': pair.get('function_name', 'unknown'),
                'low_num_instr': low_num_instr,
                'high_num_instr': high_num_instr
            })
        
        print(f"\nFiltering statistics:")
        print(f"  Original pairs: {len(gt['pairs']):,}")
        print(f"  Filtered trivial functions (<{min_instructions} instr): {trivial_count:,}")
        print(f"  Filtered query and GT with identical instructions: {same_instruction_count:,}")
        print(f"    └─ Note: Same function's different optimization versions have identical instructions, too simple")
        print(f"  Remaining valid pairs: {len(valid_pairs):,}")
        
        if len(valid_pairs) == 0:
            print(f"  ⚠️  No valid pairs, skipping")
            continue
        
        # Random shuffle
        random.shuffle(valid_pairs)
        
        # Create pool and query for each pool size
        for pool_size in pool_sizes:
            if pool_size > len(valid_pairs):
                print(f"\n  ⚠️  Pool size {pool_size} > valid pairs {len(valid_pairs)}, skipping")
                continue
            
            query_size = int(pool_size * query_ratio)
            
            print(f"\n  Creating pool_size={pool_size}, query_size={query_size}")
            
            # ========================================
            # Key: Ensure each instruction in pool is unique and maintain target count
            # ========================================
            seen_high_hashes = {}
            unique_pool_pairs = []
            duplicate_in_pool_count = 0
            
            # Select from valid_pairs, deduplicate until reaching target pool_size
            idx = 0
            while len(unique_pool_pairs) < pool_size and idx < len(valid_pairs):
                p = valid_pairs[idx]
                h = p['high_hash']
                
                if h not in seen_high_hashes:
                    seen_high_hashes[h] = p
                    unique_pool_pairs.append(p)
                else:
                    duplicate_in_pool_count += 1
                
                idx += 1
            
            if duplicate_in_pool_count > 0:
                print(f"    Dedup: Skipped {duplicate_in_pool_count} duplicate instructions, selected {len(unique_pool_pairs)} from {idx} candidates")
            
            if len(unique_pool_pairs) < pool_size:
                print(f"    ⚠️  Warning: Only {len(unique_pool_pairs)} unique functions after dedup, less than target {pool_size}")
            
            # Use deduplicated pool
            pool_pairs = unique_pool_pairs
            actual_pool_size = len(pool_pairs)
            
            # ========================================
            # Key: Count occurrences of each hash in pool
            # Used to select only queries with unique GT
            # ========================================
            hash_count_in_pool = {}
            for p in pool_pairs:
                h = p['high_hash']
                hash_count_in_pool[h] = hash_count_in_pool.get(h, 0) + 1
            
            # Create pool (high opt)
            pool_ids = [p['high_id'] for p in pool_pairs]
            
            pool_data = {
                'pool': pool_ids,
                'size': actual_pool_size,
                'opt_level': opt_high,
                'min_instructions': min_instructions,
                'filtered_trivial': True,
                'filtered_same_instruction': True,  # Always filter query=pool identical
                'pool_deduplicated': True,  # Pool already deduplicated
                'metadata': {
                    'opt_pair': f'{opt_low}_vs_{opt_high}',
                    'target_size': pool_size,
                    'actual_size': actual_pool_size,
                    'avg_instructions': sum(p['high_num_instr'] for p in pool_pairs) / len(pool_pairs),
                    'removed_duplicates': duplicate_in_pool_count
                }
            }
            
            pool_file = output_dir / f'pool_{actual_pool_size}_{opt_low}_vs_{opt_high}_filtered.json'
            with open(pool_file, 'w') as f:
                json.dump(pool_data, f, indent=2)
            print(f"    ✓ Pool saved: {pool_file.name}")
            
            # ========================================
            # Key: Only select pairs with unique GT in pool as query candidates
            # Avoid multiple functions in pool having same instruction as a query's GT
            # ========================================
            query_candidates = []
            filtered_ambiguous = 0
            
            for p in pool_pairs:
                gt_hash = p['high_hash']
                # Only use as query if GT's hash is unique in pool
                if hash_count_in_pool[gt_hash] == 1:
                    query_candidates.append(p)
                else:
                    filtered_ambiguous += 1
            
            if filtered_ambiguous > 0:
                print(f"    Filtered: {filtered_ambiguous} pairs with non-unique GT (identical instructions in pool)")
            
            print(f"    Available query candidates: {len(query_candidates)}")
            
            if len(query_candidates) == 0:
                print(f"    ⚠️  No query candidates available, skipping")
                continue
            
            # Create query (sample from candidates)
            actual_query_size = min(query_size, len(query_candidates))
            
            # Random sample queries
            random.shuffle(query_candidates)
            query_pairs = query_candidates[:actual_query_size]
            
            query_ids = [p['low_id'] for p in query_pairs]
            
            # Compute ground truth (index in pool)
            ground_truth_indices = []
            for qp in query_pairs:
                # Find position of corresponding high_id in pool_ids
                gt_idx = pool_ids.index(qp['high_id'])
                ground_truth_indices.append(gt_idx)
            
            # Final verification: Ensure each query's GT is truly unique in pool
            duplicate_gt_found = False
            for i, qp in enumerate(query_pairs):
                gt_hash = qp['high_hash']
                count = sum(1 for p in pool_pairs if p['high_hash'] == gt_hash)
                if count > 1:
                    print(f"    ⚠️⚠️ CRITICAL ERROR: Query {i}'s GT appears {count} times in pool! (logic bug)")
                    duplicate_gt_found = True
            
            if duplicate_gt_found:
                print(f"    Skipping this pool (GT duplicates exist)")
                continue
            
            query_data = {
                'queries': query_ids,
                'size': len(query_ids),
                'opt_level': opt_low,
                'pool_size': actual_pool_size,  # Use actual pool size
                'ground_truth': ground_truth_indices,
                'min_instructions': min_instructions,
                'filtered_trivial': True,
                'filtered_same_instruction': True,  # Always filter query=pool identical
                'unique_gt_guaranteed': True,  # Guarantee GT uniqueness (no identical instructions in pool)
                'metadata': {
                    'opt_pair': f'{opt_low}_vs_{opt_high}',
                    'avg_instructions': sum(p['low_num_instr'] for p in query_pairs) / len(query_pairs),
                    'gt_range': f'[{min(ground_truth_indices)}, {max(ground_truth_indices)}]',
                    'filtered_ambiguous_gt': filtered_ambiguous
                }
            }
            
            query_file = output_dir / f'query_{actual_query_size}_from_pool_{actual_pool_size}_{opt_low}_vs_{opt_high}_filtered.json'
            with open(query_file, 'w') as f:
                json.dump(query_data, f, indent=2)
            print(f"    ✓ Query saved: {query_file.name} ({actual_query_size} queries)")
    
    print(f"\n{'='*80}")
    print(f"Complete! All pools saved in: {output_dir}")
    print(f"{'='*80}")
    print(f"\n✅ Guarantees:")
    print(f"  1. Filtered trivial functions with instruction < {min_instructions}")
    print(f"  2. Query and GT have different instructions (same function, different optimization)")
    print(f"     - Example: main_O2 and main_O3 must have different instructions")
    print(f"     - Purpose: Test semantic understanding, not exact match memorization")
    print(f"  3. Each instruction in pool is unique (deduplicated)")
    print(f"  4. Each query's GT is absolutely unique in pool ⭐⭐⭐")
    print(f"     - No other functions in pool have same instruction as a query's GT")
    print(f"     - If GT is ambiguous, that pair won't be selected as query")
    print(f"  5. Maintain target pool size as much as possible (continue adding after dedup)")
    print(f"\nThese pools are suitable for evaluating model's true semantic understanding!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create filtered baseline pools')
    parser.add_argument('--func-blocks', type=str,
                        default='/data/kun/jtrans/baseline/eval/func_blocks_baseline.json')
    parser.add_argument('--ground-truth', type=str,
                        default='/data/kun/jtrans/baseline/eval/ground_truth_baseline.json')
    parser.add_argument('--output-dir', type=str,
                        default='/data/kun/jtrans/baseline/eval/pools_filtered')
    parser.add_argument('--min-instructions', type=int, default=11,
                        help='Minimum instruction count (filter trivial functions), default 11')
    
    args = parser.parse_args()
    
    create_filtered_pools(
        args.func_blocks,
        args.ground_truth,
        args.output_dir,
        min_instructions=args.min_instructions
    )
