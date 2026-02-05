#!/usr/bin/env python3
"""
Create filtered address-aware evaluation pools

Same filtering strategy as baseline:
1. Exclude trivial functions with instruction count <= 10
2. Exclude pairs where query and GT have identical instructions
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
    Create filtered evaluation pools for address-aware model
    
    Args:
        func_blocks_path: Path to func_blocks_addr.json
        ground_truth_path: Path to ground_truth_addr.json
        output_dir: Output directory
        pool_sizes: List of pool sizes
        query_ratio: Ratio of queries to pool size
        min_instructions: Minimum instruction count (filter trivial functions)
        filter_trivial_pairs: Whether to filter identical instruction pairs
    """
    print("="*80)
    print("Creating Filtered Address-Aware Evaluation Pools")
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
            # Count instructions from 'instructions' field (tab-separated)
            if 'instructions' in low_func:
                low_num_instr = low_func['instructions'].count('\t') + 1
            elif 'num_instructions' in low_func:
                low_num_instr = low_func['num_instructions']
            else:
                low_num_instr = 0
            
            if 'instructions' in high_func:
                high_num_instr = high_func['instructions'].count('\t') + 1
            elif 'num_instructions' in high_func:
                high_num_instr = high_func['num_instructions']
            else:
                high_num_instr = 0
            
            if low_num_instr < min_instructions or high_num_instr < min_instructions:
                trivial_count += 1
                continue
            
            # Get instructions for hash computation (prefer 'instructions' over 'tokens')
            if 'instructions' in low_func:
                low_tokens = low_func['instructions']
            elif 'tokens' in low_func:
                low_tokens = low_func['tokens']
            else:
                low_tokens = ''
            
            if 'instructions' in high_func:
                high_tokens = high_func['instructions']
            elif 'tokens' in high_func:
                high_tokens = high_func['tokens']
            else:
                high_tokens = ''
            
            # Check if tokens are empty
            if not low_tokens or not high_tokens:
                trivial_count += 1
                continue
            
            # Filter 2: Query and GT have identical instructions
            low_hash = compute_instruction_hash(low_tokens)
            high_hash = compute_instruction_hash(high_tokens)
            
            if filter_trivial_pairs and low_hash == high_hash:
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
        print(f"  Filtered identical instructions: {same_instruction_count:,}")
        print(f"  Remaining valid pairs: {len(valid_pairs):,}")
        
        if len(valid_pairs) == 0:
            print(f"  ⚠️  No valid pairs, skipping")
            continue
        
        # Random shuffle
        random.shuffle(valid_pairs)
        
        # Create pools for each size
        for pool_size in pool_sizes:
            if pool_size > len(valid_pairs):
                print(f"\n  ⚠️  Pool size {pool_size} > valid pairs {len(valid_pairs)}, skipping")
                continue
            
            print(f"\n  Creating pool size {pool_size}...")
            
            # Select pool pairs with deduplication
            pool_pairs = []
            seen_hashes = set()
            idx = 0
            
            while len(pool_pairs) < pool_size and idx < len(valid_pairs):
                pair = valid_pairs[idx]
                idx += 1
                
                # Deduplication: Skip if high_id (GT) already in pool
                if pair['high_hash'] in seen_hashes:
                    continue
                
                pool_pairs.append(pair)
                seen_hashes.add(pair['low_hash'])
                seen_hashes.add(pair['high_hash'])
            
            if len(pool_pairs) < pool_size:
                print(f"    Warning: Only got {len(pool_pairs)} unique pairs (target: {pool_size})")
            
            # Calculate query count
            query_count = max(10, int(pool_size * query_ratio))
            
            # Ensure query count doesn't exceed pool size
            query_count = min(query_count, len(pool_pairs))
            
            # Select queries (first query_count pairs)
            queries = pool_pairs[:query_count]
            
            # Create pool file
            pool_file = output_dir / f"pool_{opt_low}_vs_{opt_high}_{pool_size}.json"
            pool_data = {
                'pool_size': pool_size,
                'actual_size': len(pool_pairs),
                'query_count': len(queries),
                'opt_pair': f'{opt_low}_vs_{opt_high}',
                'pool': [
                    {
                        'low_id': p['low_id'],
                        'high_id': p['high_id'],
                        'binary': p['binary'],
                        'function': p['function'],
                        'low_num_instr': p['low_num_instr'],
                        'high_num_instr': p['high_num_instr']
                    }
                    for p in pool_pairs
                ],
                'queries': [
                    {
                        'query_id': q['low_id'],
                        'gt_id': q['high_id'],
                        'binary': q['binary'],
                        'function': q['function']
                    }
                    for q in queries
                ]
            }
            
            with open(pool_file, 'w') as f:
                json.dump(pool_data, f, indent=2)
            
            print(f"    ✓ Saved: {pool_file}")
            print(f"      Pool: {len(pool_pairs)} pairs")
            print(f"      Queries: {len(queries)} queries")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create filtered address-aware evaluation pools')
    parser.add_argument('--func_blocks', required=True, help='Path to func_blocks_addr.json')
    parser.add_argument('--ground_truth', required=True, help='Path to ground_truth_addr.json')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--pool_sizes', nargs='+', type=int, default=[100, 1000, 10000],
                       help='Pool sizes to create')
    parser.add_argument('--query_ratio', type=float, default=0.2,
                       help='Ratio of queries to pool size')
    parser.add_argument('--min_instructions', type=int, default=11,
                       help='Minimum instruction count (filter trivial functions)')
    parser.add_argument('--no_filter_trivial', action='store_true',
                       help='Disable filtering of identical instruction pairs')
    
    args = parser.parse_args()
    
    create_filtered_pools(
        func_blocks_path=args.func_blocks,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        pool_sizes=args.pool_sizes,
        query_ratio=args.query_ratio,
        min_instructions=args.min_instructions,
        filter_trivial_pairs=not args.no_filter_trivial
    )
    
    print("\n" + "="*80)
    print("Pool creation completed!")
    print("="*80)
