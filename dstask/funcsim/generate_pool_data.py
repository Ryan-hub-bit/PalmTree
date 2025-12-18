"""
Generate evaluation pool data files for fair model comparison.

Creates two files:
1. pool_ids_10k.json - Pool function IDs and query IDs
2. pool_function_blocks_10k.json - Function blocks for pool (only 10k functions)
"""

import json
import random
import argparse
import os
from datetime import datetime


def generate_pool_data(function_blocks_file, funcsim_pairs_file, pool_size, output_dir, seed=42):
    """
    Generate pool data files.
    
    Args:
        function_blocks_file: Path to function_blocks.json
        funcsim_pairs_file: Path to funcsim_pairs.json
        pool_size: Total number of functions in the pool
        output_dir: Directory to save output files
        seed: Random seed for reproducibility
    """
    # Load data
    print(f"Loading function blocks from {function_blocks_file}...")
    with open(function_blocks_file, 'r') as f:
        function_blocks = json.load(f)
    print(f"Loaded {len(function_blocks)} function blocks")
    
    print(f"Loading funcsim pairs from {funcsim_pairs_file}...")
    with open(funcsim_pairs_file, 'r') as f:
        funcsim_pairs = json.load(f)
    print(f"Loaded {len(funcsim_pairs)} query functions")
    
    # Set random seed
    random.seed(seed)
    
    all_func_ids = list(function_blocks.keys())
    all_query_ids = list(funcsim_pairs.keys())
    
    # Sample queries first (aim for ~10% of pool size for queries)
    num_queries = min(max(100, pool_size // 10), len(all_query_ids))
    random.shuffle(all_query_ids)
    sampled_query_ids = all_query_ids[:num_queries]
    
    print(f"\nSampling {num_queries} queries from {len(all_query_ids)} total...")
    
    # Collect sampled queries + their ground truth (required functions)
    required_func_ids = set(sampled_query_ids)
    for query_id in sampled_query_ids:
        if query_id in funcsim_pairs:
            for gt_id in funcsim_pairs[query_id]['ground_truth']:
                if gt_id in function_blocks:
                    required_func_ids.add(gt_id)
    
    print(f"Required functions (queries + ground truth): {len(required_func_ids)}")
    
    # Add random distractors to fill pool
    remaining = [fid for fid in all_func_ids if fid not in required_func_ids]
    random.shuffle(remaining)
    num_distractors = max(0, pool_size - len(required_func_ids))
    
    if len(required_func_ids) > pool_size:
        print(f"WARNING: Required functions ({len(required_func_ids)}) exceed pool_size ({pool_size})")
        print(f"Using all required functions. Consider increasing pool_size.")
    
    selected_func_ids = list(required_func_ids) + remaining[:num_distractors]
    
    print(f"Adding {num_distractors} random distractors")
    print(f"Total pool size: {len(selected_func_ids)} functions")
    
    # Filter function blocks to pool only
    print(f"\nExtracting function blocks for pool...")
    pool_function_blocks = {fid: function_blocks[fid] for fid in selected_func_ids if fid in function_blocks}
    
    # Create pool IDs data
    pool_ids_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'seed': seed,
            'pool_size': len(selected_func_ids),
            'num_queries': len(sampled_query_ids),
            'num_required': len(required_func_ids),
            'num_distractors': num_distractors
        },
        'query_ids': sampled_query_ids,
        'pool_function_ids': selected_func_ids
    }
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    
    pool_ids_file = os.path.join(output_dir, f'pool_ids_{pool_size//1000}k.json')
    pool_blocks_file = os.path.join(output_dir, f'pool_function_blocks_{pool_size//1000}k.json')
    
    print(f"\nSaving pool IDs to {pool_ids_file}...")
    with open(pool_ids_file, 'w') as f:
        json.dump(pool_ids_data, f, indent=2)
    
    print(f"Saving pool function blocks to {pool_blocks_file}...")
    with open(pool_blocks_file, 'w') as f:
        json.dump(pool_function_blocks, f)
    
    print("\n" + "="*60)
    print("Pool Data Generated Successfully!")
    print("="*60)
    print(f"Pool size: {len(selected_func_ids)} functions")
    print(f"Queries: {len(sampled_query_ids)}")
    print(f"Required (queries + ground truth): {len(required_func_ids)}")
    print(f"Distractors: {num_distractors}")
    print(f"\nOutput files:")
    print(f"  IDs:     {pool_ids_file}")
    print(f"  Blocks:  {pool_blocks_file}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate pool data files for evaluation")
    
    parser.add_argument("--function_blocks", type=str, required=True,
                       help="Path to function_blocks.json")
    parser.add_argument("--funcsim_pairs", type=str, required=True,
                       help="Path to funcsim_pairs.json")
    parser.add_argument("--pool_size", type=int, default=10000,
                       help="Target pool size (default: 10000)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for pool files")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    generate_pool_data(
        function_blocks_file=args.function_blocks,
        funcsim_pairs_file=args.funcsim_pairs,
        pool_size=args.pool_size,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
