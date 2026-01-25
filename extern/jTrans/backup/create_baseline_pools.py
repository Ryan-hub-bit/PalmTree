#!/usr/bin/env python3
"""
Create evaluation pools for baseline model.

Generates pool and query files for different optimization level pairs:
- O0 vs O3, O1 vs O3, O2 vs O3
- Pool sizes: 100, 1000, 10000
- Minimum instruction count: 10
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Configuration
GROUND_TRUTH_FILE = '/data/kun/jtrans/baseline/eval/ground_truth_baseline.json'
FUNC_BLOCKS_FILE = '/data/kun/jtrans/baseline/eval/func_blocks_baseline.json'
OUTPUT_DIR = '/data/kun/jtrans/baseline/eval/pools'

MIN_INSTRUCTIONS = 10
# Pool sizes for retrieval
POOL_SIZES = [100, 1000, 10000]
# Query size: 20% of pool size
QUERY_RATIO = 0.2
OPT_PAIRS = [
    ('O0', 'O3'),
    ('O1', 'O3'),
    ('O2', 'O3')
]

RANDOM_SEED = 42


def count_instructions(func_data):
    """Count instructions in a function."""
    instructions = func_data.get('instructions', func_data.get('tokens', ''))
    if isinstance(instructions, str):
        # Count tab-separated instructions
        return len(instructions.split('\t')) if instructions else 0
    return 0


def main():
    print("Loading ground truth...")
    with open(GROUND_TRUTH_FILE, 'r') as f:
        ground_truth_data = json.load(f)
        ground_truth = ground_truth_data['pairs']  # Extract the pairs list
    
    print("Loading function blocks...")
    with open(FUNC_BLOCKS_FILE, 'r') as f:
        func_blocks = json.load(f)
    
    print(f"Total function pairs: {len(ground_truth)}")
    print(f"Total function blocks: {len(func_blocks)}")
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # For each optimization pair
    for opt_low, opt_high in OPT_PAIRS:
        print(f"\n{'='*70}")
        print(f"Processing {opt_low} vs {opt_high}")
        print(f"{'='*70}")
        
        # Collect valid function pairs
        valid_pairs = []
        
        print("Scanning for valid function pairs...")
        # Ground truth is a list of entries: [{binary_name, function_name, O0: id, O1: id, ...}]
        for entry in tqdm(ground_truth, desc="Functions"):
            # Check if both optimization levels exist
            if opt_low not in entry or opt_high not in entry:
                continue
            
            func_id_low = str(entry[opt_low])
            func_id_high = str(entry[opt_high])
            
            # Check both functions exist in func_blocks
            if func_id_low not in func_blocks or func_id_high not in func_blocks:
                continue
            
            # Check instruction count
            count_low = func_blocks[func_id_low]['num_instructions']
            count_high = func_blocks[func_id_high]['num_instructions']
            
            if count_low >= MIN_INSTRUCTIONS and count_high >= MIN_INSTRUCTIONS:
                valid_pairs.append({
                    'binary': entry['binary_name'],
                    'function': entry['function_name'],
                    'id_low': func_id_low,
                    'id_high': func_id_high,
                    'opt_low': opt_low,
                    'opt_high': opt_high,
                        'count_low': count_low,
                        'count_high': count_high
                    })
        
        print(f"Found {len(valid_pairs)} valid function pairs")
        
        if len(valid_pairs) == 0:
            print(f"WARNING: No valid pairs found for {opt_low} vs {opt_high}")
            continue
        
        # Shuffle for randomness
        random.shuffle(valid_pairs)
        
        # Generate pools for different sizes
        for pool_size in POOL_SIZES:
            print(f"\n  Creating POOL with {pool_size} functions...")
            
            if len(valid_pairs) < pool_size:
                print(f"  WARNING: Only {len(valid_pairs)} pairs available, requested {pool_size}")
                actual_pool_size = len(valid_pairs)
            else:
                actual_pool_size = pool_size
            
            # Select pairs for this pool (FIXED for this pool size)
            pool_pairs = valid_pairs[:actual_pool_size]
            
            # Create pool: all high-opt functions (O3)
            pool_ids = [pair['id_high'] for pair in pool_pairs]
            
            # Save pool file (ONE pool file per size)
            pool_file = output_dir / f"pool_{actual_pool_size}_{opt_low}_vs_{opt_high}.json"
            pool_data = {
                'pool': pool_ids,
                'size': len(pool_ids),
                'opt_level': opt_high,
                'min_instructions': MIN_INSTRUCTIONS,
                'metadata': {
                    'opt_low': opt_low,
                    'opt_high': opt_high,
                    'created_from': str(GROUND_TRUTH_FILE)
                }
            }
            with open(pool_file, 'w') as f:
                json.dump(pool_data, f, indent=2)
            print(f"    Saved pool: {pool_file}")
            
            # Generate query set (20% of pool)
            query_size = int(actual_pool_size * QUERY_RATIO)
            print(f"    Creating QUERY set with {query_size} samples (20% of pool)...")
            
            # Randomly sample query_size pairs from pool_pairs
            query_pairs = random.sample(pool_pairs, query_size)
            
            # Create queries: low-opt functions (O0/O1/O2)
            query_ids = [pair['id_low'] for pair in query_pairs]
            query_high_ids = [pair['id_high'] for pair in query_pairs]
            
            # Ground truth: find where each query's match is in the pool
            ground_truth_indices = []
            for high_id in query_high_ids:
                # Find this high_id's position in pool
                idx = pool_ids.index(high_id)
                ground_truth_indices.append(idx)
            
            # Save query file
            query_file = output_dir / f"query_{query_size}_from_pool_{actual_pool_size}_{opt_low}_vs_{opt_high}.json"
            query_data = {
                'queries': query_ids,
                'size': len(query_ids),
                'opt_level': opt_low,
                'pool_size': actual_pool_size,
                'ground_truth': ground_truth_indices,
                'min_instructions': MIN_INSTRUCTIONS,
                'metadata': {
                    'opt_low': opt_low,
                    'opt_high': opt_high,
                    'pool_file': f"pool_{actual_pool_size}_{opt_low}_vs_{opt_high}.json",
                    'created_from': str(GROUND_TRUTH_FILE)
                }
            }
            with open(query_file, 'w') as f:
                json.dump(query_data, f, indent=2)
            print(f"      Saved query: {query_file}")
            
            # Statistics
            avg_count_low = sum(p['count_low'] for p in query_pairs) / len(query_pairs)
            avg_count_high = sum(p['count_high'] for p in query_pairs) / len(query_pairs)
            print(f"      Avg instructions: {opt_low}={avg_count_low:.1f}, {opt_high}={avg_count_high:.1f}")
            print(f"      Ground truth indices range: [{min(ground_truth_indices)}, {max(ground_truth_indices)}]")
    
    print(f"\n{'='*70}")
    print("Pool generation complete!")
    print(f"{'='*70}")
    print(f"\nGenerated files in: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.glob("*.json")):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
