#!/usr/bin/env python3
"""
Remove cross-binary duplicate functions from training and evaluation data.

Functions that share the same name across different binaries (e.g., compiler stubs
like _init, register_tm_clones, frame_dummy) have near-identical code. This causes:
  - Training confusion: model sees nearly identical code as different functions
  - Evaluation unfairness: model penalized for matching equivalent code from wrong binary

This script:
  1. Filters ground_truth_addr.json → removes ALL pairs with cross-binary duplicate names
  2. Filters eval pool files → removes pool entries and queries with duplicate names
  3. Does NOT modify func_blocks_addr.json (too large; unused entries are harmless)

Usage:
  # Dedup training ground truth
  python dedup_cross_binary.py --mode train \
      --ground_truth /path/to/ground_truth_addr.json \
      --output_dir /path/to/output

  # Dedup eval pools
  python dedup_cross_binary.py --mode eval \
      --ground_truth /path/to/eval/ground_truth_addr.json \
      --pool_dir /path/to/eval/pools_filtered \
      --output_dir /path/to/output

  # Dedup both
  python dedup_cross_binary.py --mode both \
      --ground_truth /path/to/ground_truth_addr.json \
      --eval_ground_truth /path/to/eval/ground_truth_addr.json \
      --pool_dir /path/to/eval/pools_filtered \
      --output_dir /path/to/output
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict


def find_cross_binary_duplicates(gt_data):
    """Find function names that appear in multiple binaries."""
    func_to_bins = defaultdict(set)
    for p in gt_data['pairs']:
        func_to_bins[p['function_name']].add(p['binary_name'])
    
    dup_funcs = {fn for fn, bins in func_to_bins.items() if len(bins) > 1}
    return dup_funcs


def dedup_ground_truth(gt_path, output_path):
    """Filter ground truth: remove pairs whose function_name appears in >1 binary."""
    print(f"\n{'='*60}")
    print(f"Deduplicating ground truth: {gt_path}")
    print(f"{'='*60}")
    
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    
    total = len(gt['pairs'])
    dup_funcs = find_cross_binary_duplicates(gt)
    
    # Filter
    clean_pairs = [p for p in gt['pairs'] if p['function_name'] not in dup_funcs]
    removed = total - len(clean_pairs)
    
    # Collect removed IDs for reference
    removed_ids = set()
    for p in gt['pairs']:
        if p['function_name'] in dup_funcs:
            for key in ['O0', 'Os', 'O1', 'O2', 'O3']:
                if key in p and p[key] is not None:
                    removed_ids.add(p[key])
    
    print(f"  Total pairs:           {total}")
    print(f"  Cross-binary dup names: {len(dup_funcs)}")
    print(f"  Pairs removed:         {removed} ({removed/total*100:.1f}%)")
    print(f"  Pairs remaining:       {len(clean_pairs)}")
    print(f"  Function IDs removed:  {len(removed_ids)}")
    
    # Show top removed function names
    func_counts = defaultdict(int)
    for p in gt['pairs']:
        if p['function_name'] in dup_funcs:
            func_counts[p['function_name']] += 1
    top_removed = sorted(func_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top removed functions:")
    for fn, cnt in top_removed:
        print(f"    {fn}: {cnt} pairs")
    
    # Build new ground truth
    new_gt = {
        'pairs': clean_pairs,
        'total_pairs': len(clean_pairs),
        'statistics': gt.get('statistics', {}),
        'dedup_info': {
            'original_pairs': total,
            'removed_pairs': removed,
            'cross_binary_duplicate_names': len(dup_funcs),
            'removed_function_ids': len(removed_ids),
        }
    }
    
    # Update statistics if present
    if 'statistics' in new_gt and isinstance(new_gt['statistics'], dict):
        new_gt['statistics']['deduped'] = True
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(new_gt, f, indent=None)  # compact to save space
    
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path)/1e6:.1f} MB")
    
    return dup_funcs, removed_ids


def dedup_pools(pool_dir, output_dir, dup_funcs=None, gt_path=None):
    """Filter eval pools: remove entries with cross-binary duplicate function names.
    
    If dup_funcs is not provided, will compute from gt_path.
    """
    print(f"\n{'='*60}")
    print(f"Deduplicating eval pools: {pool_dir}")
    print(f"{'='*60}")
    
    # Get duplicate function names from eval ground truth if needed
    if dup_funcs is None:
        if gt_path is None:
            raise ValueError("Must provide either dup_funcs or gt_path")
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        dup_funcs = find_cross_binary_duplicates(gt)
    
    print(f"  Cross-binary duplicate function names: {len(dup_funcs)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    pool_files = sorted(Path(pool_dir).glob("pool_*.json"))
    print(f"  Pool files found: {len(pool_files)}")
    
    for pool_file in pool_files:
        print(f"\n  --- {pool_file.name} ---")
        
        with open(pool_file, 'r') as f:
            pool = json.load(f)
        
        orig_pool_size = len(pool['pool'])
        orig_query_count = len(pool['queries'])
        
        # Filter pool entries: remove those with duplicate function names
        clean_pool = [p for p in pool['pool'] if p['function'] not in dup_funcs]
        
        # Collect remaining high_ids for query filtering
        remaining_high_ids = {p['high_id'] for p in clean_pool}
        
        # Filter queries: remove those whose function is duplicated OR whose gt_id was removed
        clean_queries = []
        for q in pool['queries']:
            if q['function'] in dup_funcs:
                continue  # query itself is a dup function
            if q['gt_id'] not in remaining_high_ids:
                continue  # ground truth target was removed
            clean_queries.append(q)
        
        removed_pool = orig_pool_size - len(clean_pool)
        removed_queries = orig_query_count - len(clean_queries)
        
        print(f"    Pool: {orig_pool_size} → {len(clean_pool)} (removed {removed_pool})")
        print(f"    Queries: {orig_query_count} → {len(clean_queries)} (removed {removed_queries})")
        
        # Update pool metadata
        new_pool = {
            'pool_size': pool.get('pool_size', orig_pool_size),
            'actual_size': len(clean_pool),
            'query_count': len(clean_queries),
            'opt_pair': pool.get('opt_pair', ''),
            'pool': clean_pool,
            'queries': clean_queries,
            'dedup_info': {
                'original_pool_size': orig_pool_size,
                'original_query_count': orig_query_count,
                'removed_pool_entries': removed_pool,
                'removed_queries': removed_queries,
            }
        }
        
        out_path = os.path.join(output_dir, pool_file.name)
        with open(out_path, 'w') as f:
            json.dump(new_pool, f, indent=None)
        
        print(f"    Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Remove cross-binary duplicate functions')
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], required=True,
                        help='What to dedup: train (ground truth), eval (pools), or both')
    parser.add_argument('--ground_truth', help='Training ground_truth_addr.json path')
    parser.add_argument('--eval_ground_truth', help='Eval ground_truth_addr.json path (for mode=both)')
    parser.add_argument('--pool_dir', help='Eval pools directory')
    parser.add_argument('--output_dir', required=True, help='Output directory for deduped files')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ('train', 'both'):
        if not args.ground_truth:
            parser.error("--ground_truth required for mode=train/both")
        gt_out = os.path.join(args.output_dir, 'ground_truth_addr_dedup.json')
        train_dup_funcs, _ = dedup_ground_truth(args.ground_truth, gt_out)
    
    if args.mode in ('eval', 'both'):
        if not args.pool_dir:
            parser.error("--pool_dir required for mode=eval/both")
        
        # For eval, use eval ground truth to find dups
        eval_gt = args.eval_ground_truth or args.ground_truth
        if not eval_gt:
            parser.error("Need --ground_truth or --eval_ground_truth for eval dedup")
        
        pool_out = os.path.join(args.output_dir, 'pools_dedup')
        dedup_pools(args.pool_dir, pool_out, gt_path=eval_gt)
    
    print(f"\n{'='*60}")
    print("Done! Deduplication complete.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
