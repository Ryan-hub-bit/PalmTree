#!/usr/bin/env python3
"""
Create a small subset of data for testing the finetune pipeline.
Extracts first N function groups with all their optimization variants.
"""

import json
import sys

def create_test_subset(n_functions=100):
    """Create small test files with first N function groups."""
    
    print(f"Creating test subset with {n_functions} function groups...")
    
    # Load full data
    print("Loading ground truth...")
    with open('/data/kun/jtransdata/ground_truth_baseline.json', 'r') as f:
        gt = json.load(f)
    
    print("Loading function blocks...")
    with open('/data/kun/jtransdata/func_blocks_baseline.json', 'r') as f:
        fb = json.load(f)
    
    # Take first N function groups
    test_pairs = gt['pairs'][:n_functions]
    print(f"Selected {len(test_pairs)} function groups")
    
    # Collect all function IDs needed
    func_ids_needed = set()
    for pair in test_pairs:
        for opt in ['O0', 'O1', 'O2', 'O3', 'Os']:
            if opt in pair:
                func_ids_needed.add(str(pair[opt]))
    
    print(f"Need {len(func_ids_needed)} function blocks")
    
    # Create subset of func_blocks
    test_fb = {}
    for fid in func_ids_needed:
        if fid in fb:
            test_fb[fid] = fb[fid]
    
    print(f"Extracted {len(test_fb)} function blocks")
    
    # Create test ground truth
    test_gt = {
        'pairs': test_pairs,
        'total_pairs': len(test_pairs),
        'statistics': {
            'note': 'Test subset for pipeline verification'
        }
    }
    
    # Save test files
    test_fb_path = '/tmp/func_blocks_baseline_test.json'
    test_gt_path = '/tmp/ground_truth_baseline_test.json'
    
    print(f"Saving test func_blocks to {test_fb_path}...")
    with open(test_fb_path, 'w') as f:
        json.dump(test_fb, f)
    
    print(f"Saving test ground_truth to {test_gt_path}...")
    with open(test_gt_path, 'w') as f:
        json.dump(test_gt, f)
    
    print("\n" + "="*60)
    print("✓ Test data created successfully!")
    print("="*60)
    print(f"Function groups: {len(test_pairs)}")
    print(f"Function blocks: {len(test_fb)}")
    print(f"Files saved to:")
    print(f"  - {test_fb_path}")
    print(f"  - {test_gt_path}")
    print("\nTo use in finetune, update run_finetune_baseline.sh:")
    print(f"  --func_blocks {test_fb_path}")
    print(f"  --ground_truth {test_gt_path}")
    print("="*60)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    create_test_subset(n)
