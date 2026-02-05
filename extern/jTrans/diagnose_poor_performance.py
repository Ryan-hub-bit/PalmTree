#!/usr/bin/env python3
"""
Diagnose why evaluation performance is poor.
"""

import json
import torch
import numpy as np
from pathlib import Path

def main():
    print("=" * 80)
    print("DIAGNOSING POOR EVALUATION PERFORMANCE")
    print("=" * 80)
    print()
    
    # Check 1: Load a pool and inspect it
    pool_file = '/data/kun/jtrans/addressaware/eval/pools_filtered/pool_O0_vs_O3_100.json'
    print(f"1. Checking pool structure: {pool_file}")
    with open(pool_file) as f:
        pool_data = json.load(f)
    
    print(f"   Pool keys: {list(pool_data.keys())}")
    print(f"   Pool size: {len(pool_data.get('pool', []))}")
    print(f"   Num queries: {len(pool_data.get('queries', []))}")
    
    if 'queries' in pool_data and len(pool_data['queries']) > 0:
        query = pool_data['queries'][0]
        print(f"   Sample query keys: {list(query.keys())}")
        print(f"   Query ID: {query.get('query_id')}")
        print(f"   GT ID: {query.get('gt_id')}")
        print(f"   Is GT in pool: {query.get('gt_id') in pool_data.get('pool', [])}")
    print()
    
    # Check 2: Verify func_blocks has both query and GT functions
    func_blocks_path = '/data/kun/jtrans/addressaware/eval/func_blocks_addr.json'
    print(f"2. Checking function blocks: {func_blocks_path}")
    with open(func_blocks_path) as f:
        func_blocks = json.load(f)
    
    print(f"   Total functions: {len(func_blocks)}")
    
    if 'queries' in pool_data and len(pool_data['queries']) > 0:
        query = pool_data['queries'][0]
        query_id = str(query.get('query_id'))
        gt_id = str(query.get('gt_id'))
        
        query_exists = query_id in func_blocks
        gt_exists = gt_id in func_blocks
        
        print(f"   Query {query_id} exists: {query_exists}")
        print(f"   GT {gt_id} exists: {gt_exists}")
        
        if query_exists and gt_exists:
            query_func = func_blocks[query_id]
            gt_func = func_blocks[gt_id]
            
            print(f"   Query opt: {query_func.get('optimization_level')}")
            print(f"   GT opt: {gt_func.get('optimization_level')}")
            
            # Check if they have instructions field
            query_has_instr = 'instructions' in query_func
            gt_has_instr = 'instructions' in gt_func
            
            print(f"   Query has 'instructions': {query_has_instr}")
            print(f"   GT has 'instructions': {gt_has_instr}")
            
            if query_has_instr:
                query_instr = query_func['instructions']
                num_instr = query_instr.count('\t') + 1
                print(f"   Query num instructions: {num_instr}")
                print(f"   Query length: {len(query_instr)} chars")
                print(f"   Query has position info: {'(' in query_instr and ':' in query_instr}")
            
            if gt_has_instr:
                gt_instr = gt_func['instructions']
                num_instr = gt_instr.count('\t') + 1
                print(f"   GT num instructions: {num_instr}")
                print(f"   GT length: {len(gt_instr)} chars")
                print(f"   GT has position info: {'(' in gt_instr and ':' in gt_instr}")
                
            # Check if they're actually from same binary/function
            print(f"   Query binary: {query_func.get('binary_name')}")
            print(f"   GT binary: {gt_func.get('binary_name')}")
            print(f"   Query function: {query_func.get('function_name')}")
            print(f"   GT function: {gt_func.get('function_name')}")
            print(f"   Same function: {query_func.get('function_name') == gt_func.get('function_name')}")
    print()
    
    # Check 3: Look at embeddings from latest evaluation
    print("3. Checking if we have recent evaluation results...")
    output_dir = Path('/home/kun/Document/AAE/output/jtrans')
    eval_files = sorted(output_dir.glob('addressaware_eval_*.json'), reverse=True)
    
    if eval_files:
        latest = eval_files[0]
        print(f"   Latest eval: {latest.name}")
        with open(latest) as f:
            results = json.load(f)
        
        if results:
            first_result = results[0] if isinstance(results, list) else results
            print(f"   Results keys: {list(first_result.keys())}")
            print(f"   MRR: {first_result.get('MRR', 'N/A')}")
            print(f"   Recall@1: {first_result.get('Recall@1', 'N/A')}")
            print(f"   Recall@10: {first_result.get('Recall@10', 'N/A')}")
    else:
        print("   No evaluation results found")
    print()
    
    # Check 4: Compare with expected baselines
    print("4. Performance comparison:")
    print("   Your results (O0→O3, 100 pool):")
    print("     - Recall@1: 15%")
    print("     - Recall@10: 45%")
    print()
    print("   Expected performance:")
    print("     - Recall@1: 45-60%")
    print("     - Recall@10: 70-85%")
    print()
    print("   Performance gap: ~3-4x worse than expected")
    print()
    
    print("=" * 80)
    print("POSSIBLE CAUSES:")
    print("=" * 80)
    print("1. Model didn't train properly (check training logs)")
    print("2. Evaluation pools are too hard (functions are very different)")
    print("3. Data leakage between train/eval (need to compare pretrained vs finetuned)")
    print("4. Hyperparameters need tuning (learning rate, margin, epochs)")
    print("5. Projection layer losing information (try without projection)")
    print()
    print("NEXT STEPS:")
    print("1. Evaluate pretrained model (baseline)")
    print("2. Check training loss convergence")
    print("3. Try different hyperparameters")
    print("4. Inspect actual embedding similarities")

if __name__ == '__main__':
    main()
