#!/usr/bin/env python3
"""
Generate Function Similarity Pairs from Deduplicated Data

This script takes the combined_deduplicated.json and creates:
1. function_blocks.json - Maps function block IDs to their instruction content
2. funcsim_pairs.json - Ground truth pairs for function similarity

For each optimization level, the ground truth pairs are the other 3 optimization levels
of the same function:
- O0 function -> ground truth: O1, O2, O3 of same function
- O1 function -> ground truth: O0, O2, O3 of same function
- O2 function -> ground truth: O0, O1, O3 of same function
- O3 function -> ground truth: O0, O1, O2 of same function

Usage:
    python generate_funcsim_pairs.py /data/kun/funcsim_match/combined_deduplicated.json [missing_tokens_report.json]
"""

import json
import sys
import os
from typing import Dict, List, Set, Optional


def load_missing_tokens(report_path: str) -> Set[str]:
    """
    Load global missing tokens from the missing tokens report.
    
    Args:
        report_path: Path to missing_tokens_report.json
        
    Returns:
        Set of missing token strings
    """
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    missing_tokens = set(report.get("global_missing_tokens", {}).keys())
    print(f"[INFO] Loaded {len(missing_tokens)} global missing tokens from report")
    return missing_tokens


def function_has_missing_tokens(instructions: List[str], missing_tokens: Set[str]) -> bool:
    """
    Check if any instruction contains a missing token.
    
    Args:
        instructions: List of instruction strings
        missing_tokens: Set of missing tokens to check
        
    Returns:
        True if any instruction contains a missing token
    """
    for instr in instructions:
        # Split instruction into tokens and check each
        tokens = instr.split()
        for token in tokens:
            if token in missing_tokens:
                return True
    return False


def generate_function_id(base_key: str, opt_level: str, func_idx: int) -> str:
    """
    Generate unique function block ID.
    
    Format: {base_index * 4 + opt_offset}
    Where opt_offset: O0=1, O1=2, O2=3, O3=4
    
    This creates sequential IDs: 1, 2, 3, 4 (for func 1), 5, 6, 7, 8 (for func 2), etc.
    
    Args:
        base_key: Original function key (not used)
        opt_level: Optimization level (O0, O1, O2, O3)
        func_idx: Function index (1-based)
        
    Returns:
        String ID like "1", "2", "3", etc.
    """
    opt_offset = {'O0': 1, 'O1': 2, 'O2': 3, 'O3': 4}[opt_level]
    block_id = (func_idx - 1) * 4 + opt_offset
    return str(block_id)


def create_function_blocks_and_pairs(deduplicated_file: str, missing_tokens: Optional[Set[str]] = None) -> tuple:
    """
    Create function blocks and similarity pairs from deduplicated data.
    
    Args:
        deduplicated_file: Path to combined_deduplicated.json
        missing_tokens: Optional set of tokens to filter out. If any optimization
                        level of a function contains a missing token, the entire
                        function (all 4 opt levels) is skipped.
        
    Returns:
        Tuple of (function_blocks, funcsim_pairs)
    """
    # Load deduplicated data
    print(f"[INFO] Loading deduplicated data from {deduplicated_file}")
    with open(deduplicated_file, 'r') as f:
        dedup_data = json.load(f)
    
    print(f"[INFO] Loaded {len(dedup_data)} unique functions")
    
    function_blocks = {}
    funcsim_pairs = {}
    
    opt_levels = ['O0', 'O1', 'O2', 'O3']
    
    # Statistics for filtering
    skipped_count = 0
    processed_count = 0
    
    # Process each function (1-based indexing for output IDs)
    output_func_idx = 0  # Will be incremented only for functions we keep
    for func_idx, (func_key, func_data) in enumerate(dedup_data.items(), start=1):
        project = func_data['project']
        binary = func_data['binary']
        func_name = func_data['function_name']
        
        # Check if any optimization level contains missing tokens
        if missing_tokens:
            has_missing = False
            for opt in opt_levels:
                if function_has_missing_tokens(func_data[opt], missing_tokens):
                    has_missing = True
                    break
            
            if has_missing:
                skipped_count += 1
                if skipped_count <= 10:
                    print(f"[FILTER] Skipping function '{func_name}' from {project}/{binary} (contains missing tokens)")
                elif skipped_count == 11:
                    print(f"[FILTER] ... (suppressing further skip messages)")
                continue
        
        # Increment output function index only for functions we keep
        output_func_idx += 1
        processed_count += 1
        
        # Create function block IDs for each optimization level
        func_ids = {}
        for opt in opt_levels:
            func_id = generate_function_id(func_key, opt, output_func_idx)
            func_ids[opt] = func_id
            
            # Store function block with ONLY instructions (no metadata)
            function_blocks[func_id] = func_data[opt]
        
        # Create ground truth pairs for each optimization level
        for opt in opt_levels:
            current_func_id = func_ids[opt]
            
            # Ground truth: all other optimization levels of the same function
            ground_truth = [func_ids[other_opt] for other_opt in opt_levels if other_opt != opt]
            
            funcsim_pairs[current_func_id] = {
                'function_id': current_func_id,
                'index': func_idx,
                'project': project,
                'binary': binary,
                'function_name': func_name,
                'opt_level': opt,
                'ground_truth': ground_truth
            }
        
        if output_func_idx % 100 == 0:
            print(f"[INFO] Processed {output_func_idx} valid functions (scanned {func_idx} total)...")
    
    print(f"\n[SUMMARY]")
    print(f"  Total unique functions in input: {len(dedup_data)}")
    if missing_tokens:
        print(f"  Functions skipped (missing tokens): {skipped_count}")
        print(f"  Functions kept: {processed_count}")
    print(f"  Function IDs: 1 to {len(function_blocks)} (sequential numbering)")
    print(f"  Total function blocks: {len(function_blocks)} (4 opt levels × {processed_count} functions)")
    print(f"  Total similarity pairs: {len(funcsim_pairs)}")
    print(f"  Each function has 3 ground truth matches")
    print(f"  ID mapping: func_N -> IDs [(N-1)*4+1, (N-1)*4+2, (N-1)*4+3, (N-1)*4+4] for [O0, O1, O2, O3]")
    
    return function_blocks, funcsim_pairs


def save_outputs(function_blocks: Dict, funcsim_pairs: Dict, output_dir: str):
    """
    Save function blocks and similarity pairs to JSON files.
    
    Args:
        function_blocks: Dictionary of function block ID to content
        funcsim_pairs: Dictionary of function ID to ground truth pairs
        output_dir: Directory to save output files
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save function_blocks.json
    blocks_file = os.path.join(output_dir, "function_blocks.json")
    with open(blocks_file, 'w') as f:
        json.dump(function_blocks, f, indent=2)
    
    blocks_size_mb = os.path.getsize(blocks_file) / (1024 * 1024)
    print(f"\n[SAVED] function_blocks.json")
    print(f"  Path: {blocks_file}")
    print(f"  Entries: {len(function_blocks)}")
    print(f"  Size: {blocks_size_mb:.2f} MB")
    
    # Save funcsim_pairs.json
    pairs_file = os.path.join(output_dir, "funcsim_pairs.json")
    with open(pairs_file, 'w') as f:
        json.dump(funcsim_pairs, f, indent=2)
    
    pairs_size_mb = os.path.getsize(pairs_file) / (1024 * 1024)
    print(f"\n[SAVED] funcsim_pairs.json")
    print(f"  Path: {pairs_file}")
    print(f"  Entries: {len(funcsim_pairs)}")
    print(f"  Size: {pairs_size_mb:.2f} MB")
    
    # Print example
    print(f"\n[EXAMPLE] Function block structure:")
    if function_blocks:
        example_id = list(function_blocks.keys())[0]
        example_block = function_blocks[example_id]
        print(f"  ID: {example_id}")
        print(f"  Content: List of {len(example_block)} instructions")
        print(f"  First instruction: {example_block[0] if example_block else 'N/A'}")
    
    print(f"\n[EXAMPLE] Similarity pair structure:")
    if funcsim_pairs:
        example_id = list(funcsim_pairs.keys())[0]
        example_pair = funcsim_pairs[example_id]
        print(f"  Function ID: {example_pair['function_id']}")
        print(f"  Opt Level: {example_pair['opt_level']}")
        print(f"  Ground Truth: {example_pair['ground_truth']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_funcsim_pairs.py <combined_deduplicated.json> [missing_tokens_report.json]")
        print("Example: python generate_funcsim_pairs.py /data/kun/funcsim_match/combined_deduplicated.json")
        print("Example with filtering: python generate_funcsim_pairs.py /data/kun/funcsim_match/combined_deduplicated.json /data/kun/funcsim_match/missing_tokens_report.json")
        sys.exit(1)
    
    deduplicated_file = sys.argv[1]
    missing_tokens_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(deduplicated_file):
        print(f"[ERROR] File not found: {deduplicated_file}")
        sys.exit(1)
    
    # Load missing tokens if provided
    missing_tokens = None
    if missing_tokens_file:
        if not os.path.exists(missing_tokens_file):
            print(f"[ERROR] Missing tokens file not found: {missing_tokens_file}")
            sys.exit(1)
        missing_tokens = load_missing_tokens(missing_tokens_file)
    
    # Output to same directory as input file
    output_dir = os.path.dirname(deduplicated_file)
    
    print("=" * 80)
    print("Generate Function Similarity Pairs")
    print("=" * 80)
    print(f"Input file: {deduplicated_file}")
    if missing_tokens_file:
        print(f"Missing tokens filter: {missing_tokens_file}")
    else:
        print(f"Missing tokens filter: None (no filtering)")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Create function blocks and pairs
    print("\n[STEP 1] Creating function blocks and similarity pairs...")
    function_blocks, funcsim_pairs = create_function_blocks_and_pairs(deduplicated_file, missing_tokens)
    
    # Save outputs
    print("\n[STEP 2] Saving outputs...")
    save_outputs(function_blocks, funcsim_pairs, output_dir)
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)
    print("\nOutput files:")
    print(f"  1. {output_dir}/function_blocks.json - Function block ID to instructions")
    print(f"  2. {output_dir}/funcsim_pairs.json - Ground truth similarity pairs")
    print("=" * 80)


if __name__ == '__main__':
    main()
