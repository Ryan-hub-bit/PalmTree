#!/usr/bin/env python3
"""
Create baseline function dataset using jTrans tokenization.

Uses the SAME ground truth matching logic as address-aware version,
but applies baseline tokenization (JUMP_ADDR_X, var_xxx, etc.)

Outputs:
1. func_blocks_baseline.json - All functions with baseline tokenization
2. ground_truth_baseline.json - Same matching logic as address-aware
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import subprocess

# Add parent to path for readidadata
sys.path.insert(0, str(Path(__file__).parent.parent))
from readidadata import parse_asm


def parse_filename(filepath):
    """Parse filename to extract binary name and optimization level."""
    filename = Path(filepath).name
    
    if not filename.endswith('_extract.pkl'):
        return None
    
    base = filename[:-len('_extract.pkl')]
    
    # Pattern: name-OptLevel-hash
    pattern = r'^(.+?)-(O[0-3sgfast]+)-([0-9a-f]{32})$'
    match = re.match(pattern, base, re.IGNORECASE)
    
    if match:
        binary_name = match.group(1)
        opt_level = match.group(2)
        file_hash = match.group(3)
        return (binary_name, opt_level, file_hash)
    
    return None


def extract_function_names_from_binary(binary_path):
    """Extract function names from non-stripped binary using nm."""
    function_names = []
    
    try:
        result = subprocess.run(
            ['nm', '-n', str(binary_path)],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                parts = line.strip().split()
                if len(parts) >= 3 and parts[1] in ['T', 't']:
                    function_names.append(parts[2])
    except Exception as e:
        print(f"Error extracting symbols from {binary_path}: {e}", file=sys.stderr)
    
    return function_names


def tokenize_function_baseline(func_data):
    """
    Tokenize function using baseline jTrans approach.
    Returns space-separated token string.
    """
    asm_list = func_data.get('asm', [])
    
    tokens = []
    for asm_str in asm_list:
        try:
            # Use readidadata.parse_asm for baseline tokenization
            parsed = parse_asm(asm_str.strip())
            if parsed:
                tokens.append(parsed)
        except:
            continue
    
    return ' '.join(tokens)


def create_function_blocks_baseline(pickle_dir, binary_dir):
    """
    Create func_blocks_baseline.json with baseline tokenization.
    
    Returns: (func_blocks dict, mapping dict)
    """
    import pickle
    
    pickle_path = Path(pickle_dir)
    pickle_files = list(pickle_path.glob('*_extract.pkl'))
    
    func_blocks = {}
    mapping = defaultdict(lambda: defaultdict(dict))
    
    func_id = 0
    
    print(f"\nProcessing {len(pickle_files)} pickle files...")
    
    # Group by binary
    by_binary = defaultdict(list)
    for fpath in pickle_files:
        parsed = parse_filename(fpath)
        if parsed:
            binary_name, opt_level, file_hash = parsed
            by_binary[binary_name].append({
                'path': fpath,
                'opt': opt_level,
                'hash': file_hash
            })
    
    for binary_name, files in sorted(by_binary.items()):
        print(f"\n  Processing binary: {binary_name}")
        
        # Try to get function names from non-stripped binary
        function_names = None
        if binary_dir:
            binary_path = Path(binary_dir)
            for opt_info in files:
                opt_level = opt_info['opt']
                potential_binary = binary_path / f"{binary_name}-{opt_level}"
                
                if potential_binary.exists():
                    function_names = extract_function_names_from_binary(potential_binary)
                    if function_names:
                        print(f"    Found {len(function_names)} function names from {potential_binary.name}")
                        break
        
        # Process each optimization level
        for opt_info in sorted(files, key=lambda x: x['opt']):
            opt_level = opt_info['opt']
            fpath = opt_info['path']
            
            print(f"    {opt_level}: ", end='', flush=True)
            
            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                
                func_count = 0
                
                # Process each function in pickle
                for func_name, func_data in data.items():
                    # Skip special sections
                    if func_name in ['.plt', 'extern', '.init', '.fini']:
                        continue
                    
                    asm_list = func_data.get('asm', [])
                    
                    # Skip empty or very short functions
                    if len(asm_list) < 5:
                        continue
                    
                    # Tokenize using baseline approach
                    tokens = tokenize_function_baseline(func_data)
                    
                    if not tokens:
                        continue
                    
                    # Create function block entry
                    func_blocks[func_id] = {
                        'id': func_id,
                        'binary': binary_name,
                        'opt': opt_level,
                        'name': func_name,
                        'tokens': tokens,
                        'num_instructions': len(asm_list)
                    }
                    
                    # For ground truth matching
                    if function_names and func_name in function_names:
                        # Use actual function name for matching
                        mapping[binary_name][func_name][opt_level] = func_id
                    else:
                        # Use placeholder for matching (won't match across opts)
                        mapping[binary_name][f"func_{func_id}"][opt_level] = func_id
                    
                    func_id += 1
                    func_count += 1
                
                print(f"{func_count} functions")
                
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    print(f"\n✓ Total functions: {len(func_blocks)}")
    
    return func_blocks, mapping


def create_ground_truth_baseline(mapping):
    """
    Create ground truth pairs using same logic as address-aware.
    Only creates pairs where functions exist in multiple optimization levels.
    """
    ground_truth = {
        'pairs': [],
        'metadata': {
            'description': 'Baseline tokenization ground truth',
            'matching': 'Same function across optimization levels'
        }
    }
    
    pair_id = 0
    
    for binary_name, func_dict in mapping.items():
        for func_name, opt_dict in func_dict.items():
            # Only create pairs if function exists in multiple optimization levels
            if len(opt_dict) >= 2:
                opt_levels = sorted(opt_dict.keys())
                
                # Create all pairwise combinations
                for i in range(len(opt_levels)):
                    for j in range(i + 1, len(opt_levels)):
                        opt1 = opt_levels[i]
                        opt2 = opt_levels[j]
                        
                        func_id1 = opt_dict[opt1]
                        func_id2 = opt_dict[opt2]
                        
                        ground_truth['pairs'].append({
                            'pair_id': pair_id,
                            'func_id1': func_id1,
                            'func_id2': func_id2,
                            'binary': binary_name,
                            'func_name': func_name,
                            'opt1': opt1,
                            'opt2': opt2,
                            'label': 1  # Same function
                        })
                        
                        pair_id += 1
    
    print(f"\n✓ Created {len(ground_truth['pairs'])} ground truth pairs")
    
    return ground_truth


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create baseline function dataset with ground truth'
    )
    parser.add_argument('pickle_dir', help='Directory with *_extract.pkl files')
    parser.add_argument('output_dir', help='Output directory for dataset files')
    parser.add_argument('--binary-dir', help='Directory with non-stripped binaries (for function names)')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Creating Baseline Function Dataset")
    print("=" * 60)
    
    # Create function blocks with baseline tokenization
    func_blocks, mapping = create_function_blocks_baseline(
        args.pickle_dir,
        args.binary_dir
    )
    
    # Create ground truth pairs
    ground_truth = create_ground_truth_baseline(mapping)
    
    # Save outputs
    func_blocks_file = output_path / 'func_blocks_baseline.json'
    ground_truth_file = output_path / 'ground_truth_baseline.json'
    
    print(f"\nSaving files...")
    with open(func_blocks_file, 'w') as f:
        json.dump(func_blocks, f, indent=2)
    print(f"  ✓ {func_blocks_file}")
    
    with open(ground_truth_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"  ✓ {ground_truth_file}")
    
    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)
    print(f"Functions: {len(func_blocks)}")
    print(f"Ground truth pairs: {len(ground_truth['pairs'])}")
    print(f"\nUse same ground_truth pairs for fair comparison with address-aware version.")


if __name__ == '__main__':
    main()
