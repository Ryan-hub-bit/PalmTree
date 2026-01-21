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
import importlib.util
import os

# Load readidadata module from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
readidadata_path = os.path.join(parent_dir, 'readidadata.py')

spec = importlib.util.spec_from_file_location("readidadata", readidadata_path)
readidadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readidadata)


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


def build_instruction_address_map(func_data):
    """
    Build mapping from basic block addresses to instruction positions.
    Uses CFG data from pkl file.
    
    Returns:
        Tuple of (addr_to_pos dict, total_instructions)
    """
    cfg = func_data.get('cfg')
    
    if cfg is None:
        return {}, 0
    
    try:
        import networkx as nx
        if not isinstance(cfg, nx.DiGraph):
            return {}, 0
    except ImportError:
        return {}, 0
    
    addr_to_pos = {}
    current_pos = 0
    
    # Sort basic blocks by address
    sorted_blocks = sorted(cfg.nodes())
    
    for block_addr in sorted_blocks:
        block_data = cfg.nodes.get(block_addr, {})
        bb_asm = block_data.get('asm', [])
        
        # Map this block's start address to current position
        addr_to_pos[block_addr] = current_pos
        current_pos += len(bb_asm)
    
    return addr_to_pos, current_pos


def find_jump_targets(func_data):
    """
    Extract jump instruction positions and their target positions using CFG.
    
    Returns:
        Dict mapping instruction_position -> target_instruction_position
    """
    cfg = func_data.get('cfg')
    
    if cfg is None:
        return {}
    
    try:
        import networkx as nx
        if not isinstance(cfg, nx.DiGraph):
            return {}
    except ImportError:
        return {}
    
    addr_to_pos, _ = build_instruction_address_map(func_data)
    jump_map = {}
    
    # Analyze CFG edges to find jumps
    for src_addr, tgt_addr in cfg.edges():
        src_data = cfg.nodes.get(src_addr, {})
        bb_asm = src_data.get('asm', [])
        
        if not bb_asm:
            continue
        
        # Check if last instruction in block is a jump
        last_instr = bb_asm[-1]
        parts = last_instr.strip().split()
        
        if parts and parts[0].startswith('j'):
            # This is a jump - map source position to target position
            src_pos = addr_to_pos.get(src_addr)
            tgt_pos = addr_to_pos.get(tgt_addr)
            
            if src_pos is not None and tgt_pos is not None:
                # The jump is at the last instruction of the source block
                jump_instr_pos = src_pos + len(bb_asm) - 1
                jump_map[jump_instr_pos] = tgt_pos
    
    return jump_map


def tokenize_function_baseline(func_data):
    """
    Tokenize function using baseline jTrans approach with JUMP_ADDR_X normalization.
    
    Uses CFG data to properly resolve jump targets to instruction positions,
    then converts to token-level positions (JUMP_ADDR_X).
    
    Jump addresses beyond position 511 are replaced with:
    - JUMP_ADDR_EXCEEDED: if target position >= 512
    - UNK_JUMP_ADDR: if target cannot be resolved
    
    Returns space-separated token string.
    """
    asm_list = func_data.get('asm', [])
    
    if not asm_list:
        return None
    
    # Get jump mappings from CFG (instruction position -> target instruction position)
    jump_map = find_jump_targets(func_data)
    
    # First pass: build tokens and map instruction positions to token positions
    instr_to_token_pos = {}  # instruction index -> starting token index
    tokens = []
    
    for instr_idx, asm_str in enumerate(asm_list):
        instr_to_token_pos[instr_idx] = len(tokens)
        
        try:
            operator, op1, op2, op3, annotation = readidadata.parse_asm(asm_str)
            
            if operator:
                tokens.append(operator)
            if op1:
                tokens.append(op1)
            if op2:
                tokens.append(op2)
            if op3:
                tokens.append(op3)
        except:
            continue
    
    # Second pass: generate final tokens with JUMP_ADDR_X replacement
    result_tokens = []
    
    for instr_idx, asm_str in enumerate(asm_list):
        try:
            operator, op1, op2, op3, annotation = readidadata.parse_asm(asm_str)
            
            # Add operator
            if operator:
                result_tokens.append(operator)
            
            # Handle jump instructions
            if operator and operator.startswith('j'):
                # Check if we have a CFG-based target for this jump
                if instr_idx in jump_map:
                    # Get target instruction position
                    target_instr_pos = jump_map[instr_idx]
                    # Convert to token position
                    target_token_pos = instr_to_token_pos.get(target_instr_pos)
                    
                    if target_token_pos is not None:
                        # Check if exceeds vocab limit
                        if target_token_pos >= 512:
                            result_tokens.append('JUMP_ADDR_EXCEEDED')
                        else:
                            result_tokens.append(f'JUMP_ADDR_{target_token_pos}')
                    else:
                        result_tokens.append('UNK_JUMP_ADDR')
                else:
                    # No CFG info - check if it's an unresolvable jump
                    if op1 and (op1 == 'UNK_ADDR' or op1.startswith('hex_')):
                        result_tokens.append('UNK_JUMP_ADDR')
                    elif op1:
                        result_tokens.append(op1)
                
                # Add remaining operands
                if op2:
                    result_tokens.append(op2)
                if op3:
                    result_tokens.append(op3)
            else:
                # Non-jump instruction - add all operands normally
                if op1:
                    result_tokens.append(op1)
                if op2:
                    result_tokens.append(op2)
                if op3:
                    result_tokens.append(op3)
        except:
            continue
    
    return ' '.join(result_tokens)


def create_function_blocks_baseline(pickle_dir, binary_dir):
    """
    Create func_blocks_baseline.json with baseline tokenization.
    
    Returns: (func_blocks dict, mapping dict)
    """
    import pickle
    
    pickle_path = Path(pickle_dir)
    pickle_files = list(pickle_path.glob('**/*_extract.pkl'))
    
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
                file_hash = opt_info['hash']
                
                # Binary filename format: name-OptLevel-hash
                potential_binary = binary_path / f"{binary_name}-{opt_level}-{file_hash}"
                
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
                    
                    # For ground truth matching - always use actual function name
                    mapping[binary_name][func_name][opt_level] = func_id
                    
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
    Create ground truth with grouped format (SAME as address-aware).
    Groups all optimization levels for each function.
    
    Format:
    {
        "pairs": [
            {
                "binary_name": "...",
                "function_name": "...",
                "O0": func_id,
                "O1": func_id,
                ...
            }
        ]
    }
    
    Note: data_json.py automatically converts this to pair-by-pair format
    internally for training, so this is more storage-efficient.
    """
    pairs = []
    
    for binary_name, func_dict in mapping.items():
        for func_name, opt_dict in func_dict.items():
            # Only include if exists in multiple optimization levels
            if len(opt_dict) > 1:
                pair = {
                    'binary_name': binary_name,
                    'function_name': func_name
                }
                pair.update(opt_dict)
                pairs.append(pair)
    
    ground_truth = {
        'pairs': pairs,
        'total_pairs': len(pairs)
    }
    
    # Calculate statistics
    opt_coverage = defaultdict(int)
    for pair in pairs:
        for opt in ['O0', 'O1', 'O2', 'O3', 'Os', 'Og', 'Ofast']:
            if opt in pair:
                opt_coverage[opt] += 1
    
    ground_truth['statistics'] = {
        'optimization_coverage': dict(opt_coverage)
    }
    
    print(f"\n✓ Created {len(pairs)} function groups")
    
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
    print(f"Ground truth groups: {ground_truth['total_pairs']}")
    print(f"\nOptimization level coverage:")
    for opt, count in sorted(ground_truth['statistics']['optimization_coverage'].items()):
        print(f"  {opt}: {count} functions")
    print(f"\nUsing grouped format (SAME as address-aware) for consistency.")


if __name__ == '__main__':
    main()
