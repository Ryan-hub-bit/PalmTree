#!/usr/bin/env python3
"""
Create instruction-level function dataset using jTrans_instr tokenization.

Uses the SAME ground truth matching logic as baseline/address-aware,
but applies instruction-level tokenization (instr_addr_{i} instead of JUMP_ADDR_X).

Outputs:
1. func_blocks_instr.json - All functions with instruction-level tokenization
2. ground_truth_instr.json - Same matching logic as baseline/address-aware
"""

import json
import re
import sys
import pickle
from pathlib import Path
from collections import defaultdict
import subprocess
import importlib.util
import os

# Load readidadata module from jTrans_instr directory
script_dir = os.path.dirname(os.path.abspath(__file__))
jtrans_instr_dir = os.path.dirname(script_dir)
readidadata_path = os.path.join(jtrans_instr_dir, 'readidadata.py')

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


def build_instruction_index_from_cfg(func_data):
    """
    Build instruction index from CFG data.
    Returns: dict mapping basic_block_addr -> instruction_start_index
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
    
    instr_index_map = {}
    current_idx = 0
    
    # Sort basic blocks by address
    sorted_blocks = sorted(cfg.nodes())
    
    for block_addr in sorted_blocks:
        block_data = cfg.nodes.get(block_addr, {})
        bb_asm = block_data.get('asm', [])
        
        # Map this block's start address to current instruction index
        instr_index_map[block_addr] = current_idx
        current_idx += len(bb_asm)
    
    return instr_index_map


def tokenize_instruction_instr(raw_instr, jump_targets):
    """
    Tokenize single instruction with instruction-level addressing.
    Replace jump targets with instr_addr_{i} format.
    
    Args:
        raw_instr: Raw instruction string from IDA
        jump_targets: Dict mapping source_instr_idx -> target_instr_idx
        
    Returns:
        List of tokens
    """
    # Use readidadata tokenizer for base tokenization
    tokens = readidadata.tokenize_instruction(raw_instr)
    
    # Check if this is a control flow instruction with known target
    if tokens and tokens[0] in ['jmp', 'je', 'jne', 'jz', 'jnz', 
                                'ja', 'jae', 'jb', 'jbe', 'jg', 'jge', 
                                'jl', 'jle', 'jo', 'jno', 'js', 'jns']:
        # Replace address operand with instr_addr_{i}
        # The target index would be provided by caller
        pass
    
    return tokens


def find_jump_targets_instr(func_data):
    """
    Extract jump instruction indices and their target indices using CFG.
    
    Returns:
        Dict mapping instruction_index -> target_instruction_index
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
    
    instr_index_map = build_instruction_index_from_cfg(func_data)
    jump_map = {}
    
    # Analyze CFG edges to find jumps
    for src_addr, tgt_addr in cfg.edges():
        src_data = cfg.nodes.get(src_addr, {})
        bb_asm = src_data.get('asm', [])
        
        if not bb_asm:
            continue
        
        # Get source and target instruction indices
        src_idx = instr_index_map.get(src_addr, -1)
        tgt_idx = instr_index_map.get(tgt_addr, -1)
        
        if src_idx == -1 or tgt_idx == -1:
            continue
        
        # Check if last instruction in block is a control flow instruction
        last_instr = bb_asm[-1]
        parts = last_instr.strip().split()
        
        if parts and parts[0] in ['call', 'jmp', 'je', 'jne', 'jz', 'jnz',
                                   'ja', 'jae', 'jb', 'jbe', 'jg', 'jge',
                                   'jl', 'jle', 'jo', 'jno', 'js', 'jns']:
            # Last instruction index in source block
            last_instr_idx = src_idx + len(bb_asm) - 1
            jump_map[last_instr_idx] = tgt_idx
    
    return jump_map


def tokenize_function_instr(func_data):
    """
    Tokenize entire function with instruction-level addressing.
    Uses \\t to separate instructions for accurate boundary detection.
    
    Returns:
        String with instructions separated by \\t, tokens within instruction by space
    """
    cfg = func_data.get('cfg')
    if cfg is None:
        return ""
    
    try:
        import networkx as nx
        if not isinstance(cfg, nx.DiGraph):
            return ""
    except ImportError:
        return ""
    
    # Build jump target mapping
    jump_targets = find_jump_targets_instr(func_data)
    
    # Collect all instructions from all basic blocks
    all_instructions = []
    sorted_blocks = sorted(cfg.nodes())
    
    for block_addr in sorted_blocks:
        block_data = cfg.nodes.get(block_addr, {})
        bb_asm = block_data.get('asm', [])
        all_instructions.extend(bb_asm)
    
    # Tokenize each instruction
    instruction_tokens = []
    for idx, instr in enumerate(all_instructions):
        # Parse instruction using readidadata
        operator, op1, op2, op3, annotation = readidadata.parse_asm(instr)
        
        # Build token list
        tokens = []
        if operator:
            tokens.append(operator)
        
        # Check if this instruction has a jump target (only for jump instructions, NOT call)
        if idx in jump_targets and operator and operator.startswith('j'):
            # This is a jump instruction (jmp, je, jne, etc.) with a known target
            target_idx = jump_targets[idx]
            tokens.append(f'instr_addr_{target_idx}')
            # Add remaining operands if any
            if op2:
                tokens.append(op2)
            if op3:
                tokens.append(op3)
        else:
            # Not a jump with target, add all operands normally (including call instructions)
            if op1:
                tokens.append(op1)
            if op2:
                tokens.append(op2)
            if op3:
                tokens.append(op3)
        
        # Join tokens within instruction with space
        instruction_tokens.append(' '.join(tokens))

    
    # Join instructions with \t separator
    return '\t'.join(instruction_tokens)


def process_pickle_file(pkl_path, binary_dir):
    """Process a single pickle file and return function data."""
    parsed = parse_filename(pkl_path)
    if not parsed:
        print(f"Warning: Could not parse filename: {pkl_path}", file=sys.stderr)
        return None
    
    binary_name, opt_level, file_hash = parsed
    
    # Load pickle file
    try:
        with open(pkl_path, 'rb') as f:
            func_data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}", file=sys.stderr)
        return None
    
    # Get function names from non-stripped binary
    binary_path = Path(binary_dir) / f"{binary_name}-{opt_level}-{file_hash}"
    if not binary_path.exists():
        print(f"Warning: Binary not found: {binary_path}", file=sys.stderr)
        return None
    
    function_names = extract_function_names_from_binary(binary_path)
    
    # Process each function
    results = []
    for func_name, func_data in func_data_dict.items():
        # Skip if not in original binary's symbol table
        if func_name not in function_names:
            continue
        
        # Skip small functions (< 5 instructions) - same as baseline
        asm_list = func_data.get('asm', [])
        if len(asm_list) < 5:
            continue
        
        # Tokenize function with instruction-level addressing
        tokenized = tokenize_function_instr(func_data)
        
        if tokenized:
            results.append({
                'binary': binary_name,
                'opt_level': opt_level,
                'func_name': func_name,
                'tokenized': tokenized
            })
    
    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python create_instr_dataset.py <extract_dir> <output_dir> [--binary-dir <binary_dir>]")
        sys.exit(1)
    
    extract_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    # Parse optional binary directory
    binary_dir = None
    for i, arg in enumerate(sys.argv):
        if arg == '--binary-dir' and i + 1 < len(sys.argv):
            binary_dir = Path(sys.argv[i + 1])
            break
    
    if not binary_dir:
        print("Error: --binary-dir is required", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 80)
    print("Creating Instruction-Level Function Dataset")
    print("=" * 80)
    print(f"Extract directory: {extract_dir}")
    print(f"Binary directory: {binary_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find all pickle files (recursively in subdirectories)
    pkl_files = list(extract_dir.glob("**/*_extract.pkl"))
    print(f"Found {len(pkl_files)} pickle files")
    
    # Group functions by (binary, function_name)
    function_groups = defaultdict(dict)
    
    for pkl_file in pkl_files:
        print(f"Processing: {pkl_file.name}")
        results = process_pickle_file(pkl_file, binary_dir)
        
        if not results:
            continue
        
        for result in results:
            key = (result['binary'], result['func_name'])
            opt = result['opt_level']
            function_groups[key][opt] = result['tokenized']
    
    print(f"\nFound {len(function_groups)} unique functions")
    
    # Generate function blocks and ground truth
    function_blocks = {}
    pairs = []
    opt_levels = ['O0', 'O1', 'O2', 'O3']
    
    func_idx = 1
    complete_count = 0
    
    for (binary, func_name), opt_dict in function_groups.items():
        # Check if we have all 4 optimization levels
        if not all(opt in opt_dict for opt in opt_levels):
            continue
        
        complete_count += 1
        
        # Generate function IDs
        func_ids = {}
        for opt in opt_levels:
            opt_offset = {'O0': 1, 'O1': 2, 'O2': 3, 'O3': 4}[opt]
            func_id = str((func_idx - 1) * 4 + opt_offset)
            func_ids[opt] = func_id
            
            function_blocks[func_id] = {
                'binary': binary,
                'function_name': func_name,
                'opt': opt,
                'instructions': opt_dict[opt]
            }
        
        # Generate pairs
        for i, opt1 in enumerate(opt_levels):
            for opt2 in opt_levels[i+1:]:
                pair = {
                    'binary': binary,
                    'func_name': func_name,
                    'opt1': opt1,
                    'opt2': opt2,
                    'func_id1': int(func_ids[opt1]),
                    'func_id2': int(func_ids[opt2])
                }
                pairs.append(pair)
        
        func_idx += 1
    
    ground_truth = {
        'pairs': pairs,
        'metadata': {
            'total_functions': complete_count,
            'total_pairs': len(pairs),
            'optimization_levels': opt_levels
        }
    }
    
    # Save output files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    func_blocks_file = output_dir / "func_blocks_instr.json"
    ground_truth_file = output_dir / "ground_truth_instr.json"
    
    print(f"\nSaving function blocks to {func_blocks_file}...")
    with open(func_blocks_file, 'w') as f:
        json.dump(function_blocks, f, indent=2)
    
    print(f"Saving ground truth to {ground_truth_file}...")
    with open(ground_truth_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Functions with all 4 opt levels: {complete_count}")
    print(f"Function blocks created: {len(function_blocks)}")
    print(f"Pairs created: {len(pairs)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
