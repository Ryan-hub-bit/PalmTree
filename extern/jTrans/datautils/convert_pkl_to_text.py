#!/usr/bin/env python3
"""
Convert pickle files from IDA extraction to text format for jTrans pretraining.

Implements proper JUMP_ADDR_X encoding as described in jTrans paper:
- For each jump instruction, replace target address with JUMP_ADDR_X
- Where X is the 0-indexed position of the target instruction in the function
- Uses CFG to accurately resolve jump targets

Input: *_extract.pkl files 
Output: Text file with one function per line, tokenized assembly with JUMP_ADDR_X.
"""

import pickle
import os
import sys
import glob

# Add jTrans to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readidadata import parse_asm


def build_instruction_address_map(func_data):
    """
    Build mapping from basic block addresses to instruction positions.
    
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


def convert_function_to_text_with_jumps(func_data):
    """
    Convert function to text with proper JUMP_ADDR_X encoding.
    
    This implements jTrans paper's jump encoding:
    1. Identify all jump instructions
    2. Use CFG to find their targets
    3. Replace jump targets with JUMP_ADDR_X where X is target position
    
    Returns:
        Space-separated token string
    """
    asm_list = func_data.get('asm', [])
    
    if not asm_list:
        return None
    
    # Get jump mappings from CFG
    jump_map = find_jump_targets(func_data)
    
    result_tokens = []
    
    for i, asm_str in enumerate(asm_list):
        try:
            operator, op1, op2, op3, annotation = parse_asm(asm_str)
            
            tokens = []
            if operator:
                tokens.append(operator)
            
            # Check if this is a jump instruction
            if operator and operator.startswith('j'):
                # Check if we have a target for this jump
                if i in jump_map:
                    # Replace operand with JUMP_ADDR_X
                    target_pos = jump_map[i]
                    tokens.append(f'JUMP_ADDR_{target_pos}')
                else:
                    # No CFG info - keep normalized operand or mark as unknown
                    if op1 and (op1 == 'UNK_ADDR' or op1.startswith('hex_')):
                        # Can't resolve - use placeholder
                        tokens.append('JUMP_ADDR_UNK')
                    elif op1:
                        tokens.append(op1)
                
                # Add remaining operands (for conditional jumps with multiple operands)
                if op2:
                    tokens.append(op2)
                if op3:
                    tokens.append(op3)
            else:
                # Non-jump instruction - add all operands normally
                if op1:
                    tokens.append(op1)
                if op2:
                    tokens.append(op2)
                if op3:
                    tokens.append(op3)
            
            result_tokens.extend(tokens)
            
        except Exception as e:
            print(f"Warning: Failed to parse instruction '{asm_str}': {e}", file=sys.stderr)
            continue
    
    return ' '.join(result_tokens)


def convert_pickle_to_text(pkl_path, output_path, min_instructions=5, max_instructions=512):
    """Convert a pickle file to text format with JUMP_ADDR_X encoding."""
    print(f"Processing: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    total_funcs = len(data)
    skipped = 0
    written = 0
    total_jumps = 0
    
    with open(output_path, 'a') as out:  # Append mode
        for func_name, func_data in data.items():
            # Skip special sections
            if func_name in ['.plt', 'extern', '.init', '.fini']:
                skipped += 1
                continue
            
            asm_list = func_data.get('asm', [])
            
            # Filter by length
            if len(asm_list) < min_instructions:
                skipped += 1
                continue
            
            # Truncate if too long
            if len(asm_list) > max_instructions:
                func_data['asm'] = asm_list[:max_instructions]
            
            # Convert to text with jump encoding
            text = convert_function_to_text_with_jumps(func_data)
            
            if text:
                # Count jumps for statistics
                num_jumps = text.count('JUMP_ADDR_')
                total_jumps += num_jumps
                
                out.write(text + '\n')
                written += 1
    
    print(f"  ✓ {written} functions, {skipped} skipped, {total_jumps} jumps")
    return written, skipped, total_jumps


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert IDA pickle files to jTrans text format with JUMP_ADDR_X')
    parser.add_argument('input_dir', help='Directory containing *_extract.pkl files (searches recursively)')
    parser.add_argument('output_file', help='Output text file')
    parser.add_argument('--min-instructions', type=int, default=5,
                        help='Minimum instructions per function (default: 5)')
    parser.add_argument('--max-instructions', type=int, default=512,
                        help='Maximum instructions per function (default: 512)')
    parser.add_argument('--pattern', default='*_extract.pkl',
                        help='Glob pattern for pickle files (default: *_extract.pkl)')
    parser.add_argument('--recursive', action='store_true', default=True,
                        help='Search recursively in subdirectories (default: True)')
    
    args = parser.parse_args()
    
    # Find all pickle files (recursively by default)
    pkl_files = []
    
    if args.recursive:
        # Walk through all subdirectories
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith('_extract.pkl'):
                    pkl_files.append(os.path.join(root, file))
    else:
        # Only search top-level directory
        pkl_pattern = os.path.join(args.input_dir, args.pattern)
        pkl_files = glob.glob(pkl_pattern)
    
    if not pkl_files:
        print(f"Error: No pickle files found in: {args.input_dir}")
        print(f"       (Searched recursively: {args.recursive})")
        sys.exit(1)
    
    print(f"Found {len(pkl_files)} pickle files\n")
    
    # Clear output file
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    
    total_written = 0
    total_skipped = 0
    total_jumps = 0
    
    for pkl_file in sorted(pkl_files):
        written, skipped, jumps = convert_pickle_to_text(
            pkl_file,
            args.output_file,
            min_instructions=args.min_instructions,
            max_instructions=args.max_instructions
        )
        total_written += written
        total_skipped += skipped
        total_jumps += jumps
    
    print(f"\n✅ Conversion complete!")
    print(f"   Functions: {total_written} (skipped {total_skipped})")
    print(f"   Jump instructions: {total_jumps}")
    print(f"   Output: {args.output_file}")
    
    # Show sample with jumps
    print(f"\n📝 Sample with JUMP_ADDR_X:")
    with open(args.output_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            if 'JUMP_ADDR_' in line:
                print(f"   {line[:120]}...")
                print(f"      → {line.count('JUMP_ADDR_')} jump(s)")
            else:
                print(f"   {line[:100]}...")


if __name__ == '__main__':
    main()
