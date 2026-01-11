"""
Prepare probe data for jump understanding evaluation.

Extracts functions with control flow information from the main dataset.
"""

import json
import sys
import os
import argparse
import pickle
import networkx as nx
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extern', 'jTrans', 'datautils'))


def load_cfg_data(cfg_file):
    """Load CFG data from pickle file."""
    with open(cfg_file, 'rb') as f:
        return pickle.load(f)


def extract_jump_info_from_cfg(cfg_graph):
    """
    Extract jump information from CFG graph.
    
    Returns dict: {jump_instruction_pos: target_instruction_pos}
    """
    jumps = {}
    
    if isinstance(cfg_graph, nx.DiGraph):
        # NetworkX graph format
        for src, dst in cfg_graph.edges():
            # Assuming nodes have 'pos' attribute for token position
            src_pos = cfg_graph.nodes[src].get('token_pos', src)
            dst_pos = cfg_graph.nodes[dst].get('token_pos', dst)
            jumps[src_pos] = dst_pos
    elif isinstance(cfg_graph, dict):
        # Dict format
        jumps = cfg_graph
    
    return jumps


def prepare_baseline_probe_data(func_blocks_file, cfg_dir, output_file, max_funcs=1000):
    """
    Prepare probe data for baseline model.
    
    Args:
        func_blocks_file: Path to func_blocks_baseline.json
        cfg_dir: Directory containing CFG pickle files
        output_file: Output JSON file
        max_funcs: Maximum number of functions to include
    """
    
    print(f"Loading function data from {func_blocks_file}...")
    with open(func_blocks_file, 'r') as f:
        func_blocks = json.load(f)
    
    probe_data = {}
    func_count = 0
    
    print(f"Processing functions with CFG data...")
    for func_id, func_data in tqdm(func_blocks.items()):
        if func_count >= max_funcs:
            break
        
        # Look for corresponding CFG file
        cfg_file = os.path.join(cfg_dir, f"{func_id}.pkl")
        if not os.path.exists(cfg_file):
            continue
        
        # Load CFG
        try:
            cfg_data = load_cfg_data(cfg_file)
            jumps = extract_jump_info_from_cfg(cfg_data)
            
            if len(jumps) == 0:
                continue  # Skip functions without jumps
            
            # Extract tokens (ensure it's a string)
            tokens = func_data['tokens']
            if isinstance(tokens, list):
                tokens = ' '.join(tokens)
            
            # Extract data
            probe_data[func_id] = {
                'tokens': tokens,  # Store as space-separated string
                'cfg': {str(k): int(v) for k, v in jumps.items()},
                'num_jumps': len(jumps)
            }
            
            func_count += 1
            
        except Exception as e:
            print(f"Error processing function {func_id}: {e}")
            continue
    
    # Save probe data
    print(f"\nSaving {len(probe_data)} functions to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(probe_data, f, indent=2)
    
    print(f"Done! Saved {len(probe_data)} functions with control flow information.")
    
    # Print statistics
    total_jumps = sum(d['num_jumps'] for d in probe_data.values())
    print(f"\nStatistics:")
    print(f"  Total functions: {len(probe_data)}")
    print(f"  Total jumps: {total_jumps}")
    print(f"  Avg jumps per function: {total_jumps/len(probe_data):.2f}")


def prepare_addressaware_probe_data(func_blocks_file, cfg_dir, output_file, max_funcs=1000):
    """
    Prepare probe data for address-aware model.
    
    Includes position and var offset information.
    """
    
    print(f"Loading function data from {func_blocks_file}...")
    with open(func_blocks_file, 'r') as f:
        func_blocks = json.load(f)
    
    probe_data = {}
    func_count = 0
    
    print(f"Processing functions with CFG and position data...")
    for func_id, func_data in tqdm(func_blocks.items()):
        if func_count >= max_funcs:
            break
        
        # Look for corresponding CFG file
        cfg_file = os.path.join(cfg_dir, f"{func_id}.pkl")
        if not os.path.exists(cfg_file):
            continue
        
        # Load CFG
        try:
            cfg_data = load_cfg_data(cfg_file)
            jumps = extract_jump_info_from_cfg(cfg_data)
            
            if len(jumps) == 0:
                continue
            
            # Parse address-aware tokens to extract positions
            tokens_str = func_data['tokens']
            if isinstance(tokens_str, list):
                tokens_str = ' '.join(tokens_str)
            
            binary_pos, function_pos, bb_pos, var_offsets = parse_addressaware_positions(tokens_str)
            
            probe_data[func_id] = {
                'tokens': tokens_str,  # Store as space-separated string
                'cfg': {str(k): int(v) for k, v in jumps.items()},
                'binary_pos': binary_pos,  # Position lists extracted from token annotations
                'function_pos': function_pos,
                'bb_pos': bb_pos,
                'var_offsets': var_offsets,
                'num_jumps': len(jumps)
            }
            
            func_count += 1
            
        except Exception as e:
            print(f"Error processing function {func_id}: {e}")
            continue
    
    # Save probe data
    print(f"\nSaving {len(probe_data)} functions to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(probe_data, f, indent=2)
    
    print(f"Done! Saved {len(probe_data)} functions with control flow and position information.")


def parse_addressaware_positions(tokens_str):
    """Parse position information from address-aware tokens."""
    import re
    
    addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    tokens = tokens_str.split()
    
    binary_pos = []
    function_pos = []
    bb_pos = []
    var_offsets = []
    
    for token in tokens:
        # Check for opcode(address)
        match = addr_pattern.match(token)
        if match:
            binary_pos.append(float(match.group(3)))
            function_pos.append(float(match.group(4)))
            bb_pos.append(float(match.group(5)))
            var_offsets.append(-1)
            continue
        
        # Check for nested address()
        nested_match = nested_addr_pattern.match(token)
        if nested_match:
            binary_pos.append(float(nested_match.group(2)))
            function_pos.append(float(nested_match.group(3)))
            bb_pos.append(float(nested_match.group(4)))
            var_offsets.append(-1)
            continue
        
        # Check for daddr()
        daddr_match = daddr_pattern.match(token)
        if daddr_match:
            binary_pos.append(float(daddr_match.group(2)))
            function_pos.append(float(daddr_match.group(3)))
            bb_pos.append(float(daddr_match.group(4)))
            var_offsets.append(-1)
            continue
        
        # Check for var()
        var_match = var_pattern.match(token)
        if var_match:
            hex_offset = var_match.group(1)
            offset_val = int(hex_offset, 16)
            if offset_val >= 2**63:
                offset_val = offset_val - 2**64
            
            binary_pos.append(-1.0)
            function_pos.append(-1.0)
            bb_pos.append(-1.0)
            var_offsets.append(offset_val)
            continue
        
        # Regular token
        binary_pos.append(-1.0)
        function_pos.append(-1.0)
        bb_pos.append(-1.0)
        var_offsets.append(-1)
    
    return binary_pos, function_pos, bb_pos, var_offsets


def main():
    parser = argparse.ArgumentParser(description="Prepare Jump Probe Data")
    parser.add_argument("--func_blocks", type=str, required=True,
                       help="Path to func_blocks JSON file")
    parser.add_argument("--cfg_dir", type=str, required=True,
                       help="Directory containing CFG pickle files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file for probe data")
    parser.add_argument("--model_type", type=str, choices=['baseline', 'addressaware'],
                       default='baseline', help="Model type")
    parser.add_argument("--max_funcs", type=int, default=1000,
                       help="Maximum number of functions to include")
    
    args = parser.parse_args()
    
    if args.model_type == 'baseline':
        prepare_baseline_probe_data(args.func_blocks, args.cfg_dir, args.output, args.max_funcs)
    else:
        prepare_addressaware_probe_data(args.func_blocks, args.cfg_dir, args.output, args.max_funcs)


if __name__ == '__main__':
    main()
