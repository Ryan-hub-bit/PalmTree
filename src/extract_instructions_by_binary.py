"""
Extract all instructions from each binary's cfg_8.txt file for comparison.
This supports both:
1. Current model (windows8noaddr): Uses cfg_8.txt only
2. Address-aware model (windows8): Uses cfg_8.txt + cfg_8_src.txt + cfg_8_tgt.txt
"""

import os
import argparse
import json
from collections import defaultdict


def extract_binary_name(filename):
    """Extract binary name from filename like '0ad__libnvtt.so_cfg_8.txt' -> '0ad__libnvtt.so'"""
    if '_cfg_8.txt' in filename:
        return filename.replace('_cfg_8.txt', '')
    return filename


def load_instructions_from_file(filepath, max_lines=None):
    """Load instructions from a file, each line contains 8 instructions"""
    instructions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_lines and idx >= max_lines:
                break
            line = line.strip()
            if line:
                # Each line has 8 instructions separated by tabs
                insts = line.split('\t')
                instructions.extend(insts)
    return instructions


def load_address_file(filepath):
    """Load address information (src or tgt)"""
    addresses = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                addrs = line.split('\t')
                addresses.extend(addrs)
    return addresses


def extract_instructions_by_binary(test_dir, output_dir, max_lines_per_binary=None, with_addresses=False):
    """
    Extract all instructions from each binary in the test dataset.
    
    Args:
        test_dir: Directory containing cfg_8.txt files
        output_dir: Directory to save extracted instructions
        max_lines_per_binary: Maximum lines to read per binary (None = all)
        with_addresses: If True, also load address information
    """
    cfg_dir = os.path.join(test_dir, 'cfg')
    
    # Find all cfg_8.txt files
    cfg_files = [f for f in os.listdir(cfg_dir) if f.endswith('_cfg_8.txt') and 
                 not f.endswith('_src.txt') and not f.endswith('_tgt.txt')]
    
    print(f"Found {len(cfg_files)} binaries in {cfg_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    binary_data = {}
    
    for cfg_file in sorted(cfg_files):
        binary_name = extract_binary_name(cfg_file)
        cfg_path = os.path.join(cfg_dir, cfg_file)
        
        print(f"Processing: {binary_name}")
        
        # Load instructions
        instructions = load_instructions_from_file(cfg_path, max_lines_per_binary)
        
        data = {
            'binary_name': binary_name,
            'num_instructions': len(instructions),
            'instructions': instructions
        }
        
        # Load address information if requested
        if with_addresses:
            src_file = cfg_file.replace('_cfg_8.txt', '_cfg_8_src.txt')
            tgt_file = cfg_file.replace('_cfg_8.txt', '_cfg_8_tgt.txt')
            
            src_path = os.path.join(cfg_dir, src_file)
            tgt_path = os.path.join(cfg_dir, tgt_file)
            
            if os.path.exists(src_path) and os.path.exists(tgt_path):
                src_addrs = load_address_file(src_path)
                tgt_addrs = load_address_file(tgt_path)
                
                data['src_addresses'] = src_addrs
                data['tgt_addresses'] = tgt_addrs
                data['num_src_addresses'] = len(src_addrs)
                data['num_tgt_addresses'] = len(tgt_addrs)
                
                # Create augmented instructions with addresses
                augmented = []
                for i, inst in enumerate(instructions):
                    if i < len(src_addrs) and i < len(tgt_addrs):
                        augmented.append({
                            'instruction': inst,
                            'src_addr': src_addrs[i],
                            'tgt_addr': tgt_addrs[i]
                        })
                data['augmented_instructions'] = augmented
                print(f"  - Loaded {len(instructions)} instructions with addresses")
            else:
                print(f"  - WARNING: Address files not found for {binary_name}")
        else:
            print(f"  - Loaded {len(instructions)} instructions")
        
        binary_data[binary_name] = data
        
        # Save individual binary file
        binary_output_file = os.path.join(output_dir, f"{binary_name}.json")
        with open(binary_output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    # Save summary
    summary = {
        'total_binaries': len(binary_data),
        'with_addresses': with_addresses,
        'binaries': {name: {
            'num_instructions': data['num_instructions'],
            'has_addresses': 'src_addresses' in data
        } for name, data in binary_data.items()}
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Total binaries: {len(binary_data)}")
    print(f"With addresses: {with_addresses}")
    print(f"Output directory: {output_dir}")
    print(f"Summary saved to: {summary_file}")
    
    return binary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract instructions by binary name')
    parser.add_argument('--test_dir', type=str, default='/home/louie/PalmTree/datalong/test',
                        help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted instructions')
    parser.add_argument('--max_lines_per_binary', type=int, default=None,
                        help='Maximum lines to read per binary (default: all)')
    parser.add_argument('--with_addresses', action='store_true',
                        help='Also extract address information (src/tgt)')
    
    args = parser.parse_args()
    
    extract_instructions_by_binary(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        max_lines_per_binary=args.max_lines_per_binary,
        with_addresses=args.with_addresses
    )
