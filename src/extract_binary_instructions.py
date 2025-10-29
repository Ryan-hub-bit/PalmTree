#!/usr/bin/env python3
"""
Extract all instructions from each binary's cfg_8.txt file in the test dataset.
Each binary will have its own output file containing all its instructions.
"""

import os
import glob
import argparse
from pathlib import Path


def extract_instructions_by_binary(test_dir, output_dir):
    """
    Extract all instructions from each binary's cfg_8.txt file.
    
    Args:
        test_dir: Directory containing test/cfg folder with binary files
        output_dir: Directory to save extracted instructions per binary
    """
    # Find all cfg_8.txt files (not _src or _tgt)
    cfg_dir = os.path.join(test_dir, "cfg")
    pattern = os.path.join(cfg_dir, "*_cfg_8.txt")
    cfg_files = glob.glob(pattern)
    
    # Filter out _src and _tgt files
    cfg_files = [f for f in cfg_files if not f.endswith("_cfg_8_src.txt") and not f.endswith("_cfg_8_tgt.txt")]
    
    print(f"Found {len(cfg_files)} binary cfg_8.txt files")
    
    os.makedirs(output_dir, exist_ok=True)
    
    binary_stats = []
    
    for cfg_file in sorted(cfg_files):
        # Extract binary name from filename
        # Format: <binary_name>_cfg_8.txt
        filename = os.path.basename(cfg_file)
        binary_name = filename.replace("_cfg_8.txt", "")
        
        print(f"Processing: {binary_name}")
        
        # Read all instruction sequences
        with open(cfg_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse instructions: each line contains 8 instructions separated by tabs
        all_instructions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split by tab to get individual instructions
            instructions = line.split('\t')
            all_instructions.extend(instructions)
        
        # Count unique instructions and opcodes
        unique_instructions = set(all_instructions)
        opcodes = set()
        for inst in all_instructions:
            parts = inst.strip().split()
            if parts:
                opcodes.add(parts[0])  # First token is opcode
        
        # Save to output file
        output_file = os.path.join(output_dir, f"{binary_name}_instructions.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header with statistics
            f.write(f"# Binary: {binary_name}\n")
            f.write(f"# Total Instructions: {len(all_instructions)}\n")
            f.write(f"# Unique Instructions: {len(unique_instructions)}\n")
            f.write(f"# Unique Opcodes: {len(opcodes)}\n")
            f.write(f"# Source File: {cfg_file}\n")
            f.write("#" + "="*80 + "\n\n")
            
            # Write all instructions (one per line)
            for inst in all_instructions:
                f.write(inst + "\n")
        
        binary_stats.append({
            'binary': binary_name,
            'total_instructions': len(all_instructions),
            'unique_instructions': len(unique_instructions),
            'unique_opcodes': len(opcodes),
            'output_file': output_file
        })
        
        print(f"  - Total instructions: {len(all_instructions)}")
        print(f"  - Unique instructions: {len(unique_instructions)}")
        print(f"  - Unique opcodes: {len(opcodes)}")
        print(f"  - Saved to: {output_file}\n")
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, "extraction_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BINARY INSTRUCTION EXTRACTION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Binaries: {len(binary_stats)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write(f"{'Binary Name':<50} {'Total Inst':>12} {'Unique Inst':>12} {'Opcodes':>8}\n")
        f.write("-"*80 + "\n")
        
        total_inst = 0
        total_unique = 0
        all_opcodes = set()
        
        for stat in binary_stats:
            f.write(f"{stat['binary']:<50} {stat['total_instructions']:>12} "
                   f"{stat['unique_instructions']:>12} {stat['unique_opcodes']:>8}\n")
            total_inst += stat['total_instructions']
            total_unique += stat['unique_instructions']
        
        f.write("-"*80 + "\n")
        f.write(f"{'TOTAL':<50} {total_inst:>12} {total_unique:>12}\n")
        f.write("-"*80 + "\n")
        
    print(f"\n{'='*80}")
    print(f"Extraction Complete!")
    print(f"{'='*80}")
    print(f"Total binaries processed: {len(binary_stats)}")
    print(f"Total instructions extracted: {total_inst:,}")
    print(f"Summary saved to: {summary_file}")
    print(f"Individual binary files saved to: {output_dir}")
    
    return binary_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract all instructions from each binary's cfg_8.txt file"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/home/louie/PalmTree/datalong/test",
        help="Directory containing test/cfg folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/louie/PalmTree/evaluation_results/binary_instructions",
        help="Output directory for extracted instructions"
    )
    
    args = parser.parse_args()
    
    extract_instructions_by_binary(args.test_dir, args.output_dir)
