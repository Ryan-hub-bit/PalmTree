#!/usr/bin/env python3
"""
Generate probe data from hierarchical CFG format (cfg_hierarchical_icfg.py output)

This script processes CFG files with hierarchical address-based positions:
  - Position format: func_in_binary:bb_in_function:inst_in_bb (all 8 decimal precision)
  - Extracts bucket labels for BB, function, and binary level probing
  
Output: Samples with embeddings positions and bucket labels for all three levels
"""

import sys
import os
import re
from pathlib import Path
import random

# Position regex: matches addr(0xADDR:pos1:pos2:pos3)
ADDR_POS_RE = re.compile(r'^(0x[0-9a-fA-F]+):([\d.]+):([\d.]+):([\d.]+)$')

# Address operand regex: matches address(0xADDR:pos1:pos2:pos3)
ADDR_OPERAND_RE = re.compile(r'address\((0x[0-9a-fA-F]+):([\d.]+):([\d.]+):([\d.]+)\)')


def parse_hierarchical_instruction(inst_str):
    """
    Parse an instruction with hierarchical positions.
    
    Format: opcode(0xADDR:func_in_bin:bb_in_func:inst_in_bb) operands...
    
    Returns:
        dict with:
            - opcode: instruction opcode
            - addr: instruction address (hex string)
            - binary_pos: function's position in binary [0-1]
            - func_pos: BB's position in function [0-1]
            - bb_pos: instruction's position in BB [0-1]
            - operands: list of operand tokens
    """
    inst_str = inst_str.strip()
    if not inst_str:
        return None
    
    # Split opcode and operands
    parts = inst_str.split(maxsplit=1)
    if len(parts) == 0:
        return None
    
    opcode_part = parts[0]
    operands_part = parts[1] if len(parts) > 1 else ""
    
    # Parse opcode(addr:pos1:pos2:pos3)
    if '(' not in opcode_part:
        return None
    
    opcode, addr_info = opcode_part.split('(', 1)
    addr_info = addr_info.rstrip(')')
    
    # Parse address and positions
    match = ADDR_POS_RE.match(addr_info)
    if not match:
        return None
    
    addr_hex = match.group(1)
    binary_pos = float(match.group(2))
    func_pos = float(match.group(3))
    bb_pos = float(match.group(4))
    
    return {
        'opcode': opcode,
        'addr': addr_hex,
        'binary_pos': binary_pos,
        'func_pos': func_pos,
        'bb_pos': bb_pos,
        'operands': operands_part.strip().split() if operands_part.strip() else []
    }


def assign_buckets(position, num_buckets=5):
    """
    Assign a bucket label based on normalized position [0-1].
    
    Args:
        position: float in [0, 1]
        num_buckets: number of buckets (default 5)
    
    Returns:
        bucket index [0, num_buckets-1]
    """
    # Clamp to [0, 1]
    position = max(0.0, min(1.0, position))
    
    # Assign bucket
    bucket = int(position * num_buckets)
    
    # Handle edge case where position == 1.0
    if bucket >= num_buckets:
        bucket = num_buckets - 1
    
    return bucket


def process_cfg_file(cfg_path, num_bb_buckets=5, num_func_buckets=5, num_binary_buckets=5, max_samples=None):
    """
    Process a hierarchical CFG file and extract samples with bucket labels.
    
    Args:
        cfg_path: Path to CFG file (from cfg_hierarchical_icfg.py)
        num_bb_buckets: Number of buckets for BB-level positions
        num_func_buckets: Number of buckets for function-level positions
        num_binary_buckets: Number of buckets for binary-level positions
        max_samples: Maximum samples to extract (None = all)
    
    Returns:
        List of samples, each with:
            - text: instruction text
            - binary_pos, func_pos, bb_pos: hierarchical positions
            - bb_bucket, func_bucket, binary_bucket: bucket labels
    """
    samples = []
    
    print(f"[INFO] Processing {cfg_path}")
    
    with open(cfg_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Split by tabs (each instruction is tab-separated)
            instructions = line.split('\t')
            
            for inst_str in instructions:
                parsed = parse_hierarchical_instruction(inst_str)
                if parsed is None:
                    continue
                
                # Assign buckets for all three levels
                bb_bucket = assign_buckets(parsed['bb_pos'], num_bb_buckets)
                func_bucket = assign_buckets(parsed['func_pos'], num_func_buckets)
                binary_bucket = assign_buckets(parsed['binary_pos'], num_binary_buckets)
                
                sample = {
                    'text': inst_str,
                    'opcode': parsed['opcode'],
                    'addr': parsed['addr'],
                    'binary_pos': parsed['binary_pos'],
                    'func_pos': parsed['func_pos'],
                    'bb_pos': parsed['bb_pos'],
                    'bb_bucket': bb_bucket,
                    'func_bucket': func_bucket,
                    'binary_bucket': binary_bucket
                }
                
                samples.append(sample)
                
                if max_samples is not None and len(samples) >= max_samples:
                    print(f"[INFO] Reached max samples limit: {max_samples}")
                    return samples
    
    print(f"[INFO] Extracted {len(samples)} samples from {cfg_path}")
    return samples


def balance_bucket_samples(samples, bucket_key, num_buckets, samples_per_bucket=1000):
    """
    Balance samples across buckets by randomly sampling.
    
    Args:
        samples: List of sample dicts
        bucket_key: Key to use for bucketing ('bb_bucket', 'func_bucket', 'binary_bucket')
        num_buckets: Number of buckets
        samples_per_bucket: Target samples per bucket
    
    Returns:
        Balanced list of samples
    """
    # Group by bucket
    buckets = {i: [] for i in range(num_buckets)}
    for sample in samples:
        bucket = sample[bucket_key]
        buckets[bucket].append(sample)
    
    # Print distribution
    print(f"\n[INFO] Original distribution for {bucket_key}:")
    for i in range(num_buckets):
        print(f"  Bucket {i}: {len(buckets[i])} samples")
    
    # Balance by sampling
    balanced = []
    for i in range(num_buckets):
        bucket_samples = buckets[i]
        if len(bucket_samples) == 0:
            print(f"  [WARN] Bucket {i} has 0 samples!")
            continue
        
        # Sample with replacement if needed
        if len(bucket_samples) < samples_per_bucket:
            sampled = random.choices(bucket_samples, k=samples_per_bucket)
            print(f"  [WARN] Bucket {i} has only {len(bucket_samples)} samples, sampling with replacement")
        else:
            sampled = random.sample(bucket_samples, samples_per_bucket)
        
        balanced.extend(sampled)
    
    print(f"[INFO] Balanced dataset: {len(balanced)} samples ({samples_per_bucket} per bucket)")
    return balanced


def save_probe_data(samples, output_path, bucket_key):
    """
    Save probe data in format suitable for bucket probe scripts.
    
    Format per line:
        text|||binary_pos|||func_pos|||bb_pos|||bucket_label
    """
    print(f"[INFO] Saving to {output_path}")
    
    with open(output_path, 'w') as f:
        for sample in samples:
            line = f"{sample['text']}|||{sample['binary_pos']:.8f}|||{sample['func_pos']:.8f}|||{sample['bb_pos']:.8f}|||{sample[bucket_key]}\n"
            f.write(line)
    
    print(f"[INFO] Saved {len(samples)} samples")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate probe data from hierarchical CFG files')
    parser.add_argument('--cfg_dir', type=str, default='../data/cfg_2',
                        help='Directory containing hierarchical CFG files')
    parser.add_argument('--output_dir', type=str, default='data/hierarchical',
                        help='Output directory for probe data')
    parser.add_argument('--num_binaries', type=int, default=5,
                        help='Number of binaries to process')
    parser.add_argument('--bb_buckets', type=int, default=5,
                        help='Number of BB-level buckets')
    parser.add_argument('--func_buckets', type=int, default=5,
                        help='Number of function-level buckets')
    parser.add_argument('--binary_buckets', type=int, default=5,
                        help='Number of binary-level buckets')
    parser.add_argument('--samples_per_bucket', type=int, default=1000,
                        help='Target samples per bucket after balancing')
    parser.add_argument('--pattern', type=str, default='*_cfg_*_inline.txt',
                        help='File pattern to match CFG files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CFG files
    cfg_dir = Path(args.cfg_dir)
    if not cfg_dir.exists():
        print(f"[ERROR] CFG directory not found: {cfg_dir}")
        sys.exit(1)
    
    cfg_files = sorted(cfg_dir.glob(args.pattern))
    if not cfg_files:
        print(f"[ERROR] No CFG files found matching pattern: {args.pattern}")
        sys.exit(1)
    
    print(f"[INFO] Found {len(cfg_files)} CFG files")
    
    # Limit to requested number
    cfg_files = cfg_files[:args.num_binaries]
    print(f"[INFO] Processing {len(cfg_files)} binaries")
    
    # Process all files and collect samples
    all_samples = []
    for cfg_file in cfg_files:
        samples = process_cfg_file(cfg_file, args.bb_buckets, args.func_buckets, args.binary_buckets)
        all_samples.extend(samples)
    
    print(f"\n[INFO] Total samples collected: {len(all_samples)}")
    
    # Generate balanced datasets for each probe type
    print("\n" + "="*80)
    print("Generating BB-level probe data")
    print("="*80)
    bb_balanced = balance_bucket_samples(all_samples, 'bb_bucket', args.bb_buckets, args.samples_per_bucket)
    save_probe_data(bb_balanced, output_dir / f'bb_bucket_{args.bb_buckets}_balanced.txt', 'bb_bucket')
    
    print("\n" + "="*80)
    print("Generating Function-level probe data")
    print("="*80)
    func_balanced = balance_bucket_samples(all_samples, 'func_bucket', args.func_buckets, args.samples_per_bucket)
    save_probe_data(func_balanced, output_dir / f'func_bucket_{args.func_buckets}_balanced.txt', 'func_bucket')
    
    print("\n" + "="*80)
    print("Generating Binary-level probe data")
    print("="*80)
    binary_balanced = balance_bucket_samples(all_samples, 'binary_bucket', args.binary_buckets, args.samples_per_bucket)
    save_probe_data(binary_balanced, output_dir / f'binary_bucket_{args.binary_buckets}_balanced.txt', 'binary_bucket')
    
    print("\n" + "="*80)
    print("DONE! Probe data generated successfully")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {output_dir / f'bb_bucket_{args.bb_buckets}_balanced.txt'}")
    print(f"  - {output_dir / f'func_bucket_{args.func_buckets}_balanced.txt'}")
    print(f"  - {output_dir / f'binary_bucket_{args.binary_buckets}_balanced.txt'}")


if __name__ == '__main__':
    main()
