#!/usr/bin/env python3
"""
Generate comprehensive JSON files with all bucket labels for each instruction.

For each binary, creates a JSON file containing:
- Instruction text
- All three hierarchical positions (binary, function, BB)
- All three bucket labels (binary_bucket, func_bucket, bb_bucket)

Output format:
{
  "binary_name": "example.so",
  "total_instructions": 12345,
  "bucket_config": {
    "bb_buckets": 5,
    "func_buckets": 5,
    "binary_buckets": 5
  },
  "instructions": [
    {
      "addr": "0x12345",
      "opcode": "mov",
      "text": "mov(0x12345:0.5:0.3:0.1) rax rbx",
      "positions": {
        "binary_pos": 0.5,
        "func_pos": 0.3,
        "bb_pos": 0.1
      },
      "buckets": {
        "binary_bucket": 2,
        "func_bucket": 1,
        "bb_bucket": 0
      }
    },
    ...
  ]
}
"""

import sys
import os
import re
import json
from pathlib import Path
from collections import defaultdict

# Position regex: matches addr(0xADDR:pos1:pos2:pos3)
ADDR_POS_RE = re.compile(r'^(0x[0-9a-fA-F]+):([\d.]+):([\d.]+):([\d.]+)$')


def parse_hierarchical_instruction(inst_str):
    """
    Parse an instruction with hierarchical positions.
    
    Format: opcode(0xADDR:func_in_bin:bb_in_func:inst_in_bb) operands...
    
    Returns:
        dict with parsed information or None if parsing fails
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
        'text': inst_str,
        'operands': operands_part.strip()
    }


def assign_bucket(position, num_buckets=5):
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


def process_cfg_file(cfg_path, bb_buckets=5, func_buckets=5, binary_buckets=5):
    """
    Process a hierarchical CFG file and extract all instructions with bucket labels.
    
    Args:
        cfg_path: Path to CFG file
        bb_buckets: Number of BB-level buckets
        func_buckets: Number of function-level buckets
        binary_buckets: Number of binary-level buckets
    
    Returns:
        dict with binary info and all instructions
    """
    binary_name = Path(cfg_path).name.replace('_cfg_2_inline.txt', '').replace('_cfg_8_inline.txt', '')
    
    instructions = []
    
    print(f"[INFO] Processing {cfg_path}")
    
    with open(cfg_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Split by tabs (each instruction is tab-separated)
            inst_tokens = line.split('\t')
            
            for inst_str in inst_tokens:
                parsed = parse_hierarchical_instruction(inst_str)
                if parsed is None:
                    continue
                
                # Assign buckets for all three levels
                bb_bucket = assign_bucket(parsed['bb_pos'], bb_buckets)
                func_bucket = assign_bucket(parsed['func_pos'], func_buckets)
                binary_bucket = assign_bucket(parsed['binary_pos'], binary_buckets)
                
                instruction = {
                    'addr': parsed['addr'],
                    'opcode': parsed['opcode'],
                    'text': parsed['text'],
                    'positions': {
                        'binary_pos': round(parsed['binary_pos'], 8),
                        'func_pos': round(parsed['func_pos'], 8),
                        'bb_pos': round(parsed['bb_pos'], 8)
                    },
                    'buckets': {
                        'binary_bucket': binary_bucket,
                        'func_bucket': func_bucket,
                        'bb_bucket': bb_bucket
                    }
                }
                
                instructions.append(instruction)
    
    print(f"[INFO] Extracted {len(instructions)} instructions from {binary_name}")
    
    # Build result
    result = {
        'binary_name': binary_name,
        'source_file': str(cfg_path),
        'total_instructions': len(instructions),
        'bucket_config': {
            'bb_buckets': bb_buckets,
            'func_buckets': func_buckets,
            'binary_buckets': binary_buckets
        },
        'instructions': instructions
    }
    
    # Add bucket statistics
    bb_dist = defaultdict(int)
    func_dist = defaultdict(int)
    binary_dist = defaultdict(int)
    
    for inst in instructions:
        bb_dist[inst['buckets']['bb_bucket']] += 1
        func_dist[inst['buckets']['func_bucket']] += 1
        binary_dist[inst['buckets']['binary_bucket']] += 1
    
    result['bucket_statistics'] = {
        'bb_distribution': {str(k): v for k, v in sorted(bb_dist.items())},
        'func_distribution': {str(k): v for k, v in sorted(func_dist.items())},
        'binary_distribution': {str(k): v for k, v in sorted(binary_dist.items())}
    }
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate JSON files with all bucket labels for each binary')
    parser.add_argument('--cfg_dir', type=str, default='../data/cfg_2',
                        help='Directory containing hierarchical CFG files')
    parser.add_argument('--output_dir', type=str, default='data/json_labels',
                        help='Output directory for JSON files')
    parser.add_argument('--bb_buckets', type=int, default=5,
                        help='Number of BB-level buckets')
    parser.add_argument('--func_buckets', type=int, default=5,
                        help='Number of function-level buckets')
    parser.add_argument('--binary_buckets', type=int, default=5,
                        help='Number of binary-level buckets')
    parser.add_argument('--pattern', type=str, default='*_cfg_*_inline.txt',
                        help='File pattern to match CFG files')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of binaries to process (default: all)')
    
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
    
    # Limit if requested
    if args.limit is not None:
        cfg_files = cfg_files[:args.limit]
        print(f"[INFO] Processing {len(cfg_files)} binaries (limited)")
    
    # Process each file
    print("="*80)
    for i, cfg_file in enumerate(cfg_files, 1):
        print(f"\n[{i}/{len(cfg_files)}] Processing {cfg_file.name}")
        
        result = process_cfg_file(cfg_file, args.bb_buckets, args.func_buckets, args.binary_buckets)
        
        # Save to JSON
        output_file = output_dir / f"{result['binary_name']}_instructions_labels.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"[INFO] Saved to {output_file}")
        
        # Print statistics
        print(f"\n  Statistics:")
        print(f"    Total instructions: {result['total_instructions']}")
        print(f"    BB bucket distribution: {result['bucket_statistics']['bb_distribution']}")
        print(f"    Function bucket distribution: {result['bucket_statistics']['func_distribution']}")
        print(f"    Binary bucket distribution: {result['bucket_statistics']['binary_distribution']}")
    
    print("\n" + "="*80)
    print("DONE! JSON files generated successfully")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files generated: {len(cfg_files)}")


if __name__ == '__main__':
    main()
