#!/usr/bin/env python3
"""
Generate Scope Prediction Dataset - One Binary at a Time

Processes each CFG file separately to ensure pairs are from the same binary.
Prioritizes control flow pairs, adds random pairs if needed.
"""

import sys
import os
import re
import random
from pathlib import Path
from collections import defaultdict

# Position regex: matches addr(0xADDR:pos1:pos2:pos3)
ADDR_POS_RE = re.compile(r'^(0x[0-9a-fA-F]+):([\d.]+):([\d.]+):([\d.]+)$')


def parse_hierarchical_instruction(inst_str):
    """Parse instruction with hierarchical positions."""
    inst_str = inst_str.strip()
    if not inst_str:
        return None
    
    parts = inst_str.split(maxsplit=1)
    if len(parts) == 0:
        return None
    
    opcode_part = parts[0]
    
    if '(' not in opcode_part:
        return None
    
    opcode, addr_info = opcode_part.split('(', 1)
    addr_info = addr_info.rstrip(')')
    
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
        # Create scope IDs
        'func_id': f"{binary_pos:.6f}",
        'bb_id': f"{binary_pos:.6f}_{func_pos:.6f}"
    }


def extract_pairs_from_sequence(sequence):
    """Extract instruction pairs from a control flow sequence."""
    if len(sequence) < 2:
        return [], [], []
    
    same_bb_pairs = []
    diff_bb_same_func_pairs = []
    diff_func_pairs = []
    
    # Try all consecutive pairs and nearby pairs
    for i in range(len(sequence)):
        for j in range(i + 1, min(i + 10, len(sequence))):  # Look ahead up to 10 instructions
            inst1 = sequence[i]
            inst2 = sequence[j]
            
            # Determine relationship
            if inst1['bb_id'] == inst2['bb_id']:
                # Same BB
                same_bb_pairs.append((inst1, inst2))
            elif inst1['func_id'] == inst2['func_id']:
                # Different BB, Same Function
                diff_bb_same_func_pairs.append((inst1, inst2))
            else:
                # Different Function
                diff_func_pairs.append((inst1, inst2))
    
    return same_bb_pairs, diff_bb_same_func_pairs, diff_func_pairs


def process_single_cfg(cfg_path):
    """
    Process a single CFG file and extract control flow pairs + all instructions.
    All pairs from this file are guaranteed to be from the same binary.
    """
    cf_same_bb = []
    cf_diff_bb_same_func = []
    cf_diff_func = []
    all_instructions = []
    
    with open(cfg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse sequence (tab-separated instructions)
            inst_tokens = line.split('\t')
            sequence = []
            
            for inst_str in inst_tokens:
                parsed = parse_hierarchical_instruction(inst_str)
                if parsed is not None:
                    sequence.append(parsed)
                    all_instructions.append(parsed)
            
            # Extract pairs from this sequence
            if len(sequence) >= 2:
                same_bb, diff_bb, diff_func = extract_pairs_from_sequence(sequence)
                cf_same_bb.extend(same_bb)
                cf_diff_bb_same_func.extend(diff_bb)
                cf_diff_func.extend(diff_func)
    
    # Deduplicate pairs
    def deduplicate_pairs(pairs):
        seen = set()
        unique = []
        for inst1, inst2 in pairs:
            key = (inst1['text'], inst2['text'])
            if key not in seen:
                seen.add(key)
                unique.append((inst1, inst2))
        return unique
    
    cf_same_bb = deduplicate_pairs(cf_same_bb)
    cf_diff_bb_same_func = deduplicate_pairs(cf_diff_bb_same_func)
    cf_diff_func = deduplicate_pairs(cf_diff_func)
    
    return (cf_same_bb, cf_diff_bb_same_func, cf_diff_func), all_instructions


def generate_random_pairs_single_binary(instructions, target_count, pair_type='same_bb'):
    """Generate random pairs from a single binary's instructions."""
    # Group instructions by scope
    by_bb = defaultdict(list)
    by_func = defaultdict(list)
    
    for inst in instructions:
        by_bb[inst['bb_id']].append(inst)
        by_func[inst['func_id']].append(inst)
    
    pairs = []
    attempts = 0
    max_attempts = target_count * 10
    
    if pair_type == 'same_bb':
        # Sample from same BB
        bb_list = [bb_id for bb_id, insts in by_bb.items() if len(insts) >= 2]
        if len(bb_list) == 0:
            return []
        while len(pairs) < target_count and attempts < max_attempts:
            bb_id = random.choice(bb_list)
            inst1, inst2 = random.sample(by_bb[bb_id], 2)
            pairs.append((inst1, inst2))
            attempts += 1
    
    elif pair_type == 'diff_bb_same_func':
        # Sample from different BBs in same function
        func_list = []
        for func_id, func_insts in by_func.items():
            func_bbs = defaultdict(list)
            for inst in func_insts:
                func_bbs[inst['bb_id']].append(inst)
            if len(func_bbs) >= 2:
                func_list.append((func_id, func_bbs))
        
        if len(func_list) == 0:
            return []
        while len(pairs) < target_count and attempts < max_attempts:
            func_id, func_bbs = random.choice(func_list)
            bb1, bb2 = random.sample(list(func_bbs.keys()), 2)
            inst1 = random.choice(func_bbs[bb1])
            inst2 = random.choice(func_bbs[bb2])
            pairs.append((inst1, inst2))
            attempts += 1
    
    elif pair_type == 'diff_func':
        # Sample from different functions
        func_list = list(by_func.keys())
        if len(func_list) < 2:
            return []
        while len(pairs) < target_count and attempts < max_attempts:
            func1, func2 = random.sample(func_list, 2)
            inst1 = random.choice(by_func[func1])
            inst2 = random.choice(by_func[func2])
            pairs.append((inst1, inst2))
            attempts += 1
    
    return pairs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate scope prediction dataset - one binary at a time')
    parser.add_argument('--cfg_dir', type=str, default='../../data/ncfg',
                        help='Directory containing CFG files')
    parser.add_argument('--output', type=str, default='../../data/scope/all_scope.txt',
                        help='Output file')
    parser.add_argument('--target_per_class', type=int, default=50000,
                        help='Target number of pairs per class')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SCOPE PREDICTION DATASET - BINARY BY BINARY")
    print("="*80)
    print(f"Input directory: {args.cfg_dir}")
    print(f"Output: {args.output}")
    print(f"Target per class: {args.target_per_class}")
    print("="*80)
    
    cfg_dir = Path(args.cfg_dir)
    if not cfg_dir.exists():
        print(f"[ERROR] Directory not found: {cfg_dir}")
        sys.exit(1)
    
    # Find all CFG files
    cfg_files = sorted(cfg_dir.glob('*_cfg_2_inline.txt'))
    print(f"\n[INFO] Found {len(cfg_files)} CFG files")
    
    # Collect all pairs across all binaries
    all_same_bb = []
    all_diff_bb_same_func = []
    all_diff_func = []
    
    for cfg_file in cfg_files:
        binary_name = cfg_file.stem.replace('_cfg_8_inline', '')
        print(f"\n[INFO] Processing {binary_name}...")
        
        (cf_same_bb, cf_diff_bb_same_func, cf_diff_func), all_insts = process_single_cfg(cfg_file)
        
        print(f"  Instructions: {len(all_insts)}")
        print(f"  Control flow pairs:")
        print(f"    Same BB: {len(cf_same_bb)}")
        print(f"    Diff BB Same Func: {len(cf_diff_bb_same_func)}")
        print(f"    Diff Func: {len(cf_diff_func)}")
        
        # Add control flow pairs
        all_same_bb.extend([(inst1, inst2, 0) for inst1, inst2 in cf_same_bb])
        all_diff_bb_same_func.extend([(inst1, inst2, 1) for inst1, inst2 in cf_diff_bb_same_func])
        all_diff_func.extend([(inst1, inst2, 2) for inst1, inst2 in cf_diff_func])
        
        # If we need more pairs, generate random ones from this binary
        needed_same_bb = max(0, (args.target_per_class // len(cfg_files)) - len(cf_same_bb))
        needed_diff_bb = max(0, (args.target_per_class // len(cfg_files)) - len(cf_diff_bb_same_func))
        needed_diff_func = max(0, (args.target_per_class // len(cfg_files)) - len(cf_diff_func))
        
        if needed_same_bb > 0:
            random_pairs = generate_random_pairs_single_binary(all_insts, needed_same_bb, 'same_bb')
            all_same_bb.extend([(inst1, inst2, 0) for inst1, inst2 in random_pairs])
            print(f"  Generated {len(random_pairs)} random Same BB pairs")
        
        if needed_diff_bb > 0:
            random_pairs = generate_random_pairs_single_binary(all_insts, needed_diff_bb, 'diff_bb_same_func')
            all_diff_bb_same_func.extend([(inst1, inst2, 1) for inst1, inst2 in random_pairs])
            print(f"  Generated {len(random_pairs)} random Diff BB Same Func pairs")
        
        if needed_diff_func > 0:
            random_pairs = generate_random_pairs_single_binary(all_insts, needed_diff_func, 'diff_func')
            all_diff_func.extend([(inst1, inst2, 2) for inst1, inst2 in random_pairs])
            print(f"  Generated {len(random_pairs)} random Diff Func pairs")
    
    # Collect and balance final dataset
    print("\n" + "="*80)
    print("BALANCING FINAL DATASET")
    print("="*80)
    
    # Shuffle each class
    random.shuffle(all_same_bb)
    random.shuffle(all_diff_bb_same_func)
    random.shuffle(all_diff_func)
    
    # Sample to target size
    final_pairs = []
    final_pairs.extend(all_same_bb[:args.target_per_class])
    final_pairs.extend(all_diff_bb_same_func[:args.target_per_class])
    final_pairs.extend(all_diff_func[:args.target_per_class])
    
    # Shuffle all together
    random.shuffle(final_pairs)
    
    print(f"Final dataset:")
    print(f"  Label 0 (Same BB): {len(all_same_bb[:args.target_per_class])}")
    print(f"  Label 1 (Diff BB Same Func): {len(all_diff_bb_same_func[:args.target_per_class])}")
    print(f"  Label 2 (Diff Func): {len(all_diff_func[:args.target_per_class])}")
    print(f"  Total: {len(final_pairs)}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Saving to {output_path}")
    with open(output_path, 'w') as f:
        for inst1, inst2, label in final_pairs:
            f.write(f"{inst1['text']}\t{inst2['text']}\t{label}\n")
    
    print(f"[INFO] Saved successfully")
    print("="*80)


if __name__ == '__main__':
    main()
