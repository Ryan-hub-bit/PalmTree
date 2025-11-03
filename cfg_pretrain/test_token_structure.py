#!/usr/bin/env python3
"""
Test the new token structure with [ADDR_START], [ADDR_END], and [SEQ] tokens
Shows how positions are assigned to different token types
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import AddressTokenizer

def test_token_structure():
    print("=" * 70)
    print("Token Structure Test - [ADDR_START] / [ADDR_END] / [SEQ]")
    print("=" * 70)
    print()
    
    # Create tokenizer
    tokenizer = AddressTokenizer()
    
    # Read a sample BB pair
    data_file = "../bb_pairs_output/atilibusb.so_bb_pairs.txt"
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return
    
    with open(data_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) < 2:
        print("Error: Not enough data")
        return
    
    # Take first pair
    source_bb = lines[0]
    target_bb = lines[1]
    
    print("SOURCE BB:")
    print("-" * 70)
    print(source_bb[:200] + "..." if len(source_bb) > 200 else source_bb)
    print()
    
    # Extract addresses from source
    source_addrs = tokenizer.extract_addresses(source_bb)
    
    print(f"Found {len(source_addrs)} addresses in source BB:")
    for i, addr in enumerate(source_addrs, 1):
        print(f"  {i}. {addr['type']:12s} -> bin_norm={addr['binary_norm']:8.6f}, func_norm={addr['function_norm']:8.6f}")
    print()
    
    # Find start and end positions
    start_pos = (0.0, 0.0)
    end_pos = (0.0, 0.0)
    
    for addr in source_addrs:
        if addr['type'] == 'addr_start':
            start_pos = (addr['binary_norm'], addr['function_norm'])
            print(f"✓ Found addr_start: ({start_pos[0]:.6f}, {start_pos[1]:.6f})")
        elif addr['type'] == 'addr_end':
            end_pos = (addr['binary_norm'], addr['function_norm'])
            print(f"✓ Found addr_end: ({end_pos[0]:.6f}, {end_pos[1]:.6f})")
    print()
    
    print("=" * 70)
    print("Expected Token Structure:")
    print("=" * 70)
    print()
    print("[ADDR_START]  <- Gets position from addr_start:", start_pos)
    print("  mov         <- Regular token, gets (0.0, 0.0)")
    print("  rax         <- Regular token, gets (0.0, 0.0)")
    print("  [ADDR]      <- Address token, gets actual position")
    print("  [SEQ]       <- Separator, gets (0.0, 0.0)")
    print("  lea         <- Regular token, gets (0.0, 0.0)")
    print("  ...")
    print("[ADDR_END]    <- Gets position from addr_end:", end_pos)
    print()
    print("[ADDR_START]  <- Gets position from target addr_start")
    print("  ...")
    print("[ADDR_END]    <- Gets position from target addr_end")
    print()
    
    print("=" * 70)
    print("Key Points:")
    print("=" * 70)
    print("✓ ONLY [ADDR] placeholder tokens get Level 2 positions (binary_norm, func_norm)")
    print("✓ Regular instruction tokens (mov, lea, rax, etc.) get (0.0, 0.0)")
    print("✓ [ADDR_START] and [ADDR_END] get positions from BB boundaries")
    print("✓ [SEQ] tokens separate instructions, get (0.0, 0.0)")
    print("✓ This reflects assembly semantics: addresses have positions, instructions don't")
    print()

if __name__ == "__main__":
    test_token_structure()
