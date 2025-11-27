#!/usr/bin/env python3
"""
Test script for displacement handling in cfg_hierarchical_icfg_ida.py

This simulates the logic without requiring IDA Pro.
"""

import re

HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def simulate_clean_ida_disasm_logic(op_type_name, op_value, op_text):
    """
    Simulate the logic from clean_ida_disasm() for different operand types.
    
    Args:
        op_type_name: String like 'o_imm', 'o_displ', 'o_near', etc.
        op_value: Numeric value of the operand
        op_text: Original text representation
    
    Returns:
        Processed operand string
    """
    op = op_text
    
    # Check if it's an immediate value
    if op_type_name == 'o_imm':
        op = "imm"
    # Handle displacement operands (like [rax + 0x20]) - mark for special treatment
    elif op_type_name == 'o_displ':
        # Mark displacement values with special token so they won't be wrapped with address()
        if op_value != 0:
            if not op.startswith('0x') and not op.startswith('['):
                op = f"disp_{hex(op_value)}"
            # Keep the original format if it's already formatted
    # For operands that reference code/data addresses (but NOT displacements)
    elif op_type_name in ['o_near', 'o_mem', 'o_far']:
        if op_value != 0:
            # Check if it's a symbol name (not already a hex address)
            if not op.startswith('0x') and not op.startswith('['):
                op = hex(op_value)
    
    return op


def simulate_build_chunk_inline_logic(tok, min_addr=0x400000, addr_positions=None):
    """
    Simulate the logic from build_chunk_inline() for processing tokens.
    
    Args:
        tok: Token to process
        min_addr: Minimum address of binary
        addr_positions: Dict of known code addresses
    
    Returns:
        Formatted output string
    """
    if addr_positions is None:
        addr_positions = {}
    
    # Check if this is a displacement token (marked with "disp_" prefix)
    if isinstance(tok, str) and tok.startswith('disp_0x'):
        # This is a displacement operand - keep the hex value without address() wrapper
        hex_part = tok[5:]  # Remove "disp_" prefix
        return hex_part
    
    # Check token for 0x prefix
    mk_hex = None
    if isinstance(tok, str) and tok.startswith('0x') and bool(HEX_RE.fullmatch(tok)):
        mk_hex = tok
    
    if mk_hex is not None:
        try:
            tgt = int(mk_hex, 16)
        except Exception:
            tgt = None
        
        if tgt is not None and tgt >= min_addr:
            if tgt in addr_positions:
                # Code address: use hierarchical positions
                return f"address({mk_hex}:0.5:0.3:0.2)"  # Simplified for testing
            else:
                # Data address
                return f"address({mk_hex}:2.0:0.0:0.0)"  # Simplified for testing
        else:
            # It's an immediate value
            return "imm"
    else:
        if tok.startswith("var_"):
            return "var"
        elif tok.startswith("arg_"):
            return "arg"
        else:
            return tok


def test_displacement_handling():
    """Test various operand types and their handling."""
    
    print("=" * 70)
    print("TESTING DISPLACEMENT HANDLING")
    print("=" * 70)
    
    # Test cases: (op_type, op_value, op_text, expected_after_clean, expected_after_build)
    test_cases = [
        # Displacement operands (struct offsets)
        ("o_displ", 0x20, "symbol", "disp_0x20", "0x20"),
        ("o_displ", 0x10, "field", "disp_0x10", "0x10"),
        ("o_displ", 0xc, "data", "disp_0xc", "0xc"),
        ("o_displ", 0x100, "offset", "disp_0x100", "0x100"),
        
        # Immediate values
        ("o_imm", 5, "5", "imm", "imm"),
        ("o_imm", 100, "100", "imm", "imm"),
        
        # Code addresses (should get address() wrapper)
        ("o_near", 0x401000, "func_main", "0x401000", "address(0x401000:0.5:0.3:0.2)"),
        ("o_near", 0x402500, "sub_402500", "0x402500", "address(0x402500:2.0:0.0:0.0)"),
        
        # Far calls/jumps (16-bit segment:offset, rare in modern code but should be tested)
        ("o_far", 0x401000, "far_func", "0x401000", "address(0x401000:0.5:0.3:0.2)"),
        ("o_far", 0x500000, "far_target", "0x500000", "address(0x500000:2.0:0.0:0.0)"),
        
        # Memory references
        ("o_mem", 0x404000, "data_ptr", "0x404000", "address(0x404000:2.0:0.0:0.0)"),
        ("o_mem", 0x405000, "global_var", "0x405000", "address(0x405000:2.0:0.0:0.0)"),
        
        # Already formatted (should be kept as-is in clean_ida_disasm)
        ("o_displ", 0x20, "[rbp+0x20]", "[rbp+0x20]", "[rbp+0x20]"),
        ("o_near", 0x401000, "0x401000", "0x401000", "address(0x401000:0.5:0.3:0.2)"),
        ("o_far", 0x401000, "0x401000", "0x401000", "address(0x401000:0.5:0.3:0.2)"),
    ]
    
    print("\nTest Cases:\n")
    print(f"{'Type':<12} {'Value':<12} {'Input':<20} {'After clean_ida':<20} {'After build_chunk':<30}")
    print("-" * 110)
    
    # Known code addresses for testing
    addr_positions = {0x401000: True}  # Simplified - just checking existence
    
    all_passed = True
    for op_type, op_value, op_text, expected_clean, expected_build in test_cases:
        # Step 1: clean_ida_disasm
        result_clean = simulate_clean_ida_disasm_logic(op_type, op_value, op_text)
        
        # Step 2: build_chunk_inline
        result_build = simulate_build_chunk_inline_logic(result_clean, min_addr=0x400000, addr_positions=addr_positions)
        
        # Check results
        clean_ok = result_clean == expected_clean
        build_ok = result_build == expected_build
        passed = clean_ok and build_ok
        
        status = "✓" if passed else "✗"
        print(f"{status} {op_type:<12} {hex(op_value):<12} {op_text:<20} {result_clean:<20} {result_build:<30}")
        
        if not passed:
            all_passed = False
            if not clean_ok:
                print(f"  ❌ clean_ida: expected '{expected_clean}', got '{result_clean}'")
            if not build_ok:
                print(f"  ❌ build_chunk: expected '{expected_build}', got '{result_build}'")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 70)
    
    return all_passed


def test_real_world_example():
    """Test the original problematic example."""
    
    print("\n\n" + "=" * 70)
    print("REAL-WORLD EXAMPLE TEST")
    print("=" * 70)
    
    print("\nOriginal instruction: mov rax, [rax + 0x20]")
    print("\nProcessing flow:\n")
    
    # The displacement part: 0x20
    op_type = "o_displ"
    op_value = 0x20
    op_text = "symbol_offset"  # IDA might show this as a symbol
    
    print(f"1. IDA operand type: {op_type}")
    print(f"   IDA operand value: {hex(op_value)}")
    print(f"   IDA operand text: {op_text}")
    
    # Step 1: clean_ida_disasm
    result_clean = simulate_clean_ida_disasm_logic(op_type, op_value, op_text)
    print(f"\n2. After clean_ida_disasm(): {result_clean}")
    
    # Step 2: build_chunk_inline
    result_build = simulate_build_chunk_inline_logic(result_clean, min_addr=0x400000)
    print(f"3. After build_chunk_inline(): {result_build}")
    
    print(f"\n4. Final output: mov(0xa659:0.25:0.18:0.13) rax [ rax + {result_build} ]")
    
    # Check if it's correct
    if result_build == "0x20":
        print("\n✅ CORRECT: Displacement stays as plain hex (0x20), not wrapped with address()")
    else:
        print(f"\n❌ WRONG: Expected '0x20', got '{result_build}'")
    
    print("=" * 70)


if __name__ == "__main__":
    # Run tests
    passed = test_displacement_handling()
    test_real_world_example()
    
    # Exit code
    exit(0 if passed else 1)
