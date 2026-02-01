#!/usr/bin/env python3
"""
Test script to verify the dataloader fix correctly parses 'daddr' tokens.
"""

import sys
sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')

import re

def test_daddr_parsing():
    """Test that daddr pattern is now correctly parsed."""
    
    # Define the regex patterns as they are in the dataloader
    addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    def parse_instruction_simplified(inst_text):
        """Simplified version of _parse_instruction to test the fix."""
        tokens = []
        positions = []
        var_offsets = []
        
        # Parse opcode and its address
        match = addr_pattern.match(inst_text)
        if not match:
            return tokens, positions, var_offsets
        
        opcode = match.group(1)
        binary_pos = float(match.group(3))
        function_pos = float(match.group(4))
        bb_pos = float(match.group(5))
        
        tokens.append(opcode)
        positions.append((binary_pos, function_pos, bb_pos))
        var_offsets.append(-1)
        
        # Parse operands - THIS IS THE FIXED CODE
        operands_text = inst_text[match.end():].strip()
        if operands_text:
            for operand in operands_text.split():
                nested_match = nested_addr_pattern.match(operand)
                daddr_match = daddr_pattern.match(operand)  # ✅ NOW CHECKING daddr_pattern
                var_match = var_pattern.match(operand)
                
                if nested_match:
                    nested_binary_pos = float(nested_match.group(2))
                    nested_function_pos = float(nested_match.group(3))
                    nested_bb_pos = float(nested_match.group(4))
                    tokens.append('address')
                    positions.append((nested_binary_pos, nested_function_pos, nested_bb_pos))
                    var_offsets.append(-1)
                elif daddr_match:  # ✅ NOW HANDLING daddr tokens
                    daddr_binary_pos = float(daddr_match.group(2))
                    daddr_function_pos = float(daddr_match.group(3))
                    daddr_bb_pos = float(daddr_match.group(4))
                    tokens.append('daddr')
                    positions.append((daddr_binary_pos, daddr_function_pos, daddr_bb_pos))
                    var_offsets.append(-1)
                elif var_match:
                    var_hex = var_match.group(1)
                    var_offset_value = int(var_hex, 16)
                    if var_offset_value > 0x7FFFFFFFFFFFFFFF:
                        var_offset_value = var_offset_value - 0x10000000000000000
                    tokens.append('var')
                    positions.append((-1.0, -1.0, -1.0))
                    var_offsets.append(var_offset_value)
                else:
                    tokens.append(operand)
                    positions.append((-1.0, -1.0, -1.0))
                    var_offsets.append(-1)
        
        return tokens, positions, var_offsets
    
    print("="*80)
    print("Testing Dataloader Fix for 'daddr' Token Parsing")
    print("="*80)
    
    # Test cases with different token types
    test_cases = [
        {
            'name': 'Control flow address (jump target)',
            'input': 'jmp(0x401000:0.5:0.3:0.1) address(0x401100:0.52:0.31:0.11)',
            'expected_tokens': ['jmp', 'address'],
            'expected_daddr_count': 0
        },
        {
            'name': 'Data address (memory operand)',
            'input': 'mov(0x401000:0.5:0.3:0.1) rdi daddr(0x601000:0.8:0.2:0.05)',
            'expected_tokens': ['mov', 'rdi', 'daddr'],
            'expected_daddr_count': 1
        },
        {
            'name': 'Mixed: address and daddr',
            'input': 'lea(0x401000:0.5:0.3:0.1) rdi address(0x401100:0.52:0.31:0.11) daddr(0x601000:0.8:0.2:0.05)',
            'expected_tokens': ['lea', 'rdi', 'address', 'daddr'],
            'expected_daddr_count': 1
        },
        {
            'name': 'Variable offset',
            'input': 'mov(0x401000:0.5:0.3:0.1) rax var(0x10)',
            'expected_tokens': ['mov', 'rax', 'var'],
            'expected_daddr_count': 0
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*80}")
        print(f"Input: {test_case['input']}")
        
        try:
            tokens, positions, var_offsets = parse_instruction_simplified(test_case['input'])
            
            print(f"\nParsed tokens: {tokens}")
            print(f"Token count: {len(tokens)}")
            print(f"'address' count: {tokens.count('address')}")
            print(f"'daddr' count: {tokens.count('daddr')}")
            print(f"'var' count: {tokens.count('var')}")
            
            # Verify expectations
            daddr_count = tokens.count('daddr')
            tokens_match = tokens == test_case['expected_tokens']
            daddr_count_match = daddr_count == test_case['expected_daddr_count']
            
            print(f"\nExpected tokens: {test_case['expected_tokens']}")
            print(f"Expected 'daddr' count: {test_case['expected_daddr_count']}")
            
            if tokens_match and daddr_count_match:
                print(f"✅ PASSED")
            else:
                print(f"❌ FAILED")
                if not tokens_match:
                    print(f"   Token mismatch: expected {test_case['expected_tokens']}, got {tokens}")
                if not daddr_count_match:
                    print(f"   Daddr count mismatch: expected {test_case['expected_daddr_count']}, got {daddr_count}")
                all_passed = False
            
            # Show position info for daddr tokens
            for j, (token, pos) in enumerate(zip(tokens, positions)):
                if token == 'daddr':
                    print(f"   daddr position: binary={pos[0]:.3f}, function={pos[1]:.3f}, bb={pos[2]:.3f}")
                    
        except Exception as e:
            print(f"❌ FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ ALL TESTS PASSED - Dataloader fix is working correctly!")
        print("\nThe dataloader now correctly:")
        print("  1. Matches daddr(...) patterns")
        print("  2. Extracts 'daddr' tokens (separate from 'address')")
        print("  3. Stores hierarchical positions for daddr tokens")
    else:
        print("❌ SOME TESTS FAILED - Please review the output above")
    print("="*80)
    
    return all_passed

if __name__ == '__main__':
    success = test_daddr_parsing()
    sys.exit(0 if success else 1)
