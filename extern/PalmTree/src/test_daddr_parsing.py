"""
Direct test of daddr parsing from a raw line
"""

import re
from vocab import WordVocab

# Test line with daddr
test_line = "mov(0x7288e:0.76383389:0.00000000:0.51724138) esi imm\tlea(0x72893:0.76383389:0.00000000:0.60344828) rdx daddr(0x7a182:0.81367236:0.18096578:0.00000000)"

print("Test line:")
print(test_line)
print("\n" + "="*80 + "\n")

# Define patterns
addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')

# Split into instructions
instructions = test_line.split('\t')
print(f"Number of instructions: {len(instructions)}\n")

for idx, inst_text in enumerate(instructions):
    print(f"Instruction {idx}: {inst_text}")
    
    # Try to match opcode with positions
    match = addr_pattern.match(inst_text)
    if match:
        opcode = match.group(1)
        binary_pos = float(match.group(3))
        function_pos = float(match.group(4))
        bb_pos = float(match.group(5))
        
        print(f"  Opcode: {opcode}")
        print(f"  Binary pos: {binary_pos}, Function pos: {function_pos}, BB pos: {bb_pos}")
        
        # Parse operands
        operands_text = inst_text[match.end():].strip()
        print(f"  Operands text: '{operands_text}'")
        
        if operands_text:
            for operand in operands_text.split():
                print(f"    Checking operand: '{operand}'")
                
                if nested_addr_pattern.match(operand):
                    print(f"      → Matched as nested address")
                elif daddr_pattern.match(operand):
                    dm = daddr_pattern.match(operand)
                    print(f"      → Matched as DADDR!")
                    print(f"        Binary: {dm.group(2)}, Function: {dm.group(3)}, BB: {dm.group(4)}")
                elif var_pattern.match(operand):
                    print(f"      → Matched as var")
                else:
                    print(f"      → Regular operand: {operand}")
    print()

print("\n" + "="*80)
print("Testing regex directly on 'daddr(0x7a182:0.81367236:0.18096578:0.00000000)'")
test_daddr = "daddr(0x7a182:0.81367236:0.18096578:0.00000000)"
if daddr_pattern.match(test_daddr):
    print("✓ MATCHES!")
    m = daddr_pattern.match(test_daddr)
    print(f"  Groups: {m.groups()}")
else:
    print("✗ Does not match")
