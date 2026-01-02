#!/usr/bin/env python3
"""
Test script to demonstrate the tokenization improvements in process.py

This script shows how memory operands and PLT symbols are now properly tokenized.
"""

import re

def tokenize_operand_old(operand):
    """Old version - joins tokens back together"""
    tokens = re.split(r'([\[\]\+\-\*])', operand)
    result = ''.join([t.strip() for t in tokens if t.strip()])
    return result

def tokenize_operand_new(operand):
    """New version - keeps tokens separate"""
    tokens = re.split(r'([\[\]\+\-\*])', operand)
    formatted = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # Keep delimiters and other tokens separate
        formatted.append(token)
    return ' '.join(formatted)

# Test cases
test_operands = [
    "[rbp+var(0x8)]",
    "[rax+rbx*4]",
    "[rsp+0x10]",
    "[rip+0x200]",
    "qword ptr [rbp-0x8]",
    "[rax]",
]

print("=" * 80)
print("TOKENIZATION COMPARISON")
print("=" * 80)
print()

for operand in test_operands:
    old = tokenize_operand_old(operand)
    new = tokenize_operand_new(operand)
    print(f"Original:  {operand}")
    print(f"Old way:   {old}  (joined, hard to parse)")
    print(f"New way:   {new}  (space-separated tokens)")
    print()

print("=" * 80)
print("EXPECTED OUTPUT IN .txt FILES")
print("=" * 80)
print()
print("Before fix:")
print("  mov(0x1000:...) rax [rbp+var(0x8)]")
print("  ^^^ Hard to tokenize - no spaces between brackets and operands")
print()
print("After fix:")
print("  mov(0x1000:...) rax [ rbp + var(0x8) ]")
print("  ^^^ Easy to tokenize - each symbol is separate")
print()

print("=" * 80)
print("PLT SYMBOL HANDLING")
print("=" * 80)
print()
print("Before fix:")
print("  call daddr(0x1030:...)")
print("  ^^^ PLT symbols not shown")
print()
print("After fix:")
print("  call .printf  or  call .malloc")
print("  ^^^ PLT symbols preserved (following cfg_hierarchical_icfg_ida.py)")
print()

print("=" * 80)
print("KEY CHANGES IN process.py")
print("=" * 80)
print("""
1. Tokenization now keeps delimiters separate:
   - Old: formatted_parts.append(norm_token) → ''.join(formatted_parts)
   - New: formatted_parts.append(token) for each → ' '.join(formatted_parts)
   - Added: if token in ['[', ']', '+', '-', '*']: formatted_parts.append(token)

2. PLT symbol detection improved:
   - Checks section_info for '.plt' in section name
   - If address in symbol_map and is PLT, uses symbol name directly
   - Example: .printf, .malloc, .free, .calloc

3. Log files now saved to separate directory:
   - LOGROOT environment variable (default: SAVEROOT/logs/)
   - Keeps data files (.pkl, .txt) and logs (.log) organized
""")

print("=" * 80)
print("TO REGENERATE DATA WITH FIXES")
print("=" * 80)
print("""
# Option 1: Reprocess specific binaries
export SAVEROOT=/data/kun/jtransdata/addr_extract
export DATAROOT=/data/kun/jtransdata/binaries
export LOGROOT=/data/kun/jtransdata/logs

idat -A -S"process.py" /path/to/binary.strip

# Option 2: Batch reprocess (if you have a script)
for binary in /data/kun/jtransdata/binaries/*.strip; do
    idat -A -S"process.py" "$binary"
done

# Option 3: Use existing run scripts (update paths as needed)
./run_ida_generation.sh
""")
