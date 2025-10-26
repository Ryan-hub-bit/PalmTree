#!/usr/bin/env python3
"""
Convert assembly instructions to minimal semantic format.
ONLY converts instructions where we add new information:
- mov variants → LD/ST/CP (adds direction clarity)
- movsx/movzx → LD.sx/LD.zx (adds direction + extension type)

Everything else stays as-is - no unnecessary renaming.
"""

import re
from pathlib import Path
from tqdm import tqdm

# MOV variants that get converted to LD/ST/CP
MOV_OPCODES = {'mov', 'movq', 'movl', 'movw', 'movb', 'movabs'}

# Extension opcodes that get special handling
SIGN_EXTEND_OPCODES = {'movsx', 'movsxd'}
ZERO_EXTEND_OPCODES = {'movzx'}


def is_memory_operand(op):
    """Check if operand contains memory reference (has brackets)."""
    return '[' in op


def simplify_operand(op):
    """Clean operand for semantic format."""
    # Remove size hints (qword, dword, etc.) but keep [brackets]
    op = re.sub(r'\b(qword|dword|word|byte)\s+', '', op)
    # Remove 'ptr' keyword
    op = re.sub(r'\s*ptr\s*', '', op)
    return op.strip()


def instruction_to_semantic(instruction):
    """
    Convert instruction to semantic format.
    ONLY converts mov/movsx/movzx - everything else passes through unchanged.
    
    Input format: "opcode dest src" (space-separated tokens, Intel syntax)
    """
    instruction = instruction.strip()
    if not instruction:
        return ""
    
    tokens = instruction.split()
    if not tokens:
        return instruction
    
    opcode = tokens[0].lower()
    
    # ONLY handle mov variants - add direction information
    if opcode in MOV_OPCODES and len(tokens) >= 3:
        # Find memory operand boundaries
        mem_start = None
        mem_end = None
        for i in range(1, len(tokens)):
            if '[' in tokens[i] and mem_start is None:
                mem_start = i
            if ']' in tokens[i] and mem_end is None:
                mem_end = i + 1
                break
        
        if mem_start is not None and mem_end is not None:
            # Has memory operand
            mem_tokens = tokens[mem_start:mem_end]
            mem_operand = simplify_operand(' '.join(mem_tokens))
            
            # Tokens before memory (excluding size prefix like qword/dword/etc)
            before_mem = []
            for i in range(1, mem_start):
                tok_lower = tokens[i].lower()
                if tok_lower not in ['qword', 'dword', 'word', 'byte']:
                    before_mem.append(tokens[i])
            
            # Tokens after memory
            after_mem = tokens[mem_end:]
            
            # Intel syntax: mov dest, src
            # If no register before memory → memory is dest → ST
            # If register before memory → memory is src → LD
            if not before_mem and after_mem:
                # mov [mem], reg → ST reg [mem]
                reg = simplify_operand(' '.join(after_mem))
                return f"ST {reg} {mem_operand}"
            elif before_mem and not after_mem:
                # mov reg, [mem] → LD [mem] reg  
                reg = simplify_operand(' '.join(before_mem))
                return f"LD {mem_operand} {reg}"
            elif before_mem and after_mem:
                # Both before and after? Shouldn't happen in normal mov
                # Treat as LD by default
                reg = simplify_operand(' '.join(before_mem))
                return f"LD {mem_operand} {reg}"
        else:
            # No memory operand → CP (register to register or imm to reg)
            # mov dest, src → CP dest src
            dest = simplify_operand(tokens[1]) if len(tokens) > 1 else ""
            src_tokens = tokens[2:]
            src = simplify_operand(' '.join(src_tokens)) if src_tokens else ""
            return f"CP {dest} {src}"
    
    # ONLY handle sign extension
    if opcode in SIGN_EXTEND_OPCODES and len(tokens) >= 3:
        mem_start = None
        mem_end = None
        for i in range(1, len(tokens)):
            if '[' in tokens[i] and mem_start is None:
                mem_start = i
            if ']' in tokens[i] and mem_end is None:
                mem_end = i + 1
                break
        
        if mem_start is not None and mem_end is not None:
            mem_tokens = tokens[mem_start:mem_end]
            mem_operand = simplify_operand(' '.join(mem_tokens))
            
            # Get register (before memory, excluding size keywords)
            reg_tokens = []
            for i in range(1, mem_start):
                tok_lower = tokens[i].lower()
                if tok_lower not in ['qword', 'dword', 'word', 'byte']:
                    reg_tokens.append(tokens[i])
            reg = simplify_operand(' '.join(reg_tokens)) if reg_tokens else ""
            
            # movsx dest, src → LD.sx src dest
            return f"LD.sx {mem_operand} {reg}"
        else:
            # Register to register: movsx dest src
            dest = simplify_operand(tokens[1]) if len(tokens) > 1 else ""
            src = simplify_operand(' '.join(tokens[2:])) if len(tokens) > 2 else ""
            return f"sext {dest} {src}"
    
    # ONLY handle zero extension
    if opcode in ZERO_EXTEND_OPCODES and len(tokens) >= 3:
        mem_start = None
        mem_end = None
        for i in range(1, len(tokens)):
            if '[' in tokens[i] and mem_start is None:
                mem_start = i
            if ']' in tokens[i] and mem_end is None:
                mem_end = i + 1
                break
        
        if mem_start is not None and mem_end is not None:
            mem_tokens = tokens[mem_start:mem_end]
            mem_operand = simplify_operand(' '.join(mem_tokens))
            
            # Get register (before memory, excluding size keywords)
            reg_tokens = []
            for i in range(1, mem_start):
                tok_lower = tokens[i].lower()
                if tok_lower not in ['qword', 'dword', 'word', 'byte']:
                    reg_tokens.append(tokens[i])
            reg = simplify_operand(' '.join(reg_tokens)) if reg_tokens else ""
            
            # movzx dest, src → LD.zx src dest
            return f"LD.zx {mem_operand} {reg}"
        else:
            # Register to register
            dest = simplify_operand(tokens[1]) if len(tokens) > 1 else ""
            src = simplify_operand(' '.join(tokens[2:])) if len(tokens) > 2 else ""
            return f"zext {dest} {src}"
    
    # Everything else: keep as-is
    return instruction


def convert_line(line):
    """
    Convert a line with tab-separated instructions.
    Format: "instruction1\\tinstruction2\\t..."
    Each instruction is a space-separated token sequence.
    """
    line = line.strip()
    if not line:
        return ""
    
    # Split by tab to get individual instructions
    instructions = line.split('\t')
    converted_instructions = []
    
    for instruction in instructions:
        converted = instruction_to_semantic(instruction)
        converted_instructions.append(converted)
    
    return '\t'.join(converted_instructions)


def process_file(input_path, output_path, log_path=None):
    """Convert entire file to semantic format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if log_path:
        log_path = Path(log_path)
        log_file = open(log_path, 'w')
    else:
        log_file = None
    
    print(f"Converting: {input_path.name}")
    print(f"       To: {output_path.name}")
    
    # Count lines for progress bar
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    converted_count = 0
    total_instructions = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin:
        with open(output_path, 'w', encoding='utf-8') as fout:
            for line in tqdm(fin, total=total_lines, desc="Converting"):
                original = line.strip()
                converted = convert_line(line)
                
                # Count conversions
                if original != converted.strip():
                    converted_count += 1
                total_instructions += len(original.split())
                
                fout.write(converted + '\n')
    
    print(f"✓ Converted {total_lines:,} lines")
    print(f"  {converted_count:,} lines modified (mov/movsx/movzx → LD/ST/CP)")
    
    if log_file:
        log_file.write(f"Conversion Summary:\n")
        log_file.write(f"  Total lines: {total_lines:,}\n")
        log_file.write(f"  Modified lines: {converted_count:,}\n")
        log_file.write(f"\nOnly converted:\n")
        log_file.write(f"  - mov/movq/movl/movw/movb/movabs → LD/ST/CP\n")
        log_file.write(f"  - movsx/movsxd → LD.sx/sext\n")
        log_file.write(f"  - movzx → LD.zx/zext\n")
        log_file.write(f"\nAll other instructions kept as-is.\n")
    
    if log_file:
        log_file.close()


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python convert_to_semantic.py <input_file> <output_file> [log_file]")
        print("\nExample:")
        print("  python convert_to_semantic.py data/cfg_2.txt data/cfg_2_semantic.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    log_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    process_file(input_file, output_file, log_file)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
