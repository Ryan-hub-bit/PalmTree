#!/usr/bin/env python3
"""
Script to collect all unique instruction patterns from assembly instruction files.
Each line contains instructions separated by tabs.
Each instruction has tokens separated by spaces.
Example: "mov  dword  [ rax + 0x84 ]  ecx\tadd  rsp   0x70"
"""

import re
from collections import defaultdict
import os

def normalize_pattern(instruction):
    """
    Normalize an instruction to extract its pattern.
    Instructions are space-separated tokens.
    Examples:
        "mov  dword  [ rax + 0x84 ]  ecx" -> "mov SIZE [MEM_COMPLEX] REG"
        "add  rsp   0x70" -> "add REG IMM"
        "pop  rbp" -> "pop REG"
    """
    instruction = instruction.strip()
    if not instruction:
        return None
    
    # Split by spaces to get all tokens
    tokens = instruction.split()
    if len(tokens) == 0:
        return None
    
    opcode = tokens[0].lower()
    
    # Filter out non-instruction patterns
    if opcode.startswith('0x'):
        return None
    
    # Skip if opcode doesn't look like an assembly instruction
    if not re.match(r'^[a-z][a-z0-9\.]*$', opcode):
        return None
    
    if len(tokens) == 1:
        # Instruction with no operands (e.g., ret, nop)
        return opcode
    
    # Process remaining tokens to normalize operands
    normalized_parts = []
    i = 1
    
    while i < len(tokens):
        token = tokens[i].lower()
        
        # Check for size specifiers (byte, word, dword, qword, etc.)
        if token in ['byte', 'word', 'dword', 'qword', 'tbyte', 'xmmword', 'ymmword', 'ptr']:
            normalized_parts.append('SIZE')
            i += 1
            continue
        
        # Check if this is a bracket (start of memory reference)
        if token == '[':
            # Collect all tokens until closing bracket
            mem_tokens = []
            i += 1
            while i < len(tokens) and tokens[i] != ']':
                mem_tokens.append(tokens[i])
                i += 1
            if i < len(tokens):
                i += 1  # Skip closing bracket
            
            # Check if memory reference is complex (has +, -, *)
            mem_str = ' '.join(mem_tokens)
            if '+' in mem_str or '-' in mem_str or '*' in mem_str:
                normalized_parts.append('[MEM_COMPLEX]')
            else:
                normalized_parts.append('[MEM]')
        else:
            # Single token operand
            normalized_op = normalize_operand(token)
            normalized_parts.append(normalized_op)
            i += 1
    
    pattern = f"{opcode} {' '.join(normalized_parts)}"
    return pattern

def normalize_operand(operand):
    """
    Normalize a single operand token to identify its type.
    """
    operand = operand.strip()
    
    # Check if it's an immediate value (hex or decimal number)
    if re.match(r'^-?0x[0-9a-fA-F]+$', operand) or re.match(r'^-?\d+$', operand):
        return 'IMM'
    
    # Check if it's a register (common x86/x64 registers)
    register_patterns = [
        r'^r[abcd]x$', r'^e[abcd]x$', r'^[abcd]x$', r'^[abcd][hl]$',  # General purpose
        r'^r[sd]i$', r'^e[sd]i$', r'^[sd]i$', r'^[sd]il$',  # Index registers
        r'^rbp$', r'^ebp$', r'^bp$', r'^bpl$',  # Base pointer
        r'^rsp$', r'^esp$', r'^sp$', r'^spl$',  # Stack pointer
        r'^r\d+[dwb]?$',  # r8-r15 and variants
        r'^xmm\d+$', r'^ymm\d+$', r'^zmm\d+$',  # SIMD registers
        r'^[cdefgs]s$',  # Segment registers
        r'^st\(\d+\)$', r'^st\d*$',  # FPU registers
        r'^mm\d+$',  # MMX registers
    ]
    
    for pattern in register_patterns:
        if re.match(pattern, operand, re.IGNORECASE):
            return 'REG'
    
    # If it starts with a letter and contains only alphanumeric/underscore, it's likely a label
    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', operand):
        return 'LABEL'
    
    # Default to unknown operand type
    return 'OPERAND'

def process_file(filepath, patterns_dict):
    """
    Process a file and collect instruction patterns.
    Each line contains instructions separated by tabs.
    """
    print(f"Processing {filepath}...")
    line_count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_count += 1
                
                # Split by tab to get individual instructions
                instructions = line.strip().split('\t')
                
                # Process each instruction
                for instruction in instructions:
                    instruction = instruction.strip()
                    if not instruction:
                        continue
                    
                    # Normalize the instruction pattern
                    pattern = normalize_pattern(instruction)
                    if pattern:
                        patterns_dict[pattern].add(instruction)
                
                if line_count % 100000 == 0:
                    print(f"  Processed {line_count} lines, found {len(patterns_dict)} patterns so far...")
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"  Total lines processed: {line_count}")
    return line_count

def main():
    data_dir = '/home/louie/PalmTree/data'
    
    files_to_process = [
        os.path.join(data_dir, 'cfg_2.txt'),
        os.path.join(data_dir, 'dfg_2.txt')
    ]
    
    # Dictionary to store patterns and their examples
    # Key: normalized pattern, Value: set of actual instructions
    patterns = defaultdict(set)
    
    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath, patterns)
        else:
            print(f"Warning: {filepath} not found, skipping...")
    
    # Sort patterns by opcode and frequency
    sorted_patterns = sorted(patterns.items(), 
                            key=lambda x: (x[0].split()[0], -len(x[1])))
    
    # Write results to output file
    output_file = os.path.join(data_dir, 'instruction_patterns.txt')
    print(f"\nWriting results to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Instruction Patterns Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total unique patterns found: {len(patterns)}\n\n")
        f.write("=" * 80 + "\n\n")
        
        current_opcode = None
        for pattern, examples in sorted_patterns:
            opcode = pattern.split()[0]
            
            # Add separator when opcode changes
            if opcode != current_opcode:
                if current_opcode is not None:
                    f.write("\n" + "-" * 80 + "\n\n")
                current_opcode = opcode
            
            f.write(f"Pattern: {pattern}\n")
            f.write(f"Count: {len(examples)} unique variations\n")
            
            # Write up to 5 example instructions
            f.write("Examples:\n")
            for i, example in enumerate(sorted(examples)[:5]):
                f.write(f"  {example}\n")
            if len(examples) > 5:
                f.write(f"  ... and {len(examples) - 5} more\n")
            f.write("\n")
    
    print(f"\nAnalysis complete!")
    print(f"Total unique patterns: {len(patterns)}")
    print(f"Results saved to: {output_file}")
    
    # Print summary statistics
    print("\nTop 20 most common patterns:")
    for i, (pattern, examples) in enumerate(sorted_patterns[:20], 1):
        print(f"{i:2d}. {pattern:40s} - {len(examples):5d} variations")

if __name__ == '__main__':
    main()
