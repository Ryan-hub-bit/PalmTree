#!/usr/bin/env python3
"""
Convert assembly instructions to minimal semantic format.
Handles the PalmTree data format: space-separated tokens, tab-separated basic blocks.
"""

import re
from pathlib import Path
from tqdm import tqdm

# Known x86 opcodes (simplified list - add more as needed)
KNOWN_OPCODES = {
    # Data movement
    'mov', 'movq', 'movl', 'movw', 'movb', 'movabs',
    'movsx', 'movsxd', 'movzx',
    'movss', 'movsd', 'movaps', 'movapd', 'movups', 'movupd',
    'movdqa', 'movdqu',
    'lea', 'xchg',
    # Arithmetic
    'add', 'sub', 'imul', 'mul', 'idiv', 'div', 'inc', 'dec', 'neg',
    'adc', 'sbb',
    'addss', 'addsd', 'addps', 'addpd',
    'subss', 'subsd', 'subps', 'subpd',
    'mulss', 'mulsd', 'mulps', 'mulpd',
    'divss', 'divsd', 'divps', 'divpd',
    # Logical
    'and', 'or', 'xor', 'not',
    'andps', 'andpd', 'orps', 'orpd', 'xorps', 'xorpd',
    # Shifts
    'shl', 'shr', 'sal', 'sar', 'rol', 'ror', 'shld', 'shrd',
    # Comparisons
    'cmp', 'test',
    'cmpss', 'cmpsd', 'cmpps', 'cmppd',
    'comiss', 'comisd', 'ucomiss', 'ucomisd',
    # Conditional moves
    'cmove', 'cmovz', 'cmovne', 'cmovnz',
    'cmovl', 'cmovle', 'cmovg', 'cmovge',
    'cmova', 'cmovae', 'cmovb', 'cmovbe',
    'cmovs', 'cmovns',
    # Jumps
    'je', 'jz', 'jne', 'jnz',
    'jl', 'jle', 'jg', 'jge',
    'ja', 'jae', 'jb', 'jbe',
    'js', 'jns', 'jo', 'jno', 'jp', 'jnp',
    'jmp',
    # Stack
    'push', 'pop', 'pushf', 'pushfq', 'popf', 'popfq',
    # Control
    'call', 'ret', 'retn', 'leave', 'enter',
    'nop', 'hlt', 'int', 'syscall', 'sysenter', 'sysexit',
    # Bit ops
    'bt', 'bts', 'btr', 'btc', 'bsf', 'bsr', 'bswap',
    # String ops
    'rep', 'repe', 'repz', 'repne', 'repnz',
    'movsb', 'movsw', 'movsd', 'movsq',
    'stosb', 'stosw', 'stosd', 'stosq',
    'lodsb', 'lodsw', 'lodsd', 'lodsq',
    'scasb', 'scasw', 'scasd', 'scasq',
    # Set
    'sete', 'setz', 'setne', 'setnz',
    'setl', 'setle', 'setg', 'setge',
    'seta', 'setae', 'setb', 'setbe',
    'sets', 'setns',
    # Misc
    'cwd', 'cdq', 'cqo', 'cbw', 'cwde', 'cdqe',
    'cmpxchg',
}

MOV_OPCODES = {'mov', 'movq', 'movl', 'movw', 'movb', 'movabs'}
SIGN_EXTEND_OPCODES = {'movsx', 'movsxd'}
ZERO_EXTEND_OPCODES = {'movzx'}


def is_memory_operand(tokens, start_idx):
    """Check if tokens starting at start_idx form a memory operand (contains '[')."""
    if start_idx >= len(tokens):
        return False, start_idx
    
    # Check if this token or following tokens contain '['
    idx = start_idx
    has_bracket = False
    while idx < len(tokens):
        if '[' in tokens[idx]:
            has_bracket = True
        if ']' in tokens[idx]:
            return True, idx + 1
        if has_bracket:
            idx += 1
        else:
            return False, start_idx + 1
    return has_bracket, idx


def parse_instruction(tokens, start_idx):
    """
    Parse one instruction from token list starting at start_idx.
    Returns (end_idx, opcode, operand_tokens).
    """
    if start_idx >= len(tokens):
        return start_idx, None, []
    
    opcode = tokens[start_idx].lower()
    if opcode not in KNOWN_OPCODES:
        # Unknown opcode, skip this token
        return start_idx + 1, None, []
    
    idx = start_idx + 1
    operand_tokens = []
    
    # Parse operands until we hit next opcode or end
    while idx < len(tokens):
        token = tokens[idx]
        
        # Check if this is the next opcode
        if token.lower() in KNOWN_OPCODES:
            break
        
        # If we hit a '[', consume until ']'
        if '[' in token:
            operand_start = idx
            while idx < len(tokens) and ']' not in tokens[idx]:
                idx += 1
            if idx < len(tokens):
                idx += 1  # consume the ']'
            operand_tokens.append(tokens[operand_start:idx])
        else:
            operand_tokens.append([token])
            idx += 1
    
    return idx, opcode, operand_tokens


def operand_tokens_to_string(operand_tokens):
    """Convert list of token lists to operand strings."""
    operands = []
    for tok_list in operand_tokens:
        # Join tokens, clean up spaces
        op_str = ' '.join(tok_list)
        # Remove size hints
        op_str = re.sub(r'\b(qword|dword|word|byte)\s*', '', op_str)
        op_str = re.sub(r'\s*ptr\s*', '', op_str)
        operands.append(op_str.strip())
    return operands


def convert_instruction(opcode, operand_tokens):
    """Convert single instruction to semantic format."""
    operands = operand_tokens_to_string(operand_tokens)
    
    # MOV variants → LD/ST/CP
    if opcode in MOV_OPCODES:
        if len(operands) == 2:
            src, dest = operands
            src_is_mem = '[' in src
            dest_is_mem = '[' in dest
            
            if dest_is_mem and not src_is_mem:
                return f"ST {src} {dest}"
            elif src_is_mem and not dest_is_mem:
                return f"LD {src} {dest}"
            else:
                return f"CP {src} {dest}"
    
    # Sign extension
    if opcode in SIGN_EXTEND_OPCODES:
        if len(operands) == 2:
            src, dest = operands
            if '[' in src:
                return f"LD.sx {src} {dest}"
            else:
                return f"sext {src} {dest}"
    
    # Zero extension
    if opcode in ZERO_EXTEND_OPCODES:
        if len(operands) == 2:
            src, dest = operands
            if '[' in src:
                return f"LD.zx {src} {dest}"
            else:
                return f"zext {src} {dest}"
    
    # Everything else: keep as-is
    inst_str = opcode
    if operands:
        inst_str += ' ' + ' '.join(operands)
    return inst_str


def convert_bb_sequence(token_sequence):
    """Convert a basic block token sequence to semantic format."""
    tokens = token_sequence.split()
    if not tokens:
        return ""
    
    converted_tokens = []
    idx = 0
    
    while idx < len(tokens):
        end_idx, opcode, operand_tokens = parse_instruction(tokens, idx)
        
        if opcode:
            converted = convert_instruction(opcode, operand_tokens)
            converted_tokens.extend(converted.split())
        else:
            # Unknown token, keep as-is
            if idx < len(tokens):
                converted_tokens.append(tokens[idx])
        
        idx = end_idx
    
    return ' '.join(converted_tokens)


def convert_line(line):
    """Convert a line with tab-separated basic block sequences."""
    line = line.strip()
    if not line:
        return ""
    
    bb_sequences = line.split('\t')
    converted_bbs = [convert_bb_sequence(bb) for bb in bb_sequences]
    return '\t'.join(converted_bbs)


def process_file(input_path, output_path, log_path=None):
    """Convert entire file to semantic format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if log_path:
        log_path = Path(log_path)
        log_file = open(log_path, 'w', encoding='utf-8')
    else:
        log_file = None
    
    print(f"Converting: {input_path.name}")
    print(f"       To: {output_path.name}")
    
    # Count lines for progress bar
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    converted_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin:
        with open(output_path, 'w', encoding='utf-8') as fout:
            for line in tqdm(fin, total=total_lines, desc="Converting"):
                original = line.strip()
                converted = convert_line(line)
                
                if original != converted:
                    converted_count += 1
                
                fout.write(converted + '\n')
    
    print(f"✓ Converted {total_lines:,} lines")
    print(f"  {converted_count:,} lines modified")
    
    if log_file:
        log_file.write(f"Conversion Summary:\n")
        log_file.write(f"  Total lines: {total_lines:,}\n")
        log_file.write(f"  Modified lines: {converted_count:,}\n")
        log_file.write(f"\nConverted:\n")
        log_file.write(f"  - mov variants → LD/ST/CP\n")
        log_file.write(f"  - movsx/movsxd → LD.sx/sext\n")
        log_file.write(f"  - movzx → LD.zx/zext\n")
        log_file.close()


def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python convert_to_semantic_v2.py <input_file> <output_file> [log_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    log_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    process_file(input_file, output_file, log_file)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
