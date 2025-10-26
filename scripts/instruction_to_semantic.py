#!/usr/bin/env python3
"""
Script to convert assembly instructions to semantic sequences.
Makes instructions human-readable and compact.
"""

import re
import sys

# Semantic descriptions for common opcodes (compact form)
OPCODE_SEMANTICS = {
    # Data movement (direction will be added dynamically for mov variants)
    'mov': '{} {}',  # will become LD/ST/CP based on operands
    'movsx': 'sext {} {}',
    'movzx': 'zext {} {}',
    'movsxd': 'sext {} {}',
    'movabs': 'cp {} {}',
    'movq': '{} {}',  # will become LD/ST/CP
    'movd': 'cp {} {}',
    'movss': 'cp.ss {} {}',
    'movsd': 'cp.sd {} {}',
    'movaps': 'cp.aps {} {}',
    'movups': 'cp.ups {} {}',
    'movdqa': 'cp.dqa {} {}',
    'movdqu': 'cp.dqu {} {}',
    'movapd': 'cp.apd {} {}',
    'movupd': 'cp.upd {} {}',
    'lea': 'lea {} {}',
    'xchg': 'xchg {} {}',
    
    # Conditional moves (keep condition suffix)
    'cmove': 'cmov.eq {} {}',
    'cmovne': 'cmov.ne {} {}',
    'cmovz': 'cmov.z {} {}',
    'cmovnz': 'cmov.nz {} {}',
    'cmovg': 'cmov.g {} {}',
    'cmovge': 'cmov.ge {} {}',
    'cmovl': 'cmov.l {} {}',
    'cmovle': 'cmov.le {} {}',
    'cmova': 'cmov.a {} {}',
    'cmovae': 'cmov.ae {} {}',
    'cmovb': 'cmov.b {} {}',
    'cmovbe': 'cmov.be {} {}',
    'cmovs': 'cmov.s {} {}',
    'cmovns': 'cmov.ns {} {}',
    
    # Arithmetic
    'add': 'add {} {}',
    'sub': 'sub {} {}',
    'adc': 'adc {} {}',
    'sbb': 'sbb {} {}',
    'inc': 'inc {}',
    'dec': 'dec {}',
    'neg': 'neg {}',
    'mul': 'mul {}',
    'imul': 'imul {} {}',
    'div': 'div {}',
    'idiv': 'idiv {}',
    'addsd': 'add.sd {} {}',
    'addss': 'add.ss {} {}',
    'addpd': 'add.pd {} {}',
    'subsd': 'sub.sd {} {}',
    'subss': 'sub.ss {} {}',
    'mulsd': 'mul.sd {} {}',
    'mulss': 'mul.ss {} {}',
    'divsd': 'div.sd {} {}',
    'divss': 'div.ss {} {}',
    
    # Logical
    'and': 'and {} {}',
    'or': 'or {} {}',
    'xor': 'xor {} {}',
    'not': 'not {}',
    'test': 'test {} {}',
    'andpd': 'and.pd {} {}',
    'orpd': 'or.pd {} {}',
    'xorpd': 'xor.pd {} {}',
    'pxor': 'xor.p {} {}',
    'pand': 'and.p {} {}',
    'por': 'or.p {} {}',
    
    # Shift/Rotate
    'shl': 'shl {} {}',
    'shr': 'shr {} {}',
    'sal': 'sal {} {}',
    'sar': 'sar {} {}',
    'rol': 'rol {} {}',
    'ror': 'ror {} {}',
    'shld': 'shld {} {}',
    'shrd': 'shrd {} {}',
    'psllq': 'shl.pq {} {}',
    'psrlq': 'shr.pq {} {}',
    
    # Comparison
    'cmp': 'cmp {} {}',
    'cmpxchg': 'cmpxchg {} {}',
    'ucomisd': 'ucomi.sd {} {}',
    'ucomiss': 'ucomi.ss {} {}',
    'comisd': 'comi.sd {} {}',
    'comiss': 'comi.ss {} {}',
    
    # Stack operations
    'push': 'push {}',
    'pop': 'pop {}',
    'pushfq': 'push.fq',
    'popfq': 'pop.fq',
    'pusha': 'push.a',
    'popa': 'pop.a',
    
    # Control flow (keep condition suffixes)
    'call': 'call {}',
    'ret': 'ret',
    'jmp': 'jmp {}',
    'je': 'j.eq {}',
    'jne': 'j.ne {}',
    'jz': 'j.z {}',
    'jnz': 'j.nz {}',
    'jg': 'j.g {}',
    'jge': 'j.ge {}',
    'jl': 'j.l {}',
    'jle': 'j.le {}',
    'ja': 'j.a {}',
    'jae': 'j.ae {}',
    'jb': 'j.b {}',
    'jbe': 'j.be {}',
    'js': 'j.s {}',
    'jns': 'j.ns {}',
    'jo': 'j.o {}',
    'jno': 'j.no {}',
    
    # Set flags
    'sete': 'set.eq {}',
    'setne': 'set.ne {}',
    'setg': 'set.g {}',
    'setge': 'set.ge {}',
    'setl': 'set.l {}',
    'setle': 'set.le {}',
    'seta': 'set.a {}',
    'setae': 'set.ae {}',
    'setb': 'set.b {}',
    'setbe': 'set.be {}',
    
    # String operations
    'rep': 'rep {}',
    'repe': 'rep.e {}',
    'repne': 'rep.ne {}',
    'movsb': 'movs.b',
    'movsw': 'movs.w',
    'movsd': 'movs.d',
    'movsq': 'movs.q',
    'stosb': 'stos.b',
    'stosw': 'stos.w',
    'stosd': 'stos.d',
    'stosq': 'stos.q',
    
    # Conversion
    'cdq': 'cdq',
    'cqo': 'cqo',
    'cwde': 'cwde',
    'cbw': 'cbw',
    'cvtsi2sd': 'cvt.i2sd {} {}',
    'cvtsi2ss': 'cvt.i2ss {} {}',
    'cvttsd2si': 'cvtt.sd2i {} {}',
    'cvttss2si': 'cvtt.ss2i {} {}',
    
    # Bit manipulation
    'bsr': 'bsr {} {}',
    'bsf': 'bsf {} {}',
    'bt': 'bt {} {}',
    'bts': 'bts {} {}',
    'btr': 'btr {} {}',
    'btc': 'btc {} {}',
    
    # Other
    'nop': 'nop',
    'leave': 'leave',
    'endbr64': 'endbr64',
    'xchg': 'xchg {} {}',
    'cltq': 'cltq',
    'syscall': 'syscall',
    'cpuid': 'cpuid',
    'rdtsc': 'rdtsc',
    'lock': 'lock {}',
    'int3': 'int3',
    'ud2': 'ud2',
    'retn': 'ret',
    
    # Additional SSE/SSE2 instructions
    'andps': 'and.ps {} {}',
    'andnpd': 'andn.pd {} {}',
    'xorps': 'xor.ps {} {}',
    'subpd': 'sub.pd {} {}',
    'maxsd': 'max.sd {} {}',
    'minsd': 'min.sd {} {}',
    'cmpsd': 'cmp.sd {} {}',
    'shufpd': 'shuf.pd {} {}',
    'unpckhpd': 'unph.pd {} {}',
    'movlhps': 'movlh.ps {} {}',
    'movmskpd': 'movmsk.pd {} {}',
    
    # SSE integer instructions
    'paddd': 'add.pd {} {}',
    'psubd': 'sub.pd {} {}',
    'pcmpeqb': 'cmpeq.pb {} {}',
    'pcmpeqd': 'cmpeq.pd {} {}',
    'pshufd': 'shuf.pd {} {}',
    'punpckldq': 'unpld.pdq {} {}',
    'punpcklqdq': 'unpld.pqq {} {}',
    'pmovmskb': 'movmsk.pb {} {}',
    
    # x87 FPU instructions
    'fld': 'fld {} {}',
    'fld1': 'fld1',
    'fldz': 'fldz',
    'fstp': 'fstp {} {}',
    'fild': 'fild {} {}',
    'fistp': 'fistp {} {}',
    'faddp': 'fadd.p {} {}',
    'fsubp': 'fsub.p {} {}',
    'fsubrp': 'fsubr.p {} {}',
    'fmulp': 'fmul.p {} {}',
    'fdivrp': 'fdivr.p {} {}',
    'fxch': 'fxch {} {}',
    'fcmovnb': 'fcmov.nb {} {}',
    'fucomi': 'fucomi {} {}',
    'fucomip': 'fucomi.p {} {}',
    'fldcw': 'fldcw {}',
    'fnstcw': 'fnstcw {}',
    
    # Additional conversions and sign extensions
    'cdqe': 'cdqe',
    'cvtsd2ss': 'cvt.sd2ss {} {}',
    'cvtss2sd': 'cvt.ss2sd {} {}',
    
    # Additional set instructions
    'setns': 'set.ns {}',
    'seto': 'set.o {}',
    'sets': 'set.s {}',
    'setpe': 'set.pe {}',
    'setpo': 'set.po {}',
    
    # Additional jumps
    'jpe': 'j.pe {}',
    'jpo': 'j.po {}',
}

def simplify_operand(operand):
    """
    Simplify operand representation for semantic description.
    """
    operand = operand.strip()
    
    # Remove 'dword', 'qword', etc. size specifiers
    operand = re.sub(r'\b(byte|word|dword|qword|tbyte|xmmword|ymmword|ptr)\b', '', operand)
    operand = ' '.join(operand.split())  # Remove extra spaces
    
    # Simplify memory references
    if '[' in operand:
        # Extract just the register/offset part
        match = re.search(r'\[(.*?)\]', operand)
        if match:
            inside = match.group(1).strip()
            # Simplify 'rel' keyword
            inside = inside.replace('rel', '').strip()
            return f"[{inside}]"
    
    return operand

def instruction_to_semantic(instruction):
    """
    Convert an assembly instruction to a semantic description.
    """
    instruction = instruction.strip()
    if not instruction:
        return ""
    
    # Split by spaces to get tokens
    tokens = instruction.split()
    if len(tokens) == 0:
        return ""
    
    opcode = tokens[0].lower()
    
    # Get semantic template
    semantic_template = OPCODE_SEMANTICS.get(opcode)
    
    if not semantic_template:
        # Unknown opcode, return simplified version
        operands = ' '.join([simplify_operand(t) for t in tokens[1:]])
        if operands:
            return f"{opcode} {operands}"
        return opcode
    
    # Extract operands (skip size specifiers and brackets)
    operands = []
    operand_is_memory = []  # Track which operands are memory references
    i = 1
    current_operand = []
    in_brackets = False
    
    while i < len(tokens):
        token = tokens[i]
        
        # Skip size specifiers
        if token.lower() in ['byte', 'word', 'dword', 'qword', 'tbyte', 'xmmword', 'ymmword', 'ptr']:
            i += 1
            continue
        
        if token == '[':
            in_brackets = True
            current_operand.append(token)
        elif token == ']':
            in_brackets = False
            current_operand.append(token)
            # Complete this operand (it's a memory reference)
            operands.append(simplify_operand(' '.join(current_operand)))
            operand_is_memory.append(True)
            current_operand = []
        elif in_brackets:
            current_operand.append(token)
        else:
            # Simple operand
            if current_operand:
                operands.append(simplify_operand(' '.join(current_operand)))
                operand_is_memory.append(False)
                current_operand = []
            operands.append(simplify_operand(token))
            operand_is_memory.append(False)
        
        i += 1
    
    # Add any remaining operand
    if current_operand:
        operands.append(simplify_operand(' '.join(current_operand)))
        operand_is_memory.append(False)
    
    # Special handling for mov and similar data transfer instructions
    # Format: mov dest, src
    # If dest is memory: "ST src [...]"
    # If src is memory: "LD [...] dest"
    # If neither: "CP src dest"
    move_instructions = ['mov', 'movq', 'movd', 'movss', 'movsd', 'movaps', 'movups', 
                        'movdqa', 'movdqu', 'movapd', 'movupd', 'movsx', 'movzx', 'movsxd']
    
    if opcode in move_instructions and len(operands) == 2:
        dest, src = operands[0], operands[1]
        dest_is_mem = operand_is_memory[0] if len(operand_is_memory) > 0 else False
        src_is_mem = operand_is_memory[1] if len(operand_is_memory) > 1 else False
        
        if dest_is_mem and not src_is_mem:
            # Store to memory
            if opcode == 'movsx':
                return f"ST.sx {src} {dest}"
            elif opcode == 'movzx':
                return f"ST.zx {src} {dest}"
            return f"ST {src} {dest}"
        elif src_is_mem and not dest_is_mem:
            # Load from memory
            if opcode == 'movsx':
                return f"LD.sx {src} {dest}"
            elif opcode == 'movzx':
                return f"LD.zx {src} {dest}"
            return f"LD {src} {dest}"
        elif dest_is_mem and src_is_mem:
            # Memory to memory (rare)
            return f"CP.mm {src} {dest}"
        else:
            # Register to register or immediate to register
            if opcode == 'movsx':
                return f"sext {src} {dest}"
            elif opcode == 'movzx':
                return f"zext {src} {dest}"
            return f"CP {src} {dest}"
    
    # Special handling for lea (load effective address) - always loads address, never memory content
    if opcode == 'lea' and len(operands) == 2:
        dest, src = operands[0], operands[1]
        return f"lea {src} {dest}"
    
    # Special handling for xchg - exchange values
    if opcode == 'xchg' and len(operands) == 2:
        op1, op2 = operands[0], operands[1]
        op1_is_mem = operand_is_memory[0] if len(operand_is_memory) > 0 else False
        op2_is_mem = operand_is_memory[1] if len(operand_is_memory) > 1 else False
        
        if op1_is_mem:
            op1 = f"M{op1}"
        if op2_is_mem:
            op2 = f"M{op2}"
        return f"xchg {op1} {op2}"
    
    # Special handling for cmpxchg (compare and exchange)
    if opcode == 'cmpxchg' and len(operands) == 2:
        dest, src = operands[0], operands[1]
        dest_is_mem = operand_is_memory[0] if len(operand_is_memory) > 0 else False
        
        if dest_is_mem:
            return f"cmpxchg M{dest} {src}"
        return f"cmpxchg {dest} {src}"
    
    # Special handling for call with memory operand
    if opcode == 'call' and len(operands) == 1:
        target = operands[0]
        if operand_is_memory[0]:
            return f"call M{target}"
        return f"call {target}"
    
    # Special handling for jmp with memory operand
    if opcode == 'jmp' and len(operands) == 1:
        target = operands[0]
        if operand_is_memory[0]:
            return f"jmp M{target}"
        return f"jmp {target}"
    
    # Special handling for push/pop with memory
    if opcode == 'push' and len(operands) == 1:
        op = operands[0]
        if operand_is_memory[0]:
            return f"push M{op}"
        return f"push {op}"
    
    if opcode == 'pop' and len(operands) == 1:
        op = operands[0]
        if operand_is_memory[0]:
            return f"pop M{op}"
        return f"pop {op}"
    
    # Apply template
    try:
        if len(operands) == 0:
            return semantic_template
        elif len(operands) == 1:
            # For single operand instructions
            if '{}' in semantic_template:
                # Add "M" prefix if it's a memory operand
                op = operands[0]
                if operand_is_memory[0]:
                    op = f"M{op}"
                return semantic_template.format(op)
            return f"{semantic_template} {operands[0]}"
        elif len(operands) == 2:
            # Most instructions have 2 operands (src, dest order for template)
            dest, src = operands[0], operands[1]
            dest_is_mem = operand_is_memory[0] if len(operand_is_memory) > 0 else False
            src_is_mem = operand_is_memory[1] if len(operand_is_memory) > 1 else False
            
            # Add "M" prefix for memory operands
            if src_is_mem:
                src = f"M{src}"
            if dest_is_mem:
                dest = f"M{dest}"
            
            return semantic_template.format(src, dest)
        else:
            # Multiple operands
            formatted_ops = []
            for i, op in enumerate(operands[::-1]):
                if i < len(operand_is_memory) and operand_is_memory[-(i+1)]:
                    formatted_ops.append(f"M{op}")
                else:
                    formatted_ops.append(op)
            return semantic_template.format(*formatted_ops)
    except:
        # Fallback
        return f"{opcode} {' '.join(operands)}"

def process_file(input_file, output_file):
    """
    Process input file and convert instructions to semantic sequences.
    Preserves tab separation between source and target.
    """
    print(f"Processing {input_file}...")
    line_count = 0
    unknown_opcodes = {}  # Track unknown opcodes
    failed_conversions = []  # Track failed conversions with line numbers
    
    # Create log file name
    log_file = output_file.replace('.txt', '_log.txt')
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin:
        with open(output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                line_count += 1
                
                # Split by tab to preserve source/target separation
                instructions = line.strip().split('\t')
                
                # Convert each instruction to semantic
                semantic_parts = []
                for idx, instruction in enumerate(instructions):
                    instruction = instruction.strip()
                    if instruction:
                        semantic = instruction_to_semantic(instruction)
                        semantic_parts.append(semantic)
                        
                        # Check if conversion used fallback (unknown opcode)
                        tokens = instruction.split()
                        if tokens:
                            opcode = tokens[0].lower()
                            # If semantic still starts with opcode and not in our known templates, it's unknown
                            if semantic.startswith(opcode) and opcode not in OPCODE_SEMANTICS:
                                if opcode not in unknown_opcodes:
                                    unknown_opcodes[opcode] = []
                                if len(unknown_opcodes[opcode]) < 5:  # Keep up to 5 examples
                                    unknown_opcodes[opcode].append(instruction)
                    else:
                        semantic_parts.append('')
                
                # Write with tab separator
                fout.write('\t'.join(semantic_parts) + '\n')
                
                if line_count % 100000 == 0:
                    print(f"  Processed {line_count} lines...")
    
    print(f"  Total lines processed: {line_count}")
    print(f"  Output saved to: {output_file}")
    
    # Write log file
    if unknown_opcodes:
        print(f"  Found {len(unknown_opcodes)} unknown opcodes")
        print(f"  Writing log to: {log_file}")
        
        with open(log_file, 'w', encoding='utf-8') as flog:
            flog.write(f"Conversion Log for {input_file}\n")
            flog.write("=" * 80 + "\n\n")
            flog.write(f"Total lines processed: {line_count}\n")
            flog.write(f"Unknown opcodes found: {len(unknown_opcodes)}\n\n")
            flog.write("=" * 80 + "\n\n")
            
            # Sort by frequency
            sorted_opcodes = sorted(unknown_opcodes.items(), key=lambda x: len(x[1]), reverse=True)
            
            for opcode, examples in sorted_opcodes:
                flog.write(f"Opcode: {opcode}\n")
                flog.write(f"Examples:\n")
                for example in examples:
                    flog.write(f"  {example}\n")
                flog.write("\n")
        
        print(f"  Log saved to: {log_file}")
    else:
        print(f"  All instructions converted successfully!")

def main():
    import os
    
    data_dir = '/home/louie/PalmTree/data'
    
    files_to_process = [
        ('cfg_2.txt', 'cfg_2_semantic.txt'),
        ('dfg_2.txt', 'dfg_2_semantic.txt'),
    ]
    
    for input_name, output_name in files_to_process:
        input_path = os.path.join(data_dir, input_name)
        output_path = os.path.join(data_dir, output_name)
        
        if os.path.exists(input_path):
            process_file(input_path, output_path)
        else:
            print(f"Warning: {input_path} not found, skipping...")
    
    print("\nConversion complete!")

if __name__ == '__main__':
    main()
