"""
Address-Aware Dataflow Generator  
Generates DFG pairs with hierarchical address normalization (binary, function, BB levels)
Window size fixed at 2 (context of ±2 instructions)
"""

from binaryninja import *
import networkx as nx
import random
import os
import re
from collections import Counter


def get_section_for_addr(addr, sections):
    """
    Get the section that contains the given address.
    Returns section object or None.
    """
    for section in sections:
        if section.start <= addr < section.end:
            return section
    return None


def format_data_address_positions(addr, sections, binary_min, binary_max):
    """
    Format address positions for DATA addresses.
    Returns sentinel "2.00000000:0.00000000:0.00000000" for non-data sections,
    real normalized positions only for .data, .rodata, .bss sections.
    """
    section = get_section_for_addr(addr, sections)
    
    # Check if address is in a data section
    if section and section.name in ['.data', '.rodata', '.bss']:
        # Real normalized position within binary
        bnorm = (addr - binary_min) / max(1, binary_max - binary_min) if binary_max > binary_min else 0.0
        bnorm = max(0.0, min(1.0, bnorm))
        return f"{bnorm:.8f}:0.00000000:0.00000000"
    else:
        # Sentinel for non-data sections (PLT, GOT, imports, etc.)
        return "2.00000000:0.00000000:0.00000000"


def parse_instruction_addressaware(ins, addr, symbol_map, string_map, binary_min, binary_max, func_min, func_max, bb_min, bb_max, sections):
    """
    Parse instruction and add inline address information.
    Format: opcode(addr:bnorm:fnorm:bbnorm) operand1 operand2 ...
    
    For CODE addresses (within function): use hierarchical func/bb/inst positions
    For DATA addresses (data sections or external): use format_data_address_positions
    
    Returns:
        Formatted instruction string with hierarchical address normalization
    """
    ins = re.sub(r'\s+', ' ', ins.strip())
    parts = ins.split(' ', 1)
    opcode = parts[0] if parts else ''
    operands = parts[1] if len(parts) > 1 else ''
    
    # Calculate normalized positions [0, 1] for CODE instruction address
    bnorm = (addr - binary_min) / max(1, binary_max - binary_min) if binary_max > binary_min else 0.0
    fnorm = (addr - func_min) / max(1, func_max - func_min) if func_max > func_min else 0.0
    bbnorm = (addr - bb_min) / max(1, bb_max - bb_min) if bb_max > bb_min else 0.0
    
    # Clamp to [0, 1]
    bnorm = max(0.0, min(1.0, bnorm))
    fnorm = max(0.0, min(1.0, fnorm))
    bbnorm = max(0.0, min(1.0, bbnorm))
    
    # Format opcode with address (CODE format - hierarchical)
    formatted_opcode = f"{opcode}({hex(addr)}:{bnorm:.6f}:{fnorm:.6f}:{bbnorm:.6f})"
    
    # Process operands
    if operands:
        formatted_operands = []
        for operand in operands.split():
            # Check if operand contains hex address
            if '0x' in operand and len(operand) >= 6:
                try:
                    # Extract hex value
                    hex_match = re.search(r'0x[0-9a-fA-F]+', operand)
                    if hex_match:
                        hex_val = int(hex_match.group(), 16)
                        
                        # Distinguish between CODE and DATA addresses
                        if hex_val in symbol_map:
                            # Symbol address - treat as CODE (hierarchical)
                            op_bnorm = (hex_val - binary_min) / max(1, binary_max - binary_min) if binary_max > binary_min else 0.0
                            op_fnorm = (hex_val - func_min) / max(1, func_max - func_min) if func_max > func_min else 0.0
                            op_bbnorm = (hex_val - bb_min) / max(1, bb_max - bb_min) if bb_max > bb_min else 0.0
                            
                            op_bnorm = max(0.0, min(1.0, op_bnorm))
                            op_fnorm = max(0.0, min(1.0, op_fnorm))
                            op_bbnorm = max(0.0, min(1.0, op_bbnorm))
                            
                            formatted_operands.append(f"symbol({hex(hex_val)}:{op_bnorm:.6f}:{op_fnorm:.6f}:{op_bbnorm:.6f})")
                        elif hex_val in string_map:
                            # String address - treat as DATA (section-based)
                            pos_str = format_data_address_positions(hex_val, sections, binary_min, binary_max)
                            formatted_operands.append(f"string({hex(hex_val)}:{pos_str})")
                        elif func_min <= hex_val <= func_max:
                            # Address within function range - treat as CODE (hierarchical)
                            op_bnorm = (hex_val - binary_min) / max(1, binary_max - binary_min) if binary_max > binary_min else 0.0
                            op_fnorm = (hex_val - func_min) / max(1, func_max - func_min) if func_max > func_min else 0.0
                            op_bbnorm = (hex_val - bb_min) / max(1, bb_max - bb_min) if bb_max > bb_min else 0.0
                            
                            op_bnorm = max(0.0, min(1.0, op_bnorm))
                            op_fnorm = max(0.0, min(1.0, op_fnorm))
                            op_bbnorm = max(0.0, min(1.0, op_bbnorm))
                            
                            formatted_operands.append(f"address({hex(hex_val)}:{op_bnorm:.6f}:{op_fnorm:.6f}:{op_bbnorm:.6f})")
                        else:
                            # Address outside function - treat as DATA (section-based)
                            pos_str = format_data_address_positions(hex_val, sections, binary_min, binary_max)
                            formatted_operands.append(f"address({hex(hex_val)}:{pos_str})")
                    else:
                        formatted_operands.append(operand)
                except ValueError:
                    formatted_operands.append(operand)
            # Check for stack variables in various formats
            # BinaryNinja formats: [rbp-0x10], [rsp+0x8], {var_10}, etc.
            elif re.search(r'\[r[bs]p[\s]*[+-][\s]*0x[0-9a-fA-F]+\]', operand):
                # Extract the offset from [rbp-0x10] or [rsp+0x8] format
                offset_match = re.search(r'[+-][\s]*(0x[0-9a-fA-F]+)', operand)
                if offset_match:
                    offset_hex = offset_match.group(1)
                    # Replace the entire operand with var(offset)
                    formatted_operands.append(f"var({offset_hex})")
                else:
                    formatted_operands.append(operand)
            # Also handle direct offset notation like {var_10} or var_10 
            elif re.search(r'(?:var_|{var_)([0-9a-fA-F]+)', operand):
                # Extract offset from var_10 or {var_10} format
                var_match = re.search(r'(?:var_|{var_)([0-9a-fA-F]+)', operand)
                if var_match:
                    offset_hex = var_match.group(1)
                    # Ensure it has 0x prefix
                    if not offset_hex.startswith('0x'):
                        offset_hex = '0x' + offset_hex
                    formatted_operands.append(f"var({offset_hex})")
                else:
                    formatted_operands.append(operand)
            else:
                formatted_operands.append(operand)
        
        return f"{formatted_opcode} {' '.join(formatted_operands)}"
    else:
        return formatted_opcode


def random_walk_addressaware(g, length, symbol_map, string_map, binary_min, binary_max, func_min, func_max, inst_to_bb, sections):
    """
    Perform random walks in DFG and collect instruction sequences with address info.
    inst_to_bb: dict mapping instruction_addr -> (bb_min, bb_max)
    sections: list of section objects for data/code distinction
    """
    sequence = []
    for n in g:
        if n != -1 and g.node[n]['text'] is not None:
            s = []
            l = 0
            
            # Handle entry point
            if n == -1:
                s.append('entry_point')
            else:
                # Find which basic block this instruction belongs to
                bb_min, bb_max = inst_to_bb.get(n, (n, n))
                
                formatted_inst = parse_instruction_addressaware(
                    g.node[n]['text'], n, symbol_map, string_map,
                    binary_min, binary_max, func_min, func_max, bb_min, bb_max, sections
                )
                s.append(formatted_inst)
            
            cur = n
            
            while l < length:
                nbs = list(g.successors(cur))
                if len(nbs):
                    cur = random.choice(nbs)
                    if cur == -1:
                        break
                    
                    # Find BB range for current instruction
                    bb_min, bb_max = inst_to_bb.get(cur, (cur, cur))
                    
                    formatted_inst = parse_instruction_addressaware(
                        g.node[cur]['text'], cur, symbol_map, string_map,
                        binary_min, binary_max, func_min, func_max, bb_min, bb_max, sections
                    )
                    s.append(formatted_inst)
                    l += 1
                else:
                    break
            sequence.append(s)
    return sequence


def process_file(f, window_size=2):
    """
    Process a binary file and generate address-aware DFG pairs.
    window_size: context window (default=2 for ±2 instructions)
    """
    symbol_map = {}
    string_map = {}
    print(f"Processing: {f}")
    bv = BinaryViewType.get_view_of_file(f)
    
    # Collect symbols and strings
    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    for string in bv.get_strings():
        string_map[string.start] = string.value
    
    # Collect sections for data/code distinction
    sections = list(bv.sections.values())
    
    # Get binary address range
    binary_min = min([seg.start for seg in bv.segments if seg.readable])
    binary_max = max([seg.end for seg in bv.segments if seg.readable])
    
    function_graphs = {}
    
    for func in bv.functions:
        G = nx.DiGraph()
        G.add_node(-1, text='entry_point')
        label_dict = {}
        label_dict[-1] = 'entry_point'
        
        # Get function address range
        func_min = func.start
        func_max = func.highest_address
        
        # Build instruction to basic block mapping
        inst_to_bb = {}
        for block in func:
            bb_start = block.start
            bb_end = block.end
            curr = block.start
            for inst in block:
                inst_to_bb[curr] = (bb_start, bb_end)
                curr += inst[1]
        
        # Build DFG based on MLIL
        for mlil_inst in func.mlil.instructions:
            addr = mlil_inst.address
            for var in mlil_inst.vars_written:
                for use_inst in func.mlil.get_var_uses(var):
                    use_addr = use_inst.address
                    
                    # Add nodes with disassembly
                    if addr not in label_dict:
                        disasm = bv.get_disassembly(addr)
                        if disasm:
                            G.add_node(addr, text=disasm)
                            label_dict[addr] = disasm
                    
                    if use_addr not in label_dict:
                        disasm = bv.get_disassembly(use_addr)
                        if disasm:
                            G.add_node(use_addr, text=disasm)
                            label_dict[use_addr] = disasm
                    
                    # Add edge from def to use
                    if addr in G and use_addr in G:
                        G.add_edge(addr, use_addr)
        
        # Add edges from entry point to function start
        if func.start in G:
            G.add_edge(-1, func.start)
        
        if len(G.nodes) > 2:
            function_graphs[func.name] = (G, func_min, func_max, inst_to_bb)
    
    # Write address-aware DFG pairs
    with open('dfg_addressaware_train.txt', 'a') as w:
        for name, (graph, func_min, func_max, inst_to_bb) in function_graphs.items():
            sequence = random_walk_addressaware(
                graph, 40, symbol_map, string_map,
                binary_min, binary_max, func_min, func_max, inst_to_bb, sections
            )
            
            for s in sequence:
                if len(s) >= 4:
                    # Create pairs with window_size = 2
                    for idx in range(len(s)):
                        for i in range(1, window_size + 1):
                            if idx - i >= 0:
                                w.write(s[idx-i] + '\t' + s[idx] + '\n')
                            if idx + i < len(s):
                                w.write(s[idx] + '\t' + s[idx+i] + '\n')


def main():
    bin_folder = '/path/to/binaries'
    file_lst = []
    window_size = 2  # Fixed window size
    
    for parent, subdirs, files in os.walk(bin_folder):
        if files:
            for f in files:
                file_lst.append(os.path.join(parent, f))
    
    i = 0
    for f in file_lst:
        print(f"{i}/{len(file_lst)}")
        try:
            process_file(f, window_size)
        except Exception as e:
            print(f"Error processing {f}: {e}")
        i += 1


if __name__ == "__main__":
    main()
