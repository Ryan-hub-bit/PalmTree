"""
Address-Aware Control Flow Generator
Generates CFG pairs with hierarchical address normalization (binary, function, BB levels)
Window size fixed at 2 (context of ±2 instructions)
"""

from binaryninja import *
import networkx as nx
import random
import os
import re
import pickle

def parse_instruction_addressaware(ins, addr, symbol_map, string_map, binary_min, binary_max, func_min, func_max, bb_min, bb_max, sections):
    """
    Parse instruction and add inline address information.
    Format: opcode(addr:bnorm:fnorm:bbnorm) operand1 operand2 ...
    
    Returns:
        Formatted instruction string with hierarchical address normalization
    """
    
    def get_section_for_addr(tgt_addr):
        """Find which section an address belongs to."""
        for sec_start, sec_end, sec_name in sections:
            if sec_start <= tgt_addr < sec_end:
                return (sec_start, sec_end, sec_name)
        return None
    
    def format_data_address_positions(tgt_addr):
        """
        Format hierarchical positions for data/non-code addresses.
        
        Only apply section normalization for data-like sections (.data, .rodata, .bss*):
          Position 1: section's position in binary = (section_start - binary_min) / (binary_max - binary_min)
          Position 2: address position inside section = (addr - section_start) / (section_end - section_start)
          Position 3: 0.0 (no BB context for data)
        
        For other sections or no section at all:
          return sentinel "2.00000000:0.00000000:0.00000000"
        """
        section_info = get_section_for_addr(tgt_addr)
        
        # No section at all → external/garbage/special
        if section_info is None:
            return "2.00000000:0.00000000:0.00000000"
        
        sec_start, sec_end, sec_name = section_info
        sname = sec_name.lower()
        
        # Only treat true data-like segments as meaningful
        is_data_like = (
            ".data" in sname or
            ".rodata" in sname or
            ".bss" in sname
        )
        
        if not is_data_like:
            # PLT/GOT/import/debug/etc. → special category
            return "2.00000000:0.00000000:0.00000000"
        
        # ---- Real data section: compute normalized positions ----
        if binary_max > binary_min:
            sec_in_binary = (sec_start - binary_min) / float(binary_max - binary_min)
            sec_in_binary = max(0.0, min(1.0, sec_in_binary))
        else:
            sec_in_binary = 0.0
        
        if sec_end > sec_start:
            addr_in_section = (tgt_addr - sec_start) / float(sec_end - sec_start)
            addr_in_section = max(0.0, min(1.0, addr_in_section))
        else:
            addr_in_section = 0.0
        
        bb_pos = 0.0
        return f"{sec_in_binary:.8f}:{addr_in_section:.8f}:{bb_pos:.8f}"
    
    ins = re.sub(r'\s+', ' ', ins.strip())
    parts = ins.split(' ', 1)
    opcode = parts[0] if parts else ''
    operands = parts[1] if len(parts) > 1 else ''
    
    # Calculate normalized positions [0, 1] for CODE addresses
    bnorm = (addr - binary_min) / max(1, binary_max - binary_min) if binary_max > binary_min else 0.0
    fnorm = (addr - func_min) / max(1, func_max - func_min) if func_max > func_min else 0.0
    bbnorm = (addr - bb_min) / max(1, bb_max - bb_min) if bb_max > bb_min else 0.0
    
    # Clamp to [0, 1]
    bnorm = max(0.0, min(1.0, bnorm))
    fnorm = max(0.0, min(1.0, fnorm))
    bbnorm = max(0.0, min(1.0, bbnorm))
    
    # Format opcode with address (CODE address - hierarchical positions)
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
                        # Calculate normalized position for this operand address
                        op_bnorm = (hex_val - binary_min) / max(1, binary_max - binary_min) if binary_max > binary_min else 0.0
                        op_fnorm = (hex_val - func_min) / max(1, func_max - func_min) if func_max > func_min else 0.0
                        op_bbnorm = (hex_val - bb_min) / max(1, bb_max - bb_min) if bb_max > bb_min else 0.0
                        
                        op_bnorm = max(0.0, min(1.0, op_bnorm))
                        op_fnorm = max(0.0, min(1.0, op_fnorm))
                        op_bbnorm = max(0.0, min(1.0, op_bbnorm))
                        
                        if hex_val in symbol_map:
                            # Symbol (function) address - treat as CODE
                            formatted_operands.append(f"symbol({hex(hex_val)}:{op_bnorm:.6f}:{op_fnorm:.6f}:{op_bbnorm:.6f})")
                        elif hex_val in string_map:
                            # String address - treat as DATA
                            pos = format_data_address_positions(hex_val)
                            formatted_operands.append(f"string({hex(hex_val)}:{pos})")
                        else:
                            # Generic address - distinguish DATA from CODE
                            # Check if it's in a code section (use function range as proxy)
                            if func_min <= hex_val < func_max:
                                # Likely CODE address within function
                                formatted_operands.append(f"address({hex(hex_val)}:{op_bnorm:.6f}:{op_fnorm:.6f}:{op_bbnorm:.6f})")
                            else:
                                # Likely DATA address outside function - use section-based positions
                                pos = format_data_address_positions(hex_val)
                                formatted_operands.append(f"address({hex(hex_val)}:{pos})")
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


def random_walk_addressaware(g, length, symbol_map, string_map, binary_min, binary_max, func_min, func_max, bb_ranges, sections):
    """
    Perform random walks in CFG and collect instruction sequences with address info.
    bb_ranges: dict mapping bb_start_addr -> (bb_min, bb_max)
    sections: list of (sec_start, sec_end, sec_name) tuples
    """
    sequence = []
    for n in g:
        if n != -1 and 'text' in g.node[n]:
            s = []
            l = 0
            
            # Find which basic block this instruction belongs to
            bb_min, bb_max = bb_ranges.get(n, (n, n))
            
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
                    if 'text' in g.node[cur]:
                        # Find BB range for current instruction
                        bb_min, bb_max = bb_ranges.get(cur, (cur, cur))
                        
                        formatted_inst = parse_instruction_addressaware(
                            g.node[cur]['text'], cur, symbol_map, string_map,
                            binary_min, binary_max, func_min, func_max, bb_min, bb_max, sections
                        )
                        s.append(formatted_inst)
                        l += 1
                    else:
                        break
                else:
                    break
            sequence.append(s)
        if len(sequence) > 5000:
            print("early stop")
            return sequence[:5000]
    return sequence


def process_file(f, window_size=2):
    """
    Process a binary file and generate address-aware CFG pairs.
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
    
    # Get binary address range
    binary_min = min([seg.start for seg in bv.segments if seg.readable])
    binary_max = max([seg.end for seg in bv.segments if seg.readable])
    
    # Collect section information
    sections = []
    for seg in bv.segments:
        if seg.readable:
            sections.append((seg.start, seg.end, seg.name))
    
    function_graphs = {}
    
    for func in bv.functions:
        G = nx.DiGraph()
        label_dict = {}
        
        # Get function address range
        func_min = func.start
        func_max = func.highest_address
        
        # Build basic block ranges
        bb_ranges = {}
        for block in func:
            bb_start = block.start
            bb_end = block.end
            # Map all instructions in this BB to the BB range
            curr = block.start
            for inst in block:
                bb_ranges[curr] = (bb_start, bb_end)
                curr += inst[1]
        
        # Build CFG
        for block in func:
            curr = block.start
            predecessor = curr
            for inst in block:
                label_dict[curr] = bv.get_disassembly(curr)
                G.add_node(curr, text=bv.get_disassembly(curr))
                if curr != block.start:
                    G.add_edge(predecessor, curr)
                predecessor = curr
                curr += inst[1]
            for edge in block.outgoing_edges:
                G.add_edge(predecessor, edge.target.start)
        
        if len(G.nodes) > 2:
            function_graphs[func.name] = (G, func_min, func_max, bb_ranges)
    
    # Write address-aware CFG pairs
    with open('cfg_addressaware_train.txt', 'a') as w:
        for name, (graph, func_min, func_max, bb_ranges) in function_graphs.items():
            sequence = random_walk_addressaware(
                graph, 40, symbol_map, string_map,
                binary_min, binary_max, func_min, func_max, bb_ranges, sections
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
