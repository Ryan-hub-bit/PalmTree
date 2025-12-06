#!/usr/bin/env python3
"""
DFG (Data Flow Graph) generation with hierarchical positions for IDA Pro.
Based on dfg_hierarchical_idfg.py but using IDA Pro APIs.
"""

import sys
# Add conda environment path for networkx
sys.path.insert(0, '/home/kun/anaconda3/envs/palmtree/lib/python3.11/site-packages')

import networkx as nx
import random
import os
import re
from pathlib import Path

import idaapi
import idautils
import idc
from ida_funcs import get_func
from ida_hexrays import *

# ---------------------------
# Config
# ---------------------------
SEG_LEN = int(os.environ.get('SEG_LEN', 2))
WALK_LEN = 30  # max steps per random walk

HEX_RE = re.compile(r"0x[0-9a-fA-F]+")

# ---------------------------
# Instruction parser with mask
# ---------------------------
def normalize_and_mask(ins_raw: str, symbol_map: dict, string_map: dict):
    """
    Parse one disassembly line, keeping hex addresses in both normalized text and mask.
    Returns (norm_text, mask_line) where both have the same token count.
    """
    ins = re.sub(r"\s+", ", ", ins_raw, 1)
    parts = ins.split(", ")
    opcode = parts[0]
    operands = parts[1:] if len(parts) > 1 else []

    out_tokens = [opcode]
    mask_tokens = ["0"]

    for op in operands:
        pieces = re.split(r"(0x[0-9A-Fa-f]+|[A-Za-z0-9_]+|\[|\]|,|:|\(|\))", op)
        for tok in pieces:
            if not tok or tok.isspace():
                continue
            if tok.startswith("0x") and bool(HEX_RE.fullmatch(tok)):
                out_tokens.append(tok)
                mask_tokens.append(tok)
            else:
                out_tokens.append(tok)
                mask_tokens.append("0")

    return " ".join(out_tokens), " ".join(mask_tokens)


def clean_ida_disasm(ea):
    """
    Get clean disassembly preserving IDA keywords (offset, short, etc.)
    but replacing symbol names with hex addresses.
    Mark immediate values with 'imm' token.
    Mark displacement operands (o_displ) with 'disp_0xXX' prefix to prevent address() wrapping.
    """
    # Get mnemonic
    mnem = idc.print_insn_mnem(ea)
    if not mnem:
        return None
    
    # Get operands - up to 6 operands max
    operands = []
    for i in range(6):
        op = idc.print_operand(ea, i)
        if not op:
            break
        
        # Clean up IDA's duplicate offsets in var format: [rsp+60h+var_60] -> [rsp+var_60]
        # This handles the common case where IDA shows both hex offset and var_ symbol
        if 'var_' in op:
            op = re.sub(r'\+?\s*0x[0-9A-Fa-f]+\s*\+\s*(?=var_)', '+', op)
            op = re.sub(r'\+?\s*[0-9A-Fa-f]+h\s*\+\s*(?=var_)', '+', op, flags=re.IGNORECASE)
            # Clean up potential artifacts: ++ -> +, [+ -> [, +] -> ]
            op = re.sub(r'\+\s*\+', '+', op)
            op = re.sub(r'\[\s*\+', '[', op)
            op = re.sub(r'\+\s*\]', ']', op)
        
        # Get operand type and value
        op_type = idc.get_operand_type(ea, i)
        op_value = idc.get_operand_value(ea, i)
        
        # Check if it's an immediate value
        if op_type == idc.o_imm:
            # It's an immediate - mark it
            op = "imm"
        # Check if operand contains keywords or symbols that need address replacement
        # Handle "offset symbol_name" -> "offset 0xADDR"
        elif 'offset' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"offset {hex(op_value)}"
        # Handle "short symbol_name" -> "short 0xADDR"
        elif 'short' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"short {hex(op_value)}"
        # Handle "large symbol_name" -> "large 0xADDR"
        elif 'large' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"large {hex(op_value)}"
        # Handle segment prefix "cs:symbol" -> just "0xADDR" (remove cs:, ds:, etc.)
        elif any(seg in op for seg in ['cs:', 'ds:', 'es:', 'ss:', 'fs:', 'gs:']) and op_value != idaapi.BADADDR and op_value != 0:
            op = hex(op_value)
        # Handle displacement operands without var (regular offsets)
        elif op_type in [idc.o_phrase, idc.o_displ]:
            # If it's not a var (already cleaned above), mark it as displacement
            if 'var_' not in op and op_value != idaapi.BADADDR and op_value != 0:
                op = f"disp_{hex(op_value)}"
        # For operands that reference code/data addresses (but NOT displacements)
        elif op_type in [idc.o_near, idc.o_mem, idc.o_far]:
            # var_ already cleaned above, just check if it needs address replacement
            if 'var_' not in op and op_value != idaapi.BADADDR and op_value != 0:
                # Check if it's a symbol name (not already a hex address)
                if not op.startswith('0x') and not op.startswith('['):
                    op = hex(op_value)
        operands.append(op)
    
    # Build clean disassembly
    if operands:
        return f"{mnem} {', '.join(operands)}"
    else:
        return mnem


# ---------------------------
# Random walk over DFG-style graph
# ---------------------------
def random_walk(g, length, symbol_map, string_map):
    """
    Produce sequences of nodes by walking successors randomly.
    Each element is (addr, norm_text, mask_line).
    """
    seqs = []
    for n in g:
        if n == -1:
            continue
        node_text = g.nodes[n].get('text')
        node_mask = g.nodes[n].get('mask')
        if node_text is None or node_mask is None:
            continue

        s = []
        steps = 0
        s.append((n, node_text, node_mask))
        cur = n

        while steps < length:
            nbs = list(g.successors(cur))
            if not nbs:
                break
            cur = random.choice(nbs)
            node_text = g.nodes[cur].get('text')
            node_mask = g.nodes[cur].get('mask')
            if node_text is None or node_mask is None:
                break
            s.append((cur, node_text, node_mask))
            steps += 1

        if s:
            seqs.append(s)

    return seqs


# ---------------------------
# Chunk builder (inline format with hierarchical positions)
# ---------------------------
def build_chunk_inline(seq, start_idx: int, k: int, ctx: dict):
    """
    Build a chunk with HIERARCHICAL position encoding (address-based).
    
    For CODE addresses (instructions):
    - Position 1: Function's position in binary = (func_start - min_addr) / (max_addr - min_addr)
    - Position 2: Basic block's position in function = (bb_start - func_start) / (func_end - func_start)
    - Position 3: Instruction's position in basic block = (inst_addr - bb_start) / (bb_end - bb_start)
    
    For DATA addresses (not in text section):
    - Position 1: Section's position in binary
    - Position 2: Address position inside section
    - Position 3: BB position = 0.0
    """
    end_idx = start_idx + k
    if start_idx < 0 or end_idx > len(seq):
        return None
    chunk = seq[start_idx:end_idx]

    addr_positions = ctx.get('addr_positions', {})
    bb_range_map = ctx.get('bb_range_map', {})
    min_addr = ctx.get('min_addr', 0)
    max_addr = ctx.get('max_addr', min_addr)
    sections = ctx.get('sections', [])

    def get_section_for_addr(addr):
        """Find which section an address belongs to."""
        for sec_start, sec_end, sec_name in sections:
            if sec_start <= addr < sec_end:
                return (sec_start, sec_end, sec_name)
        return None
    
    def format_data_address_positions(addr):
        """
        Format hierarchical positions for data/non-code addresses.
        
        Only apply section normalization for data-like sections (.data, .rodata, .bss*):
          Position 1: section's position in binary = (section_start - min_addr) / (max_addr - min_addr)
          Position 2: address position inside section = (addr - section_start) / (section_end - section_start)
          Position 3: 0.0 (no BB context for data)
        
        For other sections or no section at all:
          return sentinel "2.00000000:0.00000000:0.00000000"
        """
        section_info = get_section_for_addr(addr)
        
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
        if max_addr > min_addr:
            sec_in_binary = (sec_start - min_addr) / float(max_addr - min_addr)
            sec_in_binary = max(0.0, min(1.0, sec_in_binary))
        else:
            sec_in_binary = 0.0
        
        if sec_end > sec_start:
            addr_in_section = (addr - sec_start) / float(sec_end - sec_start)
            addr_in_section = max(0.0, min(1.0, addr_in_section))
        else:
            addr_in_section = 0.0
        
        bb_pos = 0.0
        return f"{sec_in_binary:.8f}:{addr_in_section:.8f}:{bb_pos:.8f}"
    
    def format_hierarchical_positions(entry):
        """Format hierarchical positions for code addresses."""
        bin_counter, addr, func_name, bb_start, func_start, func_end = entry
        
        if max_addr > min_addr:
            func_binary_norm = (func_start - min_addr) / float(max_addr - min_addr)
            func_binary_norm = max(0.0, min(1.0, func_binary_norm))
        else:
            func_binary_norm = 0.0
        
        # Handle case where bb_start is None (instruction not matched to a BB)
        if bb_start is not None and func_end > func_start:
            bb_function_norm = (bb_start - func_start) / float(func_end - func_start)
            bb_function_norm = max(0.0, min(1.0, bb_function_norm))
        else:
            bb_function_norm = 0.0
        
        if bb_start is not None and bb_start in bb_range_map:
            bb_start_addr, bb_end_addr = bb_range_map[bb_start]
            if bb_end_addr > bb_start_addr:
                inst_bb_norm = (addr - bb_start_addr) / float(bb_end_addr - bb_start_addr)
                inst_bb_norm = max(0.0, min(1.0, inst_bb_norm))
            else:
                inst_bb_norm = 0.0
        else:
            inst_bb_norm = 0.0
        
        return f"{func_binary_norm:.8f}:{bb_function_norm:.8f}:{inst_bb_norm:.8f}"

    out_instrs = []
    for addr, norm_line, mask_line in chunk:
        tokens = norm_line.strip().split()
        masks = mask_line.strip().split()
        if not tokens:
            continue
        opcode = tokens[0]
        operands = tokens[1:]
        operand_masks = masks[1:]

        formatted_ops = []
        for i, tok in enumerate(operands):
            mk = operand_masks[i] if i < len(operand_masks) else "0"
            mk_hex = None
            
            # Check if this is a displacement token (marked with "disp_" prefix)
            if isinstance(tok, str) and tok.startswith('disp_0x'):
                # This is a displacement operand - keep the hex value without address() wrapper
                # hex_part = tok[5:]  # Remove "disp_" prefix
                hex_part = "disp"  # Remove "disp_" prefix
                formatted_ops.append(hex_part)
                continue
            
            # Check mask first (from normalize_and_mask)
            if isinstance(mk, str) and mk.startswith('0x'):
                mk_hex = mk
            # Check token for 0x prefix
            elif isinstance(tok, str) and tok.startswith('0x') and bool(HEX_RE.fullmatch(tok)):
                mk_hex = tok
            # Check token for Intel hex suffix (e.g., 1234ABCDh)
            elif isinstance(tok, str) and re.match(r'^[0-9A-Fa-f]+h$', tok, re.IGNORECASE):
                # Convert Intel hex to 0x format
                mk_hex = '0x' + tok[:-1]

            if mk_hex is not None:
                try:
                    tgt = int(mk_hex, 16)
                except Exception:
                    tgt = None

                # Filter out immediate values:
                # - Values below binary base (min_addr) are immediates
                # - Use max(min_addr, 0x1000) to handle cases where min_addr=0 (PIE, embedded)
                # Note: o_imm operands are already filtered in clean_ida_disasm()
                #       This is just an extra safety layer for edge cases
                #       Stack offsets are handled by o_displ/o_phrase -> disp_ in clean_ida_disasm
                
                # Check if this is a real address (within binary range)
                # Use 0x1000 as minimum threshold to filter out small constants even when min_addr=0
                addr_threshold = max(min_addr, 0x1000)
                if tgt is not None and tgt >= addr_threshold:
                    if tgt in addr_positions:
                        # Code address: use hierarchical positions (func, bb, inst)
                        entry = addr_positions[tgt]
                        pos = format_hierarchical_positions(entry)
                        formatted_ops.append(f"address({mk_hex}:{pos})")
                    else:
                        # Data address (not in text section): use section-based hierarchical positions
                        # Position 1: Section's position in binary
                        # Position 2: Address position inside section
                        # Position 3: BB position = 0
                        pos = format_data_address_positions(tgt)
                        formatted_ops.append(f"address({mk_hex}:{pos})")
                else:
                    # It's an immediate value
                    formatted_ops.append("imm")
            else:
                if tok.startswith("var_"):
                    var_offset = tok[4:]  # Get the part after 'var_'
                    formatted_ops.append(f"var(0x{var_offset})")
                # elif tok.startswith("arg_"):
                #     formatted_ops.append("arg")
                else:
                    formatted_ops.append(tok)

        if addr in addr_positions:
            addr_hdr = f"{hex(addr)}:{format_hierarchical_positions(addr_positions[addr])}"
        else:
            addr_hdr = hex(addr)

        ops_join = ' '.join(formatted_ops)
        if ops_join.count('[') > ops_join.count(']'):
            ops_join = ops_join + ' ]'

        if ops_join.strip():
            out_instrs.append(f"{opcode}({addr_hdr}) {ops_join}")
        else:
            out_instrs.append(f"{opcode}({addr_hdr})")

    return "\t".join(out_instrs)


def get_basic_blocks_ida(func_ea):
    """
    Get basic blocks for a function in IDA Pro.
    Returns list of (bb_start, bb_end) tuples.
    """
    func = get_func(func_ea)
    if not func:
        return []
    
    bbs = []
    fc = idaapi.FlowChart(func)
    for block in fc:
        bb_start = block.start_ea
        bb_end = block.end_ea
        bbs.append((bb_start, bb_end))
    
    return bbs


def process_file_ida(fpath, out_dir):
    """Process a binary file with IDA Pro to generate DFG with hierarchical positions."""
    print(f"[INFO] Processing: {fpath}")
    
    # Wait for auto-analysis to complete
    idaapi.auto_wait()
    
    # Output paths
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_name = Path(fpath).name
    dfg_inline_path = out_dir / f"{binary_name}_dfg_{SEG_LEN}_inline.txt"

    print(f"[INFO] Writing to: {dfg_inline_path}")

    # Get binary address range and sections
    min_addr = idaapi.inf_get_min_ea()
    max_addr = idaapi.inf_get_max_ea()
    
    sections = []
    for n in range(idaapi.get_segm_qty()):
        seg = idaapi.getnseg(n)
        if seg:
            sections.append((seg.start_ea, seg.end_ea, idaapi.get_segm_name(seg)))
    
    print(f"[INFO] Binary range: {hex(min_addr)} - {hex(max_addr)}")
    print(f"[INFO] Sections: {len(sections)}")

    # Check if decompiler is available
    if not init_hexrays_plugin():
        print("[WARNING] Hex-Rays decompiler not available. DFG generation may be limited.")
        hexrays_available = False
    else:
        hexrays_available = True

    # Build DFG-like graph from microcode var defs/uses
    function_graphs = {}
    addr_positions = {}
    func_range_map = {}
    bb_range_map = {}
    bin_counter = 0
    
    symbol_map = {}
    string_map = {}

    for func_ea in idautils.Functions():
        func = get_func(func_ea)
        if not func:
            continue

        func_name = idc.get_func_name(func_ea)
        func_start = func.start_ea
        func_end = func.end_ea
        
        G = nx.DiGraph()
        G.add_node(-1, text='entry_point', mask='0')
        
        # Build bb_range_map from actual assembly basic blocks
        for bb_start, bb_end in get_basic_blocks_ida(func_ea):
            # Find the last instruction address in the BB
            last_inst_addr = bb_start
            curr = bb_start
            while curr < bb_end:
                last_inst_addr = curr
                curr = idc.next_head(curr, bb_end)
            bb_range_map[bb_start] = (bb_start, last_inst_addr)

        # Try to get microcode for data-flow analysis
        if hexrays_available:
            try:
                mbr = mba_ranges_t()
                mbr.ranges.push_back(range_t(func_start, func_end))
                hf = hexrays_failure_t()
                mba = gen_microcode(mbr, hf, None, DECOMP_NO_WAIT, MMAT_GLBOPT)
                
                if mba:
                    # Process microcode blocks
                    for mblock_idx in range(mba.qty):
                        mblock = mba.get_mblock(mblock_idx)
                        
                        for ins_idx in range(mblock.head, mblock.tail):
                            mins = mba.get_minsn(ins_idx)
                            if not mins:
                                continue
                            
                            addr = mins.ea
                            if addr == idaapi.BADADDR:
                                continue
                            
                            # Get disassembly
                            disasm_raw = clean_ida_disasm(addr)
                            if not disasm_raw:
                                continue
                            
                            # Find which assembly basic block this belongs to
                            # Microcode addresses might not align perfectly with assembly BBs,
                            # so find the BB that contains this address or the closest preceding BB
                            bb_start_for_addr = None
                            bbs = list(get_basic_blocks_ida(func_ea))
                            for bb_start, bb_end in bbs:
                                if bb_start <= addr <= bb_end:
                                    bb_start_for_addr = bb_start
                                    break
                            
                            # If no exact match, find the closest preceding BB within the function
                            if bb_start_for_addr is None and bbs:
                                closest_bb = None
                                min_distance = float('inf')
                                for bb_start, bb_end in bbs:
                                    if bb_start <= addr:
                                        distance = addr - bb_start
                                        if distance < min_distance:
                                            min_distance = distance
                                            closest_bb = bb_start
                                if closest_bb is not None:
                                    bb_start_for_addr = closest_bb
                            
                            norm_text, mask_line = normalize_and_mask(disasm_raw, symbol_map, string_map)
                            G.add_node(addr, text=norm_text, mask=mask_line)
                            addr_positions[addr] = (bin_counter, addr, func_name, bb_start_for_addr, func_start, func_end)
                            bin_counter += 1
                            
                            # Build data-flow edges (simplified - IDA microcode doesn't expose defs/uses as easily as MLIL)
                            # This is a basic approximation
                            if ins_idx > mblock.head:
                                prev_mins = mba.get_minsn(ins_idx - 1)
                                if prev_mins and prev_mins.ea != idaapi.BADADDR:
                                    G.add_edge(prev_mins.ea, addr)
                
            except Exception as e:
                print(f"[WARNING] Failed to generate microcode for {func_name}: {e}")
                hexrays_available = False  # Disable for remaining functions
        
        # Fallback: if no microcode, use sequential edges
        if not hexrays_available or G.number_of_nodes() <= 1:
            # Use sequential instruction flow
            for bb_start, bb_end in get_basic_blocks_ida(func_ea):
                prev_addr = None
                curr = bb_start
                
                while curr <= bb_end:
                    disasm_raw = clean_ida_disasm(curr)
                    if disasm_raw:
                        norm_text, mask_line = normalize_and_mask(disasm_raw, symbol_map, string_map)
                        G.add_node(curr, text=norm_text, mask=mask_line)
                        addr_positions[curr] = (bin_counter, curr, func_name, bb_start, func_start, func_end)
                        bin_counter += 1
                        
                        if prev_addr is not None:
                            G.add_edge(prev_addr, curr)
                        prev_addr = curr
                    
                    curr = idc.next_head(curr, idaapi.BADADDR)
                    if curr >= bb_end:
                        break
        
        func_range_map[func_name] = (func_start, func_end)
        
        # Connect orphan sources to entry node
        for node in list(G.nodes):
            if node == -1:
                continue
            if G.in_degree(node) == 0:
                G.add_edge(-1, node)
        
        function_graphs[func_name] = G

    print(f"[INFO] Processed {len(function_graphs)} functions")
    print(f"[INFO] Total instructions: {bin_counter}")

    # Random walk and generate sequences
    all_seqs = []
    for func_name, G in function_graphs.items():
        seqs = random_walk(G, WALK_LEN, symbol_map, string_map)
        all_seqs.extend(seqs)

    print(f"[INFO] Generated {len(all_seqs)} sequences from random walks")

    # Build context for hierarchical encoding
    ctx = {
        'addr_positions': addr_positions,
        'bb_range_map': bb_range_map,
        'min_addr': min_addr,
        'max_addr': max_addr,
        'sections': sections,
    }

    # Write inline format
    with open(dfg_inline_path, 'w') as f_inline:
        for seq in all_seqs:
            for start in range(len(seq)):
                line = build_chunk_inline(seq, start, SEG_LEN, ctx)
                if line:
                    f_inline.write(line + "\n")

    print(f"[INFO] Wrote {dfg_inline_path}")
    print("[INFO] DFG generation complete!")


def main():
    """Main entry point for IDA script."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Get input file path from IDA
    input_file = idaapi.get_input_file_path()
    
    # Get output directory from environment
    output_dir = os.environ.get('OUTPUT_DIR', './testres')
    
    print("=" * 70)
    print("DFG Generation with Hierarchical Positions (IDA Pro)")
    print("=" * 70)
    print(f"Input file: {input_file}")
    print(f"Output dir: {output_dir}")
    print(f"SEG_LEN: {SEG_LEN}")
    print("=" * 70)
    
    process_file_ida(input_file, output_dir)
    
    # Exit IDA
    idaapi.qexit(0)


if __name__ == "__main__":
    main()
