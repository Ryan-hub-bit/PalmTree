
from binaryninja import load
import networkx as nx
import random
import os
import re
import sys
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
SEG_LEN  = 2   # Default value, can be overridden by command-line argument
WALK_LEN = 40  # max steps per random walk

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
            if not tok or tok.isspace():  # Skip empty and whitespace-only tokens
                continue
            if tok.startswith("0x") and bool(HEX_RE.fullmatch(tok)):
                out_tokens.append(tok)
                mask_tokens.append(tok)
            else:
                out_tokens.append(tok)
                mask_tokens.append("0")

    return " ".join(out_tokens), " ".join(mask_tokens)


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
def build_chunk_inline(seq, start, k, ctx: dict):
    """
    Build one window with inline hierarchical addresses.
    
    For CODE addresses (instructions):
    - Position 1: Function's position in binary = (func_start - min_addr) / (max_addr - min_addr)
    - Position 2: Basic block's position in function = (bb_start - func_start) / (func_end - func_start)
    - Position 3: Instruction's position in basic block = (inst_addr - bb_start) / (bb_end - bb_start)
    
    For DATA addresses (not in text section):
    - Position 1: Section's position in binary = (section_start - min_addr) / (max_addr - min_addr)
    - Position 2: Address position inside section = (addr - section_start) / (section_end - section_start)
    - Position 3: BB position = 0.0 (no BB context for data)
    
    Format: opcode(0xADDR:pos1:pos2:pos3) operand address(0xADDR:pos1:pos2:pos3)
    """
    end = start + k
    if end > len(seq):
        return None

    chunk = seq[start:end]

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
        
        Position 1: Section's position in binary = (section_start - min_addr) / (max_addr - min_addr)
        Position 2: Address position inside section = (addr - section_start) / (section_end - section_start)
        Position 3: BB position = 0.0 (no BB context for data)
        """
        section_info = get_section_for_addr(addr)
        
        if section_info is not None:
            sec_start, sec_end, sec_name = section_info
            
            # Position 1: Section's position in binary
            if max_addr > min_addr:
                sec_in_binary = (sec_start - min_addr) / float(max_addr - min_addr)
                sec_in_binary = max(0.0, min(1.0, sec_in_binary))
            else:
                sec_in_binary = 0.0
            
            # Position 2: Address position inside section
            if sec_end > sec_start:
                addr_in_section = (addr - sec_start) / float(sec_end - sec_start)
                addr_in_section = max(0.0, min(1.0, addr_in_section))
            else:
                addr_in_section = 0.0
            
            # Position 3: BB position (always 0 for data addresses)
            bb_pos = 0.0
            
            return f"{sec_in_binary:.8f}:{addr_in_section:.8f}:{bb_pos:.8f}"
        else:
            # Address not in any section, fallback to binary-level position only
            if max_addr > min_addr:
                bin_norm = (addr - min_addr) / float(max_addr - min_addr)
                bin_norm = max(0.0, min(1.0, bin_norm))
            else:
                bin_norm = 0.0
            return f"{bin_norm:.8f}:0.00000000:0.00000000"

    def format_hierarchical_positions(entry):
        """
        Returns: func_in_binary:bb_in_function:inst_in_bb
        
        Position 1: (func_start - min_addr) / (max_addr - min_addr)
        Position 2: (bb_start - func_start) / (func_end - func_start)
        Position 3: (inst_addr - bb_start) / (bb_end - bb_start)
        """
        bin_counter, addr, func_name, bb_start, func_start, func_end = entry
        
        # Position 1: Function's position in binary (address-based)
        if max_addr > min_addr:
            func_binary_norm = (func_start - min_addr) / float(max_addr - min_addr)
            func_binary_norm = max(0.0, min(1.0, func_binary_norm))
        else:
            func_binary_norm = 0.0
        
        # Position 2: Basic block's position in function (address-based)
        # Handle case where bb_start is None (instruction not matched to a BB)
        if bb_start is not None and func_end > func_start:
            bb_function_norm = (bb_start - func_start) / float(func_end - func_start)
            bb_function_norm = max(0.0, min(1.0, bb_function_norm))
        else:
            bb_function_norm = 0.0
        
        # Position 3: Instruction's position in basic block (address-based)
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
            if isinstance(mk, str) and mk.startswith('0x'):
                mk_hex = mk
            elif isinstance(tok, str) and tok.startswith('0x') and bool(HEX_RE.fullmatch(tok)):
                mk_hex = tok

            if mk_hex is not None:
                try:
                    tgt = int(mk_hex, 16)
                except Exception:
                    tgt = None

                # Filter out immediate values
                is_immediate = False
                if tgt is not None:
                    if tgt < min_addr:  # below binary base
                        is_immediate = True
                    elif tgt > 0xffffffffffff0000:  # large bit patterns/masks
                        is_immediate = True

                if tgt is not None and tgt >= min_addr and not is_immediate:
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
                    formatted_ops.append(mk_hex)
            else:
                # drop symbol/string
                if tok in ("symbol", "string"):
                    continue
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


# ---------------------------
# Per-file processing
# ---------------------------
def process_file(f):
    symbol_map = {}
    string_map = {}
    print(f"[INFO] Processing: {f}")

    bv = load(f)
    if bv is None:
        print(f"[WARN] Could not load {f}; skipping.")
        return

    # collect symbols and strings
    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    for string in bv.get_strings():
        string_map[string.start] = string.value

    # Get output directory from global or use default
    import __main__
    out_dir_str = getattr(__main__, 'OUTPUT_DIR', '/home/kun/Document/PalmTree/data/dfg')
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_name = Path(f).name
    dfg_inline_path = out_dir / f"{binary_name}_dfg_{SEG_LEN}_inline.txt"

    print(f"[INFO] Writing to: {dfg_inline_path}")

    # Build a DFG-like graph from MLIL var defs/uses
    function_graphs = {}
    addr_positions = {}
    func_range_map = {}  # func_name -> (func_start, func_end)
    bb_range_map = {}    # bb_start -> (bb_start, bb_end) - from actual assembly basic blocks
    bin_counter = 0

    for func in bv.functions:
        if not hasattr(func, "mlil") or func.mlil is None:
            continue

        G = nx.DiGraph()
        G.add_node(-1, text='entry_point', mask='0')
        
        # Get function address range
        func_start = func.start
        func_end = func.start + func.total_bytes

        # Build bb_range_map from ACTUAL assembly basic blocks
        for block in func.basic_blocks:
            bb_start_addr = block.start
            # Calculate the end address (last instruction start address in the block)
            bb_end_addr = bb_start_addr
            curr = block.start
            for inst in block:
                bb_end_addr = curr
                curr += inst[1]  # inst[1] is the instruction length
            bb_range_map[bb_start_addr] = (bb_start_addr, bb_end_addr)

        try:
            blocks = list(func.mlil)
        except Exception:
            continue

        # Build graph from MLIL for data-flow edges
        for block in blocks:
            try:
                insns = list(block)
            except Exception:
                continue

            for ins in insns:
                try:
                    addr = ins.address
                except Exception:
                    continue

                try:
                    dis = bv.get_disassembly(addr)
                except Exception:
                    continue

                # Find which assembly basic block this instruction belongs to
                bb_start_for_addr = None
                for bb in func.basic_blocks:
                    if bb.start <= addr < bb.start + bb.length:
                        bb_start_for_addr = bb.start
                        break

                norm_text, mask_line = normalize_and_mask(dis, symbol_map, string_map)
                G.add_node(addr, text=norm_text, mask=mask_line)
                # Store: (bin_counter, addr, func_name, bb_start_addr, func_start, func_end)
                addr_positions[addr] = (bin_counter, addr, func.name, bb_start_for_addr, func_start, func_end)
                bin_counter += 1

                depd = []
                # defs -> uses (data-flow)
                try:
                    for var in ins.vars_read:
                        for i in func.mlil.get_var_definitions(var):
                            if func.mlil[i].address != addr:
                                depd.append((func.mlil[i].address, addr))
                except Exception:
                    pass

                try:
                    for var in ins.vars_written:
                        for i in func.mlil.get_var_uses(var):
                            if func.mlil[i].address != addr:
                                depd.append((addr, func.mlil[i].address))
                except Exception:
                    pass

                if depd:
                    G.add_edges_from(depd)

        func_range_map[func.name] = (func_start, func_end)

        # connect orphan sources to entry node
        for node in list(G.nodes):
            if node == -1:
                continue
            if G.in_degree(node) == 0:
                G.add_edge(-1, node)

        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    total_bin = bin_counter
    
    # Collect sections first
    sections = []
    try:
        for sec in bv.sections.values():
            start = sec.start
            end = sec.end
            name = sec.name
            sections.append((start, end, name))
    except Exception as e:
        print(f"[WARNING] Could not read sections: {e}")
        sections = []
    
    # Calculate min/max to cover ALL addresses (code + data sections)
    all_addrs = list(addr_positions.keys()) if addr_positions else []
    for sec_start, sec_end, _ in sections:
        all_addrs.append(sec_start)
        all_addrs.append(sec_end)
    
    if all_addrs:
        min_addr = min(all_addrs)
        max_addr = max(all_addrs)
    else:
        min_addr = 0
        max_addr = 0

    ctx = {
        'addr_positions': addr_positions,
        'total_bin': total_bin,
        'bb_range_map': bb_range_map,
        'min_addr': min_addr,
        'max_addr': max_addr,
        'sections': sections,
    }

    written = 0
    with open(dfg_inline_path, 'w', encoding='utf-8') as w:
        for _, graph in function_graphs.items():
            seqs = random_walk(graph, WALK_LEN, symbol_map, string_map)
            for s in seqs:
                if len(s) < SEG_LEN:
                    continue
                for start in range(0, len(s) - SEG_LEN + 1):
                    built = build_chunk_inline(s, start, SEG_LEN, ctx)
                    if not built:
                        continue
                    w.write(built + '\n')
                    written += 1

    print(f"[DONE] {dfg_inline_path} (wrote {written} sequences)")
    print(f"[INFO] Position encoding:")
    print(f"  CODE: func_in_binary:bb_in_function:inst_in_bb")
    print(f"  DATA: section_in_binary:addr_in_section:0.0")


# ---------------------------
# Main
# ---------------------------
def main():
    global SEG_LEN
    
    # Parse command-line arguments
    bin_folder = '/home/kun/onebinary/'
    out_dir = '/home/kun/Document/PalmTree/src/data_generator/testres'
    
    if len(sys.argv) > 1:
        SEG_LEN = int(sys.argv[1])
    if len(sys.argv) > 2:
        bin_folder = sys.argv[2]
    if len(sys.argv) > 3:
        out_dir = sys.argv[3]
    
    print(f"[CONFIG] SEG_LEN={SEG_LEN}, BIN_FOLDER={bin_folder}, OUTPUT={out_dir}")
    print(f"[INFO] Using HIERARCHICAL position encoding (address-based):")
    print(f"  CODE addresses:")
    print(f"    - Position 1: (func_start - min_addr) / (max_addr - min_addr)")
    print(f"    - Position 2: (bb_start - func_start) / (func_end - func_start)")
    print(f"    - Position 3: (inst_addr - bb_start) / (bb_end - bb_start)")
    print(f"  DATA addresses:")
    print(f"    - Position 1: (section_start - min_addr) / (max_addr - min_addr)")
    print(f"    - Position 2: (addr - section_start) / (section_end - section_start)")
    print(f"    - Position 3: 0.0 (no BB context)")
    
    # Update the output directory globally (will be used in process_file)
    import __main__
    __main__.OUTPUT_DIR = out_dir
    
    file_lst = []
    for parent, _, files in os.walk(bin_folder):
        for f in files:
            full = os.path.join(parent, f)
            if f.endswith('.txt'):
                continue
            file_lst.append(full)

    for i, f in enumerate(sorted(file_lst)):
        print(f"\n{'='*70}")
        print(f"Processing {i+1}/{len(file_lst)}")
        print(f"{'='*70}")
        process_file(f)


if __name__ == "__main__":
    main()
