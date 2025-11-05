
from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
SEG_LEN  = 2   # one output line = 2 instructions
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
            if not tok:
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
# Chunk builder (inline format)
# ---------------------------
def build_chunk_inline(seq, start, k, ctx: dict):
    """
    Build one window with inline addresses.
    Format: opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 operand2
    """
    end = start + k
    if end > len(seq):
        return None

    chunk = seq[start:end]

    addr_positions = ctx.get('addr_positions', {})
    total_bin = ctx.get('total_bin', 1)
    bb_range_map = ctx.get('bb_range_map', {})
    min_addr = ctx.get('min_addr', 0)
    max_addr = ctx.get('max_addr', min_addr)
    sections = ctx.get('sections', [])

    def section_norm(a):
        for sstart, send, _ in sections:
            if sstart <= a < send:
                if send - sstart > 1:
                    return (a - sstart) / float(send - sstart - 1)
                return 0.0
        return 0.0

    def format_positions(entry):
        bin_counter, addr, func_name, bb_start, func_start, func_end = entry
        
        # Binary-level: position based on address range
        if max_addr > min_addr:
            bnorm = (addr - min_addr) / float(max_addr - min_addr)
            bnorm = max(0.0, min(1.0, bnorm))
        else:
            bnorm = 0.0
        
        # Function-level: position within function address range
        if func_end > func_start:
            fnorm = (addr - func_start) / float(func_end - func_start)
            fnorm = max(0.0, min(1.0, fnorm))
        else:
            fnorm = 0.0
        
        # Basic block-level: position within bb address range
        if bb_start in bb_range_map:
            bb_start_addr, bb_end_addr = bb_range_map[bb_start]
            if bb_end_addr > bb_start_addr:
                bbnorm = (addr - bb_start_addr) / float(bb_end_addr - bb_start_addr)
                bbnorm = max(0.0, min(1.0, bbnorm))
            else:
                bbnorm = 0.0
        else:
            bbnorm = 0.0
        
        return f"{bnorm:.3f}:{fnorm:.3f}:{bbnorm:.3f}"

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
                        entry = addr_positions[tgt]
                        pos = format_positions(entry)
                        formatted_ops.append(f"addr_code({mk_hex}:{pos})")
                    else:
                        # data addr: binary-normalized + section_norm, last field 0
                        if max_addr > min_addr:
                            bnorm = (tgt - min_addr) / float(max_addr - min_addr)
                            bnorm = max(0.0, min(1.0, bnorm))
                        else:
                            bnorm = 0.0
                        sn = section_norm(tgt)
                        formatted_ops.append(f"addr_data({mk_hex}:{bnorm:.3f}:{sn:.3f}:0)")
                else:
                    formatted_ops.append(mk_hex)
            else:
                # drop symbol/string
                if tok in ("symbol", "string"):
                    continue
                formatted_ops.append(tok)

        if addr in addr_positions:
            addr_hdr = f"{hex(addr)}:{format_positions(addr_positions[addr])}"
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
    min_addr = min(addr_positions.keys()) if addr_positions else 0
    max_addr = max(addr_positions.keys()) if addr_positions else min_addr

    sections = []
    try:
        for sec in bv.get_sections():
            start = getattr(sec, 'start', None)
            length = getattr(sec, 'length', None) or getattr(sec, 'size', None) or 0
            name = getattr(sec, 'name', '')
            if start is not None:
                sections.append((start, start + int(length), name))
    except Exception:
        sections = []

    ctx = {
        'addr_positions': addr_positions,
        'total_bin': total_bin,
        'bb_range_map': bb_range_map,
        'min_addr': min_addr,
        'max_addr': max_addr,
        'sections': sections,
    }

    # Output path
    out_dir = Path("/home/kun/Document/PalmTree/data/dfg")
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_name = Path(f).name
    dfg_inline_path = out_dir / f"{binary_name}_dfg_{SEG_LEN}_inline.txt"

    print(f"[INFO] Writing to: {dfg_inline_path}")

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


# ---------------------------
# Main
# ---------------------------
def main():
    bin_folder = '/home/kun/Document/PalmTree/src/data_generator/testbin'
    file_lst = []
    for parent, _, files in os.walk(bin_folder):
        for f in files:
            full = os.path.join(parent, f)
            if f.endswith('.txt'):
                continue
            file_lst.append(full)

    for i, f in enumerate(sorted(file_lst)):
        print(i, '/', len(file_lst))
        process_file(f)


if __name__ == "__main__":
    main()

