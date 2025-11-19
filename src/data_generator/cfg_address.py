from binaryninja import load
import networkx as nx
import random
import os
import re
import sys
from pathlib import Path

SEG_LEN = 2  # Default value, can be overridden by command-line argument

HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def normalize_and_mask(ins_raw: str, symbol_map: dict, string_map: dict):
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


def random_walk(g: nx.DiGraph, length: int):
    sequences = []
    for n in g:
        if n != -1 and 'text' in g.nodes[n] and 'mask' in g.nodes[n]:
            s = []
            steps = 0
            s.append((n, g.nodes[n]['text'], g.nodes[n]['mask']))
            cur = n
            while steps < length:
                succ = list(g.successors(cur))
                if not succ:
                    break
                cur = random.choice(succ)
                if 'text' in g.nodes[cur] and 'mask' in g.nodes[cur]:
                    s.append((cur, g.nodes[cur]['text'], g.nodes[cur]['mask']))
                    steps += 1
                else:
                    break
            if s:
                sequences.append(s)
        if len(sequences) >= 5000:
            break
    return sequences[:5000]


def build_chunk_inline(seq, start_idx: int, k: int, ctx: dict):
    end_idx = start_idx + k
    if start_idx < 0 or end_idx > len(seq):
        return None
    chunk = seq[start_idx:end_idx]

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
        
        return f"{bnorm:.8f}:{fnorm:.6f}:{bbnorm:.4f}"

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

                # Filter out immediate values:
                # - Values below binary base are likely immediate constants
                # - Very large values that are likely bit masks (e.g., 0xfffffffffffffff0)
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
                        formatted_ops.append(f"address({mk_hex}:{pos})")
                    else:
                        # data addr: binary-normalized + section_norm, last field 0
                        if max_addr > min_addr:
                            bnorm = (tgt - min_addr) / float(max_addr - min_addr)
                            bnorm = max(0.0, min(1.0, bnorm))
                        else:
                            bnorm = 0.0
                        sn = section_norm(tgt)
                        formatted_ops.append(f"address({mk_hex}:{bnorm:.8f}:{sn:.6f}:0)")
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


def process_file(fpath: str):
    print(f"[INFO] Processing: {fpath}")
    bv = load(fpath)
    if bv is None:
        print(f"[WARN] Could not load {fpath}; skipping.")
        return

    # Get output directory from global or use default
    import __main__
    out_dir_str = getattr(__main__, 'OUTPUT_DIR', '/home/kun/Document/PalmTree/data/cfg')
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_name = Path(fpath).name
    out_inline = out_dir / f"{binary_name}_cfg_{SEG_LEN}_inline.txt"

    symbol_map = {sym.address: sym.full_name for sym in bv.get_symbols()}
    string_map = {s.start: s.value for s in bv.get_strings()}

    function_graphs = {}
    addr_positions = {}
    func_range_map = {}  # func_name -> (func_start, func_end)
    bb_range_map = {}    # bb_start -> (bb_start, bb_end)
    bin_counter = 0

    for func in bv.functions:
        G = nx.DiGraph()
        
        # Get function /address range
        func_start = func.start
        func_end = func.start + func.total_bytes
        
        # Build bb_range_map from assembly basic blocks
        for block in func.basic_blocks:
            bb_start_addr = block.start
            bb_end_addr = bb_start_addr
            curr = block.start
            for inst in block:
                bb_end_addr = curr
                curr += inst[1]
            bb_range_map[bb_start_addr] = (bb_start_addr, bb_end_addr)
        
        for block in func:
            curr = block.start
            predecessor = curr
            for inst in block:
                disasm_raw = bv.get_disassembly(curr)
                norm_text, mask_line = normalize_and_mask(disasm_raw, symbol_map, string_map)
                G.add_node(curr, text=norm_text, mask=mask_line)
                # Store: (bin_counter, addr, func_name, bb_start, func_start, func_end)
                addr_positions[curr] = (bin_counter, curr, func.name, block.start, func_start, func_end)
                bin_counter += 1
                if curr != block.start:
                    G.add_edge(predecessor, curr)
                predecessor = curr
                curr += inst[1]
            for edge in block.outgoing_edges:
                G.add_edge(predecessor, edge.target.start)
        
        func_range_map[func.name] = (func_start, func_end)
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
    with open(out_inline, 'w', encoding='utf-8') as w:
        for _, g in function_graphs.items():
            walks = random_walk(g, length=40)
            for s in walks:
                if len(s) < SEG_LEN:
                    continue
                for start in range(0, len(s) - SEG_LEN + 1):
                    packed = build_chunk_inline(s, start, SEG_LEN, ctx)
                    if not packed:
                        continue
                    w.write(packed + "\n")
                    written += 1

    print(f"[DONE] {out_inline} (wrote {written} sequences)")


def main():
    global SEG_LEN
    
    # Parse command-line arguments
    # bin_folder = "/home/kun/smallbinary"
    bin_folder = "/home/kun/Document/PalmTree/src/data_generator/testbin"
    out_dir = "/home/kun/Document/PalmTree/src/data_generator/testres"
    
    if len(sys.argv) > 1:
        SEG_LEN = int(sys.argv[1])
    if len(sys.argv) > 2:
        bin_folder = sys.argv[2]
    if len(sys.argv) > 3:
        out_dir = sys.argv[3]
    
    print(f"[CONFIG] SEG_LEN={SEG_LEN}, BIN_FOLDER={bin_folder}, OUTPUT={out_dir}")
    
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
        print(i, '/', len(file_lst))
        process_file(f)


if __name__ == '__main__':
    main()
