from binaryninja import *
from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path

# ---------- regex ----------
HEX_RE = re.compile(r'0x[0-9a-fA-F]+')

# ---------- address-like heuristic (matches your parser rule) ----------
def is_addr_like_str(s: str) -> bool:
    return s.startswith("0x") and len(s) >= 6

# ---------- your original parser (kept) ----------
def parse_instruction(ins, symbol_map, string_map):
    ins = re.sub(r'\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    if len(parts) > 1:
        operand = parts[1:]
    for i in range(len(operand)):
        symbols = re.split(r'([0-9A-Za-z]+)', operand[i])
        for j in range(len(symbols)):
            if symbols[j][:2] == '0x' and len(symbols[j]) >= 6:
                try:
                    hv = int(symbols[j], 16)
                    if hv in symbol_map:
                        symbols[j] = "symbol"
                    elif hv in string_map:
                        symbols[j] = "string"
                    else:
                        symbols[j] = "address"
                except ValueError:
                    pass
        operand[i] = ' '.join(symbols)
    opcode = parts[0]
    return ' '.join([opcode] + operand)

# ---------- token helpers ----------
def tokenize_raw(s: str):
    """Simple whitespace tokenizer (for length alignment)."""
    return s.split()

def tokenize_with_spans(s: str):
    """Tokenize while keeping (start,end) spans in original string."""
    toks = []
    i = 0
    n = len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not s[j].isspace():
            j += 1
        toks.append((s[i:j], i, j))
        i = j
    return toks  # [(token, start, end)]

def per_token_echo_hex_or_zero(raw_text: str):
    """
    File #4: For every token, if it overlaps a hex literal AND passes the address heuristic
             (0x.... with len>=6), output that hex; else '0'.
    """
    toks = tokenize_with_spans(raw_text)
    if not toks:
        return []
    out = ["0"] * len(toks)
    for m in HEX_RE.finditer(raw_text):
        lit = m.group(0)
        if not is_addr_like_str(lit):
            continue
        hs, he = m.span()
        for idx, (_tok, ts, te) in enumerate(toks):
            if not (te <= hs or ts >= he):
                out[idx] = lit
                break
    return out

def per_token_sid_labels_all_hex(raw_text: str, instr_to_sid: dict):
    """
    File #5: For every token, if it overlaps a hex literal AND passes the address heuristic,
             and that hex equals an instruction address we assigned a sentence ID to,
             output the SID; else '-1'.
    """
    toks = tokenize_with_spans(raw_text)
    if not toks:
        return []
    out = ["-1"] * len(toks)
    for m in HEX_RE.finditer(raw_text):
        lit = m.group(0)
        if not is_addr_like_str(lit):
            continue
        try:
            addr = int(lit, 16)
        except ValueError:
            continue
        sid = instr_to_sid.get(addr, -1)
        hs, he = m.span()
        for idx, (_tok, ts, te) in enumerate(toks):
            if not (te <= hs or ts >= he):
                out[idx] = str(sid)
                break
    return out

# ---------- CFG sequence sampling (kept) ----------
def random_walk(g, length, node_has_text):
    sequences = []
    for n in g:
        if n != -1 and node_has_text(n):
            seq = [n]
            cur = n
            l = 0
            while l < length:
                nbs = list(g.successors(cur))
                if nbs:
                    cur = random.choice(nbs)
                    if node_has_text(cur):
                        seq.append(cur)
                        l += 1
                    else:
                        break
                else:
                    break
            sequences.append(seq)
        if len(sequences) > 100:
            return sequences[:100]
    return sequences

# ---------- per-binary processing ----------
def process_file(f, window_size):
    print(f"[INFO] Processing: %s" % f)
    bv = load(f)

    binary_name = Path(f).stem
    out_dir = Path("output") / binary_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # outputs
    p1 = out_dir / "raw_pairs.txt"
    p2 = out_dir / "parsed_pairs.txt"
    p3 = out_dir / "instr_addr_per_token.txt"
    p4 = out_dir / "addr_token_per_token.txt"           # every addr-like token → hex, else 0
    p5 = out_dir / "addr_sent_id_per_token.txt"         # every addr-like token → SID, else -1
    pmap = out_dir / "instruction_index_map.tsv"        # instruction addr → sentence id

    print(f"[INFO] Writing:\n 1) {p1}\n 2) {p2}\n 3) {p3}\n 4) {p4}\n 5) {p5}\n map) {pmap}")

    # parser maps
    symbol_map = {sym.address: sym.full_name for sym in bv.get_symbols()}
    string_map = {s.start: s.value for s in bv.get_strings()}

    # build per-function graphs and collect instruction addresses
    function_graphs = {}
    all_instr_addrs = set()

    for func in bv.functions:
        G = nx.DiGraph()
        for block in func:
            curr = block.start
            predecessor = curr
            for inst in block:
                raw = bv.get_disassembly(curr)
                G.add_node(curr, text=raw)   # store raw at instruction address
                all_instr_addrs.add(curr)
                if curr != block.start:
                    G.add_edge(predecessor, curr)
                predecessor = curr
                curr += inst[1]
            for edge in block.outgoing_edges:
                G.add_edge(predecessor, edge.target.start)
        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    # assign per-binary sentence ids to *instruction addresses*
    instr_addrs_sorted = sorted(all_instr_addrs)
    instr_to_sid = {addr: i for i, addr in enumerate(instr_addrs_sorted)}

    # save the mapping (instr addr -> sentence id)
    with open(pmap, "w", encoding="utf-8") as wm:
        wm.write("# instruction_addr\tsentence_id\n")
        for addr in instr_addrs_sorted:
            wm.write(f"0x{addr:x}\t{instr_to_sid[addr]}\n")

    # precompute node meta
    node_meta = {}
    for G in function_graphs.values():
        for addr in G.nodes:
            raw = G.nodes[addr]['text']
            parsed = parse_instruction(raw, symbol_map, string_map)
            raw_tok_count = len(tokenize_raw(raw))
            addr_tok = per_token_echo_hex_or_zero(raw)                    # file #4
            sid_tok  = per_token_sid_labels_all_hex(raw, instr_to_sid)    # file #5
            node_meta[addr] = {
                "raw": raw,
                "parsed": parsed,
                "raw_tok_count": raw_tok_count,
                "addr_tok": addr_tok,
                "sid_tok": sid_tok,
            }

    # emit files
    with open(p1, "w", encoding="utf-8") as w_raw, \
         open(p2, "w", encoding="utf-8") as w_parsed, \
         open(p3, "w", encoding="utf-8") as w_instr, \
         open(p4, "w", encoding="utf-8") as w_addr_tok, \
         open(p5, "w", encoding="utf-8") as w_sid_tok:

        for name, graph in function_graphs.items():
            has_text = lambda a: 'text' in graph.nodes[a]
            sequences = random_walk(graph, 40, has_text)

            for seq in sequences:
                for idx in range(len(seq)):
                    for i in range(1, window_size + 1):
                        pairs = []
                        if idx - i > 0:
                            pairs.append((seq[idx - i], seq[idx]))
                        if idx + i < len(seq):
                            pairs.append((seq[idx], seq[idx + i]))
                        for u, v in pairs:
                            mu, mv = node_meta[u], node_meta[v]

                            # 1) raw pairs
                            w_raw.write(f"{mu['raw']}\t{mv['raw']}\n")

                            # 2) parsed pairs
                            w_parsed.write(f"{mu['parsed']}\t{mv['parsed']}\n")

                            # 3) instruction address per token (repeat node address count)
                            left_addr  = f"0x{u:x}"
                            right_addr = f"0x{v:x}"
                            left_rep   = " ".join([left_addr]  * max(1, mu["raw_tok_count"]))
                            right_rep  = " ".join([right_addr] * max(1, mv["raw_tok_count"]))
                            w_instr.write(f"{left_rep}\t{right_rep}\n")

                            # 4) for every addr-like token, echo the hex; else '0'
                            w_addr_tok.write(
                                f"{' '.join(mu['addr_tok'])}\t{' '.join(mv['addr_tok'])}\n"
                            )

                            # 5) for every addr-like token, output its sentence ID; else '-1'
                            w_sid_tok.write(
                                f"{' '.join(mu['sid_tok'])}\t{' '.join(mv['sid_tok'])}\n"
                            )

    print(f"[DONE] {out_dir}")

def main():
    bin_folder = '/home/louie/PalmTree/src/data_generator/testbin'
    window_size = 1
    file_lst = []
    for parent, subdirs, files in os.walk(bin_folder):
        for f in files:
            file_lst.append(os.path.join(parent, f))
    total = len(file_lst)
    for i, f in enumerate(file_lst, 1):
        print(f"[{i}/{total}]")
        process_file(f, window_size)

if __name__ == "__main__":
    main()
