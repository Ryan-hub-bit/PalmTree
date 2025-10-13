from binaryninja import *
from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path

# ---------- regex ----------
HEX_RE = re.compile(r'0x[0-9a-fA-F]+')

# ---------- address-like heuristic ----------
def is_addr_like_str(s: str) -> bool:
    return s.startswith("0x") and len(s) >= 6

# ---------- instruction parser ----------
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
    return s.split()

def tokenize_with_spans(s: str):
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
    return toks

def per_token_echo_hex_or_zero(raw_text: str):
    """File #4: echo the hex for address-like tokens (0x..., len>=6), else '0'"""
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

def per_token_addr_seqid(raw_text: str, addr_to_seqid: dict[int, list[int]]):
    """
    File #5: for every token:
      - not address-like -> '-1'
      - address-like -> list of all line indices where this address appears
        as a sentence address; if none, '-2'
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
        seqids = addr_to_seqid.get(addr)
        if seqids:
            val = "[" + ",".join(str(x) for x in sorted(set(seqids))) + "]"
        else:
            val = "-2"
        hs, he = m.span()
        for idx, (_tok, ts, te) in enumerate(toks):
            if not (te <= hs or ts >= he):
                out[idx] = val
                break
    return out

# ---------- CFG random walk ----------
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
    p4 = out_dir / "addr_token_per_token.txt"
    p5 = out_dir / "addr_sent_seqid_per_token.txt"

    print(f"[INFO] Writing:\n 1) {p1}\n 2) {p2}\n 3) {p3}\n 4) {p4}\n 5) {p5}")

    # parser maps
    symbol_map = {sym.address: sym.full_name for sym in bv.get_symbols()}
    string_map = {s.start: s.value for s in bv.get_strings()}

    # build per-function graphs
    function_graphs = {}
    node_meta = {}
    for func in bv.functions:
        G = nx.DiGraph()
        for block in func:
            curr = block.start
            predecessor = curr
            for inst in block:
                raw = bv.get_disassembly(curr)
                parsed = parse_instruction(raw, symbol_map, string_map)
                raw_tok_count = len(tokenize_raw(raw))
                addr_tok = per_token_echo_hex_or_zero(raw)
                G.add_node(curr, text=raw)
                node_meta[curr] = {
                    "raw": raw,
                    "parsed": parsed,
                    "raw_tok_count": raw_tok_count,
                    "addr_tok": addr_tok,
                }
                if curr != block.start:
                    G.add_edge(predecessor, curr)
                predecessor = curr
                curr += inst[1]
            for edge in block.outgoing_edges:
                G.add_edge(predecessor, edge.target.start)
        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    # -------- collect forward pairs --------
    pairs = []  # list of (u_addr, v_addr)
    for name, graph in function_graphs.items():
        has_text = lambda a: 'text' in graph.nodes[a]
        sequences = random_walk(graph, 40, has_text)
        for seq in sequences:
            for i in range(1, window_size + 1):
                for idx in range(0, len(seq) - i):
                    u, v = seq[idx], seq[idx + i]
                    pairs.append((u, v))

    # -------- FIRST PASS: build addr -> list of all line indices --------
    addr_to_seqid = {}  # address -> list of line indices
    for line_idx, (u, v) in enumerate(pairs):
        addr_to_seqid.setdefault(u, []).append(line_idx)
        addr_to_seqid.setdefault(v, []).append(line_idx)

    # -------- SECOND PASS: write outputs --------
    with open(p1, "w", encoding="utf-8") as w_raw, \
         open(p2, "w", encoding="utf-8") as w_parsed, \
         open(p3, "w", encoding="utf-8") as w_instr, \
         open(p4, "w", encoding="utf-8") as w_addr_tok, \
         open(p5, "w", encoding="utf-8") as w_addr_seqid_tok:

        for line_idx, (u, v) in enumerate(pairs):
            mu, mv = node_meta[u], node_meta[v]

            # 1) raw pairs
            w_raw.write(f"{mu['raw']}\t{mv['raw']}\n")

            # 2) parsed pairs
            w_parsed.write(f"{mu['parsed']}\t{mv['parsed']}\n")

            # 3) instr_addr_per_token.txt (repeat each sentence's own instr address)
            left_addr  = f"0x{u:x}"
            right_addr = f"0x{v:x}"
            left_rep   = " ".join([left_addr]  * max(1, mu["raw_tok_count"]))
            right_rep  = " ".join([right_addr] * max(1, mv["raw_tok_count"]))
            w_instr.write(f"{left_rep}\t{right_rep}\n")

            # 4) addr_token_per_token.txt
            w_addr_tok.write(
                f"{' '.join(mu['addr_tok'])}\t{' '.join(mv['addr_tok'])}\n"
            )

            # 5) addr_sent_seqid_per_token.txt (list all line indices where addr appears)
            left_seqid_tokens  = per_token_addr_seqid(mu['raw'], addr_to_seqid)
            right_seqid_tokens = per_token_addr_seqid(mv['raw'], addr_to_seqid)
            w_addr_seqid_tok.write(
                f"{' '.join(left_seqid_tokens)}\t{' '.join(right_seqid_tokens)}\n"
            )

    print(f"[DONE] {out_dir}")

def main():
    random.seed(0)
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
