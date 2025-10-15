# -*- coding: utf-8 -*-
from binaryninja import *
from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path
from collections import defaultdict

# -------------------- regex & helpers --------------------
HEX_RE = re.compile(r'0x[0-9a-fA-F]+')

def is_addr_like_str(s: str) -> bool:
    """Treat literal as address-like only if it has enough hex digits."""
    return s.startswith("0x") and len(s) >= 2 + 5  # >= 5 hex digits after 0x (tune if you want)

def tokenize_raw(s: str):
    return s.split()

def tokenize_with_spans(s: str):
    toks = []
    i, n = 0, len(s)
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
    """
    File #4: echo the hex for address-like tokens (0x..., enough digits), else '0'
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
            if not (te <= hs or ts >= he):  # overlaps token span
                out[idx] = lit
                break
    return out

def per_token_addr_seqid(raw_text: str, addr_to_seqid: dict[int, set[int]]):
    """
    File #5: for every token:
      - not address-like -> '-1'
      - address-like -> [sorted unique line indices] if present, else '-2'
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
        val = "[" + ",".join(str(x) for x in sorted(seqids)) + "]" if seqids else "-2"
        hs, he = m.span()
        for idx, (_tok, ts, te) in enumerate(toks):
            if not (te <= hs or ts >= he):
                out[idx] = val
                break
    return out

# -------------------- instruction parser --------------------
def parse_instruction(ins: str, symbol_map: dict[int, str], string_starts: set[int]) -> str:
    """
    Replace address-like immediates with 'symbol' / 'string' / 'address'.
    Keep opcode and spacing similar to your original scheme.
    """
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
                except ValueError:
                    continue
                if hv in symbol_map:
                    symbols[j] = "symbol"
                elif hv in string_starts:
                    symbols[j] = "string"
                else:
                    symbols[j] = "address"
        operand[i] = ' '.join(symbols)
    opcode = parts[0]
    return ' '.join([opcode] + operand)

# -------------------- DFG random walk (over MLIL def-use graph) --------------------
def dfg_random_walk(g: nx.DiGraph, length: int, node_has_text):
    sequences = []
    for n in g:
        if n != -1 and node_has_text(n):
            seq = [n]
            cur = n
            l = 0
            while l < length:
                nbs = list(g.successors(cur))
                if not nbs:
                    break
                cur = random.choice(nbs)
                if node_has_text(cur):
                    seq.append(cur)
                    l += 1
                else:
                    break
            sequences.append(seq)
        # Safety cutoff to prevent explosions on pathological graphs
        if len(sequences) > 100:
            return sequences[:100]
    return sequences

# -------------------- per-binary DFG processing --------------------
def process_file(f, window_size=1, output_root="/home/louie/PalmTree/data/kun/dfg"):
    print(f"[INFO][DFG] Processing: {f}")
    bv = load(f)

    binary_name = Path(f).stem
    out_dir = Path(output_root) / binary_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # outputs (mirroring CFG pipeline)
    p1 = out_dir / "raw_pairs.txt"
    p2 = out_dir / "parsed_pairs.txt"
    p3 = out_dir / "instr_addr_per_token.txt"
    p4 = out_dir / "addr_token_per_token.txt"
    p5 = out_dir / "addr_sent_seqid_per_token.txt"
    # p_edges = out_dir / "dfg_edges.txt"     # ground-truth DFG edges
    # p_pairs = out_dir / "addr_pairs.txt"    # sampled pairs (addresses)

    # print(f"[INFO] Writing:\n 1) {p1}\n 2) {p2}\n 3) {p3}\n 4) {p4}\n 5) {p5}\n 6) {p_edges}\n 7) {p_pairs}")

    # parser maps
    symbol_map = {sym.address: sym.full_name for sym in bv.get_symbols()}
    string_starts = {s.start for s in bv.get_strings()}

    # ----- build per-function DFGs over MLIL def–use edges -----
    function_graphs = {}
    node_meta = {}  # addr -> {raw, parsed, raw_tok_count, addr_tok}

    for func in bv.functions:
        # Some functions might not have MLIL; skip safely
        if func.mlil is None:
            continue

        G = nx.DiGraph()
        # Optional entry node (not used in outputs)
        G.add_node(-1, text="entry_point")

        # Create nodes for each MLIL instruction and collect def-use edges
        for block in func.mlil:
            for ins in block:
                addr = ins.address
                # Skip weird MLIL items that lack an address
                if addr is None:
                    continue

                raw_text = bv.get_disassembly(addr)
                G.add_node(addr, text=raw_text)

        # Build DFG edges: defs -> uses; and defs of read vars -> current use
        # We must loop again to populate edges because we need all nodes present
        for block in func.mlil:
            for ins in block:
                cur_addr = ins.address
                if cur_addr is None:
                    continue

                # Edges from var definitions (that produce values read here) -> current
                for var in ins.vars_read:
                    for idx in func.mlil.get_var_definitions(var):
                        def_i = func.mlil[idx]
                        def_addr = getattr(def_i, "address", None)
                        if def_addr is not None and def_addr != cur_addr:
                            G.add_edge(def_addr, cur_addr)

                # Edges from current (when it defines/writes var) -> future uses of that var
                for var in ins.vars_written:
                    for idx in func.mlil.get_var_uses(var):
                        use_i = func.mlil[idx]
                        use_addr = getattr(use_i, "address", None)
                        if use_addr is not None and use_addr != cur_addr:
                            G.add_edge(cur_addr, use_addr)

        # Add an entry edge to any node with no predecessors (optional, keeps graph weakly connected)
        for node in list(G.nodes):
            if node == -1:
                continue
            if G.in_degree(node) == 0:
                G.add_edge(-1, node)

        # Fill node_meta for later file writing
        for node in G.nodes:
            if node == -1:
                continue
            raw = G.nodes[node].get("text", "")
            parsed = parse_instruction(raw, symbol_map, string_starts)
            raw_tok_count = len(tokenize_raw(raw))
            addr_tok = per_token_echo_hex_or_zero(raw)
            node_meta[node] = {
                "raw": raw,
                "parsed": parsed,
                "raw_tok_count": raw_tok_count,
                "addr_tok": addr_tok,
            }

        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    # -------- collect forward pairs from DFG random walks --------
    pairs = []  # list of (u_addr, v_addr)
    with open(p_edges, "w", encoding="utf-8") as w_edges:
        for name, graph in function_graphs.items():
            # Write out ground-truth DFG edges (address -> address)
            for u, v in graph.edges():
                if u == -1 or v == -1:
                    continue
                w_edges.write(f"{hex(u)}\t{hex(v)}\n")

            # Sample sequences by random walks over DFG
            has_text = lambda a: a in node_meta and node_meta[a]["raw"] != ""
            sequences = dfg_random_walk(graph, 40, has_text)
            for seq in sequences:
                # Build skip-gram style pairs within window_size (like your CFG)
                for i in range(1, window_size + 1):
                    for idx in range(0, len(seq) - i):
                        u, v = seq[idx], seq[idx + i]
                        if u in node_meta and v in node_meta:
                            pairs.append((u, v))

    # -------- FIRST PASS: build addr -> set of line indices where it appears --------
    addr_to_seqid: dict[int, set[int]] = defaultdict(set)
    for line_idx, (u, v) in enumerate(pairs):
        addr_to_seqid[u].add(line_idx)
        addr_to_seqid[v].add(line_idx)

    # -------- SECOND PASS: write outputs --------
    with open(p1, "w", encoding="utf-8") as w_raw, \
         open(p2, "w", encoding="utf-8") as w_parsed, \
         open(p3, "w", encoding="utf-8") as w_instr, \
         open(p4, "w", encoding="utf-8") as w_addr_tok, \
         open(p5, "w", encoding="utf-8") as w_addr_seqid_tok, \
         open(p_pairs, "w", encoding="utf-8") as w_pairs:

        for line_idx, (u, v) in enumerate(pairs):
            mu, mv = node_meta[u], node_meta[v]

            # 0) Address pairs (explicit)
            w_pairs.write(f"{hex(u)}\t{hex(v)}\n")

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

            # 5) addr_sent_seqid_per_token.txt
            left_seqid_tokens  = per_token_addr_seqid(mu['raw'], addr_to_seqid)
            right_seqid_tokens = per_token_addr_seqid(mv['raw'], addr_to_seqid)
            w_addr_seqid_tok.write(
                f"{' '.join(left_seqid_tokens)}\t{' '.join(right_seqid_tokens)}\n"
            )

    print(f"[DONE][DFG] {out_dir}")

def main():
    random.seed(0)
    # Change these paths for your environment
    # bin_folder = '/home/louie/PalmTree/src/data_generator/testbin'
    bin_folder = '/home/louie/smallbinary'
    output_root = '/home/louie/PalmTree/data/kun/dfg'
    window_size = 1

    file_lst = []
    for parent, _subdirs, files in os.walk(bin_folder):
        for f in files:
            file_lst.append(os.path.join(parent, f))

    total = len(file_lst)
    for i, f in enumerate(file_lst, 1):
        print(f"[{i}/{total}]")
        try:
            process_file(f, window_size=window_size, output_root=output_root)
        except Exception as e:
            print(f"[WARN] Failed on {f}: {e}")

if __name__ == "__main__":
    main()



