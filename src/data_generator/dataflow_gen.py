# -*- coding: utf-8 -*-
from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path

# -------------------- tokenization & helpers (CFG-style spacing, no commas) --------------------
TOK_RE = re.compile(r"0x[0-9A-Fa-f]+|[A-Za-z_.$][\w.$]*|[\[\]\+\-\*\(\),:]|\d+")
HEX_TOKEN = re.compile(r"^0x[0-9A-Fa-f]+$")  # strict hex token check

def tokenize_cfg_style(s: str) -> list[str]:
    """CFG-style tokenizer: split and strip commas out completely."""
    toks = TOK_RE.findall(s)
    # drop literal ',' tokens so we never keep them
    return [t for t in toks if t != ',']

def join_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)

def normalize_cfg_style(s: str) -> str:
    """CFG-style spacing: no commas, just single-space tokens."""
    toks = tokenize_cfg_style(s)
    return " ".join(toks)

def parsed_tokens_keep_addrs(ins: str) -> list[str]:
    """Like parsed but keep hex addresses, no commas."""
    return tokenize_cfg_style(ins)

def parsed_tokens_symbolic(ins: str, symbol_map: dict[int, str], string_starts: set[int]) -> list[str]:
    """Like parsed, but map hex to symbol/string/address, no commas."""
    toks = tokenize_cfg_style(ins)
    out = []
    for t in toks:
        if HEX_TOKEN.match(t) and len(t) >= 6:
            try:
                hv = int(t, 16)
            except ValueError:
                out.append(t)
                continue
            if hv in symbol_map:
                out.append("symbol")
            elif hv in string_starts:
                out.append("string")
            else:
                out.append("address")
        else:
            out.append(t)
    return out

def tokens_hex_or_zero(from_text: str) -> list[str]:
    """For addr_token_per_token.txt: echo hex token or '0'."""
    toks = from_text.split()
    return [t if HEX_TOKEN.match(t) else "0" for t in toks]

# -------------------- DFG random walk (unchanged) --------------------
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

    # outputs
    p1  = out_dir / "raw_pairs.txt"
    p2  = out_dir / "dfg_train.txt"
    p2b = out_dir / "dfg_train_addr.txt"
    p3  = out_dir / "dfg_train_src.txt"
    p4  = out_dir / "dfg_train_tgt.txt"
    p5  = out_dir / "tgt_id.txt"

    print(f"[INFO] Writing:\n 1) {p1}\n 2) {p2}\n 3) {p2b}\n 4) {p3}\n 5) {p4}\n 6) {p5}")

    # symbol maps
    symbol_map    = {sym.address: sym.full_name for sym in bv.get_symbols()}
    string_starts = {s.start for s in bv.get_strings()}

    function_graphs = {}
    node_meta = {}

    # -------- DFG build (unchanged logic) --------
    for func in bv.functions:
        if func.mlil is None:
            continue

        G = nx.DiGraph()
        G.add_node(-1, text="entry_point")

        for block in func.mlil:
            for ins in block:
                addr = ins.address
                if addr is None:
                    continue
                raw_text = bv.get_disassembly(addr)
                G.add_node(addr, text=raw_text)

        # def-use edges
        for block in func.mlil:
            for ins in block:
                cur_addr = ins.address
                if cur_addr is None:
                    continue
                for var in ins.vars_read:
                    for idx in func.mlil.get_var_definitions(var):
                        def_i = func.mlil[idx]
                        def_addr = getattr(def_i, "address", None)
                        if def_addr is not None and def_addr != cur_addr:
                            G.add_edge(def_addr, cur_addr)
                for var in ins.vars_written:
                    for idx in func.mlil.get_var_uses(var):
                        use_i = func.mlil[idx]
                        use_addr = getattr(use_i, "address", None)
                        if use_addr is not None and use_addr != cur_addr:
                            G.add_edge(cur_addr, use_addr)

        # connect entry to roots
        for node in list(G.nodes):
            if node == -1:
                continue
            if G.in_degree(node) == 0:
                G.add_edge(-1, node)

        # fill node_meta (CFG-style, comma-free)
        for node in G.nodes:
            if node == -1:
                continue
            raw_bn = G.nodes[node].get("text", "")

            # CFG-style normalization and parsing
            raw_norm = normalize_cfg_style(raw_bn)
            parsed_sym = join_tokens(parsed_tokens_symbolic(raw_bn, symbol_map, string_starts))
            parsed_keep = join_tokens(parsed_tokens_keep_addrs(raw_bn))
            tok_count_keep = len(parsed_keep.split())
            hex_or_zero = tokens_hex_or_zero(parsed_keep)

            node_meta[node] = {
                "raw_norm": raw_norm,
                "parsed_sym": parsed_sym,
                "parsed_keep": parsed_keep,
                "tok_count": tok_count_keep,
                "hex_or_zero": hex_or_zero,
            }

        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    # collect pairs
    pairs = []
    for name, graph in function_graphs.items():
        has_text = lambda a: a in node_meta and node_meta[a]["raw_norm"] != ""
        sequences = dfg_random_walk(graph, 40, has_text)
        for seq in sequences:
            for i in range(1, window_size + 1):
                for idx in range(0, len(seq) - i):
                    u, v = seq[idx], seq[idx + i]
                    if u in node_meta and v in node_meta:
                        pairs.append((u, v))

    # --- First pass: write p1, p2, p2b, p3, p4 (all comma-free, CFG-style) ---
    with open(p1,  "w", encoding="utf-8") as w_raw, \
         open(p2,  "w", encoding="utf-8") as w_parsed, \
         open(p2b, "w", encoding="utf-8") as w_parsed_keep, \
         open(p3,  "w", encoding="utf-8") as w_instr, \
         open(p4,  "w", encoding="utf-8") as w_addr_tok:

        for line_idx, (u, v) in enumerate(pairs):
            mu, mv = node_meta[u], node_meta[v]

            # 1) raw_pairs.txt (cfg-style, no commas)
            w_raw.write(f"{mu['raw_norm']}\t{mv['raw_norm']}\n")

            # 2) parsed_pairs.txt (cfg-style, no commas)
            w_parsed.write(f"{mu['parsed_sym']}\t{mv['parsed_sym']}\n")

            # 3) parsed_pairs_withaddr.txt (cfg-style, keep hex, no commas)
            w_parsed_keep.write(f"{mu['parsed_keep']}\t{mv['parsed_keep']}\n")

            # 4) instr_addr_per_token.txt (withaddr-style)
            left_addr  = f"0x{u:x}"
            right_addr = f"0x{v:x}"
            left_rep   = " ".join([left_addr]  * max(1, mu["tok_count"]))
            right_rep  = " ".join([right_addr] * max(1, mv["tok_count"]))
            w_instr.write(f"{left_rep}\t{right_rep}\n")

            # 5) addr_token_per_token.txt (withaddr-style)
            w_addr_tok.write(f"{' '.join(mu['hex_or_zero'])}\t{' '.join(mv['hex_or_zero'])}\n")

    # --- Build addr_to_lines from instr_addr_per_token.txt (1-based line numbers) ---
    addr_to_lines: dict[int, list[int]] = {}
    with open(p3, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, start=1):
            if "\t" not in line:
                continue
            left, right = line.strip().split("\t", 1)
            seen = set()
            for tok in (left.split() + right.split()):
                if HEX_TOKEN.match(tok):
                    try:
                        k = int(tok, 16)
                    except ValueError:
                        continue
                    if k in seen:
                        continue
                    seen.add(k)
                    addr_to_lines.setdefault(k, []).append(line_num)

    # --- addr_sent_seqid_per_token.txt from parsed_pairs_withaddr.txt (withaddr tokens) ---
    def line_list_for_tokens(sent_text: str) -> str:
        out = []
        for t in sent_text.split():
            if HEX_TOKEN.match(t):
                try:
                    k = int(t, 16)
                except ValueError:
                    out.append("-1"); continue
                lines = addr_to_lines.get(k)
                out.append("[" + ",".join(map(str, lines)) + "]" if lines else "-2")
            else:
                out.append("-1")
        return " ".join(out)

    with open(p2b, "r", encoding="utf-8") as r_withaddr, \
         open(p5,  "w", encoding="utf-8") as w_seqids:
        for line in r_withaddr:
            if "\t" not in line:
                w_seqids.write("\t\n")
                continue
            l, r = line.strip("\n").split("\t", 1)
            w_seqids.write(f"{line_list_for_tokens(l)}\t{line_list_for_tokens(r)}\n")

    print(f"[DONE][DFG] {out_dir}")

def main():
    random.seed(0)
    bin_folder  = '/home/louie/smallbinary'           # input binaries
    output_root = '/home/louie/PalmTree/data/kun/dfg' # output root
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
