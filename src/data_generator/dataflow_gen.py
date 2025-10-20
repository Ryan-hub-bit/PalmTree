# -*- coding: utf-8 -*-
from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path
from collections import defaultdict

# -------------------- regex & helpers --------------------
HEX_RE = re.compile(r'0x[0-9a-fA-F]+')

# Split punctuation and add spaces
TOKENIZE_RE = re.compile(
    r"0x[0-9A-Fa-f]+|[A-Za-z_.$][\w.$]*|[\[\]\+\-\*\(\),:]|\d+"
)

def normalize_spacing(s: str) -> str:
    """Turn raw disassembly into spaced tokens like 'lea rdi [ rel 0x406210 ]'."""
    toks = TOKENIZE_RE.findall(s)
    return " ".join(toks)

def tokenize_raw(s: str):
    """Split strictly by whitespace; each token gets one SRC and one TGT."""
    return s.split()

def per_space_token_targets_zero(raw_text: str):
    """File #4 (TGT): emit '0' once per space-split token."""
    n = len(tokenize_raw(raw_text))
    return ["0"] * max(1, n)

def per_space_token_default(raw_text: str, default="-1"):
    """File #5 (SeqID map): align to space-split tokens with a constant placeholder."""
    n = len(tokenize_raw(raw_text))
    return [default] * max(1, n)

# -------------------- instruction parser (for parsed_pairs.txt) --------------------
def parse_instruction(ins: str, symbol_map: dict[int, str], string_starts: set[int]) -> str:
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

# -------------------- DFG random walk --------------------
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
    p1 = out_dir / "raw_pairs.txt"
    p2 = out_dir / "parsed_pairs.txt"
    p3 = out_dir / "instr_addr_per_token.txt"
    p4 = out_dir / "addr_token_per_token.txt"
    p5 = out_dir / "addr_sent_seqid_per_token.txt"

    print(f"[INFO] Writing:\n 1) {p1}\n 2) {p2}\n 3) {p3}\n 4) {p4}\n 5) {p5}")

    # symbol maps
    symbol_map = {sym.address: sym.full_name for sym in bv.get_symbols()}
    string_starts = {s.start for s in bv.get_strings()}

    function_graphs = {}
    node_meta = {}

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

        # fill node_meta
        for node in G.nodes:
            if node == -1:
                continue
            raw = G.nodes[node].get("text", "")
            parsed = parse_instruction(raw, symbol_map, string_starts)
            normalized_raw = normalize_spacing(raw)
            raw_tok_count = len(tokenize_raw(normalized_raw))
            addr_tgt_zero = per_space_token_targets_zero(normalized_raw)
            node_meta[node] = {
                "raw": normalized_raw,
                "parsed": parsed,
                "raw_tok_count": raw_tok_count,
                "addr_tgt_zero": addr_tgt_zero,
            }

        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    # collect pairs
    pairs = []
    for name, graph in function_graphs.items():
        has_text = lambda a: a in node_meta and node_meta[a]["raw"] != ""
        sequences = dfg_random_walk(graph, 40, has_text)
        for seq in sequences:
            for i in range(1, window_size + 1):
                for idx in range(0, len(seq) - i):
                    u, v = seq[idx], seq[idx + i]
                    if u in node_meta and v in node_meta:
                        pairs.append((u, v))

    # write outputs
    with open(p1, "w", encoding="utf-8") as w_raw, \
         open(p2, "w", encoding="utf-8") as w_parsed, \
         open(p3, "w", encoding="utf-8") as w_instr, \
         open(p4, "w", encoding="utf-8") as w_addr_tok, \
         open(p5, "w", encoding="utf-8") as w_addr_seqid_tok:

        for line_idx, (u, v) in enumerate(pairs):
            mu, mv = node_meta[u], node_meta[v]

            # 1) raw_pairs.txt → normalized spacing for both sides
            w_raw.write(f"{mu['raw']}\t{mv['raw']}\n")

            # 2) parsed_pairs.txt
            w_parsed.write(f"{mu['parsed']}\t{mv['parsed']}\n")

            # 3) instr_addr_per_token.txt
            left_addr  = f"0x{u:x}"
            right_addr = f"0x{v:x}"
            left_rep   = " ".join([left_addr]  * max(1, mu["raw_tok_count"]))
            right_rep  = " ".join([right_addr] * max(1, mv["raw_tok_count"]))
            w_instr.write(f"{left_rep}\t{right_rep}\n")

            # 4) addr_token_per_token.txt
            w_addr_tok.write(
                f"{' '.join(mu['addr_tgt_zero'])}\t{' '.join(mv['addr_tgt_zero'])}\n"
            )

            # 5) addr_sent_seqid_per_token.txt
            left_seqid_tokens  = per_space_token_default(mu['raw'], default="-1")
            right_seqid_tokens = per_space_token_default(mv['raw'], default="-1")
            w_addr_seqid_tok.write(
                f"{' '.join(left_seqid_tokens)}\t{' '.join(right_seqid_tokens)}\n"
            )

    print(f"[DONE][DFG] {out_dir}")

def main():
    random.seed(0)
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
# -*- coding: utf-8 -*-
from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path
from collections import defaultdict

# -------------------- regex & helpers --------------------
HEX_RE = re.compile(r'0x[0-9a-fA-F]+')

# Split punctuation and add spaces
TOKENIZE_RE = re.compile(
    r"0x[0-9A-Fa-f]+|[A-Za-z_.$][\w.$]*|[\[\]\+\-\*\(\),:]|\d+"
)

def normalize_spacing(s: str) -> str:
    """Turn raw disassembly into spaced tokens like 'lea rdi [ rel 0x406210 ]'."""
    toks = TOKENIZE_RE.findall(s)
    return " ".join(toks)

def tokenize_raw(s: str):
    """Split strictly by whitespace; each token gets one SRC and one TGT."""
    return s.split()

def per_space_token_targets_zero(raw_text: str):
    """File #4 (TGT): emit '0' once per space-split token."""
    n = len(tokenize_raw(raw_text))
    return ["0"] * max(1, n)

def per_space_token_default(raw_text: str, default="-1"):
    """File #5 (SeqID map): align to space-split tokens with a constant placeholder."""
    n = len(tokenize_raw(raw_text))
    return [default] * max(1, n)

# -------------------- instruction parser (for parsed_pairs.txt) --------------------
def parse_instruction(ins: str, symbol_map: dict[int, str], string_starts: set[int]) -> str:
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

# -------------------- DFG random walk --------------------
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
    p1 = out_dir / "raw_pairs.txt"
    p2 = out_dir / "parsed_pairs.txt"
    p3 = out_dir / "instr_addr_per_token.txt"
    p4 = out_dir / "addr_token_per_token.txt"
    p5 = out_dir / "addr_sent_seqid_per_token.txt"

    print(f"[INFO] Writing:\n 1) {p1}\n 2) {p2}\n 3) {p3}\n 4) {p4}\n 5) {p5}")

    # symbol maps
    symbol_map = {sym.address: sym.full_name for sym in bv.get_symbols()}
    string_starts = {s.start for s in bv.get_strings()}

    function_graphs = {}
    node_meta = {}

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

        # fill node_meta
        for node in G.nodes:
            if node == -1:
                continue
            raw = G.nodes[node].get("text", "")
            parsed = parse_instruction(raw, symbol_map, string_starts)
            normalized_raw = normalize_spacing(raw)
            raw_tok_count = len(tokenize_raw(normalized_raw))
            addr_tgt_zero = per_space_token_targets_zero(normalized_raw)
            node_meta[node] = {
                "raw": normalized_raw,
                "parsed": parsed,
                "raw_tok_count": raw_tok_count,
                "addr_tgt_zero": addr_tgt_zero,
            }

        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    # collect pairs
    pairs = []
    for name, graph in function_graphs.items():
        has_text = lambda a: a in node_meta and node_meta[a]["raw"] != ""
        sequences = dfg_random_walk(graph, 40, has_text)
        for seq in sequences:
            for i in range(1, window_size + 1):
                for idx in range(0, len(seq) - i):
                    u, v = seq[idx], seq[idx + i]
                    if u in node_meta and v in node_meta:
                        pairs.append((u, v))

    # write outputs
    with open(p1, "w", encoding="utf-8") as w_raw, \
         open(p2, "w", encoding="utf-8") as w_parsed, \
         open(p3, "w", encoding="utf-8") as w_instr, \
         open(p4, "w", encoding="utf-8") as w_addr_tok, \
         open(p5, "w", encoding="utf-8") as w_addr_seqid_tok:

        for line_idx, (u, v) in enumerate(pairs):
            mu, mv = node_meta[u], node_meta[v]

            # 1) raw_pairs.txt → normalized spacing for both sides
            w_raw.write(f"{mu['raw']}\t{mv['raw']}\n")

            # 2) parsed_pairs.txt
            w_parsed.write(f"{mu['parsed']}\t{mv['parsed']}\n")

            # 3) instr_addr_per_token.txt
            left_addr  = f"0x{u:x}"
            right_addr = f"0x{v:x}"
            left_rep   = " ".join([left_addr]  * max(1, mu["raw_tok_count"]))
            right_rep  = " ".join([right_addr] * max(1, mv["raw_tok_count"]))
            w_instr.write(f"{left_rep}\t{right_rep}\n")

            # 4) addr_token_per_token.txt
            w_addr_tok.write(
                f"{' '.join(mu['addr_tgt_zero'])}\t{' '.join(mv['addr_tgt_zero'])}\n"
            )

            # 5) addr_sent_seqid_per_token.txt
            left_seqid_tokens  = per_space_token_default(mu['raw'], default="-1")
            right_seqid_tokens = per_space_token_default(mv['raw'], default="-1")
            w_addr_seqid_tok.write(
                f"{' '.join(left_seqid_tokens)}\t{' '.join(right_seqid_tokens)}\n"
            )

    print(f"[DONE][DFG] {out_dir}")

def main():
    random.seed(0)
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
