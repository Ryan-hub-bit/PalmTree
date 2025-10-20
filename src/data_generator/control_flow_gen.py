# -*- coding: utf-8 -*-
from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path

# -------------------- tokenization / normalization --------------------
# Matches hex, identifiers/opcodes, punctuation like [], commas, etc., and numbers
TOKENIZE_RE = re.compile(
    r"0x[0-9A-Fa-f]+|[A-Za-z_.$][\w.$]*|[\[\]\+\-\*\(\),:]|\d+"
)

def normalize_spacing(s: str) -> str:
    """
    Normalize raw disassembly so every symbol is separated by a single space.
    Example: 'lea     rdi, [rel 0x406210]' -> 'lea rdi [ rel 0x406210 ]'
    """
    toks = TOKENIZE_RE.findall(s)
    return " ".join(toks)

# -------------------- instruction parser (kept for parsed_pairs.txt) --------------------
HEX_RE = re.compile(r'0x[0-9a-fA-F]+')

def parse_instruction(ins, symbol_map, string_map):
    """
    Replace address-like immediates with 'symbol' / 'string' / 'address'.
    Keeps your previous parse format (first whitespace -> comma+space split).
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

# -------------------- space-split helpers (SRC/TGT alignment) --------------------
def tokenize_raw(s: str):
    """Split strictly by whitespace; each token gets one SRC and one TGT."""
    return s.split()

def per_space_token_targets_zero(raw_text: str):
    """File #4 (TGT): emit '0' once per space-split token."""
    n = len(tokenize_raw(raw_text))
    return ["0"] * max(1, n)

def per_space_token_default(raw_text: str, default="-1"):
    """File #5 (SeqID map placeholder): align to space-split tokens with a constant value."""
    n = len(tokenize_raw(raw_text))
    return [default] * max(1, n)

# -------------------- CFG random walk --------------------
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

# -------------------- per-binary processing --------------------
def process_file(f, window_size, output_root="/home/louie/PalmTree/data/kun/cfg"):
    print(f"[INFO] Processing: {f}")
    bv = load(f)

    binary_name = Path(f).stem
    out_dir = Path(output_root) / binary_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # outputs
    p1 = out_dir / "raw_pairs.txt"
    p2 = out_dir / "parsed_pairs.txt"
    p3 = out_dir / "instr_addr_per_token.txt"       # SRC per token = instruction address
    p4 = out_dir / "addr_token_per_token.txt"       # TGT per token = "0"
    p5 = out_dir / "addr_sent_seqid_per_token.txt"  # placeholder aligned to space tokens

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
                # get disassembly and normalize spacing for raw_pairs + token counts
                raw_original = bv.get_disassembly(curr)
                raw = normalize_spacing(raw_original)
                parsed = parse_instruction(raw_original, symbol_map, string_map)

                # SPACE-SPLIT token count and TGT=0s sized to it
                raw_tok_count = len(tokenize_raw(raw))
                addr_tgt_zero = per_space_token_targets_zero(raw)

                # store normalized raw
                G.add_node(curr, text=raw)
                node_meta[curr] = {
                    "raw": raw,                      # normalized spaced raw
                    "parsed": parsed,               # your parsed format
                    "raw_tok_count": raw_tok_count, # per-token counts (space-split)
                    "addr_tok": addr_tgt_zero,      # zeros per token
                }

                if curr != block.start:
                    G.add_edge(predecessor, curr)
                predecessor = curr

                # Advance by instruction length; Binary Ninja's block iteration yields tuples (il, len)
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

    # -------- SECOND PASS: write outputs --------
    with open(p1, "w", encoding="utf-8") as w_raw, \
         open(p2, "w", encoding="utf-8") as w_parsed, \
         open(p3, "w", encoding="utf-8") as w_instr, \
         open(p4, "w", encoding="utf-8") as w_addr_tok, \
         open(p5, "w", encoding="utf-8") as w_addr_seqid_tok:

        for line_idx, (u, v) in enumerate(pairs):
            mu, mv = node_meta[u], node_meta[v]

            # 1) raw_pairs.txt (normalized with one space between all symbols)
            w_raw.write(f"{mu['raw']}\t{mv['raw']}\n")

            # 2) parsed_pairs.txt
            w_parsed.write(f"{mu['parsed']}\t{mv['parsed']}\n")

            # 3) instr_addr_per_token.txt (SRC per token = instruction address, repeated per space token)
            left_addr  = f"0x{u:x}"
            right_addr = f"0x{v:x}"
            left_rep   = " ".join([left_addr]  * max(1, mu["raw_tok_count"]))
            right_rep  = " ".join([right_addr] * max(1, mv["raw_tok_count"]))
            w_instr.write(f"{left_rep}\t{right_rep}\n")

            # 4) addr_token_per_token.txt (TGT per token = "0", sized to space tokens)
            w_addr_tok.write(
                f"{' '.join(mu['addr_tok'])}\t{' '.join(mv['addr_tok'])}\n"
            )

            # 5) addr_sent_seqid_per_token.txt (aligned to space tokens; default "-1")
            left_seqid_tokens  = per_space_token_default(mu['raw'], default="-1")
            right_seqid_tokens = per_space_token_default(mv['raw'], default="-1")
            w_addr_seqid_tok.write(
                f"{' '.join(left_seqid_tokens)}\t{' '.join(right_seqid_tokens)}\n"
            )

    print(f"[DONE] {out_dir}")

def main():
    random.seed(0)
    #bin_folder = '/home/louie/PalmTree/src/data_generator/testbin'
    bin_folder = '/home/louie/smallbinary'
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

