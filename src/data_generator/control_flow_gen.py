from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path

# -------- config: 1-based (editor-style) line numbers; set to 0 for zero-based --------
LINE_BASE = 1

# ---------- robust hex detection ----------
# Matches: 0x1, 0XDEAD, -0x20, 1Ah, 0ffh; avoids identifiers like foo0x10bar
HEX_RE = re.compile(r'(?<![A-Za-z0-9_])(?:-?0[xX][0-9A-Fa-f]+|[0-9A-Fa-f]+[hH])(?![A-Za-z0-9_])')

def normalize_hex_literal(lit: str) -> str:
    """Normalize matched hex literal to 0x... form (handles ...h and uppercase). Keeps sign if present."""
    lit = lit.strip()
    sign = ''
    if lit.startswith('-'):
        sign, lit = '-', lit[1:]
    if lit.lower().endswith('h'):  # e.g., '1A3h' -> '0x1A3'
        core = lit[:-1]
        return (sign + '0x' + core).lower()
    return (sign + lit).lower()

def is_addr_like_str(s: str) -> bool:
    s = s.lower().strip()
    return s.startswith("0x") and len(s) >= 4  # accept small immediates, too

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

# ---------- instruction formatting cores ----------
def _format_parsed_style_tokens(ins: str):
    """
    Convert BN disasm into the same spacing/token layout used by 'parsed' output:
    - first whitespace -> ', '
    - split on ', ' into [opcode, operands...]
    - split operands by alnum boundaries and re-join with spaces
    Return list of tokens exactly as they'd appear space-separated.
    """
    ins2 = re.sub(r'\s+', ', ', ins, 1)
    parts = ins2.split(', ')
    opcode = parts[0] if parts else ''
    out_toks = [opcode] if opcode else []
    if len(parts) > 1:
        for op in parts[1:]:
            # break into [word][non-word]... so punctuation stays separated
            pieces = re.split(r'([0-9A-Za-z]+)', op)
            # re-join with single spaces (this mirrors parsed spacing)
            joined = ' '.join(p for p in pieces if p != '')
            # then split again to ensure clean tokens
            out_toks.extend(joined.split())
    return out_toks

def parse_tokens_symbolic(ins: str, symbol_map, string_map):
    """
    Same tokenization as parsed, but map address-like literals (after normalization)
    to 'symbol'/'string'/'address' based on maps.
    """
    toks = _format_parsed_style_tokens(ins)
    mapped = []
    for t in toks:
        m = HEX_RE.fullmatch(t)
        if m:
            lit = normalize_hex_literal(m.group(0))
            if is_addr_like_str(lit):
                try:
                    hv = int(lit, 16) if not lit.startswith('-') else int(lit, 16)
                except ValueError:
                    mapped.append(t)
                    continue
                if hv in symbol_map:
                    mapped.append('symbol'); continue
                if hv in string_map:
                    mapped.append('string'); continue
                mapped.append('address'); continue
        mapped.append(t)
    return mapped

def parse_tokens_keep_addrs(ins: str):
    """Same spacing/tokens as parsed, but DO NOT replace hex addresses."""
    return _format_parsed_style_tokens(ins)

# ---------- raw normalizer (space-separated, no commas) ----------
def normalize_raw_instruction(ins: str) -> str:
    toks = re.split(r'[,\s]+', ins.strip())
    return ' '.join(t for t in toks if t)

# ---------- per-token projections based on parsed_withaddr text ----------
def per_token_echo_hex_or_zero(from_text: str):
    """
    For each token in from_text (already space-separated like parsed_withaddr), echo the hex literal if address-like else '0'.
    """
    toks = from_text.split()
    out = []
    for t in toks:
        m = HEX_RE.fullmatch(t)
        if m:
            lit = normalize_hex_literal(m.group(0))
            out.append(lit if is_addr_like_str(lit) else "0")
        else:
            out.append("0")
    return out

def per_token_line_of_addr(from_text: str,
                           addr_to_seqid: dict[int, list[int]],
                           current_line_idx: int,
                           current_addrs: set[int]):
    """
    For each token in from_text:
      - not address-like -> '-1'
      - address-like AND is one of the current pair addresses -> current line number (same as instr_addr_per_token.txt)
      - address-like but appears elsewhere -> the FIRST line number it appears on
      - address-like but never used as a sentence address -> '-2'
    Line numbers are offset by LINE_BASE.
    """
    toks = from_text.split()
    out = []
    for t in toks:
        m = HEX_RE.fullmatch(t)
        if not m:
            out.append("-1"); continue
        lit = normalize_hex_literal(m.group(0))
        if not is_addr_like_str(lit):
            out.append("-1"); continue
        try:
            addr = int(lit, 16)
        except ValueError:
            out.append("-1"); continue

        if addr in current_addrs:
            out.append(str(current_line_idx + LINE_BASE))
            continue

        seqids = addr_to_seqid.get(addr)
        if seqids:
            out.append(str(min(seqids) + LINE_BASE))
        else:
            out.append("-2")
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
def process_file(f, window_size, output_root="/home/louie/PalmTree/data/kun/cfg"):
    print(f"[INFO] Processing: {f}")
    bv = load(f)

    binary_name = Path(f).stem
    out_dir = Path(output_root) / binary_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # outputs
    p1  = out_dir / "raw_pairs.txt"
    p2  = out_dir / "cfg_train.txt"
    p2b = out_dir / "cfg_train_addr.txt"
    p3  = out_dir / "cfg_train_src.txt"
    p4  = out_dir / "cfg_train_tgt.txt"
    p5  = out_dir / "tgt_id.txt"

    print(f"[INFO] Writing:\n 1) {p1}\n 2) {p2}\n 3) {p2b}\n 4) {p3}\n 5) {p4}\n 6) {p5}")

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

                # 1) raw variants
                raw_norm = normalize_raw_instruction(raw)

                # 2) parsed (symbol/string/address) tokens & string
                parsed_sym_toks = parse_tokens_symbolic(raw, symbol_map, string_map)
                parsed_sym_str = ' '.join(parsed_sym_toks)

                # 3) parsed_withaddr tokens & string (same spacing/tokens as parsed, but keep addresses)
                parsed_keep_toks = parse_tokens_keep_addrs(raw)
                parsed_keep_str = ' '.join(parsed_keep_toks)

                # counts and addr token projections are based on parsed_withaddr
                parsed_withaddr_tok_count = len(parsed_keep_toks)
                addr_tok_from_parsed_withaddr = per_token_echo_hex_or_zero(parsed_keep_str)

                G.add_node(curr, text=raw)
                node_meta[curr] = {
                    # raw outputs
                    "raw_norm": raw_norm,

                    # parsed variants
                    "parsed_sym": parsed_sym_str,
                    "parsed_keep": parsed_keep_str,

                    # token counts for per-token projections
                    "tok_count_for_per_token": parsed_withaddr_tok_count,

                    # per-token address echoes (from parsed_withaddr)
                    "addr_tok_from_parsed_withaddr": addr_tok_from_parsed_withaddr,
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
    addr_to_seqid = {}  # address -> list of line indices (0- or 1-based handled when writing)
    for line_idx, (u, v) in enumerate(pairs):
        addr_to_seqid.setdefault(u, []).append(line_idx)
        addr_to_seqid.setdefault(v, []).append(line_idx)

    # -------- SECOND PASS: write outputs --------
    with open(p1, "w", encoding="utf-8") as w_raw, \
         open(p2, "w", encoding="utf-8") as w_parsed, \
         open(p2b, "w", encoding="utf-8") as w_parsed_keep, \
         open(p3, "w", encoding="utf-8") as w_instr, \
         open(p4, "w", encoding="utf-8") as w_addr_tok, \
         open(p5, "w", encoding="utf-8") as w_addr_seqid_tok:

        for line_idx, (u, v) in enumerate(pairs):
            mu, mv = node_meta[u], node_meta[v]

            # 1) raw_pairs.txt (normalized, single spaces, commas removed)
            w_raw.write(f"{mu['raw_norm']}\t{mv['raw_norm']}\n")

            # 2) parsed_pairs.txt (symbol/string/address abstraction)
            w_parsed.write(f"{mu['parsed_sym']}\t{mv['parsed_sym']}\n")

            # 3) parsed_pairs_withaddr.txt (same spacing/tokens as parsed, but keep hex literals)
            w_parsed_keep.write(f"{mu['parsed_keep']}\t{mv['parsed_keep']}\n")

            # --- All per-token artifacts below are based on parsed_pairs_withaddr.txt ---

            # 4) instr_addr_per_token.txt:
            # repeat each sentence's *instruction address* for the number of tokens in parsed_withaddr
            left_addr  = f"0x{u:x}"
            right_addr = f"0x{v:x}"
            left_rep   = " ".join([left_addr]  * max(1, mu["tok_count_for_per_token"]))
            right_rep  = " ".join([right_addr] * max(1, mv["tok_count_for_per_token"]))
            w_instr.write(f"{left_rep}\t{right_rep}\n")

            # 5) addr_token_per_token.txt:
            # echo hex literal per token from parsed_withaddr, else '0'
            left_addr_toks  = ' '.join(mu['addr_tok_from_parsed_withaddr'])
            right_addr_toks = ' '.join(mv['addr_tok_from_parsed_withaddr'])
            w_addr_tok.write(f"{left_addr_toks}\t{right_addr_toks}\n")

            """ # 6) addr_sent_seqid_per_token.txt:
            # For address-like tokens: output a SINGLE line number.
            # Prefer the current pair's line number if the token's address is u or v.
            current_addrs = {u, v}
            left_seqids  = ' '.join(per_token_line_of_addr(mu['parsed_keep'],
                                                           addr_to_seqid,
                                                           current_line_idx=line_idx,
                                                           current_addrs=current_addrs))
            right_seqids = ' '.join(per_token_line_of_addr(mv['parsed_keep'],
                                                           addr_to_seqid,
                                                           current_line_idx=line_idx,
                                                           current_addrs=current_addrs))
            w_addr_seqid_tok.write(f"{left_seqids}\t{right_seqids}\n") """

    print(f"[DONE] {out_dir}")

def main():
    random.seed(0)
    # bin_folder = '/home/louie/PalmTree/src/data_generator/testbin'
    bin_folder = '/home/louie/smallbinary'  # <-- set your input folder
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

