
from binaryninja import load
import networkx as nx
import random
import os
import re
import tqdm
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
SEG_LEN  = 8   # one output line = 8 instructions
WALK_LEN = 40  # max steps per random walk


# ---------------------------
# Token helpers
# ---------------------------
def _split_tokens(s: str):
    return s.split()


# ---------------------------
# Instruction parser
# ---------------------------
def parse_instruction(ins, symbol_map, string_map):
    """
    Parse one disassembly line.

    Returns:
      text        : normalized instruction string
                    (addresses -> 'address', known -> 'symbol'/'string')
      tokens      : text.split()
      orig_tokens : original tokens (to recover real 0x... for tgt)
    """
    ins_norm = re.sub(r'\s+', ', ', ins, 1)  # compact only the first whitespace run
    parts = ins_norm.split(', ')
    opcode = parts[0]
    operand = parts[1:] if len(parts) > 1 else []

    replaced_operands = []
    original_operands = []

    for i in range(len(operand)):
        symbols = re.split(r'([0-9A-Za-z]+)', operand[i])
        orig_symbols = list(symbols)

        for j in range(len(symbols)):
            tok = symbols[j]
            if tok[:2] == '0x' and len(tok) >= 6:
                try:
                    hv = int(tok, 16)
                    if hv in symbol_map:
                        symbols[j] = "symbol"
                    elif hv in string_map:
                        symbols[j] = "string"
                    else:
                        symbols[j] = "address"
                except ValueError:
                    pass

        replaced_operands.append(' '.join(symbols))
        original_operands.append(' '.join(orig_symbols))

    text = ' '.join([opcode] + replaced_operands if replaced_operands else [opcode])
    orig_text = ' '.join([opcode] + original_operands if original_operands else [opcode])

    tokens = _split_tokens(text)
    orig_tokens = _split_tokens(orig_text)
    if len(tokens) != len(orig_tokens):
        # safety: keep alignment even if unexpected split differences occur
        orig_tokens = list(tokens)

    return text, tokens, orig_tokens


# ---------------------------
# Random walk over DFG-style graph
# ---------------------------
def random_walk(g, length, symbol_map, string_map):
    """
    Produce sequences of nodes by walking successors randomly.
    Each element in a sequence is a dict:
      { 'text': normalized_text, 'tokens': [...], 'orig_tokens': [...], 'addr': node_address }
    """
    seqs = []
    for n in g:
        if n == -1:
            continue
        node_text = g.nodes[n].get('text')
        if node_text is None:
            continue

        s = []
        steps = 0
        t, toks, otoks = parse_instruction(node_text, symbol_map, string_map)
        s.append({'text': t, 'tokens': toks, 'orig_tokens': otoks, 'addr': n})
        cur = n

        while steps < length:
            nbs = list(g.successors(cur))
            if not nbs:
                break
            cur = random.choice(nbs)
            node_text = g.nodes[cur].get('text')
            if node_text is None:
                break
            t, toks, otoks = parse_instruction(node_text, symbol_map, string_map)
            s.append({'text': t, 'tokens': toks, 'orig_tokens': otoks, 'addr': cur})
            steps += 1

        if s:
            seqs.append(s)

    return seqs


# ---------------------------
# Chunk builder (8 instructions per line)
# ---------------------------
def build_chunk(seq, start, k):
    """
    Build one 8-instruction window (tabs between instructions; spaces between tokens).
    Returns (seq_text, src_line, tgt_line)
      - seq_text: 8 instructions joined with '\t'
      - src_line: per-token instruction address (space-separated, concatenated across 8)
      - tgt_line: per-token address-or-0 (space-separated, concatenated across 8)
                  NOW records original hex for 'address', 'symbol', and 'string'
    """
    end = start + k
    if end > len(seq):
        return None

    # text: tabs between instructions
    seq_text = "\t".join(s['text'] for s in seq[start:end])

    # src: repeat instruction address for each token in that instruction
    def addr_line_for(node):
        n = len(node['tokens'])
        return " ".join([hex(node['addr'])] * n)

    src_line = " ".join(addr_line_for(s) for s in seq[start:end])

    # tgt: record real hex for normalized token in {'address','symbol','string'}, else 0
    INTERESTING = {"address", "symbol", "string"}

    def mask_for(node):
        toks, otoks = node['tokens'], node['orig_tokens']
        if len(toks) != len(otoks):
            otoks = toks
        items = [
            (ot if (t in INTERESTING and isinstance(ot, str) and ot.startswith('0x') and len(ot) >= 6) else '0')
            for t, ot in zip(toks, otoks)
        ]
        return " ".join(items)

    tgt_line = " ".join(mask_for(s) for s in seq[start:end])

    return seq_text, src_line, tgt_line


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
    for func in bv.functions:
        if not hasattr(func, "mlil") or func.mlil is None:
            continue

        G = nx.DiGraph()
        G.add_node(-1, text='entry_point')

        try:
            blocks = list(func.mlil)
        except Exception:
            continue

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

                G.add_node(addr, text=dis)

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

        # connect orphan sources to entry node
        for node in list(G.nodes):
            if node == -1:
                continue
            if G.in_degree(node) == 0:
                G.add_edge(-1, node)

        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    # Output paths
    out_dir = Path("/home/louie/PalmTree/datalong/dfg")
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_name = Path(f).name

    dfg_path = out_dir / f"{binary_name}_dfg_{SEG_LEN}.txt"       # sequence text (8 instructions, tabs between)
    src_path = out_dir / f"{binary_name}_dfg_{SEG_LEN}_src.txt"   # per-token instruction address
    tgt_path = out_dir / f"{binary_name}_dfg_{SEG_LEN}_tgt.txt"   # per-token hex for address/symbol/string; 0 otherwise

    print(f"[INFO] Writing to: {dfg_path}, {src_path}, {tgt_path}")

    written = 0
    with open(dfg_path, 'w', encoding='utf-8') as w_dfg, \
         open(src_path, 'w', encoding='utf-8') as w_src, \
         open(tgt_path, 'w', encoding='utf-8') as w_tgt:

        for _, graph in function_graphs.items():
            seqs = random_walk(graph, WALK_LEN, symbol_map, string_map)
            for s in seqs:
                if len(s) < SEG_LEN:
                    continue
                for start in range(0, len(s) - SEG_LEN + 1):  # stride 1; use step=SEG_LEN for non-overlap
                    built = build_chunk(s, start, SEG_LEN)
                    if not built:
                        continue
                    seq_txt, src_line, tgt_line = built

                    # enforce exactly 8 instructions per line (7 tabs)
                    if seq_txt.count("\t") != SEG_LEN - 1:
                        continue

                    w_dfg.write(seq_txt + '\n')
                    w_src.write(src_line + '\n')
                    w_tgt.write(tgt_line + '\n')
                    written += 1

    print(f"[DONE] {dfg_path} (wrote {written} sequences)")


# ---------------------------
# Main
# ---------------------------
def main():
    bin_folder = '/home/louie/smallbinary'
    file_lst = []
    for parent, _, files in os.walk(bin_folder):
        for f in files:
            file_lst.append(os.path.join(parent, f))

    for f in tqdm.tqdm(file_lst):
        process_file(f)


if __name__ == "__main__":
    main()
