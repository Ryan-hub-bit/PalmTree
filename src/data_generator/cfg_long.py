from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path

# =========================
# Config
# =========================
SEG_LEN = 2  # each output line = 8 consecutive instructions on a CFG path

# =========================
# Address-aware normalization
# =========================
HEX_RE = re.compile(r"0x[0-9a-fA-F]+")

def normalize_and_mask(ins_raw: str, symbol_map: dict, string_map: dict):
    """
    Produce a normalized instruction text and a per-token target mask:
      - norm_text: unknown hex immediates replaced with [addr], known with 'symbol'/'string'
      - mask_line: original hex address token if token was an address, else '0'

    Tokenization is done in one pass so norm_text and mask_line have identical token counts.
    """
    # Compact the first whitespace run (mnemonic vs operands) like your original
    ins = re.sub(r"\s+", ", ", ins_raw, 1)
    parts = ins.split(", ")
    opcode = parts[0]
    operands = parts[1:] if len(parts) > 1 else []

    out_tokens = [opcode]
    mask_tokens = ["0"]  # opcode is not an address

    for op in operands:
        # split but preserve alphanumeric chunks to examine '0x...' separately
        pieces = re.split(r"([0-9A-Za-z]+)", op)
        for tok in pieces:
            if tok == "":
                continue

            is_hex = tok.startswith("0x") and len(tok) >= 6 and bool(HEX_RE.fullmatch(tok))
            if is_hex:
                try:
                    hv = int(tok, 16)
                except ValueError:
                    hv = None

                if hv is not None and hv in symbol_map:
                    out_tokens.append("symbol")
                    mask_tokens.append(tok)  # keep original hex in the mask
                elif hv is not None and hv in string_map:
                    out_tokens.append("string")
                    mask_tokens.append(tok)
                else:
                    out_tokens.append("addr")  # replace AFTER recording mask
                    mask_tokens.append(tok)
            else:
                out_tokens.append(tok)
                mask_tokens.append("0")

    norm_text = " ".join(out_tokens)
    mask_line = " ".join(mask_tokens)
    return norm_text, mask_line

# =========================
# CFG walk + windowing
# =========================
def random_walk(g: nx.DiGraph, length: int):
    """
    Sample path sequences from the CFG. Each sequence element is (addr, norm_text, mask_line).
    Assumes nodes in g have attributes: 'text' (normalized) and 'mask' (per-token mask).
    """
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

def build_chunk_from_seq(seq, start_idx: int, k: int):
    """
    Build one 8-instruction window.
      Returns (text_joined, addr_line, tgt_mask)
      - text_joined: instructions joined with '\t' (tabs between instructions; spaces between tokens)
      - addr_line: per-token instruction-address line (space-separated)
      - tgt_mask: per-token mask ('0' or original hex), space-separated
    """
    end_idx = start_idx + k
    if start_idx < 0 or end_idx > len(seq):
        return None
    chunk = seq[start_idx:end_idx]  # (addr, norm_text, mask_line)

    # Tabs between instructions; spaces inside instructions
    text_joined = "\t".join(t for _, t, _ in chunk)

    # Per-instruction address repeated for each token in the normalized text
    def instr_addr_line(addr: int, norm_line: str) -> str:
        n_tokens = len(norm_line.strip().split())
        return " ".join([hex(addr)] * n_tokens)

    addr_line = " ".join(instr_addr_line(a, t) for a, t, _ in chunk)

    # Mask captured BEFORE replacement; just concatenate per-instruction masks
    tgt_mask  = " ".join(m for _, _, m in chunk)

    return text_joined, addr_line, tgt_mask

# =========================
# Main processing
# =========================
def process_file(fpath: str):
    print(f"[INFO] Processing: {fpath}")
    bv = load(fpath)
    if bv is None:
        print(f"[WARN] Could not load {fpath}; skipping.")
        return

    # Output files (one line per 8-instruction window)
    out_dir = Path("/home/louie/PalmTree/data/test/cfg")
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_name = Path(fpath).name

    out_seq = out_dir / f"{binary_name}_cfg_{SEG_LEN}.txt"        # 8 instructions joined by '\t'
    out_src = out_dir / f"{binary_name}_cfg_{SEG_LEN}_src.txt"    # per-token instruction address
    out_tgt = out_dir / f"{binary_name}_cfg_{SEG_LEN}_tgt.txt"    # per-token mask (hex-or-0)

    # Build symbol & string maps
    symbol_map = {sym.address: sym.full_name for sym in bv.get_symbols()}
    string_map = {s.start: s.value for s in bv.get_strings()}

    # Build per-function CFGs with normalized text + mask on each node
    function_graphs = {}
    for func in bv.functions:
        G = nx.DiGraph()
        for block in func:
            curr = block.start
            predecessor = curr
            for inst in block:
                disasm_raw = bv.get_disassembly(curr)
                norm_text, mask_line = normalize_and_mask(disasm_raw, symbol_map, string_map)
                G.add_node(curr, text=norm_text, mask=mask_line)
                if curr != block.start:
                    G.add_edge(predecessor, curr)
                predecessor = curr
                curr += inst[1]  # advance by instruction length
            for edge in block.outgoing_edges:
                G.add_edge(predecessor, edge.target.start)
        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    # Generate windows and write aligned files
    written = 0
    with open(out_seq, "a", encoding="utf-8") as w_seq, \
         open(out_src, "a", encoding="utf-8") as w_src, \
         open(out_tgt, "a", encoding="utf-8") as w_tgt:

        for _, graph in function_graphs.items():
            # longer walk improves the chance of multiple windows
            walks = random_walk(graph, length=40)
            for s in walks:
                if len(s) < SEG_LEN:
                    continue
                for start in range(0, len(s) - SEG_LEN + 1):
                    packed = build_chunk_from_seq(s, start, SEG_LEN)
                    if not packed:
                        continue
                    seq_txt, addr_line, mask_line = packed
                    w_seq.write(seq_txt + "\n")
                    w_src.write(addr_line + "\n")
                    w_tgt.write(mask_line + "\n")
                    written += 1

    print(f"[DONE] {out_seq}  (wrote {written} sequences)")

def is_output_file(fname: str) -> bool:
    """Skip our own generated text files so we don't try to load them as binaries."""
    if not fname.endswith(".txt"):
        return False
    base = Path(fname).name
    return (
        base.endswith("_cfg_train.txt")
        or base.endswith("_cfg_train_src.txt")
        or base.endswith("_cfg_train_tgt.txt")
    )

def main():
    bin_folder = "/home/louie/testbinary"
    file_lst = []

    for parent, _, files in os.walk(bin_folder):
        for f in files:
            full = os.path.join(parent, f)
            if is_output_file(full):
                continue
            if f.endswith(".txt"):
                continue
            file_lst.append(full)

    for i, f in enumerate(sorted(file_lst)):
        print(i, "/", len(file_lst))
        process_file(f)

if __name__ == "__main__":
    main()
