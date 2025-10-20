from binaryninja import *
from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path
from collections import Counter

# --- helpers ---------------------------------------------------------------
HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def parse_instruction(ins: str, symbol_map: dict, string_map: dict) -> str:
    """Normalize a Binary Ninja disassembly line.

    Minimal edits vs your original: keep whitespace compaction + symbol/string/address tagging.
    """
    ins = re.sub(r"\s+", ", ", ins, 1)  # compact the first run of spaces between mnemonic and operands
    parts = ins.split(", ")
    operand = []
    if len(parts) > 1:
        operand = parts[1:]
    for i in range(len(operand)):
        symbols = re.split(r"([0-9A-Za-z]+)", operand[i])
        for j in range(len(symbols)):
            if symbols[j][:2] == "0x" and len(symbols[j]) >= 6:
                try:
                    hv = int(symbols[j], 16)
                except ValueError:
                    continue
                if hv in symbol_map:
                    symbols[j] = "symbol"
                elif hv in string_map:
                    symbols[j] = "string"
                else:
                    pass  # keep real hex addresses so we can emit address/0 masks later
        operand[i] = " ".join(symbols)
    opcode = parts[0]
    return " ".join([opcode] + operand)


def random_walk(g: nx.DiGraph, length: int, symbol_map: dict, string_map: dict):
    """Collect short instruction sequences by walking each function CFG.
    Returns a list of (addr:int, text:str) tuples so we can emit per-token address lines like DFG.
    """
    sequence = []
    for n in g:
        if n != -1 and 'text' in g.nodes[n]:
            s = []
            l = 0
            s.append((n, parse_instruction(g.nodes[n]['text'], symbol_map, string_map)))
            cur = n
            while l < length:
                nbs = list(g.successors(cur))
                if len(nbs):
                    cur = random.choice(nbs)
                    if 'text' in g.nodes[cur]:
                        s.append((cur, parse_instruction(g.nodes[cur]['text'], symbol_map, string_map)))
                        l += 1
                    else:
                        break
                else:
                    break
            sequence.append(s)
        if len(sequence) > 5000:
            return sequence[:5000]
    return sequence[:5000]


def tokens_of(line: str):
    """Tokenize a normalized instruction line in the simplest, DFG-style way: space split."""
    return line.strip().split()


def addr_mask_of(line: str):
    """Per-token address mask: keep hex like 0x..., else '0'."""
    out = []
    for tok in tokens_of(line):
        if tok.startswith("0x") and len(tok) >= 6:
            out.append(tok)
        else:
            out.append("0")
    return " ".join(out)


def instr_addr_line(addr: int, line: str) -> str:
    """DFG-style: repeat the instruction address (hex) for each token in the line."""
    n = len(tokens_of(line))
    return " ".join([hex(addr)] * n)


# --- main pipeline ---------------------------------------------------------

def process_file(f: str, window_size: int):
    symbol_map = {}
    string_map = {}

    print(f"[INFO] Processing: {f}")
    bv = load(f)

    # Create an output directory (once)
    out_dir = Path("/home/louie/PalmTree/data/cfg/")
    out_dir.mkdir(parents=True, exist_ok=True)
     # Get the binary name without extension
    binary_name = Path(f).stem

    # Aggregate *train* files (append-only) to mirror your original logic
    out_pairs = out_dir / f"{binary_name}_cfg_train.txt"
    out_src = out_dir / f"{binary_name}_cfg_train_src.txt"
    out_tgt = out_dir / f"{binary_name}_cfg_train_tgt.txt"
    # Collect symbols and strings
    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    for string in bv.get_strings():
        string_map[string.start] = string.value

    # Build per-function CFGs
    function_graphs = {}
    for func in bv.functions:
        G = nx.DiGraph()
        for block in func:
            curr = block.start
            predecessor = curr
            for inst in block:
                disasm = bv.get_disassembly(curr)
                G.add_node(curr, text=disasm)
                if curr != block.start:
                    G.add_edge(predecessor, curr)
                predecessor = curr
                curr += inst[1]  # advance by instruction length
            for edge in block.outgoing_edges:
                G.add_edge(predecessor, edge.target.start)
        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    # Write sequences (mirroring your original windowed bi-directional pairing)
    with open(out_pairs, "a", encoding="utf-8") as w_pairs, \
         open(out_src, "a", encoding="utf-8") as w_src, \
         open(out_tgt, "a", encoding="utf-8") as w_tgt:
        for name, graph in function_graphs.items():
            sequence = random_walk(graph, 40, symbol_map, string_map)
            for s in sequence:
                if len(s) >= 4:
                    for idx in range(0, len(s)):
                        for i in range(1, window_size + 1):
                            # backward pairs (strictly idx - i > 0, per your snippet)
                            if idx - i > 0:
                                (src_addr, src_txt) = s[idx - i]
                                (tgt_addr, tgt_txt) = s[idx]
                                w_pairs.write(src_txt + "\t" + tgt_txt + "\n")
                                # SRC: instruction address repeated per token (DFG-style)
                                w_src.write(instr_addr_line(src_addr, src_txt) + "\t" + instr_addr_line(tgt_addr, tgt_txt) + "\n")
                                # TGT: address-if-token-else-0 mask
                                w_tgt.write(addr_mask_of(src_txt) + "\t" + addr_mask_of(tgt_txt) + "\n")
                            # forward pairs
                            if idx + i < len(s):
                                (src2_addr, src2_txt) = s[idx]
                                (tgt2_addr, tgt2_txt) = s[idx + i]
                                w_pairs.write(src2_txt + "\t" + tgt2_txt + "\n")
                                w_src.write(instr_addr_line(src2_addr, src2_txt) + "\t" + instr_addr_line(tgt2_addr, tgt2_txt) + "\n")
                                # TGT: address-if-token-else-0 mask
                                w_tgt.write(addr_mask_of(src2_txt) + "\t" + addr_mask_of(tgt2_txt) + "\n")

    print(f"[DONE] {out_pairs}")


def main():
    bin_folder = "/home/louie/smallbinary"
    file_lst = []
    window_size = 1  # minimal change: same default as before

    for parent, subdirs, files in os.walk(bin_folder):
        for f in files:
            file_lst.append(os.path.join(parent, f))

    for i, f in enumerate(file_lst):
        print(i, "/", len(file_lst))
        process_file(f, window_size)


if __name__ == "__main__":
    main()
