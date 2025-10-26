from binaryninja import load
import networkx as nx
import random
import os
import re
from pathlib import Path
from collections import Counter

# --- helpers ---------------------------------------------------------------
HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def parse_instruction(ins: str, symbol_map: dict, string_map: dict) -> tuple:
    """Normalize a Binary Ninja disassembly line and replace address constants with [addr].
    Returns: (normalized_instruction, address_mask)
    where address_mask contains the original hex values or '0' for each token.
    """
    ins = re.sub(r"\s+", " ", ins.strip())
    tokens = ins.split()
    
    normalized_tokens = []
    address_tokens = []
    
    for token in tokens:
        # Check if this token is a hex address
        if token.startswith("0x") and len(token) >= 6:
            original_hex = token
            try:
                hv = int(token, 16)
                # Normalize the token
                if hv in symbol_map:
                    normalized_tokens.append("symbol")
                elif hv in string_map:
                    normalized_tokens.append("string")
                else:
                    normalized_tokens.append("address")
                # Keep original hex in address mask
                address_tokens.append(original_hex)
            except ValueError:
                # Not a valid hex, treat as regular token
                normalized_tokens.append(token)
                address_tokens.append("0")
        else:
            # Not an address token
            normalized_tokens.append(token)
            address_tokens.append("0")
    
    normalized_instr = " ".join(normalized_tokens)
    address_mask = " ".join(address_tokens)
    
    return (normalized_instr, address_mask)



def random_walk(g: nx.DiGraph, length: int, symbol_map: dict, string_map: dict):
    """Collect short instruction sequences by walking each function CFG.
    Returns a list of (addr:int, text:str, mask:str) tuples where:
    - addr: instruction address
    - text: normalized instruction (with 'address', 'symbol', 'string' labels)
    - mask: address mask (original hex values or '0')
    """
    sequence = []
    for n in g:
        if n != -1 and 'text' in g.nodes[n]:
            s = []
            l = 0
            text, mask = parse_instruction(g.nodes[n]['text'], symbol_map, string_map)
            s.append((n, text, mask))
            cur = n
            while l < length:
                nbs = list(g.successors(cur))
                if len(nbs):
                    cur = random.choice(nbs)
                    if 'text' in g.nodes[cur]:
                        text, mask = parse_instruction(g.nodes[cur]['text'], symbol_map, string_map)
                        s.append((cur, text, mask))
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


# addr_mask_of is no longer needed since we capture masks during parsing


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
    out_dir = Path("/home/louie/PalmTree/data/cfg")
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
                                (src_addr, src_txt, src_mask) = s[idx - i]
                                (tgt_addr, tgt_txt, tgt_mask) = s[idx]
                                w_pairs.write(src_txt + "\t" + tgt_txt + "\n")
                                # SRC: instruction address repeated per token (DFG-style)
                                w_src.write(instr_addr_line(src_addr, src_txt) + "\t" + instr_addr_line(tgt_addr, tgt_txt) + "\n")
                                # TGT: use the pre-captured address mask
                                w_tgt.write(src_mask + "\t" + tgt_mask + "\n")
                            # forward pairs
                            if idx + i < len(s):
                                (src2_addr, src2_txt, src2_mask) = s[idx]
                                (tgt2_addr, tgt2_txt, tgt2_mask) = s[idx + i]
                                w_pairs.write(src2_txt + "\t" + tgt2_txt + "\n")
                                w_src.write(instr_addr_line(src2_addr, src2_txt) + "\t" + instr_addr_line(tgt2_addr, tgt2_txt) + "\n")
                                # TGT: use the pre-captured address mask
                                w_tgt.write(src2_mask + "\t" + tgt2_mask + "\n")

    print(f"[DONE] {out_pairs}")


def main():
    bin_folder = "/home/louie/smallbinary"
    file_lst = []
    print(1)
    window_size = 1  # minimal change: same default as before

    for parent, subdirs, files in os.walk(bin_folder):
        for f in files:
            file_lst.append(os.path.join(parent, f))

    for i, f in enumerate(file_lst):
        print(i, "/", len(file_lst))
        process_file(f, window_size)


if __name__ == "__main__":
    main()
