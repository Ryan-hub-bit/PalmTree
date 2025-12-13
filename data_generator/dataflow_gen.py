from binaryninja import load 
import networkx as nx
import random
import os
import re
import tqdm
from collections import Counter
from pathlib import Path

def _split_tokens(s: str):
    return s.split()

def parse_instruction(ins, symbol_map, string_map):
    """
    Return:
      text        : normalized instruction string (unchanged format)
      tokens      : text.split()
      orig_tokens : original tokens (for real 0x... addresses)
    """
    ins_norm = re.sub(r'\s+', ', ', ins, 1)
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
        orig_tokens = list(tokens)

    return text, tokens, orig_tokens


def random_walk(g, length, symbol_map, string_map):
    seq = []
    for n in g:
        if n != -1 and g.nodes[n].get('text') is not None:
            s = []
            l = 0
            t, toks, otoks = parse_instruction(g.nodes[n]['text'], symbol_map, string_map)
            s.append({'text': t, 'tokens': toks, 'orig_tokens': otoks, 'addr': n})
            cur = n
            while l < length:
                nbs = list(g.successors(cur))
                if nbs:
                    cur = random.choice(nbs)
                    t, toks, otoks = parse_instruction(g.nodes[cur]['text'], symbol_map, string_map)
                    s.append({'text': t, 'tokens': toks, 'orig_tokens': otoks, 'addr': cur})
                    l += 1
                else:
                    break
            seq.append(s)
    return seq


def process_file(f):
    symbol_map = {}
    string_map = {}
    print(f)
    bv = load(f)

    # collect symbols and strings
    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    for string in bv.get_strings():
        string_map[string.start] = string.value

    function_graphs = {}
    for func in bv.functions:
        G = nx.DiGraph()
        G.add_node(-1, text='entry_point')
        for block in func.mlil:
            for ins in block:
                G.add_node(ins.address, text=bv.get_disassembly(ins.address))
                depd = []
                for var in ins.vars_read:
                    depd = [(func.mlil[i].address, ins.address)
                            for i in func.mlil.get_var_definitions(var)
                            if func.mlil[i].address != ins.address]
                for var in ins.vars_written:
                    depd += [(ins.address, func.mlil[i].address)
                             for i in func.mlil.get_var_uses(var)
                             if func.mlil[i].address != ins.address]
                if depd:
                    G.add_edges_from(depd)
        for node in list(G.nodes):
            if not G.in_degree(node):
                G.add_edge(-1, node)
        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    out_dir = Path("/home/louie/PalmTree/data/test/dfg")
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_name = Path(f).stem

    dfg_path = out_dir / f"{binary_name}_dfg_test.txt"
    src_path = out_dir / f"{binary_name}_dfg_test_src.txt"   # SRC
    tgt_path = out_dir / f"{binary_name}_dfg_test_tgt.txt"   # TGT

    print(f"[INFO] Writing to: {dfg_path}, {src_path}, {tgt_path}")

    with open(dfg_path, 'w', encoding='utf-8') as w_dfg, \
         open(src_path, 'w', encoding='utf-8') as w_src, \
         open(tgt_path, 'w', encoding='utf-8') as w_tgt:

        for name, graph in function_graphs.items():
            seqs = random_walk(graph, 40, symbol_map, string_map)
            for s in seqs:
                if len(s) >= 2:
                    for idx in range(1, len(s)):
                        left = s[idx-1]
                        right = s[idx]

                        # 1. DFG pairs (unchanged)
                        w_dfg.write(left['text'] + '\t' + right['text'] + '\n')

                        # 2. SRC = instruction address repeated per token (line-aligned)
                        l_src = ' '.join([hex(left['addr'])] * len(left['tokens']))
                        r_src = ' '.join([hex(right['addr'])] * len(right['tokens']))
                        w_src.write(l_src + '\t' + r_src + '\n')

                        # --- TGT: token-aligned actual literal addresses (0 if not address) ---
                        # (no 'tags' needed; we key off the normalized token == 'address')

                        # optional safety if lengths ever mismatch
                        if len(left['tokens']) != len(left['orig_tokens']):
                            left['orig_tokens'] = left['tokens']
                        if len(right['tokens']) != len(right['orig_tokens']):
                            right['orig_tokens'] = right['tokens']

                        l_tgt_items = [
                            (ot if tok == 'address' and isinstance(ot, str) and ot.startswith('0x') and len(ot) >= 6 else '0')
                            for tok, ot in zip(left['tokens'], left['orig_tokens'])
                        ]

                        r_tgt_items = [
                            (ot if tok == 'address' and isinstance(ot, str) and ot.startswith('0x') and len(ot) >= 6 else '0')
                            for tok, ot in zip(right['tokens'], right['orig_tokens'])
                        ]

                        w_tgt.write(' '.join(l_tgt_items) + '\t' + ' '.join(r_tgt_items) + '\n')





def main():
    bin_folder = '/home/louie/testbinary'
    file_lst = []
    for parent, _, files in os.walk(bin_folder):
        for f in files:
            file_lst.append(os.path.join(parent, f))
    for f in tqdm.tqdm(file_lst):
        process_file(f)


if __name__ == "__main__":
    main()
