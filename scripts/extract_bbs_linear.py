#!/usr/bin/env python3
"""Extract basic blocks in linear order (one line per BB) using bb_flow helpers.

This script follows the same loading and formatting logic as `scripts/bb_flow.py`.
For each basic block it writes a single line with the format:

  <addr_start:0xSTART:bin_norm:func_norm> INST1\tINST2\t... <addr_end:0xEND:bin_norm:func_norm>

Usage:
  python3 scripts/extract_bbs_linear.py /path/to/binary /path/to/output.txt

It imports `get_basic_block_seq` and `format_bb_with_addr` from `scripts.bb_flow` so
it will behave consistently with that script's heuristics.
"""
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

try:
    from scripts.bb_flow import get_basic_block_seq, format_bb_with_addr
except Exception as e:
    # If import fails, provide a helpful message
    raise RuntimeError("Could not import helpers from scripts/bb_flow.py. Make sure you're running this from the repo root and Binary Ninja is available if bb_flow imports it.") from e

def process_binary(binpath: str, outpath: str):
    import binaryninja

    with binaryninja.load(binpath) as bv:
        bv.update_analysis_and_wait()

        binary_base = bv.start
        binary_end = bv.end
        binary_length = max(1, binary_end - binary_base)

        with open(outpath, 'w', encoding='utf-8') as outf:
            for func in bv.functions:
                func_start = func.start
                func_length = func.total_bytes if func.total_bytes > 0 else 1

                # iterate basic blocks in linear order (by appearance in function)
                for bb in func:
                    inst_seq, start_addr, end_addr = get_basic_block_seq(bb)
                    if not inst_seq:
                        continue

                    line = format_bb_with_addr(inst_seq, start_addr, end_addr,
                                               binary_base=binary_base,
                                               binary_length=binary_length,
                                               func_start=func_start,
                                               func_length=func_length,
                                               bv=bv)
                    outf.write(line + '\n')

    print(f"Wrote basic blocks (linear order) to {outpath}")


def main():
    if len(sys.argv) < 3:
        print('Usage: scripts/extract_bbs_linear.py <binary> <output_file>')
        sys.exit(1)
    binpath = sys.argv[1]
    outpath = sys.argv[2]
    process_binary(binpath, outpath)


if __name__ == '__main__':
    main()
