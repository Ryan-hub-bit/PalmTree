#!/usr/bin/env python3
"""
Check func_blocks JSON for functions with num_instructions below a threshold.
Usage:
  python3 check_func_blocks_min_instr.py /data/kun/jtrans/baseline/func_blocks_baseline.json --threshold 5 --sample 10
"""
import json
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func_blocks', help='Path to func_blocks json')
    parser.add_argument('--threshold', type=int, default=5, help='Instruction count threshold')
    parser.add_argument('--sample', type=int, default=10, help='Number of examples to show')
    args = parser.parse_args()

    p = Path(args.func_blocks)
    if not p.exists():
        print(f'File not found: {p}')
        return

    with open(p, 'r') as f:
        data = json.load(f)

    total = len(data)
    below = []

    for fid, info in data.items():
        num = info.get('num_instructions')
        if num is None:
            # try to infer from tokens or asm
            if 'tokens' in info:
                num = len(info.get('tokens','').split('\t')) if info.get('tokens') else 0
            elif 'asm' in info:
                num = len(info.get('asm', []))
            else:
                num = 0
        if num < args.threshold:
            below.append((fid, info.get('binary',''), info.get('name',''), num))

    print('Summary')
    print('-------')
    print(f'Total functions: {total}')
    print(f'Functions with num_instructions < {args.threshold}: {len(below)} ({len(below)/total*100:.2f}%)')

    if below:
        print('\nExamples:')
        for i, (fid, binary, name, num) in enumerate(below[:args.sample]):
            print(f'  {i+1}. id={fid} binary={binary} name={name} instr={num}')

if __name__ == '__main__':
    main()
