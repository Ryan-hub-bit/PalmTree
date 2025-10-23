#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge all *_dfg.txt, *_dfg_src.txt, and *_dfg_tgt.txt files
under a given folder into unified files:
  - dfg_train.txt
  - dfg_train_src.txt
  - dfg_train_tgt.txt
"""

import os
from pathlib import Path

# Root directory containing all subfolders
root_dir = Path("/home/louie/PalmTree/data/test/dfg")

parent_dir = root_dir.parent  # removes the last element

out_train = parent_dir / "dfg_test.txt"
out_src   = parent_dir / "dfg_test_src.txt"
out_tgt   = parent_dir / "dfg_test_tgt.txt"
# Collect all matching files recursively
train_files = list(root_dir.rglob("*_dfg_test.txt"))
src_files   = list(root_dir.rglob("*_dfg_test_src.txt"))
tgt_files   = list(root_dir.rglob("*_dfg_test_tgt.txt"))

def merge_files(file_list, output_file):
    """Concatenate all text files in file_list into output_file."""
    with open(output_file, "w", encoding="utf-8") as fout:
        for i, fpath in enumerate(sorted(file_list)):
            try:
                with open(fpath, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                print(f"[{i+1}/{len(file_list)}] Merged: {fpath}")
            except Exception as e:
                print(f"[WARN] Failed reading {fpath}: {e}")
    print(f"[DONE] Wrote {output_file} ({len(file_list)} files)")

def main():
    print(f"[INFO] Found {len(train_files)} *_dfg.txt files")
    print(f"[INFO] Found {len(src_files)} *_dfg_src.txt files")
    print(f"[INFO] Found {len(tgt_files)} *_dfg_tgt.txt files")

    merge_files(train_files, out_train)
    merge_files(src_files, out_src)
    merge_files(tgt_files, out_tgt)

if __name__ == "__main__":
    main()

