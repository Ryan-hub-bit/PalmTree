#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge all cfg_train*.txt files under /home/louie/PalmTree/data/kun/cfg
into three unified files at the root:
  - merged_cfg_train.txt
  - merged_cfg_train_src.txt
  - merged_cfg_train_tgt.txt
"""
import os
from pathlib import Path

# root directory containing all subfolders
root_dir = Path("/home/louie/PalmTree/data/kun/dfg")

# output file paths (placed in root_dir)
out_train = root_dir / "dfg_train.txt"
out_src   = root_dir / "dfg_train_src.txt"
out_tgt   = root_dir / "dfg_train_tgt.txt"

# collect all matching files
train_files = list(root_dir.rglob("dfg_train.txt"))
src_files   = list(root_dir.rglob("dfg_train_src.txt"))
tgt_files   = list(root_dir.rglob("dfg_train_tgt.txt"))

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
    print(f"[INFO] Found {len(train_files)} dfg_train.txt files")
    print(f"[INFO] Found {len(src_files)} dfg_train_src.txt files")
    print(f"[INFO] Found {len(tgt_files)} dfg_train_tgt.txt files")

    merge_files(train_files, out_train)
    merge_files(src_files, out_src)
    merge_files(tgt_files, out_tgt)

if __name__ == "__main__":
    main()
