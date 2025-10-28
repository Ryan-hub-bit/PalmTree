#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge all *_cfg.txt, *_cfg_src.txt, and *_cfg_tgt.txt files
under a given folder into unified files:
  - cfg_train.txt
  - cfg_train_src.txt
  - cfg_train_tgt.txt
"""

import os
from pathlib import Path

# Root directory containing all subfolders
root_dir = Path("/home/louie/PalmTree/datalong/test/cfg")
seg_len = 8

parent_dir = root_dir.parent  # removes the last element

out_train = parent_dir / f"cfg_{seg_len}.txt"
out_src   = parent_dir / f"cfg_{seg_len}_src.txt"
out_tgt   = parent_dir / f"cfg_{seg_len}_tgt.txt"
# Collect all matching files recursively
train_files = list(root_dir.rglob(f"*_cfg_{seg_len}.txt"))
src_files   = list(root_dir.rglob(f"*_cfg_{seg_len}_src.txt"))
tgt_files   = list(root_dir.rglob(f"*_cfg_{seg_len}_tgt.txt"))

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
    print(f"[INFO] Found {len(train_files)} *_cfg.txt files")
    print(f"[INFO] Found {len(src_files)} *_cfg_src.txt files")
    print(f"[INFO] Found {len(tgt_files)} *_cfg_tgt.txt files")

    merge_files(train_files, out_train)
    merge_files(src_files, out_src)
    merge_files(tgt_files, out_tgt)

if __name__ == "__main__":
    main()

