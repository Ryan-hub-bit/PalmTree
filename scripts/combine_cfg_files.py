#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

ROOT = Path("/home/louie/PalmTree/data/kun/dfg")

# The five per-binary files we’ll merge
FILENAMES = [
    "raw_pairs.txt",
    "parsed_pairs.txt",
    "instr_addr_per_token.txt",
    "addr_token_per_token.txt",
    "addr_sent_seqid_per_token.txt",
]

# Where to write the line-range index
INDEX_FILE = ROOT / "combined_line_ranges.txt"


def iter_binary_dirs(root: Path):
    """Yield immediate subdirectories (binaries), sorted for determinism."""
    for p in sorted([d for d in root.iterdir() if d.is_dir()]):
        yield p


def count_lines(path: Path) -> int:
    """Fast-ish line counter. Returns 0 if file doesn’t exist."""
    if not path.exists():
        return 0
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def combine_one(filename: str):
    """
    Combine filename across all binary subdirs into ROOT/filename.
    Return a list of (binary_name, start_line, end_line) for the index.
    """
    out_path = ROOT / filename
    out_path_tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    ranges = []
    current_line = 1  # 1-based inclusive line numbering in the combined file

    with out_path_tmp.open("w", encoding="utf-8", newline="") as w:
        for bindir in iter_binary_dirs(ROOT):
            src = bindir / filename
            if not src.exists():
                # Skip missing files; not all binaries need to have every file
                continue

            lines_written = 0
            with src.open("r", encoding="utf-8", errors="replace") as r:
                for line in r:
                    # Ensure each line ends with \n
                    if not line.endswith("\n"):
                        line = line + "\n"
                    w.write(line)
                    lines_written += 1

            if lines_written > 0:
                start = current_line
                end = current_line + lines_written - 1
                ranges.append((bindir.name, start, end))
                current_line = end + 1

    # Atomic-ish replace
    out_path_tmp.replace(out_path)
    return out_path, ranges


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    index_lines = []
    index_lines.append("# Combined line ranges for concatenated files\n")
    index_lines.append(f"# Root: {ROOT}\n\n")

    for fname in FILENAMES:
        out_path, ranges = combine_one(fname)
        total_lines = count_lines(out_path)

        index_lines.append(f"[{fname}] -> {out_path} (total_lines={total_lines})\n")
        if not ranges:
            index_lines.append("  (no inputs found)\n\n")
            continue
        for bin_name, start, end in ranges:
            index_lines.append(f"  {bin_name}\t{start}\t{end}\n")
        index_lines.append("\n")

    with INDEX_FILE.open("w", encoding="utf-8") as w:
        w.writelines(index_lines)

    print("[DONE]")
    print(f"- Wrote combined files in: {ROOT}")
    print(f"- Wrote index: {INDEX_FILE}")


if __name__ == "__main__":
    main()
