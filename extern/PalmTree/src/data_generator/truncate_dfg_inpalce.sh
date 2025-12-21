#!/usr/bin/env bash
set -euo pipefail

CFG="/data/kun/palmtreedata/cfg_train_2.txt"
DFG="/data/kun/palmtreedata/dfg_train_2.txt"

# Count lines in CFG
CFG_LINES=$(wc -l < "$CFG")

# Truncate DFG in place (using temp file to be safe)
TMP=$(mktemp)

head -n "$CFG_LINES" "$DFG" > "$TMP"
mv "$TMP" "$DFG"

echo "Truncated $DFG to $CFG_LINES lines (matched CFG)"
