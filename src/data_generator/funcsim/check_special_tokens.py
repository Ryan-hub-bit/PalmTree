#!/usr/bin/env python3
"""
Check for special tokens in funcsim JSON files that are not in vocabulary.

This script analyzes opcode and operand tokens (excluding content within parentheses)
and reports any tokens not found in the vocabulary file.

Usage:
    python check_special_tokens.py /data/kun/funcsim_match_old2 /path/to/vocab_mapping.txt [output_file]
"""

import json
import re
import os
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime


def load_vocabulary(vocab_file):
    """Load vocabulary tokens from vocab_mapping.txt"""
    vocab_tokens = set()
    
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines until we find data rows
    in_data = False
    for line in lines:
        line = line.strip()
        # Skip empty lines and header lines
        if not line or line.startswith('=') or line.startswith('-'):
            continue
        if 'TOKEN-TO-ID' in line or 'Vocabulary size' in line:
            continue
        if line.startswith('ID') and 'Token' in line:
            in_data = True
            continue
        
        if in_data:
            # Parse data row: ID  Token  Frequency
            parts = line.split()
            if len(parts) >= 2:
                try:
                    int(parts[0])  # Verify it's a number (ID)
                    token = parts[1]
                    vocab_tokens.add(token)
                except ValueError:
                    pass
    
    return vocab_tokens


def extract_tokens_from_instruction(inst):
    """
    Extract opcode and operand tokens from instruction.
    Excludes content within parentheses (addresses, positions).
    
    Example:
        "mov(0x1234:0.5:0.5:0.5) rax address(0x5678:0.1:0.2:0.3)"
        -> ['mov', 'rax', 'address']
    """
    # Remove everything within parentheses
    cleaned = re.sub(r'\([^)]*\)', '', inst)
    
    # Extract alphanumeric tokens (opcodes, registers, keywords)
    # Also include special tokens like [ ] + - *
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[\[\]+\-\*]', cleaned)
    
    return tokens


def analyze_json_file(json_file, vocab_tokens):
    """Analyze a single JSON file for missing tokens."""
    missing_tokens = Counter()
    file_has_missing = False
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for func_name, opt_data in data.items():
            for opt_level in ['O0', 'O1', 'O2', 'O3']:
                if opt_level not in opt_data:
                    continue
                
                instructions = opt_data[opt_level]
                for inst in instructions:
                    tokens = extract_tokens_from_instruction(inst)
                    for token in tokens:
                        if token not in vocab_tokens:
                            missing_tokens[token] += 1
                            file_has_missing = True
    except Exception as e:
        print(f"[ERROR] Failed to process {json_file}: {e}")
        return None, False
    
    return missing_tokens, file_has_missing


def main():
    if len(sys.argv) < 3:
        print("Usage: python check_special_tokens.py <funcsim_dir> <vocab_file> [output_file]")
        print("Example: python check_special_tokens.py /data/kun/funcsim_match_old2 ./vocab_mapping.txt missing_tokens.txt")
        sys.exit(1)
    
    funcsim_dir = sys.argv[1]
    vocab_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "missing_tokens_report.json"
    
    # Load vocabulary
    print(f"[INFO] Loading vocabulary from {vocab_file}")
    vocab_tokens = load_vocabulary(vocab_file)
    print(f"[INFO] Loaded {len(vocab_tokens)} tokens from vocabulary")
    
    # Find all JSON files
    json_files = list(Path(funcsim_dir).glob("*/*.json"))
    print(f"[INFO] Found {len(json_files)} JSON files to analyze")
    
    # Analyze each file
    all_missing = Counter()
    files_with_missing = []
    
    for i, json_file in enumerate(json_files):
        if (i + 1) % 50 == 0:
            print(f"[INFO] Processed {i + 1}/{len(json_files)} files...")
        
        missing, has_missing = analyze_json_file(json_file, vocab_tokens)
        if missing is None:
            continue
        
        if has_missing:
            files_with_missing.append((str(json_file), missing))
            all_missing.update(missing)
    
    # Build JSON output
    output_data = {
        "metadata": {
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "source_directory": funcsim_dir,
            "vocab_file": vocab_file,
            "total_files_scanned": len(json_files),
            "files_with_issues": len(files_with_missing),
            "total_unique_missing_tokens": len(all_missing),
            "total_missing_occurrences": sum(all_missing.values())
        },
        "global_missing_tokens": dict(all_missing),
        "files_with_issues": {}
    }

    for filepath, missing in files_with_missing:
        # Use relative path for cleaner output
        try:
            rel_path = os.path.relpath(filepath, funcsim_dir)
        except ValueError:
            rel_path = filepath
            
        output_data["files_with_issues"][rel_path] = {
            "missing_tokens": list(missing.keys()),
            "counts": dict(missing)
        }
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[INFO] JSON report written to: {output_file}")
    
    # Print summary to console
    print(f"Summary:")
    print(f"  Total files analyzed: {len(json_files)}")
    print(f"  Files with missing tokens: {len(files_with_missing)}")
    print(f"  Unique missing tokens: {len(all_missing)}")
    
    return 0 if not files_with_missing else 1


if __name__ == "__main__":
    sys.exit(main())
