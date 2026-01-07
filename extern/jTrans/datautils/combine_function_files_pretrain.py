#!/usr/bin/env python3
"""
Combine function export files for PRETRAINING (no train/val/test split) - BASELINE VERSION.
Combines ALL functions from all directories into a single file.
Includes smart_merge: keeps only top 100 most common symbols, replaces others with .plt
"""

import os
import random
import re
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def collect_all_files(base_dir):
    """Collect all *_functions.pkl files from a directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"[WARNING] Directory not found: {base_dir}")
        return []
    
    files = list(base_path.glob("*_functions.pkl"))
    print(f"[INFO] Found {len(files)} files in {base_dir}")
    return files


def get_top_symbols(file_list, top_n=100, output_file=None):
    """
    Analyze all files and find the top N most common symbols (dot-prefixed names).
    Pattern matches: .symbol_name (e.g., .plt, .got, .text, etc.)
    Saves to file for consistency across runs.
    
    Args:
        file_list: List of file paths to analyze
        top_n: Number of top symbols to keep (default: 100)
        output_file: Path to save top symbols (optional)
    
    Returns:
        Set of top N symbol names, compiled regex pattern
    """
    import pickle
    
    # Regex: Matches . followed by a letter or underscore, then any word characters
    pattern = re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)')
    counts = Counter()
    
    print(f"[INFO] Analyzing symbol frequencies across {len(file_list)} files...")
    for filepath in tqdm(file_list, desc="Analyzing symbols"):
        with open(filepath, 'rb') as f:
            functions = pickle.load(f)
            for func in functions:
                # func is a string with tokens separated by spaces
                counts.update(pattern.findall(func))
    
    # Get top N symbols
    top_symbols = set(symbol for symbol, _ in counts.most_common(top_n))
    
    print(f"\n[INFO] Found {len(counts)} unique symbols")
    print(f"[INFO] Keeping top {top_n} symbols, replacing {len(counts) - top_n} with .plt")
    print(f"\n[INFO] Top 10 symbols:")
    for symbol, count in counts.most_common(10):
        status = "KEEP" if symbol in top_symbols else "REPLACE"
        print(f"  .{symbol}: {count:,} occurrences [{status}]")
    
    # Save to file
    if output_file:
        with open(output_file, 'w') as f:
            for symbol, count in counts.most_common(top_n):
                f.write(f".{symbol}\t{count}\n")
        print(f"\n[INFO] Saved top {top_n} symbols to: {output_file}")
    
    return top_symbols, pattern


def load_top_symbols(symbols_file):
    """Load top symbols from file."""
    try:
        with open(symbols_file, 'r') as f:
            # Extract symbol names (without the leading .)
            top_symbols = set(line.split('\t')[0].lstrip('.') for line in f if line.strip())
        pattern = re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)')
        print(f"[INFO] Loaded {len(top_symbols)} symbols from {symbols_file}")
        return top_symbols, pattern
    except Exception as e:
        print(f"[ERROR] Failed to load symbols: {e}")
        return None, None


def smart_merge_line(line, top_symbols, pattern):
    """Replace low-frequency symbols with .plt in a single line."""
    def replace_symbol(match):
        symbol_name = match.group(1)
        if symbol_name in top_symbols:
            return match.group(0)  # Keep as-is
        else:
            return '.plt'  # Replace with .plt
    
    return pattern.sub(replace_symbol, line)


def count_functions_in_file(filepath):
    """Count functions in a pickle file."""
    import pickle
    with open(filepath, 'rb') as f:
        return len(pickle.load(f))


def sample_functions_from_file(filepath, sample_ratio, top_symbols=None, pattern=None):
    """
    Read functions from a pickle file, apply optional smart_merge, and sample.
    
    Args:
        filepath: Path to the pickle file
        sample_ratio: Ratio of functions to sample (0.0 to 1.0)
        top_symbols: Set of symbols to keep (for smart_merge)
        pattern: Compiled regex pattern (for smart_merge)
    
    Returns:
        List of sampled function strings
    """
    import pickle
    
    with open(filepath, 'rb') as f:
        functions = pickle.load(f)
    
    sampled = []
    for func in functions:
        # Apply smart_merge if enabled
        if top_symbols is not None and pattern is not None:
            func = smart_merge_line(func, top_symbols, pattern)
        
        # Sample with probability = sample_ratio
        if sample_ratio >= 1.0 or random.random() < sample_ratio:
            sampled.append(func)
    
    return sampled


def write_pickle(functions, output_path):
    """Write functions to pickle file."""
    import pickle
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(functions, f)
    print(f"[INFO] Wrote {len(functions):,} functions to {output_path}")


def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Combine function exports for pretraining (baseline)")
    parser.add_argument("--input_dir", required=True, help="Directory containing *_functions.pkl files")
    parser.add_argument("--output_dir", required=True, help="Output directory for combined file")
    parser.add_argument("--top_n", type=int, default=100, help="Number of top symbols to keep (default: 100)")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Sampling ratio (0.0-1.0, default: 1.0 for all data)")
    parser.add_argument("--enable_smart_merge", action="store_true", help="Enable smart merge (replace low-freq symbols with .plt)")
    parser.add_argument("--output_name", default="baseline_pretrain.pkl", help="Output file name (default: baseline_pretrain.pkl)")
    
    args = parser.parse_args()
    
    # Paths
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_file = os.path.join(output_dir, args.output_name)
    symbols_file = os.path.join(output_dir, "top_symbols.txt")
    
    # Configuration
    TOP_N_SYMBOLS = args.top_n
    SAMPLE_RATIO = args.sample_ratio
    ENABLE_SMART_MERGE = args.enable_smart_merge
    
    # Print configuration
    print("="*70)
    print("PRETRAINING DATA COMBINATION (BASELINE)")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Sample ratio: {SAMPLE_RATIO:.1%} (for pretraining, typically use 1.0 = all data)")
    print(f"Smart merge: {'ENABLED' if ENABLE_SMART_MERGE else 'DISABLED'}")
    if ENABLE_SMART_MERGE:
        print(f"  - Keep top {TOP_N_SYMBOLS} symbols, replace others with .plt")
    print("="*70)
    
    # Step 1: Collect all files from input directory
    print("\n[STEP 1] Collecting files from input directory...")
    all_files = collect_all_files(input_dir)
    
    if len(all_files) == 0:
        print("[ERROR] No *_functions.pkl files found in input directory!")
        sys.exit(1)
    
    print(f"\n[INFO] Files collected: {len(all_files)} files")
    
    # Step 2: Get top symbols for smart_merge (analyze ALL files together)
    top_symbols = None
    pattern = None
    if ENABLE_SMART_MERGE:
        print("\n[STEP 2] Analyzing symbols for smart_merge...")
        
        # Check if symbol file exists
        if os.path.exists(symbols_file):
            print(f"[INFO] Found existing symbol file: {symbols_file}")
            top_symbols, pattern = load_top_symbols(symbols_file)
            if top_symbols is None:
                print(f"[WARNING] Failed to load symbols, re-analyzing...")
                top_symbols, pattern = get_top_symbols(all_files, TOP_N_SYMBOLS, symbols_file)
        else:
            print(f"[INFO] No existing symbol file, analyzing and saving to: {symbols_file}")
            top_symbols, pattern = get_top_symbols(all_files, TOP_N_SYMBOLS, symbols_file)
    
    # Step 3: Count total functions
    step_num = 3 if ENABLE_SMART_MERGE else 2
    print(f"\n[STEP {step_num}] Counting total functions...")
    
    total_functions = sum(count_functions_in_file(f) for f in tqdm(all_files, desc="Counting"))
    
    print(f"\n[INFO] Total functions: {total_functions:,}")
    print(f"[INFO] Expected output (with {SAMPLE_RATIO:.1%} sampling): ~{int(total_functions * SAMPLE_RATIO):,} functions")
    
    # Step 4: Sample from ALL files and combine
    step_num = 4 if ENABLE_SMART_MERGE else 3
    print(f"\n[STEP {step_num}] Sampling and combining all files...")
    if ENABLE_SMART_MERGE:
        print(f"  Applying smart_merge: keeping top {TOP_N_SYMBOLS} symbols, replacing others with .plt")
    
    all_functions = []
    
    for filepath in tqdm(all_files, desc="Processing"):
        all_functions.extend(sample_functions_from_file(filepath, SAMPLE_RATIO, top_symbols=top_symbols, pattern=pattern))
    
    print(f"\n[INFO] Collected {len(all_functions):,} functions")
    
    # Step 5: Shuffle to mix different binaries/optimizations
    print(f"\n[STEP 5] Shuffling data...")
    random.shuffle(all_functions)
    print(f"[INFO] Data shuffled")
    
    # Step 6: Write to output file
    print(f"\n[STEP 6] Writing output file...")
    write_pickle(all_functions, output_file)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Input statistics:")
    print(f"  - Total files: {len(all_files)}")
    print(f"  - Total functions: {total_functions:,}")
    print(f"\nSampling:")
    print(f"  - Sampling ratio: {SAMPLE_RATIO:.1%}")
    print(f"  - Functions sampled: {len(all_functions):,} ({len(all_functions)/total_functions:.1%})")
    if ENABLE_SMART_MERGE:
        print(f"\nSmart merge:")
        print(f"  - Kept top {TOP_N_SYMBOLS} symbols")
        print(f"  - Replaced others with .plt")
        print(f"  - Symbol list saved to: {symbols_file}")
    print(f"\nOutput file:")
    print(f"  {output_file}")
    print(f"  Functions: {len(all_functions):,}")
    print("="*70)
    print("[SUCCESS] Pretraining data ready!")


if __name__ == "__main__":
    main()
