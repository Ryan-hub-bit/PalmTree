#!/usr/bin/env python3
"""
Combine function export files from train/val/test directories into single files.
Samples lines (functions) proportionally from each file to achieve 0.8:0.1:0.1 ratio for train:val:test.
Includes smart_merge: keeps only top 100 most common symbols, replaces others with .plt
"""

import os
import random
import re
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def collect_all_files(base_dir):
    """Collect all *_functions.txt files from a directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"[WARNING] Directory not found: {base_dir}")
        return []
    
    files = list(base_path.glob("*_functions.txt"))
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
    # Regex: Matches . followed by a letter or underscore, then any word characters
    pattern = re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)')
    counts = Counter()
    
    print(f"[INFO] Analyzing symbol frequencies across {len(file_list)} files...")
    for filepath in tqdm(file_list, desc="Analyzing symbols"):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                counts.update(pattern.findall(line))
    
    top_symbols_list = counts.most_common(top_n)
    top_symbols = set(name for name, count in top_symbols_list)
    print(f"[INFO] Found {len(counts)} unique symbols, keeping top {len(top_symbols)}")
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Top {top_n} symbols (ranked by frequency)\n")
            f.write(f"# Format: symbol_name count\n")
            f.write("#" + "="*68 + "\n")
            for name, count in top_symbols_list:
                f.write(f"{name} {count}\n")
        print(f"[INFO] Saved top {top_n} symbols to: {output_file}")
    
    return top_symbols, pattern


def load_top_symbols(symbol_file):
    """
    Load top symbols from file.
    
    Args:
        symbol_file: Path to symbol file
    
    Returns:
        Set of symbol names, compiled regex pattern
    """
    top_symbols = set()
    pattern = re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)')
    
    try:
        with open(symbol_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse: symbol_name count
                parts = line.split()
                if parts:
                    top_symbols.add(parts[0])
        
        print(f"[INFO] Loaded {len(top_symbols)} symbols from: {symbol_file}")
    except Exception as e:
        print(f"[ERROR] Failed to load symbols from {symbol_file}: {e}")
        return None, pattern
    
    return top_symbols, pattern


def count_lines_in_file(filepath):
    """Count total lines (functions) in a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def sample_lines_from_file(filepath, sample_ratio, chunk_size=100, top_symbols=None, pattern=None):
    """
    Sample lines (functions) from a file with specified ratio using chunked random sampling.
    Divides file into chunks and randomly samples from each chunk proportionally.
    
    Args:
        filepath: Path to the file
        sample_ratio: Ratio of lines to sample (e.g., 0.03 for 3%)
        chunk_size: Size of each chunk (default: 100 lines)
        top_symbols: Set of top symbols to keep (optional, for smart_merge)
        pattern: Compiled regex pattern for symbol replacement (optional, for smart_merge)
    
    Returns:
        List of sampled lines
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    if total_lines == 0:
        return []
    
    sampled_lines = []
    
    # Process file in chunks
    for chunk_start in range(0, total_lines, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_lines)
        chunk = lines[chunk_start:chunk_end]
        
        # Calculate how many lines to sample from this chunk
        chunk_sample_size = int(len(chunk) * sample_ratio)
        
        if chunk_sample_size > 0:
            # Randomly sample from this chunk
            sampled_indices = random.sample(range(len(chunk)), min(chunk_sample_size, len(chunk)))
            sampled_chunk = [chunk[i] for i in sorted(sampled_indices)]
            
            # Apply smart_merge if top_symbols is provided
            if top_symbols is not None and pattern is not None:
                def replace_func(match):
                    s = match.group(1)
                    return f".{s}" if s in top_symbols else ".plt"
                
                sampled_chunk = [pattern.sub(replace_func, line) for line in sampled_chunk]
            
            sampled_lines.extend(sampled_chunk)
    
    return sampled_lines


def write_lines(lines, output_file):
    """Write lines to output file."""
    print(f"[INFO] Writing {len(lines):,} functions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in tqdm(lines, desc="Writing"):
            f.write(line)
    print(f"[DONE] {output_file}")


def main():
    import sys
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python combine_function_files.py <input_dir> [output_dir]")
        print("\nArguments:")
        print("  input_dir  : Directory containing *_functions.txt files")
        print("  output_dir : (Optional) Output directory for train/val/test files")
        print("               Default: /data/kun/jtransdata")
        print("               Note: top_symbols.txt will be saved in input_dir")
        print("\nExample:")
        print("  python combine_function_files.py /data/kun/jtransdata/function_exports /data/kun/jtransdata")
        sys.exit(1)
    
    # Input directory (contains all *_functions.txt files)
    input_dir = sys.argv[1]
    
    # Output directory (for train/val/test files)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/data/kun/jtransdata"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths
    train_output = os.path.join(output_dir, "addr_train.txt")
    val_output = os.path.join(output_dir, "addr_val.txt")
    test_output = os.path.join(output_dir, "addr_test.txt")
    
    # Symbol file stays in input directory for consistency
    symbols_file = os.path.join(input_dir, "top_symbols.txt")
    
    # Target final output ratio
    TARGET_TRAIN_RATIO = 0.8
    TARGET_VAL_RATIO = 0.1
    TARGET_TEST_RATIO = 0.1
    
    # Train sampling ratio (can be adjusted as needed)
    TRAIN_SAMPLE_RATIO = 1.0  # Default to 100% (use all functions)
    
    # Smart merge: keep only top N symbols
    TOP_N_SYMBOLS = 100
    ENABLE_SMART_MERGE = True  # Set to False to disable symbol replacement
    
    print("="*70)
    print("Function Export File Combiner")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"\nOutput files:")
    print(f"  - Train: {train_output}")
    print(f"  - Val:   {val_output}")
    print(f"  - Test:  {test_output}")
    print(f"  - Symbols: {symbols_file}")
    print(f"\nTarget final ratio: {TARGET_TRAIN_RATIO:.1%}:{TARGET_VAL_RATIO:.1%}:{TARGET_TEST_RATIO:.1%}")
    print(f"Smart merge: {'ENABLED' if ENABLE_SMART_MERGE else 'DISABLED'}")
    if ENABLE_SMART_MERGE:
        print(f"  - Keep top {TOP_N_SYMBOLS} symbols, replace others with .plt")
    print("="*70)
    
    # Step 1: Collect all files from input directory
    print("\n[STEP 1] Collecting files from input directory...")
    all_files = collect_all_files(input_dir)
    
    if len(all_files) == 0:
        print("[ERROR] No *_functions.txt files found in input directory!")
        sys.exit(1)
    
    print(f"\n[INFO] Files collected: {len(all_files)} files")
    
    # For this simplified version, we treat all files as one pool
    # and split them into train/val/test based on sampling
    train_files = all_files
    val_files = all_files
    test_files = all_files
    
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
    
    # Step 3: Count total lines (functions) in each directory
    step_num = 3 if ENABLE_SMART_MERGE else 2
    print(f"\n[STEP {step_num}] Counting total functions in each directory...")
    
    train_total = sum(count_lines_in_file(f) for f in tqdm(train_files, desc="train_functions"))
    val_total = sum(count_lines_in_file(f) for f in tqdm(val_files, desc="val_functions"))
    test_total = sum(count_lines_in_file(f) for f in tqdm(test_files, desc="test_functions"))
    
    print(f"\n[INFO] Total functions per directory:")
    print(f"  - train_functions: {train_total:,} functions")
    print(f"  - val_functions:   {val_total:,} functions")
    print(f"  - test_functions:  {test_total:,} functions")
    print(f"  - Grand total: {train_total + val_total + test_total:,} functions")
    
    # Step 4: Calculate required sampling ratios for val and test
    step_num = 4 if ENABLE_SMART_MERGE else 3
    print(f"\n[STEP {step_num}] Calculating sampling ratios...")
    
    # Train will sample TRAIN_SAMPLE_RATIO from train_functions files
    expected_train_lines = int(train_total * TRAIN_SAMPLE_RATIO)
    
    # To achieve 0.8:0.1:0.1 ratio, we need:
    # train : val : test = 0.8 : 0.1 : 0.1
    # So: val = train * 0.1/0.8, test = train * 0.1/0.8
    expected_val_lines = int(expected_train_lines * TARGET_VAL_RATIO / TARGET_TRAIN_RATIO)
    expected_test_lines = int(expected_train_lines * TARGET_TEST_RATIO / TARGET_TRAIN_RATIO)
    
    # Calculate sampling ratios for val and test
    val_sample_ratio = expected_val_lines / val_total if val_total > 0 else 0
    test_sample_ratio = expected_test_lines / test_total if test_total > 0 else 0
    
    # Check if we have enough data in val/test directories
    actual_val_lines = min(expected_val_lines, val_total)
    actual_test_lines = min(expected_test_lines, test_total)
    
    # If val or test doesn't have enough lines, we need to adjust train to maintain ratio
    if val_sample_ratio > 1.0 or test_sample_ratio > 1.0:
        print(f"\n[WARNING] Not enough data in val/test directories!")
        print(f"  - Val needed: {expected_val_lines:,}, available: {val_total:,}")
        print(f"  - Test needed: {expected_test_lines:,}, available: {test_total:,}")
        
        # Use all available val/test data, then calculate train to match ratio
        actual_val_lines = val_total
        actual_test_lines = test_total
        
        # Recalculate train lines to maintain 0.8:0.1:0.1 ratio
        # train / 0.8 = val / 0.1 => train = val * 8
        # train / 0.8 = test / 0.1 => train = test * 8
        # Use the smaller constraint
        train_from_val = int(actual_val_lines * TARGET_TRAIN_RATIO / TARGET_VAL_RATIO)
        train_from_test = int(actual_test_lines * TARGET_TRAIN_RATIO / TARGET_TEST_RATIO)
        expected_train_lines = min(train_from_val, train_from_test)
        
        # Recalculate val/test to match exactly
        expected_val_lines = int(expected_train_lines * TARGET_VAL_RATIO / TARGET_TRAIN_RATIO)
        expected_test_lines = int(expected_train_lines * TARGET_TEST_RATIO / TARGET_TRAIN_RATIO)
        
        # Update sample ratios
        TRAIN_SAMPLE_RATIO = expected_train_lines / train_total
        val_sample_ratio = expected_val_lines / val_total if val_total > 0 else 0
        test_sample_ratio = expected_test_lines / test_total if test_total > 0 else 0
        
        print(f"\n[INFO] Adjusted to maintain 0.8:0.1:0.1 ratio:")
        print(f"  - Train: {expected_train_lines:,} functions ({TRAIN_SAMPLE_RATIO:.2%} sampling)")
        print(f"  - Val: {expected_val_lines:,} functions ({val_sample_ratio:.2%} sampling)")
        print(f"  - Test: {expected_test_lines:,} functions ({test_sample_ratio:.2%} sampling)")
    
    # Ensure ratios don't exceed 1.0
    val_sample_ratio = min(val_sample_ratio, 1.0)
    test_sample_ratio = min(test_sample_ratio, 1.0)
    
    print(f"\n[INFO] Calculated sampling ratios:")
    print(f"  - train_functions: {TRAIN_SAMPLE_RATIO:.2%} → ~{expected_train_lines:,} functions")
    print(f"  - val_functions:   {val_sample_ratio:.2%} → ~{expected_val_lines:,} functions")
    print(f"  - test_functions:  {test_sample_ratio:.2%} → ~{expected_test_lines:,} functions")
    print(f"  - Expected ratio: {expected_train_lines/(expected_train_lines+expected_val_lines+expected_test_lines):.1%}:{expected_val_lines/(expected_train_lines+expected_val_lines+expected_test_lines):.1%}:{expected_test_lines/(expected_train_lines+expected_val_lines+expected_test_lines):.1%}")
    
    # Step 5: Sample from each directory SEPARATELY
    step_num = 5 if ENABLE_SMART_MERGE else 4
    print(f"\n[STEP {step_num}] Sampling from each directory...")
    if ENABLE_SMART_MERGE:
        print(f"  Applying smart_merge: keeping top {TOP_N_SYMBOLS} symbols, replacing others with .plt")
    
    all_train_lines = []
    all_val_lines = []
    all_test_lines = []
    
    # Sample from train_functions → train_functions.txt ONLY
    print(f"\n  Sampling from train_functions ({TRAIN_SAMPLE_RATIO:.2%})...")
    for filepath in tqdm(train_files, desc="train_functions"):
        all_train_lines.extend(sample_lines_from_file(filepath, TRAIN_SAMPLE_RATIO, top_symbols=top_symbols, pattern=pattern))
    
    # Sample from val_functions → val_functions.txt ONLY
    print(f"\n  Sampling from val_functions ({val_sample_ratio:.2%})...")
    for filepath in tqdm(val_files, desc="val_functions"):
        all_val_lines.extend(sample_lines_from_file(filepath, val_sample_ratio, top_symbols=top_symbols, pattern=pattern))
    
    # Sample from test_functions → test_functions.txt ONLY
    print(f"\n  Sampling from test_functions ({test_sample_ratio:.2%})...")
    for filepath in tqdm(test_files, desc="test_functions"):
        all_test_lines.extend(sample_lines_from_file(filepath, test_sample_ratio, top_symbols=top_symbols, pattern=pattern))
    
    total_sampled = len(all_train_lines) + len(all_val_lines) + len(all_test_lines)
    
    print(f"\n[INFO] Sampling complete:")
    print(f"  - Train: {len(all_train_lines):,} functions ({len(all_train_lines)/total_sampled:.1%})")
    print(f"  - Val:   {len(all_val_lines):,} functions ({len(all_val_lines)/total_sampled:.1%})")
    print(f"  - Test:  {len(all_test_lines):,} functions ({len(all_test_lines)/total_sampled:.1%})")
    print(f"  - Total: {total_sampled:,} functions")
    
    # Step 6: Write to output files
    step_num = 6 if ENABLE_SMART_MERGE else 5
    print(f"\n[STEP {step_num}] Writing output files...")
    write_lines(all_train_lines, train_output)
    write_lines(all_val_lines, val_output)
    write_lines(all_test_lines, test_output)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Input statistics:")
    print(f"  - Total files: {len(train_files) + len(val_files) + len(test_files)}")
    print(f"  - Total functions: {train_total + val_total + test_total:,}")
    print(f"\nSampling method:")
    print(f"  - train_functions files → train_functions.txt ({TRAIN_SAMPLE_RATIO:.1%} sampling)")
    print(f"  - val_functions files → val_functions.txt ({val_sample_ratio:.2%} sampling)")
    print(f"  - test_functions files → test_functions.txt ({test_sample_ratio:.2%} sampling)")
    if ENABLE_SMART_MERGE:
        print(f"\nSmart merge:")
        print(f"  - Kept top {TOP_N_SYMBOLS} symbols")
        print(f"  - Replaced others with .plt")
        print(f"  - Symbol list saved to: {os.path.join(output_dir, 'top_symbols.txt')}")
    print(f"\nOutput files created:")
    print(f"  {train_output}")
    print(f"    Functions: {len(all_train_lines):,} ({len(all_train_lines)/total_sampled:.1%} of sampled)")
    print(f"  {val_output}")
    print(f"    Functions: {len(all_val_lines):,} ({len(all_val_lines)/total_sampled:.1%} of sampled)")
    print(f"  {test_output}")
    print(f"    Functions: {len(all_test_lines):,} ({len(all_test_lines)/total_sampled:.1%} of sampled)")
    print(f"\nSampling efficiency: {total_sampled / (train_total + val_total + test_total):.1%} of total data")
    print("="*70)
    print("[SUCCESS] All done!")


if __name__ == "__main__":
    main()
