#!/usr/bin/env python3
"""
Combine CFG files from train/val/test directories into single files.
Samples lines proportionally from each file to achieve 0.8:0.1:0.1 ratio for train:val:test.
"""

import os
import random
from pathlib import Path
from tqdm import tqdm


def collect_all_files(base_dir):
    """Collect all *_cfg_8_inline.txt files from a directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"[WARNING] Directory not found: {base_dir}")
        return []
    
    files = list(base_path.glob("*_cfg_8_inline.txt"))
    print(f"[INFO] Found {len(files)} files in {base_dir}")
    return files


def count_lines_in_file(filepath):
    """Count total lines in a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def sample_lines_from_file(filepath, sample_ratio, chunk_size=100):
    """
    Sample lines from a file with specified ratio using chunked random sampling.
    Divides file into chunks and randomly samples from each chunk proportionally.
    
    Args:
        filepath: Path to the file
        sample_ratio: Ratio of lines to sample (e.g., 0.03 for 3%)
        chunk_size: Size of each chunk (default: 100 lines)
    
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
            sampled_lines.extend([chunk[i] for i in sorted(sampled_indices)])
    
    return sampled_lines


def write_lines(lines, output_file):
    """Write lines to output file."""
    print(f"[INFO] Writing {len(lines):,} lines to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in tqdm(lines, desc="Writing"):
            f.write(line)
    print(f"[DONE] {output_file}")


def main():
    # Directory paths
    train_dir = "/work/kliu14/txtdataset/train_cdfg"
    val_dir = "/work/kliu14/txtdataset/val_cdfg"
    test_dir = "/work/kliu14/txtdataset/test_cdfg"
    
    # Output paths
    output_dir = "/data/kun/dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    train_output = os.path.join(output_dir, "train_cfg.txt")
    val_output = os.path.join(output_dir, "val_cfg.txt")
    test_output = os.path.join(output_dir, "test_cfg.txt")
    
    # Target final output ratio
    TARGET_TRAIN_RATIO = 0.8
    TARGET_VAL_RATIO = 0.1
    TARGET_TEST_RATIO = 0.1
    
    # Train sampling ratio (fixed at 3%)
    TRAIN_SAMPLE_RATIO = 0.03
    
    print("="*70)
    print("CFG File Combiner")
    print("="*70)
    print(f"Input directories:")
    print(f"  - Train: {train_dir}")
    print(f"  - Val:   {val_dir}")
    print(f"  - Test:  {test_dir}")
    print(f"\nOutput files:")
    print(f"  - Train: {train_output}")
    print(f"  - Val:   {val_output}")
    print(f"  - Test:  {test_output}")
    print(f"\nTarget final ratio: {TARGET_TRAIN_RATIO:.1%}:{TARGET_VAL_RATIO:.1%}:{TARGET_TEST_RATIO:.1%}")
    print("="*70)
    
    # Step 1: Collect all files from each directory
    print("\n[STEP 1] Collecting files from each directory...")
    train_files = collect_all_files(train_dir)
    val_files = collect_all_files(val_dir)
    test_files = collect_all_files(test_dir)
    
    if len(train_files) == 0 and len(val_files) == 0 and len(test_files) == 0:
        print("[ERROR] No files found!")
        return
    
    print(f"\n[INFO] Files collected:")
    print(f"  - train_cdfg: {len(train_files)} files")
    print(f"  - val_cdfg:   {len(val_files)} files")
    print(f"  - test_cdfg:  {len(test_files)} files")
    
    # Step 2: Count total lines in each directory
    print("\n[STEP 2] Counting total lines in each directory...")
    
    train_total = sum(count_lines_in_file(f) for f in tqdm(train_files, desc="train_cdfg"))
    val_total = sum(count_lines_in_file(f) for f in tqdm(val_files, desc="val_cdfg"))
    test_total = sum(count_lines_in_file(f) for f in tqdm(test_files, desc="test_cdfg"))
    
    print(f"\n[INFO] Total lines per directory:")
    print(f"  - train_cdfg: {train_total:,} lines")
    print(f"  - val_cdfg:   {val_total:,} lines")
    print(f"  - test_cdfg:  {test_total:,} lines")
    print(f"  - Grand total: {train_total + val_total + test_total:,} lines")
    
    # Step 3: Calculate required sampling ratios for val and test
    print(f"\n[STEP 3] Calculating sampling ratios...")
    
    # Train will sample 3% from train_cdfg files
    expected_train_lines = int(train_total * TRAIN_SAMPLE_RATIO)
    
    # To achieve 0.8:0.1:0.1 ratio, we need:
    # train : val : test = 0.8 : 0.1 : 0.1
    # So: val = train * 0.1/0.8, test = train * 0.1/0.8
    expected_val_lines = int(expected_train_lines * TARGET_VAL_RATIO / TARGET_TRAIN_RATIO)
    expected_test_lines = int(expected_train_lines * TARGET_TEST_RATIO / TARGET_TRAIN_RATIO)
    
    # Calculate sampling ratios for val and test
    val_sample_ratio = expected_val_lines / val_total if val_total > 0 else 0
    test_sample_ratio = expected_test_lines / test_total if test_total > 0 else 0
    
    # Ensure ratios don't exceed 1.0
    val_sample_ratio = min(val_sample_ratio, 1.0)
    test_sample_ratio = min(test_sample_ratio, 1.0)
    
    print(f"\n[INFO] Calculated sampling ratios:")
    print(f"  - train_cdfg: {TRAIN_SAMPLE_RATIO:.2%} → ~{expected_train_lines:,} lines")
    print(f"  - val_cdfg:   {val_sample_ratio:.2%} → ~{expected_val_lines:,} lines")
    print(f"  - test_cdfg:  {test_sample_ratio:.2%} → ~{expected_test_lines:,} lines")
    print(f"  - Expected ratio: {expected_train_lines/(expected_train_lines+expected_val_lines+expected_test_lines):.1%}:{expected_val_lines/(expected_train_lines+expected_val_lines+expected_test_lines):.1%}:{expected_test_lines/(expected_train_lines+expected_val_lines+expected_test_lines):.1%}")
    
    # Step 4: Sample from each directory SEPARATELY
    print(f"\n[STEP 4] Sampling from each directory...")
    
    all_train_lines = []
    all_val_lines = []
    all_test_lines = []
    
    # Sample from train_cdfg → train_cfg.txt ONLY
    print(f"\n  Sampling from train_cdfg ({TRAIN_SAMPLE_RATIO:.2%})...")
    for filepath in tqdm(train_files, desc="train_cdfg"):
        all_train_lines.extend(sample_lines_from_file(filepath, TRAIN_SAMPLE_RATIO))
    
    # Sample from val_cdfg → val_cfg.txt ONLY
    print(f"\n  Sampling from val_cdfg ({val_sample_ratio:.2%})...")
    for filepath in tqdm(val_files, desc="val_cdfg"):
        all_val_lines.extend(sample_lines_from_file(filepath, val_sample_ratio))
    
    # Sample from test_cdfg → test_cfg.txt ONLY
    print(f"\n  Sampling from test_cdfg ({test_sample_ratio:.2%})...")
    for filepath in tqdm(test_files, desc="test_cdfg"):
        all_test_lines.extend(sample_lines_from_file(filepath, test_sample_ratio))
    
    total_sampled = len(all_train_lines) + len(all_val_lines) + len(all_test_lines)
    
    print(f"\n[INFO] Sampling complete:")
    print(f"  - Train: {len(all_train_lines):,} lines ({len(all_train_lines)/total_sampled:.1%})")
    print(f"  - Val:   {len(all_val_lines):,} lines ({len(all_val_lines)/total_sampled:.1%})")
    print(f"  - Test:  {len(all_test_lines):,} lines ({len(all_test_lines)/total_sampled:.1%})")
    print(f"  - Total: {total_sampled:,} lines")
    
    # Step 5: Write to output files
    print("\n[STEP 5] Writing output files...")
    write_lines(all_train_lines, train_output)
    write_lines(all_val_lines, val_output)
    write_lines(all_test_lines, test_output)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Input statistics:")
    print(f"  - Total files: {len(train_files) + len(val_files) + len(test_files)}")
    print(f"  - Total lines: {train_total + val_total + test_total:,}")
    print(f"\nSampling method:")
    print(f"  - train_cdfg files → train_cfg.txt ({TRAIN_SAMPLE_RATIO:.1%} sampling)")
    print(f"  - val_cdfg files → val_cfg.txt ({val_sample_ratio:.2%} sampling)")
    print(f"  - test_cdfg files → test_cfg.txt ({test_sample_ratio:.2%} sampling)")
    print(f"\nOutput files created:")
    print(f"  {train_output}")
    print(f"    Lines: {len(all_train_lines):,} ({len(all_train_lines)/total_sampled:.1%} of sampled)")
    print(f"  {val_output}")
    print(f"    Lines: {len(all_val_lines):,} ({len(all_val_lines)/total_sampled:.1%} of sampled)")
    print(f"  {test_output}")
    print(f"    Lines: {len(all_test_lines):,} ({len(all_test_lines)/total_sampled:.1%} of sampled)")
    print(f"\nSampling efficiency: {total_sampled / (train_total + val_total + test_total):.1%} of total data")
    print("="*70)
    print("[SUCCESS] All done!")


if __name__ == "__main__":
    main()
