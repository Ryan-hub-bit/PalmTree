#!/usr/bin/env python3
"""
Script to verify that all optimization variants (O0, O1, O2, O3, Os) exist for each binary.
"""
import os
import re
from collections import defaultdict

BINARY_DIR = "/data/kun/jtransdata/small_train"
EXTRACT_DIR = "/data/kun/jtransdata/extract"

def parse_binary_name(filename):
    """Extract base name and optimization level from binary filename"""
    # Pattern: <name>-<opt>-<hash>
    # e.g., 2bwm-git-2bwm-O0-f9579e061f6e200bc50fdae0d8f2a873
    match = re.search(r'-(O[0-3s])-([a-f0-9]{32})$', filename)
    if match:
        opt_level = match.group(1)
        hash_val = match.group(2)
        base_name = filename[:match.start()]
        return base_name, opt_level, hash_val
    return None, None, None

def get_all_binaries(path):
    """Get all binaries and group by base name"""
    binary_variants = defaultdict(dict)
    skip_extensions = {'.i64', '.idb', '.id0', '.id1', '.id2', '.nam', '.til', '.txt', '.log', '.strip'}
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.endswith(ext) for ext in skip_extensions):
                continue
            
            base_name, opt_level, hash_val = parse_binary_name(file)
            if base_name and opt_level:
                binary_variants[base_name][opt_level] = file
    
    return binary_variants

def get_processed_binaries(extract_dir):
    """Get all processed binaries from pkl files"""
    processed = set()
    
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('_extract.pkl') and file != 'saved_index.pkl':
                binary_name = file.replace('_extract.pkl', '')
                processed.add(binary_name)
    
    return processed

if __name__ == '__main__':
    print("=" * 80)
    print("Optimization Level Coverage Check")
    print("=" * 80)
    print()
    
    # Get all binaries grouped by base name
    binary_variants = get_all_binaries(BINARY_DIR)
    
    # Get all processed binaries
    processed = get_processed_binaries(EXTRACT_DIR)
    
    # Expected optimization levels
    expected_opts = {'O0', 'O1', 'O2', 'O3', 'Os'}
    
    # Track statistics
    complete_binaries = 0
    incomplete_binaries = []
    missing_variants = []
    unprocessed_variants = []
    
    for base_name, variants in sorted(binary_variants.items()):
        available_opts = set(variants.keys())
        missing_opts = expected_opts - available_opts
        
        # Check which available variants were successfully processed
        processed_opts = set()
        unprocessed_opts = set()
        
        for opt_level, filename in variants.items():
            if filename in processed:
                processed_opts.add(opt_level)
            else:
                unprocessed_opts.add(opt_level)
                unprocessed_variants.append(filename)
        
        if len(available_opts) == 5 and len(processed_opts) == 5:
            complete_binaries += 1
        else:
            incomplete_binaries.append(base_name)
            
            if missing_opts:
                missing_variants.append({
                    'base': base_name,
                    'missing': missing_opts,
                    'available': available_opts
                })
            
            # Only report unprocessed if the file exists but wasn't processed
            if unprocessed_opts:
                for opt in unprocessed_opts:
                    print(f"[UNPROCESSED] {base_name} - {opt}: {variants[opt]}")
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total unique binaries (by base name): {len(binary_variants)}")
    print(f"Binaries with all 5 variants (O0-O3,Os) processed: {complete_binaries}")
    print(f"Binaries with missing/unprocessed variants: {len(incomplete_binaries)}")
    print()
    
    if missing_variants:
        print(f"Binaries missing optimization variants in dataset: {len(missing_variants)}")
        print("Sample missing variants:")
        for item in missing_variants[:5]:
            print(f"  {item['base']}: missing {sorted(item['missing'])}, has {sorted(item['available'])}")
        print()
    
    if unprocessed_variants:
        print(f"Variants that exist but weren't processed: {len(unprocessed_variants)}")
        print("These binaries need to be reprocessed.")
        print()
        
        # Save list to file for easy reprocessing
        with open('unprocessed_variants.txt', 'w') as f:
            for filename in sorted(unprocessed_variants):
                f.write(f"{filename}\n")
        print(f"List saved to: unprocessed_variants.txt")
    
    # Calculate overall coverage
    total_expected = len(binary_variants) * 5  # Each binary should have 5 variants
    total_available = sum(len(variants) for variants in binary_variants.values())
    total_processed = len(processed)
    
    print()
    print(f"Expected total variants (all binaries × 5): {total_expected}")
    print(f"Available variants in dataset: {total_available} ({total_available*100/total_expected:.1f}%)")
    print(f"Successfully processed variants: {total_processed} ({total_processed*100/total_available:.1f}% of available)")
    print("=" * 80)
