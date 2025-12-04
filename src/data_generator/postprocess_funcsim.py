#!/usr/bin/env python3
"""
Post-process function similarity JSON files:
1. Remove functions with <= 10 instructions (in any optimization level)
2. Remove functions that appear in ALL JSON files (likely library functions)

Usage: python3 postprocess_funcsim.py /data/kun/funcsim_dataset/openssl
"""

import json
import os
import sys
from collections import defaultdict


def load_json_safe(filepath):
    """Load JSON file, return empty dict if invalid."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except:
        return {}


def save_json(filepath, data):
    """Save JSON file with indentation."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def get_max_instruction_count(func_data):
    """Get the maximum instruction count across all optimization levels."""
    max_count = 0
    for opt, instructions in func_data.items():
        if isinstance(instructions, list):
            max_count = max(max_count, len(instructions))
    return max_count


def process_folder(folder_path):
    """Process all JSON files in a folder."""
    print(f"Processing folder: {folder_path}")
    
    # Find all JSON files
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found!")
        return
    
    print(f"Found {len(json_files)} JSON files:")
    for f in json_files:
        print(f"  - {f}")
    
    # Load all JSON files
    all_data = {}
    for json_file in json_files:
        filepath = os.path.join(folder_path, json_file)
        data = load_json_safe(filepath)
        if data:
            all_data[json_file] = data
            print(f"  Loaded {json_file}: {len(data)} functions")
        else:
            print(f"  Skipped {json_file}: empty or invalid")
    
    if len(all_data) < 2:
        print("Need at least 2 valid JSON files to find common functions")
        # Still filter by instruction count
        for json_file, data in all_data.items():
            filepath = os.path.join(folder_path, json_file)
            filtered = filter_by_instruction_count(data)
            save_json(filepath, filtered)
            print(f"  Saved {json_file}: {len(filtered)} functions (filtered by instruction count only)")
        return
    
    # Find functions that appear in ALL JSON files
    all_func_names = [set(data.keys()) for data in all_data.values()]
    common_functions = set.intersection(*all_func_names) if all_func_names else set()
    
    print(f"\nFunctions appearing in ALL {len(all_data)} JSON files: {len(common_functions)}")
    if common_functions:
        print("  Sample common functions (first 20):")
        for func in list(common_functions)[:20]:
            print(f"    - {func}")
    
    # Process each JSON file
    stats = defaultdict(lambda: {'original': 0, 'removed_short': 0, 'removed_common': 0, 'final': 0})
    
    for json_file, data in all_data.items():
        filepath = os.path.join(folder_path, json_file)
        stats[json_file]['original'] = len(data)
        
        filtered_data = {}
        for func_name, func_data in data.items():
            # Skip functions with <= 10 instructions
            max_inst = get_max_instruction_count(func_data)
            if max_inst <= 10:
                stats[json_file]['removed_short'] += 1
                continue
            
            # Skip functions that appear in ALL JSON files
            if func_name in common_functions:
                stats[json_file]['removed_common'] += 1
                continue
            
            filtered_data[func_name] = func_data
        
        stats[json_file]['final'] = len(filtered_data)
        
        # Save filtered data
        save_json(filepath, filtered_data)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"{'JSON File':<30} {'Original':>10} {'Short':>10} {'Common':>10} {'Final':>10}")
    print("-" * 70)
    
    for json_file in sorted(stats.keys()):
        s = stats[json_file]
        print(f"{json_file:<30} {s['original']:>10} {s['removed_short']:>10} {s['removed_common']:>10} {s['final']:>10}")
    
    print("-" * 70)
    total_original = sum(s['original'] for s in stats.values())
    total_short = sum(s['removed_short'] for s in stats.values())
    total_common = sum(s['removed_common'] for s in stats.values())
    total_final = sum(s['final'] for s in stats.values())
    print(f"{'TOTAL':<30} {total_original:>10} {total_short:>10} {total_common:>10} {total_final:>10}")
    
    print(f"\nRemoved {len(common_functions)} common function names from all files")


def filter_by_instruction_count(data, min_instructions=10):
    """Filter functions by minimum instruction count."""
    filtered = {}
    for func_name, func_data in data.items():
        max_inst = get_max_instruction_count(func_data)
        if max_inst > min_instructions:
            filtered[func_name] = func_data
    return filtered


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 postprocess_funcsim.py <folder_path>")
        print("Example: python3 postprocess_funcsim.py /data/kun/funcsim_dataset/openssl")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory")
        sys.exit(1)
    
    process_folder(folder_path)
