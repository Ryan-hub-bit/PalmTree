#!/usr/bin/env python3
"""
Post-process function similarity JSON files:
1. Remove functions that don't have all 4 optimization levels (O0, O1, O2, O3)
2. Remove functions where ANY optimization level has < 10 instructions
3. Remove functions that appear in ALL JSON files within a project (likely library functions)

Usage: python3 postprocess_funcsim.py /data/kun/funcsim_match/openssl
"""

import json
import os
import sys
from collections import defaultdict

# Required optimization levels
REQUIRED_OPT_LEVELS = {'O0', 'O1', 'O2', 'O3'}
# Minimum instruction count (functions with any opt level < this will be removed)
MIN_INSTRUCTIONS = 10


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


def has_all_opt_levels(func_data):
    """Check if function has all required optimization levels."""
    return REQUIRED_OPT_LEVELS.issubset(set(func_data.keys()))


def get_min_instruction_count(func_data):
    """Get the minimum instruction count across all optimization levels."""
    min_count = float('inf')
    for opt, instructions in func_data.items():
        if isinstance(instructions, list):
            min_count = min(min_count, len(instructions))
    return min_count if min_count != float('inf') else 0


def process_folder(folder_path):
    """Process all JSON files in a folder."""
    print(f"Processing folder: {folder_path}")
    print(f"Requirements:")
    print(f"  - Must have all opt levels: {REQUIRED_OPT_LEVELS}")
    print(f"  - Minimum instructions in ANY opt level: >= {MIN_INSTRUCTIONS}")
    print()
    
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
    
    if not all_data:
        print("No valid JSON files to process!")
        return
    
    # Find functions that appear in ALL JSON files (only if we have 2+ files)
    common_functions = set()
    if len(all_data) >= 2:
        all_func_names = [set(data.keys()) for data in all_data.values()]
        common_functions = set.intersection(*all_func_names) if all_func_names else set()
        
        print(f"\nFunctions appearing in ALL {len(all_data)} JSON files: {len(common_functions)}")
        if common_functions:
            print("  Sample common functions (first 20):")
            for func in sorted(list(common_functions))[:20]:
                print(f"    - {func}")
    else:
        print("\nOnly 1 JSON file - skipping common function removal")
    
    # Process each JSON file
    stats = defaultdict(lambda: {
        'original': 0, 
        'removed_incomplete': 0,  # Missing O0/O1/O2/O3
        'removed_short': 0,       # Any opt level < MIN_INSTRUCTIONS
        'removed_common': 0,      # Appears in all files
        'final': 0
    })
    
    for json_file, data in all_data.items():
        filepath = os.path.join(folder_path, json_file)
        stats[json_file]['original'] = len(data)
        
        filtered_data = {}
        for func_name, func_data in data.items():
            # 1. Skip functions without all optimization levels
            if not has_all_opt_levels(func_data):
                stats[json_file]['removed_incomplete'] += 1
                continue
            
            # 2. Skip functions where ANY opt level has < MIN_INSTRUCTIONS
            min_inst = get_min_instruction_count(func_data)
            if min_inst < MIN_INSTRUCTIONS:
                stats[json_file]['removed_short'] += 1
                continue
            
            # 3. Skip functions that appear in ALL JSON files
            if func_name in common_functions:
                stats[json_file]['removed_common'] += 1
                continue
            
            filtered_data[func_name] = func_data
        
        stats[json_file]['final'] = len(filtered_data)
        
        # Save filtered data
        save_json(filepath, filtered_data)
    
    # Print summary
    print("\n" + "=" * 90)
    print("Summary:")
    print("=" * 90)
    print(f"{'JSON File':<35} {'Original':>10} {'Incomplete':>12} {'Short(<9)':>12} {'Common':>10} {'Final':>10}")
    print("-" * 90)
    
    for json_file in sorted(stats.keys()):
        s = stats[json_file]
        print(f"{json_file:<35} {s['original']:>10} {s['removed_incomplete']:>12} {s['removed_short']:>12} {s['removed_common']:>10} {s['final']:>10}")
    
    print("-" * 90)
    total_original = sum(s['original'] for s in stats.values())
    total_incomplete = sum(s['removed_incomplete'] for s in stats.values())
    total_short = sum(s['removed_short'] for s in stats.values())
    total_common = sum(s['removed_common'] for s in stats.values())
    total_final = sum(s['final'] for s in stats.values())
    print(f"{'TOTAL':<35} {total_original:>10} {total_incomplete:>12} {total_short:>12} {total_common:>10} {total_final:>10}")
    
    print(f"\nRemoved {len(common_functions)} common function names from all files")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 postprocess_funcsim.py <folder_path>")
        print("Example: python3 postprocess_funcsim.py /data/kun/funcsim_match/openssl")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory")
        sys.exit(1)
    
    process_folder(folder_path)
