#!/usr/bin/env python3
"""
Simple check: verify every binary has a corresponding pkl file
"""
import os

BINARY_DIR = "/data/kun/jtransdata/small_train"
EXTRACT_DIR = "/data/kun/jtransdata/extract"

def get_all_binaries(path):
    """Get all binaries"""
    binaries = []
    skip_extensions = {'.i64', '.idb', '.id0', '.id1', '.id2', '.nam', '.til', '.txt', '.log', '.strip'}
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.endswith(ext) for ext in skip_extensions):
                continue
            binaries.append(file)
    
    return binaries

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
    print("Checking if every binary has a corresponding pkl file...")
    print()
    
    all_binaries = get_all_binaries(BINARY_DIR)
    processed = get_processed_binaries(EXTRACT_DIR)
    
    # Find missing pkl files
    missing = []
    for binary in all_binaries:
        if binary not in processed:
            missing.append(binary)
    
    print(f"Total binaries in dataset:     {len(all_binaries)}")
    print(f"Pkl files found:               {len(processed)}")
    print(f"Missing pkl files:             {len(missing)}")
    print()
    
    if len(missing) == 0:
        print("✓ SUCCESS: Every binary has a corresponding pkl file!")
    else:
        print(f"✗ FAILED: {len(missing)} binaries are missing pkl files")
        print()
        print("Missing pkl files for:")
        for binary in sorted(missing)[:20]:
            print(f"  - {binary}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        
        # Save full list
        with open('missing_pkl_files.txt', 'w') as f:
            for binary in sorted(missing):
                f.write(f"{binary}\n")
        print()
        print(f"Full list saved to: missing_pkl_files.txt")
