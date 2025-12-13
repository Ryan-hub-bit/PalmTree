#!/usr/bin/env python3
"""
Combine and deduplicate funcsim_match JSON files based on function hashes.

This script:
1. Loads all JSON files from /data/kun/funcsim_match/{project}/*.json
2. For each function, computes hash of instructions (ignoring positional info in parentheses)
3. Deduplicates: If ANY optimization level (O0-O3) of a function has appeared before, skip the entire function
4. Outputs combined deduplicated JSON to /data/kun/funcsim_match/combined_deduplicated.json

IMPORTANT - Positional Information Handling:
- Hash computation: Ignores content in parentheses for deduplication
  Example: mov(0x1234:0.5:0.3:0.2) rax rbx -> mov() rax rbx (for hash only)
- Final output: Preserves ORIGINAL instructions WITH all positional info
  Example: mov(0x1234:0.5:0.3:0.2) rax rbx (kept as-is in output JSON)

Usage:
    python combine_and_deduplicate_funcsim.py /data/kun/funcsim_match
"""

import json
import os
import sys
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple


def normalize_instruction(inst: str) -> str:
    """
    Normalize instruction by removing positional information in parentheses.
    
    Example:
        mov(0x1234:0.5:0.3:0.2) rax rbx
        -> mov() rax rbx
        
        address(0x5678:1.0:0.5:0.3)
        -> address()
        
        var(0x10)
        -> var()
    """
    # Remove content within parentheses (positional info)
    # Pattern: anything(content:with:colons) -> anything()
    normalized = re.sub(r'\([^)]*\)', '()', inst)
    return normalized


def compute_instruction_hash(instructions: List[str]) -> str:
    """
    Compute hash of instruction list after normalizing (removing positional info).
    
    IMPORTANT: This function is ONLY used for deduplication comparison.
    The original instructions (with positional info) are preserved in the final output.
    
    Args:
        instructions: List of ORIGINAL instruction strings (with positional info)
        
    Returns:
        SHA256 hash of NORMALIZED instructions (positional info removed for comparison)
    """
    normalized_insts = [normalize_instruction(inst) for inst in instructions]
    # Join with newlines for consistent hashing
    combined = '\n'.join(normalized_insts)
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def load_all_json_files(base_dir: str) -> Dict[str, Dict]:
    """
    Load all JSON files from all project subdirectories.
    
    Args:
        base_dir: Base directory containing project folders
        
    Returns:
        Dictionary mapping (project, binary_name) to function data
    """
    all_data = {}
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"[ERROR] Directory not found: {base_dir}")
        return all_data
    
    # Iterate through all project directories
    for project_dir in sorted(base_path.iterdir()):
        if not project_dir.is_dir():
            continue
            
        project_name = project_dir.name
        print(f"[INFO] Loading project: {project_name}")
        
        # Load all JSON files in this project
        json_files = list(project_dir.glob("*.json"))
        print(f"[INFO]   Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            binary_name = json_file.stem  # filename without .json
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                key = (project_name, binary_name)
                all_data[key] = data
                print(f"[INFO]   Loaded {binary_name}: {len(data)} functions")
                
            except Exception as e:
                print(f"[WARNING] Failed to load {json_file}: {e}")
    
    return all_data


def deduplicate_functions(all_data: Dict[Tuple[str, str], Dict]) -> Dict:
    """
    Deduplicate functions based on instruction hashes.
    
    If ANY optimization level (O0-O3) of a function has the same hash as a previously
    seen function, the ENTIRE function (all 4 opt levels) is skipped.
    
    Args:
        all_data: Dictionary mapping (project, binary) to function data
        
    Returns:
        Dictionary of deduplicated functions with metadata
    """
    seen_hashes: Set[str] = set()
    deduplicated = {}
    
    total_functions = 0
    kept_functions = 0
    removed_functions = 0
    
    opt_levels = ['O0', 'O1', 'O2', 'O3']
    
    # Process each project and binary
    for (project_name, binary_name), func_data in all_data.items():
        print(f"\n[INFO] Processing {project_name}/{binary_name}...")
        
        for func_name, opt_data in func_data.items():
            total_functions += 1
            
            # Check if function has all 4 optimization levels
            if not all(opt in opt_data for opt in opt_levels):
                print(f"[WARNING] Skipping {func_name}: missing optimization levels")
                removed_functions += 1
                continue
            
            # Compute hashes for each optimization level
            hashes = {}
            for opt in opt_levels:
                instructions = opt_data[opt]
                inst_hash = compute_instruction_hash(instructions)
                hashes[opt] = inst_hash
            
            # Check if ANY of the optimization levels has been seen before
            duplicate_found = False
            for opt, inst_hash in hashes.items():
                if inst_hash in seen_hashes:
                    duplicate_found = True
                    print(f"[DEDUP] Removing {func_name}: {opt} hash already seen")
                    break
            
            if duplicate_found:
                removed_functions += 1
                continue
            
            # Not a duplicate - add all hashes to seen set
            for inst_hash in hashes.values():
                seen_hashes.add(inst_hash)
            
            # Keep this function
            unique_key = f"{project_name}::{binary_name}::{func_name}"
            deduplicated[unique_key] = {
                'project': project_name,
                'binary': binary_name,
                'function_name': func_name,
                'opt_data': opt_data,
                'hashes': hashes  # Store hashes for reference
            }
            kept_functions += 1
    
    print(f"\n[SUMMARY] Deduplication complete:")
    print(f"  Total functions processed: {total_functions}")
    print(f"  Functions kept: {kept_functions}")
    print(f"  Functions removed: {removed_functions}")
    print(f"  Deduplication rate: {removed_functions/total_functions*100:.2f}%")
    
    return deduplicated


def save_output(deduplicated: Dict, output_file: str):
    """
    Save deduplicated data to JSON file.
    
    IMPORTANT: Save ORIGINAL instructions with positional info intact.
    The hash computation (for deduplication) ignores positional info,
    but the final output preserves it.
    
    Output format:
    {
      "project::binary::function_name": {
        "project": "openssl",
        "binary": "openssl",
        "function_name": "main",
        "O0": [...],  # ORIGINAL instructions with (0xaddr:pos1:pos2:pos3)
        "O1": [...],  # ORIGINAL instructions preserved
        "O2": [...],
        "O3": [...]
      }
    }
    """
    output_data = {}
    
    for key, data in deduplicated.items():
        # Use ORIGINAL opt_data which contains instructions WITH positional info
        output_data[key] = {
            'project': data['project'],
            'binary': data['binary'],
            'function_name': data['function_name'],
            'O0': data['opt_data']['O0'],  # Original instructions preserved
            'O1': data['opt_data']['O1'],  # Original instructions preserved
            'O2': data['opt_data']['O2'],  # Original instructions preserved
            'O3': data['opt_data']['O3']   # Original instructions preserved
        }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[DONE] Saved {len(output_data)} deduplicated functions to {output_file}")
    print(f"[INFO] All instructions contain ORIGINAL positional information (content in parentheses preserved)")
    
    # Print file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[INFO] Output file size: {file_size_mb:.2f} MB")


def main():
    if len(sys.argv) < 2:
        print("Usage: python combine_and_deduplicate_funcsim.py <funcsim_match_dir>")
        print("Example: python combine_and_deduplicate_funcsim.py /data/kun/funcsim_match")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    output_file = os.path.join(base_dir, "combined_deduplicated.json")
    
    print("=" * 80)
    print("Combine and Deduplicate FuncSim Data")
    print("=" * 80)
    print(f"Input directory: {base_dir}")
    print(f"Output file: {output_file}")
    print("=" * 80)
    
    # Load all JSON files
    print("\n[STEP 1] Loading all JSON files...")
    all_data = load_all_json_files(base_dir)
    
    if not all_data:
        print("[ERROR] No data loaded. Exiting.")
        sys.exit(1)
    
    # Deduplicate
    print("\n[STEP 2] Deduplicating functions...")
    deduplicated = deduplicate_functions(all_data)
    
    # Save output
    print("\n[STEP 3] Saving output...")
    save_output(deduplicated, output_file)
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
