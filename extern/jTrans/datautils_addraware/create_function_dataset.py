#!/usr/bin/env python3
"""
Create function-level dataset with smart merge and ground truth.

Outputs:
1. func_blocks.json - All functions with IDs, binary names, optimization levels
2. ground_truth.json - Mapping of same functions across optimization levels

This integrates smart_merge (top 100 symbols) with function matching.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
import subprocess


# Address parsing patterns (same as dataloader)
ADDR_PATTERN = re.compile(r'^(\S+?)\(0x([0-9a-f]+):([0-9.]+):([0-9.]+):([0-9.]+)\)', re.IGNORECASE)
NESTED_ADDR_PATTERN = re.compile(r'^(?:address|daddr)\(0x([0-9a-f]+):([0-9.]+):([0-9.]+):([0-9.]+)\)', re.IGNORECASE)
VAR_PATTERN = re.compile(r'^var\(0x([0-9a-f]+)\)', re.IGNORECASE)


def parse_filename(filepath):
    """Parse filename to extract binary name and optimization level."""
    filename = Path(filepath).name
    
    if not filename.endswith('_functions.txt'):
        return None
    
    base = filename[:-len('_functions.txt')]
    
    # Pattern: name-OptLevel-hash
    pattern = r'^(.+?)-(O[0-3sgfast]+)-([0-9a-f]{32})$'
    match = re.match(pattern, base, re.IGNORECASE)
    
    if match:
        binary_name = match.group(1)
        opt_level = match.group(2)
        file_hash = match.group(3)
        return (binary_name, opt_level, file_hash)
    
    return None


def extract_function_names_from_binary(binary_path):
    """Extract function names from non-stripped binary using nm."""
    function_names = []
    
    try:
        result = subprocess.run(
            ['nm', '-n', str(binary_path)],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                parts = line.strip().split()
                if len(parts) >= 3 and parts[1] in ['T', 't']:
                    function_names.append(parts[2])
    except Exception as e:
        print(f"Error extracting symbols from {binary_path}: {e}", file=sys.stderr)
    
    return function_names


def get_top_symbols(function_files, top_n=100, output_file=None):
    """
    Analyze all function files to find top N most common symbols.
    Save to file for consistency across runs.
    
    Returns: set of top symbol names (without dot prefix)
    """
    symbol_counter = Counter()
    symbol_pattern = re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)')
    
    print(f"\nAnalyzing symbols across {len(function_files)} files...")
    
    for fpath in function_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    # Find all .symbol patterns
                    symbols = symbol_pattern.findall(line)
                    symbol_counter.update(symbols)
        except Exception as e:
            print(f"Error reading {fpath}: {e}", file=sys.stderr)
    
    # Get top N symbols
    top_symbols_list = [(name, count) for name, count in symbol_counter.most_common(top_n)]
    top_symbols = set(name for name, count in top_symbols_list)
    
    print(f"Found {len(symbol_counter)} unique symbols")
    print(f"Top {top_n} symbols account for {sum(count for _, count in top_symbols_list)} occurrences")
    
    # Save to file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Top {top_n} symbols (ranked by frequency)\n")
            f.write(f"# Format: symbol_name count\n")
            f.write("#" + "="*68 + "\n")
            for name, count in top_symbols_list:
                f.write(f"{name} {count}\n")
        print(f"Saved top {top_n} symbols to: {output_file}")
    
    return top_symbols


def load_top_symbols(symbol_file):
    """
    Load top symbols from file.
    
    Returns: set of symbol names
    """
    top_symbols = set()
    
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
        
        print(f"Loaded {len(top_symbols)} symbols from: {symbol_file}")
    except Exception as e:
        print(f"Error loading symbols from {symbol_file}: {e}", file=sys.stderr)
    
    return top_symbols


def smart_merge_function(function_line, top_symbols):
    """
    Apply smart merge to a single function line.
    Replace non-top-100 symbols with .plt
    
    Returns: processed function line
    """
    def replace_symbol(match):
        symbol_name = match.group(1)
        if symbol_name in top_symbols:
            return match.group(0)  # Keep original .symbol
        else:
            return '.plt'  # Replace with .plt
    
    # Replace symbols not in top 100
    pattern = r'\.([a-zA-Z_][a-zA-Z0-9_]*)'
    processed = re.sub(pattern, replace_symbol, function_line)
    
    return processed


def parse_instruction(inst_text):
    """
    Parse a single instruction and extract tokens + positions.
    Same logic as dataloader_addressaware.py
    
    Returns:
        tokens: List of token strings
        positions: List of (binary_pos, function_pos, bb_pos) tuples
        var_offsets: List of var offset values (-1 for non-var tokens)
    """
    tokens = []
    positions = []
    var_offsets = []
    
    # Match opcode with address
    match = ADDR_PATTERN.match(inst_text)
    if not match:
        return [inst_text], [(-1.0, -1.0, -1.0)], [-1]
    
    opcode = match.group(1)
    binary_pos = float(match.group(3))
    function_pos = float(match.group(4))
    bb_pos = float(match.group(5))
    
    tokens.append(opcode)
    positions.append((binary_pos, function_pos, bb_pos))
    var_offsets.append(-1)
    
    # Parse operands
    operands_text = inst_text[match.end():].strip()
    if operands_text:
        for operand in operands_text.split():
            nested_match = NESTED_ADDR_PATTERN.match(operand)
            var_match = VAR_PATTERN.match(operand)
            
            if nested_match:
                nested_binary_pos = float(nested_match.group(2))
                nested_function_pos = float(nested_match.group(3))
                nested_bb_pos = float(nested_match.group(4))
                tokens.append('address')
                positions.append((nested_binary_pos, nested_function_pos, nested_bb_pos))
                var_offsets.append(-1)
            elif var_match:
                var_hex = var_match.group(1)
                var_offset_value = int(var_hex, 16)
                
                # Handle 64-bit negative offsets
                if var_offset_value > 0x7FFFFFFFFFFFFFFF:
                    var_offset_value = var_offset_value - 0x10000000000000000
                
                tokens.append('var')
                positions.append((-1.0, -1.0, -1.0))
                var_offsets.append(var_offset_value)
            else:
                tokens.append(operand)
                positions.append((-1.0, -1.0, -1.0))
                var_offsets.append(-1)
    
    return tokens, positions, var_offsets


def parse_function_line(function_line):
    """
    Parse a complete function line (tab-separated instructions).
    
    Returns:
        tokens: Space-separated token string
        binary_pos: List of binary position values
        function_pos: List of function position values
        bb_pos: List of basic block position values
        var_offsets: List of var offset values
        num_instructions: Number of tokens
    """
    instructions = [inst.strip() for inst in function_line.split('\t') if inst.strip()]
    
    all_tokens = []
    all_binary_pos = []
    all_function_pos = []
    all_bb_pos = []
    all_var_offsets = []
    
    for inst_text in instructions:
        tokens, positions, var_offsets = parse_instruction(inst_text)
        
        all_tokens.extend(tokens)
        for bp, fp, bbp in positions:
            all_binary_pos.append(bp)
            all_function_pos.append(fp)
            all_bb_pos.append(bbp)
        all_var_offsets.extend(var_offsets)
    
    # Join tokens with space
    tokens_str = ' '.join(all_tokens)
    
    return {
        'tokens': tokens_str,
        'binary_pos': all_binary_pos,
        'function_pos': all_function_pos,
        'bb_pos': all_bb_pos,
        'var_offsets': all_var_offsets,
        'num_instructions': len(all_tokens)
    }


def create_function_blocks(export_dir, binary_dir, top_symbols):
    """
    Create func_blocks.json with all functions.
    
    Returns: (func_blocks dict, mapping dict)
    """
    export_path = Path(export_dir)
    function_files = list(export_path.glob('*_functions.txt'))
    
    func_blocks = {}
    mapping = defaultdict(lambda: defaultdict(dict))
    
    func_id = 0
    
    print(f"\nProcessing {len(function_files)} function files...")
    
    # Group by binary
    by_binary = defaultdict(list)
    for fpath in function_files:
        parsed = parse_filename(fpath)
        if parsed:
            binary_name, opt_level, file_hash = parsed
            by_binary[binary_name].append({
                'path': fpath,
                'opt': opt_level,
                'hash': file_hash
            })
    
    for binary_name, files in sorted(by_binary.items()):
        print(f"\n  Processing binary: {binary_name}")
        
        # Try to get function names from non-stripped binary
        function_names = None
        if binary_dir:
            binary_path = Path(binary_dir)
            for opt_info in files:
                opt_level = opt_info['opt']
                file_hash = opt_info['hash']
                # Binary filename includes hash: {binary_name}-{opt}-{hash}
                potential_binary = binary_path / f"{binary_name}-{opt_level}-{file_hash}"
                
                if potential_binary.exists():
                    function_names = extract_function_names_from_binary(potential_binary)
                    if function_names:
                        print(f"    Found {len(function_names)} function names from {potential_binary.name}")
                        break
        
        # Process each optimization level
        for opt_info in files:
            opt = opt_info['opt']
            fpath = opt_info['path']
            
            print(f"    Processing {opt}...", end=' ')
            
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                for line_idx, line in enumerate(lines):
                    # Apply smart merge
                    processed_line = smart_merge_function(line, top_symbols)
                    
                    # Parse the function line to extract tokens and positions
                    parsed = parse_function_line(processed_line)
                    
                    # Determine function name
                    if function_names and line_idx < len(function_names):
                        func_name = function_names[line_idx]
                    else:
                        func_name = f"func_{line_idx}"
                    
                    # Create function block with parsed data
                    func_blocks[func_id] = {
                        'id': func_id,
                        'binary_name': binary_name,
                        'function_name': func_name,
                        'optimization_level': opt,
                        'file_hash': opt_info['hash'],
                        'line_index': line_idx,
                        'instructions': processed_line,  # Keep original for reference
                        'tokens': parsed['tokens'],  # Parsed token string
                        'binary_pos': parsed['binary_pos'],
                        'function_pos': parsed['function_pos'],
                        'bb_pos': parsed['bb_pos'],
                        'var_offsets': parsed['var_offsets'],
                        'num_instructions': parsed['num_instructions']
                    }
                    
                    # Store in mapping for ground truth
                    mapping[binary_name][func_name][opt] = func_id
                    
                    func_id += 1
                
                print(f"{len(lines)} functions")
                
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
    
    print(f"\nTotal functions created: {func_id}")
    
    return func_blocks, mapping


def create_ground_truth(mapping):
    """
    Create ground_truth.json with grouped format (more efficient).
    
    Format groups all optimization levels for each function:
    {
        "pairs": [
            {
                "binary_name": "...",
                "function_name": "...",  # Real function name from nm
                "O0": func_id,
                "O1": func_id,
                "O2": func_id,
                ...
            }
        ]
    }
    
    Note: data_json.py automatically converts this to pair-by-pair format
    internally for training, so this is more storage-efficient.
    """
    pairs = []
    
    for binary_name, binary_funcs in mapping.items():
        for func_name, opt_map in binary_funcs.items():
            if len(opt_map) > 1:  # Only include if exists in multiple optimization levels
                pair = {
                    'binary_name': binary_name,
                    'function_name': func_name
                }
                pair.update(opt_map)
                pairs.append(pair)
    
    ground_truth = {
        'pairs': pairs,
        'total_pairs': len(pairs)
    }
    
    # Calculate statistics
    opt_coverage = defaultdict(int)
    for pair in pairs:
        for opt in ['O0', 'O1', 'O2', 'O3', 'Os', 'Og', 'Ofast']:
            if opt in pair:
                opt_coverage[opt] += 1
    
    ground_truth['statistics'] = {
        'optimization_coverage': dict(opt_coverage)
    }
    
    return ground_truth


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create function dataset with smart merge')
    parser.add_argument('export_dir', help='Directory containing *_functions.txt files')
    parser.add_argument('output_dir', help='Output directory for func_blocks.json and ground_truth.json')
    parser.add_argument('--binary-dir', help='Directory with non-stripped binaries for function names')
    parser.add_argument('--symbol-file', help='Path to top symbols file (default: <output_dir>/top_symbols.txt)')
    
    args = parser.parse_args()
    
    export_dir = args.export_dir
    output_dir = args.output_dir
    binary_dir = args.binary_dir
    symbol_file = args.symbol_file
    
    print("="*70)
    print("Creating Function Dataset with Smart Merge")
    print("="*70)
    print(f"\nExport directory: {export_dir}")
    print(f"Output directory: {output_dir}")
    if binary_dir:
        print(f"Binary directory: {binary_dir}")
    
    export_path = Path(export_dir)
    output_path = Path(output_dir)
    function_files = list(export_path.glob('*_functions.txt'))
    
    if not function_files:
        print(f"\nError: No *_functions.txt files found in {export_dir}")
        sys.exit(1)
    
    # Step 1: Get or load top symbols
    print("\n" + "="*70)
    print("Step 1: Loading top symbols for smart merge")
    print("="*70)
    
    # Default symbol file location is output_dir/top_symbols.txt
    if symbol_file is None:
        symbol_file = output_path / 'top_symbols.txt'
    else:
        symbol_file = Path(symbol_file)
    
    print(f"Symbol file: {symbol_file}")
    
    # Check if symbol file exists
    if symbol_file.exists():
        print(f"Loading existing symbol file...")
        top_symbols = load_top_symbols(symbol_file)
        if not top_symbols:
            print("ERROR: Symbol file is empty or invalid!")
            print("Please run combine_function_files.py first to generate top_symbols.txt")
            print("Or delete the file to re-analyze.")
            sys.exit(1)
    else:
        print(f"ERROR: Symbol file not found: {symbol_file}")
        print("\nYou must generate top_symbols.txt first!")
        print("Run one of these commands:")
        print(f"  1. ./run_combine_functions.sh {export_dir} <output_dir>")
        print(f"  2. Or create it manually by analyzing all files")
        sys.exit(1)
    
    # Step 2: Create function blocks with smart merge
    print("\n" + "="*70)
    print("Step 2: Creating function blocks")
    print("="*70)
    func_blocks, mapping = create_function_blocks(export_dir, binary_dir, top_symbols)
    
    # Step 3: Create ground truth
    print("\n" + "="*70)
    print("Step 3: Creating ground truth")
    print("="*70)
    ground_truth = create_ground_truth(mapping)
    print(f"Created {ground_truth['total_pairs']} function groups")
    
    # Step 4: Save outputs
    print("\n" + "="*70)
    print("Step 4: Saving outputs")
    print("="*70)
    
    func_blocks_file = output_path / 'func_blocks_addr.json'
    ground_truth_file = output_path / 'ground_truth_addr.json'
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(func_blocks_file, 'w', encoding='utf-8') as f:
        json.dump(func_blocks, f, indent=2)
    print(f"Saved: {func_blocks_file}")
    print(f"  Total function blocks: {len(func_blocks)}")
    
    with open(ground_truth_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved: {ground_truth_file}")
    print(f"  Total function groups: {ground_truth['total_pairs']}")
    
    print("\n" + "="*70)
    print("Statistics")
    print("="*70)
    print(f"Used top {len(top_symbols)} symbols from: {symbol_file}")
    print(f"All other symbols replaced with .plt")
    print(f"\nOptimization level coverage:")
    for opt, count in sorted(ground_truth['statistics']['optimization_coverage'].items()):
        print(f"  {opt}: {count} functions")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)
    print(f"\nDataset files created:")
    print(f"  1. {func_blocks_file}")
    print(f"  2. {ground_truth_file}")
    print(f"\nSymbol vocabulary (shared): {symbol_file}")
    print(f"\nYou can now use these files for training function similarity models.")


if __name__ == '__main__':
    main()
