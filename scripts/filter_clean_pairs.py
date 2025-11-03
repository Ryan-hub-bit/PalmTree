#!/usr/bin/env python3
"""
Filter basic block pairs to keep only "clean" ones where addresses naturally 
fall within function boundaries.

This script removes pairs where addr_start or addr_end were capped (hit 0.0 or 1.0 
due to boundary violations). It identifies suspicious pairs by detecting addresses 
that likely crossed function boundaries.

Usage:
    python filter_clean_pairs.py input_file output_file [--violations violations_file]
    
The script works by:
1. Temporarily disabling capping in bb_flow.py
2. Regenerating uncapped data to identify violations
3. Filtering the capped data to remove those violations

Or use directly with uncapped reference:
    python filter_clean_pairs.py capped.txt filtered.txt --uncapped uncapped.txt
"""

import re
import sys
import os
import subprocess
import argparse

def parse_address_field(field):
    """Parse address field like <addr_start:0x402118:0.269055:1.0>"""
    field = field.strip('<>')
    parts = field.split(':')
    if len(parts) != 4:
        return None
    return {
        'type': parts[0],
        'hex': parts[1],
        'bin_norm': float(parts[2]),
        'func_norm': float(parts[3])
    }

def check_violation_in_line(line):
    """
    Check if line has boundary violations in uncapped data.
    Returns True if addr_start or addr_end func_norm is outside [0, 1].
    """
    addr_fields = re.findall(r'<addr_(?:start|end):[^>]+>', line)
    
    for field in addr_fields:
        parsed = parse_address_field(field)
        if parsed:
            func_norm = parsed['func_norm']
            # Check if outside [0, 1], excluding special negative markers
            if func_norm >= 0 and (func_norm > 1.0 or func_norm < 0.0):
                return True
    return False

def generate_uncapped_data(binary_path, output_path):
    """Generate uncapped data by temporarily modifying bb_flow.py"""
    script_path = 'scripts/bb_flow.py'
    
    print("Temporarily disabling capping in bb_flow.py...")
    
    # Read the file
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Comment out the capping lines
    modified = content.replace(
        'start_func_norm = max(0.0, min(1.0, start_func_norm))',
        '# start_func_norm = max(0.0, min(1.0, start_func_norm))  # TEMP DISABLED'
    ).replace(
        'end_func_norm = max(0.0, min(1.0, end_func_norm))',
        '# end_func_norm = max(0.0, min(1.0, end_func_norm))  # TEMP DISABLED'
    )
    
    # Write modified version
    with open(script_path, 'w') as f:
        f.write(modified)
    
    try:
        # Generate uncapped data
        print(f"Generating uncapped data from {binary_path}...")
        result = subprocess.run(
            ['python', script_path, binary_path, output_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error generating uncapped data: {result.stderr}")
            return False
            
        print(f"✓ Generated uncapped data: {output_path}")
        return True
        
    finally:
        # Restore original (re-enable capping)
        print("Re-enabling capping in bb_flow.py...")
        with open(script_path, 'w') as f:
            f.write(content)
        print("✓ Restored bb_flow.py")

def filter_pairs(capped_file, output_file, uncapped_file, violations_file=None):
    """Filter capped pairs using uncapped reference to identify violations"""
    
    # Read uncapped data and identify violations
    print(f"Analyzing uncapped data: {uncapped_file}")
    violation_lines = set()
    
    with open(uncapped_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if check_violation_in_line(line):
                violation_lines.add(line_num)
    
    print(f"Found {len(violation_lines)} pairs with boundary violations")
    
    # Filter capped data
    print(f"Filtering capped data: {capped_file}")
    total = 0
    kept = 0
    removed = 0
    
    with open(capped_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            vfile = open(violations_file, 'w') if violations_file else None
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                total += 1
                
                if line_num in violation_lines:
                    removed += 1
                    if vfile:
                        vfile.write(line + '\n')
                else:
                    kept += 1
                    outfile.write(line + '\n')
            
            if vfile:
                vfile.close()
    
    print(f"\n✓ Processed {total} total pairs")
    print(f"✓ Kept {kept} clean pairs ({kept/total*100:.1f}%)")
    print(f"✓ Removed {removed} violated pairs ({removed/total*100:.1f}%)")
    print(f"✓ Saved filtered pairs to: {output_file}")
    if violations_file:
        print(f"✓ Saved violations to: {violations_file}")
    
    return kept, removed

def main():
    parser = argparse.ArgumentParser(
        description='Filter BB pairs to keep only those within function boundaries',
        epilog="""
Examples:
  # Auto-generate uncapped data and filter
  python filter_clean_pairs.py input.txt output.txt --binary ~/smallbinary/atilibusb.so
  
  # Use existing uncapped data
  python filter_clean_pairs.py input.txt output.txt --uncapped uncapped.txt
        """
    )
    parser.add_argument('input_file', help='Input BB pairs file (with capping)')
    parser.add_argument('output_file', help='Output filtered BB pairs file')
    parser.add_argument('--uncapped', help='Uncapped reference file (if already generated)')
    parser.add_argument('--binary', help='Binary file to auto-generate uncapped data from')
    parser.add_argument('--violations', help='Optional file to save violated pairs')
    
    args = parser.parse_args()
    
    # Determine uncapped file
    if args.uncapped:
        uncapped_file = args.uncapped
        if not os.path.exists(uncapped_file):
            print(f"ERROR: Uncapped file not found: {uncapped_file}")
            sys.exit(1)
    elif args.binary:
        # Auto-generate uncapped data
        uncapped_file = args.input_file.replace('.txt', '_uncapped_temp.txt')
        if not generate_uncapped_data(args.binary, uncapped_file):
            print("ERROR: Failed to generate uncapped data")
            sys.exit(1)
    else:
        print("ERROR: Must provide either --uncapped or --binary")
        print("Use --uncapped to specify existing uncapped reference file")
        print("Use --binary to auto-generate uncapped data")
        sys.exit(1)
    
    # Filter pairs
    filter_pairs(args.input_file, args.output_file, uncapped_file, args.violations)
    
    # Clean up temp file if we auto-generated
    if args.binary and uncapped_file.endswith('_uncapped_temp.txt'):
        os.remove(uncapped_file)
        print(f"✓ Cleaned up temporary uncapped file")

if __name__ == '__main__':
    main()
