"""
Convert address-aware format to original PalmTree format.

Input format:  opcode(0xADDR:bnorm:fnorm:bbnorm) operands ... (tab-separated instructions)
Output format: opcode operands ... (tab-separated, addresses stripped)

PalmTree expects each line to have tab-separated instruction sequences.
"""

import re
import argparse
from tqdm import tqdm


def strip_addresses(instruction_str):
    """
    Remove address information from instruction string.
    
    Input:  lea(0x4020f0:0.25433376:0.000000:0.0000) rdi [ rel address(0x405150:0.65485123:0.000000:0) ]
    Output: lea rdi [ rel address ]
    """
    # Pattern 1: opcode(0xADDR:bnorm:fnorm:bbnorm) -> opcode
    addr_pattern = re.compile(r'(\w+)\(0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+\)')
    cleaned = addr_pattern.sub(r'\1', instruction_str)
    
    # Pattern 2: address(0xADDR:bnorm:fnorm:bbnorm) -> address
    nested_pattern = re.compile(r'address\(0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+\)')
    cleaned = nested_pattern.sub('address', cleaned)
    
    return cleaned


def convert_file(input_path, output_path):
    """Convert one file from address format to PalmTree format."""
    print(f"Converting {input_path} -> {output_path}")
    
    total_lines = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, total=total_lines, desc="Processing"):
                line = line.strip()
                if not line:
                    continue
                
                # Split by tabs to get separate instructions
                instructions = line.split('\t')
                
                # Strip addresses from each instruction
                cleaned_instructions = [strip_addresses(instr) for instr in instructions]
                
                # Join back with tabs
                output_line = '\t'.join(cleaned_instructions)
                f_out.write(output_line + '\n')
    
    print(f"Conversion complete. Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert address-format data to original PalmTree format")
    
    parser.add_argument("--input", type=str, required=True,
                       help="Input file with address format")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file for PalmTree format")
    
    args = parser.parse_args()
    
    convert_file(args.input, args.output)
    
    # Print sample
    print("\n" + "="*70)
    print("Sample output (first 3 lines):")
    print("="*70)
    with open(args.output, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            # Show first 150 chars
            display_line = line.strip()
            if len(display_line) > 150:
                display_line = display_line[:150] + "..."
            print(f"{i+1}: {display_line}")
    print("="*70)


if __name__ == "__main__":
    main()
