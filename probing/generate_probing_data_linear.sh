#!/bin/bash

# Generate Position Probing Data - Linear Relationship Testing
# 
# This script generates three separate test files:
# 1. Binary level: One instruction per line (test instruction-by-instruction linear relationship)
# 2. Function level: Longest function only (test function-level position encoding)
# 3. BB level: Longest basic block only (test BB-level position encoding)

echo "========================================================================="
echo "Generate Position Probing Data - Linear Relationship Testing"
echo "========================================================================="
echo ""

# Paths
BINARY_DIR="/home/kun/testbinary"
OUTPUT_DIR="/home/kun/Document/PalmTree/probing/test_data"
SRC_DIR="/home/kun/Document/PalmTree/src/data_generator"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Configuration:"
echo "  Binary Directory: ${BINARY_DIR}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo ""

# Check if binaries exist
if [ ! -d "${BINARY_DIR}" ]; then
    echo "ERROR: Binary directory not found: ${BINARY_DIR}"
    exit 1
fi

BINARY_COUNT=$(ls "${BINARY_DIR}" 2>/dev/null | wc -l)
if [ "${BINARY_COUNT}" -eq 0 ]; then
    echo "ERROR: No binaries found in ${BINARY_DIR}"
    exit 1
fi

echo "Found ${BINARY_COUNT} binaries:"
ls -1 "${BINARY_DIR}/"
echo ""

echo "========================================================================="
echo "Data Generation Strategy"
echo "========================================================================="
echo ""
echo "Three test files will be created:"
echo ""
echo "1. BINARY LEVEL (binary_level_test.txt)"
echo "   Format: One instruction per line with inline addresses"
echo "   Purpose: Test linear relationship instruction-by-instruction"
echo "   Structure: Each line = single instruction with (addr:bnorm:fnorm:bbnorm)"
echo ""
echo "2. FUNCTION LEVEL (function_level_test.txt)"
echo "   Format: Instructions from LONGEST function only"
echo "   Purpose: Test function-level position encoding"
echo "   Structure: One instruction per line, all from same function"
echo ""
echo "3. BB LEVEL (bb_level_test.txt)"
echo "   Format: Instructions from LONGEST basic block only"
echo "   Purpose: Test BB-level position encoding"
echo "   Structure: One instruction per line, all from same BB"
echo ""

echo "========================================================================="
echo "Processing Binaries"
echo "========================================================================="
echo ""

# We'll use Python to do the extraction with BinaryNinja
cd "${SRC_DIR}" || exit 1

# Create the Python script for position probing data generation
cat > generate_position_probing.py << 'PYTHON_SCRIPT'
from binaryninja import load
import sys
import os
import re

def normalize_position(value, min_val, max_val):
    """Normalize position to [0, 1] range."""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def extract_instruction_with_positions(func, addr, instr_text, binary_min, binary_max, func_start, func_end, bb_start, bb_end):
    """
    Format: opcode(0xADDR:bnorm:fnorm:bbnorm) operands...
    Replaces code/data addresses in operands following these rules:
    1. Only addresses >= binary_min and < 0xffffffffffff0000 are replaced
    2. Addresses below binary_min or very large (masks) are kept as hex immediates
    3. Format: address(0xHEX:bnorm:fnorm:bbnorm) for code addresses
    4. Remove all commas and ensure single space between tokens
    
    - bnorm: normalized position in entire binary [0, 1]
    - fnorm: normalized position in current function [0, 1]  
    - bbnorm: normalized position in current basic block [0, 1]
    """
    import re
    
    # Normalize positions for current instruction address
    binary_norm = normalize_position(addr, binary_min, binary_max)
    function_norm = normalize_position(addr, func_start, func_end)
    bb_norm = normalize_position(addr, bb_start, bb_end)
    
    # Remove commas from instruction text (BinaryNinja uses commas in operands)
    instr_text = instr_text.replace(',', ' ')
    
    # Add spaces around brackets to match training data format: [rax] -> [ rax ]
    # This is important because the tokenizer splits on brackets
    instr_text = instr_text.replace('[', ' [ ').replace(']', ' ] ')
    
    # Normalize multiple spaces to single space
    instr_text = ' '.join(instr_text.split())
    
    # Parse instruction (now comma-free with proper bracket spacing)
    parts = instr_text.split()
    if not parts:
        return None
    
    opcode = parts[0]
    operands = ' '.join(parts[1:]) if len(parts) > 1 else ''
    
    # Replace hex addresses in operands following cfg_address.py rules
    # Pattern to match hex addresses (0x followed by hex digits)
    hex_pattern = re.compile(r'0x[0-9a-fA-F]+')
    
    def replace_hex_addr(match):
        hex_addr_str = match.group(0)
        try:
            tgt = int(hex_addr_str, 16)
        except:
            return hex_addr_str
        
        # Filter out immediate values:
        # - Values below binary base are likely immediate constants  
        # - Very large values are likely bit masks (e.g., 0xfffffffffffffff0)
        is_immediate = False
        if tgt < binary_min:  # below binary base
            is_immediate = True
        elif tgt > 0xffffffffffff0000:  # large bit patterns/masks
            is_immediate = True
        
        # Only replace if it's a valid address (not an immediate)
        if not is_immediate and tgt >= binary_min:
            # Calculate positions for this operand address
            op_binary_norm = normalize_position(tgt, binary_min, binary_max)
            # For operand addresses, we don't have function/BB context, use 0
            return f"address({hex_addr_str}:{op_binary_norm:.8f}:0.000000:0)"
        else:
            # Keep immediates as hex strings
            return hex_addr_str
    
    operands = hex_pattern.sub(replace_hex_addr, operands)
    
    # Format with inline address for opcode
    formatted = f"{opcode}(0x{addr:x}:{binary_norm:.6f}:{function_norm:.6f}:{bb_norm:.6f})"
    if operands:
        formatted += f" {operands}"
    
    return formatted

def process_binary(binary_path, output_dir):
    """Process binary and generate three test files."""
    print(f"\nProcessing: {os.path.basename(binary_path)}")
    
    # Load binary
    bv = load(binary_path)
    if not bv:
        print(f"  ERROR: Failed to load binary")
        return False
    
    print(f"  ✓ Binary loaded")
    
    # Get binary address range
    binary_min = bv.start
    binary_max = bv.end
    
    print(f"  Binary range: 0x{binary_min:x} - 0x{binary_max:x}")
    
    # Collect all instructions for binary level
    all_instructions = []
    
    # Find longest function
    longest_func = None
    longest_func_size = 0
    
    # Find longest BB
    longest_bb = None
    longest_bb_size = 0
    longest_bb_func = None
    
    for func in bv.functions:
        func_size = func.total_bytes
        
        if func_size > longest_func_size:
            longest_func = func
            longest_func_size = func_size
        
        # Check BBs in this function
        for bb in func.basic_blocks:
            bb_size = bb.length
            if bb_size > longest_bb_size:
                longest_bb = bb
                longest_bb_size = bb_size
                longest_bb_func = func
        
        # Collect all instructions
        for bb in func.basic_blocks:
            for addr in range(bb.start, bb.end):
                # Get instruction at this address
                instr = bv.get_disassembly(addr)
                if instr and not instr.startswith('0x'):  # Skip invalid instructions
                    formatted = extract_instruction_with_positions(
                        func, addr, instr,
                        binary_min, binary_max,
                        func.start, func.start + func.total_bytes,
                        bb.start, bb.end
                    )
                    if formatted:
                        all_instructions.append((addr, formatted))
    
    # Sort by address for binary level
    all_instructions.sort(key=lambda x: x[0])
    
    print(f"  Total instructions: {len(all_instructions)}")
    print(f"  Longest function: {longest_func.name if longest_func else 'None'} ({longest_func_size} bytes)")
    bb_addr_str = f"0x{longest_bb.start:x}" if longest_bb else "None"
    print(f"  Longest BB: {bb_addr_str} ({longest_bb_size} bytes)")
    
    # 1. Write binary level (one instruction per line)
    binary_level_file = os.path.join(output_dir, f"{os.path.basename(binary_path)}_binary_level.txt")
    with open(binary_level_file, 'w') as f:
        for addr, instr in all_instructions:
            f.write(instr + '\n')
    
    print(f"  ✓ Binary level: {len(all_instructions)} instructions → {binary_level_file}")
    
    # 2. Write function level (longest function only)
    if longest_func:
        function_level_file = os.path.join(output_dir, f"{os.path.basename(binary_path)}_function_level.txt")
        func_instructions = []
        
        for bb in longest_func.basic_blocks:
            for addr in range(bb.start, bb.end):
                instr = bv.get_disassembly(addr)
                if instr and not instr.startswith('0x'):
                    formatted = extract_instruction_with_positions(
                        longest_func, addr, instr,
                        binary_min, binary_max,
                        longest_func.start, longest_func.start + longest_func.total_bytes,
                        bb.start, bb.end
                    )
                    if formatted:
                        func_instructions.append((addr, formatted))
        
        func_instructions.sort(key=lambda x: x[0])
        
        with open(function_level_file, 'w') as f:
            for addr, instr in func_instructions:
                f.write(instr + '\n')
        
        print(f"  ✓ Function level: {len(func_instructions)} instructions → {function_level_file}")
    
    # 3. Write BB level (longest BB only)
    if longest_bb:
        bb_level_file = os.path.join(output_dir, f"{os.path.basename(binary_path)}_bb_level.txt")
        bb_instructions = []
        
        for addr in range(longest_bb.start, longest_bb.end):
            instr = bv.get_disassembly(addr)
            if instr and not instr.startswith('0x'):
                formatted = extract_instruction_with_positions(
                    longest_bb_func, addr, instr,
                    binary_min, binary_max,
                    longest_bb_func.start, longest_bb_func.start + longest_bb_func.total_bytes,
                    longest_bb.start, longest_bb.end
                )
                if formatted:
                    bb_instructions.append((addr, formatted))
        
        bb_instructions.sort(key=lambda x: x[0])
        
        with open(bb_level_file, 'w') as f:
            for addr, instr in bb_instructions:
                f.write(instr + '\n')
        
        print(f"  ✓ BB level: {len(bb_instructions)} instructions → {bb_level_file}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_position_probing.py <binary_path> <output_dir>")
        sys.exit(1)
    
    binary_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(binary_path):
        print(f"ERROR: Binary not found: {binary_path}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    success = process_binary(binary_path, output_dir)
    sys.exit(0 if success else 1)
PYTHON_SCRIPT

# Process each binary
binary_files=()
function_files=()
bb_files=()

for binary_path in "${BINARY_DIR}"/*; do
    if [ ! -f "${binary_path}" ]; then
        continue
    fi
    
    binary_name=$(basename "${binary_path}")
    echo "Processing: ${binary_name}"
    
    python3 generate_position_probing.py "${binary_path}" "${OUTPUT_DIR}" 2>&1 | grep -v "^INFO"
    
    # Check outputs
    binary_file="${OUTPUT_DIR}/${binary_name}_binary_level.txt"
    function_file="${OUTPUT_DIR}/${binary_name}_function_level.txt"
    bb_file="${OUTPUT_DIR}/${binary_name}_bb_level.txt"
    
    if [ -f "${binary_file}" ]; then
        binary_files+=("${binary_file}")
    fi
    
    if [ -f "${function_file}" ]; then
        function_files+=("${function_file}")
    fi
    
    if [ -f "${bb_file}" ]; then
        bb_files+=("${bb_file}")
    fi
    
    echo ""
done

# Combine outputs - USE ONLY a52dec binary for consistent position space
echo "========================================================================="
echo "Combining Outputs (Using ONLY a52dec binary)"
echo "========================================================================="
echo ""

# Use ONLY the largest binary: a52dec__liba52.so.0.0.0
# This ensures consistent position space within one binary context
a52dec_binary="${OUTPUT_DIR}/a52dec__liba52.so.0.0.0_binary_level.txt"
a52dec_function="${OUTPUT_DIR}/a52dec__liba52.so.0.0.0_function_level.txt"
a52dec_bb="${OUTPUT_DIR}/a52dec__liba52.so.0.0.0_bb_level.txt"

if [ -f "${a52dec_binary}" ]; then
    combined_binary="${OUTPUT_DIR}/binary_level_test.txt"
    cp "${a52dec_binary}" "${combined_binary}"
    total_lines=$(wc -l < "${combined_binary}")
    echo "✓ Binary Level Test: ${total_lines} instructions → ${combined_binary}"
    echo "  (Using only a52dec binary for consistent position space)"
else
    echo "✗ a52dec binary level file not found"
fi

if [ -f "${a52dec_function}" ]; then
    combined_function="${OUTPUT_DIR}/function_level_test.txt"
    cp "${a52dec_function}" "${combined_function}"
    total_lines=$(wc -l < "${combined_function}")
    echo "✓ Function Level Test: ${total_lines} instructions → ${combined_function}"
    echo "  (Using only a52dec longest function: a52_block)"
else
    echo "✗ a52dec function level file not found"
fi

if [ -f "${a52dec_bb}" ]; then
    combined_bb="${OUTPUT_DIR}/bb_level_test.txt"
    cp "${a52dec_bb}" "${combined_bb}"
    total_lines=$(wc -l < "${combined_bb}")
    echo "✓ BB Level Test: ${total_lines} instructions → ${combined_bb}"
    echo "  (Using only a52dec longest BB)"
else
    echo "✗ a52dec BB level file not found"
fi

# Cleanup temporary Python script
rm -f generate_position_probing.py

echo ""
echo "========================================================================="
echo "Summary"
echo "========================================================================="
echo ""

if [ -f "${OUTPUT_DIR}/binary_level_test.txt" ] && \
   [ -f "${OUTPUT_DIR}/function_level_test.txt" ] && \
   [ -f "${OUTPUT_DIR}/bb_level_test.txt" ]; then
    
    echo "✓ Test data generation complete!"
    echo ""
    echo "Output files:"
    echo "  1. Binary Level:   ${OUTPUT_DIR}/binary_level_test.txt"
    echo "  2. Function Level: ${OUTPUT_DIR}/function_level_test.txt"
    echo "  3. BB Level:       ${OUTPUT_DIR}/bb_level_test.txt"
    echo ""
    echo "Data format:"
    echo "  - One instruction per line"
    echo "  - Inline address format: opcode(0xADDR:bnorm:fnorm:bbnorm) operands"
    echo "  - bnorm = binary position [0,1]"
    echo "  - fnorm = function position [0,1]"
    echo "  - bbnorm = BB position [0,1]"
    echo ""
    echo "Next step:"
    echo "  Run probing experiment with these files:"
    echo "  ./run_probing.sh"
    echo ""
else
    echo "✗ Data generation incomplete. Please check errors above."
    exit 1
fi
