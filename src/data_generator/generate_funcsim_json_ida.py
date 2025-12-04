"""
Generate Function Similarity JSON for IDA Pro

This script generates a JSON file containing function instructions from binaries
compiled at different optimization levels (O0, O1, O2, O3).

Output format:
{
    "function_name": {
        "O0": ["inst1", "inst2", ...],
        "O1": ["inst1", "inst2", ...],
        "O2": ["inst1", "inst2", ...],
        "O3": ["inst1", "inst2", ...]
    },
    ...
}

Instructions are in address order within each function.
"""

import sys
import os
import json
import re

# Add conda environment's site-packages to path
conda_env_path = '/home/kun/anaconda3/envs/palmtree/lib/python3.11/site-packages'
if os.path.exists(conda_env_path) and conda_env_path not in sys.path:
    sys.path.insert(0, conda_env_path)

import idautils
import idaapi
import idc
import ida_auto
import ida_segment
import ida_funcs

HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def clean_ida_disasm(ea):
    """
    Get clean disassembly preserving IDA keywords but replacing symbol names with hex addresses.
    Same normalization as cfg_hierarchical_icfg_ida.py
    """
    mnem = idc.print_insn_mnem(ea)
    if not mnem:
        return None
    
    operands = []
    for i in range(6):
        op = idc.print_operand(ea, i)
        if not op:
            break
        
        op_type = idc.get_operand_type(ea, i)
        op_value = idc.get_operand_value(ea, i)
        
        # Check if it's an immediate value
        if op_type == idc.o_imm:
            op = "imm"
        elif 'offset' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"offset {hex(op_value)}"
        elif 'short' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"short {hex(op_value)}"
        elif 'large' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"large {hex(op_value)}"
        elif any(seg in op for seg in ['cs:', 'ds:', 'es:', 'ss:', 'fs:', 'gs:']) and op_value != idaapi.BADADDR and op_value != 0:
            op = hex(op_value)
        elif op_type == idc.o_displ:
            if op_value != idaapi.BADADDR and op_value != 0:
                op = f"disp_{hex(op_value)}"
        elif op_type in [idc.o_near, idc.o_mem, idc.o_far]:
            if op_value != idaapi.BADADDR and op_value != 0:
                if not op.startswith('0x') and not op.startswith('['):
                    op = hex(op_value)
        operands.append(op)
    
    if operands:
        return f"{mnem} {', '.join(operands)}"
    else:
        return mnem


def normalize_instruction(ins_raw: str):
    """
    Normalize instruction similar to cfg_hierarchical_icfg_ida.py
    but simpler - just tokenize and handle var/arg
    """
    ins = re.sub(r"\s+", " ", ins_raw).strip()
    parts = ins.split(" ", 1)
    opcode = parts[0]
    operands_str = parts[1] if len(parts) > 1 else ""
    
    # Tokenize operands
    if operands_str:
        # Split by comma and process each operand
        operands = []
        for op in operands_str.split(","):
            op = op.strip()
            # Handle var_XX -> var(0xXX)
            if op.startswith("var_"):
                var_offset = op[4:]
                op = f"var(0x{var_offset})"
            # Handle arg_XX -> arg
            elif op.startswith("arg_"):
                op = "arg"
            # Handle disp_0xXX -> disp
            elif op.startswith("disp_"):
                op = "disp"
            operands.append(op)
        return f"{opcode} {', '.join(operands)}"
    else:
        return opcode


def get_function_instructions(func_ea):
    """
    Get all instructions in a function in address order.
    Returns list of normalized instruction strings.
    """
    func = ida_funcs.get_func(func_ea)
    if not func:
        return []
    
    instructions = []
    
    # Iterate through all addresses in the function in order
    curr = func.start_ea
    while curr < func.end_ea:
        # Get disassembly
        disasm = clean_ida_disasm(curr)
        if disasm:
            # Normalize the instruction
            norm_inst = normalize_instruction(disasm)
            instructions.append(norm_inst)
        
        # Move to next instruction
        next_addr = idc.next_head(curr, func.end_ea)
        if next_addr == idaapi.BADADDR or next_addr <= curr:
            break
        curr = next_addr
    
    return instructions


def process_binary():
    """
    Process the current binary and extract function instructions.
    Returns dict of {function_name: [instructions]}
    """
    # Wait for auto-analysis
    ida_auto.auto_wait()
    
    functions_data = {}
    
    # Iterate all functions
    for func_ea in idautils.Functions():
        func_name = ida_funcs.get_func_name(func_ea)
        if not func_name:
            continue
        
        # Skip library/thunk functions
        func = ida_funcs.get_func(func_ea)
        if func and (func.flags & ida_funcs.FUNC_LIB or func.flags & ida_funcs.FUNC_THUNK):
            continue
        
        # Get instructions
        instructions = get_function_instructions(func_ea)
        if instructions:
            functions_data[func_name] = instructions
    
    return functions_data


def main():
    """Main entry point when run from IDA Pro."""
    # Get parameters from environment
    opt_level = os.getenv('OPT_LEVEL', 'O0')  # O0, O1, O2, O3
    output_file = os.getenv('OUTPUT_FILE', '/tmp/funcsim_output.json')
    
    print(f"[INFO] Processing binary at optimization level: {opt_level}")
    print(f"[INFO] Output file: {output_file}")
    
    # Process the binary
    functions_data = process_binary()
    
    print(f"[INFO] Found {len(functions_data)} functions")
    
    # Load existing JSON if it exists, otherwise create new
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = {}
    
    # Update with current optimization level's data
    for func_name, instructions in functions_data.items():
        if func_name not in all_data:
            all_data[func_name] = {}
        all_data[func_name][opt_level] = instructions
    
    # Save updated JSON
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"[INFO] Saved {len(functions_data)} functions for {opt_level}")
    print(f"[INFO] Total functions in JSON: {len(all_data)}")
    
    # Exit IDA
    idc.qexit(0)


if __name__ == '__main__':
    main()
