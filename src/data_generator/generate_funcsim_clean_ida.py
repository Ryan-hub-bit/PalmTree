#!/usr/bin/env python3
"""
Generate Function Similarity JSON with CLEAN instruction format using IDA Pro

This script generates a JSON file containing raw, unmodified instructions:
- NO address position encoding
- NO var/address/imm transformations
- Just clean disassembly from IDA

Output format:
{
    "function_name": {
        "O0": ["mov rax, rbx", "add rax, 0x10", ...],
        "O1": [...],
        "O2": [...],
        "O3": [...]
    }
}
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
import ida_funcs


class CleanFunctionProcessor:
    
    def __init__(self):
        pass
    
    def get_clean_disasm(self, ea):
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
            
            if mnem.lower() in ['call', 'jmp', 'je', 'jne', 'jz', 'jnz', 'ja', 'jb', 'jg', 'jl', 
                                'jge', 'jle', 'jae', 'jbe', 'jo', 'jno', 'js', 'jns', 'jc', 'jnc',
                                'jp', 'jnp', 'jpe', 'jpo', 'jcxz', 'jecxz', 'jrcxz', 'loop', 
                                'loope', 'loopne', 'loopnz', 'loopz']:
                if op_type in [idc.o_near, idc.o_far] and op_value != idaapi.BADADDR:
                    operands.append(hex(op_value))
                else:
                    operands.append(op)
            else:
                if ';' in op:
                    op = op.split(';')[0].strip()
                operands.append(op)
        
        if operands:
            return f"{mnem} {', '.join(operands)}"
        else:
            return mnem
    
    def get_basic_blocks_ida(self, func_ea):
        """Get basic blocks for a function."""
        func = ida_funcs.get_func(func_ea)
        if not func:
            return []
        
        flowchart = idaapi.FlowChart(func)
        blocks = [(block.start_ea, block.end_ea) for block in flowchart]
        return blocks
    
    def process_function(self, func_ea):
        """
        Process a single function and return list of clean instructions.
        
        Returns:
            list: List of instruction strings (clean disassembly)
        """
        func = ida_funcs.get_func(func_ea)
        if not func:
            return []
        
        instructions = []
        bbs = self.get_basic_blocks_ida(func_ea)
        
        for bb_start, bb_end in bbs:
            curr = bb_start
            while curr < bb_end:
                disasm = self.get_clean_disasm(curr)
                if disasm:
                    instructions.append(disasm)
                
                # Move to next instruction
                curr = idc.next_head(curr, bb_end)
                if curr == idaapi.BADADDR or curr >= bb_end:
                    break
        
        return instructions
    
    def process_all_functions(self):
        """
        Process all functions in the binary.
        
        Returns:
            dict: {function_name: instruction_list}
        """
        print("[INFO] Processing all functions...")
        
        result = {}
        all_funcs = list(idautils.Functions())
        
        for idx, func_ea in enumerate(all_funcs):
            if idx % 100 == 0:
                print(f"[INFO] Progress: {idx}/{len(all_funcs)} functions")
            
            func_name = ida_funcs.get_func_name(func_ea)
            if not func_name:
                continue
            
            instructions = self.process_function(func_ea)
            
            if instructions:
                result[func_name] = instructions
        
        print(f"[INFO] Processed {len(result)} functions with instructions")
        return result


def extract_opt_level_from_path(binary_path):
    """
    Extract optimization level from binary path.
    Expected format: /path/to/project_O0/binary or /path/to/binary_O0
    """
    # First try to extract from parent directory name
    parent_dir = os.path.basename(os.path.dirname(binary_path))
    for opt in ['O0', 'O1', 'O2', 'O3']:
        if f'_{opt}' in parent_dir or f'_{opt.lower()}' in parent_dir:
            return opt
    
    # Then try from binary name itself
    binary_name = os.path.basename(binary_path)
    for opt in ['O0', 'O1', 'O2', 'O3']:
        if opt in binary_name or opt.lower() in binary_name:
            return opt
    
    # Default to O0 if not found
    print(f"[WARNING] Could not detect optimization level from {binary_path}, defaulting to O0")
    return "O0"


def main():
    """Main entry point when run from IDA Pro."""
    # Get output directory from environment or use default
    out_dir = os.getenv('OUTPUT_DIR', '/home/kun/Document/PalmTree/src/data_generator/funcsim_output')
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup logging to file
    log_file = os.path.join(out_dir, 'ida_funcsim_clean.log')
    log_f = open(log_file, 'a')
    sys.stdout = log_f
    sys.stderr = log_f
    
    print("\n" + "="*70)
    print("[INFO] Clean Function Similarity Extraction Started")
    print("="*70)
    
    # Wait for IDA's auto-analysis to complete
    print("[INFO] Waiting for IDA auto-analysis to complete...")
    ida_auto.auto_wait()
    print("[INFO] Auto-analysis complete")
    
    # Get the input file path
    input_file = idc.get_input_file_path()
    binary_name = os.path.basename(input_file)
    print(f"[INFO] Processing: {input_file}")
    
    # Extract optimization level
    opt_level = extract_opt_level_from_path(input_file)
    print(f"[INFO] Detected optimization level: {opt_level}")
    
    # Determine output JSON file name based on project/binary
    # Remove _O0, _O1, etc. suffix to get base name
    base_name = binary_name
    for opt in ['_O0', '_O1', '_O2', '_O3', '_o0', '_o1', '_o2', '_o3']:
        if base_name.endswith(opt):
            base_name = base_name[:-3]
            break
    
    output_json = os.path.join(out_dir, f"{base_name}.json")
    print(f"[INFO] Output JSON: {output_json}")
    
    # Process all functions
    processor = CleanFunctionProcessor()
    func_data = processor.process_all_functions()
    
    # Load existing JSON if it exists (to merge multiple opt levels)
    if os.path.exists(output_json):
        try:
            with open(output_json, 'r') as f:
                existing_data = json.load(f)
            print(f"[INFO] Loaded existing data from {output_json}")
        except:
            existing_data = {}
            print(f"[INFO] Could not load existing JSON, creating new")
    else:
        existing_data = {}
        print(f"[INFO] Creating new JSON file")
    
    # Merge current optimization level data
    for func_name, instructions in func_data.items():
        if func_name not in existing_data:
            existing_data[func_name] = {}
        
        existing_data[func_name][opt_level] = instructions
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"[SUCCESS] Saved {len(func_data)} functions to {output_json}")
    print(f"[INFO] Optimization level: {opt_level}")
    
    # Print statistics
    total_instructions = sum(len(inst_list) for inst_list in func_data.values())
    print(f"[INFO] Total instructions: {total_instructions}")
    
    log_f.close()
    idc.qexit(0)  # Exit IDA successfully


# Run when script is executed by IDA
if __name__ == '__main__':
    main()
