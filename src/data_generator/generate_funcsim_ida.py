"""
Generate Function Similarity JSON with CFG-style instruction format using IDA Pro

This script generates a JSON file containing function instructions formatted
exactly like the CFG generation script:
- opcode(addr:pos1:pos2:pos3) operands
- var_XX -> var(0xXX)
- code addresses -> address(0xXXXX:pos1:pos2:pos3)
- immediates -> imm
- Tokens are split by special characters like [ ] + * etc.

Output format:
{
    "function_name": {
        "O0": ["inst1", "inst2", ...],
        ...
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
import ida_segment
import ida_funcs

HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


class FunctionProcessor:
    """Process functions and format instructions like CFG script."""
    
    def __init__(self):
        self.min_addr = 0
        self.max_addr = 0
        self.addr_positions = {}
        self.bb_range_map = {}
        self.sections = []
        
    def initialize(self):
        """Initialize address ranges and section info."""
        for n in range(ida_segment.get_segm_qty()):
            seg = ida_segment.getnseg(n)
            if seg:
                start = seg.start_ea
                end = seg.end_ea
                name = ida_segment.get_segm_name(seg)
                self.sections.append((start, end, name))
        
        if self.sections:
            self.min_addr = min(s[0] for s in self.sections)
            self.max_addr = max(s[1] for s in self.sections)
        
        print(f"[INFO] Binary address range: {hex(self.min_addr)} - {hex(self.max_addr)}")
    
    def get_section_for_addr(self, addr):
        """Find which section an address belongs to."""
        for sec_start, sec_end, sec_name in self.sections:
            if sec_start <= addr < sec_end:
                return (sec_start, sec_end, sec_name)
        return None
    
    def format_data_address_positions(self, addr):
        """Format hierarchical positions for data/non-code addresses."""
        section_info = self.get_section_for_addr(addr)
        
        if section_info is None:
            return "2.00000000:0.00000000:0.00000000"
        
        sec_start, sec_end, sec_name = section_info
        sname = sec_name.lower()
        
        is_data_like = (".data" in sname or ".rodata" in sname or ".bss" in sname)
        
        if not is_data_like:
            return "2.00000000:0.00000000:0.00000000"
        
        if self.max_addr > self.min_addr:
            sec_in_binary = (sec_start - self.min_addr) / float(self.max_addr - self.min_addr)
            sec_in_binary = max(0.0, min(1.0, sec_in_binary))
        else:
            sec_in_binary = 0.0
        
        if sec_end > sec_start:
            addr_in_section = (addr - sec_start) / float(sec_end - sec_start)
            addr_in_section = max(0.0, min(1.0, addr_in_section))
        else:
            addr_in_section = 0.0
        
        return f"{sec_in_binary:.8f}:{addr_in_section:.8f}:0.00000000"
    
    def format_hierarchical_positions(self, func_start, func_end, bb_start, inst_addr):
        """Format hierarchical positions for code addresses."""
        if self.max_addr > self.min_addr:
            func_binary_norm = (func_start - self.min_addr) / float(self.max_addr - self.min_addr)
            func_binary_norm = max(0.0, min(1.0, func_binary_norm))
        else:
            func_binary_norm = 0.0
        
        if bb_start is not None and func_end > func_start:
            bb_function_norm = (bb_start - func_start) / float(func_end - func_start)
            bb_function_norm = max(0.0, min(1.0, bb_function_norm))
        else:
            bb_function_norm = 0.0
        
        if bb_start is not None and bb_start in self.bb_range_map:
            bb_start_addr, bb_end_addr = self.bb_range_map[bb_start]
            if bb_end_addr > bb_start_addr:
                inst_bb_norm = (inst_addr - bb_start_addr) / float(bb_end_addr - bb_start_addr)
                inst_bb_norm = max(0.0, min(1.0, inst_bb_norm))
            else:
                inst_bb_norm = 0.0
        else:
            inst_bb_norm = 0.0
        
        return f"{func_binary_norm:.8f}:{bb_function_norm:.8f}:{inst_bb_norm:.8f}"
    
    def normalize_and_format(self, ins_raw, func_start, func_end, bb_start, inst_addr):
        """
        Normalize instruction and format like CFG script.
        Split tokens by special characters, then process each token.
        
        Key rules:
        - DISP_0x... -> disp (displacement operand, not an address)
        - 0x values inside [...] are displacements, not addresses
        - 0x values outside [...] that are >= min_addr are addresses
        - var_XX -> var(0xXX)
        - arg_XX -> arg
        """
        # Split instruction into mnemonic and operands
        ins = re.sub(r"\s+", " ", ins_raw).strip()
        parts = ins.split(" ", 1)
        opcode = parts[0]
        operands_str = parts[1] if len(parts) > 1 else ""
        
        # Split operands into tokens using regex (same as CFG script)
        pieces = re.split(r"(DISP_0x[0-9A-Fa-f]+|0x[0-9A-Fa-f]+|[A-Za-z0-9_]+|\[|\]|,|:|\(|\)|\+|\-|\*)", operands_str)
        
        formatted_tokens = []
        inside_brackets = 0  # Track if we're inside [...]
        
        for tok in pieces:
            if not tok or tok.isspace():
                continue
            
            # Track bracket depth
            if tok == '[':
                inside_brackets += 1
                formatted_tokens.append(tok)
                continue
            elif tok == ']':
                inside_brackets -= 1
                formatted_tokens.append(tok)
                continue
            
            # Handle DISP_ marked tokens (displacement operands from IDA)
            if tok.startswith("DISP_0x"):
                formatted_tokens.append("disp")
                continue
            
            # Handle hex addresses (0x format)
            if tok.startswith("0x") and HEX_RE.fullmatch(tok):
                try:
                    addr_val = int(tok, 16)
                    
                    # If inside brackets [...], treat as displacement/offset, not address
                    if inside_brackets > 0:
                        formatted_tokens.append("disp")
                    # If outside brackets and >= min_addr, treat as address
                    elif addr_val >= self.min_addr:
                        if addr_val in self.addr_positions:
                            entry = self.addr_positions[addr_val]
                            _, _, _, target_bb, target_func_start, target_func_end = entry
                            pos = self.format_hierarchical_positions(
                                target_func_start, target_func_end, target_bb, addr_val
                            )
                            formatted_tokens.append(f"address({tok}:{pos})")
                        else:
                            pos = self.format_data_address_positions(addr_val)
                            formatted_tokens.append(f"address({tok}:{pos})")
                    else:
                        # Small value, treat as immediate
                        formatted_tokens.append("imm")
                except:
                    formatted_tokens.append(tok)
                continue
                
            # Handle Intel hex format (e.g., 28h, 0FFh)
            if re.match(r'^[0-9][0-9A-Fa-f]*h$', tok, re.IGNORECASE):
                try:
                    hex_val = int(tok[:-1], 16)
                    # Inside brackets = displacement
                    if inside_brackets > 0:
                        formatted_tokens.append("disp")
                    elif hex_val >= self.min_addr:
                        hex_str = hex(hex_val)
                        if hex_val in self.addr_positions:
                            entry = self.addr_positions[hex_val]
                            _, _, _, target_bb, target_func_start, target_func_end = entry
                            pos = self.format_hierarchical_positions(
                                target_func_start, target_func_end, target_bb, hex_val
                            )
                            formatted_tokens.append(f"address({hex_str}:{pos})")
                        else:
                            pos = self.format_data_address_positions(hex_val)
                            formatted_tokens.append(f"address({hex_str}:{pos})")
                    else:
                        formatted_tokens.append("imm")
                except:
                    formatted_tokens.append("imm")
                continue
                
            # Handle var_XX -> var(0xXX)
            if tok.startswith("var_"):
                var_offset = tok[4:]
                formatted_tokens.append(f"var(0x{var_offset})")
                continue
                
            # Handle arg_XX -> arg
            if tok.startswith("arg_"):
                formatted_tokens.append("arg")
                continue
                
            # Keep other tokens as-is
            formatted_tokens.append(tok)
        
        # Format instruction address
        inst_pos = self.format_hierarchical_positions(func_start, func_end, bb_start, inst_addr)
        addr_hdr = f"{hex(inst_addr)}:{inst_pos}"
        
        # Build final instruction string
        if formatted_tokens:
            ops_str = ' '.join(formatted_tokens)
            if ops_str.count('[') > ops_str.count(']'):
                ops_str = ops_str + ' ]'
            return f"{opcode}({addr_hdr}) {ops_str}"
        else:
            return f"{opcode}({addr_hdr})"
    
    def get_clean_disasm(self, ea):
        """
        Get clean disassembly from IDA.
        - Immediate values -> 'imm'
        - Displacement operands (o_displ) -> mark with 'DISP_' prefix
        - Near/far/mem addresses -> hex value
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
            
            # Handle immediate values
            if op_type == idc.o_imm:
                operands.append("imm")
            # Handle displacement operands (like [rax + 0x20]) - mark with DISP_ prefix
            elif op_type == idc.o_displ:
                # Mark the displacement value so it won't be treated as address
                if op_value != idaapi.BADADDR and op_value != 0:
                    operands.append(f"DISP_{hex(op_value)}")
                else:
                    operands.append(op)
            # Handle near/far/mem addresses - convert symbols to hex
            elif op_type in [idc.o_near, idc.o_mem, idc.o_far]:
                if op_value != idaapi.BADADDR and op_value != 0:
                    operands.append(hex(op_value))
                else:
                    operands.append(op)
            else:
                # Keep as-is, will be tokenized by normalize_and_format
                operands.append(op)
        
        if operands:
            return f"{mnem} {', '.join(operands)}"
        else:
            return mnem
    
    def build_address_map(self):
        """Build the address position map for all functions."""
        print("[INFO] Building address position map...")
        
        counter = 0
        for func_ea in idautils.Functions():
            func = ida_funcs.get_func(func_ea)
            if not func:
                continue
            
            func_start = func.start_ea
            func_end = func.end_ea
            func_name = ida_funcs.get_func_name(func_ea)
            
            try:
                flowchart = idaapi.FlowChart(func)
                for block in flowchart:
                    bb_start = block.start_ea
                    bb_end = block.end_ea
                    self.bb_range_map[bb_start] = (bb_start, bb_end)
                    
                    curr = bb_start
                    while curr < bb_end:
                        self.addr_positions[curr] = (counter, curr, func_name, bb_start, func_start, func_end)
                        counter += 1
                        next_addr = idc.next_head(curr, bb_end)
                        if next_addr == idaapi.BADADDR or next_addr <= curr:
                            break
                        curr = next_addr
            except:
                pass
        
        print(f"[INFO] Indexed {counter} instructions")
    
    def get_function_instructions(self, func_ea):
        """Get all instructions in a function formatted in CFG style."""
        func = ida_funcs.get_func(func_ea)
        if not func:
            return []
        
        func_start = func.start_ea
        func_end = func.end_ea
        
        instructions = []
        
        bb_map = {}
        try:
            flowchart = idaapi.FlowChart(func)
            for block in flowchart:
                curr = block.start_ea
                while curr < block.end_ea:
                    bb_map[curr] = block.start_ea
                    next_addr = idc.next_head(curr, block.end_ea)
                    if next_addr == idaapi.BADADDR or next_addr <= curr:
                        break
                    curr = next_addr
        except:
            pass
        
        curr = func_start
        while curr < func_end:
            bb_start = bb_map.get(curr, None)
            
            disasm = self.get_clean_disasm(curr)
            if disasm:
                inst = self.normalize_and_format(disasm, func_start, func_end, bb_start, curr)
                if inst:
                    instructions.append(inst)
            
            next_addr = idc.next_head(curr, func_end)
            if next_addr == idaapi.BADADDR or next_addr <= curr:
                break
            curr = next_addr
        
        return instructions


def process_binary():
    """Process the current binary and extract function instructions."""
    ida_auto.auto_wait()
    
    processor = FunctionProcessor()
    processor.initialize()
    processor.build_address_map()
    
    functions_data = {}
    
    for func_ea in idautils.Functions():
        func_name = ida_funcs.get_func_name(func_ea)
        if not func_name:
            continue
        
        func = ida_funcs.get_func(func_ea)
        if func and (func.flags & ida_funcs.FUNC_LIB or func.flags & ida_funcs.FUNC_THUNK):
            continue
        
        instructions = processor.get_function_instructions(func_ea)
        if instructions:
            functions_data[func_name] = instructions
    
    return functions_data


def main():
    """Main entry point when run from IDA Pro."""
    opt_level = os.getenv('OPT_LEVEL', 'O0')
    output_file = os.getenv('OUTPUT_FILE', '/tmp/funcsim_output.json')
    
    print(f"[INFO] Processing binary at optimization level: {opt_level}")
    print(f"[INFO] Output file: {output_file}")
    
    functions_data = process_binary()
    
    print(f"[INFO] Found {len(functions_data)} functions")
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = {}
    
    for func_name, instructions in functions_data.items():
        if func_name not in all_data:
            all_data[func_name] = {}
        all_data[func_name][opt_level] = instructions
    
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"[INFO] Saved {len(functions_data)} functions for {opt_level}")
    print(f"[INFO] Total functions in JSON: {len(all_data)}")
    
    idc.qexit(0)


if __name__ == '__main__':
    main()
