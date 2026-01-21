"""
CFG Function Export with Hierarchical Position Encoding using IDA Pro

This script exports ENTIRE FUNCTIONS (no sequence chunking) with hierarchical address-aware encoding.
It should be run from within IDA Pro using -A -S flags.
IDA Pro installation: /home/kun/ida-pro-9.0
"""

# Fix Python path to include conda environment's site-packages
import sys
import os

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
import ida_bytes
import ida_ua
import ida_name
import ida_nalt

import networkx as nx
import re
from pathlib import Path

HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def normalize_and_mask(ins_raw: str, symbol_map: dict, string_map: dict):
    ins = re.sub(r"\s+", ", ", ins_raw, 1)
    parts = ins.split(", ")
    opcode = parts[0]
    operands = parts[1:] if len(parts) > 1 else []

    out_tokens = [opcode]
    mask_tokens = ["0"]

    for op in operands:
        pieces = re.split(r"(0x[0-9A-Fa-f]+|[A-Za-z0-9_]+|\[|\]|,|:|\(|\))", op)
        for tok in pieces:
            if not tok or tok.isspace():  # Skip empty and whitespace-only tokens
                continue
            if tok.startswith("0x") and bool(HEX_RE.fullmatch(tok)):
                out_tokens.append(tok)
                mask_tokens.append(tok)
            else:
                out_tokens.append(tok)
                mask_tokens.append("0")

    return " ".join(out_tokens), " ".join(mask_tokens)


def build_function_inline(func_instructions, ctx: dict):
    """
    Build entire function with HIERARCHICAL position encoding (address-based):
    
    For CODE addresses (instructions):
    - Position 1: Function's position in binary = (func_start - min_addr) / (max_addr - min_addr)
    - Position 2: Basic block's position in function = (bb_start - func_start) / (func_end - func_start)
    - Position 3: Instruction's position in basic block = (inst_addr - bb_start) / (bb_end - bb_start)
    
    For DATA addresses (not in text section):
    - Position 1: Section's position in binary = (section_start - min_addr) / (max_addr - min_addr)
    - Position 2: Address position inside section = (addr - section_start) / (section_end - section_start)
    - Position 3: BB position = 0.0 (no BB context for data)
    
    For EXTERNAL/PLT calls:
    - Use symbol name instead of address (e.g., .plt.printf, strcmp, etc.)
    """
    addr_positions = ctx.get('addr_positions', {})
    bb_range_map = ctx.get('bb_range_map', {})
    min_addr = ctx.get('min_addr', 0)
    max_addr = ctx.get('max_addr', min_addr)
    sections = ctx.get('sections', [])
    symbol_map = ctx.get('symbol_map', {})

    def get_section_for_addr(addr):
        """Find which section an address belongs to."""
        for sec_start, sec_end, sec_name in sections:
            if sec_start <= addr < sec_end:
                return (sec_start, sec_end, sec_name)
        return None
    
    def format_data_address_positions(addr):
        """
        Format hierarchical positions for addresses not in our analyzed code.
        
        For DATA sections (.data, .rodata, .bss):
          Position 1: section's position in binary = (section_start - min_addr) / (max_addr - min_addr)
          Position 2: address position inside section = (addr - section_start) / (section_end - section_start)
          Position 3: 0.0 (no BB context for data)
        
        For CODE sections (.text, .plt) - external/unanalyzed functions:
          Position 1: section's position in binary (normalized)
          Position 2: address position inside section (normalized)
          Position 3: 0.0 (no function context)
        
        For other sections (GOT, import, debug, etc.):
          return sentinel "2.00000000:0.00000000:0.00000000"
        """
        section_info = get_section_for_addr(addr)
        
        # No section at all → external/garbage/special
        if section_info is None:
            return "2.00000000:0.00000000:0.00000000"
        
        sec_start, sec_end, sec_name = section_info
        sname = sec_name.lower()
        
        # Check if it's a code section (text, plt)
        is_code_like = (
            ".text" in sname or
            ".plt" in sname or
            "text" == sname or
            "plt" == sname
        )
        
        # Check if it's a data section
        is_data_like = (
            ".data" in sname or
            ".rodata" in sname or
            ".bss" in sname or
            ".got" in sname
        )
        
        # If it's neither code nor data (import, debug, etc.), use sentinel
        if not is_code_like and not is_data_like:
            return "2.00000000:0.00000000:0.00000000"
        
        # ---- Code or data section: compute normalized positions ----
        if max_addr > min_addr:
            sec_in_binary = (sec_start - min_addr) / float(max_addr - min_addr)
            sec_in_binary = max(0.0, min(1.0, sec_in_binary))
        else:
            sec_in_binary = 0.0
        
        if sec_end > sec_start:
            addr_in_section = (addr - sec_start) / float(sec_end - sec_start)
            addr_in_section = max(0.0, min(1.0, addr_in_section))
        else:
            addr_in_section = 0.0
        
        bb_pos = 0.0
        return f"{sec_in_binary:.8f}:{addr_in_section:.8f}:{bb_pos:.8f}"

    def format_hierarchical_positions(entry):
        """
        Returns: func_in_binary:bb_in_function:inst_in_bb
        
        Position 1: (func_start - min_addr) / (max_addr - min_addr)
        Position 2: (bb_start - func_start) / (func_end - func_start)
        Position 3: (inst_addr - bb_start) / (bb_end - bb_start)
        """
        bin_counter, addr, func_name, bb_start, func_start, func_end = entry
        
        # Position 1: Function's position in binary (address-based)
        if max_addr > min_addr:
            func_binary_norm = (func_start - min_addr) / float(max_addr - min_addr)
            func_binary_norm = max(0.0, min(1.0, func_binary_norm))
        else:
            func_binary_norm = 0.0
        
        # Position 2: Basic block's position in function (address-based)
        # Handle case where bb_start is None (instruction not matched to a BB)
        if bb_start is not None and func_end > func_start:
            bb_function_norm = (bb_start - func_start) / float(func_end - func_start)
            bb_function_norm = max(0.0, min(1.0, bb_function_norm))
        else:
            bb_function_norm = 0.0
        
        # Position 3: Instruction's position in basic block (address-based)
        if bb_start is not None and bb_start in bb_range_map:
            bb_start_addr, bb_end_addr = bb_range_map[bb_start]
            if bb_end_addr > bb_start_addr:
                inst_bb_norm = (addr - bb_start_addr) / float(bb_end_addr - bb_start_addr)
                inst_bb_norm = max(0.0, min(1.0, inst_bb_norm))
            else:
                inst_bb_norm = 0.0
        else:
            inst_bb_norm = 0.0
        
        return f"{func_binary_norm:.8f}:{bb_function_norm:.8f}:{inst_bb_norm:.8f}"

    out_instrs = []
    for addr, norm_line, mask_line in func_instructions:
        tokens = norm_line.strip().split()
        masks = mask_line.strip().split()
        if not tokens:
            continue
        opcode = tokens[0]
        operands = tokens[1:]
        operand_masks = masks[1:]
        
        # Check if this is a control flow instruction (call/jmp)
        is_control_flow = opcode.lower() in ['call', 'jmp', 'jmpq', 'callq', 'ja', 'jb', 'jc', 'je', 'jg', 'jl', 'jn', 'jo', 'jp', 'js', 'jz']

        formatted_ops = []
        for i, tok in enumerate(operands):
            mk = operand_masks[i] if i < len(operand_masks) else "0"
            mk_hex = None
            
            # Check mask first (from normalize_and_mask)
            if isinstance(mk, str) and mk.startswith('0x'):
                mk_hex = mk
            # Check token for 0x prefix
            elif isinstance(tok, str) and tok.startswith('0x') and bool(HEX_RE.fullmatch(tok)):
                mk_hex = tok
            # Check token for Intel hex suffix (e.g., 1234ABCDh)
            elif isinstance(tok, str) and re.match(r'^[0-9A-Fa-f]+h$', tok, re.IGNORECASE):
                # Convert Intel hex to 0x format
                mk_hex = '0x' + tok[:-1]

            if mk_hex is not None:
                try:
                    tgt = int(mk_hex, 16)
                except Exception:
                    tgt = None

                # Filter out immediate values
                addr_threshold = max(min_addr, 0x1000)
                if tgt is not None and tgt >= addr_threshold:
                    if tgt in addr_positions:
                        # Internal function call: use hierarchical positions (func, bb, inst)
                        entry = addr_positions[tgt]
                        pos = format_hierarchical_positions(entry)
                        formatted_ops.append(f"address({mk_hex}:{pos})")
                    else:
                        # External address - check section type
                        section_info = get_section_for_addr(tgt)
                        
                        # Check if there's a symbol name for this address
                        symbol_name = None
                        if tgt in symbol_map:
                            symbol_name = symbol_map[tgt]
                        
                        if section_info:
                            sec_start, sec_end, sec_name = section_info
                            sname = sec_name.lower()
                            
                            # Check if it's a code section (PLT/text) - external function
                            is_code_section = (
                                ".plt" in sname or
                                "plt" == sname or
                                ".text" in sname or
                                "text" == sname or
                                ".extern" in sname
                            )
                            
                            # Check if it's a data section
                            is_data_section = (
                                ".data" in sname or
                                ".rodata" in sname or
                                ".bss" in sname or
                                ".got" in sname
                            )
                            
                            # For code sections (PLT/external functions), use symbol name
                            if is_code_section and symbol_name:
                                sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                formatted_ops.append(sym)
                            # For data sections
                            elif is_data_section:
                                # If it's a control flow instruction (call/jmp), don't use daddr()
                                # This handles function pointers in data sections
                                if is_control_flow and symbol_name:
                                    # Function pointer - use symbol name
                                    sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                    formatted_ops.append(sym)
                                elif is_control_flow:
                                    # Function pointer without symbol - use plain address
                                    formatted_ops.append(mk_hex)
                                else:
                                    # Regular data reference - use daddr()
                                    pos = format_data_address_positions(tgt)
                                    formatted_ops.append(f"daddr({mk_hex}:{pos})")
                            # Other sections with symbols (external but not clearly code/data)
                            elif symbol_name:
                                sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                formatted_ops.append(sym)
                            else:
                                # No symbol, use daddr with positions
                                pos = format_data_address_positions(tgt)
                                formatted_ops.append(f"daddr({mk_hex}:{pos})")
                        else:
                            # No section info - use symbol if available
                            if symbol_name:
                                sym = symbol_name if symbol_name.startswith('.') else f'.{symbol_name}'
                                formatted_ops.append(sym)
                            else:
                                # No symbol, no section - use daddr with sentinel positions
                                formatted_ops.append(f"daddr({mk_hex}:2.00000000:0.00000000:0.00000000)")
                else:
                    # It's an immediate value or stack offset
                    formatted_ops.append("imm")
            else:
                if tok.startswith("var_"):
                    var_offset = tok[4:]  # Get the part after 'var_'
                    # Remove trailing 'h' if present (Intel hex format)
                    var_offset = var_offset.rstrip('hH')
                    # Remove leading 's' if present (IDA signed offset prefix)
                    var_offset = var_offset.lstrip('sS')
                    formatted_ops.append(f"var(0x{var_offset})")
                elif tok.startswith("arg_"):
                    # IDA auto-generated argument names - treat as stack variables
                    arg_offset = tok[4:]  # Get the part after 'arg_'
                    # Remove leading 's' if present
                    arg_offset = arg_offset.lstrip('sS')
                    formatted_ops.append(f"var(0x{arg_offset})")
                elif re.match(r'^0x[0-9a-fA-F]+$', tok):
                    # Raw hex address that slipped through - should not happen, filter it out
                    formatted_ops.append("imm")
                else:
                        formatted_ops.append(tok)

        if addr in addr_positions:
            addr_hdr = f"{hex(addr)}:{format_hierarchical_positions(addr_positions[addr])}"
        else:
            addr_hdr = hex(addr)

        ops_join = ' '.join(formatted_ops)
        if ops_join.count('[') > ops_join.count(']'):
            ops_join = ops_join + ' ]'

        if ops_join.strip():
            out_instrs.append(f"{opcode}({addr_hdr}) {ops_join}")
        else:
            out_instrs.append(f"{opcode}({addr_hdr})")

    return "\t".join(out_instrs)


def clean_ida_disasm(ea):
    """
    Get clean disassembly preserving IDA keywords (offset, short, etc.)
    but replacing symbol names with hex addresses.
    Mark immediate values with 'imm' token.
    Mark displacement operands (o_displ) with 'disp_0xXX' prefix to prevent address() wrapping.
    Global variable names -> resolved to hex addresses (e.g., wanted_samples -> 0x12340)
      This ensures debug symbols for global variables are converted to daddr(0x...) format
    """
    # Get mnemonic
    mnem = idc.print_insn_mnem(ea)
    if not mnem:
        return None
    
    # Get operands - up to 6 operands max
    operands = []
    for i in range(6):
        op = idc.print_operand(ea, i)
        if not op:
            break
        
        # Get operand type and value first
        op_type = idc.get_operand_type(ea, i)
        op_value = idc.get_operand_value(ea, i)
        
        # Force stack variables to use var(0xXX) format instead of debug symbol names
        # This prevents IDA debug symbols like i_0, f_0, delim_0 from appearing
        if op_type == idc.o_displ:
            # This is a displacement operand (stack variable or memory offset)
            # Check if it contains a debug symbol name pattern (e.g., i_0, delim_0)
            # Pattern: word characters followed by underscore and digit
            if re.search(r'\b[a-zA-Z_][a-zA-Z0-9_]*_\d+\b', op):
                # Replace debug symbol with var(0xXX) format using the offset value
                # Extract base register and reconstruct with hex offset
                base_reg_match = re.search(r'\[([a-z0-9]+)', op, re.IGNORECASE)
                if base_reg_match:
                    base_reg = base_reg_match.group(1)
                    # Use var_ format with hex offset
                    op = f"[{base_reg}+var_{op_value:X}h]" if op_value > 0 else f"[{base_reg}+var_0]"
        
        # Clean up IDA's duplicate offsets in var format: [rsp+60h+var_60] -> [rsp+var_60]
        if 'var_' in op:
            op = re.sub(r'\+?\s*0x[0-9A-Fa-f]+\s*\+\s*(?=var_)', '+', op)
            op = re.sub(r'\+?\s*[0-9A-Fa-f]+h\s*\+\s*(?=var_)', '+', op, flags=re.IGNORECASE)
            # Clean up potential artifacts: ++ -> +, [+ -> [, +] -> ]
            op = re.sub(r'\+\s*\+', '+', op)
            op = re.sub(r'\[\s*\+', '[', op)
            op = re.sub(r'\+\s*\]', ']', op)
        
        # Check if it's an immediate value
        if op_type == idc.o_imm:
            # It's an immediate - mark it
            op = "imm"
        # Check if operand contains keywords or symbols that need address replacement
        # Handle "offset symbol_name" -> "offset 0xADDR"
        elif 'offset' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"offset {hex(op_value)}"
        # Handle "short symbol_name" -> "short 0xADDR"
        elif 'short' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"short {hex(op_value)}"
        # Handle "large symbol_name" -> "large 0xADDR"
        elif 'large' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"large {hex(op_value)}"
        # Handle segment prefix "cs:symbol" -> just "0xADDR" (remove cs:, ds:, etc.)
        elif any(seg in op for seg in ['cs:', 'ds:', 'es:', 'ss:', 'fs:', 'gs:']) and op_value != idaapi.BADADDR and op_value != 0:
            op = hex(op_value)
        # Handle displacement operands without var (regular offsets)
        elif op_type in [idc.o_phrase, idc.o_displ]:
            # If it's not a var (already cleaned above), replace hex offset with 'disp' while preserving structure
            if 'var_' not in op and op_value != idaapi.BADADDR and op_value != 0:
                # Replace hex values in the operand string with 'disp' marker
                # This preserves brackets and registers: [rip+0x1000] -> [rip+disp]
                op = re.sub(r'0x[0-9A-Fa-f]+', 'disp', op)
                op = re.sub(r'[0-9A-Fa-f]+h', 'disp', op, flags=re.IGNORECASE)
        # For operands that reference code/data addresses (but NOT displacements)
        elif op_type in [idc.o_near, idc.o_mem, idc.o_far]:
            # var_ already cleaned above, just check if it needs address replacement
            if 'var_' not in op:
                # Always prefer using the address value over symbol names (especially IDA auto-generated ones)
                if op_value != idaapi.BADADDR and op_value != 0:
                    # Valid address - always use hex format to avoid IDA auto-generated names
                    op = hex(op_value)
                elif not op.startswith('0x') and not op.startswith('['):
                    # No valid op_value - try to resolve symbol name to address
                    # Remove any segment prefix first
                    clean_op = op
                    for seg_prefix in ['cs:', 'ds:', 'es:', 'ss:', 'fs:', 'gs:']:
                        if clean_op.startswith(seg_prefix):
                            clean_op = clean_op[len(seg_prefix):]
                            break
                    # Try to get address by name
                    sym_addr = idc.get_name_ea_simple(clean_op)
                    if sym_addr != idaapi.BADADDR:
                        op = hex(sym_addr)
        
        operands.append(op)
    
    # Build clean disassembly
    result = f"{mnem} {', '.join(operands)}" if operands else mnem
    
    return result


def get_basic_blocks_ida(func_ea, flowchart=None):
    """
    Get basic blocks for a function in IDA Pro.
    Returns list of (bb_start, bb_end) tuples.
    If flowchart is provided, reuse it instead of creating a new one.
    """
    func = ida_funcs.get_func(func_ea)
    if not func:
        return []
    
    if flowchart is None:
        flowchart = idaapi.FlowChart(func)
    
    blocks = [(block.start_ea, block.end_ea) for block in flowchart]
    return blocks


def process_file_ida(fpath: str, out_dir: str):
    """Process a binary file using IDA Pro and export entire functions."""
    print(f"[INFO] Processing: {fpath}")
    
    try:
        # Wait for auto-analysis
        ida_auto.auto_wait()
        
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        binary_name = Path(fpath).name
        out_file = out_dir / f"{binary_name}_functions.txt"
        
        print(f"[INFO] Output file: {out_file}")

        # Build symbol and string maps
        symbol_map = {}
        for ea_tuple in idautils.Names():
            addr, name = ea_tuple
            symbol_map[addr] = name
        print(f"[INFO] Found {len(symbol_map)} symbols")
        
        string_map = {}
        for s in idautils.Strings():
            string_map[s.ea] = str(s)
        print(f"[INFO] Found {len(string_map)} strings")
    
    except Exception as e:
        print(f"[ERROR] Initial setup failed: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        addr_positions = {}
        bb_range_map = {}    # bb_start -> (bb_start, bb_end)
        bin_counter = 0
        
        # Calculate min/max addresses from segments
        print("[INFO] Calculating binary address range from segments...")
        sections = []
        for n in range(ida_segment.get_segm_qty()):
            seg = ida_segment.getnseg(n)
            if seg:
                start = seg.start_ea
                end = seg.end_ea
                name = ida_segment.get_segm_name(seg)
                sections.append((start, end, name))
                print(f"[INFO]   Segment: {name} [{hex(start)} - {hex(end)}]")
        
        # Get min/max from all segments
        if sections:
            min_addr = min(s[0] for s in sections)
            max_addr = max(s[1] for s in sections)
        else:
            min_addr = 0
            max_addr = 0
        
        print(f"[INFO] Binary address range: {hex(min_addr)} - {hex(max_addr)}")
        
    except Exception as e:
        print(f"[ERROR] Address range calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Collect all functions
    print("[INFO] Collecting function information...")
    all_functions = list(idautils.Functions())
    print(f"[INFO] Found {len(all_functions)} functions")
    
    func_data = []  # List of (func_name, func_start, func_end, instructions)
    
    for idx, func_ea in enumerate(all_functions):
        if idx % 100 == 0:
            print(f"[INFO] Progress: {idx}/{len(all_functions)} functions")
        
        func = ida_funcs.get_func(func_ea)
        if not func:
            continue
        
        func_start = func.start_ea
        func_end = func.end_ea
        func_name = ida_funcs.get_func_name(func_ea)
        
        # Create flowchart once and reuse
        flowchart = idaapi.FlowChart(func)
        func_bbs = get_basic_blocks_ida(func_ea, flowchart)
        
        for bb_start, bb_end in func_bbs:
            bb_range_map[bb_start] = (bb_start, bb_end)
        
        # Collect all instructions in the function
        func_instructions = []
        
        for block in flowchart:
            bb_start = block.start_ea
            bb_end = block.end_ea
            
            curr = bb_start
            while curr < bb_end:
                # Get clean disassembly
                disasm_raw = clean_ida_disasm(curr)
                if disasm_raw:
                    # Normalize and mask the disassembly
                    norm_text, mask_line = normalize_and_mask(disasm_raw, symbol_map, string_map)
                    
                    # Store instruction data
                    func_instructions.append((curr, norm_text, mask_line))
                    
                    # Store position info
                    addr_positions[curr] = (bin_counter, curr, func_name, bb_start, func_start, func_end)
                    bin_counter += 1
                
                curr = idc.next_head(curr, bb_end)
                if curr == idaapi.BADADDR or curr >= bb_end:
                    break
        
        # Store function data if it has instructions
        if func_instructions:
            func_data.append((func_name, func_start, func_end, func_instructions))
    
    print(f"[INFO] Collected {len(func_data)} functions with {bin_counter} total instructions")
    
    # Build context for hierarchical encoding
    ctx = {
        'addr_positions': addr_positions,
        'total_bin': bin_counter,
        'bb_range_map': bb_range_map,
        'min_addr': min_addr,
        'max_addr': max_addr,
        'sections': sections,
        'symbol_map': symbol_map,
    }
    
    # Write entire functions to output file
    print("[INFO] Writing functions to output file...")
    written = 0
    skipped_empty = 0
    skipped_small = 0
    MIN_INSTRUCTIONS = 5
    
    with open(out_file, 'w', encoding='utf-8') as w:
        for func_name, func_start, func_end, func_instructions in func_data:
            # Skip empty functions
            if not func_instructions:
                skipped_empty += 1
                continue
            
            # Skip functions with fewer than MIN_INSTRUCTIONS (same as baseline)
            if len(func_instructions) < MIN_INSTRUCTIONS:
                skipped_small += 1
                continue
            
            # Build the entire function as a single line
            packed = build_function_inline(func_instructions, ctx)
            if packed:
                w.write(packed + "\n")
                written += 1
    
    print(f"[DONE] {out_file} (wrote {written} functions)")
    print(f"[INFO] Skipped: {skipped_empty} empty, {skipped_small} < {MIN_INSTRUCTIONS} instructions")
    print(f"[INFO] Position encoding:")
    print(f"  CODE: func_in_binary:bb_in_function:inst_in_bb")
    print(f"  DATA: section_in_binary:addr_in_section:0.0")


def main():
    """Main entry point when run from IDA Pro."""
    # Get output directory from environment or use default
    out_dir = os.getenv('OUTPUT_DIR', '/home/kun/Document/AAE/extern/jTrans/datautils_addraware/output')
    
    # Setup logging to file
    log_file = os.path.join(out_dir, 'ida_processing.log')
    os.makedirs(out_dir, exist_ok=True)
    
    # Redirect prints to log file
    import sys
    log_f = open(log_file, 'a')
    sys.stdout = log_f
    sys.stderr = log_f
    
    print("\n" + "="*70)
    print("[INFO] Function Export Script started in IDA Pro")
    print("="*70)
    
    # Wait for IDA's auto-analysis to complete
    print("[INFO] Waiting for IDA auto-analysis to complete...")
    ida_auto.auto_wait()
    print("[INFO] Auto-analysis complete")
    
    # Get the input file path
    input_file = idc.get_input_file_path()
    print(f"[INFO] Processing: {input_file}")
    
    print(f"[CONFIG] OUTPUT={out_dir}")
    print(f"[INFO] Exporting ENTIRE FUNCTIONS (no sequence chunking)")
    print(f"[INFO] Using HIERARCHICAL position encoding (address-based):")
    print(f"  CODE addresses:")
    print(f"    - Position 1: (func_start - min_addr) / (max_addr - min_addr)")
    print(f"    - Position 2: (bb_start - func_start) / (func_end - func_start)")
    print(f"    - Position 3: (inst_addr - bb_start) / (bb_end - bb_start)")
    print(f"  DATA addresses:")
    print(f"    - Position 1: (section_start - min_addr) / (max_addr - min_addr)")
    print(f"    - Position 2: (addr - section_start) / (section_end - section_start)")
    print(f"    - Position 3: 0.0 (no BB context)")
    
    # Process the file
    try:
        process_file_ida(input_file, out_dir)
        print("[SUCCESS] Processing complete")
        log_f.close()
        idc.qexit(0)  # Exit IDA successfully
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        log_f.close()
        idc.qexit(1)  # Exit IDA with error


# Run when script is executed by IDA
if __name__ == '__main__':
    main()
