"""
IDA Pro script to generate address-aware format directly.
Combines IDA binary analysis with hierarchical position encoding.

Usage:
    idat -A -S"process.py" /path/to/binary

Saves both:
    1. Pickle file with function metadata (like baseline)
    2. Address-aware text format for pretraining

Symbol Handling (following cfg_hierarchical_icfg_ida.py):
    - PLT symbols are kept as-is (e.g., .printf, .malloc, .free)
    - Symbol names are preserved for addresses in .plt section
    - Other addresses use hex format with hierarchical positions
    - combine_and_split.sh later filters to top 100 symbols

References:
    - data_generator/cfg_hierarchical_icfg_ida.py: Token preprocessing and symbol handling
    - extern/jTrans/datautils/process.py: IDA extraction
"""

import idc
import idautils
import idaapi
import ida_funcs
import ida_auto
import sys
import os
import re
import pickle
# Note: networkx is NOT needed for address-aware format generation
# (only needed for CFG generation which builds graph structures)
from pathlib import Path
from collections import defaultdict

# IMMEDIATE DEBUG: Write to file as soon as script loads
try:
    debug_file = open('/tmp/ida_debug.txt', 'w')
    debug_file.write("Script started loading\n")
    debug_file.flush()
except Exception as e:
    pass

# Add site-packages to Python path
sys.path.insert(0, '/home/kun/anaconda3/lib/python3.12/site-packages')

try:
    debug_file.write("Added Python path\n")
    debug_file.flush()
except:
    pass

# Import BinaryAI if available
try:
    debug_file.write("Importing binaryai...\n")
    debug_file.flush()
    import binaryai
    if hasattr(binaryai, 'ida'):
        HAS_BINARYAI = True
        debug_file.write("BinaryAI available\n")
    else:
        HAS_BINARYAI = False
        debug_file.write("BinaryAI found but binaryai.ida not available\n")
except ImportError:
    HAS_BINARYAI = False
    binaryai = None
    debug_file.write("BinaryAI not available\n")
    
debug_file.flush()

# Allow overriding save/data roots via environment variables
SAVEROOT = os.environ.get('SAVEROOT', './extract')
DATAROOT = os.environ.get('DATAROOT', './dataset')
# Separate log directory - defaults to 'logs' subdirectory under SAVEROOT
LOGROOT = os.environ.get('LOGROOT', os.path.join(SAVEROOT, 'logs'))

debug_file.write(f"SAVEROOT={SAVEROOT}\n")
debug_file.write(f"DATAROOT={DATAROOT}\n")
debug_file.write(f"LOGROOT={LOGROOT}\n")
debug_file.flush()

# Hex pattern for address detection
HEX_RE = re.compile(r'0x[0-9a-fA-F]+')


def get_binary_range():
    """Get min and max address of binary."""
    min_addr = idaapi.BADADDR
    max_addr = 0
    
    for seg_ea in idautils.Segments():
        seg = idaapi.getseg(seg_ea)
        if seg:
            if min_addr == idaapi.BADADDR or seg.start_ea < min_addr:
                min_addr = seg.start_ea
            if seg.end_ea > max_addr:
                max_addr = seg.end_ea
    
    return min_addr, max_addr


def get_section_for_addr(addr):
    """
    Get section information for an address.
    Returns: (section_start, section_end, section_name) or None
    """
    seg = idaapi.getseg(addr)
    if not seg:
        return None
    
    seg_name = idc.get_segm_name(seg.start_ea)
    return (seg.start_ea, seg.end_ea, seg_name)


def clean_ida_disasm(ea):
    """
    Get clean disassembly with proper handling of different operand types.
    - Immediate values -> "imm"
    - Stack variables -> keep as-is (will be converted to var(0xOFFSET) later)
    - Displacements -> "disp" (to avoid address() wrapping)
    - Addresses -> hex values (will be wrapped with hierarchical positions)
    """
    # Get mnemonic
    mnem = idc.print_insn_mnem(ea)
    if not mnem:
        return None
    
    # Get operands
    operands = []
    for i in range(6):
        op = idc.print_operand(ea, i)
        if not op:
            break
        
        # Clean up IDA's duplicate offsets: [rsp+60h+var_60] -> [rsp+var_60]
        if 'var_' in op:
            op = re.sub(r'\+?\s*0x[0-9A-Fa-f]+\s*\+\s*(?=var_)', '+', op)
            op = re.sub(r'\+?\s*[0-9A-Fa-f]+h\s*\+\s*(?=var_)', '+', op, flags=re.IGNORECASE)
            op = re.sub(r'\+\s*\+', '+', op)
            op = re.sub(r'\[\s*\+', '[', op)
            op = re.sub(r'\+\s*\]', ']', op)
        
        # Get operand type and value
        op_type = idc.get_operand_type(ea, i)
        op_value = idc.get_operand_value(ea, i)
        
        # Handle different operand types
        if op_type == idc.o_imm:
            # Immediate value
            op = "imm"
        elif 'offset' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"offset {hex(op_value)}"
        elif 'short' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"short {hex(op_value)}"
        elif 'large' in op and op_value != idaapi.BADADDR and op_value != 0:
            op = f"large {hex(op_value)}"
        elif any(seg in op for seg in ['cs:', 'ds:', 'es:', 'ss:', 'fs:', 'gs:']) and op_value != idaapi.BADADDR and op_value != 0:
            # Remove segment prefix
            op = hex(op_value)
        elif op_type in [idc.o_phrase, idc.o_displ]:
            # Displacement operand (e.g., [rbp+0x10])
            if 'var_' not in op and op_value != idaapi.BADADDR and op_value != 0:
                op = "disp"
        elif op_type in [idc.o_near, idc.o_mem, idc.o_far]:
            # Code/data address
            if 'var_' not in op and op_value != idaapi.BADADDR and op_value != 0:
                if not op.startswith('0x') and not op.startswith('['):
                    op = hex(op_value)
        
        operands.append(op)
    
    # Build disassembly
    result = f"{mnem} {', '.join(operands)}" if operands else mnem
    return result


def normalize_operand(operand, min_addr, max_addr):
    """
    Normalize an operand token.
    Returns: (normalized_token, address_value or None)
    
    - Immediate values: return ("imm", None)
    - Stack variables: return ("var(0xOFFSET)", None)
    - Addresses: return (hex_string, address_value)
    - Other tokens: return (token, None)
    """
    operand = operand.strip()
    
    # Check for displacement (already marked by clean_ida_disasm)
    if operand == "disp":
        return ("disp", None)
    
    # Check for immediate
    if operand == "imm":
        return ("imm", None)
    
    # Check for stack variable (var_XX format)
    if operand.startswith("var_"):
        var_offset = operand[4:]  # Get offset after "var_"
        return (f"var(0x{var_offset})", None)
    
    # Check for hex address
    hex_match = HEX_RE.search(operand)
    if hex_match:
        hex_str = hex_match.group(0)
        try:
            addr_value = int(hex_str, 16)
            # Filter out small values (likely immediates, not addresses)
            addr_threshold = max(min_addr, 0x1000)
            if addr_value >= addr_threshold:
                return (hex_str, addr_value)
            else:
                return ("imm", None)
        except ValueError:
            pass
    
    # Intel hex format (1234ABCDh)
    intel_hex_match = re.match(r'^[0-9A-Fa-f]+h$', operand, re.IGNORECASE)
    if intel_hex_match:
        hex_str = '0x' + operand[:-1]
        try:
            addr_value = int(hex_str, 16)
            addr_threshold = max(min_addr, 0x1000)
            if addr_value >= addr_threshold:
                return (hex_str, addr_value)
            else:
                return ("imm", None)
        except ValueError:
            pass
    
    # Return as-is
    return (operand, None)


def format_data_address_positions(addr, min_addr, max_addr):
    """
    Format hierarchical positions for data addresses (non-code).
    
    For data sections (.data, .rodata, .bss):
        Position 1: section's position in binary
        Position 2: address position inside section
        Position 3: 0.0 (no BB context)
    
    For other sections: return sentinel "2.00000000:0.00000000:0.00000000"
    """
    section_info = get_section_for_addr(addr)
    
    if section_info is None:
        return "2.00000000:0.00000000:0.00000000"
    
    sec_start, sec_end, sec_name = section_info
    sname = sec_name.lower()
    
    # Only treat data-like segments
    is_data_like = (
        ".data" in sname or
        ".rodata" in sname or
        ".bss" in sname
    )
    
    if not is_data_like:
        return "2.00000000:0.00000000:0.00000000"
    
    # Compute normalized positions
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


def format_hierarchical_positions(addr, func_start, func_end, bb_start, bb_end, min_addr, max_addr):
    """
    Format hierarchical positions for code addresses.
    
    Position 1: Function's position in binary (bnorm)
    Position 2: BB's position in function (fnorm)
    Position 3: Instruction's position in BB (bbnorm)
    """
    # Position 1: Function in binary
    if max_addr > min_addr:
        func_binary_norm = (func_start - min_addr) / float(max_addr - min_addr)
        func_binary_norm = max(0.0, min(1.0, func_binary_norm))
    else:
        func_binary_norm = 0.0
    
    # Position 2: BB in function
    if bb_start is not None and func_end > func_start:
        bb_function_norm = (bb_start - func_start) / float(func_end - func_start)
        bb_function_norm = max(0.0, min(1.0, bb_function_norm))
    else:
        bb_function_norm = 0.0
    
    # Position 3: Instruction in BB
    if bb_start is not None and bb_end is not None and bb_end > bb_start:
        inst_bb_norm = (addr - bb_start) / float(bb_end - bb_start)
        inst_bb_norm = max(0.0, min(1.0, inst_bb_norm))
    else:
        inst_bb_norm = 0.0
    
    return f"{func_binary_norm:.8f}:{bb_function_norm:.8f}:{inst_bb_norm:.8f}"


def get_instruction_bb_context(inst_addr, flowchart):
    """
    Get basic block context for an instruction.
    Returns: (bb_start, bb_end) or (None, None)
    """
    for block in flowchart:
        if block.start_ea <= inst_addr < block.end_ea:
            return (block.start_ea, block.end_ea)
    return (None, None)


def get_asm_list(func_ea):
    """Get assembly list for a function (baseline format)."""
    asm_list = []
    for inst in idautils.FuncItems(func_ea):
        asm_list.append(idc.GetDisasm(inst))
    return asm_list


def get_rawbytes(func_ea):
    """Get raw bytes for a function (baseline format)."""
    rawbytes_list = b""
    for inst in idautils.FuncItems(func_ea):
        rawbytes_list += idc.get_bytes(inst, idc.get_item_size(inst))
    return rawbytes_list


def get_cfg(func_ea):
    """Get CFG for a function (baseline format).
    
    Note: Returns None to avoid networkx dependency.
    For address-aware format, CFG is not needed in the pickle file.
    """
    return None


def get_binai_feature(func_ea):
    """Get BinaryAI features if available."""
    if HAS_BINARYAI and binaryai is not None:
        return binaryai.ida.get_func_feature(func_ea)
    else:
        return None


def process_function(func_ea, min_addr, max_addr, symbol_map=None):
    """
    Process a single function and return address-aware format.
    Returns: list of instruction strings
    
    Args:
        func_ea: Function entry address
        min_addr: Minimum address of binary
        max_addr: Maximum address of binary
        symbol_map: Dictionary mapping addresses to symbol names (for PLT entries)
    """
    if symbol_map is None:
        symbol_map = {}
    
    func = ida_funcs.get_func(func_ea)
    if not func:
        return []
    
    func_start = func.start_ea
    func_end = func.end_ea
    
    # Get flowchart for BB information
    flowchart = idaapi.FlowChart(func)
    
    # Build instruction address set for this function
    func_instrs = []
    for inst_addr in idautils.FuncItems(func_ea):
        func_instrs.append(inst_addr)
    
    # Build BB lookup map
    bb_map = {}  # inst_addr -> (bb_start, bb_end)
    for block in flowchart:
        for inst_addr in range(block.start_ea, block.end_ea):
            if idc.is_code(idc.get_full_flags(inst_addr)):
                bb_map[inst_addr] = (block.start_ea, block.end_ea)
    
    # Process each instruction
    result_tokens = []
    
    for inst_addr in func_instrs:
        # Get clean disassembly
        disasm = clean_ida_disasm(inst_addr)
        if not disasm:
            continue
        
        # Split into opcode and operands
        parts = disasm.split(' ', 1)
        opcode = parts[0]
        operands_str = parts[1] if len(parts) > 1 else ""
        
        # Get BB context
        bb_start, bb_end = bb_map.get(inst_addr, (None, None))
        
        # Format opcode with hierarchical positions
        opcode_pos = format_hierarchical_positions(
            inst_addr, func_start, func_end, bb_start, bb_end, min_addr, max_addr
        )
        opcode_token = f"{opcode}({hex(inst_addr)}:{opcode_pos})"
        
        # Process operands
        formatted_operands = []
        if operands_str:
            # Split operands by comma
            operand_list = [op.strip() for op in operands_str.split(',')]
            
            for operand in operand_list:
                # Handle complex operands (e.g., [rax+rbx*4+0x10])
                # Split by common delimiters and keep them as separate tokens
                tokens_in_operand = re.split(r'([\[\]\+\-\*])', operand)
                
                formatted_parts = []
                for token in tokens_in_operand:
                    token = token.strip()
                    if not token:
                        continue
                    
                    # Keep delimiters as-is (brackets, operators)
                    if token in ['[', ']', '+', '-', '*']:
                        formatted_parts.append(token)
                        continue
                    
                    # Normalize token
                    norm_token, addr_value = normalize_operand(token, min_addr, max_addr)
                    
                    # If it's an address, wrap with hierarchical positions
                    if addr_value is not None:
                        # Check if this address has a symbol name (PLT/GOT entries)
                        # First check symbol_map (faster), then try idc.get_name() for dynamic symbols
                        symbol_name = None
                        if addr_value in symbol_map:
                            symbol_name = symbol_map[addr_value]
                        else:
                            # Try getting name directly - works for GOT entries in stripped binaries
                            name = idc.get_name(addr_value)
                            if name and not name.startswith('sub_') and not name.startswith('loc_'):
                                symbol_name = name
                        
                        # If we have a symbol name, check if it's a library function
                        if symbol_name:
                            # Check section to see if it's PLT/GOT related
                            section_info = get_section_for_addr(addr_value)
                            if section_info is not None:
                                sec_start, sec_end, sec_name = section_info
                                sec_lower = sec_name.lower()
                                # PLT or GOT sections contain library function calls
                                if '.plt' in sec_lower or '.got' in sec_lower:
                                    # Clean up the symbol name (remove @GLIBC suffixes, _ptr suffix, add dot prefix)
                                    clean_name = symbol_name.split('@')[0]  # Remove version info
                                    clean_name = clean_name.replace('_ptr', '')  # Remove IDA's _ptr suffix
                                    if not clean_name.startswith('.'):
                                        clean_name = '.' + clean_name
                                    formatted_parts.append(clean_name)
                                    continue
                        
                        # Check if it's a code address (in text section)
                        if idc.is_code(idc.get_full_flags(addr_value)):
                            # Get target function info
                            target_func = ida_funcs.get_func(addr_value)
                            if target_func:
                                target_func_start = target_func.start_ea
                                target_func_end = target_func.end_ea
                                target_flowchart = idaapi.FlowChart(target_func)
                                target_bb_start, target_bb_end = get_instruction_bb_context(addr_value, target_flowchart)
                                
                                addr_pos = format_hierarchical_positions(
                                    addr_value, target_func_start, target_func_end,
                                    target_bb_start, target_bb_end, min_addr, max_addr
                                )
                                formatted_parts.append(f"address({norm_token}:{addr_pos})")
                            else:
                                # Code address but no function
                                formatted_parts.append(f"address({norm_token}:0.00000000:0.00000000:0.00000000)")
                        else:
                            # Data address
                            data_pos = format_data_address_positions(addr_value, min_addr, max_addr)
                            formatted_parts.append(f"daddr({norm_token}:{data_pos})")
                    else:
                        # Not an address - regular token
                        formatted_parts.append(norm_token)
                
                # Add spaces between tokens for readability
                formatted_operands.append(' '.join(formatted_parts))
        
        # Build final instruction
        if formatted_operands:
            result_tokens.append(f"{opcode_token} {' '.join(formatted_operands)}")
        else:
            result_tokens.append(opcode_token)
    
    return result_tokens


def process_binary():
    """
    Process entire binary and save both:
    1. Pickle file with function metadata (like baseline)
    2. Address-aware text format
    """
    debug_file.write("Entered process_binary()\n")
    debug_file.flush()
    
    # Setup logging to file
    binary_path = idc.get_input_file_path()
    binary_name = os.path.basename(binary_path).replace('.strip', '')
    
    debug_file.write(f"Binary: {binary_name}\n")
    debug_file.flush()
    
    # Create log directory if it doesn't exist
    log_dir = Path(LOGROOT)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = os.path.join(LOGROOT, f"{binary_name}_ida.log")
    debug_file.write(f"Log path: {log_path}\n")
    debug_file.flush()
    
    log_file = open(log_path, 'w')
    
    def log(msg):
        """Print to both console and log file"""
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()
        debug_file.write(f"LOG: {msg}\n")
        debug_file.flush()
    
    log("[INFO] Starting binary processing...")
    log(f"[INFO] SAVEROOT={SAVEROOT}")
    log(f"[INFO] DATAROOT={DATAROOT}")
    
    # Wait for auto-analysis
    log("[INFO] Waiting for auto-analysis to complete...")
    ida_auto.auto_wait()
    log("[INFO] Auto-analysis complete")
    
    # Verify directories exist
    try:
        assert os.path.exists(DATAROOT), f"DATAROOT does not exist: {DATAROOT}"
        assert os.path.exists(SAVEROOT), f"SAVEROOT does not exist: {SAVEROOT}"
    except AssertionError as e:
        log(f"[ERROR] {e}")
        log_file.close()
        raise
    
    # Get binary information
    log(f"[INFO] Binary: {binary_name}")
    log(f"[INFO] Path: {binary_path}")
    
    # Find unstripped binary for symbol information
    unstrip_path = None
    for root, dirs, files in os.walk(DATAROOT):
        if binary_name in files:
            unstrip_path = os.path.join(root, binary_name)
            break
    
    if unstrip_path is None:
        log(f"[WARNING] Could not find unstripped binary: {binary_name}")
        log(f"[WARNING] Will use function addresses as names")
    else:
        log(f"[INFO] Found unstripped binary: {unstrip_path}")
    
    # Get address range
    min_addr, max_addr = get_binary_range()
    log(f"[INFO] Address range: {hex(min_addr)} - {hex(max_addr)}")
    
    # Build symbol map (following cfg_hierarchical_icfg_ida.py)
    log("[INFO] Building symbol map...")
    symbol_map = {}
    for ea_tuple in idautils.Names():
        addr, name = ea_tuple
        symbol_map[addr] = name
    log(f"[INFO] Found {len(symbol_map)} symbols")
    
    # Prepare output paths
    saveroot_path = Path(SAVEROOT)
    saveroot_path.mkdir(parents=True, exist_ok=True)
    
    pickle_output = saveroot_path / f"{binary_name}_extract.pkl"
    text_output = saveroot_path / f"{binary_name}_addressaware.txt"
    
    # Data structures
    data_list = defaultdict(dict)
    text_lines = []
    
    func_count = 0
    total_instructions = 0
    
    # Process each function
    for func_ea in idautils.Functions():
        # Skip certain sections
        seg_name = idc.get_segm_name(func_ea)
        if seg_name in ['.plt', 'extern', '.init', '.fini']:
            continue
        
        func_name = idc.get_func_name(func_ea)
        log(f"[+] Processing: {func_name} @ {hex(func_ea)}")
        
        # 1. Extract baseline data (for pickle)
        asm_list = get_asm_list(func_ea)
        rawbytes_list = get_rawbytes(func_ea)
        cfg = get_cfg(func_ea)
        bai_feature = get_binai_feature(func_ea)
        
        # Save to pickle data structure
        data_list[func_name]['func'] = func_ea
        data_list[func_name]['asm'] = asm_list
        data_list[func_name]['raw'] = rawbytes_list
        data_list[func_name]['cfg'] = cfg
        data_list[func_name]['bai'] = bai_feature
        
        # 2. Generate address-aware format (for text)
        inst_tokens = process_function(func_ea, min_addr, max_addr, symbol_map)
        
        if inst_tokens:
            text_lines.append(' '.join(inst_tokens))
            total_instructions += len(inst_tokens)
        
        func_count += 1
    
    # Save pickle file (baseline format)
    log(f"[INFO] Saving pickle file: {pickle_output}")
    with open(pickle_output, 'wb') as f:
        pickle.dump(data_list, f)
    
    # Save text file (address-aware format)
    log(f"[INFO] Saving address-aware text file: {text_output}")
    with open(text_output, 'w', encoding='utf-8') as f:
        for line in text_lines:
            f.write(line + '\n')
    
    log(f"[SUCCESS] Processed {func_count} functions")
    log(f"[SUCCESS] Total instructions: {total_instructions}")
    log(f"[SUCCESS] Pickle output: {pickle_output}")
    log(f"[SUCCESS] Text output: {text_output}")
    
    log_file.close()


if __name__ == "__main__":
    debug_file.write("Entering __main__\n")
    debug_file.flush()
    try:
        process_binary()
        debug_file.write("process_binary() completed successfully\n")
        debug_file.flush()
        debug_file.close()
        idc.qexit(0)
    except Exception as e:
        error_msg = f"[ERROR] Exception during processing: {e}"
        print(error_msg)
        debug_file.write(f"{error_msg}\n")
        import traceback
        traceback.print_exc()
        debug_file.write(traceback.format_exc())
        debug_file.flush()
        debug_file.close()
        
        # Try to write error to log file
        try:
            binary_path = idc.get_input_file_path()
            binary_name = os.path.basename(binary_path).replace('.strip', '')
            log_path = os.path.join(LOGROOT, f"{binary_name}_ida.log")
            with open(log_path, 'a') as f:
                f.write(f"\n[ERROR] Exception: {e}\n")
                f.write(traceback.format_exc())
        except:
            pass
        
        idc.qexit(1)

