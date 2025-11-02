import sys
import binaryninja
import re

def is_immediate_value(addr_str):
    """Check if an address is actually a small immediate value (not a real address)"""
    try:
        value = int(addr_str, 16)
        # Consider values less than 0x1000 as immediate values, not addresses
        return value < 0x1000
    except:
        return False

def classify_address_in_instruction(tokens, addr_idx, bv=None):
    """
    Classify an address based on its context in the instruction and the binary section
    Returns: 'addr_data' or 'addr_code'
    """
    # Check the actual section if binary view is available
    if bv is not None and addr_idx < len(tokens):
        addr_str = tokens[addr_idx]
        try:
            addr_value = int(addr_str, 16)
            # Get the section this address belongs to
            sections = bv.get_sections_at(addr_value)
            if sections:
                section = sections[0]
                section_name = section.name.lower()
                # Check section semantics - this is authoritative
                semantics = section.semantics
                
                # Code sections: .text, executable sections
                if semantics in [binaryninja.SectionSemantics.ReadOnlyCodeSectionSemantics,
                                binaryninja.SectionSemantics.ReadExecuteCodeSectionSemantics]:
                    return 'addr_code'
                # Data sections: .data, .bss, .rodata, .got, .plt
                elif semantics in [binaryninja.SectionSemantics.ReadOnlyDataSectionSemantics,
                                  binaryninja.SectionSemantics.ReadWriteDataSectionSemantics]:
                    return 'addr_data'
                # Use section name as fallback
                elif 'text' in section_name:
                    return 'addr_code'
                elif any(x in section_name for x in ['data', 'bss', 'rodata', 'got', 'plt']):
                    return 'addr_data'
        except:
            pass
    
    # Fallback heuristic: if no section info available
    # Assume memory operands are data, others are code
    context_before = ' '.join(tokens[max(0, addr_idx-3):addr_idx])
    if '[' in context_before or 'rel' in context_before:
        return 'addr_data'
    
    # Default to code address
    return 'addr_code'

def label_addresses_in_instruction(inst_str, bv=None):
    """
    Replace addresses in instruction with labeled format: addr_type:0xVALUE
    bv: BinaryView object for section lookup
    """
    tokens = inst_str.split()
    result_tokens = []
    
    for i, token in enumerate(tokens):
        # Check if token is an address (0x followed by hex digits)
        if re.match(r'^0x[0-9a-fA-F]+$', token):
            # Check if it's an immediate value (small constant)
            if is_immediate_value(token):
                # Keep immediate values unchanged
                result_tokens.append(token)
            else:
                # Classify and label the address
                addr_type = classify_address_in_instruction(tokens, i, bv)
                result_tokens.append(f'{addr_type}:{token}')
        else:
            result_tokens.append(token)
    
    return ' '.join(result_tokens)

def get_token_from_inst(inst):
    # get tokens from an instruction
    # e.g. mov eax, [ebp+0x10] -> ['mov', 'eax', '[', 'ebp', '+', '0x10', ']']
    # Remove all commas from the instruction
    tokens = [token.text for token in inst[0]]
    inst_str = ' '.join(tokens).replace(',', '')
    return inst_str.split()

def get_basic_block_seq(bb, split_at_call=False, only_call=False, after_call=False):
    # get instruction sequence from a basic block
    # the instructions are split by '\t'
    # split_at_call: if True and BB contains call, only return instructions up to and including the call
    # only_call: return only the call instruction
    # after_call: return only instructions after the call
    # Returns: (instruction_string, start_addr, end_addr) or ('', None, None) if empty
    inst_seq = []
    
    # Collect all instruction information: (tokens, addr, length)
    all_inst_info = []
    for inst_data in bb:
        tokens = inst_data[0]
        # Get instruction address and length from the function's disassembly
        # We'll use the BB start as base and calculate offsets
        all_inst_info.append(inst_data)
    
    if not all_inst_info:
        return '', None, None
    
    instructions_to_use = all_inst_info
    
    if split_at_call or only_call or after_call:
        # Find if there's a call instruction
        call_idx = -1
        for i, inst_data in enumerate(all_inst_info):
            tokens = [token.text for token in inst_data[0]]
            inst_str = ' '.join(tokens).replace(',', '')
            tokens_clean = inst_str.split()
            if tokens_clean and tokens_clean[0].lower() == 'call':
                call_idx = i
                break
        
        if call_idx >= 0:
            if only_call:
                # Return only the call instruction
                instructions_to_use = [all_inst_info[call_idx]]
            elif after_call:
                # Return instructions after the call
                if call_idx + 1 < len(all_inst_info):
                    instructions_to_use = all_inst_info[call_idx + 1:]
                else:
                    return '', None, None  # No instructions after call
            else:  # split_at_call
                # Return up to and including the call
                instructions_to_use = all_inst_info[:call_idx + 1]
    
    if not instructions_to_use:
        return '', None, None
    
    # Get addresses by disassembling the actual instructions
    # Binary Ninja gives us instruction addresses through the architecture
    func = bb.function
    bv = func.view
    
    # Start from BB start and calculate instruction addresses
    current_addr = bb.start
    inst_addresses = []
    
    for inst_data in all_inst_info:
        inst_addresses.append(current_addr)
        # Get instruction length by reading and getting info
        inst_info = bv.get_disassembly(current_addr)
        # Use instruction text length or calculate from next instruction
        # More reliable: use arch to get instruction info
        data = bv.read(current_addr, 15)
        inst_info_tuple = func.arch.get_instruction_info(data, current_addr)
        if inst_info_tuple and inst_info_tuple.length > 0:
            current_addr += inst_info_tuple.length
        else:
            current_addr += 1  # Fallback
    
    # Find indices of instructions we're using
    start_idx = all_inst_info.index(instructions_to_use[0])
    end_idx = all_inst_info.index(instructions_to_use[-1])
    
    start_addr = inst_addresses[start_idx]
    last_inst_addr = inst_addresses[end_idx]
    
    # Calculate end address as start of NEXT instruction (fall-through address)
    # This is where execution would continue sequentially
    data = bv.read(last_inst_addr, 15)
    inst_info_tuple = func.arch.get_instruction_info(data, last_inst_addr)
    if inst_info_tuple and inst_info_tuple.length > 0:
        end_addr = last_inst_addr + inst_info_tuple.length  # Next instruction starts here
    else:
        end_addr = last_inst_addr + 1
    
    # Build instruction sequence with labeled addresses
    for inst_data in instructions_to_use:
        # Get all tokens and join them
        tokens = [token.text for token in inst_data[0]]
        inst_str_raw = ' '.join(tokens)
        # Remove ALL commas from the instruction string
        inst_str_raw = inst_str_raw.replace(',', '')
        # Label addresses in this instruction (pass bv for section lookup)
        inst_str_labeled = label_addresses_in_instruction(inst_str_raw, bv)
        inst_seq.append(inst_str_labeled)
    
    inst_str = '\t'.join(inst_seq)
    return inst_str, start_addr, end_addr

def format_bb_with_addr(inst_str, start_addr, end_addr):
    """Format BB sequence with start and end addresses"""
    if not inst_str or start_addr is None or end_addr is None:
        return inst_str
    return f"<{hex(start_addr)}> {inst_str} <{hex(end_addr)}>"

def is_indirect_branch(bb):
    # Check if basic block ends with an indirect call or jump
    # Returns True for: call rax, jmp rax, call [mem], jmp [mem], etc.
    # Returns False for: call 0x401000, jmp 0x401000, je 0x401000, etc.
    if len(bb) == 0:
        return False
    
    # Get the last instruction
    last_inst = list(bb)[-1]
    tokens = [token.text for token in last_inst[0]]
    inst_str = ' '.join(tokens).replace(',', '')
    tokens_clean = inst_str.split()
    
    if not tokens_clean:
        return False
    
    # Check if it's a call or jump instruction
    mnemonic = tokens_clean[0].lower()
    if mnemonic not in ['call', 'jmp', 'ret', 'retn']:
        return False
    
    # ret/retn are always indirect
    if mnemonic in ['ret', 'retn']:
        return False  # Don't treat ret as indirect branch to [Unknown]
    
    # For call/jmp, check if the target is indirect
    # Direct: call 0x401000 (has 0x prefix)
    # Indirect: call rax, call [mem], jmp rax, etc.
    has_direct_address = any(token.startswith('0x') for token in tokens_clean[1:])
    
    return not has_direct_address

def ends_with_call(bb):
    # Check if basic block ends with a call instruction (direct or indirect)
    if len(bb) == 0:
        return False
    
    # Get the last instruction
    last_inst = list(bb)[-1]
    tokens = [token.text for token in last_inst[0]]
    inst_str = ' '.join(tokens).replace(',', '')
    tokens_clean = inst_str.split()
    
    if not tokens_clean:
        return False
    
    return tokens_clean[0].lower() == 'call'

def ends_with_ret(bb):
    """Check if basic block ends with a return instruction"""
    if len(bb) == 0:
        return False
    
    # Get the last instruction
    last_inst = list(bb)[-1]
    tokens = [token.text for token in last_inst[0]]
    inst_str = ' '.join(tokens).replace(',', '')
    tokens_clean = inst_str.split()
    
    if not tokens_clean:
        return False
    
    mnemonic = tokens_clean[0].lower()
    return mnemonic in ['ret', 'retn']

def contains_call(bb):
    # Check if basic block contains a call instruction (anywhere, not just at the end)
    for inst in bb:
        tokens = [token.text for token in inst[0]]
        inst_str = ' '.join(tokens).replace(',', '')
        tokens_clean = inst_str.split()
        if tokens_clean and tokens_clean[0].lower() == 'call':
            return True
    return False

def get_all_call_indices(bb):
    """Get indices of all call instructions in a basic block"""
    call_indices = []
    all_insts = list(bb)
    for i, inst in enumerate(all_insts):
        tokens = [token.text for token in inst[0]]
        inst_str = ' '.join(tokens).replace(',', '')
        tokens_clean = inst_str.split()
        if tokens_clean and tokens_clean[0].lower() == 'call':
            call_indices.append(i)
    return call_indices

def get_instruction_range(bb, start_idx, end_idx):
    """
    Get a range of instructions from a basic block [start_idx, end_idx] inclusive
    Returns: (instruction_string, start_addr, end_addr) or ('', None, None) if empty
    """
    all_inst_info = list(bb)
    
    if start_idx < 0 or end_idx >= len(all_inst_info) or start_idx > end_idx:
        return '', None, None
    
    instructions_to_use = all_inst_info[start_idx:end_idx+1]
    
    if not instructions_to_use:
        return '', None, None
    
    func = bb.function
    bv = func.view
    
    # Calculate instruction addresses
    current_addr = bb.start
    inst_addresses = []
    
    for inst_data in all_inst_info:
        inst_addresses.append(current_addr)
        data = bv.read(current_addr, 15)
        inst_info_tuple = func.arch.get_instruction_info(data, current_addr)
        if inst_info_tuple and inst_info_tuple.length > 0:
            current_addr += inst_info_tuple.length
        else:
            current_addr += 1
    
    start_addr = inst_addresses[start_idx]
    last_inst_addr = inst_addresses[end_idx]
    
    # Calculate end address as start of NEXT instruction
    data = bv.read(last_inst_addr, 15)
    inst_info_tuple = func.arch.get_instruction_info(data, last_inst_addr)
    if inst_info_tuple and inst_info_tuple.length > 0:
        end_addr = last_inst_addr + inst_info_tuple.length
    else:
        end_addr = last_inst_addr + 1
    
    # Build instruction sequence with labeled addresses
    inst_seq = []
    for inst_data in instructions_to_use:
        # Get all tokens and join them
        tokens = [token.text for token in inst_data[0]]
        inst_str_raw = ' '.join(tokens)
        # Remove ALL commas from the instruction string
        inst_str_raw = inst_str_raw.replace(',', '')
        inst_str_labeled = label_addresses_in_instruction(inst_str_raw, bv)
        inst_seq.append(inst_str_labeled)
    
    inst_str = '\t'.join(inst_seq)
    return inst_str, start_addr, end_addr

def get_next_instruction_bb(bv, call_bb):
    # Get the basic block containing the instruction right after the call
    # call_bb is the basic block ending with a call instruction
    if len(call_bb) == 0:
        return None
    
    last_inst = list(call_bb)[-1]
    call_addr = last_inst[1]  # instruction address
    
    # Calculate the instruction length by looking at tokens
    # The end address is call_bb.end
    call_len = call_bb.end - call_addr
    next_addr = call_addr + call_len
    
    # Find the basic block at this address
    funcs = bv.get_functions_containing(next_addr)
    if funcs:
        for func in funcs:
            for bb in func:
                if bb.start == next_addr:
                    return bb
    return None

def get_call_targets(bb):
    # Extract call targets from a basic block
    # Returns a list of addresses that are called
    call_targets = []
    for inst in bb:
        # Check if this is a call instruction
        tokens = [token.text for token in inst[0]]
        inst_str = ' '.join(tokens).replace(',', '')
        tokens_clean = inst_str.split()
        if tokens_clean and tokens_clean[0].lower() == 'call':
            # Try to get the call target address
            # inst[1] is the instruction address
            # inst[2] is the instruction length
            # We need to check the operands
            for token in inst[0]:
                # Check if token is an address (starts with 0x)
                if token.text.startswith('0x'):
                    try:
                        call_targets.append(int(token.text, 16))
                    except:
                        pass
    return call_targets

def main():
    if len(sys.argv) < 2:
        print("Usage: python bb_flow.py <binary_path> [output_file]")
        sys.exit(1)

    binary_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "bb_pairs.txt"
    
    # To prevent the "Core is not running" error, use a with statement
    # which handles the lifecycle of the binary view object.
    with binaryninja.load(binary_path) as bv:
        bv.update_analysis_and_wait()

        print(f"Extracting basic block pairs from {binary_path}...")
        
        # First pass: collect all call sites and map returns to call sites
        call_return_map = {}  # Maps (func_addr, ret_bb) -> [(inst_str, start_addr, end_addr)]
        
        for func in bv.functions:
            for bb in func:
                call_indices = get_all_call_indices(bb)
                
                if call_indices:
                    # Process each call in this BB
                    all_insts = list(bb)
                    total_insts = len(all_insts)
                    
                    for call_idx in call_indices:
                        # Get the instructions after this specific call
                        after_call_inst = None
                        after_start = None
                        after_end = None
                        
                        if call_idx + 1 < total_insts:
                            # There are instructions after this call in the same BB
                            after_call_inst, after_start, after_end = get_instruction_range(bb, call_idx + 1, total_insts - 1)
                        
                        # Get the call target from this specific call instruction
                        tokens = [token.text for token in all_insts[call_idx][0]]
                        inst_str = ' '.join(tokens).replace(',', '')
                        call_tokens = inst_str.split()
                        call_target = None
                        
                        for token in call_tokens[1:]:
                            if token.startswith('0x') or token.startswith('addr_'):
                                # Extract address
                                if token.startswith('addr_'):
                                    call_target = int(token.split(':')[1], 16)
                                else:
                                    call_target = int(token, 16)
                                break
                        
                        if call_target:
                            target_funcs = bv.get_functions_containing(call_target)
                            if target_funcs:
                                for target_func in target_funcs:
                                    # Check if this function has any return blocks
                                    has_returns = any(ends_with_ret(ret_bb) for ret_bb in target_func.basic_blocks)
                                    
                                    if has_returns:
                                        # Normal function with returns - map all return blocks to after-call sequence
                                        if target_func.start not in call_return_map:
                                            call_return_map[target_func.start] = {}
                                        
                                        for ret_bb in target_func.basic_blocks:
                                            if ends_with_ret(ret_bb):
                                                if ret_bb not in call_return_map[target_func.start]:
                                                    call_return_map[target_func.start][ret_bb] = []
                                                
                                                # The return should go to the after-call sequence (instructions after the call)
                                                if after_call_inst:
                                                    # Return to instructions after call in same BB
                                                    call_return_map[target_func.start][ret_bb].append(
                                                        (after_call_inst, after_start, after_end))
                                                else:
                                                    # No instructions after call in same BB, use next BB
                                                    # But split it if it contains a call too
                                                    next_bb = get_next_instruction_bb(bv, bb)
                                                    if next_bb:
                                                        if contains_call(next_bb):
                                                            next_inst, next_start, next_end = get_basic_block_seq(next_bb, split_at_call=True)
                                                        else:
                                                            next_inst, next_start, next_end = get_basic_block_seq(next_bb)
                                                        if next_inst:
                                                            call_return_map[target_func.start][ret_bb].append(
                                                                (next_inst, next_start, next_end))
                                    else:
                                        # PLT stub or external function with no returns (e.g., just a jmp)
                                        # Use a special marker to indicate we should create direct call->continuation edge
                                        if target_func.start not in call_return_map:
                                            call_return_map[target_func.start] = {}
                                        
                                        # Use None as a special key to indicate "no return block, direct continuation"
                                        if None not in call_return_map[target_func.start]:
                                            call_return_map[target_func.start][None] = []
                                        
                                        if after_call_inst:
                                            call_return_map[target_func.start][None].append(
                                                (after_call_inst, after_start, after_end))
                                        else:
                                            next_bb = get_next_instruction_bb(bv, bb)
                                            if next_bb:
                                                if contains_call(next_bb):
                                                    next_inst, next_start, next_end = get_basic_block_seq(next_bb, split_at_call=True)
                                                else:
                                                    next_inst, next_start, next_end = get_basic_block_seq(next_bb)
                                                if next_inst:
                                                    call_return_map[target_func.start][None].append(
                                                        (next_inst, next_start, next_end))
                                    break
                            break        # Second pass: generate basic block pairs
        with open(output_file, 'w') as f:
            pair_count = 0
            for func in bv.functions:
                for bb in func:
                    # Check if BB contains calls - if so, split at each call
                    call_indices = get_all_call_indices(bb)
                    
                    if call_indices:
                        # Process each call and the segments between them
                        all_insts = list(bb)
                        total_insts = len(all_insts)
                        
                        # Start from beginning
                        current_idx = 0
                        
                        for call_idx in call_indices:
                            # Part 1: Instructions up to and including this call
                            call_inst, call_start, call_end = get_instruction_range(bb, current_idx, call_idx)
                            
                            if call_inst:
                                call_seq = format_bb_with_addr(call_inst, call_start, call_end)
                                
                                # Check if this specific call is indirect
                                tokens = [token.text for token in all_insts[call_idx][0]]
                                inst_str = ' '.join(tokens).replace(',', '')
                                call_tokens = inst_str.split()
                                is_indirect = not any(token.startswith('0x') for token in call_tokens[1:])
                                
                                if is_indirect:
                                    f.write(f"{call_seq} -> <addr_start> unknown <addr_end>\n")
                                    pair_count += 1
                                else:
                                    # Direct call - add edge to function entry
                                    # Extract call target from this specific call instruction
                                    call_target = None
                                    for token in call_tokens[1:]:
                                        if token.startswith('0x') or token.startswith('addr_'):
                                            # Extract address
                                            if token.startswith('addr_'):
                                                call_target = int(token.split(':')[1], 16)
                                            else:
                                                call_target = int(token, 16)
                                            break
                                    
                                    if call_target:
                                        target_funcs = bv.get_functions_containing(call_target)
                                        if target_funcs:
                                            for target_func in target_funcs:
                                                target_bb = target_func.get_basic_block_at(call_target)
                                                if target_bb:
                                                    # Check if target BB also contains a call - if so, split it
                                                    if contains_call(target_bb):
                                                        succ_inst, succ_start, succ_end = get_basic_block_seq(target_bb, split_at_call=True)
                                                    else:
                                                        succ_inst, succ_start, succ_end = get_basic_block_seq(target_bb)
                                                    
                                                    if succ_inst:
                                                        succ_seq = format_bb_with_addr(succ_inst, succ_start, succ_end)
                                                        f.write(f"{call_seq} -> {succ_seq}\n")
                                                        pair_count += 1
                                                
                                                # Check if this is a PLT stub (no returns) - if so, create direct edge to continuation
                                                if target_func.start in call_return_map and None in call_return_map[target_func.start]:
                                                    # This is a PLT function, create edge to after-call continuation
                                                    for return_inst, return_start, return_end in call_return_map[target_func.start][None]:
                                                        if return_inst:
                                                            return_seq = format_bb_with_addr(return_inst, return_start, return_end)
                                                            f.write(f"{call_seq} -> {return_seq}\n")
                                                            pair_count += 1
                                                break
                                                break
                            
                            # Move to next segment (after this call)
                            current_idx = call_idx + 1
                        
                        # Part 2: Instructions after the last call (if any)
                        if current_idx < total_insts:
                            after_inst, after_start, after_end = get_instruction_range(bb, current_idx, total_insts - 1)
                            if after_inst:
                                after_call_seq = format_bb_with_addr(after_inst, after_start, after_end)
                                # This becomes a new pseudo-BB, handle its outgoing edges
                                for edge in bb.outgoing_edges:
                                    successor_bb = edge.target
                                    # Check if successor also contains a call - if so, split it
                                    if contains_call(successor_bb):
                                        succ_inst, succ_start, succ_end = get_basic_block_seq(successor_bb, split_at_call=True)
                                    else:
                                        succ_inst, succ_start, succ_end = get_basic_block_seq(successor_bb)
                                    
                                    if succ_inst:
                                        succ_seq = format_bb_with_addr(succ_inst, succ_start, succ_end)
                                        f.write(f"{after_call_seq} -> {succ_seq}\n")
                                        pair_count += 1
                    
                    elif ends_with_ret(bb):
                        # For return instructions: add edges to call sites
                        bb_inst, bb_start, bb_end = get_basic_block_seq(bb)
                        bb_seq = format_bb_with_addr(bb_inst, bb_start, bb_end)
                        if func.start in call_return_map and bb in call_return_map[func.start]:
                            for return_inst, return_start, return_end in call_return_map[func.start][bb]:
                                if return_inst:
                                    return_seq = format_bb_with_addr(return_inst, return_start, return_end)
                                    f.write(f"{bb_seq} -> {return_seq}\n")
                                    pair_count += 1
                    
                    elif is_indirect_branch(bb):
                        # Indirect branch
                        bb_inst, bb_start, bb_end = get_basic_block_seq(bb)
                        bb_seq = format_bb_with_addr(bb_inst, bb_start, bb_end)
                        f.write(f"{bb_seq} -> <addr_start> unknown <addr_end>\n")
                        pair_count += 1
                    
                    elif not contains_call(bb):
                        # Regular control flow (no call, no ret, no indirect)
                        bb_inst, bb_start, bb_end = get_basic_block_seq(bb)
                        bb_seq = format_bb_with_addr(bb_inst, bb_start, bb_end)
                        for edge in bb.outgoing_edges:
                            successor_bb = edge.target
                            # Check if successor contains a call - if so, split it
                            if contains_call(successor_bb):
                                succ_inst, succ_start, succ_end = get_basic_block_seq(successor_bb, split_at_call=True)
                            else:
                                succ_inst, succ_start, succ_end = get_basic_block_seq(successor_bb)
                            
                            if succ_inst:
                                succ_seq = format_bb_with_addr(succ_inst, succ_start, succ_end)
                                f.write(f"{bb_seq} -> {succ_seq}\n")
                                pair_count += 1
        
        print(f"✓ Extracted {pair_count} basic block pairs")
        print(f"✓ Saved to: {output_file}")

if __name__ == "__main__":
    main()
