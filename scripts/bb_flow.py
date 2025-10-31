import sys
import binaryninja

def get_token_from_inst(inst):
    # get tokens from an instruction
    # e.g. mov eax, [ebp+0x10] -> ['mov', 'eax', ',', '[', 'ebp', '+', '0x10', ']']
    # We will join them with spaces, removing the comma.
    tokens = [token.text for token in inst[0] if token.text != ',']
    return tokens

def get_basic_block_seq(bb, split_at_call=False, only_call=False, after_call=False):
    # get instruction sequence from a basic block
    # the instructions are split by '\t'
    # split_at_call: if True and BB contains call, only return instructions up to and including the call
    # only_call: return only the call instruction
    # after_call: return only instructions after the call
    inst_seq = []
    instructions = list(bb)
    
    if split_at_call or only_call or after_call:
        # Find if there's a call instruction
        call_idx = -1
        for i, inst in enumerate(instructions):
            tokens = [token.text for token in inst[0] if token.text != ',']
            if tokens and tokens[0].lower() == 'call':
                call_idx = i
                break
        
        if call_idx >= 0:
            if only_call:
                # Return only the call instruction
                instructions = [instructions[call_idx]]
            elif after_call:
                # Return instructions after the call
                if call_idx + 1 < len(instructions):
                    instructions = instructions[call_idx + 1:]
                else:
                    return ''  # No instructions after call
            else:  # split_at_call
                # Return up to and including the call
                instructions = instructions[:call_idx + 1]
    
    # Iterate through each instruction
    for inst in instructions:
        # Get disassembly tokens for this instruction
        tokens = [token.text for token in inst[0] if token.text != ',']
        inst_seq.append(' '.join(tokens))
    return '\t'.join(inst_seq)

def is_indirect_branch(bb):
    # Check if basic block ends with an indirect call or jump
    # Returns True for: call rax, jmp rax, call [mem], jmp [mem], etc.
    # Returns False for: call 0x401000, jmp 0x401000, je 0x401000, etc.
    if len(bb) == 0:
        return False
    
    # Get the last instruction
    last_inst = list(bb)[-1]
    tokens = [token.text for token in last_inst[0]]
    
    if not tokens:
        return False
    
    # Check if it's a call or jump instruction
    mnemonic = tokens[0].lower()
    if mnemonic not in ['call', 'jmp', 'ret', 'retn']:
        return False
    
    # ret/retn are always indirect
    if mnemonic in ['ret', 'retn']:
        return False  # Don't treat ret as indirect branch to [Unknown]
    
    # For call/jmp, check if the target is indirect
    # Direct: call 0x401000 (has 0x prefix)
    # Indirect: call rax, call [mem], jmp rax, etc.
    has_direct_address = any(token.startswith('0x') for token in tokens[1:])
    
    return not has_direct_address

def ends_with_call(bb):
    # Check if basic block ends with a call instruction (direct or indirect)
    if len(bb) == 0:
        return False
    
    # Get the last instruction
    last_inst = list(bb)[-1]
    tokens = [token.text for token in last_inst[0] if token.text != ',']
    
    if not tokens:
        return False
    
    return tokens[0].lower() == 'call'

def ends_with_ret(bb):
    # Check if basic block ends with a return instruction
    if len(bb) == 0:
        return False
    
    # Get the last instruction
    last_inst = list(bb)[-1]
    tokens = [token.text for token in last_inst[0] if token.text != ',']
    
    if not tokens:
        return False
    
    mnemonic = tokens[0].lower()
    return mnemonic in ['ret', 'retn']

def contains_call(bb):
    # Check if basic block contains a call instruction (anywhere, not just at the end)
    for inst in bb:
        tokens = [token.text for token in inst[0] if token.text != ',']
        if tokens and tokens[0].lower() == 'call':
            return True
    return False

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
        if tokens and tokens[0].lower() == 'call':
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
        call_return_map = {}  # Maps (func_addr, ret_bb) -> [return_target_sequences]
        
        for func in bv.functions:
            for bb in func:
                if contains_call(bb):
                    # Get the instructions after the call (within same BB)
                    after_call_seq = get_basic_block_seq(bb, after_call=True)
                    
                    # Get the call target
                    call_addrs = get_call_targets(bb)
                    if call_addrs:
                        for call_addr in call_addrs:
                            target_funcs = bv.get_functions_containing(call_addr)
                            if target_funcs:
                                for target_func in target_funcs:
                                    # Map all return blocks in this function to the after-call sequence
                                    if target_func.start not in call_return_map:
                                        call_return_map[target_func.start] = {}
                                    
                                    for ret_bb in target_func.basic_blocks:
                                        if ends_with_ret(ret_bb):
                                            if ret_bb not in call_return_map[target_func.start]:
                                                call_return_map[target_func.start][ret_bb] = []
                                            
                                            # The return should go to the after-call sequence (instructions after the call)
                                            if after_call_seq:
                                                # Return to instructions after call in same BB
                                                call_return_map[target_func.start][ret_bb].append(after_call_seq)
                                            else:
                                                # No instructions after call in same BB, use next BB
                                                # But split it if it contains a call too
                                                next_bb = get_next_instruction_bb(bv, bb)
                                                if next_bb:
                                                    if contains_call(next_bb):
                                                        next_seq = get_basic_block_seq(next_bb, split_at_call=True)
                                                    else:
                                                        next_seq = get_basic_block_seq(next_bb)
                                                    if next_seq:
                                                        call_return_map[target_func.start][ret_bb].append(next_seq)
                                    break
                            break
        
        # Second pass: generate basic block pairs
        with open(output_file, 'w') as f:
            pair_count = 0
            for func in bv.functions:
                for bb in func:
                    # Check if BB contains a call - if so, split it
                    if contains_call(bb):
                        # Part 1: Instructions up to and including the call
                        call_seq = get_basic_block_seq(bb, split_at_call=True)
                        
                        if call_seq:
                            # Check if call is indirect
                            if is_indirect_branch(bb) if ends_with_call(bb) else False:
                                f.write(f"{call_seq} -> [Unknown]\n")
                                pair_count += 1
                            else:
                                # Direct call - add edge to function entry
                                call_addrs = get_call_targets(bb)
                                if call_addrs:
                                    for call_addr in call_addrs:
                                        target_funcs = bv.get_functions_containing(call_addr)
                                        if target_funcs:
                                            for target_func in target_funcs:
                                                target_bb = target_func.get_basic_block_at(call_addr)
                                                if target_bb:
                                                    # Check if target BB also contains a call - if so, split it
                                                    if contains_call(target_bb):
                                                        succ_seq = get_basic_block_seq(target_bb, split_at_call=True)
                                                    else:
                                                        succ_seq = get_basic_block_seq(target_bb)
                                                    
                                                    if succ_seq:
                                                        f.write(f"{call_seq} -> {succ_seq}\n")
                                                        pair_count += 1
                                                        break
                                            break
                        
                        # Part 2: Instructions after the call (if any)
                        after_call_seq = get_basic_block_seq(bb, after_call=True)
                        if after_call_seq:
                            # This becomes a new pseudo-BB, handle its outgoing edges
                            for edge in bb.outgoing_edges:
                                successor_bb = edge.target
                                # Check if successor also contains a call - if so, split it
                                if contains_call(successor_bb):
                                    succ_seq = get_basic_block_seq(successor_bb, split_at_call=True)
                                else:
                                    succ_seq = get_basic_block_seq(successor_bb)
                                
                                if succ_seq:
                                    f.write(f"{after_call_seq} -> {succ_seq}\n")
                                    pair_count += 1
                    
                    elif ends_with_ret(bb):
                        # For return instructions: add edges to call sites
                        bb_seq = get_basic_block_seq(bb)
                        if func.start in call_return_map and bb in call_return_map[func.start]:
                            for return_seq in call_return_map[func.start][bb]:
                                if return_seq:
                                    f.write(f"{bb_seq} -> {return_seq}\n")
                                    pair_count += 1
                    
                    elif is_indirect_branch(bb):
                        # Indirect branch
                        bb_seq = get_basic_block_seq(bb)
                        f.write(f"{bb_seq} -> [Unknown]\n")
                        pair_count += 1
                    
                    elif not contains_call(bb):
                        # Regular control flow (no call, no ret, no indirect)
                        bb_seq = get_basic_block_seq(bb)
                        for edge in bb.outgoing_edges:
                            successor_bb = edge.target
                            # Check if successor contains a call - if so, split it
                            if contains_call(successor_bb):
                                succ_seq = get_basic_block_seq(successor_bb, split_at_call=True)
                            else:
                                succ_seq = get_basic_block_seq(successor_bb)
                            
                            if succ_seq:
                                f.write(f"{bb_seq} -> {succ_seq}\n")
                                pair_count += 1
        
        print(f"✓ Extracted {pair_count} basic block pairs")
        print(f"✓ Saved to: {output_file}")

if __name__ == "__main__":
    main()
