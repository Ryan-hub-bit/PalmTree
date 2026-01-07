#!/usr/bin/env python3
"""Simple verification of JTP labels - show each jump and its target."""

import sys
import torch
import pickle
import re
sys.path.append('/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')

from dataloader_addressaware import AddressAwareDataset

def main():
    print("="*80)
    print("JTP Label Verification - Real Function Analysis")
    print("="*80)
    
    # Load vocabulary
    vocab_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab_addr.pkl'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Create dataset with no masking
    dataset = AddressAwareDataset(
        corpus_path='/data/kun/jtransdata/addr_pretrain.txt',
        vocab=vocab,
        seq_len=512,
        token_mask_prob=0.0,  # No masking
        data_percentage=0.05,
        is_train=True
    )
    
    print(f"\nSearching for function with multiple jumps...")
    
    # Find a good sample
    for idx in range(len(dataset)):
        sample = dataset[idx]
        jtp_labels = sample['jtp_labels']
        num_jumps = (jtp_labels != -100).sum().item()
        
        if num_jumps >= 5:
            print(f"Found function with {num_jumps} jumps (index {idx})\n")
            
            tokens = sample['bert_input']
            seq_len = (tokens != vocab.stoi['<pad>']).sum().item()
            original_line = dataset.lines[idx]
            
            # Parse original to get instruction addresses
            instructions = original_line.split('\t')
            addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):[^)]+\)')
            nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):')
            
            # Build address -> instruction mapping
            addr_to_inst = {}
            for inst_idx, inst in enumerate(instructions):
                match = addr_pattern.match(inst.strip())
                if match:
                    opcode = match.group(1)
                    addr = match.group(2)
                    addr_to_inst[addr] = (inst_idx, opcode, inst.strip())
            
            print("="*80)
            print("TOKENIZED FUNCTION")
            print("="*80)
            for pos in range(min(seq_len, 100)):
                token = vocab.itos[tokens[pos].item()]
                jtp = jtp_labels[pos].item()
                if jtp != -100:
                    target = vocab.itos[tokens[jtp].item()]
                    print(f"{pos:3d}: {token:12s} [JTP→{jtp:3d} ({target})]")
                else:
                    print(f"{pos:3d}: {token:12s}")
            
            print("\n" + "="*80)
            print("JUMP VERIFICATION")
            print("="*80)
            
            # For each jump, verify it
            for pos in range(seq_len):
                token = vocab.itos[tokens[pos].item()]
                jtp = jtp_labels[pos].item()
                
                if token == 'address' and jtp != -100:
                    # Find the jump opcode before this address
                    jump_opcode = None
                    for back in range(pos-1, max(0, pos-5), -1):
                        t = vocab.itos[tokens[back].item()]
                        if t in ['jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jge', 'jl', 'jle',
                                'ja', 'jae', 'jb', 'jbe', 'jo', 'jno', 'js', 'jns',
                                'jp', 'jnp', 'jcxz', 'jecxz', 'jrcxz', 'call']:
                            jump_opcode = t
                            break
                    
                    # Find what instruction this position belongs to
                    # Count <eos> tokens to find instruction index
                    inst_idx = 0
                    for i in range(pos):
                        if vocab.itos[tokens[i].item()] == '<eos>':
                            inst_idx += 1
                    
                    # Get the original instruction
                    if inst_idx < len(instructions):
                        orig_inst = instructions[inst_idx].strip()
                        
                        # Extract target address from original instruction
                        target_match = nested_addr_pattern.search(orig_inst)
                        if target_match:
                            target_addr = target_match.group(1)
                            
                            # What instruction does this address point to?
                            if target_addr in addr_to_inst:
                                target_inst_idx, target_opcode, target_inst = addr_to_inst[target_addr]
                                
                                # What token does JTP point to?
                                jtp_token = vocab.itos[tokens[jtp].item()]
                                
                                # Find jump opcode position
                                jump_pos = None
                                for back in range(pos-1, max(0, pos-5), -1):
                                    t = vocab.itos[tokens[back].item()]
                                    if t == jump_opcode:
                                        jump_pos = back
                                        break
                                
                                # Show context around the jump
                                print(f"\n{'='*70}")
                                print(f"JUMP at token positions {jump_pos}-{pos}")
                                print(f"{'='*70}")
                                
                                # Show jump instruction tokens
                                print(f"  Jump instruction tokens:")
                                for p in range(max(0, jump_pos), min(pos+2, seq_len)):
                                    t = vocab.itos[tokens[p].item()]
                                    if p == pos:
                                        print(f"    Position {p:3d}: '{t}' ← ADDRESS TOKEN (has JTP label)")
                                    elif p == jump_pos:
                                        print(f"    Position {p:3d}: '{t}' ← JUMP OPCODE")
                                    else:
                                        print(f"    Position {p:3d}: '{t}'")
                                
                                print(f"\n  Original assembly instruction #{inst_idx}:")
                                print(f"    {orig_inst}")
                                
                                print(f"\n  Target information:")
                                print(f"    Target address: {target_addr}")
                                print(f"    Target instruction #{target_inst_idx}: {target_opcode}")
                                print(f"    {target_inst}")
                                
                                print(f"\n  JTP label verification:")
                                print(f"    JTP label value: {jtp}")
                                print(f"    Token at position {jtp}: '{jtp_token}'")
                                
                                # Show context around target
                                print(f"\n  Target instruction tokens:")
                                for p in range(max(0, jtp), min(jtp+5, seq_len)):
                                    t = vocab.itos[tokens[p].item()]
                                    if p == jtp:
                                        print(f"    Position {p:3d}: '{t}' ← JTP POINTS HERE (target opcode)")
                                    else:
                                        print(f"    Position {p:3d}: '{t}'")
                                
                                if jtp_token == target_opcode:
                                    print(f"\n  ✓ CORRECT: JTP label points to '{jtp_token}' which matches target opcode")
                                else:
                                    print(f"\n  ✗ MISMATCH: JTP points to '{jtp_token}' but target opcode is '{target_opcode}'")
            
            break

if __name__ == "__main__":
    main()
