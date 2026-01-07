#!/usr/bin/env python3
"""Verify JTP labels are correct by loading real functions and checking each jump."""

import sys
import torch
import pickle
import re
sys.path.append('/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')

from dataloader_addressaware import AddressAwareDataset

def main():
    print("="*80)
    print("Loading real functions to verify JTP labels")
    print("="*80)
    
    # Load vocabulary
    vocab_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab_addr.pkl'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"\nVocabulary size: {len(vocab)}")
    
    # Create dataset
    dataset = AddressAwareDataset(
        corpus_path='/data/kun/jtransdata/addr_pretrain.txt',
        vocab=vocab,
        seq_len=512,
        token_mask_prob=0.0,  # No masking to see original tokens
        data_percentage=0.05,
        is_train=True
    )
    
    print(f"Dataset size: {len(dataset)} samples\n")
    
    # Find a sample with multiple jumps
    print("Searching for a function with multiple jump instructions...")
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        jtp_labels = sample['jtp_labels']
        
        # Count jumps
        num_jumps = (jtp_labels != -100).sum().item()
        
        if num_jumps >= 3:  # Find function with at least 3 jumps
            print(f"\nFound function at index {idx} with {num_jumps} jumps\n")
            
            tokens = sample['bert_input']
            seq_len = (tokens != vocab.stoi['<pad>']).sum().item()
            
            # Get original line for comparison
            original_line = dataset.lines[idx]
            
            print("="*80)
            print("ORIGINAL ASSEMBLY CODE")
            print("="*80)
            
            # Parse and display original assembly
            instructions = original_line.split('\t')
            addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):[^)]+\)')
            
            inst_addrs = []
            for inst_idx, inst in enumerate(instructions):
                match = addr_pattern.match(inst.strip())
                if match:
                    opcode = match.group(1)
                    addr = match.group(2)
                    inst_addrs.append((inst_idx, addr, opcode, inst.strip()))
                    print(f"{inst_idx:3d}. [{addr}] {inst.strip()}")
            
            print("\n" + "="*80)
            print("TOKENIZED SEQUENCE WITH JTP LABELS")
            print("="*80)
            
            # Track instruction boundaries
            current_inst = None
            inst_start_pos = {}
            
            for pos in range(seq_len):
                token_id = tokens[pos].item()
                token_str = vocab.itos[token_id]
                jtp_label = jtp_labels[pos].item()
                
                # Track instruction starts (when we see opcode-like tokens)
                if token_str not in ['<sos>', '<eos>', '<pad>'] and pos > 0:
                    prev_token = vocab.itos[tokens[pos-1].item()]
                    if prev_token == '<eos>' or prev_token == '<sos>':
                        current_inst = pos
                        inst_start_pos[pos] = current_inst
                
                marker = ""
                if jtp_label != -100:
                    target_token = vocab.itos[tokens[jtp_label].item()] if jtp_label < seq_len else "OUT_OF_BOUNDS"
                    marker = f" --> JTP: points to position {jtp_label} ({target_token})"
                
                print(f"{pos:3d}: {token_str:15s}{marker}")
            
            print("\n" + "="*80)
            print("JUMP VERIFICATION")
            print("="*80)
            
            # Verify each jump
            nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):')
            
            for pos in range(seq_len):
                token_id = tokens[pos].item()
                token_str = vocab.itos[token_id]
                jtp_label = jtp_labels[pos].item()
                
                if jtp_label != -100 and token_str == 'address':
                    # Find the jump instruction this address belongs to
                    # Look backward to find the opcode
                    jump_opcode = None
                    for back_pos in range(pos-1, max(0, pos-5), -1):
                        back_token = vocab.itos[tokens[back_pos].item()]
                        if back_token in ['jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jge', 'jl', 'jle',
                                         'ja', 'jae', 'jb', 'jbe', 'jo', 'jno', 'js', 'jns', 
                                         'jp', 'jnp', 'jcxz', 'jecxz', 'jrcxz', 'call']:
                            jump_opcode = back_token
                            jump_pos = back_pos
                            break
                    
                    # Find what address this jump is targeting in original code
                    # Look for the instruction containing this position
                    inst_idx = None
                    for idx, (i_idx, addr, opcode, inst_text) in enumerate(inst_addrs):
                        if jump_opcode and opcode == jump_opcode:
                            # Check if this instruction has an address operand
                            if 'address(' in inst_text:
                                match = nested_addr_pattern.search(inst_text)
                                if match:
                                    target_addr = match.group(1)
                                    
                                    # Find which instruction this address points to
                                    target_inst_idx = None
                                    for t_idx, (ti_idx, taddr, topcode, tinst) in enumerate(inst_addrs):
                                        if taddr == target_addr:
                                            target_inst_idx = ti_idx
                                            target_opcode = topcode
                                            break
                                    
                                    # Check if JTP label matches
                                    target_token = vocab.itos[tokens[jtp_label].item()] if jtp_label < seq_len else "OUT_OF_BOUNDS"
                                    
                                    print(f"\n✓ Jump at position {pos} ({token_str}):")
                                    print(f"  Original: {inst_text[:80]}...")
                                    print(f"  Jump opcode: {jump_opcode} at position {jump_pos}")
                                    print(f"  Target address: {target_addr}")
                                    if target_inst_idx is not None:
                                        print(f"  Target instruction: {target_opcode} (instruction #{target_inst_idx})")
                                    print(f"  JTP label: {jtp_label} (token: {target_token})")
                                    
                                    # Verify correctness
                                    if target_opcode and target_token == target_opcode:
                                        print(f"  ✓ CORRECT: JTP label points to '{target_token}' which matches target opcode '{target_opcode}'")
                                    elif target_opcode:
                                        print(f"  ✗ ERROR: JTP label points to '{target_token}' but target opcode is '{target_opcode}'")
                                    
                                    inst_idx = i_idx
                                    break
            
            print("\n" + "="*80)
            print("VERIFICATION COMPLETE")
            print("="*80)
            
            break

if __name__ == "__main__":
    main()
