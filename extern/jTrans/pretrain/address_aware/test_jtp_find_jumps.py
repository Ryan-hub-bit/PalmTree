#!/usr/bin/env python3
"""Test script to find and verify JTP labels for jump instructions."""

import sys
import torch
import pickle
sys.path.append('/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')

from dataloader_addressaware import AddressAwareDataset

def main():
    print("="*80)
    print("Finding samples with jump instructions...")
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
        token_mask_prob=0.15,
        data_percentage=0.05,  # Search 5% of data
        is_train=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print("\nSearching for samples with jump instructions...")
    
    jump_opcodes = {'jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jge', 'jl', 'jle', 
                   'ja', 'jae', 'jb', 'jbe', 'jo', 'jno', 'js', 'jns', 'jp', 
                   'jnp', 'jcxz', 'jecxz', 'jrcxz', 'call'}
    
    samples_found = 0
    max_samples = 5
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        
        # Get JTP labels
        jtp_labels = sample['jtp_labels']
        
        # Check if any non-padding JTP labels exist
        if torch.any(jtp_labels != -100):
            samples_found += 1
            
            print("\n" + "="*80)
            print(f"SAMPLE {idx} (with jumps)")
            print("="*80)
            
            # Get tokens
            tokens = sample['bert_input']
            mlm_labels = sample['bert_label']
            
            # Find sequence length (before padding)
            seq_len = (tokens != vocab.stoi['<pad>']).sum().item()
            
            print(f"\nSequence length: {seq_len} tokens")
            
            # Show jump instructions
            print("\n" + "-"*80)
            print("Jump Instructions Found:")
            print("-"*80)
            
            for pos in range(seq_len):
                token_id = tokens[pos].item()
                jtp_label = jtp_labels[pos].item()
                
                if jtp_label != -100:
                    token_str = vocab.itos[token_id]
                    target_token = vocab.itos[tokens[jtp_label].item()] if jtp_label < seq_len else "INVALID"
                    
                    print(f"  Position {pos:3d}: {token_str:15s} → target position {jtp_label:3d} ({target_token})")
            
            # Show surrounding context
            print("\n" + "-"*80)
            print("Full Sequence Context:")
            print("-"*80)
            
            for pos in range(min(seq_len, 50)):  # Show first 50 tokens
                token_id = tokens[pos].item()
                token_str = vocab.itos[token_id]
                jtp_label = jtp_labels[pos].item()
                
                marker = ""
                if jtp_label != -100:
                    marker = f" [JUMP→{jtp_label}]"
                
                print(f"  {pos:3d}: {token_str:15s}{marker}")
            
            if samples_found >= max_samples:
                break
    
    print("\n" + "="*80)
    print(f"Found {samples_found} samples with jump instructions")
    print("="*80)

if __name__ == "__main__":
    main()
