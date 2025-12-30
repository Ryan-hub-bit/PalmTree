"""
Debug script to check masking logic
"""

import torch
import sys
sys.path.insert(0, 'src')

from vocab import WordVocab
from address_aware.dataloader_addressaware import InstructionMaskingDataset

def main():
    # Load vocab
    print("Loading vocabulary...")
    vocab = WordVocab.load_vocab("./vocab_addr")
    print(f"Vocab size: {len(vocab)}")
    
    # Create dataset with minimal masking for testing
    print("\nCreating dataset with instruction_mask_prob=0.25...")
    dataset = InstructionMaskingDataset(
        cfg_corpus_path='/data/kun/palmtreedata/cfg_train_2.txt',
        dfg_corpus_path=None,  # No DFG needed for this test
        vocab=vocab,
        seq_len=512,
        encoding="utf-8",
        on_memory=True,
        instruction_mask_prob=0.25,  # Mask 25% of instructions
        token_mask_prob=0.0,  # No token-level masking for IMC
        data_percentage=0.01,  # Only load 1% for quick testing
        enable_imd=False  # Disable DFG instruction masking
    )
    
    # Get one sample
    print("\nGetting sample 0...")
    sample = dataset[0]
    
    imc_input = sample['imc']['bert_input']
    imc_label = sample['imc']['bert_label']
    
    print("\n" + "="*80)
    print("IMC Sample 0 - Detailed Analysis")
    print("="*80)
    
    print("\nFirst 30 tokens:")
    print("Index | Token ID | Token         | Label   | Masked?")
    print("-" * 60)
    
    for i in range(min(30, len(imc_input))):
        tid = imc_input[i].item()
        label = imc_label[i].item()
        
        if tid == 0:  # padding
            break
        
        token = vocab.itos[tid] if 0 <= tid < len(vocab.itos) else f'UNK:{tid}'
        is_masked = "YES" if label != -1 else "NO"
        
        print(f"{i:5d} | {tid:8d} | {token:13s} | {label:7d} | {is_masked}")
    
    # Count statistics
    non_pad_count = 0
    masked_count = 0
    
    for i in range(len(imc_input)):
        tid = imc_input[i].item()
        label = imc_label[i].item()
        
        if tid == 0:  # padding
            break
        
        non_pad_count += 1
        if label != -1:
            masked_count += 1
    
    print(f"\n{'='*80}")
    print("STATISTICS:")
    print(f"  Total non-padding tokens: {non_pad_count}")
    print(f"  Masked tokens: {masked_count}")
    print(f"  Masking rate: {masked_count/non_pad_count*100:.1f}%")
    print(f"{'='*80}")
    
    # Check specific tokens
    sos_idx = vocab.stoi.get('<sos>', -1)
    eos_idx = vocab.stoi.get('<eos>', -1)
    mask_idx = vocab.stoi.get('<mask>', -1)
    
    print(f"\nSpecial token IDs:")
    print(f"  <sos>: {sos_idx}")
    print(f"  <eos>: {eos_idx}")
    print(f"  <mask>: {mask_idx}")
    
    # Check if <sos> and <eos> are masked
    print(f"\nChecking special tokens in sample:")
    for i in range(min(30, len(imc_input))):
        tid = imc_input[i].item()
        label = imc_label[i].item()
        
        if tid == sos_idx:
            print(f"  Position {i}: <sos> - label={label} {'(CORRECTLY UNMASKED)' if label == -1 else '(ERROR: SHOULD NOT BE MASKED!)'}")
        elif tid == eos_idx:
            print(f"  Position {i}: <eos> - label={label} {'(CORRECTLY UNMASKED)' if label == -1 else '(ERROR: SHOULD NOT BE MASKED!)'}")

if __name__ == "__main__":
    main()
