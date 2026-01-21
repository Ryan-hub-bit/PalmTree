#!/usr/bin/env python3
"""
Build vocabulary from jTrans_instr pretraining text file.

Extracts all unique tokens from the data and creates vocab.txt with:
1. instr_addr_0 to instr_addr_200 (instruction-level jump addresses)
2. Special tokens: instr_addr_exceed, unknown_instr_addr
3. All unique assembly tokens found in the data
4. Special BERT tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
"""

import argparse
from collections import Counter
from tqdm import tqdm


def build_vocab_from_text(input_file, output_file, min_freq=1):
    """
    Build vocabulary from text file.
    
    Args:
        input_file: Path to instr_pretrain.txt
        output_file: Path to output vocab.txt
        min_freq: Minimum frequency for a token to be included
    """
    print(f"Reading tokens from: {input_file}")
    
    # Count all tokens
    token_counts = Counter()
    total_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting tokens"):
            tokens = line.strip().split()
            token_counts.update(tokens)
            total_lines += 1
    
    print(f"\nProcessed {total_lines:,} functions")
    print(f"Found {len(token_counts):,} unique tokens")
    
    # Build vocabulary list
    vocab = []
    
    # 1. Add instruction jump addresses (0-200)
    print("\nAdding instruction jump addresses...")
    for i in range(201):  # 0 to 200 inclusive
        vocab.append(f'instr_addr_{i}')
    
    # 2. Add special jump tokens
    vocab.append('instr_addr_exceed')
    vocab.append('unknown_instr_addr')
    
    print(f"  Added {len(vocab)} jump address tokens")
    
    # 3. Add BERT special tokens
    bert_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    vocab.extend(bert_tokens)
    print(f"  Added {len(bert_tokens)} BERT special tokens")
    
    # 4. Add frequent assembly tokens from data
    # Filter out the jump address tokens we already added
    jump_tokens = {f'instr_addr_{i}' for i in range(201)}
    jump_tokens.update(['instr_addr_exceed', 'unknown_instr_addr'])
    
    assembly_tokens = []
    for token, count in token_counts.most_common():
        if token not in jump_tokens and token not in bert_tokens:
            if count >= min_freq:
                assembly_tokens.append(token)
    
    vocab.extend(assembly_tokens)
    print(f"  Added {len(assembly_tokens)} assembly tokens (min_freq={min_freq})")
    
    # Write vocabulary
    print(f"\nWriting vocabulary to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(token + '\n')
    
    print(f"\n✅ Vocabulary created!")
    print(f"   Total vocabulary size: {len(vocab):,}")
    print(f"   - Instruction addresses: 201 (instr_addr_0 to instr_addr_200)")
    print(f"   - Special jump tokens: 2 (instr_addr_exceed, unknown_instr_addr)")
    print(f"   - BERT tokens: {len(bert_tokens)}")
    print(f"   - Assembly tokens: {len(assembly_tokens):,}")
    
    # Show token frequency statistics
    print(f"\n📊 Token Statistics:")
    print(f"   Most common tokens:")
    for token, count in token_counts.most_common(10):
        print(f"     {token:20s} {count:,}")
    
    # Check jump address usage
    instr_addr_counts = {f'instr_addr_{i}': token_counts.get(f'instr_addr_{i}', 0) 
                         for i in range(201)}
    used_addrs = sum(1 for c in instr_addr_counts.values() if c > 0)
    print(f"\n   Jump address usage:")
    print(f"     Used: {used_addrs}/201 instruction addresses")
    print(f"     instr_addr_exceed: {token_counts.get('instr_addr_exceed', 0):,}")
    print(f"     unknown_instr_addr: {token_counts.get('unknown_instr_addr', 0):,}")
    
    return len(vocab)


def main():
    parser = argparse.ArgumentParser(
        description='Build vocabulary from jTrans_instr pretraining data'
    )
    parser.add_argument(
        '--input-file',
        default='/data/kun/jtrans_instr/instr_pretrain.txt',
        help='Input text file with tokenized functions'
    )
    parser.add_argument(
        '--output-file',
        default='/home/kun/Document/AAE/extern/jTrans_instr/jtrans_tokenizer/vocab.txt',
        help='Output vocabulary file'
    )
    parser.add_argument(
        '--min-freq',
        type=int,
        default=1,
        help='Minimum frequency for a token to be included (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    import os
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print("="*70)
    print("jTrans_instr Vocabulary Builder")
    print("="*70)
    
    vocab_size = build_vocab_from_text(
        args.input_file,
        args.output_file,
        args.min_freq
    )
    
    print(f"\n{'='*70}")
    print(f"Vocabulary saved to: {args.output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
