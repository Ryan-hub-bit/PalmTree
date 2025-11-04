#!/usr/bin/env python3
"""
Extend PalmTree vocabulary with address tokens
"""

import pickle
import sys
import os

def extend_palmtree_vocab(original_vocab_path: str, output_vocab_path: str):
    """
    Add address tokens to PalmTree vocabulary (WordVocab class)
    
    Args:
        original_vocab_path: Path to PalmTree's vocab file
        output_vocab_path: Path to save extended vocab
    """
    
    # Load original vocabulary (WordVocab object)
    print(f"Loading vocabulary from: {original_vocab_path}")
    with open(original_vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    original_size = len(vocab)
    print(f"Original vocabulary size: {original_size}")
    print(f"Vocabulary type: {type(vocab).__name__}")
    
    # Check special tokens
    print(f"\nSpecial tokens:")
    print(f"  <pad>: {vocab.stoi.get('<pad>', 'NOT FOUND')}")
    print(f"  <unk>: {vocab.stoi.get('<unk>', 'NOT FOUND')}")
    print(f"  <mask>: {vocab.stoi.get('<mask>', 'NOT FOUND')}")
    
    # Address tokens to add
    new_tokens = [
        '<seq>',        # Instruction separator
        'addr_start',
        'addr_end', 
        'addr_code',
        'addr_data'
    ]
    
    # Add new tokens to vocab
    print(f"\nAdding new tokens:")
    for token in new_tokens:
        if token not in vocab.stoi:
            # Add to itos (index to string)
            vocab.itos.append(token)
            # Add to stoi (string to index)
            vocab.stoi[token] = len(vocab.itos) - 1
            print(f"  Added: {token} -> {vocab.stoi[token]}")
        else:
            print(f"  Already exists: {token} -> {vocab.stoi[token]}")
    
    # Save extended vocabulary
    print(f"\nSaving extended vocabulary to: {output_vocab_path}")
    with open(output_vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    new_size = len(vocab)
    print(f"New vocabulary size: {new_size}")
    print(f"Added {new_size - original_size} new tokens")
    
    return vocab

def verify_extended_vocab(vocab_path: str):
    """Verify the extended vocabulary"""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Type: {type(vocab).__name__}")
    print("\nSpecial and address tokens:")
    for token in ['<seq>', 'addr_start', 'addr_end', 'addr_code', 'addr_data']:
        if token in vocab.stoi:
            print(f"  {token}: {vocab.stoi[token]}")
        else:
            print(f"  {token}: NOT FOUND")
    
    print(f"\nSample vocabulary (first 10 and last 10 tokens):")
    print(f"  First 10: {vocab.itos[:10]}")
    print(f"  Last 10: {vocab.itos[-10:]}")

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python extend_vocab.py <path_to_palmtree_vocab> [output_path]")
        print("\nExample:")
        print("  python extend_vocab.py ../pre-trained_model/palmtree/vocab")
        print("  python extend_vocab.py ../pre-trained_model/palmtree/vocab vocab_extended")
        sys.exit(1)
    
    original_vocab = sys.argv[1]
    output_vocab = sys.argv[2] if len(sys.argv) > 2 else "vocab_extended"
    
    if not os.path.exists(original_vocab):
        print(f"Error: Vocabulary file not found: {original_vocab}")
        sys.exit(1)
    
    # Extend vocabulary
    extended_vocab = extend_palmtree_vocab(original_vocab, output_vocab)
    
    # Verify
    verify_extended_vocab(output_vocab)
    
    print(f"\n✓ Done! Use '{output_vocab}' as your vocabulary file.")
