"""
Export vocabulary to human-readable text file

Usage:
    python export_vocab.py --vocab_path ./vocab_base --output vocab_base.txt
"""

import argparse
import sys
import os

from vocab import WordVocab


def export_vocab_to_txt(vocab_path, output_path):
    """
    Load vocabulary and export to text file with format:
    id\ttoken
    """
    print(f"Loading vocabulary from {vocab_path}...")
    vocab = WordVocab.load_vocab(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")
    
    print(f"Exporting to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("ID\tToken\n")
        f.write("-" * 40 + "\n")
        
        # Write each token with its ID
        for idx, token in enumerate(vocab.itos):
            f.write(f"{idx}\t{token}\n")
    
    print(f"Successfully exported {len(vocab)} tokens to {output_path}")
    
    # Print some statistics
    print("\nVocabulary Statistics:")
    print(f"  Total tokens: {len(vocab)}")
    print(f"  Special tokens:")
    print(f"    <pad>: {vocab.stoi.get('<pad>', 'N/A')}")
    print(f"    <unk>: {vocab.stoi.get('<unk>', 'N/A')}")
    print(f"    <eos>: {vocab.stoi.get('<eos>', 'N/A')}")
    print(f"    <sos>: {vocab.stoi.get('<sos>', 'N/A')}")
    print(f"    <mask>: {vocab.stoi.get('<mask>', 'N/A')}")
    
    # Show first few tokens
    print("\nFirst 10 tokens:")
    for i in range(min(10, len(vocab))):
        print(f"  {i}: {vocab.itos[i]}")


def main():
    parser = argparse.ArgumentParser(description='Export vocabulary to text file')
    parser.add_argument('--vocab_path', type=str, default='./vocab_base',
                       help='Path to vocabulary file (pickle)')
    parser.add_argument('--output', type=str, default='vocab_base.txt',
                       help='Output text file path')
    
    args = parser.parse_args()
    
    # Check if vocab exists
    if not os.path.exists(args.vocab_path):
        print(f"Error: Vocabulary file not found: {args.vocab_path}")
        return 1
    
    export_vocab_to_txt(args.vocab_path, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
