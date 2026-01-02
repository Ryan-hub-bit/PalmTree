"""
Simple vocabulary coverage test without transformers library.

Checks if all tokens in pretrain_train.txt are covered by baseline_vocab.txt
"""

import sys
from collections import Counter

def load_vocab(vocab_path):
    """Load vocabulary from file."""
    vocab = set()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token:
                vocab.add(token)
    return vocab

def test_vocab_coverage(data_path, vocab_path, num_samples=10000):
    """Test vocabulary coverage on training data."""
    
    print(f"Loading vocabulary from {vocab_path}...")
    vocab = load_vocab(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Check special tokens
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    print("\nSpecial tokens:")
    for token in special_tokens:
        status = "✓" if token in vocab else "✗"
        print(f"  {status} {token}")
    print()
    
    # Sample data
    print(f"Reading {num_samples} samples from {data_path}...")
    total_tokens = 0
    unknown_tokens = Counter()
    all_tokens = Counter()
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            tokens = line.strip().split()
            all_tokens.update(tokens)
            total_tokens += len(tokens)
            
            # Check each token
            for token in tokens:
                if token not in vocab:
                    unknown_tokens[token] += 1
    
    # Report results
    print(f"\n{'='*80}")
    print(f"VOCABULARY COVERAGE REPORT")
    print(f"{'='*80}")
    print(f"Samples analyzed: {min(i+1, num_samples)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Unique tokens: {len(all_tokens)}")
    print(f"Unknown tokens (UNK): {sum(unknown_tokens.values())} ({100*sum(unknown_tokens.values())/total_tokens:.2f}%)")
    print(f"Unique unknown types: {len(unknown_tokens)}")
    print(f"Coverage: {100*(1-sum(unknown_tokens.values())/total_tokens):.2f}%")
    
    if unknown_tokens:
        print(f"\n{'='*80}")
        print(f"TOP 30 UNKNOWN TOKENS:")
        print(f"{'='*80}")
        for token, count in unknown_tokens.most_common(30):
            print(f"{token:40s} : {count:6d} occurrences")
        
        print(f"\n{'='*80}")
        if len(unknown_tokens) > 100:
            print(f"❌ VOCABULARY INCOMPLETE - {len(unknown_tokens)} token types are missing!")
        else:
            print(f"⚠️  VOCABULARY MOSTLY COMPLETE - Only {len(unknown_tokens)} token types missing")
            print(f"   Coverage is {100*(1-sum(unknown_tokens.values())/total_tokens):.2f}%")
        print(f"{'='*80}")
        return sum(unknown_tokens.values())/total_tokens < 0.01  # Less than 1% unknown is acceptable
    else:
        print(f"\n{'='*80}")
        print(f"✓ VOCABULARY COMPLETE - All tokens are covered!")
        print(f"{'='*80}")
        return True


if __name__ == '__main__':
    data_path = '/data/kun/jtransdata/pretrain_train.txt'
    vocab_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/baseline/baseline_vocab.txt'
    
    # Test with 10000 samples for thorough coverage test
    success = test_vocab_coverage(data_path, vocab_path, num_samples=10000)
    
    sys.exit(0 if success else 1)
