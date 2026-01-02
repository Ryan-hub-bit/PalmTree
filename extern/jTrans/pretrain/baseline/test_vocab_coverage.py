"""
Test vocabulary coverage for baseline training data.

Checks if all tokens in pretrain_train.txt are covered by baseline_vocab.txt
"""

import sys
from collections import Counter
from transformers import BertTokenizer

def test_vocab_coverage(data_path, tokenizer_path, num_samples=1000):
    """Test vocabulary coverage on training data."""
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: PAD={tokenizer.pad_token_id}, UNK={tokenizer.unk_token_id}, "
          f"CLS={tokenizer.cls_token_id}, SEP={tokenizer.sep_token_id}, MASK={tokenizer.mask_token_id}")
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
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id == tokenizer.unk_token_id:
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
    
    if unknown_tokens:
        print(f"\n{'='*80}")
        print(f"TOP 20 UNKNOWN TOKENS:")
        print(f"{'='*80}")
        for token, count in unknown_tokens.most_common(20):
            print(f"{token:30s} : {count:6d} occurrences")
        
        print(f"\n{'='*80}")
        print(f"❌ VOCABULARY INCOMPLETE - {len(unknown_tokens)} token types are missing!")
        print(f"{'='*80}")
        return False
    else:
        print(f"\n{'='*80}")
        print(f"✓ VOCABULARY COMPLETE - All tokens are covered!")
        print(f"{'='*80}")
        return True


if __name__ == '__main__':
    data_path = '/data/kun/jtransdata/pretrain_train.txt'
    tokenizer_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/baseline'
    
    # Test with 10000 samples for thorough coverage test
    success = test_vocab_coverage(data_path, tokenizer_path, num_samples=10000)
    
    sys.exit(0 if success else 1)
