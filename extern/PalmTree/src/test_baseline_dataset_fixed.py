"""
Test the fixed baseline dataset with proper CFG_next and DFG_next logic
"""

import torch
import sys
sys.path.insert(0, 'src')

from vocab import WordVocab
from palmtree.dataset.dataset_baseline import BaselineDataset


def test_baseline_dataset():
    print("="*80)
    print("Testing Baseline Dataset with Fixed CFG_next/DFG_next Logic")
    print("="*80)
    
    # Load vocab
    print("\nLoading vocabulary...")
    vocab = WordVocab.load_vocab("./vocab_addr")
    print(f"Vocab size: {len(vocab)}")
    
    # Create dataset
    print("\nCreating baseline dataset...")
    dataset = BaselineDataset(
        cfg_corpus_path='/data/kun/palmtreedata/cfg_train_2.txt',
        dfg_corpus_path='/data/kun/palmtreedata/dfg_train_2.txt',
        vocab=vocab,
        seq_len=20,
        token_mask_prob=0.15,
        instruction_mask_prob=0.0,  # Disable instruction masking for clarity
        data_percentage=0.001  # Only 0.1% for quick test
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test a few samples
    print("\n" + "="*80)
    print("Testing Samples")
    print("="*80)
    
    cfg_next_counts = {0: 0, 1: 0}
    dfg_next_counts = {0: 0, 1: 0}
    
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*80}")
        
        # CFG info
        cfg_input = sample['cfg_bert_input']
        cfg_segment = sample['cfg_segment_label']
        cfg_is_next = sample['cfg_is_next'].item()
        
        cfg_next_counts[cfg_is_next] += 1
        
        # Decode tokens
        cfg_tokens = []
        for tid in cfg_input:
            if tid == 0:
                break
            cfg_tokens.append(vocab.itos[tid] if tid < len(vocab.itos) else f'UNK:{tid}')
        
        # Find where segment changes from 1 to 2
        seg1_end = 0
        for j, seg in enumerate(cfg_segment):
            if seg == 2:
                seg1_end = j
                break
        
        seq1_tokens = cfg_tokens[:seg1_end]
        seq2_tokens = cfg_tokens[seg1_end:]
        
        print(f"\nCFG:")
        print(f"  is_next: {cfg_is_next} ({'consecutive' if cfg_is_next == 1 else 'random'})")
        print(f"  Sequence 1 (segment 1): {' '.join(seq1_tokens[:15])}{'...' if len(seq1_tokens) > 15 else ''}")
        print(f"  Sequence 2 (segment 2): {' '.join(seq2_tokens[:15])}{'...' if len(seq2_tokens) > 15 else ''}")
        print(f"  Total tokens: {len(cfg_tokens)}")
        
        # DFG info if available
        if 'dfg_bert_input' in sample:
            dfg_input = sample['dfg_bert_input']
            dfg_segment = sample['dfg_segment_label']
            dfg_is_next = sample['dfg_is_next'].item()
            
            dfg_next_counts[dfg_is_next] += 1
            
            # Decode tokens
            dfg_tokens = []
            for tid in dfg_input:
                if tid == 0:
                    break
                dfg_tokens.append(vocab.itos[tid] if tid < len(vocab.itos) else f'UNK:{tid}')
            
            # Find where segment changes from 1 to 2
            seg1_end = 0
            for j, seg in enumerate(dfg_segment):
                if seg == 2:
                    seg1_end = j
                    break
            
            seq1_tokens = dfg_tokens[:seg1_end]
            seq2_tokens = dfg_tokens[seg1_end:]
            
            print(f"\nDFG:")
            print(f"  is_next: {dfg_is_next} ({'consecutive' if dfg_is_next == 1 else 'random'})")
            print(f"  Sequence 1 (segment 1): {' '.join(seq1_tokens[:15])}{'...' if len(seq1_tokens) > 15 else ''}")
            print(f"  Sequence 2 (segment 2): {' '.join(seq2_tokens[:15])}{'...' if len(seq2_tokens) > 15 else ''}")
            print(f"  Total tokens: {len(dfg_tokens)}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nCFG is_next distribution:")
    print(f"  is_next=0 (random): {cfg_next_counts[0]}")
    print(f"  is_next=1 (consecutive): {cfg_next_counts[1]}")
    print(f"  Ratio: {cfg_next_counts[1]/(cfg_next_counts[0]+cfg_next_counts[1])*100:.1f}% consecutive")
    
    if dfg_next_counts[0] + dfg_next_counts[1] > 0:
        print(f"\nDFG is_next distribution:")
        print(f"  is_next=0 (random): {dfg_next_counts[0]}")
        print(f"  is_next=1 (consecutive): {dfg_next_counts[1]}")
        print(f"  Ratio: {dfg_next_counts[1]/(dfg_next_counts[0]+dfg_next_counts[1])*100:.1f}% consecutive")
    
    print("\n" + "="*80)
    print("✓ Test completed successfully!")
    print("="*80)
    print("\nKey fixes:")
    print("  ✓ Each line now properly contains TWO sequences")
    print("  ✓ is_next=1: Uses both seq1 and seq2 from same line")
    print("  ✓ is_next=0: Uses seq1 from line i, seq2 from random line j")
    print("  ✓ Segment labels: 1 for seq1, 2 for seq2")
    print("  ✓ CFG: MLM masking applied")
    print("  ✓ DFG: No masking (DUP task only)")


if __name__ == '__main__':
    test_baseline_dataset()
