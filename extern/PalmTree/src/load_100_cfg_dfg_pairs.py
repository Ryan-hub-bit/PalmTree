"""
Load 100 CFG-DFG pairs and display their text and tensor representations

This script loads pairs from the baseline dataset and shows:
- Original instruction text (decoded from tokens)
- BERT input tensors
- BERT label tensors (with -100 for non-masked, token_id for masked)
- Segment labels
- is_next values
"""

import torch
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '../../strupos')

from vocab import WordVocab
from palmtree.dataset.dataset_baseline import BaselineDataset


def decode_tokens(token_ids, vocab):
    """Decode token IDs back to words"""
    tokens = []
    for tid in token_ids:
        if tid == vocab.stoi.get('<pad>', 0):
            break  # Stop at padding
        elif tid in vocab.itos:
            tokens.append(vocab.itos[tid])
        else:
            tokens.append(f'<UNK:{tid}>')
    return tokens


def format_with_masking(tokens, labels, segments):
    """Format tokens showing masking and segment labels"""
    result = []
    for i, (token, label, seg) in enumerate(zip(tokens, labels, segments)):
        if token == '<pad>':
            break
        if label != -100:  # This token is masked
            result.append(f"[{token}→MASKED|seg{seg}]")
        else:
            result.append(f"{token}|seg{seg}")
    return ' '.join(result)


def main():
    output_file = 'cfg_dfg_100_pairs_detailed.txt'
    
    # Load vocab
    print("Loading vocab...")
    vocab = WordVocab.load_vocab("./vocab_base")
    print(f"Vocab size: {len(vocab)}")
    
    # Create dataset with proper parameters
    print("\nCreating baseline dataset...")
    dataset = BaselineDataset(
        cfg_corpus_path="/data/kun/palmtreedata/cfg_train_2.txt",
        dfg_corpus_path="/data/kun/palmtreedata/dfg_train_2.txt",
        vocab=vocab,
        seq_len=512,
        enable_imd=True,  # Enable DFG data
        instruction_mask_prob=0.25,
        token_mask_prob=0.15
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"\nLoading 100 pairs and saving to {output_file}...")
    
    with open(output_file, 'w') as f:
        # Header
        f.write("="*100 + "\n")
        f.write("100 CFG-DFG PAIRS - DETAILED TEXT AND TENSOR REPRESENTATION\n")
        f.write("="*100 + "\n")
        f.write("\nLegend:\n")
        f.write("  - [token→MASKED|segN]: Token that is masked for prediction\n")
        f.write("  - token|segN: Regular token with segment label N\n")
        f.write("  - Segment labels: 1, 2, 3... for each instruction (0 for padding)\n")
        f.write("  - is_next: 1 = consecutive sequences, 0 = random sequences\n")
        f.write("="*100 + "\n\n")
        
        # Process 100 samples
        for i in range(100):
            sample = dataset[i]
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/100 pairs...")
            
            f.write(f"\n{'='*100}\n")
            f.write(f"PAIR {i+1}/100\n")
            f.write(f"{'='*100}\n\n")
            
            # ========================================================================
            # CFG DATA
            # ========================================================================
            f.write(f"{'─'*100}\n")
            f.write("CFG (Control Flow Graph)\n")
            f.write(f"{'─'*100}\n")
            
            cfg_input = sample['cfg_bert_input']
            cfg_label = sample['bert_label']  # Note: uses 'bert_label' not 'cfg_bert_label'
            cfg_segment = sample['cfg_segment_label']
            cfg_is_next = sample['cfg_is_next'].item()
            
            # Decode CFG tokens
            cfg_tokens = decode_tokens(cfg_input, vocab)
            cfg_labels_list = cfg_label.tolist()
            cfg_segments_list = cfg_segment.tolist()
            
            # Show formatted text with masking and segments
            f.write("\nCFG Instruction Text (with masking indicators):\n")
            formatted_cfg = format_with_masking(cfg_tokens, cfg_labels_list, cfg_segments_list)
            # Wrap long lines
            words = formatted_cfg.split()
            line = ""
            for word in words:
                if len(line) + len(word) + 1 > 90:
                    f.write(f"  {line}\n")
                    line = word
                else:
                    line = line + " " + word if line else word
            if line:
                f.write(f"  {line}\n")
            
            # Show raw text (first 50 tokens)
            f.write(f"\nCFG Raw Tokens (first 50): {' '.join(cfg_tokens[:50])}\n")
            
            # Show tensors (first 30 elements)
            f.write(f"\nCFG Tensors (first 30 elements):\n")
            f.write(f"  bert_input:     {cfg_input[:30].tolist()}\n")
            f.write(f"  bert_label:     {cfg_label[:30].tolist()}\n")
            f.write(f"  segment_label:  {cfg_segment[:30].tolist()}\n")
            f.write(f"  is_next:        {cfg_is_next} ({'consecutive' if cfg_is_next == 1 else 'random'})\n")
            
            # Count masking statistics
            num_masked = (cfg_label != -100).sum().item()
            total_tokens = (cfg_input != vocab.stoi.get('<pad>', 0)).sum().item()
            mask_rate = num_masked / total_tokens * 100 if total_tokens > 0 else 0
            f.write(f"\nCFG Masking Stats: {num_masked}/{total_tokens} tokens masked ({mask_rate:.1f}%)\n")
            
            # ========================================================================
            # DFG DATA
            # ========================================================================
            if 'dfg_bert_input' in sample:
                f.write(f"\n{'─'*100}\n")
                f.write("DFG (Data Flow Graph)\n")
                f.write(f"{'─'*100}\n")
                
                dfg_input = sample['dfg_bert_input']
                dfg_label = sample['dfg_bert_label']
                dfg_segment = sample['dfg_segment_label']
                dfg_is_next = sample['dfg_is_next'].item()
                
                # Decode DFG tokens
                dfg_tokens = decode_tokens(dfg_input, vocab)
                dfg_labels_list = dfg_label.tolist()
                dfg_segments_list = dfg_segment.tolist()
                
                # Show formatted text with segments (no masking for DFG in baseline)
                f.write("\nDFG Instruction Text (with segment labels):\n")
                formatted_dfg = format_with_masking(dfg_tokens, dfg_labels_list, dfg_segments_list)
                # Wrap long lines
                words = formatted_dfg.split()
                line = ""
                for word in words:
                    if len(line) + len(word) + 1 > 90:
                        f.write(f"  {line}\n")
                        line = word
                    else:
                        line = line + " " + word if line else word
                if line:
                    f.write(f"  {line}\n")
                
                # Show raw text (first 50 tokens)
                f.write(f"\nDFG Raw Tokens (first 50): {' '.join(dfg_tokens[:50])}\n")
                
                # Show tensors (first 30 elements)
                f.write(f"\nDFG Tensors (first 30 elements):\n")
                f.write(f"  bert_input:     {dfg_input[:30].tolist()}\n")
                f.write(f"  bert_label:     {dfg_label[:30].tolist()}\n")
                f.write(f"  segment_label:  {dfg_segment[:30].tolist()}\n")
                f.write(f"  is_next:        {dfg_is_next} ({'consecutive' if dfg_is_next == 1 else 'random'})\n")
                
                # Count masking statistics (should be 0 for DFG in baseline)
                num_masked = (dfg_label != -100).sum().item()
                total_tokens = (dfg_input != vocab.stoi.get('<pad>', 0)).sum().item()
                mask_rate = num_masked / total_tokens * 100 if total_tokens > 0 else 0
                f.write(f"\nDFG Masking Stats: {num_masked}/{total_tokens} tokens masked ({mask_rate:.1f}%)\n")
                f.write(f"  Note: DFG should have 0% masking in baseline (DUP task only, no MLM/IMD)\n")
            else:
                f.write(f"\n{'─'*100}\n")
                f.write("DFG: Not available (enable_imd=False)\n")
            
            f.write(f"\n{'='*100}\n")
        
        # Summary
        f.write(f"\n\n{'='*100}\n")
        f.write("SUMMARY\n")
        f.write(f"{'='*100}\n")
        f.write(f"Total pairs processed: 100\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"\nTask Configuration:\n")
        f.write(f"  CFG: MLM (token masking ~15%) + CWP (is_next prediction)\n")
        f.write(f"  DFG: DUP (is_next prediction only, NO MLM/IMD)\n")
        f.write(f"{'='*100}\n")
    
    # Print file info
    import os
    file_size = os.path.getsize(output_file)
    print(f"\n✓ Done! Saved 100 CFG-DFG pairs to {output_file}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    # Show sample statistics
    print("\nSample Statistics (from 100 pairs):")
    cfg_next_counts = {0: 0, 1: 0}
    dfg_next_counts = {0: 0, 1: 0}
    cfg_mask_counts = []
    
    for i in range(100):
        sample = dataset[i]
        cfg_next_counts[sample['cfg_is_next'].item()] += 1
        if 'dfg_is_next' in sample:
            dfg_next_counts[sample['dfg_is_next'].item()] += 1
        
        cfg_label = sample['bert_label']  # Note: uses 'bert_label' not 'cfg_bert_label'
        num_masked = (cfg_label != -100).sum().item()
        cfg_mask_counts.append(num_masked)
    
    avg_cfg_masked = sum(cfg_mask_counts) / len(cfg_mask_counts)
    print(f"  CFG is_next=0: {cfg_next_counts[0]} ({cfg_next_counts[0]}%)")
    print(f"  CFG is_next=1: {cfg_next_counts[1]} ({cfg_next_counts[1]}%)")
    print(f"  DFG is_next=0: {dfg_next_counts[0]} ({dfg_next_counts[0]}%)")
    print(f"  DFG is_next=1: {dfg_next_counts[1]} ({dfg_next_counts[1]}%)")
    print(f"  Avg CFG masked tokens per sample: {avg_cfg_masked:.1f}")


if __name__ == "__main__":
    main()
