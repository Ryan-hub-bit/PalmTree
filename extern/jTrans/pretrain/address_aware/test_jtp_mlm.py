"""
Test script to verify MLM and JTP label generation.
Shows how tokens are masked and how jump targets are predicted.
"""

import pickle
from dataloader_addressaware import AddressAwareDataset


def test_mlm_and_jtp():
    """Test MLM and JTP label generation with a sample sequence."""
    
    # Load vocabulary
    print("="*80)
    print("Loading vocabulary...")
    print("="*80)
    with open('./vocab_addr.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens: <pad>={vocab.stoi.get('<pad>')}, <unk>={vocab.stoi.get('<unk>')}, "
          f"<sos>={vocab.stoi.get('<sos>')}, <eos>={vocab.stoi.get('<eos>')}, <mask>={vocab.stoi.get('<mask>')}")
    
    # Create dataset
    dataset = AddressAwareDataset(
        corpus_path='/data/kun/jtransdata/addr_pretrain.txt',
        vocab=vocab,
        seq_len=512,
        token_mask_prob=0.15,
        data_percentage=0.01,  # Use 1% of data to find jump examples
        is_train=True
    )
    
    print(f"\nDataset size: {len(dataset)} lines")
    
    # Test first few samples
    for sample_idx in range(min(3, len(dataset))):
        print("\n" + "="*80)
        print(f"SAMPLE {sample_idx + 1}")
        print("="*80)
        
        # Get sample
        sample = dataset[sample_idx]
        
        # Extract data
        token_ids = sample['bert_input'].tolist()
        mlm_labels = sample['bert_label'].tolist()
        jtp_labels = sample['jtp_labels'].tolist()
        segments = sample['segment_label'].tolist()
        
        # Get reverse vocab mapping
        itos = {v: k for k, v in vocab.stoi.items()}
        
        # Find non-padding tokens
        pad_idx = vocab.stoi.get('<pad>', 0)
        real_length = sum(1 for tid in token_ids if tid != pad_idx)
        
        print(f"\nSequence length: {real_length} tokens (before padding)")
        print(f"Full length: {len(token_ids)}")
        
        # Analyze MLM
        print("\n" + "-"*80)
        print("MLM (Masked Language Modeling) Analysis:")
        print("-"*80)
        
        masked_count = sum(1 for label in mlm_labels if label != -100)
        mask_idx = vocab.stoi.get('<mask>', 4)
        
        print(f"Total masked tokens: {masked_count} / {real_length} ({100*masked_count/real_length:.1f}%)")
        print("\nFirst 10 masked tokens:")
        
        shown_masks = 0
        for pos, (tid, label) in enumerate(zip(token_ids[:real_length], mlm_labels[:real_length])):
            if label != -100 and shown_masks < 10:
                token_str = itos.get(tid, f'<ID_{tid}>')
                original_str = itos.get(label, f'<ID_{label}>')
                mask_type = "MASK" if tid == mask_idx else ("RANDOM" if tid != label else "KEEP")
                print(f"  Position {pos:3d}: {token_str:15s} → original: {original_str:15s} (type: {mask_type})")
                shown_masks += 1
        
        # Analyze JTP
        print("\n" + "-"*80)
        print("JTP (Jump Target Prediction) Analysis:")
        print("-"*80)
        
        jump_count = sum(1 for label in jtp_labels if label != -100)
        print(f"Total jump instructions: {jump_count}")
        
        if jump_count > 0:
            print("\nJump instructions and their targets:")
            for pos, (tid, jtp_label, seg) in enumerate(zip(token_ids[:real_length], 
                                                             jtp_labels[:real_length], 
                                                             segments[:real_length])):
                if jtp_label != -100:
                    token_str = itos.get(tid, f'<ID_{tid}>')
                    
                    # Get target instruction info
                    if jtp_label < len(token_ids):
                        target_token = itos.get(token_ids[jtp_label], f'<ID_{token_ids[jtp_label]}>')
                        target_seg = segments[jtp_label]
                        
                        # Show context around jump
                        context_start = max(0, pos - 2)
                        context_end = min(real_length, pos + 3)
                        context = ' '.join([itos.get(token_ids[i], f'<ID_{token_ids[i]}>') 
                                           for i in range(context_start, context_end)])
                        
                        print(f"\n  Position {pos:3d} (instruction {seg}): {token_str}")
                        print(f"    → Jumps to position {jtp_label:3d} (instruction {target_seg}): {target_token}")
                        print(f"    Context: ...{context}...")
                    else:
                        print(f"\n  Position {pos:3d} (instruction {seg}): {token_str}")
                        print(f"    → Target position {jtp_label} is out of sequence bounds")
        else:
            print("  No jump instructions found in this sample")
        
        # Show instruction boundaries
        print("\n" + "-"*80)
        print("Instruction Segmentation (first 20 tokens):")
        print("-"*80)
        
        prev_seg = -1
        for pos in range(min(20, real_length)):
            tid = token_ids[pos]
            seg = segments[pos]
            token_str = itos.get(tid, f'<ID_{tid}>')
            
            if seg != prev_seg:
                print(f"\n  Instruction {seg}:")
                prev_seg = seg
            
            marker = ""
            if mlm_labels[pos] != -100:
                marker += " [MASKED]"
            if jtp_labels[pos] != -100:
                marker += f" [JUMP→{jtp_labels[pos]}]"
            
            print(f"    {pos:3d}: {token_str:15s}{marker}")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    test_mlm_and_jtp()
