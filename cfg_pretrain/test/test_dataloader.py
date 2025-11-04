"""
Test the CFG pretrain data loader
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import create_dataloader
import config

def test_dataloader():
    """Test the data loader with a small sample"""
    
    print("="*60)
    print("Testing CFG Pretrain DataLoader")
    print("="*60)
    
    # Update paths
    data_file = '../bb_pairs_output/atilibusb.so_bb_pairs.txt'
    vocab_file = '../pre-trained_model/palmtree/vocab'
    
    print(f"\nData file: {data_file}")
    print(f"Vocab file: {vocab_file}")
    
    # Create dataloader with small sample
    print("\nCreating dataloader...")
    dataloader = create_dataloader(
        data_file=data_file,
        vocab_file=vocab_file,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
        max_pairs=100
    )
    
    print(f"✓ DataLoader created with {len(dataloader)} batches")
    
    # Test first batch
    print("\n" + "="*60)
    print("Testing first batch...")
    print("="*60)
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"\n=== 3-Level Embeddings ===")
        print(f"  1. Semantic (input_ids):      {batch['input_ids'].shape}")
        print(f"  2. Address Position:")
        print(f"     - Binary positions:         {batch['binary_positions'].shape}")
        print(f"     - Function positions:       {batch['function_positions'].shape}")
        print(f"  3. Sequence Position:          {batch['sequence_positions'].shape}")
        
        print(f"\n=== Other Features ===")
        print(f"  Attention mask:                {batch['attention_mask'].shape}")
        print(f"  Segment IDs:                   {batch['segment_ids'].shape}")
        print(f"  MLM labels:                    {batch['mlm_labels'].shape}")
        print(f"  CFG label:                     {batch['cfg_label'].shape}")
        print(f"  Source addr features:          {batch['source_addr_features'].shape}")
        print(f"  Target addr features:          {batch['target_addr_features'].shape}")
        
        # Show example of 3-level embeddings for first sample
        print(f"\n=== Example: First 10 tokens of sample 0 ===")
        print(f"{'Token ID':>10} {'Bin Pos':>10} {'Func Pos':>10} {'Seq Pos':>10}")
        print("-" * 44)
        for i in range(min(10, batch['input_ids'].size(1))):
            print(f"{batch['input_ids'][0, i].item():>10} "
                  f"{batch['binary_positions'][0, i].item():>10.4f} "
                  f"{batch['function_positions'][0, i].item():>10.4f} "
                  f"{batch['sequence_positions'][0, i].item():>10}")
        
        # Show MLM statistics
        batch_size = batch['input_ids'].size(0)
        total_masked = 0
        for i in range(batch_size):
            num_masked = (batch['mlm_labels'][i] != -100).sum().item()
            total_masked += num_masked
        
        avg_masked = total_masked / batch_size
        print(f"\n  Average masked tokens per sample: {avg_masked:.1f}")
        
        # Show address position statistics
        print(f"\n=== Address Position Statistics (Sample 0) ===")
        bin_pos = batch['binary_positions'][0]
        func_pos = batch['function_positions'][0]
        mask = batch['attention_mask'][0].bool()
        
        # Only consider non-padded positions
        bin_pos_valid = bin_pos[mask]
        func_pos_valid = func_pos[mask]
        
        print(f"  Binary positions - min: {bin_pos_valid.min():.4f}, max: {bin_pos_valid.max():.4f}, mean: {bin_pos_valid.mean():.4f}")
        print(f"  Function positions - min: {func_pos_valid.min():.4f}, max: {func_pos_valid.max():.4f}, mean: {func_pos_valid.mean():.4f}")
        
        if batch_idx >= 2:  # Test first 3 batches
            break
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)

if __name__ == '__main__':
    test_dataloader()
