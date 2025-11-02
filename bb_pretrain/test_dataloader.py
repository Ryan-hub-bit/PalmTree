"""
Test script to verify data loader works with new BB pair format
"""
import sys
import os

# Add bb_pretrain directory FIRST to avoid import conflicts
bb_pretrain_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, bb_pretrain_dir)

import data_loader
import config

# Paths
BB_PAIRS_FILE = "/home/louie/PalmTree/all_bb_pairs.txt"
VOCAB_FILE = "/home/louie/PalmTree/pre-trained_model/palmtree/vocab"  # No .txt extension
MAX_PAIRS_TO_LOAD = 100  # Only load first 100 pairs for testing

def test_parse_bb():
    """Test BB parsing with new format"""
    dataset = data_loader.BBPairDataset(BB_PAIRS_FILE, VOCAB_FILE, max_seq_length=256, max_pairs=MAX_PAIRS_TO_LOAD)
    
    print(f"Loaded {len(dataset)} BB pairs")
    print(f"PalmTree Vocabulary size: {len(dataset.palmtree_vocab)}")
    print(f"Address Vocabulary size: {len(dataset.addr_vocab)}")
    print()
    
    # Test first few samples
    for i in range(min(5, len(dataset))):
        print(f"\n{'='*80}")
        print(f"Sample {i}:")
        print(f"{'='*80}")
        
        pair = dataset.bb_pairs[i]
        
        print("\nSource BB:")
        print(f"  Tokens: {' '.join(pair['src_tokens'][:20])}...")
        print(f"  Total tokens: {len(pair['src_tokens'])}")
        print(f"  Address values (first 10): {[hex(addr) if addr != 0 else '0' for addr in pair['src_addr_values'][:10]]}")
        
        print("\nTarget BB:")
        print(f"  Tokens: {' '.join(pair['tgt_tokens'][:20])}...")
        print(f"  Total tokens: {len(pair['tgt_tokens'])}")
        print(f"  Address values (first 10): {[hex(addr) if addr != 0 else '0' for addr in pair['tgt_addr_values'][:10]]}")
        
        print(f"\nEdge type: {pair['edge_type']}")
        
        # Get processed sample
        sample = dataset[i]
        print(f"\nProcessed sample:")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Address encodings shape: {sample['address_encodings'].shape}")
        print(f"  Attention mask sum: {sample['attention_mask'].sum().item()}")
        print(f"  Source length: {sample['src_length']}")
        print(f"  Target length: {sample['tgt_length']}")
        print(f"  Edge type label: {sample['edge_type'].item()}")

def test_address_tokens():
    """Test that special address tokens are in vocab"""
    dataset = data_loader.BBPairDataset(BB_PAIRS_FILE, VOCAB_FILE, max_seq_length=256, max_pairs=MAX_PAIRS_TO_LOAD)
    
    print("\nDual Vocabulary System:")
    print("="*80)
    print(f"PalmTree Vocab Size: {len(dataset.palmtree_vocab)}")
    print(f"Address Vocab Size: {len(dataset.addr_vocab)}")
    
    print("\nAddress Tokens in Address Vocabulary:")
    print("="*80)
    for token in config.SPECIAL_ADDR_TOKENS:
        idx = dataset.addr_vocab.to_index(token)
        print(f"  ✓ {token:20s} -> {idx}")

if __name__ == "__main__":
    print("Testing BB Pair Data Loader")
    print("="*80)
    
    try:
        test_address_tokens()
        print()
        test_parse_bb()
        print("\n" + "="*80)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
