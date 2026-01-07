#!/usr/bin/env python3
"""
Test that address and daddr tokens use separate MLPs.
Verify the dual MLP architecture is working correctly.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocab import WordVocab
from address_embedding import AddressPositionalEmbedding

def test_dual_mlp():
    """Test that address and daddr use different MLPs"""
    
    print("=" * 80)
    print("TESTING DUAL MLP ARCHITECTURE FOR ADDRESS vs DADDR")
    print("=" * 80)
    
    # Load vocab to get token IDs
    vocab_path = 'vocab_addr.pkl'
    print(f"\nLoading vocab from {vocab_path}...")
    vocab = WordVocab.load_vocab(vocab_path)
    
    address_id = vocab.stoi.get('address', -1)
    daddr_id = vocab.stoi.get('daddr', -1)
    pad_id = vocab.stoi['<pad>']
    
    print(f"  address token ID: {address_id}")
    print(f"  daddr token ID: {daddr_id}")
    print(f"  <pad> token ID: {pad_id}")
    
    # Create embedding module
    print("\nCreating AddressPositionalEmbedding module...")
    d_model = 768
    intermediate_size = 128
    embedding_module = AddressPositionalEmbedding(
        d_model=d_model,
        intermediate_size=intermediate_size,
        dropout=0.1
    )
    
    # Check that we have two separate MLPs
    print("\nVerifying separate MLPs exist:")
    print(f"  code_address_projection: {type(embedding_module.code_address_projection)}")
    print(f"  data_address_projection: {type(embedding_module.data_address_projection)}")
    
    # Create test batch
    batch_size = 2
    seq_len = 10
    
    # Create position tensors (normalized [0,1])
    # Make address tokens have valid positions, non-address tokens have -1
    binary_pos = torch.full((batch_size, seq_len), -1.0)
    function_pos = torch.full((batch_size, seq_len), -1.0)
    bb_pos = torch.full((batch_size, seq_len), -1.0)
    
    # Set positions for some tokens (positions 2, 5, 7 are address tokens)
    address_positions = [2, 5, 7]
    for pos in address_positions:
        binary_pos[:, pos] = 0.5
        function_pos[:, pos] = 0.3
        bb_pos[:, pos] = 0.1
    
    # Create token IDs: position 2 = address, position 5 = daddr, position 7 = address
    token_ids = torch.full((batch_size, seq_len), pad_id)
    token_ids[:, 2] = address_id  # code address
    token_ids[:, 5] = daddr_id    # data address
    token_ids[:, 7] = address_id  # code address
    
    print(f"\nTest batch:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Address token positions: {address_positions}")
    print(f"  Token IDs at position 2: {token_ids[0, 2].item()} (should be address = {address_id})")
    print(f"  Token IDs at position 5: {token_ids[0, 5].item()} (should be daddr = {daddr_id})")
    print(f"  Token IDs at position 7: {token_ids[0, 7].item()} (should be address = {address_id})")
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        embeddings = embedding_module(binary_pos, function_pos, bb_pos, token_ids, vocab.stoi)
    
    print(f"  Output shape: {embeddings.shape}")
    print(f"  Expected shape: ({batch_size}, {seq_len}, {d_model})")
    
    # Check embeddings
    print("\nVerifying embeddings:")
    
    # Position 2: address token (should use code MLP)
    emb_pos2 = embeddings[0, 2]
    print(f"  Position 2 (address - code MLP):")
    print(f"    Embedding norm: {emb_pos2.norm().item():.4f}")
    print(f"    Non-zero: {(emb_pos2 != 0).any().item()}")
    
    # Position 5: daddr token (should use data MLP)
    emb_pos5 = embeddings[0, 5]
    print(f"  Position 5 (daddr - data MLP):")
    print(f"    Embedding norm: {emb_pos5.norm().item():.4f}")
    print(f"    Non-zero: {(emb_pos5 != 0).any().item()}")
    
    # Position 7: address token (should use code MLP)
    emb_pos7 = embeddings[0, 7]
    print(f"  Position 7 (address - code MLP):")
    print(f"    Embedding norm: {emb_pos7.norm().item():.4f}")
    print(f"    Non-zero: {(emb_pos7 != 0).any().item()}")
    
    # Position 0: pad token (should be all zeros)
    emb_pos0 = embeddings[0, 0]
    print(f"  Position 0 (<pad> - no address embedding):")
    print(f"    Embedding norm: {emb_pos0.norm().item():.4f}")
    print(f"    All zeros: {(emb_pos0 == 0).all().item()}")
    
    # Check that address and daddr embeddings are different
    # (they should be different because they use different MLPs with random initialization)
    diff_address_daddr = (emb_pos2 - emb_pos5).norm().item()
    diff_address_address = (emb_pos2 - emb_pos7).norm().item()
    
    print(f"\nEmbedding differences:")
    print(f"  address vs daddr (pos 2 vs 5): {diff_address_daddr:.4f}")
    print(f"  address vs address (pos 2 vs 7): {diff_address_address:.4f}")
    print(f"    (address vs daddr should be large - different MLPs)")
    print(f"    (address vs address should be small - same MLP, same positions)")
    
    # Verify different MLPs produce different embeddings
    if diff_address_daddr > 0.01:
        print("\n✓ SUCCESS: address and daddr produce different embeddings (separate MLPs working!)")
    else:
        print("\n✗ FAILED: address and daddr produce same embeddings (MLPs may not be separate)")
    
    # Count parameters
    code_params = sum(p.numel() for p in embedding_module.code_address_projection.parameters())
    data_params = sum(p.numel() for p in embedding_module.data_address_projection.parameters())
    
    print(f"\nParameter counts:")
    print(f"  Code address MLP: {code_params:,} parameters")
    print(f"  Data address MLP: {data_params:,} parameters")
    print(f"  Total: {code_params + data_params:,} parameters")
    print(f"  (Should be equal: {code_params == data_params})")
    
    print("\n" + "=" * 80)
    print("DUAL MLP TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_dual_mlp()
