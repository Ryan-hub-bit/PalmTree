"""
Simple test to check if var_offsets are being used in embeddings.
Compares embeddings with and without var_offsets to see if they're identical.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vocab import WordVocab
from address_embedding import AddressAwareBERTEmbedding


def test_if_embeddings_identical():
    """Test if embeddings are identical with and without var_offsets"""
    
    print("="*60)
    print("TEST: Are embeddings identical with/without var_offsets?")
    print("="*60)
    
    # Load vocab
    vocab_path = "vocab.pkl"
    if not os.path.exists(vocab_path):
        print(f"❌ Vocab file not found: {vocab_path}")
        return
    
    vocab = WordVocab.load_vocab(vocab_path)
    print(f"✓ Loaded vocab with {len(vocab)} tokens\n")
    
    # Create embedding layer with var embedding enabled
    embed_size = 128
    embedding_layer = AddressAwareBERTEmbedding(
        vocab_size=len(vocab),
        embed_size=embed_size,
        dropout=0.0,  # No dropout for reproducibility
        max_len=60,
        use_address_embedding=True,
        use_var_embedding=True  # Enable var embedding
    )
    
    embedding_layer.eval()
    
    # Create test input
    batch_size = 2
    seq_len = 10
    
    token_ids = torch.randint(5, 100, (batch_size, seq_len))
    segment_labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    binary_pos = torch.rand(batch_size, seq_len)
    function_pos = torch.rand(batch_size, seq_len)
    bb_pos = torch.rand(batch_size, seq_len)
    
    # Test 1: With non-zero var_offsets
    var_offsets_nonzero = torch.tensor([
        [0, 16, 32, 0, 64, 0, 128, 0, 0, 0],
        [0, 0, 24, 48, 0, 96, 0, 0, 0, 0]
    ], dtype=torch.long)
    
    # Test 2: With zero var_offsets (equivalent to no var embedding)
    var_offsets_zero = torch.zeros_like(var_offsets_nonzero)
    
    print("Test inputs:")
    print(f"  token_ids shape: {token_ids.shape}")
    print(f"  var_offsets (non-zero): {var_offsets_nonzero[0].tolist()}")
    print(f"  var_offsets (zero):     {var_offsets_zero[0].tolist()}")
    
    with torch.no_grad():
        # Embedding WITH var_offsets
        embedding1 = embedding_layer(
            token_ids, segment_labels, 
            binary_pos, function_pos, bb_pos, 
            var_offsets_nonzero
        )
        
        # Embedding WITHOUT var_offsets (all zeros)
        embedding2 = embedding_layer(
            token_ids, segment_labels, 
            binary_pos, function_pos, bb_pos, 
            var_offsets_zero
        )
    
    # Check if embeddings are identical
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    are_identical = torch.equal(embedding1, embedding2)
    
    print(f"embedding1 shape: {embedding1.shape}")
    print(f"embedding2 shape: {embedding2.shape}")
    print(f"\nAre embeddings EXACTLY identical? {are_identical}")
    
    if are_identical:
        print("❌ embedding1 == embedding2 (IDENTICAL)")
        print("   This means var_offsets are NOT being used!")
        print("   The var embedding is NOT affecting the output.")
    else:
        print("✅ embedding1 != embedding2 (DIFFERENT)")
        print("   This means var_offsets ARE being used!")
        print("   The var embedding IS working correctly.")
    
    # Calculate difference
    diff = (embedding1 - embedding2).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    
    print(f"\nDifference metrics:")
    print(f"  Mean absolute difference: {mean_diff:.8f}")
    print(f"  Max absolute difference:  {max_diff:.8f}")
    
    # Check with tolerance
    are_close = torch.allclose(embedding1, embedding2, atol=1e-8)
    print(f"  Are close (atol=1e-8)?    {are_close}")
    
    # Show difference at positions with non-zero var offsets
    print(f"\nDifference at var token positions:")
    for b in range(batch_size):
        for i in range(seq_len):
            var_offset = var_offsets_nonzero[b, i].item()
            if var_offset > 0:
                pos_diff = diff[b, i].mean().item()
                print(f"  Batch {b}, Pos {i}: var_offset={var_offset:3d} (0x{var_offset:02x}) → diff={pos_diff:.6f}")
    
    print("="*60)
    
    # Final verdict
    if are_identical:
        print("\n🚨 PROBLEM DETECTED: Var embeddings are NOT being used!")
        return False
    else:
        print("\n✅ SUCCESS: Var embeddings ARE being used!")
        return True


if __name__ == "__main__":
    success = test_if_embeddings_identical()
    exit(0 if success else 1)
