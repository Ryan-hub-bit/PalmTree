"""
Test Address-Aware Embeddings

Tests:
1. AddressPositionalEmbedding - processes binary/function/bb positions + is_daddr
2. VarPositionalEmbedding - processes var offsets
3. AddressAwareBERTEmbedding - full embedding pipeline
"""

import torch
from vocab import WordVocab
from address_aware.dataloader_addressaware import InstructionMaskingDataset
from address_aware.address_embedding import (
    AddressPositionalEmbedding,
    VarPositionalEmbedding,
    AddressAwareBERTEmbedding
)


def test_address_positional_embedding():
    """Test AddressPositionalEmbedding with real data, including is_daddr"""
    print("="*80)
    print("TEST 1: AddressPositionalEmbedding (with is_daddr)")
    print("="*80)
    
    # Create embedding
    d_model = 768
    addr_embed = AddressPositionalEmbedding(d_model=d_model)
    
    # Test with sample data
    batch_size = 2
    seq_len = 10
    
    # Create sample positions (values between -1.0 and 1.0)
    binary_pos = torch.rand(batch_size, seq_len) * 2 - 1  # [-1, 1]
    function_pos = torch.rand(batch_size, seq_len) * 2 - 1
    bb_pos = torch.rand(batch_size, seq_len) * 2 - 1
    is_daddr = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    # Set some to -1.0 (special tokens)
    binary_pos[:, 0] = -1.0  # <sos>
    function_pos[:, 0] = -1.0
    bb_pos[:, 0] = -1.0
    
    # Set some as daddr
    is_daddr[:, 5] = 1
    
    print(f"\nInput shapes:")
    print(f"  binary_pos: {binary_pos.shape}")
    print(f"  function_pos: {function_pos.shape}")
    print(f"  bb_pos: {bb_pos.shape}")
    print(f"  is_daddr: {is_daddr.shape}")
    print(f"\nSample is_daddr: {is_daddr[0].tolist()}")
    
    # Forward pass without is_daddr
    output_no_daddr = addr_embed(binary_pos, function_pos, bb_pos)
    
    # Forward pass with is_daddr
    output_with_daddr = addr_embed(binary_pos, function_pos, bb_pos, is_daddr)
    
    print(f"\nOutput shape (without is_daddr): {output_no_daddr.shape}")
    print(f"Output shape (with is_daddr): {output_with_daddr.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {d_model}]")
    
    assert output_no_daddr.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    assert output_with_daddr.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    assert not torch.isnan(output_with_daddr).any(), "NaN values detected!"
    assert not torch.isinf(output_with_daddr).any(), "Inf values detected!"
    
    # Check that daddr makes a difference
    diff_at_daddr = (output_with_daddr[0, 5] - output_no_daddr[0, 5]).abs().mean()
    diff_at_non_daddr = (output_with_daddr[0, 0] - output_no_daddr[0, 0]).abs().mean()
    
    print(f"\n✓ Difference statistics:")
    print(f"  At daddr position (5): {diff_at_daddr.item():.4f} (should be > 0)")
    print(f"  At non-daddr position (0): {diff_at_non_daddr.item():.4f}")
    
    print(f"\n✓ Output statistics (with is_daddr):")
    print(f"  Mean: {output_with_daddr.mean().item():.4f}")
    print(f"  Std: {output_with_daddr.std().item():.4f}")
    print(f"  Min: {output_with_daddr.min().item():.4f}")
    print(f"  Max: {output_with_daddr.max().item():.4f}")
    
    print("\n✓ AddressPositionalEmbedding working correctly!\n")


def test_var_offset_embedding():
    """Test VarPositionalEmbedding"""
    print("="*80)
    print("TEST 2: VarPositionalEmbedding")
    print("="*80)
    
    d_model = 768
    max_offset = 256  # 0xFF
    var_embed = VarPositionalEmbedding(d_model=d_model, max_offset=max_offset)
    
    batch_size = 2
    seq_len = 10
    
    # Create sample var offsets
    # -1 means not a var, 0-255 are actual offsets
    var_offsets = torch.randint(-1, max_offset, (batch_size, seq_len))
    
    print(f"\nInput shape: {var_offsets.shape}")
    print(f"Sample var_offsets:\n{var_offsets[0]}")
    
    # Forward pass
    output = var_embed(var_offsets)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {d_model}]")
    
    assert output.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    assert not torch.isnan(output).any(), "NaN values detected!"
    
    print(f"\n✓ Output statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    
    print("\n✓ VarPositionalEmbedding working correctly!\n")


def test_full_embedding_pipeline():
    """Test AddressAwareBERTEmbedding - the complete embedding"""
    print("="*80)
    print("TEST 3: AddressAwareBERTEmbedding (Full Pipeline)")
    print("="*80)
    
    # Load vocab
    vocab = WordVocab.load_vocab('./vocab_addr')
    vocab_size = len(vocab)
    d_model = 768
    
    # Create full embedding
    bert_embed = AddressAwareBERTEmbedding(vocab_size=vocab_size, embed_size=d_model)
    
    batch_size = 2
    seq_len = 10
    
    # Create sample data
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    segment_labels = torch.ones(batch_size, seq_len, dtype=torch.long)
    binary_pos = torch.rand(batch_size, seq_len) * 2 - 1
    function_pos = torch.rand(batch_size, seq_len) * 2 - 1
    bb_pos = torch.rand(batch_size, seq_len) * 2 - 1
    var_offsets = torch.randint(-1, 256, (batch_size, seq_len))
    is_daddr = torch.randint(0, 2, (batch_size, seq_len))
    
    print(f"\nInput shapes:")
    print(f"  token_ids: {token_ids.shape}")
    print(f"  segment_labels: {segment_labels.shape}")
    print(f"  binary_pos: {binary_pos.shape}")
    print(f"  var_offsets: {var_offsets.shape}")
    print(f"  is_daddr: {is_daddr.shape}")
    
    # Forward pass
    output = bert_embed(token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets, is_daddr)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {d_model}]")
    
    assert output.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    assert not torch.isnan(output).any(), "NaN values detected!"
    
    print(f"\n✓ Output statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    
    print("\n✓ AddressAwareBERTEmbedding working correctly!\n")


def test_with_real_data():
    """Test embeddings with real dataloader output"""
    print("="*80)
    print("TEST 4: Integration with Real Dataloader")
    print("="*80)
    
    # Load vocab and dataset
    print("\nLoading dataset...")
    vocab = WordVocab.load_vocab('./vocab_addr')
    dataset = InstructionMaskingDataset(
        cfg_corpus_path='/data/kun/palmtreedata/cfg_train_2.txt',
        dfg_corpus_path=None,
        vocab=vocab,
        seq_len=512,
        on_memory=True,
        token_mask_prob=0.0,
        instruction_mask_prob=0.0,
        data_percentage=0.001,  # Just 0.1% for quick test
        enable_imd=False,
    )
    
    # Get a sample
    sample = dataset[0]
    imc_data = sample['imc']
    
    # Extract all fields
    bert_input = imc_data['bert_input'].unsqueeze(0)  # Add batch dim
    segment_labels = imc_data['segment_label'].unsqueeze(0)
    binary_pos = imc_data['binary_pos'].unsqueeze(0)
    function_pos = imc_data['function_pos'].unsqueeze(0)
    bb_pos = imc_data['bb_pos'].unsqueeze(0)
    var_offsets = imc_data['var_offsets'].unsqueeze(0)
    is_daddr = imc_data['is_daddr'].unsqueeze(0)
    
    print(f"\nData shapes:")
    print(f"  bert_input: {bert_input.shape}")
    print(f"  segment_labels: {segment_labels.shape}")
    print(f"  binary_pos: {binary_pos.shape}")
    print(f"  var_offsets: {var_offsets.shape}")
    print(f"  is_daddr: {is_daddr.shape}")
    
    # Create full embedding
    d_model = 768
    bert_embed = AddressAwareBERTEmbedding(vocab_size=len(vocab), embed_size=d_model)
    
    # Apply embedding
    print("\nApplying full BERT embedding...")
    output = bert_embed(bert_input, segment_labels, binary_pos, function_pos, bb_pos, var_offsets, is_daddr)
    
    print(f"\nEmbedding shape: {output.shape}")
    print(f"Expected: [1, {bert_input.shape[1]}, {d_model}]")
    
    assert output.shape == (1, bert_input.shape[1], d_model), "Shape mismatch!"
    assert not torch.isnan(output).any(), "NaN values detected!"
    
    print(f"\n✓ Combined embedding statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    
    # Show first few positions
    print(f"\nFirst 5 positions embedding norms:")
    for i in range(min(5, output.shape[1])):
        norm = output[0, i].norm().item()
        token = vocab.itos[bert_input[0, i].item()]
        print(f"  [{i}] {token:10s}: norm = {norm:.4f}")
    
    print("\n✓ Real data integration working correctly!\n")


def main():
    print("\n" + "="*80)
    print("Address-Aware Embeddings Test Suite")
    print("="*80 + "\n")
    
    try:
        test_address_positional_embedding()
        test_var_offset_embedding()
        test_full_embedding_pipeline()
        test_with_real_data()
        
        print("="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Test model forward pass with these embeddings")
        print("  2. Test trainer integration")
        print("  3. Run a quick training loop")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
