"""
Test script to verify the address-aware dataloader and model work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from palmtree.dataset.vocab import WordVocab
from dataloader import AddressAwareDataset
from model import AddressAwareBERT, AddressAwareBERTForPretraining


def test_dataloader():
    """Test that the dataloader can parse inline format correctly."""
    print("Testing dataloader...")
    
    # Create a small test vocabulary
    from collections import Counter
    counter = Counter({
        'mov': 100, 'test': 80, 'je': 60, 'call': 50, 'retn': 40,
        'rax': 90, 'rsp': 85, 'qword': 70, '[': 75, ']': 75,
        'rel': 65, 'addr_code': 55, 'addr_data': 55, '0x8': 45
    })
    vocab = WordVocab.__new__(WordVocab)
    vocab.freqs = counter
    vocab.pad_index = 0
    vocab.unk_index = 1
    vocab.eos_index = 2
    vocab.sos_index = 3
    vocab.mask_index = 4
    vocab.itos = ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"] + list(counter.keys())
    vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Test parsing
    dataset = AddressAwareDataset.__new__(AddressAwareDataset)
    dataset.vocab = vocab
    dataset.seq_len = 64
    dataset.mask_prob = 0.15
    dataset.nsp_prob = 0.5
    
    # Import regex patterns
    import re
    dataset.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    dataset.nested_addr_pattern = re.compile(r'(addr_code|addr_data)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    
    # Test instruction parsing
    test_instr = "mov(0x401008:0.022:0.296:0.444) rax qword [ rel addr_data(0x403fe8:1.000:0.000:0) ]"
    tokens, positions = dataset._parse_instruction(test_instr)
    
    print(f"\nTest instruction: {test_instr}")
    print(f"Parsed tokens ({len(tokens)}): {tokens}")
    print(f"Parsed positions ({len(positions)}):")
    for i, (tok, pos) in enumerate(zip(tokens, positions)):
        print(f"  {i}: {tok:15s} -> {pos}")
    
    assert len(tokens) == len(positions), "Token and position counts must match!"
    assert len(tokens) > 0, "Should parse at least one token!"
    
    # Verify that only opcodes and address tokens have non-zero positions
    assert positions[0] == (0.022, 0.296, 0.444), "mov should have its address position"
    assert positions[1] == (0.0, 0.0, 0.0), "rax should have zero position"
    assert positions[2] == (0.0, 0.0, 0.0), "qword should have zero position"
    assert positions[3] == (0.0, 0.0, 0.0), "[ should have zero position"
    assert positions[4] == (0.0, 0.0, 0.0), "rel should have zero position"
    assert positions[5] == (1.0, 0.0, 0.0), "addr_data should have its address position"
    assert positions[6] == (0.0, 0.0, 0.0), "] should have zero position"
    
    print("✓ Dataloader parsing works!")
    print("✓ Only opcodes and address tokens have positional embeddings!")
    return vocab


def test_model(vocab):
    """Test that the model can be created and run forward pass."""
    print("\nTesting model...")
    
    # Create model (hidden must be divisible by 3 and by attn_heads)
    bert = AddressAwareBERT(
        vocab_size=len(vocab),
        hidden=120,  # Small for testing, divisible by 3 and 4
        n_layers=2,
        attn_heads=4,
        dropout=0.1,
        max_len=64
    )
    
    model = AddressAwareBERTForPretraining(bert, len(vocab))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create dummy batch
    batch_size = 2
    seq_len = 64
    
    token_ids = torch.randint(0, len(vocab), (batch_size, seq_len))
    segment_labels = torch.randint(0, 2, (batch_size, seq_len))
    binary_pos = torch.rand(batch_size, seq_len)
    function_pos = torch.rand(batch_size, seq_len)
    bb_pos = torch.rand(batch_size, seq_len)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        mlm_output, nsp_output = model(token_ids, segment_labels, binary_pos, function_pos, bb_pos)
    
    print(f"MLM output shape: {mlm_output.shape} (expected: [{batch_size}, {seq_len}, {len(vocab)}])")
    print(f"NSP output shape: {nsp_output.shape} (expected: [{batch_size}, 2])")
    
    assert mlm_output.shape == (batch_size, seq_len, len(vocab)), "MLM output shape mismatch!"
    assert nsp_output.shape == (batch_size, 2), "NSP output shape mismatch!"
    
    print("✓ Model forward pass works!")


def test_address_embedding():
    """Test address positional embedding."""
    print("\nTesting address positional embedding...")
    
    from address_embedding import AddressPositionalEmbedding, SequencePositionalEmbedding
    
    d_model = 768
    batch_size = 2
    seq_len = 16
    
    # Test sequence positional embedding (like PalmTree)
    seq_emb = SequencePositionalEmbedding(d_model=d_model, max_len=512)
    dummy_input = torch.zeros(batch_size, seq_len)
    seq_output = seq_emb(dummy_input)
    print(f"Sequence positional embedding:")
    print(f"  Input: {dummy_input.shape}")
    print(f"  Output: {seq_output.shape}")
    # Note: Sequence pos returns [1, seq_len, d_model] and broadcasts to batch
    assert seq_output.shape[1] == seq_len and seq_output.shape[2] == d_model, "Sequence pos shape mismatch!"
    
    # Test address positional embedding (NEW)
    addr_emb = AddressPositionalEmbedding(d_model=d_model, max_len=512)
    binary_pos = torch.rand(batch_size, seq_len)
    function_pos = torch.rand(batch_size, seq_len)
    bb_pos = torch.rand(batch_size, seq_len)
    addr_output = addr_emb(binary_pos, function_pos, bb_pos)
    
    print(f"\nAddress positional embedding:")
    print(f"  Input shapes: binary_pos={binary_pos.shape}, function_pos={function_pos.shape}, bb_pos={bb_pos.shape}")
    print(f"  Output: {addr_output.shape} (expected: [{batch_size}, {seq_len}, {d_model}])")
    print(f"  Level weights: {addr_emb.level_weights.data}")
    assert addr_output.shape == (batch_size, seq_len, d_model), "Address pos shape mismatch!"
    
    print("✓ Sequence positional embedding works!")
    print("✓ Address positional embedding works!")


def main():
    print("="*60)
    print("Address-Aware PalmTree Test Suite")
    print("="*60)
    
    try:
        vocab = test_dataloader()
        test_address_embedding()
        test_model(vocab)
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        print("\nThe address-aware components are working correctly.")
        print("You can now proceed to train on real data.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
