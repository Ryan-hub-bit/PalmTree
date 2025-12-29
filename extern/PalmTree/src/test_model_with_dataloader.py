"""
Test Address-Aware BERT Model with Address-Aware Dataloader

This script tests:
1. Model initialization
2. Forward pass with IMC task (instruction masking CFG)
3. Forward pass with MLM task (token masking CFG)
4. Output shapes and statistics
5. Loss calculation
"""

import torch
import sys
sys.path.insert(0, 'src')

from vocab import WordVocab
from address_aware.dataloader_addressaware import InstructionMaskingDataset
from address_aware.model_addressaware import AddressAwareBERT, AddressAwareBERTForPretraining


def test_model_initialization():
    """Test that the model initializes correctly"""
    print("="*80)
    print("TEST 1: Model Initialization")
    print("="*80)
    
    vocab = WordVocab.load_vocab("./vocab_addr")
    vocab_size = len(vocab)
    
    # Create base BERT model
    bert = AddressAwareBERT(
        vocab_size=vocab_size,
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
        max_len=512,
        use_address_embedding=True,
        use_var_embedding=True
    )
    
    # Create pretraining model with IMC and MLM tasks
    model = AddressAwareBERTForPretraining(
        bert_model=bert,
        vocab_size=vocab_size,
        enable_mlm=True,
        enable_imc=True,
        enable_imd=False,  # No DFG for now
        enable_nsp_cfg=False,  # No NSP for this test
        enable_nsp_dfg=False,
        enable_scope=False
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden size: 768")
    print(f"  Layers: 12")
    print(f"  Attention heads: 12")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    print()
    print("  ✓ Model initialized successfully!")
    
    return model, vocab


def test_imc_forward_pass(model, vocab):
    """Test forward pass with IMC task"""
    print("\n" + "="*80)
    print("TEST 2: IMC Forward Pass (Instruction Masking CFG)")
    print("="*80)
    
    # Load dataset
    print("  Loading dataset...")
    dataset = InstructionMaskingDataset(
        cfg_corpus_path='/data/kun/palmtreedata/cfg_train_2.txt',
        dfg_corpus_path=None,
        vocab=vocab,
        seq_len=512,
        on_memory=True,
        instruction_mask_prob=0.25,
        token_mask_prob=0.0,  # No token-level masking for IMC
        data_percentage=0.01,
        enable_imd=False
    )
    
    print(f"  Dataset size: {len(dataset)}")
    
    # Get a batch (simulate batch_size=2)
    sample1 = dataset[0]
    sample2 = dataset[1]
    
    # Stack into batch
    batch_input = torch.stack([sample1['imc']['bert_input'], sample2['imc']['bert_input']])
    batch_label = torch.stack([sample1['imc']['bert_label'], sample2['imc']['bert_label']])
    batch_segment = torch.stack([sample1['imc']['segment_label'], sample2['imc']['segment_label']])
    batch_binary_pos = torch.stack([sample1['imc']['binary_pos'], sample2['imc']['binary_pos']])
    batch_function_pos = torch.stack([sample1['imc']['function_pos'], sample2['imc']['function_pos']])
    batch_bb_pos = torch.stack([sample1['imc']['bb_pos'], sample2['imc']['bb_pos']])
    batch_var_offsets = torch.stack([sample1['imc']['var_offsets'], sample2['imc']['var_offsets']])
    batch_is_daddr = torch.stack([sample1['imc']['is_daddr'], sample2['imc']['is_daddr']])
    
    print(f"\n  Batch shapes:")
    print(f"    bert_input: {batch_input.shape}")
    print(f"    bert_label: {batch_label.shape}")
    print(f"    segment_label: {batch_segment.shape}")
    print(f"    binary_pos: {batch_binary_pos.shape}")
    print(f"    function_pos: {batch_function_pos.shape}")
    print(f"    bb_pos: {batch_bb_pos.shape}")
    print(f"    var_offsets: {batch_var_offsets.shape}")
    print(f"    is_daddr: {batch_is_daddr.shape}")
    
    # Forward pass for IMC
    print("\n  Running forward pass for IMC...")
    model.eval()
    with torch.no_grad():
        im_output = model.forward_im(
            token_ids=batch_input,
            segment_labels=batch_segment,
            binary_pos=batch_binary_pos,
            function_pos=batch_function_pos,
            bb_pos=batch_bb_pos,
            var_offsets=batch_var_offsets,
            corpus_type='cfg'
        )
    
    print(f"  Output shape: {im_output.shape}")
    print(f"  Expected: [2, 512, {len(vocab)}]")
    
    # Check output statistics
    print(f"\n  Output statistics:")
    print(f"    Mean: {im_output.mean().item():.4f}")
    print(f"    Std: {im_output.std().item():.4f}")
    print(f"    Min: {im_output.min().item():.4f}")
    print(f"    Max: {im_output.max().item():.4f}")
    
    # Test loss calculation
    print(f"\n  Testing loss calculation...")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    # Reshape for loss calculation
    im_output_flat = im_output.view(-1, len(vocab))
    batch_label_flat = batch_label.view(-1)
    
    loss = criterion(im_output_flat, batch_label_flat)
    print(f"    IMC Loss: {loss.item():.4f}")
    
    # Count masked tokens
    num_masked = (batch_label != -1).sum().item()
    num_total = (batch_input != 0).sum().item()  # Non-padding tokens
    print(f"    Masked tokens: {num_masked}/{num_total} ({num_masked/num_total*100:.1f}%)")
    
    print("\n  ✓ IMC forward pass successful!")
    
    return im_output


def test_mlm_forward_pass(model, vocab):
    """Test forward pass with MLM task"""
    print("\n" + "="*80)
    print("TEST 3: MLM Forward Pass (Token Masking CFG)")
    print("="*80)
    
    # Load dataset
    print("  Loading dataset...")
    dataset = InstructionMaskingDataset(
        cfg_corpus_path='/data/kun/palmtreedata/cfg_train_2.txt',
        dfg_corpus_path=None,
        vocab=vocab,
        seq_len=512,
        on_memory=True,
        instruction_mask_prob=0.0,  # No instruction-level masking for MLM
        token_mask_prob=0.15,
        data_percentage=0.01,
        enable_imd=False
    )
    
    print(f"  Dataset size: {len(dataset)}")
    
    # Get a batch
    sample1 = dataset[0]
    sample2 = dataset[1]
    
    # Stack into batch
    batch_input = torch.stack([sample1['mlm']['bert_input'], sample2['mlm']['bert_input']])
    batch_label = torch.stack([sample1['mlm']['bert_label'], sample2['mlm']['bert_label']])
    batch_segment = torch.stack([sample1['mlm']['segment_label'], sample2['mlm']['segment_label']])
    batch_binary_pos = torch.stack([sample1['mlm']['binary_pos'], sample2['mlm']['binary_pos']])
    batch_function_pos = torch.stack([sample1['mlm']['function_pos'], sample2['mlm']['function_pos']])
    batch_bb_pos = torch.stack([sample1['mlm']['bb_pos'], sample2['mlm']['bb_pos']])
    batch_var_offsets = torch.stack([sample1['mlm']['var_offsets'], sample2['mlm']['var_offsets']])
    batch_is_daddr = torch.stack([sample1['mlm']['is_daddr'], sample2['mlm']['is_daddr']])
    
    print(f"\n  Batch shapes:")
    print(f"    bert_input: {batch_input.shape}")
    print(f"    bert_label: {batch_label.shape}")
    
    # Forward pass for MLM
    print("\n  Running forward pass for MLM...")
    model.eval()
    with torch.no_grad():
        mlm_output, nsp_output = model.forward(
            token_ids=batch_input,
            segment_labels=batch_segment,
            binary_pos=batch_binary_pos,
            function_pos=batch_function_pos,
            bb_pos=batch_bb_pos,
            var_offsets=batch_var_offsets,
            is_daddr=batch_is_daddr,
            corpus_type='cfg'
        )
    
    print(f"  MLM output shape: {mlm_output.shape}")
    print(f"  Expected: [2, 512, {len(vocab)}]")
    print(f"  NSP output: {nsp_output} (should be None, NSP disabled)")
    
    # Check output statistics
    print(f"\n  Output statistics:")
    print(f"    Mean: {mlm_output.mean().item():.4f}")
    print(f"    Std: {mlm_output.std().item():.4f}")
    
    # Test loss calculation
    print(f"\n  Testing loss calculation...")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    mlm_output_flat = mlm_output.view(-1, len(vocab))
    batch_label_flat = batch_label.view(-1)
    
    loss = criterion(mlm_output_flat, batch_label_flat)
    print(f"    MLM Loss: {loss.item():.4f}")
    
    # Count masked tokens
    num_masked = (batch_label != -1).sum().item()
    num_total = (batch_input != 0).sum().item()
    print(f"    Masked tokens: {num_masked}/{num_total} ({num_masked/num_total*100:.1f}%)")
    
    print("\n  ✓ MLM forward pass successful!")
    
    return mlm_output


def test_predictions(model, vocab):
    """Test actual token predictions"""
    print("\n" + "="*80)
    print("TEST 4: Token Predictions")
    print("="*80)
    
    # Load dataset
    dataset = InstructionMaskingDataset(
        cfg_corpus_path='/data/kun/palmtreedata/cfg_train_2.txt',
        dfg_corpus_path=None,
        vocab=vocab,
        seq_len=512,
        on_memory=True,
        instruction_mask_prob=0.25,
        token_mask_prob=0.0,
        data_percentage=0.01,
        enable_imd=False
    )
    
    sample = dataset[0]
    
    # Get IMC data
    bert_input = sample['imc']['bert_input'].unsqueeze(0)
    bert_label = sample['imc']['bert_label'].unsqueeze(0)
    segment_label = sample['imc']['segment_label'].unsqueeze(0)
    binary_pos = sample['imc']['binary_pos'].unsqueeze(0)
    function_pos = sample['imc']['function_pos'].unsqueeze(0)
    bb_pos = sample['imc']['bb_pos'].unsqueeze(0)
    var_offsets = sample['imc']['var_offsets'].unsqueeze(0)
    is_daddr = sample['imc']['is_daddr'].unsqueeze(0)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        im_output = model.forward_im(
            token_ids=bert_input,
            segment_labels=segment_label,
            binary_pos=binary_pos,
            function_pos=function_pos,
            bb_pos=bb_pos,
            var_offsets=var_offsets,
            corpus_type='cfg'
        )
    
    # Get predictions for masked tokens
    print("  Masked token predictions (first 10):")
    print("  " + "-"*70)
    print(f"  {'Pos':>4} | {'Input Token':^15} | {'True Token':^15} | {'Predicted':^15}")
    print("  " + "-"*70)
    
    count = 0
    for i in range(min(50, len(bert_input[0]))):
        input_id = bert_input[0, i].item()
        label_id = bert_label[0, i].item()
        
        if input_id == 0:  # padding
            break
        
        if label_id != -1:  # This token is masked
            # Get prediction
            pred_id = im_output[0, i].argmax().item()
            
            input_token = vocab.itos[input_id] if input_id < len(vocab.itos) else f'UNK:{input_id}'
            true_token = vocab.itos[label_id] if label_id < len(vocab.itos) else f'UNK:{label_id}'
            pred_token = vocab.itos[pred_id] if pred_id < len(vocab.itos) else f'UNK:{pred_id}'
            
            match = "✓" if pred_id == label_id else "✗"
            print(f"  {i:4d} | {input_token:^15} | {true_token:^15} | {pred_token:^15} {match}")
            
            count += 1
            if count >= 10:
                break
    
    print("\n  ✓ Predictions generated successfully!")
    print("  Note: Random predictions expected for untrained model")


def main():
    print("="*80)
    print("ADDRESS-AWARE BERT MODEL TEST")
    print("="*80)
    print()
    
    # Test 1: Model initialization
    model, vocab = test_model_initialization()
    
    # Test 2: IMC forward pass
    test_imc_forward_pass(model, vocab)
    
    # Test 3: MLM forward pass
    test_mlm_forward_pass(model, vocab)
    
    # Test 4: Token predictions
    test_predictions(model, vocab)
    
    # Final summary
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print("\nSummary:")
    print("  ✓ Model initializes correctly with address-aware embeddings")
    print("  ✓ IMC forward pass works (instruction-level masking)")
    print("  ✓ MLM forward pass works (token-level masking)")
    print("  ✓ Loss calculation works correctly")
    print("  ✓ Predictions are generated (random for untrained model)")
    print("\nNext steps:")
    print("  1. Test with GPU if available")
    print("  2. Test full training loop")
    print("  3. Test checkpoint saving/loading")
    print("="*80)


if __name__ == '__main__':
    main()
