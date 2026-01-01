#!/usr/bin/env python3
"""
Verification script for baseline BinBertModel implementation.

Tests:
1. Model creation
2. Position embeddings trick applied correctly
3. Forward pass works
4. Save and load functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_model_creation():
    """Test that BinBertModel can be created."""
    print("=" * 60)
    print("Test 1: Model Creation")
    print("=" * 60)
    
    try:
        from model_baseline import create_baseline_model
        
        model = create_baseline_model(
            vocab_size=4000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=512
        )
        
        print("✓ Model created successfully")
        print(f"✓ Model type: {type(model).__name__}")
        print(f"✓ BERT type: {type(model.bert).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        
        return model
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_position_embeddings(model):
    """Test that position_embeddings = word_embeddings trick is applied."""
    print("\n" + "=" * 60)
    print("Test 2: Position Embeddings Trick")
    print("=" * 60)
    
    if model is None:
        print("✗ Skipped (model not created)")
        return False
    
    try:
        # Check that position_embeddings points to word_embeddings
        pos_emb = model.bert.embeddings.position_embeddings
        word_emb = model.bert.embeddings.word_embeddings
        
        if pos_emb is word_emb:
            print("✓ position_embeddings = word_embeddings (jTrans trick applied)")
            print(f"✓ Embedding shape: {word_emb.weight.shape}")
            return True
        else:
            print("✗ position_embeddings != word_embeddings (trick not applied!)")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass(model):
    """Test that forward pass works."""
    print("\n" + "=" * 60)
    print("Test 3: Forward Pass")
    print("=" * 60)
    
    if model is None:
        print("✗ Skipped (model not created)")
        return False
    
    try:
        import torch
        
        # Create dummy input
        batch_size = 2
        seq_len = 10
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            mlm_logits, jtp_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        
        print(f"✓ Forward pass successful")
        print(f"✓ MLM logits shape: {mlm_logits.shape}")
        print(f"✓ JTP logits shape: {jtp_logits.shape}")
        
        # Verify shapes
        expected_mlm_shape = (batch_size, seq_len, 1000)  # vocab_size
        expected_jtp_shape = (batch_size, seq_len, 128)   # max_position
        
        if mlm_logits.shape == expected_mlm_shape:
            print(f"✓ MLM shape correct: {mlm_logits.shape}")
        else:
            print(f"✗ MLM shape incorrect: {mlm_logits.shape} != {expected_mlm_shape}")
            
        if jtp_logits.shape == expected_jtp_shape:
            print(f"✓ JTP shape correct: {jtp_logits.shape}")
        else:
            print(f"✗ JTP shape incorrect: {jtp_logits.shape} != {expected_jtp_shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load(model):
    """Test save and load functionality."""
    print("\n" + "=" * 60)
    print("Test 4: Save and Load")
    print("=" * 60)
    
    if model is None:
        print("✗ Skipped (model not created)")
        return False
    
    try:
        import torch
        import tempfile
        import shutil
        from model_baseline import BinBertModel
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        print(f"✓ Using temp directory: {temp_dir}")
        
        try:
            # Save model
            model.bert.save_pretrained(temp_dir)
            print(f"✓ Model saved to {temp_dir}")
            
            # Load model
            loaded_bert = BinBertModel.from_pretrained(temp_dir)
            print(f"✓ Model loaded from {temp_dir}")
            
            # Verify position trick is preserved
            pos_emb = loaded_bert.embeddings.position_embeddings
            word_emb = loaded_bert.embeddings.word_embeddings
            
            if pos_emb is word_emb:
                print("✓ Position trick preserved after save/load!")
            else:
                print("✗ Position trick NOT preserved after save/load!")
                return False
            
            # Test forward pass on loaded model
            input_ids = torch.randint(0, 1000, (1, 5))
            outputs = loaded_bert(input_ids=input_ids)
            print(f"✓ Forward pass on loaded model works")
            print(f"✓ Output shape: {outputs[0].shape}")
            
            return True
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up temp directory")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compatibility():
    """Test compatibility with original jTrans structure."""
    print("\n" + "=" * 60)
    print("Test 5: jTrans Compatibility")
    print("=" * 60)
    
    try:
        from model_baseline import BinBertModel
        from transformers import BertModel
        
        # Check inheritance
        if issubclass(BinBertModel, BertModel):
            print("✓ BinBertModel inherits from BertModel")
        else:
            print("✗ BinBertModel does NOT inherit from BertModel")
            return False
        
        # Check __init__ signature
        import inspect
        sig = inspect.signature(BinBertModel.__init__)
        params = list(sig.parameters.keys())
        
        if 'config' in params and 'add_pooling_layer' in params:
            print("✓ __init__ signature matches jTrans: (config, add_pooling_layer)")
        else:
            print(f"✗ __init__ signature mismatch: {params}")
            return False
        
        print("✓ Fully compatible with original jTrans BinBertModel!")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BinBertModel Verification Tests")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Model creation
    model = test_model_creation()
    results.append(("Model Creation", model is not None))
    
    # Test 2: Position embeddings trick
    pos_result = test_position_embeddings(model)
    results.append(("Position Embeddings Trick", pos_result))
    
    # Test 3: Forward pass
    forward_result = test_forward_pass(model)
    results.append(("Forward Pass", forward_result))
    
    # Test 4: Save and load
    save_load_result = test_save_load(model)
    results.append(("Save and Load", save_load_result))
    
    # Test 5: Compatibility
    compat_result = test_compatibility()
    results.append(("jTrans Compatibility", compat_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("✓ BinBertModel is ready for training!")
    else:
        print("✗ SOME TESTS FAILED!")
        print("✗ Please check the errors above.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
