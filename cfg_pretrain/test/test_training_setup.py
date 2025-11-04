"""
Test script to verify training setup before full training
"""

import torch
import os
import pickle
import sys

# Test imports
try:
    from data_loader import CFGPretrainDataset
    from config import (
        VOCAB_FILE, BB_PAIRS_FILE, MAX_SEQ_LEN, EMBED_DIM,
        NUM_HEADS, NUM_LAYERS, BATCH_SIZE, LEARNING_RATE,
        NUM_EPOCHS, MLM_PROBABILITY
    )
    print("✓ Successfully imported data_loader and config")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def main():
    print("\n" + "="*80)
    print("CFG PRETRAINING SETUP VERIFICATION")
    print("="*80)
    
    # 1. Check vocabulary
    print("\n1. Checking Extended Vocabulary...")
    vocab_path = os.path.join(os.path.dirname(__file__), VOCAB_FILE)
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"   ✓ Vocabulary loaded: {len(vocab)} tokens")
        
        # Check address tokens
        addr_tokens = ['addr_start', 'addr_end', 'addr_code', 'addr_data']
        for token in addr_tokens:
            if token in vocab.stoi:
                print(f"   ✓ {token}: ID {vocab.stoi[token]}")
            else:
                print(f"   ✗ {token}: NOT FOUND")
    else:
        print(f"   ✗ Vocabulary file not found: {vocab_path}")
        return
    
    # 2. Check PalmTree model
    print("\n2. Checking PalmTree Pre-trained Model...")
    palmtree_path = os.path.join(os.path.dirname(__file__), "..", "pre-trained_model", "palmtree", "transformer.ep19")
    if os.path.exists(palmtree_path):
        print(f"   ✓ PalmTree model found: {palmtree_path}")
        try:
            # Allow loading of PalmTree's BERT model (trust the source)
            model = torch.load(palmtree_path, map_location='cpu', weights_only=False)
            if hasattr(model, 'embedding'):
                emb_weight = model.embedding.token.weight
                print(f"   ✓ Embeddings shape: {emb_weight.shape}")
                print(f"     - Vocab size: {emb_weight.shape[0]}")
                print(f"     - Embed dim: {emb_weight.shape[1]}")
            else:
                print(f"   ✗ Model structure unexpected")
        except Exception as e:
            print(f"   ✗ Error loading model: {e}")
    else:
        print(f"   ✗ PalmTree model not found: {palmtree_path}")
    
    # 3. Check training data
    print("\n3. Checking Training Data...")
    data_path = os.path.join(os.path.dirname(__file__), "..", BB_PAIRS_FILE)
    if os.path.exists(data_path):
        print(f"   ✓ Training data found: {data_path}")
        with open(data_path, 'r') as f:
            lines = f.readlines()
        print(f"   ✓ Number of BB pairs: {len(lines):,}")
        
        # Show sample
        if lines:
            sample = lines[0][:200]
            print(f"   ✓ Sample (first 200 chars): {sample}...")
    else:
        print(f"   ✗ Training data not found: {data_path}")
    
    # 4. Test data loader
    print("\n4. Testing Data Loader...")
    try:
        dataset = CFGPretrainDataset(
            data_file=data_path,
            vocab_file=vocab_path,
            max_seq_length=MAX_SEQ_LEN,
            mlm_probability=MLM_PROBABILITY
        )
        print(f"   ✓ Dataset created: {len(dataset)} samples")
        
        # Test loading one sample
        sample = dataset[0]
        print(f"   ✓ Sample loaded successfully")
        print(f"     - Keys: {list(sample.keys())}")
        for key, value in sample.items():
            if torch.is_tensor(value):
                print(f"     - {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"     - {key}: {type(value)}")
    except Exception as e:
        print(f"   ✗ Data loader error: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Configuration summary
    print("\n5. Training Configuration:")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Max sequence length: {MAX_SEQ_LEN}")
    print(f"   - Embedding dimension: {EMBED_DIM}")
    print(f"   - Attention heads: {NUM_HEADS}")
    print(f"   - Transformer layers: {NUM_LAYERS}")
    print(f"   - MLM probability: {MLM_PROBABILITY}")
    
    # 6. GPU availability
    print("\n6. Hardware:")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available")
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"   ⚠ CUDA not available, will use CPU")
    
    print("\n" + "="*80)
    print("SETUP VERIFICATION COMPLETE")
    print("="*80)
    print("\nIf all checks passed, you can run: python train.py")
    print()

if __name__ == "__main__":
    main()
