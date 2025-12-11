#!/usr/bin/env python3
"""
Debug script for FunctionSimilarityDataset

Tests the dataloader to verify:
1. Instruction parsing (tokens, positions, var offsets)
2. Function processing
3. Pair generation
"""

import sys
import os
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'strupos'))

from vocab import WordVocab
import importlib.util

# Load the dataloader module
current_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("funcsim_dataloader", os.path.join(current_dir, "dataloader.py"))
funcsim_dataloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(funcsim_dataloader)

FunctionSimilarityDataset = funcsim_dataloader.FunctionSimilarityDataset


def test_instruction_parsing():
    """Test parsing of different instruction formats"""
    print("=" * 80)
    print("TEST 1: Instruction Parsing")
    print("=" * 80)
    
    # Load vocab
    vocab = WordVocab.load_vocab("../../strupos/vocab.pkl")
    
    # Create a minimal dataset instance just to access parsing method
    dataset = FunctionSimilarityDataset.__new__(FunctionSimilarityDataset)
    dataset.vocab = vocab
    dataset.addr_pattern = funcsim_dataloader.re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    dataset.nested_addr_pattern = funcsim_dataloader.re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    dataset.var_pattern = funcsim_dataloader.re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    # Test cases
    test_instructions = [
        "mov(0x401000:0.5:0.2:0.1) rax rbx",
        "add(0x401004:0.5:0.25:0.15) rax 0x10",
        "mov(0x401008:0.5:0.3:0.2) rax address(0x404000:0.6:0.0:0.0)",
        "lea(0x40100c:0.5:0.35:0.25) rax [ rbp + var(0x08) ]",
        "call(0x401010:0.5:0.4:0.3) address(0x402000:0.7:0.0:0.0)",
    ]
    
    for i, inst in enumerate(test_instructions, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {inst}")
        
        try:
            tokens, positions, var_offsets = dataset._parse_instruction(inst)
            print(f"Tokens: {tokens}")
            print(f"Positions: {positions}")
            print(f"Var offsets: {var_offsets}")
            
            # Verify lengths match
            assert len(tokens) == len(positions) == len(var_offsets), \
                f"Length mismatch! tokens={len(tokens)}, positions={len(positions)}, var_offsets={len(var_offsets)}"
            print("✓ Lengths match")
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()


def test_function_processing():
    """Test processing a sample function"""
    print("\n" + "=" * 80)
    print("TEST 2: Function Processing")
    print("=" * 80)
    
    # Load vocab
    vocab = WordVocab.load_vocab("../../strupos/vocab.pkl")
    
    # Create sample function blocks
    sample_blocks = {
        "1": [
            "push(0x401000:0.1:0.0:0.0) rbp",
            "mov(0x401001:0.1:0.05:0.05) rbp rsp",
            "sub(0x401004:0.1:0.1:0.1) rsp 0x20",
            "mov(0x401008:0.1:0.15:0.15) address(0x404000:0.9:0.0:0.0) rax",
            "lea(0x40100c:0.1:0.2:0.2) rax [ rbp + var(0x08) ]",
        ],
        "2": [
            "push(0x402000:0.2:0.0:0.0) rbp",
            "mov(0x402001:0.2:0.05:0.05) rbp rsp",
        ]
    }
    
    sample_pairs = {
        "1": {"ground_truth": ["2"]},
        "2": {"ground_truth": ["1"]},
    }
    
    # Save to temp files
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_blocks, f)
        blocks_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_pairs, f)
        pairs_file = f.name
    
    try:
        # Create dataset
        print("\nCreating dataset...")
        dataset = FunctionSimilarityDataset(
            function_blocks_file=blocks_file,
            funcsim_pairs_file=pairs_file,
            vocab=vocab,
            seq_len=64,
            negative_samples=1
        )
        
        print(f"\n✓ Dataset created successfully")
        print(f"  Function blocks: {len(dataset.function_blocks)}")
        print(f"  Training pairs: {len(dataset.training_pairs)}")
        
        # Test getting a sample
        print("\n--- Testing __getitem__ ---")
        sample = dataset[0]
        
        print(f"\nSample keys: {sample.keys()}")
        print(f"func1_input shape: {sample['func1_input'].shape}")
        print(f"func1_binary_pos shape: {sample['func1_binary_pos'].shape}")
        print(f"func1_function_pos shape: {sample['func1_function_pos'].shape}")
        print(f"func1_bb_pos shape: {sample['func1_bb_pos'].shape}")
        print(f"Label: {sample['label'].item()}")
        
        # Show first few tokens and positions
        print(f"\nFirst 10 tokens (func1): {sample['func1_input'][:10].tolist()}")
        print(f"First 10 binary_pos: {sample['func1_binary_pos'][:10].tolist()}")
        print(f"First 10 function_pos: {sample['func1_function_pos'][:10].tolist()}")
        print(f"First 10 bb_pos: {sample['func1_bb_pos'][:10].tolist()}")
        
        # Decode tokens
        print(f"\nDecoded tokens (func1):")
        for i in range(min(10, len(sample['func1_input']))):
            token_id = sample['func1_input'][i].item()
            if token_id == 0:  # padding
                break
            token = vocab.itos.get(token_id, '<UNK>')
            bin_pos = sample['func1_binary_pos'][i].item()
            func_pos = sample['func1_function_pos'][i].item()
            bb_pos = sample['func1_bb_pos'][i].item()
            print(f"  [{i}] {token:15s} | bin:{bin_pos:.3f} func:{func_pos:.3f} bb:{bb_pos:.3f}")
        
        print("\n✓ Function processing works!")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        os.unlink(blocks_file)
        os.unlink(pairs_file)


def test_real_data():
    """Test with actual funcsim data if available"""
    print("\n" + "=" * 80)
    print("TEST 3: Real Data Loading")
    print("=" * 80)
    
    blocks_file = "/data/kun/funcsim_match/function_blocks.json"
    pairs_file = "/data/kun/funcsim_match/funcsim_pairs.json"
    
    if not os.path.exists(blocks_file):
        print(f"⊘ Skipping - blocks file not found: {blocks_file}")
        return
    
    if not os.path.exists(pairs_file):
        print(f"⊘ Skipping - pairs file not found: {pairs_file}")
        return
    
    print(f"Loading vocab...")
    vocab = WordVocab.load_vocab("../../strupos/vocab.pkl")
    
    print(f"Creating dataset with real data...")
    print(f"  Blocks: {blocks_file}")
    print(f"  Pairs: {pairs_file}")
    
    try:
        dataset = FunctionSimilarityDataset(
            function_blocks_file=blocks_file,
            funcsim_pairs_file=pairs_file,
            vocab=vocab,
            seq_len=512,
            negative_samples=3
        )
        
        print(f"\n✓ Real dataset loaded successfully!")
        print(f"  Total training pairs: {len(dataset)}")
        
        # Get a sample
        print(f"\nTesting random sample...")
        import random
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        
        print(f"  Sample index: {idx}")
        print(f"  Label: {'POSITIVE' if sample['label'].item() == 1 else 'NEGATIVE'}")
        print(f"  func1 non-padding tokens: {(sample['func1_input'] != 0).sum().item()}")
        print(f"  func2 non-padding tokens: {(sample['func2_input'] != 0).sum().item()}")
        
        print("\n✓ Real data test passed!")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "=" * 80)
    print("FunctionSimilarityDataset Debug Script")
    print("=" * 80)
    
    # Run tests
    test_instruction_parsing()
    test_function_processing()
    test_real_data()
    
    print("\n" + "=" * 80)
    print("Debug Complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
