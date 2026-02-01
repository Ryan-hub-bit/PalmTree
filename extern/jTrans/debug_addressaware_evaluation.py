#!/usr/bin/env python3
"""
Debug script for address-aware evaluation
Checks: model loading, tokenization, embeddings, similarities
"""

import torch
import torch.nn.functional as F
import json
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Import from finetune.py
import sys
sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans')
sys.path.insert(0, '/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware')
from finetune import AddressAwareBertWrapper
from transformers import BertConfig, BertModel, BertTokenizer
from model_addressaware import AddressAwareBERTEmbedding


def load_vocab_and_tokenizer(vocab_path):
    """Load vocab_stoi and tokenizer"""
    vocab_file = Path(vocab_path) / 'vocab.txt'
    with open(vocab_file, 'r') as f:
        vocab = [line.strip() for line in f]
    
    vocab_stoi = {}
    for idx, token in enumerate(vocab):
        vocab_stoi[token] = idx
    
    # Use BertTokenizer like evaluate_addressaware_pools.py
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    
    return vocab_stoi, tokenizer


def tokenize_function(func_block, tokenizer, vocab_stoi, max_len=512):
    """Tokenize a single function with address-aware features"""
    # Tokens are already space-separated string
    tokens = func_block['tokens'].strip().split()
    
    # Get pre-computed position encodings
    binary_pos = func_block.get('binary_pos', [])
    function_pos = func_block.get('function_pos', [])
    bb_pos = func_block.get('bb_pos', [])
    var_offsets = func_block.get('var_offsets', [])
    
    # Add [CLS] and [SEP]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    binary_pos = [-1.0] + binary_pos + [-1.0]
    function_pos = [-1.0] + function_pos + [-1.0]
    bb_pos = [-1.0] + bb_pos + [-1.0]
    var_offsets = [-1] + var_offsets + [-1]
    
    # Truncate if needed
    if len(tokens) > max_len:
        tokens = tokens[:max_len-1] + ['[SEP]']
        binary_pos = binary_pos[:max_len-1] + [-1.0]
        function_pos = function_pos[:max_len-1] + [-1.0]
        bb_pos = bb_pos[:max_len-1] + [-1.0]
        var_offsets = var_offsets[:max_len-1] + [-1]
    
    # Convert tokens to IDs using vocab_stoi
    token_ids = [vocab_stoi.get(t, vocab_stoi.get('[UNK]', 1)) for t in tokens]
    
    # Pad to max_len
    padding_len = max_len - len(token_ids)
    token_ids += [vocab_stoi.get('[PAD]', 0)] * padding_len
    binary_pos += [-1.0] * padding_len
    function_pos += [-1.0] * padding_len
    bb_pos += [-1.0] * padding_len
    var_offsets += [-1] * padding_len
    
    return token_ids, binary_pos, function_pos, bb_pos, var_offsets


def generate_embeddings(model, func_ids, func_blocks, tokenizer, vocab_stoi, device, max_len=512, batch_size=32):
    """Generate embeddings for a list of function IDs"""
    embeddings = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(func_ids), batch_size), desc="Generating embeddings"):
            batch_ids = func_ids[i:i+batch_size]
            
            # Tokenize batch
            batch_input_ids = []
            batch_binary_pos = []
            batch_function_pos = []
            batch_bb_pos = []
            batch_var_offsets = []
            batch_attention_mask = []
            
            for fid in batch_ids:
                token_ids, bin_pos, func_pos, bb_p, var_off = tokenize_function(
                    func_blocks[str(fid)], tokenizer, vocab_stoi, max_len
                )
                
                batch_input_ids.append(token_ids)
                batch_binary_pos.append(bin_pos)
                batch_function_pos.append(func_pos)
                batch_bb_pos.append(bb_p)
                batch_var_offsets.append(var_off)
                
                # Create attention mask
                attention_mask = [1 if tid != vocab_stoi.get('[PAD]', 0) else 0 for tid in token_ids]
                batch_attention_mask.append(attention_mask)
            
            # Convert to tensors
            input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(device)
            binary_pos = torch.tensor(batch_binary_pos, dtype=torch.float).to(device)
            function_pos = torch.tensor(batch_function_pos, dtype=torch.float).to(device)
            bb_pos = torch.tensor(batch_bb_pos, dtype=torch.float).to(device)
            var_offsets = torch.tensor(batch_var_offsets, dtype=torch.long).to(device)
            attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(device)
            token_type_ids = torch.zeros_like(input_ids)
            
            # Get embeddings
            batch_outputs = model(
                token_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                binary_pos=binary_pos,
                function_pos=function_pos,
                bb_pos=bb_pos,
                var_offsets=var_offsets
            )
            
            # Extract pooler_output (CLS token embeddings)
            batch_embeddings = batch_outputs['pooler_output']
            embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)


def debug_tokenization(func_blocks, func_ids, vocab_stoi):
    """Debug tokenization for sample functions"""
    print("\n" + "="*80)
    print("STEP 1: Check Tokenization")
    print("="*80)
    
    for func_id in func_ids[:3]:
        func = func_blocks[str(func_id)]
        tokens_str = func['tokens']
        tokens = tokens_str.strip().split()
        
        print(f"\nFunction ID: {func_id}")
        print(f"  Binary: {func.get('binary_name', 'N/A')}")
        print(f"  Function: {func.get('function_name', 'N/A')}")
        print(f"  Tokens: {len(tokens)}")
        print(f"  First 20 tokens: {' '.join(tokens[:20])}")
        
        # Check address/daddr tokens (processed format)
        addr_tokens = [t for t in tokens if t in ['address', 'daddr']]
        print(f"  Address/daddr tokens: {len(addr_tokens)}")
        
        # Check position encodings
        binary_pos = func.get('binary_pos', [])
        function_pos = func.get('function_pos', [])
        bb_pos = func.get('bb_pos', [])
        
        print(f"  Position encodings:")
        print(f"    binary_pos: {len(binary_pos)} values, range=[{min(binary_pos) if binary_pos else 'N/A'}..{max(binary_pos) if binary_pos else 'N/A'}]")
        print(f"    function_pos: {len(function_pos)} values, range=[{min(function_pos) if function_pos else 'N/A'}..{max(function_pos) if function_pos else 'N/A'}]")
        print(f"    bb_pos: {len(bb_pos)} values, range=[{min(bb_pos) if bb_pos else 'N/A'}..{max(bb_pos) if bb_pos else 'N/A'}]")
        
        # Check if in vocab
        if addr_tokens:
            print(f"  ✓ Address tokens in vocab: 'address' and 'daddr'")
        
        # Sample a few tokens with their positions
        print(f"  Sample (token, binary_pos, function_pos, bb_pos):")
        for i in range(min(5, len(tokens))):
            bp = binary_pos[i] if i < len(binary_pos) else 'N/A'
            fp = function_pos[i] if i < len(function_pos) else 'N/A'
            bbp = bb_pos[i] if i < len(bb_pos) else 'N/A'
            print(f"    [{i}] {tokens[i]:12s} {bp:8.4f} {fp:8.4f} {bbp:8.4f}" if isinstance(bp, float) else f"    [{i}] {tokens[i]:12s} N/A")


def debug_model_loading(model, device):
    """Debug model loading and parameters"""
    print("\n" + "="*80)
    print("STEP 2: Check Model Loading")
    print("="*80)
    
    # Check model type
    print(f"Model type: {type(model.bert.embeddings).__name__}")
    
    # Check if vocab_stoi is properly loaded
    if hasattr(model.bert.embeddings, 'vocab_stoi'):
        vocab_stoi = model.bert.embeddings.vocab_stoi
        print(f"✓ vocab_stoi loaded: {len(vocab_stoi)} entries")
        
        # Check for address tokens
        addr_keys = [k for k in vocab_stoi.keys() if 'address' in k or 'daddr' in k]
        print(f"  Address tokens in vocab_stoi: {len(addr_keys)}")
        if addr_keys:
            print(f"  Sample: {list(addr_keys)[:5]}")
    else:
        print("✗ WARNING: vocab_stoi NOT found in model!")
    
    # Check address position embedding
    if hasattr(model.bert.embeddings, 'address_position'):
        print("✓ address_position embedding exists")
        addr_pos = model.bert.embeddings.address_position
        print(f"  Type: {type(addr_pos).__name__}")
    else:
        print("✗ WARNING: address_position NOT found!")
    
    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def debug_embeddings(model, func_blocks, func_ids, tokenizer, vocab_stoi, device, max_len=512):
    """Debug embedding generation using proper evaluation pipeline"""
    print("\n" + "="*80)
    print("STEP 3: Check Embedding Generation")
    print("="*80)
    
    # Use the proper generate_embeddings function
    test_ids = func_ids[:10]
    print(f"Generating embeddings for {len(test_ids)} test functions...")
    embeddings = generate_embeddings(
        model, test_ids, func_blocks, tokenizer, vocab_stoi, device, max_len, batch_size=4
    )
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    
    # Check embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")
    
    # Check if embeddings are all the same (collapsed)
    emb_norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding norms:")
    print(f"  Mean norm: {emb_norms.mean():.6f}")
    print(f"  Std norm: {emb_norms.std():.6f}")
    
    # Check pairwise similarities
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    
    print(f"\nPairwise cosine similarities:")
    print(f"  Mean: {sim_matrix.mean():.6f}")
    print(f"  Std: {sim_matrix.std():.6f}")
    print(f"  Min (off-diagonal): {np.min(sim_matrix - np.eye(len(sim_matrix))):.6f}")
    print(f"  Max (off-diagonal): {np.max(sim_matrix - np.eye(len(sim_matrix)) * 2):.6f}")
    
    # WARNING: Check for embedding collapse
    if sim_matrix.std() < 0.01:
        print("\n⚠️  WARNING: Embeddings are very similar! Possible collapse!")
    
    if emb_norms.std() < 0.1:
        print("⚠️  WARNING: Embedding norms are very uniform!")
    
    return embeddings


def debug_ground_truth_pairs(model, func_blocks, ground_truth, tokenizer, vocab_stoi, device, max_len=512):
    """Check similarity for actual ground truth pairs"""
    print("\n" + "="*80)
    print("STEP 4: Check Ground Truth Pair Similarities")
    print("="*80)
    
    # Sample some ground truth pairs
    sample_pairs = []
    for opt_pair, pairs in ground_truth.items():
        sample_pairs.extend(pairs[:5])  # 5 pairs per opt level
        if len(sample_pairs) >= 20:
            break
    
    print(f"Checking {len(sample_pairs[:20])} ground truth pairs...")
    
    # Get embeddings for low and high opt functions
    low_ids = [pair[0] for pair in sample_pairs[:20]]
    high_ids = [pair[1] for pair in sample_pairs[:20]]
    
    low_embeddings = generate_embeddings(
        model, low_ids, func_blocks, tokenizer, vocab_stoi, device, max_len, batch_size=4
    )
    high_embeddings = generate_embeddings(
        model, high_ids, func_blocks, tokenizer, vocab_stoi, device, max_len, batch_size=4
    )
    
    # Compute similarities
    similarities = []
    for i in range(len(low_embeddings)):
        low_emb = torch.tensor(low_embeddings[i:i+1])
        high_emb = torch.tensor(high_embeddings[i:i+1])
        sim = F.cosine_similarity(low_emb, high_emb).item()
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    print(f"\nGround truth pair similarities:")
    print(f"  Mean: {similarities.mean():.6f}")
    print(f"  Std: {similarities.std():.6f}")
    print(f"  Min: {similarities.min():.6f}")
    print(f"  Max: {similarities.max():.6f}")
    print(f"  Median: {np.median(similarities):.6f}")
    
    # Check distribution
    high_sim = (similarities > 0.8).sum()
    medium_sim = ((similarities > 0.5) & (similarities <= 0.8)).sum()
    low_sim = (similarities <= 0.5).sum()
    
    print(f"\nSimilarity distribution:")
    print(f"  High (>0.8): {high_sim}/{len(similarities)} ({high_sim/len(similarities)*100:.1f}%)")
    print(f"  Medium (0.5-0.8): {medium_sim}/{len(similarities)} ({medium_sim/len(similarities)*100:.1f}%)")
    print(f"  Low (<0.5): {low_sim}/{len(similarities)} ({low_sim/len(similarities)*100:.1f}%)")
    
    if similarities.mean() < 0.3:
        print("\n⚠️  WARNING: Ground truth pairs have LOW similarity!")
        print("   This suggests the model hasn't learned to recognize similar functions.")


def debug_random_pairs(model, func_blocks, func_ids, tokenizer, vocab_stoi, device, max_len=512):
    """Check similarity for random (non-matching) pairs"""
    print("\n" + "="*80)
    print("STEP 5: Check Random Pair Similarities (Negative Samples)")
    print("="*80)
    
    np.random.seed(42)
    indices1 = np.random.choice(min(len(func_ids), 100), 20, replace=False)
    indices2 = np.random.choice(min(len(func_ids), 100), 20, replace=False)
    
    # Make sure not the same indices
    while np.any(indices1 == indices2):
        indices2 = np.random.choice(min(len(func_ids), 100), 20, replace=False)
    
    ids1 = [func_ids[i] for i in indices1]
    ids2 = [func_ids[i] for i in indices2]
    
    print(f"Checking 20 random pairs...")
    
    # Get embeddings
    embeddings1 = generate_embeddings(
        model, ids1, func_blocks, tokenizer, vocab_stoi, device, max_len, batch_size=4
    )
    embeddings2 = generate_embeddings(
        model, ids2, func_blocks, tokenizer, vocab_stoi, device, max_len, batch_size=4
    )
    
    # Compute similarities
    similarities = []
    for i in range(len(embeddings1)):
        emb1 = torch.tensor(embeddings1[i:i+1])
        emb2 = torch.tensor(embeddings2[i:i+1])
        sim = F.cosine_similarity(emb1, emb2).item()
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    print(f"\nRandom pair similarities:")
    print(f"  Mean: {similarities.mean():.6f}")
    print(f"  Std: {similarities.std():.6f}")
    print(f"  Min: {similarities.min():.6f}")
    print(f"  Max: {similarities.max():.6f}")
    
    if similarities.mean() > 0.7:
        print("\n⚠️  WARNING: Random pairs have HIGH similarity!")
        print("   Model may be producing similar embeddings for all functions.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--vocab_path', required=True)
    parser.add_argument('--func_blocks', required=True)
    parser.add_argument('--ground_truth', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    print("="*80)
    print("ADDRESS-AWARE EVALUATION DEBUG")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Vocab: {args.vocab_path}")
    print(f"Device: {args.device}")
    
    # Load data
    print("\nLoading data...")
    with open(args.func_blocks, 'r') as f:
        func_blocks = json.load(f)
    with open(args.ground_truth, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"Loaded {len(func_blocks)} functions")
    
    # Load vocab_stoi and tokenizer
    vocab_stoi, tokenizer = load_vocab_and_tokenizer(args.vocab_path)
    print(f"Loaded vocab: {len(vocab_stoi)} tokens")
    
    # Load model
    print("\nLoading model...")
    device = torch.device(args.device)
    
    # Load model config and weights (imports already at top of file)
    config_path = os.path.join(args.model_path, 'config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = BertConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_hidden_layers=config_dict['num_hidden_layers'],
        num_attention_heads=config_dict['num_attention_heads'],
        intermediate_size=config_dict['hidden_size'] * 4,
        max_position_embeddings=config_dict['max_position_embeddings'],
        type_vocab_size=config_dict.get('type_vocab_size', 2),
    )
    
    bert_model = BertModel(config, add_pooling_layer=False)
    
    # Replace embeddings with address-aware version
    bert_model.embeddings = AddressAwareBERTEmbedding(
        vocab_size=config_dict['vocab_size'],
        embed_size=config_dict['hidden_size'],
        dropout=0.1,
        max_len=config_dict['max_position_embeddings'],
        use_address_embedding=True,
        use_var_embedding=True,
        segment_types=256,
        vocab_stoi=vocab_stoi
    )
    
    # Load pretrained weights
    weights_path = os.path.join(args.model_path, 'pytorch_model.bin')
    state_dict = torch.load(weights_path, map_location='cpu')
    bert_model.load_state_dict(state_dict)
    
    # Wrap for finetuning interface
    model = AddressAwareBertWrapper(bert_model)
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully")
    
    # Sample function IDs
    func_ids = list(map(int, list(func_blocks.keys())[:1000]))
    
    # Run debug steps
    debug_tokenization(func_blocks, func_ids, vocab_stoi)
    debug_model_loading(model, device)
    debug_embeddings(model, func_blocks, func_ids, tokenizer, vocab_stoi, device)
    debug_ground_truth_pairs(model, func_blocks, ground_truth, tokenizer, vocab_stoi, device)
    debug_random_pairs(model, func_blocks, func_ids, tokenizer, vocab_stoi, device)
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
    print("\nKey indicators:")
    print("  1. If ground truth pairs have LOW similarity (<0.3): Model not trained properly")
    print("  2. If random pairs have HIGH similarity (>0.7): Embedding collapse")
    print("  3. If embedding std is very low (<0.01): All embeddings are similar")
    print("  4. If vocab_stoi is missing: Address embedding not working")


if __name__ == '__main__':
    main()
