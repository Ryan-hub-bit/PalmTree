#!/usr/bin/env python3
"""
Evaluate finetuned model on baseline pools.

Metrics:
- MRR (Mean Reciprocal Rank)
- Recall@1
- Recall@5
- Recall@10
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import os
import pickle

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'strupos'))

from vocab import TorchVocab


def load_vocab(vocab_path):
    """Load vocabulary"""
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def load_model(checkpoint_path, vocab, device='cuda'):
    """Load finetuned model"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Import here to avoid circular dependency
    from model import IMCModel
    
    # Create a simple config object
    class ModelConfig:
        def __init__(self):
            self.vocab_size = len(vocab)
            self.hidden = 768
            self.n_layers = 12
            self.attn_heads = 12
            self.dropout = 0.1
    
    config = ModelConfig()
    
    # Load model
    model = IMCModel(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded successfully")
    return model, config


def tokenize_function(func_block, vocab, max_len=100):
    """Tokenize function block using vocabulary"""
    # Parse tokens from func_block['tokens']
    tokens_str = func_block['tokens']
    tokens = tokens_str.split('\t')  # Instructions separated by tab
    
    # Flatten all tokens from all instructions
    all_tokens = []
    for instr in tokens:
        instr_tokens = instr.split()
        all_tokens.extend(instr_tokens)
    
    # Convert to vocab indices
    token_ids = []
    for token in all_tokens[:max_len]:
        token_ids.append(vocab.stoi.get(token, vocab.stoi.get('<unk>', 1)))
    
    # Pad to max_len
    pad_idx = vocab.stoi.get('<pad>', 0)
    while len(token_ids) < max_len:
        token_ids.append(pad_idx)
    
    return token_ids[:max_len]


def generate_embeddings(model, func_ids, func_blocks, vocab, device, max_len=100, batch_size=32):
    """Generate embeddings for a list of function IDs"""
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(func_ids), batch_size), desc="Generating embeddings"):
            batch_ids = func_ids[i:i+batch_size]
            
            # Tokenize batch
            batch_tokens = []
            for fid in batch_ids:
                token_ids = tokenize_function(func_blocks[fid], vocab, max_len)
                batch_tokens.append(token_ids)
            
            # Convert to tensor
            input_tensor = torch.tensor(batch_tokens, dtype=torch.long).to(device)
            
            # Get embeddings (model returns CLS token embedding)
            outputs = model(input_tensor)
            batch_embeddings = outputs.cpu().numpy()
            embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings)
    return embeddings


def compute_similarity(query_embeddings, pool_embeddings):
    """Compute cosine similarity between queries and pool"""
    # Normalize
    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    pool_norm = pool_embeddings / (np.linalg.norm(pool_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix (queries x pool)
    similarity = np.matmul(query_norm, pool_norm.T)
    return similarity


def calculate_metrics(similarity_matrix, ground_truth, k_values=[1, 5, 10]):
    """Calculate retrieval metrics"""
    num_queries = len(ground_truth)
    
    # Get top-k predictions for each query
    top_k_indices = np.argsort(-similarity_matrix, axis=1)  # Sort descending
    
    # Calculate metrics
    reciprocal_ranks = []
    recall_at_k = {k: 0 for k in k_values}
    
    for i, gt_idx in enumerate(ground_truth):
        # Find rank of ground truth
        predicted_indices = top_k_indices[i]
        
        # Find where ground truth appears in predictions
        try:
            rank = np.where(predicted_indices == gt_idx)[0][0] + 1  # 1-based rank
            reciprocal_ranks.append(1.0 / rank)
        except:
            reciprocal_ranks.append(0.0)
            rank = float('inf')
        
        # Check recall@k
        for k in k_values:
            if gt_idx in predicted_indices[:k]:
                recall_at_k[k] += 1
    
    # Compute final metrics
    mrr = np.mean(reciprocal_ranks)
    recall_metrics = {f"Recall@{k}": (count / num_queries) * 100 for k, count in recall_at_k.items()}
    
    return {
        'MRR': mrr,
        **recall_metrics
    }


def evaluate_pool(model, pool_file, query_file, func_blocks, vocab, device, max_len=100, batch_size=32):
    """Evaluate on a single pool/query pair"""
    # Load pool and query
    with open(pool_file, 'r') as f:
        pool_data = json.load(f)
    with open(query_file, 'r') as f:
        query_data = json.load(f)
    
    pool_ids = pool_data['pool']
    query_ids = query_data['queries']
    ground_truth = query_data['ground_truth']
    
    print(f"\nEvaluating: {query_file.name}")
    print(f"  Pool size: {len(pool_ids)}, Query size: {len(query_ids)}")
    
    # Generate embeddings
    print("  Generating pool embeddings...")
    pool_embeddings = generate_embeddings(model, pool_ids, func_blocks, vocab, device, max_len, batch_size)
    
    print("  Generating query embeddings...")
    query_embeddings = generate_embeddings(model, query_ids, func_blocks, vocab, device, max_len, batch_size)
    
    # Compute similarity
    print("  Computing similarities...")
    similarity_matrix = compute_similarity(query_embeddings, pool_embeddings)
    
    # Calculate metrics
    print("  Calculating metrics...")
    metrics = calculate_metrics(similarity_matrix, ground_truth)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline pools')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to finetuned model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--pool-dir', type=str, 
                        default='/data/kun/jtrans/baseline/eval/pools',
                        help='Directory containing pool and query files')
    parser.add_argument('--func-blocks', type=str,
                        default='/data/kun/jtrans/baseline/eval/func_blocks_baseline.json',
                        help='Path to function blocks file')
    parser.add_argument('--max-len', type=int, default=100,
                        help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding generation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--pool-size', type=int, choices=[100, 1000, 10000],
                        help='Evaluate specific pool size only')
    parser.add_argument('--opt-pair', type=str, choices=['O0_vs_O3', 'O1_vs_O3', 'O2_vs_O3'],
                        help='Evaluate specific optimization pair only')
    
    args = parser.parse_args()
    
    # Load vocabulary
    vocab = load_vocab(args.vocab)
    
    # Load model
    model, config = load_model(args.model, vocab, args.device)
    
    # Load function blocks
    print(f"\nLoading function blocks from: {args.func_blocks}")
    with open(args.func_blocks, 'r') as f:
        func_blocks = json.load(f)
    print(f"Loaded {len(func_blocks)} function blocks")
    
    # Find pool/query files
    pool_dir = Path(args.pool_dir)
    pool_files = sorted(pool_dir.glob('pool_*.json'))
    
    # Filter by pool size and opt pair if specified
    if args.pool_size:
        pool_files = [f for f in pool_files if f'pool_{args.pool_size}_' in f.name]
    if args.opt_pair:
        pool_files = [f for f in pool_files if args.opt_pair in f.name]
    
    print(f"\nFound {len(pool_files)} pool files to evaluate")
    
    # Evaluate each pool
    all_results = []
    for pool_file in pool_files:
        # Find corresponding query file
        pool_size = pool_file.stem.split('_')[1]
        opt_pair = '_'.join(pool_file.stem.split('_')[2:])
        
        # Calculate query size (20% of pool)
        query_size = int(int(pool_size) * 0.2)
        query_file = pool_dir / f"query_{query_size}_from_pool_{pool_size}_{opt_pair}.json"
        
        if not query_file.exists():
            print(f"Warning: Query file not found for {pool_file.name}")
            continue
        
        # Evaluate
        metrics = evaluate_pool(model, pool_file, query_file, func_blocks, vocab, 
                               args.device, args.max_len, args.batch_size)
        
        # Store results
        result = {
            'pool_file': pool_file.name,
            'pool_size': int(pool_size),
            'query_size': query_size,
            'opt_pair': opt_pair,
            **metrics
        }
        all_results.append(result)
        
        # Print results
        print(f"\n  Results:")
        print(f"    MRR:       {metrics['MRR']:.4f}")
        print(f"    Recall@1:  {metrics['Recall@1']:.2f}%")
        print(f"    Recall@5:  {metrics['Recall@5']:.2f}%")
        print(f"    Recall@10: {metrics['Recall@10']:.2f}%")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL EVALUATIONS")
    print("="*80)
    print(f"{'Pool Size':<12} {'Opt Pair':<12} {'MRR':<10} {'R@1':<10} {'R@5':<10} {'R@10':<10}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['pool_size']:<12} {result['opt_pair']:<12} "
              f"{result['MRR']:<10.4f} "
              f"{result['Recall@1']:<10.2f} "
              f"{result['Recall@5']:<10.2f} "
              f"{result['Recall@10']:<10.2f}")
    
    # Save results to JSON
    output_file = pool_dir / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
