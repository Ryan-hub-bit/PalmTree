#!/usr/bin/env python3
"""
Evaluate finetuned addressaware model on addressaware pools.

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
from transformers import BertTokenizer, BertModel, BertConfig


def load_vocab(vocab_path):
    """Load vocabulary using BertTokenizer"""
    print(f"Loading tokenizer from: {vocab_path}")
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    print(f"Vocabulary size: {len(tokenizer)}")
    return tokenizer


class AddressAwareBertWrapper(torch.nn.Module):
    """
    Wrapper for address-aware BERT encoder to match BERT interface for evaluation.
    Extracts [CLS] token as pooled output.
    """
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model  # The BERT model with AddressAwareBERTEmbedding
        
    def forward(self, token_ids, attention_mask, token_type_ids,
                binary_pos, function_pos, bb_pos, var_offsets=None):
        """
        Forward pass returning pooled output (CLS token).
        
        Returns:
            Dict with:
                - pooler_output: [batch_size, hidden_size]
                - last_hidden_state: [batch_size, seq_len, hidden_size]
        """
        # Get embeddings
        embeddings = self.bert.embeddings(
            token_ids,
            token_type_ids,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets
        )
        
        # Pass through transformer encoder
        outputs = self.bert.encoder(
            embeddings,
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden]
        pooler_output = sequence_output[:, 0, :]  # CLS token
        
        # Return as dict
        return {
            'pooler_output': pooler_output,
            'last_hidden_state': sequence_output
        }


def load_model(checkpoint_path, vocab_path, device='cuda'):
    """Load finetuned addressaware model"""
    print(f"Loading model from: {checkpoint_path}")
    
    from pretrain.address_aware.address_embedding import AddressAwareBERTEmbedding
    
    # Load vocab for address vs daddr distinction
    vocab_file = os.path.join(vocab_path, 'vocab.txt')
    vocab_stoi = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            token = line.strip()
            vocab_stoi[token] = idx
    
    # Load config
    config_path = os.path.join(checkpoint_path, 'config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create BERT model with config
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
        vocab_stoi=vocab_stoi  # Pass vocab mapping for address vs daddr distinction
    )
    
    # Load weights
    weights_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
    state_dict = torch.load(weights_path, map_location=device)
    bert_model.load_state_dict(state_dict)
    
    # Wrap for evaluation
    model = AddressAwareBertWrapper(bert_model).to(device)
    model.eval()
    
    print(f"Model loaded successfully (AddressAware)")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    
    return model, config


def tokenize_function(func_block, tokenizer, max_len=512):
    """Tokenize function block using addressaware format"""
    # Get token string (support both 'tokens' and 'instructions')
    func_str = func_block.get('tokens') or func_block.get('instructions', '')
    
    # Use encode_plus
    encoded = tokenizer.encode_plus(
        func_str,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Extract token IDs
    token_ids = encoded['input_ids'].squeeze(0).tolist()
    
    # Get address info if available
    binary_pos = func_block.get('binary_pos', [0] * max_len)
    function_pos = func_block.get('function_pos', [0] * max_len)
    bb_pos = func_block.get('bb_pos', [0] * max_len)
    var_offsets = func_block.get('var_offsets', [0] * max_len)
    
    # Pad or truncate to max_len
    if len(binary_pos) < max_len:
        binary_pos = binary_pos + [0] * (max_len - len(binary_pos))
    else:
        binary_pos = binary_pos[:max_len]
    
    if len(function_pos) < max_len:
        function_pos = function_pos + [0] * (max_len - len(function_pos))
    else:
        function_pos = function_pos[:max_len]
        
    if len(bb_pos) < max_len:
        bb_pos = bb_pos + [0] * (max_len - len(bb_pos))
    else:
        bb_pos = bb_pos[:max_len]
        
    if len(var_offsets) < max_len:
        var_offsets = var_offsets + [0] * (max_len - len(var_offsets))
    else:
        var_offsets = var_offsets[:max_len]
    
    return {
        'token_ids': token_ids,
        'binary_pos': binary_pos,
        'function_pos': function_pos,
        'bb_pos': bb_pos,
        'var_offsets': var_offsets
    }


def generate_embeddings(model, func_ids, func_blocks, tokenizer, device, max_len=512, batch_size=32, model_vocab_size=None):
    """Generate embeddings for a list of function IDs"""
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(func_ids), batch_size), desc="Generating embeddings"):
            batch_ids = func_ids[i:i+batch_size]
            
            # Tokenize batch
            batch_data = {
                'token_ids': [],
                'attention_mask': [],
                'token_type_ids': [],
                'binary_pos': [],
                'function_pos': [],
                'bb_pos': [],
                'var_offsets': []
            }
            
            for fid in batch_ids:
                tokenized = tokenize_function(func_blocks[fid], tokenizer, max_len)
                token_ids = tokenized['token_ids']
                
                # CRITICAL: Clamp token IDs to model's vocab size
                if model_vocab_size is not None:
                    token_ids = [min(tid, model_vocab_size - 1) for tid in token_ids]
                
                batch_data['token_ids'].append(token_ids)
                
                # Create attention mask
                attention_mask = [1 if tid != tokenizer.pad_token_id else 0 for tid in token_ids]
                batch_data['attention_mask'].append(attention_mask)
                
                # Create token_type_ids (all 0s)
                token_type_ids = [0] * max_len
                batch_data['token_type_ids'].append(token_type_ids)
                
                # Add address info
                batch_data['binary_pos'].append(tokenized['binary_pos'])
                batch_data['function_pos'].append(tokenized['function_pos'])
                batch_data['bb_pos'].append(tokenized['bb_pos'])
                batch_data['var_offsets'].append(tokenized['var_offsets'])
            
            # Convert to tensors
            token_ids = torch.tensor(batch_data['token_ids'], dtype=torch.long).to(device)
            attention_mask = torch.tensor(batch_data['attention_mask'], dtype=torch.long).to(device)
            token_type_ids = torch.tensor(batch_data['token_type_ids'], dtype=torch.long).to(device)
            binary_pos = torch.tensor(batch_data['binary_pos'], dtype=torch.long).to(device)
            function_pos = torch.tensor(batch_data['function_pos'], dtype=torch.long).to(device)
            bb_pos = torch.tensor(batch_data['bb_pos'], dtype=torch.long).to(device)
            var_offsets = torch.tensor(batch_data['var_offsets'], dtype=torch.long).to(device)
            
            # Get embeddings from AddressAware model
            outputs = model(
                token_ids=token_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                binary_pos=binary_pos,
                function_pos=function_pos,
                bb_pos=bb_pos,
                var_offsets=var_offsets
            )
            
            # Use pooler_output (CLS token)
            batch_embeddings = outputs['pooler_output'].cpu().numpy()
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


def evaluate_pool(model, pool_file, query_file, func_blocks, tokenizer, device, max_len=512, batch_size=32, model_vocab_size=None):
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
    pool_embeddings = generate_embeddings(model, pool_ids, func_blocks, tokenizer, device, max_len, batch_size, model_vocab_size)
    
    print("  Generating query embeddings...")
    query_embeddings = generate_embeddings(model, query_ids, func_blocks, tokenizer, device, max_len, batch_size, model_vocab_size)
    
    # Compute similarity
    print("  Computing similarities...")
    similarity_matrix = compute_similarity(query_embeddings, pool_embeddings)
    
    # Calculate metrics
    print("  Calculating metrics...")
    metrics = calculate_metrics(similarity_matrix, ground_truth)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate addressaware pools')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to finetuned model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--pool-dir', type=str, 
                        default='/data/kun/jtrans/addressaware/eval/pools_filtered',
                        help='Directory containing pool and query files')
    parser.add_argument('--func-blocks', type=str,
                        default='/data/kun/jtrans/addressaware/eval/func_blocks_addr.json',
                        help='Path to function blocks file')
    parser.add_argument('--max-len', type=int, default=512,
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
    
    # Load tokenizer
    tokenizer = load_vocab(args.vocab)
    
    # Load model
    model, config = load_model(args.model, args.vocab, args.device)
    
    # Get model's actual vocab size
    model_vocab_size = config.vocab_size
    tokenizer_vocab_size = len(tokenizer)
    
    print(f"\nVocab size check:")
    print(f"  Tokenizer vocab size: {tokenizer_vocab_size}")
    print(f"  Model vocab size: {model_vocab_size}")
    if tokenizer_vocab_size != model_vocab_size:
        print(f"  WARNING: Vocab size mismatch! Will clamp token IDs to [0, {model_vocab_size-1}]")
    
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
        
        # Calculate query size (50% of pool)
        query_size = int(int(pool_size) * 0.5)
        query_file = pool_dir / f"query_{query_size}_from_pool_{pool_size}_{opt_pair}.json"
        
        if not query_file.exists():
            print(f"Warning: Query file not found for {pool_file.name}")
            continue
        
        # Evaluate
        metrics = evaluate_pool(model, pool_file, query_file, func_blocks, tokenizer, 
                               args.device, args.max_len, args.batch_size, model_vocab_size)
        
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
