#!/usr/bin/env python
"""
Evaluate baseline model using pre-generated fair pool and query JSON files.

Usage:
    python evaluate_baseline_with_pools.py \
        --model_path /home/kun/Document/AAE/output/jtrans/baseline_finetune \
        --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/baseline \
        --func_blocks /data/kun/jtransdata/func_blocks_baseline.json \
        --pool_file /data/kun/jtransdata/fair_pools/pool_100_O0_vs_O1.json \
        --query_file /data/kun/jtransdata/fair_pools/query_pool_100_O0_vs_O1.json \
        --output_file results_baseline.txt
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertConfig, BertModel
from tqdm import tqdm
import sys
import os
import torch.nn as nn


# Define BinBertModel directly (from finetune.py)
class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        
        # Baseline model: position_embeddings = word_embeddings (share weights)
        # This replicates the pretraining behavior
        self.embeddings.position_embeddings.weight = self.embeddings.word_embeddings.weight
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, **kwargs):
        # Use input_ids as position_ids to index into the shared embedding matrix
        if position_ids is None:
            position_ids = input_ids.clone()
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            **kwargs
        )


def load_model(model_path):
    """Load baseline model."""
    print(f"Loading baseline model from {model_path}...")
    config_path = Path(model_path) / 'config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Override max_position_embeddings to match vocab_size (baseline behavior)
    config_dict['max_position_embeddings'] = config_dict['vocab_size']
    config = BertConfig(**config_dict)
    model = BinBertModel(config)
    
    weights_path = Path(model_path) / 'pytorch_model.bin'
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def tokenize_function(func_str, tokenizer, max_length=512):
    """Tokenize for baseline model."""
    token_type_ids = torch.zeros(max_length, dtype=torch.long)
    
    encoded = tokenizer.encode_plus(
        func_str,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    vocab_size = len(tokenizer)
    input_ids = encoded['input_ids'].squeeze(0)
    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    
    return {
        'input_ids': input_ids,
        'attention_mask': encoded['attention_mask'].squeeze(0),
        'token_type_ids': token_type_ids
    }


def generate_embedding(model, tokenized, device='cuda'):
    """Generate embedding for baseline model."""
    input_ids = tokenized['input_ids'].unsqueeze(0).to(device)
    attention_mask = tokenized['attention_mask'].unsqueeze(0).to(device)
    token_type_ids = tokenized['token_type_ids'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        if isinstance(output, dict):
            embedding = output['pooler_output'].cpu().numpy()
        else:
            embedding = output.pooler_output.cpu().numpy()
    
    return embedding[0]


def evaluate_retrieval(query_embeddings, pool_embeddings):
    """
    Evaluate retrieval metrics.
    
    Args:
        query_embeddings: List of query embeddings
        pool_embeddings: List of pool embeddings (first len(queries) are ground truths)
    
    Returns:
        dict: Metrics (MRR, Recall@1, Recall@5, Recall@10)
    """
    num_queries = len(query_embeddings)
    
    reciprocal_ranks = []
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    
    for i, query_emb in enumerate(query_embeddings):
        # Ground truth is at position i in pool
        gt_position = i
        
        # Compute similarities
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        pool_norm = pool_embeddings / (np.linalg.norm(pool_embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(pool_norm, query_norm)
        
        # Rank
        sorted_indices = np.argsort(-similarities)
        rank = np.where(sorted_indices == gt_position)[0][0] + 1
        
        reciprocal_ranks.append(1.0 / rank)
        recall_at_1.append(1.0 if rank == 1 else 0.0)
        recall_at_5.append(1.0 if rank <= 5 else 0.0)
        recall_at_10.append(1.0 if rank <= 10 else 0.0)
    
    return {
        'mrr': np.mean(reciprocal_ranks),
        'recall@1': np.mean(recall_at_1),
        'recall@5': np.mean(recall_at_5),
        'recall@10': np.mean(recall_at_10),
        'num_queries': num_queries
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline model with pre-generated fair pools')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to baseline model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to baseline tokenizer')
    parser.add_argument('--func_blocks', type=str, required=True,
                        help='Path to baseline func_blocks.json')
    parser.add_argument('--pool_file', type=str, required=True,
                        help='Path to pool JSON file')
    parser.add_argument('--query_file', type=str, required=True,
                        help='Path to query JSON file')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--output_file', type=str, default='results_baseline.txt')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    
    # Load pool and query definitions
    print(f"\nLoading pool from {args.pool_file}")
    with open(args.pool_file, 'r') as f:
        pool_data = json.load(f)
    
    print(f"Loading queries from {args.query_file}")
    with open(args.query_file, 'r') as f:
        query_data = json.load(f)
    
    print(f"  Pool size: {len(pool_data)}")
    print(f"  Query size: {len(query_data)}")
    
    # Load function blocks
    print(f"\nLoading function blocks from {args.func_blocks}")
    with open(args.func_blocks, 'r') as f:
        func_blocks = json.load(f)
    
    # Load model
    model = load_model(args.model_path).to(device)
    
    # Generate query embeddings
    print("\nGenerating query embeddings...")
    query_embeddings = []
    
    for query in tqdm(query_data, desc="Queries"):
        func_id = str(query['baseline_func_id'])
        func_str = func_blocks[func_id].get('tokens', func_blocks[func_id].get('instructions', ''))
        tokenized = tokenize_function(func_str, tokenizer, args.max_length)
        embedding = generate_embedding(model, tokenized, device)
        query_embeddings.append(embedding)
    
    # Generate pool embeddings
    print("\nGenerating pool embeddings...")
    pool_embeddings = []
    
    for pool_entry in tqdm(pool_data, desc="Pool"):
        func_id = str(pool_entry['baseline_func_id'])
        func_str = func_blocks[func_id].get('tokens', func_blocks[func_id].get('instructions', ''))
        tokenized = tokenize_function(func_str, tokenizer, args.max_length)
        embedding = generate_embedding(model, tokenized, device)
        pool_embeddings.append(embedding)
    
    query_embeddings = np.array(query_embeddings)
    pool_embeddings = np.array(pool_embeddings)
    
    # Evaluate
    print("\nEvaluating baseline model...")
    metrics = evaluate_retrieval(query_embeddings, pool_embeddings)
    
    # Print results
    print(f"\n{'='*80}")
    print("BASELINE MODEL RESULTS")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Pool: {args.pool_file}")
    print(f"Queries: {len(query_data)}")
    print(f"Pool size: {len(pool_data)}")
    print(f"\n{'Metric':<15} {'Value':<15}")
    print("-" * 30)
    print(f"{'MRR':<15} {metrics['mrr']:<15.4f}")
    print(f"{'Recall@1':<15} {metrics['recall@1']:<15.4f}")
    print(f"{'Recall@5':<15} {metrics['recall@5']:<15.4f}")
    print(f"{'Recall@10':<15} {metrics['recall@10']:<15.4f}")
    print(f"{'='*80}\n")
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"Baseline Model Evaluation Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Pool file: {args.pool_file}\n")
        f.write(f"Query file: {args.query_file}\n")
        f.write(f"Queries: {len(query_data)}\n")
        f.write(f"Pool size: {len(pool_data)}\n")
        f.write(f"\n{'='*80}\n\n")
        f.write(f"{'Metric':<15} {'Value':<15}\n")
        f.write(f"{'-'*30}\n")
        f.write(f"{'MRR':<15} {metrics['mrr']:<15.4f}\n")
        f.write(f"{'Recall@1':<15} {metrics['recall@1']:<15.4f}\n")
        f.write(f"{'Recall@5':<15} {metrics['recall@5']:<15.4f}\n")
        f.write(f"{'Recall@10':<15} {metrics['recall@10']:<15.4f}\n")
    
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
