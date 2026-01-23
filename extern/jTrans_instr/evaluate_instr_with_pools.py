#!/usr/bin/env python3
"""
Evaluate jTrans_instr model using fair pools.
Uses the SAME pool/query files as baseline and addressaware for fair comparison.
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer

# Import model
sys.path.insert(0, str(Path(__file__).parent / 'pretrain'))
from model_instr import InstrBertModel


def generate_instruction_ids(tokenizer, input_ids, func_text, maxlen):
    """
    Generate instruction_ids tensor.
    Uses \t separator to assign instruction index to each token.
    """
    # Split by \t to get instructions
    instructions = func_text.split('\t')
    
    # Count tokens for each instruction
    instruction_token_counts = []
    for instr in instructions:
        if not instr.strip():
            continue
        # Tokenize without special tokens to get count
        instr_enc = tokenizer(instr, add_special_tokens=False)
        instruction_token_counts.append(len(instr_enc['input_ids']))
    
    # Build instruction_ids
    instruction_ids = [0]  # [CLS] -> instruction 0
    current_instr_idx = 0
    
    for count in instruction_token_counts:
        instruction_ids.extend([current_instr_idx] * count)
        current_instr_idx += 1
        
        if len(instruction_ids) >= maxlen - 1:
            break
    
    # Truncate and add [SEP]
    instruction_ids = instruction_ids[:maxlen - 1]
    instruction_ids.append(current_instr_idx)
    
    # Pad to maxlen
    while len(instruction_ids) < maxlen:
        instruction_ids.append(0)
    
    return torch.tensor(instruction_ids[:maxlen], dtype=torch.long)


def load_model(model_path, tokenizer, device):
    """Load finetuned model."""
    print(f"Loading model from {model_path}...")
    
    # Try to load as finetuned model first
    checkpoint_path = Path(model_path) / "pytorch_model.bin"
    if not checkpoint_path.exists():
        checkpoint_path = Path(model_path)
    
    model = InstrBertModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


def load_function_data(func_blocks_path, func_ids):
    """Load function data for specified function IDs."""
    print(f"Loading function data from {func_blocks_path}...")
    
    with open(func_blocks_path, 'r') as f:
        all_func_blocks = json.load(f)
    
    func_data = {}
    for func_id in func_ids:
        func_id_str = str(func_id)
        if func_id_str in all_func_blocks:
            func_data[func_id] = all_func_blocks[func_id_str]
    
    print(f"✓ Loaded {len(func_data)} functions")
    return func_data


def encode_function(model, tokenizer, func_text, device, max_length=512):
    """Encode a function into an embedding."""
    # Use pretrain tokenizer
    encoding = tokenizer(
        func_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Generate instruction_ids based on \t separator
    instruction_ids = generate_instruction_ids(tokenizer, input_ids[0].cpu().tolist(), func_text, max_length)
    instruction_ids = instruction_ids.unsqueeze(0).to(device)
    
    # Get embedding
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            instruction_ids=instruction_ids
        )
        # Use [CLS] token embedding
        embedding = outputs[0][:, 0, :].cpu().numpy()
    
    return embedding[0]


def compute_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def evaluate_retrieval(model, tokenizer, pool_funcs, query_funcs, device):
    """
    Evaluate retrieval performance.
    
    Returns:
        dict with MRR and Recall@K metrics
    """
    print("\nEncoding pool functions...")
    pool_embeddings = {}
    for func_id, func_data in tqdm(pool_funcs.items()):
        func_text = func_data['instructions']
        pool_embeddings[func_id] = encode_function(model, tokenizer, func_text, device)
    
    print("Encoding query functions...")
    query_embeddings = {}
    for func_id, func_data in tqdm(query_funcs.items()):
        func_text = func_data['instructions']
        query_embeddings[func_id] = encode_function(model, tokenizer, func_text, device)
    
    print("\nComputing similarities and ranking...")
    
    reciprocal_ranks = []
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    
    for query_id, query_emb in tqdm(query_embeddings.items()):
        # Compute similarities to all pool functions
        similarities = []
        pool_ids = []
        
        for pool_id, pool_emb in pool_embeddings.items():
            sim = compute_similarity(query_emb, pool_emb)
            similarities.append(sim)
            pool_ids.append(pool_id)
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_pool_ids = [pool_ids[i] for i in sorted_indices]
        
        # Find rank of correct match (same function name, different opt level)
        query_func_name = query_funcs[query_id]['function_name']
        
        rank = None
        for i, pool_id in enumerate(sorted_pool_ids):
            pool_func_name = pool_funcs[pool_id]['function_name']
            if pool_func_name == query_func_name:
                rank = i + 1  # 1-indexed
                break
        
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            
            if rank <= 1:
                recall_at_1 += 1
            if rank <= 5:
                recall_at_5 += 1
            if rank <= 10:
                recall_at_10 += 1
    
    # Compute metrics
    num_queries = len(query_embeddings)
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    recall_1 = recall_at_1 / num_queries
    recall_5 = recall_at_5 / num_queries
    recall_10 = recall_at_10 / num_queries
    
    return {
        'MRR': mrr,
        'Recall@1': recall_1,
        'Recall@5': recall_5,
        'Recall@10': recall_10,
        'num_queries': num_queries
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to finetuned model')
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer')
    parser.add_argument('--func_blocks', required=True, help='Path to func_blocks_instr.json')
    parser.add_argument('--pool_file', required=True, help='Path to pool JSON file')
    parser.add_argument('--query_file', required=True, help='Path to query JSON file')
    parser.add_argument('--output_file', required=True, help='Path to output results file')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    
    # Load model
    model = load_model(args.model_path, tokenizer, device)
    
    # Load pool and query function IDs
    print(f"\nLoading pool from {args.pool_file}...")
    with open(args.pool_file, 'r') as f:
        pool_data = json.load(f)
    pool_func_ids = pool_data['pool']
    
    print(f"Loading queries from {args.query_file}...")
    with open(args.query_file, 'r') as f:
        query_data = json.load(f)
    query_func_ids = query_data['queries']
    
    print(f"Pool size: {len(pool_func_ids)}")
    print(f"Query size: {len(query_func_ids)}")
    
    # Load function data
    pool_funcs = load_function_data(args.func_blocks, pool_func_ids)
    query_funcs = load_function_data(args.func_blocks, query_func_ids)
    
    # Evaluate
    results = evaluate_retrieval(model, tokenizer, pool_funcs, query_funcs, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Pool file: {Path(args.pool_file).name}")
    print(f"Number of queries: {results['num_queries']}")
    print(f"MRR:       {results['MRR']:.4f}")
    print(f"Recall@1:  {results['Recall@1']:.4f}")
    print(f"Recall@5:  {results['Recall@5']:.4f}")
    print(f"Recall@10: {results['Recall@10']:.4f}")
    print("=" * 60)
    
    # Save results
    os.makedirs(Path(args.output_file).parent, exist_ok=True)
    with open(args.output_file, 'w') as f:
        f.write(f"Pool file: {Path(args.pool_file).name}\n")
        f.write(f"Number of queries: {results['num_queries']}\n")
        f.write(f"MRR:       {results['MRR']:.4f}\n")
        f.write(f"Recall@1:  {results['Recall@1']:.4f}\n")
        f.write(f"Recall@5:  {results['Recall@5']:.4f}\n")
        f.write(f"Recall@10: {results['Recall@10']:.4f}\n")
    
    print(f"\n✓ Results saved to {args.output_file}")


if __name__ == '__main__':
    main()
