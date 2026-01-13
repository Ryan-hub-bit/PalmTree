#!/usr/bin/env python
"""
Evaluate address-aware model using pre-generated fair pool and query JSON files.

Usage:
    python evaluate_addressaware_with_pools.py \
        --model_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune \
        --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
        --func_blocks /data/kun/jtransdata/func_blocks_addr.json \
        --pool_file /data/kun/jtransdata/fair_pools/pool_100_O0_vs_O1.json \
        --query_file /data/kun/jtransdata/fair_pools/query_pool_100_O0_vs_O1.json \
        --output_file results_addressaware.txt
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
import pickle
import re
import torch.nn as nn


# Define AddressAwareBertWrapper directly (from finetune.py)
class AddressAwareBertWrapper(nn.Module):
    """
    Wrapper for address-aware BERT encoder to match BERT interface for finetuning.
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
        
        # Return as dict (compatible with DataParallel)
        return {
            'pooler_output': pooler_output,
            'last_hidden_state': sequence_output
        }


def load_model(model_path, tokenizer_path):
    """Load address-aware model."""
    print(f"Loading address-aware model from {model_path}...")
    
    # Load vocab for address vs daddr distinction
    vocab_path = Path(tokenizer_path) / 'vocab_addr.pkl'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_stoi = vocab.stoi
    
    # Load config
    config_path = Path(model_path) / 'config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create BERT model with config
    config = BertConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_hidden_layers=config_dict['num_hidden_layers'],
        num_attention_heads=config_dict['num_attention_heads'],
        intermediate_size=config_dict.get('intermediate_size', config_dict['hidden_size'] * 4),
        max_position_embeddings=config_dict['max_position_embeddings'],
        type_vocab_size=config_dict.get('type_vocab_size', 2),
    )
    
    # Add pretrain/address_aware to path for importing address embedding
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pretrain', 'address_aware'))
    from address_embedding import AddressAwareBERTEmbedding
    
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
    
    # Load weights
    weights_path = Path(model_path) / 'pytorch_model.bin'
    state_dict = torch.load(weights_path, map_location='cpu')
    bert_model.load_state_dict(state_dict)
    
    # Wrap for inference
    model = AddressAwareBertWrapper(bert_model)
    model.eval()
    return model


def tokenize_function(func_str, tokenizer, max_length=512):
    """Tokenize for address-aware model."""
    addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    # Parse instructions and extract position info
    instructions = func_str.strip().split('\n')
    tokens = []
    binary_positions = []
    function_positions = []
    bb_positions = []
    var_offsets = []
    
    for inst in instructions:
        parts = inst.strip().split()
        for part in parts:
            match = addr_pattern.match(part)
            if match:
                tokens.append(match.group(1))
                binary_positions.append(float(match.group(3)))
                function_positions.append(float(match.group(4)))
                bb_positions.append(float(match.group(5)))
                var_offsets.append(-1)
                continue
            
            nested_match = nested_addr_pattern.match(part)
            if nested_match:
                tokens.append('address')
                binary_positions.append(float(nested_match.group(2)))
                function_positions.append(float(nested_match.group(3)))
                bb_positions.append(float(nested_match.group(4)))
                var_offsets.append(-1)
                continue
            
            daddr_match = daddr_pattern.match(part)
            if daddr_match:
                tokens.append('daddr')
                binary_positions.append(float(daddr_match.group(2)))
                function_positions.append(float(daddr_match.group(3)))
                bb_positions.append(float(daddr_match.group(4)))
                var_offsets.append(-1)
                continue
            
            var_match = var_pattern.match(part)
            if var_match:
                tokens.append('var')
                binary_positions.append(0.0)
                function_positions.append(0.0)
                bb_positions.append(0.0)
                offset = int(var_match.group(1), 16)
                var_offsets.append(offset)
                continue
            
            tokens.append(part)
            binary_positions.append(0.0)
            function_positions.append(0.0)
            bb_positions.append(0.0)
            var_offsets.append(-1)
    
    # Tokenize
    input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_length])
    seq_len = len(input_ids)
    
    # Pad
    input_ids += [tokenizer.pad_token_id] * (max_length - seq_len)
    binary_positions += [0.0] * (max_length - seq_len)
    function_positions += [0.0] * (max_length - seq_len)
    bb_positions += [0.0] * (max_length - seq_len)
    var_offsets += [-1] * (max_length - seq_len)
    attention_mask = [1] * seq_len + [0] * (max_length - seq_len)
    token_type_ids = [0] * max_length
    
    return {
        'input_ids': torch.tensor(input_ids[:max_length], dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask[:max_length], dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids[:max_length], dtype=torch.long),
        'binary_pos': torch.tensor(binary_positions[:max_length], dtype=torch.float),
        'function_pos': torch.tensor(function_positions[:max_length], dtype=torch.float),
        'bb_pos': torch.tensor(bb_positions[:max_length], dtype=torch.float),
        'var_offsets': torch.tensor(var_offsets[:max_length], dtype=torch.long)
    }


def generate_embedding(model, tokenized, device='cuda'):
    """Generate embedding for address-aware model."""
    input_ids = tokenized['input_ids'].unsqueeze(0).to(device)
    attention_mask = tokenized['attention_mask'].unsqueeze(0).to(device)
    token_type_ids = tokenized['token_type_ids'].unsqueeze(0).to(device)
    binary_pos = tokenized['binary_pos'].unsqueeze(0).to(device)
    function_pos = tokenized['function_pos'].unsqueeze(0).to(device)
    bb_pos = tokenized['bb_pos'].unsqueeze(0).to(device)
    var_offsets = tokenized['var_offsets'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(
            token_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            binary_pos=binary_pos,
            function_pos=function_pos,
            bb_pos=bb_pos,
            var_offsets=var_offsets
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
    parser = argparse.ArgumentParser(description='Evaluate address-aware model with pre-generated fair pools')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to address-aware model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to address-aware tokenizer')
    parser.add_argument('--func_blocks', type=str, required=True,
                        help='Path to address-aware func_blocks.json')
    parser.add_argument('--pool_file', type=str, required=True,
                        help='Path to pool JSON file')
    parser.add_argument('--query_file', type=str, required=True,
                        help='Path to query JSON file')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--output_file', type=str, default='results_addressaware.txt')
    
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
    model = load_model(args.model_path, args.tokenizer).to(device)
    
    # Generate query embeddings
    print("\nGenerating query embeddings...")
    query_embeddings = []
    
    for query in tqdm(query_data, desc="Queries"):
        func_id = str(query['addressaware_func_id'])
        func_str = func_blocks[func_id].get('instructions', func_blocks[func_id].get('tokens', ''))
        tokenized = tokenize_function(func_str, tokenizer, args.max_length)
        embedding = generate_embedding(model, tokenized, device)
        query_embeddings.append(embedding)
    
    # Generate pool embeddings
    print("\nGenerating pool embeddings...")
    pool_embeddings = []
    
    for pool_entry in tqdm(pool_data, desc="Pool"):
        func_id = str(pool_entry['addressaware_func_id'])
        func_str = func_blocks[func_id].get('instructions', func_blocks[func_id].get('tokens', ''))
        tokenized = tokenize_function(func_str, tokenizer, args.max_length)
        embedding = generate_embedding(model, tokenized, device)
        pool_embeddings.append(embedding)
    
    query_embeddings = np.array(query_embeddings)
    pool_embeddings = np.array(pool_embeddings)
    
    # Evaluate
    print("\nEvaluating address-aware model...")
    metrics = evaluate_retrieval(query_embeddings, pool_embeddings)
    
    # Print results
    print(f"\n{'='*80}")
    print("ADDRESS-AWARE MODEL RESULTS")
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
        f.write(f"Address-Aware Model Evaluation Results\n")
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
