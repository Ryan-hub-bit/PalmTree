#!/usr/bin/env python
"""Debug evaluation to check similarity scores."""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertConfig, BertModel
import torch.nn as nn


class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings.weight = self.embeddings.word_embeddings.weight
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, **kwargs):
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
    config_path = Path(model_path) / 'config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config_dict['max_position_embeddings'] = config_dict['vocab_size']
    config = BertConfig(**config_dict)
    model = BinBertModel(config)
    
    weights_path = Path(model_path) / 'pytorch_model.bin'
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def tokenize_function(func_str, tokenizer, max_length=512):
    token_type_ids = torch.zeros(max_length, dtype=torch.long)
    encoded = tokenizer.encode_plus(
        func_str, max_length=max_length, padding='max_length',
        truncation=True, return_tensors='pt'
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


# Load data
model_path = '/home/kun/Document/AAE/output/jtrans/baseline_finetune/finetune_epoch_3'
tokenizer_path = '/home/kun/Document/AAE/extern/jTrans/pretrain/baseline'
func_blocks_path = '/data/kun/jtransdata/func_blocks_baseline.json'
pool_file = '/data/kun/jtransdata/fair_pools/pool_100_O0_vs_O1.json'
query_file = '/data/kun/jtransdata/fair_pools/query_pool_100_O0_vs_O1.json'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = load_model(model_path).to(device)

with open(pool_file) as f:
    pool_data = json.load(f)
with open(query_file) as f:
    query_data = json.load(f)
with open(func_blocks_path) as f:
    func_blocks = json.load(f)

# Test with first query
print("\nTesting Query 0:")
q = query_data[0]
print(f"  Binary: {q['binary']}")
print(f"  Function: {q['function_name']}")
print(f"  Opt: {q['opt']}")
print(f"  Func ID: {q['baseline_func_id']}")

# Generate query embedding
func_str = func_blocks[str(q['baseline_func_id'])].get('tokens', '')
tokenized = tokenize_function(func_str, tokenizer)
query_emb = generate_embedding(model, tokenized, device)

# Generate pool embeddings for first 10 entries
print("\nGenerating embeddings for first 10 pool entries...")
pool_embs = []
for i, p in enumerate(pool_data[:10]):
    func_str = func_blocks[str(p['baseline_func_id'])].get('tokens', '')
    tokenized = tokenize_function(func_str, tokenizer)
    emb = generate_embedding(model, tokenized, device)
    pool_embs.append(emb)

pool_embs = np.array(pool_embs)

# Compute similarities
query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
pool_norm = pool_embs / (np.linalg.norm(pool_embs, axis=1, keepdims=True) + 1e-8)
similarities = np.dot(pool_norm, query_norm)

print("\nSimilarity scores (Query 0 vs first 10 pool entries):")
for i, sim in enumerate(similarities):
    p = pool_data[i]
    is_gt = (p['binary'] == q['binary'] and p['function_name'] == q['function_name'])
    print(f"  Pool {i} ({'GT' if is_gt else 'NEG'}): {sim:.6f} - {p['binary'][:30]}... {p['function_name'][:30]}... {p['opt']}")

sorted_indices = np.argsort(-similarities)
print(f"\nRanking: {sorted_indices}")
print(f"Ground truth position: {sorted_indices.tolist().index(0) if 0 in sorted_indices else 'NOT IN TOP 10'}")
