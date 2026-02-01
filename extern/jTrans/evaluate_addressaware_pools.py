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
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
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
    """
    Tokenize function block using addressaware format.
    
    CRITICAL: MATCH PRETRAIN!
    - Always use 'instructions' field (raw format with address annotations)
    - Parse address/daddr patterns inline
    - Both address() and daddr() become 'address' token
    """
    # MATCH PRETRAIN: Always use 'instructions' field
    func_str = func_block.get('instructions', '')
    
    if not func_str:
        print(f"Warning: Function {func_block.get('id')} missing 'instructions' field")
        return {
            'token_ids': [tokenizer.pad_token_id] * max_len,
            'attention_mask': [0] * max_len,
            'token_type_ids': [0] * max_len,
            'binary_pos': [-1.0] * max_len,
            'function_pos': [-1.0] * max_len,
            'bb_pos': [-1.0] * max_len,
            'var_offsets': [-1] * max_len
        }
    
    # Parse address-aware function (SAME as pretrain)
    result = _parse_address_aware_function(func_str, tokenizer, max_len)
    
    return {
        'token_ids': result['input_ids'].tolist() if hasattr(result['input_ids'], 'tolist') else result['input_ids'],
        'attention_mask': result['attention_mask'].tolist() if hasattr(result['attention_mask'], 'tolist') else result['attention_mask'],
        'token_type_ids': result['token_type_ids'].tolist() if hasattr(result['token_type_ids'], 'tolist') else result['token_type_ids'],
        'binary_pos': result['binary_pos'].tolist() if hasattr(result['binary_pos'], 'tolist') else result['binary_pos'],
        'function_pos': result['function_pos'].tolist() if hasattr(result['function_pos'], 'tolist') else result['function_pos'],
        'bb_pos': result['bb_pos'].tolist() if hasattr(result['bb_pos'], 'tolist') else result['bb_pos'],
        'var_offsets': result['var_offsets'].tolist() if hasattr(result['var_offsets'], 'tolist') else result['var_offsets']
    }


def _parse_instruction(inst_text):
    """
    Parse a single address-aware instruction.
    Format: opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 operand2 ...
    
    Returns:
        tokens: List of token strings
        positions: List of (binary_pos, function_pos, bb_pos) tuples
        var_offsets: List of var offset values (-1 for non-var tokens)
    """
    import re
    
    addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    tokens = []
    positions = []
    var_offsets = []
    
    parts = inst_text.split()
    
    for part in parts:
        if not part or part.isspace():
            continue
        
        # Check for opcode with address: opcode(0xADDR:bnorm:fnorm:bbnorm)
        opcode_match = addr_pattern.match(part)
        if opcode_match:
            opcode = opcode_match.group(1)
            binary_norm = float(opcode_match.group(3))
            function_norm = float(opcode_match.group(4))
            bb_norm = float(opcode_match.group(5))
            
            tokens.append(opcode)
            positions.append((binary_norm, function_norm, bb_norm))
            var_offsets.append(-1)
            continue
        
        # Check for address() pattern
        addr_match = nested_addr_pattern.match(part)
        if addr_match:
            binary_norm = float(addr_match.group(2))
            function_norm = float(addr_match.group(3))
            bb_norm = float(addr_match.group(4))
            
            tokens.append('address')
            positions.append((binary_norm, function_norm, bb_norm))
            var_offsets.append(-1)
            continue
        
        # Check for daddr() pattern - data address (uses separate MLP in embedding layer)
        daddr_match = daddr_pattern.match(part)
        if daddr_match:
            binary_norm = float(daddr_match.group(2))
            function_norm = float(daddr_match.group(3))
            bb_norm = float(daddr_match.group(4))
            
            tokens.append('daddr')  # Keep 'daddr' - embedding layer distinguishes from 'address'
            positions.append((binary_norm, function_norm, bb_norm))
            var_offsets.append(-1)
            continue
        
        # Check for var() pattern
        var_match = var_pattern.match(part)
        if var_match:
            hex_offset = var_match.group(1)
            offset_val = int(hex_offset, 16)
            
            # Handle 64-bit negative offsets (two's complement)
            if offset_val > 0x7FFFFFFFFFFFFFFF:
                offset_val = offset_val - 0x10000000000000000
            
            tokens.append('var')
            positions.append((-1.0, -1.0, -1.0))
            var_offsets.append(offset_val)
            continue
        
        # Regular token (no address info)
        tokens.append(part)
        positions.append((-1.0, -1.0, -1.0))
        var_offsets.append(-1)
    
    return tokens, positions, var_offsets


def _parse_address_aware_function(func_str, tokenizer, max_length):
    """
    Parse an entire address-aware function string (SAME as finetune).
    Format: inst1\tinst2\tinst3...
    
    Returns dict with:
        - input_ids: Token IDs (LongTensor)
        - attention_mask: Attention mask
        - token_type_ids: Segment labels
        - binary_pos: Binary position embeddings (FloatTensor)
        - function_pos: Function position embeddings (FloatTensor)
        - bb_pos: Basic block position embeddings (FloatTensor)
        - var_offsets: Variable offset values (LongTensor)
    """
    instructions = func_str.split('\t')
    
    all_tokens = []
    all_positions = []
    all_var_offsets = []
    all_segments = []
    
    # Add <sos> at beginning (segment 1) - MATCH FINETUNE
    all_tokens.append('<sos>')
    all_positions.append((-1.0, -1.0, -1.0))
    all_var_offsets.append(-1)
    all_segments.append(1)
    
    for inst_idx, inst_text in enumerate(instructions):
        inst_text = inst_text.strip()
        if not inst_text:
            continue
        
        tokens, positions, var_offsets = _parse_instruction(inst_text)
        
        # Segment label = instruction number (1-indexed) - MATCH FINETUNE
        inst_segment = inst_idx + 1
        all_tokens.extend(tokens)
        all_positions.extend(positions)
        all_var_offsets.extend(var_offsets)
        all_segments.extend([inst_segment] * len(tokens))
    
    # Add <eos> at the end (gets last instruction's segment) - MATCH FINETUNE
    last_segment = inst_idx + 1 if instructions else 1
    all_tokens.append('<eos>')
    all_positions.append((-1.0, -1.0, -1.0))
    all_var_offsets.append(-1)
    all_segments.append(last_segment)
    
    # Convert tokens to IDs using tokenizer's vocabulary
    token_ids = []
    unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 1
    for tok in all_tokens:
        token_id = tokenizer.convert_tokens_to_ids(tok)
        if token_id is None:
            token_id = unk_token_id
        token_ids.append(token_id)
    
    # Truncate or pad to max_length
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
        all_positions = all_positions[:max_length]
        all_var_offsets = all_var_offsets[:max_length]
        all_segments = all_segments[:max_length]
    else:
        padding_len = max_length - len(token_ids)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        token_ids += [pad_token_id] * padding_len
        all_positions += [(-1.0, -1.0, -1.0)] * padding_len
        all_var_offsets += [-1] * padding_len
        all_segments += [0] * padding_len
    
    # Clamp segment labels to valid range [0, 255]
    all_segments = [min(seg, 255) for seg in all_segments]
    
    # Create attention mask
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    attention_mask = [1 if tid != pad_token_id else 0 for tid in token_ids]
    
    # Split positions into separate lists
    binary_pos = [p[0] for p in all_positions]
    function_pos = [p[1] for p in all_positions]
    bb_pos = [p[2] for p in all_positions]
    
    return {
        'input_ids': torch.LongTensor(token_ids),
        'attention_mask': torch.LongTensor(attention_mask),
        'token_type_ids': torch.LongTensor(all_segments),
        'binary_pos': torch.FloatTensor(binary_pos),
        'function_pos': torch.FloatTensor(function_pos),
        'bb_pos': torch.FloatTensor(bb_pos),
        'var_offsets': torch.LongTensor(all_var_offsets)
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
            binary_pos = torch.tensor(batch_data['binary_pos'], dtype=torch.float).to(device)
            function_pos = torch.tensor(batch_data['function_pos'], dtype=torch.float).to(device)
            bb_pos = torch.tensor(batch_data['bb_pos'], dtype=torch.float).to(device)
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
