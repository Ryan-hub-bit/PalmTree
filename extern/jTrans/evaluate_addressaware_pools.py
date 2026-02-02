#!/usr/bin/env python3
"""
Evaluate finetuned address-aware model on evaluation pools.

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
import re
from transformers import BertTokenizer, BertModel, BertConfig


def load_vocab(vocab_path):
    """Load vocabulary using BertTokenizer"""
    print(f"Loading tokenizer from: {vocab_path}")
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    print(f"Vocabulary size: {len(tokenizer)}")
    return tokenizer


class AddressAwareBertWrapper(torch.nn.Module):
    """
    Wrapper for address-aware BERT (matching finetune.py).
    Extracts [CLS] token as pooled output.
    """
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        
    def forward(self, token_ids, attention_mask, token_type_ids,
                binary_pos, function_pos, bb_pos, var_offsets=None):
        """Forward pass returning pooled output (CLS token)."""
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
        
        return pooler_output


def load_model(checkpoint_path, vocab_stoi, device='cuda'):
    """Load finetuned address-aware model"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Import address-aware embedding
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pretrain', 'address_aware'))
    from address_embedding import AddressAwareBERTEmbedding
    
    # Load config
    config_path = os.path.join(checkpoint_path, 'config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = BertConfig(**config_dict)
    
    # Create BERT model
    bert_model = BertModel(config, add_pooling_layer=False)
    
    # Replace embeddings with address-aware version
    bert_model.embeddings = AddressAwareBERTEmbedding(
        vocab_size=config.vocab_size,
        embed_size=config.hidden_size,
        dropout=0.1,
        max_len=config.max_position_embeddings,
        use_address_embedding=True,
        use_var_embedding=True,
        segment_types=256,
        vocab_stoi=vocab_stoi
    )
    
    # Load weights
    weights_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
    state_dict = torch.load(weights_path, map_location=device)
    bert_model.load_state_dict(state_dict)
    
    # CRITICAL: Re-set vocab_stoi after loading state_dict
    # load_state_dict() restores parameters/buffers but NOT Python attributes like vocab_stoi
    bert_model.embeddings.vocab_stoi = vocab_stoi
    print(f"  vocab_stoi restored (address={vocab_stoi.get('address')}, daddr={vocab_stoi.get('daddr')})")
    
    # Wrap model
    model = AddressAwareBertWrapper(bert_model).to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    return model, config


def parse_address_aware_function(func_str, max_len=512):
    """
    Parse address-aware function into tokens and position arrays.
    Format: opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 operand2 ...
    """
    # Regex patterns (same as dataloader)
    addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
    var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
    
    tokens = []
    binary_positions = []
    function_positions = []
    bb_positions = []
    var_offsets = []
    
    parts = func_str.strip().split()
    
    for part in parts:
        # Check for opcode(address) pattern
        match = addr_pattern.match(part)
        if match:
            tokens.append(match.group(1))
            binary_positions.append(float(match.group(3)))
            function_positions.append(float(match.group(4)))
            bb_positions.append(float(match.group(5)))
            var_offsets.append(-1)
            continue
        
        # Check for nested address() pattern
        nested_match = nested_addr_pattern.match(part)
        if nested_match:
            tokens.append('address')
            binary_positions.append(float(nested_match.group(2)))
            function_positions.append(float(nested_match.group(3)))
            bb_positions.append(float(nested_match.group(4)))
            var_offsets.append(-1)
            continue
        
        # Check for daddr() pattern
        daddr_match = daddr_pattern.match(part)
        if daddr_match:
            tokens.append('daddr')
            binary_positions.append(float(daddr_match.group(2)))
            function_positions.append(float(daddr_match.group(3)))
            bb_positions.append(float(daddr_match.group(4)))
            var_offsets.append(-1)
            continue
        
        # Check for var() pattern
        var_match = var_pattern.match(part)
        if var_match:
            tokens.append('var')
            var_hex = var_match.group(1)
            var_offset = int(var_hex, 16)
            # Convert to signed
            if var_offset >= 0x80000000:
                var_offset = var_offset - 0x100000000
            binary_positions.append(-1.0)
            function_positions.append(-1.0)
            bb_positions.append(-1.0)
            var_offsets.append(var_offset)
            continue
        
        # Regular token
        tokens.append(part)
        binary_positions.append(-1.0)
        function_positions.append(-1.0)
        bb_positions.append(-1.0)
        var_offsets.append(-1)
    
    return tokens, binary_positions, function_positions, bb_positions, var_offsets


def tokenize_function(func_block, tokenizer, vocab_stoi, max_len=512):
    """
    Tokenize address-aware function block.
    
    CRITICAL: Must match pretrain/finetune format:
    - Use <sos> and <eos> (NOT [CLS] and [SEP])
    - Generate instruction-level segment labels (1, 2, 3, ...)
    - Use <pad> for padding (NOT [PAD])
    """
    func_str = func_block['tokens']
    
    # Split by tabs to get instructions (matching pretrain format)
    instructions = func_str.split('\t') if '\t' in func_str else [func_str]
    
    all_tokens = []
    all_binary_pos = []
    all_function_pos = []
    all_bb_pos = []
    all_var_offsets = []
    all_segments = []
    
    # Add <sos> at beginning (segment 1) - MATCH PRETRAIN
    all_tokens.append('<sos>')
    all_binary_pos.append(-1.0)
    all_function_pos.append(-1.0)
    all_bb_pos.append(-1.0)
    all_var_offsets.append(-1)
    all_segments.append(1)
    
    for inst_idx, inst_text in enumerate(instructions):
        inst_text = inst_text.strip()
        if not inst_text:
            continue
        
        # Parse this instruction
        tokens, binary_pos, function_pos, bb_pos, var_offsets = parse_address_aware_function(inst_text, max_len)
        
        # Segment label = instruction number (1-indexed)
        inst_segment = inst_idx + 1
        
        all_tokens.extend(tokens)
        all_binary_pos.extend(binary_pos)
        all_function_pos.extend(function_pos)
        all_bb_pos.extend(bb_pos)
        all_var_offsets.extend(var_offsets)
        all_segments.extend([inst_segment] * len(tokens))
    
    # Add <eos> at end (gets last instruction's segment) - MATCH PRETRAIN
    last_segment = inst_idx + 1 if instructions else 1
    all_tokens.append('<eos>')
    all_binary_pos.append(-1.0)
    all_function_pos.append(-1.0)
    all_bb_pos.append(-1.0)
    all_var_offsets.append(-1)
    all_segments.append(last_segment)
    
    # Truncate if needed
    if len(all_tokens) > max_len:
        all_tokens = all_tokens[:max_len]
        all_binary_pos = all_binary_pos[:max_len]
        all_function_pos = all_function_pos[:max_len]
        all_bb_pos = all_bb_pos[:max_len]
        all_var_offsets = all_var_offsets[:max_len]
        all_segments = all_segments[:max_len]
    
    # Convert tokens to IDs (use <unk> for unknown, not [UNK])
    token_ids = [vocab_stoi.get(t, vocab_stoi.get('<unk>', 1)) for t in all_tokens]
    
    # Pad to max_len (use <pad>, not [PAD])
    padding_len = max_len - len(token_ids)
    token_ids += [vocab_stoi.get('<pad>', 0)] * padding_len
    all_binary_pos += [-1.0] * padding_len
    all_function_pos += [-1.0] * padding_len
    all_bb_pos += [-1.0] * padding_len
    all_var_offsets += [-1] * padding_len
    all_segments += [0] * padding_len  # Padding gets segment 0
    
    # Clamp segment labels to valid range [0, 255] (segment_types=256)
    all_segments = [min(seg, 255) for seg in all_segments]
    
    return token_ids, all_binary_pos, all_function_pos, all_bb_pos, all_var_offsets, all_segments


def generate_embeddings(model, func_ids, func_blocks, tokenizer, vocab_stoi, device, max_len=512, batch_size=32):
    """Generate embeddings for a list of function IDs"""
    embeddings = []
    
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
            
            batch_segments = []
            
            for fid in batch_ids:
                token_ids, bin_pos, func_pos, bb_p, var_off, segments = tokenize_function(
                    func_blocks[fid], tokenizer, vocab_stoi, max_len
                )
                
                batch_input_ids.append(token_ids)
                batch_binary_pos.append(bin_pos)
                batch_function_pos.append(func_pos)
                batch_bb_pos.append(bb_p)
                batch_var_offsets.append(var_off)
                batch_segments.append(segments)
                
                # Create attention mask (use <pad> not [PAD])
                attention_mask = [1 if tid != vocab_stoi.get('<pad>', 0) else 0 for tid in token_ids]
                batch_attention_mask.append(attention_mask)
            
            # Convert to tensors
            input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(device)
            binary_pos = torch.tensor(batch_binary_pos, dtype=torch.float).to(device)
            function_pos = torch.tensor(batch_function_pos, dtype=torch.float).to(device)
            bb_pos = torch.tensor(batch_bb_pos, dtype=torch.float).to(device)
            var_offsets = torch.tensor(batch_var_offsets, dtype=torch.long).to(device)
            attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(device)
            token_type_ids = torch.tensor(batch_segments, dtype=torch.long).to(device)  # Proper segment labels!
            
            # Get embeddings
            batch_embeddings = model(
                token_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                binary_pos=binary_pos,
                function_pos=function_pos,
                bb_pos=bb_pos,
                var_offsets=var_offsets
            )
            
            embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)


def evaluate_pool(pool_path, model, func_blocks, tokenizer, vocab_stoi, device, max_len=512, batch_size=32):
    """Evaluate on a single pool"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {Path(pool_path).name}")
    print(f"{'='*80}")
    
    with open(pool_path, 'r') as f:
        pool_data = json.load(f)
    
    # Detect pool format: old format vs new format
    # Old format: {'pool': [id1, id2, ...], 'size': N, ...}
    # New format: {'pool': [{'high_id': ..., 'low_id': ...}], 'queries': [...], ...}
    
    if 'queries' in pool_data:
        # NEW FORMAT: separate pool and queries
        actual_size = pool_data.get('actual_size', len(pool_data['pool']))
        print(f"Pool size: {actual_size}")
        print(f"Queries: {len(pool_data['queries'])}")
        
        # Generate embeddings for all pool functions
        pool_func_ids = [p['high_id'] for p in pool_data['pool']]
        print(f"\nGenerating pool embeddings ({len(pool_func_ids)} functions)...")
        pool_embeddings = generate_embeddings(
            model, pool_func_ids, func_blocks, tokenizer, vocab_stoi, device, max_len, batch_size
        )
        
        # Generate embeddings for queries
        query_func_ids = [q['query_id'] for q in pool_data['queries']]
        print(f"\nGenerating query embeddings ({len(query_func_ids)} functions)...")
        query_embeddings = generate_embeddings(
            model, query_func_ids, func_blocks, tokenizer, vocab_stoi, device, max_len, batch_size
        )
        
    else:
        # OLD FORMAT: pool is list of function IDs, need to load separate query file
        actual_size = pool_data.get('metadata', {}).get('actual_size', pool_data.get('size', len(pool_data['pool'])))
        print(f"Pool size: {actual_size}")
        
        # Find corresponding query file
        pool_name = Path(pool_path).stem  # e.g., 'pool_100_O0_vs_O3_filtered'
        query_file = Path(pool_path).parent / f"query_{pool_name.replace('pool_', '')}.json"
        
        if not query_file.exists():
            print(f"ERROR: Query file not found: {query_file}")
            print("Old format pools require a separate query file.")
            return None
        
        with open(query_file, 'r') as f:
            query_data = json.load(f)
        
        print(f"Queries: {len(query_data['queries'])}")
        
        # Generate embeddings for pool (old format: direct list of IDs)
        pool_func_ids = pool_data['pool']
        print(f"\nGenerating pool embeddings ({len(pool_func_ids)} functions)...")
        pool_embeddings = generate_embeddings(
            model, pool_func_ids, func_blocks, tokenizer, vocab_stoi, device, max_len, batch_size
        )
        
        # Generate embeddings for queries (from separate query file)
        query_func_ids = [q['query_id'] for q in query_data['queries']]
        print(f"\nGenerating query embeddings ({len(query_func_ids)} functions)...")
        query_embeddings = generate_embeddings(
            model, query_func_ids, func_blocks, tokenizer, vocab_stoi, device, max_len, batch_size
        )
        
        # Use query data for evaluation
        pool_data['queries'] = query_data['queries']
    
    # Compute similarities and ranks
    print("\nComputing similarities and ranks...")
    
    # Normalize embeddings for cosine similarity (CRITICAL: same as finetune!)
    print("  Normalizing embeddings...")
    pool_norm = pool_embeddings / (np.linalg.norm(pool_embeddings, axis=1, keepdims=True) + 1e-8)
    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    
    mrr_sum = 0.0
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    
    for i, query in enumerate(tqdm(pool_data['queries'])):
        query_emb = query_norm[i:i+1]
        
        # Compute cosine similarities (normalized dot product = cosine similarity)
        similarities = np.dot(pool_norm, query_emb.T).flatten()
        
        # Get ranks (descending similarity)
        ranks = np.argsort(-similarities)
        
        # Find ground truth position
        gt_id = query['gt_id']
        gt_idx = pool_func_ids.index(gt_id)
        rank_position = np.where(ranks == gt_idx)[0][0] + 1  # 1-indexed
        
        # Update metrics
        mrr_sum += 1.0 / rank_position
        if rank_position == 1:
            recall_at_1 += 1
        if rank_position <= 5:
            recall_at_5 += 1
        if rank_position <= 10:
            recall_at_10 += 1
    
    num_queries = len(pool_data['queries'])
    results = {
        'pool_file': Path(pool_path).name,
        'pool_size': pool_data['actual_size'],
        'num_queries': num_queries,
        'MRR': mrr_sum / num_queries,
        'Recall@1': recall_at_1 / num_queries,
        'Recall@5': recall_at_5 / num_queries,
        'Recall@10': recall_at_10 / num_queries
    }
    
    print(f"\nResults:")
    print(f"  MRR: {results['MRR']:.4f}")
    print(f"  Recall@1: {results['Recall@1']:.4f}")
    print(f"  Recall@5: {results['Recall@5']:.4f}")
    print(f"  Recall@10: {results['Recall@10']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate address-aware model on pools')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--vocab_path', required=True, help='Vocabulary path')
    parser.add_argument('--pool_dir', required=True, help='Pool directory')
    parser.add_argument('--func_blocks', required=True, help='Function blocks JSON')
    parser.add_argument('--max_len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--pool_size', type=int, help='Evaluate specific pool size only')
    parser.add_argument('--opt_pair', help='Evaluate specific opt pair only (e.g., O0_vs_O3)')
    parser.add_argument('--output', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Load tokenizer and build vocab_stoi
    tokenizer = load_vocab(args.vocab_path)
    vocab_path = os.path.join(args.vocab_path, 'vocab.txt')
    vocab_stoi = {}
    with open(vocab_path, 'r') as f:
        for idx, line in enumerate(f):
            token = line.strip()
            vocab_stoi[token] = idx
    
    # Load model
    model, config = load_model(args.checkpoint, vocab_stoi, args.device)
    
    # Load func blocks
    print(f"\nLoading func blocks: {args.func_blocks}")
    with open(args.func_blocks, 'r') as f:
        func_blocks = json.load(f)
    print(f"Loaded {len(func_blocks)} functions")
    
    # Find pool files
    pool_dir = Path(args.pool_dir)
    pool_files = sorted(pool_dir.glob('pool_*.json'))
    
    # Filter by pool_size and opt_pair if specified
    if args.pool_size:
        pool_files = [p for p in pool_files if f'_{args.pool_size}.json' in str(p)]
    if args.opt_pair:
        pool_files = [p for p in pool_files if args.opt_pair in str(p)]
    
    print(f"\nFound {len(pool_files)} pool files to evaluate")
    
    # Evaluate each pool
    all_results = []
    for pool_file in pool_files:
        results = evaluate_pool(
            pool_file, model, func_blocks, tokenizer, vocab_stoi,
            args.device, args.max_len, args.batch_size
        )
        all_results.append(results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for res in all_results:
        print(f"\n{res['pool_file']}:")
        print(f"  MRR: {res['MRR']:.4f}")
        print(f"  Recall@1: {res['Recall@1']:.4f}")
        print(f"  Recall@5: {res['Recall@5']:.4f}")
        print(f"  Recall@10: {res['Recall@10']:.4f}")
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
