"""
Evaluate Pre-trained BERT (without fine-tuning) on Function Similarity

Uses the pre-trained BERT checkpoint directly with:
- Chunking (8 instructions/chunk, 60 tokens max)
- Mean pooling over chunks
- No projection layer, no L2 normalization
- Raw 768-dim BERT embeddings
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import argparse
import json
import logging
from datetime import datetime
from tqdm import tqdm
import sys
import os
import numpy as np
import importlib.util

# Add strupos to path first
strupos_path = os.path.join(os.path.dirname(__file__), '..', '..', 'strupos')
if strupos_path not in sys.path:
    sys.path.insert(0, strupos_path)

# Now import from strupos
from vocab import WordVocab

# Import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import BERT model directly
spec = importlib.util.spec_from_file_location("strupos_model", os.path.join(strupos_path, "model.py"))
strupos_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strupos_model)
AddressAwareBERT = strupos_model.AddressAwareBERT

# Import local dataloader
spec = importlib.util.spec_from_file_location("funcsim_dataloader", os.path.join(current_dir, "dataloader.py"))
funcsim_dataloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(funcsim_dataloader)
FunctionSimilarityDataset = funcsim_dataloader.FunctionSimilarityDataset


def setup_logging(log_dir, experiment_name):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'eval_pretrained_{experiment_name}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def compute_retrieval_metrics(bert_model, function_blocks, funcsim_pairs, vocab, device, logger, seq_len=512, top_k=[1, 5, 10], pool_size=None,
                            checkpoint_path=None, task_name=None, eval_pool_path=None):
    """
    Compute retrieval metrics using pretrained BERT.
    
    Uses chunking + mean pooling only (no projection, no normalization).
    Returns raw 768-dim BERT embeddings.
    """
    bert_model.eval()
    
    # Load pre-generated pool if specified
    if eval_pool_path is not None:
        logger.info(f"Loading pre-generated evaluation pool from: {eval_pool_path}")
        with open(eval_pool_path, 'r') as f:
            pool_data = json.load(f)
        
        sampled_query_ids = pool_data['query_ids']
        selected_func_ids = pool_data['pool_function_ids']
        
        # Filter to selected functions
        function_blocks = {fid: function_blocks[fid] for fid in selected_func_ids if fid in function_blocks}
        funcsim_pairs = {fid: funcsim_pairs[fid] for fid in sampled_query_ids if fid in funcsim_pairs}
        
        logger.info(f"Loaded pre-generated pool:")
        logger.info(f"  - Pool size: {len(function_blocks)} functions")
        logger.info(f"  - Queries: {len(funcsim_pairs)}")
        logger.info(f"  - Metadata: seed={pool_data['metadata'].get('seed')}, generated={pool_data['metadata'].get('generated_at')}")
    
    # Limit pool size if specified (and no pre-generated pool)
    elif pool_size is not None and pool_size < len(function_blocks):
        import random
        random.seed(42)
        
        all_func_ids = list(function_blocks.keys())
        all_query_ids = list(funcsim_pairs.keys())
        
        # Sample queries
        num_queries = min(max(100, pool_size // 10), len(all_query_ids))
        random.shuffle(all_query_ids)
        sampled_query_ids = all_query_ids[:num_queries]
        
        # Collect required functions
        required_func_ids = set(sampled_query_ids)
        for query_id in sampled_query_ids:
            if query_id in funcsim_pairs:
                for gt_id in funcsim_pairs[query_id]['ground_truth']:
                    if gt_id in function_blocks:
                        required_func_ids.add(gt_id)
        
        # Add distractors
        remaining = [fid for fid in all_func_ids if fid not in required_func_ids]
        random.shuffle(remaining)
        num_distractors = max(0, pool_size - len(required_func_ids))
        
        if len(required_func_ids) > pool_size:
            logger.warning(f"Required functions ({len(required_func_ids)}) exceed pool_size ({pool_size})")
        
        selected_func_ids = list(required_func_ids) + remaining[:num_distractors]
        
        function_blocks = {fid: function_blocks[fid] for fid in selected_func_ids if fid in function_blocks}
        funcsim_pairs = {fid: funcsim_pairs[fid] for fid in sampled_query_ids if fid in funcsim_pairs}
        
        logger.info(f"Sampled pool: {len(function_blocks)} functions")
        logger.info(f"  - Queries: {len(funcsim_pairs)}")
        logger.info(f"  - Required: {len(required_func_ids)}")
        logger.info(f"  - Distractors: {num_distractors}")
    else:
        logger.info(f"Pool size: {len(function_blocks)} functions")
        logger.info(f"Query functions: {len(funcsim_pairs)}")
    
    # Check cache
    embeddings_cache_path = None
    if task_name and checkpoint_path:
        embeddings_dir = os.path.join(os.path.dirname(checkpoint_path), 'embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)
        pool_suffix = f'_pool{len(function_blocks)}' if pool_size else '_full'
        embeddings_cache_path = os.path.join(embeddings_dir, f'{task_name}_pretrained{pool_suffix}_embeddings.pt')
        
        if os.path.exists(embeddings_cache_path):
            logger.info(f"Loading cached embeddings from: {embeddings_cache_path}")
            cached_data = torch.load(embeddings_cache_path, map_location='cpu')
            all_embeddings = cached_data['embeddings']
            logger.info(f"Loaded {len(all_embeddings)} cached embeddings")
            
            all_func_ids = list(function_blocks.keys())
            missing_ids = [fid for fid in all_func_ids if fid not in all_embeddings]
            
            if len(missing_ids) > 0:
                logger.warning(f"Cache incomplete: {len(missing_ids)} missing")
                all_embeddings = None
            else:
                logger.info("All embeddings found in cache!")
        else:
            logger.info(f"No cache found at: {embeddings_cache_path}")
            all_embeddings = None
    else:
        all_embeddings = None
    
    # Compute embeddings if not cached
    if all_embeddings is None:
        logger.info("Computing embeddings using pretrained BERT (chunking + mean pool only)...")
        
        dataset = FunctionSimilarityDataset(
            function_blocks_file=None,
            funcsim_pairs_file=None,
            vocab=vocab,
            seq_len=seq_len,
            negative_samples=0
        )
        dataset.function_blocks = function_blocks
        dataset.funcsim_pairs = funcsim_pairs
        
        all_embeddings = {}
        all_func_ids = list(function_blocks.keys())
        
        with torch.no_grad():
            for func_id in tqdm(all_func_ids, desc="Encoding functions"):
                result = dataset._process_function(func_id)
                func_input, func_segment, func_bin_pos, func_func_pos, func_bb_pos, func_var_offsets, num_instr, boundaries = result
                
                # Move to device and add batch dimension
                func_input = func_input.unsqueeze(0).to(device)
                func_segment = func_segment.unsqueeze(0).to(device)
                func_bin_pos = func_bin_pos.unsqueeze(0).to(device)
                func_func_pos = func_func_pos.unsqueeze(0).to(device)
                func_bb_pos = func_bb_pos.unsqueeze(0).to(device)
                func_var_offsets = func_var_offsets.unsqueeze(0).to(device)
                
                # Get BERT output
                bert_output = bert_model(func_input, func_segment, func_bin_pos, func_func_pos, func_bb_pos, func_var_offsets)
                
                # Chunk and mean pool (8 instructions per chunk)
                chunk_embeddings = []
                for i in range(num_instr):
                    start_idx = boundaries[i]
                    # End is either the start of next instruction or end of sequence
                    end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else func_input.size(1)
                    
                    if end_idx > start_idx:
                        chunk_emb = bert_output[0, start_idx:end_idx, :].mean(dim=0)
                        chunk_embeddings.append(chunk_emb)
                
                if len(chunk_embeddings) > 0:
                    # Mean pool over all chunks (no normalization)
                    func_emb = torch.stack(chunk_embeddings).mean(dim=0)
                else:
                    # Fallback to CLS token
                    func_emb = bert_output[0, 0, :]
                
                all_embeddings[func_id] = func_emb.cpu()
        
        logger.info(f"Computed {len(all_embeddings)} embeddings")
        
        # Cache embeddings
        if embeddings_cache_path:
            logger.info(f"Saving to cache: {embeddings_cache_path}")
            torch.save({
                'embeddings': all_embeddings,
                'checkpoint': checkpoint_path,
                'vocab_size': len(vocab),
                'num_functions': len(all_embeddings)
            }, embeddings_cache_path)
            logger.info("Cached successfully!")
    else:
        logger.info("Using cached embeddings")
    
    # Compute retrieval metrics
    recall_at_k = {k: [] for k in top_k}
    reciprocal_ranks = []
    
    logger.info("Computing retrieval metrics...")
    
    for query_id, pair_data in tqdm(funcsim_pairs.items(), desc="Evaluating queries"):
        ground_truth = set(pair_data['ground_truth'])
        
        if query_id not in all_embeddings:
            continue
        
        query_emb = all_embeddings[query_id].to(device)
        
        # Compute similarities
        similarities = {}
        for candidate_id in all_func_ids:
            if candidate_id == query_id:
                continue
            if candidate_id not in all_embeddings:
                continue
            
            candidate_emb = all_embeddings[candidate_id].to(device)
            sim = torch.cosine_similarity(query_emb.unsqueeze(0), candidate_emb.unsqueeze(0), dim=1).item()
            similarities[candidate_id] = sim
        
        # Rank candidates
        ranked_candidates = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        ranked_ids = [cand_id for cand_id, _ in ranked_candidates]
        
        # Recall@K
        for k in top_k:
            top_k_ids = set(ranked_ids[:k])
            if len(top_k_ids & ground_truth) > 0:
                recall_at_k[k].append(1.0)
            else:
                recall_at_k[k].append(0.0)
        
        # MRR
        found_rank = None
        for rank, cand_id in enumerate(ranked_ids, start=1):
            if cand_id in ground_truth:
                found_rank = rank
                break
        
        if found_rank is not None:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)
    
    # Compute averages
    results = {}
    for k in top_k:
        results[f'recall@{k}'] = np.mean(recall_at_k[k]) if recall_at_k[k] else 0.0
    results['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    results['num_queries'] = len(reciprocal_ranks)
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info("Retrieval Metrics (Pretrained BERT):")
    logger.info(f"{'='*60}")
    logger.info(f"Number of queries: {results['num_queries']}")
    for k in top_k:
        logger.info(f"Recall@{k}: {results[f'recall@{k}']:.4f}")
    logger.info(f"MRR: {results['mrr']:.4f}")
    logger.info(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained BERT on function similarity")
    
    # Data
    parser.add_argument("--function_blocks", type=str, required=True, help="Path to function_blocks.json")
    parser.add_argument("--funcsim_pairs", type=str, required=True, help="Path to funcsim_pairs.json")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab.pkl")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained BERT checkpoint")
    parser.add_argument("--hidden", type=int, default=768, help="Hidden size")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--attn_heads", type=int, default=12, help="Number of attention heads")
    
    # Evaluation
    parser.add_argument("--seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--pool_size", type=int, default=None, help="Limit retrieval pool size")
    parser.add_argument("--eval_pool", type=str, default=None, help="Path to pre-generated eval pool JSON (overrides pool_size)")
    
    # Output
    parser.add_argument("--output", type=str, default="../../output/funcsim/pretrained_results.json", help="Output file")
    parser.add_argument("--log_dir", type=str, default="../../log/funcsim", help="Log directory")
    parser.add_argument("--task_name", type=str, default="mlm", help="Task name for caching")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger = setup_logging(args.log_dir, "pretrained_eval")
    
    logger.info("=" * 80)
    logger.info("Pretrained BERT Evaluation (No Fine-tuning)")
    logger.info("Method: Chunking + Mean Pooling (768-dim, no projection/normalization)")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("=" * 80)
    
    # Load vocabulary
    logger.info(f"Loading vocabulary from {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Detect checkpoint capabilities
    logger.info(f"Detecting checkpoint capabilities...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'bert_state_dict' in checkpoint:
            state_dict = checkpoint['bert_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()
    
    has_address = any('address_position' in k or 'address_pos' in k for k in state_dict.keys())
    has_var = any('var_position' in k or 'var_offset' in k for k in state_dict.keys())
    
    logger.info("=" * 80)
    logger.info("CHECKPOINT CAPABILITIES")
    logger.info("=" * 80)
    logger.info(f"Has address embeddings: {has_address}")
    logger.info(f"Has var embeddings: {has_var}")
    logger.info("=" * 80)
    
    # Create AddressAwareBERT (no projection layer)
    logger.info("Creating BERT model...")
    bert_model = AddressAwareBERT(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        max_len=args.seq_len,
        use_address_embedding=has_address,
        use_var_embedding=has_var
    )
    
    # Load pretrained BERT weights
    logger.info("Loading pretrained BERT weights...")
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'bert_state_dict' in checkpoint_data:
        bert_model.load_state_dict(checkpoint_data['bert_state_dict'])
    elif 'model_state_dict' in checkpoint_data:
        bert_model.load_state_dict(checkpoint_data['model_state_dict'])
    else:
        bert_model.load_state_dict(checkpoint_data)
    
    bert_model = bert_model.to(device)
    logger.info("Model ready! (BERT: pretrained, no projection layer)")
    
    # Load function data
    logger.info("Loading function blocks and pairs...")
    
    # If using pre-generated pool, try to load filtered function blocks first (much faster)
    if args.eval_pool:
        filtered_blocks_file = args.eval_pool.replace('.json', '_function_blocks.json')
        if os.path.exists(filtered_blocks_file):
            logger.info(f"Loading filtered function blocks from {filtered_blocks_file}...")
            with open(filtered_blocks_file, 'r') as f:
                function_blocks = json.load(f)
            logger.info(f"Loaded {len(function_blocks)} functions (filtered for pool)")
        else:
            logger.info(f"Loading full function blocks (this may take a while)...")
            with open(args.function_blocks, 'r') as f:
                function_blocks = json.load(f)
            logger.info(f"Loaded {len(function_blocks)} functions (full dataset)")
    else:
        with open(args.function_blocks, 'r') as f:
            function_blocks = json.load(f)
        logger.info(f"Loaded {len(function_blocks)} functions")
    
    with open(args.funcsim_pairs, 'r') as f:
        funcsim_pairs = json.load(f)
    
    logger.info(f"Loaded {len(function_blocks)} function blocks")
    logger.info(f"Loaded {len(funcsim_pairs)} query functions")
    
    # Create dataset and set flag
    dataset = FunctionSimilarityDataset(
        function_blocks_file=None,
        funcsim_pairs_file=None,
        vocab=vocab,
        seq_len=args.seq_len,
        negative_samples=0
    )
    dataset.use_address_var = has_address and has_var
    dataset.function_blocks = function_blocks
    dataset.funcsim_pairs = funcsim_pairs
    
    if dataset.use_address_var:
        logger.info("✓ Dataloader will use parsed address/var info")
    else:
        logger.info("✗ Dataloader will mask address/var to -1")
    
    # Compute retrieval metrics
    logger.info("\n" + "="*80)
    logger.info("RETRIEVAL EVALUATION")
    logger.info("="*80)
    
    retrieval_results = compute_retrieval_metrics(
        bert_model=bert_model,
        function_blocks=function_blocks,
        funcsim_pairs=funcsim_pairs,
        vocab=vocab,
        device=device,
        logger=logger,
        seq_len=args.seq_len,
        top_k=[1, 5, 10, 20],
        pool_size=args.pool_size,
        checkpoint_path=args.checkpoint,
        task_name=args.task_name,
        eval_pool_path=args.eval_pool
    )
    
    # Save results
    output_data = {
        'checkpoint': args.checkpoint,
        'method': 'pretrained_bert',
        'retrieval_results': retrieval_results,
        'checkpoint_info': {
            'has_address': has_address,
            'has_var': has_var
        }
    }
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to: {args.output}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
