"""
Evaluate PalmTree Models (Baseline and Address-Aware) on Function Similarity

This script evaluates PalmTree models on function similarity retrieval task:
- BASELINE: Original BERT without address features
- ADDRESS_AWARE: BERT with address embeddings

Uses the same evaluation protocol as evaluate_pretrained.py:
- Chunking (instruction-based)
- Mean pooling
- Cosine similarity for retrieval
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

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
strupos_path = os.path.join(current_dir, '..', '..', 'strupos')
palmtree_path = os.path.join(current_dir, '..', '..', 'extern', 'PalmTree', 'src')

if strupos_path not in sys.path:
    sys.path.insert(0, strupos_path)
if palmtree_path not in sys.path:
    sys.path.insert(0, palmtree_path)

# Import vocab
from vocab import WordVocab

# Import PalmTree models
from palmtree.model.bert import BERT
from palmtree.model.bert_addressaware import AddressAwareBERT

# Import local dataloader
spec = importlib.util.spec_from_file_location("funcsim_dataloader", os.path.join(current_dir, "dataloader.py"))
funcsim_dataloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(funcsim_dataloader)
FunctionSimilarityDataset = funcsim_dataloader.FunctionSimilarityDataset


def setup_logging(log_dir, experiment_name):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'eval_palmtree_{experiment_name}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def compute_retrieval_metrics(model, function_blocks, funcsim_pairs, vocab, device, logger, 
                             model_type, seq_len=512, top_k=[1, 5, 10], pool_size=None,
                             checkpoint_path=None, task_name=None, eval_pool_path=None):
    """
    Compute retrieval metrics using PalmTree model.
    
    Args:
        model: BERT or AddressAwareBERT model
        function_blocks: Dict of function_id -> instructions
        funcsim_pairs: Dict of function_id -> ground_truth list
        vocab: WordVocab instance
        device: torch device
        logger: Logger instance
        model_type: 'baseline' or 'address_aware'
        seq_len: Maximum sequence length
        top_k: List of K values for Recall@K
        pool_size: Maximum number of functions in retrieval pool
        checkpoint_path: Path to model checkpoint (for caching)
        task_name: Task name for organizing cached embeddings
        eval_pool_path: Path to pre-generated evaluation pool
        
    Returns:
        Dictionary with recall@k and MRR metrics
    """
    model.eval()
    
    # Load pre-generated pool if specified
    if eval_pool_path is not None:
        logger.info(f"Loading pre-generated evaluation pool from: {eval_pool_path}")
        with open(eval_pool_path, 'r') as f:
            pool_data = json.load(f)
        
        sampled_query_ids = pool_data['query_ids']
        selected_func_ids = set(pool_data['pool_function_ids'])
        
        # Filter queries first
        funcsim_pairs_initial = {fid: funcsim_pairs[fid] for fid in sampled_query_ids if fid in funcsim_pairs}
        
        # CRITICAL: Add ALL ground truth functions to the pool
        logger.info("Ensuring all ground truth functions are in the pool...")
        added_gt_count = 0
        funcsim_pairs = {}
        missing_from_data = 0
        
        for query_id, pair_data in funcsim_pairs_initial.items():
            ground_truth = pair_data['ground_truth']
            
            # Add all ground truth functions to pool
            for gt_id in ground_truth:
                if gt_id not in selected_func_ids:
                    if gt_id in function_blocks:
                        selected_func_ids.add(gt_id)
                        added_gt_count += 1
                    else:
                        missing_from_data += 1
            
            # Only keep queries where ALL ground truth functions are available
            available_gt = [gt_id for gt_id in ground_truth if gt_id in function_blocks]
            if len(available_gt) == len(ground_truth):
                funcsim_pairs[query_id] = {
                    'ground_truth': available_gt,
                    'metadata': pair_data.get('metadata', {})
                }
        
        # Filter function_blocks to selected functions
        function_blocks = {fid: function_blocks[fid] for fid in selected_func_ids if fid in function_blocks}
        
        logger.info(f"Loaded pre-generated pool:")
        logger.info(f"  - Pool size (original): {len(pool_data['pool_function_ids'])}")
        logger.info(f"  - Pool size (with GT): {len(function_blocks)}")
        logger.info(f"  - Added ground truth: {added_gt_count}")
        logger.info(f"  - Queries (initial): {len(funcsim_pairs_initial)}")
        logger.info(f"  - Queries (valid): {len(funcsim_pairs)}")
        if missing_from_data > 0:
            logger.warning(f"  - ⚠ {missing_from_data} ground truth functions missing from function_blocks data")
        if len(funcsim_pairs) < len(funcsim_pairs_initial):
            logger.warning(f"  - ⚠ Removed {len(funcsim_pairs_initial) - len(funcsim_pairs)} queries with incomplete ground truth")
        logger.info(f"  - Metadata: seed={pool_data['metadata'].get('seed')}, generated={pool_data['metadata'].get('generated_at')}")
    
    # Limit pool size if specified (and no pre-generated pool)
    elif pool_size is not None and pool_size < len(function_blocks):
        import random
        random.seed(42)
        
        all_func_ids = list(function_blocks.keys())
        all_query_ids = list(funcsim_pairs.keys())
        
        # Sample queries first
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
        
        # Add random distractors
        remaining = [fid for fid in all_func_ids if fid not in required_func_ids]
        random.shuffle(remaining)
        num_distractors = max(0, pool_size - len(required_func_ids))
        
        selected_func_ids = list(required_func_ids) + remaining[:num_distractors]
        
        # Filter to selected functions
        function_blocks = {fid: function_blocks[fid] for fid in selected_func_ids if fid in function_blocks}
        funcsim_pairs = {fid: funcsim_pairs[fid] for fid in sampled_query_ids if fid in funcsim_pairs}
        
        logger.info(f"Sampled pool: {len(function_blocks)} functions")
        logger.info(f"  - Queries: {len(funcsim_pairs)}")
        logger.info(f"  - Required: {len(required_func_ids)}")
        logger.info(f"  - Distractors: {num_distractors}")
    else:
        logger.info(f"Using all functions: {len(function_blocks)}")
        logger.info(f"  - Queries: {len(funcsim_pairs)}")
    
    all_func_ids = list(function_blocks.keys())
    
    # Check for cached embeddings
    embeddings_cache_path = None
    if checkpoint_path and task_name:
        cache_dir = os.path.join(os.path.dirname(checkpoint_path), 'embedding_cache', task_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Use pool hash if available
        if eval_pool_path:
            import hashlib
            with open(eval_pool_path, 'rb') as f:
                pool_hash = hashlib.md5(f.read()).hexdigest()[:8]
            embeddings_cache_path = os.path.join(cache_dir, f'embeddings_pool_{pool_hash}.pt')
        else:
            embeddings_cache_path = os.path.join(cache_dir, 'embeddings.pt')
    
    all_embeddings = None
    if embeddings_cache_path and os.path.exists(embeddings_cache_path):
        logger.info(f"Loading cached embeddings from: {embeddings_cache_path}")
        try:
            cache_data = torch.load(embeddings_cache_path, map_location='cpu', weights_only=False)
            all_embeddings = cache_data['embeddings']
            
            # Verify cache completeness
            missing_ids = [fid for fid in all_func_ids if fid not in all_embeddings]
            
            if len(missing_ids) > 0:
                logger.warning(f"Cache incomplete: {len(missing_ids)} missing")
                all_embeddings = None
            else:
                logger.info("All embeddings found in cache!")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            all_embeddings = None
    
    # Compute embeddings if not cached
    if all_embeddings is None:
        logger.info(f"Computing embeddings using PalmTree {model_type} model...")
        
        # Create dataset
        dataset = FunctionSimilarityDataset(
            function_blocks_file=None,
            funcsim_pairs_file=None,
            vocab=vocab,
            seq_len=seq_len,
            negative_samples=0
        )
        
        # Set address/var usage flag
        if model_type == 'address_aware':
            dataset.use_address_var = True
            logger.info("✓ Using address/var embeddings")
        else:
            dataset.use_address_var = False
            logger.info("✗ Masking address/var (baseline mode)")
        
        dataset.function_blocks = function_blocks
        dataset.funcsim_pairs = funcsim_pairs
        
        all_embeddings = {}
        
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
                
                # Get model output
                if model_type == 'baseline':
                    # Baseline BERT: only token IDs and segment labels
                    model_output = model(func_input, func_segment)
                else:
                    # Address-aware BERT: include all positional features
                    model_output = model(func_input, func_segment, func_bin_pos, func_func_pos, func_bb_pos, func_var_offsets)
                
                # Chunk and mean pool (instruction-based)
                chunk_embeddings = []
                for i in range(num_instr):
                    start_idx = boundaries[i]
                    end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else func_input.size(1)
                    
                    if end_idx > start_idx:
                        chunk_emb = model_output[0, start_idx:end_idx, :].mean(dim=0)
                        chunk_embeddings.append(chunk_emb)
                
                if len(chunk_embeddings) > 0:
                    # Mean pool over all chunks
                    func_emb = torch.stack(chunk_embeddings).mean(dim=0)
                else:
                    # Fallback to CLS token
                    func_emb = model_output[0, 0, :]
                
                all_embeddings[func_id] = func_emb.cpu()
        
        logger.info(f"Computed {len(all_embeddings)} embeddings")
        
        # Cache embeddings
        if embeddings_cache_path:
            logger.info(f"Saving to cache: {embeddings_cache_path}")
            torch.save({
                'embeddings': all_embeddings,
                'checkpoint': checkpoint_path,
                'vocab_size': len(vocab),
                'num_functions': len(all_embeddings),
                'model_type': model_type
            }, embeddings_cache_path)
            logger.info("Cached successfully!")
    else:
        logger.info("Using cached embeddings")
    
    # Compute retrieval metrics
    recall_at_k = {k: [] for k in top_k}
    reciprocal_ranks = []
    
    logger.info("Computing retrieval metrics...")
    
    # Validate ground truth availability in pool
    pool_func_set = set(all_func_ids)
    queries_with_valid_gt = 0
    total_gt_in_pool = 0
    total_gt_count = 0
    
    for query_id, pair_data in funcsim_pairs.items():
        ground_truth = pair_data['ground_truth']
        total_gt_count += len(ground_truth)
        available_gt = [gt_id for gt_id in ground_truth if gt_id in pool_func_set]
        total_gt_in_pool += len(available_gt)
        if len(available_gt) > 0:
            queries_with_valid_gt += 1
    
    logger.info(f"Ground truth validation:")
    logger.info(f"  - Queries with ≥1 GT in pool: {queries_with_valid_gt}/{len(funcsim_pairs)}")
    logger.info(f"  - Total GT in pool: {total_gt_in_pool}/{total_gt_count} ({100*total_gt_in_pool/max(total_gt_count,1):.1f}%)")
    
    if total_gt_in_pool < total_gt_count * 0.5:
        logger.error("⚠️  CRITICAL: Less than 50% of ground truth functions are in the pool!")
        logger.error("    This will result in artificially low metrics. Check your pool generation!")
    
    for query_id, pair_data in tqdm(funcsim_pairs.items(), desc="Evaluating queries"):
        ground_truth = set(pair_data['ground_truth'])
        
        # Filter ground truth to only those in pool
        ground_truth = ground_truth & pool_func_set
        
        if len(ground_truth) == 0:
            # Skip queries with no valid ground truth
            continue
        
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
            
            # Cosine similarity
            sim = torch.nn.functional.cosine_similarity(
                query_emb.unsqueeze(0),
                candidate_emb.unsqueeze(0)
            ).item()
            
            similarities[candidate_id] = sim
        
        # Rank candidates
        ranked_candidates = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        ranked_ids = [cid for cid, _ in ranked_candidates]
        
        # Compute recall@k
        for k in top_k:
            top_k_ids = set(ranked_ids[:k])
            hit = len(top_k_ids & ground_truth) > 0
            recall_at_k[k].append(1.0 if hit else 0.0)
        
        # Compute MRR
        for rank, cid in enumerate(ranked_ids, 1):
            if cid in ground_truth:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    # Aggregate metrics
    results = {
        'recall@k': {k: np.mean(recall_at_k[k]) for k in top_k},
        'mrr': np.mean(reciprocal_ranks),
        'num_queries': len(funcsim_pairs),
        'pool_size': len(function_blocks)
    }
    
    logger.info("\n" + "="*80)
    logger.info("RETRIEVAL RESULTS")
    logger.info("="*80)
    for k in top_k:
        logger.info(f"Recall@{k}: {results['recall@k'][k]:.4f}")
    logger.info(f"MRR: {results['mrr']:.4f}")
    logger.info(f"Queries: {results['num_queries']}")
    logger.info(f"Pool size: {results['pool_size']}")
    logger.info("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PalmTree models on function similarity")
    
    # Data
    parser.add_argument("--function_blocks", type=str, required=True, help="Path to function_blocks.json")
    parser.add_argument("--funcsim_pairs", type=str, required=True, help="Path to funcsim_pairs.json")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab file")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PalmTree checkpoint")
    parser.add_argument("--model_type", type=str, required=True, choices=['baseline', 'address_aware'], 
                       help="Model type: baseline or address_aware")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--address_embed_dim", type=int, default=128, help="Address embedding dimension (address_aware only)")
    parser.add_argument("--var_embed_dim", type=int, default=32, help="Var embedding dimension (address_aware only)")
    
    # Evaluation
    parser.add_argument("--seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--pool_size", type=int, default=None, help="Limit retrieval pool size")
    parser.add_argument("--eval_pool", type=str, default=None, help="Path to pre-generated eval pool JSON")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Output
    parser.add_argument("--output", type=str, required=True, help="Output file")
    parser.add_argument("--log_dir", type=str, default="../../log/funcsim", help="Log directory")
    parser.add_argument("--task_name", type=str, required=True, help="Task name for caching")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger = setup_logging(args.log_dir, args.task_name)
    
    logger.info("=" * 80)
    logger.info(f"PalmTree {args.model_type.upper()} Evaluation")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Model type: {args.model_type}")
    logger.info("=" * 80)
    
    # Load vocabulary
    logger.info(f"Loading vocabulary from {args.vocab}")
    if args.vocab.endswith('.pkl'):
        vocab = WordVocab.load_vocab(args.vocab)
    else:
        # Load from text file (PalmTree format)
        vocab = WordVocab.load_vocab(args.vocab)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    if args.model_type == 'baseline':
        model = BERT(
            vocab_size=len(vocab),
            hidden=args.hidden,
            n_layers=args.n_layers,
            attn_heads=args.attn_heads,
            dropout=0.1
        )
    else:  # address_aware
        model = AddressAwareBERT(
            vocab_size=len(vocab),
            hidden=args.hidden,
            n_layers=args.n_layers,
            attn_heads=args.attn_heads,
            dropout=0.1,
            use_address_embedding=True,
            use_var_embedding=True,
            max_len=args.seq_len,
            segment_types=16  # Support instruction-level segments (1-8) plus padding
        )
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint_data, dict):
        if 'model_state_dict' in checkpoint_data:
            state_dict = checkpoint_data['model_state_dict']
        elif 'bert_state_dict' in checkpoint_data:
            state_dict = checkpoint_data['bert_state_dict']
        else:
            state_dict = checkpoint_data
    else:
        state_dict = checkpoint_data.state_dict()
    
    # If state_dict has "bert." prefix (from FunctionSimilarityModel), extract only BERT weights
    if any(k.startswith('bert.') for k in state_dict.keys()):
        logger.info("Detected FunctionSimilarityModel checkpoint, extracting BERT weights...")
        bert_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('bert.'):
                # Remove "bert." prefix
                new_key = k[5:]  # len('bert.') = 5
                bert_state_dict[new_key] = v
        state_dict = bert_state_dict
        logger.info(f"Extracted {len(state_dict)} BERT parameters")
    
    # Load checkpoint - should now match exactly
    model.load_state_dict(state_dict)
    logger.info("Checkpoint loaded successfully!")
    
    model = model.to(device)
    logger.info("Model loaded successfully!")
    
    # Load function data
    logger.info("Loading function blocks and pairs...")
    
    # If using pre-generated pool, try to load filtered function blocks first
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
    
    # Compute retrieval metrics
    logger.info("\n" + "="*80)
    logger.info("RETRIEVAL EVALUATION")
    logger.info("="*80)
    
    retrieval_results = compute_retrieval_metrics(
        model=model,
        function_blocks=function_blocks,
        funcsim_pairs=funcsim_pairs,
        vocab=vocab,
        device=device,
        logger=logger,
        model_type=args.model_type,
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
        'model_type': args.model_type,
        'method': f'palmtree_{args.model_type}',
        'retrieval_results': retrieval_results,
        'model_config': {
            'hidden': args.hidden,
            'n_layers': args.n_layers,
            'attn_heads': args.attn_heads
        }
    }
    
    if args.model_type == 'address_aware':
        output_data['model_config']['address_embed_dim'] = args.address_embed_dim
        output_data['model_config']['var_embed_dim'] = args.var_embed_dim
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to: {args.output}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
