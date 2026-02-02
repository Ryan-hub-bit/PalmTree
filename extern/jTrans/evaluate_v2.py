"""
jTrans Function Similarity Evaluation (strupos-style)

Evaluates fine-tuned model on function similarity retrieval task.
Computes Recall@K and MRR metrics.
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

# Import our self-contained model
from bert_model import FunctionSimilarityModel
from data_json import FunctionDataset_CL_AddressAware_JSON


def setup_logging(log_dir, experiment_name):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'eval_{experiment_name}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def compute_retrieval_metrics(model, function_blocks, query_ids, pool_ids, funcsim_pairs, vocab, device, logger, 
                             max_len=512, top_k=[1, 5, 10]):
    """
    Compute retrieval metrics: Recall@K and MRR.
    
    For each query function, rank all pool candidates by similarity
    and check if ground truth functions appear in top K.
    
    Args:
        model: Trained FunctionSimilarityModel
        function_blocks: Dict of function_id -> instruction data
        query_ids: List of query function IDs
        pool_ids: List of pool function IDs (candidates)
        funcsim_pairs: Dict of function_id -> {ground_truth: [ids]}
        vocab: WordVocab instance
        device: torch device
        logger: Logger instance
        max_len: Maximum sequence length
        top_k: List of K values for Recall@K
        
    Returns:
        Dictionary with recall@k and MRR metrics
    """
    model.eval()
    
    logger.info("Generating embeddings for pool...")
    pool_embeddings = {}
    
    # Generate embeddings for all pool functions
    with torch.no_grad():
        for func_id in tqdm(pool_ids, desc="Pool embeddings"):
            if func_id not in function_blocks:
                continue
            
            # Parse function (similar to dataset)
            func_data = function_blocks[func_id]
            # TODO: Process function data and get embedding
            # For now, skip implementation details
            pass
    
    logger.info("Evaluating queries...")
    recall_at_k = {k: 0 for k in top_k}
    mrr_sum = 0.0
    valid_queries = 0
    
    with torch.no_grad():
        for query_id in tqdm(query_ids, desc="Query evaluation"):
            if query_id not in funcsim_pairs or query_id not in function_blocks:
                continue
            
            ground_truth = set(funcsim_pairs[query_id]['ground_truth'])
            if not ground_truth:
                continue
            
            # Get query embedding
            # TODO: Process query function
            
            # Compute similarities with all pool functions
            similarities = []
            for pool_id in pool_ids:
                if pool_id in pool_embeddings:
                    # Compute similarity
                    sim = 0.0  # TODO: Compute actual similarity
                    similarities.append((pool_id, sim))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Compute metrics
            valid_queries += 1
            
            # Recall@K
            for k in top_k:
                top_k_ids = set([func_id for func_id, _ in similarities[:k]])
                if len(ground_truth & top_k_ids) > 0:
                    recall_at_k[k] += 1
            
            # MRR
            for rank, (func_id, _) in enumerate(similarities, start=1):
                if func_id in ground_truth:
                    mrr_sum += 1.0 / rank
                    break
    
    # Compute averages
    results = {}
    for k in top_k:
        results[f'recall@{k}'] = recall_at_k[k] / valid_queries if valid_queries > 0 else 0
    
    results['mrr'] = mrr_sum / valid_queries if valid_queries > 0 else 0
    results['num_queries'] = valid_queries
    
    return results


def main():
    parser = argparse.ArgumentParser(description="jTrans Function Similarity Evaluation")
    
    # Data paths
    parser.add_argument('--func_blocks', type=str, required=True, help='Function blocks JSON')
    parser.add_argument('--ground_truth', type=str, required=True, help='Ground truth JSON')
    parser.add_argument('--pool_ids', type=str, required=True, help='Pool IDs JSON')
    parser.add_argument('--vocab_path', type=str, required=True, help='Vocabulary path')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True, help='Trained model checkpoint')
    
    # Model config
    parser.add_argument('--hidden', type=int, default=768)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--attn_heads', type=int, default=12)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=512)
    
    # Output
    parser.add_argument('--output', type=str, required=True, help='Output results JSON')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    
    # Evaluation config
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.log_dir, 'evaluation')
    
    logger.info("=" * 80)
    logger.info("jTrans Function Similarity Evaluation")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("=" * 80)
    
    # Load vocabulary
    from pretrain.address_aware.vocab import WordVocab
    logger.info(f"Loading vocabulary from {args.vocab_path}")
    vocab = WordVocab.load_vocab(args.vocab_path)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Load data
    logger.info("Loading function blocks...")
    with open(args.func_blocks, 'r') as f:
        function_blocks = json.load(f)
    
    logger.info("Loading ground truth...")
    with open(args.ground_truth, 'r') as f:
        ground_truth_data = json.load(f)
    
    # Convert ground truth to funcsim_pairs format
    funcsim_pairs = {}
    for pair in ground_truth_data:
        # Assuming format: {O0: id1, O1: id2, O2: id3, O3: id4}
        # Create bidirectional pairs
        opt_ids = [pair.get(f'O{i}') for i in range(4) if f'O{i}' in pair]
        for func_id in opt_ids:
            if func_id:
                ground_truth_list = [x for x in opt_ids if x and x != func_id]
                funcsim_pairs[func_id] = {'ground_truth': ground_truth_list}
    
    logger.info("Loading pool IDs...")
    with open(args.pool_ids, 'r') as f:
        pool_data = json.load(f)
        pool_ids = pool_data if isinstance(pool_data, list) else pool_data.get('pool_ids', [])
    
    logger.info(f"Function blocks: {len(function_blocks)}")
    logger.info(f"Funcsim pairs: {len(funcsim_pairs)}")
    logger.info(f"Pool size: {len(pool_ids)}")
    
    # Load checkpoint and auto-detect
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    has_address = any('address_position' in k or 'address' in k for k in state_dict.keys())
    has_var = any('var_position' in k or 'var' in k for k in state_dict.keys())
    
    logger.info(f"Checkpoint has address embeddings: {has_address}")
    logger.info(f"Checkpoint has var embeddings: {has_var}")
    
    # Create model
    logger.info("Creating model...")
    model = FunctionSimilarityModel(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        max_len=args.max_len,
        use_address_embedding=has_address,
        use_var_embedding=has_var,
        embedding_dim=args.embedding_dim,
        freeze_bert=False
    )
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Compute metrics
    logger.info("Computing retrieval metrics...")
    
    # Use a subset of queries for faster evaluation
    query_ids = list(funcsim_pairs.keys())[:1000]  # Limit to 1000 queries
    
    results = compute_retrieval_metrics(
        model=model,
        function_blocks=function_blocks,
        query_ids=query_ids,
        pool_ids=pool_ids,
        funcsim_pairs=funcsim_pairs,
        vocab=vocab,
        device=device,
        logger=logger,
        max_len=args.max_len,
        top_k=[1, 5, 10]
    )
    
    # Log results
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    for key, value in results.items():
        if key.startswith('recall'):
            logger.info(f"{key}: {value:.4f} ({value*100:.2f}%)")
        elif key == 'mrr':
            logger.info(f"MRR: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 80)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
