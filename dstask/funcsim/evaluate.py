"""
Evaluation script for Function Similarity

Tests the fine-tuned model on the held-out test set.
Computes retrieval metrics: Recall@K, MRR, accuracy, etc.
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

# Import local modules using importlib to avoid conflicts
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load local model.py
spec = importlib.util.spec_from_file_location("funcsim_model", os.path.join(current_dir, "model.py"))
funcsim_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(funcsim_model)

# Load local dataloader.py
spec = importlib.util.spec_from_file_location("funcsim_dataloader", os.path.join(current_dir, "dataloader.py"))
funcsim_dataloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(funcsim_dataloader)

# Extract classes
FunctionSimilarityModel = funcsim_model.FunctionSimilarityModel
FunctionSimilarityDataset = funcsim_dataloader.FunctionSimilarityDataset


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


def compute_retrieval_metrics(model, function_blocks, funcsim_pairs, vocab, device, logger, seq_len=512, top_k=[1, 5, 10], pool_size=None):
    """
    Compute retrieval metrics: Recall@K and MRR.
    
    For each query function, rank all candidates by similarity and check
    if ground truth functions appear in top K.
    
    Args:
        model: Trained FunctionSimilarityModel
        function_blocks: Dict of function_id -> instructions
        funcsim_pairs: Dict of function_id -> ground_truth list
        vocab: WordVocab instance
        device: torch device
        logger: Logger instance
        seq_len: Maximum sequence length
        top_k: List of K values for Recall@K
        pool_size: Maximum number of functions in retrieval pool (None = use all)
        
    Returns:
        Dictionary with recall@k and MRR metrics
    """
    model.eval()
    
    # Limit pool size if specified
    if pool_size is not None and pool_size < len(function_blocks):
        import random
        all_func_ids = list(function_blocks.keys())
        query_ids = set(funcsim_pairs.keys())
        
        # Randomly sample pool_size functions from all available
        random.shuffle(all_func_ids)
        selected_ids = all_func_ids[:pool_size]
        
        # Keep only queries that are in the sampled pool
        sampled_query_ids = [fid for fid in selected_ids if fid in query_ids]
        
        function_blocks = {fid: function_blocks[fid] for fid in selected_ids if fid in function_blocks}
        funcsim_pairs = {fid: funcsim_pairs[fid] for fid in sampled_query_ids if fid in funcsim_pairs}
        
        logger.info(f"Sampled pool size: {len(function_blocks)} functions (from {len(all_func_ids)} total)")
        logger.info(f"Sampled queries: {len(funcsim_pairs)} query functions (from {len(query_ids)} total)")
    else:
        logger.info(f"Pool size: {len(function_blocks)} functions")
        logger.info(f"Query functions: {len(funcsim_pairs)}")
    
    logger.info("Computing embeddings for all function blocks...")
    
    # Create a temporary dataset to process all function blocks
    dataset = FunctionSimilarityDataset(
        function_blocks_file=None,  # We'll override
        funcsim_pairs_file=None,
        vocab=vocab,
        seq_len=seq_len,
        negative_samples=0
    )
    dataset.function_blocks = function_blocks
    dataset.funcsim_pairs = funcsim_pairs
    
    # Compute embeddings for all function blocks
    all_embeddings = {}
    all_func_ids = list(function_blocks.keys())
    
    with torch.no_grad():
        for func_id in tqdm(all_func_ids, desc="Encoding functions"):
            # Process function
            func_input, func_segment, func_bin_pos, func_func_pos, func_bb_pos, func_var_offsets = dataset._process_function(func_id)
            
            # Move to device and add batch dimension
            func_input = func_input.unsqueeze(0).to(device)
            func_segment = func_segment.unsqueeze(0).to(device)
            func_bin_pos = func_bin_pos.unsqueeze(0).to(device)
            func_func_pos = func_func_pos.unsqueeze(0).to(device)
            func_bb_pos = func_bb_pos.unsqueeze(0).to(device)
            
            # Compute embedding
            emb = model(func_input, func_segment, func_bin_pos, func_func_pos, func_bb_pos)
            all_embeddings[func_id] = emb.squeeze(0).cpu()  # [embedding_dim]
    
    logger.info(f"Computed embeddings for {len(all_embeddings)} functions")
    
    # Compute retrieval metrics
    recall_at_k = {k: [] for k in top_k}
    reciprocal_ranks = []
    
    logger.info("Computing retrieval metrics...")
    
    for query_id, pair_data in tqdm(funcsim_pairs.items(), desc="Evaluating queries"):
        ground_truth = set(pair_data['ground_truth'])
        
        # Skip if query doesn't have embeddings
        if query_id not in all_embeddings:
            continue
        
        query_emb = all_embeddings[query_id].to(device)
        
        # Compute similarities to all other functions
        similarities = {}
        for candidate_id in all_func_ids:
            if candidate_id == query_id:  # Skip self
                continue
            if candidate_id not in all_embeddings:
                continue
            
            candidate_emb = all_embeddings[candidate_id].to(device)
            sim = torch.cosine_similarity(query_emb.unsqueeze(0), candidate_emb.unsqueeze(0), dim=1).item()
            similarities[candidate_id] = sim
        
        # Rank candidates by similarity (descending)
        ranked_candidates = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        ranked_ids = [cand_id for cand_id, _ in ranked_candidates]
        
        # Compute Recall@K
        for k in top_k:
            top_k_ids = set(ranked_ids[:k])
            # Check if any ground truth is in top K
            if len(top_k_ids & ground_truth) > 0:
                recall_at_k[k].append(1.0)
            else:
                recall_at_k[k].append(0.0)
        
        # Compute reciprocal rank (MRR)
        found_rank = None
        for rank, cand_id in enumerate(ranked_ids, start=1):
            if cand_id in ground_truth:
                found_rank = rank
                break
        
        if found_rank is not None:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)
    
    # Compute average metrics
    results = {}
    for k in top_k:
        results[f'recall@{k}'] = np.mean(recall_at_k[k]) if recall_at_k[k] else 0.0
    results['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    results['num_queries'] = len(reciprocal_ranks)
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info("Retrieval Metrics:")
    logger.info(f"{'='*60}")
    logger.info(f"Number of queries: {results['num_queries']}")
    for k in top_k:
        logger.info(f"Recall@{k}: {results[f'recall@{k}']:.4f}")
    logger.info(f"MRR (Mean Reciprocal Rank): {results['mrr']:.4f}")
    logger.info(f"{'='*60}")
    
    return results


def evaluate_model(model, test_loader, device, logger):
    """Evaluate model on test set."""
    model.eval()
    
    correct = 0
    total = 0
    
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            func1_input = batch['func1_input'].to(device)
            func1_segment = batch['func1_segment'].to(device)
            func1_bin_pos = batch['func1_binary_pos'].to(device)
            func1_func_pos = batch['func1_function_pos'].to(device)
            func1_bb_pos = batch['func1_bb_pos'].to(device)
            
            func2_input = batch['func2_input'].to(device)
            func2_segment = batch['func2_segment'].to(device)
            func2_bin_pos = batch['func2_binary_pos'].to(device)
            func2_func_pos = batch['func2_function_pos'].to(device)
            func2_bb_pos = batch['func2_bb_pos'].to(device)
            
            labels = batch['label'].to(device)
            
            # Forward pass
            emb1 = model(func1_input, func1_segment, func1_bin_pos, func1_func_pos, func1_bb_pos)
            emb2 = model(func2_input, func2_segment, func2_bin_pos, func2_func_pos, func2_bb_pos)
            
            # Compute similarity
            similarity = model.compute_similarity(emb1, emb2, metric='cosine')
            predictions = (similarity > 0.5).long()
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_similarities.extend(similarity.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Compute average similarity for positive and negative pairs
    pos_similarities = [s for s, l in zip(all_similarities, all_labels) if l == 1]
    neg_similarities = [s for s, l in zip(all_similarities, all_labels) if l == 0]
    
    avg_pos_sim = np.mean(pos_similarities) if pos_similarities else 0.0
    avg_neg_sim = np.mean(neg_similarities) if neg_similarities else 0.0
    
    logger.info(f"\n{'='*60}")
    logger.info("Test Set Results:")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    logger.info(f"Avg Positive Similarity: {avg_pos_sim:.4f}")
    logger.info(f"Avg Negative Similarity: {avg_neg_sim:.4f}")
    logger.info(f"Similarity Gap: {avg_pos_sim - avg_neg_sim:.4f}")
    logger.info(f"{'='*60}")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_pos_similarity': avg_pos_sim,
        'avg_neg_similarity': avg_neg_sim,
        'similarity_gap': avg_pos_sim - avg_neg_sim
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate function similarity model")
    
    # Data
    parser.add_argument("--function_blocks", type=str, required=True, help="Path to function_blocks.json")
    parser.add_argument("--funcsim_pairs", type=str, required=True, help="Path to funcsim_pairs.json")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab.pkl")
    parser.add_argument("--test_indices", type=str, required=True, help="Path to test_indices.json")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--hidden", type=int, default=768, help="Hidden size")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--attn_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Function embedding dimension")
    parser.add_argument("--no_address", action="store_true", help="Disable address embeddings")
    
    # Evaluation
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--negative_samples", type=int, default=3, help="Negative samples per positive")
    parser.add_argument("--pool_size", type=int, default=None, help="Limit retrieval pool size (None = use all)")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of test data to use (0.0-1.0)")
    
    # Output
    parser.add_argument("--output", type=str, default="../../output/funcsim/test_results.json", help="Output file")
    parser.add_argument("--log_dir", type=str, default="../../log/funcsim", help="Log directory")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger = setup_logging(args.log_dir, "eval")
    
    logger.info("=" * 80)
    logger.info("Function Similarity Evaluation")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test indices: {args.test_indices}")
    logger.info("=" * 80)
    
    # Load vocabulary
    logger.info(f"Loading vocabulary from {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Load full dataset
    logger.info("Loading dataset...")
    full_dataset = FunctionSimilarityDataset(
        function_blocks_file=args.function_blocks,
        funcsim_pairs_file=args.funcsim_pairs,
        vocab=vocab,
        seq_len=args.seq_len,
        negative_samples=args.negative_samples
    )
    
    # Load test indices
    logger.info(f"Loading test indices from {args.test_indices}")
    with open(args.test_indices, 'r') as f:
        test_info = json.load(f)
    test_indices = test_info['test_indices']
    
    # Sample test data if data_fraction < 1.0
    if args.data_fraction < 1.0:
        import random
        sample_size = int(len(test_indices) * args.data_fraction)
        random.seed(42)  # For reproducibility
        test_indices = random.sample(test_indices, sample_size)
        logger.info(f"Sampled {sample_size} test indices ({args.data_fraction*100:.1f}% of {len(test_info['test_indices'])} total)")
    
    # Create test dataset
    test_dataset = Subset(full_dataset, test_indices)
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = FunctionSimilarityModel(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        max_len=args.seq_len,
        use_address_embedding=not args.no_address,
        embedding_dim=args.embedding_dim,
        freeze_bert=False
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"Checkpoint val accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    
    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("PAIRWISE EVALUATION (Accuracy)")
    logger.info("="*80)
    pairwise_results = evaluate_model(model, test_loader, device, logger)
    
    # Compute retrieval metrics (Recall@K, MRR)
    logger.info("\n" + "="*80)
    logger.info("RETRIEVAL EVALUATION (Recall@K, MRR)")
    logger.info("="*80)
    
    # Load function blocks and pairs for retrieval evaluation
    logger.info("Loading function blocks and pairs for retrieval evaluation...")
    with open(args.function_blocks, 'r') as f:
        function_blocks = json.load(f)
    with open(args.funcsim_pairs, 'r') as f:
        funcsim_pairs = json.load(f)
    
    # Filter to only test set functions
    test_func_ids = set()
    for idx in test_indices:
        # Get the function IDs from the test pairs
        if idx < len(full_dataset.training_pairs):
            func1_id, func2_id, _ = full_dataset.training_pairs[idx]
            test_func_ids.add(func1_id)
            test_func_ids.add(func2_id)
    
    # Filter function blocks and pairs to test set only
    test_function_blocks = {fid: function_blocks[fid] for fid in test_func_ids if fid in function_blocks}
    test_funcsim_pairs = {fid: funcsim_pairs[fid] for fid in test_func_ids if fid in funcsim_pairs}
    
    logger.info(f"Test set has {len(test_funcsim_pairs)} query functions")
    
    retrieval_results = compute_retrieval_metrics(
        model=model,
        function_blocks=test_function_blocks,
        funcsim_pairs=test_funcsim_pairs,
        vocab=vocab,
        device=device,
        logger=logger,
        seq_len=args.seq_len,
        top_k=[1, 5, 10, 20],
        pool_size=args.pool_size
    )
    
    # Save results
    output_data = {
        'checkpoint': args.checkpoint,
        'test_size': len(test_dataset),
        'pairwise_results': pairwise_results,
        'retrieval_results': retrieval_results,
        'checkpoint_info': {
            'epoch': checkpoint.get('epoch', None),
            'train_acc': checkpoint.get('train_acc', None),
            'val_acc': checkpoint.get('val_acc', None)
        }
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to: {args.output}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
