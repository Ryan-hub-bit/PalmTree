"""
Enhanced evaluation with pre-computed embeddings and configurable pools.

This module provides:
1. Generate and save embeddings for all functions (one-time cost)
2. Load pre-computed embeddings for fast evaluation
3. Create evaluation pools with positive + negative samples
4. Compute retrieval metrics (MRR, Recall@K)
"""

import os
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path


def generate_embeddings(model, dataset, model_type='baseline', save_path=None, batch_size=64):
    """
    Generate embeddings for all functions in the dataset.
    
    Args:
        model: The trained model
        dataset: FunctionDataset_CL_Load_JSON or FunctionDataset_CL_AddressAware_JSON instance
        model_type: 'baseline' or 'addressaware'
        save_path: Path to save embeddings (optional)
        batch_size: Batch size for embedding generation
        
    Returns:
        dict: {
            'embeddings': List of embeddings for each function group
            'metadata': List of metadata dicts (binary, func_name, opts)
        }
    """
    model.eval()
    all_embeddings = []
    all_metadata = []
    
    print(f"Generating embeddings for {len(dataset)} function groups...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            # Get all optimization variants for this function
            # Handle both tokenized_datas (baseline) and processed_datas (address-aware)
            if hasattr(dataset, 'tokenized_datas'):
                func_group = dataset.tokenized_datas[idx]
            elif hasattr(dataset, 'processed_datas'):
                func_group = dataset.processed_datas[idx]
            else:
                raise AttributeError("Dataset must have either 'tokenized_datas' or 'processed_datas' attribute")
            
            func_metadata = dataset.ebds[idx] if hasattr(dataset, 'ebds') else None
            
            # Collect embeddings for all opts in this group
            group_embeddings = []
            
            for func_data in func_group:
                if model_type == 'baseline':
                    input_ids = func_data['input_ids'].unsqueeze(0).cuda()
                    attention_mask = func_data['attention_mask'].unsqueeze(0).cuda()
                    token_type_ids = func_data['token_type_ids'].unsqueeze(0).cuda()
                    
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    # Handle both dict and object output
                    if isinstance(output, dict):
                        embedding = output['pooler_output'].cpu().numpy()
                    else:
                        embedding = output.pooler_output.cpu().numpy()
                    
                elif model_type == 'addressaware':
                    # Address-aware model needs additional position inputs
                    input_ids = func_data['input_ids'].unsqueeze(0).cuda()
                    attention_mask = func_data['attention_mask'].unsqueeze(0).cuda()
                    token_type_ids = func_data['token_type_ids'].unsqueeze(0).cuda()
                    binary_pos = func_data['binary_pos'].unsqueeze(0).cuda()
                    function_pos = func_data['function_pos'].unsqueeze(0).cuda()
                    bb_pos = func_data['bb_pos'].unsqueeze(0).cuda()
                    var_offsets = func_data['var_offsets'].unsqueeze(0).cuda()
                    
                    output = model(
                        token_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        binary_pos=binary_pos,
                        function_pos=function_pos,
                        bb_pos=bb_pos,
                        var_offsets=var_offsets
                    )
                    # Handle dict output (address-aware returns dict)
                    if isinstance(output, dict):
                        embedding = output['pooler_output'].cpu().numpy()
                    else:
                        embedding = output.pooler_output.cpu().numpy()
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
                
                group_embeddings.append(embedding[0])  # Shape: (768,)
            
            all_embeddings.append(group_embeddings)
            all_metadata.append(func_metadata)
    
    result = {
        'embeddings': all_embeddings,
        'metadata': all_metadata,
        'num_groups': len(all_embeddings)
    }
    
    # Save if path provided
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle to handle variable-length lists properly
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"Embeddings saved to {save_path}")
    
    return result


def load_embeddings(load_path):
    """
    Load pre-computed embeddings.
    
    Args:
        load_path: Path to pickle file with embeddings
        
    Returns:
        dict: Same format as generate_embeddings output
    """
    import pickle
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    return data


def evaluate_with_pool(embeddings_data, pool_size=100, num_negatives=None, 
                       opt_pairs=[('O0', 'O3')], all_opts_are_ground_truth=True):
    """
    Evaluate using pre-computed embeddings with configurable pool size.
    
    IMPORTANT: By default, ALL optimization levels of the same function are ground truths
    (matching the training behavior where any opt pair from same function is positive).
    
    Args:
        embeddings_data: Output from generate_embeddings or load_embeddings
        pool_size: Total size of candidate pool (ground truths + negatives)
        num_negatives: Number of negative samples per query (default: pool_size - num_ground_truths)
        opt_pairs: List of (anchor_opt, positive_opt) tuples to evaluate
        all_opts_are_ground_truth: If True, all opts of same function count as ground truth
                                    (e.g., O0→[O1,O2,O3] all correct). If False, only
                                    the specific pair in opt_pairs counts.
        
    Returns:
        dict: Evaluation metrics (MRR, Recall@1, Recall@5, Recall@10)
    """
    all_embeddings = embeddings_data['embeddings']
    metadata = embeddings_data['metadata']
    num_groups = len(all_embeddings)
    
    # Map optimization level to index
    opt_to_idx = {'O0': 0, 'O1': 1, 'O2': 2, 'O3': 3, 'Os': 4}
    
    print(f"Evaluating {num_groups} function groups with pool_size={pool_size}")
    if all_opts_are_ground_truth:
        print(f"  Ground truth: ALL optimization levels of same function (matches training)")
    else:
        print(f"  Ground truth: Only the specific target opt level")
    
    # Store metrics for each pair separately
    pair_metrics = {}
    overall_reciprocal_ranks = []
    overall_recall_at_1 = []
    overall_recall_at_5 = []
    overall_recall_at_10 = []
    
    for anchor_opt, target_opt in opt_pairs:
        anchor_idx = opt_to_idx.get(anchor_opt)
        target_idx = opt_to_idx.get(target_opt)
        
        if anchor_idx is None or target_idx is None:
            print(f"Skipping invalid opt pair: {anchor_opt}, {target_opt}")
            continue
        
        print(f"\nEvaluating {anchor_opt} vs {target_opt}...")
        
        # Metrics for this specific pair
        pair_reciprocal_ranks = []
        pair_recall_at_1 = []
        pair_recall_at_5 = []
        pair_recall_at_10 = []
        
        for query_idx in tqdm(range(num_groups)):
            query_group = all_embeddings[query_idx]
            
            # Skip if this group doesn't have the anchor opt
            if len(query_group) <= anchor_idx:
                continue
            
            anchor_emb = np.array(query_group[anchor_idx])
            
            # Determine ground truth indices
            if all_opts_are_ground_truth:
                # All optimization levels except the anchor itself are ground truths
                ground_truth_indices = [i for i in range(len(query_group)) if i != anchor_idx]
            else:
                # Only the specific target opt is ground truth
                if len(query_group) <= target_idx:
                    continue
                ground_truth_indices = [target_idx]
            
            num_ground_truths = len(ground_truth_indices)
            if num_ground_truths == 0:
                continue
            
            # Calculate number of negatives needed
            if num_negatives is None:
                negatives_needed = max(1, pool_size - num_ground_truths)
            else:
                negatives_needed = num_negatives
            
            # Sample negative candidates (from other function groups)
            negative_indices = [i for i in range(num_groups) if i != query_idx]
            sampled_negatives = np.random.choice(
                negative_indices,
                size=min(negatives_needed, len(negative_indices)),
                replace=False
            )
            
            # Build candidate pool: [ground truths] + [negatives]
            candidates = []
            ground_truth_positions = []  # Track which positions in pool are ground truths
            
            # Add ground truths first
            for gt_idx in ground_truth_indices:
                candidates.append(np.array(query_group[gt_idx]))
                ground_truth_positions.append(len(candidates) - 1)
            
            # Add negatives
            for neg_idx in sampled_negatives:
                neg_group = all_embeddings[neg_idx]
                # Use random opt level for negatives (same as training)
                neg_opt_idx = random.randint(0, min(len(neg_group) - 1, 3))
                if len(neg_group) > neg_opt_idx:
                    candidates.append(np.array(neg_group[neg_opt_idx]))
            
            candidates = np.array(candidates)
            
            # Validate pool
            assert len(candidates) >= num_ground_truths, "Pool must contain ground truths"
            
            # Compute similarities
            anchor_emb_normalized = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-8)
            candidates_normalized = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(candidates_normalized, anchor_emb_normalized)
            
            # Find rank of highest-ranked ground truth
            sorted_indices = np.argsort(-similarities)
            best_gt_rank = float('inf')
            for gt_pos in ground_truth_positions:
                rank = np.where(sorted_indices == gt_pos)[0][0] + 1
                best_gt_rank = min(best_gt_rank, rank)
            
            # Compute metrics using best ground truth rank
            pair_reciprocal_ranks.append(1.0 / best_gt_rank)
            pair_recall_at_1.append(1.0 if best_gt_rank == 1 else 0.0)
            pair_recall_at_5.append(1.0 if best_gt_rank <= 5 else 0.0)
            pair_recall_at_10.append(1.0 if best_gt_rank <= 10 else 0.0)
        
        # Calculate metrics for this pair
        pair_key = f"{anchor_opt}_vs_{target_opt}"
        pair_metrics[pair_key] = {
            'mrr': np.mean(pair_reciprocal_ranks) if pair_reciprocal_ranks else 0.0,
            'recall@1': np.mean(pair_recall_at_1) if pair_recall_at_1 else 0.0,
            'recall@5': np.mean(pair_recall_at_5) if pair_recall_at_5 else 0.0,
            'recall@10': np.mean(pair_recall_at_10) if pair_recall_at_10 else 0.0,
            'num_queries': len(pair_reciprocal_ranks)
        }
        
        # Print metrics for this pair
        print(f"\nResults for {anchor_opt} vs {target_opt}:")
        print(f"  MRR: {pair_metrics[pair_key]['mrr']:.4f}")
        print(f"  Recall@1: {pair_metrics[pair_key]['recall@1']:.4f}")
        print(f"  Recall@5: {pair_metrics[pair_key]['recall@5']:.4f}")
        print(f"  Recall@10: {pair_metrics[pair_key]['recall@10']:.4f}")
        print(f"  Queries: {pair_metrics[pair_key]['num_queries']}")
        
        # Add to overall metrics
        overall_reciprocal_ranks.extend(pair_reciprocal_ranks)
        overall_recall_at_1.extend(pair_recall_at_1)
        overall_recall_at_5.extend(pair_recall_at_5)
        overall_recall_at_10.extend(pair_recall_at_10)
    
    # Calculate overall metrics (average across all pairs)
    overall_metrics = {
        'mrr': np.mean(overall_reciprocal_ranks) if overall_reciprocal_ranks else 0.0,
        'recall@1': np.mean(overall_recall_at_1) if overall_recall_at_1 else 0.0,
        'recall@5': np.mean(overall_recall_at_5) if overall_recall_at_5 else 0.0,
        'recall@10': np.mean(overall_recall_at_10) if overall_recall_at_10 else 0.0,
        'num_queries': len(overall_reciprocal_ranks),
        'per_pair': pair_metrics
    }
    
    print(f"\n{'='*60}")
    print(f"Overall Evaluation Results (pool_size={pool_size}):")
    print(f"  MRR: {overall_metrics['mrr']:.4f}")
    print(f"  Recall@1: {overall_metrics['recall@1']:.4f}")
    print(f"  Recall@5: {overall_metrics['recall@5']:.4f}")
    print(f"  Recall@10: {overall_metrics['recall@10']:.4f}")
    print(f"  Total Queries: {overall_metrics['num_queries']}")
    print(f"{'='*60}")
    
    return overall_metrics


def finetune_eval_cached(model, dataset, model_type='baseline', 
                         embedding_cache_path=None, pool_size=100,
                         force_regenerate=False):
    """
    Evaluation with embedding caching.
    
    Args:
        model: The trained model
        dataset: FunctionDataset_CL_Load_JSON instance
        model_type: 'baseline' or 'addressaware'
        embedding_cache_path: Path to cache embeddings
        pool_size: Size of evaluation pool
        force_regenerate: Force re-generation even if cache exists
        
    Returns:
        dict: Evaluation metrics
    """
    # Check if embeddings exist
    if embedding_cache_path and Path(embedding_cache_path).exists() and not force_regenerate:
        print(f"Loading cached embeddings from {embedding_cache_path}...")
        embeddings_data = load_embeddings(embedding_cache_path)
    else:
        print("Generating embeddings (this may take a while)...")
        embeddings_data = generate_embeddings(
            model, dataset, model_type=model_type,
            save_path=embedding_cache_path
        )
    
    # Evaluate
    metrics = evaluate_with_pool(
        embeddings_data,
        pool_size=pool_size,
        opt_pairs=[('O0', 'O3'), ('O1', 'O3'), ('O2', 'O3')],
        all_opts_are_ground_truth=False
    )
    
    return metrics
