#!/usr/bin/env python
"""
Evaluate fine-tuned model with different pool sizes.

This script:
1. Loads a fine-tuned model
2. Generates embeddings for all functions (cached)
3. Evaluates with different pool sizes (100, 1000, 10000)
4. Uses a subset of queries (10% by default)
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import BertTokenizer
from data_json import FunctionDataset_CL_Load_JSON
from finetune_eval_with_pool import generate_embeddings, evaluate_with_pool, load_embeddings
from finetune import BinBertModel
import logging

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def load_model(model_path, tokenizer_path, model_type='baseline'):
    """Load fine-tuned model."""
    print(f"Loading model from {model_path}...")
    
    if model_type == 'baseline':
        from transformers import BertConfig
        
        # Load config
        config = BertConfig.from_pretrained(model_path)
        
        # Create model
        model = BinBertModel(config)
        
        # Load weights
        weights_path = Path(model_path) / 'pytorch_model.bin'
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded baseline model with vocab size {config.vocab_size}")
    
    elif model_type == 'addressaware':
        from transformers import BertModel, BertConfig
        from pretrain.address_aware.address_embedding import AddressAwareBERTEmbedding
        from finetune import AddressAwareBertWrapper
        import pickle
        import json
        
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
        
        # Wrap for evaluation
        model = AddressAwareBertWrapper(bert_model)
        
        print(f"Loaded address-aware model with vocab size {config_dict['vocab_size']}")
    
    else:
        raise NotImplementedError(f"Model type {model_type} not yet supported")
    
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='Evaluate with different pool sizes')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to tokenizer')
    parser.add_argument('--func_blocks', type=str, required=True,
                        help='Path to func_blocks.json')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground_truth.json')
    parser.add_argument('--model_type', type=str, default='baseline',
                        choices=['baseline', 'addressaware'])
    parser.add_argument('--embedding_cache', type=str, required=True,
                        help='Path to save/load embeddings')
    parser.add_argument('--pool_sizes', type=int, nargs='+', 
                        default=[100, 1000, 10000],
                        help='Pool sizes to evaluate')
    parser.add_argument('--query_ratio', type=float, default=0.1,
                        help='Ratio of queries to use (0.1 = 10%%)')
    parser.add_argument('--data_ratio', type=float, default=1.0,
                        help='Ratio of data to load')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Max sequence length')
    parser.add_argument('--regenerate', action='store_true',
                        help='Force regenerate embeddings')
    parser.add_argument('--output_file', type=str, default='pool_eval_results.txt',
                        help='File to save results')
    
    args = parser.parse_args()
    
    logger = setup_logger('pool_eval')
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    
    # Load dataset
    logger.info(f"Loading dataset (data_ratio={args.data_ratio})...")
    if args.model_type == 'addressaware':
        from data_json import FunctionDataset_CL_AddressAware_JSON
        dataset = FunctionDataset_CL_AddressAware_JSON(
            tokenizer=tokenizer,
            func_blocks_path=args.func_blocks,
            ground_truth_path=args.ground_truth,
            opt=['O0', 'O1', 'O2', 'O3'],
            add_ebd=True,
            max_length=args.max_length,
            data_ratio=args.data_ratio
        )
    else:
        dataset = FunctionDataset_CL_Load_JSON(
            tokenizer=tokenizer,
            func_blocks_path=args.func_blocks,
            ground_truth_path=args.ground_truth,
            opt=['O0', 'O1', 'O2', 'O3'],
            add_ebd=True,
            max_length=args.max_length,
            data_ratio=args.data_ratio
        )
    logger.info(f"Loaded {len(dataset)} function groups")
    
    # Check if embeddings exist
    embedding_path = Path(args.embedding_cache)
    
    if embedding_path.exists() and not args.regenerate:
        logger.info(f"Loading cached embeddings from {embedding_path}")
        embeddings_data = load_embeddings(str(embedding_path))
    else:
        # Load model
        model = load_model(args.model_path, args.tokenizer, args.model_type)
        model = model.cuda()
        
        # Generate embeddings
        logger.info("Generating embeddings (this may take a while)...")
        embeddings_data = generate_embeddings(
            model=model,
            dataset=dataset,
            model_type=args.model_type,
            save_path=str(embedding_path)
        )
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Evaluate with different pool sizes
    results = []
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating with query_ratio={args.query_ratio} ({int(args.query_ratio*100)}% of data)")
    logger.info(f"{'='*60}\n")
    
    for pool_size in args.pool_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating with pool_size={pool_size}")
        logger.info(f"{'='*60}")
        
        # Sample queries
        num_total_queries = embeddings_data['num_groups']
        num_queries = max(1, int(num_total_queries * args.query_ratio))
        
        # Create subset of embeddings for evaluation
        query_indices = np.random.choice(num_total_queries, num_queries, replace=False)
        
        # Create subset embeddings
        subset_embeddings = {
            'embeddings': [embeddings_data['embeddings'][i] for i in query_indices],
            'metadata': [embeddings_data['metadata'][i] for i in query_indices] if embeddings_data['metadata'] else None,
            'num_groups': num_queries
        }
        
        logger.info(f"Using {num_queries} / {num_total_queries} queries")
        
        # Evaluate
        metrics = evaluate_with_pool(
            embeddings_data=subset_embeddings,
            pool_size=pool_size,
            opt_pairs=[('O0', 'O3'), ('O1', 'O3'), ('O2', 'O3')],
            all_opts_are_ground_truth=False
        )
        
        result_entry = {
            'pool_size': pool_size,
            'num_queries': num_queries,
            **metrics
        }
        results.append(result_entry)
        
        logger.info(f"\nResults for pool_size={pool_size}:")
        logger.info(f"  MRR: {metrics['mrr']:.4f}")
        logger.info(f"  Recall@1: {metrics['recall@1']:.4f}")
        logger.info(f"  Recall@5: {metrics['recall@5']:.4f}")
        logger.info(f"  Recall@10: {metrics['recall@10']:.4f}")
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Query Ratio: {args.query_ratio} ({int(args.query_ratio*100)}%)\n")
        f.write(f"Data Ratio: {args.data_ratio}\n")
        f.write(f"Total Function Groups: {embeddings_data['num_groups']}\n")
        f.write(f"\n{'='*80}\n\n")
        
        f.write(f"{'Pool Size':<12} {'Queries':<10} {'MRR':<10} {'R@1':<10} {'R@5':<10} {'R@10':<10}\n")
        f.write(f"{'-'*80}\n")
        
        for result in results:
            f.write(f"{result['pool_size']:<12} "
                   f"{result['num_queries']:<10} "
                   f"{result['mrr']:<10.4f} "
                   f"{result['recall@1']:<10.4f} "
                   f"{result['recall@5']:<10.4f} "
                   f"{result['recall@10']:<10.4f}\n")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to {output_path}")
    logger.info(f"{'='*60}\n")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Pool Size':<12} {'Queries':<10} {'MRR':<10} {'R@1':<10} {'R@5':<10} {'R@10':<10}")
    print("-"*80)
    for result in results:
        print(f"{result['pool_size']:<12} "
              f"{result['num_queries']:<10} "
              f"{result['mrr']:<10.4f} "
              f"{result['recall@1']:<10.4f} "
              f"{result['recall@5']:<10.4f} "
              f"{result['recall@10']:<10.4f}")
    print("="*80)

if __name__ == '__main__':
    main()
