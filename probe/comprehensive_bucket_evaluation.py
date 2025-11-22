#!/usr/bin/env python3
"""
Comprehensive Bucket Prediction Evaluation

Tests BOTH models (address-aware and baseline) on all three bucket tasks:
- BB-level bucket prediction
- Function-level bucket prediction  
- Binary-level bucket prediction

Uses JSON label files as ground truth and evaluates model predictions.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'addressaware'))

from palmtree.dataset.vocab import WordVocab
import bert_pytorch


def load_vocab(vocab_path):
    """Load vocabulary."""
    print(f"[INFO] Loading vocabulary from {vocab_path}")
    vocab = WordVocab.load_vocab(vocab_path)
    print(f"[INFO] Vocabulary size: {len(vocab)}")
    return vocab


def load_baseline_model(checkpoint_path, vocab_size, hidden=128, n_layers=12, attn_heads=8, device='cuda'):
    """Load baseline BERT model."""
    print(f"[INFO] Loading baseline model from {checkpoint_path}")
    
    # Create model
    bert = bert_pytorch.BERT(
        vocab_size=vocab_size,
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads
    )
    
    # Load checkpoint (weights_only=False needed for PyTorch 2.6+)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # It's a dictionary with state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'bert_state_dict' in checkpoint:
            state_dict = checkpoint['bert_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'bert.' prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('bert.'):
                new_key = key[5:]  # Remove 'bert.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        bert.load_state_dict(new_state_dict, strict=False)
    else:
        # It's a model object directly - just return it
        bert = checkpoint.to(device)
        bert.eval()
    
    bert = bert.to(device)
    bert.eval()
    
    print(f"[INFO] Baseline model loaded successfully")
    return bert


def load_addressaware_model(checkpoint_path, vocab_size, hidden=128, n_layers=12, attn_heads=8, device='cuda'):
    """Load address-aware BERT model."""
    print(f"[INFO] Loading address-aware model from {checkpoint_path}")
    
    from model import AddressAwareBERT
    
    # Create model
    bert = AddressAwareBERT(
        vocab_size=vocab_size,
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads
    )
    
    # Load checkpoint (weights_only=False needed for PyTorch 2.6+)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # It's a dictionary with state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'bert_state_dict' in checkpoint:
            state_dict = checkpoint['bert_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'bert.' prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('bert.'):
                new_key = key[5:]  # Remove 'bert.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        bert.load_state_dict(new_state_dict, strict=False)
    else:
        # It's a model object directly - just return it
        bert = checkpoint.to(device)
        bert.eval()
    
    bert = bert.to(device)
    bert.eval()
    
    print(f"[INFO] Address-aware model loaded successfully")
    return bert


def tokenize_instruction(inst_text, vocab):
    """
    Tokenize instruction text.
    
    Returns list of token IDs.
    """
    # Parse instruction to extract opcode and operands
    tokens = inst_text.strip().split()
    
    token_ids = []
    for token in tokens:
        # Handle special tokens
        if token in vocab.stoi:
            token_ids.append(vocab.stoi[token])
        else:
            token_ids.append(vocab.stoi.get('[UNK]', 0))
    
    return token_ids


def get_embedding_baseline(model, token_ids, vocab, device='cuda'):
    """
    Get embedding from baseline model using mean pooling.
    
    Args:
        model: Baseline BERT model
        token_ids: List of token IDs
        vocab: Vocabulary
        device: Device to use
    
    Returns:
        embedding: numpy array of shape (hidden_size,)
    """
    # Add [CLS] and [SEP]
    cls_id = vocab.stoi.get('[CLS]', 1)
    sep_id = vocab.stoi.get('[SEP]', 2)
    
    input_ids = [cls_id] + token_ids + [sep_id]
    segment_labels = [0] * len(input_ids)
    
    # Convert to tensors
    input_tensor = torch.LongTensor([input_ids]).to(device)
    segment_tensor = torch.LongTensor([segment_labels]).to(device)
    
    # Get embeddings
    with torch.no_grad():
        output = model(input_tensor, segment_tensor)  # [1, seq_len, hidden]
        
        # Mean pooling (excluding padding)
        mask = (input_tensor > 0).float().unsqueeze(-1)  # [1, seq_len, 1]
        masked_output = output * mask
        sum_output = masked_output.sum(dim=1)  # [1, hidden]
        count = mask.sum(dim=1)  # [1, 1]
        embedding = sum_output / count  # [1, hidden]
    
    return embedding.cpu().numpy()[0]


def get_embedding_addressaware(model, token_ids, positions, vocab, device='cuda'):
    """
    Get embedding from address-aware model using mean pooling.
    
    Args:
        model: Address-aware BERT model
        token_ids: List of token IDs
        positions: Dict with 'binary_pos', 'func_pos', 'bb_pos'
        vocab: Vocabulary
        device: Device to use
    
    Returns:
        embedding: numpy array of shape (hidden_size,)
    """
    # Add [CLS] and [SEP]
    cls_id = vocab.stoi.get('[CLS]', 1)
    sep_id = vocab.stoi.get('[SEP]', 2)
    
    input_ids = [cls_id] + token_ids + [sep_id]
    segment_labels = [0] * len(input_ids)
    
    # Position embeddings (same for all tokens in single instruction)
    binary_positions = [positions['binary_pos']] * len(input_ids)
    func_positions = [positions['func_pos']] * len(input_ids)
    bb_positions = [positions['bb_pos']] * len(input_ids)
    
    # Convert to tensors
    input_tensor = torch.LongTensor([input_ids]).to(device)
    segment_tensor = torch.LongTensor([segment_labels]).to(device)
    binary_tensor = torch.FloatTensor([binary_positions]).to(device)
    func_tensor = torch.FloatTensor([func_positions]).to(device)
    bb_tensor = torch.FloatTensor([bb_positions]).to(device)
    
    # Get embeddings
    with torch.no_grad():
        output = model(input_tensor, segment_tensor, binary_tensor, func_tensor, bb_tensor)  # [1, seq_len, hidden]
        
        # Mean pooling (excluding padding)
        mask = (input_tensor > 0).float().unsqueeze(-1)  # [1, seq_len, 1]
        masked_output = output * mask
        sum_output = masked_output.sum(dim=1)  # [1, hidden]
        count = mask.sum(dim=1)  # [1, 1]
        embedding = sum_output / count  # [1, hidden]
    
    return embedding.cpu().numpy()[0]


def load_json_labels(json_path, max_samples=None):
    """Load instruction labels from JSON file."""
    print(f"[INFO] Loading labels from {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    instructions = data['instructions']
    
    if max_samples is not None:
        instructions = instructions[:max_samples]
    
    print(f"[INFO] Loaded {len(instructions)} instructions from {data['binary_name']}")
    return instructions, data


def balance_samples(instructions, bucket_key, num_buckets, samples_per_bucket=500):
    """Balance samples across buckets."""
    import random
    
    # Group by bucket
    buckets = defaultdict(list)
    for inst in instructions:
        bucket = inst['buckets'][bucket_key]
        buckets[bucket].append(inst)
    
    # Print distribution
    print(f"\n[INFO] Original distribution for {bucket_key}:")
    for i in range(num_buckets):
        print(f"  Bucket {i}: {len(buckets[i])} samples")
    
    # Balance
    balanced = []
    for i in range(num_buckets):
        bucket_samples = buckets[i]
        if len(bucket_samples) == 0:
            print(f"  [WARN] Bucket {i} has 0 samples, skipping")
            continue
        
        if len(bucket_samples) < samples_per_bucket:
            sampled = random.choices(bucket_samples, k=samples_per_bucket)
        else:
            sampled = random.sample(bucket_samples, samples_per_bucket)
        
        balanced.extend(sampled)
    
    print(f"[INFO] Balanced: {len(balanced)} samples")
    return balanced


def evaluate_bucket_task(model, instructions, vocab, bucket_key, model_type='baseline', device='cuda'):
    """
    Evaluate model on a bucket prediction task.
    
    Args:
        model: BERT model (baseline or address-aware)
        instructions: List of instruction dicts with labels
        vocab: Vocabulary
        bucket_key: 'bb_bucket', 'func_bucket', or 'binary_bucket'
        model_type: 'baseline' or 'addressaware'
        device: Device to use
    
    Returns:
        dict with results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {model_type.upper()} on {bucket_key.upper()}")
    print(f"{'='*80}")
    
    # Extract embeddings and labels
    embeddings = []
    labels = []
    
    print(f"[INFO] Extracting embeddings for {len(instructions)} instructions...")
    
    for i, inst in enumerate(instructions):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(instructions)}")
        
        # Tokenize
        token_ids = tokenize_instruction(inst['text'], vocab)
        
        # Get embedding
        if model_type == 'baseline':
            emb = get_embedding_baseline(model, token_ids, vocab, device)
        else:
            emb = get_embedding_addressaware(model, token_ids, inst['positions'], vocab, device)
        
        embeddings.append(emb)
        labels.append(inst['buckets'][bucket_key])
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    print(f"[INFO] Labels shape: {labels.shape}")
    
    # Split train/test (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"[INFO] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train logistic regression
    print(f"[INFO] Training logistic regression probe...")
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n[RESULT] {model_type.upper()} - {bucket_key.upper()}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return {
        'model_type': model_type,
        'bucket_key': bucket_key,
        'accuracy': accuracy,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'y_test': y_test,
        'y_pred': y_pred
    }


def main():
    parser = argparse.ArgumentParser(description='Comprehensive bucket prediction evaluation')
    parser.add_argument('--json_dir', type=str, default='data/json_labels',
                        help='Directory with JSON label files')
    parser.add_argument('--vocab', type=str, default='../pre-trained_model/palmtree/vocab',
                        help='Vocabulary file')
    parser.add_argument('--baseline_checkpoint', type=str, default='../addressaware/output_baseline_new/best_bert.pt',
                        help='Baseline model checkpoint')
    parser.add_argument('--addressaware_checkpoint', type=str, default='../addressaware/output_address_new/best_bert.pt',
                        help='Address-aware model checkpoint')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--attn_heads', type=int, default=8)
    parser.add_argument('--samples_per_bucket', type=int, default=2000,
                        help='Samples per bucket for balanced evaluation (default: 2000)')
    parser.add_argument('--binary_limit', type=int, default=5,
                        help='Number of binaries to evaluate (default: 5)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='results/bucket_comparison.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPREHENSIVE BUCKET PREDICTION EVALUATION")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Samples per bucket: {args.samples_per_bucket}")
    print(f"Binary limit: {args.binary_limit}")
    print("="*80)
    
    # Load vocabulary
    vocab = load_vocab(args.vocab)
    
    # Load models
    baseline_model = load_baseline_model(
        args.baseline_checkpoint, len(vocab), args.hidden, args.n_layers, args.attn_heads, args.device
    )
    
    addressaware_model = load_addressaware_model(
        args.addressaware_checkpoint, len(vocab), args.hidden, args.n_layers, args.attn_heads, args.device
    )
    
    # Find JSON files
    json_dir = Path(args.json_dir)
    json_files = sorted(json_dir.glob('*_instructions_labels.json'))
    
    if not json_files:
        print(f"[ERROR] No JSON files found in {json_dir}")
        return
    
    print(f"\n[INFO] Found {len(json_files)} JSON files")
    json_files = json_files[:args.binary_limit]
    print(f"[INFO] Loading {len(json_files)} binaries")
    
    # Collect all instructions from all binaries into one combined dataset
    print(f"\n{'='*80}")
    print("Combining instructions from all binaries")
    print(f"{'='*80}")
    
    all_instructions = []
    binary_names = []
    
    for json_file in json_files:
        instructions, data = load_json_labels(json_file)
        all_instructions.extend(instructions)
        binary_names.append(data['binary_name'])
        print(f"[INFO] Added {len(instructions)} instructions from {data['binary_name']}")
    
    print(f"\n[INFO] Total instructions across all binaries: {len(all_instructions)}")
    
    # Collect all results
    all_results = []
    
    # Evaluate on all three bucket tasks using combined dataset
    for bucket_key in ['bb_bucket', 'func_bucket', 'binary_bucket']:
        print(f"\n{'='*80}")
        print(f"Task: {bucket_key.upper()} (Combined dataset from {len(binary_names)} binaries)")
        print(f"{'='*80}")
        
        # Balance samples from combined dataset
        balanced = balance_samples(all_instructions, bucket_key, 5, args.samples_per_bucket)
        
        if len(balanced) == 0:
            print(f"[WARN] No samples for {bucket_key}, skipping")
            continue
            
        # Evaluate baseline
        result_baseline = evaluate_bucket_task(
            baseline_model, balanced, vocab, bucket_key, 'baseline', args.device
        )
        result_baseline['binary'] = f"combined_{len(binary_names)}_binaries"
        result_baseline['binary_list'] = binary_names
        all_results.append(result_baseline)
        
        # Evaluate address-aware
        result_addressaware = evaluate_bucket_task(
            addressaware_model, balanced, vocab, bucket_key, 'addressaware', args.device
        )
        result_addressaware['binary'] = f"combined_{len(binary_names)}_binaries"
        result_addressaware['binary_list'] = binary_names
        all_results.append(result_addressaware)
    
    # Print summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Evaluated on {len(binary_names)} binaries: {', '.join(binary_names)}")
    print(f"Combined dataset size: {len(all_instructions)} instructions")
    print(f"Samples per bucket: {args.samples_per_bucket}")
    print(f"{'='*80}\n")
    
    summary = defaultdict(lambda: {'baseline': [], 'addressaware': []})
    
    for result in all_results:
        bucket_key = result['bucket_key']
        model_type = result['model_type']
        summary[bucket_key][model_type].append(result['accuracy'])
    
    print(f"\nResults from combined dataset of {len(binary_names)} binaries:\n")
    
    for bucket_key in ['bb_bucket', 'func_bucket', 'binary_bucket']:
        if bucket_key not in summary:
            continue
        
        baseline_accs = summary[bucket_key]['baseline']
        addressaware_accs = summary[bucket_key]['addressaware']
        
        if not baseline_accs or not addressaware_accs:
            continue
        
        # Since we only evaluate once on combined data, just take the first result
        baseline_avg = baseline_accs[0]
        addressaware_avg = addressaware_accs[0]
        improvement = ((addressaware_avg - baseline_avg) / baseline_avg) * 100
        
        print(f"{bucket_key.upper()}:")
        print(f"  Baseline:       {baseline_avg:.4f} ({baseline_avg*100:.2f}%)")
        print(f"  Address-Aware:  {addressaware_avg:.4f} ({addressaware_avg*100:.2f}%)")
        print(f"  Improvement:    {improvement:+.2f}%")
        print()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON
    results_serializable = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k not in ['y_test', 'y_pred']}
        results_serializable.append(r_copy)
    
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                bucket_key: {
                    'baseline_avg': float(np.mean(summary[bucket_key]['baseline'])) if summary[bucket_key]['baseline'] else 0,
                    'addressaware_avg': float(np.mean(summary[bucket_key]['addressaware'])) if summary[bucket_key]['addressaware'] else 0,
                    'improvement_pct': float(((np.mean(summary[bucket_key]['addressaware']) - np.mean(summary[bucket_key]['baseline'])) / np.mean(summary[bucket_key]['baseline'])) * 100) if summary[bucket_key]['baseline'] else 0
                }
                for bucket_key in ['bb_bucket', 'func_bucket', 'binary_bucket'] if bucket_key in summary
            },
            'all_results': results_serializable
        }, f, indent=2)
    
    print(f"[INFO] Results saved to {output_path}")


if __name__ == '__main__':
    main()
