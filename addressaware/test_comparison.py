"""
Fair Comparison Test Script

Compares:
1. Address-Aware BERT (with 3-level address embeddings)
2. Baseline BERT (with sequential sinusoidal positions)

Both trained on SAME data with SAME hyperparameters.
Only difference: Address embeddings vs sequential positions.

Metrics:
- MLM: Loss, Perplexity, Top-1/Top-5 Accuracy (CFG only)
- NSP_CFG: Loss, Accuracy, Precision, Recall, F1
- NSP_DFG: Loss, Accuracy, Precision, Recall, F1
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
import numpy as np
import random
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from palmtree.dataset.vocab import WordVocab
from dataloader_paired import PairedAddressAwareDataset
from dataloader_baseline import PairedBaselineDataset
from model import AddressAwareBERTForPretraining, AddressAwareBERT
from model_baseline import create_baseline_model


class MLMMetrics:
    """Metrics for Masked Language Modeling."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0
        self.top1_correct = 0
        self.top5_correct = 0
        self.masked_tokens = 0
    
    def update(self, loss, predictions, labels):
        """Update metrics (only for masked tokens where labels != -1)."""
        mask = labels != -1
        if mask.sum() == 0:
            return
        
        masked_preds = predictions[mask]
        masked_labels = labels[mask]
        
        self.total_loss += loss.item() * mask.sum().item()
        self.total_tokens += mask.sum().item()
        
        # Top-1 accuracy
        top1_pred = torch.argmax(masked_preds, dim=-1)
        self.top1_correct += (top1_pred == masked_labels).sum().item()
        
        # Top-5 accuracy
        top5_pred = torch.topk(masked_preds, k=min(5, masked_preds.size(-1)), dim=-1)[1]
        top5_match = (top5_pred == masked_labels.unsqueeze(-1)).any(dim=-1)
        self.top5_correct += top5_match.sum().item()
        
        self.masked_tokens += mask.sum().item()
    
    def compute(self):
        """Compute final metrics."""
        if self.masked_tokens == 0:
            return {
                'loss': 0.0,
                'perplexity': float('inf'),
                'top1_acc': 0.0,
                'top5_acc': 0.0,
            }
        
        avg_loss = self.total_loss / self.total_tokens
        perplexity = np.exp(min(avg_loss, 50))
        top1_acc = self.top1_correct / self.masked_tokens
        top5_acc = self.top5_correct / self.masked_tokens
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
        }


class NSPMetrics:
    """Metrics for Next Sentence Prediction."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
    
    def update(self, loss, predictions, labels):
        """Update metrics."""
        pred_labels = torch.argmax(predictions, dim=-1)
        
        self.total_loss += loss.item() * len(labels)
        self.total_samples += len(labels)
        
        # Binary classification metrics
        for pred, true in zip(pred_labels, labels):
            if pred == 1 and true == 1:
                self.true_positives += 1
            elif pred == 1 and true == 0:
                self.false_positives += 1
            elif pred == 0 and true == 0:
                self.true_negatives += 1
            elif pred == 0 and true == 1:
                self.false_negatives += 1
    
    def compute(self):
        """Compute final metrics."""
        if self.total_samples == 0:
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }
        
        avg_loss = self.total_loss / self.total_samples
        accuracy = (self.true_positives + self.true_negatives) / self.total_samples
        
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0.0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }


def evaluate_addressaware_model(model, data_loader, device):
    """Evaluate address-aware model on test set."""
    model.eval()
    
    mlm_metrics = MLMMetrics()
    nsp_cfg_metrics = NSPMetrics()
    nsp_dfg_metrics = NSPMetrics()
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    nsp_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Address-Aware"):
            # CFG data
            cfg_input = batch['cfg_bert_input'].to(device)
            cfg_segment = batch['cfg_segment_label'].to(device)
            cfg_binary_pos = batch['cfg_binary_pos'].to(device)
            cfg_function_pos = batch['cfg_function_pos'].to(device)
            cfg_bb_pos = batch['cfg_bb_pos'].to(device)
            cfg_mlm_labels = batch['cfg_bert_label'].to(device)
            cfg_nsp_labels = batch['cfg_is_next'].to(device)
            
            # DFG data
            dfg_input = batch['dfg_bert_input'].to(device)
            dfg_segment = batch['dfg_segment_label'].to(device)
            dfg_binary_pos = batch['dfg_binary_pos'].to(device)
            dfg_function_pos = batch['dfg_function_pos'].to(device)
            dfg_bb_pos = batch['dfg_bb_pos'].to(device)
            dfg_nsp_labels = batch['dfg_is_next'].to(device)
            
            # Forward pass for CFG (MLM + NSP)
            cfg_mlm_output, cfg_nsp_output = model(
                cfg_input, cfg_segment,
                cfg_binary_pos, cfg_function_pos, cfg_bb_pos,
                corpus_type='cfg'
            )
            
            # Forward pass for DFG (NSP only)
            _, dfg_nsp_output = model(
                dfg_input, dfg_segment,
                dfg_binary_pos, dfg_function_pos, dfg_bb_pos,
                corpus_type='dfg'
            )
            
            # Calculate losses
            mlm_loss = mlm_criterion(cfg_mlm_output.transpose(1, 2), cfg_mlm_labels)
            nsp_cfg_loss = nsp_criterion(cfg_nsp_output, cfg_nsp_labels)
            nsp_dfg_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels)
            
            # Update metrics
            mlm_metrics.update(mlm_loss, cfg_mlm_output, cfg_mlm_labels)
            nsp_cfg_metrics.update(nsp_cfg_loss, cfg_nsp_output, cfg_nsp_labels)
            nsp_dfg_metrics.update(nsp_dfg_loss, dfg_nsp_output, dfg_nsp_labels)
    
    # Compute final metrics
    mlm_results = mlm_metrics.compute()
    nsp_cfg_results = nsp_cfg_metrics.compute()
    nsp_dfg_results = nsp_dfg_metrics.compute()
    
    return {
        'mlm': mlm_results,
        'nsp_cfg': nsp_cfg_results,
        'nsp_dfg': nsp_dfg_results,
        'total_loss': mlm_results['loss'] + nsp_cfg_results['loss'] + nsp_dfg_results['loss']
    }


def evaluate_baseline_model(model, data_loader, device):
    """Evaluate baseline model on test set."""
    model.eval()
    
    mlm_metrics = MLMMetrics()
    nsp_cfg_metrics = NSPMetrics()
    nsp_dfg_metrics = NSPMetrics()
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    nsp_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Baseline"):
            # CFG data
            cfg_input = batch['bert_input'].to(device)
            cfg_segment = batch['segment_label'].to(device)
            cfg_mlm_labels = batch['bert_label'].to(device)
            cfg_nsp_labels = batch['is_next'].to(device)
            
            # DFG data
            dfg_input = batch['dfg_input'].to(device)
            dfg_segment = batch['dfg_segment'].to(device)
            dfg_nsp_labels = batch['dfg_is_next'].to(device)
            
            # Forward pass
            mlm_output, cwp_output, dup_output = model(
                cfg_input, cfg_segment,
                dfg_input, dfg_segment
            )
            
            # Calculate losses
            mlm_loss = mlm_criterion(mlm_output.transpose(1, 2), cfg_mlm_labels)
            nsp_cfg_loss = nsp_criterion(cwp_output, cfg_nsp_labels)
            nsp_dfg_loss = nsp_criterion(dup_output, dfg_nsp_labels)
            
            # Update metrics
            mlm_metrics.update(mlm_loss, mlm_output, cfg_mlm_labels)
            nsp_cfg_metrics.update(nsp_cfg_loss, cwp_output, cfg_nsp_labels)
            nsp_dfg_metrics.update(nsp_dfg_loss, dup_output, dfg_nsp_labels)
    
    # Compute final metrics
    mlm_results = mlm_metrics.compute()
    nsp_cfg_results = nsp_cfg_metrics.compute()
    nsp_dfg_results = nsp_dfg_metrics.compute()
    
    return {
        'mlm': mlm_results,
        'nsp_cfg': nsp_cfg_results,
        'nsp_dfg': nsp_dfg_results,
        'total_loss': mlm_results['loss'] + nsp_cfg_results['loss'] + nsp_dfg_results['loss']
    }


def load_addressaware_checkpoint(checkpoint_path, vocab_size, hidden, n_layers, attn_heads, dropout, max_len, device):
    """Load address-aware model from checkpoint."""
    print(f"[Address-Aware] Loading from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Remove DataParallel prefix if exists
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    
    # Check if model has learnable position embeddings
    has_learnable_pos = 'bert.embedding.position_embedding.weight' in new_state_dict
    
    if has_learnable_pos:
        print("[Address-Aware] Using learnable position embeddings")
        pretrained_pos_emb = new_state_dict['bert.embedding.position_embedding.weight']
        checkpoint_max_len = pretrained_pos_emb.size(0)
        print(f"[Address-Aware] Max length: {checkpoint_max_len}")
        
        bert = AddressAwareBERT(
            vocab_size=vocab_size,
            hidden=hidden,
            n_layers=n_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            max_len=checkpoint_max_len,
            pretrained_position_emb=pretrained_pos_emb
        )
    else:
        bert = AddressAwareBERT(
            vocab_size=vocab_size,
            hidden=hidden,
            n_layers=n_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            max_len=max_len
        )
    
    model = AddressAwareBERTForPretraining(bert, vocab_size)
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def load_baseline_checkpoint(checkpoint_path, vocab_size, hidden, n_layers, attn_heads, dropout, device):
    """Load baseline model from checkpoint."""
    print(f"[Baseline] Loading from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Remove DataParallel prefix if exists
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    
    # Create baseline model
    model = create_baseline_model(
        vocab_size=vocab_size,
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads,
        dropout=dropout
    )
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Fair comparison between Address-Aware and Baseline models")
    
    # Data paths
    parser.add_argument('--cfg_test', required=True, help='Path to CFG test data')
    parser.add_argument('--dfg_test', required=True, help='Path to DFG test data')
    parser.add_argument('--vocab', required=True, help='Path to vocabulary file')
    
    # Model checkpoints
    parser.add_argument('--addressaware_checkpoint', help='Path to address-aware model checkpoint')
    parser.add_argument('--baseline_checkpoint', help='Path to baseline model checkpoint')
    
    # Model config
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--attn_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    
    # Evaluation config
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--output_file', default='comparison_results.json', help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible masking')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Set random seeds for reproducible masking
    print(f"Setting random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print("✓ Random seeds set for reproducible masking\n")
    
    # Load vocabulary
    print(f"Loading vocabulary from: {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}\n")
    
    results = {}
    
    # === Evaluate Address-Aware Model ===
    if args.addressaware_checkpoint:
        print("="*80)
        print("EVALUATING ADDRESS-AWARE MODEL")
        print("="*80)
        
        # Reset random seed for reproducible masking
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Create test dataset with address info
        print("Creating test dataset (with address positions)...")
        addressaware_dataset = PairedAddressAwareDataset(
            cfg_corpus_path=args.cfg_test,
            dfg_corpus_path=args.dfg_test,
            vocab=vocab,
            seq_len=args.seq_len,
            on_memory=True,
            nsp_prob=0.5,
            mask_prob=0.15,
            data_percentage=1.0,
            train_split=1.0,
            is_train=True
        )
        
        addressaware_loader = DataLoader(
            addressaware_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(f"Test dataset: {len(addressaware_dataset)} samples\n")
        
        # Load model
        model = load_addressaware_checkpoint(
            args.addressaware_checkpoint,
            vocab_size, args.hidden, args.n_layers,
            args.attn_heads, args.dropout, args.seq_len * 12, device
        )
        
        # Evaluate
        addressaware_results = evaluate_addressaware_model(model, addressaware_loader, device)
        results['address_aware'] = addressaware_results
        
        print("\n" + "-"*80)
        print("ADDRESS-AWARE RESULTS:")
        print("-"*80)
        print(f"Total Loss:     {addressaware_results['total_loss']:.4f}")
        print(f"\nMLM (CFG):")
        print(f"  Loss:         {addressaware_results['mlm']['loss']:.4f}")
        print(f"  Perplexity:   {addressaware_results['mlm']['perplexity']:.4f}")
        print(f"  Top-1 Acc:    {addressaware_results['mlm']['top1_acc']:.4f} ({addressaware_results['mlm']['top1_acc']*100:.2f}%)")
        print(f"  Top-5 Acc:    {addressaware_results['mlm']['top5_acc']:.4f} ({addressaware_results['mlm']['top5_acc']*100:.2f}%)")
        print(f"\nNSP_CFG:")
        print(f"  Loss:         {addressaware_results['nsp_cfg']['loss']:.4f}")
        print(f"  Accuracy:     {addressaware_results['nsp_cfg']['accuracy']:.4f} ({addressaware_results['nsp_cfg']['accuracy']*100:.2f}%)")
        print(f"  Precision:    {addressaware_results['nsp_cfg']['precision']:.4f}")
        print(f"  Recall:       {addressaware_results['nsp_cfg']['recall']:.4f}")
        print(f"  F1:           {addressaware_results['nsp_cfg']['f1']:.4f}")
        print(f"\nNSP_DFG:")
        print(f"  Loss:         {addressaware_results['nsp_dfg']['loss']:.4f}")
        print(f"  Accuracy:     {addressaware_results['nsp_dfg']['accuracy']:.4f} ({addressaware_results['nsp_dfg']['accuracy']*100:.2f}%)")
        print(f"  Precision:    {addressaware_results['nsp_dfg']['precision']:.4f}")
        print(f"  Recall:       {addressaware_results['nsp_dfg']['recall']:.4f}")
        print(f"  F1:           {addressaware_results['nsp_dfg']['f1']:.4f}")
        print("="*80 + "\n")
    
    # === Evaluate Baseline Model ===
    if args.baseline_checkpoint:
        print("="*80)
        print("EVALUATING BASELINE MODEL")
        print("="*80)
        
        # Reset random seed for reproducible masking (SAME SEED as address-aware!)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Create test dataset WITHOUT address info
        print("Creating test dataset (sequential positions only)...")
        baseline_dataset = PairedBaselineDataset(
            cfg_corpus_path=args.cfg_test,
            dfg_corpus_path=args.dfg_test,
            vocab=vocab,
            seq_len=args.seq_len,
            on_memory=True,
            nsp_prob=0.5,
            mask_prob=0.15,
            data_percentage=1.0,
            train_split=1.0,
            is_train=True
        )
        
        baseline_loader = DataLoader(
            baseline_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(f"Test dataset: {len(baseline_dataset)} samples\n")
        
        # Load model
        model = load_baseline_checkpoint(
            args.baseline_checkpoint,
            vocab_size, args.hidden, args.n_layers,
            args.attn_heads, args.dropout, device
        )
        
        # Evaluate
        baseline_results = evaluate_baseline_model(model, baseline_loader, device)
        results['baseline'] = baseline_results
        
        print("\n" + "-"*80)
        print("BASELINE RESULTS:")
        print("-"*80)
        print(f"Total Loss:     {baseline_results['total_loss']:.4f}")
        print(f"\nMLM (CFG):")
        print(f"  Loss:         {baseline_results['mlm']['loss']:.4f}")
        print(f"  Perplexity:   {baseline_results['mlm']['perplexity']:.4f}")
        print(f"  Top-1 Acc:    {baseline_results['mlm']['top1_acc']:.4f} ({baseline_results['mlm']['top1_acc']*100:.2f}%)")
        print(f"  Top-5 Acc:    {baseline_results['mlm']['top5_acc']:.4f} ({baseline_results['mlm']['top5_acc']*100:.2f}%)")
        print(f"\nNSP_CFG:")
        print(f"  Loss:         {baseline_results['nsp_cfg']['loss']:.4f}")
        print(f"  Accuracy:     {baseline_results['nsp_cfg']['accuracy']:.4f} ({baseline_results['nsp_cfg']['accuracy']*100:.2f}%)")
        print(f"  Precision:    {baseline_results['nsp_cfg']['precision']:.4f}")
        print(f"  Recall:       {baseline_results['nsp_cfg']['recall']:.4f}")
        print(f"  F1:           {baseline_results['nsp_cfg']['f1']:.4f}")
        print(f"\nNSP_DFG:")
        print(f"  Loss:         {baseline_results['nsp_dfg']['loss']:.4f}")
        print(f"  Accuracy:     {baseline_results['nsp_dfg']['accuracy']:.4f} ({baseline_results['nsp_dfg']['accuracy']*100:.2f}%)")
        print(f"  Precision:    {baseline_results['nsp_dfg']['precision']:.4f}")
        print(f"  Recall:       {baseline_results['nsp_dfg']['recall']:.4f}")
        print(f"  F1:           {baseline_results['nsp_dfg']['f1']:.4f}")
        print("="*80 + "\n")
    
    # === Comparison Summary ===
    if 'address_aware' in results and 'baseline' in results:
        print("="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"Note: Both models evaluated with SAME random seed ({args.seed})")
        print("      → Same tokens masked at same positions (fair comparison)")
        print("")
        
        aa = results['address_aware']
        bl = results['baseline']
        
        print(f"\nMLM Performance:")
        print(f"  Address-Aware Top-1: {aa['mlm']['top1_acc']*100:.2f}%  |  Baseline Top-1: {bl['mlm']['top1_acc']*100:.2f}%  |  Δ: {(aa['mlm']['top1_acc']-bl['mlm']['top1_acc'])*100:+.2f}%")
        print(f"  Address-Aware Top-5: {aa['mlm']['top5_acc']*100:.2f}%  |  Baseline Top-5: {bl['mlm']['top5_acc']*100:.2f}%  |  Δ: {(aa['mlm']['top5_acc']-bl['mlm']['top5_acc'])*100:+.2f}%")
        print(f"  Address-Aware PPL:   {aa['mlm']['perplexity']:.2f}  |  Baseline PPL:   {bl['mlm']['perplexity']:.2f}  |  Δ: {aa['mlm']['perplexity']-bl['mlm']['perplexity']:+.2f}")
        
        print(f"\nNSP_CFG Performance:")
        print(f"  Address-Aware Acc:   {aa['nsp_cfg']['accuracy']*100:.2f}%  |  Baseline Acc:   {bl['nsp_cfg']['accuracy']*100:.2f}%  |  Δ: {(aa['nsp_cfg']['accuracy']-bl['nsp_cfg']['accuracy'])*100:+.2f}%")
        print(f"  Address-Aware F1:    {aa['nsp_cfg']['f1']:.4f}  |  Baseline F1:    {bl['nsp_cfg']['f1']:.4f}  |  Δ: {aa['nsp_cfg']['f1']-bl['nsp_cfg']['f1']:+.4f}")
        
        print(f"\nNSP_DFG Performance:")
        print(f"  Address-Aware Acc:   {aa['nsp_dfg']['accuracy']*100:.2f}%  |  Baseline Acc:   {bl['nsp_dfg']['accuracy']*100:.2f}%  |  Δ: {(aa['nsp_dfg']['accuracy']-bl['nsp_dfg']['accuracy'])*100:+.2f}%")
        print(f"  Address-Aware F1:    {aa['nsp_dfg']['f1']:.4f}  |  Baseline F1:    {bl['nsp_dfg']['f1']:.4f}  |  Δ: {aa['nsp_dfg']['f1']-bl['nsp_dfg']['f1']:+.4f}")
        
        print(f"\nOverall:")
        print(f"  Address-Aware Loss:  {aa['total_loss']:.4f}  |  Baseline Loss:  {bl['total_loss']:.4f}  |  Δ: {aa['total_loss']-bl['total_loss']:+.4f}")
        
        # Determine winner
        aa_wins = 0
        bl_wins = 0
        
        if aa['mlm']['top1_acc'] > bl['mlm']['top1_acc']:
            aa_wins += 1
        else:
            bl_wins += 1
            
        if aa['nsp_cfg']['accuracy'] > bl['nsp_cfg']['accuracy']:
            aa_wins += 1
        else:
            bl_wins += 1
            
        if aa['nsp_dfg']['accuracy'] > bl['nsp_dfg']['accuracy']:
            aa_wins += 1
        else:
            bl_wins += 1
        
        print(f"\n{'='*80}")
        if aa_wins > bl_wins:
            print("✓ Address-Aware model WINS! Address embeddings improve performance.")
        elif bl_wins > aa_wins:
            print("✗ Baseline model WINS. Address embeddings may not help or hurt performance.")
        else:
            print("≈ TIE. Address embeddings have mixed impact.")
        print("="*80)
    
    # Save results to JSON
    print(f"\nSaving results to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()
