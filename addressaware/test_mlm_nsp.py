"""
Comprehensive Test Script for MLM and NSP Evaluation

Evaluates both:
1. Address-Aware BERT (trained model)
2. Baseline PalmTree (pre-trained model)

On tasks:
- MLM (Masked Language Modeling) - CFG only
- NSP_CFG (Next Sentence Prediction for CFG - order coherence)
- NSP_DFG (Next Sentence Prediction for DFG - trace membership)

Metrics:
- MLM: Loss, Perplexity, Top-1/Top-5 Accuracy
- NSP: Loss, Accuracy, Precision, Recall, F1
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from palmtree.dataset.vocab import WordVocab
from palmtree.model.bert import BERT as PalmTreeBERT
from dataloader_paired import PairedAddressAwareDataset
from model import AddressAwareBERT, AddressAwareBERTForPretraining


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
        """
        Update metrics.
        
        Args:
            loss: Scalar loss value
            predictions: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
        """
        # Only consider masked tokens (labels != -1)
        mask = labels != -1
        if mask.sum() == 0:
            return
        
        masked_preds = predictions[mask]  # [num_masked, vocab_size]
        masked_labels = labels[mask]  # [num_masked]
        
        # Update loss
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
                'mlm_loss': 0.0,
                'mlm_perplexity': float('inf'),
                'mlm_top1_acc': 0.0,
                'mlm_top5_acc': 0.0,
            }
        
        avg_loss = self.total_loss / self.total_tokens
        perplexity = np.exp(min(avg_loss, 50))  # Cap to avoid overflow
        top1_acc = self.top1_correct / self.masked_tokens
        top5_acc = self.top5_correct / self.masked_tokens
        
        return {
            'mlm_loss': avg_loss,
            'mlm_perplexity': perplexity,
            'mlm_top1_acc': top1_acc,
            'mlm_top5_acc': top5_acc,
        }


class NSPMetrics:
    """Metrics for Next Sentence Prediction."""
    
    def __init__(self, name='NSP'):
        self.name = name
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0
        self.tp = 0  # True Positives
        self.tn = 0  # True Negatives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
    
    def update(self, loss, predictions, labels):
        """
        Update metrics.
        
        Args:
            loss: Scalar loss value
            predictions: [batch_size, 2] - logits
            labels: [batch_size] - binary labels (0 or 1)
        """
        batch_size = labels.size(0)
        
        # Update loss
        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size
        
        # Get predictions
        pred_labels = torch.argmax(predictions, dim=-1)
        
        # Compute confusion matrix components
        self.tp += ((pred_labels == 1) & (labels == 1)).sum().item()
        self.tn += ((pred_labels == 0) & (labels == 0)).sum().item()
        self.fp += ((pred_labels == 1) & (labels == 0)).sum().item()
        self.fn += ((pred_labels == 0) & (labels == 1)).sum().item()
    
    def compute(self):
        """Compute final metrics."""
        if self.total_samples == 0:
            return {
                f'{self.name}_loss': 0.0,
                f'{self.name}_acc': 0.0,
                f'{self.name}_precision': 0.0,
                f'{self.name}_recall': 0.0,
                f'{self.name}_f1': 0.0,
            }
        
        avg_loss = self.total_loss / self.total_samples
        accuracy = (self.tp + self.tn) / self.total_samples
        
        # Precision: TP / (TP + FP)
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        
        # Recall: TP / (TP + FN)
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            f'{self.name}_loss': avg_loss,
            f'{self.name}_acc': accuracy,
            f'{self.name}_precision': precision,
            f'{self.name}_recall': recall,
            f'{self.name}_f1': f1,
        }


def evaluate_addressaware_model(model, dataloader, device):
    """
    Evaluate Address-Aware BERT model.
    
    Returns metrics for MLM, NSP_CFG, and NSP_DFG.
    """
    model.eval()
    
    mlm_metrics = MLMMetrics()
    nsp_cfg_metrics = NSPMetrics(name='nsp_cfg')
    nsp_dfg_metrics = NSPMetrics(name='nsp_dfg')
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    nsp_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # === CFG Forward (MLM + NSP) ===
            cfg_mlm_output, cfg_nsp_output = model(
                batch['cfg_bert_input'],
                batch['cfg_segment_label'],
                batch['cfg_binary_pos'],
                batch['cfg_function_pos'],
                batch['cfg_bb_pos'],
                corpus_type='cfg'
            )
            
            # CFG MLM
            mlm_loss = mlm_criterion(cfg_mlm_output.transpose(1, 2), batch['cfg_bert_label'])
            mlm_metrics.update(mlm_loss, cfg_mlm_output, batch['cfg_bert_label'])
            
            # CFG NSP
            cfg_nsp_loss = nsp_criterion(cfg_nsp_output, batch['cfg_is_next'])
            nsp_cfg_metrics.update(cfg_nsp_loss, cfg_nsp_output, batch['cfg_is_next'])
            
            # === DFG Forward (NSP only) ===
            _, dfg_nsp_output = model(
                batch['dfg_bert_input'],
                batch['dfg_segment_label'],
                batch['dfg_binary_pos'],
                batch['dfg_function_pos'],
                batch['dfg_bb_pos'],
                corpus_type='dfg'
            )
            
            # DFG NSP
            dfg_nsp_loss = nsp_criterion(dfg_nsp_output, batch['dfg_is_next'])
            nsp_dfg_metrics.update(dfg_nsp_loss, dfg_nsp_output, batch['dfg_is_next'])
    
    # Compute all metrics
    results = {}
    results.update(mlm_metrics.compute())
    results.update(nsp_cfg_metrics.compute())
    results.update(nsp_dfg_metrics.compute())
    
    # Compute total loss (same as training)
    results['total_loss'] = results['mlm_loss'] + results['nsp_cfg_loss'] + results['nsp_dfg_loss']
    
    return results


def evaluate_palmtree_model(model, dataloader, device, vocab_size):
    """
    Evaluate baseline PalmTree model.
    
    IMPORTANT: Uses the SAME token sequences as address-aware model,
    but ignores address information (uses only token + segment + sequential position).
    This ensures a fair comparison on identical inputs.
    """
    model.eval()
    
    mlm_metrics = MLMMetrics()
    nsp_cfg_metrics = NSPMetrics(name='nsp_cfg')
    nsp_dfg_metrics = NSPMetrics(name='nsp_dfg')
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    nsp_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # Create MLM and NSP heads for PalmTree
    # NOTE: These are randomly initialized, which is a limitation.
    # For truly fair comparison, you'd want to train PalmTree with these heads.
    mlm_head = nn.Sequential(
        nn.Linear(model.hidden, model.hidden),
        nn.GELU(),
        nn.LayerNorm(model.hidden),
        nn.Linear(model.hidden, vocab_size)
    ).to(device)
    
    nsp_cfg_head = nn.Sequential(
        nn.Linear(model.hidden, model.hidden),
        nn.Tanh(),
        nn.Linear(model.hidden, 2)
    ).to(device)
    
    nsp_dfg_head = nn.Sequential(
        nn.Linear(model.hidden, model.hidden),
        nn.Tanh(),
        nn.Linear(model.hidden, 2)
    ).to(device)
    
    print("[NOTE] PalmTree uses SAME token sequences but NO address information")
    print("[WARNING] PalmTree checkpoint does NOT contain trained MLM/NSP heads!")
    print("[WARNING] Creating randomly initialized heads - comparison is UNFAIR")
    print("[WARNING] PalmTree saves only the BERT encoder, not the BERTLM wrapper")
    print("[INFO] For fair comparison, you should:")
    print("       1. Train PalmTree baseline with same task heads, OR")
    print("       2. Compare encoder representations instead of task performance")
    print()    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PalmTree"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # === CFG Forward (MLM + NSP) ===
            # PalmTree BERT forward: (input, segment_label)
            # Uses SAME tokens as address-aware model, but NO address positions
            cfg_output = model.forward(batch['cfg_bert_input'], batch['cfg_segment_label'])
            
            # CFG MLM - SAME masked tokens as address-aware model
            cfg_mlm_output = mlm_head(cfg_output)
            mlm_loss = mlm_criterion(cfg_mlm_output.transpose(1, 2), batch['cfg_bert_label'])
            mlm_metrics.update(mlm_loss, cfg_mlm_output, batch['cfg_bert_label'])
            
            # CFG NSP - SAME NSP labels as address-aware model
            cfg_nsp_output = nsp_cfg_head(cfg_output[:, 0, :])
            cfg_nsp_loss = nsp_criterion(cfg_nsp_output, batch['cfg_is_next'])
            nsp_cfg_metrics.update(cfg_nsp_loss, cfg_nsp_output, batch['cfg_is_next'])
            
            # === DFG Forward (NSP only) ===
            # Uses SAME tokens as address-aware model, but NO address positions
            dfg_output = model.forward(batch['dfg_bert_input'], batch['dfg_segment_label'])
            
            # DFG NSP - SAME NSP labels as address-aware model
            dfg_nsp_output = nsp_dfg_head(dfg_output[:, 0, :])
            dfg_nsp_loss = nsp_criterion(dfg_nsp_output, batch['dfg_is_next'])
            nsp_dfg_metrics.update(dfg_nsp_loss, dfg_nsp_output, batch['dfg_is_next'])
    
    # Compute all metrics
    results = {}
    results.update(mlm_metrics.compute())
    results.update(nsp_cfg_metrics.compute())
    results.update(nsp_dfg_metrics.compute())
    
    # Compute total loss
    results['total_loss'] = results['mlm_loss'] + results['nsp_cfg_loss'] + results['nsp_dfg_loss']
    
    return results


def load_addressaware_checkpoint(checkpoint_path, vocab_size, hidden, n_layers, attn_heads, dropout, max_len, device):
    """Load address-aware model from checkpoint."""
    print(f"Loading address-aware model from: {checkpoint_path}")
    
    # Load checkpoint first
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel wrapper (remove 'module.' prefix if present)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Check if position_embedding is learnable (has .weight) or sinusoidal (has .pe)
    has_learnable_pos = 'bert.embedding.position_embedding.weight' in new_state_dict
    
    if has_learnable_pos:
        # Extract the position embedding from checkpoint and pass it as pretrained
        print("[INFO] Checkpoint uses learnable position embeddings")
        pretrained_pos_emb = new_state_dict['bert.embedding.position_embedding.weight']
        
        # Use the actual max_len from the checkpoint (not the arg max_len)
        checkpoint_max_len = pretrained_pos_emb.size(0)
        print(f"[INFO] Using checkpoint max_len: {checkpoint_max_len}")
        
        bert = AddressAwareBERT(
            vocab_size=vocab_size,
            hidden=hidden,
            n_layers=n_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            max_len=checkpoint_max_len,  # Use checkpoint's max_len
            pretrained_position_emb=pretrained_pos_emb  # This triggers nn.Embedding creation
        )
    else:
        # Model uses sinusoidal position embeddings
        print("[INFO] Checkpoint uses sinusoidal position embeddings")
        bert = AddressAwareBERT(
            vocab_size=vocab_size,
            hidden=hidden,
            n_layers=n_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            max_len=max_len
        )
    
    model = AddressAwareBERTForPretraining(bert, vocab_size)
    
    # Load the cleaned state dict
    model.load_state_dict(new_state_dict)
    
    model = model.to(device)
    model.eval()
    
    return model


def load_palmtree_checkpoint(checkpoint_path, device):
    """Load PalmTree baseline model from checkpoint."""
    print(f"Loading PalmTree model from: {checkpoint_path}")
    
    # Load full model object (PalmTree saves the model, not state_dict)
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if hasattr(model, 'bert'):
        # It's a BERTLM wrapper, extract BERT
        model = model.bert
    
    model = model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Test MLM and NSP for both models")
    
    # Data paths
    parser.add_argument('--cfg_test', required=True, help='Path to CFG test data')
    parser.add_argument('--dfg_test', required=True, help='Path to DFG test data')
    parser.add_argument('--vocab', required=True, help='Path to vocabulary file')
    
    # Model paths
    parser.add_argument('--addressaware_checkpoint', help='Path to address-aware model checkpoint')
    parser.add_argument('--palmtree_checkpoint', help='Path to PalmTree baseline checkpoint')
    
    # Model config (for address-aware model)
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--attn_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    
    # Evaluation config
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--output_dir', default='test_results', help='Output directory for results')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print(f"Loading vocabulary from: {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = PairedAddressAwareDataset(
        cfg_corpus_path=args.cfg_test,
        dfg_corpus_path=args.dfg_test,
        vocab=vocab,
        seq_len=args.seq_len,
        on_memory=True,
        nsp_prob=0.5,
        mask_prob=0.15,
        data_percentage=1.0,  # Use all test data
        train_split=1.0,  # No split for test data
        is_train=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Test dataset: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    # === Evaluate Address-Aware Model ===
    if args.addressaware_checkpoint:
        print("\n" + "="*70)
        print("Evaluating Address-Aware BERT")
        print("="*70)
        
        model = load_addressaware_checkpoint(
            args.addressaware_checkpoint,
            vocab_size=vocab_size,
            hidden=args.hidden,
            n_layers=args.n_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
            max_len=args.seq_len * 12,  # Conservative max_len
            device=device
        )
        
        addressaware_results = evaluate_addressaware_model(model, test_loader, device)
        all_results['address_aware'] = addressaware_results
        
        print("\nAddress-Aware BERT Results:")
        print(f"  Total Loss: {addressaware_results['total_loss']:.4f}")
        print(f"  MLM Loss: {addressaware_results['mlm_loss']:.4f}")
        print(f"  MLM Perplexity: {addressaware_results['mlm_perplexity']:.4f}")
        print(f"  MLM Top-1 Acc: {addressaware_results['mlm_top1_acc']:.4f}")
        print(f"  MLM Top-5 Acc: {addressaware_results['mlm_top5_acc']:.4f}")
        print(f"  NSP_CFG Acc: {addressaware_results['nsp_cfg_acc']:.4f}")
        print(f"  NSP_CFG F1: {addressaware_results['nsp_cfg_f1']:.4f}")
        print(f"  NSP_DFG Acc: {addressaware_results['nsp_dfg_acc']:.4f}")
        print(f"  NSP_DFG F1: {addressaware_results['nsp_dfg_f1']:.4f}")
    
    # === Evaluate PalmTree Baseline ===
    if args.palmtree_checkpoint:
        print("\n" + "="*70)
        print("Evaluating PalmTree Baseline")
        print("="*70)
        
        palmtree_model = load_palmtree_checkpoint(args.palmtree_checkpoint, device)
        
        palmtree_results = evaluate_palmtree_model(palmtree_model, test_loader, device, vocab_size)
        all_results['palmtree_baseline'] = palmtree_results
        
        print("\nPalmTree Baseline Results:")
        print(f"  Total Loss: {palmtree_results['total_loss']:.4f}")
        print(f"  MLM Loss: {palmtree_results['mlm_loss']:.4f}")
        print(f"  MLM Perplexity: {palmtree_results['mlm_perplexity']:.4f}")
        print(f"  MLM Top-1 Acc: {palmtree_results['mlm_top1_acc']:.4f}")
        print(f"  MLM Top-5 Acc: {palmtree_results['mlm_top5_acc']:.4f}")
        print(f"  NSP_CFG Acc: {palmtree_results['nsp_cfg_acc']:.4f}")
        print(f"  NSP_CFG F1: {palmtree_results['nsp_cfg_f1']:.4f}")
        print(f"  NSP_DFG Acc: {palmtree_results['nsp_dfg_acc']:.4f}")
        print(f"  NSP_DFG F1: {palmtree_results['nsp_dfg_f1']:.4f}")
    
    # === Save Results ===
    results_file = os.path.join(args.output_dir, 'comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'results': all_results,
            'config': {
                'cfg_test': args.cfg_test,
                'dfg_test': args.dfg_test,
                'vocab': args.vocab,
                'batch_size': args.batch_size,
                'seq_len': args.seq_len,
                'hidden': args.hidden,
                'n_layers': args.n_layers,
                'attn_heads': args.attn_heads,
            }
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")
    
    # === Print Comparison ===
    if len(all_results) == 2:
        print("\n" + "="*70)
        print("FAIR COMPARISON: Address-Aware vs PalmTree Baseline")
        print("="*70)
        print("IMPORTANT: Both models evaluated on IDENTICAL token sequences")
        print("  - Address-Aware: Uses token + segment + ADDRESS positions")
        print("  - PalmTree:      Uses token + segment + SEQUENTIAL positions only")
        print("  - Difference shows the impact of address-aware embeddings")
        print("="*70)
        
        aa_res = all_results['address_aware']
        pt_res = all_results['palmtree_baseline']
        
        metrics = [
            ('Total Loss', 'total_loss', 'lower'),
            ('MLM Loss', 'mlm_loss', 'lower'),
            ('MLM Perplexity', 'mlm_perplexity', 'lower'),
            ('MLM Top-1 Acc', 'mlm_top1_acc', 'higher'),
            ('MLM Top-5 Acc', 'mlm_top5_acc', 'higher'),
            ('NSP_CFG Acc', 'nsp_cfg_acc', 'higher'),
            ('NSP_CFG F1', 'nsp_cfg_f1', 'higher'),
            ('NSP_DFG Acc', 'nsp_dfg_acc', 'higher'),
            ('NSP_DFG F1', 'nsp_dfg_f1', 'higher'),
        ]
        
        print(f"\n{'Metric':<20} {'Address-Aware':<15} {'PalmTree':<15} {'Improvement':<15} {'Winner'}")
        print("-" * 80)
        
        for metric_name, metric_key, better in metrics:
            aa_val = aa_res[metric_key]
            pt_val = pt_res[metric_key]
            diff = aa_val - pt_val
            
            if better == 'lower':
                winner = 'Address-Aware ✓' if aa_val < pt_val else 'PalmTree'
                improvement = ((pt_val - aa_val) / pt_val * 100) if pt_val != 0 else 0
                improvement_str = f"{improvement:+.2f}%"
            else:
                winner = 'Address-Aware ✓' if aa_val > pt_val else 'PalmTree'
                improvement = ((aa_val - pt_val) / pt_val * 100) if pt_val != 0 else 0
                improvement_str = f"{improvement:+.2f}%"
            
            print(f"{metric_name:<20} {aa_val:<15.4f} {pt_val:<15.4f} {improvement_str:<15} {winner}")
        
        print("="*70)
        print("NOTE: Positive improvement % means Address-Aware performs better")
        print("="*70)
        print("\n⚠️  WARNING: This comparison is UNFAIR!")
        print("   - Address-Aware: Trained encoder + Trained MLM/NSP heads")
        print("   - PalmTree: Pre-trained encoder + RANDOM MLM/NSP heads")
        print("   - PalmTree checkpoint does NOT save the trained task heads")
        print("\n   The huge performance gap is mainly due to random vs trained heads,")
        print("   NOT due to address-aware embeddings.")
        print("\n   For fair comparison, you should:")
        print("   1. Train a baseline model (without address) on same data with same heads")
        print("   2. Compare encoder representations on downstream tasks")
        print("   3. Or compare only the encoder embeddings (cosine similarity, clustering)")
        print("="*70)


if __name__ == "__main__":
    main()
