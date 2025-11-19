"""
Position Probing Experiment

Probe the learned embeddings from address-aware and baseline models
to evaluate how well they encode positional information at different levels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json
import os
import sys
from tqdm import tqdm
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from addressaware.dataloader_paired import PairedAddressAwareDataset
from addressaware.model import AddressAwareBERT
from addressaware.model_baseline import BaselineBERT
from pre_trained_model.vocab import WordVocab
from position_predictor import MultiLevelPositionProbe


class ProbingMetrics:
    """Track metrics for position prediction."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_samples = 0
        self.binary_mse = 0.0
        self.binary_mae = 0.0
        self.function_mse = 0.0
        self.function_mae = 0.0
        self.bb_mse = 0.0
        self.bb_mae = 0.0
    
    def update(self, predictions, targets, mask):
        """
        Update metrics with batch predictions.
        
        Args:
            predictions: dict with 'binary', 'function', 'bb' keys
            targets: dict with 'binary', 'function', 'bb' keys
            mask: (batch_size, seq_len) - True for valid positions
        """
        mask = mask.bool()
        batch_size = mask.sum().item()
        self.total_samples += batch_size
        
        for level in ['binary', 'function', 'bb']:
            pred = predictions[level].squeeze(-1)[mask]
            target = targets[level][mask]
            
            mse = ((pred - target) ** 2).sum().item()
            mae = (pred - target).abs().sum().item()
            
            if level == 'binary':
                self.binary_mse += mse
                self.binary_mae += mae
            elif level == 'function':
                self.function_mse += mse
                self.function_mae += mae
            elif level == 'bb':
                self.bb_mse += mse
                self.bb_mae += mae
    
    def get_metrics(self):
        """Return average metrics."""
        if self.total_samples == 0:
            return {}
        
        return {
            'binary': {
                'mse': self.binary_mse / self.total_samples,
                'rmse': np.sqrt(self.binary_mse / self.total_samples),
                'mae': self.binary_mae / self.total_samples
            },
            'function': {
                'mse': self.function_mse / self.total_samples,
                'rmse': np.sqrt(self.function_mse / self.total_samples),
                'mae': self.function_mae / self.total_samples
            },
            'bb': {
                'mse': self.bb_mse / self.total_samples,
                'rmse': np.sqrt(self.bb_mse / self.total_samples),
                'mae': self.bb_mae / self.total_samples
            }
        }


def extract_embeddings_addressaware(model, batch, device):
    """
    Extract contextualized embeddings from address-aware model.
    
    Returns:
        embeddings: (batch_size, seq_len, hidden_size)
        positions: dict with 'binary', 'function', 'bb' ground truth
        mask: (batch_size, seq_len) - valid token mask
    """
    # Move batch to device
    cfg_input = batch['cfg_bert_input'].to(device)
    cfg_segment = batch['cfg_segment_label'].to(device)
    cfg_binary_pos = batch['cfg_binary_pos'].to(device)
    cfg_function_pos = batch['cfg_function_pos'].to(device)
    cfg_bb_pos = batch['cfg_bb_pos'].to(device)
    
    # Get embeddings from model
    with torch.no_grad():
        embeddings = model.bert.embedding(
            cfg_input, 
            cfg_segment,
            cfg_binary_pos,
            cfg_function_pos,
            cfg_bb_pos
        )
        
        # Pass through transformer layers
        for layer in model.bert.transformer_blocks:
            embeddings = layer(embeddings, mask=None)
    
    # Create mask (exclude padding tokens)
    mask = (cfg_input != model.bert.embedding.token.padding_idx)
    
    # Ground truth positions
    positions = {
        'binary': cfg_binary_pos,
        'function': cfg_function_pos,
        'bb': cfg_bb_pos
    }
    
    return embeddings, positions, mask


def extract_embeddings_baseline(model, batch, device):
    """
    Extract contextualized embeddings from baseline model.
    
    Returns:
        embeddings: (batch_size, seq_len, hidden_size)
        positions: dict with 'binary', 'function', 'bb' - all zeros (no ground truth)
        mask: (batch_size, seq_len) - valid token mask
    """
    # Move batch to device
    cfg_input = batch['bert_input'].to(device)
    cfg_segment = batch['segment_label'].to(device)
    
    # Get embeddings from model
    with torch.no_grad():
        embeddings = model.bert.embedding(cfg_input, cfg_segment)
        
        # Pass through transformer layers
        for layer in model.bert.transformer_blocks:
            embeddings = layer(embeddings, mask=None)
    
    # Create mask
    mask = (cfg_input != model.bert.embedding.token.padding_idx)
    
    # Baseline has no position ground truth - use zeros
    batch_size, seq_len = cfg_input.shape
    positions = {
        'binary': torch.zeros(batch_size, seq_len, device=device),
        'function': torch.zeros(batch_size, seq_len, device=device),
        'bb': torch.zeros(batch_size, seq_len, device=device)
    }
    
    return embeddings, positions, mask


def train_probe(model, probe, dataloader, device, epochs=10, lr=0.001, is_addressaware=True):
    """
    Train a linear probe to predict positions from embeddings.
    
    Args:
        model: Pretrained BERT model (frozen)
        probe: Position probe model (trainable)
        dataloader: DataLoader for training data
        device: torch device
        epochs: Number of training epochs
        lr: Learning rate
        is_addressaware: Whether model is address-aware (for data extraction)
    
    Returns:
        trained_probe: Trained probe model
        train_metrics: Training metrics
    """
    probe = probe.to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')  # Per-sample loss
    
    model.eval()  # Freeze pretrained model
    
    print(f"\nTraining probe for {epochs} epochs...")
    
    for epoch in range(epochs):
        probe.train()
        metrics = ProbingMetrics()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # Extract embeddings from frozen model
            if is_addressaware:
                embeddings, positions, mask = extract_embeddings_addressaware(model, batch, device)
            else:
                embeddings, positions, mask = extract_embeddings_baseline(model, batch, device)
            
            # Forward through probe
            predictions = probe(embeddings)
            
            # Compute loss (only on valid positions)
            total_loss = 0
            for level in ['binary', 'function', 'bb']:
                pred = predictions[level].squeeze(-1)
                target = positions[level]
                loss = criterion(pred, target)
                
                # Mask out padding
                loss = (loss * mask.float()).sum() / mask.float().sum()
                total_loss += loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update metrics
            metrics.update(predictions, positions, mask)
            
            # Update progress bar
            current_metrics = metrics.get_metrics()
            pbar.set_postfix({
                'loss': total_loss.item(),
                'binary_mae': current_metrics['binary']['mae'],
                'func_mae': current_metrics['function']['mae'],
                'bb_mae': current_metrics['bb']['mae']
            })
        
        # Print epoch summary
        epoch_metrics = metrics.get_metrics()
        print(f"\nEpoch {epoch+1} Results:")
        for level in ['binary', 'function', 'bb']:
            print(f"  {level.upper()}: MAE={epoch_metrics[level]['mae']:.4f}, "
                  f"RMSE={epoch_metrics[level]['rmse']:.4f}")
    
    return probe, metrics.get_metrics()


def evaluate_probe(model, probe, dataloader, device, is_addressaware=True):
    """
    Evaluate trained probe on test data.
    
    Returns:
        metrics: Evaluation metrics
    """
    model.eval()
    probe.eval()
    
    metrics = ProbingMetrics()
    
    print("\nEvaluating probe...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Extract embeddings
            if is_addressaware:
                embeddings, positions, mask = extract_embeddings_addressaware(model, batch, device)
            else:
                embeddings, positions, mask = extract_embeddings_baseline(model, batch, device)
            
            # Predict
            predictions = probe(embeddings)
            
            # Update metrics
            metrics.update(predictions, positions, mask)
    
    return metrics.get_metrics()


def main():
    parser = argparse.ArgumentParser(description='Probe position embeddings from BERT models')
    parser.add_argument('--addressaware_model', type=str, required=True,
                        help='Path to address-aware model checkpoint')
    parser.add_argument('--baseline_model', type=str, required=True,
                        help='Path to baseline model checkpoint')
    parser.add_argument('--test_cfg', type=str, 
                        default='./test_data/all_cfg_probing.txt',
                        help='Path to test CFG data')
    parser.add_argument('--test_dfg', type=str,
                        default='./test_data/all_dfg_probing.txt',
                        help='Path to test DFG data')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size of BERT models')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='Sequence length')
    parser.add_argument('--probe_epochs', type=int, default=10,
                        help='Number of epochs to train probe')
    parser.add_argument('--probe_lr', type=float, default=0.001,
                        help='Learning rate for probe training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print(f"\nLoading vocabulary from: {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load test data (address-aware format for ground truth positions)
    print(f"\nLoading test data...")
    print(f"  CFG: {args.test_cfg}")
    print(f"  DFG: {args.test_dfg}")
    
    test_dataset = PairedAddressAwareDataset(
        cfg_corpus_path=args.test_cfg,
        dfg_corpus_path=args.test_dfg,
        vocab=vocab,
        seq_len=args.seq_len,
        on_memory=True,
        nsp_prob=0.5,
        mask_prob=0.15,
        data_percentage=0.1,  # Use 10% of test data for faster probing
        train_split=1.0,
        is_train=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # ========================================================================
    # PROBE ADDRESS-AWARE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("PROBING ADDRESS-AWARE MODEL")
    print("="*80)
    
    # Load address-aware model
    print(f"\nLoading address-aware model from: {args.addressaware_model}")
    addressaware_model = AddressAwareBERT(
        len(vocab),
        hidden=args.hidden_size,
        n_layers=12,
        attn_heads=8,
        max_len=args.seq_len
    ).to(device)
    
    checkpoint = torch.load(args.addressaware_model, map_location=device)
    addressaware_model.load_state_dict(checkpoint['model_state_dict'])
    addressaware_model.eval()
    print("✓ Address-aware model loaded")
    
    # Create and train probe
    addressaware_probe = MultiLevelPositionProbe(hidden_size=args.hidden_size)
    addressaware_probe, train_metrics = train_probe(
        addressaware_model,
        addressaware_probe,
        test_loader,
        device,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        is_addressaware=True
    )
    
    # Evaluate probe
    addressaware_results = evaluate_probe(
        addressaware_model,
        addressaware_probe,
        test_loader,
        device,
        is_addressaware=True
    )
    
    print("\n" + "-"*80)
    print("ADDRESS-AWARE MODEL - Final Results:")
    print("-"*80)
    for level in ['binary', 'function', 'bb']:
        print(f"{level.upper():10s}: MAE={addressaware_results[level]['mae']:.4f}, "
              f"RMSE={addressaware_results[level]['rmse']:.4f}")
    
    # ========================================================================
    # PROBE BASELINE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("PROBING BASELINE MODEL")
    print("="*80)
    
    # Load baseline model
    print(f"\nLoading baseline model from: {args.baseline_model}")
    baseline_model = BaselineBERT(
        len(vocab),
        hidden=args.hidden_size,
        n_layers=12,
        attn_heads=8,
        max_len=args.seq_len
    ).to(device)
    
    checkpoint = torch.load(args.baseline_model, map_location=device)
    baseline_model.load_state_dict(checkpoint['model_state_dict'])
    baseline_model.eval()
    print("✓ Baseline model loaded")
    
    # Create and train probe
    baseline_probe = MultiLevelPositionProbe(hidden_size=args.hidden_size)
    baseline_probe, train_metrics = train_probe(
        baseline_model,
        baseline_probe,
        test_loader,
        device,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        is_addressaware=False  # Baseline model
    )
    
    # Evaluate probe
    baseline_results = evaluate_probe(
        baseline_model,
        baseline_probe,
        test_loader,
        device,
        is_addressaware=False
    )
    
    print("\n" + "-"*80)
    print("BASELINE MODEL - Final Results:")
    print("-"*80)
    for level in ['binary', 'function', 'bb']:
        print(f"{level.upper():10s}: MAE={baseline_results[level]['mae']:.4f}, "
              f"RMSE={baseline_results[level]['rmse']:.4f}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    results = {
        'addressaware': addressaware_results,
        'baseline': baseline_results,
        'config': vars(args)
    }
    
    output_path = os.path.join(args.output_dir, 'probing_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON: Address-Aware vs Baseline")
    print("="*80)
    
    for level in ['binary', 'function', 'bb']:
        aa_mae = addressaware_results[level]['mae']
        bl_mae = baseline_results[level]['mae']
        improvement = ((bl_mae - aa_mae) / bl_mae) * 100 if bl_mae > 0 else 0
        
        print(f"\n{level.upper()} Position Prediction:")
        print(f"  Address-Aware MAE: {aa_mae:.4f}")
        print(f"  Baseline MAE:      {bl_mae:.4f}")
        print(f"  Improvement:       {improvement:+.2f}%")


if __name__ == "__main__":
    main()
