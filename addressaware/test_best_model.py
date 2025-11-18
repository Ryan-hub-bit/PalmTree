"""
Test the best Address-Aware model on CFG and DFG data.

This script:
1. Loads the best model checkpoint
2. Evaluates on CFG data (MLM + NSP)
3. Evaluates on DFG data (NSP only)
4. Reports detailed metrics for both
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import argparse
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from palmtree.dataset.vocab import WordVocab
from dataloader_paired import PairedAddressAwareDataset
from model import AddressAwareBERT, AddressAwareBERTForPretraining


def evaluate_model(model, data_loader, device, verbose=True):
    """
    Evaluate the model on validation data.
    
    Returns detailed metrics for CFG and DFG separately.
    """
    model.eval()
    
    # Loss accumulators
    total_loss = 0
    mlm_loss_total = 0
    nsp_cfg_loss_total = 0
    nsp_dfg_loss_total = 0
    
    # Accuracy accumulators
    mlm_correct = 0
    mlm_total = 0
    nsp_cfg_correct = 0
    nsp_cfg_total = 0
    nsp_dfg_correct = 0
    nsp_dfg_total = 0
    
    # Loss functions
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        iterator = tqdm(data_loader, desc="Evaluating") if verbose else data_loader
        
        for batch in iterator:
            # === CFG Processing ===
            cfg_token_ids = batch['cfg_bert_input'].to(device)
            cfg_segment_labels = batch['cfg_segment_label'].to(device)
            cfg_binary_pos = batch['cfg_binary_pos'].to(device)
            cfg_function_pos = batch['cfg_function_pos'].to(device)
            cfg_bb_pos = batch['cfg_bb_pos'].to(device)
            cfg_mlm_labels = batch['cfg_bert_label'].to(device)
            cfg_nsp_labels = batch['cfg_is_next'].to(device)
            
            # CFG forward pass
            cfg_mlm_output, cfg_nsp_output = model(
                cfg_token_ids, cfg_segment_labels,
                cfg_binary_pos, cfg_function_pos, cfg_bb_pos,
                corpus_type='cfg'
            )
            
            # CFG losses
            mlm_loss = mlm_criterion(cfg_mlm_output.transpose(1, 2), cfg_mlm_labels)
            nsp_cfg_loss = nsp_criterion(cfg_nsp_output, cfg_nsp_labels)
            
            # CFG accuracy - MLM
            mask = cfg_mlm_labels != -1
            if mask.any():
                mlm_pred = torch.argmax(cfg_mlm_output[mask], dim=-1)
                mlm_correct += (mlm_pred == cfg_mlm_labels[mask]).sum().item()
                mlm_total += mask.sum().item()
            
            # CFG accuracy - NSP
            nsp_cfg_pred = torch.argmax(cfg_nsp_output, dim=-1)
            nsp_cfg_correct += (nsp_cfg_pred == cfg_nsp_labels).sum().item()
            nsp_cfg_total += len(cfg_nsp_labels)
            
            # === DFG Processing ===
            dfg_token_ids = batch['dfg_bert_input'].to(device)
            dfg_segment_labels = batch['dfg_segment_label'].to(device)
            dfg_binary_pos = batch['dfg_binary_pos'].to(device)
            dfg_function_pos = batch['dfg_function_pos'].to(device)
            dfg_bb_pos = batch['dfg_bb_pos'].to(device)
            dfg_nsp_labels = batch['dfg_is_next'].to(device)
            
            # DFG forward pass (NSP only)
            _, dfg_nsp_output = model(
                dfg_token_ids, dfg_segment_labels,
                dfg_binary_pos, dfg_function_pos, dfg_bb_pos,
                corpus_type='dfg'
            )
            
            # DFG loss
            nsp_dfg_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels)
            
            # DFG accuracy - NSP
            nsp_dfg_pred = torch.argmax(dfg_nsp_output, dim=-1)
            nsp_dfg_correct += (nsp_dfg_pred == dfg_nsp_labels).sum().item()
            nsp_dfg_total += len(dfg_nsp_labels)
            
            # Accumulate losses
            combined_loss = mlm_loss + nsp_cfg_loss + nsp_dfg_loss
            total_loss += combined_loss.item()
            mlm_loss_total += mlm_loss.item()
            nsp_cfg_loss_total += nsp_cfg_loss.item()
            nsp_dfg_loss_total += nsp_dfg_loss.item()
    
    # Calculate metrics
    n_batches = len(data_loader)
    
    results = {
        'total_loss': total_loss / n_batches,
        'mlm_loss': mlm_loss_total / n_batches,
        'mlm_accuracy': mlm_correct / mlm_total if mlm_total > 0 else 0.0,
        'mlm_correct': mlm_correct,
        'mlm_total': mlm_total,
        'nsp_cfg_loss': nsp_cfg_loss_total / n_batches,
        'nsp_cfg_accuracy': nsp_cfg_correct / nsp_cfg_total if nsp_cfg_total > 0 else 0.0,
        'nsp_cfg_correct': nsp_cfg_correct,
        'nsp_cfg_total': nsp_cfg_total,
        'nsp_dfg_loss': nsp_dfg_loss_total / n_batches,
        'nsp_dfg_accuracy': nsp_dfg_correct / nsp_dfg_total if nsp_dfg_total > 0 else 0.0,
        'nsp_dfg_correct': nsp_dfg_correct,
        'nsp_dfg_total': nsp_dfg_total,
    }
    
    return results


def print_results(results, title="Evaluation Results"):
    """Pretty print evaluation results."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)
    
    print(f"\n{'Combined Loss:':<25} {results['total_loss']:.4f}")
    
    print(f"\n{'CFG - Masked Language Modeling (MLM)':}")
    print(f"  {'Loss:':<23} {results['mlm_loss']:.4f}")
    print(f"  {'Accuracy:':<23} {results['mlm_accuracy']:.2%} ({results['mlm_correct']:,} / {results['mlm_total']:,})")
    
    print(f"\n{'CFG - Next Sentence Prediction (NSP)':}")
    print(f"  {'Loss:':<23} {results['nsp_cfg_loss']:.4f}")
    print(f"  {'Accuracy:':<23} {results['nsp_cfg_accuracy']:.2%} ({results['nsp_cfg_correct']:,} / {results['nsp_cfg_total']:,})")
    
    print(f"\n{'DFG - Next Sentence Prediction (NSP)':}")
    print(f"  {'Loss:':<23} {results['nsp_dfg_loss']:.4f}")
    print(f"  {'Accuracy:':<23} {results['nsp_dfg_accuracy']:.2%} ({results['nsp_dfg_correct']:,} / {results['nsp_dfg_total']:,})")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test Address-Aware BERT best model")
    
    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, 
                       default="output_addressaware/best_model.pt",
                       help="Path to model checkpoint")
    
    # Data paths
    parser.add_argument("--cfg_data", type=str,
                       default="../data/test/cfg/all_cfg_combined.txt",
                       help="CFG data file")
    parser.add_argument("--dfg_data", type=str,
                       default="../data/test/dfg/all_dfg_combined.txt",
                       help="DFG data file")
    parser.add_argument("--vocab", type=str,
                       default="../pre-trained_model/palmtree/vocab",
                       help="Vocabulary file")
    
    # Test configuration
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--data_percentage", type=float, default=1.0,
                       help="Percentage of data to test on (0.0-1.0)")
    parser.add_argument("--use_val_split", action="store_true",
                       help="Use validation split from training (last 10%)")
    
    # Device
    parser.add_argument("--cuda", action="store_true", default=True,
                       help="Use CUDA if available")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load training args to get model architecture
    args_path = os.path.join(os.path.dirname(args.checkpoint), "args.json")
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            train_args = json.load(f)
        print(f"Loaded training configuration from: {args_path}")
    else:
        print(f"Warning: Could not find {args_path}, using defaults")
        train_args = {
            'hidden': 128,
            'layers': 12,
            'attn_heads': 8,
            'seq_len': 20,
            'dropout': 0.1,
        }
    
    # Load vocabulary
    print(f"\nLoading vocabulary: {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create model with same architecture as training
    print(f"\nCreating model...")
    print(f"  Hidden: {train_args['hidden']}")
    print(f"  Layers: {train_args['layers']}")
    print(f"  Attention heads: {train_args['attn_heads']}")
    print(f"  Sequence length: {train_args['seq_len']}")
    
    bert = AddressAwareBERT(
        vocab_size=len(vocab),
        hidden=train_args['hidden'],
        n_layers=train_args['layers'],
        attn_heads=train_args['attn_heads'],
        dropout=train_args.get('dropout', 0.1),
        max_len=train_args['seq_len']
    )
    
    model = AddressAwareBERTForPretraining(bert, len(vocab))
    
    # Load weights
    print("Loading model weights...")
    state_dict = checkpoint['model_state_dict']
    
    # Handle DataParallel wrapper (remove "module." prefix if present)
    if list(state_dict.keys())[0].startswith('module.'):
        print("Removing DataParallel 'module.' prefix from state dict...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load with strict=False to handle minor architecture differences
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys (using random init): {len(missing_keys)} keys")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"  - {key}")
    
    if unexpected_keys:
        print(f"Warning: Unexpected keys (ignored): {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys:
                print(f"  - {key}")
    
    model = model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch'] + 1}")
    if 'val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    
    # Create test dataset
    print(f"\nLoading test data...")
    
    if args.use_val_split:
        # Use the validation split (last 10% from training)
        print("Using validation split (last 10% of training data)")
        test_dataset = PairedAddressAwareDataset(
            cfg_corpus_path=args.cfg_data,
            dfg_corpus_path=args.dfg_data,
            vocab=vocab,
            seq_len=train_args['seq_len'],
            on_memory=True,
            nsp_prob=train_args.get('nsp_prob', 0.5),
            mask_prob=train_args.get('mask_prob', 0.15),
            data_percentage=train_args.get('data_percentage', 1.0),
            train_split=train_args.get('train_split', 0.9),
            is_train=False  # Get validation split
        )
    else:
        # Use all data or specified percentage
        print(f"Using {args.data_percentage*100:.0f}% of full dataset")
        test_dataset = PairedAddressAwareDataset(
            cfg_corpus_path=args.cfg_data,
            dfg_corpus_path=args.dfg_data,
            vocab=vocab,
            seq_len=train_args['seq_len'],
            on_memory=True,
            nsp_prob=train_args.get('nsp_prob', 0.5),
            mask_prob=train_args.get('mask_prob', 0.15),
            data_percentage=args.data_percentage,
            train_split=1.0,  # Use all as "training" (no split)
            is_train=True
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    print(f"Test batches: {len(test_loader)}")
    
    # Evaluate
    print("\n" + "="*70)
    print("Starting evaluation...")
    print("="*70)
    
    results = evaluate_model(model, test_loader, device, verbose=True)
    
    # Print results
    print_results(results, "Address-Aware BERT - Best Model Results")
    
    # Save results to file
    output_file = os.path.join(os.path.dirname(args.checkpoint), "test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
