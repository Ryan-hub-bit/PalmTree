"""
Comparison Training Script - Original vs Address-Aware BERT

This script allows training and comparing two versions:
1. BASELINE: Original BERT without address embeddings (masks position numbers)
2. ADDRESS_AWARE: Your strupos version with address embeddings and position encodings

Usage:
    # Train baseline (original, no address features)
    python train_comparison.py --mode baseline
    
    # Train address-aware (your strupos version)
    python train_comparison.py --mode address_aware
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, 'src')
sys.path.insert(0, '../../strupos')  # For your address-aware components

from config import *

# ============================================================================
# Parse Arguments
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='PalmTree Comparison Training')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['baseline', 'address_aware'],
                        help='Training mode: baseline (no address) or address_aware (with address)')
    
    # Data paths
    parser.add_argument('--train_cfg', type=str, 
                        default='/data/kun/palmtreedata/cfg_train_2.txt',
                        help='CFG training data')
    parser.add_argument('--train_dfg', type=str,
                        default='/data/kun/palmtreedata/dfg_train_2.txt', 
                        help='DFG training data')
    parser.add_argument('--test_cfg', type=str,
                        default='/data/kun/palmtreedata/cfg_test_2.txt',
                        help='CFG test data')
    parser.add_argument('--test_dfg', type=str,
                        default='/data/kun/palmtreedata/dfg_test_2.txt',
                        help='DFG test data')
    parser.add_argument('--vocab_path', type=str,
                        default='../../strupos/vocab.pkl',
                        help='Vocabulary file')
    
    # Model hyperparameters
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='Hidden size')
    parser.add_argument('--n_layers', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--attn_heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Address-aware specific parameters
    parser.add_argument('--address_embed_dim', type=int, default=64,
                        help='Address embedding dimension (address_aware mode)')
    parser.add_argument('--var_embed_dim', type=int, default=32,
                        help='Variable embedding dimension (address_aware mode)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=10000,
                        help='Warmup steps')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Maximum sequence length')
    
    # Task configuration
    parser.add_argument('--enable_imc', action='store_true', default=True,
                        help='Enable instruction masking for CFG')
    parser.add_argument('--enable_imd', action='store_true', default=False,
                        help='Enable instruction masking for DFG')
    parser.add_argument('--enable_mlm', action='store_true', default=True,
                        help='Enable token-level MLM')
    parser.add_argument('--instruction_mask_prob', type=float, default=0.25,
                        help='Instruction masking probability')
    parser.add_argument('--token_mask_prob', type=float, default=0.15,
                        help='Token masking probability')
    
    # Training settings
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Logging frequency')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for checkpoints')
    parser.add_argument('--cuda_devices', type=int, nargs='+', default=[0],
                        help='CUDA device IDs')
    parser.add_argument('--data_percentage', type=float, default=1.0,
                        help='Percentage of data to use (for quick testing)')
    
    return parser.parse_args()


# ============================================================================
# Baseline Mode Setup (Original BERT without address features)
# ============================================================================
def setup_baseline_mode(args, vocab):
    """
    Setup baseline mode: Original BERT without address embeddings
    - Masks all position numbers in address(...) format
    - Uses standard BERT architecture
    """
    print("\n" + "="*80)
    print("BASELINE MODE: Original BERT (No Address Features)")
    print("="*80)
    print("This mode:")
    print("  - Masks position numbers in address() tokens")
    print("  - Does NOT use address embeddings")
    print("  - Uses standard BERT architecture")
    print("="*80 + "\n")
    
    # Import baseline components
    from palmtree.dataset.dataset_baseline import BaselineDataset
    from palmtree.model import BERT
    from palmtree.trainer import BERTTrainer
    
    # Create dataset with position masking
    print("Loading baseline dataset (masking positions)...")
    train_dataset = BaselineDataset(
        cfg_corpus_path=args.train_cfg,
        dfg_corpus_path=args.train_dfg if args.enable_imd else None,
        vocab=vocab,
        seq_len=args.seq_len,
        token_mask_prob=args.token_mask_prob,
        instruction_mask_prob=args.instruction_mask_prob,
        enable_imd=args.enable_imd,
        data_percentage=args.data_percentage,
    )
    
    test_dataset = None
    if args.test_cfg and os.path.exists(args.test_cfg):
        test_dataset = BaselineDataset(
            cfg_corpus_path=args.test_cfg,
            dfg_corpus_path=args.test_dfg if args.enable_imd else None,
            vocab=vocab,
            seq_len=args.seq_len,
            token_mask_prob=args.token_mask_prob,
            instruction_mask_prob=args.instruction_mask_prob,
            enable_imd=args.enable_imd,
            data_percentage=args.data_percentage,
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Create standard BERT model (no address features)
    print(f"\nBuilding standard BERT model...")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Attention heads: {args.attn_heads}")
    
    model = BERT(
        vocab_size=len(vocab),
        hidden=args.hidden_size,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout
    )
    
    # Create trainer
    trainer = BERTTrainer(
        bert=model,
        vocab_size=len(vocab),
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        with_cuda=torch.cuda.is_available(),
        cuda_devices=args.cuda_devices,
        log_freq=args.log_freq,
        mode='baseline'
    )
    
    return trainer, train_dataset, test_dataset


# ============================================================================
# Address-Aware Mode Setup (Your strupos version)
# ============================================================================
def setup_address_aware_mode(args, vocab):
    """
    Setup address-aware mode: Your strupos version with address embeddings
    - Uses position numbers for address embeddings
    - Uses your custom BERT with address-aware layers
    """
    print("\n" + "="*80)
    print("ADDRESS-AWARE MODE: Your Strupos Version")
    print("="*80)
    print("This mode:")
    print("  - Uses position numbers for address embeddings")
    print("  - Uses address-aware BERT architecture")
    print("  - Includes variable offset embeddings")
    print("="*80 + "\n")
    
    # Import your address-aware components
    sys.path.insert(0, '../../strupos')
    from dataloader import InstructionMaskingDataset
    from model import AddressAwareBERT
    from train import AddressAwareTrainer
    
    # Create your address-aware dataset
    print("Loading address-aware dataset (using positions)...")
    train_dataset = InstructionMaskingDataset(
        cfg_corpus_path=args.train_cfg,
        dfg_corpus_path=args.train_dfg if args.enable_imd else None,
        vocab=vocab,
        seq_len=args.seq_len,
        token_mask_prob=args.token_mask_prob,
        instruction_mask_prob=args.instruction_mask_prob,
        enable_imd=args.enable_imd,
        data_percentage=args.data_percentage,
    )
    
    test_dataset = None
    if args.test_cfg and os.path.exists(args.test_cfg):
        test_dataset = InstructionMaskingDataset(
            cfg_corpus_path=args.test_cfg,
            dfg_corpus_path=args.test_dfg if args.enable_imd else None,
            vocab=vocab,
            seq_len=args.seq_len,
            token_mask_prob=args.token_mask_prob,
            instruction_mask_prob=args.instruction_mask_prob,
            enable_imd=args.enable_imd,
            data_percentage=args.data_percentage,
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Create your address-aware BERT model
    print(f"\nBuilding address-aware BERT model...")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Attention heads: {args.attn_heads}")
    print(f"  Address embedding dim: {args.address_embed_dim}")
    print(f"  Variable embedding dim: {args.var_embed_dim}")
    
    model = AddressAwareBERT(
        vocab_size=len(vocab),
        hidden=args.hidden_size,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        address_embed_dim=args.address_embed_dim,
        var_embed_dim=args.var_embed_dim,
    )
    
    # Create your trainer
    trainer = AddressAwareTrainer(
        model=model,
        vocab_size=len(vocab),
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        with_cuda=torch.cuda.is_available(),
        cuda_devices=args.cuda_devices,
        log_freq=args.log_freq,
    )
    
    return trainer, train_dataset, test_dataset


# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("PalmTree Comparison Training")
    print("="*80)
    print(f"Mode: {args.mode.upper()}")
    print(f"CFG Train: {args.train_cfg}")
    print(f"DFG Train: {args.train_dfg if args.enable_imd else 'Not used'}")
    print(f"Vocab: {args.vocab_path}")
    print("="*80 + "\n")
    
    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab_path}...")
    if args.vocab_path.endswith('.pkl'):
        # Load your strupos vocab
        import pickle
        with open(args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    else:
        # Load original palmtree vocab
        from palmtree.dataset import WordVocab
        vocab = WordVocab.load_vocab(args.vocab_path)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Setup based on mode
    if args.mode == 'baseline':
        trainer, train_dataset, test_dataset = setup_baseline_mode(args, vocab)
    else:  # address_aware
        trainer, train_dataset, test_dataset = setup_address_aware_mode(args, vocab)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Training loop
    print("\n" + "="*80)
    print("Training Start")
    print("="*80 + "\n")
    
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss = trainer.train(epoch)
        
        # Test
        test_loss = None
        if test_dataset is not None:
            print("\nRunning test evaluation...")
            test_loss = trainer.test(epoch)
        
        # Save checkpoint
        checkpoint_path = output_dir / f"epoch_{epoch}.pt"
        trainer.save(epoch, str(output_dir))
        print(f"Model saved to {checkpoint_path}")
        
        # Save best model
        current_loss = test_loss if test_loss is not None else train_loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_path = output_dir / "best_model.pt"
            trainer.save(epoch, str(output_dir), save_path="best_model.pt")
            print(f"Best model updated! Loss: {best_loss:.4f}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
