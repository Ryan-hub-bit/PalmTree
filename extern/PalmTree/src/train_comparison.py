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
# Import baseline components
from palmtree.dataset.dataset_baseline import BaselineDataset
from palmtree.model import BERT
from palmtree.trainer import BERTTrainer

# Import regex for preprocessing
import re

# Add paths
sys.path.insert(0, 'src')

from config import *
from vocab import WordVocab

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
    parser.add_argument('--token_mask_prob', type=float, default=0.15,
                        help='Token masking probability (MLM) - used by both modes')
    parser.add_argument('--instruction_mask_prob', type=float, default=0.25,
                        help='Instruction masking probability (IMC) - only used by address-aware mode')
    parser.add_argument('--enable_imc', action='store_true',
                        help='Enable Instruction Masking Classification task - only used by address-aware mode')
    parser.add_argument('--enable_mlm', action='store_true',
                        help='Enable Masked Language Modeling task - only used by address-aware mode')
    
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
                        help='Percentage of data to use (for quick testing) - used by both modes')
    
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
    print(f"  - Token masking probability: {args.token_mask_prob}")
    print(f"  - Data percentage: {args.data_percentage*100}%")
    print("="*80 + "\n")
    
    
    # Create dataset with position masking
    print("Loading baseline dataset (masking positions)...")
    train_dataset = BaselineDataset(
        dfg_corpus_path=args.train_dfg,
        cfg_corpus_path=args.train_cfg,
        vocab=vocab,
        seq_len=args.seq_len,
        encoding="utf-8",
        corpus_lines=None,
        on_memory=True,
        token_mask_prob=args.token_mask_prob,
        data_percentage=args.data_percentage
    )
    
    test_dataset = None
    if args.test_cfg and os.path.exists(args.test_cfg):
        test_dataset = BaselineDataset(
            dfg_corpus_path=args.test_dfg,
            cfg_corpus_path=args.test_cfg,
            vocab=vocab,
            seq_len=args.seq_len,
            encoding="utf-8",
            corpus_lines=None,
            on_memory=True,
            token_mask_prob=args.token_mask_prob,
            data_percentage=args.data_percentage
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
        log_freq=args.log_freq
    )
    
    return trainer


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
    
    # Import address-aware components (copied from strupos)
    from address_aware.dataloader_addressaware import InstructionMaskingDataset
    from address_aware.model_addressaware import AddressAwareBERT
    from address_aware.train_addressaware import AddressAwareTrainer
    
    # Create your address-aware dataset
    print("Loading address-aware dataset (using positions)...")
    train_dataset = InstructionMaskingDataset(
        cfg_corpus_path=args.train_cfg,
        dfg_corpus_path=args.train_dfg,
        vocab=vocab,
        seq_len=args.seq_len,
        token_mask_prob=args.token_mask_prob,
        instruction_mask_prob=args.instruction_mask_prob,
        data_percentage=args.data_percentage,
        enable_imd=False,  # DFG only for DUP task, not for instruction masking
    )
    
    test_dataset = None
    if args.test_cfg and os.path.exists(args.test_cfg):
        test_dataset = InstructionMaskingDataset(
            cfg_corpus_path=args.test_cfg,
            dfg_corpus_path=args.test_dfg,
            vocab=vocab,
            seq_len=args.seq_len,
            token_mask_prob=args.token_mask_prob,
            instruction_mask_prob=args.instruction_mask_prob,
            data_percentage=args.data_percentage,
            enable_imd=False,  # DFG only for DUP task, not for instruction masking
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
    
    model = AddressAwareBERT(
        vocab_size=len(vocab),
        hidden=args.hidden_size,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        max_len=args.seq_len,
        use_address_embedding=True,
        use_var_embedding=True,
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
    
    return trainer


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
    print(f"Vocab: {args.vocab_path}")
    print("="*80 + "\n")
    
    # Load or create vocabulary
    if not os.path.exists(args.vocab_path):
        print(f"Vocabulary file not found at {args.vocab_path}")
        print("Building vocabulary from training data...")
        
        
        # Create vocab directory if needed
        vocab_dir = os.path.dirname(args.vocab_path)
        if vocab_dir and not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
        
        if args.mode == 'baseline':
            # Preprocessing wrapper for baseline vocab (same as baseline dataset preprocessing)
            class BaselinePreprocessedFileWrapper:
                """Preprocess lines to match baseline format during vocab creation"""
                def __init__(self, file_handle):
                    self.file_handle = file_handle
                    self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
                    self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
                    self.daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
                    self.var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
                
                def __iter__(self):
                    for line in self.file_handle:
                        # Preprocess line same as baseline dataset
                        # 1. daddr(...) -> address
                        line = self.daddr_pattern.sub('address', line)
                        # 2. address(...) -> address
                        line = self.nested_addr_pattern.sub('address', line)
                        # 3. opcode(...) -> opcode
                        line = self.addr_pattern.sub(r'\1', line)
                        # 4. var(0xOFFSET) -> var_0xOFFSET
                        line = self.var_pattern.sub(r'var_\1', line)
                        # Return tokens
                        yield line.split()
            
            # Build vocabulary from training files with baseline preprocessing
            with open(args.train_cfg, "r", encoding="utf-8") as f1:
                if args.train_dfg:
                    with open(args.train_dfg, "r", encoding="utf-8") as f2:
                        vocab = WordVocab([BaselinePreprocessedFileWrapper(f1), BaselinePreprocessedFileWrapper(f2)], 
                                         max_size=5000, min_freq=2)
                else:
                    vocab = WordVocab([BaselinePreprocessedFileWrapper(f1)], max_size=5000, min_freq=2)
        
        else:  # address_aware
            # Preprocessing for address-aware vocab (from create_vocab.py)
            # Remove address info in parentheses but keep token structure
            class AddressAwarePreprocessedFile:
                """Remove all address information in parentheses during vocab creation"""
                def __init__(self, file_handle):
                    self.file_handle = file_handle
                
                def __iter__(self):
                    for line in self.file_handle:
                        # Remove all patterns like (0xADDR:pos1:pos2:pos3) - address positions
                        cleaned = re.sub(r'\(0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+\)', '', line)
                        # Remove all patterns like (0xADDR) - var offsets
                        cleaned = re.sub(r'\(0x[0-9a-fA-F]+\)', '', cleaned)
                        # Split by whitespace and filter empty strings
                        tokens = [tok for tok in cleaned.replace('\t', ' ').split() if tok]
                        yield tokens
            
            # Build vocabulary from training files with address-aware preprocessing
            with open(args.train_cfg, "r", encoding="utf-8") as f1:
                if args.train_dfg:
                    with open(args.train_dfg, "r", encoding="utf-8") as f2:
                        vocab = WordVocab([AddressAwarePreprocessedFile(f1), AddressAwarePreprocessedFile(f2)], 
                                         max_size=5000, min_freq=2)
                else:
                    vocab = WordVocab([AddressAwarePreprocessedFile(f1)], max_size=5000, min_freq=2)
        
        print(f"VOCAB SIZE: {len(vocab)}")
        vocab.save_vocab(args.vocab_path)
        print(f"Vocabulary saved to {args.vocab_path}")
    
    # Load vocabulary
    print(f"\nLoading Vocab from {args.vocab_path}")
    vocab = WordVocab.load_vocab(args.vocab_path)
    print(f"Vocab Size: {len(vocab)}")
    
    # Setup based on mode
    if args.mode == 'baseline':
        trainer = setup_baseline_mode(args, vocab)
    else:  # address_aware
        trainer = setup_address_aware_mode(args, vocab)
    
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
        if trainer.test_data is not None:
            print("\nRunning test evaluation...")
            test_loss = trainer.test(epoch)
        
        # Save epoch checkpoint
        checkpoint_path = output_dir / f"epoch_{epoch}.pt"
        trainer.save(epoch, str(checkpoint_path))
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model (only best_model.pt, no epoch number)
        current_loss = test_loss if test_loss is not None else train_loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_path = output_dir / "best_model.pt"
            trainer.save(epoch, str(best_path))
            print(f"Best model updated! Loss: {best_loss:.4f}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
