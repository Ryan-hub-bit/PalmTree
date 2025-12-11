"""
Training script for PalmTree with support for:
1. Original BERT (ignore address info, just use tokens)
2. Address-Aware BERT (use both tokens and address information)

Usage:
    # Original BERT training
    python train_palmtree_addressaware.py --mode original
    
    # Address-Aware BERT training
    python train_palmtree_addressaware.py --mode addressaware
"""

import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from config import *
import palmtree
from palmtree import dataset
from palmtree.trainer import pretrain_addressaware
from palmtree.model import BERT, AddressAwareBERT

print(palmtree.__file__)


def main():
    parser = argparse.ArgumentParser(description='PalmTree Training with Address-Aware Support')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='original', choices=['original', 'addressaware'],
                        help='Training mode: original (ignore address) or addressaware (use address)')
    
    # Data paths
    parser.add_argument('--vocab_path', type=str, default='cdfg_bert_1/vocab',
                        help='Path to save/load vocabulary')
    parser.add_argument('--train_cfg', type=str, required=True,
                        help='Path to CFG training data')
    parser.add_argument('--train_dfg', type=str, required=True,
                        help='Path to DFG training data')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data (optional)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save model checkpoints')
    
    # Model hyperparameters
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden size')
    parser.add_argument('--n_layers', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--attn_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='Maximum sequence length')
    
    # Address-aware specific parameters
    parser.add_argument('--address_hidden', type=int, default=64,
                        help='Hidden size for address embedding (for addressaware mode)')
    parser.add_argument('--var_size', type=int, default=256,
                        help='Variable offset vocabulary size (for addressaware mode)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Number of data loading workers')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Logging frequency')
    
    # Vocab parameters
    parser.add_argument('--vocab_max_size', type=int, default=13000,
                        help='Maximum vocabulary size')
    parser.add_argument('--vocab_min_freq', type=int, default=1,
                        help='Minimum token frequency for vocabulary')
    
    # GPU settings
    parser.add_argument('--cuda_devices', type=int, nargs='+', default=[0],
                        help='CUDA device IDs')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Training Mode: {args.mode.upper()}")
    print(f"{'='*60}\n")
    
    # Step 1: Build/Load Vocabulary
    print(f"Building vocabulary from training data...")
    with open(args.train_cfg, "r", encoding="utf-8") as f1:
        with open(args.train_dfg, "r", encoding="utf-8") as f2:
            vocab = dataset.WordVocab([f1, f2], max_size=args.vocab_max_size, min_freq=args.vocab_min_freq)
    
    print(f"VOCAB SIZE: {len(vocab)}")
    vocab.save_vocab(args.vocab_path)
    
    print(f"Loading Vocab from {args.vocab_path}")
    vocab = dataset.WordVocab.load_vocab(args.vocab_path)
    print(f"Vocab Size: {len(vocab)}")
    
    # Step 2: Load Dataset
    if args.mode == 'original':
        print("\n[ORIGINAL MODE] Loading dataset without address information...")
        print("Using BERTDataset (tokens only)")
        
        train_dataset = dataset.BERTDataset(
            args.train_cfg, args.train_dfg, vocab, 
            seq_len=args.seq_len, corpus_lines=None, on_memory=True
        )
        
        test_dataset = dataset.BERTDataset(
            args.test_data, args.test_data, vocab, 
            seq_len=args.seq_len, on_memory=True
        ) if args.test_data is not None else None
        
    elif args.mode == 'addressaware':
        print("\n[ADDRESS-AWARE MODE] Loading dataset with address information...")
        print("Using BERTDatasetAddressAware (tokens + binary/function/BB positions + var offsets)")
        
        train_dataset = dataset.BERTDatasetAddressAware(
            args.train_cfg, args.train_dfg, vocab, 
            seq_len=args.seq_len, corpus_lines=None, on_memory=True
        )
        
        test_dataset = dataset.BERTDatasetAddressAware(
            args.test_data, args.test_data, vocab, 
            seq_len=args.seq_len, on_memory=True
        ) if args.test_data is not None else None
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Step 3: Create DataLoader
    print("Creating DataLoader...")
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=True
    )
    
    test_data_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    ) if test_dataset is not None else None
    
    # Step 4: Build Model
    if args.mode == 'original':
        print("\n[ORIGINAL MODE] Building standard BERT model...")
        model = BERT(
            len(vocab), 
            hidden=args.hidden, 
            n_layers=args.n_layers, 
            attn_heads=args.attn_heads, 
            dropout=args.dropout
        )
        
    elif args.mode == 'addressaware':
        print("\n[ADDRESS-AWARE MODE] Building AddressAwareBERT model...")
        print(f"  - Token vocab size: {len(vocab)}")
        print(f"  - Hidden size: {args.hidden}")
        print(f"  - Address hidden size: {args.address_hidden}")
        print(f"  - Var offset vocab size: {args.var_size}")
        
        model = AddressAwareBERT(
            vocab_size=len(vocab),
            hidden=args.hidden,
            n_layers=args.n_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
            address_hidden=args.address_hidden,
            var_size=args.var_size
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 5: Create Trainer
    print("\nCreating BERT Trainer...")
    bert_trainer = pretrain_addressaware.BERTTrainer(
        model, 
        len(vocab), 
        train_dataloader=train_data_loader, 
        test_dataloader=test_data_loader,
        lr=args.lr, 
        betas=(0.9, 0.999), 
        weight_decay=args.weight_decay,
        with_cuda=True, 
        cuda_devices=args.cuda_devices, 
        log_freq=args.log_freq,
        mode=args.mode  # Pass mode to trainer
    )
    
    # Step 6: Training Loop
    print(f"\n{'='*60}")
    print(f"Starting Training for {args.epochs} epochs")
    print(f"Output path: {args.output_path}")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        bert_trainer.train(epoch)
        bert_trainer.save(epoch, args.output_path)
        
        # if test_data_loader is not None:
        #     bert_trainer.test(epoch)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Model saved to: {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
