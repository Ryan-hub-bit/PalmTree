"""
Training script for Baseline BERT (NO address-aware embeddings)

This is for FAIR COMPARISON with Address-Aware BERT:
- Same data
- Same MLM/NSP tasks
- Same hyperparameters
- Same model architecture (EXCEPT no address embeddings)

The ONLY difference: Uses sequential positional embeddings instead of address-aware embeddings
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import argparse
import os
import sys
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from palmtree.dataset.vocab import WordVocab
from dataloader_baseline import PairedBaselineDataset
from model_baseline import create_baseline_model


def train_epoch(model, data_loader, optimizer, device, log_freq=100):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    mlm_loss_total = 0
    nsp_cfg_loss_total = 0
    nsp_dfg_loss_total = 0
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    
    progress = tqdm(data_loader, desc="Training")
    
    for i, batch in enumerate(progress):
        # Move to device
        cfg_token_ids = batch['bert_input'].to(device)
        cfg_segment_labels = batch['segment_label'].to(device)
        cfg_mlm_labels = batch['bert_label'].to(device)
        cfg_nsp_labels = batch['is_next'].to(device)
        
        dfg_token_ids = batch['dfg_input'].to(device)
        dfg_segment_labels = batch['dfg_segment'].to(device)
        dfg_nsp_labels = batch['dfg_is_next'].to(device)
        
        # Forward pass: model(cfg_input, cfg_segment, dfg_input, dfg_segment)
        # Returns: mlm_output, cwp_output, dup_output
        mlm_output, cfg_nsp_output, dfg_nsp_output = model(
            cfg_token_ids, cfg_segment_labels,
            dfg_token_ids, dfg_segment_labels
        )
        
        # Calculate losses
        mlm_loss = mlm_criterion(mlm_output.transpose(1, 2), cfg_mlm_labels)
        nsp_cfg_loss = nsp_criterion(cfg_nsp_output, cfg_nsp_labels)
        nsp_dfg_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels)
        
        # Combined loss
        loss = mlm_loss + nsp_cfg_loss + nsp_dfg_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        mlm_loss_total += mlm_loss.item()
        nsp_cfg_loss_total += nsp_cfg_loss.item()
        nsp_dfg_loss_total += nsp_dfg_loss.item()
        
        # Update progress bar
        if i % log_freq == 0:
            avg_loss = total_loss / (i + 1)
            avg_mlm = mlm_loss_total / (i + 1)
            avg_nsp_cfg = nsp_cfg_loss_total / (i + 1)
            avg_nsp_dfg = nsp_dfg_loss_total / (i + 1)
            progress.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'mlm': f'{avg_mlm:.4f}',
                'nsp_cfg': f'{avg_nsp_cfg:.4f}',
                'nsp_dfg': f'{avg_nsp_dfg:.4f}'
            })
    
    return {
        'total_loss': total_loss / len(data_loader),
        'mlm_loss': mlm_loss_total / len(data_loader),
        'nsp_cfg_loss': nsp_cfg_loss_total / len(data_loader),
        'nsp_dfg_loss': nsp_dfg_loss_total / len(data_loader)
    }


def validate_epoch(model, data_loader, device):
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0
    mlm_loss_total = 0
    nsp_cfg_loss_total = 0
    nsp_dfg_loss_total = 0
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            # Move to device
            cfg_token_ids = batch['bert_input'].to(device)
            cfg_segment_labels = batch['segment_label'].to(device)
            cfg_mlm_labels = batch['bert_label'].to(device)
            cfg_nsp_labels = batch['is_next'].to(device)
            
            dfg_token_ids = batch['dfg_input'].to(device)
            dfg_segment_labels = batch['dfg_segment'].to(device)
            dfg_nsp_labels = batch['dfg_is_next'].to(device)
            
            # Forward pass
            mlm_output, cfg_nsp_output, dfg_nsp_output = model(
                cfg_token_ids, cfg_segment_labels,
                dfg_token_ids, dfg_segment_labels
            )
            
            # Calculate losses
            mlm_loss = mlm_criterion(mlm_output.transpose(1, 2), cfg_mlm_labels)
            nsp_cfg_loss = nsp_criterion(cfg_nsp_output, cfg_nsp_labels)
            nsp_dfg_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels)
            
            loss = mlm_loss + nsp_cfg_loss + nsp_dfg_loss
            
            # Accumulate
            total_loss += loss.item()
            mlm_loss_total += mlm_loss.item()
            nsp_cfg_loss_total += nsp_cfg_loss.item()
            nsp_dfg_loss_total += nsp_dfg_loss.item()
    
    return {
        'total_loss': total_loss / len(data_loader),
        'mlm_loss': mlm_loss_total / len(data_loader),
        'nsp_cfg_loss': nsp_cfg_loss_total / len(data_loader),
        'nsp_dfg_loss': nsp_dfg_loss_total / len(data_loader)
    }


def main():
    parser = argparse.ArgumentParser(description="Train Baseline BERT (no address embeddings)")
    
    # Data paths
    parser.add_argument('--cfg_train', required=True, help='Path to CFG training data')
    parser.add_argument('--dfg_train', required=True, help='Path to DFG training data')
    parser.add_argument('--vocab', required=True, help='Path to vocabulary file')
    parser.add_argument('--data_percentage', type=float, default=1.0, help='Percentage of data to use')
    parser.add_argument('--train_split', type=float, default=0.9, help='Train/val split ratio')
    
    # Model config (MUST match address-aware model)
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--attn_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Pre-trained initialization
    parser.add_argument('--palmtree_checkpoint', help='Path to pre-trained PalmTree checkpoint for initialization')
    
    # Training config
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='Masking probability')
    parser.add_argument('--nsp_prob', type=float, default=0.5, help='NSP negative probability')
    
    # Output
    parser.add_argument('--output_dir', default='output_baseline', help='Output directory')
    parser.add_argument('--log_freq', type=int, default=50, help='Logging frequency')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    
    # Hardware
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print(f"Loading vocabulary from: {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = PairedBaselineDataset(
        cfg_corpus_path=args.cfg_train,
        dfg_corpus_path=args.dfg_train,
        vocab=vocab,
        seq_len=args.seq_len,
        on_memory=True,
        nsp_prob=args.nsp_prob,
        mask_prob=args.mask_prob,
        data_percentage=args.data_percentage,
        train_split=args.train_split,
        is_train=True
    )
    
    print("Creating validation dataset...")
    val_dataset = PairedBaselineDataset(
        cfg_corpus_path=args.cfg_train,
        dfg_corpus_path=args.dfg_train,
        vocab=vocab,
        seq_len=args.seq_len,
        on_memory=True,
        nsp_prob=args.nsp_prob,
        mask_prob=args.mask_prob,
        data_percentage=args.data_percentage,
        train_split=args.train_split,
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Train dataset: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val dataset: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Load pre-trained PalmTree weights if provided
    pretrained_weights = None
    if args.palmtree_checkpoint:
        print(f"\nLoading pre-trained PalmTree from: {args.palmtree_checkpoint}")
        pretrained_bert = torch.load(args.palmtree_checkpoint, map_location=device, weights_only=False)
        
        if hasattr(pretrained_bert, 'state_dict'):
            pretrained_state = pretrained_bert.state_dict()
        else:
            pretrained_state = pretrained_bert
        
        # Extract relevant weights for baseline model
        pretrained_weights = {
            'token_embedding': pretrained_state.get('embedding.token.weight'),
            'segment_embedding': pretrained_state.get('embedding.segment.weight'),
            'transformer_blocks': {k: v for k, v in pretrained_state.items() if k.startswith('transformer_blocks.')}
        }
        print(f"[BASELINE] Loaded pre-trained weights for initialization")
    
    # Create model (Baseline BERT - NO address embeddings)
    print("\nCreating Baseline BERT model (NO address embeddings)...")
    model = create_baseline_model(
        vocab_size=vocab_size,
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout
    )
    
    # Initialize with pre-trained weights if available
    if pretrained_weights is not None:
        model_state = model.state_dict()
        
        # Load token embedding
        if pretrained_weights['token_embedding'] is not None:
            if model_state['bert.embedding.token.weight'].shape == pretrained_weights['token_embedding'].shape:
                model_state['bert.embedding.token.weight'] = pretrained_weights['token_embedding']
                print("[BASELINE] Initialized token embeddings from PalmTree")
        
        # Load segment embedding
        if pretrained_weights['segment_embedding'] is not None:
            if 'bert.embedding.segment.weight' in model_state:
                if model_state['bert.embedding.segment.weight'].shape == pretrained_weights['segment_embedding'].shape:
                    model_state['bert.embedding.segment.weight'] = pretrained_weights['segment_embedding']
                    print("[BASELINE] Initialized segment embeddings from PalmTree")
        
        # Load transformer blocks
        for k, v in pretrained_weights['transformer_blocks'].items():
            new_key = f'bert.{k}'
            if new_key in model_state and model_state[new_key].shape == v.shape:
                model_state[new_key] = v
        print(f"[BASELINE] Initialized transformer blocks from PalmTree")
        
        model.load_state_dict(model_state)
    
    model = model.to(device)
    
    # Multi-GPU
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.log_freq)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device)
        
        # Log
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f} "
              f"(MLM: {train_metrics['mlm_loss']:.4f}, "
              f"NSP_CFG: {train_metrics['nsp_cfg_loss']:.4f}, "
              f"NSP_DFG: {train_metrics['nsp_dfg_loss']:.4f})")
        print(f"  Val Loss:   {val_metrics['total_loss']:.4f} "
              f"(MLM: {val_metrics['mlm_loss']:.4f}, "
              f"NSP_CFG: {val_metrics['nsp_cfg_loss']:.4f}, "
              f"NSP_DFG: {val_metrics['nsp_dfg_loss']:.4f})")
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics
        })
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'baseline_bert.ep{epoch}')
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['total_loss'],
            'val_loss': val_metrics['total_loss']
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            
            # Save full model state
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss']
            }, best_model_path)
            
            # Save BERT encoder separately (for embedding extraction like original PalmTree)
            best_bert_path = os.path.join(args.output_dir, 'best_bert.pt')
            torch.save(model_to_save.bert, best_bert_path)
            
            print(f"  ✓ New best model saved: {best_model_path}")
            print(f"  ✓ BERT encoder saved: {best_bert_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
