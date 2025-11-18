"""
Training script for Address-Aware BERT pretraining

Trains a BERT model with address-aware positional embeddings on:
1. Masked Language Modeling (MLM)
2. Next Sentence Prediction (NSP)

Using the inline address format from cfg_address.py and dfg_address.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import sys
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from palmtree.dataset.vocab import WordVocab
from dataloader_paired import PairedAddressAwareDataset
from model import AddressAwareBERT, AddressAwareBERTForPretraining


def train_epoch(model, data_loader, optimizer, device, log_freq=100):
    """Train for one epoch with paired CFG+DFG samples."""
    model.train()
    
    total_loss = 0
    mlm_loss_total = 0
    nsp_cfg_loss_total = 0
    nsp_dfg_loss_total = 0
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    
    progress = tqdm(data_loader, desc="Training")
    
    for i, batch in enumerate(progress):
        # === Process CFG (MLM + NSP) ===
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
        
        # === Process DFG (NSP only) ===
        dfg_token_ids = batch['dfg_bert_input'].to(device)
        dfg_segment_labels = batch['dfg_segment_label'].to(device)
        dfg_binary_pos = batch['dfg_binary_pos'].to(device)
        dfg_function_pos = batch['dfg_function_pos'].to(device)
        dfg_bb_pos = batch['dfg_bb_pos'].to(device)
        dfg_nsp_labels = batch['dfg_is_next'].to(device)
        
        # DFG forward pass (NO MLM)
        _, dfg_nsp_output = model(
            dfg_token_ids, dfg_segment_labels,
            dfg_binary_pos, dfg_function_pos, dfg_bb_pos,
            corpus_type='dfg'
        )
        
        # DFG loss (NSP only)
        nsp_dfg_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels)
        
        # Combined loss: MLM(CFG) + NSP(CFG) + NSP(DFG)
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
        'nsp_dfg_loss': nsp_dfg_loss_total / len(data_loader),
    }


def validate(model, data_loader, device):
    """Validate the model on validation set with paired CFG+DFG samples."""
    model.eval()
    
    total_loss = 0
    mlm_loss_total = 0
    nsp_cfg_loss_total = 0
    nsp_dfg_loss_total = 0
    
    mlm_correct = 0
    mlm_total = 0
    nsp_cfg_correct = 0
    nsp_cfg_total = 0
    nsp_dfg_correct = 0
    nsp_dfg_total = 0
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # === Process CFG (MLM + NSP) ===
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
            
            # CFG accuracy
            mask = cfg_mlm_labels != -1
            if mask.any():
                mlm_pred = torch.argmax(cfg_mlm_output[mask], dim=-1)
                mlm_correct += (mlm_pred == cfg_mlm_labels[mask]).sum().item()
                mlm_total += mask.sum().item()
            
            nsp_cfg_pred = torch.argmax(cfg_nsp_output, dim=-1)
            nsp_cfg_correct += (nsp_cfg_pred == cfg_nsp_labels).sum().item()
            nsp_cfg_total += len(cfg_nsp_labels)
            
            # === Process DFG (NSP only) ===
            dfg_token_ids = batch['dfg_bert_input'].to(device)
            dfg_segment_labels = batch['dfg_segment_label'].to(device)
            dfg_binary_pos = batch['dfg_binary_pos'].to(device)
            dfg_function_pos = batch['dfg_function_pos'].to(device)
            dfg_bb_pos = batch['dfg_bb_pos'].to(device)
            dfg_nsp_labels = batch['dfg_is_next'].to(device)
            
            # DFG forward pass
            _, dfg_nsp_output = model(
                dfg_token_ids, dfg_segment_labels,
                dfg_binary_pos, dfg_function_pos, dfg_bb_pos,
                corpus_type='dfg'
            )
            
            # DFG loss
            nsp_dfg_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels)
            
            # DFG accuracy
            nsp_dfg_pred = torch.argmax(dfg_nsp_output, dim=-1)
            nsp_dfg_correct += (nsp_dfg_pred == dfg_nsp_labels).sum().item()
            nsp_dfg_total += len(dfg_nsp_labels)
            
            # Combined loss
            loss = mlm_loss + nsp_cfg_loss + nsp_dfg_loss
            
            # Accumulate losses
            total_loss += loss.item()
            mlm_loss_total += mlm_loss.item()
            nsp_cfg_loss_total += nsp_cfg_loss.item()
            nsp_dfg_loss_total += nsp_dfg_loss.item()
    
    # Calculate accuracies
    mlm_acc = mlm_correct / mlm_total if mlm_total > 0 else 0
    nsp_cfg_acc = nsp_cfg_correct / nsp_cfg_total if nsp_cfg_total > 0 else 0
    nsp_dfg_acc = nsp_dfg_correct / nsp_dfg_total if nsp_dfg_total > 0 else 0
    
    return {
        'total_loss': total_loss / len(data_loader),
        'mlm_loss': mlm_loss_total / len(data_loader),
        'nsp_cfg_loss': nsp_cfg_loss_total / len(data_loader),
        'nsp_dfg_loss': nsp_dfg_loss_total / len(data_loader),
        'mlm_acc': mlm_acc,
        'nsp_cfg_acc': nsp_cfg_acc,
        'nsp_dfg_acc': nsp_dfg_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    
    # Data args
    parser.add_argument("--cfg_train", type=str, required=True, help="Path to CFG training data")
    parser.add_argument("--dfg_train", type=str, required=True, help="Path to DFG training data")
    parser.add_argument("--cfg_val", type=str, help="Path to CFG validation data (optional)")
    parser.add_argument("--dfg_val", type=str, help="Path to DFG validation data (optional)")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--data_percentage", type=float, default=1.0, help="Percentage of dataset to use (0.0-1.0)")
    parser.add_argument("--train_split", type=float, default=0.9, help="Train/val split ratio (e.g., 0.9 = 90%% train, 10%% val)")
    
    # Model args
    parser.add_argument("--hidden", type=int, default=768, help="Hidden size")
    parser.add_argument("--layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--attn_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=512, help="Maximum sequence length")
    
    # Pre-trained PalmTree args
    parser.add_argument("--palmtree_checkpoint", type=str, help="Path to pre-trained PalmTree checkpoint")
    parser.add_argument("--freeze_token_emb", action="store_true", help="Freeze token embeddings from PalmTree")
    parser.add_argument("--freeze_position_emb", action="store_true", help="Freeze position embeddings from PalmTree")
    parser.add_argument("--freeze_segment_emb", action="store_true", help="Freeze segment embeddings from PalmTree")
    parser.add_argument("--freeze_transformer", action="store_true", help="Freeze transformer blocks from PalmTree")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Masking args
    parser.add_argument("--mask_prob", type=float, default=0.15, help="Probability of masking a token")
    parser.add_argument("--nsp_prob", type=float, default=0.5, help="Probability of negative NSP sample")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--save_freq", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--log_freq", type=int, default=100, help="Log every N batches")
    
    # Device args
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = PairedAddressAwareDataset(
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = None
    if args.cfg_val and args.dfg_val:
        # Use separate validation files if provided
        print("Creating validation dataset from separate files...")
        val_dataset = PairedAddressAwareDataset(
            cfg_corpus_path=args.cfg_val,
            dfg_corpus_path=args.dfg_val,
            vocab=vocab,
            seq_len=args.seq_len,
            on_memory=True,
            nsp_prob=args.nsp_prob,
            mask_prob=args.mask_prob,
            data_percentage=1.0,  # Use all validation data
            train_split=1.0,
            is_train=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    elif args.train_split < 1.0:
        # Automatically create validation set from training data
        print("Creating validation dataset (automatic split)...")
        val_dataset = PairedAddressAwareDataset(
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
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    # Create model
    print("Creating model...")
    
    # Load pre-trained PalmTree weights if provided
    pretrained_token_emb = None
    pretrained_position_emb = None
    pretrained_segment_emb = None
    pretrained_transformer = None
    
    if args.palmtree_checkpoint:
        print(f"Loading pre-trained PalmTree from: {args.palmtree_checkpoint}")
        from load_pretrained import load_palmtree_weights
        
        weights = load_palmtree_weights(args.palmtree_checkpoint, len(vocab), device)
        pretrained_token_emb = weights['token_embedding']
        pretrained_position_emb = weights['position_embedding']
        pretrained_segment_emb = weights['segment_embedding']
        pretrained_transformer = weights['transformer_state']
    
    bert = AddressAwareBERT(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        max_len=args.seq_len,
        pretrained_token_emb=pretrained_token_emb,
        pretrained_position_emb=pretrained_position_emb,
        pretrained_segment_emb=pretrained_segment_emb,
        pretrained_transformer=pretrained_transformer
    )
    
    # Freeze components based on flags
    if args.freeze_token_emb and pretrained_token_emb is not None:
        print("[INFO] Freezing token embeddings")
        bert.embedding.token_embedding.weight.requires_grad = False
    
    if args.freeze_position_emb and pretrained_position_emb is not None:
        print("[INFO] Freezing position embeddings")
        bert.embedding.position_embedding.weight.requires_grad = False
    
    if args.freeze_segment_emb and pretrained_segment_emb is not None:
        print("[INFO] Freezing segment embeddings")
        bert.embedding.segment_embedding.weight.requires_grad = False
    
    if args.freeze_transformer and pretrained_transformer is not None:
        print("[INFO] Freezing transformer blocks")
        for param in bert.transformer_blocks.parameters():
            param.requires_grad = False
    
    model = AddressAwareBERTForPretraining(bert, len(vocab))
    model = model.to(device)
    
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.log_freq)
        scheduler.step()
        
        print(f"Train Loss: {train_metrics['total_loss']:.4f} | "
              f"MLM: {train_metrics['mlm_loss']:.4f} | "
              f"NSP_CFG: {train_metrics['nsp_cfg_loss']:.4f} | "
              f"NSP_DFG: {train_metrics['nsp_dfg_loss']:.4f}")
        
        # Validate
        if val_loader is not None:
            val_metrics = validate(model, val_loader, device)
            print(f"Val Loss: {val_metrics['total_loss']:.4f} | "
                  f"MLM: {val_metrics['mlm_loss']:.4f} ({val_metrics['mlm_acc']:.2%}) | "
                  f"NSP_CFG: {val_metrics['nsp_cfg_loss']:.4f} ({val_metrics['nsp_cfg_acc']:.2%}) | "
                  f"NSP_DFG: {val_metrics['nsp_dfg_loss']:.4f} ({val_metrics['nsp_dfg_acc']:.2%})")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, os.path.join(args.output_dir, "best_model.pt"))
                print("Saved best model")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"))
            print(f"Saved checkpoint for epoch {epoch + 1}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
