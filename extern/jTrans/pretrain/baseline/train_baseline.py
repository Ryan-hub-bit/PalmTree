"""
Baseline Training Script for jTrans Pretraining

Pure MLM + JTP (Jump-Target Prediction) without address-aware features.

Tasks:
1. MLM: Mask regular tokens (15%) and predict them
2. JTP: Mask jump tokens (20%) and predict target positions

Usage:
    python train_baseline.py \\
        --train_path /path/to/train.txt \\
        --test_path /path/to/test.txt \\
        --tokenizer_path /path/to/tokenizer \\
        --output_dir ./output_baseline \\
        --batch_size 32 \\
        --learning_rate 1e-4 \\
        --num_epochs 10
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import logging
import sys
import os
from tqdm import tqdm
from datetime import datetime
import json

# Add parent directory for jTrans imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model_baseline import create_baseline_model
from dataloader_baseline import create_baseline_dataloaders


def setup_logging(output_dir):
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'train_baseline_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, scheduler, device, logger):
    """Train for one epoch with MLM + JTP tasks."""
    model.train()
    total_mlm_loss = 0
    total_jtp_loss = 0
    total_loss = 0
    mlm_correct = 0
    jtp_correct = 0
    mlm_tokens = 0
    jtp_tokens = 0
    
    progress = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        mlm_labels = batch['mlm_labels'].to(device)
        jtp_labels = batch['jtp_labels'].to(device)
        
        # Forward pass
        mlm_logits, jtp_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Compute MLM loss (only on masked tokens)
        loss_fct = nn.CrossEntropyLoss()
        mlm_loss = loss_fct(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
        
        # Compute JTP loss (only on masked jump tokens)
        jtp_loss = loss_fct(jtp_logits.view(-1, jtp_logits.size(-1)), jtp_labels.view(-1))
        
        # Combined loss
        loss = mlm_loss + jtp_loss
        
        # Compute accuracies
        # MLM accuracy
        mlm_mask = (mlm_labels != -100)
        mlm_predictions = torch.argmax(mlm_logits, dim=-1)
        mlm_correct += ((mlm_predictions == mlm_labels) & mlm_mask).sum().item()
        mlm_tokens += mlm_mask.sum().item()
        
        # JTP accuracy
        jtp_mask = (jtp_labels != -100)
        jtp_predictions = torch.argmax(jtp_logits, dim=-1)
        jtp_correct += ((jtp_predictions == jtp_labels) & jtp_mask).sum().item()
        jtp_tokens += jtp_mask.sum().item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_mlm_loss += mlm_loss.item()
        total_jtp_loss += jtp_loss.item()
        total_loss += loss.item()
        
        # Update progress bar
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            mlm_acc = mlm_correct / mlm_tokens if mlm_tokens > 0 else 0
            jtp_acc = jtp_correct / jtp_tokens if jtp_tokens > 0 else 0
            progress.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'mlm_acc': f'{mlm_acc:.4f}',
                'jtp_acc': f'{jtp_acc:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
    
    avg_mlm_loss = total_mlm_loss / len(dataloader)
    avg_jtp_loss = total_jtp_loss / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    mlm_accuracy = mlm_correct / mlm_tokens if mlm_tokens > 0 else 0
    jtp_accuracy = jtp_correct / jtp_tokens if jtp_tokens > 0 else 0
    
    return avg_loss, avg_mlm_loss, avg_jtp_loss, mlm_accuracy, jtp_accuracy


def validate_epoch(model, dataloader, device, logger):
    """Validate for one epoch with MLM + JTP tasks."""
    model.eval()
    total_mlm_loss = 0
    total_jtp_loss = 0
    total_loss = 0
    mlm_correct = 0
    jtp_correct = 0
    mlm_tokens = 0
    jtp_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)
            jtp_labels = batch['jtp_labels'].to(device)
            
            # Forward pass
            mlm_logits, jtp_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Compute losses
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            jtp_loss = loss_fct(jtp_logits.view(-1, jtp_logits.size(-1)), jtp_labels.view(-1))
            loss = mlm_loss + jtp_loss
            
            # Compute accuracies
            # MLM accuracy
            mlm_mask = (mlm_labels != -100)
            mlm_predictions = torch.argmax(mlm_logits, dim=-1)
            mlm_correct += ((mlm_predictions == mlm_labels) & mlm_mask).sum().item()
            mlm_tokens += mlm_mask.sum().item()
            
            # JTP accuracy
            jtp_mask = (jtp_labels != -100)
            jtp_predictions = torch.argmax(jtp_logits, dim=-1)
            jtp_correct += ((jtp_predictions == jtp_labels) & jtp_mask).sum().item()
            jtp_tokens += jtp_mask.sum().item()
            
            total_mlm_loss += mlm_loss.item()
            total_jtp_loss += jtp_loss.item()
            total_loss += loss.item()
    
    avg_mlm_loss = total_mlm_loss / len(dataloader)
    avg_jtp_loss = total_jtp_loss / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    mlm_accuracy = mlm_correct / mlm_tokens if mlm_tokens > 0 else 0
    jtp_accuracy = jtp_correct / jtp_tokens if jtp_tokens > 0 else 0
    
    return avg_loss, avg_mlm_loss, avg_jtp_loss, mlm_accuracy, jtp_accuracy


def main():
    parser = argparse.ArgumentParser(description='Baseline jTrans Pretraining (MLM + JTP)')
    
    # Data paths
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--test_path', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                       help='Path to tokenizer directory')
    
    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=768,
                       help='Hidden size')
    parser.add_argument('--num_hidden_layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--num_attention_heads', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--intermediate_size', type=int, default=3072,
                       help='FFN intermediate size')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1,
                       help='Dropout probability')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=10000,
                       help='Warmup steps')
    parser.add_argument('--max_len', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--mlm_probability', type=float, default=0.15,
                       help='MLM masking probability')
    parser.add_argument('--jtp_probability', type=float, default=0.20,
                       help='JTP masking probability')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./output_baseline',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save model every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.output_dir)
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("jTrans BASELINE Pretraining (MLM + JTP)")
    logger.info("=" * 80)
    logger.info(f"Train data: {args.train_path}")
    logger.info(f"Test data: {args.test_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Max length: {args.max_len}")
    logger.info(f"MLM probability: {args.mlm_probability}")
    logger.info(f"JTP probability: {args.jtp_probability}")
    logger.info(f"Hidden size: {args.hidden_size}")
    logger.info(f"Layers: {args.num_hidden_layers}")
    logger.info(f"Attention heads: {args.num_attention_heads}")
    logger.info("=" * 80)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    vocab_size = len(tokenizer)
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, test_loader = create_baseline_dataloaders(
        train_path=args.train_path,
        test_path=args.test_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        mlm_probability=args.mlm_probability,
        jtp_probability=args.jtp_probability,
        num_workers=args.num_workers
    )
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("Creating baseline model...")
    model = create_baseline_model(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        max_position_embeddings=args.max_len
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    training_history = []
    
    logger.info("\nStarting training...")
    logger.info("=" * 80)
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_mlm_loss, train_jtp_loss, train_mlm_acc, train_jtp_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, logger
        )
        logger.info(f"Train - Loss: {train_loss:.4f}, MLM: {train_mlm_loss:.4f} (acc: {train_mlm_acc:.4f}), "
                   f"JTP: {train_jtp_loss:.4f} (acc: {train_jtp_acc:.4f})")
        
        # Validate
        val_loss, val_mlm_loss, val_jtp_loss, val_mlm_acc, val_jtp_acc = validate_epoch(
            model, test_loader, device, logger
        )
        logger.info(f"Val - Loss: {val_loss:.4f}, MLM: {val_mlm_loss:.4f} (acc: {val_mlm_acc:.4f}), "
                   f"JTP: {val_jtp_loss:.4f} (acc: {val_jtp_acc:.4f})")
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_mlm_loss': train_mlm_loss,
            'train_jtp_loss': train_jtp_loss,
            'train_mlm_acc': train_mlm_acc,
            'train_jtp_acc': train_jtp_acc,
            'val_loss': val_loss,
            'val_mlm_loss': val_mlm_loss,
            'val_jtp_loss': val_jtp_loss,
            'val_mlm_acc': val_mlm_acc,
            'val_jtp_acc': val_jtp_acc
        })
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_dir = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model
            model.bert.save_pretrained(checkpoint_dir)
            
            # Save training info
            training_info = {
                'epoch': epoch + 1,
                'mode': 'baseline',
                'train_loss': train_loss,
                'train_mlm_loss': train_mlm_loss,
                'train_jtp_loss': train_jtp_loss,
                'val_loss': val_loss,
                'val_mlm_loss': val_mlm_loss,
                'val_jtp_loss': val_jtp_loss,
                'architecture': {
                    'hidden_size': args.hidden_size,
                    'num_hidden_layers': args.num_hidden_layers,
                    'num_attention_heads': args.num_attention_heads,
                    'vocab_size': vocab_size
                }
            }
            with open(os.path.join(checkpoint_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=2)
            
            logger.info(f"✓ Checkpoint saved to {checkpoint_dir}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_dir = os.path.join(args.output_dir, 'best_model')
            os.makedirs(best_model_dir, exist_ok=True)
            
            model.bert.save_pretrained(best_model_dir)
            
            best_info = {
                'epoch': epoch + 1,
                'mode': 'baseline',
                'best_val_loss': best_val_loss,
                'val_mlm_acc': val_mlm_acc,
                'val_jtp_acc': val_jtp_acc,
                'architecture': {
                    'hidden_size': args.hidden_size,
                    'num_hidden_layers': args.num_hidden_layers,
                    'num_attention_heads': args.num_attention_heads,
                    'vocab_size': vocab_size
                }
            }
            with open(os.path.join(best_model_dir, 'training_info.json'), 'w') as f:
                json.dump(best_info, f, indent=2)
            
            logger.info(f"✓ Best model saved! Val loss: {best_val_loss:.4f}")
    
    # Save training history
    history_file = os.path.join(args.output_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Training history saved to {history_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
