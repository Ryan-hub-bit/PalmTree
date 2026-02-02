"""
jTrans Function Similarity Fine-tuning (strupos-style)

Fine-tunes a pre-trained AddressAwareBERT model for function similarity
using contrastive learning, following dstask/funcsim approach.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
import sys
import random
import numpy as np

# Import our self-contained model
from bert_model import FunctionSimilarityModel, ContrastiveLoss
from data_json import FunctionDataset_CL_AddressAware_JSON


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def custom_collate_fn(batch):
    """
    Custom collate function for address-aware triplets.
    Handles variable-length instruction boundaries.
    """
    # Unpack batch - each item is (anchor, positive, negative) × 7 fields
    anchors, positives, negatives = [], [], []
    
    for triplet in batch:
        # Each triplet contains processed data for anchor, pos, neg
        anchor, pos, neg = triplet
        anchors.append(anchor)
        positives.append(pos)
        negatives.append(neg)
    
    def stack_samples(samples):
        """Stack a list of samples into batch tensors."""
        return {
            'input_ids': torch.stack([s['input_ids'] for s in samples]),
            'segment_labels': torch.stack([s['segment_labels'] for s in samples]),
            'binary_pos': torch.stack([s['binary_pos'] for s in samples]),
            'function_pos': torch.stack([s['function_pos'] for s in samples]),
            'bb_pos': torch.stack([s['bb_pos'] for s in samples]),
            'var_offsets': torch.stack([s['var_offsets'] for s in samples]),
        }
    
    return {
        'anchor': stack_samples(anchors),
        'positive': stack_samples(positives),
        'negative': stack_samples(negatives)
    }


def setup_logging(log_dir, experiment_name):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{experiment_name}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress = tqdm(dataloader, desc="Training")
    
    for batch in progress:
        # Move to device
        anchor = {k: v.to(device) for k, v in batch['anchor'].items()}
        positive = {k: v.to(device) for k, v in batch['positive'].items()}
        negative = {k: v.to(device) for k, v in batch['negative'].items()}
        
        # Forward pass - get embeddings
        emb_anchor = model(
            anchor['input_ids'], anchor['segment_labels'],
            anchor['binary_pos'], anchor['function_pos'], anchor['bb_pos'],
            anchor['var_offsets']
        )
        
        emb_positive = model(
            positive['input_ids'], positive['segment_labels'],
            positive['binary_pos'], positive['function_pos'], positive['bb_pos'],
            positive['var_offsets']
        )
        
        emb_negative = model(
            negative['input_ids'], negative['segment_labels'],
            negative['binary_pos'], negative['function_pos'], negative['bb_pos'],
            negative['var_offsets']
        )
        
        # Compute triplet loss
        # Positive pairs should be similar (label=1)
        # Negative pairs should be dissimilar (label=0)
        loss_pos = criterion(emb_anchor, emb_positive, torch.ones(emb_anchor.size(0)).to(device))
        loss_neg = criterion(emb_anchor, emb_negative, torch.zeros(emb_anchor.size(0)).to(device))
        loss = loss_pos + loss_neg
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        
        # Compute accuracy (positive closer than negative?)
        sim_pos = torch.sum(emb_anchor * emb_positive, dim=1)
        sim_neg = torch.sum(emb_anchor * emb_negative, dim=1)
        correct += (sim_pos > sim_neg).sum().item()
        total += emb_anchor.size(0)
        
        progress.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device, logger):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move to device
            anchor = {k: v.to(device) for k, v in batch['anchor'].items()}
            positive = {k: v.to(device) for k, v in batch['positive'].items()}
            negative = {k: v.to(device) for k, v in batch['negative'].items()}
            
            # Forward pass
            emb_anchor = model(
                anchor['input_ids'], anchor['segment_labels'],
                anchor['binary_pos'], anchor['function_pos'], anchor['bb_pos'],
                anchor['var_offsets']
            )
            
            emb_positive = model(
                positive['input_ids'], positive['segment_labels'],
                positive['binary_pos'], positive['function_pos'], positive['bb_pos'],
                positive['var_offsets']
            )
            
            emb_negative = model(
                negative['input_ids'], negative['segment_labels'],
                negative['binary_pos'], negative['function_pos'], negative['bb_pos'],
                negative['var_offsets']
            )
            
            # Compute loss
            loss_pos = criterion(emb_anchor, emb_positive, torch.ones(emb_anchor.size(0)).to(device))
            loss_neg = criterion(emb_anchor, emb_negative, torch.zeros(emb_anchor.size(0)).to(device))
            loss = loss_pos + loss_neg
            
            total_loss += loss.item()
            
            # Accuracy
            sim_pos = torch.sum(emb_anchor * emb_positive, dim=1)
            sim_neg = torch.sum(emb_anchor * emb_negative, dim=1)
            correct += (sim_pos > sim_neg).sum().item()
            total += emb_anchor.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="jTrans Function Similarity Fine-tuning")
    
    # Data paths
    parser.add_argument('--func_blocks', type=str, required=True, help='Function blocks JSON file')
    parser.add_argument('--ground_truth', type=str, required=True, help='Ground truth JSON file')
    parser.add_argument('--vocab_path', type=str, required=True, help='Vocabulary path (.txt file)')
    
    # Pre-trained model
    parser.add_argument('--model_path', type=str, required=True, help='Pre-trained BERT checkpoint')
    
    # Output
    parser.add_argument('--output_path', type=str, required=True, help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='jtrans_funcsim', help='Experiment name')
    
    # Model config
    parser.add_argument('--hidden', type=int, default=768, help='Hidden size')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--attn_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Function embedding dimension')
    parser.add_argument('--max_len', type=int, default=512, help='Max sequence length')
    
    # Training config
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epoch', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--triplet_margin', type=float, default=0.2, help='Triplet margin')
    parser.add_argument('--freeze_bert', action='store_true', help='Freeze BERT weights')
    parser.add_argument('--freeze_cnt', type=int, default=-1, help='Number of layers to freeze (-1 for none)')
    
    # Data sampling
    parser.add_argument('--data_ratio', type=float, default=1.0, help='Ratio of data to use')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--val_split', type=float, default=0.125, help='Val split from train')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    log_dir = os.path.join(args.output_path, 'logs')
    logger = setup_logging(log_dir, args.experiment_name)
    
    logger.info("=" * 80)
    logger.info("jTrans Function Similarity Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Pre-trained model: {args.model_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info("=" * 80)
    
    # Load vocabulary
    from pretrain.address_aware.vocab import WordVocab
    logger.info(f"Loading vocabulary from {args.vocab_path}")
    vocab = WordVocab.load_vocab(args.vocab_path)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Load dataset
    logger.info("Loading dataset...")
    full_dataset = FunctionDataset_CL_AddressAware_JSON(
        tokenizer=None,  # We'll use vocab directly in dataset
        func_blocks_path=args.func_blocks,
        ground_truth_path=args.ground_truth,
        max_length=args.max_len,
        data_ratio=args.data_ratio
    )
    
    # Set vocab in dataset
    full_dataset.vocab = vocab
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_val_size = int(dataset_size * args.train_split)
    test_size = dataset_size - train_val_size
    
    train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size])
    
    val_size = int(train_val_size * args.val_split)
    train_size = train_val_size - val_size
    
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Auto-detect checkpoint capabilities
    logger.info(f"Loading checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'bert_state_dict' in checkpoint:
            state_dict = checkpoint['bert_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()
    
    # Detect address/var embeddings
    has_address = any('address_position' in k or 'address' in k for k in state_dict.keys())
    has_var = any('var_position' in k or 'var' in k for k in state_dict.keys())
    
    logger.info("=" * 80)
    logger.info("CHECKPOINT CAPABILITIES")
    logger.info("=" * 80)
    logger.info(f"Has address embeddings: {has_address}")
    logger.info(f"Has var embeddings: {has_var}")
    logger.info("=" * 80)
    
    # Create model
    logger.info("Creating model...")
    model = FunctionSimilarityModel(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        max_len=args.max_len,
        use_address_embedding=has_address,
        use_var_embedding=has_var,
        embedding_dim=args.embedding_dim,
        freeze_bert=args.freeze_bert
    )
    
    # Load pre-trained weights
    logger.info("Loading pre-trained BERT weights...")
    result = model.load_pretrained_bert(args.model_path)
    
    # Freeze layers if requested
    if args.freeze_cnt > 0:
        logger.info(f"Freezing first {args.freeze_cnt} transformer layers...")
        for i in range(args.freeze_cnt):
            if i < len(model.bert.transformer_blocks):
                for param in model.bert.transformer_blocks[i].parameters():
                    param.requires_grad = False
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = ContrastiveLoss(margin=args.triplet_margin, metric='cosine')
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    
    # Training loop
    best_val_acc = 0.0
    
    logger.info("\nStarting training...")
    logger.info("=" * 80)
    
    for epoch in range(args.epoch):
        logger.info(f"\nEpoch {epoch+1}/{args.epoch}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, logger)
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Update learning rate
        scheduler.step()
        logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        os.makedirs(args.output_path, exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        
        # Save latest
        latest_path = os.path.join(args.output_path, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.output_path, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved! Val Acc: {val_acc:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
