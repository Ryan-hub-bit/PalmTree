"""
Training script for Address-Aware PalmTree
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(__file__))

import config
from data_loader import create_dataloaders
from models.addr_palmtree import AddressAwarePalmTree, load_pretrained_palmtree


class MultiTaskLoss(nn.Module):
    """Combined loss for multiple tasks"""
    
    def __init__(self, task_weights=None):
        super().__init__()
        self.task_weights = task_weights or config.TASK_WEIGHTS
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=1)  # Ignore PAD token
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with task predictions
            targets: dict with ground truth
        """
        losses = {}
        total_loss = 0
        
        # Next BB prediction loss
        if 'next_bb_logits' in outputs:
            next_bb_loss = self.ce_loss(
                outputs['next_bb_logits'].view(-1, outputs['next_bb_logits'].size(-1)),
                targets['next_bb_ids'].view(-1)
            )
            losses['next_bb'] = next_bb_loss
            total_loss += self.task_weights['next_bb'] * next_bb_loss
        
        # Address type classification loss
        if 'addr_type_logits' in outputs:
            addr_type_loss = self.ce_loss(
                outputs['addr_type_logits'].view(-1, outputs['addr_type_logits'].size(-1)),
                targets['addr_type_labels'].view(-1)
            )
            losses['addr_type'] = addr_type_loss
            total_loss += self.task_weights['addr_type'] * addr_type_loss
        
        # Edge type classification loss
        if 'edge_type_logits' in outputs:
            edge_type_loss = self.ce_loss(
                outputs['edge_type_logits'],
                targets['edge_type']
            )
            losses['edge_type'] = edge_type_loss
            total_loss += self.task_weights['edge_type'] * edge_type_loss
        
        losses['total'] = total_loss
        return losses


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        address_encodings = batch['address_encodings'].to(device)
        addr_type_ids = batch['addr_type_ids'].to(device)
        edge_type = batch['edge_type'].to(device)
        
        # Forward pass with 3-level embeddings
        outputs = model(input_ids, attention_mask, address_encodings, addr_type_ids)
        
        # Prepare targets for next BB prediction
        # Shift target by 1 for autoregressive prediction
        next_bb_ids = input_ids[:, 1:].contiguous()
        next_bb_ids = torch.cat([next_bb_ids, torch.zeros_like(input_ids[:, :1])], dim=1)
        
        # Address type labels (already provided in addr_type_ids)
        addr_type_labels = addr_type_ids
        
        targets = {
            'next_bb_ids': next_bb_ids,
            'addr_type_labels': addr_type_labels,
            'edge_type': edge_type,
        }
        
        # Compute loss
        losses = criterion(outputs, targets)
        loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': loss.item(),
            'avg_loss': total_loss / (batch_idx + 1)
        })
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            address_encodings = batch['address_encodings'].to(device)
            addr_type_ids = batch['addr_type_ids'].to(device)
            edge_type = batch['edge_type'].to(device)
            
            outputs = model(input_ids, attention_mask, address_encodings, addr_type_ids)
            
            next_bb_ids = input_ids[:, 1:].contiguous()
            next_bb_ids = torch.cat([next_bb_ids, torch.zeros_like(input_ids[:, :1])], dim=1)
            addr_type_labels = addr_type_ids
            
            targets = {
                'next_bb_ids': next_bb_ids,
                'addr_type_labels': addr_type_labels,
                'edge_type': edge_type,
            }
            
            losses = criterion(outputs, targets)
            total_loss += losses['total'].item()
    
    return total_loss / len(dataloader)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load vocabulary and create dataloaders
    print('Loading data...')
    train_loader, val_loader, test_loader, palmtree_vocab, addr_vocab = create_dataloaders(
        bb_pairs_file=args.bb_pairs_file,
        vocab_file=args.vocab_file,
        batch_size=config.BATCH_SIZE
    )
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')
    
    # Get vocab size (handle both dict and WordVocab object)
    if hasattr(palmtree_vocab, '__len__'):
        vocab_size = len(palmtree_vocab)
    elif hasattr(palmtree_vocab, 'stoi'):
        vocab_size = len(palmtree_vocab.stoi)
    else:
        vocab_size = len(palmtree_vocab)
    
    print(f'PalmTree Vocab Size: {vocab_size}')
    print(f'Address Vocab Size: {len(addr_vocab)}')
    
    # Load pre-trained PalmTree
    print('Loading pre-trained PalmTree...')
    pretrained_model_path = os.path.join(config.PRETRAINED_MODEL_PATH, 'palmtree', 'transformer.ep19')
    palmtree_bert = load_pretrained_palmtree(
        model_path=pretrained_model_path,
        vocab_size=vocab_size,
        hidden=768,
        n_layers=12,
        attn_heads=12
    )
    
    # Create address-aware model
    model = AddressAwarePalmTree(palmtree_bert).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Setup optimization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = MultiTaskLoss()
    
    # Setup logging
    writer = SummaryWriter(os.path.join(config.OUTPUT_DIR, 'logs'))
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{config.NUM_EPOCHS}')
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f}')
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config.OUTPUT_DIR, 'best_model.pt'))
            print('Saved best model')
    
    # Test
    print('\nTesting...')
    checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Address-Aware PalmTree')
    parser.add_argument('--bb_pairs_file', type=str, required=True,
                        help='Path to BB pairs file')
    parser.add_argument('--vocab_file', type=str, 
                        default=os.path.join(config.PRETRAINED_MODEL_PATH, 'vocab.txt'),
                        help='Path to vocabulary file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    main(args)
