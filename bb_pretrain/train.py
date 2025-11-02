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
from streaming_data_loader import create_streaming_dataloader
from models.addr_palmtree import AddressAwarePalmTree, load_pretrained_palmtree


class MultiTaskLoss(nn.Module):
    """Combined loss for multiple tasks focused on address semantics"""
    
    def __init__(self, task_weights=None):
        super().__init__()
        self.task_weights = task_weights or config.TASK_WEIGHTS
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore masked positions
        
    def forward(self, outputs, targets, model=None):
        """
        Args:
            outputs: dict with task predictions
            targets: dict with ground truth
            model: model instance for computing address distance
        """
        losses = {}
        total_loss = 0
        
        # Task 1: Masked Address Type Prediction (PRIMARY TASK)
        # Only compute loss on masked positions
        if 'addr_type_logits' in outputs and 'masked_addr_type_labels' in targets:
            addr_type_loss = self.ce_loss(
                outputs['addr_type_logits'].view(-1, outputs['addr_type_logits'].size(-1)),
                targets['masked_addr_type_labels'].view(-1)
            )
            losses['addr_type'] = addr_type_loss
            total_loss += self.task_weights['addr_type'] * addr_type_loss
        
        # Task 2: Edge Type Classification
        if 'edge_type_logits' in outputs and 'edge_type' in targets:
            edge_type_loss = self.ce_loss(
                outputs['edge_type_logits'],
                targets['edge_type']
            )
            losses['edge_type'] = edge_type_loss
            total_loss += self.task_weights['edge_type'] * edge_type_loss
        
        # Task 3: Address Distance Prediction
        if model is not None and 'addr_pairs' in targets:
            # Compute distance logits for sampled address pairs
            addr_pairs = targets['addr_pairs']  # [num_pairs, 2] positions
            batch_indices = targets['addr_batch_indices']  # [num_pairs] batch indices
            if addr_pairs.size(0) > 0:
                distance_logits = model.compute_address_distance_logits(
                    outputs['hidden_states'],
                    batch_indices,
                    addr_pairs[:, 0],
                    addr_pairs[:, 1]
                )
                distance_loss = self.ce_loss(
                    distance_logits,
                    targets['addr_distance_labels']
                )
                losses['addr_distance'] = distance_loss
                total_loss += self.task_weights['addr_distance'] * distance_loss
        
        # Task 4: Next BB prediction (optional, lower weight)
        if 'next_bb_logits' in outputs and 'next_bb_ids' in targets:
            next_bb_loss = self.ce_loss(
                outputs['next_bb_logits'].view(-1, outputs['next_bb_logits'].size(-1)),
                targets['next_bb_ids'].view(-1)
            )
            losses['next_bb'] = next_bb_loss
            total_loss += self.task_weights['next_bb'] * next_bb_loss
        
        losses['total'] = total_loss
        return losses


def mask_address_types(addr_type_ids, mask_prob=0.15):
    """
    Mask address types for prediction (like BERT masked LM)
    
    Args:
        addr_type_ids: [B, L] - address type IDs
        mask_prob: probability of masking each address
    
    Returns:
        masked_ids: [B, L] - address types with some masked (set to unknown=5)
        labels: [B, L] - original types (-100 for non-masked positions)
    """
    masked_ids = addr_type_ids.clone()
    labels = torch.full_like(addr_type_ids, -100)  # -100 = ignore in loss
    
    # Only mask positions that actually have addresses (not unknown=5)
    has_address = (addr_type_ids != 5)  # 5 = unknown/no-address
    
    # Random mask
    mask = (torch.rand_like(addr_type_ids.float()) < mask_prob) & has_address
    
    # Set masked positions to unknown
    masked_ids[mask] = 5  # unknown type
    labels[mask] = addr_type_ids[mask]  # store original for loss
    
    return masked_ids, labels


def sample_address_pairs(addr_type_ids, address_values=None, num_pairs=1):
    """
    Sample pairs of addresses for distance prediction
    
    Args:
        addr_type_ids: [B, L] - address type IDs
        address_values: [B, L] - actual address values (optional)
        num_pairs: number of pairs to sample per batch
    
    Returns:
        pairs: [num_pairs, 2] - positions of address pairs
        batch_indices: [num_pairs] - which batch each pair belongs to
        distance_labels: [num_pairs] - distance class (0=same_bb, 1=near, 2=medium, 3=far)
    """
    batch_size, seq_len = addr_type_ids.shape
    
    # Find positions with addresses
    has_address = (addr_type_ids != 5)  # not unknown
    
    pairs_list = []
    batch_indices_list = []
    labels_list = []
    
    for b in range(min(batch_size, num_pairs)):  # Sample from first few batches
        addr_positions = torch.where(has_address[b])[0]
        
        if len(addr_positions) < 2:
            continue
        
        # Sample two random address positions
        indices = torch.randperm(len(addr_positions))[:2]
        pos1, pos2 = addr_positions[indices[0]], addr_positions[indices[1]]
        
        # Determine distance class
        type1, type2 = addr_type_ids[b, pos1].item(), addr_type_ids[b, pos2].item()
        
        # Simple heuristic for distance (can be improved with actual address values)
        if type1 == 2 and type2 == 3:  # addr_start and addr_end
            distance_class = 0  # same_bb
        elif type1 == type2:
            distance_class = 1  # near (same type, likely nearby)
        elif (type1 == 0 or type2 == 0):  # involves code address
            distance_class = 2  # medium
        else:
            distance_class = 3  # far
        
        pairs_list.append(torch.tensor([pos1, pos2]))
        batch_indices_list.append(b)
        labels_list.append(distance_class)
    
    if len(pairs_list) == 0:
        return None, None, None
    
    pairs = torch.stack(pairs_list)
    batch_indices = torch.tensor(batch_indices_list, dtype=torch.long)
    labels = torch.tensor(labels_list, dtype=torch.long)
    
    return pairs, batch_indices, labels


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
        
        # Mask some address types for prediction (like BERT masked LM)
        masked_addr_type_ids, masked_labels = mask_address_types(
            addr_type_ids, 
            mask_prob=config.ADDR_MASK_PROB
        )
        masked_addr_type_ids = masked_addr_type_ids.to(device)
        masked_labels = masked_labels.to(device)
        
        # Sample address pairs for distance prediction
        addr_pairs, batch_indices, distance_labels = sample_address_pairs(
            addr_type_ids,
            batch.get('address_values', None)
        )
        if addr_pairs is not None:
            addr_pairs = addr_pairs.to(device)
            batch_indices = batch_indices.to(device)
            distance_labels = distance_labels.to(device)
        
        # Forward pass with 3-level embeddings (use masked address types)
        outputs = model(input_ids, attention_mask, address_encodings, masked_addr_type_ids)
        
        # Prepare targets
        next_bb_ids = input_ids[:, 1:].contiguous()
        next_bb_ids = torch.cat([next_bb_ids, torch.full_like(input_ids[:, :1], -100)], dim=1)
        
        targets = {
            'masked_addr_type_labels': masked_labels,
            'edge_type': edge_type,
            'next_bb_ids': next_bb_ids,
        }
        
        if addr_pairs is not None:
            targets['addr_pairs'] = addr_pairs
            targets['addr_batch_indices'] = batch_indices
            targets['addr_distance_labels'] = distance_labels
        
        # Compute loss (pass model for address distance computation)
        losses = criterion(outputs, targets, model=model)
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
            
            # Mask address types (same as training)
            masked_addr_type_ids, masked_labels = mask_address_types(
                addr_type_ids, 
                mask_prob=config.ADDR_MASK_PROB
            )
            masked_addr_type_ids = masked_addr_type_ids.to(device)
            masked_labels = masked_labels.to(device)
            
            # Sample address pairs
            addr_pairs, distance_labels = sample_address_pairs(
                addr_type_ids,
                batch.get('address_values', None)
            )
            if addr_pairs is not None:
                addr_pairs = addr_pairs.to(device)
                distance_labels = distance_labels.to(device)
            
            outputs = model(input_ids, attention_mask, address_encodings, masked_addr_type_ids)
            
            next_bb_ids = input_ids[:, 1:].contiguous()
            next_bb_ids = torch.cat([next_bb_ids, torch.full_like(input_ids[:, :1], -100)], dim=1)
            
            targets = {
                'masked_addr_type_labels': masked_labels,
                'edge_type': edge_type,
                'next_bb_ids': next_bb_ids,
            }
            
            if addr_pairs is not None:
                targets['addr_pairs'] = addr_pairs
                targets['addr_distance_labels'] = distance_labels
            
            losses = criterion(outputs, targets, model=model)
            total_loss += losses['total'].item()
    
    return total_loss / len(dataloader)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load vocabulary and create dataloaders
    print('Loading data...')
    
    if args.use_streaming:
        # Use streaming dataloader for large files
        print("Using streaming dataloader (memory efficient)")
        train_loader = create_streaming_dataloader(
            bb_pairs_file=args.bb_pairs_file,
            vocab_file=args.vocab_file,
            batch_size=config.BATCH_SIZE,
            num_workers=4,
            shuffle_buffer_size=10000
        )
        # For streaming, we'll skip val/test for now (or create separate streaming loaders)
        val_loader = None
        test_loader = None
        palmtree_vocab = train_loader.dataset.palmtree_vocab
        addr_vocab = train_loader.dataset.addr_vocab
        print(f'Train: Streaming from file')
    else:
        # Regular dataloader (loads everything into memory)
        print("Using regular dataloader (loads into memory)")
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
        
        # Validate (skip if streaming without val set)
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f'Val Loss: {val_loss:.4f}')
        else:
            val_loss = train_loss  # Use train loss if no val set
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        if val_loader is not None:
            writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config.OUTPUT_DIR, 'best_model_sum.pt'))
            print('Saved best model (sum fusion)')
    
    # Test (skip if streaming without test set)
    if test_loader is not None:
        print('\nTesting...')
        checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model_sum.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.4f}')
    else:
        print('\nSkipping test (no test set for streaming mode)')
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Address-Aware PalmTree')
    parser.add_argument('--bb_pairs_file', type=str, required=True,
                        help='Path to BB pairs file')
    parser.add_argument('--vocab_file', type=str, 
                        default=os.path.join(config.PRETRAINED_MODEL_PATH, 'vocab.txt'),
                        help='Path to vocabulary file')
    parser.add_argument('--use_streaming', action='store_true',
                        help='Use streaming dataloader for large files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    main(args)
