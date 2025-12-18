"""
Training script for Function Similarity Fine-tuning

Fine-tunes a pre-trained AddressAwareBERT model for function similarity
using contrastive learning on funcsim dataset.
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
import os
import importlib.util

# Add strupos to path first
strupos_path = os.path.join(os.path.dirname(__file__), '..', '..', 'strupos')
if strupos_path not in sys.path:
    sys.path.insert(0, strupos_path)

# Now import from strupos
from vocab import WordVocab

# Import local modules using importlib to avoid conflicts
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load local model.py
spec = importlib.util.spec_from_file_location("funcsim_model", os.path.join(current_dir, "model.py"))
funcsim_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(funcsim_model)

# Load local dataloader.py
spec = importlib.util.spec_from_file_location("funcsim_dataloader", os.path.join(current_dir, "dataloader.py"))
funcsim_dataloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(funcsim_dataloader)

# Extract classes
FunctionSimilarityModel = funcsim_model.FunctionSimilarityModel
ContrastiveLoss = funcsim_model.ContrastiveLoss
FunctionSimilarityDataset = funcsim_dataloader.FunctionSimilarityDataset


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length instruction_boundaries lists.
    
    Since different functions have different numbers of instructions, we keep
    num_instructions and instruction_boundaries as lists instead of tensors.
    """
    # Separate the dictionary items
    func1_input = torch.stack([item['func1_input'] for item in batch])
    func1_segment = torch.stack([item['func1_segment'] for item in batch])
    func1_binary_pos = torch.stack([item['func1_binary_pos'] for item in batch])
    func1_function_pos = torch.stack([item['func1_function_pos'] for item in batch])
    func1_bb_pos = torch.stack([item['func1_bb_pos'] for item in batch])
    func1_var_offsets = torch.stack([item['func1_var_offsets'] for item in batch])
    func1_num_instructions = [item['func1_num_instructions'] for item in batch]  # Keep as list
    func1_boundaries = [item['func1_boundaries'] for item in batch]  # Keep as list
    
    func2_input = torch.stack([item['func2_input'] for item in batch])
    func2_segment = torch.stack([item['func2_segment'] for item in batch])
    func2_binary_pos = torch.stack([item['func2_binary_pos'] for item in batch])
    func2_function_pos = torch.stack([item['func2_function_pos'] for item in batch])
    func2_bb_pos = torch.stack([item['func2_bb_pos'] for item in batch])
    func2_var_offsets = torch.stack([item['func2_var_offsets'] for item in batch])
    func2_num_instructions = [item['func2_num_instructions'] for item in batch]  # Keep as list
    func2_boundaries = [item['func2_boundaries'] for item in batch]  # Keep as list
    
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'func1_input': func1_input,
        'func1_segment': func1_segment,
        'func1_binary_pos': func1_binary_pos,
        'func1_function_pos': func1_function_pos,
        'func1_bb_pos': func1_bb_pos,
        'func1_var_offsets': func1_var_offsets,
        'func1_num_instructions': func1_num_instructions,
        'func1_boundaries': func1_boundaries,
        'func2_input': func2_input,
        'func2_segment': func2_segment,
        'func2_binary_pos': func2_binary_pos,
        'func2_function_pos': func2_function_pos,
        'func2_bb_pos': func2_bb_pos,
        'func2_var_offsets': func2_var_offsets,
        'func2_num_instructions': func2_num_instructions,
        'func2_boundaries': func2_boundaries,
        'label': labels
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
        func1_input = batch['func1_input'].to(device)
        func1_segment = batch['func1_segment'].to(device)
        func1_bin_pos = batch['func1_binary_pos'].to(device)
        func1_func_pos = batch['func1_function_pos'].to(device)
        func1_bb_pos = batch['func1_bb_pos'].to(device)
        func1_var_offsets = batch['func1_var_offsets'].to(device)
        
        func2_input = batch['func2_input'].to(device)
        func2_segment = batch['func2_segment'].to(device)
        func2_bin_pos = batch['func2_binary_pos'].to(device)
        func2_func_pos = batch['func2_function_pos'].to(device)
        func2_bb_pos = batch['func2_bb_pos'].to(device)
        func2_var_offsets = batch['func2_var_offsets'].to(device)
        
        labels = batch['label'].to(device)
        
        # Forward pass
        emb1 = model(func1_input, func1_segment, func1_bin_pos, func1_func_pos, func1_bb_pos, func1_var_offsets)
        emb2 = model(func2_input, func2_segment, func2_bin_pos, func2_func_pos, func2_bb_pos, func2_var_offsets)
        
        # Compute loss
        loss = criterion(emb1, emb2, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Compute accuracy (cosine similarity > 0.5 for positive, < 0.5 for negative)
        similarity = model.compute_similarity(emb1, emb2, metric='cosine')
        predictions = (similarity > 0.5).long()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        total_loss += loss.item()
        
        # Update progress bar
        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
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
            func1_input = batch['func1_input'].to(device)
            func1_segment = batch['func1_segment'].to(device)
            func1_bin_pos = batch['func1_binary_pos'].to(device)
            func1_func_pos = batch['func1_function_pos'].to(device)
            func1_bb_pos = batch['func1_bb_pos'].to(device)
            func1_var_offsets = batch['func1_var_offsets'].to(device)
            
            func2_input = batch['func2_input'].to(device)
            func2_segment = batch['func2_segment'].to(device)
            func2_bin_pos = batch['func2_binary_pos'].to(device)
            func2_func_pos = batch['func2_function_pos'].to(device)
            func2_bb_pos = batch['func2_bb_pos'].to(device)
            func2_var_offsets = batch['func2_var_offsets'].to(device)
            
            labels = batch['label'].to(device)
            
            # Forward pass
            emb1 = model(func1_input, func1_segment, func1_bin_pos, func1_func_pos, func1_bb_pos, func1_var_offsets)
            emb2 = model(func2_input, func2_segment, func2_bin_pos, func2_func_pos, func2_bb_pos, func2_var_offsets)
            
            # Compute loss
            loss = criterion(emb1, emb2, labels)
            
            # Compute accuracy
            similarity = model.compute_similarity(emb1, emb2, metric='cosine')
            predictions = (similarity > 0.5).long()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Fine-tune for function similarity")
    
    # Data
    parser.add_argument("--function_blocks", type=str, required=True, help="Path to function_blocks.json")
    parser.add_argument("--funcsim_pairs", type=str, required=True, help="Path to funcsim_pairs.json")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab.pkl")
    
    # Model
    parser.add_argument("--pretrained_bert", type=str, required=True, help="Path to pre-trained BERT checkpoint")
    parser.add_argument("--hidden", type=int, default=768, help="Hidden size")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--attn_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Function embedding dimension")
    parser.add_argument("--freeze_bert", action="store_true", help="Freeze BERT weights")
    parser.add_argument("--no_address", action="store_true", help="Disable address embeddings")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=60, help="Maximum sequence length")
    parser.add_argument("--negative_samples", type=int, default=3, help="Negative samples per positive")
    parser.add_argument("--margin", type=float, default=1.0, help="Contrastive loss margin")
    parser.add_argument("--dataset_fraction", type=float, default=1.0, help="Fraction of dataset to use (e.g., 0.2 for 20%)")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training split ratio (rest is test)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio (from training set)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="../../output/funcsim", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="../../log/funcsim", help="Log directory")
    parser.add_argument("--experiment_name", type=str, default="funcsim", help="Experiment name")
    parser.add_argument("--task_name", type=str, default="mlm", help="Task identifier for caching (e.g., 'mlm', 'mlm_addr_var')")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger = setup_logging(args.log_dir, args.experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Function Similarity Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Pre-trained BERT: {args.pretrained_bert}")
    logger.info(f"Function blocks: {args.function_blocks}")
    logger.info(f"Funcsim pairs: {args.funcsim_pairs}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Save arguments
    args_file = os.path.join(args.output_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Arguments saved to: {args_file}")
    
    # Load vocabulary
    logger.info(f"Loading vocabulary from {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset
    logger.info("Creating dataset...")
    full_dataset = FunctionSimilarityDataset(
        function_blocks_file=args.function_blocks,
        funcsim_pairs_file=args.funcsim_pairs,
        vocab=vocab,
        seq_len=args.seq_len,
        negative_samples=args.negative_samples
    )
    
    # Apply dataset fraction if specified (for faster experiments)
    if args.dataset_fraction < 1.0:
        original_size = len(full_dataset)
        subset_size = int(original_size * args.dataset_fraction)
        remaining_size = original_size - subset_size
        
        logger.info(f"Using {args.dataset_fraction*100:.1f}% of dataset: {subset_size}/{original_size} samples")
        
        full_dataset, _ = random_split(
            full_dataset,
            [subset_size, remaining_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    # Split into train+val and test
    test_size = int(len(full_dataset) * (1 - args.train_split))
    train_val_size = len(full_dataset) - test_size
    train_val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Further split train_val into train and val
    val_size = int(len(train_val_dataset) * args.val_split)
    train_size = len(train_val_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_val_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Dataset samples: {len(full_dataset)}")
    logger.info(f"Training samples: {len(train_dataset)} ({len(train_dataset)/len(full_dataset)*100:.1f}%)")
    logger.info(f"Validation samples: {len(val_dataset)} ({len(val_dataset)/len(full_dataset)*100:.1f}%)")
    logger.info(f"Test samples: {len(test_dataset)} ({len(test_dataset)/len(full_dataset)*100:.1f}%)")
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Save test indices for later evaluation
    test_indices_file = os.path.join(args.output_dir, 'test_indices.json')
    test_indices = test_dataset.indices if hasattr(test_dataset, 'indices') else list(range(len(test_dataset)))
    with open(test_indices_file, 'w') as f:
        json.dump({'test_indices': test_indices, 'test_size': len(test_dataset)}, f)
    logger.info(f"Test indices saved to: {test_indices_file}")
    
    # STEP 1: Detect what the pretrained checkpoint has BEFORE creating model
    logger.info(f"Detecting capabilities of pretrained checkpoint: {args.pretrained_bert}")
    checkpoint = torch.load(args.pretrained_bert, map_location='cpu', weights_only=False)
    
    # Extract state dict from checkpoint
    if isinstance(checkpoint, dict):
        if 'bert_state_dict' in checkpoint:
            state_dict = checkpoint['bert_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()
    
    # Detect if checkpoint has address/var embeddings
    has_address = any('address_position' in k or 'address_pos' in k for k in state_dict.keys())
    has_var = any('var_position' in k or 'var_offset' in k for k in state_dict.keys())
    has_address_var = has_address and has_var
    
    logger.info("=" * 80)
    logger.info("PRETRAINED CHECKPOINT CAPABILITIES")
    logger.info("=" * 80)
    logger.info(f"Checkpoint has address embeddings: {has_address}")
    logger.info(f"Checkpoint has var embeddings: {has_var}")
    logger.info("=" * 80)
    
    # Override --no_address flag based on what checkpoint actually has
    use_address_embedding = has_address_var
    if args.no_address and has_address_var:
        logger.warning("--no_address was set but checkpoint HAS address/var embeddings!")
        logger.warning("Ignoring --no_address flag to match checkpoint")
    elif not args.no_address and not has_address_var:
        logger.warning("--no_address was NOT set but checkpoint LACKS address/var embeddings!")
        logger.warning("Disabling address embeddings to match checkpoint")
    
    # STEP 2: Create model matching checkpoint capabilities
    logger.info("Creating model to match checkpoint...")
    model = FunctionSimilarityModel(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        max_len=args.seq_len,
        use_address_embedding=use_address_embedding,
        use_var_embedding=use_address_embedding,  # Keep them together
        embedding_dim=args.embedding_dim,
        freeze_bert=args.freeze_bert
    )
    
    # STEP 3: Load pretrained weights (should have no missing/unexpected keys now)
    logger.info(f"Loading pre-trained BERT from {args.pretrained_bert}")
    bert_capabilities = model.load_pretrained_bert(args.pretrained_bert)
    
    # STEP 4: Configure dataset based on what BERT has
    # Get the underlying dataset object (handles both Dataset and Subset)
    def get_base_dataset(ds):
        """Recursively get the base dataset from a Subset."""
        if hasattr(ds, 'dataset'):
            return get_base_dataset(ds.dataset)
        return ds
    
    base_dataset = get_base_dataset(full_dataset)
    base_dataset.use_address_var = has_address_var
    
    if has_address_var:
        logger.info("[INFO] BERT has address/var embeddings → Dataloader will use parsed address/var info")
    else:
        logger.info("[INFO] BERT does NOT have address/var embeddings → All positions/offsets masked to -1")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = ContrastiveLoss(margin=args.margin, metric='cosine')
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_acc = 0.0
    
    logger.info("\nStarting training...")
    logger.info("=" * 80)
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
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
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'task_name': args.task_name,  # Save task name for evaluation
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(args.output_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved! Val Acc: {val_acc:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
