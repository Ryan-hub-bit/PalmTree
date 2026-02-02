"""
Address-Aware jTrans Training Script

Trains with hierarchical address embeddings instead of position=word trick.
Uses MLM + JTP tasks like baseline but with address-aware tokenization.
"""

import os
import sys

# GPU MUST be set BEFORE importing torch
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print(f"[INFO] Using GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    print("[WARNING] CUDA_VISIBLE_DEVICES not set!")
    print("[WARNING] Please run via shell script that sets GPU.")
    sys.exit(1)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
import json

# Import address-aware components
from model_addressaware import create_addressaware_model
from dataloader_addressaware import AddressAwareDataset
from vocab import WordVocab


def setup_logging(output_dir):
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'train_addressaware_{timestamp}.log')
    
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
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_mlm_loss = 0
    total_jtp_loss = 0
    
    mlm_correct = 0
    mlm_total = 0
    jtp_correct = 0
    jtp_total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # VALIDATION: Check all inputs on CPU before moving to GPU
        token_ids_cpu = batch['bert_input']
        segment_labels_cpu = batch['segment_label']
        mlm_labels_cpu = batch['bert_label']
        
        max_id = token_ids_cpu.max().item()
        min_id = token_ids_cpu.min().item()
        # Handle DataParallel wrapper
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        vocab_size = actual_model.bert.embeddings.token_embedding.num_embeddings
        
        max_seg = segment_labels_cpu.max().item()
        min_seg = segment_labels_cpu.min().item()
        segment_vocab_size = actual_model.bert.embeddings.segment_embedding.num_embeddings
        
        # Filter out ignore_index (-100) for MLM label validation
        valid_mlm_mask = (mlm_labels_cpu != -100)
        if valid_mlm_mask.any():
            valid_mlm_labels = mlm_labels_cpu[valid_mlm_mask]
            max_mlm = valid_mlm_labels.max().item()
            min_mlm = valid_mlm_labels.min().item()
        else:
            max_mlm = -1
            min_mlm = -1
        
        # Check MLM labels
        if max_mlm >= vocab_size or (min_mlm < 0 and min_mlm != -100):
            print(f"\n{'='*80}")
            print(f"INVALID MLM LABEL IN BATCH {batch_idx}")
            print(f"{'='*80}")
            print(f"Vocab size: {vocab_size} (IDs 0-{vocab_size-1})")
            print(f"Max MLM label (excluding -100): {max_mlm}")
            print(f"Min MLM label (excluding -100): {min_mlm}")
            print(f"MLM labels shape: {mlm_labels_cpu.shape}")
            
            # Find invalid positions
            invalid_mask = ((mlm_labels_cpu >= vocab_size) | (mlm_labels_cpu < -100))
            invalid_positions = torch.nonzero(invalid_mask, as_tuple=False)
            print(f"\nInvalid MLM label count: {invalid_mask.sum().item()}")
            print(f"First 20 invalid positions:")
            for i, (seq_idx, tok_idx) in enumerate(invalid_positions[:20]):
                invalid_label = mlm_labels_cpu[seq_idx, tok_idx].item()
                token_id = token_ids_cpu[seq_idx, tok_idx].item()
                print(f"  Seq {seq_idx.item()}, Token {tok_idx.item()}: Label = {invalid_label}, Token ID = {token_id}")
            
            print(f"{'='*80}\n")
            raise ValueError(f"Invalid MLM label detected in batch {batch_idx}: max={max_mlm}, allowed=[0, {vocab_size-1}]")
        
        # Check JTP labels (if present)
        jtp_labels_cpu = batch.get('jtp_labels', None)
        if jtp_labels_cpu is not None:
            valid_jtp_mask = (jtp_labels_cpu != -100)
            if valid_jtp_mask.any():
                valid_jtp_labels = jtp_labels_cpu[valid_jtp_mask]
                max_jtp = valid_jtp_labels.max().item()
                min_jtp = valid_jtp_labels.min().item()
                
                # JTP labels should be in range [0, seq_len-1]
                seq_len = token_ids_cpu.shape[1]  # Get sequence length from batch
                if max_jtp >= seq_len or min_jtp < 0:
                    print(f"\n{'='*80}")
                    print(f"INVALID JTP LABEL IN BATCH {batch_idx}")
                    print(f"{'='*80}")
                    print(f"Sequence length: {seq_len} (valid positions 0-{seq_len-1})")
                    print(f"Max JTP label (excluding -100): {max_jtp}")
                    print(f"Min JTP label (excluding -100): {min_jtp}")
                    print(f"JTP labels shape: {jtp_labels_cpu.shape}")
                    
                    # Find invalid positions
                    invalid_mask = ((jtp_labels_cpu >= seq_len) | ((jtp_labels_cpu < 0) & (jtp_labels_cpu != -100)))
                    invalid_positions = torch.nonzero(invalid_mask, as_tuple=False)
                    print(f"\nInvalid JTP label count: {invalid_mask.sum().item()}")
                    print(f"First 20 invalid positions:")
                    for i, (seq_idx, tok_idx) in enumerate(invalid_positions[:20]):
                        invalid_label = jtp_labels_cpu[seq_idx, tok_idx].item()
                        token_id = token_ids_cpu[seq_idx, tok_idx].item()
                        print(f"  Seq {seq_idx.item()}, Token {tok_idx.item()}: JTP Label = {invalid_label}, Token ID = {token_id}")
                    
                    print(f"{'='*80}\n")
                    raise ValueError(f"Invalid JTP label detected in batch {batch_idx}: max={max_jtp}, allowed=[0, {seq_len-1}]")
        
        # Check segment labels
        if max_seg >= segment_vocab_size or min_seg < 0:
            print(f"\n{'='*80}")
            print(f"INVALID SEGMENT LABEL IN BATCH {batch_idx}")
            print(f"{'='*80}")
            print(f"Segment embedding size: {segment_vocab_size} (IDs 0-{segment_vocab_size-1})")
            print(f"Max segment label: {max_seg}")
            print(f"Min segment label: {min_seg}")
            print(f"Segment labels shape: {segment_labels_cpu.shape}")
            
            # Find invalid positions
            invalid_mask = (segment_labels_cpu >= segment_vocab_size) | (segment_labels_cpu < 0)
            invalid_positions = torch.nonzero(invalid_mask, as_tuple=False)
            print(f"\nInvalid segment count: {invalid_mask.sum().item()}")
            print(f"First 20 invalid positions:")
            for i, (seq_idx, tok_idx) in enumerate(invalid_positions[:20]):
                invalid_seg = segment_labels_cpu[seq_idx, tok_idx].item()
                token_id = token_ids_cpu[seq_idx, tok_idx].item()
                print(f"  Seq {seq_idx.item()}, Token {tok_idx.item()}: Segment = {invalid_seg}, Token ID = {token_id}")
            
            # Show context
            seq_idx, tok_idx = invalid_positions[0][0].item(), invalid_positions[0][1].item()
            start = max(0, tok_idx - 5)
            end = min(segment_labels_cpu.shape[1], tok_idx + 6)
            print(f"\nContext (seq {seq_idx}, tokens {start}:{end}):")
            print(f"  Token IDs: {token_ids_cpu[seq_idx, start:end].tolist()}")
            print(f"  Segments: {segment_labels_cpu[seq_idx, start:end].tolist()}")
            print(f"{'='*80}\n")
            
            raise ValueError(f"Invalid segment label detected in batch {batch_idx}: max={max_seg}, allowed=[0, {segment_vocab_size-1}]")
        
        # Check token IDs
        if max_id >= vocab_size or min_id < 0:
            print(f"\n{'='*80}")
            print(f"INVALID TOKEN ID IN BATCH {batch_idx}")
            print(f"{'='*80}")
            print(f"Vocab size: {vocab_size}")
            print(f"Max token ID: {max_id}")
            print(f"Min token ID: {min_id}")
            print(f"Token IDs shape: {token_ids_cpu.shape}")
            
            # Find invalid positions
            invalid_mask = (token_ids_cpu >= vocab_size) | (token_ids_cpu < 0)
            invalid_positions = torch.nonzero(invalid_mask, as_tuple=False)
            print(f"\nInvalid token count: {invalid_mask.sum().item()}")
            print(f"First 10 invalid positions:")
            for i, (seq_idx, tok_idx) in enumerate(invalid_positions[:10]):
                invalid_id = token_ids_cpu[seq_idx, tok_idx].item()
                print(f"  Seq {seq_idx.item()}, Token {tok_idx.item()}: ID = {invalid_id}")
            
            # Show context
            seq_idx, tok_idx = invalid_positions[0][0].item(), invalid_positions[0][1].item()
            start = max(0, tok_idx - 5)
            end = min(token_ids_cpu.shape[1], tok_idx + 6)
            print(f"\nContext (seq {seq_idx}, tokens {start}:{end}):")
            print(f"  Token IDs: {token_ids_cpu[seq_idx, start:end].tolist()}")
            print(f"  Segments: {batch['segment_label'][seq_idx, start:end].tolist()}")
            print(f"{'='*80}\n")
            
            raise ValueError(f"Invalid token ID detected in batch {batch_idx}")
        
        # Move to device
        token_ids = token_ids_cpu.to(device)
        token_type_ids = batch['segment_label'].to(device)
        mlm_labels = batch['bert_label'].to(device)
        
        # Address positions
        binary_pos = batch['binary_pos'].to(device)
        function_pos = batch['function_pos'].to(device)
        bb_pos = batch['bb_pos'].to(device)
        var_offsets = batch.get('var_offsets', None)
        if var_offsets is not None:
            var_offsets = var_offsets.to(device)
        
        # JTP labels (if present)
        jtp_labels = batch.get('jtp_labels', None)
        if jtp_labels is not None:
            jtp_labels = jtp_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (no longer needs attention_mask parameter)
        mlm_logits, jtp_logits = model(
            token_ids=token_ids,
            token_type_ids=token_type_ids,
            binary_pos=binary_pos,
            function_pos=function_pos,
            bb_pos=bb_pos,
            var_offsets=var_offsets
        )
        
        # MLM loss
        mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        mlm_loss = mlm_loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
        
        # JTP loss (if labels provided)
        if jtp_labels is not None:
            jtp_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            jtp_loss = jtp_loss_fn(jtp_logits.view(-1, jtp_logits.size(-1)), jtp_labels.view(-1))
            loss = mlm_loss + jtp_loss
        else:
            jtp_loss = torch.tensor(0.0).to(device)
            loss = mlm_loss
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Stats
        total_loss += loss.item()
        total_mlm_loss += mlm_loss.item()
        total_jtp_loss += jtp_loss.item()
        
        # Accuracy (only for masked positions)
        mlm_mask = mlm_labels != -100
        if mlm_mask.sum() > 0:
            mlm_preds = mlm_logits.argmax(dim=-1)
            mlm_correct += ((mlm_preds == mlm_labels) & mlm_mask).sum().item()
            mlm_total += mlm_mask.sum().item()
        
        if jtp_labels is not None:
            jtp_mask = jtp_labels != -100
            if jtp_mask.sum() > 0:
                jtp_preds = jtp_logits.argmax(dim=-1)
                jtp_correct += ((jtp_preds == jtp_labels) & jtp_mask).sum().item()
                jtp_total += jtp_mask.sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mlm': f'{mlm_loss.item():.4f}',
            'jtp': f'{jtp_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_mlm_loss = total_mlm_loss / len(dataloader)
    avg_jtp_loss = total_jtp_loss / len(dataloader)
    mlm_acc = mlm_correct / mlm_total if mlm_total > 0 else 0
    jtp_acc = jtp_correct / jtp_total if jtp_total > 0 else 0
    
    return avg_loss, avg_mlm_loss, avg_jtp_loss, mlm_acc, jtp_acc


def validate_epoch(model, dataloader, device, logger):
    """Validate one epoch."""
    model.eval()
    total_loss = 0
    total_mlm_loss = 0
    total_jtp_loss = 0
    
    mlm_correct = 0
    mlm_total = 0
    jtp_correct = 0
    jtp_total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            token_ids = batch['bert_input'].to(device)
            token_type_ids = batch['segment_label'].to(device)
            mlm_labels = batch['bert_label'].to(device)
            
            binary_pos = batch['binary_pos'].to(device)
            function_pos = batch['function_pos'].to(device)
            bb_pos = batch['bb_pos'].to(device)
            var_offsets = batch.get('var_offsets', None)
            if var_offsets is not None:
                var_offsets = var_offsets.to(device)
            
            jtp_labels = batch.get('jtp_labels', None)
            if jtp_labels is not None:
                jtp_labels = jtp_labels.to(device)
            
            mlm_logits, jtp_logits = model(
                token_ids=token_ids,
                token_type_ids=token_type_ids,
                binary_pos=binary_pos,
                function_pos=function_pos,
                bb_pos=bb_pos,
                var_offsets=var_offsets
            )
            
            mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            
            if jtp_labels is not None:
                jtp_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                jtp_loss = jtp_loss_fn(jtp_logits.view(-1, jtp_logits.size(-1)), jtp_labels.view(-1))
                loss = mlm_loss + jtp_loss
            else:
                jtp_loss = torch.tensor(0.0).to(device)
                loss = mlm_loss
            
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_jtp_loss += jtp_loss.item()
            
            mlm_mask = mlm_labels != -100
            if mlm_mask.sum() > 0:
                mlm_preds = mlm_logits.argmax(dim=-1)
                mlm_correct += ((mlm_preds == mlm_labels) & mlm_mask).sum().item()
                mlm_total += mlm_mask.sum().item()
            
            if jtp_labels is not None:
                jtp_mask = jtp_labels != -100
                if jtp_mask.sum() > 0:
                    jtp_preds = jtp_logits.argmax(dim=-1)
                    jtp_correct += ((jtp_preds == jtp_labels) & jtp_mask).sum().item()
                    jtp_total += jtp_mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mlm_loss = total_mlm_loss / len(dataloader)
    avg_jtp_loss = total_jtp_loss / len(dataloader)
    mlm_acc = mlm_correct / mlm_total if mlm_total > 0 else 0
    jtp_acc = jtp_correct / jtp_total if jtp_total > 0 else 0
    
    return avg_loss, avg_mlm_loss, avg_jtp_loss, mlm_acc, jtp_acc


def main():
    parser = argparse.ArgumentParser(description="Address-Aware jTrans Pretraining")
    
    # Data paths
    parser.add_argument('--train_path', type=str, required=True, help='Training data path')
    parser.add_argument('--vocab_path', type=str, required=True, help='Vocabulary path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    # Model hyperparameters
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size')
    parser.add_argument('--num_hidden_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--num_attention_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--max_len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='Warmup steps')
    parser.add_argument('--save_every', type=int, default=1, help='Save every N epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    # Masking parameters
    parser.add_argument('--token_mask_prob', type=float, default=0.15, help='Token masking probability')
    
    # Data sampling
    parser.add_argument('--data_ratio', type=float, default=1.0, help='Ratio of training data to use (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.output_dir)
    
    # Log GPU info
    if torch.cuda.is_available():
        logger.info("=" * 80)
        logger.info("GPU INFORMATION")
        logger.info("=" * 80)
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        logger.info(f"Number of visible GPUs: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"PyTorch device: {device}")
    
    logger.info("=" * 80)
    logger.info("jTrans ADDRESS-AWARE Pretraining")
    logger.info("=" * 80)
    logger.info(f"Train data: {args.train_path}")
    logger.info(f"Vocab: {args.vocab_path}")
    logger.info(f"Data ratio: {args.data_ratio:.1%} of training data")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    # Load vocabulary
    logger.info(f"Loading vocabulary from {args.vocab_path}...")
    vocab = WordVocab.load_vocab(args.vocab_path)
    vocab_size = len(vocab)
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataset = AddressAwareDataset(
        corpus_path=args.train_path,
        vocab=vocab,
        seq_len=args.max_len,
        token_mask_prob=args.token_mask_prob,
        on_memory=True,
        data_percentage=args.data_ratio
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train batches: {len(train_dataloader)}")
    
    # Create model
    logger.info("Creating address-aware model...")
    model = create_addressaware_model(
        vocab_size=vocab_size,
        hidden=args.hidden_size,
        n_layers=args.num_hidden_layers,
        attn_heads=args.num_attention_heads,
        dropout=args.dropout,
        max_len=args.max_len
    )
    
    # Set vocab_stoi for address/daddr distinction in embeddings
    model.bert.embeddings.vocab_stoi = vocab.stoi
    
    model = model.to(device)
    
    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training with DataParallel")
        model = nn.DataParallel(model)
    
    # Verify embedding layer size matches vocab
    # Handle DataParallel wrapper: .module gives access to the actual model
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    actual_embedding_size = actual_model.bert.embeddings.token_embedding.weight.shape[0]
    logger.info(f"Model token embedding size: {actual_embedding_size}")
    logger.info(f"Expected vocab size: {vocab_size}")
    if actual_embedding_size != vocab_size:
        logger.error(f"MISMATCH! Embedding size {actual_embedding_size} != vocab size {vocab_size}")
        raise ValueError(f"Model embedding size mismatch")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    from transformers import get_linear_schedule_with_warmup, AdamW
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Check for existing checkpoints to resume training
    start_epoch = 0
    if os.path.exists(args.output_dir):
        # Find all checkpoint directories
        checkpoint_dirs = [d for d in os.listdir(args.output_dir) 
                          if d.startswith('checkpoint_epoch_') and 
                          os.path.isdir(os.path.join(args.output_dir, d))]
        
        if checkpoint_dirs:
            # Extract epoch numbers and find the latest
            epoch_nums = [int(d.split('_')[-1]) for d in checkpoint_dirs]
            latest_epoch = max(epoch_nums)
            latest_checkpoint_dir = os.path.join(args.output_dir, f'checkpoint_epoch_{latest_epoch}')
            
            logger.info("=" * 80)
            logger.info(f"Found existing checkpoint at epoch {latest_epoch}")
            logger.info(f"Loading checkpoint from: {latest_checkpoint_dir}")
            
            try:
                # Load model state
                checkpoint_path = os.path.join(latest_checkpoint_dir, 'pytorch_model.bin')
                if os.path.exists(checkpoint_path):
                    actual_model = model.module if isinstance(model, nn.DataParallel) else model
                    actual_model.bert.load_state_dict(torch.load(checkpoint_path, map_location=device))
                    logger.info("✓ Model weights loaded")
                
                # Load optimizer state if exists
                optimizer_path = os.path.join(latest_checkpoint_dir, 'optimizer.pt')
                if os.path.exists(optimizer_path):
                    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                    logger.info("✓ Optimizer state loaded")
                
                # Load scheduler state if exists
                scheduler_path = os.path.join(latest_checkpoint_dir, 'scheduler.pt')
                if os.path.exists(scheduler_path):
                    scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
                    logger.info("✓ Scheduler state loaded")
                
                start_epoch = latest_epoch
                logger.info(f"Resuming training from epoch {start_epoch + 1}")
                logger.info("=" * 80)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                logger.warning("Starting training from scratch")
                start_epoch = 0
    
    # Training loop
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_mlm_loss, train_jtp_loss, train_mlm_acc, train_jtp_acc = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, logger
        )
        
        logger.info(f"Train - Loss: {train_loss:.4f}, MLM Loss: {train_mlm_loss:.4f}, JTP Loss: {train_jtp_loss:.4f}")
        logger.info(f"Train - MLM Acc: {train_mlm_acc:.4f}, JTP Acc: {train_jtp_acc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_dir = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model weights (BERT part only for compatibility)
            # Unwrap DataParallel if needed
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(actual_model.bert.state_dict(), os.path.join(checkpoint_dir, 'pytorch_model.bin'))
            
            # Save optimizer and scheduler states for resumption
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pt'))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler.pt'))
            
            # Save config
            config = {
                'vocab_size': vocab_size,
                'hidden_size': args.hidden_size,
                'num_hidden_layers': args.num_hidden_layers,
                'num_attention_heads': args.num_attention_heads,
                'max_position_embeddings': args.max_len,
                'type_vocab_size': 2,
                'model_type': 'address_aware_jtrans'
            }
            with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save training info
            training_info = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_mlm_acc': train_mlm_acc,
            }
            with open(os.path.join(checkpoint_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=2)
            
            logger.info(f"✓ Checkpoint saved to {checkpoint_dir}")
        
        logger.info("=" * 80)
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
