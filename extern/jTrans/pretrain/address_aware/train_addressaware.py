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


def train_epoch(model, dataloader, optimizer, scheduler, device, logger,
                scaler=None, jtp_weight=1.0, debug=False, tb_writer=None, global_step=0):
    """Train one epoch with optional AMP and loss weighting."""
    model.train()
    total_loss = 0
    total_mlm_loss = 0
    total_jtp_loss = 0
    
    mlm_correct = 0
    mlm_total = 0
    jtp_correct = 0
    jtp_total = 0
    
    use_amp = scaler is not None
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Debug validation (only first 5 batches unless --debug)
        if debug or batch_idx < 5:
            token_ids_cpu = batch['bert_input']
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            vocab_size = actual_model.bert.embeddings.token_embedding.num_embeddings
            max_id = token_ids_cpu.max().item()
            min_id = token_ids_cpu.min().item()
            if max_id >= vocab_size or min_id < 0:
                raise ValueError(f"Invalid token ID in batch {batch_idx}: min={min_id}, max={max_id}, vocab={vocab_size}")
        
        # Move to device
        token_ids = batch['bert_input'].to(device)
        attention_mask = batch['attention_mask'].to(device)
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
        
        optimizer.zero_grad()
        
        # Forward pass with optional AMP
        with torch.cuda.amp.autocast(enabled=use_amp):
            mlm_logits, jtp_logits = model(
                token_ids=token_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                binary_pos=binary_pos,
                function_pos=function_pos,
                bb_pos=bb_pos,
                var_offsets=var_offsets
            )
            
            # MLM loss
            mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            
            # JTP loss with configurable weight
            if jtp_logits is not None and jtp_labels is not None:
                jtp_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                jtp_loss = jtp_loss_fn(jtp_logits.view(-1, jtp_logits.size(-1)), jtp_labels.view(-1))
                loss = mlm_loss + jtp_weight * jtp_loss
            else:
                jtp_loss = torch.tensor(0.0).to(device)
                loss = mlm_loss
        
        # Backward with AMP
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        
        # Stats
        total_loss += loss.item()
        total_mlm_loss += mlm_loss.item()
        total_jtp_loss += jtp_loss.item()
        
        # Accuracy (only for masked positions)
        with torch.no_grad():
            mlm_mask = mlm_labels != -100
            if mlm_mask.sum() > 0:
                mlm_preds = mlm_logits.argmax(dim=-1)
                mlm_correct += ((mlm_preds == mlm_labels) & mlm_mask).sum().item()
                mlm_total += mlm_mask.sum().item()
            
            if jtp_logits is not None and jtp_labels is not None:
                jtp_mask = jtp_labels != -100
                if jtp_mask.sum() > 0:
                    jtp_preds = jtp_logits.argmax(dim=-1)
                    jtp_correct += ((jtp_preds == jtp_labels) & jtp_mask).sum().item()
                    jtp_total += jtp_mask.sum().item()
        
        global_step += 1
        
        # TensorBoard logging
        if tb_writer is not None and global_step % 50 == 0:
            tb_writer.add_scalar('train/loss', loss.item(), global_step)
            tb_writer.add_scalar('train/mlm_loss', mlm_loss.item(), global_step)
            tb_writer.add_scalar('train/jtp_loss', jtp_loss.item(), global_step)
            tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
        
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
    
    return avg_loss, avg_mlm_loss, avg_jtp_loss, mlm_acc, jtp_acc, global_step


def validate_epoch(model, dataloader, device, logger, jtp_weight=1.0):
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
            attention_mask = batch['attention_mask'].to(device)
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
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                binary_pos=binary_pos,
                function_pos=function_pos,
                bb_pos=bb_pos,
                var_offsets=var_offsets
            )
            
            mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            
            if jtp_logits is not None and jtp_labels is not None:
                jtp_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                jtp_loss = jtp_loss_fn(jtp_logits.view(-1, jtp_logits.size(-1)), jtp_labels.view(-1))
                loss = mlm_loss + jtp_weight * jtp_loss
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
            
            if jtp_logits is not None and jtp_labels is not None:
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
    parser.add_argument('--warmup_ratio', type=float, default=0.06, help='Warmup ratio of total steps (default: 6%)')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Override warmup with fixed steps (overrides warmup_ratio if set)')
    parser.add_argument('--save_every', type=int, default=1, help='Save every N epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    # Masking parameters
    parser.add_argument('--token_mask_prob', type=float, default=0.15, help='Token masking probability')
    parser.add_argument('--instruction_mask_prob', type=float, default=0.15, help='Instruction masking probability')
    parser.add_argument('--masking_strategy', type=str, default='mixed', choices=['token', 'instruction', 'mixed'],
                        help='Masking strategy: token (standard MLM), instruction (mask whole instructions), mixed (50/50)')
    
    # Loss weighting
    parser.add_argument('--jtp_weight', type=float, default=1.0, help='Weight for JTP loss (loss = MLM + jtp_weight * JTP)')
    
    # Experimental flags (both enabled by default)
    parser.add_argument('--no_jtp', action='store_true', help='Disable JTP task')
    parser.add_argument('--no_binary_pos', action='store_true', help='Disable binary position for code addresses')
    
    # Data sampling
    parser.add_argument('--data_ratio', type=float, default=1.0, help='Ratio of training data to use (0.0-1.0)')
    parser.add_argument('--val_split', type=float, default=0.05, help='Validation split ratio (default: 5%)')
    
    # Training efficiency
    parser.add_argument('--amp', action='store_true', default=True, help='Enable mixed precision training (default: True)')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--debug', action='store_true', help='Enable verbose per-batch validation checks')
    
    args = parser.parse_args()
    
    # Handle AMP flag
    use_amp = args.amp and not args.no_amp and torch.cuda.is_available()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.output_dir)
    
    # TensorBoard
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, 'tensorboard')
        tb_writer = SummaryWriter(tb_dir)
        logger.info(f"TensorBoard logging to {tb_dir}")
    except ImportError:
        logger.warning("TensorBoard not available, skipping.")
    
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
    logger.info("jTrans ADDRESS-AWARE Pretraining (Improved)")
    logger.info("=" * 80)
    logger.info(f"Train data: {args.train_path}")
    logger.info(f"Vocab: {args.vocab_path}")
    logger.info(f"Data ratio: {args.data_ratio:.1%} of training data")
    logger.info(f"Validation split: {args.val_split:.1%}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"JTP task: {'Disabled (MLM only)' if args.no_jtp else 'Enabled'}")
    logger.info(f"JTP weight: {args.jtp_weight}")
    logger.info(f"Binary position: {'Disabled' if args.no_binary_pos else 'Enabled'}")
    logger.info(f"Masking strategy: {args.masking_strategy}")
    logger.info(f"Mixed precision (AMP): {use_amp}")
    logger.info(f"Debug mode: {args.debug}")
    
    # Load vocabulary
    logger.info(f"Loading vocabulary from {args.vocab_path}...")
    vocab = WordVocab.load_vocab(args.vocab_path)
    vocab_size = len(vocab)
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Create datasets with train/val split
    logger.info("Creating dataloaders...")
    train_split = 1.0 - args.val_split
    
    train_dataset = AddressAwareDataset(
        corpus_path=args.train_path,
        vocab=vocab,
        seq_len=args.max_len,
        token_mask_prob=args.token_mask_prob,
        instruction_mask_prob=args.instruction_mask_prob,
        masking_strategy=args.masking_strategy,
        on_memory=True,
        data_percentage=args.data_ratio,
        train_split=train_split,
        is_train=True,
    )
    
    val_dataset = AddressAwareDataset(
        corpus_path=args.train_path,
        vocab=vocab,
        seq_len=args.max_len,
        token_mask_prob=args.token_mask_prob,
        instruction_mask_prob=args.instruction_mask_prob,
        masking_strategy='token',  # Val always uses token masking for consistency
        on_memory=True,
        data_percentage=args.data_ratio,
        train_split=train_split,
        is_train=False,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ) if len(val_dataset) > 0 else None
    
    logger.info(f"Train samples: {len(train_dataset)}, Train batches: {len(train_dataloader)}")
    if val_dataloader:
        logger.info(f"Val samples: {len(val_dataset)}, Val batches: {len(val_dataloader)}")
    
    # Create model
    logger.info("Creating address-aware model...")
    model = create_addressaware_model(
        vocab_size=vocab_size,
        hidden=args.hidden_size,
        n_layers=args.num_hidden_layers,
        attn_heads=args.num_attention_heads,
        dropout=args.dropout,
        max_len=args.max_len,
        use_jtp=not args.no_jtp,
        use_binary_pos=not args.no_binary_pos
    )
    
    # Set vocab_stoi for address/daddr distinction in embeddings
    model.bert.embeddings.vocab_stoi = vocab.stoi
    
    model = model.to(device)
    
    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training with DataParallel")
        model = nn.DataParallel(model)
    
    # Verify embedding layer size matches vocab
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
    
    # Warmup: use fixed steps if specified, otherwise use ratio
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps} ({warmup_steps/total_steps:.1%})")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # AMP GradScaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Check for existing checkpoints to resume training
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if os.path.exists(args.output_dir):
        checkpoint_dirs = [d for d in os.listdir(args.output_dir) 
                          if d.startswith('checkpoint_epoch_') and 
                          os.path.isdir(os.path.join(args.output_dir, d))]
        
        if checkpoint_dirs:
            epoch_nums = [int(d.split('_')[-1]) for d in checkpoint_dirs]
            latest_epoch = max(epoch_nums)
            latest_checkpoint_dir = os.path.join(args.output_dir, f'checkpoint_epoch_{latest_epoch}')
            
            logger.info("=" * 80)
            logger.info(f"Found existing checkpoint at epoch {latest_epoch}")
            logger.info(f"Loading checkpoint from: {latest_checkpoint_dir}")
            
            try:
                # Load model state (BERT part for finetune compatibility)
                checkpoint_path = os.path.join(latest_checkpoint_dir, 'pytorch_model.bin')
                if os.path.exists(checkpoint_path):
                    actual_model = model.module if isinstance(model, nn.DataParallel) else model
                    actual_model.bert.load_state_dict(torch.load(checkpoint_path, map_location=device))
                    logger.info("✓ Model weights loaded")
                
                # Try to load full model (includes MLM/JTP heads) for proper resume
                full_path = os.path.join(latest_checkpoint_dir, 'full_model.bin')
                if os.path.exists(full_path):
                    actual_model = model.module if isinstance(model, nn.DataParallel) else model
                    actual_model.load_state_dict(torch.load(full_path, map_location=device))
                    logger.info("✓ Full model weights loaded (includes MLM/JTP heads)")
                
                optimizer_path = os.path.join(latest_checkpoint_dir, 'optimizer.pt')
                if os.path.exists(optimizer_path):
                    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                    logger.info("✓ Optimizer state loaded")
                
                scheduler_path = os.path.join(latest_checkpoint_dir, 'scheduler.pt')
                if os.path.exists(scheduler_path):
                    scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
                    logger.info("✓ Scheduler state loaded")
                
                # Load training info for best_val_loss
                info_path = os.path.join(latest_checkpoint_dir, 'training_info.json')
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    best_val_loss = info.get('best_val_loss', float('inf'))
                    global_step = info.get('global_step', 0)
                    logger.info(f"✓ Training info loaded (best_val_loss={best_val_loss:.4f})")
                
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
        train_loss, train_mlm_loss, train_jtp_loss, train_mlm_acc, train_jtp_acc, global_step = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, logger,
            scaler=scaler, jtp_weight=args.jtp_weight, debug=args.debug,
            tb_writer=tb_writer, global_step=global_step
        )
        
        logger.info(f"Train - Loss: {train_loss:.4f}, MLM Loss: {train_mlm_loss:.4f}, JTP Loss: {train_jtp_loss:.4f}")
        logger.info(f"Train - MLM Acc: {train_mlm_acc:.4f}, JTP Acc: {train_jtp_acc:.4f}")
        
        # Log embedding scale factors
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        emb = actual_model.bert.embeddings
        logger.info(f"Embedding scales - token: {emb.token_scale.item():.3f}, pos: {emb.pos_scale.item():.3f}, "
                     f"addr: {emb.addr_scale.item():.3f}, seg: {emb.seg_scale.item():.3f}, var: {emb.var_scale.item():.3f}")
        
        # Validate
        val_loss = float('inf')
        if val_dataloader is not None:
            val_loss, val_mlm_loss, val_jtp_loss, val_mlm_acc, val_jtp_acc = validate_epoch(
                model, val_dataloader, device, logger, jtp_weight=args.jtp_weight
            )
            logger.info(f"Val   - Loss: {val_loss:.4f}, MLM Loss: {val_mlm_loss:.4f}, JTP Loss: {val_jtp_loss:.4f}")
            logger.info(f"Val   - MLM Acc: {val_mlm_acc:.4f}, JTP Acc: {val_jtp_acc:.4f}")
            
            if tb_writer is not None:
                tb_writer.add_scalar('val/loss', val_loss, epoch + 1)
                tb_writer.add_scalar('val/mlm_loss', val_mlm_loss, epoch + 1)
                tb_writer.add_scalar('val/mlm_acc', val_mlm_acc, epoch + 1)
        
        if tb_writer is not None:
            tb_writer.add_scalar('train/epoch_loss', train_loss, epoch + 1)
            tb_writer.add_scalar('train/epoch_mlm_acc', train_mlm_acc, epoch + 1)
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_dir = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            
            # Save BERT part only (for finetune.py compatibility)
            torch.save(actual_model.bert.state_dict(), os.path.join(checkpoint_dir, 'pytorch_model.bin'))
            
            # Save full model (for training resume with MLM/JTP heads)
            torch.save(actual_model.state_dict(), os.path.join(checkpoint_dir, 'full_model.bin'))
            
            # Save optimizer and scheduler states
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pt'))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler.pt'))
            
            # Save config
            config = {
                'vocab_size': vocab_size,
                'hidden_size': args.hidden_size,
                'num_hidden_layers': args.num_hidden_layers,
                'num_attention_heads': args.num_attention_heads,
                'max_position_embeddings': args.max_len,
                'type_vocab_size': 256,
                'model_type': 'address_aware_jtrans',
                'use_jtp': not args.no_jtp,
                'use_binary_pos': not args.no_binary_pos,
                'masking_strategy': args.masking_strategy,
                'jtp_weight': args.jtp_weight,
            }
            with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save training info
            training_info = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'train_loss': train_loss,
                'train_mlm_acc': train_mlm_acc,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }
            with open(os.path.join(checkpoint_dir, 'training_info.json'), 'w') as f:
                json.dump(training_info, f, indent=2)
            
            logger.info(f"✓ Checkpoint saved to {checkpoint_dir}")
        
        # Save best model (based on val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = os.path.join(args.output_dir, 'best_model')
            os.makedirs(best_dir, exist_ok=True)
            
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(actual_model.bert.state_dict(), os.path.join(best_dir, 'pytorch_model.bin'))
            
            config = {
                'vocab_size': vocab_size,
                'hidden_size': args.hidden_size,
                'num_hidden_layers': args.num_hidden_layers,
                'num_attention_heads': args.num_attention_heads,
                'max_position_embeddings': args.max_len,
                'type_vocab_size': 256,
                'model_type': 'address_aware_jtrans',
                'use_jtp': not args.no_jtp,
                'use_binary_pos': not args.no_binary_pos,
                'masking_strategy': args.masking_strategy,
                'jtp_weight': args.jtp_weight,
            }
            with open(os.path.join(best_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"★ NEW BEST model saved (val_loss={val_loss:.4f})")
        
        logger.info("=" * 80)
    
    if tb_writer is not None:
        tb_writer.close()
    
    logger.info(f"Training completed! Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved at: {os.path.join(args.output_dir, 'best_model')}")


if __name__ == '__main__':
    main()
