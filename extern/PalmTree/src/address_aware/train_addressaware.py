"""
Training script for Address-Aware BERT with Instruction Masking

Trains a BERT model with:
1. Instruction Masking CFG (IMC) - masks entire instructions in CFG
2. Instruction Masking DFG (IMD) - masks entire instructions in DFG
3. Masked Language Modeling (MLM) - masks individual tokens  
4. Scope Prediction

New tasks: IMC/IMD (Instruction Masking for CFG/DFG)
- Masks entire instructions at a configurable rate
- Model must predict all tokens in the masked instruction
- Different from MLM which masks individual tokens
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
import random
import numpy as np


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


# Import local modules
from vocab import WordVocab
# from create_vocab import create_vocab  # Not needed for train_comparison.py
from address_aware.dataloader_addressaware import InstructionMaskingDataset
# from dataloader_scope import ScopeDataset  # Not needed for train_comparison.py
from address_aware.model_addressaware import AddressAwareBERT, AddressAwareBERTForPretraining


def train_epoch(model, data_loader, scope_loader, optimizer, device, log_freq=1000, logger=None,
                enable_imc=False, enable_imd=False, enable_mlm=True, enable_scope=False):
    """
    Train for one epoch with IMC, IMD, MLM, and Scope tasks.
    
    Args:
        model: AddressAwareBERTForPretraining model
        data_loader: DataLoader for IMC+IMD+MLM data
        scope_loader: DataLoader for scope prediction data
        optimizer: Optimizer
        device: Device to train on
        log_freq: Log every N batches
        logger: Logger instance
        enable_imc: Enable instruction masking for CFG
        enable_imd: Enable instruction masking for DFG
        enable_mlm: Enable masked language modeling task
        enable_scope: Enable scope prediction task
    
    Returns:
        Dictionary with average losses
    """
    model.train()
    
    total_imc_loss = 0
    total_imd_loss = 0
    total_mlm_loss = 0
    total_scope_loss = 0
    total_loss = 0
    
    # Loss functions
    imc_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore non-masked tokens
    imd_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    scope_criterion = nn.NLLLoss()
    
    # Scope iterator
    scope_iter = iter(scope_loader) if (enable_scope and scope_loader) else None
    
    # Create progress bar with manual updates
    progress_bar = tqdm(total=len(data_loader), desc="Training", disable=False)
    
    for batch_idx, batch in enumerate(data_loader):
        # === Process IMC (Instruction Masking CFG) ===
        imc_loss = torch.tensor(0.0, device=device)
        if enable_imc:
            imc_batch = batch['imc']
            
            # Validate token IDs BEFORE moving to GPU
            imc_input_cpu = imc_batch['bert_input']
            imc_segment_cpu = imc_batch['segment_label']
            vocab_size = model.module.bert.embedding.token_embedding.num_embeddings if hasattr(model, 'module') else model.bert.embedding.token_embedding.num_embeddings
            segment_vocab_size = model.module.bert.embedding.segment_embedding.num_embeddings if hasattr(model, 'module') else model.bert.embedding.segment_embedding.num_embeddings
            
            # Check token IDs
            invalid_mask = (imc_input_cpu < 0) | (imc_input_cpu >= vocab_size)
            if invalid_mask.any():
                invalid_ids = imc_input_cpu[invalid_mask].unique()
                logger.error(f"Invalid token IDs found in IMC input: {invalid_ids.tolist()}")
                logger.error(f"Vocab size: {vocab_size}, Min ID: {imc_input_cpu.min()}, Max ID: {imc_input_cpu.max()}")
                # Print first few samples with invalid IDs
                for i in range(min(3, imc_input_cpu.size(0))):
                    if invalid_mask[i].any():
                        logger.error(f"Sample {i} invalid tokens: {imc_input_cpu[i][invalid_mask[i]].tolist()}")
                raise ValueError(f"Token IDs out of bounds: {invalid_ids.tolist()}")
            
            # Check segment IDs
            invalid_seg_mask = (imc_segment_cpu < 0) | (imc_segment_cpu >= segment_vocab_size)
            if invalid_seg_mask.any():
                invalid_seg_ids = imc_segment_cpu[invalid_seg_mask].unique()
                logger.error(f"Invalid segment IDs found in IMC input: {invalid_seg_ids.tolist()}")
                logger.error(f"Segment vocab size: {segment_vocab_size}, Min ID: {imc_segment_cpu.min()}, Max ID: {imc_segment_cpu.max()}")
                raise ValueError(f"Segment IDs out of bounds: {invalid_seg_ids.tolist()}")
            
            imc_input = imc_input_cpu.to(device)
            imc_labels = imc_batch['bert_label'].to(device)
            imc_segment = imc_segment_cpu.to(device)
            imc_binary_pos = imc_batch['binary_pos'].to(device)
            imc_function_pos = imc_batch['function_pos'].to(device)
            imc_bb_pos = imc_batch['bb_pos'].to(device)
            imc_var_offsets = imc_batch['var_offsets'].to(device)
            
            if hasattr(model, 'module'):
                imc_output = model.module.forward_im(
                    imc_input, imc_segment,
                    imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets
                )
            else:
                imc_output = model.forward_im(
                    imc_input, imc_segment,
                    imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets
                )
            
            imc_output = imc_output.view(-1, imc_output.size(-1))
            imc_labels_flat = imc_labels.view(-1)
            imc_loss = imc_criterion(imc_output, imc_labels_flat)
        
        # === Process IMD (Instruction Masking DFG) ===
        imd_loss = torch.tensor(0.0, device=device)
        if enable_imd and 'imd' in batch:
            imd_batch = batch['imd']
            
            imd_input = imd_batch['bert_input'].to(device)
            imd_labels = imd_batch['bert_label'].to(device)
            imd_segment = imd_batch['segment_label'].to(device)
            imd_binary_pos = imd_batch['binary_pos'].to(device)
            imd_function_pos = imd_batch['function_pos'].to(device)
            imd_bb_pos = imd_batch['bb_pos'].to(device)
            imd_var_offsets = imd_batch['var_offsets'].to(device)
            
            if hasattr(model, 'module'):
                imd_output = model.module.forward_im(
                    imd_input, imd_segment,
                    imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets
                )
            else:
                imd_output = model.forward_im(
                    imd_input, imd_segment,
                    imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets
                )
            
            imd_output = imd_output.view(-1, imd_output.size(-1))
            imd_labels_flat = imd_labels.view(-1)
            imd_loss = imd_criterion(imd_output, imd_labels_flat)
        
        # === Process MLM (Token-level Masking) ===
        mlm_loss = torch.tensor(0.0, device=device)
        if enable_mlm:
            mlm_batch = batch['mlm']
            
            mlm_input = mlm_batch['bert_input'].to(device)
            mlm_labels = mlm_batch['bert_label'].to(device)
            mlm_segment = mlm_batch['segment_label'].to(device)
            mlm_binary_pos = mlm_batch['binary_pos'].to(device)
            mlm_function_pos = mlm_batch['function_pos'].to(device)
            mlm_bb_pos = mlm_batch['bb_pos'].to(device)
            mlm_var_offsets = mlm_batch['var_offsets'].to(device)
            
            # Use standard forward pass for MLM
            mlm_output, _ = model(
                mlm_input, mlm_segment,
                mlm_binary_pos, mlm_function_pos, mlm_bb_pos, mlm_var_offsets
            )
            
            mlm_output = mlm_output.view(-1, mlm_output.size(-1))
            mlm_labels_flat = mlm_labels.view(-1)
            mlm_loss = mlm_criterion(mlm_output, mlm_labels_flat)
        
        # === Process Scope (if available and enabled) ===
        scope_loss = torch.tensor(0.0, device=device)
        if enable_scope and scope_iter is not None:
            try:
                scope_batch = next(scope_iter)
            except StopIteration:
                scope_iter = iter(scope_loader)
                scope_batch = next(scope_iter)
            
            scope_token_ids = scope_batch['bert_input'].to(device)
            scope_segment_labels = scope_batch['segment_label'].to(device)
            scope_binary_pos = scope_batch['binary_pos'].to(device)
            scope_function_pos = scope_batch['function_pos'].to(device)
            scope_bb_pos = scope_batch['bb_pos'].to(device)
            scope_var_offsets = scope_batch['var_offsets'].to(device)
            scope_labels = scope_batch['scope_label'].to(device)
            
            if hasattr(model, 'module'):
                scope_output = model.module.forward_scope(
                    scope_token_ids, scope_segment_labels,
                    scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets
                )
            else:
                scope_output = model.forward_scope(
                    scope_token_ids, scope_segment_labels,
                    scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets
                )
            
            scope_loss = scope_criterion(scope_output, scope_labels)
        
        # Combined loss
        loss = imc_loss + imd_loss + mlm_loss + scope_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate losses
        total_imc_loss += imc_loss.item()
        total_imd_loss += imd_loss.item()
        total_mlm_loss += mlm_loss.item()
        total_scope_loss += scope_loss.item()
        total_loss += loss.item()
        
        # Update progress bar every 100 iterations
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'imc': f'{total_imc_loss/(batch_idx+1):.4f}' if enable_imc else '0',
                'imd': f'{total_imd_loss/(batch_idx+1):.4f}' if enable_imd else '0',
                'mlm': f'{total_mlm_loss/(batch_idx+1):.4f}' if enable_mlm else '0',
                'scope': f'{total_scope_loss/(batch_idx+1):.4f}' if enable_scope else '0',
            })
            progress_bar.update(100 if batch_idx > 0 else 1)
        
        # Log to file periodically
        if (batch_idx + 1) % log_freq == 0 and logger:
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(f"Batch {batch_idx+1}/{len(data_loader)} - "
                      f"Loss: {avg_loss:.4f} | IMC: {total_imc_loss/(batch_idx+1):.4f} | "
                      f"IMD: {total_imd_loss/(batch_idx+1):.4f} | "
                      f"MLM: {total_mlm_loss/(batch_idx+1):.4f} | "
                      f"SCOPE: {total_scope_loss/(batch_idx+1):.4f}")
    
    # Close progress bar
    progress_bar.close()
    
    n_batches = len(data_loader)
    return {
        'total_loss': total_loss / n_batches,
        'imc_loss': total_imc_loss / n_batches,
        'imd_loss': total_imd_loss / n_batches,
        'mlm_loss': total_mlm_loss / n_batches,
        'scope_loss': total_scope_loss / n_batches,
    }


def validate_epoch(model, data_loader, scope_loader, device, logger=None,
                   enable_imc=False, enable_imd=False, enable_mlm=True, enable_scope=False):
    """Validate for one epoch"""
    model.eval()
    
    total_imc_loss = 0
    total_imd_loss = 0
    total_mlm_loss = 0
    total_scope_loss = 0
    total_loss = 0
    
    imc_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    imd_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    scope_criterion = nn.NLLLoss()
    
    scope_iter = iter(scope_loader) if (enable_scope and scope_loader) else None
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # === Process IMC ===
            imc_loss = torch.tensor(0.0, device=device)
            if enable_imc:
                imc_batch = batch['imc']
                
                imc_input = imc_batch['bert_input'].to(device)
                imc_labels = imc_batch['bert_label'].to(device)
                imc_segment = imc_batch['segment_label'].to(device)
                imc_binary_pos = imc_batch['binary_pos'].to(device)
                imc_function_pos = imc_batch['function_pos'].to(device)
                imc_bb_pos = imc_batch['bb_pos'].to(device)
                imc_var_offsets = imc_batch['var_offsets'].to(device)
                
                if hasattr(model, 'module'):
                    imc_output = model.module.forward_im(
                        imc_input, imc_segment,
                        imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets
                    )
                else:
                    imc_output = model.forward_im(
                        imc_input, imc_segment,
                        imc_binary_pos, imc_function_pos, imc_bb_pos, imc_var_offsets
                    )
                
                imc_output = imc_output.view(-1, imc_output.size(-1))
                imc_labels_flat = imc_labels.view(-1)
                imc_loss = imc_criterion(imc_output, imc_labels_flat)
            
            # === Process IMD ===
            imd_loss = torch.tensor(0.0, device=device)
            if enable_imd and 'imd' in batch:
                imd_batch = batch['imd']
                
                imd_input = imd_batch['bert_input'].to(device)
                imd_labels = imd_batch['bert_label'].to(device)
                imd_segment = imd_batch['segment_label'].to(device)
                imd_binary_pos = imd_batch['binary_pos'].to(device)
                imd_function_pos = imd_batch['function_pos'].to(device)
                imd_bb_pos = imd_batch['bb_pos'].to(device)
                imd_var_offsets = imd_batch['var_offsets'].to(device)
                
                if hasattr(model, 'module'):
                    imd_output = model.module.forward_im(
                        imd_input, imd_segment,
                        imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets
                    )
                else:
                    imd_output = model.forward_im(
                        imd_input, imd_segment,
                        imd_binary_pos, imd_function_pos, imd_bb_pos, imd_var_offsets
                    )
                
                imd_output = imd_output.view(-1, imd_output.size(-1))
                imd_labels_flat = imd_labels.view(-1)
                imd_loss = imd_criterion(imd_output, imd_labels_flat)
            
            # === Process MLM ===
            mlm_loss = torch.tensor(0.0, device=device)
            if enable_mlm:
                mlm_batch = batch['mlm']
                
                mlm_input = mlm_batch['bert_input'].to(device)
                mlm_labels = mlm_batch['bert_label'].to(device)
                mlm_segment = mlm_batch['segment_label'].to(device)
                mlm_binary_pos = mlm_batch['binary_pos'].to(device)
                mlm_function_pos = mlm_batch['function_pos'].to(device)
                mlm_bb_pos = mlm_batch['bb_pos'].to(device)
                mlm_var_offsets = mlm_batch['var_offsets'].to(device)
                
                mlm_output, _ = model(
                    mlm_input, mlm_segment,
                    mlm_binary_pos, mlm_function_pos, mlm_bb_pos, mlm_var_offsets
                )
                
                mlm_output = mlm_output.view(-1, mlm_output.size(-1))
                mlm_labels_flat = mlm_labels.view(-1)
                mlm_loss = mlm_criterion(mlm_output, mlm_labels_flat)
            
            # Scope
            scope_loss = torch.tensor(0.0, device=device)
            if enable_scope and scope_iter is not None:
                try:
                    scope_batch = next(scope_iter)
                except StopIteration:
                    scope_iter = iter(scope_loader)
                    scope_batch = next(scope_iter)
                
                scope_token_ids = scope_batch['bert_input'].to(device)
                scope_segment_labels = scope_batch['segment_label'].to(device)
                scope_binary_pos = scope_batch['binary_pos'].to(device)
                scope_function_pos = scope_batch['function_pos'].to(device)
                scope_bb_pos = scope_batch['bb_pos'].to(device)
                scope_var_offsets = scope_batch['var_offsets'].to(device)
                scope_labels = scope_batch['scope_label'].to(device)
                
                if hasattr(model, 'module'):
                    scope_output = model.module.forward_scope(
                        scope_token_ids, scope_segment_labels,
                        scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets
                    )
                else:
                    scope_output = model.forward_scope(
                        scope_token_ids, scope_segment_labels,
                        scope_binary_pos, scope_function_pos, scope_bb_pos, scope_var_offsets
                    )
                
                scope_loss = scope_criterion(scope_output, scope_labels)
            
            loss = imc_loss + imd_loss + mlm_loss + scope_loss
            
            total_imc_loss += imc_loss.item()
            total_imd_loss += imd_loss.item()
            total_mlm_loss += mlm_loss.item()
            total_scope_loss += scope_loss.item()
            total_loss += loss.item()
    
    n_batches = len(data_loader)
    return {
        'total_loss': total_loss / n_batches,
        'imc_loss': total_imc_loss / n_batches,
        'imd_loss': total_imd_loss / n_batches,
        'mlm_loss': total_mlm_loss / n_batches,
        'scope_loss': total_scope_loss / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Address-Aware BERT with Instruction Masking')
    
    # Data args
    parser.add_argument("--cfg_train", type=str, required=True, help="Training CFG corpus")
    parser.add_argument("--dfg_train", type=str, required=True, help="Training DFG corpus")
    parser.add_argument("--cfg_val", type=str, help="Validation CFG corpus")
    parser.add_argument("--dfg_val", type=str, help="Validation DFG corpus")
    parser.add_argument("--scope_train", type=str, help="Training scope corpus")
    parser.add_argument("--scope_val", type=str, help="Validation scope corpus")
    # Test data (only used for vocab generation, not training)
    parser.add_argument("--cfg_test", type=str, help="Test CFG corpus (for vocab only)")
    parser.add_argument("--dfg_test", type=str, help="Test DFG corpus (for vocab only)")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary file")
    
    # Model args
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size")
    parser.add_argument("--layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=60, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    
    # Masking args
    parser.add_argument("--token_mask_prob", type=float, default=0.15, help="Token-level mask probability (MLM)")
    parser.add_argument("--instruction_mask_prob", type=float, default=0.15, help="Instruction-level mask probability (IM)")
    
    # Data sampling
    parser.add_argument("--data_percentage", type=float, default=1.0, help="Percentage of training data to use")
    parser.add_argument("--val_percentage", type=float, default=1.0, help="Percentage of validation data to use")
    
    # Task flags
    parser.add_argument("--enable_imc", action="store_true", help="Enable Instruction Masking for CFG")
    parser.add_argument("--disable_imc", action="store_true", help="Disable Instruction Masking for CFG")
    parser.add_argument("--enable_imd", action="store_true", help="Enable Instruction Masking for DFG")
    parser.add_argument("--enable_mlm", action="store_true", default=True, help="Enable MLM")
    parser.add_argument("--disable_mlm", action="store_true", help="Disable MLM")
    parser.add_argument("--enable_scope", action="store_true", help="Enable Scope Prediction")
    parser.add_argument("--use_address_embedding", action="store_true", default=True, help="Use address embeddings")
    parser.add_argument("--disable_address_embedding", action="store_true", help="Disable address embeddings")
    parser.add_argument("--use_var_embedding", action="store_true", default=True, help="Use var offset embeddings for var(0xXX) tokens")
    parser.add_argument("--disable_var_embedding", action="store_true", help="Disable var offset embeddings")
    
    # Output args
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory")
    parser.add_argument("--save_freq", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--log_freq", type=int, default=1000, help="Log every N batches")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from latest checkpoint if available")
    
    # Device args
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Handle disable flags
    if args.disable_imc:
        args.enable_imc = False
    if args.disable_mlm:
        args.enable_mlm = False
    if args.disable_address_embedding:
        args.use_address_embedding = False
    if args.disable_var_embedding:
        args.use_var_embedding = False
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Setup device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Arguments saved to: {os.path.join(args.output_dir, 'args.json')}")
    
    # Load or create vocabulary
    # Include train, val, and test data for vocab generation
    # Test data is only used for vocab, not for training
    data_files = [
        args.cfg_train,
        args.dfg_train,
        args.cfg_val,
        args.dfg_val,
        args.scope_train,
        args.scope_val,
        args.cfg_test,  # Only for vocab generation
        args.dfg_test,  # Only for vocab generation
    ]
    vocab = create_vocab(data_files, args.vocab, logger=logger)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    logger.info("Creating training dataset with instruction masking...")
    
    # Only load data if at least one task is enabled
    if not (args.enable_imc or args.enable_imd or args.enable_mlm):
        logger.error("At least one task must be enabled (IMC, IMD, or MLM)")
        return
    
    train_dataset = InstructionMaskingDataset(
        cfg_corpus_path=args.cfg_train if args.enable_imc or args.enable_mlm else None,
        dfg_corpus_path=args.dfg_train if args.enable_imd else None,
        vocab=vocab,
        seq_len=args.seq_len,
        on_memory=True,
        token_mask_prob=args.token_mask_prob,
        instruction_mask_prob=args.instruction_mask_prob,
        data_percentage=args.data_percentage,
        train_split=1.0,
        is_train=True,
        enable_imd=args.enable_imd
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Create validation dataset
    val_loader = None
    if args.cfg_val and args.dfg_val:
        logger.info("Creating validation dataset...")
        val_dataset = InstructionMaskingDataset(
            cfg_corpus_path=args.cfg_val if args.enable_imc or args.enable_mlm else None,
            dfg_corpus_path=args.dfg_val if args.enable_imd else None,
            vocab=vocab,
            seq_len=args.seq_len,
            on_memory=True,
            token_mask_prob=args.token_mask_prob,
            instruction_mask_prob=args.instruction_mask_prob,
            data_percentage=args.val_percentage,
            train_split=1.0,
            is_train=True,
            enable_imd=args.enable_imd
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    # Create scope datasets if enabled
    scope_train_loader = None
    scope_val_loader = None
    if args.enable_scope and args.scope_train:
        logger.info(f"Creating scope training dataset from {args.scope_train}...")
        scope_train_dataset = ScopeDataset(
            scope_corpus_path=args.scope_train,
            vocab=vocab,
            seq_len=args.seq_len,
            encoding="utf-8",
            on_memory=True,
            data_percentage=args.data_percentage
        )
        scope_train_loader = DataLoader(
            scope_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        if args.scope_val:
            logger.info(f"Creating scope validation dataset from {args.scope_val}...")
            scope_val_dataset = ScopeDataset(
                scope_corpus_path=args.scope_val,
                vocab=vocab,
                seq_len=args.seq_len,
                encoding="utf-8",
                on_memory=True,
                data_percentage=args.val_percentage
            )
            scope_val_loader = DataLoader(
                scope_val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
    
    # Create model
    logger.info("Creating model (training from scratch)...")
    logger.info("Task configuration:")
    logger.info(f"  - IMC: {args.enable_imc}")
    logger.info(f"  - IMD: {args.enable_imd}")
    logger.info(f"  - MLM: {args.enable_mlm}")
    logger.info(f"  - SCOPE: {args.enable_scope}")
    logger.info(f"  - Address Embedding: {args.use_address_embedding}")
    logger.info(f"  - Var Embedding: {args.use_var_embedding}")
    
    bert = AddressAwareBERT(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        max_len=args.seq_len,
        use_address_embedding=args.use_address_embedding,
        use_var_embedding=args.use_var_embedding
    )
    
    model = AddressAwareBERTForPretraining(
        bert,
        vocab_size=len(vocab),
        enable_mlm=args.enable_mlm,
        enable_imc=args.enable_imc,
        enable_imd=args.enable_imd,
        enable_nsp_cfg=False,
        enable_nsp_dfg=False,
        enable_scope=args.enable_scope
    )
    
    model = model.to(device)
    
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if exists
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint_latest.pt')
    if args.resume and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        logger.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    else:
        logger.info("Resume requested but no checkpoint found at " + checkpoint_path)
        logger.info("Starting training from scratch.")
    
    # Training loop
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 80)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, scope_train_loader,
            optimizer, device, args.log_freq, logger,
            enable_imc=args.enable_imc,
            enable_imd=args.enable_imd,
            enable_mlm=args.enable_mlm,
            enable_scope=args.enable_scope
        )
        
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"  - IMC: {train_metrics['imc_loss']:.4f}")
        logger.info(f"  - IMD: {train_metrics['imd_loss']:.4f}")
        logger.info(f"  - MLM: {train_metrics['mlm_loss']:.4f}")
        logger.info(f"  - Scope: {train_metrics['scope_loss']:.4f}")
        
        # Validate
        if val_loader:
            val_metrics = validate_epoch(
                model, val_loader, scope_val_loader, device, logger,
                enable_imc=args.enable_imc,
                enable_imd=args.enable_imd,
                enable_mlm=args.enable_mlm,
                enable_scope=args.enable_scope
            )
            
            logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
            logger.info(f"  - IMC: {val_metrics['imc_loss']:.4f}")
            logger.info(f"  - IMD: {val_metrics['imd_loss']:.4f}")
            logger.info(f"  - MLM: {val_metrics['mlm_loss']:.4f}")
            logger.info(f"  - Scope: {val_metrics['scope_loss']:.4f}")
            
            # Check for improvement
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                epochs_without_improvement = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'val_metrics': val_metrics,
                }, os.path.join(args.output_dir, 'best_model.pt'))
                
                # Save best BERT
                bert_state = model.module.bert.state_dict() if hasattr(model, 'module') else model.bert.state_dict()
                torch.save(bert_state, os.path.join(args.output_dir, 'best_bert.pt'))
                
                logger.info(f"✓ New best model saved! Val loss: {best_val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'epochs_without_improvement': epochs_without_improvement,
            'train_metrics': train_metrics,
        }, checkpoint_path)
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            periodic_checkpoint = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, periodic_checkpoint)
            logger.info(f"Saved checkpoint to {periodic_checkpoint}")
        
        # Early stopping
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break
        
        scheduler.step()
    
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to: {args.output_dir}")


# ============================================================================
# Trainer Class for train_comparison.py (MLM only)
# ============================================================================
class AddressAwareTrainer:
    """
    Trainer wrapper for Address-Aware BERT with MLM only
    """
    def __init__(self, model, vocab_size, train_dataloader, test_dataloader=None,
                 lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, warmup_steps=10000,
                 with_cuda=True, cuda_devices=None, log_freq=100):
        """
        Args:
            model: AddressAwareBERT model
            vocab_size: Size of vocabulary
            train_dataloader: Training data loader
            test_dataloader: Test data loader (optional)
            lr: Learning rate
            betas: Adam optimizer betas
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            with_cuda: Whether to use CUDA
            cuda_devices: CUDA device IDs
            log_freq: Logging frequency
        """
        self.device = torch.device("cuda:0" if with_cuda and torch.cuda.is_available() else "cpu")
        self.model = AddressAwareBERTForPretraining(model, vocab_size).to(self.device)
        
        # DataParallel if multiple GPUs
        if with_cuda and torch.cuda.is_available() and cuda_devices and len(cuda_devices) > 1:
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.log_freq = log_freq
        
        # Optimizer with warmup
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.warmup_steps = warmup_steps
        self.step = 0
        
        # Loss function for MLM only
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        print(f"Training device: {self.device}")
        print(f"Using CUDA: {with_cuda and torch.cuda.is_available()}")
    
    def train(self, epoch):
        """Train for one epoch with MLM only"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_data, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Update learning rate with warmup
            if self.step < self.warmup_steps:
                lr_scale = min(1., float(self.step + 1) / self.warmup_steps)
                for pg in self.optim.param_groups:
                    pg['lr'] = lr_scale * pg['initial_lr'] if 'initial_lr' in pg else pg['lr']
            self.step += 1
            
            self.optim.zero_grad()
            
            # Process MLM (Masked Language Modeling) only
            # Dataset returns dict with 'mlm' key
            mlm_batch = batch['mlm']
            mlm_input = mlm_batch['bert_input'].to(self.device)
            mlm_segment = mlm_batch['segment_label'].to(self.device)
            mlm_label = mlm_batch['bert_label'].to(self.device)
            binary_pos = mlm_batch['binary_pos'].to(self.device)
            function_pos = mlm_batch['function_pos'].to(self.device)
            bb_pos = mlm_batch['bb_pos'].to(self.device)
            var_offsets = mlm_batch['var_offsets'].to(self.device)
            is_daddr = mlm_batch['is_daddr'].to(self.device)
            
            mlm_output, nsp_output = self.model(mlm_input, mlm_segment, binary_pos, function_pos, bb_pos, var_offsets, is_daddr, corpus_type='cfg')
            mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), mlm_label)
            
            # Backward pass
            mlm_loss.backward()
            self.optim.step()
            
            # Accumulate loss
            total_loss += mlm_loss.item()
            
            # Update progress bar
            if (batch_idx + 1) % self.log_freq == 0:
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        avg_loss = total_loss / len(self.train_data)
        print(f"\nEpoch {epoch} - Avg MLM Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def test(self, epoch):
        """Test for one epoch with MLM only"""
        if self.test_data is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_data, desc=f"Test {epoch}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Process MLM only
                # Dataset returns dict with 'mlm' key
                mlm_batch = batch['mlm']
                mlm_input = mlm_batch['bert_input'].to(self.device)
                mlm_segment = mlm_batch['segment_label'].to(self.device)
                mlm_label = mlm_batch['bert_label'].to(self.device)
                binary_pos = mlm_batch['binary_pos'].to(self.device)
                function_pos = mlm_batch['function_pos'].to(self.device)
                bb_pos = mlm_batch['bb_pos'].to(self.device)
                var_offsets = mlm_batch['var_offsets'].to(self.device)
                is_daddr = mlm_batch['is_daddr'].to(self.device)
                
                mlm_output = self.model(mlm_input, mlm_segment, binary_pos, function_pos, bb_pos, var_offsets, is_daddr)
                mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), mlm_label)
                
                total_loss += mlm_loss.item()
        
        avg_loss = total_loss / len(self.test_data)
        print(f"Test {epoch} - Avg MLM Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path
        
        :param epoch: current epoch number
        :param file_path: model output path (used as-is, no epoch suffix added)
        :return: final_output_path
        """
        output_path = file_path
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get the actual model (unwrap DataParallel if needed)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Save full checkpoint with optimizer state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, output_path)
        
        # Also save just the BERT part (for compatibility)
        bert_state = model_to_save.bert.state_dict()
        bert_path = output_path.replace('.pt', '_bert.pt')
        torch.save(bert_state, bert_path)
        
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


if __name__ == '__main__':
    main()
