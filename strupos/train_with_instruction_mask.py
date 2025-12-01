"""
Training script for Address-Aware BERT with Instruction Masking

Trains a BERT model with:
1. Instruction Masking (IM) - masks entire instructions
2. Masked Language Modeling (MLM) - masks individual tokens  
3. Next Sentence Prediction (NSP-CFG and NSP-DFG)
4. Scope Prediction

New task: IM (Instruction Masking)
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

# Import local modules
from vocab import WordVocab
from dataloader_instruction_mask import InstructionMaskingDataset
from dataloader_scope import ScopeDataset
from model import AddressAwareBERT, AddressAwareBERTForPretraining


def train_epoch(model, data_loader, scope_loader, optimizer, device, log_freq=1000, logger=None,
                enable_im=True, enable_mlm=True, enable_nsp_cfg=False, enable_nsp_dfg=False, enable_scope=False):
    """
    Train for one epoch with IM, MLM, NSP, and Scope tasks.
    
    Args:
        model: AddressAwareBERTForPretraining model
        data_loader: DataLoader for IM+MLM+NSP data
        scope_loader: DataLoader for scope prediction data
        optimizer: Optimizer
        device: Device to train on
        log_freq: Log every N batches
        logger: Logger instance
        enable_im: Enable instruction masking task
        enable_mlm: Enable masked language modeling task
        enable_nsp_cfg: Enable CFG NSP task
        enable_nsp_dfg: Enable DFG NSP task
        enable_scope: Enable scope prediction task
    
    Returns:
        Dictionary with average losses
    """
    model.train()
    
    total_im_loss = 0
    total_mlm_loss = 0
    total_nsp_cfg_loss = 0
    total_nsp_dfg_loss = 0
    total_scope_loss = 0
    total_loss = 0
    
    # Loss functions
    im_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore non-masked tokens
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    scope_criterion = nn.NLLLoss()
    
    # Scope iterator
    scope_iter = iter(scope_loader) if (enable_scope and scope_loader) else None
    
    # Create progress bar with manual updates
    progress_bar = tqdm(total=len(data_loader), desc="Training", disable=False)
    
    for batch_idx, batch in enumerate(data_loader):
        # === Process IM (Instruction Masking) ===
        im_loss = torch.tensor(0.0, device=device)
        if enable_im:
            im_batch = batch['im']
            
            im_input = im_batch['bert_input'].to(device)
            im_labels = im_batch['bert_label'].to(device)
            im_segment = im_batch['segment_label'].to(device)
            im_binary_pos = im_batch['binary_pos'].to(device)
            im_function_pos = im_batch['function_pos'].to(device)
            im_bb_pos = im_batch['bb_pos'].to(device)
            
            if hasattr(model, 'module'):
                im_output = model.module.forward_im(
                    im_input, im_segment,
                    im_binary_pos, im_function_pos, im_bb_pos
                )
            else:
                im_output = model.forward_im(
                    im_input, im_segment,
                    im_binary_pos, im_function_pos, im_bb_pos
                )
            
            im_output = im_output.view(-1, im_output.size(-1))
            im_labels_flat = im_labels.view(-1)
            im_loss = im_criterion(im_output, im_labels_flat)
        
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
            
            # Use standard forward pass for MLM
            mlm_output, _ = model(
                mlm_input, mlm_segment,
                mlm_binary_pos, mlm_function_pos, mlm_bb_pos
            )
            
            mlm_output = mlm_output.view(-1, mlm_output.size(-1))
            mlm_labels_flat = mlm_labels.view(-1)
            mlm_loss = mlm_criterion(mlm_output, mlm_labels_flat)
        
        # === Process NSP-CFG ===
        nsp_cfg_loss = torch.tensor(0.0, device=device)
        if enable_nsp_cfg:
            cfg_nsp_batch = batch['nsp_cfg']
            
            cfg_nsp_input = cfg_nsp_batch['bert_input'].to(device)
            cfg_segment_labels = cfg_nsp_batch['segment_label'].to(device)
            cfg_nsp_binary_pos = cfg_nsp_batch['binary_pos'].to(device)
            cfg_nsp_function_pos = cfg_nsp_batch['function_pos'].to(device)
            cfg_nsp_bb_pos = cfg_nsp_batch['bb_pos'].to(device)
            cfg_nsp_labels = cfg_nsp_batch['is_next'].to(device)
            
            _, cfg_nsp_output = model(
                cfg_nsp_input, cfg_segment_labels,
                cfg_nsp_binary_pos, cfg_nsp_function_pos, cfg_nsp_bb_pos,
                corpus_type='cfg'
            )
            
            nsp_cfg_loss = nsp_criterion(cfg_nsp_output, cfg_nsp_labels.squeeze())
        
        # === Process NSP-DFG ===
        nsp_dfg_loss = torch.tensor(0.0, device=device)
        if enable_nsp_dfg:
            dfg_nsp_batch = batch['nsp_dfg']
            
            dfg_nsp_input = dfg_nsp_batch['bert_input'].to(device)
            dfg_segment_labels = dfg_nsp_batch['segment_label'].to(device)
            dfg_nsp_binary_pos = dfg_nsp_batch['binary_pos'].to(device)
            dfg_nsp_function_pos = dfg_nsp_batch['function_pos'].to(device)
            dfg_nsp_bb_pos = dfg_nsp_batch['bb_pos'].to(device)
            dfg_nsp_labels = dfg_nsp_batch['is_next'].to(device)
            
            _, dfg_nsp_output = model(
                dfg_nsp_input, dfg_segment_labels,
                dfg_nsp_binary_pos, dfg_nsp_function_pos, dfg_nsp_bb_pos,
                corpus_type='dfg'
            )
            
            nsp_dfg_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels.squeeze())
        
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
            scope_labels = scope_batch['scope_label'].to(device)
            
            if hasattr(model, 'module'):
                scope_output = model.module.forward_scope(
                    scope_token_ids, scope_segment_labels,
                    scope_binary_pos, scope_function_pos, scope_bb_pos
                )
            else:
                scope_output = model.forward_scope(
                    scope_token_ids, scope_segment_labels,
                    scope_binary_pos, scope_function_pos, scope_bb_pos
                )
            
            scope_loss = scope_criterion(scope_output, scope_labels)
        
        # Combined loss
        loss = im_loss + mlm_loss + nsp_cfg_loss + nsp_dfg_loss + scope_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate losses
        total_im_loss += im_loss.item()
        total_mlm_loss += mlm_loss.item()
        total_nsp_cfg_loss += nsp_cfg_loss.item()
        total_nsp_dfg_loss += nsp_dfg_loss.item()
        total_scope_loss += scope_loss.item()
        total_loss += loss.item()
        
        # Update progress bar every 100 iterations
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'im': f'{total_im_loss/(batch_idx+1):.4f}' if enable_im else '0',
                'mlm': f'{total_mlm_loss/(batch_idx+1):.4f}' if enable_mlm else '0',
                'nsp_cfg': f'{total_nsp_cfg_loss/(batch_idx+1):.4f}' if enable_nsp_cfg else '0',
                'nsp_dfg': f'{total_nsp_dfg_loss/(batch_idx+1):.4f}' if enable_nsp_dfg else '0',
                'scope': f'{total_scope_loss/(batch_idx+1):.4f}' if enable_scope else '0',
            })
            progress_bar.update(100 if batch_idx > 0 else 1)
        
        # Log to file periodically
        if (batch_idx + 1) % log_freq == 0 and logger:
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(f"Batch {batch_idx+1}/{len(data_loader)} - "
                      f"Loss: {avg_loss:.4f} | IM: {total_im_loss/(batch_idx+1):.4f} | "
                      f"MLM: {total_mlm_loss/(batch_idx+1):.4f} | "
                      f"NSP_CFG: {total_nsp_cfg_loss/(batch_idx+1):.4f} | "
                      f"NSP_DFG: {total_nsp_dfg_loss/(batch_idx+1):.4f} | "
                      f"SCOPE: {total_scope_loss/(batch_idx+1):.4f}")
    
    # Close progress bar
    progress_bar.close()
    
    n_batches = len(data_loader)
    return {
        'total_loss': total_loss / n_batches,
        'im_loss': total_im_loss / n_batches,
        'mlm_loss': total_mlm_loss / n_batches,
        'nsp_cfg_loss': total_nsp_cfg_loss / n_batches,
        'nsp_dfg_loss': total_nsp_dfg_loss / n_batches,
        'scope_loss': total_scope_loss / n_batches,
    }


def validate_epoch(model, data_loader, scope_loader, device, logger=None,
                   enable_im=True, enable_mlm=True, enable_nsp_cfg=False, enable_nsp_dfg=False, enable_scope=False):
    """Validate for one epoch"""
    model.eval()
    
    total_im_loss = 0
    total_mlm_loss = 0
    total_nsp_cfg_loss = 0
    total_nsp_dfg_loss = 0
    total_scope_loss = 0
    total_loss = 0
    
    im_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    scope_criterion = nn.NLLLoss()
    
    scope_iter = iter(scope_loader) if (enable_scope and scope_loader) else None
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # === Process IM ===
            im_loss = torch.tensor(0.0, device=device)
            if enable_im:
                im_batch = batch['im']
                
                im_input = im_batch['bert_input'].to(device)
                im_labels = im_batch['bert_label'].to(device)
                im_segment = im_batch['segment_label'].to(device)
                im_binary_pos = im_batch['binary_pos'].to(device)
                im_function_pos = im_batch['function_pos'].to(device)
                im_bb_pos = im_batch['bb_pos'].to(device)
                
                if hasattr(model, 'module'):
                    im_output = model.module.forward_im(
                        im_input, im_segment,
                        im_binary_pos, im_function_pos, im_bb_pos
                    )
                else:
                    im_output = model.forward_im(
                        im_input, im_segment,
                        im_binary_pos, im_function_pos, im_bb_pos
                    )
                
                im_output = im_output.view(-1, im_output.size(-1))
                im_labels_flat = im_labels.view(-1)
                im_loss = im_criterion(im_output, im_labels_flat)
            
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
                
                mlm_output, _ = model(
                    mlm_input, mlm_segment,
                    mlm_binary_pos, mlm_function_pos, mlm_bb_pos
                )
                
                mlm_output = mlm_output.view(-1, mlm_output.size(-1))
                mlm_labels_flat = mlm_labels.view(-1)
                mlm_loss = mlm_criterion(mlm_output, mlm_labels_flat)
            
            # NSP-CFG
            nsp_cfg_loss = torch.tensor(0.0, device=device)
            if enable_nsp_cfg:
                cfg_nsp_batch = batch['nsp_cfg']
                cfg_nsp_input = cfg_nsp_batch['bert_input'].to(device)
                cfg_segment_labels = cfg_nsp_batch['segment_label'].to(device)
                cfg_nsp_binary_pos = cfg_nsp_batch['binary_pos'].to(device)
                cfg_nsp_function_pos = cfg_nsp_batch['function_pos'].to(device)
                cfg_nsp_bb_pos = cfg_nsp_batch['bb_pos'].to(device)
                cfg_nsp_labels = cfg_nsp_batch['is_next'].to(device)
                
                _, cfg_nsp_output = model(
                    cfg_nsp_input, cfg_segment_labels,
                    cfg_nsp_binary_pos, cfg_nsp_function_pos, cfg_nsp_bb_pos,
                    corpus_type='cfg'
                )
                nsp_cfg_loss = nsp_criterion(cfg_nsp_output, cfg_nsp_labels.squeeze())
            
            # NSP-DFG
            nsp_dfg_loss = torch.tensor(0.0, device=device)
            if enable_nsp_dfg:
                dfg_nsp_batch = batch['nsp_dfg']
                dfg_nsp_input = dfg_nsp_batch['bert_input'].to(device)
                dfg_segment_labels = dfg_nsp_batch['segment_label'].to(device)
                dfg_nsp_binary_pos = dfg_nsp_batch['binary_pos'].to(device)
                dfg_nsp_function_pos = dfg_nsp_batch['function_pos'].to(device)
                dfg_nsp_bb_pos = dfg_nsp_batch['bb_pos'].to(device)
                dfg_nsp_labels = dfg_nsp_batch['is_next'].to(device)
                
                _, dfg_nsp_output = model(
                    dfg_nsp_input, dfg_segment_labels,
                    dfg_nsp_binary_pos, dfg_nsp_function_pos, dfg_nsp_bb_pos,
                    corpus_type='dfg'
                )
                nsp_dfg_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels.squeeze())
            
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
                scope_labels = scope_batch['scope_label'].to(device)
                
                if hasattr(model, 'module'):
                    scope_output = model.module.forward_scope(
                        scope_token_ids, scope_segment_labels,
                        scope_binary_pos, scope_function_pos, scope_bb_pos
                    )
                else:
                    scope_output = model.forward_scope(
                        scope_token_ids, scope_segment_labels,
                        scope_binary_pos, scope_function_pos, scope_bb_pos
                    )
                
                scope_loss = scope_criterion(scope_output, scope_labels)
            
            loss = im_loss + mlm_loss + nsp_cfg_loss + nsp_dfg_loss + scope_loss
            
            total_im_loss += im_loss.item()
            total_mlm_loss += mlm_loss.item()
            total_nsp_cfg_loss += nsp_cfg_loss.item()
            total_nsp_dfg_loss += nsp_dfg_loss.item()
            total_scope_loss += scope_loss.item()
            total_loss += loss.item()
    
    n_batches = len(data_loader)
    return {
        'total_loss': total_loss / n_batches,
        'im_loss': total_im_loss / n_batches,
        'mlm_loss': total_mlm_loss / n_batches,
        'nsp_cfg_loss': total_nsp_cfg_loss / n_batches,
        'nsp_dfg_loss': total_nsp_dfg_loss / n_batches,
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
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary file")
    
    # Model args
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size")
    parser.add_argument("--layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=60, help="Maximum sequence length")
    parser.add_argument("--nsp_content_max", type=int, default=20, help="Max tokens per NSP segment")
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
    parser.add_argument("--nsp_prob", type=float, default=0.5, help="NSP negative sampling probability")
    
    # Data sampling
    parser.add_argument("--data_percentage", type=float, default=1.0, help="Percentage of training data to use")
    parser.add_argument("--val_percentage", type=float, default=1.0, help="Percentage of validation data to use")
    
    # Task flags
    parser.add_argument("--enable_im", action="store_true", default=True, help="Enable Instruction Masking")
    parser.add_argument("--disable_im", action="store_true", help="Disable Instruction Masking")
    parser.add_argument("--enable_mlm", action="store_true", default=True, help="Enable MLM")
    parser.add_argument("--disable_mlm", action="store_true", help="Disable MLM")
    parser.add_argument("--enable_nsp_cfg", action="store_true", help="Enable NSP-CFG")
    parser.add_argument("--enable_nsp_dfg", action="store_true", help="Enable NSP-DFG")
    parser.add_argument("--enable_scope", action="store_true", help="Enable Scope Prediction")
    parser.add_argument("--use_address_embedding", action="store_true", default=True, help="Use address embeddings")
    parser.add_argument("--disable_address_embedding", action="store_true", help="Disable address embeddings")
    
    # Output args
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory")
    parser.add_argument("--save_freq", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--log_freq", type=int, default=1000, help="Log every N batches")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from latest checkpoint if available")
    
    # Device args
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs")
    
    args = parser.parse_args()
    
    # Handle disable flags
    if args.disable_im:
        args.enable_im = False
    if args.disable_mlm:
        args.enable_mlm = False
    if args.disable_address_embedding:
        args.use_address_embedding = False
    
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
    
    # Load vocabulary
    logger.info(f"Loading vocabulary from {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    logger.info("Creating training dataset with instruction masking...")
    train_dataset = InstructionMaskingDataset(
        cfg_corpus_path=args.cfg_train,
        dfg_corpus_path=args.dfg_train,
        vocab=vocab,
        seq_len=args.seq_len,
        nsp_content_max=args.nsp_content_max,
        on_memory=True,
        nsp_prob=args.nsp_prob,
        token_mask_prob=args.token_mask_prob,
        instruction_mask_prob=args.instruction_mask_prob,
        data_percentage=args.data_percentage,
        train_split=1.0,
        is_train=True
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
            cfg_corpus_path=args.cfg_val,
            dfg_corpus_path=args.dfg_val,
            vocab=vocab,
            seq_len=args.seq_len,
            nsp_content_max=args.nsp_content_max,
            on_memory=True,
            nsp_prob=args.nsp_prob,
            token_mask_prob=args.token_mask_prob,
            instruction_mask_prob=args.instruction_mask_prob,
            data_percentage=args.val_percentage,
            train_split=1.0,
            is_train=True
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
            corpus_path=args.scope_train,
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
                corpus_path=args.scope_val,
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
    logger.info(f"  - IM: {args.enable_im}")
    logger.info(f"  - MLM: {args.enable_mlm}")
    logger.info(f"  - NSP-CFG: {args.enable_nsp_cfg}")
    logger.info(f"  - NSP-DFG: {args.enable_nsp_dfg}")
    logger.info(f"  - SCOPE: {args.enable_scope}")
    logger.info(f"  - Address Embedding: {args.use_address_embedding}")
    
    bert = AddressAwareBERT(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        max_len=args.seq_len,
        use_address_embedding=args.use_address_embedding
    )
    
    model = AddressAwareBERTForPretraining(
        bert,
        vocab_size=len(vocab),
        enable_mlm=args.enable_mlm,
        enable_im=args.enable_im,
        enable_nsp_cfg=args.enable_nsp_cfg,
        enable_nsp_dfg=args.enable_nsp_dfg,
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
            enable_im=args.enable_im,
            enable_mlm=args.enable_mlm,
            enable_nsp_cfg=args.enable_nsp_cfg,
            enable_nsp_dfg=args.enable_nsp_dfg,
            enable_scope=args.enable_scope
        )
        
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"  - IM: {train_metrics['im_loss']:.4f}")
        logger.info(f"  - MLM: {train_metrics['mlm_loss']:.4f}")
        logger.info(f"  - NSP-CFG: {train_metrics['nsp_cfg_loss']:.4f}")
        logger.info(f"  - NSP-DFG: {train_metrics['nsp_dfg_loss']:.4f}")
        logger.info(f"  - Scope: {train_metrics['scope_loss']:.4f}")
        
        # Validate
        if val_loader:
            val_metrics = validate_epoch(
                model, val_loader, scope_val_loader, device, logger,
                enable_im=args.enable_im,
                enable_mlm=args.enable_mlm,
                enable_nsp_cfg=args.enable_nsp_cfg,
                enable_nsp_dfg=args.enable_nsp_dfg,
                enable_scope=args.enable_scope
            )
            
            logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
            logger.info(f"  - IM: {val_metrics['im_loss']:.4f}")
            logger.info(f"  - MLM: {val_metrics['mlm_loss']:.4f}")
            logger.info(f"  - NSP-CFG: {val_metrics['nsp_cfg_loss']:.4f}")
            logger.info(f"  - NSP-DFG: {val_metrics['nsp_dfg_loss']:.4f}")
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


if __name__ == '__main__':
    main()
