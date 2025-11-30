"""
Training script for Address-Aware BERT pretraining with Multi-to-One NSP

Trains a BERT model with address-aware positional embeddings on:
1. Masked Language Modeling (MLM)
2. Next Sentence Prediction (NSP) - Multi-to-One strategy
   - Uses N-1 instructions to predict the Nth instruction (e.g., 7 instructions predict 8th)

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
import pickle
from tqdm import tqdm
import json
import logging
from datetime import datetime
import re

# Import local modules
from vocab import WordVocab
from dataloader_multi_to_one import MultiToOneDataset
from dataloader_scope import ScopeDataset
from model import AddressAwareBERT, AddressAwareBERTForPretraining


def preprocess_line(line):
    """
    Remove all address information in parentheses from a line.
    
    Examples:
        mov(0x401000:0.5:0.3:0.2) eax ebx -> mov eax ebx
        address(0x123:0.5:0.3:0.2) -> address
    
    Returns cleaned tokens as a list.
    """
    # Remove all patterns like (0xADDR:pos1:pos2:pos3)
    cleaned = re.sub(r'\(0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+\)', '', line)
    # Split by whitespace and tab, filter out empty strings
    tokens = [tok for tok in cleaned.replace('\t', ' ').split() if tok]
    return tokens


class PreprocessedFile:
    """
    Wrapper that preprocesses lines from a file by removing address info.
    Acts as an iterator that yields lists of tokens.
    """
    def __init__(self, file_handle):
        self.file_handle = file_handle
    
    def __iter__(self):
        for line in self.file_handle:
            yield preprocess_line(line)
            
def create_vocab_if_needed(vocab_path, logger, train_cfg_dataset, train_dfg_dataset, val_cfg_dataset, val_dfg_dataset, test_cfg_dataset, test_dfg_dataset):
    """
    Create vocabulary if it doesn't exist using components from create_vocab.py.
    Also validates existing vocab file is a valid pickle file.
    
    Args:
        vocab_path: Path to vocabulary file
        logger: Logger instance
        *_dataset: Paths to dataset files
    """
    if os.path.exists(vocab_path):
        # Check if it's a valid pickle file
        try:
            with open(vocab_path, "rb") as f:
                pickle.load(f)
            if logger:
                logger.info(f"Vocabulary already exists at {vocab_path}")
            return
        except (pickle.UnpicklingError, UnicodeDecodeError) as e:
            # Invalid pickle file (probably old text format), delete and recreate
            msg = f"Existing vocab file '{vocab_path}' is not a valid pickle file (probably old text format). Deleting and recreating..."
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            os.remove(vocab_path)
    
    if logger:
        logger.info(f"Vocabulary not found at {vocab_path}, creating it now...")
        logger.info("Using WordVocab with max_size=13000, min_freq=1")
    else:
        print(f"Vocabulary not found at {vocab_path}, creating it now...")
        print("Using WordVocab with max_size=13000, min_freq=1")
    
    
    # Check if files exist
    files_to_check = [
        train_cfg_dataset, train_dfg_dataset,
        val_cfg_dataset, val_dfg_dataset,
        test_cfg_dataset, test_dfg_dataset
    ]
    
    for fpath in files_to_check:
        if not os.path.exists(fpath):
            msg = f"ERROR: File not found: {fpath}"
            if logger:
                logger.error(msg)
            else:
                print(msg)
            raise FileNotFoundError(fpath)
        
        msg = f"Found: {fpath}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # Open all files and create vocabulary
    with open(train_cfg_dataset, "r", encoding="utf-8") as f1, \
         open(train_dfg_dataset, "r", encoding="utf-8") as f2, \
         open(val_cfg_dataset, "r", encoding="utf-8") as f3, \
         open(val_dfg_dataset, "r", encoding="utf-8") as f4, \
         open(test_cfg_dataset, "r", encoding="utf-8") as f5, \
         open(test_dfg_dataset, "r", encoding="utf-8") as f6:
        
        # Wrap each file with preprocessing (removes address info)
        preprocessed_files = [
            PreprocessedFile(f1),
            PreprocessedFile(f2),
            PreprocessedFile(f3),
            PreprocessedFile(f4),
            PreprocessedFile(f5),
            PreprocessedFile(f6)
        ]
        
        vocab = WordVocab(
            preprocessed_files,
            max_size=50000,
            min_freq=2
        )
    
    msg = f"VOCAB SIZE: {len(vocab)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # Save vocabulary
    vocab.save_vocab(vocab_path)
    
    msg = f"Vocabulary saved to: {vocab_path}"
    if logger:
        logger.info(msg)
    else:
        print(msg)



def train_epoch(model, data_loader, scope_loader, optimizer, device, log_freq=100, logger=None, enable_mlm=True, enable_nsp_cfg=False, enable_nsp_dfg=False, enable_scope=False):
    """Train for one epoch with paired CFG+DFG samples and scope prediction."""
    model.train()
    
    total_loss = 0
    mlm_loss_total = 0
    nsp_cfg_loss_total = 0
    nsp_dfg_loss_total = 0
    scope_loss_total = 0
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    scope_criterion = nn.CrossEntropyLoss()
    
    progress = tqdm(data_loader, desc="Training", file=sys.stdout)
    
    # Create scope iterator
    scope_iter = iter(scope_loader) if scope_loader is not None else None
    
    for i, batch in enumerate(progress):
        # === Process CFG (MLM + NSP) ===
        # MLM uses cfg_mlm_* keys
        cfg_mlm_input = batch['cfg_mlm_input'].to(device)
        cfg_mlm_binary_pos = batch['cfg_mlm_binary_pos'].to(device)
        cfg_mlm_function_pos = batch['cfg_mlm_function_pos'].to(device)
        cfg_mlm_bb_pos = batch['cfg_mlm_bb_pos'].to(device)
        cfg_mlm_labels = batch['cfg_mlm_label'].to(device)
        
        # NSP uses cfg_nsp_* keys
        cfg_nsp_input = batch['cfg_nsp_input'].to(device)
        cfg_segment_labels = batch['cfg_segment_label'].to(device)
        cfg_nsp_binary_pos = batch['cfg_nsp_binary_pos'].to(device)
        cfg_nsp_function_pos = batch['cfg_nsp_function_pos'].to(device)
        cfg_nsp_bb_pos = batch['cfg_nsp_bb_pos'].to(device)
        cfg_nsp_labels = batch['cfg_is_next'].to(device)
        
        # CFG MLM forward pass (if enabled)
        mlm_loss = torch.tensor(0.0, device=device)
        if enable_mlm:
            cfg_mlm_output, _ = model(
                cfg_mlm_input, 
                torch.zeros_like(cfg_mlm_input),  # No segment labels for MLM
                cfg_mlm_binary_pos, cfg_mlm_function_pos, cfg_mlm_bb_pos, 
                corpus_type='cfg'
            )
            mlm_loss = mlm_criterion(cfg_mlm_output.transpose(1, 2), cfg_mlm_labels)
        
        # CFG NSP forward pass (if enabled)
        nsp_cfg_loss = torch.tensor(0.0, device=device)
        if enable_nsp_cfg:
            _, cfg_nsp_output = model(
                cfg_nsp_input, cfg_segment_labels,
                cfg_nsp_binary_pos, cfg_nsp_function_pos, cfg_nsp_bb_pos, 
                corpus_type='cfg'
            )
            nsp_cfg_loss = nsp_criterion(cfg_nsp_output, cfg_nsp_labels)
        
        # === Process DFG (NSP only, if enabled) ===
        nsp_dfg_loss = torch.tensor(0.0, device=device)
        if enable_nsp_dfg:
            dfg_nsp_input = batch['dfg_nsp_input'].to(device)
            dfg_segment_labels = batch['dfg_segment_label'].to(device)
            dfg_nsp_binary_pos = batch['dfg_nsp_binary_pos'].to(device)
            dfg_nsp_function_pos = batch['dfg_nsp_function_pos'].to(device)
            dfg_nsp_bb_pos = batch['dfg_nsp_bb_pos'].to(device)
            dfg_nsp_labels = batch['dfg_is_next'].to(device)
            
            # DFG forward pass (NO MLM)
            _, dfg_nsp_output = model(
                dfg_nsp_input, dfg_segment_labels,
                dfg_nsp_binary_pos, dfg_nsp_function_pos, dfg_nsp_bb_pos,
                corpus_type='dfg'
            )
            
            # DFG loss (NSP only)
            nsp_dfg_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels)
        
        # === Process Scope (if available and enabled) ===
        scope_loss = torch.tensor(0.0, device=device)
        if enable_scope and scope_iter is not None:
            try:
                scope_batch = next(scope_iter)
            except StopIteration:
                # Restart scope iterator if exhausted
                scope_iter = iter(scope_loader)
                scope_batch = next(scope_iter)
            
            scope_token_ids = scope_batch['bert_input'].to(device)
            scope_segment_labels = scope_batch['segment_label'].to(device)
            scope_binary_pos = scope_batch['binary_pos'].to(device)
            scope_function_pos = scope_batch['function_pos'].to(device)
            scope_bb_pos = scope_batch['bb_pos'].to(device)
            scope_labels = scope_batch['scope_label'].to(device)
            
            # Scope forward pass
            scope_output = model.forward_scope(
                scope_token_ids, scope_segment_labels,
                scope_binary_pos, scope_function_pos, scope_bb_pos
            )
            
            # Scope loss
            scope_loss = scope_criterion(scope_output, scope_labels)
        
        # Combined loss: MLM(CFG) + NSP(CFG) + NSP(DFG) + SCOPE
        loss = mlm_loss + nsp_cfg_loss + nsp_dfg_loss + scope_loss
        
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
        scope_loss_total += scope_loss.item()
        
        # Calculate running averages for display
        avg_loss = total_loss / (i + 1)
        avg_mlm = mlm_loss_total / (i + 1)
        avg_nsp_cfg = nsp_cfg_loss_total / (i + 1)
        avg_nsp_dfg = nsp_dfg_loss_total / (i + 1)
        avg_scope = scope_loss_total / (i + 1)
        
        # Update progress bar every iteration
        progress.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'mlm': f'{avg_mlm:.4f}',
            'nsp_cfg': f'{avg_nsp_cfg:.4f}',
            'nsp_dfg': f'{avg_nsp_dfg:.4f}',
            'scope': f'{avg_scope:.4f}'
        })
        
        # Log to file periodically
        if i % log_freq == 0 and logger:
            logger.info(f"Batch {i}/{len(data_loader)} - "
                      f"Loss: {avg_loss:.4f} | MLM: {avg_mlm:.4f} | "
                      f"NSP_CFG: {avg_nsp_cfg:.4f} | NSP_DFG: {avg_nsp_dfg:.4f} | "
                      f"SCOPE: {avg_scope:.4f}")
    
    return {
        'total_loss': total_loss / len(data_loader),
        'mlm_loss': mlm_loss_total / len(data_loader),
        'nsp_cfg_loss': nsp_cfg_loss_total / len(data_loader),
        'nsp_dfg_loss': nsp_dfg_loss_total / len(data_loader),
        'scope_loss': scope_loss_total / len(data_loader)
    }


def validate(model, data_loader, scope_loader, device, logger=None, enable_mlm=True, enable_nsp_cfg=False, enable_nsp_dfg=False, enable_scope=False):
    """Validate the model on validation set with paired CFG+DFG samples and scope."""
    model.eval()
    
    total_loss = 0
    mlm_loss_total = 0
    nsp_cfg_loss_total = 0
    nsp_dfg_loss_total = 0
    scope_loss_total = 0
    
    mlm_correct = 0
    mlm_total = 0
    nsp_cfg_correct = 0
    nsp_cfg_total = 0
    nsp_dfg_correct = 0
    nsp_dfg_total = 0
    scope_correct = 0
    scope_total = 0
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    nsp_criterion = nn.CrossEntropyLoss()
    scope_criterion = nn.CrossEntropyLoss()
    
    # Create scope iterator
    scope_iter = iter(scope_loader) if scope_loader is not None else None
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation", file=sys.stdout):
            # === Process CFG (MLM + NSP) ===
            # MLM uses cfg_mlm_* keys
            cfg_mlm_input = batch['cfg_mlm_input'].to(device)
            cfg_mlm_binary_pos = batch['cfg_mlm_binary_pos'].to(device)
            cfg_mlm_function_pos = batch['cfg_mlm_function_pos'].to(device)
            cfg_mlm_bb_pos = batch['cfg_mlm_bb_pos'].to(device)
            cfg_mlm_labels = batch['cfg_mlm_label'].to(device)
            
            # NSP uses cfg_nsp_* keys
            cfg_nsp_input = batch['cfg_nsp_input'].to(device)
            cfg_segment_labels = batch['cfg_segment_label'].to(device)
            cfg_nsp_binary_pos = batch['cfg_nsp_binary_pos'].to(device)
            cfg_nsp_function_pos = batch['cfg_nsp_function_pos'].to(device)
            cfg_nsp_bb_pos = batch['cfg_nsp_bb_pos'].to(device)
            cfg_nsp_labels = batch['cfg_is_next'].to(device)
            
            # CFG MLM forward pass (if enabled)
            mlm_loss = torch.tensor(0.0, device=device)
            if enable_mlm:
                cfg_mlm_output, _ = model(
                    cfg_mlm_input,
                    torch.zeros_like(cfg_mlm_input),  # No segment labels for MLM
                    cfg_mlm_binary_pos, cfg_mlm_function_pos, cfg_mlm_bb_pos,
                    corpus_type='cfg'
                )
                mlm_loss = mlm_criterion(cfg_mlm_output.transpose(1, 2), cfg_mlm_labels)
                
                # CFG MLM accuracy
                mask = cfg_mlm_labels != -1
                if mask.any():
                    mlm_pred = torch.argmax(cfg_mlm_output[mask], dim=-1)
                    mlm_correct += (mlm_pred == cfg_mlm_labels[mask]).sum().item()
                    mlm_total += mask.sum().item()
            
            # CFG NSP forward pass (if enabled)
            nsp_cfg_loss = torch.tensor(0.0, device=device)
            if enable_nsp_cfg:
                _, cfg_nsp_output = model(
                    cfg_nsp_input, cfg_segment_labels,
                    cfg_nsp_binary_pos, cfg_nsp_function_pos, cfg_nsp_bb_pos,
                    corpus_type='cfg'
                )
                nsp_cfg_loss = nsp_criterion(cfg_nsp_output, cfg_nsp_labels)
                
                # CFG NSP accuracy
                nsp_cfg_pred = torch.argmax(cfg_nsp_output, dim=-1)
                nsp_cfg_correct += (nsp_cfg_pred == cfg_nsp_labels).sum().item()
                nsp_cfg_total += len(cfg_nsp_labels)
            
            # === Process DFG (NSP only, if enabled) ===
            nsp_dfg_loss = torch.tensor(0.0, device=device)
            if enable_nsp_dfg:
                dfg_nsp_input = batch['dfg_nsp_input'].to(device)
                dfg_segment_labels = batch['dfg_segment_label'].to(device)
                dfg_nsp_binary_pos = batch['dfg_nsp_binary_pos'].to(device)
                dfg_nsp_function_pos = batch['dfg_nsp_function_pos'].to(device)
                dfg_nsp_bb_pos = batch['dfg_nsp_bb_pos'].to(device)
                dfg_nsp_labels = batch['dfg_is_next'].to(device)
                
                # DFG forward pass (NO MLM)
                _, dfg_nsp_output = model(
                    dfg_nsp_input, dfg_segment_labels,
                    dfg_nsp_binary_pos, dfg_nsp_function_pos, dfg_nsp_bb_pos,
                    corpus_type='dfg'
                )
                
                # DFG loss
                nsp_dfg_loss = nsp_criterion(dfg_nsp_output, dfg_nsp_labels)
                
                # DFG accuracy
                nsp_dfg_pred = torch.argmax(dfg_nsp_output, dim=-1)
                nsp_dfg_correct += (nsp_dfg_pred == dfg_nsp_labels).sum().item()
                nsp_dfg_total += len(dfg_nsp_labels)
            
            # === Process Scope (if available and enabled) ===
            scope_loss = torch.tensor(0.0, device=device)
            if enable_scope and scope_iter is not None:
                try:
                    scope_batch = next(scope_iter)
                except StopIteration:
                    # Restart scope iterator if exhausted
                    scope_iter = iter(scope_loader)
                    scope_batch = next(scope_iter)
                
                scope_token_ids = scope_batch['bert_input'].to(device)
                scope_segment_labels = scope_batch['segment_label'].to(device)
                scope_binary_pos = scope_batch['binary_pos'].to(device)
                scope_function_pos = scope_batch['function_pos'].to(device)
                scope_bb_pos = scope_batch['bb_pos'].to(device)
                scope_labels = scope_batch['scope_label'].to(device)
                
                # Scope forward pass
                scope_output = model.forward_scope(
                    scope_token_ids, scope_segment_labels,
                    scope_binary_pos, scope_function_pos, scope_bb_pos
                )
                
                # Scope loss
                scope_loss = scope_criterion(scope_output, scope_labels)
                
                # Scope accuracy
                scope_pred = torch.argmax(scope_output, dim=-1)
                scope_correct += (scope_pred == scope_labels).sum().item()
                scope_total += len(scope_labels)
            
            # Combined loss
            loss = mlm_loss + nsp_cfg_loss + nsp_dfg_loss + scope_loss
            
            # Accumulate losses
            total_loss += loss.item()
            mlm_loss_total += mlm_loss.item()
            nsp_cfg_loss_total += nsp_cfg_loss.item()
            nsp_dfg_loss_total += nsp_dfg_loss.item()
            scope_loss_total += scope_loss.item()
    
    # Calculate accuracies
    mlm_acc = mlm_correct / mlm_total if mlm_total > 0 else 0
    nsp_cfg_acc = nsp_cfg_correct / nsp_cfg_total if nsp_cfg_total > 0 else 0
    nsp_dfg_acc = nsp_dfg_correct / nsp_dfg_total if nsp_dfg_total > 0 else 0
    scope_acc = scope_correct / scope_total if scope_total > 0 else 0
    
    return {
        'total_loss': total_loss / len(data_loader),
        'mlm_loss': mlm_loss_total / len(data_loader),
        'nsp_cfg_loss': nsp_cfg_loss_total / len(data_loader),
        'nsp_dfg_loss': nsp_dfg_loss_total / len(data_loader),
        'scope_loss': scope_loss_total / len(data_loader),
        'mlm_acc': mlm_acc,
        'nsp_cfg_acc': nsp_cfg_acc,
        'nsp_dfg_acc': nsp_dfg_acc,
        'scope_acc': scope_acc
    }


def main():
    # Create args namespace from command-line arguments
    parser = argparse.ArgumentParser()
    
    # Data args
    parser.add_argument("--cfg_train", type=str, required=True, help="Path to CFG training data")
    parser.add_argument("--dfg_train", type=str, required=True, help="Path to DFG training data")
    parser.add_argument("--scope_train", type=str, default=None, help="Path to scope training data (optional)")
    parser.add_argument("--cfg_val", type=str, default=None, help="Path to CFG validation data (optional)")
    parser.add_argument("--dfg_val", type=str, default=None, help="Path to DFG validation data (optional)")
    parser.add_argument("--scope_val", type=str, default=None, help="Path to scope validation data (optional)")
    parser.add_argument("--cfg_test", type=str, default=None, help="Path to CFG test data (optional)")
    parser.add_argument("--dfg_test", type=str, default=None, help="Path to DFG test data (optional)")
    parser.add_argument("--vocab", type=str, default="./vocab.pkl", help="Path to vocabulary file (pickle format)")
    parser.add_argument("--data_percentage", type=float, default=1.0, help="Percentage of training dataset to use (0.0-1.0)")
    parser.add_argument("--val_percentage", type=float, default=1.0, help="Percentage of validation dataset to use (0.0-1.0)")
    
    # Model args
    parser.add_argument("--hidden", type=int, default=768, help="Hidden size")
    parser.add_argument("--layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--attn_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--nsp_content_max", type=int, default=20, help="Maximum content length for NSP pairs (CFG/DFG)")
    parser.add_argument("--instruction_level_segment", action="store_true", default=False, help="Use instruction-level segment IDs (each instruction gets unique segment)")
    
    # Task selection args (for ablation studies)
    parser.add_argument("--enable_mlm", action="store_true", default=True, help="Enable Masked Language Modeling")
    parser.add_argument("--disable_mlm", action="store_true", help="Disable Masked Language Modeling")
    parser.add_argument("--enable_nsp_cfg", action="store_true", default=False, help="Enable NSP for CFG")
    parser.add_argument("--enable_nsp_dfg", action="store_true", default=False, help="Enable NSP for DFG")
    parser.add_argument("--enable_scope", action="store_true", default=False, help="Enable Scope prediction")
    parser.add_argument("--use_address_embedding", action="store_true", default=False, help="Use address-aware embeddings")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")
    
    # Masking args
    parser.add_argument("--mask_prob", type=float, default=0.15, help="Probability of masking a token")
    parser.add_argument("--nsp_prob", type=float, default=0.5, help="Probability of negative NSP sample")
    
    # Output args
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory")
    parser.add_argument("--save_freq", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--log_freq", type=int, default=100, help="Log every N batches")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from latest checkpoint if available")
    
    # Device args
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs")
    
    args = parser.parse_args()
    
    # Handle disable_mlm flag
    if args.disable_mlm:
        args.enable_mlm = False
    
    # Setup device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output and log directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"train_{timestamp}.log")
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Using device: {device}")
    
    # Save args
    args_file = os.path.join(args.output_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Arguments saved to: {args_file}")
    
    # Create vocabulary if it doesn't exist (runs create_vocab.py)
    create_vocab_if_needed(args.vocab,logger,args.cfg_train, args.dfg_train, args.cfg_val, args.dfg_val, args.cfg_test, args.dfg_test)
    
    # Load vocabulary
    logger.info(f"Loading vocabulary from {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    logger.info("Creating training dataset...")
    
    # Use MultiToOneDataset for multi-to-one NSP strategy
    logger.info("Using MULTI-TO-ONE NSP strategy (first 7 instructions predict 8th)")
    if args.instruction_level_segment:
        logger.info("  Segment mode: INSTRUCTION-LEVEL (each instruction gets unique segment ID)")
    else:
        logger.info("  Segment mode: STANDARD (context=segment1, target=segment2)")
    
    train_dataset = MultiToOneDataset(
        cfg_corpus_path=args.cfg_train,
        dfg_corpus_path=args.dfg_train,
        vocab=vocab,
        seq_len=args.seq_len,
        nsp_content_max=args.nsp_content_max,
        on_memory=True,
        nsp_prob=args.nsp_prob,
        mask_prob=args.mask_prob,
        data_percentage=args.data_percentage,
        train_split=1.0,  # Always 1.0 since data is pre-split into separate files
        is_train=True,
        instruction_level_segment=args.instruction_level_segment
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Create validation dataset from separate files
    val_loader = None
    if args.cfg_val and args.dfg_val:
        logger.info("Creating validation dataset from separate files...")
        val_dataset = MultiToOneDataset(
            cfg_corpus_path=args.cfg_val,
            dfg_corpus_path=args.dfg_val,
            vocab=vocab,
            seq_len=args.seq_len,
            nsp_content_max=args.nsp_content_max,
            on_memory=True,
            nsp_prob=args.nsp_prob,
            mask_prob=args.mask_prob,
            data_percentage=args.val_percentage,
            train_split=1.0,  # Always 1.0 since data is pre-split
            is_train=True,
            instruction_level_segment=args.instruction_level_segment
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        logger.warning("No validation data provided. Training without validation.")
    
    # Create scope datasets if provided
    scope_train_loader = None
    scope_val_loader = None
    if args.scope_train:
        logger.info(f"Creating scope training dataset from {args.scope_train}...")
        scope_train_dataset = ScopeDataset(
            scope_corpus_path=args.scope_train,
            vocab=vocab,
            seq_len=args.seq_len,
            on_memory=True,
            data_percentage=args.data_percentage,
            train_split=1.0,  # Always 1.0 since data is pre-split
            is_train=True
        )
        
        scope_train_loader = DataLoader(
            scope_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
    
    # Create scope validation dataset if provided
    if args.scope_val:
        logger.info(f"Creating scope validation dataset from {args.scope_val}...")
        scope_val_dataset = ScopeDataset(
            scope_corpus_path=args.scope_val,
            vocab=vocab,
            seq_len=args.seq_len,
            on_memory=True,
            data_percentage=args.val_percentage,
            train_split=1.0,  # Always 1.0 since data is pre-split
            is_train=False
        )
        
        scope_val_loader = DataLoader(
            scope_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    # Create test dataset if provided
    test_loader = None
    if args.cfg_test and args.dfg_test:
        logger.info("Creating test dataset...")
        test_dataset = MultiToOneDataset(
            cfg_corpus_path=args.cfg_test,
            dfg_corpus_path=args.dfg_test,
            vocab=vocab,
            seq_len=args.seq_len,
            nsp_content_max=args.nsp_content_max,
            on_memory=True,
            nsp_prob=args.nsp_prob,
            mask_prob=args.mask_prob,
            data_percentage=1.0,  # Use all test data
            train_split=1.0,
            is_train=True,
            instruction_level_segment=args.instruction_level_segment
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create model
    logger.info("Creating model (training from scratch - NO pre-trained PalmTree)...")
    logger.info(f"Task configuration:")
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
        len(vocab),
        enable_mlm=args.enable_mlm,
        enable_nsp_cfg=args.enable_nsp_cfg,
        enable_nsp_dfg=args.enable_nsp_dfg,
        enable_scope=args.enable_scope
    )
    model = model.to(device)
    
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    
    # Resume from checkpoint if requested and exists
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
    
    if args.resume and os.path.exists(checkpoint_path):
        logger.info(f"Found checkpoint: {checkpoint_path}")
        logger.info("Resuming training from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
        logger.info(f"Best validation loss so far: {best_val_loss:.4f}")
        logger.info(f"Epochs without improvement: {epochs_without_improvement}")
    elif args.resume:
        logger.info(f"Resume requested but no checkpoint found at {checkpoint_path}")
        logger.info("Starting training from scratch.")
    else:
        logger.info("Starting training from scratch.")
    
    # Training loop
    logger.info("Starting training...")
    logger.info("="*80)
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-"*80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, scope_train_loader, optimizer, device, args.log_freq, logger, 
                                    args.enable_mlm, args.enable_nsp_cfg, args.enable_nsp_dfg, args.enable_scope)
        scheduler.step()
        
        train_log = (f"Train Loss: {train_metrics['total_loss']:.4f} | "
                    f"MLM: {train_metrics['mlm_loss']:.4f} | "
                    f"NSP_CFG: {train_metrics['nsp_cfg_loss']:.4f} | "
                    f"NSP_DFG: {train_metrics['nsp_dfg_loss']:.4f} | "
                    f"SCOPE: {train_metrics['scope_loss']:.4f}")
        logger.info(train_log)
        
        # Validate
        if val_loader is not None:
            val_metrics = validate(model, val_loader, scope_val_loader, device, logger,
                                  args.enable_mlm, args.enable_nsp_cfg, args.enable_nsp_dfg, args.enable_scope)
            val_log = (f"Val Loss: {val_metrics['total_loss']:.4f} | "
                      f"MLM: {val_metrics['mlm_loss']:.4f} ({val_metrics['mlm_acc']:.2%}) | "
                      f"NSP_CFG: {val_metrics['nsp_cfg_loss']:.4f} ({val_metrics['nsp_cfg_acc']:.2%}) | "
                      f"NSP_DFG: {val_metrics['nsp_dfg_loss']:.4f} ({val_metrics['nsp_dfg_acc']:.2%}) | "
                      f"SCOPE: {val_metrics['scope_loss']:.4f} ({val_metrics['scope_acc']:.2%})")
            logger.info(val_log)
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                
                # Save full model state
                best_model_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, best_model_path)
                
                # Save BERT encoder separately (for embedding extraction like original PalmTree)
                if hasattr(model, 'module'):
                    # DataParallel case
                    bert_to_save = model.module.bert
                else:
                    bert_to_save = model.bert
                best_bert_path = os.path.join(args.output_dir, "best_bert.pt")
                torch.save(bert_to_save, best_bert_path)
                
                logger.info(f"Saved best model (val_loss: {best_val_loss:.4f})")
                logger.info(f"  - Full model: {best_model_path}")
                logger.info(f"  - BERT encoder: {best_bert_path}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")
                
                # Early stopping check
                if epochs_without_improvement >= args.early_stopping_patience:
                    logger.info("="*80)
                    logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                    logger.info(f"Best validation loss: {best_val_loss:.4f}")
                    logger.info("="*80)
                    break
        
        # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'epochs_without_improvement': epochs_without_improvement,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Optionally save periodic checkpoints
        if (epoch + 1) % args.save_freq == 0:
            periodic_checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'epochs_without_improvement': epochs_without_improvement,
            }, periodic_checkpoint_path)
            logger.info(f"Saved periodic checkpoint: {periodic_checkpoint_path}")
    
    
    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final test evaluation if test data provided
    if test_loader is not None:
        logger.info("\n" + "="*80)
        logger.info("Running final evaluation on test set...")
        logger.info("="*80)
        
        # Load best model
        best_model_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Best model from epoch {checkpoint['epoch']} (val_loss: {checkpoint['val_loss']:.4f})")
        else:
            logger.info("WARNING: Best model not found, using current model state")
        
        # Evaluate on test set
        test_metrics = validate(model, test_loader, scope_val_loader, device, logger,
                               args.enable_mlm, args.enable_nsp_cfg, args.enable_nsp_dfg, args.enable_scope)
        
        logger.info("\n" + "="*80)
        logger.info("TEST SET RESULTS:")
        logger.info("="*80)
        logger.info(f"Total Loss: {test_metrics['total_loss']:.4f}")
        if args.enable_mlm:
            logger.info(f"MLM Loss: {test_metrics['mlm_loss']:.4f} | Accuracy: {test_metrics['mlm_acc']:.2%}")
        if args.enable_nsp_cfg:
            logger.info(f"NSP-CFG Loss: {test_metrics['nsp_cfg_loss']:.4f} | Accuracy: {test_metrics['nsp_cfg_acc']:.2%}")
        if args.enable_nsp_dfg:
            logger.info(f"NSP-DFG Loss: {test_metrics['nsp_dfg_loss']:.4f} | Accuracy: {test_metrics['nsp_dfg_acc']:.2%}")
        if args.enable_scope:
            logger.info(f"SCOPE Loss: {test_metrics['scope_loss']:.4f} | Accuracy: {test_metrics['scope_acc']:.2%}")
        logger.info("="*80)
        
        # Save test results to file
        test_results_file = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test results saved to: {test_results_file}")
    
    logger.info(f"\nLogs saved to: {log_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
