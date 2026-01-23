"""
Finetune jTrans_instr model on function similarity task.
Based on baseline finetune.py but adapted for instruction-level model.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader

# Import instruction-level data loader
from data_json_instr import load_paired_data_json_instr, FunctionDataset_CL_Load_JSON_Instr

# Import instruction-level model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pretrain'))
from model_instr import InstrBertModel


def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=name
    )
    logger = logging.getLogger(__name__)
    s_handle = logging.StreamHandler(sys.stdout)
    s_handle.setLevel(logging.INFO)
    s_handle.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s")
    )
    logger.addHandler(s_handle)
    return logger


class TripletCosLoss(nn.Module):
    """Triplet loss with cosine similarity."""
    
    def __init__(self, margin=0.2):
        super(TripletCosLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)
        loss = (self.margin - (pos_sim - neg_sim)).clamp(min=1e-6).mean()
        return loss


class InstrBertWrapper(nn.Module):
    """Wrapper for InstrBertModel to be compatible with finetune interface."""
    
    def __init__(self, model):
        super(InstrBertWrapper, self).__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask, token_type_ids, instruction_ids):
        """Forward pass that returns pooler_output."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            instruction_ids=instruction_ids
        )
        # Return dict with pooler_output for compatibility
        return type('obj', (object,), {'pooler_output': outputs[1]})()


def train_epoch(model, train_loader, optimizer, loss_fn, epoch, args, logger):
    """Train for one epoch."""
    model.train()
    train_iterator = tqdm(train_loader)
    loss_list = []
    
    for i, batch_data in enumerate(train_iterator):
        # Unpack batch: 3 × (input_ids, attention_mask, token_type_ids, instruction_ids)
        (anchor_ids, pos_ids, neg_ids,
         anchor_mask, pos_mask, neg_mask,
         anchor_seg, pos_seg, neg_seg,
         anchor_instr_ids, pos_instr_ids, neg_instr_ids) = batch_data
        
        # Move to GPU
        anchor_ids = anchor_ids.cuda()
        pos_ids = pos_ids.cuda()
        neg_ids = neg_ids.cuda()
        anchor_mask = anchor_mask.cuda()
        pos_mask = pos_mask.cuda()
        neg_mask = neg_mask.cuda()
        anchor_seg = anchor_seg.cuda()
        pos_seg = pos_seg.cuda()
        neg_seg = neg_seg.cuda()
        anchor_instr_ids = anchor_instr_ids.cuda()
        pos_instr_ids = pos_instr_ids.cuda()
        neg_instr_ids = neg_instr_ids.cuda()
        
        optimizer.zero_grad()
        
        # Forward pass
        anchor_output = model(anchor_ids, anchor_mask, anchor_seg, anchor_instr_ids)
        anchor_emb = anchor_output.pooler_output
        
        pos_output = model(pos_ids, pos_mask, pos_seg, pos_instr_ids)
        pos_emb = pos_output.pooler_output
        
        neg_output = model(neg_ids, neg_mask, neg_seg, neg_instr_ids)
        neg_emb = neg_output.pooler_output
        
        # Compute loss
        loss = loss_fn(anchor_emb, pos_emb, neg_emb)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
        
        # Logging
        if (i + 1) % args.log_every == 0:
            avg_loss = sum(loss_list[-args.log_every:]) / min(args.log_every, len(loss_list))
            lr = optimizer.param_groups[0]["lr"]
            train_iterator.set_description(
                f"[Epoch {epoch}/{args.epoch}] Loss: {avg_loss:.4f}, LR: {lr:.2e}"
            )
    
    avg_epoch_loss = sum(loss_list) / len(loss_list)
    logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


def main():
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument('--func_blocks', required=True, help='Path to func_blocks_instr.json')
    parser.add_argument('--ground_truth', required=True, help='Path to ground_truth_instr.json')
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer directory')
    
    # Model paths
    parser.add_argument('--model_path', required=True, help='Path to pretrained model checkpoint')
    parser.add_argument('--output_path', required=True, help='Output directory for finetuned models')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=5, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--freeze_cnt', type=int, default=10, help='Number of layers to freeze')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='Ratio of data to use (0.001 for quick test)')
    parser.add_argument('--log_every', type=int, default=100, help='Log every N steps')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Logger
    logger = get_logger(os.path.join(args.output_path, 'finetune.log'))
    logger.info("=" * 80)
    logger.info("jTrans_instr Finetuning")
    logger.info("=" * 80)
    logger.info(f"Arguments: {args}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    
    # Load data
    logger.info("Loading training data...")
    train_funcs, train_meta = load_paired_data_json_instr(
        args.func_blocks,
        args.ground_truth,
        opt=['O0', 'O1', 'O2', 'O3'],
        data_ratio=args.data_ratio
    )
    
    train_dataset = FunctionDataset_CL_Load_JSON_Instr(train_funcs, tokenizer, maxlen=512)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        prefetch_factor=2
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} samples")
    
    # Load model
    logger.info(f"Loading pretrained model from {args.model_path}...")
    base_model = InstrBertModel.from_pretrained(args.model_path)
    model = InstrBertWrapper(base_model)
    
    # Freeze layers
    if args.freeze_cnt > 0:
        logger.info(f"Freezing first {args.freeze_cnt} layers...")
        for param in base_model.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(args.freeze_cnt):
            for param in base_model.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    # Loss function
    loss_fn = TripletCosLoss(margin=0.2)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(1, args.epoch + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, loss_fn, epoch, args, logger)
        
        # Save checkpoint
        save_path = os.path.join(args.output_path, f"finetune_epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        
        # Save the base model (unwrap from DataParallel if needed)
        model_to_save = model.module.model if hasattr(model, 'module') else model.model
        model_to_save.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        logger.info(f"Saved checkpoint to {save_path}")
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
