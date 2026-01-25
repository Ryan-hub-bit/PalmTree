"""
Baseline Dataloader for jTrans Pretraining

Pure MLM + JTP (Jump-Target Prediction) without address-aware features.
Uses original jTrans approach: position_embeddings = word_embeddings

Tasks:
1. MLM: Mask regular tokens (15% probability) and predict them
2. JTP: Mask JUMP_ADDR_X tokens (20% probability) and predict target position X
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import sys
import os
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from readidadata import parse_asm


class BaselinePretrainingDataset(Dataset):
    """
    Baseline dataset for pretraining with MLM and JTP.
    
    No address-aware features - just standard MLM + Jump-Target Prediction.
    """
    
    def __init__(
        self,
        data_path,
        tokenizer,
        max_len=512,
        mlm_probability=0.15,
        jtp_probability=0.20,
        convert_jump_addr=True,
        on_memory=True
    ):
        """
        Args:
            data_path: Path to preprocessed data file (one function per line)
            tokenizer: jTrans tokenizer
            max_len: Maximum sequence length
            mlm_probability: Probability of masking regular tokens
            jtp_probability: Probability of masking jump tokens
            convert_jump_addr: Whether to convert jump addresses to JUMP_ADDR_X
            on_memory: Load all data into memory
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_probability = mlm_probability
        self.jtp_probability = jtp_probability
        self.convert_jump_addr = convert_jump_addr
        self.on_memory = on_memory
        
        # Special tokens
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.unk_token_id = tokenizer.unk_token_id
        
        # Load data
        if on_memory:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.lines = [line.strip() for line in f if line.strip()]
        else:
            # Count lines for __len__
            with open(data_path, 'r', encoding='utf-8') as f:
                self.num_lines = sum(1 for line in f if line.strip())
    
    def __len__(self):
        if self.on_memory:
            return len(self.lines)
        else:
            return self.num_lines
    
    def _get_line(self, idx):
        """Get a line from the dataset."""
        if self.on_memory:
            return self.lines[idx]
        else:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == idx:
                        return line.strip()
    
    def _extract_jump_positions(self, tokens):
        """
        Extract jump tokens and their target positions.
        
        Note: tokens already include [CLS] at position 0, so we need to adjust
        target positions by +1 to account for CLS insertion.
        
        Returns:
            jump_positions: List of (token_idx, target_position)
        """
        jump_positions = []
        jump_pattern = re.compile(r'JUMP_ADDR_(\d+)')
        
        for i, token in enumerate(tokens):
            match = jump_pattern.match(token)
            if match:
                target_pos = int(match.group(1))
                # Adjust target position to account for CLS token at position 0
                adjusted_target_pos = target_pos + 1
                
                # Ensure target position is within sequence length
                if adjusted_target_pos < self.max_len:
                    jump_positions.append((i, adjusted_target_pos))
        
        return jump_positions
    
    def _apply_mlm_mask(self, input_ids, labels, jump_positions):
        """
        Apply MLM masking to regular tokens (not jump tokens).
        
        Args:
            input_ids: Token IDs
            labels: Labels for MLM loss
            jump_positions: List of (idx, target) to avoid masking
            
        Returns:
            Masked input_ids and labels
        """
        jump_indices = set([idx for idx, _ in jump_positions])
        
        for i in range(len(input_ids)):
            # Skip special tokens and jump tokens
            if input_ids[i] in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                continue
            if i in jump_indices:
                continue
            
            # Mask with probability
            if random.random() < self.mlm_probability:
                labels[i] = input_ids[i]
                
                # 80% replace with [MASK]
                prob = random.random()
                if prob < 0.8:
                    input_ids[i] = self.mask_token_id
                # 10% replace with random token
                elif prob < 0.9:
                    input_ids[i] = random.randint(0, len(self.tokenizer) - 1)
                # 10% keep original
        
        return input_ids, labels
    
    def _apply_jtp_mask(self, input_ids, jtp_labels, jump_positions):
        """
        Apply JTP masking to jump tokens.
        
        Args:
            input_ids: Token IDs
            jtp_labels: Labels for JTP loss (target positions)
            jump_positions: List of (idx, target_position)
            
        Returns:
            Masked input_ids and jtp_labels
        """
        for idx, target_pos in jump_positions:
            if random.random() < self.jtp_probability:
                # Set label to target position
                jtp_labels[idx] = target_pos
                
                # Mask the jump token
                # 80% replace with [MASK]
                prob = random.random()
                if prob < 0.8:
                    input_ids[idx] = self.mask_token_id
                # 10% replace with random token
                elif prob < 0.9:
                    input_ids[idx] = random.randint(0, len(self.tokenizer) - 1)
                # 10% keep original
        
        return input_ids, jtp_labels
    
    def __getitem__(self, idx):
        """
        Get a training example.
        
        Returns:
            Dictionary with:
                - input_ids: Masked token IDs
                - attention_mask: Attention mask
                - token_type_ids: Token type IDs (all 0s)
                - mlm_labels: Labels for MLM (-100 for non-masked)
                - jtp_labels: Labels for JTP (-100 for non-jump)
        """
        # Get function text
        line = self._get_line(idx)
        
        # Tokenize
        tokens = line.split()
        
        # Truncate if needed
        if len(tokens) > self.max_len - 2:  # Reserve space for [CLS] and [SEP]
            tokens = tokens[:self.max_len - 2]
        
        # Add special tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Extract jump positions (before converting to IDs)
        jump_positions = self._extract_jump_positions(tokens)
        
        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Create labels (initialized to -100 for ignore)
        mlm_labels = [-100] * len(input_ids)
        jtp_labels = [-100] * len(input_ids)
        
        # Apply MLM masking (excludes jump tokens)
        input_ids, mlm_labels = self._apply_mlm_mask(input_ids, mlm_labels, jump_positions)
        
        # Apply JTP masking (only jump tokens)
        input_ids, jtp_labels = self._apply_jtp_mask(input_ids, jtp_labels, jump_positions)
        
        # Padding
        seq_len = len(input_ids)
        attention_mask = [1] * seq_len
        token_type_ids = [0] * seq_len
        
        # Pad to max_len
        padding_len = self.max_len - seq_len
        if padding_len > 0:
            input_ids += [self.pad_token_id] * padding_len
            attention_mask += [0] * padding_len
            token_type_ids += [0] * padding_len
            mlm_labels += [-100] * padding_len
            jtp_labels += [-100] * padding_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
            'jtp_labels': torch.tensor(jtp_labels, dtype=torch.long)
        }


def create_baseline_dataloaders(
    train_path,
    tokenizer,
    batch_size=32,
    max_len=512,
    mlm_probability=0.15,
    jtp_probability=0.20,
    num_workers=4
):
    """
    Create train dataloader for baseline pretraining.
    
    Args:
        train_path: Path to training data
        tokenizer: jTrans tokenizer
        batch_size: Batch size
        max_len: Maximum sequence length
        mlm_probability: MLM masking probability
        jtp_probability: JTP masking probability
        num_workers: Number of data loading workers
        
    Returns:
        train_loader
    """
    train_dataset = BaselinePretrainingDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_len=max_len,
        mlm_probability=mlm_probability,
        jtp_probability=jtp_probability,
        on_memory=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader
