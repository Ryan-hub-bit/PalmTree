"""
Instruction-Level Dataloader for jTrans_instr Pretraining

MLM + JTP (Jump-Target Prediction) with instruction-level embeddings.

Key differences from baseline:
1. Generates instruction_ids tensor mapping each token to its instruction index
2. Jump addresses are instr_addr_{i} (instruction indices) not JUMP_ADDR_X (token positions)
3. JTP predicts instruction index (0-200) not token position (0-511)

Tasks:
1. MLM: Mask regular tokens (15% probability) and predict them
2. JTP: Mask instr_addr_{i} tokens (20% probability) and predict target instruction index i
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import re


class InstrPretrainingDataset(Dataset):
    """
    Instruction-level dataset for pretraining with MLM and JTP.
    
    Generates instruction_ids tensor to map each token to its parent instruction.
    """
    
    def __init__(
        self,
        data_path,
        tokenizer,
        max_len=512,
        max_instructions=201,
        mlm_probability=0.15,
        jtp_probability=0.20,
        on_memory=True,
        data_ratio=1.0
    ):
        """
        Args:
            data_path: Path to preprocessed data file (one function per line)
            tokenizer: jTrans tokenizer with instr_addr tokens
            max_len: Maximum sequence length (token-level)
            max_instructions: Maximum number of instructions (default: 201 for 0-200)
            mlm_probability: Probability of masking regular tokens
            jtp_probability: Probability of masking instr_addr tokens
            on_memory: Load all data into memory
            data_ratio: Fraction of data to use (0.0001 = 0.01%, 1.0 = 100%)
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_instructions = max_instructions
        self.mlm_probability = mlm_probability
        self.jtp_probability = jtp_probability
        self.on_memory = on_memory
        self.data_ratio = data_ratio
        
        # Special tokens
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.unk_token_id = tokenizer.unk_token_id
        
        # Load data
        if on_memory:
            with open(data_path, 'r', encoding='utf-8') as f:
                all_lines = [line.strip() for line in f if line.strip()]
            # Sample data according to ratio
            if data_ratio < 1.0:
                import random
                num_samples = max(1, int(len(all_lines) * data_ratio))
                self.lines = random.sample(all_lines, num_samples)
                print(f"[INFO] Using {num_samples}/{len(all_lines)} samples ({data_ratio*100:.4f}%)")
            else:
                self.lines = all_lines
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
    
    def _parse_instruction_boundaries(self, tokens):
        """
        Parse instruction boundaries from tokens.
        
        Assumes tokens are formatted as:
        [CLS] instr0_token1 instr0_token2 ... instr1_token1 ... [SEP]
        
        Instructions are separated by instruction address markers or control flow tokens.
        For simplicity, we'll use a heuristic: new instruction starts with:
        - Assembly mnemonics (mov, add, push, pop, jmp, call, ret, etc.)
        - After instr_addr_* tokens
        
        Returns:
            instruction_ids: List mapping each token position to instruction index
            jump_info: List of (token_idx, target_instruction_idx) for instr_addr tokens
        """
        instruction_ids = []
        jump_info = []
        current_instr_idx = 0
        
        # Assembly instruction mnemonics (common ones)
        mnemonics = {
            'mov', 'add', 'sub', 'push', 'pop', 'lea', 'xor', 'or', 'and', 'not',
            'jmp', 'je', 'jne', 'jz', 'jnz', 'ja', 'jb', 'jg', 'jl', 'call', 'ret',
            'test', 'cmp', 'inc', 'dec', 'mul', 'div', 'shl', 'shr', 'sal', 'sar',
            'leave', 'enter', 'nop', 'int', 'syscall', 'sysenter'
        }
        
        instr_addr_pattern = re.compile(r'instr_addr_(\d+)')
        
        for i, token in enumerate(tokens):
            # [CLS] and [SEP] get their own instruction index
            if token == '[CLS]':
                instruction_ids.append(0)
                current_instr_idx = 0
                continue
            elif token == '[SEP]':
                instruction_ids.append(current_instr_idx)
                continue
            
            # Check if this is an instr_addr token
            match = instr_addr_pattern.match(token)
            if match:
                target_instr_idx = int(match.group(1))
                # Clamp to valid range
                if target_instr_idx >= self.max_instructions:
                    target_instr_idx = self.max_instructions - 1
                jump_info.append((i, target_instr_idx))
                instruction_ids.append(current_instr_idx)
                continue
            
            # Check if this starts a new instruction
            if token.lower() in mnemonics:
                current_instr_idx += 1
                # Clamp to max_instructions
                if current_instr_idx >= self.max_instructions:
                    current_instr_idx = self.max_instructions - 1
            
            instruction_ids.append(current_instr_idx)
        
        return instruction_ids, jump_info
    
    def _apply_mlm_mask(self, input_ids, labels, jump_indices):
        """
        Apply MLM masking to regular tokens (not jump tokens).
        
        Args:
            input_ids: Token IDs
            labels: Labels for MLM loss
            jump_indices: Set of indices to avoid masking (jump tokens)
            
        Returns:
            Masked input_ids and labels
        """
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
    
    def _apply_jtp_mask(self, input_ids, jtp_labels, jump_info):
        """
        Apply JTP masking to instr_addr tokens.
        
        Args:
            input_ids: Token IDs
            jtp_labels: Labels for JTP loss (target instruction indices)
            jump_info: List of (token_idx, target_instruction_idx)
            
        Returns:
            Masked input_ids and jtp_labels
        """
        for token_idx, target_instr_idx in jump_info:
            if random.random() < self.jtp_probability:
                # Set label to target instruction index
                jtp_labels[token_idx] = target_instr_idx
                
                # Mask the instr_addr token
                # 80% replace with [MASK]
                prob = random.random()
                if prob < 0.8:
                    input_ids[token_idx] = self.mask_token_id
                # 10% replace with random token
                elif prob < 0.9:
                    input_ids[token_idx] = random.randint(0, len(self.tokenizer) - 1)
                # 10% keep original
        
        return input_ids, jtp_labels
    
    def __getitem__(self, idx):
        """
        Get a training example.
        
        Returns:
            Dictionary with:
                - input_ids: Masked token IDs
                - attention_mask: Attention mask
                - instruction_ids: Instruction index for each token
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
        
        # Parse instruction boundaries and extract jump info
        instruction_ids, jump_info = self._parse_instruction_boundaries(tokens)
        
        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Create labels (initialized to -100 for ignore)
        mlm_labels = [-100] * len(input_ids)
        jtp_labels = [-100] * len(input_ids)
        
        # Get jump token indices
        jump_indices = set([idx for idx, _ in jump_info])
        
        # Apply MLM masking (excludes jump tokens)
        input_ids, mlm_labels = self._apply_mlm_mask(input_ids, mlm_labels, jump_indices)
        
        # Apply JTP masking (only instr_addr tokens)
        input_ids, jtp_labels = self._apply_jtp_mask(input_ids, jtp_labels, jump_info)
        
        # Padding
        seq_len = len(input_ids)
        attention_mask = [1] * seq_len
        
        # Pad to max_len
        padding_len = self.max_len - seq_len
        if padding_len > 0:
            input_ids += [self.pad_token_id] * padding_len
            attention_mask += [0] * padding_len
            instruction_ids += [0] * padding_len  # Pad instruction_ids with 0
            mlm_labels += [-100] * padding_len
            jtp_labels += [-100] * padding_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'instruction_ids': torch.tensor(instruction_ids, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
            'jtp_labels': torch.tensor(jtp_labels, dtype=torch.long)
        }


def create_instr_dataloaders(
    train_path,
    test_path,
    tokenizer,
    batch_size=32,
    max_len=512,
    max_instructions=201,
    mlm_probability=0.15,
    jtp_probability=0.20,
    num_workers=4,
    data_ratio=1.0
):
    """
    Create train and test dataloaders for instruction-level pretraining.
    
    Args:
        train_path: Path to training data (instr_pretrain.txt)
        test_path: Path to test data (instr_test.txt), or None to skip validation
        tokenizer: jTrans tokenizer
        batch_size: Batch size
        max_len: Maximum sequence length (token-level)
        max_instructions: Maximum number of instructions
        mlm_probability: MLM masking probability
        jtp_probability: JTP masking probability
        num_workers: Number of data loading workers
        data_ratio: Fraction of data to use (0.0001 = 0.01%, 1.0 = 100%)
        
    Returns:
        train_loader, test_loader (test_loader is None if test_path is None)
    """
    train_dataset = InstrPretrainingDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_len=max_len,
        max_instructions=max_instructions,
        mlm_probability=mlm_probability,
        jtp_probability=jtp_probability,
        on_memory=True,
        data_ratio=data_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create test loader only if test_path is provided
    test_loader = None
    if test_path is not None:
        test_dataset = InstrPretrainingDataset(
            data_path=test_path,
            tokenizer=tokenizer,
            max_len=max_len,
            max_instructions=max_instructions,
            mlm_probability=mlm_probability,
            jtp_probability=jtp_probability,
            on_memory=True,
            data_ratio=data_ratio  # Apply same ratio to test set
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, test_loader

    
    return train_loader, test_loader
