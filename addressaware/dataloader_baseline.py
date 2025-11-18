"""
Baseline Paired CFG+DFG DataLoader for PalmTree (without address embeddings)

This dataloader:
- Loads the SAME data as address-aware version
- Strips address information from inline format
- Uses sequential positions instead of address positions
- Used for fair comparison: same data, same tasks, but no address info
"""

import torch
from torch.utils.data import Dataset
import re
import random


class PairedBaselineDataset(Dataset):
    """
    Paired CFG+DFG dataset for baseline PalmTree (no address embeddings).
    
    Following PalmTree's approach:
    - Returns both CFG and DFG in each sample
    - CFG: MLM (15% masking) + NSP (order checking)
    - DFG: NO MLM + NSP only (trace membership)
    - Strips address information from inline format
    - Uses sequential positions (0, 1, 2, ..., seq_len-1)
    """
    
    def __init__(
        self,
        cfg_corpus_path,
        dfg_corpus_path,
        vocab,
        seq_len=512,
        encoding="utf-8",
        on_memory=True,
        nsp_prob=0.5,
        mask_prob=0.15,
        data_percentage=1.0,
        train_split=1.0,
        is_train=True,
    ):
        """
        Args:
            cfg_corpus_path: Path to CFG inline format file
            dfg_corpus_path: Path to DFG inline format file
            vocab: Vocabulary object
            seq_len: Maximum sequence length
            encoding: File encoding
            on_memory: Load all data into memory
            nsp_prob: Probability of negative NSP pair
            mask_prob: Probability of masking a token for MLM
            data_percentage: Percentage of dataset to use (0.0-1.0)
            train_split: Train/val split ratio
            is_train: True for training set, False for validation set
        """
        self.vocab = vocab
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.encoding = encoding
        self.nsp_prob = nsp_prob
        self.mask_prob = mask_prob
        self.data_percentage = max(0.0, min(1.0, data_percentage))
        self.train_split = max(0.0, min(1.0, train_split))
        self.is_train = is_train
        
        # Regex patterns for parsing inline format
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        # Pattern for address operands
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        
        # Load data
        print(f"[BASELINE] Loading CFG corpus from {cfg_corpus_path}")
        self.cfg_lines = self._load_corpus(cfg_corpus_path)
        print(f"[BASELINE] Loaded {len(self.cfg_lines)} CFG lines")
        
        print(f"[BASELINE] Loading DFG corpus from {dfg_corpus_path}")
        self.dfg_lines = self._load_corpus(dfg_corpus_path)
        print(f"[BASELINE] Loaded {len(self.dfg_lines)} DFG lines")
        
        # Apply data percentage filter
        if self.data_percentage < 1.0:
            cfg_size = int(len(self.cfg_lines) * self.data_percentage)
            dfg_size = int(len(self.dfg_lines) * self.data_percentage)
            self.cfg_lines = self.cfg_lines[:cfg_size]
            self.dfg_lines = self.dfg_lines[:dfg_size]
            print(f"[BASELINE] Using {self.data_percentage*100:.1f}% of data: {len(self.cfg_lines)} CFG, {len(self.dfg_lines)} DFG")
        
        # Apply train/validation split
        if self.train_split < 1.0:
            cfg_train_size = int(len(self.cfg_lines) * self.train_split)
            dfg_train_size = int(len(self.dfg_lines) * self.train_split)
            
            if self.is_train:
                self.cfg_lines = self.cfg_lines[:cfg_train_size]
                self.dfg_lines = self.dfg_lines[:dfg_train_size]
                print(f"[BASELINE] Training set: {len(self.cfg_lines)} CFG, {len(self.dfg_lines)} DFG ({self.train_split*100:.1f}%)")
            else:
                self.cfg_lines = self.cfg_lines[cfg_train_size:]
                self.dfg_lines = self.dfg_lines[dfg_train_size:]
                print(f"[BASELINE] Validation set: {len(self.cfg_lines)} CFG, {len(self.dfg_lines)} DFG ({(1-self.train_split)*100:.1f}%)")
        
        # Dataset length driven by CFG (larger corpus)
        self.n_cfg = len(self.cfg_lines)
        self.n_dfg = len(self.dfg_lines)
        print(f"[BASELINE] Dataset size: {self.n_cfg} samples (CFG-driven, with DFG cycling)")
    
    def _load_corpus(self, path):
        """Load corpus file into memory."""
        lines = []
        with open(path, 'r', encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines
    
    def _parse_instruction(self, instruction_str):
        """
        Parse instruction and STRIP address information.
        Returns only tokens (no address positions).
        """
        tokens = []
        
        parts = instruction_str.split()
        for part in parts:
            # Check for opcode with inline address: opcode(0xADDR:bnorm:fnorm:bbnorm)
            addr_match = self.addr_pattern.match(part)
            if addr_match:
                # Extract opcode only, discard address info
                opcode = addr_match.group(1)
                tokens.append(opcode)
                continue
            
            # Check for nested address: address(0xADDR:bnorm:fnorm:bbnorm)
            nested_match = self.nested_addr_pattern.match(part)
            if nested_match:
                # Keep 'address' token, discard address values
                tokens.append('address')
                continue
            
            # Regular token (no address)
            tokens.append(part)
        
        return tokens
    
    def _get_random_line(self, corpus):
        """Get a random line from corpus."""
        return corpus[random.randint(0, len(corpus) - 1)]
    
    def _get_nsp_pair(self, index, corpus):
        """Get NSP pair from specified corpus."""
        line1 = corpus[index % len(corpus)]
        t1_tokens = self._parse_instruction(line1)
        
        if random.random() < self.nsp_prob:
            # Negative sample: random line
            line2 = self._get_random_line(corpus)
            is_next = 0
        else:
            # Positive sample: next line
            if (index + 1) % len(corpus) < len(corpus):
                line2 = corpus[(index + 1) % len(corpus)]
            else:
                line2 = corpus[0]
            is_next = 1
        
        t2_tokens = self._parse_instruction(line2)
        return t1_tokens, t2_tokens, is_next
    
    def _mask_tokens(self, tokens):
        """Apply MLM masking (for CFG only)."""
        output_tokens = []
        output_labels = []
        
        for token in tokens:
            prob = random.random()
            
            if prob < self.mask_prob:
                prob = random.random()
                
                if prob < 0.8:
                    # 80%: Replace with [MASK]
                    output_tokens.append(self.vocab.mask_index)
                elif prob < 0.9:
                    # 10%: Replace with random token
                    output_tokens.append(random.randint(0, len(self.vocab) - 1))
                else:
                    # 10%: Keep original token
                    output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                
                output_labels.append(self.vocab.stoi.get(token, self.vocab.unk_index))
            else:
                # Not masked
                output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                output_labels.append(-1)
        
        return output_tokens, output_labels
    
    def _pad_sequence(self, sequence, max_len, pad_value):
        """Pad or truncate sequence to max_len."""
        if len(sequence) > max_len:
            return sequence[:max_len]
        else:
            return sequence + [pad_value] * (max_len - len(sequence))
    
    def __len__(self):
        return self.n_cfg
    
    def __getitem__(self, index):
        """
        Get paired CFG+DFG sample WITHOUT address information.
        
        Returns both CFG and DFG sequences in one sample.
        Uses sequential positions (0, 1, 2, ...) instead of address positions.
        """
        # Get CFG NSP pair
        cfg_t1_tokens, cfg_t2_tokens, cfg_is_next = self._get_nsp_pair(index, self.cfg_lines)
        
        # Get DFG NSP pair (cycle if needed)
        dfg_index = index % self.n_dfg if self.n_dfg > 0 else 0
        if self.n_dfg > 0:
            dfg_t1_tokens, dfg_t2_tokens, dfg_is_next = self._get_nsp_pair(dfg_index, self.dfg_lines)
        else:
            # No DFG data - use dummy
            dfg_t1_tokens, dfg_t2_tokens = [], []
            dfg_is_next = 0
        
        # === Process CFG (with MLM) ===
        cfg_combined_tokens = ['<sos>'] + cfg_t1_tokens + ['<eos>'] + cfg_t2_tokens + ['<eos>']
        cfg_segment_labels = [0] * (len(cfg_t1_tokens) + 2) + [1] * (len(cfg_t2_tokens) + 1)
        
        # Truncate if needed
        if len(cfg_combined_tokens) > self.seq_len:
            cfg_combined_tokens = cfg_combined_tokens[:self.seq_len]
            cfg_segment_labels = cfg_segment_labels[:self.seq_len]
        
        # Apply MLM masking for CFG
        cfg_bert_input, cfg_bert_label = self._mask_tokens(cfg_combined_tokens)
        
        # Pad CFG sequences
        cfg_bert_input = self._pad_sequence(cfg_bert_input, self.seq_len, self.vocab.pad_index)
        cfg_bert_label = self._pad_sequence(cfg_bert_label, self.seq_len, -1)
        cfg_segment_labels = self._pad_sequence(cfg_segment_labels, self.seq_len, 0)
        
        # === Process DFG (NO MLM) ===
        if self.n_dfg > 0:
            dfg_combined_tokens = ['<sos>'] + dfg_t1_tokens + ['<eos>'] + dfg_t2_tokens + ['<eos>']
            dfg_segment_labels = [0] * (len(dfg_t1_tokens) + 2) + [1] * (len(dfg_t2_tokens) + 1)
            
            # Truncate if needed
            if len(dfg_combined_tokens) > self.seq_len:
                dfg_combined_tokens = dfg_combined_tokens[:self.seq_len]
                dfg_segment_labels = dfg_segment_labels[:self.seq_len]
            
            # NO masking for DFG
            dfg_bert_input = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in dfg_combined_tokens]
            
            # Pad DFG sequences
            dfg_bert_input = self._pad_sequence(dfg_bert_input, self.seq_len, self.vocab.pad_index)
            dfg_segment_labels = self._pad_sequence(dfg_segment_labels, self.seq_len, 0)
        else:
            # No DFG - use dummy padded sequences
            dfg_bert_input = [self.vocab.pad_index] * self.seq_len
            dfg_segment_labels = [0] * self.seq_len
        
        # Return paired CFG+DFG sample (NO address positions)
        return {
            # CFG data (with MLM)
            'bert_input': torch.tensor(cfg_bert_input, dtype=torch.long),
            'bert_label': torch.tensor(cfg_bert_label, dtype=torch.long),
            'segment_label': torch.tensor(cfg_segment_labels, dtype=torch.long),
            'is_next': torch.tensor(cfg_is_next, dtype=torch.long),
            
            # DFG data (NO MLM, only NSP)
            'dfg_input': torch.tensor(dfg_bert_input, dtype=torch.long),
            'dfg_segment': torch.tensor(dfg_segment_labels, dtype=torch.long),
            'dfg_is_next': torch.tensor(dfg_is_next, dtype=torch.long),
        }
