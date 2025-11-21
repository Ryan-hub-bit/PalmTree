"""
Paired CFG+DFG DataLoader for Address-Aware PalmTree

Matches PalmTree's approach:
- Each sample contains BOTH CFG and DFG sequences
- Three tasks per sample: MLM(CFG) + NSP(CFG) + NSP(DFG)
- Fallback to CFG-only when DFG runs out
"""

import torch
from torch.utils.data import Dataset
import re
import random


class PairedAddressAwareDataset(Dataset):
    """
    Paired CFG+DFG dataset for address-aware pretraining.
    
    Following PalmTree's approach:
    - Returns both CFG and DFG in each sample
    - CFG: MLM (15% masking) + NSP (order checking)
    - DFG: NO MLM + NSP only (trace membership)
    - Fallback to CFG-only if DFG exhausted
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
        # Pattern for address operands (now using 'address' instead of 'addr_code'/'addr_data')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        
        # Load data
        print(f"Loading CFG corpus from {cfg_corpus_path}")
        self.cfg_lines = self._load_corpus(cfg_corpus_path)
        print(f"Loaded {len(self.cfg_lines)} CFG lines")
        
        print(f"Loading DFG corpus from {dfg_corpus_path}")
        self.dfg_lines = self._load_corpus(dfg_corpus_path)
        print(f"Loaded {len(self.dfg_lines)} DFG lines")
        
        # Apply data percentage filter
        if self.data_percentage < 1.0:
            cfg_size = int(len(self.cfg_lines) * self.data_percentage)
            dfg_size = int(len(self.dfg_lines) * self.data_percentage)
            self.cfg_lines = self.cfg_lines[:cfg_size]
            self.dfg_lines = self.dfg_lines[:dfg_size]
            print(f"Using {self.data_percentage*100:.1f}% of data: {len(self.cfg_lines)} CFG, {len(self.dfg_lines)} DFG")
        
        # Apply train/validation split
        if self.train_split < 1.0:
            cfg_train_size = int(len(self.cfg_lines) * self.train_split)
            dfg_train_size = int(len(self.dfg_lines) * self.train_split)
            
            if self.is_train:
                self.cfg_lines = self.cfg_lines[:cfg_train_size]
                self.dfg_lines = self.dfg_lines[:dfg_train_size]
                print(f"Training set: {len(self.cfg_lines)} CFG, {len(self.dfg_lines)} DFG ({self.train_split*100:.1f}%)")
            else:
                self.cfg_lines = self.cfg_lines[cfg_train_size:]
                self.dfg_lines = self.dfg_lines[dfg_train_size:]
                print(f"Validation set: {len(self.cfg_lines)} CFG, {len(self.dfg_lines)} DFG ({(1-self.train_split)*100:.1f}%)")
        
        # Dataset length driven by CFG (larger corpus)
        self.n_cfg = len(self.cfg_lines)
        self.n_dfg = len(self.dfg_lines)
        print(f"Dataset size: {self.n_cfg} samples (CFG-driven, with DFG cycling)")
    
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
        """Parse instruction and extract tokens with address positions."""
        tokens = []
        addr_positions = []
        
        parts = instruction_str.split()
        for part in parts:
            # Check for opcode with inline address: opcode(0xADDR:bnorm:fnorm:bbnorm)
            addr_match = self.addr_pattern.match(part)
            if addr_match:
                opcode = addr_match.group(1)
                bnorm = float(addr_match.group(3))
                fnorm = float(addr_match.group(4))
                bbnorm = float(addr_match.group(5))
                tokens.append(opcode)
                addr_positions.append((bnorm, fnorm, bbnorm))
                continue
            
            # Check for nested address: address(0xADDR:bnorm:fnorm:bbnorm)
            nested_match = self.nested_addr_pattern.match(part)
            if nested_match:
                bnorm = float(nested_match.group(2))
                fnorm = float(nested_match.group(3))
                bbnorm = float(nested_match.group(4))
                tokens.append('address')  # Use PalmTree's 'address' token
                addr_positions.append((bnorm, fnorm, bbnorm))
                continue
            
            # Regular token (no address)
            tokens.append(part)
            addr_positions.append((0.0, 0.0, 0.0))
        
        return tokens, addr_positions
    
    def _get_random_line(self, corpus):
        """Get a random line from corpus."""
        return corpus[random.randint(0, len(corpus) - 1)]
    
    def _split_line_into_instructions(self, tokens, positions):
        """
        Split a line into two instruction sequences.
        
        For 2-instruction format: each line has 2 instructions
        For 8-instruction format: each line has 8 instructions
        
        Strategy: Split at midpoint to get two equal halves
        Returns: (inst1_tokens, inst1_pos, inst2_tokens, inst2_pos)
        """
        mid_point = len(tokens) // 2
        return tokens[:mid_point], positions[:mid_point], tokens[mid_point:], positions[mid_point:]
    
    def _get_nsp_pair(self, index, corpus):
        """
        Get NSP pair using WITHIN-LINE vs CROSS-LINE strategy for CFG.
        
        CFG Semantics: Control Flow CONTINUITY
        
        POSITIVE (is_next=1): 
          - Sentence A = First half of Line N (C1)
          - Sentence B = Second half of SAME Line N (C2)
          - Meaning: "C2 continues the control flow from C1" (same CFG path)
          
        NEGATIVE (is_next=0):
          - Sentence A = First half of Line N (C1)
          - Sentence B = First half of RANDOM Line (C_random)
          - Meaning: "C_random does NOT continue from C1" (different CFG path)
          
        This teaches: "Does sentence B continue the control flow from sentence A?"
        """
        line = corpus[index % len(corpus)]
        all_tokens, all_positions = self._parse_instruction(line)
        
        # Split line into two halves
        first_half_tokens, first_half_pos, second_half_tokens, second_half_pos = \
            self._split_line_into_instructions(all_tokens, all_positions)
        
        if random.random() < self.nsp_prob:
            # NEGATIVE: pair C1 with random instruction (broken control flow)
            random_line = self._get_random_line(corpus)
            random_tokens, random_positions = self._parse_instruction(random_line)
            # Use first half of random line as the disconnected instruction
            random_first_half, random_first_pos, _, _ = \
                self._split_line_into_instructions(random_tokens, random_positions)
            
            t1_tokens = first_half_tokens
            t1_positions = first_half_pos
            t2_tokens = random_first_half
            t2_positions = random_first_pos
            is_next = 0
        else:
            # POSITIVE: C1 → C2 (correct control flow continuity within same line)
            t1_tokens = first_half_tokens
            t1_positions = first_half_pos
            t2_tokens = second_half_tokens
            t2_positions = second_half_pos
            is_next = 1
        
        return t1_tokens, t1_positions, t2_tokens, t2_positions, is_next
    
    def _get_nsp_pair_dfg(self, index, corpus):
        """
        Get NSP pair for DFG using WITHIN-LINE vs CROSS-LINE strategy.
        
        DFG Semantics: Data Dependency CORRECTNESS
        
        POSITIVE (is_next=1): 
          - Sentence A = First half of Line N (D1)
          - Sentence B = Second half of SAME Line N (D2)
          - Meaning: "D2 has the CORRECT data dependency on D1"
          
        NEGATIVE (is_next=0):
          - Sentence A = First half of Line N (D1)
          - Sentence B = First half of RANDOM Line (D_random, NOT D2)
          - Meaning: "D_random is NOT the correct dependency for D1"
          
        This teaches: "Is this the correct data flow dependency?"
        """
        line = corpus[index % len(corpus)]
        all_tokens, all_positions = self._parse_instruction(line)
        
        # Split line into two halves
        first_half_tokens, first_half_pos, second_half_tokens, second_half_pos = \
            self._split_line_into_instructions(all_tokens, all_positions)
        
        if random.random() < self.nsp_prob:
            # NEGATIVE: pair D1 with random instruction (NOT D2)
            random_line = self._get_random_line(corpus)
            random_tokens, random_positions = self._parse_instruction(random_line)
            # Use first half of random line as the wrong dependency
            random_first_half, random_first_pos, _, _ = \
                self._split_line_into_instructions(random_tokens, random_positions)
            
            t1_tokens = first_half_tokens
            t1_positions = first_half_pos
            t2_tokens = random_first_half
            t2_positions = random_first_pos
            is_next = 0
        else:
            # POSITIVE: D1 → D2 (correct dependency within same line)
            t1_tokens = first_half_tokens
            t1_positions = first_half_pos
            t2_tokens = second_half_tokens
            t2_positions = second_half_pos
            is_next = 1
        
        return t1_tokens, t1_positions, t2_tokens, t2_positions, is_next
    
    def _mask_tokens(self, tokens):
        """Apply MLM masking (for CFG only).
        
        NOTE: Address positions are KEPT even when tokens are masked.
        Rationale: In binary analysis, address context is always available and provides
        crucial location-dependent information (function boundaries, prologue/epilogue
        patterns, etc.). The model learns: "given this address context, what instruction?"
        """
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
        Get paired CFG+DFG sample.
        
        Returns both CFG and DFG sequences in one sample.
        CFG: Control flow ordering NSP
        DFG: Data dependency NSP
        """
        # Get CFG NSP pair (control flow ordering)
        cfg_t1_tokens, cfg_t1_pos, cfg_t2_tokens, cfg_t2_pos, cfg_is_next = self._get_nsp_pair(index, self.cfg_lines)
        
        # Get DFG NSP pair (data dependency) - cycle if needed
        dfg_index = index % self.n_dfg if self.n_dfg > 0 else 0
        if self.n_dfg > 0:
            dfg_t1_tokens, dfg_t1_pos, dfg_t2_tokens, dfg_t2_pos, dfg_is_next = self._get_nsp_pair_dfg(dfg_index, self.dfg_lines)
        else:
            # No DFG data - use dummy
            dfg_t1_tokens, dfg_t1_pos = [], []
            dfg_t2_tokens, dfg_t2_pos = [], []
            dfg_is_next = 0
        
        # === Process CFG (with MLM) ===
        cfg_combined_tokens = ['<sos>'] + cfg_t1_tokens + ['<eos>'] + cfg_t2_tokens + ['<eos>']
        cfg_combined_pos = [(0.0, 0.0, 0.0)] + cfg_t1_pos + [(0.0, 0.0, 0.0)] + cfg_t2_pos + [(0.0, 0.0, 0.0)]
        cfg_segment_labels = [0] * (len(cfg_t1_tokens) + 2) + [1] * (len(cfg_t2_tokens) + 1)
        
        # Truncate if needed
        if len(cfg_combined_tokens) > self.seq_len:
            cfg_combined_tokens = cfg_combined_tokens[:self.seq_len]
            cfg_combined_pos = cfg_combined_pos[:self.seq_len]
            cfg_segment_labels = cfg_segment_labels[:self.seq_len]
        
        # Apply MLM masking for CFG
        cfg_bert_input, cfg_bert_label = self._mask_tokens(cfg_combined_tokens)
        
        # Pad CFG sequences
        padding_len = self.seq_len - len(cfg_bert_input)
        cfg_bert_input = self._pad_sequence(cfg_bert_input, self.seq_len, self.vocab.pad_index)
        cfg_bert_label = self._pad_sequence(cfg_bert_label, self.seq_len, -1)
        cfg_segment_labels = self._pad_sequence(cfg_segment_labels, self.seq_len, 0)
        cfg_combined_pos = cfg_combined_pos + [(0.0, 0.0, 0.0)] * padding_len
        
        # Extract CFG position components
        cfg_binary_pos = [pos[0] for pos in cfg_combined_pos]
        cfg_function_pos = [pos[1] for pos in cfg_combined_pos]
        cfg_bb_pos = [pos[2] for pos in cfg_combined_pos]
        
        # === Process DFG (NO MLM) ===
        if self.n_dfg > 0:
            dfg_combined_tokens = ['<sos>'] + dfg_t1_tokens + ['<eos>'] + dfg_t2_tokens + ['<eos>']
            dfg_combined_pos = [(0.0, 0.0, 0.0)] + dfg_t1_pos + [(0.0, 0.0, 0.0)] + dfg_t2_pos + [(0.0, 0.0, 0.0)]
            dfg_segment_labels = [0] * (len(dfg_t1_tokens) + 2) + [1] * (len(dfg_t2_tokens) + 1)
            
            # Truncate if needed
            if len(dfg_combined_tokens) > self.seq_len:
                dfg_combined_tokens = dfg_combined_tokens[:self.seq_len]
                dfg_combined_pos = dfg_combined_pos[:self.seq_len]
                dfg_segment_labels = dfg_segment_labels[:self.seq_len]
            
            # NO masking for DFG
            dfg_bert_input = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in dfg_combined_tokens]
            
            # Pad DFG sequences
            padding_len = self.seq_len - len(dfg_bert_input)
            dfg_bert_input = self._pad_sequence(dfg_bert_input, self.seq_len, self.vocab.pad_index)
            dfg_segment_labels = self._pad_sequence(dfg_segment_labels, self.seq_len, 0)
            dfg_combined_pos = dfg_combined_pos + [(0.0, 0.0, 0.0)] * padding_len
            
            # Extract DFG position components
            dfg_binary_pos = [pos[0] for pos in dfg_combined_pos]
            dfg_function_pos = [pos[1] for pos in dfg_combined_pos]
            dfg_bb_pos = [pos[2] for pos in dfg_combined_pos]
        else:
            # No DFG - use dummy padded sequences
            dfg_bert_input = [self.vocab.pad_index] * self.seq_len
            dfg_segment_labels = [0] * self.seq_len
            dfg_binary_pos = [0.0] * self.seq_len
            dfg_function_pos = [0.0] * self.seq_len
            dfg_bb_pos = [0.0] * self.seq_len
        
        # Return paired CFG+DFG sample
        return {
            # CFG data (with MLM)
            'cfg_bert_input': torch.tensor(cfg_bert_input, dtype=torch.long),
            'cfg_bert_label': torch.tensor(cfg_bert_label, dtype=torch.long),
            'cfg_segment_label': torch.tensor(cfg_segment_labels, dtype=torch.long),
            'cfg_is_next': torch.tensor(cfg_is_next, dtype=torch.long),
            'cfg_binary_pos': torch.tensor(cfg_binary_pos, dtype=torch.float),
            'cfg_function_pos': torch.tensor(cfg_function_pos, dtype=torch.float),
            'cfg_bb_pos': torch.tensor(cfg_bb_pos, dtype=torch.float),
            
            # DFG data (NO MLM, only NSP)
            'dfg_bert_input': torch.tensor(dfg_bert_input, dtype=torch.long),
            'dfg_segment_label': torch.tensor(dfg_segment_labels, dtype=torch.long),
            'dfg_is_next': torch.tensor(dfg_is_next, dtype=torch.long),
            'dfg_binary_pos': torch.tensor(dfg_binary_pos, dtype=torch.float),
            'dfg_function_pos': torch.tensor(dfg_function_pos, dtype=torch.float),
            'dfg_bb_pos': torch.tensor(dfg_bb_pos, dtype=torch.float),
        }
