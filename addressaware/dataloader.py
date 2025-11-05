"""
Address-Aware DataLoader for PalmTree

This dataloader handles the new inline address format:
  opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 operand2 ...

It extracts:
1. Token sequences for vocabulary embedding
2. Address normalization values (binary, function, basic block levels)
3. Segment labels for NSP task
"""

import torch
from torch.utils.data import Dataset
import re
import random


class AddressAwareDataset(Dataset):
    """
    Dataset for address-aware binary code pretraining.
    
    Reads from combined CFG/DFG inline format files where each instruction has:
    - opcode(0xADDR:bnorm:fnorm:bbnorm) operands
    
    Extracts three levels of embeddings:
    1. Token embeddings (standard vocabulary)
    2. Address position embeddings (sin/cos on bnorm, fnorm, bbnorm)
    3. Segment embeddings (for NSP task)
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
            vocab: Vocabulary object with stoi, itos, pad_index, mask_index
            seq_len: Maximum sequence length
            encoding: File encoding
            on_memory: Load all data into memory
            nsp_prob: Probability of negative NSP pair
            mask_prob: Probability of masking a token for MLM
            data_percentage: Percentage of dataset to use (0.0-1.0)
            train_split: Percentage of data for training (rest is validation), e.g., 0.9 = 90% train, 10% val
            is_train: True for training set, False for validation set
        """
        self.vocab = vocab
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.encoding = encoding
        self.nsp_prob = nsp_prob
        self.mask_prob = mask_prob
        self.data_percentage = max(0.0, min(1.0, data_percentage))  # Clamp to [0, 1]
        self.train_split = max(0.0, min(1.0, train_split))  # Clamp to [0, 1]
        self.is_train = is_train
        
        # Regex patterns for parsing inline format
        # Match: opcode(0xADDR:bnorm:fnorm:bbnorm)
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        # Match nested addresses: addr_code(...) or addr_data(...)
        self.nested_addr_pattern = re.compile(r'(addr_code|addr_data)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        
        # Load data
        print(f"Loading CFG corpus from {cfg_corpus_path}")
        self.cfg_lines = self._load_corpus(cfg_corpus_path)
        print(f"Loaded {len(self.cfg_lines)} CFG lines")
        
        print(f"Loading DFG corpus from {dfg_corpus_path}")
        self.dfg_lines = self._load_corpus(dfg_corpus_path)
        print(f"Loaded {len(self.dfg_lines)} DFG lines")
        
        # Apply data percentage filter if needed
        if self.data_percentage < 1.0:
            cfg_size = int(len(self.cfg_lines) * self.data_percentage)
            dfg_size = int(len(self.dfg_lines) * self.data_percentage)
            self.cfg_lines = self.cfg_lines[:cfg_size]
            self.dfg_lines = self.dfg_lines[:dfg_size]
            print(f"Using {self.data_percentage*100:.1f}% of data: {len(self.cfg_lines)} CFG lines, {len(self.dfg_lines)} DFG lines")
        
        # Apply train/validation split if needed
        if self.train_split < 1.0:
            cfg_train_size = int(len(self.cfg_lines) * self.train_split)
            dfg_train_size = int(len(self.dfg_lines) * self.train_split)
            
            if self.is_train:
                # Use first train_split% for training
                self.cfg_lines = self.cfg_lines[:cfg_train_size]
                self.dfg_lines = self.dfg_lines[:dfg_train_size]
                print(f"Training set: {len(self.cfg_lines)} CFG lines, {len(self.dfg_lines)} DFG lines ({self.train_split*100:.1f}%)")
            else:
                # Use remaining (1-train_split)% for validation
                self.cfg_lines = self.cfg_lines[cfg_train_size:]
                self.dfg_lines = self.dfg_lines[dfg_train_size:]
                print(f"Validation set: {len(self.cfg_lines)} CFG lines, {len(self.dfg_lines)} DFG lines ({(1-self.train_split)*100:.1f}%)")

        
        # Use the larger corpus as the main driver
        if len(self.cfg_lines) >= len(self.dfg_lines):
            self.main_corpus = self.cfg_lines
            self.aux_corpus = self.dfg_lines
            self.corpus_type = "cfg"
        else:
            self.main_corpus = self.dfg_lines
            self.aux_corpus = self.cfg_lines
            self.corpus_type = "dfg"
        
        print(f"Dataset size: {len(self.main_corpus)} (driven by {self.corpus_type.upper()})")
    
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
        Parse a single instruction string.
        
        Returns:
            tokens: List of token strings
            addr_positions: List of (bnorm, fnorm, bbnorm) tuples, same length as tokens
        """
        tokens = []
        addr_positions = []
        
        # Split by tabs first (multiple instructions per line)
        instructions = instruction_str.split('\t')
        
        for instr in instructions:
            instr = instr.strip()
            if not instr:
                continue
            
            # Try to match main instruction format: opcode(addr)
            match = self.addr_pattern.match(instr)
            if match:
                opcode = match.group(1)
                addr_hex = match.group(2)
                bnorm = float(match.group(3))
                fnorm = float(match.group(4))
                bbnorm = float(match.group(5))
                
                tokens.append(opcode)
                addr_positions.append((bnorm, fnorm, bbnorm))
                
                # Get the operands part (after the closing parenthesis)
                operands_start = match.end()
                operands_str = instr[operands_start:].strip()
                
                if operands_str:
                    # Parse operands - need to handle nested addresses
                    operand_tokens, operand_positions = self._parse_operands(operands_str)
                    tokens.extend(operand_tokens)
                    # Use positions from parsed operands (will be (0,0,0) for regular operands)
                    addr_positions.extend(operand_positions)
            else:
                # Fallback: just split by spaces if format is unexpected
                parts = instr.split()
                tokens.extend(parts)
                # Use default position (0, 0, 0) for unparseable instructions
                addr_positions.extend([(0.0, 0.0, 0.0)] * len(parts))
        
        return tokens, addr_positions
    
    def _parse_operands(self, operands_str):
        """
        Parse operands, handling nested addresses like addr_code(...).
        
        Returns:
            tokens: List of token strings
            positions: List of (bnorm, fnorm, bbnorm) tuples
        """
        tokens = []
        positions = []
        
        # Split by whitespace first
        parts = operands_str.split()
        
        for part in parts:
            if not part:
                continue
            
            # Check if this part has address info: addr_code(0xADDR:bnorm:fnorm:bbnorm) or addr_data(...)
            match = self.nested_addr_pattern.match(part)
            if match:
                # This is addr_code or addr_data with position info
                token = match.group(1)  # e.g., 'addr_code' or 'addr_data'
                addr_hex = match.group(2)
                bnorm = float(match.group(3))
                fnorm = float(match.group(4))
                bbnorm = float(match.group(5))
                
                tokens.append(token)
                positions.append((bnorm, fnorm, bbnorm))
            else:
                # Regular operand (rax, qword, [, ], etc.) - no position info
                tokens.append(part)
                positions.append((0.0, 0.0, 0.0))
        
        return tokens, positions
    
    def _get_random_line(self, corpus):
        """Get a random line from a corpus."""
        return corpus[random.randint(0, len(corpus) - 1)]
    
    def _get_nsp_pair(self, index):
        """
        Get a pair of sequences for Next Sentence Prediction.
        
        Returns:
            t1_tokens, t1_positions: First sequence tokens and positions
            t2_tokens, t2_positions: Second sequence tokens and positions
            is_next: 1 if consecutive, 0 if random
            corpus_type: 'cfg' or 'dfg' (which corpus this sample is from)
        """
        # Get first sequence from main corpus
        line1 = self.main_corpus[index]
        t1_tokens, t1_positions = self._parse_instruction(line1)
        
        # Decide if we should get the next sentence or a random one
        if random.random() < self.nsp_prob:
            # Get a random sentence (negative sample)
            line2 = self._get_random_line(self.aux_corpus if len(self.aux_corpus) > 0 else self.main_corpus)
            is_next = 0
        else:
            # Get the next sentence (positive sample)
            if index + 1 < len(self.main_corpus):
                line2 = self.main_corpus[index + 1]
            else:
                line2 = self.main_corpus[0]
            is_next = 1
        
        t2_tokens, t2_positions = self._parse_instruction(line2)
        
        return t1_tokens, t1_positions, t2_tokens, t2_positions, is_next, self.corpus_type
    
    def _mask_tokens(self, tokens):
        """
        Apply masking for Masked Language Model (MLM) task.
        
        Returns:
            bert_input: Token indices with masks
            bert_label: Original token indices (or -1 for non-masked)
        """
        output_tokens = []
        output_labels = []
        
        for token in tokens:
            prob = random.random()
            
            # 15% chance of masking
            if prob < self.mask_prob:
                prob = random.random()
                
                # 80% replace with [MASK]
                if prob < 0.8:
                    output_tokens.append(self.vocab.mask_index)
                # 10% replace with random token
                elif prob < 0.9:
                    output_tokens.append(random.randint(0, len(self.vocab) - 1))
                # 10% keep original
                else:
                    output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                
                # Label is the original token
                output_labels.append(self.vocab.stoi.get(token, self.vocab.unk_index))
            else:
                # No masking
                output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                output_labels.append(-1)  # -1 means don't compute loss
        
        return output_tokens, output_labels
    
    def _pad_sequence(self, sequence, max_len, pad_value):
        """Pad sequence to max_len."""
        if len(sequence) > max_len:
            return sequence[:max_len]
        else:
            return sequence + [pad_value] * (max_len - len(sequence))
    
    def __len__(self):
        return len(self.main_corpus)
    
    def __getitem__(self, index):
        """
        Get a training sample.
        
        Following PalmTree's approach:
        - CFG: MLM (15% masking) + NSP (order checking)
        - DFG: NO MLM (no masking) + NSP only (trace membership)
        
        Returns dict with:
            bert_input: Token indices [seq_len]
            bert_label: MLM labels [seq_len] (only for CFG, all -1 for DFG)
            segment_label: Segment labels (0 for first seq, 1 for second) [seq_len]
            is_next: NSP label (0 or 1)
            binary_pos: Binary-level address positions [seq_len]
            function_pos: Function-level address positions [seq_len]
            bb_pos: Basic block-level address positions [seq_len]
        """
        # Get NSP pair
        t1_tokens, t1_pos, t2_tokens, t2_pos, is_next, corpus_type = self._get_nsp_pair(index)
        
        # Combine sequences with [SEP] token
        # [CLS] t1 [SEP] t2 [SEP]
        combined_tokens = ['<sos>'] + t1_tokens + ['<eos>'] + t2_tokens + ['<eos>']
        combined_pos = [(0.0, 0.0, 0.0)] + t1_pos + [(0.0, 0.0, 0.0)] + t2_pos + [(0.0, 0.0, 0.0)]
        
        # Segment labels: 0 for first sequence, 1 for second sequence
        segment_labels = [0] * (len(t1_tokens) + 2) + [1] * (len(t2_tokens) + 1)
        
        # Truncate if too long
        if len(combined_tokens) > self.seq_len:
            combined_tokens = combined_tokens[:self.seq_len]
            combined_pos = combined_pos[:self.seq_len]
            segment_labels = segment_labels[:self.seq_len]
        
        # Apply masking ONLY for CFG (PalmTree approach)
        if corpus_type == 'cfg':
            # CFG: Apply MLM (15% masking)
            bert_input, bert_label = self._mask_tokens(combined_tokens)
        else:
            # DFG: NO masking, only NSP
            bert_input = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in combined_tokens]
            bert_label = [-1] * len(combined_tokens)  # No MLM labels for DFG
        
        # Pad sequences
        padding_len = self.seq_len - len(bert_input)
        bert_input = self._pad_sequence(bert_input, self.seq_len, self.vocab.pad_index)
        bert_label = self._pad_sequence(bert_label, self.seq_len, -1)
        segment_labels = self._pad_sequence(segment_labels, self.seq_len, 0)
        
        # Pad position arrays
        combined_pos = combined_pos + [(0.0, 0.0, 0.0)] * padding_len
        
        # Extract position components
        binary_pos = [pos[0] for pos in combined_pos]
        function_pos = [pos[1] for pos in combined_pos]
        bb_pos = [pos[2] for pos in combined_pos]
        
        # Convert to tensors
        output = {
            "bert_input": torch.tensor(bert_input, dtype=torch.long),
            "bert_label": torch.tensor(bert_label, dtype=torch.long),
            "segment_label": torch.tensor(segment_labels, dtype=torch.long),
            "is_next": torch.tensor(is_next, dtype=torch.long),
            "binary_pos": torch.tensor(binary_pos, dtype=torch.float),
            "function_pos": torch.tensor(function_pos, dtype=torch.float),
            "bb_pos": torch.tensor(bb_pos, dtype=torch.float),
            "corpus_type": corpus_type,  # 'cfg' or 'dfg' - determines which NSP head to use
        }
        
        return output

