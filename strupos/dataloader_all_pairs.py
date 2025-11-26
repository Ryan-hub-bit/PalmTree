"""
All Consecutive Pairs DataLoader for Address-Aware BERT

Creates ALL consecutive pairs from each CFG line for NSP training.

For a line with 8 instructions:
- MLM: Uses all 8 instructions (one sample per line)
- NSP-CFG: Creates ALL 7 pairs: (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8)
- NSP-DFG: Uses DFG as-is (already in pair format)

This means each CFG line produces:
- 1 MLM sample
- 7 NSP samples

Total samples = len(cfg_lines) * 7 for NSP
"""

import torch
from torch.utils.data import Dataset
import re
import random
from tqdm import tqdm


class AllConsecutivePairsDataset(Dataset):
    """
    Dataset that creates ALL consecutive instruction pairs for NSP.
    
    Strategy:
    - MLM: Iterate through CFG lines (one MLM sample per line)
    - NSP-CFG: Create ALL consecutive pairs from each line (7 pairs per 8-inst line)
    - NSP-DFG: Use DFG lines as-is (already in pair format)
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
        self.vocab = vocab
        self.seq_len = seq_len
        self.nsp_prob = nsp_prob
        self.mask_prob = mask_prob
        
        # Regex patterns for parsing inline format
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        
        # Load CFG data
        print(f"Loading CFG corpus from {cfg_corpus_path}")
        self.cfg_lines = self._load_corpus(cfg_corpus_path)
        print(f"Loaded {len(self.cfg_lines)} CFG lines")
        
        # Load DFG data
        print(f"Loading DFG corpus from {dfg_corpus_path}")
        self.dfg_lines = self._load_corpus(dfg_corpus_path)
        print(f"Loaded {len(self.dfg_lines)} DFG lines")
        
        # Apply data percentage
        if data_percentage < 1.0:
            cfg_size = int(len(self.cfg_lines) * data_percentage)
            dfg_size = int(len(self.dfg_lines) * data_percentage)
            self.cfg_lines = self.cfg_lines[:cfg_size]
            self.dfg_lines = self.dfg_lines[:dfg_size]
        
        # Apply train/val split
        if train_split < 1.0:
            cfg_train_size = int(len(self.cfg_lines) * train_split)
            dfg_train_size = int(len(self.dfg_lines) * train_split)
            
            if is_train:
                self.cfg_lines = self.cfg_lines[:cfg_train_size]
                self.dfg_lines = self.dfg_lines[:dfg_train_size]
            else:
                self.cfg_lines = self.cfg_lines[cfg_train_size:]
                self.dfg_lines = self.dfg_lines[dfg_train_size:]
        
        # Create ALL consecutive NSP pairs from CFG
        print("Creating ALL consecutive NSP pairs from CFG...")
        self.cfg_nsp_pairs = []
        for line_idx, line in enumerate(tqdm(self.cfg_lines, desc="CFG NSP")):
            pairs = self._create_all_consecutive_pairs(line, line_idx)
            self.cfg_nsp_pairs.extend(pairs)
        
        print(f"Dataset size:")
        print(f"  CFG lines (for MLM): {len(self.cfg_lines)}")
        print(f"  CFG NSP pairs: {len(self.cfg_nsp_pairs)}")
        print(f"  DFG lines (NSP only): {len(self.dfg_lines)}")
    
    def _load_corpus(self, path):
        """Load corpus file"""
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines
    
    def _create_all_consecutive_pairs(self, line, line_idx):
        """
        Create ALL consecutive pairs from a line.
        
        Args:
            line: Tab-separated instructions
            line_idx: Index of this line in cfg_lines
        
        Returns:
            List of (line_idx, inst1, inst2, label) tuples
        """
        instructions = line.split('\t')
        pairs = []
        
        if len(instructions) < 2:
            return pairs
        
        # Create ALL consecutive pairs: (0,1), (1,2), (2,3), ..., (n-2, n-1)
        for i in range(len(instructions) - 1):
            inst1 = instructions[i]
            inst2_original = instructions[i + 1]
            
            # Decide if positive or negative
            if random.random() < self.nsp_prob:
                # NEGATIVE: Replace inst2 with random instruction (ensure it's different)
                inst2 = inst2_original
                max_attempts = 10
                attempts = 0
                while inst2 == inst2_original and attempts < max_attempts:
                    random_line = random.choice(self.cfg_lines)
                    random_instructions = random_line.split('\t')
                    inst2 = random.choice(random_instructions)
                    attempts += 1
                label = 0
            else:
                # POSITIVE: Keep consecutive
                inst2 = inst2_original
                label = 1
            
            pairs.append((line_idx, inst1, inst2, label))
        
        return pairs
    
    def _parse_instruction(self, inst_text):
        """Parse a single instruction and extract tokens + positions"""
        tokens = []
        positions = []
        
        # Match opcode with address
        match = self.addr_pattern.match(inst_text)
        if not match:
            return [inst_text], [(0.0, 0.0, 0.0)]
        
        opcode = match.group(1)
        binary_pos = float(match.group(3))
        function_pos = float(match.group(4))
        bb_pos = float(match.group(5))
        
        tokens.append(opcode)
        positions.append((binary_pos, function_pos, bb_pos))
        
        # Parse operands
        operands_text = inst_text[match.end():].strip()
        if operands_text:
            for operand in operands_text.split():
                nested_match = self.nested_addr_pattern.match(operand)
                if nested_match:
                    tokens.append('address')
                else:
                    tokens.append(operand)
                positions.append((binary_pos, function_pos, bb_pos))
        
        return tokens, positions
    
    def _parse_line(self, line):
        """Parse a full line into tokens and positions"""
        instructions = line.split('\t')
        all_tokens = []
        all_positions = []
        
        for inst in instructions:
            inst = inst.strip()
            if inst:
                tokens, positions = self._parse_instruction(inst)
                all_tokens.extend(tokens)
                all_positions.extend(positions)
        
        return all_tokens, all_positions
    
    def _parse_line_with_separators(self, line):
        """
        Parse a full line keeping instructions separate with <eos> between them.
        Used for MLM task to maintain instruction boundaries.
        
        Format: <sos> inst1_tokens <eos> inst2_tokens <eos> ... instN_tokens <eos>
        
        Returns:
            tokens: List of tokens with <sos> at start and <eos> between instructions
            positions: Corresponding position tuples (0,0,0) for special tokens
        """
        instructions = line.split('\t')
        all_tokens = ['<sos>']
        all_positions = [(0.0, 0.0, 0.0)]
        
        for inst in instructions:
            inst = inst.strip()
            if inst:
                tokens, positions = self._parse_instruction(inst)
                all_tokens.extend(tokens)
                all_positions.extend(positions)
                # Add <eos> after each instruction
                all_tokens.append('<eos>')
                all_positions.append((0.0, 0.0, 0.0))
        
        return all_tokens, all_positions
    
    def __len__(self):
        """Length = number of NSP pairs (so we train on ALL pairs)"""
        return len(self.cfg_nsp_pairs)
    
    def __getitem__(self, index):
        """
        Get one NSP training sample.
        
        Returns paired CFG and DFG NSP data.
        For CFG MLM, you need to iterate cfg_lines separately.
        """
        # Get CFG NSP pair
        line_idx, cfg_inst1, cfg_inst2, cfg_is_next = self.cfg_nsp_pairs[index]
        
        # Parse the two instructions for CFG NSP
        inst1_tokens, inst1_pos = self._parse_instruction(cfg_inst1)
        inst2_tokens, inst2_pos = self._parse_instruction(cfg_inst2)
        
        # For CFG MLM: Use the full line with instruction separators
        cfg_mlm_line = self.cfg_lines[line_idx]
        cfg_mlm_tokens, cfg_mlm_positions = self._parse_line_with_separators(cfg_mlm_line)
        
        # Get DFG NSP pair (cycle through DFG lines)
        dfg_idx = index % len(self.dfg_lines)
        dfg_line = self.dfg_lines[dfg_idx]
        
        # Parse DFG line as tab-separated instructions (same as CFG)
        dfg_instructions = dfg_line.split('\t')
        
        if len(dfg_instructions) >= 2:
            # Get consecutive pair based on position within line
            # Use modulo to cycle through all pairs in this line
            pair_idx = (index // len(self.dfg_lines)) % max(1, len(dfg_instructions) - 1)
            
            dfg_inst1 = dfg_instructions[pair_idx]
            dfg_inst2_original = dfg_instructions[pair_idx + 1]
            
            # Parse the two consecutive instructions
            dfg_inst1_tokens, dfg_inst1_pos = self._parse_instruction(dfg_inst1)
            dfg_inst2_tokens, dfg_inst2_pos = self._parse_instruction(dfg_inst2_original)
            
            # DFG NSP label
            if random.random() < self.nsp_prob:
                # NEGATIVE: Replace second instruction with random (ensure it's different)
                dfg_inst2 = dfg_inst2_original
                max_attempts = 10
                attempts = 0
                while dfg_inst2 == dfg_inst2_original and attempts < max_attempts:
                    random_dfg = random.choice(self.dfg_lines)
                    random_instructions = random_dfg.split('\t')
                    dfg_inst2 = random.choice(random_instructions)
                    attempts += 1
                dfg_inst2_tokens, dfg_inst2_pos = self._parse_instruction(dfg_inst2)
                dfg_is_next = 0
            else:
                # POSITIVE: Keep consecutive
                dfg_is_next = 1
        else:
            # Fallback for lines with < 2 instructions
            dfg_inst1_tokens, dfg_inst1_pos = self._parse_line(dfg_line)
            dfg_inst2_tokens, dfg_inst2_pos = [], []
            dfg_is_next = 1
        
        # Now convert to BERT format and return...
        # (Rest of the processing - tokenization, masking, padding, etc.)
        # This is the same as in dataloader_paired.py
        
        return self._create_bert_input(
            cfg_mlm_tokens, cfg_mlm_positions,
            inst1_tokens, inst1_pos, inst2_tokens, inst2_pos, cfg_is_next,
            dfg_inst1_tokens, dfg_inst1_pos, dfg_inst2_tokens, dfg_inst2_pos, dfg_is_next
        )
    
    def _create_bert_input(self, cfg_mlm_tokens, cfg_mlm_positions,
                          cfg_nsp1_tokens, cfg_nsp1_pos, cfg_nsp2_tokens, cfg_nsp2_pos, cfg_is_next,
                          dfg_nsp1_tokens, dfg_nsp1_pos, dfg_nsp2_tokens, dfg_nsp2_pos, dfg_is_next):
        """Create BERT input format with MLM and NSP tasks"""
        
        # === Process CFG for MLM (uses full line with instruction separators) ===
        # cfg_mlm_tokens already has <sos> at start and <eos> after each instruction
        cfg_mlm_combined = cfg_mlm_tokens
        cfg_mlm_pos_combined = cfg_mlm_positions
        
        # Truncate if needed
        if len(cfg_mlm_combined) > self.seq_len:
            cfg_mlm_combined = cfg_mlm_combined[:self.seq_len]
            cfg_mlm_pos_combined = cfg_mlm_pos_combined[:self.seq_len]
        
        # Apply MLM masking
        cfg_mlm_input, cfg_mlm_label = self._mask_tokens(cfg_mlm_combined)
        
        # Pad CFG MLM
        mlm_padding = self.seq_len - len(cfg_mlm_input)
        cfg_mlm_input = self._pad_sequence(cfg_mlm_input, self.seq_len, self.vocab.pad_index)
        cfg_mlm_label = self._pad_sequence(cfg_mlm_label, self.seq_len, -1)
        cfg_mlm_pos_combined = cfg_mlm_pos_combined + [(0.0, 0.0, 0.0)] * mlm_padding
        
        cfg_mlm_binary_pos = [pos[0] for pos in cfg_mlm_pos_combined]
        cfg_mlm_function_pos = [pos[1] for pos in cfg_mlm_pos_combined]
        cfg_mlm_bb_pos = [pos[2] for pos in cfg_mlm_pos_combined]
        
        # === Process CFG for NSP (uses consecutive pair) ===
        cfg_nsp_combined = ['<sos>'] + cfg_nsp1_tokens + ['<eos>'] + cfg_nsp2_tokens + ['<eos>']
        cfg_nsp_pos_combined = [(0.0, 0.0, 0.0)] + cfg_nsp1_pos + [(0.0, 0.0, 0.0)] + cfg_nsp2_pos + [(0.0, 0.0, 0.0)]
        cfg_segment_labels = [0] * (len(cfg_nsp1_tokens) + 2) + [1] * (len(cfg_nsp2_tokens) + 1)
        
        # Truncate if needed
        if len(cfg_nsp_combined) > self.seq_len:
            cfg_nsp_combined = cfg_nsp_combined[:self.seq_len]
            cfg_nsp_pos_combined = cfg_nsp_pos_combined[:self.seq_len]
            cfg_segment_labels = cfg_segment_labels[:self.seq_len]
        
        # Convert to indices (no masking for NSP)
        cfg_nsp_input = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in cfg_nsp_combined]
        
        # Pad CFG NSP
        nsp_padding = self.seq_len - len(cfg_nsp_input)
        cfg_nsp_input = self._pad_sequence(cfg_nsp_input, self.seq_len, self.vocab.pad_index)
        cfg_segment_labels = self._pad_sequence(cfg_segment_labels, self.seq_len, 0)
        cfg_nsp_pos_combined = cfg_nsp_pos_combined + [(0.0, 0.0, 0.0)] * nsp_padding
        
        cfg_nsp_binary_pos = [pos[0] for pos in cfg_nsp_pos_combined]
        cfg_nsp_function_pos = [pos[1] for pos in cfg_nsp_pos_combined]
        cfg_nsp_bb_pos = [pos[2] for pos in cfg_nsp_pos_combined]
        
        # === Process DFG for NSP ===
        dfg_nsp_combined = ['<sos>'] + dfg_nsp1_tokens + ['<eos>'] + dfg_nsp2_tokens + ['<eos>']
        dfg_nsp_pos_combined = [(0.0, 0.0, 0.0)] + dfg_nsp1_pos + [(0.0, 0.0, 0.0)] + dfg_nsp2_pos + [(0.0, 0.0, 0.0)]
        dfg_segment_labels = [0] * (len(dfg_nsp1_tokens) + 2) + [1] * (len(dfg_nsp2_tokens) + 1)
        
        # Truncate if needed
        if len(dfg_nsp_combined) > self.seq_len:
            dfg_nsp_combined = dfg_nsp_combined[:self.seq_len]
            dfg_nsp_pos_combined = dfg_nsp_pos_combined[:self.seq_len]
            dfg_segment_labels = dfg_segment_labels[:self.seq_len]
        
        # Convert to indices (no masking for DFG)
        dfg_nsp_input = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in dfg_nsp_combined]
        
        # Pad DFG NSP
        dfg_padding = self.seq_len - len(dfg_nsp_input)
        dfg_nsp_input = self._pad_sequence(dfg_nsp_input, self.seq_len, self.vocab.pad_index)
        dfg_segment_labels = self._pad_sequence(dfg_segment_labels, self.seq_len, 0)
        dfg_nsp_pos_combined = dfg_nsp_pos_combined + [(0.0, 0.0, 0.0)] * dfg_padding
        
        dfg_nsp_binary_pos = [pos[0] for pos in dfg_nsp_pos_combined]
        dfg_nsp_function_pos = [pos[1] for pos in dfg_nsp_pos_combined]
        dfg_nsp_bb_pos = [pos[2] for pos in dfg_nsp_pos_combined]
        
        # Return all data
        return {
            # CFG MLM task
            'cfg_mlm_input': torch.tensor(cfg_mlm_input, dtype=torch.long),
            'cfg_mlm_label': torch.tensor(cfg_mlm_label, dtype=torch.long),
            'cfg_mlm_binary_pos': torch.tensor(cfg_mlm_binary_pos, dtype=torch.float),
            'cfg_mlm_function_pos': torch.tensor(cfg_mlm_function_pos, dtype=torch.float),
            'cfg_mlm_bb_pos': torch.tensor(cfg_mlm_bb_pos, dtype=torch.float),
            
            # CFG NSP task
            'cfg_nsp_input': torch.tensor(cfg_nsp_input, dtype=torch.long),
            'cfg_segment_label': torch.tensor(cfg_segment_labels, dtype=torch.long),
            'cfg_is_next': torch.tensor(cfg_is_next, dtype=torch.long),
            'cfg_nsp_binary_pos': torch.tensor(cfg_nsp_binary_pos, dtype=torch.float),
            'cfg_nsp_function_pos': torch.tensor(cfg_nsp_function_pos, dtype=torch.float),
            'cfg_nsp_bb_pos': torch.tensor(cfg_nsp_bb_pos, dtype=torch.float),
            
            # DFG NSP task
            'dfg_nsp_input': torch.tensor(dfg_nsp_input, dtype=torch.long),
            'dfg_segment_label': torch.tensor(dfg_segment_labels, dtype=torch.long),
            'dfg_is_next': torch.tensor(dfg_is_next, dtype=torch.long),
            'dfg_nsp_binary_pos': torch.tensor(dfg_nsp_binary_pos, dtype=torch.float),
            'dfg_nsp_function_pos': torch.tensor(dfg_nsp_function_pos, dtype=torch.float),
            'dfg_nsp_bb_pos': torch.tensor(dfg_nsp_bb_pos, dtype=torch.float),
        }
    
    def _mask_tokens(self, tokens):
        """Apply MLM masking to tokens, but never mask special tokens"""
        output_tokens = []
        output_labels = []
        
        # Special tokens that should never be masked
        special_tokens = {'<sos>', '<eos>', '[PAD]', '[CLS]', '[SEP]'}
        
        for token in tokens:
            # Skip masking for special tokens
            if token in special_tokens:
                output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                output_labels.append(-1)  # No label for special tokens
                continue
            
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
                    # 10%: Keep original
                    output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                
                output_labels.append(self.vocab.stoi.get(token, self.vocab.unk_index))
            else:
                # Not masked
                output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                output_labels.append(-1)
        
        return output_tokens, output_labels
    
    def _pad_sequence(self, sequence, max_len, pad_value):
        """Pad or truncate sequence to max_len"""
        if len(sequence) > max_len:
            return sequence[:max_len]
        else:
            return sequence + [pad_value] * (max_len - len(sequence))
