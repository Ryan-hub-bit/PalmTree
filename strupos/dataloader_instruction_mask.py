"""
Instruction Masking DataLoader for Address-Aware BERT

Implements instruction-level masking where entire instructions are masked at once.
This is different from token-level MLM which masks individual tokens.

For a line with 8 instructions and instruction_mask_rate=0.15:
- ~1 instruction will be fully masked (all its tokens replaced with [MASK])
- The model must predict all tokens of the masked instruction
- Also supports standard token-level MLM

Tasks:
- IMC (Instruction Masking CFG): Mask entire instructions in CFG sequences
- IMD (Instruction Masking DFG): Mask entire instructions in DFG sequences  
- MLM (Masked Language Model): Mask individual tokens  
"""

import torch
from torch.utils.data import Dataset
import re
import random
from tqdm import tqdm


class InstructionMaskingDataset(Dataset):
    """
    Dataset that implements instruction-level masking.
    
    Strategy:
    - IMC: Mask entire instructions in CFG at instruction_mask_rate (e.g., 15%)
    - IMD: Mask entire instructions in DFG at instruction_mask_rate (e.g., 15%)
    - MLM: Mask individual tokens at token_mask_rate (e.g., 15%)
    """
    
    def __init__(
        self,
        cfg_corpus_path,
        dfg_corpus_path,
        vocab,
        seq_len=512,
        encoding="utf-8",
        on_memory=True,
        token_mask_prob=0.15,  # Standard MLM masking rate
        instruction_mask_prob=0.25,  # Instruction-level masking rate
        data_percentage=1.0,
        train_split=1.0,
        is_train=True,
        enable_imd=False,  # Enable DFG instruction masking (controlled by args)
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.token_mask_prob = token_mask_prob
        self.instruction_mask_prob = instruction_mask_prob
        self.enable_imd = enable_imd
        
        # Special token IDs
        self.pad_idx = vocab.stoi.get('<pad>', 0)
        self.unk_idx = vocab.stoi.get('<unk>', 1)
        self.eos_idx = vocab.stoi.get('<eos>', 2)
        self.sos_idx = vocab.stoi.get('<sos>', 3)
        self.mask_idx = vocab.stoi.get('<mask>', 4)
        
        # Regex patterns for parsing inline format
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        
        # Load CFG data
        print(f"Loading CFG corpus from {cfg_corpus_path}")
        self.cfg_lines = self._load_corpus(cfg_corpus_path)
        print(f"Loaded {len(self.cfg_lines)} CFG lines")
        
        # Load DFG data if enabled
        self.dfg_lines = []
        if self.enable_imd and dfg_corpus_path:
            print(f"Loading DFG corpus from {dfg_corpus_path}")
            self.dfg_lines = self._load_corpus(dfg_corpus_path)
            print(f"Loaded {len(self.dfg_lines)} DFG lines")
        
        # Apply data percentage
        if data_percentage < 1.0:
            cfg_size = int(len(self.cfg_lines) * data_percentage)
            self.cfg_lines = self.cfg_lines[:cfg_size]
            if self.dfg_lines:
                dfg_size = int(len(self.dfg_lines) * data_percentage)
                self.dfg_lines = self.dfg_lines[:dfg_size]
        
        # Apply train/val split
        if train_split < 1.0:
            cfg_train_size = int(len(self.cfg_lines) * train_split)
            
            if is_train:
                self.cfg_lines = self.cfg_lines[:cfg_train_size]
                if self.dfg_lines:
                    dfg_train_size = int(len(self.dfg_lines) * train_split)
                    self.dfg_lines = self.dfg_lines[:dfg_train_size]
            else:
                self.cfg_lines = self.cfg_lines[cfg_train_size:]
                if self.dfg_lines:
                    dfg_train_size = int(len(self.dfg_lines) * train_split)
                    self.dfg_lines = self.dfg_lines[dfg_train_size:]
        
        print(f"Dataset size:")
        print(f"  CFG lines (for IMC+MLM): {len(self.cfg_lines)}")
        if self.dfg_lines:
            print(f"  DFG lines (for IMD): {len(self.dfg_lines)}")
        print(f"  Instruction mask rate: {instruction_mask_prob}")
        print(f"  Token mask rate: {token_mask_prob}")
    
    def _load_corpus(self, path):
        """Load corpus file"""
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines
    
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
                    nested_binary_pos = float(nested_match.group(2))
                    nested_function_pos = float(nested_match.group(3))
                    nested_bb_pos = float(nested_match.group(4))
                    tokens.append('address')
                    positions.append((nested_binary_pos, nested_function_pos, nested_bb_pos))
                else:
                    tokens.append(operand)
                    positions.append((0.0, 0.0, 0.0))
        
        return tokens, positions
    
    def _mask_instruction(self, tokens, positions):
        """
        Mask entire instruction (all tokens become [MASK]).
        
        Returns:
            masked_tokens: All tokens replaced with [MASK]
            positions: Original positions preserved
            original_tokens: Ground truth labels
        """
        masked_tokens = [self.mask_idx] * len(tokens)
        original_tokens = [self.vocab.stoi.get(t, self.unk_idx) for t in tokens]
        
        return masked_tokens, positions, original_tokens
    
    def _mask_tokens(self, tokens, positions):
        """
        Apply standard MLM token-level masking.
        
        Returns:
            masked_tokens: Tokens with some randomly masked
            positions: Original positions
            labels: -1 for unmasked, original token ID for masked
        """
        masked_tokens = []
        labels = []
        
        for token, pos in zip(tokens, positions):
            token_id = self.vocab.stoi.get(token, self.unk_idx)
            
            # Randomly mask this token
            if random.random() < self.token_mask_prob:
                # 80% replace with [MASK]
                # 10% replace with random token
                # 10% keep original
                prob = random.random()
                if prob < 0.8:
                    masked_tokens.append(self.mask_idx)
                elif prob < 0.9:
                    masked_tokens.append(random.randint(5, len(self.vocab) - 1))
                else:
                    masked_tokens.append(token_id)
                labels.append(token_id)
            else:
                masked_tokens.append(token_id)
                labels.append(-1)  # Not masked
        
        return masked_tokens, labels
    
    def _process_line_for_instruction_masking(self, line):
        """
        Process a full line and apply instruction-level masking.
        
        Returns:
            bert_input: Token IDs with instruction-level masking
            bert_label: Labels for masked instructions (-1 for unmasked)
            segment_label: Segment IDs
            binary_pos, function_pos, bb_pos: Position embeddings
            instruction_mask: 1 for masked instructions, 0 otherwise
        """
        instructions = line.split('\t')
        
        all_tokens = []
        all_positions = []
        all_labels = []
        instruction_boundaries = []  # Track where each instruction starts/ends
        instruction_mask_labels = []  # 1 if instruction is masked, 0 otherwise
        
        current_pos = 0
        for inst_text in instructions:
            inst_text = inst_text.strip()
            if not inst_text:
                continue
            
            tokens, positions = self._parse_instruction(inst_text)
            
            # Decide if this instruction should be masked
            if random.random() < self.instruction_mask_prob:
                # Mask entire instruction
                masked_tokens, positions, labels = self._mask_instruction(tokens, positions)
                instruction_mask_labels.append(1)
            else:
                # No instruction-level masking (but still convert to IDs)
                masked_tokens = [self.vocab.stoi.get(t, self.unk_idx) for t in tokens]
                labels = [-1] * len(tokens)  # Not masked at instruction level
                instruction_mask_labels.append(0)
            
            instruction_boundaries.append((current_pos, current_pos + len(tokens)))
            current_pos += len(tokens)
            
            all_tokens.extend(masked_tokens)
            all_positions.extend(positions)
            all_labels.extend(labels)
        
        # Add [SOS] and [EOS]
        all_tokens = [self.sos_idx] + all_tokens + [self.eos_idx]
        all_labels = [-1] + all_labels + [-1]
        all_positions = [(0.0, 0.0, 0.0)] + all_positions + [(0.0, 0.0, 0.0)]
        
        # Truncate or pad to seq_len
        if len(all_tokens) > self.seq_len:
            all_tokens = all_tokens[:self.seq_len]
            all_labels = all_labels[:self.seq_len]
            all_positions = all_positions[:self.seq_len]
        else:
            padding_len = self.seq_len - len(all_tokens)
            all_tokens += [self.pad_idx] * padding_len
            all_labels += [-1] * padding_len
            all_positions += [(0.0, 0.0, 0.0)] * padding_len
        
        # Create segment labels (all 0 for single sequence)
        segment_label = [0] * self.seq_len
        
        # Split positions
        binary_pos = [p[0] for p in all_positions]
        function_pos = [p[1] for p in all_positions]
        bb_pos = [p[2] for p in all_positions]
        
        return {
            'bert_input': torch.LongTensor(all_tokens),
            'bert_label': torch.LongTensor(all_labels),
            'segment_label': torch.LongTensor(segment_label),
            'binary_pos': torch.FloatTensor(binary_pos),
            'function_pos': torch.FloatTensor(function_pos),
            'bb_pos': torch.FloatTensor(bb_pos),
        }
    
    def _process_line_for_token_masking(self, line):
        """
        Process a full line and apply token-level MLM masking.
        This is standard BERT-style MLM where individual tokens are randomly masked.
        
        Returns:
            bert_input: Token IDs with token-level masking
            bert_label: Labels for masked tokens (-1 for unmasked)
            segment_label: Segment IDs
            binary_pos, function_pos, bb_pos: Position embeddings
        """
        instructions = line.split('\t')
        
        all_tokens = []
        all_positions = []
        all_labels = []
        
        for inst_text in instructions:
            inst_text = inst_text.strip()
            if not inst_text:
                continue
            
            tokens, positions = self._parse_instruction(inst_text)
            
            # Apply token-level masking
            masked_tokens, labels = self._mask_tokens(tokens, positions)
            
            all_tokens.extend(masked_tokens)
            all_positions.extend(positions)
            all_labels.extend(labels)
        
        # Add [SOS] and [EOS]
        all_tokens = [self.sos_idx] + all_tokens + [self.eos_idx]
        all_labels = [-1] + all_labels + [-1]
        all_positions = [(0.0, 0.0, 0.0)] + all_positions + [(0.0, 0.0, 0.0)]
        
        # Truncate or pad to seq_len
        if len(all_tokens) > self.seq_len:
            all_tokens = all_tokens[:self.seq_len]
            all_labels = all_labels[:self.seq_len]
            all_positions = all_positions[:self.seq_len]
        else:
            padding_len = self.seq_len - len(all_tokens)
            all_tokens += [self.pad_idx] * padding_len
            all_labels += [-1] * padding_len
            all_positions += [(0.0, 0.0, 0.0)] * padding_len
        
        # Create segment labels (all 0 for single sequence)
        segment_label = [0] * self.seq_len
        
        # Split positions
        binary_pos = [p[0] for p in all_positions]
        function_pos = [p[1] for p in all_positions]
        bb_pos = [p[2] for p in all_positions]
        
        return {
            'bert_input': torch.LongTensor(all_tokens),
            'bert_label': torch.LongTensor(all_labels),
            'segment_label': torch.LongTensor(segment_label),
            'binary_pos': torch.FloatTensor(binary_pos),
            'function_pos': torch.FloatTensor(function_pos),
            'bb_pos': torch.FloatTensor(bb_pos),
        }
    
    def __len__(self):
        """Length = number of CFG lines"""
        return len(self.cfg_lines)
    
    def __getitem__(self, index):
        """
        Get one training sample with IMC, IMD, and MLM.
        
        Returns:
            - imc_data: Dict with instruction-masked data from full CFG line
            - imd_data: Dict with instruction-masked data from full DFG line (if enabled)
            - mlm_data: Dict with token-masked data from full CFG line (different masking)
        """
        # IMC: Process the full CFG line with instruction-level masking
        imc_data = self._process_line_for_instruction_masking(self.cfg_lines[index])
        
        # MLM: Process the full CFG line with token-level masking (independent from IMC)
        mlm_data = self._process_line_for_token_masking(self.cfg_lines[index])
        
        result = {
            'imc': imc_data,
            'mlm': mlm_data,
        }
        
        # IMD: Process the full DFG line with instruction-level masking (if enabled)
        if self.enable_imd and self.dfg_lines:
            dfg_index = index % len(self.dfg_lines)
            imd_data = self._process_line_for_instruction_masking(self.dfg_lines[dfg_index])
            result['imd'] = imd_data
        
        return result
