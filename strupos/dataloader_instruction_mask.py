"""
Instruction Masking DataLoader for Address-Aware BERT

Implements instruction-level masking where entire instructions are masked at once.
This is different from token-level MLM which masks individual tokens.

For a line with 8 instructions and instruction_mask_rate=0.15:
- ~1 instruction will be fully masked (all its tokens replaced with [MASK])
- The model must predict all tokens of the masked instruction
- Also supports standard token-level MLM and NSP

Tasks:
- IM (Instruction Masking): Mask entire instructions
- MLM (Masked Language Model): Mask individual tokens  
- NSP-CFG: Next sentence prediction on CFG pairs
- NSP-DFG: Next sentence prediction on DFG pairs
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
    - IM: Mask entire instructions at instruction_mask_rate (e.g., 15%)
    - MLM: Mask individual tokens at token_mask_rate (e.g., 15%)
    - NSP-CFG: Create one random consecutive pair per line
    - NSP-DFG: Use DFG lines as-is
    """
    
    def __init__(
        self,
        cfg_corpus_path,
        dfg_corpus_path,
        vocab,
        seq_len=512,
        nsp_content_max=20,
        encoding="utf-8",
        on_memory=True,
        nsp_prob=0.5,
        token_mask_prob=0.15,  # Standard MLM masking rate
        instruction_mask_prob=0.25,  # Instruction-level masking rate
        data_percentage=1.0,
        train_split=1.0,
        is_train=True,
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.nsp_content_max = nsp_content_max
        self.nsp_prob = nsp_prob
        self.token_mask_prob = token_mask_prob
        self.instruction_mask_prob = instruction_mask_prob
        
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
        
        # Create one random NSP pair per CFG line
        print("Creating one random NSP pair per CFG line...")
        self.cfg_nsp_pairs = []
        for line_idx, line in enumerate(tqdm(self.cfg_lines, desc="CFG NSP")):
            pair = self._create_random_consecutive_pair(line, line_idx)
            if pair is not None:
                self.cfg_nsp_pairs.append(pair)
        
        print(f"Dataset size:")
        print(f"  CFG lines (for IM+MLM): {len(self.cfg_lines)}")
        print(f"  CFG NSP pairs: {len(self.cfg_nsp_pairs)}")
        print(f"  DFG lines (NSP only): {len(self.dfg_lines)}")
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
    
    def _create_random_consecutive_pair(self, line, line_idx):
        """Create one random consecutive pair from a line"""
        instructions = line.split('\t')
        
        if len(instructions) < 2:
            return None
        
        # Randomly pick a starting position
        i = random.randint(0, len(instructions) - 2)
        
        inst1 = instructions[i]
        inst2_original = instructions[i + 1]
        
        # Decide if positive or negative
        if random.random() < self.nsp_prob:
            # NEGATIVE: Replace inst2 with random instruction
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
        
        return (line_idx, inst1, inst2, label)
    
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
    
    def _process_nsp_pair(self, inst1_text, inst2_text, is_next):
        """Process a pair of instructions for NSP"""
        inst1_tokens, inst1_pos = self._parse_instruction(inst1_text)
        inst2_tokens, inst2_pos = self._parse_instruction(inst2_text)
        
        # Convert to IDs (no masking for NSP)
        inst1_ids = [self.vocab.stoi.get(t, self.unk_idx) for t in inst1_tokens]
        inst2_ids = [self.vocab.stoi.get(t, self.unk_idx) for t in inst2_tokens]
        
        # Combine sequences
        tokens = [self.sos_idx] + inst1_ids + [self.eos_idx] + inst2_ids + [self.eos_idx]
        positions = [(0.0, 0.0, 0.0)] + inst1_pos + [(0.0, 0.0, 0.0)] + inst2_pos + [(0.0, 0.0, 0.0)]
        segment_label = [0] * (len(inst1_ids) + 2) + [1] * (len(inst2_ids) + 1)
        
        # Truncate or pad
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
            positions = positions[:self.seq_len]
            segment_label = segment_label[:self.seq_len]
        else:
            padding_len = self.seq_len - len(tokens)
            tokens += [self.pad_idx] * padding_len
            positions += [(0.0, 0.0, 0.0)] * padding_len
            segment_label += [0] * padding_len
        
        binary_pos = [p[0] for p in positions]
        function_pos = [p[1] for p in positions]
        bb_pos = [p[2] for p in positions]
        
        return {
            'bert_input': torch.LongTensor(tokens),
            'segment_label': torch.LongTensor(segment_label),
            'is_next': torch.LongTensor([is_next]),
            'binary_pos': torch.FloatTensor(binary_pos),
            'function_pos': torch.FloatTensor(function_pos),
            'bb_pos': torch.FloatTensor(bb_pos),
        }
    
    def __len__(self):
        """Length = number of NSP pairs"""
        return len(self.cfg_nsp_pairs)
    
    def __getitem__(self, index):
        """
        Get one training sample with IM, MLM, and NSP.
        
        Returns:
            - im_data: Dict with instruction-masked data from full CFG line
            - mlm_data: Dict with token-masked data from full CFG line (different masking)
            - nsp_cfg_data: Dict with NSP pair from CFG
            - nsp_dfg_data: Dict with NSP pair from DFG
        """
        # Get CFG NSP pair
        line_idx, cfg_inst1, cfg_inst2, cfg_is_next = self.cfg_nsp_pairs[index]
        
        # IM: Process the full line with instruction-level masking
        im_data = self._process_line_for_instruction_masking(self.cfg_lines[line_idx])
        
        # MLM: Process the full line with token-level masking (independent from IM)
        mlm_data = self._process_line_for_token_masking(self.cfg_lines[line_idx])
        
        # NSP-CFG: Process the pair
        nsp_cfg_data = self._process_nsp_pair(cfg_inst1, cfg_inst2, cfg_is_next)
        
        # NSP-DFG: Get corresponding DFG pair
        dfg_idx = index % len(self.dfg_lines)
        dfg_line = self.dfg_lines[dfg_idx]
        dfg_instructions = dfg_line.split('\t')
        
        if len(dfg_instructions) >= 2:
            # DFG is already in pair format, take first two
            dfg_inst1, dfg_inst2 = dfg_instructions[0], dfg_instructions[1]
            dfg_is_next = 1  # Assume positive
        else:
            # Fallback
            dfg_inst1 = dfg_instructions[0] if dfg_instructions else ""
            dfg_inst2 = dfg_instructions[0] if dfg_instructions else ""
            dfg_is_next = 1
        
        nsp_dfg_data = self._process_nsp_pair(dfg_inst1, dfg_inst2, dfg_is_next)
        
        return {
            'im': im_data,
            'mlm': mlm_data,
            'nsp_cfg': nsp_cfg_data,
            'nsp_dfg': nsp_dfg_data,
        }
