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


class AddressAwareDataset(Dataset):
    """
    Dataset for jTrans with address-aware tokenization.
    
    Expects data format from jTrans datautils with address annotations:
    opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 operand2 ...
    
    Strategy:
    - MLM: Mask individual tokens at token_mask_rate (e.g., 15%)
    - No separate CFG/DFG - just one unified corpus from jTrans datautils
    """
    
    def __init__(
        self,
        corpus_path,
        vocab,
        seq_len=512,
        encoding="utf-8",
        on_memory=True,
        token_mask_prob=0.15,  # Standard MLM masking rate
        data_percentage=1.0,
        train_split=1.0,
        is_train=True,
    ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.token_mask_prob = token_mask_prob
        
        # Special token IDs
        self.pad_idx = vocab.stoi.get('<pad>', 0)
        self.unk_idx = vocab.stoi.get('<unk>', 1)
        self.eos_idx = vocab.stoi.get('<eos>', 2)
        self.sos_idx = vocab.stoi.get('<sos>', 3)
        self.mask_idx = vocab.stoi.get('<mask>', 4)
        
        # Jump/branch instruction opcodes for JTP task
        self.jump_opcodes = {
            'jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jge', 'jl', 'jle', 
            'ja', 'jae', 'jb', 'jbe', 'jo', 'jno', 'js', 'jns', 'jp', 'jnp',
            'jcxz', 'jecxz', 'jrcxz', 'call', 'ret', 'retn', 'retf'
        }
        
        # Regex patterns for parsing inline format from jTrans datautils
        # opcode(0xADDR:bnorm:fnorm:bbnorm)
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        # address(0xADDR:bnorm:fnorm:bbnorm) for jump targets
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        # daddr(0xADDR:bnorm:fnorm:bbnorm) for data addresses
        self.daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        # var(0xXX) for stack variables
        self.var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
        
        # Load data from jTrans datautils output
        self.lines = []
        if corpus_path:
            print(f"Loading corpus from {corpus_path}")
            self.lines = self._load_corpus(corpus_path)
            print(f"Loaded {len(self.lines)} lines")
        
        # Apply data percentage
        if data_percentage < 1.0:
            size = int(len(self.lines) * data_percentage)
            self.lines = self.lines[:size]
        
        # Apply train/val split
        if train_split < 1.0:
            train_size = int(len(self.lines) * train_split)
            
            if is_train:
                self.lines = self.lines[:train_size]
            else:
                self.lines = self.lines[train_size:]
        
        print(f"Dataset size: {len(self.lines)} lines")
        print(f"Token mask rate: {token_mask_prob}")
    
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
        """Parse a single instruction and extract tokens + positions + var offsets"""
        tokens = []
        positions = []
        var_offsets = []  # Store var offset for each token (-1 for non-var tokens, >=0 for var tokens)
        
        # Match opcode with address
        match = self.addr_pattern.match(inst_text)
        if not match:
            return [inst_text], [(-1.0, -1.0, -1.0)], [-1]
        
        opcode = match.group(1)
        binary_pos = float(match.group(3))
        function_pos = float(match.group(4))
        bb_pos = float(match.group(5))
        
        tokens.append(opcode)
        positions.append((binary_pos, function_pos, bb_pos))
        var_offsets.append(-1)  # Opcode is not a var, use -1 as sentinel
        
        # Parse operands
        operands_text = inst_text[match.end():].strip()
        if operands_text:
            for operand in operands_text.split():
                nested_match = self.nested_addr_pattern.match(operand)
                var_match = self.var_pattern.match(operand)
                
                if nested_match:
                    nested_binary_pos = float(nested_match.group(2))
                    nested_function_pos = float(nested_match.group(3))
                    nested_bb_pos = float(nested_match.group(4))
                    tokens.append('address')
                    positions.append((nested_binary_pos, nested_function_pos, nested_bb_pos))
                    var_offsets.append(-1)  # address is not a var, use -1 as sentinel
                elif var_match:
                    # Extract var offset from var(0xXX)
                    var_hex = var_match.group(1)
                    var_offset_value = int(var_hex, 16)
                    
                    # Handle 64-bit negative offsets (two's complement)
                    # Values > 0x7FFFFFFFFFFFFFFF are negative in two's complement
                    if var_offset_value > 0x7FFFFFFFFFFFFFFF:
                        # Convert to signed 64-bit integer
                        var_offset_value = var_offset_value - 0x10000000000000000
                    
                    tokens.append('var')  # Token is just 'var'
                    positions.append((-1.0, -1.0, -1.0))  # var has no address position, use -1 as sentinel
                    var_offsets.append(var_offset_value)  # Store the actual offset (can be negative)
                else:
                    tokens.append(operand)
                    positions.append((-1.0, -1.0, -1.0))  # non-address operand, use -1 as sentinel
                    var_offsets.append(-1)  # Not a var, use -1 as sentinel
        
        return tokens, positions, var_offsets
    
    def _mask_instruction(self, tokens, positions, var_offsets):
        """
        Mask entire instruction (all tokens become [MASK]).
        
        Returns:
            masked_tokens: All tokens replaced with [MASK]
            positions: Original positions preserved
            var_offsets: Zeroed out to prevent data leakage (var is masked)
            original_tokens: Ground truth labels
        """
        masked_tokens = [self.mask_idx] * len(tokens)
        original_tokens = [self.vocab.stoi.get(t, self.unk_idx) for t in tokens]
        # Set var_offsets to -1 to prevent data leakage - if model sees offset >= 0,
        # it would trivially know the token is 'var'
        masked_var_offsets = [-1] * len(tokens)
        
        return masked_tokens, positions, masked_var_offsets, original_tokens
    
    def _mask_tokens(self, tokens, positions, var_offsets):
        """
        Apply standard MLM token-level masking.
        
        Returns:
            masked_tokens: Tokens with some randomly masked
            masked_var_offsets: Var offsets zeroed for masked tokens (prevent data leakage)
            labels: -1 for unmasked, original token ID for masked
        """
        masked_tokens = []
        masked_var_offsets = []
        labels = []
        
        for i, (token, pos) in enumerate(zip(tokens, positions)):
            token_id = self.vocab.stoi.get(token, self.unk_idx)
            var_offset = var_offsets[i] if i < len(var_offsets) else -1
            
            # Randomly mask this token
            if random.random() < self.token_mask_prob:
                # 80% replace with [MASK]
                # 10% replace with random token
                # 10% keep original
                prob = random.random()
                if prob < 0.8:
                    masked_tokens.append(self.mask_idx)
                elif prob < 0.9:
                    # Random token - ensure it's within valid vocab range
                    max_valid_id = len(self.vocab) - 1
                    random_token_id = random.randint(5, max_valid_id)
                    assert random_token_id <= max_valid_id, f"Random token {random_token_id} > {max_valid_id}"
                    masked_tokens.append(random_token_id)
                else:
                    masked_tokens.append(token_id)
                labels.append(token_id)
                # Set var_offset to -1 for masked tokens to prevent data leakage
                masked_var_offsets.append(-1)
            else:
                masked_tokens.append(token_id)
                labels.append(-100)  # Not masked - use -100 to match ignore_index in CrossEntropyLoss
                # Keep var_offset for non-masked tokens
                masked_var_offsets.append(var_offset)
        
        return masked_tokens, masked_var_offsets, labels
    
    def _process_line_for_instruction_masking(self, line):
        """
        Process a full line and apply instruction-level masking.
        Format: <sos> inst1 inst2 inst3 ... <eos>
        NO separators between instructions, just <sos> at start and <eos> at end.
        Segment labels: All tokens get segment 1 (single sequence).
        
        Strategy: Calculate number of instructions to mask based on instruction_mask_prob,
        then randomly select which instructions to mask.
        
        Returns:
            bert_input: Token IDs with instruction-level masking
            bert_label: Labels for masked instructions (-1 for unmasked)
            segment_label: All 1s (single sequence)
            binary_pos, function_pos, bb_pos: Position embeddings
            var_offsets: Var offset values (-1 for non-var tokens)
        """
        instructions = [inst.strip() for inst in line.split('\t') if inst.strip()]
        
        # Calculate how many instructions to mask
        num_instructions = len(instructions)
        if num_instructions == 0:
            # Empty line, return minimal valid output
            return self._create_empty_sample()
        
        num_to_mask = max(1, int(num_instructions * self.instruction_mask_prob))
        # Ensure we don't mask all instructions (if more than 1)
        if num_instructions > 1:
            num_to_mask = min(num_to_mask, num_instructions - 1)
        
        # Randomly select which instructions to mask
        instructions_to_mask = set(random.sample(range(num_instructions), num_to_mask))
        
        all_tokens = []
        all_positions = []
        all_labels = []
        all_segments = []
        all_var_offsets = []
        
        # Add <sos> at the beginning (segment 1)
        all_tokens.append(self.sos_idx)
        all_positions.append((-1.0, -1.0, -1.0))
        all_labels.append(-100)  # SOS is not masked - use -100 to match ignore_index
        all_segments.append(1)
        all_var_offsets.append(-1)  # SOS is not a var, use -1 as sentinel
        
        for inst_idx, inst_text in enumerate(instructions):
            tokens, positions, var_offsets = self._parse_instruction(inst_text)
            
            # Mask this instruction if it was selected
            if inst_idx in instructions_to_mask:
                # Mask entire instruction
                masked_tokens, positions, var_offsets, labels = self._mask_instruction(tokens, positions, var_offsets)
            else:
                # No instruction-level masking (but still convert to IDs)
                masked_tokens = [self.vocab.stoi.get(t, self.unk_idx) for t in tokens]
                labels = [-100] * len(tokens)  # Not masked - use -100 to match ignore_index
            
            # All tokens in the same segment (segment 1)
            all_tokens.extend(masked_tokens)
            all_positions.extend(positions)
            all_labels.extend(labels)
            all_segments.extend([1] * len(masked_tokens))
            all_var_offsets.extend(var_offsets)
            
            # NO <eos> after each instruction - we'll add it only at the very end
        
        # Add <eos> at the end (segment 1)
        all_tokens.append(self.eos_idx)
        all_positions.append((-1.0, -1.0, -1.0))
        all_labels.append(-100)  # EOS is not masked - use -100 to match ignore_index
        all_segments.append(1)
        all_var_offsets.append(-1)  # EOS is not a var, use -1 as sentinel
        
        # Generate JTP labels
        jtp_labels = self._create_jtp_labels(instructions, all_tokens_info)
        
        # Truncate or pad to seq_len
        if len(all_tokens) > self.seq_len:
            all_tokens = all_tokens[:self.seq_len]
            all_labels = all_labels[:self.seq_len]
            all_positions = all_positions[:self.seq_len]
            all_segments = all_segments[:self.seq_len]
            all_var_offsets = all_var_offsets[:self.seq_len]
            jtp_labels = jtp_labels[:self.seq_len]
        else:
            padding_len = self.seq_len - len(all_tokens)
            all_tokens += [self.pad_idx] * padding_len
            all_labels += [-100] * padding_len  # Use -100 to match CrossEntropyLoss ignore_index
            all_positions += [(-1.0, -1.0, -1.0)] * padding_len
            all_segments += [0] * padding_len  # Padding gets segment 0
            all_var_offsets += [-1] * padding_len  # Padding is not a var, use -1 as sentinel
            jtp_labels += [-100] * padding_len  # Padding tokens have no JTP label
        
        # Use segment labels as instruction IDs
        segment_label = all_segments
        
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
            'var_offsets': torch.LongTensor(all_var_offsets),
        }
    
    def _create_jtp_labels(self, instructions, all_tokens_info):
        """
        Create Jump Target Prediction labels.
        
        Args:
            instructions: List of instruction strings
            all_tokens_info: List of (token_id, inst_idx, is_opcode, raw_text) for each token position
            
        Returns:
            jtp_labels: List of labels (-100 for non-jumps, target_token_position for jumps)
        """
        # Build mapping: instruction address -> instruction index
        addr_to_inst_idx = {}
        inst_addresses = []
        
        for inst_idx, inst_text in enumerate(instructions):
            inst_text = inst_text.strip()
            if not inst_text:
                continue
            
            # Extract instruction address from opcode(0xADDR:...)
            match = self.addr_pattern.match(inst_text)
            if match:
                inst_addr = match.group(2)  # Get 0xADDR
                addr_to_inst_idx[inst_addr] = inst_idx
                inst_addresses.append(inst_addr)
        
        # Build mapping: instruction index -> first token position (opcode position)
        inst_idx_to_token_pos = {}
        for token_pos, (token_id, inst_idx, is_opcode, raw_text) in enumerate(all_tokens_info):
            if is_opcode and inst_idx >= 0 and inst_idx not in inst_idx_to_token_pos:
                inst_idx_to_token_pos[inst_idx] = token_pos
        
        # First pass: identify jump instructions and their targets
        jump_inst_targets = {}  # inst_idx -> target_token_pos
        for token_pos, (token_id, inst_idx, is_opcode, raw_text) in enumerate(all_tokens_info):
            if is_opcode and raw_text in self.jump_opcodes:
                if inst_idx < len(instructions):
                    inst_text = instructions[inst_idx].strip()
                    
                    # Check if instruction has address operand
                    if 'address(' in inst_text:
                        # Extract target address from address(0xADDR:...)
                        nested_match = self.nested_addr_pattern.search(inst_text)
                        if nested_match:
                            target_addr = nested_match.group(1)  # 0xADDR
                            target_inst_idx = addr_to_inst_idx.get(target_addr, -1)
                            
                            # Convert instruction index to token position
                            if target_inst_idx >= 0:
                                target_token_pos = inst_idx_to_token_pos.get(target_inst_idx, -100)
                                jump_inst_targets[inst_idx] = target_token_pos
        
        # Second pass: assign JTP labels to 'address' tokens in jump instructions
        jtp_labels = []
        for token_pos, (token_id, inst_idx, is_opcode, raw_text) in enumerate(all_tokens_info):
            # Place label on the 'address' token that contains the jump target
            if raw_text == 'address' and inst_idx in jump_inst_targets:
                jtp_labels.append(jump_inst_targets[inst_idx])
            else:
                # Not a jump target address - use -100 (ignored in loss)
                jtp_labels.append(-100)
        
        return jtp_labels
    
    def _process_line_for_token_masking(self, line):
        """
        Process a full line and apply token-level MLM masking.
        Format: [SOS] inst1 [EOS] inst2 [EOS] inst3 [EOS] ...
        Segment labels: [SOS] gets segment 1, inst1 tokens get 1, [EOS] gets 1, 
                        inst2 tokens get 2, [EOS] gets 2, etc.
        
        Returns:
            bert_input: Token IDs with token-level masking
            bert_label: Labels for masked tokens (-1 for unmasked)
            segment_label: Instruction ID (1 for inst1, 2 for inst2, ...)
            binary_pos, function_pos, bb_pos: Position embeddings
            var_offsets: Var offset values (0 for non-var tokens)
        """
        instructions = line.split('\t')
        
        all_tokens = []
        all_positions = []
        all_labels = []
        all_segments = []
        all_var_offsets = []
        all_tokens_info = []  # Track (token_id, inst_idx, is_opcode, raw_text) for JTP
        
        # Add [SOS] at the beginning (gets segment 1 - same as first instruction)
        all_tokens.append(self.sos_idx)
        all_positions.append((-1.0, -1.0, -1.0))
        all_labels.append(-100)  # SOS is not masked - use -100 to match ignore_index
        all_segments.append(1)
        all_var_offsets.append(-1)  # SOS is not a var, use -1 as sentinel
        all_tokens_info.append((self.sos_idx, -1, False, '<sos>'))
        
        for inst_idx, inst_text in enumerate(instructions):
            inst_text = inst_text.strip()
            if not inst_text:
                continue
            
            tokens, positions, var_offsets = self._parse_instruction(inst_text)
            
            # Apply token-level masking
            masked_tokens, var_offsets, labels = self._mask_tokens(tokens, positions, var_offsets)
            # Segment label = instruction number (1-indexed)
            inst_segment = inst_idx + 1
            
            # Track token info for JTP (before masking)
            for token_idx, (masked_tok, orig_tok) in enumerate(zip(masked_tokens, tokens)):
                is_opcode = (token_idx == 0)  # First token is opcode
                all_tokens_info.append((masked_tok, inst_idx, is_opcode, orig_tok))
            
            all_tokens.extend(masked_tokens)
            all_positions.extend(positions)
            all_labels.extend(labels)
            all_segments.extend([inst_segment] * len(masked_tokens))
            all_var_offsets.extend(var_offsets)
            
            # Add [EOS] after each instruction (same segment as the instruction)
            all_tokens.append(self.eos_idx)
            all_positions.append((-1.0, -1.0, -1.0))
            all_labels.append(-100)  # EOS is not masked - use -100 to match ignore_index
            all_segments.append(inst_segment)
            all_var_offsets.append(-1)  # EOS is not a var, use -1 as sentinel
            all_tokens_info.append((self.eos_idx, inst_idx, False, '<eos>'))
        
        # Verify lengths match before generating JTP labels
        assert len(all_tokens) == len(all_tokens_info), f"Length mismatch: {len(all_tokens)} tokens vs {len(all_tokens_info)} token_info"
        
        # Generate JTP labels
        jtp_labels = self._create_jtp_labels(instructions, all_tokens_info)
        
        # Truncate or pad to seq_len
        if len(all_tokens) > self.seq_len:
            all_tokens = all_tokens[:self.seq_len]
            all_labels = all_labels[:self.seq_len]
            all_positions = all_positions[:self.seq_len]
            all_segments = all_segments[:self.seq_len]
            all_var_offsets = all_var_offsets[:self.seq_len]
            jtp_labels = jtp_labels[:self.seq_len]
            
            # Invalidate JTP labels that point beyond truncated sequence
            jtp_labels = [-100 if (label >= self.seq_len and label != -100) else label for label in jtp_labels]
        else:
            padding_len = self.seq_len - len(all_tokens)
            all_tokens += [self.pad_idx] * padding_len
            all_labels += [-100] * padding_len  # Use -100 to match CrossEntropyLoss ignore_index
            all_positions += [(-1.0, -1.0, -1.0)] * padding_len
            all_segments += [0] * padding_len  # Padding gets segment 0
            all_var_offsets += [-1] * padding_len  # Padding is not a var, use -1 as sentinel
            jtp_labels += [-100] * padding_len  # Padding tokens have no JTP label
        
        # Use segment labels as instruction IDs
        segment_label = all_segments
        
        # Split positions
        binary_pos = [p[0] for p in all_positions]
        function_pos = [p[1] for p in all_positions]
        bb_pos = [p[2] for p in all_positions]
        
        # Create attention mask: 1 for real tokens, 0 for padding
        attention_mask = [1 if token != self.vocab.pad_index else 0 for token in all_tokens]
        
        return {
            'bert_input': torch.LongTensor(all_tokens),
            'bert_label': torch.LongTensor(all_labels),
            'segment_label': torch.LongTensor(segment_label),
            'attention_mask': torch.LongTensor(attention_mask),
            'binary_pos': torch.FloatTensor(binary_pos),
            'function_pos': torch.FloatTensor(function_pos),
            'bb_pos': torch.FloatTensor(bb_pos),
            'var_offsets': torch.LongTensor(all_var_offsets),
            'jtp_labels': torch.LongTensor(jtp_labels),
        }
    
    def __len__(self):
        """Length = number of lines from jTrans datautils"""
        return len(self.lines)
    
    def __getitem__(self, index):
        """
        Get one training sample with MLM masking.
        
        Returns dict with standard BERT format:
        - bert_input: token IDs
        - bert_label: MLM labels (-100 for non-masked)
        - segment_label: all zeros (single sequence)
        - attention_mask: 1 for real tokens, 0 for padding
        - binary_pos, function_pos, bb_pos: hierarchical positions
        - var_offsets: var(0xXX) offsets
        """
        # Process line with token-level MLM masking
        result = self._process_line_for_token_masking(self.lines[index])
        
        # Validate token IDs are within vocab range (before returning tensors)
        bert_input = result['bert_input']
        vocab_size = len(self.vocab)
        
        # Convert to numpy for CPU-side checking
        if hasattr(bert_input, 'numpy'):
            ids = bert_input.numpy()
        else:
            ids = bert_input
        
        max_id = ids.max()
        if max_id >= vocab_size:
            print(f"\n[ERROR] Token ID {max_id} >= vocab size {vocab_size}")
            print(f"Index: {index}")
            print(f"Line preview: {self.lines[index][:300]}...")
            print(f"First 50 token IDs: {ids[:50]}")
            # Find invalid tokens
            invalid_positions = (ids >= vocab_size).nonzero()[0]
            print(f"Invalid token count: {len(invalid_positions)}")
            print(f"Invalid token positions (first 20): {invalid_positions[:20]}")
            print(f"Invalid token IDs: {ids[invalid_positions][:20]}")
            raise ValueError(f"Token ID {max_id} out of vocab range [0, {vocab_size-1}]")
        
        return result
