"""
DataLoader for Function Similarity Task

Loads function blocks and creates training pairs from ground truth data.
"""

import torch
from torch.utils.data import Dataset
import json
import random
import re
import sys
import os

# Add strupos to path for vocab
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'strupos'))

from vocab import WordVocab


class FunctionSimilarityDataset(Dataset):
    """
    Dataset for function similarity training.
    
    Creates pairs of functions:
    - Positive pairs: Same function at different optimization levels
    - Negative pairs: Different functions
    """
    
    def __init__(
        self,
        function_blocks_file,
        funcsim_pairs_file,
        vocab,
        seq_len=512,
        negative_samples=1,
        encoding="utf-8"
    ):
        """
        Args:
            function_blocks_file: Path to function_blocks.json
            funcsim_pairs_file: Path to funcsim_pairs.json
            vocab: WordVocab instance
            seq_len: Maximum sequence length
            negative_samples: Number of negative samples per positive pair
            encoding: File encoding
        """
        self.vocab = vocab
        self.seq_len = seq_len
        self.negative_samples = negative_samples
        
        # Flag to control whether to load address/var info or set to 0
        # Will be set based on what the loaded BERT model has
        self.use_address_var = True
        
        # Special token IDs
        self.pad_idx = vocab.stoi.get('<pad>', 0)
        self.unk_idx = vocab.stoi.get('<unk>', 1)
        self.eos_idx = vocab.stoi.get('<eos>', 2)
        self.sos_idx = vocab.stoi.get('<sos>', 3)
        
        # Regex patterns for parsing
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        # Pattern to match var(0xXX) tokens
        self.var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
        
        # Load function blocks
        if function_blocks_file is not None:
            print(f"[INFO] Loading function blocks from {function_blocks_file}")
            with open(function_blocks_file, 'r') as f:
                self.function_blocks = json.load(f)
        else:
            self.function_blocks = {}
        
        # Load ground truth pairs
        if funcsim_pairs_file is not None:
            print(f"[INFO] Loading ground truth pairs from {funcsim_pairs_file}")
            with open(funcsim_pairs_file, 'r') as f:
                self.funcsim_pairs = json.load(f)
        else:
            self.funcsim_pairs = {}
        
        # Create training pairs
        self.training_pairs = self._create_training_pairs()
        
        print(f"[INFO] Loaded {len(self.function_blocks)} function blocks")
        print(f"[INFO] Created {len(self.training_pairs)} training pairs")
    
    def _create_training_pairs(self):
        """
        Create training pairs from ground truth.
        
        For each function, create:
        - Positive pairs: (anchor, positive) where positive is ground truth
        - Negative pairs: (anchor, negative) where negative is random different function
        
        Returns:
            List of (anchor_id, pair_id, label) tuples
            label=1 for positive, label=0 for negative
        """
        pairs = []
        all_function_ids = list(self.function_blocks.keys())
        
        for func_id, pair_data in self.funcsim_pairs.items():
            ground_truth = pair_data['ground_truth']
            
            # Create positive pairs with each ground truth
            for gt_id in ground_truth:
                pairs.append((func_id, gt_id, 1))  # Positive pair
            
            # Create negative pairs
            for _ in range(self.negative_samples):
                # Sample a random function that's not in ground truth
                negative_id = random.choice(all_function_ids)
                while negative_id == func_id or negative_id in ground_truth:
                    negative_id = random.choice(all_function_ids)
                pairs.append((func_id, negative_id, 0))  # Negative pair
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        return pairs
    
    def _parse_instruction(self, inst):
        """
        Parse instruction to extract tokens, address positions, and var offsets.
        
        Returns:
            tokens: List of token strings
            positions: List of (binary_pos, function_pos, bb_pos) tuples for each token
            var_offsets: List of var offset values (0 for non-var tokens)
        """
        # Extract opcode and positions from opcode(addr:pos1:pos2:pos3)
        match = self.addr_pattern.match(inst)
        if match:
            opcode = match.group(1)
            addr = match.group(2)
            binary_pos_val = float(match.group(3))
            function_pos_val = float(match.group(4))
            bb_pos_val = float(match.group(5))
            
            # Remove the matched part to get operands
            operands = inst[match.end():].strip()
        else:
            # Fallback: use default positions
            opcode = inst.split()[0] if inst else ''
            operands = ' '.join(inst.split()[1:]) if len(inst.split()) > 1 else ''
            binary_pos_val = 0.0
            function_pos_val = 0.0
            bb_pos_val = 0.0
        
        # Start with opcode
        tokens = [opcode]
        positions = [(binary_pos_val, function_pos_val, bb_pos_val)]
        var_offsets = [0]  # Opcode is not a var
        
        # Parse operands
        if operands:
            pos = 0
            while pos < len(operands):
                # Check for address(...)
                nested_match = self.nested_addr_pattern.match(operands[pos:])
                if nested_match:
                    # Extract address positions from nested address token
                    nested_binary_pos = float(nested_match.group(2))
                    nested_function_pos = float(nested_match.group(3))
                    nested_bb_pos = float(nested_match.group(4))
                    tokens.append('address')
                    positions.append((nested_binary_pos, nested_function_pos, nested_bb_pos))
                    var_offsets.append(0)  # address is not a var
                    pos += nested_match.end()
                    continue
                
                # Check for var(0xXX)
                var_match = self.var_pattern.match(operands[pos:])
                if var_match:
                    # Extract var offset from var(0xXX)
                    var_hex = var_match.group(1)
                    var_offset_value = int(var_hex, 16)
                    tokens.append('var')
                    positions.append((0.0, 0.0, 0.0))  # var has no address position
                    var_offsets.append(var_offset_value)  # Store the offset
                    pos += var_match.end()
                    continue
                
                # Regular character
                char = operands[pos]
                if char.isspace():
                    pos += 1
                elif char in '[]+-*,':
                    tokens.append(char)
                    positions.append((0.0, 0.0, 0.0))
                    var_offsets.append(0)
                    pos += 1
                else:
                    # Read word
                    end = pos
                    while end < len(operands) and operands[end] not in ' []+-*,':
                        end += 1
                    tokens.append(operands[pos:end])
                    positions.append((0.0, 0.0, 0.0))
                    var_offsets.append(0)
                    pos = end
        
        return tokens, positions, var_offsets
    
    def _process_function(self, func_id):
        """
        Process a function block into token IDs and positions.
        
        Returns:
            bert_input: [seq_len]
            segment_label: [seq_len]
            binary_pos: [seq_len]
            function_pos: [seq_len]
            bb_pos: [seq_len]
            var_offsets: [seq_len]
        """
        instructions = self.function_blocks[func_id]
        
        # Parse all instructions
        all_tokens = []
        all_binary_pos = []
        all_function_pos = []
        all_bb_pos = []
        all_var_offsets = []
        
        for inst in instructions:
            tokens, positions, var_offsets = self._parse_instruction(inst)
            all_tokens.extend(tokens)
            all_var_offsets.extend(var_offsets)
            # Extract positions for each token
            for pos in positions:
                all_binary_pos.append(pos[0])
                all_function_pos.append(pos[1])
                all_bb_pos.append(pos[2])
        
        # Truncate or pad to seq_len
        if len(all_tokens) > self.seq_len:
            all_tokens = all_tokens[:self.seq_len]
            all_binary_pos = all_binary_pos[:self.seq_len]
            all_function_pos = all_function_pos[:self.seq_len]
            all_bb_pos = all_bb_pos[:self.seq_len]
            all_var_offsets = all_var_offsets[:self.seq_len]
        
        # Convert tokens to IDs
        token_ids = [self.vocab.stoi.get(t, self.unk_idx) for t in all_tokens]
        
        # Pad
        padding_len = self.seq_len - len(token_ids)
        token_ids = token_ids + [self.pad_idx] * padding_len
        
        # If model doesn't have address/var embeddings, set everything to 0
        if not self.use_address_var:
            all_binary_pos = [0.0] * self.seq_len
            all_function_pos = [0.0] * self.seq_len
            all_bb_pos = [0.0] * self.seq_len
            all_var_offsets = [0] * self.seq_len
        else:
            all_binary_pos = all_binary_pos + [0.0] * padding_len
            all_function_pos = all_function_pos + [0.0] * padding_len
            all_bb_pos = all_bb_pos + [0.0] * padding_len
            all_var_offsets = all_var_offsets + [0] * padding_len
        
        # Segment labels (all 0 for single function)
        segment_labels = [0] * self.seq_len
        
        return (
            torch.tensor(token_ids),
            torch.tensor(segment_labels),
            torch.tensor(all_binary_pos, dtype=torch.float),
            torch.tensor(all_function_pos, dtype=torch.float),
            torch.tensor(all_bb_pos, dtype=torch.float),
            torch.tensor(all_var_offsets, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        """
        Get a training pair.
        
        Returns:
            Dictionary with:
            - func1_*: First function data (input, segment, binary_pos, function_pos, bb_pos, var_offsets)
            - func2_*: Second function data (input, segment, binary_pos, function_pos, bb_pos, var_offsets)
            - label: 1 for positive pair, 0 for negative pair
        """
        func1_id, func2_id, label = self.training_pairs[idx]
        
        # Process both functions (now returns 6 values including var_offsets)
        func1_input, func1_segment, func1_bin_pos, func1_func_pos, func1_bb_pos, func1_var_offsets = self._process_function(func1_id)
        func2_input, func2_segment, func2_bin_pos, func2_func_pos, func2_bb_pos, func2_var_offsets = self._process_function(func2_id)
        
        return {
            'func1_input': func1_input,
            'func1_segment': func1_segment,
            'func1_binary_pos': func1_bin_pos,
            'func1_function_pos': func1_func_pos,
            'func1_bb_pos': func1_bb_pos,
            'func1_var_offsets': func1_var_offsets,
            'func2_input': func2_input,
            'func2_segment': func2_segment,
            'func2_binary_pos': func2_bin_pos,
            'func2_function_pos': func2_func_pos,
            'func2_bb_pos': func2_bb_pos,
            'func2_var_offsets': func2_var_offsets,
            'label': torch.tensor(label, dtype=torch.long)
        }

