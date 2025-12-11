"""
Scope DataLoader for Address-Aware PalmTree

Loads instruction pairs with scope labels for scope prediction task.
Scope indicates the relationship between two instructions (0, 1, or 2).
"""

import torch
from torch.utils.data import Dataset
import re
import random


class ScopeDataset(Dataset):
    """
    Scope dataset for training scope prediction.
    
    Each sample contains two instructions and a scope label (0, 1, or 2).
    """
    
    def __init__(
        self,
        scope_corpus_path,
        vocab,
        seq_len=512,
        encoding="utf-8",
        on_memory=True,
        data_percentage=1.0,
        train_split=1.0,
        is_train=True,
    ):
        """
        Args:
            scope_corpus_path: Path to scope file (two instructions + label per line)
            vocab: Vocabulary object
            seq_len: Maximum sequence length
            encoding: File encoding
            on_memory: Load all data into memory
            data_percentage: Percentage of dataset to use (0.0-1.0)
            train_split: Train/val split ratio
            is_train: True for training set, False for validation set
        """
        self.vocab = vocab
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.encoding = encoding
        self.data_percentage = max(0.0, min(1.0, data_percentage))
        self.train_split = max(0.0, min(1.0, train_split))
        self.is_train = is_train
        
        # Regex patterns for parsing inline format
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        
        # Load data
        print(f"Loading scope corpus from {scope_corpus_path}")
        self.lines = self._load_corpus(scope_corpus_path)
        print(f"Loaded {len(self.lines)} scope samples")
        
        # Apply data percentage filter
        if self.data_percentage < 1.0:
            size = int(len(self.lines) * self.data_percentage)
            self.lines = self.lines[:size]
            print(f"Using {self.data_percentage*100:.1f}% of data: {len(self.lines)} samples")
        
        # Apply train/validation split
        if self.train_split < 1.0:
            train_size = int(len(self.lines) * self.train_split)
            
            if self.is_train:
                self.lines = self.lines[:train_size]
                print(f"Training set: {len(self.lines)} samples ({self.train_split*100:.1f}%)")
            else:
                self.lines = self.lines[train_size:]
                print(f"Validation set: {len(self.lines)} samples ({(1-self.train_split)*100:.1f}%)")
    
    def _load_corpus(self, path):
        """Load corpus from file."""
        with open(path, "r", encoding=self.encoding) as f:
            if self.on_memory:
                lines = [line.strip() for line in f if line.strip()]
            else:
                lines = None  # Not implemented for large files
        return lines
    
    def __len__(self):
        return len(self.lines)
    
    def _parse_instruction(self, inst_str):
        """
        Parse instruction in inline format.
        
        Format: opcode(addr:bnorm:fnorm:bbnorm) operand1 operand2 ...
        Example: mov(0x401000:0.5:0.2:0.1) rax rbx
        
        Returns:
            tokens: List of token strings
            positions: List of (binary_pos, function_pos, bb_pos) tuples for each token
            var_offsets: List of var offset values (0 for non-var tokens)
        """
        inst_str = inst_str.strip()
        if not inst_str:
            return [], [], []
        
        # Match main instruction pattern
        main_match = self.addr_pattern.match(inst_str)
        if not main_match:
            # No address info, treat as regular tokens
            tokens = inst_str.split()
            return tokens, [(0.0, 0.0, 0.0)] * len(tokens), [0] * len(tokens)
        
        opcode = main_match.group(1)
        bnorm = float(main_match.group(3))
        fnorm = float(main_match.group(4))
        bbnorm = float(main_match.group(5))
        
        # Start with opcode and its position
        tokens = [opcode]
        positions = [(bnorm, fnorm, bbnorm)]
        var_offsets = [0]  # Opcode is not a var
        
        # Get operands (everything after the address part)
        operands_str = inst_str[main_match.end():].strip()
        
        # Parse operands
        if operands_str:
            for operand in operands_str.split():
                # Check if operand is address(...)
                nested_match = self.nested_addr_pattern.match(operand)
                if nested_match:
                    # Extract position info from the address(...) wrapper
                    nested_bnorm = float(nested_match.group(2))
                    nested_fnorm = float(nested_match.group(3))
                    nested_bbnorm = float(nested_match.group(4))
                    tokens.append('address')
                    positions.append((nested_bnorm, nested_fnorm, nested_bbnorm))
                    var_offsets.append(0)  # address is not a var
                elif operand.startswith('var(') and operand.endswith(')'):
                    # Extract var offset: var(0x10) -> offset = 16
                    var_hex = operand[4:-1]  # Remove 'var(' and ')'
                    try:
                        var_offset_value = int(var_hex, 16)
                    except ValueError:
                        var_offset_value = 0
                    tokens.append(operand)
                    positions.append((0.0, 0.0, 0.0))  # var has no position info
                    var_offsets.append(var_offset_value)
                else:
                    # Regular operand (rax, imm, etc.) - no position info
                    tokens.append(operand)
                    positions.append((0.0, 0.0, 0.0))
                    var_offsets.append(0)  # Not a var
        
        return tokens, positions, var_offsets
    
    def _convert_to_ids(self, tokens):
        """Convert token strings to vocabulary IDs."""
        return [self.vocab.stoi.get(token, self.vocab.unk_index) for token in tokens]
    
    def __getitem__(self, index):
        """
        Get a scope sample.
        
        Returns:
            Dictionary with:
                - bert_input: Token IDs for both instructions
                - segment_label: Segment labels (0 for first inst, 1 for second)
                - scope_label: Scope label (0, 1, or 2)
                - binary_pos: Binary-level positions
                - function_pos: Function-level positions
                - bb_pos: Basic block-level positions
                - var_offsets: Var offset values (0 for non-var tokens)
        """
        line = self.lines[index]
        
        # Parse line: inst1 \t inst2 \t label
        parts = line.split('\t')
        if len(parts) != 3:
            # Fallback for malformed lines
            parts = line.split()
            if len(parts) >= 3:
                inst1_str = parts[0]
                inst2_str = parts[1]
                scope_label = int(parts[2]) if parts[2].isdigit() else 0
            else:
                inst1_str = ""
                inst2_str = ""
                scope_label = 0
        else:
            inst1_str = parts[0]
            inst2_str = parts[1]
            scope_label = int(parts[2]) if parts[2].isdigit() else 0
        
        # Parse both instructions
        tokens1, positions1, var_offsets1 = self._parse_instruction(inst1_str)
        tokens2, positions2, var_offsets2 = self._parse_instruction(inst2_str)
        
        # Convert to IDs
        ids1 = self._convert_to_ids(tokens1)
        ids2 = self._convert_to_ids(tokens2)
        
        # Truncate if needed
        max_len_per_inst = (self.seq_len - 3) // 2  # Reserve 3 for [SOS], [EOS], [EOS]
        ids1 = ids1[:max_len_per_inst]
        ids2 = ids2[:max_len_per_inst]
        positions1 = positions1[:max_len_per_inst]
        positions2 = positions2[:max_len_per_inst]
        var_offsets1 = var_offsets1[:max_len_per_inst]
        var_offsets2 = var_offsets2[:max_len_per_inst]
        
        # Build sequence: [SOS] inst1 [EOS] inst2 [EOS]
        bert_input = [self.vocab.sos_index] + ids1 + [self.vocab.eos_index] + ids2 + [self.vocab.eos_index]
        segment_labels = [0] * (1 + len(ids1) + 1) + [1] * (len(ids2) + 1)
        
        # Build position sequences
        # Special tokens ([SOS], [EOS]) get (0.0, 0.0, 0.0) - no position info
        all_positions = [(0.0, 0.0, 0.0)] + positions1 + [(0.0, 0.0, 0.0)] + positions2 + [(0.0, 0.0, 0.0)]
        
        binary_pos = [pos[0] for pos in all_positions]
        function_pos = [pos[1] for pos in all_positions]
        bb_pos = [pos[2] for pos in all_positions]
        
        # Build var_offsets sequence
        # Special tokens ([SOS], [EOS]) get 0 - not a var
        all_var_offsets = [0] + var_offsets1 + [0] + var_offsets2 + [0]
        
        # Pad to seq_len
        padding_len = self.seq_len - len(bert_input)
        bert_input += [self.vocab.pad_index] * padding_len
        segment_labels += [0] * padding_len
        binary_pos += [0.0] * padding_len
        function_pos += [0.0] * padding_len
        bb_pos += [0.0] * padding_len
        all_var_offsets += [0] * padding_len
        
        return {
            'bert_input': torch.tensor(bert_input, dtype=torch.long),
            'segment_label': torch.tensor(segment_labels, dtype=torch.long),
            'scope_label': torch.tensor(scope_label, dtype=torch.long),
            'binary_pos': torch.tensor(binary_pos, dtype=torch.float),
            'function_pos': torch.tensor(function_pos, dtype=torch.float),
            'bb_pos': torch.tensor(bb_pos, dtype=torch.float),
            'var_offsets': torch.tensor(all_var_offsets, dtype=torch.long),
        }
