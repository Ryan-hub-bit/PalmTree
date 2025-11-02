"""
Data loader for Basic Block pretraining
Uses TWO separate vocabularies:
1. PalmTree vocab (instruction tokens) - pre-trained
2. Address vocab (address tokens) - trained from scratch
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import re

try:
    from . import config
    from .addr_vocab import AddrVocab
except ImportError:
    import config
    from addr_vocab import AddrVocab


class BBPairDataset(Dataset):
    """
    Dataset for basic block pairs with address information
    Uses TWO vocabularies:
    - palmtree_vocab: For instruction tokens (pre-trained)
    - addr_vocab: For address tokens (separate)
    """
    
    def __init__(self, bb_pairs_file: str, vocab_file: str, max_seq_length: int = config.MAX_SEQ_LENGTH, max_pairs: int = None):
        """
        Args:
            bb_pairs_file: Path to BB pairs file (e.g., all_bb_pairs.txt)
            vocab_file: Path to PalmTree vocabulary
            max_seq_length: Maximum sequence length
            max_pairs: Maximum number of pairs to load (for testing, use None for all)
        """
        self.max_seq_length = max_seq_length
        self.max_pairs = max_pairs
        
        # Load TWO separate vocabularies
        self.palmtree_vocab = self._load_vocab(vocab_file)  # Pre-trained instruction vocab
        self.addr_vocab = AddrVocab()  # Address-specific vocab
        
        self.bb_pairs = self._load_bb_pairs(bb_pairs_file)
        
    def _load_vocab(self, vocab_file: str):
        """Load PalmTree vocabulary - supports both pickle and text formats"""
        
        # Try loading as pickle first (PalmTree format)
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pre-trained_model'))
            from vocab import WordVocab
            
            vocab_obj = WordVocab.load_vocab(vocab_file)
            return vocab_obj
        except Exception as e:
            print(f"Failed to load vocab with WordVocab: {e}")
            # Fall back to dictionary format
            vocab = {}
            try:
                import pickle
                with open(vocab_file, 'rb') as f:
                    vocab_obj = pickle.load(f)
                    # PalmTree vocab has stoi (string to index) attribute
                    if hasattr(vocab_obj, 'stoi'):
                        vocab = vocab_obj.stoi
                    elif hasattr(vocab_obj, 'itos'):
                        # Build from itos (index to string)
                        vocab = {token: idx for idx, token in enumerate(vocab_obj.itos)}
                    else:
                        vocab = vocab_obj
            except:
                # Fall back to text format
                with open(vocab_file, 'r') as f:
                    for idx, line in enumerate(f):
                        token = line.strip()
                        vocab[token] = idx
            
            return vocab
    
    def _tokenize(self, tokens: List[str]) -> Tuple[List[int], List[int]]:
        """
        Convert tokens to token IDs using TWO vocabularies
        
        Returns:
            palmtree_ids: IDs in PalmTree vocab (for Level 1 semantic embeddings)
            addr_ids: IDs in Address vocab (to identify which tokens are addresses)
        """
        palmtree_ids = []
        addr_ids = []
        
        # Check if vocab is WordVocab object or dict
        has_stoi = hasattr(self.palmtree_vocab, 'stoi')
        unk_index = self.palmtree_vocab.unk_index if has_stoi else 1
        
        for token in tokens:
            # Check if it's an address token
            if self.addr_vocab.is_address_token(token):
                # Address token: use <unk> in PalmTree, real ID in addr vocab
                palmtree_ids.append(unk_index)
                addr_ids.append(self.addr_vocab.to_index(token))
            else:
                # Regular instruction token: use PalmTree vocab, <unk> in addr vocab
                if has_stoi:
                    palmtree_ids.append(self.palmtree_vocab.stoi.get(token, unk_index))
                else:
                    palmtree_ids.append(self.palmtree_vocab.get(token, unk_index))
                addr_ids.append(self.addr_vocab.unk_index)
        
        return palmtree_ids, addr_ids
    def _load_bb_pairs(self, bb_pairs_file: str) -> List[Dict]:
        """Load BB pairs from file"""
        bb_pairs = []
        with open(bb_pairs_file, 'r') as f:
            for line_num, line in enumerate(f):
                # Stop if we've loaded enough pairs for testing
                if self.max_pairs is not None and len(bb_pairs) >= self.max_pairs:
                    break
                
                line = line.strip()
                if not line or '->' not in line or line.startswith('#'):
                    continue
                
                # Split source and target BB
                parts = line.split(' -> ')
                if len(parts) != 2:
                    continue
                
                src_bb, tgt_bb = parts[0].strip(), parts[1].strip()
                
                # Parse addresses and tokens
                src_tokens, src_addr_values, src_addr_types = self._parse_bb(src_bb)
                tgt_tokens, tgt_addr_values, tgt_addr_types = self._parse_bb(tgt_bb)
                
                # Determine edge type
                edge_type = self._classify_edge_type(src_tokens, tgt_tokens)
                
                bb_pairs.append({
                    'src_tokens': src_tokens,
                    'src_addr_values': src_addr_values,
                    'src_addr_types': src_addr_types,
                    'tgt_tokens': tgt_tokens,
                    'tgt_addr_values': tgt_addr_values,
                    'tgt_addr_types': tgt_addr_types,
                    'edge_type': edge_type,
                })
        
        return bb_pairs
    
    def _parse_bb(self, bb_str: str) -> Tuple[List[str], List[int], List[int]]:
        """
        Parse basic block string into tokens and extract addresses
        
        Format: <0x401000> inst1 inst2 addr_code:0x401016 <0x401014>
        
        Returns:
            tokens: List of token strings (with <addr_*> placeholders for addresses)
            addr_values: List of address values (one per token, 0 for regular tokens)
            addr_types: List of address type IDs (one per token)
        """
        if bb_str == '<addr_start> unknown <addr_end>' or 'unknown' in bb_str:
            # Unknown indirect branch: use sentinel value 0xFFFFFFFFFFFFFFFF
            return (
                ['<addr_start>', 'unknown', '<addr_end>'], 
                [0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF],
                [config.ADDR_TYPE_LABELS['start'], config.ADDR_TYPE_LABELS['unknown'], config.ADDR_TYPE_LABELS['end']]
            )
        
        # Split by tabs (instructions)
        instructions = bb_str.split('\t')
        tokens = []
        addr_values = []  # Address value for each token
        addr_types = []   # Address type ID for each token
        
        bb_start_addr = None
        bb_end_addr = None
        
        for inst_idx, inst in enumerate(instructions):
            inst = inst.strip()
            if not inst:
                continue
            
            # Split instruction into tokens
            inst_tokens = inst.split()
            
            for token in inst_tokens:
                # Check for BB start marker: <0x...>
                if token.startswith('<0x') and token.endswith('>'):
                    addr_str = token[1:-1]  # Remove < and >
                    addr_value = int(addr_str, 16)
                    
                    # First <0x...> is BB start, last one is BB end
                    if bb_start_addr is None:
                        bb_start_addr = addr_value
                        tokens.append('<addr_start>')
                        addr_values.append(addr_value)
                        addr_types.append(config.ADDR_TYPE_LABELS['start'])
                    else:
                        # This might be BB end (will be replaced if we see another)
                        bb_end_addr = addr_value
                
                # Check for labeled address: addr_code:0x... or addr_data:0x...
                elif ':' in token and token.startswith('addr_'):
                    addr_type, addr_str = token.split(':', 1)
                    addr_value = int(addr_str, 16)
                    
                    # Add token placeholder (Level 1: semantic embedding)
                    if addr_type == 'addr_code':
                        tokens.append('<addr_code>')
                        addr_types.append(config.ADDR_TYPE_LABELS['code'])
                    elif addr_type == 'addr_data':
                        tokens.append('<addr_data>')
                        addr_types.append(config.ADDR_TYPE_LABELS['data'])
                    else:
                        # Fallback for any other addr_* types
                        tokens.append(f'<{addr_type}>')
                        addr_types.append(config.ADDR_TYPE_LABELS.get(addr_type.replace('addr_', ''), config.ADDR_TYPE_LABELS['unknown']))
                    
                    # Store actual address value (Level 2: sin/cos encoding)
                    addr_values.append(addr_value)
                
                else:
                    # Regular token (instruction mnemonic, register, etc.)
                    tokens.append(token)
                    addr_values.append(0)  # Regular tokens get 0 address
                    addr_types.append(config.ADDR_TYPE_LABELS['unknown'])  # Regular tokens have no specific address type
            
            # Add <seq> separator after each instruction (except the last one)
            if inst_idx < len(instructions) - 1:
                tokens.append('<seq>')
                addr_values.append(0)
                addr_types.append(config.ADDR_TYPE_LABELS['unknown'])
        
        # Add BB end marker
        if bb_end_addr is not None:
            tokens.append('<addr_end>')
            addr_values.append(bb_end_addr)
            addr_types.append(config.ADDR_TYPE_LABELS['end'])
        else:
            tokens.append('<addr_end>')
            addr_values.append(0)
            addr_types.append(config.ADDR_TYPE_LABELS['end'])
        
        return tokens, addr_values, addr_types
    
    
    def _classify_edge_type(self, src_tokens: List[str], tgt_tokens: List[str]) -> str:
        """Classify edge type between source and target BB"""
        if '<addr_unknown>' in src_tokens or '<addr_unknown>' in tgt_tokens:
            return 'indirect'
        
        # Check last instruction of source BB
        src_str = ' '.join(src_tokens)
        
        if 'call' in src_str:
            return 'call'
        elif any(x in src_str for x in ['retn', 'ret ']):
            return 'return'
        elif any(x in src_str for x in ['je', 'jne', 'jmp', 'jg', 'jl', 'ja', 'jb']):
            return 'branch'
        else:
            return 'fallthrough'
    
    def _sinusoidal_encoding(self, address: int) -> np.ndarray:
        """Generate sinusoidal encoding for address"""
        encoding = np.zeros(config.ADDRESS_ENCODING_DIM)
        
        for i in range(config.ADDRESS_ENCODING_DIM):
            if i % 2 == 0:
                encoding[i] = np.sin(address / (config.ADDRESS_FREQ_BASE ** (i / config.ADDRESS_ENCODING_DIM)))
            else:
                encoding[i] = np.cos(address / (config.ADDRESS_FREQ_BASE ** (i / config.ADDRESS_ENCODING_DIM)))
        
        return encoding
    
    def __len__(self) -> int:
        return len(self.bb_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single BB pair with three-level embeddings using dual vocabularies"""
        pair = self.bb_pairs[idx]
        
        # Tokenize using BOTH vocabularies
        src_palmtree_ids, src_addr_ids = self._tokenize(pair['src_tokens'])
        tgt_palmtree_ids, tgt_addr_ids = self._tokenize(pair['tgt_tokens'])
        
        # Level 2: Create sin/cos address encodings for each token
        src_addr_encodings = []
        for addr_value in pair['src_addr_values']:
            src_addr_encodings.append(self._sinusoidal_encoding(addr_value))
        
        tgt_addr_encodings = []
        for addr_value in pair['tgt_addr_values']:
            tgt_addr_encodings.append(self._sinusoidal_encoding(addr_value))
        
        # Level 2: Also get address type IDs
        src_addr_type_ids = pair['src_addr_types']
        tgt_addr_type_ids = pair['tgt_addr_types']
        
        # Combine source and target with [SEP] token
        # Handle both WordVocab and dict
        if hasattr(self.palmtree_vocab, 'stoi'):
            sep_palmtree = self.palmtree_vocab.stoi.get('<eos>', self.palmtree_vocab.eos_index)
            pad_palmtree = self.palmtree_vocab.pad_index
        else:
            sep_palmtree = self.palmtree_vocab.get('[SEP]', 2)
            pad_palmtree = self.palmtree_vocab.get('[PAD]', 1)
        
        sep_addr = self.addr_vocab.unk_index
        
        combined_palmtree_ids = src_palmtree_ids + [sep_palmtree] + tgt_palmtree_ids
        combined_addr_ids = src_addr_ids + [sep_addr] + tgt_addr_ids
        combined_addr_encodings = src_addr_encodings + [np.zeros(config.ADDRESS_ENCODING_DIM)] + tgt_addr_encodings
        combined_addr_types = src_addr_type_ids + [config.ADDR_TYPE_LABELS['unknown']] + tgt_addr_type_ids
        
        # Truncate if too long
        if len(combined_palmtree_ids) > self.max_seq_length:
            combined_palmtree_ids = combined_palmtree_ids[:self.max_seq_length]
            combined_addr_ids = combined_addr_ids[:self.max_seq_length]
            combined_addr_encodings = combined_addr_encodings[:self.max_seq_length]
            combined_addr_types = combined_addr_types[:self.max_seq_length]
        
        # Pad if too short
        padding_length = self.max_seq_length - len(combined_palmtree_ids)
        pad_addr = self.addr_vocab.pad_index
        
        combined_palmtree_ids += [pad_palmtree] * padding_length
        combined_addr_ids += [pad_addr] * padding_length
        combined_addr_encodings += [np.zeros(config.ADDRESS_ENCODING_DIM)] * padding_length
        combined_addr_types += [config.ADDR_TYPE_LABELS['unknown']] * padding_length
        
        # Create attention mask
        attention_mask = [1] * (self.max_seq_length - padding_length) + [0] * padding_length
        
        # Edge type label
        edge_type_label = config.EDGE_TYPE_LABELS[pair['edge_type']]
        
        # Level 3 (positional embeddings) will be added by the model itself
        return {
            'input_ids': torch.tensor(combined_palmtree_ids, dtype=torch.long),  # Level 1: PalmTree vocab IDs
            'addr_ids': torch.tensor(combined_addr_ids, dtype=torch.long),  # Address vocab IDs (for identification)
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'address_encodings': torch.tensor(np.array(combined_addr_encodings), dtype=torch.float),  # Level 2: sin/cos address
            'addr_type_ids': torch.tensor(combined_addr_types, dtype=torch.long),  # Level 2: address type IDs
            'src_length': len(src_palmtree_ids),
            'tgt_length': len(tgt_palmtree_ids),
            'edge_type': torch.tensor(edge_type_label, dtype=torch.long),
        }


def create_dataloaders(bb_pairs_file: str, vocab_file: str, batch_size: int = config.BATCH_SIZE):
    """
    Create train/val/test dataloaders
    Returns both PalmTree vocab and Address vocab
    """
    dataset = BBPairDataset(bb_pairs_file, vocab_file)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * config.TRAIN_SPLIT)
    val_size = int(total_size * config.VAL_SPLIT)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, dataset.palmtree_vocab, dataset.addr_vocab
