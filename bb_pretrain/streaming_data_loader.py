"""
Streaming Data Loader for Large BB Pairs File
Does NOT load entire file into memory - reads line by line
"""
import os
import torch
from torch.utils.data import IterableDataset, DataLoader
import random
import re
from typing import Optional
import pickle

try:
    from . import config
    from .addr_vocab import AddrVocab
except ImportError:
    import config
    from addr_vocab import AddrVocab


class StreamingBBPairDataset(IterableDataset):
    """
    Streaming dataset that reads BB pairs line by line
    Suitable for very large files (e.g., 21GB all_bb_pairs.txt)
    """
    
    def __init__(self, bb_pairs_file: str, vocab_file: str, 
                 max_seq_length: int = config.MAX_SEQ_LENGTH,
                 shuffle_buffer_size: int = 10000,
                 epoch_size: Optional[int] = None):
        """
        Args:
            bb_pairs_file: Path to BB pairs file
            vocab_file: Path to PalmTree vocabulary
            max_seq_length: Maximum sequence length
            shuffle_buffer_size: Buffer size for shuffling (larger = more random)
            epoch_size: Number of samples per epoch (None = full file)
        """
        super().__init__()
        self.bb_pairs_file = bb_pairs_file
        self.max_seq_length = max_seq_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.epoch_size = epoch_size
        
        # Load vocabularies
        self.palmtree_vocab = self._load_vocab(vocab_file)
        self.addr_vocab = AddrVocab()
        
        # Get file size info
        self.file_size = os.path.getsize(bb_pairs_file)
        self.num_lines = self._count_lines()
        
        print(f"Streaming dataset initialized:")
        print(f"  File: {bb_pairs_file}")
        print(f"  Size: {self.file_size / (1024**3):.2f} GB")
        print(f"  Lines: {self.num_lines:,}")
        print(f"  Shuffle buffer: {shuffle_buffer_size:,}")
        print(f"  Epoch size: {epoch_size if epoch_size else 'Full file'}")
    
    def _load_vocab(self, vocab_file: str):
        """Load PalmTree vocabulary"""
        # Try pickle first
        if os.path.exists(vocab_file):
            try:
                with open(vocab_file, 'rb') as f:
                    vocab = pickle.load(f)
                return vocab
            except:
                pass
        
        # Try text format
        vocab = {'<pad>': 1, '<unk>': 0, '<eos>': 2, '<sos>': 3, '<mask>': 4}
        if os.path.exists(vocab_file + '.txt'):
            vocab_file = vocab_file + '.txt'
        
        if os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                for idx, line in enumerate(f, start=5):
                    token = line.strip()
                    if token:
                        vocab[token] = idx
        
        print(f"Loaded PalmTree vocab: {len(vocab)} tokens")
        return vocab
    
    def _count_lines(self):
        """Count total lines in file (for progress tracking)"""
        print("Counting lines in file...")
        count = 0
        with open(self.bb_pairs_file, 'r') as f:
            for _ in f:
                count += 1
        return count
    
    def _parse_bb_pair(self, line: str):
        """Parse a BB pair line"""
        try:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#') or '->' not in line:
                return None
            
            # Split by arrow: "BB1 -> BB2"
            parts = line.split(' -> ')
            if len(parts) != 2:
                return None
            
            bb1_str, bb2_str = parts[0].strip(), parts[1].strip()
            
            # Parse BBs (format: <0x401000> instructions <0x401003>)
            bb1 = self._parse_bb(bb1_str)
            bb2 = self._parse_bb(bb2_str)
            
            if bb1 is None or bb2 is None:
                return None
            
            # Infer edge type from instructions
            edge_type = self._infer_edge_type(bb1_str, bb2_str)
            
            return {
                'bb1': bb1,
                'bb2': bb2,
                'edge_type': edge_type
            }
        except:
            return None
    
    def _infer_edge_type(self, bb1_str: str, bb2_str: str):
        """Infer edge type from instruction strings"""
        bb1_lower = bb1_str.lower()
        
        if 'unknown' in bb2_str:
            return 'indirect'
        elif 'call' in bb1_lower:
            return 'call'
        elif 'ret' in bb1_lower or 'retn' in bb1_lower:
            return 'return'
        elif any(x in bb1_lower for x in ['je', 'jne', 'jmp', 'jg', 'jl', 'ja', 'jb']):
            return 'branch'
        else:
            return 'fallthrough'
    
    def _parse_bb(self, bb_str: str):
        """Parse a single BB string (format: <0x401000> instructions <0x401003>)"""
        try:
            # Extract start address (first <0x...>)
            start_match = re.search(r'<(0x[0-9a-fA-F]+)>', bb_str)
            if not start_match:
                return None
            start_addr = int(start_match.group(1), 16)
            
            # Extract end address (last <0x...>)
            end_matches = list(re.finditer(r'<(0x[0-9a-fA-F]+)>', bb_str))
            if len(end_matches) < 2:
                end_addr = start_addr  # Single instruction BB
            else:
                end_addr = int(end_matches[-1].group(1), 16)
            
            # Extract instructions (everything between first and last address markers)
            if len(end_matches) >= 2:
                start_pos = start_match.end()
                end_pos = end_matches[-1].start()
                instructions = bb_str[start_pos:end_pos].strip()
            else:
                instructions = bb_str[start_match.end():].strip()
            
            return {
                'start_addr': start_addr,
                'end_addr': end_addr,
                'instructions': instructions
            }
        except:
            return None
    
    def _tokenize_instructions(self, instructions: str):
        """Tokenize instructions into PalmTree tokens and address info"""
        tokens = []
        addr_types = []
        addr_values = []
        
        # Check vocab type
        has_stoi = hasattr(self.palmtree_vocab, 'stoi')
        if has_stoi:
            unk_idx = self.palmtree_vocab.unk_index
            vocab_dict = self.palmtree_vocab.stoi
        else:
            unk_idx = self.palmtree_vocab.get('<unk>', 0)
            vocab_dict = self.palmtree_vocab
        
        # Split by whitespace and process
        parts = instructions.split()
        
        for part in parts:
            # Check if it's an address
            if part.startswith('0x') or part.startswith('addr_'):
                # This is an address
                addr_value, addr_type = self._parse_address_token(part)
                
                # Get token ID from PalmTree vocab (might be <unk>)
                token_id = vocab_dict.get(part, unk_idx)
                tokens.append(token_id)
                addr_types.append(config.ADDR_TYPE_LABELS.get(addr_type, 5))  # 5 = unknown
                addr_values.append(addr_value)
            else:
                # Regular instruction token
                token_id = vocab_dict.get(part, unk_idx)
                tokens.append(token_id)
                addr_types.append(5)  # No address
                addr_values.append(0)  # No value
        
        return tokens, addr_types, addr_values
    
    def _parse_address_token(self, token: str):
        """Parse address token to get value and type"""
        # Extract type
        if 'addr_code' in token:
            addr_type = 'code'
        elif 'addr_data' in token:
            addr_type = 'data'
        elif 'addr_start' in token:
            addr_type = 'start'
        elif 'addr_end' in token:
            addr_type = 'end'
        elif 'addr_tgt' in token:
            addr_type = 'tgt'
        else:
            addr_type = 'unknown'
        
        # Extract value
        value_match = re.search(r'0x([0-9a-fA-F]+)', token)
        if value_match:
            addr_value = int(value_match.group(1), 16)
        else:
            addr_value = 0
        
        return addr_value, addr_type
    
    def _encode_address_value(self, addr_value: int):
        """Sinusoidal encoding of address value"""
        if addr_value == 0:
            return torch.zeros(config.ADDRESS_ENCODING_DIM)
        
        # Sinusoidal encoding (like positional encoding)
        dim = config.ADDRESS_ENCODING_DIM
        encoding = torch.zeros(dim)
        
        position = addr_value / 10000.0  # Scale down
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
        
        encoding[0::2] = torch.sin(position * div_term)
        encoding[1::2] = torch.cos(position * div_term)
        
        return encoding
    
    def _shuffle_buffer(self, buffer):
        """Shuffle buffer in place"""
        random.shuffle(buffer)
        return buffer
    
    def __iter__(self):
        """Iterate through BB pairs"""
        # Get worker info for distributed training
        worker_info = torch.utils.data.get_worker_info()
        
        # Open file
        with open(self.bb_pairs_file, 'r') as f:
            # Skip to worker's portion if using multiple workers
            if worker_info is not None:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers
                
                # Simple strategy: each worker reads every Nth line
                lines_per_worker = self.num_lines // num_workers
                start_line = worker_id * lines_per_worker
                
                # Skip to start line
                for _ in range(start_line):
                    next(f)
                
                max_lines = lines_per_worker
            else:
                max_lines = self.epoch_size if self.epoch_size else self.num_lines
            
            # Streaming with shuffle buffer
            buffer = []
            lines_read = 0
            
            for line in f:
                if lines_read >= max_lines:
                    break
                
                # Parse line
                pair_data = self._parse_bb_pair(line)
                if pair_data is None:
                    continue
                
                # Add to buffer
                buffer.append(pair_data)
                lines_read += 1
                
                # When buffer is full, shuffle and yield
                if len(buffer) >= self.shuffle_buffer_size:
                    buffer = self._shuffle_buffer(buffer)
                    
                    while len(buffer) > self.shuffle_buffer_size // 2:
                        item = buffer.pop(0)
                        processed = self._process_pair(item)
                        if processed is not None:
                            yield processed
            
            # Yield remaining buffer
            buffer = self._shuffle_buffer(buffer)
            for item in buffer:
                processed = self._process_pair(item)
                if processed is not None:
                    yield processed
    
    def _process_pair(self, pair_data):
        """Process a BB pair into model inputs"""
        try:
            bb1 = pair_data['bb1']
            bb2 = pair_data['bb2']
            
            # Tokenize both BBs
            tokens1, addr_types1, addr_values1 = self._tokenize_instructions(bb1['instructions'])
            tokens2, addr_types2, addr_values2 = self._tokenize_instructions(bb2['instructions'])
            
            # Add special tokens for BB boundaries
            # [CLS] <addr_start> BB1 <addr_end> <addr_start> BB2 <addr_end> [SEP]
            input_ids = [self.palmtree_vocab['<sos>']]
            addr_type_ids = [5]  # unknown for CLS
            address_encodings = [self._encode_address_value(0)]
            
            # BB1 start marker
            input_ids.append(self.palmtree_vocab.get('<addr_start>', self.palmtree_vocab['<unk>']))
            addr_type_ids.append(2)  # start
            address_encodings.append(self._encode_address_value(bb1['start_addr']))
            
            # BB1 instructions
            input_ids.extend(tokens1[:self.max_seq_length//2 - 4])
            addr_type_ids.extend(addr_types1[:self.max_seq_length//2 - 4])
            for val in addr_values1[:self.max_seq_length//2 - 4]:
                address_encodings.append(self._encode_address_value(val))
            
            # BB1 end marker
            input_ids.append(self.palmtree_vocab.get('<addr_end>', self.palmtree_vocab['<unk>']))
            addr_type_ids.append(3)  # end
            address_encodings.append(self._encode_address_value(bb1['end_addr']))
            
            # BB2 start marker
            input_ids.append(self.palmtree_vocab.get('<addr_start>', self.palmtree_vocab['<unk>']))
            addr_type_ids.append(2)  # start
            address_encodings.append(self._encode_address_value(bb2['start_addr']))
            
            # BB2 instructions
            input_ids.extend(tokens2[:self.max_seq_length//2 - 4])
            addr_type_ids.extend(addr_types2[:self.max_seq_length//2 - 4])
            for val in addr_values2[:self.max_seq_length//2 - 4]:
                address_encodings.append(self._encode_address_value(val))
            
            # BB2 end marker
            input_ids.append(self.palmtree_vocab.get('<addr_end>', self.palmtree_vocab['<unk>']))
            addr_type_ids.append(3)  # end
            address_encodings.append(self._encode_address_value(bb2['end_addr']))
            
            # EOS
            input_ids.append(self.palmtree_vocab['<eos>'])
            addr_type_ids.append(5)
            address_encodings.append(self._encode_address_value(0))
            
            # Truncate if too long
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
                addr_type_ids = addr_type_ids[:self.max_seq_length]
                address_encodings = address_encodings[:self.max_seq_length]
            
            # Pad to max length
            pad_len = self.max_seq_length - len(input_ids)
            if pad_len > 0:
                input_ids.extend([self.palmtree_vocab['<pad>']] * pad_len)
                addr_type_ids.extend([5] * pad_len)
                address_encodings.extend([self._encode_address_value(0)] * pad_len)
            
            # Create attention mask
            attention_mask = [1] * (self.max_seq_length - pad_len) + [0] * pad_len
            
            # Edge type
            edge_type = config.EDGE_TYPE_LABELS.get(pair_data['edge_type'], 0)
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'address_encodings': torch.stack(address_encodings),
                'addr_type_ids': torch.tensor(addr_type_ids, dtype=torch.long),
                'edge_type': torch.tensor(edge_type, dtype=torch.long),
            }
        except Exception as e:
            return None


def create_streaming_dataloader(bb_pairs_file: str, vocab_file: str, 
                                  batch_size: int = 8,
                                  num_workers: int = 2,
                                  shuffle_buffer_size: int = 10000,
                                  epoch_size: Optional[int] = None):
    """
    Create streaming dataloader for large BB pairs file
    
    Args:
        bb_pairs_file: Path to BB pairs file
        vocab_file: Path to vocabulary
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle_buffer_size: Size of shuffle buffer (larger = more random)
        epoch_size: Number of samples per epoch (None = full file)
    """
    dataset = StreamingBBPairDataset(
        bb_pairs_file=bb_pairs_file,
        vocab_file=vocab_file,
        shuffle_buffer_size=shuffle_buffer_size,
        epoch_size=epoch_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == '__main__':
    # Test streaming loader
    import numpy as np
    np.log = np.log  # Fix for sinusoidal encoding
    
    dataloader = create_streaming_dataloader(
        bb_pairs_file='../all_bb_pairs.txt',
        vocab_file='../pre-trained_model/palmtree/vocab',
        batch_size=4,
        num_workers=2,
        shuffle_buffer_size=1000,
        epoch_size=100  # Just 100 samples for testing
    )
    
    print("\nTesting streaming dataloader...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  address_encodings: {batch['address_encodings'].shape}")
        
        if batch_idx >= 2:
            break
    
    print("\nStreaming dataloader works!")
