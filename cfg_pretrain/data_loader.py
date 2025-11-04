"""
Data loader for CFG-based pretraining with multiple tasks:
1. Masked Language Modeling (MLM)
2. CFG Edge Prediction (predict successor basic block)
3. Address Value Prediction
"""

import os
import re
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from collections import Counter

try:
    from . import config
except ImportError:
    import config


class AddressTokenizer:
    """Tokenizer for address format: <type:hex:bin_norm:func_norm>"""
    
    def __init__(self):
        self.addr_types = {
            'addr_start': 0,
            'addr_end': 1,
            'addr_code': 2,
            'addr_data': 3,
            'addr_unknown': 4
        }
        self.special_markers = {
            -3.0: 'DATA_MARKER',
            -4.0: 'UNKNOWN_MARKER',
            -5.0: 'BINARY_UNKNOWN_MARKER',
            -6.0: 'FUNCTION_UNKNOWN_MARKER'
        }
    
    def parse_address(self, addr_str: str) -> Dict:
        """
        Parse address string like <addr_start:0x402118:0.269055:0.5>
        Returns dict with type, hex, binary_norm, function_norm
        """
        # Remove angle brackets
        addr_str = addr_str.strip('<>')
        parts = addr_str.split(':')
        
        if len(parts) != 4:
            return None
        
        addr_type, hex_addr, bin_norm, func_norm = parts
        
        try:
            return {
                'type': addr_type,
                'type_id': self.addr_types.get(addr_type, 4),  # 4 for unknown
                'hex': hex_addr,
                'binary_norm': float(bin_norm),
                'function_norm': float(func_norm),
                'is_special': float(func_norm) < 0  # Negative values are special markers
            }
        except ValueError:
            return None
    
    def extract_addresses(self, text: str) -> List[Dict]:
        """Extract all addresses from instruction text"""
        pattern = r'<addr_[^>]+>'
        matches = re.findall(pattern, text)
        addresses = []
        for match in matches:
            parsed = self.parse_address(match)
            if parsed:
                addresses.append(parsed)
        return addresses


class CFGPretrainDataset(Dataset):
    """
    Dataset for CFG-based pretraining with multiple tasks
    Each sample contains a pair of basic blocks (source -> target)
    """
    
    def __init__(
        self,
        data_file: str,
        vocab_file: str,
        max_seq_length: int = config.MAX_SEQ_LENGTH,
        mlm_probability: float = config.MLM_PROBABILITY,
        max_pairs: Optional[int] = None
    ):
        """
        Args:
            data_file: Path to BB pairs file
            vocab_file: Path to vocabulary file
            max_seq_length: Maximum sequence length
            mlm_probability: Probability of masking tokens for MLM
            max_pairs: Maximum pairs to load (None for all)
        """
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        
        # Load vocabulary
        self.vocab = self._load_vocab(vocab_file)
        self.vocab_size = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.mask_token = '<mask>'
        self.seq_token = '<seq>'  # Separator between instructions
        
        self.pad_id = self.vocab.get(self.pad_token, 0)
        self.mask_id = self.vocab.get(self.mask_token, 1)
        self.seq_id = self.vocab.get(self.seq_token, 2)
        
        # Address tokenizer
        self.addr_tokenizer = AddressTokenizer()
        
        # Load data
        self.pairs = self._load_pairs(data_file, max_pairs)
        
        print(f"Loaded {len(self.pairs)} BB pairs")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Load vocabulary from file"""
        vocab = {}
        
        # Try pickle format first (handles both dict and WordVocab)
        try:
            import pickle
            with open(vocab_file, 'rb') as f:
                loaded = pickle.load(f)
            
            # If it's a dict, use it directly
            if isinstance(loaded, dict):
                return loaded
            
            # If it's a WordVocab object (has stoi attribute), use stoi
            if hasattr(loaded, 'stoi') and isinstance(loaded.stoi, dict):
                return loaded.stoi
        except Exception as e:
            print(f"Pickle load attempt failed: {e}")
        
        # Try text format
        try:
            with open(vocab_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        token, idx = parts[0], int(parts[1])
                        vocab[token] = idx
            return vocab
        except Exception as e:
            print(f"Text load attempt failed: {e}")
        
        raise ValueError(f"Could not load vocabulary from {vocab_file}")
    
    def _load_pairs(self, data_file: str, max_pairs: Optional[int]) -> List[Tuple[str, str]]:
        """
        Load BB pairs from file
        Each line: <addr_start...> inst1 inst2 ... <addr_end...> <addr_start...> inst3 ... <addr_end...>
        Split by <addr_end><addr_start> pattern
        """
        pairs = []
        
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split into two BBs
                # Pattern: <addr_end:...> <addr_start:...>
                split_pattern = r'(<addr_end:[^>]+>)\s+(<addr_start:[^>]+>)'
                matches = list(re.finditer(split_pattern, line))
                
                if len(matches) >= 1:
                    # Split at the first transition
                    match = matches[0]
                    split_pos = match.start(2)  # Start of second addr_start
                    
                    source_bb = line[:match.end(1)].strip()
                    target_bb = line[split_pos:].strip()
                    
                    if source_bb and target_bb:
                        pairs.append((source_bb, target_bb))
                
                if max_pairs and len(pairs) >= max_pairs:
                    break
        
        return pairs
    
    def _tokenize_bb(self, bb_text: str) -> Tuple[List[int], List[Dict], List[Tuple[float, float]], Tuple[float, float], Tuple[float, float]]:
        """
        Tokenize basic block text into token IDs with address position info
        Returns: (token_ids, address_info, position_embeddings, start_position, end_position)
        
        position_embeddings: List of (binary_norm, function_norm) for each token
        ONLY addr_* tokens get real positions, other tokens get (0.0, 0.0)
        start_position: (binary_norm, function_norm) from addr_start
        end_position: (binary_norm, function_norm) from addr_end
        
        Instructions are separated by [SEQ] tokens
        """
        # Extract all addresses
        addresses = self.addr_tokenizer.extract_addresses(bb_text)
        
        # Find start and end positions from addresses
        start_position = (0.0, 0.0)
        end_position = (0.0, 0.0)
        
        for addr in addresses:
            if addr['type'] == 'addr_start':
                start_position = (addr['binary_norm'], addr['function_norm'])
            elif addr['type'] == 'addr_end':
                end_position = (addr['binary_norm'], addr['function_norm'])
        
        # Split by tabs to get individual instructions
        instructions = bb_text.split('\t')
        
        token_ids = []
        position_info = []
        
        addr_pattern = r'<addr_[^>]+>'
        
        for instr_idx, instruction in enumerate(instructions):
            instruction = instruction.strip()
            if not instruction:
                continue
            
            # Split instruction by address tags
            parts = re.split(f'({addr_pattern})', instruction)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                # Check if this part is an address tag
                if re.match(addr_pattern, part):
                    # This is an address - parse it
                    parsed = self.addr_tokenizer.parse_address(part)
                    if parsed:
                        # Use the address type as the token (addr_start, addr_end, addr_code, addr_data)
                        token_ids.append(self.vocab.get(parsed['type'], self.vocab.get('<unk>', 0)))
                        position_info.append((parsed['binary_norm'], parsed['function_norm']))
                    else:
                        # Failed to parse, use <unk>
                        token_ids.append(self.vocab.get('<unk>', 0))
                        position_info.append((0.0, 0.0))
                else:
                    # Regular instruction tokens - no address position
                    tokens = part.split()
                    for token in tokens:
                        token_id = self.vocab.get(token, self.vocab.get('<unk>', 0))
                        token_ids.append(token_id)
                        position_info.append((0.0, 0.0))  # No address position for regular tokens
            
            # Add [SEQ] separator between instructions (but not after the last one)
            if instr_idx < len(instructions) - 1:
                token_ids.append(self.seq_id)
                position_info.append((0.0, 0.0))
        
        return token_ids, addresses, position_info, start_position, end_position
    
    def _apply_mlm(self, token_ids: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """
        Apply masked language modeling
        Returns: (masked_token_ids, labels, mask_positions)
        
        Can focus on address tokens only by checking token names
        """
        masked_ids = token_ids.copy()
        labels = [-100] * len(token_ids)  # -100 is ignored in loss
        mask_positions = []
        
        # Get address token IDs
        addr_token_ids = {
            self.vocab.get('addr_start', -1),
            self.vocab.get('addr_end', -1),
            self.vocab.get('addr_code', -1),
            self.vocab.get('addr_data', -1),
        }
        addr_token_ids.discard(-1)  # Remove if not found in vocab
        
        for i in range(len(token_ids)):
            # Skip padding
            if token_ids[i] == self.pad_id:
                continue
            
            # Option 1: Mask ALL tokens (original behavior)
            # should_mask = True
            
            # Option 2: Mask ONLY address tokens (focus on addresses)
            should_mask = token_ids[i] in addr_token_ids
            
            # Option 3: Mask address tokens with HIGHER probability
            # if token_ids[i] in addr_token_ids:
            #     should_mask = random.random() < (self.mlm_probability * 2)  # 2x more likely
            # else:
            #     should_mask = random.random() < self.mlm_probability
            
            if should_mask and random.random() < self.mlm_probability:
                labels[i] = token_ids[i]  # Original token as label
                mask_positions.append(i)
                
                rand = random.random()
                if rand < config.MLM_MASK_TOKEN_PROB:
                    masked_ids[i] = self.mask_id
                elif rand < config.MLM_MASK_TOKEN_PROB + config.MLM_RANDOM_TOKEN_PROB:
                    masked_ids[i] = random.randint(0, self.vocab_size - 1)
                # else: keep original (UNCHANGED_PROB)
        
        return masked_ids, labels, mask_positions
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample with 3-level embeddings:
        1. Semantic: token IDs for instruction meaning
        2. Address position: binary_norm and function_norm for position in binary/function
        3. Sequence position: position in the instruction sequence (handled by model)
        
        Token structure: source_tokens target_tokens
        - addr_start and addr_end tokens already mark BB boundaries
        - Address tokens (addr_start, addr_end, addr_code, addr_data) get positions
        - Regular tokens get (0.0, 0.0)
        
        Negative pair sampling: with probability config.NEGATIVE_PAIR_PROB, replace target BB with a random BB (not the true successor) and set cfg_label=0.
        """
        source_bb, target_bb = self.pairs[idx]
        
        # Decide if this sample will be negative
        if random.random() < getattr(config, 'NEGATIVE_PAIR_PROB', 0.5):
            # Sample a random target BB (not the true successor)
            neg_idx = random.randrange(len(self.pairs))
            while neg_idx == idx:
                neg_idx = random.randrange(len(self.pairs))
            _, negative_target_bb = self.pairs[neg_idx]
            target_bb = negative_target_bb
            cfg_label = 0
        else:
            cfg_label = 1
        
        # Tokenize both BBs with position information
        source_ids, source_addrs, source_positions, source_start_pos, source_end_pos = self._tokenize_bb(source_bb)
        target_ids, target_addrs, target_positions, target_start_pos, target_end_pos = self._tokenize_bb(target_bb)
        
        # Combine: source target (addr_start and addr_end are already in the token sequences)
        combined_ids = source_ids + target_ids
        
        # Combine position embeddings (binary_norm, function_norm)
        combined_positions = source_positions + target_positions
        
        # Truncate if too long
        if len(combined_ids) > self.max_seq_length:
            combined_ids = combined_ids[:self.max_seq_length]
            combined_positions = combined_positions[:self.max_seq_length]
        
        # Apply MLM
        masked_ids, mlm_labels, mask_positions = self._apply_mlm(combined_ids)
        
        # Padding
        seq_length = len(masked_ids)
        padding_length = self.max_seq_length - seq_length
        
        masked_ids += [self.pad_id] * padding_length
        mlm_labels += [-100] * padding_length
        attention_mask = [1] * seq_length + [0] * padding_length
        combined_positions += [(0.0, 0.0)] * padding_length  # Pad positions
        
        # Segment IDs (0 for source, 1 for target)
        segment_ids = [0] * len(source_ids) + [1] * len(target_ids)
        segment_ids += [0] * padding_length
        segment_ids = segment_ids[:self.max_seq_length]
        
        # Sequence positions (0 to seq_length-1, then padding with 0)
        sequence_positions = list(range(seq_length)) + [0] * padding_length
        
        # Split position embeddings into separate arrays
        binary_positions = [pos[0] for pos in combined_positions]
        function_positions = [pos[1] for pos in combined_positions]
        
        # Address features (aggregated) for auxiliary tasks
        source_addr_features = self._aggregate_address_features(source_addrs)
        target_addr_features = self._aggregate_address_features(target_addrs)
        
        return {
            # Level 1: Semantic embedding (token IDs)
            'input_ids': torch.tensor(masked_ids, dtype=torch.long),
            
            # Level 2: Address position embedding (global/local position)
            'binary_positions': torch.tensor(binary_positions, dtype=torch.float),
            'function_positions': torch.tensor(function_positions, dtype=torch.float),
            
            # Level 3: Sequence position (handled by positional encoding in model)
            'sequence_positions': torch.tensor(sequence_positions, dtype=torch.long),
            
            # Other features
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
            
            # Task labels
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
            'cfg_label': torch.tensor(cfg_label, dtype=torch.long),  # 1 = valid CFG edge, 0 = negative pair
            
            # Aggregated address features for auxiliary tasks
            'source_addr_features': torch.tensor(source_addr_features, dtype=torch.float),
            'target_addr_features': torch.tensor(target_addr_features, dtype=torch.float),
        }
    
    def _aggregate_address_features(self, addresses: List[Dict]) -> List[float]:
        """
        Aggregate address information into feature vector
        Returns: [avg_binary_norm, avg_function_norm, num_code, num_data, num_special]
        """
        if not addresses:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        bin_norms = []
        func_norms = []
        num_code = 0
        num_data = 0
        num_special = 0
        
        for addr in addresses:
            bin_norms.append(addr['binary_norm'])
            func_norms.append(addr['function_norm'])
            
            if addr['type'] == 'addr_code':
                num_code += 1
            elif addr['type'] == 'addr_data':
                num_data += 1
            
            if addr['is_special']:
                num_special += 1
        
        return [
            np.mean(bin_norms) if bin_norms else 0.0,
            np.mean(func_norms) if func_norms else 0.0,
            float(num_code),
            float(num_data),
            float(num_special)
        ]


def create_dataloader(
    data_file: str,
    vocab_file: str,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 4,
    max_pairs: Optional[int] = None
) -> DataLoader:
    """Create DataLoader for CFG pretraining"""
    
    dataset = CFGPretrainDataset(
        data_file=data_file,
        vocab_file=vocab_file,
        max_seq_length=config.MAX_SEQ_LENGTH,
        mlm_probability=config.MLM_PROBABILITY,
        max_pairs=max_pairs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == '__main__':
    # Test the data loader
    print("Testing CFG Pretrain DataLoader...")
    
    data_file = '../bb_pairs_output/atilibusb.so_bb_pairs.txt'
    vocab_file = '../pre-trained_model/palmtree/vocab'
    
    dataloader = create_dataloader(
        data_file=data_file,
        vocab_file=vocab_file,
        batch_size=4,
        max_pairs=100
    )
    
    print(f"\nDataLoader created with {len(dataloader)} batches")
    
    # Test one batch
    for batch in dataloader:
        print("\nBatch keys:", batch.keys())
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention mask shape:", batch['attention_mask'].shape)
        print("MLM labels shape:", batch['mlm_labels'].shape)
        print("Source addr features shape:", batch['source_addr_features'].shape)
        print("Target addr features shape:", batch['target_addr_features'].shape)
        
        # Show some MLM examples
        for i in range(min(2, batch['input_ids'].size(0))):
            masked_positions = (batch['mlm_labels'][i] != -100).nonzero(as_tuple=True)[0]
            print(f"\nSample {i}: {len(masked_positions)} masked tokens")
        
        break
    
    print("\n✓ DataLoader test passed!")
