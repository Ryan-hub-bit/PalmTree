"""
Data loader for JSON-based function datasets (baseline and address-aware).

Compatible with existing finetune.py and fasteval.py interfaces.
"""

import json
import random
import re
import torch
from pathlib import Path


def load_paired_data_json(func_blocks_path, ground_truth_path, opt=['O0', 'O1', 'O2', 'O3'], add_ebd=False, data_ratio=1.0):
    """
    Load function pairs from JSON format (baseline or address-aware).
    
    Returns format compatible with original load_paired_data:
    - functions: List of lists, each inner list contains function strings for different opts
    - func_emb_data: List of dicts with metadata for evaluation
    
    Args:
        func_blocks_path: Path to func_blocks.json or func_blocks_baseline.json
        ground_truth_path: Path to ground_truth.json or ground_truth_baseline.json
        opt: List of optimization levels to include
        add_ebd: Whether to add embedding metadata
        data_ratio: Ratio of data to use (0.0 to 1.0), default 1.0 for all data
    """
    # Load ground truth
    print(f'Loading ground truth from {ground_truth_path}...')
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    all_pairs = ground_truth['pairs']
    total_pairs = len(all_pairs)
    
    # Apply data_ratio to limit number of pairs BEFORE processing
    if data_ratio < 1.0:
        # For quick testing with address-aware (millions of pairs), cap at reasonable limit
        if data_ratio <= 0.001:
            num_pairs = 500  # Fixed cap for quick testing
            print(f'Quick test mode: using {num_pairs} pairs (data_ratio={data_ratio}, total available={total_pairs})')
        else:
            num_pairs = int(total_pairs * data_ratio)
            print(f'Using {num_pairs} / {total_pairs} pairs (data_ratio={data_ratio})')
        
        num_pairs = max(1, min(num_pairs, total_pairs))
        all_pairs = all_pairs[:num_pairs]
    else:
        print(f'Using all {total_pairs} pairs')
    
    # Collect only the function IDs we need
    needed_func_ids = set()
    print(f'Collecting needed function IDs from {len(all_pairs)} pairs...')
    for pair in all_pairs:
        # Handle both baseline format (binary_name, function_name) and address-aware format (binary, func_name)
        binary = pair.get('binary_name', pair.get('binary'))
        func_name = pair.get('function_name', pair.get('func_name'))
        
        # Baseline format: has O0, O1, O2, O3, Os fields directly
        # Address-aware format: has opt1, opt2, func_id1, func_id2
        if 'opt1' in pair and 'opt2' in pair:
            # Address-aware format
            needed_func_ids.add(str(pair['func_id1']))
            needed_func_ids.add(str(pair['func_id2']))
        else:
            # Baseline format - all optimization levels in one entry
            for opt_level in ['O0', 'O1', 'O2', 'O3', 'Os']:
                if opt_level in pair:
                    needed_func_ids.add(str(pair[opt_level]))
    
    print(f'Loading only {len(needed_func_ids)} needed functions from {func_blocks_path}...')
    
    # Load ONLY the needed function blocks
    func_blocks = {}
    with open(func_blocks_path, 'r') as f:
        all_blocks = json.load(f)
        for func_id in needed_func_ids:
            if func_id in all_blocks:
                func_blocks[func_id] = all_blocks[func_id]
    
    print(f'Loaded {len(func_blocks)} function blocks')
    
    # Build mapping: (binary, func_name) -> {opt: func_id}
    func_mapping = {}
    for pair in all_pairs:
        # Handle both baseline format (binary_name, function_name) and address-aware format (binary, func_name)
        binary = pair.get('binary_name', pair.get('binary'))
        func_name = pair.get('function_name', pair.get('func_name'))
        key = (binary, func_name)
        
        if key not in func_mapping:
            func_mapping[key] = {}
        
        # Baseline format: has O0, O1, O2, O3, Os fields directly
        # Address-aware format: has opt1, opt2, func_id1, func_id2
        if 'opt1' in pair and 'opt2' in pair:
            # Address-aware format
            opt1, opt2 = pair['opt1'], pair['opt2']
            func_id1, func_id2 = pair['func_id1'], pair['func_id2']
            func_mapping[key][opt1] = func_id1
            func_mapping[key][opt2] = func_id2
        else:
            # Baseline format - all optimization levels in one entry
            for opt_level in ['O0', 'O1', 'O2', 'O3', 'Os']:
                if opt_level in pair:
                    func_mapping[key][opt_level] = pair[opt_level]
    
    # Convert to original format
    functions = []
    func_emb_data = []
    
    for (binary, func_name), opt_dict in func_mapping.items():
        # Only include if we have functions for requested opts
        has_opts = [o for o in opt if o in opt_dict]
        
        if len(has_opts) < 2:  # Need at least 2 opts for pairing
            continue
        
        func_list = []
        
        if add_ebd:
            ebd_entry = {'proj': binary, 'funcname': func_name}
        
        for o in opt:
            if o in opt_dict:
                func_id = opt_dict[o]
                func_data = func_blocks[str(func_id)]  # JSON keys are strings
                
                # Handle both baseline format (tokens) and address-aware format (instructions)
                if 'tokens' in func_data:
                    func_str = func_data['tokens']
                elif 'instructions' in func_data:
                    func_str = func_data['instructions']
                else:
                    continue
                
                if add_ebd:
                    ebd_entry[o] = len(func_list)
                
                func_list.append(func_str)
        
        if len(func_list) >= 2:
            functions.append(func_list)
            
            if add_ebd:
                func_emb_data.append(ebd_entry)
    
    print(f'TOTAL {sum(len(f) for f in functions)} function variants across {len(functions)} unique functions')
    
    return functions, func_emb_data


class FunctionDataset_CL_JSON(torch.utils.data.Dataset):
    """
    Contrastive learning dataset for JSON-based function data.
    Compatible with original FunctionDataset_CL interface.
    """
    def __init__(self, tokenizer, func_blocks_path, ground_truth_path, 
                 opt=['O0', 'O1', 'O2', 'O3'], add_ebd=True):
        functions, ebds = load_paired_data_json(
            func_blocks_path, ground_truth_path, opt=opt, add_ebd=add_ebd
        )
        self.datas = functions
        self.ebds = ebds
        self.tokenizer = tokenizer
        self.opt = opt
    
    def __getitem__(self, idx):
        """Return (anchor, positive, negative) triplet."""
        pairs = self.datas[idx]
        
        # Select anchor and positive from same function
        pos = random.randint(0, len(pairs) - 1)
        pos2 = random.randint(0, len(pairs) - 1)
        
        # Select negative from different function
        neg_idx = random.randint(0, len(self.datas) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.datas) - 1)
        
        neg_pairs = self.datas[neg_idx]
        neg_pos = random.randint(0, len(neg_pairs) - 1)
        
        # Return anchor, positive, negative
        return pairs[pos], pairs[pos2], neg_pairs[neg_pos]
    
    def __len__(self):
        return len(self.datas)


class FunctionDataset_CL_Load_JSON(torch.utils.data.Dataset):
    """
    Pre-tokenized dataset version for JSON data.
    """
    def __init__(self, tokenizer, func_blocks_path, ground_truth_path,
                 opt=['O0', 'O1', 'O2', 'O3'], add_ebd=True, max_length=512, data_ratio=1.0):
        functions, ebds = load_paired_data_json(
            func_blocks_path, ground_truth_path, opt=opt, add_ebd=add_ebd, data_ratio=data_ratio
        )
        
        # Pre-tokenize all functions
        print("Pre-tokenizing functions...")
        self.tokenized_datas = []
        
        # Get vocab size for validation
        vocab_size = len(tokenizer)
        
        for func_list in functions:
            tokenized_list = []
            for func_str in func_list:
                # Create segment labels based on instruction boundaries (tabs)
                token_type_ids = self._create_segment_labels(func_str, tokenizer, max_length)
                
                encoded = tokenizer.encode_plus(
                    func_str,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Clip token IDs to vocab size to prevent CUDA errors
                input_ids = encoded['input_ids'].squeeze(0)
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                
                tokenized_list.append({
                    'input_ids': input_ids,
                    'attention_mask': encoded['attention_mask'].squeeze(0),
                    'token_type_ids': token_type_ids
                })
            self.tokenized_datas.append(tokenized_list)
        
        self.ebds = ebds
        self.opt = opt
        
        print(f"Pre-tokenized {len(self.tokenized_datas)} function groups")
    
    def _create_segment_labels(self, func_str, tokenizer, max_length):
        """
        Create segment labels (token_type_ids) for baseline model.
        Baseline uses simple 0/1 alternating segments or all zeros.
        
        Returns:
            torch.Tensor: segment labels matching tokenized sequence length
        """
        import torch
        
        # For baseline model, use simple segment labels (all 0s)
        # BERT's token_type_embeddings only has 2 types (0 and 1)
        segment_labels = [0] * max_length
        
        return torch.tensor(segment_labels, dtype=torch.long)
    
    def __getitem__(self, idx):
        """Return tokenized (anchor, positive, negative) triplet."""
        pairs = self.tokenized_datas[idx]
        
        # Select anchor and positive from same function
        pos = random.randint(0, len(pairs) - 1)
        pos2 = random.randint(0, len(pairs) - 1)
        
        # Select negative from different function
        neg_idx = random.randint(0, len(self.tokenized_datas) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.tokenized_datas) - 1)
        
        neg_pairs = self.tokenized_datas[neg_idx]
        neg_pos = random.randint(0, len(neg_pairs) - 1)
        
        anchor = pairs[pos]
        positive = pairs[pos2]
        negative = neg_pairs[neg_pos]
        
        return (
            anchor['input_ids'], positive['input_ids'], negative['input_ids'],
            anchor['attention_mask'], positive['attention_mask'], negative['attention_mask'],
            anchor['token_type_ids'], positive['token_type_ids'], negative['token_type_ids']
        )
    
    def __len__(self):
        return len(self.tokenized_datas)


class FunctionDataset_CL_AddressAware_JSON(torch.utils.data.Dataset):
    """
    Contrastive learning dataset for address-aware JSON-based function data.
    Parses hierarchical position information from address-aware tokens.
    
    Compatible with AddressAwareJTransForMLM model.
    """
    def __init__(self, tokenizer, func_blocks_path, ground_truth_path,
                 opt=['O0', 'O1', 'O2', 'O3'], add_ebd=True, max_length=512, data_ratio=1.0):
        functions, ebds = load_paired_data_json(
            func_blocks_path, ground_truth_path, opt=opt, add_ebd=add_ebd, data_ratio=data_ratio
        )
        
        # Regex patterns from address-aware pretrain dataloader
        self.addr_pattern = re.compile(r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.nested_addr_pattern = re.compile(r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.daddr_pattern = re.compile(r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)')
        self.var_pattern = re.compile(r'var\((0x[0-9a-fA-F]+)\)')
        
        # Pre-process all functions
        print("Pre-processing address-aware functions...")
        self.processed_datas = []
        
        for func_list in functions:
            processed_list = []
            for func_str in func_list:
                # Parse address-aware tokens
                processed = self._parse_address_aware_function(func_str, tokenizer, max_length)
                processed_list.append(processed)
            self.processed_datas.append(processed_list)
        
        self.ebds = ebds
        self.opt = opt
        self.tokenizer = tokenizer
        
        print(f"Pre-processed {len(self.processed_datas)} address-aware function groups")
    
    def _parse_instruction(self, inst_text):
        """
        Parse a single address-aware instruction.
        Format: opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 operand2 ...
        
        Returns:
            tokens: List of token strings
            positions: List of (binary_pos, function_pos, bb_pos) tuples
            var_offsets: List of var offset values (-1 for non-var tokens)
        """
        tokens = []
        positions = []
        var_offsets = []
        
        # Split instruction into space-separated parts
        parts = inst_text.strip().split()
        
        for part in parts:
            # Check for opcode(address) pattern
            match = self.addr_pattern.match(part)
            if match:
                opcode = match.group(1)
                binary_pos = float(match.group(3))
                function_pos = float(match.group(4))
                bb_pos = float(match.group(5))
                
                tokens.append(opcode)
                positions.append((binary_pos, function_pos, bb_pos))
                var_offsets.append(-1)
                continue
            
            # Check for nested address() pattern
            nested_match = self.nested_addr_pattern.match(part)
            if nested_match:
                binary_pos = float(nested_match.group(2))
                function_pos = float(nested_match.group(3))
                bb_pos = float(nested_match.group(4))
                
                tokens.append('address')
                positions.append((binary_pos, function_pos, bb_pos))
                var_offsets.append(-1)
                continue
            
            # Check for daddr() pattern
            daddr_match = self.daddr_pattern.match(part)
            if daddr_match:
                binary_pos = float(daddr_match.group(2))
                function_pos = float(daddr_match.group(3))
                bb_pos = float(daddr_match.group(4))
                
                tokens.append('daddr')
                positions.append((binary_pos, function_pos, bb_pos))
                var_offsets.append(-1)
                continue
            
            # Check for var() pattern
            var_match = self.var_pattern.match(part)
            if var_match:
                hex_offset = var_match.group(1)
                offset_val = int(hex_offset, 16)
                
                # Handle 64-bit negative offsets (two's complement)
                # Values > 0x7FFFFFFFFFFFFFFF are negative in two's complement
                if offset_val > 0x7FFFFFFFFFFFFFFF:
                    # Convert to signed 64-bit integer
                    offset_val = offset_val - 0x10000000000000000
                
                tokens.append('var')
                positions.append((-1.0, -1.0, -1.0))  # var tokens don't have positions
                var_offsets.append(offset_val)
                continue
            
            # Regular token (no address info)
            tokens.append(part)
            positions.append((-1.0, -1.0, -1.0))
            var_offsets.append(-1)
        
        return tokens, positions, var_offsets
    
    def _parse_address_aware_function(self, func_str, tokenizer, max_length):
        """
        Parse an entire address-aware function string.
        Format: inst1\tinst2\tinst3...
        
        Returns dict with:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - token_type_ids: Segment labels
            - binary_pos: Binary position embeddings
            - function_pos: Function position embeddings
            - bb_pos: Basic block position embeddings
            - var_offsets: Variable offset values
        """
        instructions = func_str.split('\t')
        
        all_tokens = []
        all_positions = []
        all_var_offsets = []
        all_segments = []
        
        # Add <sos> at beginning (segment 1) - MATCH PRETRAIN
        all_tokens.append('<sos>')
        all_positions.append((-1.0, -1.0, -1.0))
        all_var_offsets.append(-1)
        all_segments.append(1)  # SOS gets segment 1
        
        for inst_idx, inst_text in enumerate(instructions):
            inst_text = inst_text.strip()
            if not inst_text:
                continue
            
            tokens, positions, var_offsets = self._parse_instruction(inst_text)
            
            # All tokens in segment 1 - MATCH PRETRAIN (not per-instruction segments)
            all_tokens.extend(tokens)
            all_positions.extend(positions)
            all_var_offsets.extend(var_offsets)
            all_segments.extend([1] * len(tokens))
            
            # NO <eos> after each instruction - MATCH PRETRAIN
        
        # Add <eos> at the end ONLY (segment 1) - MATCH PRETRAIN
        all_tokens.append('<eos>')
        all_positions.append((-1.0, -1.0, -1.0))
        all_var_offsets.append(-1)
        all_segments.append(1)
        
        # Convert tokens to IDs using tokenizer's vocabulary
        token_ids = []
        unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 1  # Default to 1 (<unk>)
        for tok in all_tokens:
            # Use tokenizer's convert_tokens_to_ids method for proper handling
            token_id = tokenizer.convert_tokens_to_ids(tok)
            # Handle None return for unknown tokens
            if token_id is None:
                token_id = unk_token_id
            token_ids.append(token_id)
        
        # Truncate or pad to max_length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            all_positions = all_positions[:max_length]
            all_var_offsets = all_var_offsets[:max_length]
            all_segments = all_segments[:max_length]
        else:
            padding_len = max_length - len(token_ids)
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            token_ids += [pad_token_id] * padding_len
            all_positions += [(-1.0, -1.0, -1.0)] * padding_len
            all_var_offsets += [-1] * padding_len
            all_segments += [0] * padding_len  # Padding gets segment 0
        
        # Create attention mask
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        attention_mask = [1 if tid != pad_token_id else 0 for tid in token_ids]
        
        # Split positions into separate lists
        binary_pos = [p[0] for p in all_positions]
        function_pos = [p[1] for p in all_positions]
        bb_pos = [p[2] for p in all_positions]
        
        return {
            'input_ids': torch.LongTensor(token_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'token_type_ids': torch.LongTensor(all_segments),
            'binary_pos': torch.FloatTensor(binary_pos),
            'function_pos': torch.FloatTensor(function_pos),
            'bb_pos': torch.FloatTensor(bb_pos),
            'var_offsets': torch.LongTensor(all_var_offsets)
        }
    
    def __getitem__(self, idx):
        """
        Return address-aware (anchor, positive, negative) triplet.
        
        Returns tuple of:
            - input_ids (3 tensors)
            - attention_mask (3 tensors)
            - token_type_ids (3 tensors)
            - binary_pos (3 tensors)
            - function_pos (3 tensors)
            - bb_pos (3 tensors)
            - var_offsets (3 tensors)
        """
        pairs = self.processed_datas[idx]
        
        # Select anchor and positive from same function
        pos = random.randint(0, len(pairs) - 1)
        pos2 = random.randint(0, len(pairs) - 1)
        
        # Select negative from different function
        neg_idx = random.randint(0, len(self.processed_datas) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.processed_datas) - 1)
        
        neg_pairs = self.processed_datas[neg_idx]
        neg_pos = random.randint(0, len(neg_pairs) - 1)
        
        anchor = pairs[pos]
        positive = pairs[pos2]
        negative = neg_pairs[neg_pos]
        
        return (
            anchor['input_ids'], positive['input_ids'], negative['input_ids'],
            anchor['attention_mask'], positive['attention_mask'], negative['attention_mask'],
            anchor['token_type_ids'], positive['token_type_ids'], negative['token_type_ids'],
            anchor['binary_pos'], positive['binary_pos'], negative['binary_pos'],
            anchor['function_pos'], positive['function_pos'], negative['function_pos'],
            anchor['bb_pos'], positive['bb_pos'], negative['bb_pos'],
            anchor['var_offsets'], positive['var_offsets'], negative['var_offsets']
        )
    
    def __len__(self):
        return len(self.processed_datas)

