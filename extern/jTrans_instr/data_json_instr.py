"""
Data loader for instruction-level jTrans_instr model.
Compatible with finetune_instr.py interface.
"""

import json
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset


def load_paired_data_json_instr(func_blocks_path, ground_truth_path, opt=['O0', 'O1', 'O2', 'O3'], data_ratio=1.0):
    """
    Load function pairs from instruction-level JSON format.
    
    Returns:
    - functions: List of lists, each inner list contains function strings for different opts
    - func_emb_data: List of dicts with metadata for evaluation
    """
    # Load ground truth
    print(f'Loading ground truth from {ground_truth_path}...')
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    all_pairs = ground_truth['pairs']
    total_pairs = len(all_pairs)
    
    # Apply data_ratio
    if data_ratio < 1.0:
        if data_ratio <= 0.001:
            num_pairs = 500
            print(f'Quick test mode: using {num_pairs} pairs (data_ratio={data_ratio})')
        else:
            num_pairs = int(total_pairs * data_ratio)
            print(f'Using {num_pairs} / {total_pairs} pairs (data_ratio={data_ratio})')
        
        num_pairs = max(1, min(num_pairs, total_pairs))
        all_pairs = all_pairs[:num_pairs]
    else:
        print(f'Using all {total_pairs} pairs')
    
    # Collect needed function IDs
    needed_func_ids = set()
    for pair in all_pairs:
        needed_func_ids.add(str(pair['func_id1']))
        needed_func_ids.add(str(pair['func_id2']))
    
    print(f'Loading {len(needed_func_ids)} needed functions from {func_blocks_path}...')
    
    # Load function blocks
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
        binary = pair['binary']
        func_name = pair['func_name']
        key = (binary, func_name)
        
        if key not in func_mapping:
            func_mapping[key] = {}
        
        opt1, opt2 = pair['opt1'], pair['opt2']
        func_id1, func_id2 = pair['func_id1'], pair['func_id2']
        
        func_mapping[key][opt1] = func_id1
        func_mapping[key][opt2] = func_id2
    
    # Build final data structure
    functions = []
    func_emb_data = []
    
    for (binary, func_name), opt_dict in func_mapping.items():
        # Only include if we have all required optimization levels
        if all(o in opt_dict for o in opt):
            func_list = []
            for o in opt:
                func_id = str(opt_dict[o])
                if func_id in func_blocks:
                    func_list.append(func_blocks[func_id]['instructions'])
                else:
                    break
            
            if len(func_list) == len(opt):
                functions.append(func_list)
                func_emb_data.append({
                    'binary': binary,
                    'function_name': func_name,
                    'opts': opt
                })
    
    print(f'Final dataset: {len(functions)} functions with all {len(opt)} optimization levels')
    return functions, func_emb_data


class FunctionDataset_CL_Load_JSON_Instr(Dataset):
    """Dataset for instruction-level contrastive learning from JSON."""
    
    def __init__(self, funcs, tokenizer, maxlen=512):
        self.funcs = funcs
        self.tokenizer = tokenizer
        self.maxlen = maxlen
    
    def __len__(self):
        return len(self.funcs)
    
    def _tokenize_with_instruction_ids(self, func_str):
        """
        Tokenize and generate instruction_ids.
        Uses pretrain tokenizer, then assigns instruction index based on \t separator.
        """
        # Use pretrain tokenizer (already handles everything correctly)
        encoding = self.tokenizer(
            func_str,
            max_length=self.maxlen,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids)).squeeze(0)
        
        # Generate instruction_ids by counting \t in original string
        # Split by \t to get instructions
        instructions = func_str.split('\t')
        
        # Tokenize each instruction to find token boundaries
        instruction_token_counts = []
        for instr in instructions:
            if not instr.strip():
                continue
            # Count tokens for this instruction (without special tokens)
            instr_enc = self.tokenizer(instr, add_special_tokens=False)
            instruction_token_counts.append(len(instr_enc['input_ids']))
        
        # Build instruction_ids tensor
        instruction_ids = [0]  # [CLS] token -> instruction 0
        current_instr_idx = 0
        
        for count in instruction_token_counts:
            # Assign same instruction index to all tokens in this instruction
            instruction_ids.extend([current_instr_idx] * count)
            current_instr_idx += 1
            
            # Stop if we exceed maxlen (account for [CLS] and [SEP])
            if len(instruction_ids) >= self.maxlen - 1:
                break
        
        # Truncate and add [SEP]
        instruction_ids = instruction_ids[:self.maxlen - 1]
        instruction_ids.append(current_instr_idx)  # [SEP] token -> last instruction
        
        # Pad to maxlen
        while len(instruction_ids) < self.maxlen:
            instruction_ids.append(0)  # Padding -> instruction 0
        
        instruction_ids = torch.tensor(instruction_ids, dtype=torch.long)
        
        return input_ids, attention_mask, token_type_ids, instruction_ids
    
    def __getitem__(self, index):
        """Get triplet: (anchor, positive, negative)."""
        # Select anchor function
        anchor_idx = index
        anchor_funcs = self.funcs[anchor_idx]  # List of same function with different opts
        
        # Random select anchor and positive from same function
        if len(anchor_funcs) >= 2:
            anchor_opt_idx, pos_opt_idx = random.sample(range(len(anchor_funcs)), 2)
        else:
            anchor_opt_idx = pos_opt_idx = 0
        
        anchor_str = anchor_funcs[anchor_opt_idx]
        pos_str = anchor_funcs[pos_opt_idx]
        
        # Random select negative from different function
        while True:
            neg_idx = random.randint(0, len(self.funcs) - 1)
            if neg_idx != anchor_idx:
                break
        
        neg_funcs = self.funcs[neg_idx]
        neg_opt_idx = random.randint(0, len(neg_funcs) - 1)
        neg_str = neg_funcs[neg_opt_idx]
        
        # Tokenize all three
        anchor_ids, anchor_mask, anchor_seg, anchor_instr_ids = self._tokenize_with_instruction_ids(anchor_str)
        pos_ids, pos_mask, pos_seg, pos_instr_ids = self._tokenize_with_instruction_ids(pos_str)
        neg_ids, neg_mask, neg_seg, neg_instr_ids = self._tokenize_with_instruction_ids(neg_str)
        
        return (
            anchor_ids, pos_ids, neg_ids,
            anchor_mask, pos_mask, neg_mask,
            anchor_seg, pos_seg, neg_seg,
            anchor_instr_ids, pos_instr_ids, neg_instr_ids
        )
