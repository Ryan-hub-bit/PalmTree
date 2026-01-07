"""
Data loader for JSON-based function datasets (baseline and address-aware).

Compatible with existing finetune.py and fasteval.py interfaces.
"""

import json
import random
import torch
from pathlib import Path


def load_paired_data_json(func_blocks_path, ground_truth_path, opt=['O0', 'O1', 'O2', 'O3'], add_ebd=False):
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
    """
    # Load data
    with open(func_blocks_path, 'r') as f:
        func_blocks = json.load(f)
    
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Build mapping: (binary, func_name) -> {opt: func_id}
    func_mapping = {}
    for pair in ground_truth['pairs']:
        key = (pair['binary'], pair['func_name'])
        
        if key not in func_mapping:
            func_mapping[key] = {}
        
        # Add both functions from the pair
        opt1, opt2 = pair['opt1'], pair['opt2']
        func_id1, func_id2 = pair['func_id1'], pair['func_id2']
        
        func_mapping[key][opt1] = func_id1
        func_mapping[key][opt2] = func_id2
    
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
                func_str = func_data['tokens']
                
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
                 opt=['O0', 'O1', 'O2', 'O3'], add_ebd=True, max_length=512):
        functions, ebds = load_paired_data_json(
            func_blocks_path, ground_truth_path, opt=opt, add_ebd=add_ebd
        )
        
        # Pre-tokenize all functions
        print("Pre-tokenizing functions...")
        self.tokenized_datas = []
        
        for func_list in functions:
            tokenized_list = []
            for func_str in func_list:
                encoded = tokenizer.encode_plus(
                    func_str,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                tokenized_list.append({
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0)
                })
            self.tokenized_datas.append(tokenized_list)
        
        self.ebds = ebds
        self.opt = opt
        
        print(f"Pre-tokenized {len(self.tokenized_datas)} function groups")
    
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
            anchor['attention_mask'], positive['attention_mask'], negative['attention_mask']
        )
    
    def __len__(self):
        return len(self.tokenized_datas)
