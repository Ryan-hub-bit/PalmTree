#!/usr/bin/env python3
"""
Diagnostic script to identify why address-aware model performs worse than baseline.
"""

import json
import sys
import pickle
from pathlib import Path
from transformers import BertTokenizer

def check_func_blocks_format(path, model_type):
    """Check the structure of function blocks."""
    print(f"\n{'='*60}")
    print(f"Checking {model_type} function blocks: {path}")
    print(f"{'='*60}")
    
    try:
        with open(path, 'r') as f:
            func_blocks = json.load(f)
        
        print(f"Total functions: {len(func_blocks)}")
        
        # Sample first 3 functions
        keys = list(func_blocks.keys())[:3]
        for key in keys:
            func_data = func_blocks[key]
            print(f"\nFunction ID: {key}")
            print(f"  Keys: {list(func_data.keys())}")
            
            # Get function string
            func_str = func_data.get('instructions', func_data.get('tokens', ''))
            print(f"  Length: {len(func_str)} characters")
            print(f"  First 150 chars: {func_str[:150]}")
            
    except Exception as e:
        print(f"ERROR: {e}")

def check_pool_query_files(pool_path, query_path):
    """Check pool and query file structure."""
    print(f"\n{'='*60}")
    print(f"Checking pool and query files")
    print(f"{'='*60}")
    
    try:
        with open(pool_path, 'r') as f:
            pool_data = json.load(f)
        
        with open(query_path, 'r') as f:
            query_data = json.load(f)
        
        print(f"\nPool size: {len(pool_data)}")
        print(f"Query size: {len(query_data)}")
        
        # Sample first entry
        if pool_data:
            print(f"\nSample pool entry keys: {list(pool_data[0].keys())}")
            print(f"Sample pool entry: {pool_data[0]}")
        
        if query_data:
            print(f"\nSample query entry keys: {list(query_data[0].keys())}")
            print(f"Sample query entry: {query_data[0]}")
            
    except Exception as e:
        print(f"ERROR: {e}")

def check_tokenizer_vocab(baseline_path, addressaware_path):
    """Compare tokenizer vocabularies."""
    print(f"\n{'='*60}")
    print(f"Checking tokenizer vocabularies")
    print(f"{'='*60}")
    
    try:
        baseline_tokenizer = BertTokenizer.from_pretrained(baseline_path)
        print(f"\nBaseline vocab size: {len(baseline_tokenizer)}")
        print(f"Sample baseline tokens: {list(baseline_tokenizer.vocab.keys())[:20]}")
        
        # Check for address-aware tokenizer
        addressaware_tokenizer = BertTokenizer.from_pretrained(addressaware_path)
        print(f"\nAddress-aware vocab size: {len(addressaware_tokenizer)}")
        print(f"Sample address-aware tokens: {list(addressaware_tokenizer.vocab.keys())[:20]}")
        
        # Check for special tokens
        print(f"\nAddress-aware special tokens:")
        if 'address' in addressaware_tokenizer.vocab:
            print(f"  'address': {addressaware_tokenizer.vocab['address']}")
        if 'daddr' in addressaware_tokenizer.vocab:
            print(f"  'daddr': {addressaware_tokenizer.vocab['daddr']}")
        if 'var' in addressaware_tokenizer.vocab:
            print(f"  'var': {addressaware_tokenizer.vocab['var']}")
            
        # Check vocab_addr.pkl
        vocab_addr_path = Path(addressaware_path) / 'vocab_addr.pkl'
        if vocab_addr_path.exists():
            sys.path.insert(0, str(Path(addressaware_path).resolve()))
            with open(vocab_addr_path, 'rb') as f:
                vocab_obj = pickle.load(f)
            print(f"\nVocab object type: {type(vocab_obj)}")
            if hasattr(vocab_obj, 'stoi'):
                print(f"Vocab stoi size: {len(vocab_obj.stoi)}")
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def check_model_config(baseline_path, addressaware_path):
    """Compare model configurations."""
    print(f"\n{'='*60}")
    print(f"Checking model configurations")
    print(f"{'='*60}")
    
    try:
        baseline_config_path = Path(baseline_path) / 'config.json'
        with open(baseline_config_path, 'r') as f:
            baseline_config = json.load(f)
        print(f"\nBaseline config:")
        for key in ['vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads']:
            print(f"  {key}: {baseline_config.get(key, 'N/A')}")
        
        addressaware_config_path = Path(addressaware_path) / 'config.json'
        with open(addressaware_config_path, 'r') as f:
            addressaware_config = json.load(f)
        print(f"\nAddress-aware config:")
        for key in ['vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads']:
            print(f"  {key}: {addressaware_config.get(key, 'N/A')}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == '__main__':
    # Paths
    DATA_DIR = "/data/kun/jtransdata"
    
    # Function blocks
    FUNC_BLOCKS_BASELINE = f"{DATA_DIR}/func_blocks_baseline.json"
    FUNC_BLOCKS_ADDR = f"{DATA_DIR}/func_blocks_addr.json"
    
    # Pool and query
    POOL_FILE = f"{DATA_DIR}/fair_pools/pool_100_O0_vs_O3.json"
    QUERY_FILE = f"{DATA_DIR}/fair_pools/query_pool_100_O0_vs_O3.json"
    
    # Tokenizers
    BASELINE_TOKENIZER = "/home/kun/Document/AAE/extern/jTrans/pretrain/baseline"
    ADDRESSAWARE_TOKENIZER = "/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware"
    
    # Models
    BASELINE_MODEL = "/home/kun/Document/AAE/output/jtrans/baseline_finetune/finetune_epoch_5"
    ADDRESSAWARE_MODEL = "/home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_4"
    
    # Run diagnostics
    check_func_blocks_format(FUNC_BLOCKS_BASELINE, "BASELINE")
    check_func_blocks_format(FUNC_BLOCKS_ADDR, "ADDRESS-AWARE")
    check_pool_query_files(POOL_FILE, QUERY_FILE)
    check_tokenizer_vocab(BASELINE_TOKENIZER, ADDRESSAWARE_TOKENIZER)
    check_model_config(BASELINE_MODEL, ADDRESSAWARE_MODEL)
    
    print(f"\n{'='*60}")
    print("Diagnostic complete!")
    print(f"{'='*60}\n")
