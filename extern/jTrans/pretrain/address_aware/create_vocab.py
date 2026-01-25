"""
Generate vocabulary from address-aware training data

Uses the WordVocab class for fast vocabulary creation.
"""

import os
import sys
import re

# Import local vocab module
from vocab import WordVocab


def preprocess_line(line):
    """
    Remove all address information in parentheses from a line and normalize tokens.
    
    Token normalization rules:
    - All var(0xXX) -> "var" (single token for all stack variables)
    - address(0xADDR:pos1:pos2:pos3) -> "address" (remove position info)
    - daddr(0xADDR:pos1:pos2:pos3) -> "daddr" (remove position info)  
    - mov(0xADDR:pos1:pos2:pos3) -> "mov" (remove position info from opcodes)
    - arg_XXXX -> "arg" (IDA auto-generated argument names)
    - Raw hex addresses 0xXXXX -> filtered out (shouldn't be in vocab)
    - Filter out: jump table labels like (jpt_XXXX
    - Filter out: Intel hex format like 4000h)
    - Remove segment prefix: ds:dword_0 -> dword_0
    - Intel hex XXXXh) -> "imm" (treat as immediate)
    
    Examples:
        mov(0x401000:0.5:0.3:0.2) eax ebx -> mov eax ebx
        address(0x123:0.5:0.3:0.2) -> address
        var(0x10) -> var
        arg_55C8 -> arg
        0x12340 -> (filtered out)
        ds:dword_0 -> dword_0
    
    Returns cleaned tokens as a list.
    """
    # Remove all patterns like word(0xADDR:pos1:pos2:pos3) - address positions
    # This captures the full token including the closing paren
    cleaned = re.sub(r'(\w+)\(0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+\)', r'\1', line)
    
    # Normalize var(0xXX) to just "var" - handle both with and without closing paren
    # This catches: var(0x10) var(0x10 var(0xFFFFFFFFFFFFFDB4h)
    cleaned = re.sub(r'var\(0x[0-9a-fA-FhH]+\)?', 'var', cleaned)
    
    # Split by whitespace and tab
    tokens = [tok for tok in cleaned.replace('\t', ' ').split() if tok]
    
    # Additional token-level filtering and normalization
    normalized_tokens = []
    for tok in tokens:
        # Filter out jump table labels: (jpt_XXXX
        if tok.startswith('(jpt_'):
            continue
        
        # Filter out raw hex addresses: 0xXXXX (these shouldn't be in tokenized output)
        if re.match(r'^0x[0-9a-fA-F]+$', tok):
            continue
        
        # Normalize arg_XXXX (IDA auto-generated argument names) to just "arg"
        if re.match(r'^arg_[0-9A-Fa-f]+$', tok):
            normalized_tokens.append('arg')
            continue
        
        # Convert Intel hex format with closing paren to imm: XXXXh)
        if re.match(r'^[0-9A-Fa-f]+h\)$', tok):
            normalized_tokens.append('imm')
            continue
        
        # Remove segment prefixes: ds:dword_0 -> dword_0
        if re.match(r'^[cdefgs]s:', tok):
            tok = tok.split(':', 1)[1]  # Keep everything after the colon
        
        # Normalize .plt.plt to .plt (malformed double PLT)
        if tok == '.plt.plt':
            normalized_tokens.append('.plt')
            continue
        
        # Keep the token
        normalized_tokens.append(tok)
    
    return normalized_tokens


class PreprocessedFile:
    """
    Wrapper that preprocesses lines from a file by removing address info.
    Acts as an iterator that yields lists of tokens.
    """
    def __init__(self, file_handle):
        self.file_handle = file_handle
    
    def __iter__(self):
        for line in self.file_handle:
            yield preprocess_line(line)


def create_vocab(data_files, vocab_path, max_size=10000, min_freq=2, logger=None):
    """
    Create vocabulary from data files if it doesn't exist.
    
    Args:
        data_files: List of data file paths to build vocabulary from
        vocab_path: Path to save vocabulary
        max_size: Maximum vocabulary size
        min_freq: Minimum token frequency
        logger: Optional logger instance
    
    Returns:
        WordVocab instance
    """
    # Check if vocab already exists
    if os.path.exists(vocab_path):
        if logger:
            logger.info(f"Loading existing vocabulary from {vocab_path}")
        else:
            print(f"Loading existing vocabulary from {vocab_path}")
        return WordVocab.load_vocab(vocab_path)
    
    # Vocab doesn't exist, create it
    if logger:
        logger.info(f"Vocabulary file not found: {vocab_path}")
        logger.info("Creating vocabulary from data files...")
    else:
        print(f"Vocabulary file not found: {vocab_path}")
        print("Creating vocabulary from data files...")
    
    # Filter existing files
    files_to_use = [f for f in data_files if f and os.path.exists(f)]
    
    if not files_to_use:
        raise FileNotFoundError("No data files found to build vocabulary!")
    
    for fpath in files_to_use:
        if logger:
            logger.info(f"  Using: {fpath}")
        else:
            print(f"  Using: {fpath}")
    
    # Open all files and build vocabulary
    file_handles = [open(f, 'r', encoding='utf-8') for f in files_to_use]
    preprocessed = [PreprocessedFile(fh) for fh in file_handles]
    
    vocab = WordVocab(preprocessed, max_size=max_size, min_freq=min_freq)
    
    # Close file handles
    for fh in file_handles:
        fh.close()
    
    # Save vocabulary
    vocab.save_vocab(vocab_path)
    
    if logger:
        logger.info(f"Vocabulary created with {len(vocab)} tokens")
        logger.info(f"Vocabulary saved to: {vocab_path}")
    else:
        print(f"Vocabulary created with {len(vocab)} tokens")
        print(f"Vocabulary saved to: {vocab_path}")
    
    return vocab


if __name__ == "__main__":
    # Data files from /data/kun/jtransdata/
    train_dataset = "/data/kun/jtransdata/addr_pretrain.txt"
    
    vocab_path = "/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/vocab_addr.pkl"
    
    print("=" * 60)
    print("Creating Vocabulary from Address-Aware Training Data")
    print("=" * 60)
    print(f"Using WordVocab with max_size=10000, min_freq=50")
    print(f"Output: {vocab_path} (pickle format)")
    print()
    
    # Check if files exist
    files_to_check = [train_dataset]
    
    # Optionally include test dataset
    for fpath in files_to_check:
        if not os.path.exists(fpath):
            print(f"ERROR: File not found: {fpath}")
            sys.exit(1)
        # Get file size
        file_size = os.path.getsize(fpath) / (1024 * 1024)  # MB
        print(f"Found: {fpath} ({file_size:.1f} MB)")
    
    print()
    print("Building vocabulary (this may take a few minutes)...")
    print()
    
    # Open all files and pass to WordVocab with preprocessing 
    # Wrap each file handle with PreprocessedFile to remove address info
    file_handles = []
    preprocessed_files = []
    
    try:
        for fpath in files_to_check:
            fh = open(fpath, "r", encoding="utf-8")
            file_handles.append(fh)
            preprocessed_files.append(PreprocessedFile(fh))
         
        vocab = WordVocab(
            preprocessed_files,
            max_size=10000,
            min_freq=2
        )
        
    finally:
        # Close all file handles
        for fh in file_handles:
            fh.close()
    
    print()
    print("VOCAB SIZE:", len(vocab))
    print()
    
    # Save vocabulary
    vocab.save_vocab(vocab_path)
    print(f"Vocabulary saved to: {vocab_path}")
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print()
    print("You can now use this vocabulary in your training:")
    print("  from vocab import WordVocab")
    print("  vocab = WordVocab.load_vocab('vocab.pkl')")
    print()
