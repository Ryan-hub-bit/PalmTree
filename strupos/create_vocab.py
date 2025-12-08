"""
Generate vocabulary from all CFG and DFG data (train, val, test)

Uses the WordVocab class for fast vocabulary creation.
"""

import os
import sys
import re

# Import local vocab module
from vocab import WordVocab


def preprocess_line(line):
    """
    Remove all address information in parentheses from a line.
    
    Examples:
        mov(0x401000:0.5:0.3:0.2) eax ebx -> mov eax ebx
        address(0x123:0.5:0.3:0.2) -> address
        var(0x10028) -> var
    
    Returns cleaned tokens as a list.
    """
    # Remove all patterns like (0xADDR:pos1:pos2:pos3) - address positions
    cleaned = re.sub(r'\(0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+\)', '', line)
    # Remove all patterns like (0xADDR) - var offsets, e.g., var(0x10028) -> var
    cleaned = re.sub(r'\(0x[0-9a-fA-F]+\)', '', cleaned)
    # Split by whitespace and tab, filter out empty strings
    tokens = [tok for tok in cleaned.replace('\t', ' ').split() if tok]
    return tokens


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


def create_vocab(data_files, vocab_path, max_size=13000, min_freq=2, logger=None):
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
    # Data files from /work/kliu14/vardataset/
    train_cfg_dataset = "/work/kliu14/vardataset/train_cfg.txt"
    train_dfg_dataset = "/work/kliu14/vardataset/train_dfg.txt"
    val_cfg_dataset = "/work/kliu14/vardataset/val_cfg.txt"
    val_dfg_dataset = "/work/kliu14/vardataset/val_dfg.txt"
    test_cfg_dataset = "/work/kliu14/vardataset/test_cfg.txt"
    test_dfg_dataset = "/work/kliu14/vardataset/test_dfg.txt"
    
    vocab_path = "./vocab.pkl"
    
    print("=" * 60)
    print("Creating Vocabulary from Train + Val + Test Data")
    print("=" * 60)
    print(f"Using WordVocab with max_size=13000, min_freq=2")
    print(f"Output: {vocab_path} (pickle format)")
    print()
    
    # Check if files exist
    files_to_check = [
        train_cfg_dataset, train_dfg_dataset,
        val_cfg_dataset, val_dfg_dataset,
        test_cfg_dataset, test_dfg_dataset
    ]
    
    for fpath in files_to_check:
        if not os.path.exists(fpath):
            print(f"ERROR: File not found: {fpath}")
            sys.exit(1)
        print(f"Found: {fpath}")
    
    print()
    
    # Open all files and pass to WordVocab with preprocessing
    # Wrap each file handle with PreprocessedFile to remove address info
    with open(train_cfg_dataset, "r", encoding="utf-8") as f1, \
         open(train_dfg_dataset, "r", encoding="utf-8") as f2, \
         open(val_cfg_dataset, "r", encoding="utf-8") as f3, \
         open(val_dfg_dataset, "r", encoding="utf-8") as f4, \
         open(test_cfg_dataset, "r", encoding="utf-8") as f5, \
         open(test_dfg_dataset, "r", encoding="utf-8") as f6:
        
        # Wrap each file with preprocessing
        preprocessed_files = [
            PreprocessedFile(f1),
            PreprocessedFile(f2),
            PreprocessedFile(f3),
            PreprocessedFile(f4),
            PreprocessedFile(f5),
            PreprocessedFile(f6)
        ]
        
        vocab = WordVocab(
            preprocessed_files,
            max_size=13000,
            min_freq=1
        )
    
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
