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
    
    Returns cleaned tokens as a list.
    """
    # Remove all patterns like (0xADDR:pos1:pos2:pos3)
    cleaned = re.sub(r'\(0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+\)', '', line)
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


if __name__ == "__main__":
    # Data files from /data/kun/dataset/
    train_cfg_dataset = "/data/kun/dataset/train_cfg.txt"
    train_dfg_dataset = "/data/kun/dataset/train_dfg.txt"
    val_cfg_dataset = "/data/kun/dataset/val_cfg.txt"
    val_dfg_dataset = "/data/kun/dataset/val_dfg.txt"
    test_cfg_dataset = "/data/kun/dataset/test_cfg.txt"
    test_dfg_dataset = "/data/kun/dataset/test_dfg.txt"
    
    vocab_path = "./vocab.pkl"
    
    print("=" * 60)
    print("Creating Vocabulary from Train + Val + Test Data")
    print("=" * 60)
    print(f"Using WordVocab with max_size=13000, min_freq=1")
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
