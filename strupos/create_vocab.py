"""
Generate vocabulary from all CFG and DFG data (train, val, test)

NOTE: Token normalization (stack_var, imm_small, imm_mid, imm_large) 
is now done during CFG/DFG generation in IDA Pro.
This script just collects the already-normalized tokens.
"""

import os
import re
from collections import Counter
from tqdm import tqdm


def parse_instruction(instruction):
    """
    Parse instruction and extract tokens, removing address info in parentheses.
    
    Format: opcode(addr:pos1:pos2:pos3) operand1 operand2 ...
    Output: [opcode, operand1, operand2, ...]
    
    Also handles operands with addresses like: address(0x123:0.5:0.3:0.2)
    Output: [address] (without the parentheses content)
    
    NOTE: Normalization (stack_var, imm_small, etc.) is done during data generation,
    so tokens here are already normalized.
    """
    # Pattern: opcode(hex_addr:pos1:pos2:pos3) operands...
    pattern = r'^([a-zA-Z0-9._]+)\((0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+)\)\s*(.*?)$'
    match = re.match(pattern, instruction.strip())
    
    if not match:
        # Fallback: treat entire instruction as single token
        if instruction.strip():
            return [instruction.strip()]
        return []
    
    opcode = match.group(1)
    operands_str = match.group(3)
    
    tokens = [opcode]
    
    if operands_str:
        # Split operands by whitespace
        operands = [op.strip() for op in operands_str.split() if op.strip()]
        
        # For each operand, remove address info if present
        # e.g., address(0x123:0.5:0.3:0.2) -> address
        for operand in operands:
            # Remove anything in parentheses (address positions)
            clean_operand = re.sub(r'\(0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+\)', '', operand)
            if clean_operand:  # Only add if not empty after cleaning
                tokens.append(clean_operand)
    
    return tokens


def collect_tokens_from_file(file_path):
    """Collect all tokens from a single file"""
    tokens = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by tab to get instructions
                instructions = line.split('\t')
                for inst in instructions:
                    inst_tokens = parse_instruction(inst)
                    tokens.extend(inst_tokens)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return tokens


def create_vocab_from_files(file_paths, output_file, min_freq=2, max_vocab_size=50000):
    """
    Create vocabulary from specific text files.
    
    Args:
        file_paths: List of file paths
        output_file: Path to save vocab.txt
        min_freq: Minimum frequency for a token to be included
        max_vocab_size: Maximum vocabulary size
    """
    token_counter = Counter()
    
    print("Collecting tokens from all data files...")
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Processing: {file_path}")
        tokens = collect_tokens_from_file(file_path)
        token_counter.update(tokens)
    
    print(f"\nTotal unique tokens: {len(token_counter):,}")
    print(f"Total token occurrences: {sum(token_counter.values()):,}")
    
    # Filter by minimum frequency
    filtered_tokens = {token: count for token, count in token_counter.items() 
                      if count >= min_freq}
    print(f"After filtering (min_freq={min_freq}): {len(filtered_tokens):,} tokens")
    
    # Create vocabulary with special tokens
    vocab = []
    
    # Special tokens (must be first)
    special_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<sos>", "<eos>",
        # Normalized tokens for immediates and stack variables
        "stack_var", "imm_small", "imm_mid", "imm_large"
    ]
    vocab.extend(special_tokens)
    
    # Add most common tokens
    most_common = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)
    
    for token, count in most_common:
        if token not in special_tokens and len(vocab) < max_vocab_size:
            vocab.append(token)
    
    print(f"Final vocabulary size: {len(vocab):,}")
    
    # Save vocabulary
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(f"{token}\n")
    
    print(f"\nVocabulary saved to: {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Vocabulary Statistics:")
    print("="*60)
    print(f"Special tokens: {len(special_tokens)}")
    print(f"Regular tokens: {len(vocab) - len(special_tokens):,}")
    
    # Show normalized token counts
    print(f"\nNormalized token statistics:")
    for norm_token in ["stack_var", "imm_small", "imm_mid", "imm_large"]:
        if norm_token in token_counter:
            print(f"  {norm_token:20s} : {token_counter[norm_token]:,}")
    
    print(f"\nMost common tokens:")
    for token, count in most_common[:20]:
        if token in vocab:
            print(f"  {token:30s} : {count:,}")


if __name__ == "__main__":
    # Data files from /data/kun/dataset/
    data_files = [
        "/data/kun/dataset/train_cfg.txt",
        "/data/kun/dataset/train_dfg.txt",
        "/data/kun/dataset/val_cfg.txt",
        "/data/kun/dataset/val_dfg.txt",
        "/data/kun/dataset/test_cfg.txt",
        "/data/kun/dataset/test_dfg.txt",
    ]
    
    output_file = "./vocab.txt"
    
    print("="*60)
    print("Creating Vocabulary from Train + Val + Test Data")
    print("="*60)
    
    create_vocab_from_files(
        file_paths=data_files,
        output_file=output_file,
        min_freq=10,
        max_vocab_size=10000
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
