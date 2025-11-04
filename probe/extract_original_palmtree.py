#!/usr/bin/env python3
"""Extract embeddings from ORIGINAL PalmTree model using cleaned BB format.

This tests whether PalmTree can predict binary-level position without ANY positional info.
The input files (hello_palmtree.txt, redis-cli_palmtree.txt) have NO address tags.

Usage:
  python probe/extract_original_palmtree.py \
    --checkpoint pre-trained_model/palmtree/transformer.ep19 \
    --vocab pre-trained_model/palmtree/vocab_extended \
    --bb_file hello_palmtree.txt \
    --bb_file_with_positions hello.txt \
    --out_emb /tmp/palmtree_clean_h_BB.npy \
    --out_pos /tmp/palmtree_clean_p_binary.npy
"""
import argparse
import re
import sys
import os
import numpy as np
import torch

# Add repo paths
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('.'))

# Compatibility shim for old pickled vocab
import palmtree
import palmtree.dataset
sys.modules['bert_pytorch'] = palmtree
sys.modules['bert_pytorch.dataset'] = palmtree.dataset

from palmtree.dataset.vocab import Vocab

# Regex to extract positions from the original file
BB_LINE_RE = re.compile(
    r'<addr_start:(0x[0-9a-fA-F]+):([0-9eE+-.]+):([0-9eE+-.]+)>\s*(.*?)\s*<addr_end:(0x[0-9a-fA-F]+):([0-9eE+-.]+):([0-9eE+-.]+)>',
    re.S
)


def parse_positions_from_original(line):
    """Extract positions from original format file."""
    m = BB_LINE_RE.search(line)
    if not m:
        return None
    
    start_bin = float(m.group(2))
    start_func = float(m.group(3))
    end_bin = float(m.group(6))
    end_func = float(m.group(7))
    
    return {
        'bin': (start_bin + end_bin) / 2.0,
        'func': (start_func + end_func) / 2.0,
        'start_bin': start_bin,
        'start_func': start_func
    }


def parse_clean_bb_line(line):
    """Parse cleaned BB line (no address tags, just instructions).
    
    Returns list of tokens or None
    """
    line = line.strip()
    if not line:
        return None
    
    # Split on whitespace to get tokens
    tokens = line.split()
    
    return tokens


def build_id_seqs(bb_file_clean, bb_file_positions, vocab, seq_len=512, filter_sentinel=True):
    """Parse cleaned BB file and get positions from original file.
    
    Args:
        bb_file_clean: File with cleaned BB format (no address info)
        bb_file_positions: Original file with position information
        
    Returns:
        - ids: (N, seq_len) token IDs
        - bin_positions: (N,) binary positions
        - func_positions: (N,) function positions
    """
    ids = []
    bin_positions = []
    func_positions_list = []
    
    with open(bb_file_clean, 'r', encoding='utf-8', errors='ignore') as f_clean, \
         open(bb_file_positions, 'r', encoding='utf-8', errors='ignore') as f_pos:
        
        for line_clean, line_pos in zip(f_clean, f_pos):
            # Parse tokens from cleaned file
            tokens = parse_clean_bb_line(line_clean)
            if tokens is None:
                continue
            
            # Parse positions from original file
            pos_info = parse_positions_from_original(line_pos)
            if pos_info is None:
                continue
            
            # Filter sentinel values if requested
            if filter_sentinel:
                if pos_info['start_func'] < 0:
                    continue
            
            # Convert tokens to ids
            token_ids = [vocab.stoi.get(t, vocab.unk_index) for t in tokens]
            
            # Pad/truncate
            if len(token_ids) < seq_len:
                token_ids += [vocab.pad_index] * (seq_len - len(token_ids))
            else:
                token_ids = token_ids[:seq_len]
            
            ids.append(token_ids)
            bin_positions.append(pos_info['bin'])
            func_positions_list.append(pos_info['func'])
    
    return np.array(ids, dtype=np.int64), np.array(bin_positions), np.array(func_positions_list)


def extract_embeddings(model, ids, batch_size=64, pooling='cls', device='cuda'):
    """Extract embeddings from PalmTree model.
    
    Args:
        pooling: 'cls' or 'mean'
    """
    model.eval()
    model = model.to(device)
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            x = torch.LongTensor(batch_ids).to(device)
            
            # PalmTree forward: just token embeddings, no explicit positions
            # segment_info is None for single-sequence input
            output = model.forward(x, None)
            
            if pooling == 'cls':
                # Use first token (CLS)
                h = output[:, 0, :]
            elif pooling == 'mean':
                # Mean pooling over non-padding tokens
                # Assuming padding index is 0
                mask = (x != 0).float().unsqueeze(-1)
                masked_output = output * mask
                sum_output = masked_output.sum(dim=1)
                count = mask.sum(dim=1).clamp(min=1)
                h = sum_output / count
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            
            all_embeddings.append(h.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='PalmTree checkpoint path')
    parser.add_argument('--vocab', required=True, help='Vocab file path')
    parser.add_argument('--bb_file', required=True, help='Cleaned BB file (no address info)')
    parser.add_argument('--bb_file_with_positions', required=True, help='Original BB file (for extracting positions)')
    parser.add_argument('--out_emb', required=True, help='Output embeddings .npy file')
    parser.add_argument('--out_pos_binary', required=True, help='Output binary positions .npy file')
    parser.add_argument('--out_pos_func', default=None, help='Output function positions .npy file')
    parser.add_argument('--seq_len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--pooling', choices=['cls', 'mean'], default='cls', help='Pooling strategy')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print(f"Loading vocab from {args.vocab}")
    vocab = Vocab.load_vocab(args.vocab)
    print(f"Vocab size: {len(vocab)}")
    
    print(f"Loading checkpoint from {args.checkpoint}")
    model = torch.load(args.checkpoint, map_location='cpu')
    print(f"Model loaded: {type(model)}")
    
    print(f"Parsing BBs from cleaned file: {args.bb_file}")
    print(f"Getting positions from: {args.bb_file_with_positions}")
    ids, bin_positions, func_positions = build_id_seqs(
        args.bb_file,
        args.bb_file_with_positions,
        vocab,
        seq_len=args.seq_len
    )
    print(f"Total BBs: {len(ids)}")
    print(f"Binary position range: [{bin_positions.min():.4f}, {bin_positions.max():.4f}]")
    print(f"Function position range: [{func_positions.min():.4f}, {func_positions.max():.4f}]")
    
    print(f"\nExtracting embeddings with {args.pooling} pooling on {args.device}...")
    embeddings = extract_embeddings(
        model,
        ids,
        batch_size=args.batch_size,
        pooling=args.pooling,
        device=args.device
    )
    print(f"Embeddings shape: {embeddings.shape}")
    
    print(f"\nSaving embeddings to {args.out_emb}")
    np.save(args.out_emb, embeddings)
    
    print(f"Saving binary positions to {args.out_pos_binary}")
    np.save(args.out_pos_binary, bin_positions)
    
    if args.out_pos_func:
        print(f"Saving function positions to {args.out_pos_func}")
        np.save(args.out_pos_func, func_positions)
    
    print("\nDone!")
    print(f"\nTo test linear relationship:")
    print(f"  python probe/probe_linear_relationship.py \\")
    print(f"    --embeddings {args.out_emb} \\")
    print(f"    --positions {args.out_pos_binary}")


if __name__ == '__main__':
    main()
