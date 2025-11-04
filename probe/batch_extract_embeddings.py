#!/usr/bin/env python3
"""Extract embeddings and positions from a BB file for linear probing.

Usage:
  python probe/batch_extract_embeddings.py \
    --checkpoint cfg_pretrain/checkpoints/cfg_pretrain_mlm_only_best.pth \
    --vocab cfg_pretrain/vocab_extended \
    --bb_file hello.txt \
    --out_emb /tmp/h_BB.npy \
    --out_pos /tmp/p_func.npy
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

from probe.load_checkpoint import load_model_and_vocab

BB_LINE_RE = re.compile(
    r'<addr_start:(0x[0-9a-fA-F]+):([0-9eE+-.]+):([0-9eE+-.]+)>\s*(.*?)\s*<addr_end:(0x[0-9a-fA-F]+):([0-9eE+-.]+):([0-9eE+-.]+)>',
    re.S
)


def parse_bb_line(line):
    """Parse one BB line and return (tokens, pos, start_bin, start_func, end_bin, end_func) or None."""
    m = BB_LINE_RE.search(line)
    if not m:
        return None
    
    start_bin = float(m.group(2))
    start_func = float(m.group(3))
    insts_str = m.group(4).strip()
    end_bin = float(m.group(6))
    end_func = float(m.group(7))
    
    # Extract tokens: split on whitespace across all instructions
    tokens = []
    for inst in insts_str.split('\t'):
        inst = inst.strip()
        if inst:
            tokens.extend(inst.split())
    
    # Position: mean of start and end func_norm
    pos = (start_func + end_func) / 2.0
    
    return tokens, pos, start_bin, start_func, end_bin, end_func


def build_id_seqs(bb_file, vocab, seq_len=128, filter_sentinel=True):
    """Parse BB file and return (id_array, position_array, bin_pos_array, func_pos_array)."""
    ids = []
    positions = []
    bin_positions = []
    func_positions_list = []
    
    with open(bb_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_bb_line(line)
            if parsed is None:
                continue
            
            tokens, pos, start_bin, start_func, end_bin, end_func = parsed
            
            # Filter sentinel values if requested
            if filter_sentinel:
                if start_func < 0 or end_func < 0:
                    continue
            
            # Convert tokens to ids
            token_ids = [vocab.stoi.get(t, vocab.unk_index) for t in tokens]
            
            # Pad/truncate
            if len(token_ids) < seq_len:
                token_ids = token_ids + [vocab.pad_index] * (seq_len - len(token_ids))
            else:
                token_ids = token_ids[:seq_len]
            
            ids.append(token_ids)
            positions.append(pos)
            bin_positions.append((start_bin + end_bin) / 2.0)
            func_positions_list.append(pos)
    
    return (np.array(ids, dtype=np.int64), 
            np.array(positions, dtype=np.float32),
            np.array(bin_positions, dtype=np.float32),
            np.array(func_positions_list, dtype=np.float32))


def extract_embeddings(checkpoint, vocab_path, bb_file, out_emb, out_pos, out_bin_pos=None,
                       seq_len=128, batch_size=256, filter_sentinel=True, pooling='cls'):
    """Extract embeddings and positions, save to .npy files."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Loading model from {checkpoint}')
    print(f'Loading vocab from {vocab_path}')
    model, vocab = load_model_and_vocab(checkpoint, vocab_path, device=device)
    
    print(f'Parsing BB file: {bb_file}')
    X_ids, p, bin_pos, func_pos = build_id_seqs(bb_file, vocab, seq_len=seq_len, filter_sentinel=filter_sentinel)
    
    N = X_ids.shape[0]
    print(f'Found {N} basic blocks (after filtering), seq_len={seq_len}')
    
    if N == 0:
        print('No valid basic blocks found!')
        return
    
    # Extract embeddings in batches
    model.eval()
    emb_list = []
    
    print('Extracting embeddings...')
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = torch.from_numpy(X_ids[i:i+batch_size]).to(device)
            batch_bin_pos = torch.from_numpy(bin_pos[i:i+batch_size]).to(device)
            batch_func_pos = torch.from_numpy(func_pos[i:i+batch_size]).to(device)
            
            # CFGPretrainModel.forward(token_ids, binary_positions, function_positions, sequence_positions, attention_mask)
            # For embeddings, we need to broadcast positions to match seq_len
            B = batch.size(0)
            # Binary and function positions: use the same value for all tokens in the sequence
            binary_positions = batch_bin_pos.unsqueeze(1).expand(B, seq_len)
            function_positions = batch_func_pos.unsqueeze(1).expand(B, seq_len)
            # Sequence positions: 0, 1, 2, ..., seq_len-1
            sequence_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, seq_len)
            # Attention mask: 1 for valid tokens, 0 for padding
            attention_mask = (batch != vocab.pad_index).long()
            
            out = model(batch, binary_positions, function_positions, sequence_positions, attention_mask)  # (B, seq_len, hidden)
            
            if pooling == 'cls':
                emb = out[:, 0, :].cpu().numpy()
            elif pooling == 'mean':
                mask = (batch != vocab.pad_index).float().unsqueeze(-1)
                sum_vec = (out * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                emb = (sum_vec / counts).cpu().numpy()
            else:
                raise ValueError(f'Unknown pooling: {pooling}')
            
            emb_list.append(emb)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f'  Processed {i + batch_size}/{N} blocks')
    
    H = np.vstack(emb_list)
    
    print(f'\nEmbeddings shape: {H.shape}')
    print(f'Function positions shape: {p.shape}')
    print(f'Function position range: [{p.min():.6f}, {p.max():.6f}]')
    print(f'Binary positions shape: {bin_pos.shape}')
    print(f'Binary position range: [{bin_pos.min():.6f}, {bin_pos.max():.6f}]')
    
    np.save(out_emb, H)
    np.save(out_pos, p)
    
    print(f'\nSaved embeddings to {out_emb}')
    print(f'Saved function positions to {out_pos}')
    
    if out_bin_pos is not None:
        np.save(out_bin_pos, bin_pos)
        print(f'Saved binary positions to {out_bin_pos}')


def main():
    parser = argparse.ArgumentParser(description='Extract BB embeddings for linear probe')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab', required=True, help='Path to vocab file')
    parser.add_argument('--bb_file', required=True, help='BB file (one line per BB)')
    parser.add_argument('--out_emb', default='/tmp/h_BB.npy', help='Output embeddings .npy')
    parser.add_argument('--out_pos', default='/tmp/p_func.npy', help='Output function positions .npy')
    parser.add_argument('--out_bin_pos', default=None, help='Output binary positions .npy (optional)')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--pooling', choices=['cls', 'mean'], default='cls', help='Pooling method')
    parser.add_argument('--no_filter', action='store_true', help='Do not filter sentinel positions')
    args = parser.parse_args()
    
    extract_embeddings(
        args.checkpoint,
        args.vocab,
        args.bb_file,
        args.out_emb,
        args.out_pos,
        out_bin_pos=args.out_bin_pos,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        filter_sentinel=not args.no_filter,
        pooling=args.pooling
    )


if __name__ == '__main__':
    main()
