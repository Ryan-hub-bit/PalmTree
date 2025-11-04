"""Extract embedding for a specific basic block (by address) from all_bb_pairs.txt.

Usage example:
  python probe/extract_bb_embedding.py \
    --checkpoint cfg_pretrain/checkpoints/cfg_pretrain_mlm_only_best.pth \
    --vocab cfg_pretrain/vocab_extended \
    --bb_file all_bb_pairs.txt \
    --addr 0x401000 \
    --seq_len 128 \
    --pooling cls

The script scans `all_bb_pairs.txt`, finds the first BB whose <addr_start:...>
matches `--addr`, tokenizes it according to the repo's vocab, pads/truncates to
`--seq_len`, runs the model, and prints/saves the embedding.
"""
import argparse
import re
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from probe.load_checkpoint import load_model_and_vocab
import torch


BB_PATTERN = re.compile(r'(<addr_start:[^>]+?>\s.*?\s<addr_end:[^>]+?>)')
ADDR_START_RE = re.compile(r'<addr_start:(0x[0-9a-fA-F]+):')


def parse_bb_tokens(bb_str: str):
    """Return list of token strings for a basic block string as produced by bb_flow.py"""
    # bb_str already has commas removed upstream. Instructions are separated by tabs.
    # We'll split on whitespace to match training-time tokenization.
    # Remove the addr_start and addr_end wrappers from token stream (we keep them as tokens)
    tokens = bb_str.strip().split()
    return tokens


def find_bb_by_addr(bb_file: str, target_addr: str):
    """Scan file for a BB whose addr_start hex equals target_addr (string like '0x401000').
    Returns the token list or None if not found."""
    with open(bb_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            matches = BB_PATTERN.findall(line)
            if not matches:
                continue
            # Typically there will be two matches per line (source BB and target BB)
            for bb_str in matches:
                m = ADDR_START_RE.search(bb_str)
                if m:
                    addr = m.group(1).lower()
                    if addr == target_addr.lower():
                        return parse_bb_tokens(bb_str)
    return None


def tokens_to_ids(tokens, vocab, seq_len):
    ids = [vocab.stoi.get(tok, vocab.unk_index) for tok in tokens]
    if len(ids) < seq_len:
        ids = ids + [vocab.pad_index] * (seq_len - len(ids))
    else:
        ids = ids[:seq_len]
    return ids


def main():
    parser = argparse.ArgumentParser(description="Extract embedding for BB address from all_bb_pairs.txt")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--bb_file', default='all_bb_pairs.txt')
    parser.add_argument('--addr', required=True, help='hex address of BB start, e.g. 0x401000')
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--pooling', choices=['cls','mean'], default='cls')
    parser.add_argument('--out', default=None, help='optional path to save embedding .npy')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading model and vocab...')
    model, vocab = load_model_and_vocab(args.checkpoint, args.vocab, device=device)

    print(f"Scanning {args.bb_file} for address {args.addr}...")
    tokens = find_bb_by_addr(args.bb_file, args.addr)
    if tokens is None:
        print('Basic block not found')
        return

    print('Found BB tokens sample:', ' '.join(tokens[:20]))

    ids = tokens_to_ids(tokens, vocab, args.seq_len)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    seg = torch.zeros_like(x, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(x, seg)  # (1, seq_len, hidden)

    if args.pooling == 'cls':
        emb = out[0,0,:].cpu().numpy()
    else:
        mask = (x != vocab.pad_index).float().unsqueeze(-1)
        sum_vec = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        emb = (sum_vec / counts).cpu().numpy()[0]

    print('Embedding shape:', emb.shape)
    if args.out:
        np.save(args.out, emb)
        print('Saved embedding to', args.out)
    else:
        # Print first 8 values
        print('Embedding (first 8 dims):', emb[:8].tolist())


if __name__ == '__main__':
    main()
