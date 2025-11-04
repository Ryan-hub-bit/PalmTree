#!/usr/bin/env python3
"""Test if PalmTree embeddings preserve the line order from the input file.

This tests the key question: Without positional encoding, can PalmTree know 
the ORDER of basic blocks as they appear in the file?

Usage:
  python probe/test_palmtree_order.py \
    --bb_file hello_palmtree.txt \
    --model_path pre-trained_model/palmtree/transformer.ep19 \
    --vocab_path pre-trained_model/palmtree/vocab \
    --out_emb /tmp/palmtree_order_emb.npy \
    --out_order /tmp/palmtree_order_indices.npy
"""
import argparse
import sys
import os
import numpy as np
import torch

# Add paths
sys.path.insert(0, os.path.abspath('pre-trained_model'))
sys.path.insert(0, os.path.abspath('src'))

# Compatibility shim for old pickled vocab
import palmtree
import palmtree.dataset
sys.modules['bert_pytorch'] = palmtree
sys.modules['bert_pytorch.dataset'] = palmtree.dataset

import eval_utils as utils


def read_bb_file(bb_file):
    """Read basic blocks from file, one BB per line.
    
    Converts tabs to spaces to match PalmTree's expected format.
    
    Returns:
        list of strings (one per BB)
    """
    bbs = []
    with open(bb_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Replace tabs with spaces and normalize whitespace
            line = line.replace('\t', ' ')
            # Replace multiple spaces with single space
            line = ' '.join(line.split())
            if line:
                bbs.append(line)
    return bbs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bb_file', required=True, help='Basic block file (one BB per line)')
    parser.add_argument('--model_path', default='pre-trained_model/palmtree/transformer.ep19')
    parser.add_argument('--vocab_path', default='pre-trained_model/palmtree/vocab')
    parser.add_argument('--out_emb', required=True, help='Output embeddings .npy')
    parser.add_argument('--out_order', required=True, help='Output order indices .npy')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TESTING: Can PalmTree embeddings preserve file order?")
    print("="*60)
    
    print(f"\nLoading PalmTree model...")
    palmtree = utils.UsableTransformer(
        model_path=args.model_path,
        vocab_path=args.vocab_path
    )
    
    print(f"\nReading basic blocks from: {args.bb_file}")
    bbs = read_bb_file(args.bb_file)
    print(f"Total basic blocks: {len(bbs)}")
    
    if len(bbs) > 0:
        print(f"\nFirst BB: {bbs[0][:100]}...")
        print(f"Last BB:  {bbs[-1][:100]}...")
    
    print(f"\nEncoding basic blocks in batches of {args.batch_size}...")
    all_embeddings = []
    
    for i in range(0, len(bbs), args.batch_size):
        batch = bbs[i:i+args.batch_size]
        embeddings = palmtree.encode(batch)
        all_embeddings.append(embeddings)
        
        if (i // args.batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch)}/{len(bbs)} BBs...")
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    # Create order indices (0, 1, 2, ..., N-1)
    order_indices = np.arange(len(bbs), dtype=np.float32)
    # Normalize to [0, 1]
    if len(bbs) > 1:
        order_indices = order_indices / (len(bbs) - 1)
    
    print(f"Order indices range: [{order_indices.min():.4f}, {order_indices.max():.4f}]")
    
    print(f"\nSaving embeddings to: {args.out_emb}")
    np.save(args.out_emb, embeddings)
    
    print(f"Saving order indices to: {args.out_order}")
    np.save(args.out_order, order_indices)
    
    print("\n" + "="*60)
    print("NEXT STEP: Test linear relationship with probe")
    print("="*60)
    print(f"\n  python probe/probe_linear_relationship.py \\")
    print(f"    --embeddings {args.out_emb} \\")
    print(f"    --positions {args.out_order}")
    print("\nIf R² is high → PalmTree knows the order")
    print("If R² is low → PalmTree does NOT know the order")
    print("="*60)


if __name__ == '__main__':
    main()
