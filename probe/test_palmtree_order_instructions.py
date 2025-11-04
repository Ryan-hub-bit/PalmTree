#!/usr/bin/env python3
"""Test if PalmTree embeddings preserve the line order from the input file.

IMPORTANT: Each BB line contains multiple instructions separated by tabs.
We encode EACH INSTRUCTION separately, then average them per BB.

Usage:
  python probe/test_palmtree_order_instructions.py \
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

# Force CPU mode before importing config
import config
config.USE_CUDA = False

# Compatibility shim for old pickled vocab
import palmtree
import palmtree.dataset
sys.modules['bert_pytorch'] = palmtree
sys.modules['bert_pytorch.dataset'] = palmtree.dataset

import eval_utils as utils


def read_bb_file_as_instructions(bb_file):
    """Read basic blocks and split into instructions.
    
    Each BB line has instructions separated by tabs.
    We need to encode each instruction separately.
    
    Returns:
        list of lists: [[inst1, inst2, ...], [inst1, inst2, ...], ...]
    """
    bb_instructions = []
    with open(bb_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by tab to get individual instructions
            instructions = line.split('\t')
            # Clean each instruction: normalize whitespace
            instructions = [' '.join(inst.split()) for inst in instructions if inst.strip()]
            
            if instructions:
                bb_instructions.append(instructions)
    
    return bb_instructions


def encode_bbs_as_instructions(palmtree, bb_instructions, batch_size=64):
    """Encode each BB by encoding its instructions separately and averaging.
    
    Args:
        palmtree: UsableTransformer instance
        bb_instructions: list of lists of instructions
        batch_size: batch size for palmtree.encode()
    
    Returns:
        numpy array of shape (num_bbs, emb_dim)
    """
    all_bb_embeddings = []
    
    for bb_idx, instructions in enumerate(bb_instructions):
        if (bb_idx + 1) % 100 == 0:
            print(f"  Processing BB {bb_idx + 1}/{len(bb_instructions)}...")
        
        # Encode all instructions in this BB at once
        # palmtree.encode() can handle batches
        inst_embeddings = palmtree.encode(instructions)
        
        # Average across instructions to get BB-level embedding
        bb_embedding = np.mean(inst_embeddings, axis=0)
        all_bb_embeddings.append(bb_embedding)
    
    return np.array(all_bb_embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bb_file', required=True, help='Basic block file (one BB per line, instructions separated by tabs)')
    parser.add_argument('--model_path', default='pre-trained_model/palmtree/transformer.ep19')
    parser.add_argument('--vocab_path', default='pre-trained_model/palmtree/vocab')
    parser.add_argument('--out_emb', required=True, help='Output embeddings .npy')
    parser.add_argument('--out_order', required=True, help='Output order indices .npy')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for encoding')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TESTING: Can PalmTree embeddings preserve file order?")
    print("ENCODING: Each instruction separately, then averaging per BB")
    print("="*70)
    
    print(f"\nLoading PalmTree model...")
    palmtree = utils.UsableTransformer(
        model_path=args.model_path,
        vocab_path=args.vocab_path
    )
    
    print(f"\nReading basic blocks from: {args.bb_file}")
    bb_instructions = read_bb_file_as_instructions(args.bb_file)
    print(f"Total basic blocks: {len(bb_instructions)}")
    
    if len(bb_instructions) > 0:
        print(f"\nFirst BB has {len(bb_instructions[0])} instructions:")
        for i, inst in enumerate(bb_instructions[0][:3]):
            print(f"  Inst {i+1}: {inst}")
        if len(bb_instructions[0]) > 3:
            print(f"  ... ({len(bb_instructions[0])-3} more)")
        
        print(f"\nLast BB has {len(bb_instructions[-1])} instructions:")
        for i, inst in enumerate(bb_instructions[-1][:3]):
            print(f"  Inst {i+1}: {inst}")
        if len(bb_instructions[-1]) > 3:
            print(f"  ... ({len(bb_instructions[-1])-3} more)")
    
    print(f"\nEncoding basic blocks (instruction-by-instruction)...")
    embeddings = encode_bbs_as_instructions(palmtree, bb_instructions, args.batch_size)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    # Create order indices (0, 1, 2, ..., N-1)
    order_indices = np.arange(len(bb_instructions), dtype=np.float32)
    # Normalize to [0, 1]
    if len(bb_instructions) > 1:
        order_indices = order_indices / (len(bb_instructions) - 1)
    
    print(f"Order indices range: [{order_indices.min():.4f}, {order_indices.max():.4f}]")
    
    print(f"\nSaving embeddings to: {args.out_emb}")
    np.save(args.out_emb, embeddings)
    
    print(f"Saving order indices to: {args.out_order}")
    np.save(args.out_order, order_indices)
    
    print("\n" + "="*70)
    print("NEXT STEP: Test linear relationship with probe")
    print("="*70)
    print(f"\n  python probe/probe_linear_relationship.py \\")
    print(f"    --embeddings {args.out_emb} \\")
    print(f"    --positions {args.out_order}")
    print("\nIf R² is high → PalmTree knows the order")
    print("If R² is low → PalmTree does NOT know the order")
    print("="*70)


if __name__ == '__main__':
    main()
