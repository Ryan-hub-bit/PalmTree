"""
Test original PalmTree model using the original dataloader and approach.

This uses:
- Original BERTDataset (expects tab-separated files with src/tgt address files)
- Original BERTLM model wrapper
- Original trainer's test method

Note: This requires the original data format with src/tgt address files.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from palmtree.dataset import WordVocab, BERTDataset
from palmtree.model import BERT, BERTLM
from palmtree.trainer import BERTTrainer


def main():
    parser = argparse.ArgumentParser(description="Test original PalmTree using original dataloader")
    
    parser.add_argument("--checkpoint", type=str,
                       default="../pre-trained_model/palmtree/transformer.ep19",
                       help="Path to PalmTree BERT checkpoint")
    parser.add_argument("--vocab", type=str,
                       default="../pre-trained_model/palmtree/vocab",
                       help="Vocabulary file")
    
    # Data paths - need BOTH instruction files AND address files
    parser.add_argument("--cfg_corpus", type=str, required=True,
                       help="CFG corpus file (instructions)")
    parser.add_argument("--dfg_corpus", type=str, required=True,
                       help="DFG corpus file (instructions)")
    parser.add_argument("--cfg_src", type=str, required=True,
                       help="CFG source address file")
    parser.add_argument("--dfg_src", type=str, required=True,
                       help="DFG source address file")
    parser.add_argument("--cfg_tgt", type=str, required=True,
                       help="CFG target address file")
    parser.add_argument("--dfg_tgt", type=str, required=True,
                       help="DFG target address file")
    
    parser.add_argument("--seq_len", type=int, default=20,
                       help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Data loader workers")
    parser.add_argument("--cuda", action="store_true", default=True,
                       help="Use CUDA")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Testing Original PalmTree Model")
    print("="*70)
    
    # Check if all required files exist
    required_files = [
        args.checkpoint, args.vocab,
        args.cfg_corpus, args.dfg_corpus,
        args.cfg_src, args.dfg_src,
        args.cfg_tgt, args.dfg_tgt
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"ERROR: Required file not found: {f}")
            print("\nThis test requires the original PalmTree data format:")
            print("  - Instruction files (CFG/DFG corpus)")
            print("  - Source address files (CFG/DFG src)")
            print("  - Target address files (CFG/DFG tgt)")
            return 1
    
    # Load vocabulary
    print(f"\nLoading vocabulary: {args.vocab}")
    vocab = WordVocab.load_vocab(args.vocab)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load BERT model
    print(f"\nLoading BERT checkpoint: {args.checkpoint}")
    bert = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f"BERT hidden size: {bert.hidden}")
    print(f"BERT layers: {bert.n_layers}")
    print(f"BERT attention heads: {bert.attn_heads}")
    
    # Create test dataset using original BERTDataset
    print(f"\nCreating test dataset...")
    test_dataset = BERTDataset(
        cfg_corpus_path=args.cfg_corpus,
        dfg_corpus_path=args.dfg_corpus,
        cfg_src_path=args.cfg_src,
        dfg_src_path=args.dfg_src,
        cfg_tgt_path=args.cfg_tgt,
        dfg_tgt_path=args.dfg_tgt,
        vocab=vocab,
        seq_len=args.seq_len,
        on_memory=True,
        drive_mode="min"  # Use minimum of CFG/DFG sizes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create trainer (will handle the testing)
    print(f"\nCreating trainer...")
    trainer = BERTTrainer(
        bert=bert,
        vocab_size=len(vocab),
        train_dataloader=None,  # No training
        test_dataloader=test_loader,
        lr=1e-4,
        with_cuda=args.cuda,
        log_freq=50
    )
    
    # Run test using original trainer's test method
    print("\n" + "="*70)
    print("Running test (epoch 0)...")
    print("="*70)
    trainer.test(epoch=0)
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
    

if __name__ == "__main__":
    main()
