"""Helpers to load a Palmtree model checkpoint and vocabulary.

This is the same loader as in `src/palmtree/probe/load_checkpoint.py` but
placed at repository root so it's next to `cfg_pretrain/`.

Usage:
    from probe.load_checkpoint import load_model_and_vocab
    model, vocab = load_model_and_vocab(checkpoint_path, vocab_path, device)
"""
from typing import Optional, Tuple
import torch
import os
import sys

# Compatibility shim for old pickled vocab files that reference 'bert_pytorch'
# Need to import palmtree modules first, then alias them
import palmtree
import palmtree.dataset
sys.modules['bert_pytorch'] = palmtree
sys.modules['bert_pytorch.dataset'] = palmtree.dataset


def load_model_and_vocab(checkpoint_path: str, vocab_path: str, device: Optional[torch.device] = None):
    """Load model and vocab.

    Args:
        checkpoint_path: path to a .pth checkpoint. This repo commonly saved full
            model objects via torch.save(model, path). The loader will attempt
            to load that object and return it in eval() mode.
        vocab_path: path to the pickled vocab object (palmtree.dataset.vocab.Vocab)
        device: torch.device or None (auto-select CPU/GPU)

    Returns:
        (model, vocab)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    # Load vocab first (needed for model reconstruction)
    from palmtree.dataset.vocab import Vocab
    vocab = Vocab.load_vocab(vocab_path)

    # Prefer map_location so loading works without GPU
    data = torch.load(checkpoint_path, map_location=device)

    # If the checkpoint is a model object, return it directly
    if hasattr(data, "eval"):
        model = data
    elif isinstance(data, dict) and "model_state_dict" in data:
        # Common pattern: a dict with 'model_state_dict' and metadata
        # Try to reconstruct model from cfg_pretrain config using CFGPretrainModel
        print("Checkpoint contains state_dict only; reconstructing CFGPretrainModel from cfg_pretrain config...")
        try:
            # Add both repo root and cfg_pretrain to path
            sys.path.insert(0, os.path.abspath('.'))
            sys.path.insert(0, os.path.abspath('./cfg_pretrain'))
            from cfg_pretrain import config as cfg
            from cfg_pretrain.train import CFGPretrainModel
            
            vocab_size = len(vocab)
            model = CFGPretrainModel(
                vocab_size=vocab_size,
                embed_dim=cfg.EMBED_DIM,
                num_heads=cfg.NUM_HEADS,
                num_layers=cfg.NUM_LAYERS,
                max_seq_len=cfg.MAX_SEQ_LEN,
                palmtree_embeddings=None  # Don't need PalmTree embeddings for inference
            )
            model.load_state_dict(data["model_state_dict"])
            print(f"Reconstructed CFGPretrainModel: embed_dim={cfg.EMBED_DIM}, layers={cfg.NUM_LAYERS}, heads={cfg.NUM_HEADS}")
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint contains state_dict only and auto-reconstruction failed: {e}"
            )
    else:
        # Fallback: try to return whatever was saved (may raise later when used)
        model = data

    model.to(device)
    model.eval()

    return model, vocab
