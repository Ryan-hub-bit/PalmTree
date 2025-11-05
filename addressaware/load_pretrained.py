"""
Utility to load pre-trained PalmTree model and extract components
for Address-Aware BERT initialization.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from palmtree.model.bert import BERT
from model import AddressAwareBERT, AddressAwareBERTForPretraining


def load_palmtree_weights(checkpoint_path, vocab_size, device='cuda'):
    """
    Load pre-trained PalmTree checkpoint and extract weights.
    
    Args:
        checkpoint_path: Path to PalmTree checkpoint (.pt file)
        vocab_size: Size of vocabulary (must match checkpoint)
        device: Device to load onto
        
    Returns:
        Dictionary with:
            - token_embedding: Token embedding weights
            - position_embedding: Position embedding weights
            - segment_embedding: Segment embedding weights
            - transformer_state: State dict for transformer blocks
            - full_state: Full state dict (optional, for reference)
    """
    print(f"[INFO] Loading PalmTree checkpoint from: {checkpoint_path}")
    
    # Load checkpoint (weights_only=False for backward compatibility)
    # PalmTree saves the entire model object, not a state_dict
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"[INFO] Loaded model type: {type(model).__name__}")
    
    # The model is a BERT object, extract its state_dict
    state_dict = model.state_dict()
    print(f"[INFO] Extracted state_dict with {len(state_dict)} keys")
    
    # Extract token embedding weights
    # PalmTree structure: embedding.token.weight (not embedding.token.embedding.weight)
    token_emb_key = 'embedding.token.weight'
    if token_emb_key in state_dict:
        token_embedding = state_dict[token_emb_key]
        print(f"[INFO] Extracted token embedding: {token_embedding.shape}")
    else:
        print(f"[WARNING] Token embedding not found (looking for '{token_emb_key}')")
        print(f"[DEBUG] Available keys: {list(state_dict.keys())[:5]}")
        token_embedding = None
    
    # Extract position embedding weights
    # PalmTree structure: embedding.position.pe (sinusoidal, not learnable)
    position_emb_key = 'embedding.position.pe'
    if position_emb_key in state_dict:
        position_embedding = state_dict[position_emb_key]
        print(f"[INFO] Extracted position embedding: {position_embedding.shape}")
    else:
        print(f"[WARNING] Position embedding not found (looking for '{position_emb_key}')")
        position_embedding = None
    
    # Extract segment embedding weights
    # PalmTree structure: embedding.segment.weight (not embedding.segment.embedding.weight)
    segment_emb_key = 'embedding.segment.weight'
    if segment_emb_key in state_dict:
        segment_embedding = state_dict[segment_emb_key]
        print(f"[INFO] Extracted segment embedding: {segment_embedding.shape}")
    else:
        print(f"[WARNING] Segment embedding not found (looking for '{segment_emb_key}')")
        segment_embedding = None
    
    # Extract transformer blocks
    # PalmTree structure: transformer_blocks.0.*, transformer_blocks.1.*, etc.
    transformer_state = {}
    for key, value in state_dict.items():
        if key.startswith('transformer_blocks.'):
            transformer_state[key] = value
    
    if transformer_state:
        print(f"[INFO] Extracted {len(transformer_state)} transformer parameters")
    else:
        print(f"[WARNING] No transformer blocks found, will use random initialization")
        transformer_state = None
    
    return {
        'token_embedding': token_embedding,
        'position_embedding': position_embedding,
        'segment_embedding': segment_embedding,
        'transformer_state': transformer_state,
        'full_state': state_dict,
    }


def create_addressaware_from_palmtree(
    palmtree_checkpoint,
    vocab_size,
    hidden=768,
    n_layers=12,
    attn_heads=12,
    dropout=0.1,
    max_len=512,
    device='cuda'
):
    """
    Create Address-Aware BERT model initialized with pre-trained PalmTree weights.
    
    ALL PalmTree components are FROZEN (token emb, seq pos, segment emb, transformer).
    ONLY the new address positional embedding and task heads are TRAINABLE.
    
    Args:
        palmtree_checkpoint: Path to PalmTree checkpoint
        vocab_size: Vocabulary size
        hidden: Hidden dimension
        n_layers: Number of transformer layers
        attn_heads: Number of attention heads
        dropout: Dropout rate
        max_len: Maximum sequence length
        device: Device to use
        
    Returns:
        AddressAwareBERTForPretraining model with frozen PalmTree components
    """
    print("=" * 70)
    print("Creating Address-Aware BERT from Pre-trained PalmTree")
    print("Strategy: FREEZE all PalmTree, TRAIN only address components")
    print("=" * 70)
    
    # Load PalmTree weights
    weights = load_palmtree_weights(palmtree_checkpoint, vocab_size, device)
    
    # Create Address-Aware BERT
    # This will automatically FREEZE token embeddings and transformer blocks
    bert = AddressAwareBERT(
        vocab_size=vocab_size,
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads,
        dropout=dropout,
        max_len=max_len,
        pretrained_token_emb=weights['token_embedding'],
        pretrained_segment_emb=weights['segment_embedding'],
        pretrained_transformer=weights['transformer_state']
    )
    
    # Create full model with pretraining heads
    model = AddressAwareBERTForPretraining(bert, vocab_size)
    model = model.to(device)
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[INFO] Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Load pre-trained PalmTree into Address-Aware BERT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PalmTree checkpoint")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--hidden", type=int, default=768, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--attn_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--freeze_token", action="store_true", help="Freeze token embeddings")
    parser.add_argument("--freeze_transformer", action="store_true", help="Freeze transformer blocks")
    parser.add_argument("--output", type=str, help="Optional: save initialized model")
    
    args = parser.parse_args()
    
    # Create model
    model = create_addressaware_from_palmtree(
        palmtree_checkpoint=args.checkpoint,
        vocab_size=args.vocab_size,
        hidden=args.hidden,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        freeze_token_emb=args.freeze_token,
        freeze_transformer=args.freeze_transformer
    )
    
    # Optionally save
    if args.output:
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': args.vocab_size,
            'hidden': args.hidden,
            'n_layers': args.n_layers,
            'attn_heads': args.attn_heads,
        }, args.output)
        print(f"\n[SAVED] Model saved to: {args.output}")
