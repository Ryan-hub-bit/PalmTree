"""
Address-Aware jTrans Model

Self-contained implementation using strupos-style architecture.
No HuggingFace dependencies - uses our own TransformerBlock implementation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import transformer_components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from transformer_components import TransformerBlock
from address_embedding import AddressAwareBERTEmbedding


class AddressAwareJTransForMLM(nn.Module):
    """
    Address-aware BERT with MLM and JTP heads for pretraining.
    
    Similar to baseline but uses hierarchical address embeddings.
    """
    
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, 
                 dropout=0.1, max_len=512):
        super().__init__()
        
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Address-aware embedding layer
        self.embeddings = AddressAwareBERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            dropout=dropout,
            max_len=max_len,
            use_address_embedding=True,
            use_var_embedding=True,
            segment_types=256,  # Support up to 256 instructions per sequence
            vocab_stoi=None  # Will be set by train script after vocab is loaded
        )
        
        # Transformer blocks (self-contained implementation)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
            for _ in range(n_layers)
        ])
        
        # MLM head (predicts masked tokens)
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, vocab_size)
        )
        
        # JTP head (predicts jump target positions)
        # Outputs position index [0, max_len-1]
        self.jtp_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, max_len)
        )
    
    def forward(self, token_ids, token_type_ids, 
                binary_pos, function_pos, bb_pos, var_offsets=None):
        """
        Forward pass.
        
        Args:
            token_ids: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            binary_pos: [batch_size, seq_len] normalized [0,1] positions
            function_pos: [batch_size, seq_len] normalized [0,1] positions
            bb_pos: [batch_size, seq_len] normalized [0,1] positions
            var_offsets: [batch_size, seq_len] var(0xXX) offsets, -1 for non-var
            
        Returns:
            mlm_logits: [batch_size, seq_len, vocab_size]
            jtp_logits: [batch_size, seq_len, max_len]
        """
        # Create attention mask (mask out padding tokens)
        mask = (token_ids > 0).unsqueeze(1).repeat(1, token_ids.size(1), 1).unsqueeze(1)
        
        # Get embeddings with address awareness
        x = self.embeddings(
            token_ids,
            token_type_ids,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets
        )
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        # MLM predictions
        mlm_logits = self.mlm_head(x)
        
        # JTP predictions
        jtp_logits = self.jtp_head(x)
        
        return mlm_logits, jtp_logits


def create_addressaware_model(vocab_size, hidden=768, n_layers=12, attn_heads=12, 
                               dropout=0.1, max_len=512):
    """
    Factory function to create address-aware model.
    """
    model = AddressAwareJTransForMLM(
        vocab_size=vocab_size,
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads,
        dropout=dropout,
        max_len=max_len
    )
    
    # Initialize weights
    model.apply(lambda module: _init_weights(module))
    
    return model


def _init_weights(module):
    """Initialize weights (same as BERT)"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    if isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
