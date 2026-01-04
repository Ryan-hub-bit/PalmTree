"""
Address-Aware BERT Model

Modified BERT architecture that incorporates address normalization
at three hierarchical levels: binary, function, and basic block.
"""

import torch.nn as nn

from .transformer import TransformerBlock
from .embedding.address_embedding import AddressAwareBERTEmbedding


class AddressAwareBERT(nn.Module):
    """
    Address-aware BERT for binary code understanding.
    
    Uses hierarchical address positions (binary, function, basic block)
    in addition to standard token embeddings.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, 
                 use_address_embedding=True, use_var_embedding=True, max_len=512, segment_types=3):
        """
        Args:
            vocab_size: Size of the vocabulary
            hidden: Hidden size / embedding dimension
            n_layers: Number of transformer layers
            attn_heads: Number of attention heads
            dropout: Dropout rate
            use_address_embedding: Whether to use address-aware positional embeddings
            use_var_embedding: Whether to use var offset embeddings for var(0xXX) tokens
            max_len: Maximum sequence length for positional embeddings
            segment_types: Number of segment types
        """
        super().__init__()
        
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.use_address_embedding = use_address_embedding
        self.use_var_embedding = use_var_embedding

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # Address-aware embedding layer
        self.embedding = AddressAwareBERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            dropout=dropout,
            use_address_embedding=use_address_embedding,
            use_var_embedding=use_var_embedding,
            max_len=max_len,
            segment_types=segment_types
        )

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info, binary_pos, function_pos, bb_pos, var_offsets=None):
        """
        Forward pass.
        
        Args:
            x: [batch_size, seq_len] token IDs
            segment_info: [batch_size, seq_len] segment labels
            binary_pos: [batch_size, seq_len] binary-level positions
            function_pos: [batch_size, seq_len] function-level positions
            bb_pos: [batch_size, seq_len] basic block-level positions
            var_offsets: [batch_size, seq_len] (optional) var offset values
            
        Returns:
            output: [batch_size, seq_len, hidden]
        """
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info, binary_pos, function_pos, bb_pos, var_offsets)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def encode(self, x, segment_info, binary_pos, function_pos, bb_pos, var_offsets=None):
        """
        Encoding without the last transformer layer (for embeddings).
        
        Args:
            x: [batch_size, seq_len] token IDs
            segment_info: [batch_size, seq_len] segment labels
            binary_pos: [batch_size, seq_len] binary-level positions
            function_pos: [batch_size, seq_len] function-level positions
            bb_pos: [batch_size, seq_len] basic block-level positions
            var_offsets: [batch_size, seq_len] (optional) var offset values
            
        Returns:
            output: [batch_size, seq_len, hidden]
        """
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info, binary_pos, function_pos, bb_pos, var_offsets)

        # running over multiple transformer blocks (except last one)
        for transformer in self.transformer_blocks[:-1]:
            x = transformer.forward(x, mask)

        return x
