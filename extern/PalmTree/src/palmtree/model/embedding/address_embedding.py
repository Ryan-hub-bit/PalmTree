"""
Address-Aware Positional Embedding

Uses sin/cos encoding on three levels of address normalization:
1. Binary-level position (bnorm): Position within entire binary
2. Function-level position (fnorm): Position within function  
3. Basic block-level position (bbnorm): Position within basic block

These are added to create a rich positional encoding that captures
hierarchical structure of binary code.
"""

import torch
import torch.nn as nn
import math


class SequencePositionalEmbedding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence order (like original PalmTree).
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model] or [batch_size, seq_len]
        Returns:
            Positional encoding: [batch_size, seq_len, d_model]
        """
        return self.pe[:, :x.size(1)]


class AddressPositionalEmbedding(nn.Module):
    """
    Address-aware positional embedding using sin/cos encoding + MLP projection.
    
    Applies sinusoidal position encoding to three normalized address values:
    - binary_pos: [0, 1] position in binary
    - function_pos: [0, 1] position in function
    - bb_pos: [0, 1] position in basic block
    
    Each level gets its own sin/cos encoding, then concatenated and projected
    to d_model via MLP.
    """
    
    def __init__(self, d_model, max_len=512, intermediate_size=128, dropout=0.1):
        """
        Args:
            d_model: Embedding dimension (output size)
            max_len: Maximum sequence length
            intermediate_size: Size of encoding per address level before concatenation
            dropout: Dropout rate for MLP projection
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_per_level = intermediate_size  # Dimensions per address level
        
        # Learnable weight to balance the three levels
        self.level_weights = nn.Parameter(torch.ones(3))
        
        # MLP to project concatenated encodings (3 * intermediate_size) to d_model
        concat_size = 3 * intermediate_size
        self.projection = nn.Sequential(
            nn.Linear(concat_size, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
    def _sinusoidal_encoding(self, positions, d_model):
        """
        Create sinusoidal encoding for a batch of positions.
        
        Args:
            positions: [batch_size, seq_len] normalized positions in [0, 1]
            d_model: Embedding dimension for this level
            
        Returns:
            encoding: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = positions.shape
        
        # Create dimension indices
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=positions.device) 
                            * -(math.log(10000.0) / d_model))
        
        # Scale positions to [0, 10000] range for better frequency distribution
        positions_scaled = positions.unsqueeze(-1) * 10000.0  # [batch, seq, 1]
        
        # Create encoding
        encoding = torch.zeros(batch_size, seq_len, d_model, device=positions.device)
        
        # Apply sin to even indices
        encoding[:, :, 0::2] = torch.sin(positions_scaled * div_term)
        
        # Apply cos to odd indices
        if d_model % 2 == 0:
            encoding[:, :, 1::2] = torch.cos(positions_scaled * div_term)
        else:
            encoding[:, :, 1::2] = torch.cos(positions_scaled * div_term[:-1])
        
        return encoding
    
    def forward(self, binary_pos, function_pos, bb_pos):
        """
        Forward pass with three levels of address positions.
        
        Args:
            binary_pos: [batch_size, seq_len] binary-level normalized positions
            function_pos: [batch_size, seq_len] function-level normalized positions
            bb_pos: [batch_size, seq_len] basic block-level normalized positions
            
        Returns:
            embedding: [batch_size, seq_len, d_model] combined positional embedding
        """
        # Generate sinusoidal encoding for each level
        binary_enc = self._sinusoidal_encoding(binary_pos, self.d_per_level)
        function_enc = self._sinusoidal_encoding(function_pos, self.d_per_level)
        bb_enc = self._sinusoidal_encoding(bb_pos, self.d_per_level)
        
        # Apply learnable weights (softmax normalized)
        weights = torch.softmax(self.level_weights, dim=0)
        
        # Weight each encoding
        binary_enc = binary_enc * weights[0]
        function_enc = function_enc * weights[1]
        bb_enc = bb_enc * weights[2]
        
        # Concatenate along embedding dimension: [batch, seq, 3*intermediate_size]
        concatenated = torch.cat([binary_enc, function_enc, bb_enc], dim=-1)
        
        # Project to d_model: [batch, seq, d_model]
        embedding = self.projection(concatenated)
        
        return embedding


class VarPositionalEmbedding(nn.Module):
    """
    Positional embedding for var(0xXX) tokens based on their offset values.
    
    Uses sinusoidal encoding on the variable offset (e.g., 0x10 -> 16).
    Only var tokens get the encoding, all other tokens get zeros.
    """
    
    def __init__(self, d_model, max_offset=4096):
        """
        Args:
            d_model: Embedding dimension
            max_offset: Maximum expected variable offset (for normalization)
        """
        super().__init__()
        self.d_model = d_model
        self.max_offset = max_offset
        
        # Precompute div_term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) 
                            * -(math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
    
    def forward(self, var_offsets):
        """
        Forward pass.
        
        Args:
            var_offsets: [batch_size, seq_len] variable offsets (0 for non-var tokens)
                         e.g., for var(0x10), offset = 16; for non-var, offset = 0
            
        Returns:
            encoding: [batch_size, seq_len, d_model] sinusoidal encoding for var offsets
                      (zeros for non-var tokens)
        """
        batch_size, seq_len = var_offsets.shape
        device = var_offsets.device
        
        # Create output tensor initialized to zeros
        encoding = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Create mask for var tokens (offset > 0)
        var_mask = (var_offsets > 0).unsqueeze(-1)  # [batch, seq, 1]
        
        # Scale offsets for sinusoidal encoding
        offsets_scaled = var_offsets.float().unsqueeze(-1)  # [batch, seq, 1]
        
        # Apply sin to even indices
        encoding[:, :, 0::2] = torch.sin(offsets_scaled * self.div_term)
        
        # Apply cos to odd indices
        if self.d_model % 2 == 0:
            encoding[:, :, 1::2] = torch.cos(offsets_scaled * self.div_term)
        else:
            encoding[:, :, 1::2] = torch.cos(offsets_scaled * self.div_term[:-1])
        
        # Zero out non-var tokens
        encoding = encoding * var_mask.float()
        
        return encoding


class AddressAwareBERTEmbedding(nn.Module):
    """
    Address-aware BERT Embedding.
    
    Combines embeddings:
    1. Token embedding
    2. Sequence positional embedding (sinusoidal, standard BERT style)
    3. Address positional embedding (sin/cos on binary_pos, function_pos, bb_pos)
    4. Segment embedding (for NSP task)
    5. Var positional embedding (sin/cos on var offsets for var(0xXX) tokens)
    """
    
    def __init__(self, vocab_size, embed_size, dropout=0.1, use_address_embedding=True, use_var_embedding=True, max_len=512, segment_types=3):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            dropout: Dropout rate
            use_address_embedding: Whether to use address-aware positional embeddings
            use_var_embedding: Whether to use var offset embeddings
            max_len: Maximum sequence length for positional embeddings
            segment_types: Number of segment types (default 3: padding, sent_A, sent_B)
        """
        super().__init__()
        
        self.embed_size = embed_size
        self.use_address_embedding = use_address_embedding
        self.use_var_embedding = use_var_embedding
        
        # 1. Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # 2. Sequence positional embedding - sinusoidal (standard BERT style)
        self.position_embedding = SequencePositionalEmbedding(embed_size, max_len=max_len)
        
        # 3. Address-aware positional embedding - sin/cos encoding on 3 levels
        if self.use_address_embedding:
            self.address_position = AddressPositionalEmbedding(embed_size, max_len=max_len, dropout=dropout)
        else:
            self.address_position = None
        
        # 4. Segment embedding (for NSP)
        self.segment_embedding = nn.Embedding(segment_types, embed_size, padding_idx=0)
        
        # 5. Var positional embedding - sin/cos encoding on var offsets
        if self.use_var_embedding:
            self.var_position = VarPositionalEmbedding(embed_size)
        else:
            self.var_position = None
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_size)
    
    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets=None):
        """
        Forward pass.
        
        Args:
            token_ids: [batch_size, seq_len] token indices
            segment_labels: [batch_size, seq_len] segment labels (0, 1, or 2)
            binary_pos: [batch_size, seq_len] binary-level positions
            function_pos: [batch_size, seq_len] function-level positions
            bb_pos: [batch_size, seq_len] basic block-level positions
            var_offsets: [batch_size, seq_len] var offset values (0 for non-var tokens)
            
        Returns:
            embedding: [batch_size, seq_len, embed_size]
        """
        # 1. Get token embeddings
        token_emb = self.token_embedding(token_ids)
        
        # 2. Get sequence positional embeddings (sinusoidal)
        seq_pos_emb = self.position_embedding(token_ids)
        
        # 3. Get address positional embeddings (3-level sinusoidal)
        if self.use_address_embedding:
            addr_pos_emb = self.address_position(binary_pos, function_pos, bb_pos)
        else:
            addr_pos_emb = 0
        
        # 4. Get var positional embeddings
        if self.use_var_embedding and var_offsets is not None:
            var_pos_emb = self.var_position(var_offsets)
        else:
            var_pos_emb = 0
        
        # Combine all embeddings
        embedding = token_emb + seq_pos_emb + addr_pos_emb + self.segment_embedding(segment_labels) + var_pos_emb
        
        # Apply layer norm and dropout
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        
        return embedding
