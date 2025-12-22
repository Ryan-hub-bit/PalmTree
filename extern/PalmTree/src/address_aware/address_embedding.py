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
    Standard sinusoidal positional encoding for sequence order (like PalmTree).
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
        
        # MLP to project concatenated encodings (3 * intermediate_size) to d_model FOR REGULAR ADDRESS
        concat_size = 3 * intermediate_size
        self.projection = nn.Sequential(
            nn.Linear(concat_size, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),  # Add dropout after activation
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)   # Add dropout at output
        )
        
        # SEPARATE MLP for daddr tokens (data addresses have different semantics)
        self.daddr_projection = nn.Sequential(
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
    
    def forward(self, binary_pos, function_pos, bb_pos, is_daddr=None):
        """
        Forward pass with three levels of address positions.
        
        Args:
            binary_pos: [batch_size, seq_len] binary-level normalized positions (-1 for non-address tokens)
            function_pos: [batch_size, seq_len] function-level normalized positions (-1 for non-address tokens)
            bb_pos: [batch_size, seq_len] basic block-level normalized positions (-1 for non-address tokens)
            is_daddr: [batch_size, seq_len] binary indicator (1 for daddr, 0 for others), optional
            
        Returns:
            embedding: [batch_size, seq_len, d_model] combined positional embedding
                      (zeros for non-address tokens where positions are -1)
                      (uses daddr_projection for daddr tokens when is_daddr provided)
        """
        # Create mask for address tokens (all three positions >= 0)
        # If any position is -1, the token is not an address-like token
        address_mask = ((binary_pos >= 0) & (function_pos >= 0) & (bb_pos >= 0)).unsqueeze(-1)  # [batch, seq, 1]
        
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
        
        # Project to d_model with separate MLPs for daddr vs regular address
        if is_daddr is not None:
            # Apply regular projection
            embedding_regular = self.projection(concatenated)
            # Apply daddr-specific projection
            embedding_daddr = self.daddr_projection(concatenated)
            
            # Mix based on is_daddr mask: [batch, seq, 1]
            daddr_mask = (is_daddr == 1).unsqueeze(-1).float()
            embedding = embedding_regular * (1 - daddr_mask) + embedding_daddr * daddr_mask
        else:
            # No is_daddr provided, use regular projection for all
            embedding = self.projection(concatenated)
        
        # Zero out embedding for non-address tokens (where address_mask is False)
        embedding = embedding * address_mask.float()
        
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
            var_offsets: [batch_size, seq_len] variable offsets
                         -1 for non-var tokens
                         >= 0 for var tokens (e.g., for var(0x10), offset = 16; for var(0x0), offset = 0)
            
        Returns:
            encoding: [batch_size, seq_len, d_model] sinusoidal encoding for var offsets
                      (zeros for non-var tokens where offset = -1)
        """
        batch_size, seq_len = var_offsets.shape
        device = var_offsets.device
        
        # Create output tensor initialized to zeros
        encoding = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Create mask for var tokens (offset >= 0)
        var_mask = (var_offsets >= 0).unsqueeze(-1)  # [batch, seq, 1]
        
        # For var tokens: use their actual offset value for sinusoidal encoding
        # For non-var tokens (offset = -1): the encoding will be zeroed out by the mask anyway
        offsets_scaled = var_offsets.float().unsqueeze(-1)  # [batch, seq, 1]
        
        # Apply sin to even indices
        encoding[:, :, 0::2] = torch.sin(offsets_scaled * self.div_term)
        
        # Apply cos to odd indices
        if self.d_model % 2 == 0:
            encoding[:, :, 1::2] = torch.cos(offsets_scaled * self.div_term)
        else:
            encoding[:, :, 1::2] = torch.cos(offsets_scaled * self.div_term[:-1])
        
        # Zero out encoding for non-var tokens (where var_mask is False)
        # This ensures non-var tokens (offset = -1) get all zeros regardless of sin/cos(-1)
        encoding = encoding * var_mask.float()
        
        return encoding


class AddressAwareBERTEmbedding(nn.Module):
    """
    Address-aware BERT Embedding.
    
    Combines embeddings (ALL trained from scratch):
    1. Token embedding
    2. Sequence positional embedding (sinusoidal)
    3. Address positional embedding (sin/cos on binary_pos, function_pos, bb_pos) - OPTIONAL
    4. Segment embedding (for NSP task)
    5. Var positional embedding (sin/cos on var offsets for var(0xXX) tokens) - OPTIONAL
    """
    
    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=512, use_address_embedding=True, use_var_embedding=True):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            use_address_embedding: Whether to use address-aware positional embeddings
            use_var_embedding: Whether to use var offset embeddings
        """
        super().__init__()
        
        self.embed_size = embed_size
        self.use_address_embedding = use_address_embedding
        self.use_var_embedding = use_var_embedding
        
        # 1. Token embedding - trained from scratch
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # 2. Sequence positional embedding - sinusoidal (standard BERT style)
        self.position_embedding = SequencePositionalEmbedding(embed_size, max_len)
        
        # 3. Address-aware positional embedding - sin/cos encoding on 3 levels (OPTIONAL)
        if self.use_address_embedding:
            self.address_position = AddressPositionalEmbedding(embed_size, max_len, dropout=dropout)
        else:
            self.address_position = None
        
        # 4. Segment embedding (supports both NSP and instruction-level segments)
        # Increased from 2 to 16 to support instruction IDs (1-8) plus padding (0)
        self.segment_embedding = nn.Embedding(16, embed_size, padding_idx=0)
        
        # 5. Var positional embedding - sin/cos encoding on var offsets (OPTIONAL)
        if self.use_var_embedding:
            self.var_position = VarPositionalEmbedding(embed_size)
        else:
            self.var_position = None
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_size)
    
    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets=None, is_daddr=None):
        """
        Forward pass.
        
        Args:
            token_ids: [batch_size, seq_len] token indices
            segment_labels: [batch_size, seq_len] segment labels (0 or 1)
            binary_pos: [batch_size, seq_len] binary-level positions
            function_pos: [batch_size, seq_len] function-level positions
            bb_pos: [batch_size, seq_len] basic block-level positions
            var_offsets: [batch_size, seq_len] var offset values (0 for non-var tokens)
            is_daddr: [batch_size, seq_len] binary indicator (1 for daddr, 0 for others), optional
            
        Returns:
            embedding: [batch_size, seq_len, embed_size]
        """
        batch_size, seq_len = token_ids.size()
        
        # 1. Get token embeddings (trained from scratch)
        token_emb = self.token_embedding(token_ids)
        
        # 2. Get sequence positional embeddings (sinusoidal)
        seq_pos_emb = self.position_embedding(token_ids)
        
        # 3. Get address-aware positional embeddings if enabled (with daddr support)
        if self.use_address_embedding and self.address_position is not None:
            addr_pos_emb = self.address_position(binary_pos, function_pos, bb_pos, is_daddr)
        else:
            addr_pos_emb = 0
        
        # 4. Get segment embeddings
        segment_emb = self.segment_embedding(segment_labels)
        
        # 5. Get var positional embeddings if enabled
        if self.use_var_embedding and self.var_position is not None and var_offsets is not None:
            var_pos_emb = self.var_position(var_offsets)
        else:
            var_pos_emb = 0
        
        # Combine all embeddings
        embedding = token_emb + seq_pos_emb + addr_pos_emb + segment_emb + var_pos_emb
        
        embedding = self.dropout(self.layer_norm(embedding))
        
        return embedding
        
        # 3. Get address positional embeddings (3-level sinusoidal) - OPTIONAL
        if self.use_address_embedding:
            addr_pos_emb = self.address_position(binary_pos, function_pos, bb_pos)
        else:
            addr_pos_emb = 0
        
        # 4. Get var positional embeddings - OPTIONAL
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
