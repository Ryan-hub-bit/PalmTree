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
        
        # MLP to project concatenated encodings (3 * intermediate_size) to d_model
        concat_size = 3 * intermediate_size
        self.projection = nn.Sequential(
            nn.Linear(concat_size, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),  # Add dropout after activation
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)   # Add dropout at output
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


class AddressAwareBERTEmbedding(nn.Module):
    """
    Address-aware BERT Embedding.
    
    Combines FOUR types of embeddings:
    1. Token embedding (from pre-trained PalmTree - FROZEN)
    2. Sequence positional embedding (from PalmTree - FROZEN)
    3. Address positional embedding (NEW - sin/cos on bnorm, fnorm, bbnorm - TRAINABLE)
    4. Segment embedding (for NSP task - can use pre-trained or new)
    """
    
    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=512, 
                 pretrained_token_emb=None, pretrained_position_emb=None, pretrained_segment_emb=None):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            pretrained_token_emb: Pre-trained token embedding from PalmTree (will be FROZEN)
            pretrained_position_emb: Pre-trained position embedding from PalmTree (will be FROZEN)
            pretrained_segment_emb: Pre-trained segment embedding from PalmTree (optional)
        """
        super().__init__()
        
        self.embed_size = embed_size
        
        # 1. Token embedding - from PalmTree (FROZEN)
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        if pretrained_token_emb is not None:
            print(f"[INFO] Loading pre-trained token embeddings: {pretrained_token_emb.shape}")
            self.token_embedding.weight.data.copy_(pretrained_token_emb)
        
        # 2. Sequence positional embedding - from PalmTree (FROZEN)
        if pretrained_position_emb is not None:
            print(f"[INFO] Loading pre-trained position embeddings: {pretrained_position_emb.shape}")
            # Create learnable embedding and load pre-trained weights
            self.position_embedding = nn.Embedding(max_len, embed_size)
            # PalmTree position embedding has shape [1, max_len, embed_size], squeeze the batch dim
            if pretrained_position_emb.dim() == 3:
                pretrained_position_emb_squeezed = pretrained_position_emb.squeeze(0)
            else:
                pretrained_position_emb_squeezed = pretrained_position_emb
            # Copy weights (slice to max_len if needed)
            with torch.no_grad():
                self.position_embedding.weight.copy_(pretrained_position_emb_squeezed[:max_len])
        else:
            # Create standard sinusoidal position embedding
            self.position_embedding = SequencePositionalEmbedding(embed_size, max_len)
        
        # 3. Address-aware positional embedding - NEW component (TRAINABLE)
        self.address_position = AddressPositionalEmbedding(embed_size, max_len, dropout=dropout)
        print(f"[INFO] Address positional embeddings (3-level) - TRAINABLE")
        
        # 4. Segment embedding (for NSP)
        self.segment_embedding = nn.Embedding(2, embed_size)
        if pretrained_segment_emb is not None:
            print(f"[INFO] Loading pre-trained segment embeddings: {pretrained_segment_emb.shape}")
            # Only copy first 2 segment embeddings (PalmTree may have more)
            with torch.no_grad():
                self.segment_embedding.weight.data.copy_(pretrained_segment_emb[:2])
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_size)
    
    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos):
        """
        Forward pass.
        
        Args:
            token_ids: [batch_size, seq_len] token indices
            segment_labels: [batch_size, seq_len] segment labels (0 or 1)
            binary_pos: [batch_size, seq_len] binary-level positions
            function_pos: [batch_size, seq_len] function-level positions
            bb_pos: [batch_size, seq_len] basic block-level positions
            
        Returns:
            embedding: [batch_size, seq_len, embed_size]
        """
        batch_size, seq_len = token_ids.size()
        
        # 1. Get token embeddings (FROZEN - from PalmTree)
        token_emb = self.token_embedding(token_ids)
        
        # 2. Get sequence positional embeddings (FROZEN - from PalmTree)
        if isinstance(self.position_embedding, nn.Embedding):
            # Learnable position embedding loaded from PalmTree
            position_ids = torch.arange(seq_len, dtype=torch.long, device=token_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            seq_pos_emb = self.position_embedding(position_ids)
        else:
            # Sinusoidal position embedding
            seq_pos_emb = self.position_embedding(token_ids)
        
        # 3. Get address positional embeddings (TRAINABLE - NEW component)
        addr_pos_emb = self.address_position(binary_pos, function_pos, bb_pos)
        
        # 4. Get segment embeddings
        segment_emb = self.segment_embedding(segment_labels)
        
        # Combine all four: token + seq_position + addr_position + segment
        embedding = token_emb + seq_pos_emb + addr_pos_emb + segment_emb
        
        # Apply layer norm and dropout
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        
        return embedding
