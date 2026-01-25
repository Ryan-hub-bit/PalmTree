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
    Address-aware positional embedding using direct MLP projection on 3 hierarchical positions.
    
    Uses dual MLPs to distinguish between:
    - 'address' tokens (code addresses, control flow)
    - 'daddr' tokens (data addresses, data flow)
    
    Takes three normalized position values:
    - binary_pos: [0, 1] position in binary
    - function_pos: [0, 1] position in function  
    - bb_pos: [0, 1] position in basic block
    
    Each token type gets projected through its own MLP (3 floats → d_model).
    """
    
    def __init__(self, d_model, max_len=512, dropout=0.1):
        """
        Args:
            d_model: Embedding dimension (output size)
            max_len: Maximum sequence length (unused, kept for compatibility)
            dropout: Dropout rate for MLP projection
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Dual MLPs: separate projections for code vs data addresses
        # Input: 3 floats (binary_pos, function_pos, bb_pos)
        # Output: d_model dimensions
        
        # MLP for 'address' tokens (code addresses, control flow)
        self.code_address_projection = nn.Sequential(
            nn.Linear(3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        # MLP for 'daddr' tokens (data addresses, data flow)
        self.data_address_projection = nn.Sequential(
            nn.Linear(3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, binary_pos, function_pos, bb_pos, token_ids, vocab_stoi):
        """
        Forward pass with three levels of address positions.
        Uses dual MLPs to distinguish code addresses from data addresses.
        
        Args:
            binary_pos: [batch_size, seq_len] binary-level normalized positions (-1 for non-address tokens)
            function_pos: [batch_size, seq_len] function-level normalized positions (-1 for non-address tokens)
            bb_pos: [batch_size, seq_len] basic block-level normalized positions (-1 for non-address tokens)
            token_ids: [batch_size, seq_len] token indices
            vocab_stoi: dict mapping token strings to indices (for distinguishing address vs daddr)
            
        Returns:
            embedding: [batch_size, seq_len, d_model] combined positional embedding
                      (zeros for non-address/daddr tokens where positions are -1)
        """
        # Create mask for address-like tokens (all three positions >= 0)
        address_mask = ((binary_pos >= 0) & (function_pos >= 0) & (bb_pos >= 0)).unsqueeze(-1)  # [batch, seq, 1]
        
        # Stack the 3 position values as a single input vector
        # [batch_size, seq_len, 3] where each position is in [0, 1]
        positions = torch.stack([binary_pos, function_pos, bb_pos], dim=-1)
        
        # Determine which tokens are 'address' vs 'daddr'
        # Get indices for 'address' and 'daddr' tokens from vocab
        address_token_id = vocab_stoi.get('address', -1)
        daddr_token_id = vocab_stoi.get('daddr', -1)
        
        # Create masks for code and data addresses
        is_code_address = (token_ids == address_token_id).unsqueeze(-1).float()  # [batch, seq, 1]
        is_data_address = (token_ids == daddr_token_id).unsqueeze(-1).float()   # [batch, seq, 1]
        
        # Apply appropriate MLP based on token type
        code_embedding = self.code_address_projection(positions)  # [batch, seq, d_model]
        data_embedding = self.data_address_projection(positions)  # [batch, seq, d_model]
        
        # Combine embeddings based on token type
        embedding = code_embedding * is_code_address + data_embedding * is_data_address
        
        # Zero out embedding for non-address tokens (where address_mask is False)
        embedding = embedding * address_mask.float()
        
        return embedding
        


class VarPositionalEmbedding(nn.Module):
    """
    Positional embedding for var(0xXX) tokens based on their offset values.
    
    Uses direct MLP projection on raw offset values.
    Clamps extremely large offsets to prevent numerical issues.
    Only var tokens get the encoding, all other tokens get zeros.
    """
    
    def __init__(self, d_model, max_offset=8192, dropout=0.1):
        """
        Args:
            d_model: Embedding dimension
            max_offset: Maximum offset value (larger values are clamped)
            dropout: Dropout rate for MLP
        """
        super().__init__()
        self.d_model = d_model
        self.max_offset = max_offset
        
        # Direct MLP projection: 1 scalar (raw offset) -> d_model
        self.projection = nn.Sequential(
            nn.Linear(1, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, var_offsets):
        """
        Forward pass with raw offset values (clamped to max_offset).
        
        Args:
            var_offsets: [batch_size, seq_len] variable offsets
                         -1 for non-var tokens
                         >= 0 for var tokens (e.g., for var(0x10), offset = 16; for var(0x0), offset = 0)
            
        Returns:
            encoding: [batch_size, seq_len, d_model] MLP-projected encoding for var offsets
                      (zeros for non-var tokens where offset = -1)
        """
        batch_size, seq_len = var_offsets.shape
        device = var_offsets.device
        
        # Create mask for var tokens (offset != -1, sentinel value for non-var)
        var_mask = (var_offsets != -1).unsqueeze(-1)  # [batch, seq, 1]
        
        # Use raw offset values, clamped to [-max_offset, max_offset]
        # This preserves exact differences for nearby offsets (e.g., var(16) vs var(22))
        # and supports negative offsets for local variables (e.g., var(-49) for [rbp-0x31])
        offsets_float = var_offsets.float()
        offsets_float = torch.clamp(offsets_float, min=-float(self.max_offset), max=float(self.max_offset))
        # Set sentinel value -1 to 0 to avoid affecting the MLP
        offsets_float = torch.where(var_offsets == -1, torch.zeros_like(offsets_float), offsets_float)
        encoded = offsets_float.unsqueeze(-1)  # [batch, seq, 1]
        
        # Project through MLP
        encoding = self.projection(encoded)  # [batch, seq, d_model]
        
        # Zero out encoding for non-var tokens (where var_mask is False)
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
    
    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=512, use_address_embedding=True, use_var_embedding=True, segment_types=256, vocab_stoi=None):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            use_address_embedding: Whether to use address-aware positional embeddings
            use_var_embedding: Whether to use var offset embeddings
            segment_types: Number of segment types (instruction IDs). Default 256 to handle long sequences.
            vocab_stoi: Vocabulary string-to-index mapping (needed for address vs daddr distinction)
        """
        super().__init__()
        
        self.embed_size = embed_size
        self.use_address_embedding = use_address_embedding
        self.use_var_embedding = use_var_embedding
        self.vocab_stoi = vocab_stoi  # Will be set after vocab is loaded
        
        # 1. Token embedding - trained from scratch
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # 2. Sequence positional embedding - sinusoidal (standard BERT style)
        self.position_embedding = SequencePositionalEmbedding(embed_size, max_len)
        
        # 3. Address-aware positional embedding - sin/cos encoding on 3 levels (OPTIONAL)
        if self.use_address_embedding:
            self.address_position = AddressPositionalEmbedding(embed_size, max_len, dropout=dropout)
        else:
            self.address_position = None
        
        # 4. Segment embedding (supports instruction-level segments)
        # Supports up to segment_types instruction IDs (default 256 to handle long sequences)
        self.segment_embedding = nn.Embedding(segment_types, embed_size)
        
        # 5. Var positional embedding - direct MLP on var offsets (OPTIONAL)
        if self.use_var_embedding:
            self.var_position = VarPositionalEmbedding(embed_size, dropout=dropout)
        else:
            self.var_position = None
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_size)
    
    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets=None):
        """
        Forward pass.
        
        Args:
            token_ids: [batch_size, seq_len] token indices
            segment_labels: [batch_size, seq_len] segment labels (0 or 1)
            binary_pos: [batch_size, seq_len] binary-level positions
            function_pos: [batch_size, seq_len] function-level positions
            bb_pos: [batch_size, seq_len] basic block-level positions
            var_offsets: [batch_size, seq_len] var offset values (0 for non-var tokens)
            
        Returns:
            embedding: [batch_size, seq_len, embed_size]
        """
        batch_size, seq_len = token_ids.size()
        
        # VALIDATION: Check for invalid token IDs before embedding lookup
        max_token_id = token_ids.max().item()
        min_token_id = token_ids.min().item()
        vocab_size = self.token_embedding.num_embeddings
        
        if max_token_id >= vocab_size or min_token_id < 0:
            print(f"\n{'='*80}")
            print(f"INVALID TOKEN ID DETECTED IN FORWARD PASS")
            print(f"{'='*80}")
            print(f"Vocab size: {vocab_size}")
            print(f"Max token ID in batch: {max_token_id}")
            print(f"Min token ID in batch: {min_token_id}")
            print(f"Token IDs shape: {token_ids.shape}")
            
            # Find all invalid positions
            invalid_mask = (token_ids >= vocab_size) | (token_ids < 0)
            if invalid_mask.any():
                invalid_positions = torch.nonzero(invalid_mask, as_tuple=False)
                print(f"\nNumber of invalid token IDs: {invalid_mask.sum().item()}")
                print(f"First 10 invalid positions (batch_idx, seq_idx):")
                for i, (batch_idx, seq_idx) in enumerate(invalid_positions[:10]):
                    invalid_id = token_ids[batch_idx, seq_idx].item()
                    print(f"  [{batch_idx.item()}, {seq_idx.item()}] = {invalid_id}")
                
                # Show context around first invalid token
                batch_idx, seq_idx = invalid_positions[0][0].item(), invalid_positions[0][1].item()
                start_idx = max(0, seq_idx - 5)
                end_idx = min(seq_len, seq_idx + 6)
                print(f"\nContext around first invalid token (batch {batch_idx}, position {seq_idx}):")
                print(f"Token IDs: {token_ids[batch_idx, start_idx:end_idx].tolist()}")
                print(f"Segment labels: {segment_labels[batch_idx, start_idx:end_idx].tolist()}")
            print(f"{'='*80}\n")
            
            raise ValueError(f"Token ID out of range: min={min_token_id}, max={max_token_id}, vocab_size={vocab_size}")
        
        # 1. Get token embeddings (trained from scratch)
        token_emb = self.token_embedding(token_ids)
        
        # 2. Get sequence positional embeddings (sinusoidal)
        seq_pos_emb = self.position_embedding(token_ids)
        
        # 3. Get address positional embeddings (direct MLP on 3 hierarchical positions) - OPTIONAL
        if self.use_address_embedding:
            if self.vocab_stoi is None:
                raise ValueError("vocab_stoi must be provided to distinguish address vs daddr tokens")
            addr_pos_emb = self.address_position(binary_pos, function_pos, bb_pos, token_ids, self.vocab_stoi)
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
