"""
Address-Aware Positional Embedding

Architecture follows a clear pipeline:
1. **Norm**: Normalize raw addresses to [0, 1] range at three hierarchical levels
   - Binary-level position (bnorm): Position within entire binary
   - Function-level position (fnorm): Position within function
   - Basic block-level position (bbnorm): Position within basic block

2. **Sin/Cos**: Apply sinusoidal encoding at multiple frequencies
   - Acts like a "microscope" that reveals both:
     * Low frequencies → global structure/contours
     * High frequencies → local details/textures
   - Each hierarchical level is encoded independently

3. **Concat**: Concatenate (not sum) embeddings from all 3 levels
   - Preserves distinct information from each hierarchy (function/block/instruction)
   - Like "assembling components" rather than "melting them together"

4. **MLP**: Project concatenated features to final embedding
   - Acts as a "controller" that reads the multi-level dashboard
   - Separate MLPs for 'address' (code/control flow) vs 'daddr' (data flow)
   - Translates hierarchical position readings into executable embeddings
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
    Address-aware positional embedding with hierarchical encoding.
    
    Pipeline:
    1. Norm: Normalize positions to [0, 1] (already normalized in input)
    2. Sin/Cos: Apply sinusoidal encoding to each hierarchical level independently
                Captures both low-freq (global structure) and high-freq (local details)
    3. Concat: Concatenate all 3 levels (binary, function, bb) instead of summing
               Preserves distinct information from each hierarchy level
    4. MLP: Project concatenated features to d_model dimension
            Separate MLPs for 'address' (code/control flow) vs 'daddr' (data flow)
    
    Takes three normalized position values:
    - binary_pos: [0, 1] position in binary (global context)
    - function_pos: [0, 1] position in function (medium context)
    - bb_pos: [0, 1] position in basic block (local context)
    """
    
    def __init__(self, d_model, max_len=512, dropout=0.1, num_sin_cos_features=8):
        """
        Args:
            d_model: Embedding dimension (output size)
            max_len: Maximum sequence length (unused, kept for compatibility)
            dropout: Dropout rate for MLP projection
            num_sin_cos_features: Number of sin/cos pairs per position level (default: 8)
                                 Each level gets 2*num_sin_cos_features dimensions (sin + cos)
        
        Note: Code addresses ('address' tokens) always use learnable null embedding for binary position.
              Data addresses ('daddr' tokens) always use real binary position.
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_sin_cos_features = num_sin_cos_features
        
        # Sin/Cos frequencies for multi-scale encoding
        # Lower frequencies capture global structure, higher frequencies capture fine details
        # Using exponentially spaced frequencies: 2^0, 2^1, 2^2, ..., 2^(num_features-1)
        self.register_buffer('frequencies', 
                           torch.pow(2.0, torch.arange(num_sin_cos_features).float()))
        
        # Learnable "null" embedding for binary position (used for code addresses)
        # This allows the model to explicitly learn "no binary position" representation
        # Different from zeros or encoding of position 0.0
        # Code addresses don't need global binary context, data addresses do
        self.null_binary_embedding = nn.Parameter(torch.randn(1, 1, 2 * num_sin_cos_features) * 0.02)
        
        # Each hierarchical level gets 2*num_sin_cos_features dimensions (sin + cos)
        # Total dimensions: 3 * 2 * num_sin_cos_features (always use 3 levels for MLP compatibility)
        # - num_sin_cos_features sin features
        # - num_sin_cos_features cos features
        # Total per level: 2 * num_sin_cos_features
        # Total concatenated: 3 levels * 2 * num_sin_cos_features
        sincos_dim_per_level = 2 * num_sin_cos_features
        total_sincos_dim = 3 * sincos_dim_per_level  # Always 3 hierarchical levels
        
        # Dual MLPs: separate projections for code vs data addresses
        # Input: concatenated sin/cos encodings from 3 hierarchical levels
        # Output: d_model dimensions
        
        # MLP for 'address' tokens (code addresses, control flow)
        self.code_address_projection = nn.Sequential(
            nn.Linear(total_sincos_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        # MLP for 'daddr' tokens (data addresses, data flow)
        self.data_address_projection = nn.Sequential(
            nn.Linear(total_sincos_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
    
    def _apply_sincos_encoding(self, normalized_pos):
        """
        Apply multi-scale sin/cos encoding to normalized positions.
        
        Args:
            normalized_pos: [batch_size, seq_len] positions in [0, 1]
            
        Returns:
            encoded: [batch_size, seq_len, 2*num_sin_cos_features] sin/cos features
        """
        # normalized_pos: [batch, seq]
        # frequencies: [num_sin_cos_features]
        # Result: [batch, seq, num_sin_cos_features]
        angles = normalized_pos.unsqueeze(-1) * self.frequencies * math.pi  # Scale by pi
        
        # Compute sin and cos
        sin_features = torch.sin(angles)  # [batch, seq, num_sin_cos_features]
        cos_features = torch.cos(angles)  # [batch, seq, num_sin_cos_features]
        
        # Interleave sin and cos: [sin0, cos0, sin1, cos1, ...]
        encoded = torch.stack([sin_features, cos_features], dim=-1)  # [batch, seq, num_features, 2]
        encoded = encoded.reshape(normalized_pos.shape[0], normalized_pos.shape[1], -1)  # [batch, seq, 2*num_features]
        
        return encoded
    
    def forward(self, binary_pos, function_pos, bb_pos, token_ids, vocab_stoi):
        """
        Forward pass with hierarchical address encoding.
        
        Pipeline:
        1. Norm: Input positions are already normalized to [0, 1]
        2. Sin/Cos: Apply sinusoidal encoding to each level independently
        3. Concat: Concatenate all 3 levels (preserves hierarchy information)
        4. MLP: Project through appropriate MLP (code vs data address)
        
        Args:
            binary_pos: [batch_size, seq_len] binary-level normalized positions (-1 for non-address tokens)
            function_pos: [batch_size, seq_len] function-level normalized positions (-1 for non-address tokens)
            bb_pos: [batch_size, seq_len] basic block-level normalized positions (-1 for non-address tokens)
            token_ids: [batch_size, seq_len] token indices
            vocab_stoi: dict mapping token strings to indices (for distinguishing address vs daddr)
            
        Returns:
            embedding: [batch_size, seq_len, d_model] hierarchical positional embedding
                      (zeros for non-address/daddr tokens where positions are -1)
        """
        # Create mask for address-like tokens (all three positions >= 0)
        address_mask = ((binary_pos >= 0) & (function_pos >= 0) & (bb_pos >= 0)).unsqueeze(-1)  # [batch, seq, 1]
        
        # Step 1: Norm - Clamp normalized positions to [0, 1] (they should already be normalized)
        binary_pos_norm = torch.clamp(binary_pos, 0.0, 1.0)
        function_pos_norm = torch.clamp(function_pos, 0.0, 1.0)
        bb_pos_norm = torch.clamp(bb_pos, 0.0, 1.0)
        
        # Step 2: Sin/Cos - Apply sinusoidal encoding to each hierarchical level
        # Each level is encoded independently to capture its specific frequency characteristics
        binary_sincos = self._apply_sincos_encoding(binary_pos_norm)      # [batch, seq, 2*num_features]
        function_sincos = self._apply_sincos_encoding(function_pos_norm)  # [batch, seq, 2*num_features]
        bb_sincos = self._apply_sincos_encoding(bb_pos_norm)              # [batch, seq, 2*num_features]
        
        # Step 3: Conditional binary_pos based on position value
        # If binary_pos was -1 (set by dataloader for code addresses), use null embedding
        # Otherwise use real sin/cos encoding (for data addresses)
        batch_size, seq_len = token_ids.shape
        null_embed = self.null_binary_embedding.expand(batch_size, seq_len, -1)  # [batch, seq, 2*num_features]
        
        # Create mask: True where binary_pos < 0 (i.e., -1.0 from dataloader)
        use_null_mask = (binary_pos < 0.0).unsqueeze(-1)  # [batch, seq, 1]
        
        # Use null embedding where binary_pos < 0, otherwise use sin/cos encoding
        binary_sincos = null_embed * use_null_mask + binary_sincos * (~use_null_mask)
        
        # Step 4: Concat - Concatenate all 3 hierarchical levels
        # This preserves distinct information from each level (unlike summing which mixes them)
        hierarchical_features = torch.cat([binary_sincos, function_sincos, bb_sincos], dim=-1)
        
        # Determine which projection to use based on token type:
        # - 'daddr' tokens → data_address_projection (memory operands, data flow)
        # - All other tokens with valid positions → code_address_projection
        #   This includes:
        #   * 'address' tokens (explicit code addresses)
        #   * Opcodes at addresses (mov, call, jmp, etc. at their instruction addresses)
        #   These are paired: opcode and its 'address' token share the same position,
        #   so they should use the same projection for consistency.
        daddr_token_id = vocab_stoi.get('daddr', -1)
        is_data_address = (token_ids == daddr_token_id).unsqueeze(-1).float()  # [batch, seq, 1]
        
        # Code address: any token with valid position info that is NOT daddr
        # This ensures opcodes and their paired 'address' tokens use the same embedding
        is_code_address = (address_mask.float() * (1.0 - is_data_address))  # [batch, seq, 1]
        
        # Step 5: MLP - Apply appropriate controller/projection based on token type
        # The MLP acts as a "controller" that translates hierarchical readings into embeddings
        code_embedding = self.code_address_projection(hierarchical_features)  # [batch, seq, d_model]
        data_embedding = self.data_address_projection(hierarchical_features)  # [batch, seq, d_model]
        
        # Combine embeddings based on token type
        # Note: is_code_address and is_data_address are mutually exclusive
        embedding = code_embedding * is_code_address + data_embedding * is_data_address
        
        return embedding
        


class VarPositionalEmbedding(nn.Module):
    """
    Positional embedding for var(0xXX) tokens with hybrid Embedding + MLP design.
    
    Design Philosophy:
    1. **In-Distribution (Precise)**: [-128, 128] uses Embedding Table
       - Each integer has independent semantics
       - 257 discrete values mapped to learnable embeddings
    
    2. **Out-of-Distribution (Approximate)**: |x| > 128 uses MLP
       - Only cares about "roughly how far"
       - Uses log-distance for smooth compression: log2(|x| - 128 + 1)
    
    Vocabulary Structure (259 tokens):
    - ID 0-256: Offsets [-128, +128] (257 values)
    - ID 257: [NEG_OVERFLOW] for x < -128
    - ID 258: [POS_OVERFLOW] for x > 128
    
    Final Embedding: E_final = Embedding(token_id) + MLP(log_distance)
    """
    
    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: Embedding dimension
            dropout: Dropout rate for MLP
        """
        super().__init__()
        self.d_model = d_model
        
        # Vocabulary parameters
        self.in_dist_min = -128
        self.in_dist_max = 128
        self.vocab_size = 259  # 257 precise values + 2 overflow tokens
        self.neg_overflow_id = 257
        self.pos_overflow_id = 258
        
        # Embedding table for in-distribution values and overflow tokens
        # ID 0-256: corresponds to offsets [-128, +128]
        # ID 257: [NEG_OVERFLOW]
        # ID 258: [POS_OVERFLOW]
        self.embedding_table = nn.Embedding(self.vocab_size, d_model, padding_idx=None)
        
        # MLP for out-of-distribution distance encoding
        # Input: 1 scalar (log-distance)
        # Output: d_model dimensions
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def _compute_token_id(self, var_offsets):
        """
        Convert var offsets to token IDs.
        
        Formula:
            ID = 257                if x < -128  (negative overflow)
            ID = x + 128            if -128 <= x <= 128  (in-distribution)
            ID = 258                if x > 128   (positive overflow)
        
        Args:
            var_offsets: [batch_size, seq_len] raw offset values
                        -1 for non-var tokens (sentinel value)
        
        Returns:
            token_ids: [batch_size, seq_len] mapped to vocab [0-258]
                      -1 for non-var tokens (will be masked out)
        """
        # Create a copy to avoid modifying input
        token_ids = var_offsets.clone()
        
        # Mask for valid var tokens (offset != -1)
        valid_mask = (var_offsets != -1)
        
        # Apply mapping only to valid tokens
        # Negative overflow: x < -128 -> ID 257
        neg_overflow_mask = valid_mask & (var_offsets < self.in_dist_min)
        token_ids[neg_overflow_mask] = self.neg_overflow_id
        
        # In-distribution: -128 <= x <= 128 -> ID = x + 128
        in_dist_mask = valid_mask & (var_offsets >= self.in_dist_min) & (var_offsets <= self.in_dist_max)
        token_ids[in_dist_mask] = var_offsets[in_dist_mask] + 128
        
        # Positive overflow: x > 128 -> ID 258
        pos_overflow_mask = valid_mask & (var_offsets > self.in_dist_max)
        token_ids[pos_overflow_mask] = self.pos_overflow_id
        
        return token_ids
    
    def _compute_log_distance(self, var_offsets):
        """
        Compute log-distance for out-of-distribution offsets.
        
        Formula:
            Dist = 0                        if -128 <= x <= 128
            Dist = log2(|x| - 128 + 1)      if |x| > 128
        
        Why subtract 128? To make overflow part start from 0 for smooth transition.
        Why use log? To compress large numbers (e.g., 65536 -> 16) for gradient stability.
        
        Args:
            var_offsets: [batch_size, seq_len] raw offset values
        
        Returns:
            log_distances: [batch_size, seq_len, 1] log-distance values
        """
        batch_size, seq_len = var_offsets.shape
        device = var_offsets.device
        
        # Initialize distances to 0
        distances = torch.zeros_like(var_offsets, dtype=torch.float32)
        
        # Compute distance only for out-of-distribution values
        # For negative overflow: dist = log2(|x| - 128 + 1)
        neg_overflow_mask = var_offsets < self.in_dist_min
        if neg_overflow_mask.any():
            neg_dist = torch.abs(var_offsets[neg_overflow_mask]) - 128 + 1
            distances[neg_overflow_mask] = torch.log2(neg_dist.float())
        
        # For positive overflow: dist = log2(|x| - 128 + 1)
        pos_overflow_mask = var_offsets > self.in_dist_max
        if pos_overflow_mask.any():
            pos_dist = var_offsets[pos_overflow_mask] - 128 + 1
            distances[pos_overflow_mask] = torch.log2(pos_dist.float())
        
        # In-distribution values keep distance = 0
        
        return distances.unsqueeze(-1)  # [batch, seq, 1]
    
    def forward(self, var_offsets):
        """
        Forward pass with hybrid Embedding + MLP encoding.
        
        Pipeline:
        1. Map offsets to token IDs ([-128,128] -> [0,256], overflow -> [257,258])
        2. Look up embeddings from table
        3. Compute log-distance for overflow values
        4. Project log-distance through MLP
        5. Combine: E_final = Embedding(ID) + MLP(Dist)
        
        Args:
            var_offsets: [batch_size, seq_len] variable offsets
                         -1 for non-var tokens (sentinel value)
                         Any integer for var tokens (e.g., var(0x10) -> 16, var(-0x31) -> -49)
            
        Returns:
            encoding: [batch_size, seq_len, d_model] hybrid embedding
                      (zeros for non-var tokens where offset = -1)
        """
        batch_size, seq_len = var_offsets.shape
        device = var_offsets.device
        
        # Create mask for var tokens (offset != -1)
        var_mask = (var_offsets != -1).unsqueeze(-1)  # [batch, seq, 1]
        
        # Step 1: Compute token IDs for embedding lookup
        token_ids = self._compute_token_id(var_offsets)  # [batch, seq]
        
        # Step 2: Look up embeddings (for non-var tokens, use ID 0 as placeholder)
        lookup_ids = torch.where(var_offsets == -1, 
                                torch.zeros_like(token_ids), 
                                token_ids)
        embedding_part = self.embedding_table(lookup_ids)  # [batch, seq, d_model]
        
        # Step 3: Compute log-distance for MLP
        log_distances = self._compute_log_distance(var_offsets)  # [batch, seq, 1]
        
        # Step 4: Project through MLP
        distance_part = self.distance_mlp(log_distances)  # [batch, seq, d_model]
        
        # Step 5: Combine embeddings
        # E_final = Embedding(ID) + MLP(Dist)
        encoding = embedding_part + distance_part
        
        # Zero out encoding for non-var tokens
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
    
    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=512, use_address_embedding=True, use_var_embedding=True, use_binary_pos=True, segment_types=256, vocab_stoi=None):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            use_address_embedding: Whether to use address-aware positional embeddings
            use_var_embedding: Whether to use var offset embeddings
            use_binary_pos: DEPRECATED - kept for backward compatibility, has no effect.
                           Code addresses always use null embedding, data addresses use real binary_pos.
            segment_types: Number of segment types (instruction IDs). Default 256 to handle long sequences.
            vocab_stoi: Vocabulary string-to-index mapping (needed for address vs daddr distinction)
        
        Note: Code addresses ('address' tokens) always use learnable null embedding for binary position.
              Data addresses ('daddr' tokens) always use real binary position from dataloader.
              This is controlled by dataloader setting binary_pos=-1 for code addresses.
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
        # Note: AddressPositionalEmbedding always creates null_binary_embedding
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
