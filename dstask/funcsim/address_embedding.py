import torch
import torch.nn as nn
import math

class SequencePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class AddressPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512, intermediate_size=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_per_level = intermediate_size
        self.level_weights = nn.Parameter(torch.ones(3))
        concat_size = 3 * intermediate_size
        self.projection = nn.Sequential(
            nn.Linear(concat_size, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
    def _sinusoidal_encoding(self, positions, d_model):
        batch_size, seq_len = positions.shape
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=positions.device) * -(math.log(10000.0) / d_model))
        positions_scaled = positions.unsqueeze(-1) * 10000.0
        encoding = torch.zeros(batch_size, seq_len, d_model, device=positions.device)
        encoding[:, :, 0::2] = torch.sin(positions_scaled * div_term)
        if d_model % 2 == 0:
            encoding[:, :, 1::2] = torch.cos(positions_scaled * div_term)
        else:
            encoding[:, :, 1::2] = torch.cos(positions_scaled * div_term[:-1])
        return encoding
    def forward(self, binary_pos, function_pos, bb_pos):
        binary_enc = self._sinusoidal_encoding(binary_pos, self.d_per_level)
        function_enc = self._sinusoidal_encoding(function_pos, self.d_per_level)
        bb_enc = self._sinusoidal_encoding(bb_pos, self.d_per_level)
        weights = torch.softmax(self.level_weights, dim=0)
        binary_enc = binary_enc * weights[0]
        function_enc = function_enc * weights[1]
        bb_enc = bb_enc * weights[2]
        concatenated = torch.cat([binary_enc, function_enc, bb_enc], dim=-1)
        embedding = self.projection(concatenated)
        return embedding

class VarPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_offset=4096):
        super().__init__()
        self.d_model = d_model
        self.max_offset = max_offset
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
    def forward(self, var_offsets):
        batch_size, seq_len = var_offsets.shape
        device = var_offsets.device
        encoding = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        var_mask = (var_offsets > 0).unsqueeze(-1)
        offsets_scaled = var_offsets.float().unsqueeze(-1)
        encoding[:, :, 0::2] = torch.sin(offsets_scaled * self.div_term)
        if self.d_model % 2 == 0:
            encoding[:, :, 1::2] = torch.cos(offsets_scaled * self.div_term)
        else:
            encoding[:, :, 1::2] = torch.cos(offsets_scaled * self.div_term[:-1])
        encoding = encoding * var_mask.float()
        return encoding

class AddressAwareBERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=512, use_address_embedding=True, use_var_embedding=True):
        super().__init__()
        self.embed_size = embed_size
        self.use_address_embedding = use_address_embedding
        self.use_var_embedding = use_var_embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_embedding = SequencePositionalEmbedding(embed_size, max_len)
        if self.use_address_embedding:
            self.address_position = AddressPositionalEmbedding(embed_size, max_len, dropout=dropout)
        else:
            self.address_position = None
        self.segment_embedding = nn.Embedding(16, embed_size, padding_idx=0)
        if self.use_var_embedding:
            self.var_position = VarPositionalEmbedding(embed_size)
        else:
            self.var_position = None
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_size)
    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets=None):
        batch_size, seq_len = token_ids.size()
        token_emb = self.token_embedding(token_ids)
        seq_pos_emb = self.position_embedding(token_ids)
        if self.use_address_embedding:
            addr_pos_emb = self.address_position(binary_pos, function_pos, bb_pos)
        else:
            addr_pos_emb = 0
        segment_emb = self.segment_embedding(segment_labels)
        if self.use_var_embedding and var_offsets is not None:
            var_pos_emb = self.var_position(var_offsets)
        else:
            var_pos_emb = 0
        embedding = token_emb + seq_pos_emb + addr_pos_emb + segment_emb + var_pos_emb
        embedding = self.layer_norm(embedding)var_pos_emb
        embedding = self.dropout(embedding)
        return embedding
