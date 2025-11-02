"""
Address-Aware PalmTree Model
Extends PalmTree with address encoding
"""
import torch
import torch.nn as nn
import sys
import os

# Add PalmTree to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from palmtree.model.bert import BERT
import config


class AddressAwarePalmTree(nn.Module):
    """
    PalmTree model with 3-level embeddings:
    1. Semantic level (token embeddings from PalmTree)
    2. Address level (address type and value encodings)
    3. Positional level (position in sequence)
    """
    
    def __init__(self, palmtree_bert: BERT):
        super().__init__()
        self.palmtree = palmtree_bert
        self.hidden_size = config.HIDDEN_SIZE
        
        # Level 1: Semantic embeddings (from PalmTree - already has token embeddings)
        # We'll use palmtree's token embeddings directly
        
        # Level 2: Address-level embeddings
        # Address type embedding (code, data, start, end, etc.)
        self.addr_type_embedding = nn.Embedding(
            num_embeddings=len(config.ADDR_TYPE_LABELS),
            embedding_dim=self.hidden_size
        )
        # Address value encoding projection (sinusoidal encoding of actual address)
        self.addr_value_projection = nn.Linear(config.ADDRESS_ENCODING_DIM, self.hidden_size)
        
        # Level 3: Positional embeddings
        # Enhanced positional encoding (beyond standard transformer positional encoding)
        self.position_embedding = nn.Embedding(
            num_embeddings=config.MAX_SEQ_LENGTH,
            embedding_dim=self.hidden_size
        )
        
        # Combine the three levels with learned weights
        self.level_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # Task-specific heads
        self.next_bb_head = nn.Linear(self.hidden_size, config.PALMTREE_VOCAB_SIZE)
        self.addr_type_head = nn.Linear(self.hidden_size, len(config.ADDR_TYPE_LABELS))
        self.edge_type_head = nn.Linear(self.hidden_size, len(config.EDGE_TYPE_LABELS))
        
    def forward(self, input_ids, attention_mask, address_encodings, addr_type_ids, position_ids=None):
        """
        3-level embedding forward pass
        
        Args:
            input_ids: [batch_size, seq_len] - token IDs
            attention_mask: [batch_size, seq_len] - attention mask
            address_encodings: [batch_size, seq_len, addr_enc_dim] - sinusoidal address encodings
            addr_type_ids: [batch_size, seq_len] - address type IDs (code/data/start/end/etc)
            position_ids: [batch_size, seq_len] - position indices (auto-generated if None)
        
        Returns:
            dict with task outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Level 1: Semantic embeddings from PalmTree BERT
        # PalmTree BERT uses segment_info (all ones for valid tokens)
        segment_info = attention_mask.clone()
        semantic_embeddings = self.palmtree(input_ids, segment_info)  # [B, L, H]
        
        # Level 2: Address-level embeddings
        # 2a. Address type embeddings
        addr_type_embeds = self.addr_type_embedding(addr_type_ids)  # [B, L, H]
        # 2b. Address value encodings (sinusoidal encoding of actual addresses)
        addr_value_embeds = self.addr_value_projection(address_encodings)  # [B, L, H]
        # Combine address type and value
        address_level_embeddings = addr_type_embeds + addr_value_embeds  # [B, L, H]
        
        # Level 3: Positional embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # [B, L]
        positional_embeddings = self.position_embedding(position_ids)  # [B, L, H]
        
        # Fuse all three levels
        # Concatenate: [semantic | address | positional] -> [B, L, 3*H]
        combined = torch.cat([
            semantic_embeddings,
            address_level_embeddings,
            positional_embeddings
        ], dim=-1)
        
        # Fuse with learned transformation
        fused_features = self.level_fusion(combined)  # [B, L, H]
        
        # Task-specific predictions
        next_bb_logits = self.next_bb_head(fused_features)
        addr_type_logits = self.addr_type_head(fused_features)
        
        # Use [CLS] token for sequence-level prediction
        cls_features = fused_features[:, 0, :]
        edge_type_logits = self.edge_type_head(cls_features)
        
        return {
            'next_bb_logits': next_bb_logits,
            'addr_type_logits': addr_type_logits,
            'edge_type_logits': edge_type_logits,
            'hidden_states': fused_features,
            'semantic_embeddings': semantic_embeddings,
            'address_embeddings': address_level_embeddings,
            'positional_embeddings': positional_embeddings,
        }


def load_pretrained_palmtree(model_path: str, vocab_size: int, hidden: int = 768, n_layers: int = 12, attn_heads: int = 12):
    """Load pre-trained PalmTree BERT model"""
    
    # Load pre-trained checkpoint (the checkpoint itself is a BERT model)
    # Use weights_only=False for older PyTorch checkpoints
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # The checkpoint is already a BERT model, just return it
    # Set to eval mode for using pre-trained features
    checkpoint.eval()
    
    return checkpoint
