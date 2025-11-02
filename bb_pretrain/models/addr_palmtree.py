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
    
    def __init__(self, palmtree_bert: BERT, extended_vocab_size: int = None):
        super().__init__()
        self.palmtree = palmtree_bert
        self.hidden_size = config.HIDDEN_SIZE
        
        # Level 1: Semantic embeddings
        # For regular tokens: use PalmTree's pre-trained embeddings (frozen)
        # For address tokens: add NEW trainable embeddings
        
        # Get original vocab size from PalmTree
        self.palmtree_vocab_size = self.palmtree.embedding.token.num_embeddings
        
        # Extended vocab: original + new address tokens
        self.extended_vocab_size = extended_vocab_size or (self.palmtree_vocab_size + len(config.SPECIAL_ADDR_TOKENS))
        
        # NEW: Trainable embeddings for address tokens (initialized randomly)
        # These will be trained through your pretraining tasks!
        num_new_tokens = self.extended_vocab_size - self.palmtree_vocab_size
        if num_new_tokens > 0:
            self.addr_token_embeddings = nn.Embedding(
                num_embeddings=num_new_tokens,
                embedding_dim=self.hidden_size
            )
            print(f"Added {num_new_tokens} trainable address token embeddings")
        else:
            self.addr_token_embeddings = None
        
        # Level 2: Address-level embeddings (VALUE only, not type)
        # No more addr_type_embedding - type is now in semantic level!
        # Only keep address VALUE encoding
        self.addr_value_projection = nn.Linear(config.ADDRESS_ENCODING_DIM, self.hidden_size)
        
        # Level 3: Positional embeddings
        # Enhanced positional encoding (beyond standard transformer positional encoding)
        self.position_embedding = nn.Embedding(
            num_embeddings=config.MAX_SEQ_LENGTH,
            embedding_dim=self.hidden_size
        )
        
        # Simple addition fusion (like BERT: token + position)
        # No MLP, just sum the three levels
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Task-specific heads for learning address semantics and relationships
        # Task 1: Masked Token Prediction (regular tokens + address tokens)
        # Predict over EXTENDED vocabulary
        self.next_bb_head = nn.Linear(self.hidden_size, self.extended_vocab_size)
        
        # Task 2: Control Flow Edge Type Prediction
        self.edge_type_head = nn.Linear(self.hidden_size, len(config.EDGE_TYPE_LABELS))
        
        # Task 3: Address Distance Prediction (learn spatial relationships)
        # Classify distance as: same-bb, near (<1KB), medium (<64KB), far (>64KB)
        self.addr_distance_head = nn.Linear(self.hidden_size * 2, 4)  # Takes 2 address embeddings
    
    def _get_token_embeddings(self, input_ids):
        """
        Get token embeddings:
        - Regular tokens (< palmtree_vocab_size): use PalmTree's pre-trained embeddings
        - Address tokens (>= palmtree_vocab_size): use our trainable embeddings
        
        Args:
            input_ids: [B, L] - token IDs
        
        Returns:
            embeddings: [B, L, H] - combined embeddings
        """
        batch_size, seq_len = input_ids.shape
        embeddings = torch.zeros(batch_size, seq_len, self.hidden_size, device=input_ids.device)
        
        # Mask for regular tokens vs address tokens
        regular_mask = input_ids < self.palmtree_vocab_size
        addr_mask = input_ids >= self.palmtree_vocab_size
        
        # Get PalmTree embeddings for regular tokens (pre-trained)
        if regular_mask.any():
            regular_ids = input_ids.clone()
            regular_ids[~regular_mask] = 0  # Set non-regular to 0 (will be replaced)
            regular_embeds = self.palmtree.embedding.token(regular_ids)  # [B, L, H]
            embeddings[regular_mask] = regular_embeds[regular_mask]
        
        # Get our trainable embeddings for address tokens
        if addr_mask.any() and self.addr_token_embeddings is not None:
            # Map extended vocab IDs to address token embedding IDs (0, 1, 2, ...)
            addr_ids = input_ids.clone()
            addr_ids[~addr_mask] = 0  # Set non-address to 0
            addr_ids[addr_mask] = addr_ids[addr_mask] - self.palmtree_vocab_size  # Offset to 0-based
            
            addr_embeds = self.addr_token_embeddings(addr_ids)  # [B, L, H]
            embeddings[addr_mask] = addr_embeds[addr_mask]
        
        return embeddings
        
    def forward(self, input_ids, attention_mask, address_encodings, addr_type_ids=None, position_ids=None):
        """
        3-level embedding forward pass
        
        Args:
            input_ids: [batch_size, seq_len] - token IDs (including new address token IDs)
            attention_mask: [batch_size, seq_len] - attention mask
            address_encodings: [batch_size, seq_len, addr_enc_dim] - sinusoidal address VALUE encodings
            addr_type_ids: DEPRECATED - address types are now in input_ids as tokens
            position_ids: [batch_size, seq_len] - position indices (auto-generated if None)
        
        Returns:
            dict with task outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Level 1: Semantic embeddings
        # Pass input_ids through PalmTree BERT (it does embedding + transformer)
        segment_info = attention_mask.clone()
        palmtree_output = self.palmtree(input_ids, segment_info)  # [B, L, H]
        
        # Now replace embeddings for address tokens with our trainable ones
        # For tokens >= 6631 (address tokens), use our addr_token_embeddings
        semantic_embeddings = palmtree_output.clone()
        for token_id in range(self.palmtree_vocab_size, self.extended_vocab_size):
            # Find positions where this address token appears
            mask = (input_ids == token_id)
            if mask.any():
                # Get the trainable embedding for this address token
                addr_token_emb = self.addr_token_embeddings(
                    torch.tensor([token_id - self.palmtree_vocab_size], device=input_ids.device)
                )  # [1, H]
                # Replace in semantic_embeddings
                semantic_embeddings[mask] = addr_token_emb
        
        # Level 2: Address-level embeddings (VALUE only, type is in semantic now!)
        # Only use address value encodings
        address_level_embeddings = self.addr_value_projection(address_encodings)  # [B, L, H]
        
        # Level 3: Positional embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # [B, L]
        positional_embeddings = self.position_embedding(position_ids)  # [B, L, H]
        
        # Fuse all three levels with simple addition (like BERT)
        # semantic + address + position
        fused_features = semantic_embeddings + address_level_embeddings + positional_embeddings  # [B, L, H]
        fused_features = self.layer_norm(fused_features)  # Normalize after addition
        
        # Task-specific predictions
        # Task 1: Masked Token Prediction (including address tokens!)
        # Predict over EXTENDED vocabulary (PalmTree + address tokens)
        token_prediction_logits = self.next_bb_head(fused_features)  # [B, L, extended_vocab]
        
        # Task 2: Edge Type Prediction (sequence-level, use [CLS])
        cls_features = fused_features[:, 0, :]
        edge_type_logits = self.edge_type_head(cls_features)  # [B, num_edge_types]
        
        # Task 3: Address Distance (computed externally, but return features for pairing)
        # We'll compute pairwise distances in the loss function
        
        return {
            'token_prediction_logits': token_prediction_logits,  # Renamed from next_bb_logits
            'next_bb_logits': token_prediction_logits,  # Keep for backward compatibility
            'addr_type_logits': token_prediction_logits,  # Keep for backward compatibility (now same as token prediction)
            'edge_type_logits': edge_type_logits,
            'hidden_states': fused_features,
            'semantic_embeddings': semantic_embeddings,
            'address_embeddings': address_level_embeddings,
            'positional_embeddings': positional_embeddings,
        }
    
    def compute_address_distance_logits(self, fused_features, batch_indices, pos1, pos2):
        """
        Compute distance classification between two addresses
        
        Args:
            fused_features: [B, L, H] - fused embeddings
            batch_indices: [num_pairs] - which batch each pair belongs to
            pos1, pos2: [num_pairs] - positions of two addresses to compare
        
        Returns:
            distance_logits: [num_pairs, 4] - same-bb/near/medium/far classification
        """
        # Extract features at the two positions for each pair
        feat1 = fused_features[batch_indices, pos1]  # [num_pairs, H]
        feat2 = fused_features[batch_indices, pos2]  # [num_pairs, H]
        
        # Concatenate and predict distance class
        paired_features = torch.cat([feat1, feat2], dim=-1)  # [num_pairs, 2*H]
        distance_logits = self.addr_distance_head(paired_features)  # [num_pairs, 4]
        
        return distance_logits


def load_pretrained_palmtree(model_path: str, vocab_size: int, hidden: int = 768, n_layers: int = 12, attn_heads: int = 12):
    """Load pre-trained PalmTree BERT model"""
    
    # Load pre-trained checkpoint (the checkpoint itself is a BERT model)
    # Use weights_only=False for older PyTorch checkpoints
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # The checkpoint is already a BERT model, just return it
    # Set to eval mode for using pre-trained features
    checkpoint.eval()
    
    return checkpoint
