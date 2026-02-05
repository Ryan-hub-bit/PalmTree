"""
Address-Aware jTrans Model

Integrates hierarchical address embeddings (binary/function/bb positions)
with separate daddr vocabulary and var offset tracking.

Key difference from baseline:
- Uses AddressAwareBERTEmbedding instead of position=word embedding
- Supports daddr, var, and imm tokens with proper semantic encoding
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from address_embedding import AddressAwareBERTEmbedding


class AddressAwareJTransForMLM(nn.Module):
    """
    Address-aware BERT with MLM and JTP heads for pretraining.
    
    Similar to baseline but uses hierarchical address embeddings.
    """
    
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, 
                 dropout=0.1, max_len=512, use_jtp=True, use_binary_pos=True):
        super().__init__()
        
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.use_jtp = use_jtp
        
        # Create BERT config
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden,
            num_hidden_layers=n_layers,
            num_attention_heads=attn_heads,
            intermediate_size=hidden * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_len,
            type_vocab_size=256,  # Must match segment_types in AddressAwareBERTEmbedding
        )
        
        # Create base BERT model
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # Replace standard embeddings with address-aware embeddings
        self.bert.embeddings = AddressAwareBERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            dropout=dropout,
            max_len=max_len,
            use_address_embedding=True,
            use_var_embedding=True,
            use_binary_pos=use_binary_pos,
            segment_types=256,  # Support up to 256 instructions per sequence
            vocab_stoi=None  # Will be set by train script after vocab is loaded
        )
        
        # MLM head (predicts masked tokens)
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, vocab_size)
        )
        
        # JTP head (predicts jump target positions) - OPTIONAL
        # Outputs position index [0, max_len-1]
        if self.use_jtp:
            self.jtp_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, max_len)
            )
        else:
            self.jtp_head = None
    
    def forward(self, token_ids, attention_mask, token_type_ids, 
                binary_pos, function_pos, bb_pos, var_offsets=None):
        """
        Forward pass.
        
        Args:
            token_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            binary_pos: [batch_size, seq_len] normalized [0,1] positions
            function_pos: [batch_size, seq_len] normalized [0,1] positions
            bb_pos: [batch_size, seq_len] normalized [0,1] positions
            var_offsets: [batch_size, seq_len] var(0xXX) offsets, -1 for non-var
            
        Returns:
            mlm_logits: [batch_size, seq_len, vocab_size]
            jtp_logits: [batch_size, seq_len, max_len]
        """
        # Get embeddings with address awareness
        # AddressAwareBERTEmbedding expects: token_ids, segment_labels, binary_pos, function_pos, bb_pos, var_offsets
        embeddings = self.bert.embeddings(
            token_ids,
            token_type_ids,
            binary_pos,
            function_pos,
            bb_pos,
            var_offsets
        )
        
        # Pass through transformer
        outputs = self.bert.encoder(
            embeddings,
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden]
        
        # MLM predictions
        mlm_logits = self.mlm_head(sequence_output)
        
        # JTP predictions (optional)
        if self.use_jtp:
            jtp_logits = self.jtp_head(sequence_output)
        else:
            jtp_logits = None
        
        return mlm_logits, jtp_logits


def create_addressaware_model(vocab_size, hidden=768, n_layers=12, attn_heads=12, 
                               dropout=0.1, max_len=512, use_jtp=True, use_binary_pos=True):
    """
    Factory function to create address-aware model.
    
    Args:
        use_jtp: Whether to include JTP head (if False, MLM only)
        use_binary_pos: Whether to use binary-level position embeddings
    """
    model = AddressAwareJTransForMLM(
        vocab_size=vocab_size,
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads,
        dropout=dropout,
        max_len=max_len,
        use_jtp=use_jtp,
        use_binary_pos=use_binary_pos
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
