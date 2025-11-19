"""
Address-Aware BERT Model

Modified BERT architecture that incorporates address normalization
at three hierarchical levels: binary, function, and basic block.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from palmtree.model.transformer import TransformerBlock
from palmtree.model.utils.layer_norm import LayerNorm

# Use relative import for address_embedding (works when running from addressaware dir)
try:
    from addressaware.address_embedding import AddressAwareBERTEmbedding
except ModuleNotFoundError:
    from address_embedding import AddressAwareBERTEmbedding


class AddressAwareBERT(nn.Module):
    """
    Address-aware BERT for binary code understanding.
    
    Uses hierarchical address positions (binary, function, basic block)
    in addition to standard token embeddings.
    
    Can be initialized with pre-trained PalmTree weights.
    """
    
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, max_len=512,
                 pretrained_token_emb=None, pretrained_position_emb=None, pretrained_segment_emb=None, pretrained_transformer=None):
        """
        Args:
            vocab_size: Size of the vocabulary
            hidden: Hidden size / embedding dimension
            n_layers: Number of transformer layers
            attn_heads: Number of attention heads
            dropout: Dropout rate
            max_len: Maximum sequence length
            pretrained_token_emb: Pre-trained token embedding from PalmTree (will be FROZEN)
            pretrained_position_emb: Pre-trained position embedding from PalmTree (will be FROZEN)
            pretrained_segment_emb: Pre-trained segment embedding from PalmTree
            pretrained_transformer: Pre-trained transformer blocks from PalmTree (will be FROZEN)
        """
        super().__init__()
        
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        
        # Address-aware embedding layer
        # PalmTree components (token, sequence_pos) will be FROZEN
        # Address position embedding is NEW and TRAINABLE
        self.embedding = AddressAwareBERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            dropout=dropout,
            max_len=max_len,
            pretrained_token_emb=pretrained_token_emb,
            pretrained_position_emb=pretrained_position_emb,
            pretrained_segment_emb=pretrained_segment_emb
        )
        
        # Transformer blocks (can be loaded from pre-trained PalmTree)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
            for _ in range(n_layers)
        ])
        
        # Load pre-trained transformer blocks if provided and FREEZE them
        if pretrained_transformer is not None:
            print(f"[INFO] Loading pre-trained transformer blocks")
            self.transformer_blocks.load_state_dict(pretrained_transformer, strict=False)
            # FREEZE all transformer parameters
            """ for param in self.transformer_blocks.parameters():
                param.requires_grad = False
            print(f"[INFO] Transformer blocks FROZEN") """
    
    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos):
        """
        Forward pass.
        
        Args:
            token_ids: [batch_size, seq_len]
            segment_labels: [batch_size, seq_len]
            binary_pos: [batch_size, seq_len]
            function_pos: [batch_size, seq_len]
            bb_pos: [batch_size, seq_len]
            
        Returns:
            output: [batch_size, seq_len, hidden]
        """
        # Create attention mask (mask out padding tokens)
        mask = (token_ids > 0).unsqueeze(1).repeat(1, token_ids.size(1), 1).unsqueeze(1)
        
        # Get embeddings with address awareness
        x = self.embedding(token_ids, segment_labels, binary_pos, function_pos, bb_pos)
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x


class AddressAwareBERTForPretraining(nn.Module):
    """
    Address-aware BERT with MLM and dual NSP heads for pretraining.
    
    Following PalmTree's approach:
    - MLM head (for CFG only)
    - NSP_CFG head (for CFG order coherence)
    - NSP_DFG head (for DFG trace coherence)
    """
    
    def __init__(self, bert_model, vocab_size):
        """
        Args:
            bert_model: AddressAwareBERT model
            vocab_size: Vocabulary size
        """
        super().__init__()
        
        self.bert = bert_model
        self.vocab_size = vocab_size
        self.hidden = bert_model.hidden
        
        # Masked Language Model head (for CFG only)
        self.mlm_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, vocab_size)
        )
        
        # CFG Next Sentence Prediction head (order coherence)
        self.nsp_cfg_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 2)
        )
        
        # DFG Next Sentence Prediction head (trace coherence)
        self.nsp_dfg_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 2)
        )
        
        # Address Distance Prediction head (learns address relationships)
        # Predicts relative distance category between two instruction segments
        self.adp_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 categories: very_close, close, medium, far, very_far
        )
        
        # Address Order Prediction head (predicts if instruction A comes before B)
        self.aop_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 2)  # binary: before (1) or after (0)
        )
    
    def forward(self, token_ids, segment_labels, binary_pos, function_pos, bb_pos, corpus_type='cfg', 
                return_address_tasks=False):
        """
        Forward pass for pretraining.
        
        Args:
            token_ids: [batch_size, seq_len]
            segment_labels: [batch_size, seq_len]
            binary_pos: [batch_size, seq_len]
            function_pos: [batch_size, seq_len]
            bb_pos: [batch_size, seq_len]
            corpus_type: 'cfg' or 'dfg' - determines which NSP head to use
            return_address_tasks: if True, also return address distance and order predictions
            
        Returns:
            mlm_output: [batch_size, seq_len, vocab_size] - predictions for each token (CFG only)
            nsp_output: [batch_size, 2] - binary classification for NSP
            adp_output: [batch_size, 5] - address distance prediction (if return_address_tasks=True)
            aop_output: [batch_size, 2] - address order prediction (if return_address_tasks=True)
        """
        # Get BERT output
        sequence_output = self.bert(token_ids, segment_labels, binary_pos, function_pos, bb_pos)
        
        # MLM prediction for all tokens (only meaningful for CFG)
        mlm_output = self.mlm_head(sequence_output)
        
        # NSP prediction using [CLS] token (first token)
        cls_output = sequence_output[:, 0, :]
        
        # Use appropriate NSP head based on corpus type
        if corpus_type == 'cfg':
            nsp_output = self.nsp_cfg_head(cls_output)
        else:  # dfg
            nsp_output = self.nsp_dfg_head(cls_output)
        
        if return_address_tasks:
            # Address Distance Prediction using [CLS] token
            adp_output = self.adp_head(cls_output)
            
            # Address Order Prediction using [CLS] token
            aop_output = self.aop_head(cls_output)
            
            return mlm_output, nsp_output, adp_output, aop_output
        
        return mlm_output, nsp_output
