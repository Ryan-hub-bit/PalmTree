"""
Baseline Model for jTrans Pretraining

Standard BERT with jTrans trick: position_embeddings = word_embeddings
No address-aware features.

Tasks:
1. MLM (Masked Language Modeling): Predict masked tokens
2. JTP (Jump-Target Prediction): Predict jump target positions
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BinBertModel(BertModel):
    """
    Baseline jTrans BERT model with position_embeddings = word_embeddings trick.
    
    This is the original jTrans BinBertModel approach.
    Inherits from BertModel and modifies embeddings after initialization.
    """
    
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.config = config
        
        # jTrans trick: Use word embeddings as position embeddings
        # This allows the model to learn position information from token context
        self.embeddings.position_embeddings = self.embeddings.word_embeddings


class BaselinePretrainingModel(nn.Module):
    """
    Baseline pretraining model with MLM + JTP tasks.
    """
    
    def __init__(self, bert_model, vocab_size, max_position=512):
        super().__init__()
        self.bert = bert_model
        self.vocab_size = vocab_size
        self.max_position = max_position
        
        hidden_size = bert_model.config.hidden_size
        
        # MLM head: Predict token vocabulary
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # JTP head: Predict jump target position (0 to max_position)
        self.jtp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, max_position)
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Forward pass for baseline pretraining.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            
        Returns:
            mlm_logits: [batch_size, seq_len, vocab_size]
            jtp_logits: [batch_size, seq_len, max_position]
        """
        # Get BERT output
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # MLM prediction
        mlm_logits = self.mlm_head(sequence_output)
        
        # JTP prediction (for JUMP_ADDR tokens)
        jtp_logits = self.jtp_head(sequence_output)
        
        return mlm_logits, jtp_logits


def create_baseline_model(
    vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512
):
    """
    Create a baseline jTrans model for pretraining with MLM + JTP tasks.
    
    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: FFN intermediate size
        hidden_dropout_prob: Dropout probability
        attention_probs_dropout_prob: Attention dropout
        max_position_embeddings: Maximum sequence length
        
    Returns:
        BaselinePretrainingModel with MLM + JTP heads
    """
    # Create config
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
    )
    
    # Create baseline BERT model (BinBertModel)
    bert = BinBertModel(config, add_pooling_layer=False)
    
    # Wrap with pretraining heads
    model = BaselinePretrainingModel(
        bert_model=bert,
        vocab_size=vocab_size,
        max_position=max_position_embeddings
    )
    
    return model
