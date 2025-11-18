"""
Baseline PalmTree Model (without address embeddings)

This model:
- Uses standard PalmTree BERT with sequential position embeddings
- Same architecture as address-aware model but NO address positions
- Used for fair comparison: same data, same tasks, but no address info
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import PalmTree modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from palmtree.model.bert import BERT
from palmtree.model.language_model import NextSentencePrediction, MaskedLanguageModel


class BaselineBERTForPretraining(nn.Module):
    """
    Baseline BERT for Pretraining (standard PalmTree without address embeddings).
    
    Architecture:
    - Standard BERT with token + position + segment embeddings
    - MLM head for CFG masked language modeling
    - CWP head for CFG next sentence prediction
    - DUP head for DFG next sentence prediction
    
    Same as AddressAwareBERT but uses sequential positions instead of address positions.
    """
    
    def __init__(self, bert: BERT, vocab_size):
        """
        Args:
            bert: Standard PalmTree BERT model
            vocab_size: Total vocabulary size for MLM
        """
        super().__init__()
        self.bert = bert
        
        # Task heads (same as address-aware model)
        self.MLM = MaskedLanguageModel(self.bert.hidden, vocab_size)
        self.CWP = NextSentencePrediction(self.bert.hidden)  # CFG NSP
        self.DUP = NextSentencePrediction(self.bert.hidden)  # DFG NSP
    
    def forward(self, cfg_input, cfg_segment, dfg_input, dfg_segment):
        """
        Forward pass for baseline model.
        
        Args:
            cfg_input: CFG token IDs [batch_size, seq_len]
            cfg_segment: CFG segment labels [batch_size, seq_len]
            dfg_input: DFG token IDs [batch_size, seq_len]
            dfg_segment: DFG segment labels [batch_size, seq_len]
        
        Returns:
            mlm_output: MLM predictions for CFG [batch_size, seq_len, vocab_size]
            cwp_output: CWP predictions for CFG [batch_size, 2]
            dup_output: DUP predictions for DFG [batch_size, 2]
        """
        # Encode CFG with standard BERT (sequential positions)
        cfg_encoded = self.bert(cfg_input, cfg_segment)
        
        # Encode DFG with standard BERT (sequential positions)
        dfg_encoded = self.bert(dfg_input, dfg_segment)
        
        # Apply task heads
        mlm_output = self.MLM(cfg_encoded)
        cwp_output = self.CWP(cfg_encoded)
        dup_output = self.DUP(dfg_encoded)
        
        return mlm_output, cwp_output, dup_output


def create_baseline_model(vocab_size, hidden=128, n_layers=12, attn_heads=8, dropout=0.1):
    """
    Create baseline PalmTree model for fair comparison.
    
    Args:
        vocab_size: Vocabulary size
        hidden: Hidden dimension size
        n_layers: Number of transformer layers
        attn_heads: Number of attention heads
        dropout: Dropout rate
    
    Returns:
        BaselineBERTForPretraining model
    """
    # Create standard PalmTree BERT
    bert = BERT(
        vocab_size=vocab_size,
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads,
        dropout=dropout
    )
    
    # Wrap in pretraining model with task heads
    model = BaselineBERTForPretraining(bert, vocab_size)
    
    return model
