"""
Linear Probe Models for Position Prediction

This module defines simple linear probe classifiers to predict
position values from learned embeddings.
"""

import torch
import torch.nn as nn


class PositionProbe(nn.Module):
    """
    Linear probe for predicting a single position value (0-1 normalized).
    
    Given: Contextualized embedding from BERT model
    Predict: Position value in [0, 1] range
    """
    
    def __init__(self, hidden_size=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()  # Ensure output in [0, 1]
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, seq_len, hidden_size)
        Returns:
            predictions: (batch_size, seq_len, 1) - predicted positions
        """
        x = self.dropout(embeddings)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class MultiLevelPositionProbe(nn.Module):
    """
    Linear probe for predicting all three position levels simultaneously.
    
    Predicts:
    - Binary position (position in entire binary)
    - Function position (position in current function)
    - Basic block position (position in current basic block)
    """
    
    def __init__(self, hidden_size=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Separate linear layers for each position level
        self.binary_linear = nn.Linear(hidden_size, 1)
        self.function_linear = nn.Linear(hidden_size, 1)
        self.bb_linear = nn.Linear(hidden_size, 1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, seq_len, hidden_size)
        Returns:
            dict with keys:
                'binary': (batch_size, seq_len, 1)
                'function': (batch_size, seq_len, 1)
                'bb': (batch_size, seq_len, 1)
        """
        x = self.dropout(embeddings)
        
        binary_pred = self.sigmoid(self.binary_linear(x))
        function_pred = self.sigmoid(self.function_linear(x))
        bb_pred = self.sigmoid(self.bb_linear(x))
        
        return {
            'binary': binary_pred,
            'function': function_pred,
            'bb': bb_pred
        }


class PositionClassificationProbe(nn.Module):
    """
    Linear probe for classifying positions into bins (categorical prediction).
    
    Useful for analyzing whether the model learns discrete position categories
    rather than continuous values.
    """
    
    def __init__(self, hidden_size=128, num_bins=10, dropout=0.1):
        """
        Args:
            hidden_size: Dimension of input embeddings
            num_bins: Number of position bins (e.g., 10 for deciles)
            dropout: Dropout rate
        """
        super().__init__()
        self.num_bins = num_bins
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_bins)
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, seq_len, hidden_size)
        Returns:
            logits: (batch_size, seq_len, num_bins)
        """
        x = self.dropout(embeddings)
        logits = self.linear(x)
        return logits
    
    @staticmethod
    def position_to_bin(position, num_bins=10):
        """Convert continuous position [0, 1] to bin index [0, num_bins-1]."""
        bin_idx = (position * num_bins).long()
        bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)
        return bin_idx


def get_probe_model(probe_type='single', hidden_size=128, num_bins=10, dropout=0.1):
    """
    Factory function to create probe models.
    
    Args:
        probe_type: 'single', 'multi', or 'classification'
        hidden_size: Hidden dimension of BERT model
        num_bins: Number of bins for classification probe
        dropout: Dropout rate
    
    Returns:
        Probe model instance
    """
    if probe_type == 'single':
        return PositionProbe(hidden_size, dropout)
    elif probe_type == 'multi':
        return MultiLevelPositionProbe(hidden_size, dropout)
    elif probe_type == 'classification':
        return PositionClassificationProbe(hidden_size, num_bins, dropout)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
