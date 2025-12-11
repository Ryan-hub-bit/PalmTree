"""
Address-Aware BERT Language Model
Next Sentence Prediction Model + Masked Language Model with Address Information
"""

import torch.nn as nn
import torch

from .bert_addressaware import AddressAwareBERT


class AddressAwareBERTLM(nn.Module):
    """
    Address-Aware BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: AddressAwareBERT, vocab_size):
        """
        Args:
            bert: AddressAwareBERT model which should be trained
            vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.bert = bert
        self.CWP = NextSentencePrediction(self.bert.hidden)  # Control Flow Walk Prediction
        self.DUP = NextSentencePrediction(self.bert.hidden)  # Data Use Prediction
        self.MLM = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, d, d_segment_label, d_binary_pos, d_function_pos, d_bb_pos, d_var_offsets,
                c, c_segment_label, c_binary_pos, c_function_pos, c_bb_pos, c_var_offsets):
        """
        Forward pass with address information.
        
        Args:
            d: [batch_size, seq_len] DFG token IDs
            d_segment_label: [batch_size, seq_len] DFG segment labels
            d_binary_pos: [batch_size, seq_len] DFG binary-level positions
            d_function_pos: [batch_size, seq_len] DFG function-level positions
            d_bb_pos: [batch_size, seq_len] DFG basic block-level positions
            d_var_offsets: [batch_size, seq_len] DFG var offsets
            c: [batch_size, seq_len] CFG token IDs
            c_segment_label: [batch_size, seq_len] CFG segment labels
            c_binary_pos: [batch_size, seq_len] CFG binary-level positions
            c_function_pos: [batch_size, seq_len] CFG function-level positions
            c_bb_pos: [batch_size, seq_len] CFG basic block-level positions
            c_var_offsets: [batch_size, seq_len] CFG var offsets
            
        Returns:
            dup_output: DFG next sentence prediction
            cwp_output: CFG next sentence prediction  
            mlm_output: Masked language model predictions
        """
        d = self.bert(d, d_segment_label, d_binary_pos, d_function_pos, d_bb_pos, d_var_offsets)
        c = self.bert(c, c_segment_label, c_binary_pos, c_function_pos, c_bb_pos, c_var_offsets)

        return self.DUP(d), self.CWP(c), self.MLM(d)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model: is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        Args:
            hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # Use [CLS] token representation (first token)
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    Predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        Args:
            hidden: output size of BERT model
            vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
