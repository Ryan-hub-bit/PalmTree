import torch.nn as nn
import torch

from .bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    
    Tasks:
    - CFG: CWP (Control flow Walk Prediction) + MLM (Masked Language Modeling)
    - DFG: DUP (Data Use Prediction) only
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.CWP= NextSentencePrediction(self.bert.hidden)  # CFG walk prediction
        self.DUP = NextSentencePrediction(self.bert.hidden)  # DFG use prediction
        self.MLM = MaskedLanguageModel(self.bert.hidden, vocab_size)  # CFG token masking

    def forward(self, d, d_segment_label, c, c_segment_label):
        d = self.bert(d, d_segment_label)  # DFG encoding
        c = self.bert(c, c_segment_label)  # CFG encoding

        return self.DUP(d), self.CWP(c), self.MLM(c)  # MLM on CFG, not DFG


class NextSentencePrediction(nn.Module):
    """
    From NSP task, now used for DUP and CWP
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


