# Baseline Training Tasks Configuration

## Overview
The baseline model (without address embeddings) performs the following tasks:

### CFG Tasks:
1. **MLM (Masked Language Modeling)**: Mask and predict individual tokens at CFG level
2. **CWP (Control flow Walk Prediction)**: Predict if two instruction sequences are consecutive in the control flow

### DFG Tasks:
1. **DUP (Data Use Prediction)**: Predict if two instruction sequences are consecutive in the data flow
2. **NO MLM**: DFG does not use masked language modeling
3. **NO IMD**: No instruction-level masking for DFG

## Implementation Details

### Dataset (`dataset_baseline.py`)
- **CFG is_next**: 50% consecutive (is_next=1), 50% random (is_next=0) → for CWP task
- **CFG MLM**: Applied to CFG tokens (15% masking probability)
- **DFG is_next**: 50% consecutive (is_next=1), 50% random (is_next=0) → for DUP task
- **DFG MLM**: NOT applied to DFG (no token masking)
- **IMD**: NOT applied to DFG (no instruction-level masking)

### Model (`language_model.py`)
```python
class BERTLM(nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        self.bert = bert
        self.CWP = NextSentencePrediction(self.bert.hidden)  # CFG walk prediction
        self.DUP = NextSentencePrediction(self.bert.hidden)  # DFG use prediction
        self.MLM = MaskedLanguageModel(self.bert.hidden, vocab_size)  # CFG token masking
    
    def forward(self, d, d_segment_label, c, c_segment_label):
        d = self.bert(d, d_segment_label)  # DFG encoding
        c = self.bert(c, c_segment_label)  # CFG encoding
        return self.DUP(d), self.CWP(c), self.MLM(c)  # MLM on CFG only!
```

### Trainer (`pretrain.py`)
- **Loss functions**:
  - `cfg_next_criterion`: NLLLoss for CWP task (CFG is_next prediction)
  - `dfg_next_criterion`: NLLLoss for DUP task (DFG is_next prediction)  
  - `masked_criterion`: NLLLoss for MLM task (CFG token prediction only)
  - All use `ignore_index=-100` to skip non-masked tokens
- **Total Loss**: `loss = cfg_next_loss + dfg_next_loss + mask_loss`
  - Note: mask_loss uses `data["cfg_bert_label"]` (CFG only, not DFG)

## Task Comparison: Baseline vs Address-Aware

| Task | Baseline | Address-Aware |
|------|----------|---------------|
| CFG MLM | ✓ Token-level masking | ✓ Token-level masking |
| CFG IMC | ✗ No instruction masking | ✓ Instruction-level masking |
| CFG CWP | ✓ Next sequence prediction | ✓ Next sequence prediction |
| DFG MLM | ✗ No token masking | ✗ No token masking |
| DFG IMD | ✗ No instruction masking | ✗ No instruction masking |
| DFG DUP | ✓ Next sequence prediction | ✓ Next sequence prediction |
| Address Embeddings | ✗ Stripped out | ✓ Hierarchical positions |

## Verification

Run the following to verify is_next distribution:
```bash
cd /home/kun/Document/PalmTree/extern/PalmTree/src
python3 -c "
import torch
import sys
sys.path.insert(0, '../../strupos')
from vocab import WordVocab
from palmtree.dataset.dataset_baseline import BaselineDataset

vocab = WordVocab.load_vocab('./vocab_base')
dataset = BaselineDataset(
    cfg_corpus_path='/data/kun/palmtreedata/cfg_train_2.txt',
    dfg_corpus_path='/data/kun/palmtreedata/dfg_train_2.txt',
    vocab=vocab,
    seq_len=512,
    enable_imd=True,
    instruction_mask_prob=0.25,
    token_mask_prob=0.15
)

cfg_next = [dataset[i]['cfg_is_next'].item() for i in range(100)]
dfg_next = [dataset[i]['dfg_is_next'].item() for i in range(100)]

print(f'CFG is_next=0: {cfg_next.count(0)}%, is_next=1: {cfg_next.count(1)}%')
print(f'DFG is_next=0: {dfg_next.count(0)}%, is_next=1: {dfg_next.count(1)}%')
"
```

Expected output: ~50/50 distribution for both CFG and DFG is_next values.
