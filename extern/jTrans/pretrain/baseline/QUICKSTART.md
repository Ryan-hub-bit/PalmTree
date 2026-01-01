# Baseline jTrans Pretraining - Quick Start Guide

Get started with baseline jTrans pretraining in 5 minutes!

## Prerequisites

1. **Python environment**: Python 3.7+
2. **Dependencies**: PyTorch, transformers, tqdm
3. **Data**: Preprocessed assembly functions (one per line)
4. **Tokenizer**: jTrans BertTokenizer

## Quick Setup

### Step 1: Prepare Your Data

Create a data file with one function per line:

```bash
# Example: train_data.txt
push rbp mov rbp rsp sub rsp CONST call JUMP_ADDR_15 test eax eax
mov rdi CONST call JUMP_ADDR_10 add rsp CONST pop rbp ret
lea rax [rbp+CONST] mov [rbp+CONST] rax mov eax CONST
...
```

**Key format requirements**:
- Tokens separated by spaces
- Jump targets as `JUMP_ADDR_X` where X is the target position
- Constants normalized to `CONST`
- Variables normalized to `var_xxx` or `arg_xxx`

### Step 2: Set Up Tokenizer

You need a jTrans tokenizer with special tokens:

```python
from transformers import BertTokenizer

# If you already have a tokenizer
tokenizer = BertTokenizer.from_pretrained('/path/to/tokenizer')

# Or create one (if needed)
# See jTrans/jtrans_tokenizer/ for tokenizer creation
```

### Step 3: Configure Training

Edit `run_baseline.sh`:

```bash
#!/bin/bash

# Set your paths
TRAIN_PATH="./data/train.txt"         # Your training data
TEST_PATH="./data/test.txt"           # Your test data
TOKENIZER_PATH="./tokenizer"          # Your tokenizer
OUTPUT_DIR="./output_baseline"        # Where to save results

# Default hyperparameters work well!
BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_EPOCHS=10
```

### Step 4: Run Training

```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
chmod +x run_baseline.sh
./run_baseline.sh
```

**That's it!** Training will start and you'll see:

```
================================================================================
jTrans BASELINE Pretraining (MLM + JTP)
================================================================================
Train data: ./data/train.txt
Test data: ./data/test.txt
Device: cuda
Batch size: 32
Learning rate: 0.0001
...

Epoch 1/10
────────────────────────────────────────────────────────────────────────────────
Training: 100%|██████████| 1250/1250 [05:23<00:00, 3.86it/s, loss=4.2341, mlm_acc=0.6234, jtp_acc=0.3456]
Train - Loss: 4.2341, MLM: 2.1234 (acc: 0.6234), JTP: 2.1107 (acc: 0.3456)
Validation: 100%|██████████| 312/312 [00:45<00:00, 6.89it/s]
Val - Loss: 3.9876, MLM: 1.9543 (acc: 0.6543), JTP: 2.0333 (acc: 0.3789)
✓ Best model saved! Val loss: 3.9876
...
```

## What Gets Trained?

### Two Tasks Run Simultaneously

1. **MLM (Masked Language Modeling)**
   - 15% of regular tokens are masked
   - Model predicts what token was masked
   - Example: `mov [MASK] rsp` → predict `rbp`

2. **JTP (Jump-Target Prediction)**
   - 20% of jump tokens are masked
   - Model predicts where the jump goes
   - Example: `call [MASK]` → predict position `15` (from `JUMP_ADDR_15`)

## Checking Results

### During Training

Monitor in real-time:
- **loss**: Combined MLM + JTP loss (lower is better)
- **mlm_acc**: MLM token prediction accuracy
- **jtp_acc**: JTP position prediction accuracy
- **lr**: Current learning rate

### After Training

Check the output directory:

```bash
output_baseline/
├── best_model/              # Best checkpoint (lowest val loss)
│   ├── config.json
│   ├── pytorch_model.bin
│   └── training_info.json
├── training_history.json    # All metrics per epoch
└── train_baseline_*.log     # Full training log
```

View training history:

```python
import json

with open('output_baseline/training_history.json') as f:
    history = json.load(f)

for epoch in history:
    print(f"Epoch {epoch['epoch']}: "
          f"Val Loss={epoch['val_loss']:.4f}, "
          f"MLM Acc={epoch['val_mlm_acc']:.4f}, "
          f"JTP Acc={epoch['val_jtp_acc']:.4f}")
```

## Expected Results

### Typical Performance (10 epochs)

- **MLM Accuracy**: 60-70% ✅
- **JTP Accuracy**: 30-50% (harder task)
- **Training Time**: ~3 hours/epoch on V100
- **Memory**: ~10GB GPU

### What's Good Performance?

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| **MLM Acc** | <40% | 40-60% | 60-75% | >75% |
| **JTP Acc** | <15% | 15-30% | 30-50% | >50% |
| **Val Loss** | >6.0 | 4.0-6.0 | 2.5-4.0 | <2.5 |

## Common Issues & Fixes

### 1. Out of Memory
```bash
# Solution: Reduce batch size or sequence length
python train_baseline.py ... --batch_size 16 --max_len 256
```

### 2. Data Loading Error
```bash
# Check your data format
head -n 5 data/train.txt

# Should see tokenized functions, one per line
```

### 3. Poor JTP Performance
```bash
# JTP is harder - this is normal!
# Position prediction is more difficult than token prediction
# JTP accuracy 30-40% is actually good
```

### 4. Training Too Slow
```bash
# Reduce model size for faster experimentation
python train_baseline.py ... \
    --hidden_size 512 \
    --num_hidden_layers 6 \
    --num_epochs 5
```

## Next Steps

### 1. Use Pretrained Model

Load your trained model:

```python
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('output_baseline/best_model')
tokenizer = BertTokenizer.from_pretrained('./tokenizer')

# Get embeddings for a function
tokens = ['push', 'rbp', 'mov', 'rbp', 'rsp']
inputs = tokenizer(tokens, is_split_into_words=True, return_tensors='pt')
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # [1, seq_len, 768]
```

### 2. Fine-tune on Downstream Tasks

Use the pretrained model for:
- Function similarity
- Vulnerability detection
- Binary diffing
- Malware classification

### 3. Compare with Address-Aware

Try the address-aware version:

```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain
./train_address_aware_enhanced.sh
```

### 4. Experiment with Hyperparameters

Try different settings:
- Increase JTP probability: `--jtp_probability 0.30`
- Larger model: `--hidden_size 1024 --num_hidden_layers 24`
- More epochs: `--num_epochs 20`

## Advanced Usage

### Custom Training Script

```python
from baseline import create_baseline_model, create_baseline_dataloaders
from transformers import BertTokenizer
import torch

# Setup
device = torch.device('cuda')
tokenizer = BertTokenizer.from_pretrained('./tokenizer')

# Create model
model = create_baseline_model(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_hidden_layers=12
).to(device)

# Create dataloaders
train_loader, test_loader = create_baseline_dataloaders(
    train_path='./data/train.txt',
    test_path='./data/test.txt',
    tokenizer=tokenizer,
    batch_size=32
)

# Training loop
for epoch in range(10):
    for batch in train_loader:
        mlm_logits, jtp_logits = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            token_type_ids=batch['token_type_ids'].to(device)
        )
        # ... compute loss and backprop
```

## Help & Support

### Documentation
- Full README: `README.md`
- Code comments in source files
- jTrans paper for theoretical background

### Debugging
- Check logs: `output_baseline/train_baseline_*.log`
- Validate data format
- Start with small dataset to test pipeline

### Performance Tips
1. Use mixed precision training (FP16) for 2x speedup
2. Increase batch size on larger GPUs
3. Use multiple GPUs with `torch.nn.DataParallel`
4. Profile with `torch.profiler` to find bottlenecks

---

**Happy Training!** 🚀

For questions or issues, check the README.md or jTrans documentation.
