# Address-Aware PalmTree Training Guide

This document explains how to train PalmTree with **address-aware embeddings** that incorporate hierarchical position information and variable offset encodings.

## Overview

The address-aware version enhances the standard PalmTree model with:

1. **Hierarchical Address Embeddings**: Sinusoidal encodings at three levels
   - Binary-level position (where in the entire binary)
   - Function-level position (where in the current function)
   - Basic block-level position (where in the current basic block)

2. **Variable Offset Embeddings**: Sinusoidal encodings for `var(0xXX)` tokens based on their stack offsets

3. **Token-Specific Masking**: 
   - Address embeddings only applied to opcodes and `address(...)` tokens
   - Var embeddings only applied to `var(0xXX)` tokens
   - Other tokens (registers, special tokens) get zero vectors for these embeddings

## Key Differences from Standard PalmTree

| Aspect | Standard PalmTree | Address-Aware PalmTree |
|--------|-------------------|------------------------|
| **Input Format** | `mov rax rbx` | `mov(0x1000:0.123:0.234:0.345) rax rbx` |
| **Positional Info** | Sequence position only | Sequence + hierarchical addresses |
| **Variable Tokens** | `var(0x10)` treated as token | `var(0x10)` gets offset embedding (16) |
| **Model** | `BERT` | `AddressAwareBERT` |
| **Dataset** | `BERTDataset` | `BERTDatasetAddressAware` |
| **Trainer Mode** | `mode='original'` | `mode='addressaware'` |

## Data Format Requirements

### Input File Format

Your training data must include **inline address information** in this format:

```
opcode(addr:bnorm:fnorm:bbnorm) operand1 operand2 var(0xXX)
```

**Example lines:**

```
mov(0x401000:0.123456:0.234567:0.345678) rax rbx
call(0x401005:0.125000:0.236000:0.348000) address(0x402000:0.234567:0.345678:0.456789)
lea(0x40100a:0.127000:0.238000:0.350000) rax var(0x10)
```

**Format breakdown:**
- `addr`: Hexadecimal instruction address (e.g., `0x401000`)
- `bnorm`: Binary-level normalized position, range [0.0, 1.0]
- `fnorm`: Function-level normalized position, range [0.0, 1.0]  
- `bbnorm`: Basic block-level normalized position, range [0.0, 1.0]

### Generating Address-Aware Data

Use the data generation scripts with hierarchical position tracking:

```bash
# CFG with address info
python data_generator/cfg_hierarchical_icfg_ida.py \
    --binary path/to/binary \
    --output cfg_train.txt

# DFG with address info
python data_generator/dfg_hierarchical_idfg_ida.py \
    --binary path/to/binary \
    --output dfg_train.txt
```

These scripts automatically generate the `(addr:bnorm:fnorm:bbnorm)` annotations.

## Training Setup

### 1. Prepare Directory Structure

```bash
cd /path/to/PalmTree/extern/PalmTree
mkdir -p data/training/cdfg_bert_addressaware
mkdir -p cdfg_bert_addressaware
```

### 2. Place Your Training Data

```bash
data/training/cdfg_bert_addressaware/
├── cfg_train.txt  # CFG sequences with address info
├── dfg_train.txt  # DFG sequences with address info
├── cfg_test.txt   # (Optional) Test CFG sequences
└── dfg_test.txt   # (Optional) Test DFG sequences
```

### 3. Configure Hyperparameters

The training script uses parameters from `src/config.py` where available, with fallback defaults.

**Edit `src/config.py` for global settings:**
```python
VOCAB_SIZE = 10000      # Maximum vocabulary size
USE_CUDA = True         # Use GPU if available
DEVICES = [0]           # CUDA device IDs
MAXLEN = 10            # Default sequence length
```

**Edit `train_palmtree_addressaware.py` for training-specific settings:**
```python
# Model hyperparameters (uses config.py values with fallbacks)
VOCAB_SIZE = VOCAB_SIZE if 'VOCAB_SIZE' in dir() else 13000
SEQ_LEN = MAXLEN if 'MAXLEN' in dir() else 20
BATCH_SIZE = 256       # Batch size
HIDDEN_SIZE = 128      # Embedding dimension
N_LAYERS = 12          # Number of transformer layers
ATTN_HEADS = 8         # Number of attention heads
DROPOUT = 0.1          # Dropout rate

# Training hyperparameters
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10000
NUM_EPOCHS = 20

# Address-aware flags
USE_ADDRESS_EMBEDDING = True  # Use hierarchical address positions
USE_VAR_EMBEDDING = True      # Use var(0xXX) offset embeddings

# CUDA settings (uses config.py values with fallbacks)
USE_CUDA = USE_CUDA if 'USE_CUDA' in dir() else True
CUDA_DEVICES = DEVICES if 'DEVICES' in dir() else [0]
```

### 4. Run Training

```bash
cd extern/PalmTree
python train_palmtree_addressaware.py
```

## Model Architecture

### Embedding Layer

The `AddressAwareBERTEmbedding` combines multiple embeddings:

```python
total_embedding = token_emb + seq_pos_emb + addr_pos_emb + segment_emb + var_pos_emb
```

**Components:**

1. **Token Embedding** (`nn.Embedding`): Standard vocabulary lookup
   - All tokens: `rax`, `mov`, `var(0x10)`, etc.

2. **Sequence Position Embedding** (sinusoidal): Standard BERT-style sequence position
   - All tokens: 0, 1, 2, 3, ...

3. **Address Position Embedding** (sinusoidal, **3 levels**):
   - **Opcodes/addresses**: Full hierarchical encoding
   - **Other tokens**: Zero vector (masked out)
   - Uses `-1` as sentinel value for non-address tokens

4. **Segment Embedding** (`nn.Embedding`): For NSP task
   - Segment A vs Segment B

5. **Var Position Embedding** (sinusoidal):
   - **var(0xXX) tokens**: Sinusoidal encoding of offset value
   - **Other tokens**: Zero vector (masked out)
   - Uses `-1` as sentinel value for non-var tokens

### Masking Strategy

**Address Position Masking:**
```python
address_mask = ((binary_pos >= 0) & (function_pos >= 0) & (bb_pos >= 0))
embedding = embedding * address_mask.float()
```

**Var Position Masking:**
```python
var_mask = (var_offsets >= 0)
encoding = encoding * var_mask.float()
```

| Token Type | Example | binary_pos | function_pos | bb_pos | var_offset | address_emb | var_emb |
|------------|---------|-----------|--------------|--------|------------|-------------|---------|
| Opcode | `mov` | 0.123 | 0.234 | 0.345 | -1 | ✅ Full | ❌ Zeros |
| Address | `address(0x401000)` | 0.234 | 0.345 | 0.456 | -1 | ✅ Full | ❌ Zeros |
| Var | `var(0x10)` | -1 | -1 | -1 | 16 | ❌ Zeros | ✅ Full |
| Register | `rax` | -1 | -1 | -1 | -1 | ❌ Zeros | ❌ Zeros |
| Special | `[SOS]` | -1 | -1 | -1 | -1 | ❌ Zeros | ❌ Zeros |

## Training Tasks

The model is trained on three tasks simultaneously:

1. **MLM (Masked Language Model)**: Predict masked tokens in DFG
2. **DUP (Data Use Prediction)**: Next sentence prediction for DFG sequences  
3. **CWP (Control Walk Prediction)**: Next sentence prediction for CFG sequences

Loss function:
```python
total_loss = mlm_loss + dup_loss + cwp_loss
```

## Output

Training produces checkpoints in `cdfg_bert_addressaware/transformer/`:

```
bert_trained_0.model
bert_trained_1.model
...
bert_trained_19.model
```

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Epoch number
- Training configuration

## Using Trained Models

### Load Model for Inference

```python
from palmtree.model import AddressAwareBERT
import torch

# Initialize model
model = AddressAwareBERT(
    vocab_size=13000,
    hidden=128,
    n_layers=12,
    attn_heads=8,
    use_address_embedding=True,
    use_var_embedding=True
)

# Load checkpoint
checkpoint = torch.load('cdfg_bert_addressaware/transformer/bert_trained_19.model')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Extract Function Embeddings

```python
# Prepare input with address info
tokens = encode_with_vocab(instruction_sequence)  # [1, seq_len]
segment = torch.zeros_like(tokens)
binary_pos = torch.tensor([[0.1, 0.2, 0.3, ...]])
function_pos = torch.tensor([[0.01, 0.02, 0.03, ...]])
bb_pos = torch.tensor([[0.001, 0.002, 0.003, ...]])
var_offsets = torch.tensor([[-1, -1, 16, ...]])  # -1 for non-var, offset for var

# Forward pass
with torch.no_grad():
    embeddings = model(tokens, segment, binary_pos, function_pos, bb_pos, var_offsets)
    
# Use [CLS] token embedding as function representation
function_embedding = embeddings[:, 0, :]  # [1, hidden_size]
```

## Troubleshooting

### Import Errors

If you see `Import "palmtree" could not be resolved`:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/PalmTree/extern/PalmTree/src"
```

### Data Format Errors

If training fails with parsing errors, verify your data format:

```python
# Quick check
with open('data/training/cdfg_bert_addressaware/cfg_train.txt', 'r') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"Line {i}: {line.strip()}")
```

Expected output should have `opcode(0xADDR:float:float:float)` format.

### Memory Issues

Reduce batch size and/or number of workers:

```python
BATCH_SIZE = 128  # Reduce from 256
NUM_WORKERS = 4   # Reduce from 10
```

### CUDA Out of Memory

Reduce model size:

```python
HIDDEN_SIZE = 64   # Reduce from 128
N_LAYERS = 6       # Reduce from 12
```

## Comparison: Standard vs Address-Aware

### Standard PalmTree Training

```bash
python train_palmtree.py
```

**Pros:**
- Simpler data format
- Faster training (no extra embeddings)
- Works with existing data

**Cons:**
- No hierarchical position awareness
- Cannot distinguish `var(0x0)` from `var(0x10)` structurally
- Loses spatial information about code structure

### Address-Aware PalmTree Training

```bash
python train_palmtree_addressaware.py
```

**Pros:**
- Rich positional information (binary/function/BB levels)
- Explicit var offset encoding
- Better for tasks requiring structural understanding
- Clean token-type separation via masking

**Cons:**
- Requires annotated data with address info
- Slightly slower training (more embeddings)
- More complex data pipeline

## Next Steps

After training, you can:

1. **Fine-tune for downstream tasks**: Function similarity, type recovery, etc.
2. **Extract embeddings**: Use for similarity search, clustering
3. **Probe analysis**: Evaluate what the model learned about code structure

See `dstask/funcsim/` for function similarity fine-tuning examples.

## References

- Original PalmTree paper: [Link to paper]
- Standard training script: `train_palmtree.py`
- Address embedding implementation: `src/palmtree/model/embedding/address_embedding.py`
- Dataset implementation: `src/palmtree/dataset/dataset_addressaware.py`
