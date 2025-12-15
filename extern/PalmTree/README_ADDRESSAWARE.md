# Address-Aware PalmTree: Quick Start Guide

## What Was Added

I've integrated **address-aware training** into the PalmTree framework at `extern/PalmTree/`. The address-aware version enhances the model with:

1. **Hierarchical Address Embeddings** - Sinusoidal positional encodings at three levels:
   - Binary-level (where in the entire binary)
   - Function-level (where in the current function)  
   - Basic block-level (where in the current basic block)

2. **Variable Offset Embeddings** - Sinusoidal encodings for `var(0xXX)` tokens based on their stack offsets

3. **Smart Masking** - Token-specific embedding application:
   - Opcodes/addresses → Get address embeddings
   - Variables → Get var offset embeddings
   - Registers/special tokens → Get zeros for both

## Files Created

### 1. Training Script
**`extern/PalmTree/train_palmtree_addressaware.py`**
- Main training script using `AddressAwareBERT` model
- Uses `BERTDatasetAddressAware` for data loading
- Trains with MLM + DUP + CWP tasks
- Configurable hyperparameters

### 2. Documentation
**`extern/PalmTree/TRAIN_ADDRESSAWARE.md`**
- Complete training guide
- Architecture explanation
- Configuration options
- Troubleshooting tips
- Model usage examples

**`extern/PalmTree/DATA_FORMAT_EXAMPLES.md`**
- Detailed data format specification
- Real examples with calculations
- Token type breakdown
- Validation scripts

**`extern/PalmTree/compare_training.sh`**
- Side-by-side comparison of standard vs address-aware training
- Run: `./compare_training.sh`

## Quick Start

### Step 1: Prepare Your Data

Your training data must have inline address information:

```
mov(0x401000:0.000000:0.000000:0.000000) rax rbx
call(0x401003:0.000015:0.012500:0.166667) symbol(0x402000:0.000500:0.000000:0.000000)
lea(0x401008:0.000040:0.033333:0.444444) rax var(0x10)
```

Generate using:
```bash
python data_generator/cfg_hierarchical_icfg_ida.py --input binary --output cfg_train.txt
python data_generator/dfg_hierarchical_idfg_ida.py --input binary --output dfg_train.txt
```

### Step 2: Place Data Files

```bash
mkdir -p extern/PalmTree/data/training/cdfg_bert_addressaware
cp cfg_train.txt extern/PalmTree/data/training/cdfg_bert_addressaware/
cp dfg_train.txt extern/PalmTree/data/training/cdfg_bert_addressaware/
```

### Step 3: Configure Training

The script uses parameters from `src/config.py` where available:

**Edit `src/config.py` for global settings:**
```python
VOCAB_SIZE = 10000      # Vocabulary size
USE_CUDA = True         # Use GPU
DEVICES = [0]           # GPU device IDs  
MAXLEN = 10            # Sequence length
```

**Edit `train_palmtree_addressaware.py` for training-specific settings:**
```python
# Key settings (with config.py integration)
VOCAB_SIZE = VOCAB_SIZE if 'VOCAB_SIZE' in dir() else 13000  # Uses config.py
SEQ_LEN = MAXLEN if 'MAXLEN' in dir() else 20                # Uses config.py
BATCH_SIZE = 256
HIDDEN_SIZE = 128
N_LAYERS = 12
ATTN_HEADS = 8
NUM_EPOCHS = 20

# Address-aware flags
USE_ADDRESS_EMBEDDING = True  # Hierarchical positions
USE_VAR_EMBEDDING = True      # Var offsets

# CUDA settings (uses config.py)
USE_CUDA = USE_CUDA if 'USE_CUDA' in dir() else True
CUDA_DEVICES = DEVICES if 'DEVICES' in dir() else [0]
```

### Step 4: Run Training

```bash
cd extern/PalmTree
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python train_palmtree_addressaware.py
```

## Model Architecture

```
Input Tokens: mov rax var(0x10)
     ↓
Token Embedding
     +
Sequence Position Embedding (standard BERT)
     +
Address Position Embedding (hierarchical, masked)
     +
Var Position Embedding (offsets, masked)
     +
Segment Embedding (NSP task)
     ↓
12 Transformer Layers
     ↓
Output: [batch, seq_len, hidden_size]
```

### Masking Logic

| Token | Address Emb | Var Emb | Why |
|-------|------------|---------|-----|
| `mov` | ✅ Full encoding | ❌ Zeros | Has address info in data |
| `address(0x401000)` | ✅ Full encoding | ❌ Zeros | Has address info |
| `var(0x10)` | ❌ Zeros | ✅ Full encoding (offset=16) | Has var offset |
| `rax` | ❌ Zeros | ❌ Zeros | Register, no position info |
| `[SOS]` | ❌ Zeros | ❌ Zeros | Special token |

**Sentinel values:**
- Address positions use `-1` for non-address tokens
- Var offsets use `-1` for non-var tokens
- Embeddings are masked to zeros where sentinels appear

## Comparison: Standard vs Address-Aware

| Aspect | Standard | Address-Aware |
|--------|----------|---------------|
| **Input** | `mov rax rbx` | `mov(0x401000:0.1:0.2:0.3) rax rbx` |
| **Model** | `BERT` | `AddressAwareBERT` |
| **Embeddings** | 3 types | 5 types (+ address + var) |
| **Training Speed** | Faster | ~10% slower |
| **Model Size** | Smaller | Larger (more params) |
| **Best For** | General tasks | Structure-aware tasks |

## Existing Infrastructure

The address-aware components were **already implemented** in the PalmTree codebase:

- ✅ `src/palmtree/model/bert_addressaware.py` - AddressAwareBERT model
- ✅ `src/palmtree/model/language_model_addressaware.py` - AddressAwareBERTLM wrapper
- ✅ `src/palmtree/dataset/dataset_addressaware.py` - BERTDatasetAddressAware
- ✅ `src/palmtree/trainer/pretrain_addressaware.py` - BERTTrainer with mode='addressaware'
- ✅ `src/palmtree/model/embedding/address_embedding.py` - AddressAwareBERTEmbedding

**What I added:**
- 📄 Training script that uses these components
- 📄 Comprehensive documentation
- 📄 Data format examples
- 📄 Comparison tools

## Next Steps

### 1. Train Your Model

```bash
cd extern/PalmTree
python train_palmtree_addressaware.py
```

### 2. Monitor Training

Checkpoints saved in `cdfg_bert_addressaware/transformer/`:
```
bert_trained_0.model
bert_trained_1.model
...
```

### 3. Use Trained Model

```python
from palmtree.model import AddressAwareBERT
import torch

model = AddressAwareBERT(vocab_size=13000, hidden=128, n_layers=12, attn_heads=8)
checkpoint = torch.load('cdfg_bert_addressaware/transformer/bert_trained_19.model')
model.load_state_dict(checkpoint['model_state_dict'])

# Extract embeddings
embeddings = model(tokens, segment, binary_pos, function_pos, bb_pos, var_offsets)
```

### 4. Fine-tune for Downstream Tasks

See `dstask/funcsim/` for examples of fine-tuning on function similarity.

## Troubleshooting

### "Import palmtree could not be resolved"

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/PalmTree/extern/PalmTree/src"
```

### Data format errors

Verify your data has the correct format:
```bash
head -1 data/training/cdfg_bert_addressaware/cfg_train.txt
```

Should see: `opcode(0xADDR:float:float:float) operand1 operand2`

### Out of memory

Reduce batch size:
```python
BATCH_SIZE = 128  # Instead of 256
```

## References

- **Training Script**: `extern/PalmTree/train_palmtree_addressaware.py`
- **Full Guide**: `extern/PalmTree/TRAIN_ADDRESSAWARE.md`
- **Data Format**: `extern/PalmTree/DATA_FORMAT_EXAMPLES.md`
- **Comparison**: Run `extern/PalmTree/compare_training.sh`

## Summary

You now have a **complete address-aware training pipeline** that:
1. Uses hierarchical address positions (binary/function/BB)
2. Encodes variable offsets explicitly
3. Applies smart masking based on token types
4. Is fully integrated with existing PalmTree infrastructure
5. Has comprehensive documentation and examples

The key innovation is that **address tokens get address embeddings**, **var tokens get var embeddings**, and **other tokens get neither** - creating clean semantic separation in the embedding space.
