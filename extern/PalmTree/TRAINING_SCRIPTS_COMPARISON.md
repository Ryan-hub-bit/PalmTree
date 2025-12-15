# Training Scripts Comparison

Both `train_palmtree.py` (standard) and `train_palmtree_addressaware.py` (address-aware) now use **identical structure and parameters** from `config.py`, with only these differences:

## Key Differences

### 1. Output Paths

**Standard:**
```python
vocab_path = "cdfg_bert_1/vocab"
train_cfg_dataset = "data/training/cdfg_bert_1/cfg_train.txt"
train_dfg_dataset = "data/training/cdfg_bert_1/dfg_train.txt"
test_cfg_dataset = "data/training/cdfg_bert_1/cfg_test.txt"
test_dfg_dataset = "data/training/cdfg_bert_1/dfg_test.txt"
output_path = "cdfg_bert_1/transformer"
```

**Address-Aware:**
```python
vocab_path = "cdfg_bert_addressaware/vocab"
train_cfg_dataset = "data/training/cdfg_bert_addressaware/cfg_train.txt"
train_dfg_dataset = "data/training/cdfg_bert_addressaware/dfg_train.txt"
test_cfg_dataset = "data/training/cdfg_bert_addressaware/cfg_test.txt"
test_dfg_dataset = "data/training/cdfg_bert_addressaware/dfg_test.txt"
output_path = "cdfg_bert_addressaware/transformer"
```

### 2. Model Import

**Standard:**
```python
from palmtree.model import BERT
```

**Address-Aware:**
```python
from palmtree.model import AddressAwareBERT
```

### 3. Dataset Import

**Standard:**
```python
# Uses built-in dataset
from palmtree import dataset
# ... later ...
train_dataset = dataset.BERTDataset(...)
```

**Address-Aware:**
```python
from palmtree.dataset.dataset_addressaware import BERTDatasetAddressAware
# ... later ...
train_dataset = BERTDatasetAddressAware(...)
```

### 4. Model Instantiation

**Standard:**
```python
bert = BERT(
    vocab_size=len(vocab),
    hidden=HIDDEN_SIZE,
    n_layers=N_LAYERS,
    attn_heads=ATTN_HEADS,
    dropout=DROPOUT
)
```

**Address-Aware:**
```python
bert = AddressAwareBERT(
    vocab_size=len(vocab),
    hidden=HIDDEN_SIZE,
    n_layers=N_LAYERS,
    attn_heads=ATTN_HEADS,
    dropout=DROPOUT,
    use_address_embedding=USE_ADDRESS_EMBEDDING,  # Additional parameter
    use_var_embedding=USE_VAR_EMBEDDING           # Additional parameter
)
```

### 5. Additional Flags (Address-Aware Only)

**Address-Aware:**
```python
# Address-aware specific flags
USE_ADDRESS_EMBEDDING = True  # Use hierarchical address positions
USE_VAR_EMBEDDING = True      # Use var(0xXX) offset embeddings
```

**Standard:**
```python
# No additional flags needed
```

### 6. Trainer Mode

**Standard:**
```python
trainer_instance = trainer.BERTTrainer(
    # ... all parameters same ...
    mode='original'  # KEY: Use standard mode
)
```

**Address-Aware:**
```python
trainer_instance = trainer.BERTTrainer(
    # ... all parameters same ...
    mode='addressaware'  # KEY: Use address-aware mode
)
```

## Shared Configuration

Both scripts use **identical parameters** from `config.py`:

```python
# From config.py (with fallbacks)
VOCAB_SIZE = VOCAB_SIZE if 'VOCAB_SIZE' in dir() else 13000
SEQ_LEN = MAXLEN if 'MAXLEN' in dir() else 20
USE_CUDA = USE_CUDA if 'USE_CUDA' in dir() else True
CUDA_DEVICES = DEVICES if 'DEVICES' in dir() else [0]

# Model hyperparameters (identical)
HIDDEN_SIZE = 128
N_LAYERS = 12
ATTN_HEADS = 8
DROPOUT = 0.1

# Training hyperparameters (identical)
LEARNING_RATE = 1e-5
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10000
NUM_EPOCHS = 20

# Data loading (identical)
BATCH_SIZE = 256
NUM_WORKERS = 10
MIN_FREQ = 1

# Logging (identical)
LOG_FREQ = 100
```

## Identical Structure

Both scripts follow the **exact same structure**:

```python
# 1. Imports
# 2. Configuration
# 3. Build Vocabulary
# 4. Load Vocabulary
# 5. Load Training Dataset
# 6. Load Test Dataset (Optional)
# 7. Create DataLoaders
# 8. Build BERT Model
# 9. Create BERT Trainer
# 10. Training Loop
```

## Summary Table

| Aspect | Standard | Address-Aware |
|--------|----------|---------------|
| **Structure** | ✓ Same | ✓ Same |
| **Config Integration** | ✓ Same | ✓ Same |
| **Hyperparameters** | ✓ Same | ✓ Same |
| **Model Class** | `BERT` | `AddressAwareBERT` |
| **Dataset Class** | `BERTDataset` | `BERTDatasetAddressAware` |
| **Output Directory** | `cdfg_bert_1/` | `cdfg_bert_addressaware/` |
| **Trainer Mode** | `'original'` | `'addressaware'` |
| **Address Embedding** | ✗ No | ✓ Yes |
| **Var Embedding** | ✗ No | ✓ Yes |
| **Data Format** | Simple tokens | Tokens + positions |

## Usage

**Standard PalmTree:**
```bash
cd extern/PalmTree
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python train_palmtree.py
```

**Address-Aware PalmTree:**
```bash
cd extern/PalmTree
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python train_palmtree_addressaware.py
```

## Data Requirements

**Standard:**
- Input: `mov rax rbx`
- No position information needed
- Works with existing data

**Address-Aware:**
- Input: `mov(0x401000:0.123:0.234:0.345) rax rbx`
- Requires address-annotated data
- Generate using `cfg_hierarchical_icfg_ida.py` and `dfg_hierarchical_idfg_ida.py`

## Switching Between Modes

To switch from standard to address-aware training:

1. **Generate address-annotated data** (if not already done)
2. **Change the script**: Use `train_palmtree_addressaware.py` instead of `train_palmtree.py`
3. **Same config.py**: No changes needed to configuration

To switch back to standard training:

1. **Use standard data** (without address annotations)
2. **Change the script**: Use `train_palmtree.py` instead of `train_palmtree_addressaware.py`
3. **Same config.py**: No changes needed to configuration

## Benefits of Unified Structure

✅ **Easy Configuration**: Both scripts use the same `config.py`  
✅ **Consistent Hyperparameters**: Guaranteed same training settings  
✅ **Easy Switching**: Just change the script name  
✅ **Maintainable**: Updates to one script template apply to both  
✅ **Clear Differences**: Only model and data handling differ  

## Code Maintenance

When updating hyperparameters or configuration:

1. **Edit `src/config.py`** for global settings (affects both scripts)
2. **Edit specific script** for script-specific settings (rare)
3. **Keep structure identical** when modifying either script

Both scripts should always have the same:
- Import pattern
- Configuration structure
- Section organization
- Logging format
- Training loop structure

Only the model class, dataset class, and output paths should differ.
