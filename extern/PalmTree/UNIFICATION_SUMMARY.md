# Training Scripts Unification Summary

## Changes Completed (December 15, 2025)

Both `train_palmtree.py` and `train_palmtree_addressaware.py` have been **unified** to use identical structure and configuration.

## What Changed

### train_palmtree.py (Standard)
- ✅ Updated to use `config.py` parameters
- ✅ Aligned structure with address-aware version
- ✅ Added proper sections and comments
- ✅ Consistent error handling and logging
- ✅ Same hyperparameters and training loop

### train_palmtree_addressaware.py (Address-Aware)
- ✅ Already using `config.py` parameters
- ✅ Structure matches standard version
- ✅ Only differences: model, dataset, and flags
- ✅ No dependencies on strupos module

## Verification

### ✅ No External Dependencies
```bash
$ grep -r "from strupos\|import strupos" src/ train_*.py
✓ No strupos imports found
```

### ✅ Identical Configuration Structure
Both scripts use the same configuration pattern:
```python
# From config.py (with fallbacks)
VOCAB_SIZE = VOCAB_SIZE if 'VOCAB_SIZE' in dir() else 13000
SEQ_LEN = MAXLEN if 'MAXLEN' in dir() else 20
USE_CUDA = USE_CUDA if 'USE_CUDA' in dir() else True
CUDA_DEVICES = DEVICES if 'DEVICES' in dir() else [0]

# Identical hyperparameters
HIDDEN_SIZE = 128
N_LAYERS = 12
ATTN_HEADS = 8
DROPOUT = 0.1
LEARNING_RATE = 1e-5
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10000
NUM_EPOCHS = 20
BATCH_SIZE = 256
NUM_WORKERS = 10
```

### ✅ Only 6 Key Differences

1. **Output directory names**: `cdfg_bert_1/` vs `cdfg_bert_addressaware/`
2. **Model import**: `BERT` vs `AddressAwareBERT`
3. **Dataset import**: `BERTDataset` vs `BERTDatasetAddressAware`
4. **Model flags**: None vs `use_address_embedding`, `use_var_embedding`
5. **Trainer mode**: `'original'` vs `'addressaware'`
6. **Print messages**: "Standard" vs "Address-Aware"

Everything else is **100% identical**.

## File Structure

```
extern/PalmTree/
├── src/
│   ├── config.py                          # Shared configuration
│   └── palmtree/                          # All self-contained
│       ├── model/
│       │   ├── bert.py                    # Standard BERT
│       │   ├── bert_addressaware.py       # Address-aware BERT
│       │   └── embedding/
│       │       └── address_embedding.py   # Address embeddings
│       ├── dataset/
│       │   ├── dataset.py                 # Standard dataset
│       │   └── dataset_addressaware.py    # Address-aware dataset
│       └── trainer/
│           └── pretrain_addressaware.py   # Unified trainer
│
├── train_palmtree.py                      # Standard training
└── train_palmtree_addressaware.py         # Address-aware training
```

## Configuration Flow

```
config.py (Global Settings)
    ↓
    ├→ train_palmtree.py
    │      ├─ Uses: VOCAB_SIZE, MAXLEN, USE_CUDA, DEVICES
    │      ├─ Model: BERT
    │      ├─ Dataset: BERTDataset
    │      └─ Mode: 'original'
    │
    └→ train_palmtree_addressaware.py
           ├─ Uses: VOCAB_SIZE, MAXLEN, USE_CUDA, DEVICES (same)
           ├─ Model: AddressAwareBERT
           ├─ Dataset: BERTDatasetAddressAware
           ├─ Additional: USE_ADDRESS_EMBEDDING, USE_VAR_EMBEDDING
           └─ Mode: 'addressaware'
```

## Usage Examples

### Edit Configuration Once, Affects Both

**Edit `src/config.py`:**
```python
VOCAB_SIZE = 15000    # Both scripts will use this
USE_CUDA = True       # Both scripts will use this
DEVICES = [0, 1, 2]   # Both scripts will use this
MAXLEN = 30          # Both scripts will use this
```

### Run Standard Training
```bash
cd extern/PalmTree
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python train_palmtree.py
# Uses config.py values
# Output: cdfg_bert_1/
```

### Run Address-Aware Training
```bash
cd extern/PalmTree
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python train_palmtree_addressaware.py
# Uses same config.py values
# Output: cdfg_bert_addressaware/
```

## Benefits

### 1. Consistency
- ✅ Same hyperparameters guaranteed
- ✅ Same training settings
- ✅ Easy to compare results

### 2. Maintainability
- ✅ Update config once, affects both
- ✅ Identical structure easy to maintain
- ✅ Clear separation of concerns

### 3. Flexibility
- ✅ Easy to switch between modes
- ✅ Same data pipeline (except dataset class)
- ✅ Compatible with same trainer

### 4. Self-Contained
- ✅ No external dependencies
- ✅ No strupos imports needed
- ✅ All code in palmtree module

## Testing

### Verify Configuration
```bash
cd extern/PalmTree
python -c "
import sys
sys.path.insert(0, 'src')
from config import *
print(f'VOCAB_SIZE: {VOCAB_SIZE}')
print(f'USE_CUDA: {USE_CUDA}')
print(f'DEVICES: {DEVICES}')
print(f'MAXLEN: {MAXLEN}')
"
```

### Verify No External Dependencies
```bash
cd extern/PalmTree
grep -r "from strupos\|import strupos" src/ train_*.py || echo "✓ Clean"
```

### Verify Structure Alignment
```bash
cd extern/PalmTree
diff <(grep "^# ====" train_palmtree.py) \
     <(grep "^# ====" train_palmtree_addressaware.py)
# Should show identical section markers
```

## Quick Reference

| Feature | train_palmtree.py | train_palmtree_addressaware.py |
|---------|-------------------|--------------------------------|
| Config Integration | ✅ Yes | ✅ Yes |
| Uses config.py | ✅ Yes | ✅ Yes |
| Self-Contained | ✅ Yes | ✅ Yes |
| Structure | ✅ Unified | ✅ Unified |
| Hyperparameters | ✅ Identical | ✅ Identical |
| Model | BERT | AddressAwareBERT |
| Dataset | BERTDataset | BERTDatasetAddressAware |
| Mode | 'original' | 'addressaware' |
| Output Dir | cdfg_bert_1/ | cdfg_bert_addressaware/ |
| Address Emb | ❌ No | ✅ Yes |
| Var Emb | ❌ No | ✅ Yes |

## Next Steps

1. **Configure** `src/config.py` with your desired settings
2. **Choose** which training mode you need:
   - Standard: Use `train_palmtree.py`
   - Address-Aware: Use `train_palmtree_addressaware.py`
3. **Run** the training script
4. **Monitor** training progress (both use same logging format)
5. **Compare** results from both modes if needed

## Documentation

- **Training Comparison**: `TRAINING_SCRIPTS_COMPARISON.md`
- **Implementation Notes**: `IMPLEMENTATION_NOTES.md`
- **Address-Aware Guide**: `TRAIN_ADDRESSAWARE.md`
- **Quick Start**: `README_ADDRESSAWARE.md`
- **Master Index**: `INDEX.md`

---

**Status**: ✅ **Complete**  
**Date**: December 15, 2025  
**Summary**: Both training scripts now share identical structure and use the same configuration, with only model-specific differences.
