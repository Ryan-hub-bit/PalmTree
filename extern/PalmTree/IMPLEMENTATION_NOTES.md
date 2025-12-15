# Address-Aware PalmTree Setup - Implementation Notes

## Changes Made (December 15, 2025)

### Configuration Integration

The training script now properly integrates with `src/config.py`:

```python
# src/config.py values are used where available
VOCAB_SIZE = VOCAB_SIZE if 'VOCAB_SIZE' in dir() else 13000
SEQ_LEN = MAXLEN if 'MAXLEN' in dir() else 20
USE_CUDA = USE_CUDA if 'USE_CUDA' in dir() else True
CUDA_DEVICES = DEVICES if 'DEVICES' in dir() else [0]
```

### Self-Contained Implementation

**No dependencies on strupos:**
- All address-aware components are in `extern/PalmTree/src/palmtree/`
- The `address_embedding.py` is self-contained with torch/torch.nn only
- No cross-repository imports required

### Module Verification

Checked all address-aware modules for external dependencies:

✅ `src/palmtree/model/bert_addressaware.py`
- Imports: torch.nn, .transformer, .embedding.address_embedding
- Self-contained: YES

✅ `src/palmtree/model/language_model_addressaware.py`
- Imports: torch, torch.nn, .bert_addressaware
- Self-contained: YES

✅ `src/palmtree/model/embedding/address_embedding.py`
- Imports: torch, torch.nn, math
- Self-contained: YES

✅ `src/palmtree/dataset/dataset_addressaware.py`
- Imports: torch, tqdm, random, re
- Self-contained: YES

✅ `src/palmtree/trainer/pretrain_addressaware.py`
- Imports: torch, torch.nn, torch.optim, ..model, .optim_schedule
- Self-contained: YES

### Configuration Flow

```
src/config.py
    ↓
train_palmtree_addressaware.py (reads config)
    ↓
Uses config values:
    - VOCAB_SIZE (with fallback: 13000)
    - MAXLEN → SEQ_LEN (with fallback: 20)
    - USE_CUDA (with fallback: True)
    - DEVICES → CUDA_DEVICES (with fallback: [0])
```

### File Structure

```
extern/PalmTree/
├── src/
│   ├── config.py                          # Global configuration
│   └── palmtree/
│       ├── model/
│       │   ├── bert_addressaware.py       # ✅ Self-contained
│       │   ├── language_model_addressaware.py  # ✅ Self-contained
│       │   └── embedding/
│       │       └── address_embedding.py   # ✅ Self-contained
│       ├── dataset/
│       │   └── dataset_addressaware.py    # ✅ Self-contained
│       └── trainer/
│           └── pretrain_addressaware.py   # ✅ Self-contained
│
└── train_palmtree_addressaware.py         # Uses config.py values
```

### Key Implementation Details

1. **Config Integration:**
   - Script reads from `config.py` using `from config import *`
   - Fallback values ensure script works even if config is minimal
   - Conditional assignment: `VAR = CONFIG_VAR if 'CONFIG_VAR' in dir() else DEFAULT`

2. **No External Dependencies:**
   - No imports from `strupos/`
   - All address-aware logic is within `palmtree/` module
   - Standard library + PyTorch only

3. **Embedding Components:**
   - `SequencePositionalEmbedding`: Standard sinusoidal (BERT-style)
   - `AddressPositionalEmbedding`: 3-level hierarchical with masking
   - `VarPositionalEmbedding`: Offset-based with masking
   - All use sentinel value `-1` for "not applicable"

4. **Masking Strategy:**
   - Address mask: `(binary_pos >= 0) & (function_pos >= 0) & (bb_pos >= 0)`
   - Var mask: `var_offsets >= 0`
   - Masked tokens get zero vectors via multiplication

### Usage

**1. Configure global settings:**
```bash
vim extern/PalmTree/src/config.py
```

**2. Configure training settings:**
```bash
vim extern/PalmTree/train_palmtree_addressaware.py
```

**3. Run training:**
```bash
cd extern/PalmTree
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python train_palmtree_addressaware.py
```

### Verification

Run the verification script to check setup:
```bash
cd extern/PalmTree
python verify_setup.py
```

Checks:
- ✓ All files exist
- ✓ Data format is correct
- ✓ Imports work (no strupos dependencies)
- ✓ Model can be instantiated
- ✓ PyTorch + CUDA available

### Configuration Example

**Minimal `src/config.py`:**
```python
VOCAB_SIZE = 13000
USE_CUDA = True
DEVICES = [0, 1]  # Multi-GPU
MAXLEN = 30
```

**Training script adapts automatically:**
```python
VOCAB_SIZE = 13000  # From config
SEQ_LEN = 30        # From MAXLEN in config
USE_CUDA = True     # From config
CUDA_DEVICES = [0, 1]  # From DEVICES in config
```

### Testing

To verify no strupos dependencies:
```bash
cd extern/PalmTree
grep -r "from strupos\|import strupos" src/palmtree/
# Should return: No matches
```

Result: ✅ **PASSED** - No strupos imports found

### Summary

✅ Training script uses `config.py` values with sensible fallbacks  
✅ All modules are self-contained (no strupos dependencies)  
✅ Address embedding logic is fully within palmtree package  
✅ Configuration is flexible and modular  
✅ Documentation updated to reflect config integration  

The address-aware PalmTree implementation is now:
- **Self-contained**: No external dependencies
- **Configurable**: Uses config.py for global settings
- **Flexible**: Fallback defaults ensure it always works
- **Production-ready**: Clean separation of concerns
