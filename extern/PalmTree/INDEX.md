# Address-Aware PalmTree Training - File Index

## Summary

Address-aware training has been fully integrated into `extern/PalmTree/`. This enhancement adds hierarchical address positional embeddings and variable offset embeddings to the standard PalmTree model.

**Key Innovation:** Token-specific masking ensures opcodes get address embeddings, variables get offset embeddings, and other tokens get zeros - creating clean semantic separation.

---

## Created Files

### 📜 Training Scripts

#### `train_palmtree_addressaware.py`
**Main training script for address-aware PalmTree**

- Uses `AddressAwareBERT` model
- Uses `BERTDatasetAddressAware` for data loading
- Trains with MLM + DUP + CWP tasks
- Fully configurable hyperparameters
- Supports CUDA multi-GPU training

**Usage:**
```bash
cd extern/PalmTree
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python train_palmtree_addressaware.py
```

---

### 📚 Documentation

#### `README_ADDRESSAWARE.md`
**Quick start guide**

- What was added (overview)
- Quick start (4 steps)
- Model architecture diagram
- Masking logic table
- Comparison: standard vs address-aware
- Troubleshooting

**Read this first!**

#### `TRAIN_ADDRESSAWARE.md`
**Comprehensive training guide**

- Overview of address-aware features
- Data format requirements
- Training setup (step-by-step)
- Model architecture details
- Masking strategy explanation
- Output and checkpoint usage
- Troubleshooting section

**Complete reference documentation**

#### `DATA_FORMAT_EXAMPLES.md`
**Data format specification with examples**

- Standard vs address-aware format comparison
- Token format breakdown
- Position normalization examples (with calculations)
- Variable offset examples
- Real assembly function example
- Token type summary table
- Validation scripts

**Essential for data preparation**

---

### 🔧 Utility Scripts

#### `compare_training.sh`
**Side-by-side comparison tool**

Displays feature comparison between standard and address-aware training.

**Usage:**
```bash
cd extern/PalmTree
./compare_training.sh
```

**Output:**
- Model comparison
- Feature comparison table
- Usage instructions

#### `verify_setup.py`
**Setup verification script**

Checks if everything is properly configured before training.

**Usage:**
```bash
cd extern/PalmTree
python verify_setup.py
```

**Checks:**
- ✓ Required files exist
- ✓ Data format is correct
- ✓ Imports work properly
- ✓ Model can be instantiated
- ✓ PyTorch + CUDA available

---

## Existing Infrastructure (Already in Codebase)

The following components were **already implemented** in PalmTree:

### Models
- `src/palmtree/model/bert_addressaware.py` - AddressAwareBERT implementation
- `src/palmtree/model/language_model_addressaware.py` - AddressAwareBERTLM wrapper

### Embeddings
- `src/palmtree/model/embedding/address_embedding.py` - AddressAwareBERTEmbedding
  - SequencePositionalEmbedding (standard)
  - AddressPositionalEmbedding (3-level hierarchical)
  - VarPositionalEmbedding (offset-based)

### Dataset
- `src/palmtree/dataset/dataset_addressaware.py` - BERTDatasetAddressAware
  - Parses inline address information
  - Extracts positions and var offsets
  - Handles masking for MLM task

### Trainer
- `src/palmtree/trainer/pretrain_addressaware.py` - BERTTrainer with mode='addressaware'
  - Supports both standard and address-aware modes
  - Handles MLM + NSP + DUP losses
  - Checkpoint saving and loading

---

## File Dependency Graph

```
train_palmtree_addressaware.py
    │
    ├─→ palmtree.model.AddressAwareBERT
    │       └─→ AddressAwareBERTEmbedding
    │               ├─→ TokenEmbedding
    │               ├─→ SequencePositionalEmbedding
    │               ├─→ AddressPositionalEmbedding (3 levels)
    │               ├─→ VarPositionalEmbedding (offsets)
    │               └─→ SegmentEmbedding
    │
    ├─→ palmtree.dataset.dataset_addressaware.BERTDatasetAddressAware
    │       └─→ parse_instruction_with_address()
    │               ├─→ Extract tokens
    │               ├─→ Extract positions (binary, function, BB)
    │               └─→ Extract var offsets
    │
    └─→ palmtree.trainer.BERTTrainer (mode='addressaware')
            └─→ AddressAwareBERTLM
                    ├─→ MLM (Masked Language Model)
                    ├─→ DUP (Data Use Prediction)
                    └─→ CWP (Control Walk Prediction)
```

---

## Quick Reference

### Data Format

**Standard:**
```
mov rax rbx	push rbp
```

**Address-Aware:**
```
mov(0x401000:0.000000:0.000000:0.000000) rax rbx	push(0x401003:0.000015:0.012500:0.166667) rbp
```

### Generate Data

```bash
python data_generator/cfg_hierarchical_icfg_ida.py --input binary --output cfg_train.txt
python data_generator/dfg_hierarchical_idfg_ida.py --input binary --output dfg_train.txt
```

### Training Command

```bash
cd extern/PalmTree
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python train_palmtree_addressaware.py
```

### Model Usage

```python
from palmtree.model import AddressAwareBERT

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

# Extract embeddings
embeddings = model(tokens, segment, binary_pos, function_pos, bb_pos, var_offsets)
```

---

## Configuration Reference

### Global Configuration (`src/config.py`)

```python
VOCAB_SIZE = 10000      # Maximum vocabulary size (used by training script)
USE_CUDA = True         # Enable CUDA (used by training script)
DEVICES = [0]           # CUDA device IDs (used by training script)
MAXLEN = 10            # Default sequence length (used by training script)
```

### Model Hyperparameters (`train_palmtree_addressaware.py`)

```python
# Uses config.py values with fallbacks
VOCAB_SIZE = VOCAB_SIZE if 'VOCAB_SIZE' in dir() else 13000
SEQ_LEN = MAXLEN if 'MAXLEN' in dir() else 20
HIDDEN_SIZE = 128      # Embedding dimension
N_LAYERS = 12          # Number of transformer layers
ATTN_HEADS = 8         # Number of attention heads
DROPOUT = 0.1          # Dropout rate
```

### Training Hyperparameters

```python
BATCH_SIZE = 256       # Batch size
LEARNING_RATE = 1e-5   # Learning rate
WEIGHT_DECAY = 0.01    # AdamW weight decay
WARMUP_STEPS = 10000   # Learning rate warmup
NUM_EPOCHS = 20        # Training epochs
```

### Address-Aware Flags

```python
USE_ADDRESS_EMBEDDING = True  # Hierarchical address positions
USE_VAR_EMBEDDING = True      # Var(0xXX) offset embeddings

# CUDA settings (uses config.py values with fallbacks)
USE_CUDA = USE_CUDA if 'USE_CUDA' in dir() else True
CUDA_DEVICES = DEVICES if 'DEVICES' in dir() else [0]
```

---

## Directory Structure

```
extern/PalmTree/
├── train_palmtree_addressaware.py    # Main training script
├── README_ADDRESSAWARE.md             # Quick start guide
├── TRAIN_ADDRESSAWARE.md              # Complete training guide
├── DATA_FORMAT_EXAMPLES.md            # Data format specification
├── compare_training.sh                # Comparison tool
├── verify_setup.py                    # Setup verification
│
├── data/
│   └── training/
│       └── cdfg_bert_addressaware/
│           ├── cfg_train.txt          # CFG training data
│           ├── dfg_train.txt          # DFG training data
│           ├── cfg_test.txt           # (Optional) CFG test data
│           └── dfg_test.txt           # (Optional) DFG test data
│
├── cdfg_bert_addressaware/
│   ├── vocab                          # Vocabulary file
│   └── transformer/
│       ├── bert_trained_0.model       # Checkpoint epoch 0
│       ├── bert_trained_1.model       # Checkpoint epoch 1
│       └── ...                        # More checkpoints
│
└── src/
    └── palmtree/
        ├── model/
        │   ├── bert_addressaware.py
        │   ├── language_model_addressaware.py
        │   └── embedding/
        │       └── address_embedding.py
        ├── dataset/
        │   └── dataset_addressaware.py
        └── trainer/
            └── pretrain_addressaware.py
```

---

## Workflow

```
1. Generate Data
   ├─ cfg_hierarchical_icfg_ida.py
   └─ dfg_hierarchical_idfg_ida.py
        ↓
2. Verify Setup
   └─ verify_setup.py
        ↓
3. Train Model
   └─ train_palmtree_addressaware.py
        ↓
4. Checkpoints
   └─ cdfg_bert_addressaware/transformer/*.model
        ↓
5. Use Model
   ├─ Fine-tune for downstream tasks
   ├─ Extract embeddings
   └─ Evaluate performance
```

---

## Key Features

### 1. Hierarchical Address Embeddings

Three levels of sinusoidal positional encoding:
- **Binary-level**: Position in entire binary (0.0 to 1.0)
- **Function-level**: Position in function (0.0 to 1.0)
- **Basic block-level**: Position in basic block (0.0 to 1.0)

### 2. Variable Offset Embeddings

Sinusoidal encoding based on stack offset:
- `var(0x0)` → offset = 0
- `var(0x10)` → offset = 16
- `var(0x20)` → offset = 32

### 3. Smart Masking

Token-specific embedding application:
- **Opcodes/addresses**: Get address embeddings (full encoding)
- **Variables**: Get var embeddings (offset-based encoding)
- **Registers/special**: Get zeros for both

Uses `-1` as sentinel value for "not applicable".

---

## Next Steps

1. **Read** `README_ADDRESSAWARE.md` for quick start
2. **Verify** setup with `verify_setup.py`
3. **Review** `DATA_FORMAT_EXAMPLES.md` for data format
4. **Generate** training data with hierarchical position tracking
5. **Configure** `train_palmtree_addressaware.py` hyperparameters
6. **Train** model with `python train_palmtree_addressaware.py`
7. **Use** trained model for downstream tasks

---

## Support

For issues or questions:
1. Check `TRAIN_ADDRESSAWARE.md` troubleshooting section
2. Run `verify_setup.py` to diagnose problems
3. Review `DATA_FORMAT_EXAMPLES.md` for data format issues
4. Compare with standard training using `compare_training.sh`

---

**Created:** December 15, 2025  
**Purpose:** Add address-aware training capability to PalmTree  
**Status:** Complete and ready to use
