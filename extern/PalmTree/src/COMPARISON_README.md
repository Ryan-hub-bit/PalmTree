# PalmTree Comparison Training

This directory contains scripts to train and compare two versions of PalmTree:

1. **BASELINE**: Original BERT without address features
2. **ADDRESS-AWARE**: Your strupos version with address embeddings

## Overview

The comparison is designed to validate whether your address embeddings and position encodings actually improve the model's performance.

### BASELINE Mode
- **Model**: Standard BERT architecture (no address features)
- **Data Processing**: Position numbers in `address()` are **MASKED** (set to 0.0)
- **Purpose**: Show performance without address information
- **Key Difference**: Cannot use position information

Example:
```
Input:  call(0x401234:0.12345678:0.45678901:0.78901234)
After:  call(0x401234:0.00000000:0.00000000:0.00000000)
```

### ADDRESS-AWARE Mode
- **Model**: Your custom AddressAwareBERT with address layers
- **Data Processing**: Position numbers are **USED** for embeddings
- **Purpose**: Show performance with your address features
- **Key Difference**: Uses position embeddings, variable embeddings, and address-aware layers

Example:
```
Input:  call(0x401234:0.12345678:0.45678901:0.78901234)
Model uses: binary_pos=0.12345678, function_pos=0.45678901, bb_pos=0.78901234
```

## Data Format

Both modes use the same data files:
- `/data/kun/palmtreedata/cfg_train_2.txt` - CFG training data
- `/data/kun/palmtreedata/dfg_train_2.txt` - DFG training data  
- `/data/kun/palmtreedata/cfg_test_2.txt` - CFG test data
- `/data/kun/palmtreedata/dfg_test_2.txt` - DFG test data

The data format includes:
- Instructions in inline format with hierarchical positions
- `opcode(0xADDR:binary_pos:function_pos:bb_pos)` format
- `address(0xADDR:binary_pos:function_pos:bb_pos)` for operands
- `var(0xOFFSET)` for variable references

## Usage

### 1. Train Baseline Model

```bash
cd /home/kun/Document/PalmTree/extern/PalmTree/src
./train_baseline.sh
```

This will:
- Mask all position numbers (set to 0.0)
- Train standard BERT without address features
- Save checkpoints to `./output_comparison/baseline/`

### 2. Train Address-Aware Model

```bash
cd /home/kun/Document/PalmTree/extern/PalmTree/src
./train_address_aware.sh
```

This will:
- Use position numbers for address embeddings
- Train your AddressAwareBERT with custom layers
- Save checkpoints to `./output_comparison/address_aware/`

### 3. Compare Results

After training both models, compare:
- Training loss curves
- Test loss values
- Task-specific metrics (IMC, MLM accuracy)
- Convergence speed

Expected outcome: If your address features help, the address-aware model should:
- Achieve lower loss
- Learn faster (fewer epochs to converge)
- Perform better on instruction masking tasks

## Configuration

You can modify the training scripts to adjust:

**Model Size:**
- `HIDDEN_SIZE`: Hidden dimension (default: 768)
- `N_LAYERS`: Number of transformer layers (default: 12)
- `ATTN_HEADS`: Number of attention heads (default: 12)
- `ADDRESS_EMBED_DIM`: Address embedding dimension (default: 64, address-aware only)
- `VAR_EMBED_DIM`: Variable embedding dimension (default: 32, address-aware only)

**Training:**
- `BATCH_SIZE`: Batch size (default: 16)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `NUM_EPOCHS`: Number of training epochs (default: 20)
- `SEQ_LEN`: Maximum sequence length (default: 512)

**Masking:**
- `INSTRUCTION_MASK_PROB`: Probability of masking entire instructions (default: 0.25)
- `TOKEN_MASK_PROB`: Probability of masking individual tokens (default: 0.15)

**Data:**
- `DATA_PERCENTAGE`: Fraction of data to use (default: 1.0 = 100%)
  - Set to 0.1 for quick testing with 10% of data

## Quick Testing

For quick testing with small data:

```bash
# Edit the script and change:
DATA_PERCENTAGE=0.1  # Use 10% of data
NUM_EPOCHS=5         # Train for 5 epochs

# Then run normally:
./train_baseline.sh
./train_address_aware.sh
```

## Output Structure

```
output_comparison/
├── baseline/
│   ├── epoch_0.pt
│   ├── epoch_1.pt
│   ├── ...
│   └── best_model.pt
└── address_aware/
    ├── epoch_0.pt
    ├── epoch_1.pt
    ├── ...
    └── best_model.pt
```

Each checkpoint contains:
- Model state dict
- Optimizer state
- Training metrics
- Configuration

## Implementation Details

### Baseline Dataset (`palmtree/dataset/dataset_baseline.py`)
- Inherits from PyTorch Dataset
- Parses inline format with regex
- **Masks positions**: Replaces all position numbers with 0.0
- Implements instruction-level masking (IMC)
- Implements token-level masking (MLM)
- Returns: `{bert_input, bert_label}`

### Address-Aware Dataset (`strupos/dataloader.py`)
- Parses inline format with regex
- **Extracts positions**: Creates position embeddings from numbers
- **Extracts var offsets**: Creates variable embeddings
- Implements instruction-level masking (IMC, IMD)
- Implements token-level masking (MLM)
- Returns: `{bert_input, bert_label, positions, var_offsets, ...}`

### Training Script (`train_comparison.py`)
- Single unified script for both modes
- Mode selection via `--mode` argument
- Dynamically imports appropriate components
- Saves models to separate directories
- Tracks best model based on loss

## Comparison Metrics

To validate your address features, check:

1. **Loss Comparison**: Address-aware should have lower final loss
2. **Convergence Speed**: Address-aware might converge faster
3. **Task Performance**: 
   - IMC accuracy (instruction masking)
   - MLM accuracy (token masking)
4. **Generalization**: Test set performance

## Troubleshooting

**Import errors:**
- Make sure strupos is in the correct path
- Check that all required modules are installed

**Memory errors:**
- Reduce `BATCH_SIZE`
- Reduce `SEQ_LEN`
- Use smaller model (`HIDDEN_SIZE`, `N_LAYERS`)

**Data not found:**
- Check data paths in the shell scripts
- Make sure data files exist and are readable

## Next Steps

After training:
1. Compare training curves
2. Evaluate on downstream tasks
3. Analyze attention patterns
4. Test on real binaries

If address-aware performs better → Your features work! ✓
If baseline performs similarly → Might need to tune hyperparameters or try different tasks
