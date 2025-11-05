# Address-Aware PalmTree Setup Guide

This guide walks you through the complete workflow from binary analysis to model training.

## Quick Start

### Step 1: Generate Training Data (Required)

First, generate CFG and DFG sequences with address information from your binaries:

```bash
cd /home/kun/Document/PalmTree/scripts
./generate_data.sh
```

This will:
- Process all binaries in the configured folder
- Generate CFG sequences with address normalization
- Generate DFG sequences with address normalization
- Output: `combined_output.txt` in the configured output directory

**Default configuration** (edit the script to change):
- SEG_LEN=300 (sequence length)
- BIN_FOLDER="/home/kun/test_bins"
- Output to separate CFG/DFG directories

### Step 2: Test the System (Recommended)

Verify all components work correctly:

```bash
cd /home/kun/Document/PalmTree/addressaware
python3 test.py
```

Expected output:
```
============================================================
Address-Aware PalmTree Test Suite
============================================================
Testing dataloader...
✓ Dataloader parsing works!

Testing address positional embedding...
✓ Address positional embedding works!

Testing model...
✓ Model forward pass works!

============================================================
✓ All tests passed!
============================================================
```

### Step 3: Train the Model

```bash
cd /home/kun/Document/PalmTree/addressaware
./train_example.sh
```

This will:
1. Check if vocabulary exists, build it if not
2. Train the address-aware BERT model
3. Save checkpoints to `output/` directory
4. Print training metrics every epoch

## Important Configuration Notes

### Hidden Size Requirements

The `hidden` parameter (d_model) must meet TWO requirements:

1. **Divisible by 3**: For the three address levels (binary, function, basic block)
2. **Divisible by attention heads**: For multi-head attention

**Recommended values**:
- **768** (divisible by 3, 6, 12) ✓ Default, works well
- **120** (divisible by 3, 4) ✓ Good for testing
- ~~128~~ (not divisible by 3) ✗ Will fail

Example error if not divisible by 3:
```
AssertionError: d_model should be divisible by 3 for three address levels
```

### Data Format

The system expects inline address format:
```
opcode(0xADDR:bnorm:fnorm:bbnorm) operands
```

Where:
- **ADDR**: Actual instruction address in hex
- **bnorm**: Binary-level position [0, 1]
- **fnorm**: Function-level position [0, 1]
- **bbnorm**: Basic block-level position [0, 1]

Example:
```
mov(0x401008:0.022:0.296:0.444) rax qword [ rel addr_data(0x403fe8:1.000:0.000:0) ]
```

**Positional Embedding Strategy:**

Only tokens with explicit address information receive position embeddings:
- **Opcodes** (e.g., `mov`, `test`, `call`) → Use their address positions
- **Address tokens** (e.g., `addr_code`, `addr_data`) → Use their address positions  
- **Regular operands** (e.g., `rax`, `qword`, `[`, `]`, `rel`) → Use (0, 0, 0)

This selective strategy focuses the model on learning structural patterns from instruction locations while allowing operand tokens to learn their semantics independently.

## Customizing Training

### Edit train_example.sh

Open `train_example.sh` and modify the configuration at the top:

```bash
# Data paths
CFG_DATA="../data/cfg_output/combined_output.txt"
DFG_DATA="../data/dfg_output/combined_output.txt"
VOCAB_FILE="vocab.pkl"
OUTPUT_DIR="output"

# Model configuration
HIDDEN=768          # Must be divisible by 3 and attn_heads
N_LAYERS=12
ATTN_HEADS=12       # Must divide HIDDEN evenly
MAX_LEN=512

# Training configuration
EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=0.0001
DROPOUT=0.1
MASK_PROB=0.15      # MLM masking probability
NSP_PROB=0.5        # NSP negative sample rate
```

### Memory Considerations

If you run out of GPU memory:
- Reduce `BATCH_SIZE` (e.g., 16 or 8)
- Reduce `MAX_LEN` (e.g., 256 or 128)
- Reduce `HIDDEN` (e.g., 384, but must still be divisible by 3)

### For Multiple GPUs

Uncomment in `train_example.sh`:
```bash
USE_MULTI_GPU="--multi_gpu"
```

## Project Structure

```
addressaware/
├── dataloader.py              # Dataset loader
├── address_embedding.py       # Three-level positional encoding
├── model.py                   # Address-aware BERT
├── train.py                   # Training script
├── test.py                    # Test suite
├── train_example.sh           # Training launcher (executable)
├── README.md                  # Architecture documentation
└── SETUP_GUIDE.md            # This file
```

## Troubleshooting

### "d_model should be divisible by 3"
- Your `HIDDEN` parameter is not divisible by 3
- Change it to 768, 384, 120, or another multiple of 3

### "data file not found"
- Run `generate_data.sh` first to create training data
- Check paths in `train_example.sh` match your output directories

### "Import errors"
- Make sure you're running from the `addressaware/` directory
- The scripts set up paths to import from `../src/palmtree`

### Out of memory
- Reduce `BATCH_SIZE` in `train_example.sh`
- Reduce `MAX_LEN` or `HIDDEN`

### Test fails
- Check if PyTorch is installed: `python3 -c "import torch; print(torch.__version__)"`
- Check if Binary Ninja is available (for data generation only)

## Next Steps

After training completes:
1. Checkpoints are saved in `output/checkpoint_epoch_*.pt`
2. Use the trained model for downstream tasks:
   - Binary function similarity
   - Malware classification
   - Vulnerability detection
   - Code clone detection

See `README.md` for architecture details and extending the model.

## Performance Tips

1. **Use a vocabulary**: The example script builds it automatically from CFG data
2. **Monitor validation**: The training script reports MLM and NSP accuracy
3. **Adjust learning rate**: Default is 0.0001, may need tuning
4. **Save disk space**: Training generates ~1GB checkpoints, delete old ones if needed

## Questions?

Check:
- `README.md` - Architecture and design details
- `test.py` - Example usage of each component
- `train.py` - All available command-line arguments
