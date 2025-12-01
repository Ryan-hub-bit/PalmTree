# Instruction Masking (IM) Task

## Overview

The Instruction Masking (IM) task is a new pretraining objective that masks **entire instructions** instead of individual tokens. This is different from the standard MLM (Masked Language Model) task which only masks individual tokens.

## Key Differences from MLM

| Feature | MLM (Token Masking) | IM (Instruction Masking) |
|---------|-------------------|-------------------------|
| **Masking Unit** | Individual tokens | Entire instructions |
| **Masking Rate** | ~15% of tokens | ~15% of instructions |
| **Example** | `mov [MASK] ebx` | `[MASK] [MASK] [MASK]` (all tokens in instruction) |
| **Objective** | Predict masked tokens | Predict all tokens in masked instruction |
| **Granularity** | Fine-grained | Coarse-grained |

## Architecture

### New Components

1. **InstructionMaskingDataset** (`dataloader_instruction_mask.py`)
   - Randomly selects instructions to mask at `instruction_mask_prob` rate
   - Replaces all tokens in selected instructions with `[MASK]`
   - Preserves position embeddings for masked instructions
   - Also supports standard MLM, NSP-CFG, NSP-DFG, and Scope tasks

2. **IM Prediction Head** (in `model.py`)
   - Added `enable_im` flag to `AddressAwareBERTForPretraining`
   - Dedicated `forward_im()` method for instruction masking
   - Shares same architecture as MLM head but trained separately

3. **Training Script** (`train_with_instruction_mask.py`)
   - Handles IM loss alongside other tasks
   - Supports DataParallel for multi-GPU training
   - Resume/checkpoint functionality

## Usage

### Quick Start

```bash
cd /home/kun/Document/PalmTree/strupos
./run_instruction_mask.sh
```

### Configuration

Edit `run_instruction_mask.sh` to customize:

```bash
# Masking rates
TOKEN_MASK_PROB=0.15          # Token-level masking (MLM)
INSTRUCTION_MASK_PROB=0.15     # Instruction-level masking (IM)

# Enable/disable tasks
TASKS="--enable_im"            # Instruction Masking
# TASKS="${TASKS} --enable_mlm"        # Token-level MLM
# TASKS="${TASKS} --enable_nsp_cfg"    # NSP on CFG
# TASKS="${TASKS} --enable_nsp_dfg"    # NSP on DFG
# TASKS="${TASKS} --enable_scope"      # Scope prediction
```

### Command-Line Arguments

```bash
python train_with_instruction_mask.py \
  --cfg_train /data/kun/dataset/train_cfg.txt \
  --dfg_train /data/kun/dataset/train_dfg.txt \
  --vocab ./vocab.pkl \
  --enable_im \
  --instruction_mask_prob 0.15 \
  --token_mask_prob 0.15 \
  --output_dir ../output/im_address \
  --log_dir ../log/im_address \
  --cuda
```

## Example

### Input (CFG line with 8 instructions)
```
mov(0x401000:0.1:0.2:0.3) eax ebx	add(0x401005:0.1:0.2:0.4) eax 1	...
```

### After Instruction Masking (15% rate, instruction 3 selected)
```
mov(0x401000:0.1:0.2:0.3) eax ebx	add(0x401005:0.1:0.2:0.4) eax 1	[MASK] [MASK] [MASK]	...
```

### Model Objective
Predict all tokens of the masked instruction (instruction 3).

## Benefits

1. **Holistic Understanding**: Model learns to predict entire instruction patterns, not just individual tokens
2. **Structural Learning**: Captures dependencies between opcode and operands within instructions
3. **Complementary to MLM**: Can be used together with MLM for multi-granularity learning
4. **Binary Code Specific**: Better suited for assembly where instructions are semantic units

## Output Files

After training, you'll find:

```
output/im_address/
├── best_model.pt           # Full model with best validation loss
├── best_bert.pt            # BERT encoder only (for downstream tasks)
├── checkpoint_latest.pt    # Latest checkpoint for resuming
├── checkpoint_epoch_N.pt   # Periodic checkpoints
└── args.json               # Training arguments

log/im_address/
└── train_YYYYMMDD_HHMMSS.log  # Training logs
```

## Experiments

### Recommended Configurations

**1. IM Only (Instruction-level understanding)**
```bash
TASKS="--enable_im"
```

**2. IM + Address Embeddings (Structure-aware)**
```bash
TASKS="--enable_im"
ADDRESS_FLAG=""  # Use address embeddings
```

**3. IM + MLM (Multi-granularity)**
```bash
TASKS="--enable_im --enable_mlm"
```

**4. Full Multi-task (All objectives)**
```bash
TASKS="--enable_im --enable_mlm --enable_nsp_cfg --enable_nsp_dfg --enable_scope"
```

### Ablation Studies

Test different masking rates:

```bash
# Low rate (10%)
--instruction_mask_prob 0.10

# Standard rate (15%)
--instruction_mask_prob 0.15

# High rate (25%)
--instruction_mask_prob 0.25
```

## Comparison with Existing Tasks

| Task | Granularity | Input Format | Objective |
|------|------------|--------------|-----------|
| **MLM** | Token | Single sequence | Predict masked tokens |
| **IM** | Instruction | Single sequence | Predict masked instructions |
| **NSP-CFG** | Pair | Two sequences | Predict if consecutive |
| **NSP-DFG** | Pair | Two sequences | Predict data flow relation |
| **SCOPE** | Pair | Two sequences | Classify scope level (3-class) |

## Implementation Details

### DataLoader

The `InstructionMaskingDataset` processes data as follows:

1. **Parse line into instructions** (tab-separated)
2. **For each instruction:**
   - Roll dice with `instruction_mask_prob`
   - If selected: Replace all tokens with `[MASK]`, store original as labels
   - If not selected: Store labels as `-1` (ignore in loss)
3. **Combine instructions** into sequence with `[SOS]` and `[EOS]`
4. **Pad/truncate** to `seq_len`
5. **Extract position embeddings** for address-aware training

### Loss Function

```python
# Cross-entropy with ignore_index=-1
im_criterion = nn.CrossEntropyLoss(ignore_index=-1)

# Only masked instructions contribute to loss
im_loss = im_criterion(im_output.view(-1, vocab_size), labels.view(-1))
```

### Multi-GPU Support

Works with `nn.DataParallel`:

```python
if hasattr(model, 'module'):
    im_output = model.module.forward_im(...)
else:
    im_output = model.forward_im(...)
```

## Performance Tips

1. **Batch Size**: Start with 256, increase if GPU memory allows
2. **Data Percentage**: Use 0.2 (20%) for quick experiments, 1.0 (100%) for full training
3. **Workers**: Set `--num_workers` to 4-8 for faster data loading
4. **Resume**: Always use `--resume` to continue from checkpoints
5. **Early Stopping**: Use `--early_stopping_patience 5` to prevent overfitting

## Citation

If you use this Instruction Masking task, please cite:

```bibtex
@misc{instruction_masking_2025,
  title={Instruction Masking for Binary Code Understanding},
  author={Your Name},
  year={2025},
  note={Extension to Address-Aware BERT}
}
```

## See Also

- `train_from_scratch.py` - Standard MLM+NSP training
- `train_multi_to_one.py` - Multi-to-one NSP variant
- `dataloader_all_pairs.py` - Standard dataloader
- `dataloader_scope.py` - Scope prediction dataloader
