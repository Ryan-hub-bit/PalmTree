# MLM and IM Task Separation

## Overview
The dataloader and training script now provide **separate** MLM (Masked Language Modeling) and IM (Instruction Masking) tasks, allowing the model to learn both token-level and instruction-level representations.

## Key Changes

### 1. Dataloader (`dataloader_instruction_mask.py`)

#### New Method: `_process_line_for_token_masking()`
- Applies **token-level masking** (standard BERT MLM)
- Randomly masks 15% of tokens (configurable via `token_mask_prob`)
- Masking strategy:
  - 80% → Replace with [MASK]
  - 10% → Replace with random token
  - 10% → Keep original
- Returns labels: original token IDs for masked positions, -1 for unmasked

#### Updated Method: `__getitem__()`
Now returns **4 data batches** per sample:
```python
{
    'im': {...},        # Instruction-level masking (25% of instructions)
    'mlm': {...},       # Token-level masking (15% of tokens)
    'nsp_cfg': {...},   # NSP on CFG pairs
    'nsp_dfg': {...},   # NSP on DFG pairs
}
```

**Important**: `im` and `mlm` come from the **same CFG line** but with **different random masking**, providing independent training signals.

### 2. Training Script (`train_with_instruction_mask.py`)

#### Updated `train_epoch()` Function
Now properly implements MLM:

**IM Processing** (lines ~76-103):
```python
if enable_im:
    im_batch = batch['im']
    # ... forward through model.forward_im()
    im_loss = criterion(im_output, im_labels)
```

**MLM Processing** (lines ~105-132):
```python
if enable_mlm:
    mlm_batch = batch['mlm']
    # ... forward through model()
    mlm_loss = criterion(mlm_output, mlm_labels)
```

Both losses are combined:
```python
loss = im_loss + mlm_loss + nsp_cfg_loss + nsp_dfg_loss + scope_loss
```

#### Updated `validate_epoch()` Function
Same structure as training - processes IM and MLM separately.

## Task Comparison

| Feature | MLM (Token-Level) | IM (Instruction-Level) |
|---------|-------------------|------------------------|
| **Granularity** | Individual tokens | Entire instructions |
| **Masking Rate** | 15% of tokens | 25% of instructions |
| **Masking Strategy** | 80/10/10 (MASK/random/keep) | 100% MASK all tokens |
| **Forward Method** | `model()` (standard) | `model.forward_im()` |
| **Prediction Head** | `model.mask_lm` | `model.IM` |
| **Learning Goal** | Token-level semantics | Instruction-level patterns |

## Training Flow

For each batch:
1. **IM Sample**: Same CFG line, ~25% instructions fully masked
   - Model predicts all tokens in masked instructions
   
2. **MLM Sample**: Same CFG line, ~15% tokens randomly masked
   - Model predicts individual masked tokens
   
3. **NSP-CFG**: Consecutive CFG instruction pairs
   - Model predicts if inst2 follows inst1
   
4. **NSP-DFG**: DFG instruction pairs
   - Model predicts data flow relationships

5. **Scope** (optional): Directory scope prediction
   - Model predicts which directory a function belongs to

## Usage

### Enable Both Tasks
```bash
python train_with_instruction_mask.py \
    --enable_im \
    --enable_mlm \
    --cfg_corpus ../data/cfg_2/all_cfg_combined.txt \
    --dfg_corpus ../data/dfg_2/all_dfg_combined.txt \
    --vocab_path ../pre-trained_model/palmtree/vocab_1157.pkl
```

### Enable Only IM
```bash
python train_with_instruction_mask.py \
    --enable_im \
    --cfg_corpus ../data/cfg_2/all_cfg_combined.txt \
    --dfg_corpus ../data/dfg_2/all_dfg_combined.txt
```

### Enable Only MLM
```bash
python train_with_instruction_mask.py \
    --enable_mlm \
    --cfg_corpus ../data/cfg_2/all_cfg_combined.txt \
    --dfg_corpus ../data/dfg_2/all_dfg_combined.txt
```

## Expected Output

With `--enable_im --enable_mlm`:
```
Training:   3%| | 357/11784 [00:43<22:14,  8.56it/s, loss=5.2341, im=2.8123, mlm=2.4218, nsp_cfg=0, nsp_dfg=0, scope=0]
```

- **loss**: Total combined loss
- **im**: Instruction masking loss (should be > 0)
- **mlm**: Token masking loss (should be > 0)
- **nsp_cfg**: CFG NSP loss (if enabled)
- **nsp_dfg**: DFG NSP loss (if enabled)
- **scope**: Scope prediction loss (if enabled)

## Benefits of Dual Tasks

1. **Complementary Learning**:
   - MLM learns fine-grained token semantics
   - IM learns coarse-grained instruction patterns

2. **Robustness**:
   - Multiple masking strategies improve generalization
   - Model sees same data with different masks

3. **Hierarchical Understanding**:
   - Token-level: Individual opcode/operand meanings
   - Instruction-level: Complete instruction semantics

4. **Better Representations**:
   - MLM captures local dependencies
   - IM captures instruction-level structure

## Configuration

### Dataloader Parameters
```python
InstructionMaskingDataset(
    token_mask_prob=0.15,        # MLM: 15% of tokens
    instruction_mask_prob=0.25,  # IM: 25% of instructions
    ...
)
```

### Model Requirements
- Must have `forward_im()` method for IM task
- Must have standard `forward()` method for MLM task
- Both use same BERT backbone, different prediction heads

## Troubleshooting

### MLM loss is 0
- Check `--enable_mlm` flag is set
- Verify `batch['mlm']` exists in dataloader output
- Ensure token_mask_prob > 0

### IM loss is 0
- Check `--enable_im` flag is set
- Verify model has `forward_im()` method
- Ensure instruction_mask_prob > 0

### Both losses are high
- Normal at start of training
- IM typically higher than MLM (predicting whole instructions vs individual tokens)
- Should decrease over epochs
