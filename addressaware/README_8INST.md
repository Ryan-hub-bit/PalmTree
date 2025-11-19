# 8-Instruction Training Setup - Quick Start Guide

## Overview

Created complete training infrastructure for 8-instruction windows with 7:1 split strategy.

---

## Files Created

### Dataloaders
1. **`dataloader_paired_8inst.py`** - Address-aware dataloader
2. **`dataloader_baseline_8inst.py`** - Baseline dataloader (no address info)

### Training Scripts
3. **`train_8inst.py`** - Address-aware training script
4. **`train_baseline_8inst.py`** - Baseline training script

### Shell Wrappers
5. **`train_addressaware_8inst.sh`** - Address-aware launcher
6. **`train_baseline_8inst.sh`** - Baseline launcher

---

## NSP Strategy (7:1 Split)

### CFG: Control Flow Order Verification
- **Split**: First 7/8 (C1) + Last 1/8 (C2)
- **POSITIVE**: C1 → C2 (correct order)
- **NEGATIVE**: C2 → C1 (reversed order)
- **Goal**: Learn if execution order is correct

### DFG: Data Dependency Verification
- **Split**: First 7/8 (D1) + Last 1/8 (D2)
- **POSITIVE**: D1 → D2 (same line, correct dependency)
- **NEGATIVE**: D1 → D_random (different line, wrong dependency)
- **Goal**: Learn if data dependency is correct

---

## Quick Start

### 1. Create Combined Data Files

```bash
# CFG data
cd /home/kun/Document/PalmTree/data/cfg
cat *_cfg_8_inline.txt > all_cfg_8inst_combined.txt

# DFG data
cd /home/kun/Document/PalmTree/data/dfg
cat *_dfg_8_inline.txt > all_dfg_8inst_combined.txt
```

### 2. Train Address-Aware Model

```bash
cd /home/kun/Document/PalmTree/addressaware
./train_addressaware_8inst.sh
```

**Output**: `output_addressaware_8inst/`

### 3. Train Baseline Model

```bash
cd /home/kun/Document/PalmTree/addressaware
./train_baseline_8inst.sh
```

**Output**: `output_baseline_8inst/`

---

## Configuration

### Model Architecture
- **Hidden Size**: 128
- **Layers**: 12
- **Attention Heads**: 8
- **Max Sequence Length**: 80 (~8 instructions × 10 tokens)

### Training Hyperparameters
- **Epochs**: 20
- **Batch Size**: 128 (reduced from 512 due to 4× longer sequences)
- **Learning Rate**: 0.001
- **Dropout**: 0.1
- **Mask Probability**: 0.15
- **NSP Probability**: 0.5

### Data Settings
- **Data Percentage**: 100%
- **Train/Val Split**: 90% / 10%

---

## Key Differences: 2-Inst vs 8-Inst

| Aspect | 2-Instruction | 8-Instruction |
|--------|---------------|---------------|
| Window Size | 2 insts (~20 tokens) | 8 insts (~80 tokens) |
| Split Ratio | 1:1 (50/50) | 7:1 (87.5%/12.5%) |
| Max Length | 20 | 80 |
| Batch Size | 512 | 128 |
| Context | Minimal | Rich (7 insts) |

### Why 7:1 Split?
- **More context**: First part has 7 instructions (vs 1 in 1:1 split)
- **Better learning**: Model sees longer patterns before prediction
- **Long-range dependencies**: Learns relationships across 7 instructions
- **Realistic**: Mimics real-world scenario where you need to predict next instruction from context

---

## Expected Outputs

### Address-Aware Model
```
output_addressaware_8inst/
├── best_model.pt           # Best validation checkpoint
├── final_model.pt          # Final epoch checkpoint
├── training.log            # Training logs
├── args.json               # Training arguments
└── metrics/
    ├── train_metrics.json
    └── val_metrics.json
```

### Baseline Model
```
output_baseline_8inst/
├── best_model.pt
├── final_model.pt
├── training.log
├── args.json
└── metrics/
    ├── train_metrics.json
    └── val_metrics.json
```

---

## Data Format

Each line contains 8 instructions in inline address format:

```
inst1(0xADDR1:b1:f1:bb1) op1 op2  inst2(0xADDR2:b2:f2:bb2) op3 ... inst8(0xADDR8:b8:f8:bb8) ops
```

### Address-Aware Processing
- Extracts tokens: `['inst1', 'op1', 'op2', 'inst2', 'op3', ..., 'inst8', 'ops']`
- Extracts positions: `[(b1,f1,bb1), (0,0,0), (0,0,0), (b2,f2,bb2), ...]`

### Baseline Processing
- Extracts tokens: `['inst1', 'op1', 'op2', 'inst2', 'op3', ..., 'inst8', 'ops']`
- Uses sequential positions: `[0, 1, 2, 3, 4, ..., n]`

**Result**: Identical token sequences, only positional encoding differs!

---

## Fair Comparison Checklist

Both models must have:

✅ **IDENTICAL:**
1. Training data files
2. Vocabulary
3. Model architecture (hidden, layers, heads)
4. Sequence length (80)
5. Batch size (128)
6. Learning rate (0.001)
7. Number of epochs (20)
8. NSP strategy (7:1 split)
9. MLM masking (15%)
10. PalmTree initialization

❌ **DIFFERENT:**
1. **Position encoding**:
   - Address-Aware: 3-level learnable embeddings (binary, function, basic block)
   - Baseline: Sinusoidal sequential positions

This ensures any performance difference is attributable to address embeddings only!

---

## Troubleshooting

### Data Files Not Found
**Error**: "CFG data file not found: ../data/cfg/all_cfg_8inst_combined.txt"

**Solution**: Create combined files:
```bash
cd /home/kun/Document/PalmTree/data/cfg
cat *_cfg_8_inline.txt > all_cfg_8inst_combined.txt
```

### Out of Memory (OOM)
**Solution**: Reduce batch size in shell scripts:
```bash
BATCH_SIZE=64  # or 32
```

### Import Errors
**Error**: "ModuleNotFoundError: No module named 'dataloader_paired_8inst'"

**Solution**: Ensure you're running from the `addressaware/` directory:
```bash
cd /home/kun/Document/PalmTree/addressaware
./train_addressaware_8inst.sh
```

---

## Monitoring Training

### Check Progress
```bash
# Address-aware
tail -f output_addressaware_8inst/training.log

# Baseline
tail -f output_baseline_8inst/training.log
```

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Expected Training Time
- **Per epoch**: 30-60 minutes (depends on GPU and dataset size)
- **Total (20 epochs)**: 10-20 hours per model
- **Both models**: 20-40 hours total

---

## Next Steps

1. ✅ Create combined data files
2. ✅ Run address-aware training
3. ✅ Run baseline training
4. ⏳ Create evaluation/comparison script (if needed)
5. ⏳ Analyze results

---

## Summary

You now have a complete training infrastructure for 8-instruction windows with:

✨ **Address-Aware Model**: Uses 3-level address embeddings
✨ **Baseline Model**: Uses sequential positions
✨ **7:1 Split**: Rich context (7 insts) → prediction (1 inst)
✨ **Fair Comparison**: Identical setup except position encoding
✨ **CFG NSP**: Tests execution order (reversed negatives)
✨ **DFG NSP**: Tests data dependency (random negatives)

**Ready to train!** 🚀
