# Model Comparison Testing Framework

This folder contains tools to test and compare the performance of **PalmTree (baseline)** and **AddressAware** models on MLM and NSP tasks.

## ⚠️ Important: Input Consistency

**Both models receive IDENTICAL token sequences** to ensure fair comparison. The only difference is that AddressAware receives additional hierarchical address positional information.

📖 See [INPUT_CONSISTENCY.md](INPUT_CONSISTENCY.md) for detailed explanation of how inputs are kept consistent.

Quick summary:
- ✓ Same token sequences (e.g., `mov`, `rax`, `addr_code`)
- ✓ Same vocabulary (6,631 tokens from PalmTree)
- ✓ Same masking and NSP pairing strategy
- ✓ Only difference: AddressAware gets extra address positions
- ✓ All tokens are in PalmTree's vocabulary (no unknowns)

## Overview

The testing framework evaluates both models on the same test data and provides:
- **MLM (Masked Language Model)** loss and accuracy for CFG sequences
- **NSP (Next Sentence Prediction)** loss and accuracy for both CFG and DFG sequences
- Side-by-side comparison showing performance differences
- JSON output for further analysis

## Files

- `test_models.py` - Main testing script that loads and evaluates both models
- `run_comparison.sh` - Shell script for convenient testing with default parameters
- `results/` - Output directory for test results (created automatically)

## Quick Start

### 1. Basic Test (Fresh AddressAware Model)

Test PalmTree against a freshly initialized AddressAware model (before training):

```bash
cd /home/kun/Document/PalmTree/addressaware/test_comparison
./run_comparison.sh
```

This shows the baseline comparison and how much the address-aware architecture changes the loss landscape.

### 2. Test with Trained AddressAware Model

After training your AddressAware model, test it against PalmTree:

```bash
# Edit run_comparison.sh and set:
ADDRESSAWARE_CHECKPOINT="../output_addressaware/checkpoint_epoch_5.pth"

# Then run:
./run_comparison.sh
```

### 3. Custom Test Parameters

Run with custom parameters:

```bash
python test_models.py \
    --cfg_data ../data/cfg/all_cfg_combined.txt \
    --dfg_data ../data/dfg/all_dfg_combined.txt \
    --vocab ../../pre-trained_model/palmtree/vocab \
    --palmtree_checkpoint ../../pre-trained_model/palmtree/transformer.ep19 \
    --addressaware_checkpoint ../output_addressaware/checkpoint_epoch_5.pth \
    --batch_size 512 \
    --seq_len 20 \
    --test_samples 10000 \
    --output results/my_test.json
```

## Parameters

### Required:
- `--cfg_data`: Path to CFG corpus (inline format with addresses)
- `--dfg_data`: Path to DFG corpus (inline format with addresses)
- `--vocab`: Path to vocabulary file
- `--palmtree_checkpoint`: Path to PalmTree pre-trained model

### Optional:
- `--addressaware_checkpoint`: Path to trained AddressAware checkpoint (omit for fresh model)
- `--batch_size`: Batch size (default: 512)
- `--seq_len`: Sequence length (default: 20)
- `--num_workers`: DataLoader workers (default: 4)
- `--test_samples`: Number of samples to test (default: 10000)
- `--output`: Output JSON file path (default: comparison_results.json)

## Output

The script produces:

### 1. Console Output

```
========================================================================
PalmTree (CFG) Test Results
========================================================================

📊 CFG Performance:
  Total Loss:  5.2340
  MLM Loss:    4.1234
  MLM Acc:     28.45%
  NSP Loss:    1.1106
  NSP Acc:     52.30%

========================================================================
📈 Comparison: PalmTree vs AddressAware
========================================================================

🔍 CFG (MLM + NSP):
Metric               PalmTree        AddressAware    Difference     
----------------------------------------------------------------------
MLM Loss             4.1234          3.8456          -0.2778
MLM Accuracy         28.45           31.23           +2.78%
NSP Loss             1.1106          0.9845          -0.1261
NSP Accuracy         52.30           55.67           +3.37%

🔍 DFG (NSP only):
Metric               PalmTree        AddressAware    Difference     
----------------------------------------------------------------------
NSP Loss             0.7234          0.6891          -0.0343
NSP Accuracy         68.90           71.23           +2.33%
```

### 2. JSON Output

Detailed results saved to `results/comparison_YYYYMMDD_HHMMSS.json`:

```json
{
  "palmtree": {
    "cfg": {
      "total_loss": 5.2340,
      "mlm_loss": 4.1234,
      "mlm_acc": 0.2845,
      "nsp_loss": 1.1106,
      "nsp_acc": 0.5230
    },
    "dfg": {
      "nsp_loss": 0.7234,
      "nsp_acc": 0.6890
    }
  },
  "addressaware": {
    "cfg": {
      "total_loss": 4.8301,
      "mlm_loss": 3.8456,
      "mlm_acc": 0.3123,
      "nsp_loss": 0.9845,
      "nsp_acc": 0.5567
    },
    "dfg": {
      "nsp_loss": 0.6891,
      "nsp_acc": 0.7123
    }
  },
  "test_config": {
    "test_samples": 10000,
    "batch_size": 512,
    "seq_len": 20
  }
}
```

## What to Look For

### 1. Fresh AddressAware Model (Before Training)

When testing a freshly initialized AddressAware model:
- **MLM Loss**: Should be similar or slightly higher (random address embeddings)
- **NSP Loss**: Should be similar (uses PalmTree's frozen transformer)
- **Accuracy**: Should be similar (random guess ~50% for NSP, low for MLM)

This establishes the baseline and shows the impact of address-aware architecture.

### 2. Trained AddressAware Model (After Training)

After training with address-aware embeddings:
- **MLM Loss**: Should decrease (model learns address-token relationships)
- **NSP Loss**: Should decrease (address context helps sequence ordering)
- **Accuracy**: Should increase for both MLM and NSP
- **CFG vs DFG**: Different improvements expected (CFG has more training data)

### 3. Expected Improvements

Address-aware embeddings should particularly help with:
- **Instruction ordering** (NSP for CFG) - addresses provide strong ordering cues
- **Data flow coherence** (NSP for DFG) - addresses link memory operations
- **Operand prediction** (MLM) - address context disambiguates operands

## Testing Strategy

### Stage 1: Baseline Comparison
Test fresh AddressAware model to understand architectural impact:
```bash
ADDRESSAWARE_CHECKPOINT=""  # Empty - fresh model
./run_comparison.sh
```

### Stage 2: During Training
Test at each epoch to track learning progress:
```bash
for epoch in 1 2 3 4 5; do
    ADDRESSAWARE_CHECKPOINT="../output_addressaware/checkpoint_epoch_${epoch}.pth"
    python test_models.py ... --output results/epoch_${epoch}.json
done
```

### Stage 3: Final Evaluation
Test final trained model with larger sample size:
```bash
python test_models.py \
    --addressaware_checkpoint ../output_addressaware/checkpoint_epoch_5.pth \
    --test_samples 50000 \
    --output results/final_comparison.json
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
--batch_size 256  # or 128
```

### Test Takes Too Long
Reduce test samples:
```bash
--test_samples 5000  # or 1000 for quick tests
```

### Model Loading Errors
Check that checkpoint paths are correct:
```bash
ls -lh ../../pre-trained_model/palmtree/transformer.ep19
ls -lh ../output_addressaware/checkpoint_epoch_5.pth
```

## Notes

- The test uses the **same paired dataloader** as training, ensuring fair comparison
- PalmTree model doesn't use address positions (they're ignored in forward pass)
- AddressAware model uses all four embedding components (token + position + address + segment)
- Test data is masked/paired the same way as training data
- Results are deterministic (no shuffling in test loader)

## Future Enhancements

Potential additions to the testing framework:
- Per-token-type accuracy breakdown (opcodes vs operands)
- Address-bearing vs non-address-bearing token comparison
- Visualization of loss curves over training
- Statistical significance testing
- Per-basic-block or per-function analysis
