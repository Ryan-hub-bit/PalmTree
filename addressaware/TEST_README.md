# MLM and NSP Testing for Address-Aware BERT

## Overview

This test evaluates both **Address-Aware BERT** and **PalmTree Baseline** on identical data to measure the impact of address-aware embeddings.

## Fair Comparison Design

Both models receive **exactly the same token sequences** from the test data:

| Model | Input Components | What It Uses |
|-------|-----------------|--------------|
| **Address-Aware BERT** | Token + Segment + **Address Positions** (binary, function, bb) | Token embeddings + Segment embeddings + **Address-aware positional embeddings** + Transformer |
| **PalmTree Baseline** | Token + Segment + Sequential Positions | Token embeddings + Segment embeddings + Sequential positional embeddings + Transformer |

**Key Point**: The only difference is the **positional information**:
- Address-Aware uses: `(binary_pos, function_pos, bb_pos)` - captures hierarchical structure
- PalmTree uses: Sequential position `(0, 1, 2, 3, ...)` - captures only order

This ensures any performance difference is due to **address-aware embeddings**, not different inputs.

## Tasks Evaluated

### 1. MLM (Masked Language Model) - CFG Only
- **Task**: Predict masked tokens in control flow graph sequences
- **Metrics**: 
  - Loss (lower is better)
  - Perplexity (lower is better)
  - Top-1 Accuracy (higher is better)
  - Top-5 Accuracy (higher is better)

### 2. NSP_CFG (Next Sentence Prediction - CFG)
- **Task**: Determine if two basic blocks are consecutive in control flow
- **Metrics**:
  - Loss (lower is better)
  - Accuracy (higher is better)
  - Precision, Recall, F1 (higher is better)

### 3. NSP_DFG (Next Sentence Prediction - DFG)
- **Task**: Determine if two instructions belong to the same data flow trace
- **Metrics**:
  - Loss (lower is better)
  - Accuracy (higher is better)
  - Precision, Recall, F1 (higher is better)

## Running the Test

### Prerequisites

1. **Test data prepared** in `../data/test/`:
   - `all_cfg_combined.txt` - CFG sequences with inline addresses
   - `all_dfg_combined.txt` - DFG sequences with inline addresses

2. **Trained models**:
   - Address-Aware: `output_addressaware/transformer.ep19`
   - PalmTree baseline: `../pre-trained_model/palmtree/transformer.ep19`

3. **Vocabulary**: `../pre-trained_model/palmtree/vocab`

### Quick Start

```bash
cd addressaware
./test_mlm_nsp.sh
```

### Manual Execution

```bash
python3 test_mlm_nsp.py \
    --cfg_test ../data/test/all_cfg_combined.txt \
    --dfg_test ../data/test/all_dfg_combined.txt \
    --vocab ../pre-trained_model/palmtree/vocab \
    --addressaware_checkpoint output_addressaware/transformer.ep19 \
    --palmtree_checkpoint ../pre-trained_model/palmtree/transformer.ep19 \
    --hidden 128 \
    --n_layers 12 \
    --attn_heads 8 \
    --seq_len 20 \
    --batch_size 512 \
    --output_dir test_results_mlm_nsp \
    --cuda
```

### Testing Individual Models

Test only Address-Aware:
```bash
python3 test_mlm_nsp.py \
    --cfg_test ../data/test/all_cfg_combined.txt \
    --dfg_test ../data/test/all_dfg_combined.txt \
    --vocab ../pre-trained_model/palmtree/vocab \
    --addressaware_checkpoint output_addressaware/transformer.ep19 \
    --hidden 128 --n_layers 12 --attn_heads 8 --seq_len 20 \
    --batch_size 512 --cuda
```

Test only PalmTree:
```bash
python3 test_mlm_nsp.py \
    --cfg_test ../data/test/all_cfg_combined.txt \
    --dfg_test ../data/test/all_dfg_combined.txt \
    --vocab ../pre-trained_model/palmtree/vocab \
    --palmtree_checkpoint ../pre-trained_model/palmtree/transformer.ep19 \
    --batch_size 512 --cuda
```

## Output

### Console Output

You'll see:
1. Individual results for each model
2. Side-by-side comparison table with improvement percentages

Example:
```
FAIR COMPARISON: Address-Aware vs PalmTree Baseline
======================================================================
IMPORTANT: Both models evaluated on IDENTICAL token sequences
  - Address-Aware: Uses token + segment + ADDRESS positions
  - PalmTree:      Uses token + segment + SEQUENTIAL positions only
  - Difference shows the impact of address-aware embeddings
======================================================================

Metric               Address-Aware   PalmTree        Improvement     Winner
--------------------------------------------------------------------------------
Total Loss           2.3456          2.8901          -18.84%         Address-Aware ✓
MLM Loss             1.2345          1.5678          -21.25%         Address-Aware ✓
MLM Top-1 Acc        0.4521          0.3892          +16.16%         Address-Aware ✓
NSP_CFG Acc          0.8765          0.7234          +21.17%         Address-Aware ✓
NSP_DFG Acc          0.9012          0.8123          +10.94%         Address-Aware ✓
```

### JSON Output

Results saved to `test_results_mlm_nsp/comparison_results.json`:

```json
{
  "results": {
    "address_aware": {
      "total_loss": 2.3456,
      "mlm_loss": 1.2345,
      "mlm_perplexity": 3.4567,
      "mlm_top1_acc": 0.4521,
      "mlm_top5_acc": 0.7654,
      "nsp_cfg_loss": 0.5678,
      "nsp_cfg_acc": 0.8765,
      "nsp_cfg_precision": 0.8901,
      "nsp_cfg_recall": 0.8634,
      "nsp_cfg_f1": 0.8765,
      "nsp_dfg_loss": 0.2345,
      "nsp_dfg_acc": 0.9012,
      "nsp_dfg_precision": 0.9123,
      "nsp_dfg_recall": 0.8901,
      "nsp_dfg_f1": 0.9010
    },
    "palmtree_baseline": {
      ...
    }
  },
  "config": {
    "cfg_test": "../data/test/all_cfg_combined.txt",
    "dfg_test": "../data/test/all_dfg_combined.txt",
    ...
  }
}
```

## Interpreting Results

### What to Look For

1. **MLM Improvements**: If Address-Aware has:
   - Lower MLM loss/perplexity → Better at predicting masked instructions
   - Higher Top-1/Top-5 accuracy → More accurate predictions
   - **Why it matters**: Shows address context helps understand instruction semantics

2. **NSP_CFG Improvements**: If Address-Aware has:
   - Higher accuracy/F1 → Better at recognizing CFG structure
   - **Why it matters**: Shows address positions help understand control flow

3. **NSP_DFG Improvements**: If Address-Aware has:
   - Higher accuracy/F1 → Better at recognizing data flow traces
   - **Why it matters**: Shows address positions help track data dependencies

### Expected Results

If address-aware embeddings are beneficial, you should see:
- **Lower losses** across all tasks
- **Higher accuracies** for both NSP tasks
- **Better MLM predictions** (higher Top-1/Top-5)

The improvement percentage shows the relative gain from adding address information.

## Important Notes

### PalmTree Baseline Limitation

⚠️ **The PalmTree baseline in this test uses randomly initialized MLM/NSP heads**, not pre-trained heads. This is because:
- PalmTree was pre-trained with different task heads
- We need to ensure both models use the same head architecture

For a fully fair comparison, you could:
1. Pre-train PalmTree's heads on the same data (without addresses)
2. Then compare both models

However, since both models' BERT encoders are trained/pre-trained, the comparison still shows the impact of address-aware positional embeddings on the encoder representations.

### Test Data Requirements

The test data must use the **inline address format**:
```
opcode(0xADDR:bnorm:fnorm:bbnorm) operand1 operand2
```

If you don't have test data yet, prepare it using your data generation scripts.

## Troubleshooting

### "Test data not found"
Create test data or update paths in `test_mlm_nsp.sh`

### "Checkpoint not found"
Ensure models are trained, or update checkpoint paths

### Out of memory
Reduce `--batch_size` (try 256 or 128)

### Mismatch in model config
Ensure `--hidden`, `--n_layers`, `--attn_heads` match your trained model

## Next Steps

After testing:

1. **Analyze results**: Which tasks benefit most from address embeddings?
2. **Visualize**: Create plots comparing metrics across epochs
3. **Ablation study**: Test different address normalization strategies
4. **Downstream tasks**: Use these embeddings for binary analysis tasks
