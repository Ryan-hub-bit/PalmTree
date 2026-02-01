# AddressAware Evaluation Pipeline

This guide explains how to generate evaluation data and evaluate the addressaware model.

## Prerequisites

- AddressAware model has been pretrained
- AddressAware model has been finetuned
- Small_test dataset exists at `/data/kun/jtrans/small_test` and `/data/kun/jtrans/small_test_strip`

## Step-by-Step Evaluation Process

### Step 1: Generate AddressAware Evaluation Data

Generate `func_blocks_addr.json` and `ground_truth_addr.json` from small_test binaries:

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_addraware

# This will:
# 1. Extract functions from stripped binaries using IDA Pro
# 2. Combine pickle files to create addr_pretrain.txt
# 3. Generate func_blocks_addr.json and ground_truth_addr.json
./generate_addressaware_eval_data.sh
```

**Output files:**
- `/data/kun/jtrans/addressaware/eval/func_blocks_addr.json`
- `/data/kun/jtrans/addressaware/eval/ground_truth_addr.json`
- `/data/kun/jtrans/addressaware/eval/extract/` (pickle files)

**Note:** This step can take several hours depending on the size of small_test dataset.

### Step 2: Create Filtered Evaluation Pools

Create filtered pools that ensure:
1. Query and Pool instructions differ (non-memorization test)
2. Each instruction in pool is unique (deduplicated)
3. Each query has only one unique GT in pool (unambiguous)

```bash
cd /home/kun/Document/AAE/extern/jTrans

# This creates pool files for O0_vs_O3, O1_vs_O3, O2_vs_O3
# with sizes 100, 1000, 10000
./run_create_addressaware_filtered_pools.sh
```

**Output files:**
- `/data/kun/jtrans/addressaware/eval/pools_filtered/pool_*.json`
- `/data/kun/jtrans/addressaware/eval/pools_filtered/query_*.json`

### Step 3: Evaluate AddressAware Model

Evaluate the finetuned addressaware model on the filtered pools:

```bash
cd /home/kun/Document/AAE/extern/jTrans

# Evaluate with your finetuned model
# Replace with actual checkpoint path and epoch number
./run_addressaware_pool_evaluation.sh \
    /home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_10 \
    /home/kun/Document/AAE/extern/jTrans/pretrain/addressaware
```

**Optional: Evaluate specific pool size or optimization pair:**

```bash
# Only evaluate 1000-size pools
./run_addressaware_pool_evaluation.sh \
    /home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_10 \
    /home/kun/Document/AAE/extern/jTrans/pretrain/addressaware \
    1000

# Only evaluate O0->O3 pairs
./run_addressaware_pool_evaluation.sh \
    /home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_10 \
    /home/kun/Document/AAE/extern/jTrans/pretrain/addressaware \
    10000 \
    O0_vs_O3
```

**Output:**
- Metrics printed to console (MRR, Recall@1, Recall@5, Recall@10)
- `/data/kun/jtrans/addressaware/eval/pools_filtered/evaluation_results.json`

## Comparing with Baseline

To compare addressaware with baseline, run the baseline evaluation:

```bash
cd /home/kun/Document/AAE/extern/jTrans

# Baseline evaluation
./run_baseline_pool_evaluation.sh \
    /home/kun/Document/AAE/output/jtrans/baseline_finetune/finetune_epoch_10 \
    /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
```

Then compare the results:
- Baseline results: `/data/kun/jtrans/baseline/eval/pools_filtered/evaluation_results.json`
- AddressAware results: `/data/kun/jtrans/addressaware/eval/pools_filtered/evaluation_results.json`

## Directory Structure

```
/data/kun/jtrans/
├── baseline/
│   └── eval/
│       ├── func_blocks_baseline.json
│       ├── ground_truth_baseline.json
│       └── pools_filtered/
│           ├── pool_*.json
│           ├── query_*.json
│           └── evaluation_results.json
└── addressaware/
    └── eval/
        ├── func_blocks_addr.json
        ├── ground_truth_addr.json
        ├── extract/
        │   └── *_functions.pkl
        └── pools_filtered/
            ├── pool_*.json
            ├── query_*.json
            └── evaluation_results.json
```

## Evaluation Metrics

- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for all queries
- **Recall@K**: Percentage of queries where GT appears in top-K results
  - Recall@1: Exact match (GT is rank 1)
  - Recall@5: GT appears in top 5
  - Recall@10: GT appears in top 10

## Notes

1. **Filtered pools guarantee:**
   - No memorization: Query and GT have different instruction sequences
   - Unambiguous: Each query has exactly one unique GT in the pool
   - Deduplicated: Each instruction sequence appears only once in pool

2. **Model checkpoint:**
   - Use the best finetuned checkpoint (check validation loss)
   - Typical checkpoints: `finetune_epoch_3`, `finetune_epoch_5`, `finetune_epoch_10`

3. **Troubleshooting:**
   - If vocab size mismatch warning appears, it's expected - token IDs will be clamped
   - If CUDA out of memory, reduce `--batch-size` in the evaluation script
   - If no pools found, ensure Step 2 completed successfully

## Quick Reference

```bash
# Full pipeline (all steps)
cd /home/kun/Document/AAE/extern/jTrans/datautils_addraware
./generate_addressaware_eval_data.sh

cd /home/kun/Document/AAE/extern/jTrans
./run_create_addressaware_filtered_pools.sh
./run_addressaware_pool_evaluation.sh \
    output/jtrans/addressaware_finetune/finetune_epoch_10 \
    pretrain/addressaware
```
