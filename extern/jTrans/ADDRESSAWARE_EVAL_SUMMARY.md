# AddressAware Evaluation Setup - Summary

## What Has Been Created

I have set up the complete evaluation pipeline for the **addressaware** model, mirroring the baseline evaluation setup. Here's what was created:

### 1. Directory Structure
```
/data/kun/jtrans/addressaware/eval/
├── func_blocks_addr.json         (to be generated)
├── ground_truth_addr.json        (to be generated)
├── extract/                      (IDA Pro outputs)
└── pools_filtered/               (filtered evaluation pools)
    ├── pool_*.json
    ├── query_*.json
    └── evaluation_results.json
```

### 2. Data Generation Scripts

#### `/extern/jTrans/datautils_addraware/generate_addressaware_eval_data.sh`
Generates addressaware evaluation data from small_test binaries:
- Extracts functions using IDA Pro from stripped binaries
- Creates `func_blocks_addr.json` and `ground_truth_addr.json`
- Stores pickle files in `/data/kun/jtrans/addressaware/eval/extract/`

### 3. Pool Creation Scripts

#### `/extern/jTrans/create_filtered_addressaware_pools.py`
Python script that creates filtered evaluation pools with guarantees:
- Query and GT have different instruction sequences (non-memorization)
- Each instruction in pool is unique (deduplicated)
- Each query has exactly one unique GT in pool (unambiguous)

#### `/extern/jTrans/run_create_addressaware_filtered_pools.sh`
Shell wrapper to run the pool creation script

### 4. Evaluation Scripts

#### `/extern/jTrans/evaluate_addressaware_pools.py`
Python evaluation script that:
- Loads the finetuned addressaware model
- Generates embeddings for pool and queries
- Computes similarity and calculates metrics (MRR, Recall@K)
- Saves results to JSON

#### `/extern/jTrans/run_addressaware_pool_evaluation.sh`
Shell wrapper to run the evaluation with proper arguments

### 5. Pipeline Automation

#### `/extern/jTrans/run_addressaware_full_eval_pipeline.sh`
**Quick start script** that runs all three steps:
1. Generate evaluation data
2. Create filtered pools  
3. Run evaluation

### 6. Documentation

#### `/extern/jTrans/ADDRESSAWARE_EVALUATION_PIPELINE.md`
Complete documentation with:
- Step-by-step instructions
- Directory structure
- Metrics explanation
- Troubleshooting tips

## How to Use

### Option 1: Run Full Pipeline (Recommended)
```bash
cd /home/kun/Document/AAE/extern/jTrans
./run_addressaware_full_eval_pipeline.sh
```

This runs all steps automatically and handles existing data intelligently.

### Option 2: Run Steps Individually

```bash
# Step 1: Generate eval data
cd /home/kun/Document/AAE/extern/jTrans/datautils_addraware
./generate_addressaware_eval_data.sh

# Step 2: Create filtered pools
cd /home/kun/Document/AAE/extern/jTrans
./run_create_addressaware_filtered_pools.sh

# Step 3: Evaluate
./run_addressaware_pool_evaluation.sh \
    /home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_10 \
    /home/kun/Document/AAE/extern/jTrans/pretrain/addressaware
```

## Key Features

### 1. **Filtered Pools**
Unlike training data, evaluation pools guarantee:
- **No memorization**: Query ≠ GT instruction sequences
- **Unambiguous**: Each query has exactly 1 unique GT
- **Deduplicated**: Each instruction sequence appears once

### 2. **Multiple Pool Sizes**
Evaluates on pools of different sizes:
- 100 (easy)
- 1000 (medium)
- 10000 (hard)

### 3. **Multiple Optimization Pairs**
Tests across different optimization levels:
- O0 → O3 (hardest - maximum optimization difference)
- O1 → O3 (medium)
- O2 → O3 (easier)

### 4. **Comprehensive Metrics**
- **MRR**: Mean Reciprocal Rank (overall ranking quality)
- **Recall@1**: Exact matches (GT is rank 1)
- **Recall@5**: GT in top 5
- **Recall@10**: GT in top 10

## Comparison with Baseline

Both baseline and addressaware now have identical evaluation setups:

| Feature | Baseline | AddressAware |
|---------|----------|--------------|
| Eval data location | `/data/kun/jtrans/baseline/eval/` | `/data/kun/jtrans/addressaware/eval/` |
| Pool creation | `run_create_baseline_filtered_pools.sh` | `run_create_addressaware_filtered_pools.sh` |
| Evaluation script | `evaluate_baseline_pools.py` | `evaluate_addressaware_pools.py` |
| Results file | `baseline/eval/pools_filtered/evaluation_results.json` | `addressaware/eval/pools_filtered/evaluation_results.json` |

You can directly compare the `evaluation_results.json` files to see which model performs better!

## Next Steps

1. **Update model checkpoint path** in `run_addressaware_full_eval_pipeline.sh` if needed
   - Default: `/home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_10`
   - Change to your best checkpoint

2. **Run the evaluation**:
   ```bash
   cd /home/kun/Document/AAE/extern/jTrans
   ./run_addressaware_full_eval_pipeline.sh
   ```

3. **Compare results**:
   - Baseline: `/data/kun/jtrans/baseline/eval/pools_filtered/evaluation_results.json`
   - AddressAware: `/data/kun/jtrans/addressaware/eval/pools_filtered/evaluation_results.json`

4. **Analyze**:
   - Which model has higher MRR?
   - Which has better Recall@K?
   - How does performance vary with pool size?
   - Which optimization pair is hardest?

## Files Created

### Scripts (8 files)
1. `datautils_addraware/generate_addressaware_eval_data.sh`
2. `create_filtered_addressaware_pools.py`
3. `run_create_addressaware_filtered_pools.sh`
4. `evaluate_addressaware_pools.py`
5. `run_addressaware_pool_evaluation.sh`
6. `run_addressaware_full_eval_pipeline.sh`

### Documentation (2 files)
7. `ADDRESSAWARE_EVALUATION_PIPELINE.md`
8. `ADDRESSAWARE_EVAL_SUMMARY.md` (this file)

All scripts have been made executable and are ready to use!

## Troubleshooting

### Issue: Model checkpoint not found
**Solution**: Update the checkpoint path in the evaluation script or pipeline script

### Issue: Vocab size mismatch warning
**Solution**: This is expected - token IDs will be automatically clamped to valid range

### Issue: CUDA out of memory
**Solution**: Reduce `--batch-size` in the evaluation script (default: 64)

### Issue: No function files created in Step 1
**Solution**: Check IDA Pro logs in `./log/` directory, ensure stripped binaries exist

## Important Notes

- **Step 1 takes time**: IDA Pro extraction can take hours depending on dataset size
- **Step 2 is fast**: Pool creation takes seconds to minutes
- **Step 3 depends on hardware**: GPU evaluation is much faster than CPU

## Questions?

Refer to:
- Detailed guide: [ADDRESSAWARE_EVALUATION_PIPELINE.md](ADDRESSAWARE_EVALUATION_PIPELINE.md)
- Baseline comparison: [BASELINE_PIPELINE_CHECK.md](BASELINE_PIPELINE_CHECK.md)
