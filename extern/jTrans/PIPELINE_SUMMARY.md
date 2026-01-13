# Fine-tuning Pipeline Scripts - Summary

## Available Scripts

### Baseline Model
- **run_finetune_baseline.sh** - Basic fine-tuning (1 epoch, 0.1% data)
- **run_quick_pipeline.sh** - Quick test (2 epochs, 0.1% data + pool eval)
- **run_full_pipeline.sh** - Full training (10 epochs, 10% data + pool eval)

### Address-Aware Model
- **run_finetune_addressaware.sh** - Basic fine-tuning (1 epoch, 0.1% data)
- **run_quick_pipeline_addressaware.sh** - Quick test (2 epochs, 0.1% data + pool eval)
- **run_full_pipeline_addressaware.sh** - Full training (10 epochs, 10% data + pool eval)

## Quick Start

### Test Baseline Model
```bash
cd /home/kun/Document/AAE/extern/jTrans
CUDA_VISIBLE_DEVICES=0 ./run_quick_pipeline.sh
```

### Test Address-Aware Model
```bash
cd /home/kun/Document/AAE/extern/jTrans
CUDA_VISIBLE_DEVICES=0 ./run_quick_pipeline_addressaware.sh
```

## Configuration Details

### Baseline
- Pretrained: `/home/kun/Document/AAE/output/jtrans/baseline_pretrain/checkpoint_epoch_10`
- Tokenizer: `/home/kun/Document/AAE/extern/jTrans/pretrain/baseline`
- Data: `/data/kun/jtransdata/func_blocks_baseline.json`
- Ground Truth: `/data/kun/jtransdata/ground_truth_baseline.json`

### Address-Aware
- Pretrained: `/home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_10`
- Tokenizer: `/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware`
- Data: `/data/kun/jtransdata/func_blocks_addr.json`
- Ground Truth: `/data/kun/jtransdata/ground_truth_addr.json`

## Evaluation Details

### Pool-Based Evaluation
- Pool sizes: 100, 1000, 10000
- Query ratio: 10% of functions
- Ground truths per query: 3 (O1, O2, O3 for O0 query)
- Pool composition example (pool=100):
  - 3 ground truths (same function, different opts)
  - 97 negatives (different functions)

### Metrics Reported
- MRR (Mean Reciprocal Rank)
- Recall@1
- Recall@5
- Recall@10

## Pipeline Steps

Both pipelines follow the same pattern:

1. **Fine-tune model** on function similarity task
   - Uses triplet loss (anchor, positive, negative)
   - Positive = same function, different opt
   - Negative = different function

2. **Generate embeddings** (cached for reuse)
   - One-time embedding generation
   - Saved to `embeddings_cache/embeddings_all.npz`

3. **Evaluate with multiple pool sizes**
   - Reuses cached embeddings
   - Tests retrieval performance at different scales
   - Results saved to `pool_evaluation_results.txt`

## Output Locations

### Baseline
- Models: `/home/kun/Document/AAE/output/jtrans/baseline_finetune/finetune_epoch_*`
- Results: `/home/kun/Document/AAE/output/jtrans/baseline_finetune/pool_evaluation_results.txt`
- Embeddings: `/home/kun/Document/AAE/output/jtrans/baseline_finetune/embeddings_cache/`

### Address-Aware
- Models: `/home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_*`
- Results: `/home/kun/Document/AAE/output/jtrans/addressaware_finetune/pool_evaluation_results.txt`
- Embeddings: `/home/kun/Document/AAE/output/jtrans/addressaware_finetune/embeddings_cache/`
