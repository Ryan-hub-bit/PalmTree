# Baseline jTrans Pipeline: Complete Guide

This document provides a comprehensive guide for the **baseline jTrans model** pipeline, from pretraining through finetuning to evaluation.

---

## Table of Contents
1. [Overview](#overview)
2. [Data Requirements](#data-requirements)
3. [Step 1: Pretraining](#step-1-pretraining)
4. [Step 2: Finetuning](#step-2-finetuning)
5. [Step 3: Evaluation](#step-3-evaluation)
6. [Complete Example](#complete-example)
7. [Troubleshooting](#troubleshooting)

---

## Overview

**Baseline Model Architecture:**
- Standard BERT with MLM (Masked Language Modeling) + JTP (Jump-Target Prediction)
- No address-aware features
- Position embeddings follow standard BERT approach
- Type vocab size: 2 (standard segment embeddings)

**Pipeline Flow:**
```
Raw Assembly → Pretraining Data → Pretrained Model → Finetuning → Finetuned Model → Evaluation
```

---

## Data Requirements

### Pretraining Data
- **Format:** `.pkl` file containing tokenized sequences
- **Location:** `/data/kun/jtransdata/baseline_pretrain.pkl`
- **Vocabulary:** `/data/kun/jtransdata/vocab_baseline.txt`
- **Content:** Assembly instructions without address annotations
- **Example format:**
  ```
  mov rdi rsp
  call sub_140001000
  test eax eax
  jz loc_140001234
  ```

### Finetuning Data
- **Function blocks:** `/data/kun/jtransdata/func_blocks_baseline.json`
  - JSON format with function representations
  - Keys: function IDs mapping to token sequences
- **Ground truth:** `/data/kun/jtransdata/ground_truth_baseline.json`
  - Pairs of similar functions for training
  - Format: `{"func1_id": "func2_id", ...}`

### Evaluation Data
- **Pool files:** `/data/kun/jtransdata/fair_pools/pool_<size>_<opt_pair>.json`
  - Pool sizes: 100, 1000, 10000
  - Optimization pairs: O0_vs_O3, O1_vs_O3, O2_vs_O3
- **Query files:** `/data/kun/jtransdata/fair_pools/query_pool_<size>_<opt_pair>.json`

---

## Step 1: Pretraining

### Script Location
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
```

### Basic Usage
```bash
# Use 100% of training data (default)
./run_baseline_pretrain.sh

# Use 10% of training data
./run_baseline_pretrain.sh 0.1

# Use 50% of training data
./run_baseline_pretrain.sh 0.5
```

### Manual Command
```bash
python3 train_baseline.py \
    --train_data /data/kun/jtransdata/baseline_pretrain.pkl \
    --vocab /data/kun/jtransdata/vocab_baseline.txt \
    --output_dir /home/kun/Document/AAE/output/jtrans/baseline_pretrain \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --mask_prob 0.15 \
    --save_best \
    --data_ratio 1.0
```

### Key Arguments

| Argument | Description | Default | Notes |
|----------|-------------|---------|-------|
| `--train_data` | Training data path | Required | `.pkl` file |
| `--vocab` | Vocabulary file | Required | `.txt` file |
| `--output_dir` | Output directory | Required | Saves checkpoints here |
| `--batch_size` | Batch size | 32 | Adjust based on GPU memory |
| `--epochs` | Number of epochs | 100 | Full pretraining |
| `--lr` | Learning rate | 1e-4 | Adam optimizer |
| `--mask_prob` | MLM mask probability | 0.15 | Standard BERT masking |
| `--save_best` | Save best model | False | Use flag to enable |
| `--data_ratio` | Data sampling ratio | 1.0 | 0.0-1.0, for quick tests |

### Checkpoint Resumption
The training script **automatically resumes** from the latest checkpoint if found:
- Checks `output_dir` for existing `checkpoint_epoch_*` directories
- Loads the latest checkpoint (model + optimizer + scheduler states)
- Continues training from the next epoch
- **No additional flags needed** - just re-run the same command

### Expected Output
```
/home/kun/Document/AAE/output/jtrans/baseline_pretrain/
├── checkpoint_epoch_10/
│   ├── pytorch_model.bin      # Model weights
│   ├── config.json             # Model configuration
│   ├── training_info.json      # Training metrics
│   ├── optimizer.pt            # Optimizer state (for resumption)
│   └── scheduler.pt            # Scheduler state (for resumption)
├── checkpoint_epoch_20/
├── ...
└── logs/
    └── training_YYYYMMDD_HHMMSS.log
```

### Training Time
- **Full dataset (100%):** ~24-48 hours per epoch on single GPU
- **10% dataset:** ~2-5 hours per epoch
- **Total:** 100 epochs for full pretraining

---

## Step 2: Finetuning

### Script Location
```bash
cd /home/kun/Document/AAE/extern/jTrans
```

### Basic Usage
```bash
./run_finetune_baseline.sh
```

### Manual Command
```bash
python finetune.py \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks_baseline.json \
    --ground_truth /data/kun/jtransdata/ground_truth_baseline.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/baseline \
    --model_path /home/kun/Document/AAE/output/jtrans/baseline_pretrain/checkpoint_epoch_10 \
    --output_path /home/kun/Document/AAE/output/jtrans/baseline_finetune \
    --batch_size 32 \
    --eval_batch_size 64 \
    --lr 1e-5 \
    --epoch 5 \
    --weight_decay 0.01 \
    --freeze_cnt 10 \
    --data_ratio 1.0
```

### Key Arguments

| Argument | Description | Default | Notes |
|----------|-------------|---------|-------|
| `--data_type` | Data format | Required | Use `json` |
| `--func_blocks` | Function representations | Required | JSON file |
| `--ground_truth` | Training pairs | Required | JSON file |
| `--tokenizer` | Tokenizer directory | Required | Contains vocab |
| `--model_path` | Pretrained checkpoint | Required | From Step 1 |
| `--output_path` | Output directory | Required | Saves finetuned model |
| `--batch_size` | Training batch size | 32 | Adjust for GPU |
| `--eval_batch_size` | Evaluation batch size | 64 | Can be larger |
| `--lr` | Learning rate | 1e-5 | Lower than pretraining |
| `--epoch` | Finetuning epochs | 5 | Usually 3-10 |
| `--weight_decay` | L2 regularization | 0.01 | Prevents overfitting |
| `--freeze_cnt` | Freeze first N layers | 10 | 0=no freezing |
| `--data_ratio` | Data sampling ratio | 1.0 | For quick tests |

### Expected Output
```
/home/kun/Document/AAE/output/jtrans/baseline_finetune/
├── finetune_epoch_1/
│   ├── pytorch_model.bin
│   └── config.json
├── finetune_epoch_2/
├── ...
├── finetune_epoch_5/
└── training_log.txt
```

### Training Time
- **Per epoch:** ~30 minutes to 2 hours (depends on dataset size)
- **Total:** ~2-10 hours for 5 epochs

---

## Step 3: Evaluation

### Evaluation with Pools

#### Script Location
```bash
cd /home/kun/Document/AAE/extern/jTrans
```

#### Basic Usage
```bash
./run_baseline_pool_evaluation.sh
```

This evaluates across:
- **Pool sizes:** 100, 1000, 10000 functions
- **Optimization pairs:** O0 vs O3, O1 vs O3, O2 vs O3
- **Metrics:** Recall@1, Recall@10, MRR (Mean Reciprocal Rank)

#### Manual Command (Single Pool)
```bash
python evaluate_baseline_with_pools.py \
    --model_path /home/kun/Document/AAE/output/jtrans/baseline_finetune/finetune_epoch_5 \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/baseline \
    --func_blocks /data/kun/jtransdata/func_blocks_baseline.json \
    --pool_file /data/kun/jtransdata/fair_pools/pool_1000_O0_vs_O3.json \
    --query_file /data/kun/jtransdata/fair_pools/query_pool_1000_O0_vs_O3.json \
    --output_file /data/kun/jtransdata/fair_pool_results/baseline_1000_O0_vs_O3.txt \
    --batch_size 64 \
    --use_cache
```

#### Key Arguments

| Argument | Description | Notes |
|----------|-------------|-------|
| `--model_path` | Finetuned model checkpoint | From Step 2 |
| `--tokenizer` | Tokenizer directory | Same as finetuning |
| `--func_blocks` | Function representations | Same as finetuning |
| `--pool_file` | Pool of candidate functions | JSON file |
| `--query_file` | Query functions to search | JSON file |
| `--output_file` | Results output path | `.txt` file |
| `--batch_size` | Batch size | Larger = faster |
| `--use_cache` | Cache embeddings | Speeds up evaluation |

### Expected Output
```
/data/kun/jtransdata/fair_pool_results/
├── baseline_100_O0_vs_O3.txt
├── baseline_100_O1_vs_O3.txt
├── baseline_100_O2_vs_O3.txt
├── baseline_1000_O0_vs_O3.txt
├── baseline_1000_O1_vs_O3.txt
├── baseline_1000_O2_vs_O3.txt
├── baseline_10000_O0_vs_O3.txt
├── baseline_10000_O1_vs_O3.txt
└── baseline_10000_O2_vs_O3.txt
```

#### Results Format
```
Pool Size: 1000
Optimization Pair: O0_vs_O3
Recall@1: 0.7234
Recall@10: 0.8912
MRR: 0.7891
```

### Evaluation Time
- **Per pool:** ~5-30 minutes (depends on pool size)
- **All 9 configurations:** ~1-4 hours

---

## Complete Example

### Full Pipeline Walkthrough

```bash
# ============================================
# STEP 1: Pretraining (10% data for quick test)
# ============================================
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
./run_baseline_pretrain.sh 0.1

# Wait for completion (~20-50 hours for 100 epochs)
# Or use checkpoint_epoch_10 for quick iteration

# ============================================
# STEP 2: Finetuning
# ============================================
cd /home/kun/Document/AAE/extern/jTrans

# Edit run_finetune_baseline.sh to point to your pretrained checkpoint
# Update --model_path to: .../baseline_pretrain/checkpoint_epoch_10

./run_finetune_baseline.sh

# Wait for completion (~2-10 hours)

# ============================================
# STEP 3: Evaluation
# ============================================
cd /home/kun/Document/AAE/extern/jTrans

# Edit run_baseline_pool_evaluation.sh to point to your finetuned model
# Update BASELINE_MODEL to: .../baseline_finetune/finetune_epoch_5

./run_baseline_pool_evaluation.sh

# Check results in /data/kun/jtransdata/fair_pool_results/
```

### Quick Test Pipeline (Small Scale)

```bash
# 1. Pretrain with 1% data for 10 epochs
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
python3 train_baseline.py \
    --train_data /data/kun/jtransdata/baseline_pretrain.pkl \
    --vocab /data/kun/jtransdata/vocab_baseline.txt \
    --output_dir /home/kun/Document/AAE/output/jtrans/baseline_pretrain_test \
    --batch_size 32 \
    --epochs 10 \
    --lr 1e-4 \
    --data_ratio 0.01

# 2. Finetune for 3 epochs
cd /home/kun/Document/AAE/extern/jTrans
python finetune.py \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks_baseline.json \
    --ground_truth /data/kun/jtransdata/ground_truth_baseline.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/baseline \
    --model_path /home/kun/Document/AAE/output/jtrans/baseline_pretrain_test/checkpoint_epoch_10 \
    --output_path /home/kun/Document/AAE/output/jtrans/baseline_finetune_test \
    --batch_size 32 \
    --lr 1e-5 \
    --epoch 3 \
    --data_ratio 0.1

# 3. Evaluate on smallest pool
python evaluate_baseline_with_pools.py \
    --model_path /home/kun/Document/AAE/output/jtrans/baseline_finetune_test/finetune_epoch_3 \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/baseline \
    --func_blocks /data/kun/jtransdata/func_blocks_baseline.json \
    --pool_file /data/kun/jtransdata/fair_pools/pool_100_O0_vs_O3.json \
    --query_file /data/kun/jtransdata/fair_pools/query_pool_100_O0_vs_O3.json \
    --output_file test_results.txt \
    --batch_size 64
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
**Symptom:** CUDA out of memory error during training

**Solutions:**
```bash
# Reduce batch size
--batch_size 16  # or even 8

# Use gradient accumulation (if supported)
--gradient_accumulation_steps 2

# Reduce sequence length
--max_len 256  # instead of 512
```

#### 2. Vocabulary Mismatch
**Symptom:** `Embedding size mismatch` error

**Check:**
```bash
# Verify vocab size matches
grep "vocab_size" /path/to/checkpoint/config.json
wc -l /data/kun/jtransdata/vocab_baseline.txt
```

**Solution:** Ensure you're using the same vocabulary for pretrain, finetune, and eval

#### 3. Checkpoint Not Found
**Symptom:** `FileNotFoundError` when loading checkpoint

**Check:**
```bash
ls /home/kun/Document/AAE/output/jtrans/baseline_pretrain/
```

**Solution:** Verify checkpoint path and epoch number exist

#### 4. Training Interrupted
**Symptom:** Want to resume training after interruption

**Solution:** Just re-run the same command - the script automatically resumes:
```bash
# No special flags needed - automatic resumption
./run_baseline_pretrain.sh
```

The script will:
- Find the latest checkpoint (e.g., `checkpoint_epoch_6`)
- Load model, optimizer, and scheduler states
- Continue from epoch 7

#### 5. Slow Evaluation
**Symptom:** Evaluation takes very long

**Solutions:**
```bash
# Use caching
--use_cache

# Increase batch size
--batch_size 128  # if GPU allows

# Evaluate on smaller pools first
# Start with pool_100 before pool_10000
```

#### 6. Poor Performance
**Symptom:** Low recall scores

**Check:**
- Pretraining: Did it run for enough epochs? (at least 10-20)
- Finetuning: Using correct learning rate? (1e-5 typical)
- Data: Are func_blocks and ground_truth aligned?
- Model: Using the correct checkpoint epoch?

**Debug:**
```bash
# Check training logs
tail -100 /home/kun/Document/AAE/output/jtrans/baseline_pretrain/logs/*.log

# Verify finetuning metrics
grep "Recall" /home/kun/Document/AAE/output/jtrans/baseline_finetune/training_log.txt
```

---

## Performance Expectations

### Typical Results (Full Training)

**Pool Size 1000, O0 vs O3:**
- Recall@1: 0.65-0.75
- Recall@10: 0.85-0.92
- MRR: 0.72-0.80

**Pool Size 10000:**
- Recall@1: 0.50-0.65
- Recall@10: 0.75-0.85
- MRR: 0.60-0.72

### Training Requirements

**Hardware:**
- GPU: NVIDIA GPU with 16GB+ VRAM (single GPU)
- RAM: 32GB+ system memory
- Storage: 100GB+ for data and checkpoints

**Time Investment:**
- Pretraining: 2-5 days (100 epochs, full data)
- Finetuning: 2-10 hours (5 epochs)
- Evaluation: 1-4 hours (all pools)
- **Total:** ~3-6 days for complete pipeline

---

## Additional Resources

### Related Files
- Pretraining script: `/home/kun/Document/AAE/extern/jTrans/pretrain/baseline/train_baseline.py`
- Finetuning script: `/home/kun/Document/AAE/extern/jTrans/finetune.py`
- Evaluation script: `/home/kun/Document/AAE/extern/jTrans/evaluate_baseline_with_pools.py`
- Data loader: `/home/kun/Document/AAE/extern/jTrans/data_json.py`

### Model Details
- Architecture: BERT-base (12 layers, 768 hidden, 12 heads)
- Vocabulary: ~20k-30k tokens (assembly mnemonics + operands)
- Max sequence length: 512 tokens
- Training objectives: MLM + JTP (Jump-Target Prediction)

### Contact & Support
For issues or questions:
1. Check logs in `output_dir/logs/`
2. Verify data paths and formats
3. Review this documentation
4. Check GPU availability and memory

---

**Last Updated:** January 2026
