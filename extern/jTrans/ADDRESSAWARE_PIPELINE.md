# Address-Aware jTrans Pipeline: Complete Guide

This document provides a comprehensive guide for the **address-aware jTrans model** pipeline, from pretraining through finetuning to evaluation.

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

**Address-Aware Model Architecture:**
- Enhanced BERT with hierarchical address embeddings
- Position embeddings at 4 levels: function, basic block, instruction, token
- Address-aware attention mechanism
- Instruction-based segment IDs (1, 2, 3, ... for each instruction)
- MLM (Masked Language Modeling) + JTP (Jump-Target Prediction)

**Key Difference from Baseline:**
- Uses actual memory addresses to create hierarchical position embeddings
- Distinguishes between `address` and `daddr` (data addresses) tokens
- Better captures structural relationships in binary code

**Pipeline Flow:**
```
Raw Assembly + Addresses → Pretraining Data → Pretrained Model → Finetuning → Finetuned Model → Evaluation
```

---

## Data Requirements

### Pretraining Data
- **Format:** `.txt` file with tab-separated instructions
- **Location:** `/data/kun/jtransdata/addr_pretrain.txt`
- **Vocabulary:** `./vocab_addr.pkl` (in pretrain/address_aware/)
- **Content:** Assembly with hierarchical address annotations
- **Data format:**
  ```
  inst1\tinst2\tinst3\tinst4
  ```
  - Instructions separated by `\t` (tabs)
  - Tokens within instructions separated by spaces
  - Example instruction: `mov(0x1000:0:0:0) rdi(0x1000:0:0:1) rsp(0x1000:0:0:2)`

### Finetuning Data
- **Function blocks:** `/data/kun/jtransdata/func_blocks_addr.json`
  - JSON format with address-aware function representations
  - Keys: function IDs mapping to token sequences with hierarchical addresses
- **Ground truth:** `/data/kun/jtransdata/ground_truth_addr.json`
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
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
```

### Basic Usage
```bash
# Use 10% of training data (default in script)
./run_addressaware_pretrain.sh

# Use 100% of training data
./run_addressaware_pretrain.sh 1.0

# Use 1% for quick testing
./run_addressaware_pretrain.sh 0.01
```

### Manual Command
```bash
python3 train_addressaware.py \
    --train_path /data/kun/jtransdata/addr_pretrain.txt \
    --vocab_path ./vocab_addr.pkl \
    --output_dir /home/kun/Document/AAE/output/jtrans/addressaware_pretrain \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --warmup_steps 10000 \
    --save_every 2 \
    --token_mask_prob 0.15 \
    --data_ratio 1.0 \
    --num_workers 4 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --max_len 512
```

### Key Arguments

| Argument | Description | Default | Notes |
|----------|-------------|---------|-------|
| `--train_path` | Training data path | Required | `.txt` file with tabs |
| `--vocab_path` | Vocabulary file | Required | `.pkl` file |
| `--output_dir` | Output directory | Required | Saves checkpoints here |
| `--batch_size` | Batch size | 128 | Can be larger than baseline |
| `--learning_rate` | Learning rate | 1e-4 | Adam optimizer |
| `--num_epochs` | Number of epochs | 10 | Can train longer (50-100) |
| `--warmup_steps` | Warmup steps | 10000 | Linear warmup |
| `--save_every` | Save every N epochs | 2 | Checkpoint frequency |
| `--token_mask_prob` | MLM mask probability | 0.15 | Standard BERT |
| `--data_ratio` | Data sampling ratio | 1.0 | 0.0-1.0 for quick tests |
| `--num_workers` | DataLoader workers | 4 | Parallel data loading |
| `--hidden_size` | Hidden dimension | 768 | BERT-base size |
| `--num_hidden_layers` | Transformer layers | 12 | BERT-base depth |
| `--num_attention_heads` | Attention heads | 12 | Must divide hidden_size |
| `--max_len` | Max sequence length | 512 | Truncate longer sequences |

### Important: Segment ID Logic
The address-aware model uses **instruction-based segment IDs**:
- Each instruction gets a unique segment ID (1, 2, 3, ...)
- All tokens within the same instruction share the same segment ID
- `[SOS]` token gets segment ID 1
- `[EOS]` token gets the last instruction's segment ID
- `[PAD]` tokens get segment ID 0
- **One `[EOS]` at the very end only** (not after each instruction)

This is consistent across pretraining, finetuning, and evaluation.

### Multi-GPU Training
The script automatically uses **all available GPUs** with DataParallel:
```bash
# Uses GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1
./run_addressaware_pretrain.sh
```

### Checkpoint Resumption
The training script **automatically resumes** from the latest checkpoint if found:
- Checks `output_dir` for existing `checkpoint_epoch_*` directories
- Loads the latest checkpoint (model + optimizer + scheduler states)
- Continues training from the next epoch
- **No additional flags needed** - just re-run the same command

Example:
```bash
# First run: trains epochs 1-10
./run_addressaware_pretrain.sh

# Interrupted at epoch 6, re-run same command
./run_addressaware_pretrain.sh
# Automatically loads checkpoint_epoch_6 and continues from epoch 7
```

### Expected Output
```
/home/kun/Document/AAE/output/jtrans/addressaware_pretrain/
├── checkpoint_epoch_2/
│   ├── pytorch_model.bin      # Model weights (BERT part)
│   ├── config.json             # Model configuration
│   ├── training_info.json      # Training metrics (loss, acc)
│   ├── optimizer.pt            # Optimizer state (for resumption)
│   └── scheduler.pt            # Scheduler state (for resumption)
├── checkpoint_epoch_4/
├── checkpoint_epoch_6/
├── checkpoint_epoch_8/
├── checkpoint_epoch_10/
└── logs/
    └── training_YYYYMMDD_HHMMSS.log
```

### Training Time
- **Full dataset (100%):** ~12-24 hours per epoch on 2x GPUs
- **10% dataset:** ~1-2 hours per epoch
- **1% dataset:** ~10-15 minutes per epoch
- **Total (10 epochs, 100% data):** ~5-10 days

---

## Step 2: Finetuning

### Script Location
```bash
cd /home/kun/Document/AAE/extern/jTrans
```

### Basic Usage
```bash
./run_finetune_addressaware.sh
```

### Manual Command
```bash
python finetune.py \
    --model_type addressaware \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks_addr.json \
    --ground_truth /data/kun/jtransdata/ground_truth_addr.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
    --model_path /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_10 \
    --output_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune \
    --batch_size 32 \
    --eval_batch_size 64 \
    --lr 1e-5 \
    --epoch 10 \
    --weight_decay 0.01 \
    --freeze_cnt 10 \
    --data_ratio 0.001
```

### Key Arguments

| Argument | Description | Default | Notes |
|----------|-------------|---------|-------|
| `--model_type` | Model architecture | Required | Must be `addressaware` |
| `--data_type` | Data format | Required | Use `json` |
| `--func_blocks` | Function representations | Required | JSON with addresses |
| `--ground_truth` | Training pairs | Required | JSON file |
| `--tokenizer` | Tokenizer directory | Required | Contains vocab |
| `--model_path` | Pretrained checkpoint | Required | From Step 1 |
| `--output_path` | Output directory | Required | Saves finetuned model |
| `--batch_size` | Training batch size | 32 | Adjust for GPU |
| `--eval_batch_size` | Evaluation batch size | 64 | Can be larger |
| `--lr` | Learning rate | 1e-5 | Lower than pretraining |
| `--epoch` | Finetuning epochs | 10 | Usually 5-10 |
| `--weight_decay` | L2 regularization | 0.01 | Prevents overfitting |
| `--freeze_cnt` | Freeze first N layers | 10 | 0=no freezing |
| `--data_ratio` | Data sampling ratio | 0.001 | For quick tests |

### Important Notes
1. **Must specify `--model_type addressaware`** to use address-aware architecture
2. Uses address-aware data loader that handles hierarchical positions
3. Segment IDs follow instruction-based numbering (same as pretraining)
4. Data ratio of 0.001 in script is for testing - use 1.0 for full training

### Expected Output
```
/home/kun/Document/AAE/output/jtrans/addressaware_finetune/
├── finetune_epoch_1/
│   ├── pytorch_model.bin
│   └── config.json
├── finetune_epoch_2/
├── ...
├── finetune_epoch_10/
└── training_log.txt
```

### Training Time
- **Per epoch:** ~30 minutes to 3 hours (depends on dataset size and data_ratio)
- **Total (10 epochs, 0.001 ratio):** ~1-2 hours
- **Total (10 epochs, 1.0 ratio):** ~5-30 hours

---

## Step 3: Evaluation

### Evaluation with Pools

#### Script Location
```bash
cd /home/kun/Document/AAE/extern/jTrans
```

#### Basic Usage
```bash
./run_addressaware_pool_evaluation.sh
```

This evaluates across:
- **Pool sizes:** 100, 1000, 10000 functions
- **Optimization pairs:** O0 vs O3, O1 vs O3, O2 vs O3
- **Metrics:** Recall@1, Recall@10, MRR (Mean Reciprocal Rank)

#### Manual Command (Single Pool)
```bash
python evaluate_addressaware_with_pools.py \
    --model_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_4 \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
    --func_blocks /data/kun/jtransdata/func_blocks_addr.json \
    --pool_file /data/kun/jtransdata/fair_pools/pool_1000_O0_vs_O3.json \
    --query_file /data/kun/jtransdata/fair_pools/query_pool_1000_O0_vs_O3.json \
    --output_file /data/kun/jtransdata/fair_pool_addr_results/addressaware_1000_O0_vs_O3.txt \
    --batch_size 64 \
    --use_cache
```

#### Key Arguments

| Argument | Description | Notes |
|----------|-------------|-------|
| `--model_path` | Finetuned model checkpoint | From Step 2 |
| `--tokenizer` | Tokenizer directory | Same as finetuning |
| `--func_blocks` | Function representations | Must be address-aware version |
| `--pool_file` | Pool of candidate functions | JSON file |
| `--query_file` | Query functions to search | JSON file |
| `--output_file` | Results output path | `.txt` file |
| `--batch_size` | Batch size | Larger = faster |
| `--use_cache` | Cache embeddings | Speeds up evaluation |

### Expected Output
```
/data/kun/jtransdata/fair_pool_addr_results/
├── addressaware_100_O0_vs_O3.txt
├── addressaware_100_O1_vs_O3.txt
├── addressaware_100_O2_vs_O3.txt
├── addressaware_1000_O0_vs_O3.txt
├── addressaware_1000_O1_vs_O3.txt
├── addressaware_1000_O2_vs_O3.txt
├── addressaware_10000_O0_vs_O3.txt
├── addressaware_10000_O1_vs_O3.txt
└── addressaware_10000_O2_vs_O3.txt
```

#### Results Format
```
Pool Size: 1000
Optimization Pair: O0_vs_O3
Recall@1: 0.7823
Recall@10: 0.9145
MRR: 0.8234
```

### Evaluation Time
- **Per pool:** ~5-30 minutes (depends on pool size)
- **All 9 configurations:** ~1-4 hours

---

## Complete Example

### Full Pipeline Walkthrough

```bash
# ============================================
# STEP 0: Environment Setup
# ============================================
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jtrans

# ============================================
# STEP 1: Pretraining (10% data)
# ============================================
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware

# Quick test with 1% data
./run_addressaware_pretrain.sh 0.01

# Or full pretraining with 100% data
./run_addressaware_pretrain.sh 1.0

# Wait for completion or use existing checkpoint
# Checkpoint will be at: 
# /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_10

# ============================================
# STEP 2: Finetuning
# ============================================
cd /home/kun/Document/AAE/extern/jTrans

# Edit run_finetune_addressaware.sh to point to your checkpoint
# Update --model_path to: .../addressaware_pretrain/checkpoint_epoch_10
# Update --data_ratio to 1.0 for full finetuning (currently 0.001)

./run_finetune_addressaware.sh

# Wait for completion
# Finetuned model will be at:
# /home/kun/Document/AAE/output/jtrans/addressaware_finetune/finetune_epoch_4

# ============================================
# STEP 3: Evaluation
# ============================================
cd /home/kun/Document/AAE/extern/jTrans

# Edit run_addressaware_pool_evaluation.sh to point to finetuned model
# Update ADDRESSAWARE_MODEL to: .../addressaware_finetune/finetune_epoch_4

./run_addressaware_pool_evaluation.sh

# Check results in /data/kun/jtransdata/fair_pool_addr_results/
ls -lh /data/kun/jtransdata/fair_pool_addr_results/

# View a result
cat /data/kun/jtransdata/fair_pool_addr_results/addressaware_1000_O0_vs_O3.txt
```

### Quick Test Pipeline (Ultra-Small Scale)

```bash
# ============================================
# Ultra-fast test (~30 minutes total)
# ============================================

# 1. Pretrain with 0.1% data for 2 epochs (~5 minutes)
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
python3 train_addressaware.py \
    --train_path /data/kun/jtransdata/addr_pretrain.txt \
    --vocab_path ./vocab_addr.pkl \
    --output_dir /home/kun/Document/AAE/output/jtrans/addressaware_pretrain_test \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --num_epochs 2 \
    --save_every 2 \
    --data_ratio 0.001

# 2. Finetune for 2 epochs (~10 minutes)
cd /home/kun/Document/AAE/extern/jTrans
python finetune.py \
    --model_type addressaware \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks_addr.json \
    --ground_truth /data/kun/jtransdata/ground_truth_addr.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
    --model_path /home/kun/Document/AAE/output/jtrans/addressaware_pretrain_test/checkpoint_epoch_2 \
    --output_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune_test \
    --batch_size 32 \
    --lr 1e-5 \
    --epoch 2 \
    --data_ratio 0.001

# 3. Evaluate on smallest pool (~5 minutes)
python evaluate_addressaware_with_pools.py \
    --model_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune_test/finetune_epoch_2 \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
    --func_blocks /data/kun/jtransdata/func_blocks_addr.json \
    --pool_file /data/kun/jtransdata/fair_pools/pool_100_O0_vs_O3.json \
    --query_file /data/kun/jtransdata/fair_pools/query_pool_100_O0_vs_O3.json \
    --output_file test_addr_results.txt \
    --batch_size 64 \
    --use_cache

# Check results
cat test_addr_results.txt
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
**Symptom:** CUDA out of memory error

**Solutions:**
```bash
# Reduce batch size
--batch_size 64  # or 32, 16

# Use single GPU instead of multi-GPU
export CUDA_VISIBLE_DEVICES=0

# Reduce sequence length
--max_len 256  # instead of 512
```

#### 2. Segment ID / Type Embeddings Error
**Symptom:** Error related to token_type_ids or segment embeddings

**Check:** Ensure all three components use instruction-based segment IDs:
- Pretraining: `dataloader_addressaware.py` line 446
- Finetuning: `data_json.py` line 433
- Evaluation: Uses `token_type_ids` from dataset

**Verify:**
```bash
grep "inst_segment = inst_idx + 1" pretrain/address_aware/dataloader_addressaware.py
grep "inst_segment = inst_idx + 1" data_json.py
```

#### 3. Vocabulary Mismatch
**Symptom:** `Embedding size mismatch` error

**Debug:**
```python
# Check vocabulary size
import pickle
vocab = pickle.load(open('./vocab_addr.pkl', 'rb'))
print(f"Vocab size: {len(vocab)}")
print(f"Max token ID: {max(vocab.stoi.values())}")
```

**Solution:** Ensure same vocabulary used across all stages

#### 4. Data Format Error
**Symptom:** Error parsing address annotations

**Check data format:**
```bash
# Should have tabs between instructions
head -1 /data/kun/jtransdata/addr_pretrain.txt | cat -A
# Look for ^I (tab characters) between instructions
```

**Data format rules:**
- Instructions separated by `\t` (tabs)
- Tokens within instructions separated by spaces
- Final sequence has only spaces (tabs removed during processing)

#### 5. Training Not Resuming
**Symptom:** Training starts from epoch 1 even though checkpoints exist

**Check:**
```bash
# Verify checkpoint directory structure
ls -la /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/

# Should see checkpoint_epoch_* directories
# Each should contain: pytorch_model.bin, optimizer.pt, scheduler.pt
```

**Solution:** Ensure output_dir matches between runs

#### 6. Address vs Daddr Distinction
**Symptom:** Model not distinguishing address types

**Verify vocabulary has both:**
```python
import pickle
vocab = pickle.load(open('./vocab_addr.pkl', 'rb'))
print('address' in vocab.stoi)  # Should be True
print('daddr' in vocab.stoi)    # Should be True
```

The model uses `vocab.stoi` to distinguish these during embedding lookup.

#### 7. Multi-GPU Issues
**Symptom:** Error with DataParallel or GPU synchronization

**Solutions:**
```bash
# Use single GPU
export CUDA_VISIBLE_DEVICES=0

# For debugging, make CUDA synchronous (slower but clearer errors)
export CUDA_LAUNCH_BLOCKING=1

# Reduce batch size for multi-GPU
# Each GPU processes batch_size/num_gpus samples
```

#### 8. Slow Data Loading
**Symptom:** GPU utilization low, data loading bottleneck

**Solutions:**
```bash
# Increase worker threads
--num_workers 8  # or more

# Pre-load data to faster storage (SSD)
# Check disk I/O with: iostat -x 1
```

---

## Performance Expectations

### Typical Results (Full Training)

**Address-aware typically outperforms baseline by 3-8%**

**Pool Size 1000, O0 vs O3:**
- Recall@1: 0.72-0.82
- Recall@10: 0.88-0.95
- MRR: 0.78-0.87

**Pool Size 10000:**
- Recall@1: 0.58-0.72
- Recall@10: 0.78-0.88
- MRR: 0.66-0.78

### Comparison: Address-Aware vs Baseline

| Metric | Baseline | Address-Aware | Improvement |
|--------|----------|---------------|-------------|
| Recall@1 (1K pool) | 0.70 | 0.77 | +10% |
| Recall@10 (1K pool) | 0.90 | 0.93 | +3.3% |
| MRR (1K pool) | 0.77 | 0.83 | +7.8% |

### Training Requirements

**Hardware:**
- GPU: 2x NVIDIA GPUs with 16GB+ VRAM each (can use 1 GPU with smaller batch)
- RAM: 64GB+ system memory (address data is larger)
- Storage: 200GB+ for data and checkpoints

**Time Investment:**
- Pretraining: 5-10 days (10 epochs, full data, 2 GPUs)
- Finetuning: 5-30 hours (10 epochs)
- Evaluation: 1-4 hours (all pools)
- **Total:** ~6-12 days for complete pipeline

**Quick Test (0.1% data):**
- Pretraining: 1-2 hours (2 epochs)
- Finetuning: 30 minutes (2 epochs)
- Evaluation: 15 minutes (single pool)
- **Total:** ~2-3 hours

---

## Key Differences from Baseline

### Architecture
| Feature | Baseline | Address-Aware |
|---------|----------|---------------|
| Position Embeddings | Standard BERT (0-511) | Hierarchical (func:bb:inst:tok) |
| Address Handling | None | Distinguishes address/daddr |
| Segment IDs | Binary (0/1) | Instruction-based (1,2,3,...) |
| Special Tokens | Standard | Address-aware vocab |

### Data Format
| Aspect | Baseline | Address-Aware |
|--------|----------|---------------|
| File Format | `.pkl` | `.txt` with tabs |
| Annotations | None | Hierarchical addresses |
| Separator | N/A | `\t` between instructions |
| Token Format | Plain | `token(addr:f:b:i:t)` |

### Training
| Parameter | Baseline | Address-Aware |
|-----------|----------|---------------|
| Default Batch Size | 32 | 128 |
| Typical Epochs | 100 | 10-50 |
| GPU Count | 1 | 1-2 |
| Data Ratio (testing) | 0.01-0.1 | 0.001-0.01 |

---

## Additional Resources

### Related Files
- **Pretraining:**
  - Script: `/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/train_addressaware.py`
  - Data loader: `/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/dataloader_addressaware.py`
  - Model: `/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware/model_addressaware.py`

- **Finetuning:**
  - Script: `/home/kun/Document/AAE/extern/jTrans/finetune.py`
  - Data loader: `/home/kun/Document/AAE/extern/jTrans/data_json.py` (address-aware section)

- **Evaluation:**
  - Script: `/home/kun/Document/AAE/extern/jTrans/evaluate_addressaware_with_pools.py`
  - Pool eval: `/home/kun/Document/AAE/extern/jTrans/finetune_eval_with_pool.py`

### Model Architecture Details
- **Base Model:** BERT-base (12 layers, 768 hidden, 12 heads)
- **Position Embedding Levels:**
  1. Function-level: 0-127 (128 functions max)
  2. Basic block-level: 0-127 (128 BBs per function)
  3. Instruction-level: 0-511 (512 instructions per BB)
  4. Token-level: 0-15 (16 tokens per instruction)
- **Total Position Space:** 128 × 128 × 512 × 16 = ~134M unique positions
- **Vocabulary:** ~25k-35k tokens (including address variants)
- **Max Sequence Length:** 512 tokens
- **Training Objectives:** MLM (80% mask, 10% random, 10% keep) + JTP

### Segment ID Consistency
**Critical:** All three stages must use the same segment ID logic:

```python
# Instruction-based segment IDs (consistent across pretrain/finetune/eval)
inst_segment = inst_idx + 1  # 1, 2, 3, ...

# SOS gets segment 1
# Each instruction's tokens get that instruction's segment
# EOS gets last instruction's segment
# PAD gets segment 0
```

### Contact & Support
For issues or questions:
1. Check logs: `output_dir/logs/training_*.log`
2. Verify data format (tabs between instructions)
3. Ensure vocabulary consistency
4. Check GPU availability: `nvidia-smi`
5. Review this documentation
6. Compare with baseline pipeline if issues persist

---

## Checklist: Before You Start

- [ ] Environment: `conda activate jtrans`
- [ ] GPUs available: `nvidia-smi`
- [ ] Data exists:
  - [ ] `/data/kun/jtransdata/addr_pretrain.txt`
  - [ ] `./vocab_addr.pkl`
  - [ ] `/data/kun/jtransdata/func_blocks_addr.json`
  - [ ] `/data/kun/jtransdata/ground_truth_addr.json`
- [ ] Disk space: 200GB+ free
- [ ] RAM: 64GB+ available
- [ ] Understanding: Read "Key Differences from Baseline" section

---

**Last Updated:** January 2026
**Version:** 2.0 (with checkpoint resumption)
