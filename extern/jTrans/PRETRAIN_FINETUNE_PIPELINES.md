# Pretrain and Finetune Pipelines

This document describes the complete pretrain and finetune pipelines for both baseline and address-aware models.

---

## Table of Contents
1. [Baseline Pipeline](#baseline-pipeline)
2. [Address-Aware Pipeline](#address-aware-pipeline)
3. [Quick Reference](#quick-reference)

---

## Baseline Pipeline

### 1. Pretraining (Baseline)

**Purpose**: Train the model using MLM (Masked Language Modeling) + JTP (Jump-Target Prediction)

**Script**: `pretrain/baseline/run_baseline_pretrain.sh`

**Usage**:
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
./run_baseline_pretrain.sh [data_ratio]

# Examples:
./run_baseline_pretrain.sh 0.1   # Use 10% of data (quick test)
./run_baseline_pretrain.sh 1.0   # Use 100% of data (full training)
```

**Configuration**:
- **Input Data**: `/data/kun/jtransdata/baseline_pretrain.pkl`
- **Vocabulary**: `/data/kun/jtransdata/vocab_baseline.txt`
- **Output**: `/home/kun/Document/AAE/output/jtrans/baseline_pretrain/`
- **GPU**: Controlled by `CUDA_VISIBLE_DEVICES=0`
- **Hyperparameters**:
  - Batch size: 32
  - Epochs: 100
  - Learning rate: 1e-4
  - Mask probability: 0.15

**What it does**:
- Masks 15% of tokens randomly
- Predicts masked tokens (MLM)
- Predicts jump targets for control flow (JTP)
- Saves checkpoints after each epoch
- Saves best model based on validation loss

**Output Files**:
- `checkpoint_epoch_N/` - Model checkpoints
- `training.log` - Training logs
- `best_model/` - Best performing model

---

### 2. Finetuning (Baseline)

**Purpose**: Adapt pretrained model for binary similarity using contrastive learning

**Script**: `run_full_pipeline.sh` or `run_quick_pipeline.sh`

**Usage**:
```bash
cd /home/kun/Document/AAE/extern/jTrans

# Full pipeline (100% data)
./run_full_pipeline.sh [GPU_ID]

# Quick test (0.1% data)
./run_quick_pipeline.sh [GPU_ID]

# Examples:
./run_full_pipeline.sh 0        # Use GPU 0
./run_full_pipeline.sh 1        # Use GPU 1
```

**Configuration**:

**Full Pipeline** (`run_full_pipeline.sh`):
- **Pretrained Model**: `/home/kun/Document/AAE/output/jtrans/baseline_pretrain/checkpoint_epoch_10`
- **Tokenizer**: `/home/kun/Document/AAE/extern/jTrans/pretrain/baseline`
- **Input Data**: 
  - Function blocks: `/data/kun/jtransdata/func_blocks_baseline.json`
  - Ground truth: `/data/kun/jtransdata/ground_truth_baseline.json`
- **Output**: `/home/kun/Document/AAE/output/jtrans/baseline_finetune/`
- **Data ratio**: 1.0 (100%)
- **Epochs**: 5
- **Hyperparameters**:
  - Batch size: 32
  - Learning rate: 1e-5
  - Weight decay: 0.01
  - Freeze layers: 10

**Quick Pipeline** (`run_quick_pipeline.sh`):
- Same as full but:
  - **Data ratio**: 0.001 (0.1%)
  - **Epochs**: 2

**What it does**:
1. **Finetuning**:
   - Loads pretrained model
   - Freezes bottom N layers
   - Trains with contrastive loss (InfoNCE)
   - Positive pairs: Same function, different optimization levels
   - Negative pairs: Different functions

2. **Evaluation**:
   - Tests with pool sizes: 100, 1000, 10000
   - Evaluates 3 pairs: O0→O3, O1→O3, O2→O3
   - Metrics: MRR, Recall@1, Recall@5, Recall@10
   - Uses 10% of queries for evaluation

**Output Files**:
- `pytorch_model.bin` - Finetuned model
- `config.json` - Model configuration
- `embeddings_cache/` - Cached embeddings for evaluation
- `evaluation_results.txt` - Performance metrics

---

## Address-Aware Pipeline

### 1. Pretraining (Address-Aware)

**Purpose**: Train model with hierarchical address embeddings using MLM + JTP

**Script**: `pretrain/address_aware/run_addressaware_pretrain.sh`

**Usage**:
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
./run_addressaware_pretrain.sh [data_ratio]

# Examples:
./run_addressaware_pretrain.sh 0.1   # Use 10% of data
./run_addressaware_pretrain.sh 1.0   # Use 100% of data
```

**Configuration**:
- **Input Data**: `/data/kun/jtransdata/addr_pretrain.txt`
- **Vocabulary**: `./vocab_addr.pkl`
- **Output**: `/home/kun/Document/AAE/output/jtrans/addressaware_pretrain/`
- **GPU**: Controlled by `CUDA_VISIBLE_DEVICES=0,1` (multi-GPU)
- **Hyperparameters**:
  - Batch size: 128
  - Epochs: 10
  - Learning rate: 1e-4
  - Mask probability: 0.15
  - Warmup steps: 10000
  - Model: 768 hidden, 12 layers, 12 heads

**Input Format**:
```
opcode(0xADDR:func_pos:bb_pos:inst_pos) operand1 operand2 ...
```
- Uses hierarchical position encoding: function → basic block → instruction
- Address tokens: `address()` and `daddr()` for direct/indirect addressing

**What it does**:
- Same as baseline but with hierarchical address embeddings
- Separates position embeddings from word embeddings
- Uses dual MLP for address() and daddr() tokens

**Output Files**:
- `checkpoint_epoch_N/` - Model checkpoints
- `training.log` - Training logs

---

### 2. Finetuning (Address-Aware)

**Purpose**: Adapt address-aware pretrained model for binary similarity

**Script**: `run_full_pipeline_addressaware.sh` or `run_quick_pipeline_addressaware.sh`

**Usage**:
```bash
cd /home/kun/Document/AAE/extern/jTrans

# Full pipeline (100% data)
./run_full_pipeline_addressaware.sh [GPU_ID]

# Quick test (0.1% data)
./run_quick_pipeline_addressaware.sh [GPU_ID]

# Examples:
./run_full_pipeline_addressaware.sh 0
./run_quick_pipeline_addressaware.sh 1
```

**Configuration**:

**Full Pipeline** (`run_full_pipeline_addressaware.sh`):
- **Pretrained Model**: `/home/kun/Document/AAE/output/jtrans/addressaware_pretrain/checkpoint_epoch_10`
- **Tokenizer**: `/home/kun/Document/AAE/extern/jTrans/pretrain/address_aware`
- **Input Data**: 
  - Function blocks: `/data/kun/jtransdata/func_blocks_addressaware.json`
  - Ground truth: `/data/kun/jtransdata/ground_truth_addressaware.json`
- **Output**: `/home/kun/Document/AAE/output/jtrans/addressaware_finetune/`
- **Data ratio**: 1.0 (100%)
- **Epochs**: 5

**Quick Pipeline** (`run_quick_pipeline_addressaware.sh`):
- **Data ratio**: 0.001 (0.1%)
- **Epochs**: 2

**What it does**:
- Same as baseline finetuning but with address-aware model
- Uses AddressAwareBertWrapper with hierarchical embeddings
- Evaluation format same as baseline

**Output Files**:
- Same structure as baseline finetuning

---

## Quick Reference

### Complete Training Flow

#### Baseline:
```bash
# 1. Pretrain
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
./run_baseline_pretrain.sh 1.0

# 2. Finetune
cd /home/kun/Document/AAE/extern/jTrans
./run_full_pipeline.sh 0
```

#### Address-Aware:
```bash
# 1. Pretrain
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
./run_addressaware_pretrain.sh 1.0

# 2. Finetune
cd /home/kun/Document/AAE/extern/jTrans
./run_full_pipeline_addressaware.sh 0
```

### Quick Testing Flow

#### Baseline:
```bash
# Quick test with small data
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
./run_baseline_pretrain.sh 0.1

cd /home/kun/Document/AAE/extern/jTrans
./run_quick_pipeline.sh 0
```

#### Address-Aware:
```bash
# Quick test with small data
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
./run_addressaware_pretrain.sh 0.1

cd /home/kun/Document/AAE/extern/jTrans
./run_quick_pipeline_addressaware.sh 0
```

---

## Evaluation Details

### Pool-based Evaluation

After finetuning, the model is evaluated with different pool sizes to test retrieval performance:

**Pool Sizes**: 100, 1000, 10000

**Evaluation Pairs**: 
- O0 → O3 (Query: O0-compiled, Ground Truth: O3-compiled)
- O1 → O3 (Query: O1-compiled, Ground Truth: O3-compiled)
- O2 → O3 (Query: O2-compiled, Ground Truth: O3-compiled)

**For each query**:
1. Take anchor (e.g., O0 version of function)
2. Create pool: 1 ground truth (O3) + (pool_size - 1) negatives from other functions
3. Compute similarity between anchor and all candidates
4. Find rank of ground truth

**Metrics**:
- **MRR** (Mean Reciprocal Rank): Average of 1/rank
- **Recall@1**: % of queries where ground truth is rank 1
- **Recall@5**: % of queries where ground truth is in top 5
- **Recall@10**: % of queries where ground truth is in top 10

**Example Output**:
```
Evaluating O0 vs O3...
100%|████████| 50/50 [00:00<00:00]

Results for O0 vs O3:
  MRR: 0.3500
  Recall@1: 0.2800
  Recall@5: 0.3600
  Recall@10: 0.5000
  Queries: 50

============================================================
Overall Evaluation Results (pool_size=100):
  MRR: 0.3500
  Recall@1: 0.2800
  Recall@5: 0.3600
  Recall@10: 0.5000
  Total Queries: 150
============================================================
```

---

## Data Requirements

### Pretraining Data

**Baseline**:
- Format: Pickled list of tokenized functions
- File: `/data/kun/jtransdata/baseline_pretrain.pkl`
- Vocabulary: `/data/kun/jtransdata/vocab_baseline.txt`
- Tokens separated by tabs between instructions, spaces within instructions

**Address-Aware**:
- Format: Text file with hierarchical position annotations
- File: `/data/kun/jtransdata/addr_pretrain.txt`
- Vocabulary: Pickle file with stoi/itos mappings
- Format: `opcode(0xADDR:func_pos:bb_pos:inst_pos) operands`

### Finetuning Data

**Both Models**:
- **func_blocks JSON**: Maps function IDs to their tokenized representations
  ```json
  {
    "func_id": {
      "O0": "tokenized_function_O0",
      "O1": "tokenized_function_O1",
      "O2": "tokenized_function_O2",
      "O3": "tokenized_function_O3"
    }
  }
  ```

- **ground_truth JSON**: Defines which functions are semantically equivalent
  ```json
  {
    "binary_name": {
      "func_name": ["func_id1", "func_id2", ...]
    }
  }
  ```

---

## Key Differences: Baseline vs Address-Aware

| Feature | Baseline | Address-Aware |
|---------|----------|---------------|
| Position Encoding | position_embeddings = word_embeddings | Hierarchical (func→bb→inst) |
| Address Handling | Generic tokens (JUMP_ADDR_X) | address() and daddr() with dual MLP |
| Data Format | Tab/space separated tokens | Tokens with position annotations |
| Vocabulary Size | ~50K | ~50K |
| Pretraining | MLM + JTP | MLM + JTP + hierarchical positions |
| Model Wrapper | Direct BinBertModel | AddressAwareBertWrapper |
| GPU Usage | Single GPU (pretrain) | Multi-GPU (pretrain) |

---

## Troubleshooting

### Common Issues

1. **Out of Memory during pretraining**:
   - Reduce batch size
   - Use gradient accumulation
   - Reduce sequence length

2. **Pool size limited during evaluation**:
   - Actual pool size = min(requested_size, available_functions)
   - Use more data for larger pools
   - Quick test has ~150 functions, full data has 10K+

3. **Model not loading**:
   - Check checkpoint path exists
   - Verify epoch number in path
   - Ensure vocab/config files present

4. **Low performance**:
   - Check if using pretrained model (not random init)
   - Verify data format matches model type
   - Ensure sufficient training epochs
   - Check learning rate not too high/low

---

## Performance Expectations

### Pretraining
- **Baseline**: ~100 epochs, several hours on single GPU
- **Address-Aware**: ~10 epochs, several hours on 2 GPUs

### Finetuning
- **Full**: ~5 epochs, 1-2 hours
- **Quick**: ~2 epochs, 5-10 minutes

### Evaluation
- **First run**: Slow (generates embeddings)
- **Subsequent runs**: Fast (uses cached embeddings)
- **Speed**: ~3000 queries/second with cached embeddings

### Expected Results (pool_size=10000, full data)
- **MRR**: 0.30-0.40
- **Recall@1**: 0.25-0.35
- **Recall@10**: 0.35-0.45
- Address-aware typically slightly better than baseline
