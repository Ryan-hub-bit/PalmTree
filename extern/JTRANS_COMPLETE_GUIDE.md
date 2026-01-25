# jTrans Complete Guide: Baseline, AddressAware & Instruction-Level Models

This guide covers all three jTrans model variants for binary code similarity detection.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architectures](#model-architectures)
3. [Data Generation Pipeline](#data-generation-pipeline)
4. [Pretrain Phase](#pretrain-phase)
5. [Finetune Phase](#finetune-phase)
6. [Evaluation](#evaluation)
7. [Complete Workflow Examples](#complete-workflow-examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Three Model Variants

| Model | Position Encoding | Jump Addressing | Granularity | Max Length |
|-------|------------------|-----------------|-------------|------------|
| **Baseline** | Shared with word embeddings | JUMP_ADDR_X (token position) | Token-level | 512 tokens |
| **AddressAware** | Hierarchical (binary+function+BB+offset) | JUMP_ADDR_X (token position) | Token-level | 512 tokens |
| **jTrans_instr** | Standard BERT + instruction_embeddings | instr_addr_{i} (instruction index) | Instruction-level | 201 instructions |

### Key Differences

- **Baseline**: Simple shared position embeddings
- **AddressAware**: Hierarchical position embeddings capture binary/function/basic-block structure
- **jTrans_instr**: Instruction-level embeddings, different addressing scheme for jumps

---

## Model Architectures

### 1. Baseline Architecture

```python
class BaselineBertModel:
    word_embeddings: 768-dim
    position_embeddings: 768-dim (SHARED with word_embeddings)
    token_type_embeddings: 768-dim
    
    transformer_layers: 12 × (768-dim, 12 heads)
    
    max_position_embeddings: 512
```

**Key Feature**: Position embeddings are shared/tied with word embeddings to save parameters.

**Jump Encoding**: `JUMP_ADDR_X` where X is the token position (0-indexed) of the jump target.

---

### 2. AddressAware Architecture

```python
class AddressAwareBertModel:
    word_embeddings: 768-dim
    
    # Hierarchical position embeddings
    binary_pos_embeddings: 192-dim (max 10 binaries)
    function_pos_embeddings: 192-dim (max 200 functions)
    bb_pos_embeddings: 192-dim (max 50 basic blocks)
    var_offset_embeddings: 192-dim (max 512 positions)
    # Total: 4 × 192 = 768-dim
    
    token_type_embeddings: 768-dim
    
    transformer_layers: 12 × (768-dim, 12 heads)
    
    max_position_embeddings: 512
```

**Key Feature**: Position information encoded hierarchically across binary/function/BB/token levels.

**Jump Encoding**: Same as baseline - `JUMP_ADDR_X` where X is the token position.

---

### 3. jTrans_instr Architecture

```python
class InstrBertModel:
    word_embeddings: 768-dim
    position_embeddings: 768-dim (standard BERT)
    instruction_embeddings: 768-dim (NEW - instruction-level)
    token_type_embeddings: 768-dim
    
    transformer_layers: 12 × (768-dim, 12 heads)
    
    max_position_embeddings: 512 tokens
    max_instruction_embeddings: 201 instructions
```

**Key Feature**: Separate instruction embeddings layer to distinguish instruction boundaries.

**Jump Encoding**: `instr_addr_{i}` where i is the instruction index (not token position).

**Instruction Boundary**: Uses `\t` separator between instructions (NOT mnemonics-based detection).

---

## Data Generation Pipeline

### Prerequisites

- IDA Pro 9.0 (with Python3 support)
- Binary files in `/data/kun/jtransdata/small_train/` (or your binary directory)
- Python 3.x with networkx, pickle

### Step 1: IDA Pro Extraction (Shared by All Models)

All three models use the same pickle files from IDA extraction.

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils

# Run IDA Pro extraction
python3 run.py
```

**Output**: `/data/kun/jtransdata/extract/*_extract.pkl`

**Pickle Format**:
```python
{
    'func_name': {
        'asm': ['push rbp', 'mov rsp , rbp', ...],
        'cfg': NetworkX DiGraph,
        'data_refs': [...],
        ...
    }
}
```

---

### Step 2A: Baseline Data Generation

#### Pretrain Data

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils

# Convert pickles to pretrain text
python3 convert_pkl_to_text.py \
    /data/kun/jtransdata/extract \
    /data/kun/jtransdata/baseline_pretrain.txt \
    --min-instructions 5 \
    --max-instructions 512
```

**Output Format** (one function per line, space-separated):
```
push rbp mov rsp , rbp sub rsp , CONST jmp JUMP_ADDR_5 nop ret
```

**Build Vocabulary**:
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline

python create_vocab.py \
    --input_file /data/kun/jtransdata/baseline_pretrain.txt \
    --output_dir ./baseline_tokenizer \
    --vocab_size 50000
```

#### Finetune Data

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils

python3 create_baseline_dataset.py \
    /data/kun/jtransdata/extract \
    /data/kun/jtransdata \
    --binary-dir /data/kun/jtransdata/small_train
```

**Outputs**:
- `/data/kun/jtransdata/func_blocks_baseline.json`
- `/data/kun/jtransdata/ground_truth_baseline.json`

**Or use the complete script**:
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils
./generate_baseline.sh
```

---

### Step 2B: AddressAware Data Generation

#### Pretrain Data

AddressAware uses the same pretrain text as baseline:
```bash
# Reuse baseline_pretrain.txt
cp /data/kun/jtransdata/baseline_pretrain.txt \
   /data/kun/jtransdata/addressaware_pretrain.txt
```

**Build Vocabulary**:
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/addressaware

python create_vocab.py \
    --input_file /data/kun/jtransdata/addressaware_pretrain.txt \
    --output_dir ./addressaware_tokenizer \
    --vocab_size 50000
```

#### Finetune Data

```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_addraware

./run_pipeline.sh
```

**Outputs**:
- `/data/kun/jtransdata/func_blocks_addr.json`
- `/data/kun/jtransdata/ground_truth_addr.json`

---

### Step 2C: jTrans_instr Data Generation

#### Pretrain Data

```bash
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils

# Option 1: Generate new pickles
./generate_instr.sh

# Option 2: Reuse existing pickles
./generate_instr.sh --reuse-pickles
```

**Convert to text format**:
```bash
python3 convert_pkl_to_text.py \
    /data/kun/jtrans_instr/extract \
    /data/kun/jtrans_instr/instr_pretrain.txt \
    --min-instructions 3 \
    --max-instructions 200
```

**Output Format** (instructions separated by `\t`):
```
push rbp	mov rsp , rbp	sub rsp , CONST	jmp instr_addr_5	nop	ret
```

**Build Vocabulary**:
```bash
cd /home/kun/Document/AAE/extern/jTrans_instr/pretrain

./build_vocab.sh
```

#### Finetune Data

```bash
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils

python3 create_instr_dataset.py \
    /data/kun/jtrans_instr/extract \
    /data/kun/jtrans_instr \
    --binary-dir /data/kun/jtransdata/small_train
```

**Outputs**:
- `/data/kun/jtrans_instr/func_blocks_instr.json`
- `/data/kun/jtrans_instr/ground_truth_instr.json`

---

## Pretrain Phase

### Baseline Pretrain

```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline

./run_baseline_pretrain.sh
```

**Configuration**:
- Task: MLM (15%) + JTP (20%)
- Max length: 512 tokens
- Hidden: 768, Layers: 12, Heads: 12
- Position embeddings: Shared with word embeddings

**Output**: Model checkpoint in training output directory

---

### AddressAware Pretrain

```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/addressaware

./run_addressaware_pretrain.sh
```

**Configuration**:
- Task: MLM (15%) + JTP (20%)
- Max length: 512 tokens
- Hidden: 768, Layers: 12, Heads: 12
- Position embeddings: Hierarchical (binary+function+BB+offset)

**Output**: Model checkpoint with hierarchical position embeddings

---

### jTrans_instr Pretrain

```bash
cd /home/kun/Document/AAE/extern/jTrans_instr/pretrain

./run_instr_pretrain.sh
```

**Configuration**:
- Task: MLM (15%) + JTP (20%)
- Max length: 512 tokens, 201 instructions
- Hidden: 768, Layers: 12, Heads: 12
- Instruction embeddings: Separate layer
- Instruction boundary: `\t` separator

**Output**: Model checkpoint with instruction embeddings

---

## Finetune Phase

### Baseline Finetune

```bash
cd /home/kun/Document/AAE/extern/jTrans

./run_finetune_baseline.sh
```

**Input Data**:
- `func_blocks_baseline.json`
- `ground_truth_baseline.json`

**Key Parameters**:
- Learning rate: 1e-5
- Batch size: 32
- Epochs: 10-20

---

### AddressAware Finetune

```bash
cd /home/kun/Document/AAE/extern/jTrans

./run_finetune_addressaware.sh
```

**Input Data**:
- `func_blocks_addr.json`
- `ground_truth_addr.json`

**Key Parameters**:
- Same as baseline
- Uses hierarchical position embeddings from pretrain

---

### jTrans_instr Finetune

```bash
cd /home/kun/Document/AAE/extern/jTrans_instr

./run_finetune_instr.sh
```

**Input Data**:
- `func_blocks_instr.json`
- `ground_truth_instr.json`

**Key Parameters**:
- Same architecture as pretrain
- Instruction-level addressing maintained

---

## Evaluation

### Fair Comparison Setup

To fairly compare all three models, generate pools with identical function IDs:

```bash
cd /home/kun/Document/AAE/extern/jTrans_instr

python3 generate_fair_pools_all.py \
    --baseline-ground-truth /data/kun/jtransdata/ground_truth_baseline.json \
    --addressaware-ground-truth /data/kun/jtransdata/ground_truth_addr.json \
    --instr-ground-truth /data/kun/jtrans_instr/ground_truth_instr.json \
    --output-dir /data/kun/fair_pools
```

**Outputs**:
- `pool_*.json` - Contains baseline IDs
- `pool_*_meta.json` - Contains all three model IDs

---

### Baseline Evaluation

```bash
cd /home/kun/Document/AAE/extern/jTrans

python3 evaluate_baseline_with_pools.py \
    --model-path /path/to/baseline/checkpoint \
    --pool-dir /data/kun/fair_pools \
    --func-blocks /data/kun/jtransdata/func_blocks_baseline.json \
    --output-file baseline_results.json
```

---

### AddressAware Evaluation

```bash
cd /home/kun/Document/AAE/extern/jTrans

python3 evaluate_addressaware_with_pools.py \
    --model-path /path/to/addressaware/checkpoint \
    --pool-dir /data/kun/fair_pools \
    --func-blocks /data/kun/jtransdata/func_blocks_addr.json \
    --output-file addressaware_results.json
```

---

### jTrans_instr Evaluation

```bash
cd /home/kun/Document/AAE/extern/jTrans_instr

./run_evaluate_instr.sh
```

---

## Complete Workflow Examples

### End-to-End Baseline

```bash
# 1. Generate all data (IDA + pretrain + finetune)
cd /home/kun/Document/AAE/extern/jTrans/datautils
./generate_baseline.sh

# 2. Build vocabulary
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
python create_vocab.py \
    --input_file /data/kun/jtransdata/baseline_pretrain.txt \
    --output_dir ./baseline_tokenizer \
    --vocab_size 50000

# 3. Pretrain
./run_baseline_pretrain.sh

# 4. Finetune
cd /home/kun/Document/AAE/extern/jTrans
./run_finetune_baseline.sh

# 5. Generate fair pools
cd /home/kun/Document/AAE/extern/jTrans_instr
python3 generate_fair_pools_all.py \
    --baseline-ground-truth /data/kun/jtransdata/ground_truth_baseline.json \
    --addressaware-ground-truth /data/kun/jtransdata/ground_truth_addr.json \
    --instr-ground-truth /data/kun/jtrans_instr/ground_truth_instr.json \
    --output-dir /data/kun/fair_pools

# 6. Evaluate
cd /home/kun/Document/AAE/extern/jTrans
python3 evaluate_baseline_with_pools.py \
    --model-path ./output/baseline_finetune/checkpoint-best \
    --pool-dir /data/kun/fair_pools \
    --func-blocks /data/kun/jtransdata/func_blocks_baseline.json \
    --output-file baseline_results.json
```

---

### End-to-End AddressAware

```bash
# 1. Reuse baseline pickles and pretrain text
cd /home/kun/Document/AAE/extern/jTrans/datautils_addraware
./run_pipeline.sh

# 2. Build vocabulary
cd /home/kun/Document/AAE/extern/jTrans/pretrain/addressaware
python create_vocab.py \
    --input_file /data/kun/jtransdata/baseline_pretrain.txt \
    --output_dir ./addressaware_tokenizer \
    --vocab_size 50000

# 3. Pretrain
./run_addressaware_pretrain.sh

# 4. Finetune
cd /home/kun/Document/AAE/extern/jTrans
./run_finetune_addressaware.sh

# 5. Evaluate (uses fair pools from baseline setup)
python3 evaluate_addressaware_with_pools.py \
    --model-path ./output/addressaware_finetune/checkpoint-best \
    --pool-dir /data/kun/fair_pools \
    --func-blocks /data/kun/jtransdata/func_blocks_addr.json \
    --output-file addressaware_results.json
```

---

### End-to-End jTrans_instr

```bash
# 1. Generate data (reuse pickles from baseline)
cd /home/kun/Document/AAE/extern/jTrans_instr/datautils
./generate_instr.sh --reuse-pickles

# 2. Build vocabulary
cd /home/kun/Document/AAE/extern/jTrans_instr/pretrain
./build_vocab.sh

# 3. Pretrain
./run_instr_pretrain.sh

# 4. Finetune
cd /home/kun/Document/AAE/extern/jTrans_instr
./run_finetune_instr.sh

# 5. Evaluate (uses fair pools from baseline setup)
./run_evaluate_instr.sh
```

---

## Troubleshooting

### Issue: IDA Pro fails with "Python3 not found"

**Solution**: Switch IDA to system Python3
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils
./ida-pro-9.0/idapyswitch
```

---

### Issue: Pickle files not found

**Check**:
```bash
find /data/kun/jtransdata/extract -name "*_extract.pkl" | head -5
```

**Solution**: Run IDA extraction first
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils
python3 run.py
```

---

### Issue: Vocabulary mismatch between pretrain and finetune

**Cause**: Using different tokenizers

**Solution**: Ensure pretrain tokenizer is reused:
```python
# In finetune/eval
tokenizer = BertTokenizer.from_pretrained('../pretrain/baseline/baseline_tokenizer')
```

---

### Issue: jTrans_instr instruction boundaries not detected

**Cause**: Using mnemonics instead of `\t` separator

**Solution**: Verify data uses `\t`:
```bash
head -1 /data/kun/jtrans_instr/instr_pretrain.txt | sed 's/\t/\\t/g'
# Should see \t between instructions
```

---

### Issue: Jump addresses don't match between models

**Expected**:
- Baseline/AddressAware: `JUMP_ADDR_5` (token position 5)
- jTrans_instr: `instr_addr_2` (instruction index 2)

**Solution**: This is correct - they use different addressing schemes.

---

### Issue: Pool evaluation gives different results

**Cause**: Not using fair pools with meta.json

**Solution**: Regenerate pools with `generate_fair_pools_all.py` to ensure same function IDs across models.

---

### Issue: Out of memory during pretrain

**Solution**: Reduce batch size
```bash
# In run_*_pretrain.sh
--per_device_train_batch_size 16  # reduce from 32
--gradient_accumulation_steps 2    # add to maintain effective batch size
```

---

## Quick Reference

### File Locations

| Component | Baseline | AddressAware | jTrans_instr |
|-----------|----------|--------------|--------------|
| Pickles | `/data/kun/jtransdata/extract/` | Same | `/data/kun/jtrans_instr/extract/` |
| Pretrain text | `baseline_pretrain.txt` | `addressaware_pretrain.txt` | `instr_pretrain.txt` |
| Func blocks | `func_blocks_baseline.json` | `func_blocks_addr.json` | `func_blocks_instr.json` |
| Ground truth | `ground_truth_baseline.json` | `ground_truth_addr.json` | `ground_truth_instr.json` |
| Tokenizer | `pretrain/baseline/baseline_tokenizer/` | `pretrain/addressaware/addressaware_tokenizer/` | `pretrain/jtrans_tokenizer/` |

### Command Summary

```bash
# Data generation
./generate_baseline.sh                    # Baseline (includes pretrain text)
./run_pipeline.sh                         # AddressAware
./generate_instr.sh --reuse-pickles      # jTrans_instr

# Pretrain
./run_baseline_pretrain.sh               # Baseline
./run_addressaware_pretrain.sh           # AddressAware
./run_instr_pretrain.sh                  # jTrans_instr

# Finetune
./run_finetune_baseline.sh               # Baseline
./run_finetune_addressaware.sh           # AddressAware
./run_finetune_instr.sh                  # jTrans_instr

# Evaluate
evaluate_baseline_with_pools.py          # Baseline
evaluate_addressaware_with_pools.py      # AddressAware
./run_evaluate_instr.sh                  # jTrans_instr
```

---

## Key Takeaways

1. **All models share IDA pickle extraction** - only generate once
2. **Baseline and AddressAware use same pretrain text** - only tokenization differs
3. **jTrans_instr uses instruction-level addressing** - different from token-level
4. **Fair pools are essential** - ensure same functions evaluated across models
5. **Vocabulary must be consistent** - pretrain tokenizer reused in finetune/eval
6. **Instruction boundaries use `\t`** - not mnemonics detection

---

**Last Updated**: January 2026  
**Author**: jTrans Development Team
