# Complete Pipeline: Baseline vs Address-Aware

This document outlines the complete pipeline for training and evaluating both baseline and address-aware models on the same ground truth data.

---

## Prerequisites

- **Stripped binaries**: `/data/kun/jtransdata/small_train_strip`
- **Non-stripped binaries** (for function names): `/data/kun/jtransdata/small_train`
- IDA Pro 9.0: `/home/kun/ida-pro-9.0/idat64`
- Conda environment: `palmtree` (Python 3.11)

---

## BASELINE PIPELINE

### Step 1: IDA Extraction (Pickle Format)
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils

# Run IDA extraction on stripped binaries
./run_ida_extraction.sh /data/kun/jtransdata/small_train_strip

# Output: /data/kun/jtransdata/baseline_exports/*_functions.pkl
```

### Step 2: Convert Pickle to Text Format
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils

# Convert all pickle files to text format for pretraining
python3 convert_pkl_to_text.py \
    /data/kun/jtransdata/extract \
    /data/kun/jtransdata/baseline_pretrain.txt

# Output:
#   - /data/kun/jtransdata/baseline_pretrain.txt (all functions in text format)
```

### Step 3: Use Original Vocabulary
```bash
# Baseline uses the original jTrans tokenizer vocabulary (no need to create new vocab)
# Vocabulary location: /home/kun/Document/AAE/extern/jTrans/jtrans_tokenizer/original

# The original vocab already contains:
#   - Standard assembly opcodes and registers
#   - JUMP_ADDR_0 - JUMP_ADDR_99 for jump targets
#   - Special tokens (<pad>, <unk>, <eos>, <sos>, <mask>)
```

### Step 4: Pretrain MLM Model
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline

# Train masked language model using all pretraining data
./run_baseline_pretrain.sh

# Outputs:
#   - /home/kun/Document/AAE/output/jtrans/baseline_pretrain/model_epoch_*.pth
#   - /home/kun/Document/AAE/output/jtrans/baseline_pretrain/best_model.pth
#   - /home/kun/Document/AAE/output/jtrans/baseline_pretrain/training.log
```

### Step 5: Generate Function Dataset
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils

# Create baseline function dataset with ground truth
python3 create_baseline_dataset.py \
    /data/kun/jtransdata/extract \
    /data/kun/jtransdata \
    --binary-dir /data/kun/jtransdata/small_train

# Output:
#   - func_blocks_baseline.json (tokenized functions)
#   - ground_truth_baseline.json (same function pairs across opts)
```

### Step 6: Fine-tune for Function Similarity
```bash
cd /home/kun/Document/AAE/extern/jTrans

# Fine-tune on contrastive learning task using best pretrained model
./run_finetune_baseline.sh

# Or run directly:
python3 finetune.py \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks_baseline.json \
    --ground_truth /data/kun/jtransdata/ground_truth_baseline.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/jtrans_tokenizer/original \
    --model_path /home/kun/Document/AAE/output/jtrans/baseline_pretrain/best_model.pth \
    --batch_size 32 \
    --eval_batch_size 64 \
    --lr 2e-5 \
    --epoch 10 \
    --weight_decay 0.01 \
    --freeze_cnt 10 \
    --load_path /home/kun/Document/AAE/output/jtrans/baseline_finetune

# Output: /home/kun/Document/AAE/output/jtrans/baseline_finetune/model_epoch_*.pth
```

### Step 7: Generate Embeddings
```bash
cd /home/kun/Document/AAE/extern/jTrans

# Generate embeddings for all functions
python3 generate_embeddings.py \
    --model_path /home/kun/Document/AAE/output/jtrans/baseline_finetune/model_epoch_10.pth \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/jtrans_tokenizer/original \
    --func_blocks /data/kun/jtransdata/func_blocks_baseline.json \
    --output /home/kun/Document/AAE/output/jtrans/baseline_embeddings.pkl

# Output: baseline_embeddings.pkl
```

### Step 8: Evaluate
```bash
cd /home/kun/Document/AAE/extern/jTrans

# Evaluate on function similarity task
python3 fasteval.py \
    --experiment_path /home/kun/Document/AAE/output/jtrans/baseline_embeddings.pkl \
    --poolsize 32

# Outputs: MRR, Recall@1 for O0-O3, O0-Os, O1-O3, etc.
```

---

## ADDRESS-AWARE PIPELINE

### Step 1: IDA Function Export (Hierarchical Positions)
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_addraware

# Export functions with hierarchical address encoding
./run_ida_function_export.sh /data/kun/jtransdata/small_train_strip

# Output: /data/kun/jtransdata/function_exports/*_functions.txt
```

### Step 2: Combine & Smart Merge (For Pretraining)
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_addraware

# Combine ALL functions into single file for pretraining (no train/val/test split)
./run_combine_pretrain.sh /data/kun/jtransdata/function_exports /data/kun/jtransdata

# Output:
#   - /data/kun/jtransdata/addr_pretrain.txt (all functions, smart merged)
#   - /data/kun/jtransdata/top_symbols.txt (top 100 symbols)
```
### Step 3: Create Vocabulary
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware

# Create address-aware vocabulary from pretraining data
# Creates BOTH vocab_addr.txt (human-readable) and vocab_addr.pkl (for training)
python3 create_vocab.py \
    --input /data/kun/jtransdata/addr_pretrain.txt \
    --output /data/kun/jtransdata/vocab_addr.txt \
    --min_freq 50

# Outputs:
#   - /data/kun/jtransdata/vocab_addr.txt (human-readable vocabulary)
#   - /data/kun/jtransdata/vocab_addr.pkl (pickle format for training)
```

### Step 4: Pretrain MLM Model
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware

# Train masked language model with address-aware tokens
# Uses same file for train and test (no validation needed during pretraining)
./run_addressaware_pretrain.sh

# Outputs:
#   - /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/model_epoch_*.pth
#   - /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/best_model.pth
#   - /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/training.log
```

### Step 5: Generate Function Dataset
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_addraware

# Create address-aware function dataset with ground truth
python3 create_function_dataset.py \
    /data/kun/jtransdata/function_exports \
    /data/kun/jtransdata \
    --binary-dir /data/kun/jtransdata/small_train

# Output:
#   - func_blocks.json (tokenized functions with address encoding)
#   - ground_truth.json (same function pairs across opts - SAME as baseline!)
```

### Step 6: Fine-tune for Function Similarity
```bash
cd /home/kun/Document/AAE/extern/jTrans

# Fine-tune on contrastive learning task using best pretrained model
./run_finetune_addressaware.sh

# Or run directly:
python3 finetune.py \
    --data_type json \
    --func_blocks /data/kun/jtransdata/func_blocks.json \
    --ground_truth /data/kun/jtransdata/ground_truth.json \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
    --model_path /home/kun/Document/AAE/output/jtrans/addressaware_pretrain/best_model.pth \
    --batch_size 32 \
    --eval_batch_size 64 \
    --lr 2e-5 \
    --epoch 10 \
    --weight_decay 0.01 \
    --freeze_cnt 10 \
    --load_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune

# Output: /home/kun/Document/AAE/output/jtrans/addressaware_finetune/model_epoch_*.pth
```

### Step 7: Generate Embeddings
```bash
cd /home/kun/Document/AAE/extern/jTrans

# Generate embeddings for all functions
python3 generate_embeddings.py \
    --model_path /home/kun/Document/AAE/output/jtrans/addressaware_finetune/model_epoch_10.pth \
    --tokenizer /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware \
    --func_blocks /data/kun/jtransdata/func_blocks.json \
    --output /home/kun/Document/AAE/output/jtrans/addressaware_embeddings.pkl

# Output: addressaware_embeddings.pkl
```

### Step 8: Evaluate
```bash
cd /home/kun/Document/AAE/extern/jTrans

# Evaluate on function similarity task
python3 fasteval.py \
    --experiment_path /home/kun/Document/AAE/output/jtrans/addressaware_embeddings.pkl \
    --poolsize 32

# Outputs: MRR, Recall@1 for O0-O3, O0-Os, O1-O3, etc.
```

---

## KEY DIFFERENCES

### Data Format
| Stage | Baseline | Address-Aware |
|-------|----------|---------------|
| Raw extraction | `*_extract.pkl` | `*_functions.txt` |
| Tokenization | `JUMP_ADDR_X`, `var_xxx`, `arg_xxx` | `address()`, `daddr()`, `var(0xXX)` |
| Position encoding | Relative jump positions | Hierarchical (func:bb:inst) |
| External symbols | `callfunc_xxx` | `daddr(mk_printf:...)` |

### Ground Truth
Both use **IDENTICAL** ground truth:
- Same function matching logic (via `nm` command on non-stripped binaries)
- Same pair generation (all combinations across O0/O1/O2/O3)
- Same `ground_truth.json` structure
- **Fair comparison on same test set**

### Vocabulary Size
- **Baseline**: ~2900 tokens (JUMP_ADDR_0-99, normalized ops)
- **Address-Aware**: ~900 tokens (more compact, address tokens normalized)

---

## COMPARISON & ANALYSIS

After completing both pipelines, compare results:

```bash
# Compare vocabularies
wc -l vocab_baseline.txt vocab_addr.txt

# Compare embedding quality
python3 compare_results.py \
    --baseline /home/kun/Document/AAE/output/jtrans/baseline_embeddings.pkl \
    --addressaware /home/kun/Document/AAE/output/jtrans/addressaware_embeddings.pkl \
    --ground_truth /data/kun/jtransdata/ground_truth.json

# Expected outputs:
#   - MRR scores (baseline vs address-aware)
#   - Recall@1 scores (baseline vs address-aware)
#   - Cross-optimization performance breakdown
```

---

## TROUBLESHOOTING

### Common Issues

1. **IDA extraction hangs**: Kill process, skip problematic binary
2. **Vocabulary pollution**: Increase `--min-freq` to filter rare tokens
3. **OOM during training**: Reduce `--batch_size` or `--max-instructions`
4. **Ground truth mismatch**: Ensure same binaries used for both pipelines

### Validation Checks

```bash
# Check dataset sizes match
jq 'length' /data/kun/jtransdata/func_blocks_baseline.json
jq 'length' /data/kun/jtransdata/func_blocks.json

# Check ground truth pairs match
jq '.pairs | length' /data/kun/jtransdata/ground_truth_baseline.json
jq '.pairs | length' /data/kun/jtransdata/ground_truth.json

# Verify same binaries covered
jq '.pairs[].binary' /data/kun/jtransdata/ground_truth_baseline.json | sort -u
jq '.pairs[].binary' /data/kun/jtransdata/ground_truth.json | sort -u
```

---

## NOTES

- Both pipelines use **stripped binaries** for extraction
- **Non-stripped binaries** only used for function name extraction (ground truth matching)
- Smart merge (top 100 symbols) only applies to address-aware pipeline
- Baseline uses readidadata.py normalization, address-aware uses custom formatting
- Both models trained on same hardware/hyperparameters for fair comparison
