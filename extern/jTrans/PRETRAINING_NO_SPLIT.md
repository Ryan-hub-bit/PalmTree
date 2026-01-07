# Pretraining Data Preparation - Quick Guide

## Key Change: No Train/Val/Test Split for Pretraining

For **masked language model (MLM) pretraining**, we don't need to split data into train/val/test. We use ALL available data to learn token representations. The split only happens later during the **finetuning** phase for function similarity.

---

## Address-Aware Pipeline

### 1. Export Functions with Hierarchical Addresses
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_addraware
./run_ida_function_export.sh /data/kun/jtransdata/small_train_strip
```
**Output**: `/data/kun/jtransdata/function_exports/*_functions.txt`

### 2. Combine All Functions (No Split)
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils_addraware
./run_combine_pretrain.sh /data/kun/jtransdata/function_exports /data/kun/jtransdata
```
**Output**: 
- `/data/kun/jtransdata/addr_pretrain.txt` (ALL functions combined)
- `/data/kun/jtransdata/top_symbols.txt` (top 100 PLT symbols)

### 3. Create Vocabulary
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
python3 create_vocab.py \
    --input /data/kun/jtransdata/addr_pretrain.txt \
    --output /data/kun/jtransdata/vocab_addr.txt \
    --min_freq 50
```
**Output**: 
- `/data/kun/jtransdata/vocab_addr.txt` (human-readable)
- `/data/kun/jtransdata/vocab_addr.pkl` (for training)

### 4. Pretrain MLM
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/address_aware
./run_addressaware_pretrain.sh
```
**Output**: `/home/kun/Document/AAE/output/jtrans/addressaware_pretrain/best_model.pth`

---

## Baseline Pipeline

### 1. Export Functions (Pickle Format)
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils
./run_ida_extraction.sh /data/kun/jtransdata/small_train_strip
```
**Output**: `/data/kun/jtransdata/baseline_exports/*_functions.pkl`

### 2. Combine All Functions (No Split)
```bash
cd /home/kun/Document/AAE/extern/jTrans/datautils
./run_combine_pretrain.sh /data/kun/jtransdata/baseline_exports /data/kun/jtransdata
```
**Output**: 
- `/data/kun/jtransdata/baseline_pretrain.pkl` (ALL functions combined)
- `/data/kun/jtransdata/top_symbols.txt` (top 100 PLT symbols)

### 3. Create Vocabulary
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
python3 create_vocab.py \
    --pkl_file /data/kun/jtransdata/baseline_pretrain.pkl \
    --vocab_file /data/kun/jtransdata/vocab.txt \
    --min_freq 50
```
**Output**: 
- `/data/kun/jtransdata/vocab.txt` (human-readable)
- `/data/kun/jtransdata/vocab.pkl` (for training)

### 4. Pretrain MLM
```bash
cd /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
./run_baseline_pretrain.sh
```
**Output**: `/home/kun/Document/AAE/output/jtrans/baseline_pretrain/best_model.pth`

---

## Key Files Created

| File | Purpose | Format |
|------|---------|--------|
| `combine_function_files_pretrain.py` | Combine ALL functions (address-aware) | Python script |
| `run_combine_pretrain.sh` | Shell wrapper for address-aware combine | Bash script |
| `/datautils/combine_function_files_pretrain.py` | Combine ALL functions (baseline) | Python script |
| `/datautils/run_combine_pretrain.sh` | Shell wrapper for baseline combine | Bash script |
| `addr_pretrain.txt` | All address-aware functions | Text file |
| `baseline_pretrain.pkl` | All baseline functions | Pickle file |
| `vocab_addr.pkl` | Address-aware vocabulary | Pickle file |
| `vocab.pkl` | Baseline vocabulary | Pickle file |

---

## What Changed from Original Pipeline?

### Before (with split):
1. Export functions → separate into train/val/test directories
2. Combine files → create train.txt, val.txt, test.txt (0.8:0.1:0.1)
3. Create vocab from train.txt
4. Pretrain using train.txt for training, val.txt for validation

### Now (no split):
1. Export functions → all go to same directory
2. Combine ALL files → create single pretrain.txt/pkl
3. Create vocab from pretrain data
4. Pretrain using ALL data (same file for train and test args)

### Why?
- **Pretraining**: MLM objective doesn't need validation. More data = better representations
- **Finetuning**: Uses separate function dataset (func_blocks.json) with train/val/test split for contrastive learning

---

## Troubleshooting

### "string" token in vocabulary
Check `function_export_ida.py` line 320 - should keep tokens as-is, not replace with "string"

### CUDA index out of bounds
Vocabulary was created from different data than training data. Regenerate vocab from actual pretraining file:
```bash
# Address-aware
python3 create_vocab.py --input /data/kun/jtransdata/addr_pretrain.txt --output /data/kun/jtransdata/vocab_addr.txt --min_freq 50

# Baseline
python3 create_vocab.py --pkl_file /data/kun/jtransdata/baseline_pretrain.pkl --vocab_file /data/kun/jtransdata/vocab.txt --min_freq 50
```

### Vocab size mismatch
Ensure `min_freq` is the same when creating vocab and must match the data actually used in training
