# Baseline Model Fine-tuning and FastEval Checklist

## ✅ Available Resources

### 1. Pretrained Baseline Model
**Location:** `/home/kun/Document/AAE/output/jtrans/baseline_pretrain/best_model/`

Files:
- ✅ `config.json` (585 bytes) - Model architecture config
- ✅ `pytorch_model.bin` (334M) - Pretrained model weights
- ✅ `vocab.txt` (40K, 2898 tokens) - Baseline tokenizer vocabulary
- ✅ `training_info.json` - Training metadata (epoch 6, val_mlm_acc: 83.49%)

Model Configuration:
- Architecture: BinBertModel (BERT-based)
- Hidden size: 768
- Layers: 12
- Attention heads: 12
- Vocab size: 2902
- Max position embeddings: 512

### 2. Fine-tuning Dataset
**Location:** `/data/kun/jtransdata/`

Files:
- ✅ `func_blocks_baseline.json` (738M) - 603,787 functions with baseline tokenization
- ✅ `ground_truth_baseline.json` (44M) - **180,416 ground truth pairs**

Ground Truth Details:
- Same function across different optimization levels
- Covers O0, O1, O2, O3, Os combinations
- Example pairs: O0-O3, O1-O2, O2-Os, etc.

### 3. Fine-tuning Script
**Location:** `/home/kun/Document/AAE/extern/jTrans/finetune.py`

Features:
- ✅ Supports triplet contrastive learning
- ✅ Has JSON data loader (`FunctionDataset_CL_JSON`)
- ✅ Uses cosine similarity loss
- ✅ Supports WandB logging

### 4. Evaluation Script
**Location:** `/home/kun/Document/AAE/extern/jTrans/fasteval.py`

Features:
- ✅ Computes MRR (Mean Reciprocal Rank)
- ✅ Computes Recall@1
- ✅ Tests multiple opt-level pairs (O0-O3, O1-O3, etc.)
- ✅ Uses poolsize for ranking evaluation

---

## 📋 Required Actions for Fine-tuning

### Step 1: Prepare Fine-tuning Command

```bash
cd /home/kun/Document/AAE/extern/jTrans

python3 finetune.py \
    --model_path /home/kun/Document/AAE/output/jtrans/baseline_pretrain/best_model \
    --func_blocks /data/kun/jtransdata/func_blocks_baseline.json \
    --ground_truth /data/kun/jtransdata/ground_truth_baseline.json \
    --output_dir /home/kun/Document/AAE/output/jtrans/baseline_finetune \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 1e-5 \
    --use_json_loader
```

**Note:** Check `finetune.py` for exact argument names. May need to verify:
- Does it support `--model_path` or `--pretrained_model`?
- Does it have `--use_json_loader` flag or automatically detect JSON format?

### Step 2: Generate Embeddings

After fine-tuning, generate embeddings for evaluation:

```bash
python3 eval_save.py \
    --model_path /home/kun/Document/AAE/output/jtrans/baseline_finetune/best_model \
    --func_blocks /data/kun/jtransdata/func_blocks_baseline.json \
    --output_pkl /home/kun/Document/AAE/output/jtrans/baseline_finetune/embeddings.pkl
```

**Note:** Check if `eval_save.py` exists and supports JSON format.

### Step 3: Run FastEval

```bash
python3 fasteval.py \
    --experiment_path /home/kun/Document/AAE/output/jtrans/baseline_finetune/embeddings.pkl \
    --poolsize 32
```

This will output:
- MRR for O0-O3, O0-Os, O1-Os, O1-O3, O2-Os, O2-O3
- Recall@1 for each pair

---

## 🔍 Pre-Flight Checks Needed

### 1. Check finetune.py Arguments
```bash
cd /home/kun/Document/AAE/extern/jTrans
python3 finetune.py --help
```

Look for:
- Model loading arguments
- JSON data loader support
- Output directory options

### 2. Check if JSON Loader is Complete
```bash
grep -n "FunctionDataset_CL_JSON" /home/kun/Document/AAE/extern/jTrans/data_json.py -A 20
```

Verify:
- Can it load func_blocks_baseline.json?
- Can it load ground_truth_baseline.json?
- Does it handle baseline tokenization format?

### 3. Check Tokenizer Compatibility
The pretrained model uses:
- Token format: `JUMP_ADDR_X`, `var_xxx`, `arg_xxx`, `[reg+CONST]`
- Vocab size: 2902

The dataset should match this format (already verified via `readidadata.parse_asm()`).

### 4. Verify eval_save.py Exists
```bash
ls -lh /home/kun/Document/AAE/extern/jTrans/eval_save.py
```

If not, may need to create embedding generation script.

---

## 🎯 Expected Outcomes

After fine-tuning and evaluation:

1. **Fine-tuned Model:** Improved function similarity detection
2. **Embeddings:** Vector representations for 603,787 functions
3. **Metrics:**
   - MRR (Mean Reciprocal Rank): Higher is better
   - Recall@1: Percentage of correct top-1 matches
   - Results for 6 optimization level pairs

---

## 🚨 Potential Issues to Watch

1. **Memory:** 603,787 functions may require significant RAM/VRAM
   - Consider batching during embedding generation
   - May need to subsample for initial testing

2. **Tokenizer Loading:** Ensure vocab.txt is loaded correctly
   - Check if finetune.py expects vocab file path
   - Verify token-to-id mapping matches pretrained model

3. **JSON Format:** Baseline tokenization is space-separated tokens
   - Ensure data loader splits tokens correctly
   - Verify no position encoding mismatches

4. **Ground Truth Format:** Ensure fine-tuning script understands pair structure
   - Check if it expects `func_id1`, `func_id2` fields
   - Verify binary/opt filtering logic

---

## 🔧 Quick Validation Test

Before full fine-tuning, test data loading:

```python
from data_json import FunctionDataset_CL_JSON

# Test dataset loading
dataset = FunctionDataset_CL_JSON(
    func_blocks_path='/data/kun/jtransdata/func_blocks_baseline.json',
    ground_truth_path='/data/kun/jtransdata/ground_truth_baseline.json',
    tokenizer_path='/home/kun/Document/AAE/output/jtrans/baseline_pretrain/best_model/vocab.txt',
    max_len=512
)

print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Sample shapes: {[s.shape for s in sample]}")
```

---

## ✅ Summary

**YOU CAN PROCEED** with fine-tuning and evaluation! All required components are available:

✅ Pretrained model with proper vocab  
✅ Function dataset with baseline tokenization  
✅ Ground truth pairs (180K pairs)  
✅ Fine-tuning script with JSON support  
✅ Evaluation script for MRR/Recall@1  

**Next Step:** Verify the exact arguments for `finetune.py` and start fine-tuning.

