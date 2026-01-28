# Baseline Pipeline Consistency Check

## 1. PRETRAIN (model_baseline.py)

### BinBertModel Class:
```python
class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings.position_embeddings = self.embeddings.word_embeddings
```

### Forward Pass:
```python
outputs = self.bert(
    input_ids=input_ids,              # Token IDs
    attention_mask=attention_mask,    # 1 for real tokens, 0 for padding
    token_type_ids=token_type_ids     # Segment IDs (all 0s)
)
# position_ids NOT passed → uses sequential [0, 1, 2, 3, ...]
```

**Key Points:**
- ✅ Shares position_embeddings layer with word_embeddings
- ✅ Uses sequential position_ids: [0, 1, 2, 3, ...]
- ✅ Passes input_ids, attention_mask, token_type_ids

---

## 2. FINETUNE (finetune.py)

### BinBertModel Class:
```python
class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.embeddings.position_embeddings = self.embeddings.word_embeddings
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, **kwargs):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,  # None → sequential [0, 1, 2, ...]
            **kwargs
        )
```

### Training Forward Pass:
```python
output1 = model(
    input_ids=input_ids1,              # Token IDs
    attention_mask=attention_mask1,    # 1 for real tokens, 0 for padding
    token_type_ids=token_type_ids1     # Segment IDs (all 0s)
)
anchor = output1.pooler_output
# position_ids NOT passed → uses sequential [0, 1, 2, 3, ...]
```

**Key Points:**
- ✅ Shares position_embeddings layer with word_embeddings (SAME as pretrain)
- ✅ Uses sequential position_ids: [0, 1, 2, 3, ...] (SAME as pretrain)
- ✅ Passes input_ids, attention_mask, token_type_ids (SAME as pretrain)
- ✅ Inherits directly from BertModel (cleaner than wrapping)

---

## 3. EVALUATION (evaluate_baseline_pools.py)

### BinBertModel Class:
```python
class BinBertModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.bert.embeddings.position_embeddings = self.bert.embeddings.word_embeddings
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, **kwargs):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,  # None → sequential [0, 1, 2, ...]
            **kwargs
        )
```

### Inference Forward Pass:
```python
outputs = model(
    input_ids=input_ids,              # Token IDs
    attention_mask=attention_mask,    # 1 for real tokens, 0 for padding
    token_type_ids=token_type_ids     # Segment IDs (all 0s)
)
embeddings = outputs.pooler_output
# position_ids NOT passed → uses sequential [0, 1, 2, 3, ...]
```

**Key Points:**
- ✅ Shares position_embeddings layer with word_embeddings (SAME as pretrain/finetune)
- ✅ Uses sequential position_ids: [0, 1, 2, 3, ...] (SAME as pretrain/finetune)
- ✅ Passes input_ids, attention_mask, token_type_ids (SAME as pretrain/finetune)
- ✅ Wraps BertModel (needed for loading state_dict)

---

## CONSISTENCY VERIFICATION ✅

| Component | position_embeddings | position_ids | input_ids | attention_mask | token_type_ids |
|-----------|---------------------|--------------|-----------|----------------|----------------|
| Pretrain  | = word_embeddings   | [0,1,2,3]    | ✅        | ✅             | ✅ (all 0s)    |
| Finetune  | = word_embeddings   | [0,1,2,3]    | ✅        | ✅             | ✅ (all 0s)    |
| Evaluation| = word_embeddings   | [0,1,2,3]    | ✅        | ✅             | ✅ (all 0s)    |

**Status: ALL CONSISTENT ✅**

---

## NEXT STEPS

1. ✅ Code is now consistent across pretrain/finetune/evaluation
2. ❌ Old finetune checkpoints used WRONG behavior (position_ids=input_ids)
3. ⚠️ **MUST RE-FINETUNE** with corrected finetune.py
4. ⚠️ Delete old checkpoints: `rm -rf /home/kun/Document/AAE/output/jtrans/baseline_finetune/finetune_epoch_*`
5. ⚠️ Re-run: `bash run_baseline_finetune.sh` (or HPC equivalent)

---

## TECHNICAL DETAILS

### What does `position_embeddings = word_embeddings` do?

When you assign:
```python
self.embeddings.position_embeddings = self.embeddings.word_embeddings
```

- Both variables point to the SAME nn.Embedding object
- position_ids [0, 1, 2, 3, ...] index into word_embeddings matrix
- The model learns positional patterns from token context
- This is the original jTrans trick for binary code

### Why NOT `position_ids = input_ids`?

- That would make position index = token ID
- Token IDs are vocabulary indices (0-2902), not sequence positions
- Would break the learned patterns from pretrain
- Old finetune used this by mistake


---

## FINAL CHECKLIST BEFORE RETRAINING

### ✅ Code Consistency (All Fixed)
- [x] Pretrain: `position_embeddings = word_embeddings`, sequential position_ids
- [x] Finetune: `position_embeddings = word_embeddings`, sequential position_ids  
- [x] Evaluation: `position_embeddings = word_embeddings`, sequential position_ids
- [x] All pass: input_ids, attention_mask, token_type_ids
- [x] Evaluation now creates attention_mask and token_type_ids (was missing)

### ⚠️ Action Required
1. Backup old checkpoints (for comparison):
   ```bash
   mv /home/kun/Document/AAE/output/jtrans/baseline_finetune \
      /home/kun/Document/AAE/output/jtrans/baseline_finetune_OLD_WRONG
   ```

2. Run baseline finetune:
   ```bash
   cd /home/kun/Document/AAE/extern/jTrans
   bash run_finetune_baseline.sh
   ```
   
3. After training, run evaluation:
   ```bash
   bash run_baseline_pool_evaluation.sh \
     /home/kun/Document/AAE/output/jtrans/baseline_finetune/finetune_epoch_10 \
     /home/kun/Document/AAE/extern/jTrans/pretrain/baseline
   ```

### Expected Improvements
- Previous finetune: Wrong position embeddings → Poor performance
- New finetune: Correct position embeddings → Should match pretrain behavior
- Evaluation: Now consistent with both pretrain and finetune

---

## SUMMARY

**Everything is now consistent!** The baseline pipeline (pretrain → finetune → evaluation) all use:
- Same position embedding trick: `position_embeddings = word_embeddings`
- Same position_ids: sequential [0, 1, 2, 3, ...]
- Same inputs: input_ids, attention_mask, token_type_ids

You're ready to retrain! 🚀

---

## ADDITIONAL FIXES APPLIED

### Issue 2: Tokenization Mismatch ❌ → ✅

**Problem Found:**
- Evaluation was manually splitting tokens and using `convert_tokens_to_ids()`
- Did NOT add [CLS] and [SEP] tokens
- Different from finetune which uses `tokenizer.encode_plus()`

**Fix Applied:**
```python
# OLD (WRONG):
tokens = tokens_str.split('\t')
token_ids = tokenizer.convert_tokens_to_ids(all_tokens[:max_len])
# No [CLS] or [SEP]!

# NEW (CORRECT - same as finetune):
encoded = tokenizer.encode_plus(
    func_str,
    max_length=max_len,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
# Automatically adds [CLS] and [SEP] ✅
```

### Token Type IDs Verification ✅

Confirmed all three use **all 0s**:
- Pretrain: `token_type_ids = [0] * seq_len`
- Finetune: `token_type_ids = [0] * max_length`
- Evaluation: `encode_plus()` returns `token_type_ids = [0, 0, 0, ...]`

**All consistent!** ✅

---

## FINAL SUMMARY OF ALL FIXES

| Component  | Fix 1: position_embeddings | Fix 2: tokenization | Fix 3: attention_mask | Fix 4: token_type_ids |
|------------|---------------------------|---------------------|----------------------|----------------------|
| Pretrain   | ✅ (already correct)       | ✅ (already correct) | ✅ (already correct)  | ✅ (all 0s)          |
| Finetune   | ✅ FIXED (was wrong)       | ✅ (already correct) | ✅ (already correct)  | ✅ (all 0s)          |
| Evaluation | ✅ FIXED (was wrong)       | ✅ FIXED (was wrong) | ✅ FIXED (was missing)| ✅ FIXED (now matches)|

**Status: ALL ISSUES FIXED! Everything now consistent! ✅✅✅**

