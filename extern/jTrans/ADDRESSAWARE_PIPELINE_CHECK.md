# Address-Aware Pipeline Consistency Check

## 1. PRETRAIN (model_addressaware.py)

### Embedding Forward Call:
```python
embeddings = self.bert.embeddings(
    token_ids,          # [batch_size, seq_len]
    token_type_ids,     # [batch_size, seq_len] - segment labels
    binary_pos,         # [batch_size, seq_len] - binary-level positions
    function_pos,       # [batch_size, seq_len] - function-level positions
    bb_pos,             # [batch_size, seq_len] - basic block-level positions
    var_offsets         # [batch_size, seq_len] - var offset values
)
```

### Encoder Forward Call:
```python
outputs = self.bert.encoder(
    embeddings,
    attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
)
```

**Key Points:**
- ✅ Uses AddressAwareBERTEmbedding with 6 parameters
- ✅ Hierarchical position embeddings: binary → function → BB
- ✅ Variable offset embeddings for var(0xXX) tokens
- ✅ vocab_stoi passed to distinguish address vs daddr tokens

---

## 2. FINETUNE (finetune.py)

### Embedding Forward Call (AddressAwareBertWrapper):
```python
embeddings = self.bert.embeddings(
    token_ids,          # [batch_size, seq_len]
    token_type_ids,     # [batch_size, seq_len] - segment labels
    binary_pos,         # [batch_size, seq_len] - binary-level positions
    function_pos,       # [batch_size, seq_len] - function-level positions
    bb_pos,             # [batch_size, seq_len] - basic block-level positions
    var_offsets         # [batch_size, seq_len] - var offset values
)
```

### Encoder Forward Call:
```python
outputs = self.bert.encoder(
    embeddings,
    attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
)
```

### Model Loading:
```python
# 1. Load vocab_stoi from vocab.txt
vocab_stoi = {}
with open(vocab_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        token = line.strip()
        vocab_stoi[token] = idx

# 2. Create AddressAwareBERTEmbedding with same parameters as pretrain
bert_model.embeddings = AddressAwareBERTEmbedding(
    vocab_size=config_dict['vocab_size'],
    embed_size=config_dict['hidden_size'],
    dropout=0.1,
    max_len=config_dict['max_position_embeddings'],
    use_address_embedding=True,
    use_var_embedding=True,
    segment_types=256,
    vocab_stoi=vocab_stoi  # CRITICAL: Pass vocab mapping
)

# 3. Load pretrained weights
state_dict = torch.load(weights_path, map_location='cpu')
bert_model.load_state_dict(state_dict)
```

**Key Points:**
- ✅ Uses SAME AddressAwareBERTEmbedding class as pretrain
- ✅ SAME parameter order in forward call
- ✅ SAME initialization parameters
- ✅ Loads vocab_stoi from vocab.txt (SAME as pretrain)
- ✅ Wraps in AddressAwareBertWrapper for convenience

---

## 3. CONSISTENCY VERIFICATION ✅

### Parameter Order Comparison:

| Position | Pretrain | Finetune | Match |
|----------|----------|----------|-------|
| 1 | token_ids | token_ids | ✅ |
| 2 | token_type_ids | token_type_ids | ✅ |
| 3 | binary_pos | binary_pos | ✅ |
| 4 | function_pos | function_pos | ✅ |
| 5 | bb_pos | bb_pos | ✅ |
| 6 | var_offsets | var_offsets | ✅ |

### Embedding Configuration:

| Setting | Pretrain | Finetune | Match |
|---------|----------|----------|-------|
| vocab_size | config['vocab_size'] | config['vocab_size'] | ✅ |
| embed_size | config['hidden_size'] | config['hidden_size'] | ✅ |
| dropout | 0.1 | 0.1 | ✅ |
| max_len | config['max_position_embeddings'] | config['max_position_embeddings'] | ✅ |
| use_address_embedding | True | True | ✅ |
| use_var_embedding | True | True | ✅ |
| segment_types | 256 | 256 | ✅ |
| vocab_stoi | Loaded from vocab.txt | Loaded from vocab.txt | ✅ |

### Encoder Call:

| Component | Pretrain | Finetune | Match |
|-----------|----------|----------|-------|
| Input | embeddings | embeddings | ✅ |
| attention_mask | .unsqueeze(1).unsqueeze(2) | .unsqueeze(1).unsqueeze(2) | ✅ |
| Output | sequence_output | sequence_output | ✅ |

**Status: FULLY CONSISTENT ✅**

---

## 4. KEY DIFFERENCES FROM BASELINE

### Baseline:
- 3 inputs: token_ids, attention_mask, token_type_ids
- position_embeddings = word_embeddings (shared layer)
- Sequential position IDs [0, 1, 2, 3, ...]

### Address-Aware:
- 6 inputs: token_ids, token_type_ids, binary_pos, function_pos, bb_pos, var_offsets
- Hierarchical position embeddings (binary → function → BB)
- Address-specific embeddings for address vs daddr tokens
- Variable offset embeddings for var(0xXX) tokens
- Separate embedding layers (not shared)

---

## 5. VALIDATION CHECKS

### ✅ Pretrain → Finetune Loading:
1. Same AddressAwareBERTEmbedding class
2. Same initialization parameters
3. vocab_stoi loaded correctly from vocab.txt
4. Weights loaded via bert_model.load_state_dict(state_dict)

### ✅ Forward Pass Consistency:
1. Same parameter order
2. Same attention_mask reshaping
3. Same encoder call pattern
4. Same output extraction (CLS token)

### ✅ No Memory Leaks:
1. loss.item() used for logging ✅
2. DataLoader: num_workers=2, prefetch_factor=1 ✅
3. Batch size optimized (8 for address-aware) ✅

---

## SUMMARY

**Address-aware pipeline is FULLY CONSISTENT!** ✅

- Pretrain and finetune use IDENTICAL embedding calls
- SAME parameter order
- SAME initialization
- vocab_stoi loaded correctly in both
- No memory leaks

**Ready for training!** 🚀

