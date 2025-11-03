# ✅ Extended Vocabulary Setup Complete!

## What Was Done:

### 1. Extended PalmTree Vocabulary
- **Original size**: 6,631 tokens (PalmTree's pre-trained vocabulary)
- **New size**: 6,635 tokens (+4 address tokens)
- **Added tokens**:
  - `addr_start` → ID 6631
  - `addr_end` → ID 6632
  - `addr_code` → ID 6633
  - `addr_data` → ID 6634

### 2. Files Created/Modified:
- ✅ `vocab_extended` - Extended vocabulary file
- ✅ `extend_vocab.py` - Script to extend vocabulary
- ✅ `test_extended_vocab.py` - Verification script
- ✅ `config.py` - Updated to use extended vocabulary
- ✅ `data_loader.py` - Already configured to use address type tokens

### 3. Verified:
- ✅ All 4 address tokens correctly added
- ✅ Bidirectional lookup works (token ↔ ID)
- ✅ Address tags properly tokenized
- ✅ Regular PalmTree tokens unchanged

---

## How It Works:

### Tokenization Example:
```
Input: <addr_start:0x4020f0:0.267785:0.000000> mov rax [ rel <addr_data:0x405150:0.661077:-3.000000> ]

Tokens:
  addr_start (ID: 6631) with position (0.267785, 0.000000)  ← Level 1 + Level 2
  mov        (ID: 5)    with position (0.0, 0.0)            ← Level 1 only
  rax        (ID: 8)    with position (0.0, 0.0)            ← Level 1 only
  [          (ID: 6)    with position (0.0, 0.0)            ← Level 1 only
  rel        (ID: 52)   with position (0.0, 0.0)            ← Level 1 only
  addr_data  (ID: 6634) with position (0.661077, -3.000000) ← Level 1 + Level 2
  ]          (ID: 7)    with position (0.0, 0.0)            ← Level 1 only
```

### 3-Level Embeddings:
1. **Level 1 (Semantic)**: Token ID from extended vocabulary
   - PalmTree tokens: Pre-trained embeddings (frozen or fine-tuned)
   - Address tokens: New embeddings (trained from scratch)

2. **Level 2 (Address Position)**: Binary & Function normalized positions
   - Only non-zero for address tokens
   - Regular tokens get (0.0, 0.0)

3. **Level 3 (Sequence Position)**: Standard positional encoding
   - All tokens get sequence position

---

## Next Steps:

### Immediate:
1. ✅ **DONE**: Extended vocabulary created
2. ✅ **DONE**: Config updated to use extended vocabulary
3. ⏳ **TODO**: Test data_loader with extended vocab

### Training Setup:
4. Load PalmTree's pre-trained model
5. Extend embedding layer from 6631 → 6635 tokens
6. Initialize 4 new embeddings (random or from similar tokens)
7. Add address-specific prediction heads

### Training Strategy:
**Phase 1** (5-10 epochs):
- Freeze PalmTree layers
- Train only:
  - New address token embeddings (4 tokens)
  - Address position embeddings (Level 2)
  - Address prediction heads

**Phase 2** (10-20 epochs):
- Unfreeze everything
- Fine-tune end-to-end
- Use smaller LR for PalmTree layers (1e-5)
- Use larger LR for address components (1e-4)

---

## Files Ready to Use:

### Data Files:
- `vocab_extended` - Use this for training!
- `../bb_pairs_output/atilibusb.so_bb_pairs.txt` - Training data (426 pairs)

### Code Files:
- `data_loader.py` - Ready to load data with extended vocab
- `config.py` - Configured for extended vocab
- `model_with_palmtree.py` - Example model architecture

### Test/Utility Files:
- `test_extended_vocab.py` - Verify vocabulary
- `analyze_example.py` - See tokenization in action
- `extend_vocab.py` - Extend more vocabularies if needed

---

## Recommended Address-Focused Tasks:

### High Priority:
1. **MLM on Address Tokens Only** - Already configured in data_loader.py
   - Change `should_mask = True` to `should_mask = token_ids[i] in addr_token_ids`
   
2. **Control Flow Direction Prediction**
   - Classify: local (same function), forward, backward, external
   - Uses function_norm values directly
   
3. **CFG Reachability with Negative Samples**
   - Add random non-successor BBs (label=0)
   - Current: all pairs are valid (label=1)

### Medium Priority:
4. **Address Type Classification**
   - Predict: addr_start vs addr_end vs addr_code vs addr_data
   
5. **Address Distance Regression**
   - Predict distance between address pairs

---

## Model Architecture Notes:

```python
# Embedding layer sizes:
token_embedding:            [6635, 768]  # 6631 PalmTree + 4 new
binary_position_embedding:  [1, 768]     # Continuous position
function_position_embedding:[1, 768]     # Continuous position

# Address prediction heads:
addr_type_classifier:       [768, 4]     # 4 address types
cf_direction_classifier:    [768, 4]     # 4 directions
mlm_head:                   [768, 6635]  # Predict any token
```

---

## Summary:

✅ **You're all set!** The extended vocabulary is ready to use.

**Key Points:**
- Address tokens (`addr_start`, `addr_end`, `addr_code`, `addr_data`) are now in vocabulary
- They will learn their semantics through training
- Level 2 positions (binary_norm, function_norm) provide spatial information
- PalmTree's pre-trained embeddings give you a strong baseline for regular tokens
- Focus training on address-specific tasks for best results

**Want me to help you set up the actual training script next?**
