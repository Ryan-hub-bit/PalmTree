# Dataloader Fix & Retraining Guide

## Summary

**Bug Fixed**: [dataloader_addressaware.py](pretrain/address_aware/dataloader_addressaware.py) was defining `daddr_pattern` (line 73) but never using it during parsing (lines 130-175).

**Impact**: All generated data contained only 'address' tokens, never 'daddr' tokens, causing the `data_address_projection` MLP in the model to never be trained.

**Fix Applied**: Added `daddr_match` check in operand parsing loop (line 136-153) to properly extract 'daddr' tokens.

---

## What Changed

### Before (Buggy Code)
```python
for operand in operands_text.split():
    nested_match = self.nested_addr_pattern.match(operand)
    var_match = self.var_pattern.match(operand)
    
    if nested_match:
        tokens.append('address')
        # ...
    elif var_match:
        tokens.append('var')
        # ...
    else:
        tokens.append(operand)  # ❌ daddr(...) falls here as literal string!
```

### After (Fixed Code)
```python
for operand in operands_text.split():
    nested_match = self.nested_addr_pattern.match(operand)
    daddr_match = self.daddr_pattern.match(operand)  # ✅ NEW
    var_match = self.var_pattern.match(operand)
    
    if nested_match:
        tokens.append('address')
        # ...
    elif daddr_match:  # ✅ NEW
        tokens.append('daddr')
        positions.append((daddr_binary_pos, daddr_function_pos, daddr_bb_pos))
        var_offsets.append(-1)
    elif var_match:
        tokens.append('var')
        # ...
    else:
        tokens.append(operand)
```

---

## Verification

**Test Script**: [test_dataloader_fix.py](test_dataloader_fix.py)

```bash
cd /home/kun/Document/AAE/extern/jTrans
python3 test_dataloader_fix.py
```

**Test Results**: ✅ All 4 test cases passed
- Control flow address (jump targets) → 'address' token
- Data address (memory operands) → 'daddr' token  
- Mixed instructions → Both 'address' and 'daddr' correctly distinguished
- Variable offsets → 'var' token still works

---

## Next Steps

### ⚠️ IMPORTANT: Check Your Raw Data First

Before regenerating everything, verify if your raw corpus already contains `daddr(...)` annotations:

```bash
# Check a sample of your raw pretrain corpus
head -1000 /path/to/your/raw/corpus.txt | grep -o "daddr(" | wc -l
```

**Two scenarios:**

#### Scenario A: Raw corpus HAS `daddr(...)` annotations
- Your data generator already distinguishes code vs data addresses
- Just need to regenerate parsed data with fixed dataloader
- Retrain model from scratch

#### Scenario B: Raw corpus DOES NOT have `daddr(...)`
- Need to fix data generation pipeline first
- Check [cfg_hierarchical_icfg_ida.py](../../data_generator/cfg_hierarchical_icfg_ida.py)
- Add logic to distinguish code addresses (jump targets) vs data addresses (memory operands)
- Regenerate raw corpus → Parse with fixed dataloader → Retrain

---

## Retraining Pipeline

### Step 1: Verify Vocab Contains 'daddr'

```bash
# Check if 'daddr' is in vocabulary
grep "^daddr$" /data/kun/jtrans/addressaware/vocab.txt
```

If not present, add it:
```bash
echo "daddr" >> /data/kun/jtrans/addressaware/vocab.txt
```

### Step 2: Regenerate Training Data

**If raw corpus has daddr:**
```bash
# Regenerate parsed data with fixed dataloader
cd /home/kun/Document/AAE/data_generator
bash generate_data.sh  # Or your specific generation script
```

**If raw corpus needs fixing:**
```bash
# First fix data generator to output daddr(...) for data addresses
# Then regenerate from scratch
cd /home/kun/Document/AAE/data_generator
bash run_ida_generation.sh  # Full pipeline from binaries
```

### Step 3: Verify Generated Data

```bash
python3 << 'EOF'
import json

# Check evaluation data
with open('/data/kun/jtrans/addressaware/eval/func_blocks_addr.json', 'r') as f:
    func_blocks = json.load(f)

for i in range(5):
    func = func_blocks[str(i)]
    tokens = func['tokens'].strip().split()
    print(f"Function {i}: address={tokens.count('address')}, daddr={tokens.count('daddr')}, var={tokens.count('var')}")
EOF
```

**Expected**: You should see non-zero daddr counts if data addresses exist in your binaries.

### Step 4: Retrain Model from Scratch

```bash
cd /home/kun/Document/AAE/extern/jTrans

# Pretrain with fixed dataloader
bash pretrain/address_aware/run_pretrain.sh

# Finetune
bash run_finetune_addressaware.sh

# Evaluate
bash run_addressaware_pool_evaluation.sh
```

### Step 5: Verify Model Learned Both Address Types

After training, check if both MLPs were trained:

```python
import torch

model = torch.load('/path/to/checkpoint.pth', map_location='cpu')

# Check code_address_projection weights
code_weights = model['embedding.address_position.code_address_projection.0.weight']
data_weights = model['embedding.address_position.data_address_projection.0.weight']

print(f"Code address MLP norm: {torch.norm(code_weights).item():.4f}")
print(f"Data address MLP norm: {torch.norm(data_weights).item():.4f}")

# Both should have similar magnitudes if trained properly
```

---

## Expected Performance Improvements

With proper 'address' vs 'daddr' distinction:

1. **Richer Semantic Understanding**
   - Model learns separate representations for control flow (address) vs data flow (daddr)
   - Better captures different usage patterns

2. **Full Model Capacity Utilized**
   - Both code_address_projection and data_address_projection MLPs trained
   - Previously only 50% of address embedding capacity was used

3. **Better Evaluation Results**
   - Should improve over baseline (Recall@1: 0%, MRR: 0.05-0.09)
   - Proper semantic distinction helps with function similarity

---

## Files Modified

1. ✅ **[pretrain/address_aware/dataloader_addressaware.py](pretrain/address_aware/dataloader_addressaware.py)**
   - Lines 136-153: Added daddr_match handling

2. ✅ **[test_dataloader_fix.py](test_dataloader_fix.py)**
   - New file: Verification test script

3. 📝 **This guide**: [DATALOADER_FIX_GUIDE.md](DATALOADER_FIX_GUIDE.md)

---

## Architecture Reminder

**AddressPositionalEmbedding** ([address_embedding.py](pretrain/address_aware/address_embedding.py))

```python
# Dual MLP design:
code_address_projection  # For 'address' tokens (control flow)
data_address_projection  # For 'daddr' tokens (data flow)

# Routing logic:
is_code_address = (token_ids == vocab_stoi['address'])
is_data_address = (token_ids == vocab_stoi['daddr'])

embedding = code_embedding * is_code_address + data_embedding * is_data_address
```

**Token Distinction:**
- `address(...)`: Jump/branch targets, function call addresses (control flow)
- `daddr(...)`: Memory operand addresses, data section references (data flow)
- `var(...)`: Stack/frame pointer offsets (local variables)

---

## Troubleshooting

### Issue: Still seeing zero 'daddr' tokens after regeneration

**Diagnosis:**
```bash
# Check raw corpus
grep "daddr(" /path/to/corpus.txt | head -5
```

If no matches, your data generator doesn't output daddr annotations.

**Solution:** Fix data generator to distinguish:
- Jump/call targets → `address(...)`
- Memory operands (mov [rax], lea [rbx+8]) → `daddr(...)`

### Issue: Model checkpoint incompatible after vocab change

If you add 'daddr' to vocab, old checkpoints won't work (vocab size mismatch).

**Solution:** Train from scratch with new vocab including 'daddr'.

### Issue: Performance still poor after retraining

Check:
1. Is your dataset large enough? (Need sufficient daddr examples)
2. Are daddr tokens properly balanced? (Not 99% address, 1% daddr)
3. Are position values correct? (Should be normalized [0,1])

```python
# Verify data balance
import json
with open('func_blocks_addr.json') as f:
    data = json.load(f)
    
address_total = sum(d['tokens'].split().count('address') for d in data.values())
daddr_total = sum(d['tokens'].split().count('daddr') for d in data.values())
print(f"Ratio address:daddr = {address_total}:{daddr_total}")
```

---

## Questions?

1. **Do I need daddr distinction for my task?**
   - If your downstream task involves data flow analysis, memory dependencies, or pointer tracking → Yes
   - If you only care about control flow graph → Maybe not critical

2. **Can I just remove daddr logic instead?**
   - Yes, if you don't need the distinction
   - Remove data_address_projection MLP and dual routing logic
   - Simpler model, but less expressive

3. **How to decide if raw data needs daddr?**
   - Check a sample binary's disassembly
   - Look for memory operands: `mov rax, [rbx+8]`, `lea rdi, [rip+0x1000]`
   - These should be marked as daddr, not address

---

**Status**: ✅ Dataloader fixed and verified
**Next**: Check raw data → Regenerate → Retrain → Evaluate
