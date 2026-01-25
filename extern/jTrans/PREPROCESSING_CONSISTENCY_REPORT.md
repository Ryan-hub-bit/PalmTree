# AddressAware Preprocessing Consistency Report

## Executive Summary

**Status:** ⚠️ **CRITICAL INCONSISTENCY FOUND**

The preprocessing logic for AddressAware models has **one major inconsistency** in segment labeling between:
- **Evaluation** (evaluate_addressaware_with_pools.py): Uses segment=1 for ALL tokens
- **Finetune** (data_json.py): Uses segment=inst_idx+1 (per-instruction segments)
- **Pretrain** (dataloader_addressaware.py): Uses segment=inst_idx+1 (per-instruction segments)

This means **evaluation preprocessing does NOT match pretrain/finetune**, which could cause model performance degradation during evaluation.

---

## Detailed Comparison

### 1. Segment Labeling

| Stage | File | Segment Strategy | Code Location |
|-------|------|------------------|---------------|
| **Pretrain** | `pretrain/address_aware/dataloader_addressaware.py` | `inst_segment = inst_idx + 1` | Line 451 |
| **Finetune** | `data_json.py` | `inst_segment = inst_idx + 1` | Line 438 |
| **Evaluation** | `evaluate_addressaware_with_pools.py` | `all_segments.extend([1] * len(tokens))` | Line 251 |

**Impact:** HIGH ⚠️

Segment embeddings are learned during pretrain/finetune to distinguish different instructions within a function. Using segment=1 for all tokens during evaluation means:
1. The model cannot distinguish between instructions
2. Segment embeddings are effectively unused
3. Model performance may degrade if segment information is important

**Pretrain/Finetune Segment Strategy:**
```python
# Add <sos> at beginning (segment 1)
all_segments.append(1)

# Each instruction gets its own segment number (1-indexed)
for inst_idx, inst_text in enumerate(instructions):
    inst_segment = inst_idx + 1
    all_segments.extend([inst_segment] * len(tokens))

# Add <eos> at end (gets last instruction's segment)
last_segment = inst_idx + 1 if instructions else 1
all_segments.append(last_segment)
```

**Evaluation Segment Strategy (WRONG):**
```python
# Add <sos> at beginning (segment 1)
all_segments.append(1)

# ALL instruction tokens get segment 1 (WRONG!)
for inst_idx, inst_text in enumerate(instructions):
    all_segments.extend([1] * len(tokens))

# Add <eos> at end (segment 1)
all_segments.append(1)
```

---

### 2. Regex Patterns

| Component | Pattern | Status |
|-----------|---------|--------|
| **addr_pattern** | `r'(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)'` | ✅ IDENTICAL |
| **nested_addr_pattern** | `r'address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)'` | ✅ IDENTICAL |
| **daddr_pattern** | `r'daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)'` | ✅ IDENTICAL |
| **var_pattern** | `r'var\((0x[0-9a-fA-F]+)\)'` | ✅ IDENTICAL |

**Impact:** NONE ✅

All three stages use identical regex patterns for parsing instructions.

---

### 3. Two's Complement Conversion

All three stages use identical logic for handling 64-bit negative var offsets:

```python
# Handle 64-bit negative offsets (two's complement)
# Values > 0x7FFFFFFFFFFFFFFF are negative in two's complement
if offset_val > 0x7FFFFFFFFFFFFFFF:
    # Convert to signed 64-bit integer
    offset_val = offset_val - 0x10000000000000000
```

**Locations:**
- Pretrain: Line 152 in dataloader_addressaware.py
- Finetune: Line 385 in data_json.py
- Evaluation: Line 203 in evaluate_addressaware_with_pools.py

**Impact:** NONE ✅

---

### 4. Position Tuple Ordering

All three stages extract positions in the same order:

```python
binary_pos = float(match.group(3))
function_pos = float(match.group(4))
bb_pos = float(match.group(5))

positions.append((binary_pos, function_pos, bb_pos))
```

**Impact:** NONE ✅

---

### 5. Special Token Handling

All three stages use identical SOS/EOS placement:

```python
# Add <sos> at beginning
all_tokens.append('<sos>')
all_positions.append((-1.0, -1.0, -1.0))
all_var_offsets.append(-1)

# Process instructions...

# Add <eos> at end ONLY (NOT after each instruction)
all_tokens.append('<eos>')
all_positions.append((-1.0, -1.0, -1.0))
all_var_offsets.append(-1)
```

**Impact:** NONE ✅

---

### 6. Var Offset Sentinel Values

All three stages use `-1` as sentinel for non-var tokens:

```python
tokens.append('var')
positions.append((-1.0, -1.0, -1.0))  # var tokens don't have positions
var_offsets.append(offset_val)

# For non-var tokens
tokens.append(opcode)
positions.append((binary_pos, function_pos, bb_pos))
var_offsets.append(-1)  # NOT a var, use -1 as sentinel
```

**Impact:** NONE ✅

---

### 7. Segment Clamping

Both pretrain and finetune clamp segment labels to [0, 255]:

```python
# Clamp segment labels to valid range [0, 255] (segment_types=256)
# Functions with >255 instructions will have segments capped at 255
all_segments = [min(seg, 255) for seg in all_segments]
```

**Evaluation does NOT have this clamping** (but it doesn't matter since it only uses segment=1).

**Locations:**
- Pretrain: Line 488 in dataloader_addressaware.py
- Finetune: Line 481 in data_json.py
- Evaluation: NOT PRESENT (but not needed since all segments are 1)

**Impact:** NONE (evaluation doesn't need clamping since it only uses segment=1)

---

## Recommendations

### 🔴 HIGH PRIORITY: Fix Evaluation Segment Labeling

**Problem:** Evaluation uses segment=1 for all tokens, while pretrain/finetune use per-instruction segments.

**Solution:** Update `evaluate_addressaware_with_pools.py` line 251 to match pretrain/finetune:

**Current (WRONG):**
```python
# All tokens in segment 1 (PRETRAIN style, not per-instruction segments)
all_tokens.extend(tokens)
all_positions.extend(positions)
all_var_offsets.extend(var_offsets)
all_segments.extend([1] * len(tokens))
```

**Fix (CORRECT):**
```python
# Segment label = instruction number (1-indexed) - MATCH PRETRAIN/FINETUNE
inst_segment = inst_idx + 1
all_tokens.extend(tokens)
all_positions.extend(positions)
all_var_offsets.extend(var_offsets)
all_segments.extend([inst_segment] * len(tokens))
```

Also update EOS segment at line 257:

**Current (WRONG):**
```python
# Add <eos> at the end ONLY (segment 1)
all_tokens.append('<eos>')
all_positions.append((-1.0, -1.0, -1.0))
all_var_offsets.append(-1)
all_segments.append(1)
```

**Fix (CORRECT):**
```python
# Add <eos> at the end ONLY (gets last instruction's segment) - MATCH PRETRAIN/FINETUNE
last_segment = inst_idx + 1 if instructions else 1
all_tokens.append('<eos>')
all_positions.append((-1.0, -1.0, -1.0))
all_var_offsets.append(-1)
all_segments.append(last_segment)
```

Add segment clamping before returning (around line 268):

```python
# Clamp segment labels to valid range [0, 255] (segment_types=256)
# Functions with >255 instructions will have segments capped at 255
all_segments = [min(seg, 255) for seg in all_segments]
```

### 🟢 LOW PRIORITY: Add Consistency Tests

Create a test script to validate preprocessing consistency:

```python
def test_preprocessing_consistency():
    """Test that pretrain/finetune/evaluation use same preprocessing"""
    sample_inst = "mov(0x1000:0.5:0.3:0.2) rax, address(0x2000:0.6:0.4:0.3)"
    
    # Parse with each implementation
    pretrain_result = pretrain_dataloader._parse_instruction(sample_inst)
    finetune_result = finetune_dataset._parse_instruction(sample_inst)
    evaluation_result = evaluation._parse_instruction(sample_inst)
    
    # Verify identical results
    assert pretrain_result == finetune_result == evaluation_result
```

---

## Verification Steps

After applying the fix:

1. **Verify segment labeling:**
   ```python
   # Sample function: 3 instructions
   func_str = "mov(0x1000:0.1:0.2:0.3) rax, rbx\t" \
              "add(0x1004:0.1:0.2:0.4) rax, 0x10\t" \
              "ret(0x1008:0.1:0.2:0.5)"
   
   result = tokenize_function(func_str, tokenizer, max_length=512)
   segments = result['segment_label']
   
   # Expected:
   # <sos>: segment 1
   # mov, rax, rbx: segment 1
   # add, rax, 0x10: segment 2
   # ret: segment 3
   # <eos>: segment 3
   ```

2. **Test with real data:**
   - Load sample functions from test set
   - Compare evaluation embeddings before/after fix
   - Verify model performance doesn't degrade

3. **Verify segment clamping:**
   - Test function with >255 instructions
   - Verify segments are clamped to [0, 255]

---

## Summary

| Component | Status | Action Required |
|-----------|--------|-----------------|
| Regex Patterns | ✅ Consistent | None |
| Two's Complement | ✅ Consistent | None |
| Position Ordering | ✅ Consistent | None |
| SOS/EOS Placement | ✅ Consistent | None |
| Var Sentinel Values | ✅ Consistent | None |
| **Segment Labeling** | ⚠️ **INCONSISTENT** | **FIX REQUIRED** |

**Conclusion:** The evaluation preprocessing has a critical bug in segment labeling that causes it to diverge from pretrain/finetune. This should be fixed immediately to ensure accurate model evaluation.
