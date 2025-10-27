# Analysis: Role-Based Structured Format vs. Minimal Semantic Format

## Overview
Comparing two transformation approaches for assembly → semantic representation:

1. **Current (Minimal)**: Only add LD/ST/CP direction markers for mov instructions
2. **Proposed (Structured)**: Full operand role labeling with immediate bucketing

## Detailed Comparison

### Example Transformation

#### Original Assembly
```
mov rax qword [rbp-0x18]    add rsp 0x20
```

#### Current Minimal Format
```
LD [rbp-0x18] rax    add rsp 0x20
```
- Tokens: `LD`, `[rbp-0x18]`, `rax`, `add`, `rsp`, `0x20` (6 tokens)

#### Proposed Structured Format
```
OP=MOV DST_RAX SRC=MEM BASE_RBP DISP_pagen    OP=ADD DST_RSP SRC_[page+]
```
- Tokens: `OP=MOV`, `DST_RAX`, `SRC=MEM`, `BASE_RBP`, `DISP_pagen`, `OP=ADD`, `DST_RSP`, `SRC_[page+]` (8 tokens)

### Token Count Impact

| Approach | Example Token Count | Change vs Original |
|----------|--------------------|--------------------|
| Original | 7 tokens | baseline |
| Minimal (current) | 6 tokens | **-14% (reduction)** |
| Structured (proposed) | 8 tokens | **+14% (increase)** |

## Pros of Structured Format

✅ **Explicit Role Labeling**
- Every operand has DST/SRC/SRC2 role → no positional ambiguity
- BERT doesn't need to learn "first operand is dest" convention

✅ **Immediate Abstraction**
- `0x20` → `[page+]` generalizes across similar values
- Reduces vocabulary size (no unique constants in vocab)
- May help model learn patterns independent of specific numbers

✅ **Memory Structure**
- `BASE_RBP DISP_pagen` explicitly decomposes memory operands
- Separates register from offset (structural understanding)

✅ **Consistent Format**
- Every instruction follows same `OP=X DST_Y SRC_Z` pattern
- Easier to parse/process programmatically

## Cons of Structured Format

❌ **Token Explosion**
- Simple `mov rax rbx` becomes `OP=MOV DST_RAX SRC_RBX` (5 tokens vs 3)
- **+33% more tokens** on average → slower training, more memory

❌ **Lost Precision**
- `0x20` → `[page+]` loses exact value information
- For binary analysis, specific offsets/constants are often critical
- `lea rax [rdi+8]` vs `lea rax [rdi+16]` become same after bucketing

❌ **Vocabulary Pollution**
- Every register becomes: `DST_RAX`, `SRC_RAX`, `BASE_RAX`, etc.
- **3-4x vocabulary size** for registers alone
- More parameters → harder to train

❌ **Redundant Information**
- Intel syntax already has positional encoding: `mov dest src`
- Adding `DST_`/`SRC_` is redundant if BERT learns position
- Wastes tokens on information already present

❌ **Phase-1 Limitation**
- Current code **ignores INDEX and SCALE** in memory operands
- `[rax+rcx*8]` only captures BASE=RAX, loses rcx and scale
- Incomplete representation

## Critical Question: What Does BERT Learn?

### Hypothesis 1: Position Encoding Sufficient
If BERT's position embeddings already encode "first token is dest, second is source", then explicit DST_/SRC_ labels are **redundant**.

**Test**: Does vanilla BERT perform comparably on tasks like operand extraction?

### Hypothesis 2: Abstraction Helps Generalization
If bucketing immediates (`0x20`→`[page+]`) helps model learn patterns across different constants, structured format wins.

**Test**: Does bucketed format achieve better transfer to unseen code?

### Hypothesis 3: Precision Matters
For binary analysis (vulnerability detection, similarity), exact values are critical. Abstraction hurts.

**Test**: Downstream task performance on binary diffing/similarity.

## Recommendations

### Option A: Stick with Minimal Format ✅ **RECOMMENDED**
**Why:**
- Your current evaluation compares vanilla vs minimal semantic
- Minimal has proven token reduction (26%)
- Changing formats now invalidates all trained models (20 epochs × 2)
- **Finish current experiment first** before exploring new formats

**Next Steps:**
1. Complete evaluation (vanilla vs semantic LD/ST/CP)
2. Analyze results
3. If semantic helps, publish/deploy
4. If not helpful, THEN try structured format as alternative

### Option B: Hybrid Experiment
Test **three** formats in parallel:
1. Vanilla (baseline)
2. Minimal semantic (LD/ST/CP)
3. Structured roles (OP=, DST=, etc.)

**Cost:** 3× training time + 3× model storage

### Option C: Structured Format Improvements
If you really want to try structured format, fix these first:

1. **Add INDEX/SCALE support**:
   ```python
   # [rax+rcx*8+0x10] should produce:
   BASE_RAX IDX_RCX SCALE_8 DISP_smallp
   ```

2. **Reduce token count** with shorthand:
   ```python
   # Instead of: OP=MOV DST_RAX SRC=MEM BASE_RBP DISP_pagen
   # Use: M D:RAX S:M B:RBP D:pg-
   ```

3. **Hybrid immediate mode**: Keep exact values for small constants, bucket only large ones:
   ```python
   if abs(v) <= 128: return str(v)  # exact
   else: return bucket_imm(v)       # abstract
   ```

4. **Test on small dataset first**: Train 1 epoch on 100K samples to verify vocabulary size is reasonable

## My Opinion

**DON'T switch formats now.** Here's why:

1. **You're 90% done** with current evaluation (just need to run test_both_models.py)
2. **Sunk cost**: 20 epochs × 2 formats already trained with minimal semantic
3. **Unknown benefit**: Structured format is unproven, may hurt performance
4. **Complexity**: New format has incomplete implementation (missing IDX/SCALE)
5. **Token cost**: +33% tokens = slower training + larger models

**Better plan:**
1. ✅ Run evaluation on minimal semantic (this week)
2. ✅ Publish results if semantic helps
3. 🔄 Then explore structured format as v2.0 (next month)
4. 🔄 Compare three formats: vanilla / minimal / structured

## If You Insist on Testing Structured Format

Create a **small-scale experiment**:

```bash
# Extract 10K lines from training data
head -10000 data/cfg_2.txt > data/cfg_2_tiny.txt
head -10000 data/dfg_2.txt > data/dfg_2_tiny.txt

# Convert with both formats
python convert_to_semantic.py data/cfg_2_tiny.txt data/cfg_2_tiny_minimal.txt
python convert_to_structured.py --in data/cfg_2_tiny.txt --out data/cfg_2_tiny_structured.txt

# Train 3 models for 5 epochs each (tiny dataset)
# Compare MLM loss convergence

# If structured format converges 2× faster → worth the token cost
# If not → stick with minimal
```

**Test criteria:**
- Structured must converge **≥30% faster** to justify +33% token overhead
- Final perplexity must be **≥20% better** to justify vocabulary explosion

## Conclusion

The structured format is **theoretically interesting** but **practically risky**.

**My recommendation: Finish your current experiment first.**

You've invested significant effort in minimal semantic format. See those results before pivoting. If minimal semantic doesn't help MLM, THEN structured format becomes an interesting alternative to explore.

Don't let perfect be the enemy of good. Ship the current work, iterate later.
