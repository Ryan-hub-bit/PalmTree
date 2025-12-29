# Baseline Dataset Fixed

## Summary

The `dataset_baseline.py` now has the **exact same structure** as `dataset.py`, with only one difference:

### The ONLY Difference: `_mask_positions()` method

This method removes all position information to create a true baseline:

```python
def _mask_positions(self, text):
    """
    Transformations:
      1. daddr(0xADDR:pos:pos:pos) → address
      2. address(0xADDR:pos:pos:pos) → address  
      3. opcode(0xADDR:pos:pos:pos) → opcode
      4. var(0xOFFSET) → var_0xOFFSET
    """
```

### Structure (Same as dataset.py):

1. **Data Loading**: 
   - Each line contains TWO sequences: `seq1 \t seq2`
   - `cfg_lines[i] = [seq1, seq2]`
   - `dfg_lines[i] = [seq1, seq2]`

2. **random_sent() Logic**:
   - 25%: CFG is_next=1, DFG is_next=1 (both consecutive)
   - 25%: CFG is_next=0, DFG is_next=1 (CFG random, DFG consecutive)
   - 25%: CFG is_next=1, DFG is_next=0 (CFG consecutive, DFG random)
   - 25%: CFG is_next=0, DFG is_next=0 (both random)

3. **random_word() Logic**:
   - 15% token masking probability
   - 80% → [MASK], 10% → random token, 10% → keep original

4. **Sequence Format**:
   ```
   <sos> seq1_tokens <eos> seq2_tokens <eos> <pad> <pad> ...
   ```
   - Segment labels: 1 for seq1, 2 for seq2, 0 for padding

5. **Tasks**:
   - **CFG**: MLM (token masking) + CWP (is_next prediction)
   - **DFG**: MLM (token masking) + DUP (is_next prediction)

### Where _mask_positions() is Applied:

```python
def __getitem__(self, item):
    c1, c2, c_label, d1, d2, d_label = self.random_sent(item)

    # Apply position masking to DFG sequences (BASELINE SPECIFIC)
    d1 = self._mask_positions(d1)
    d2 = self._mask_positions(d2)

    d1_random, d1_label = self.random_word(d1)
    d2_random, d2_label = self.random_word(d2)
    
    # ... (process d1, d2)

    # Apply position masking to CFG sequences (BASELINE SPECIFIC)
    c1 = self._mask_positions(c1)
    c2 = self._mask_positions(c2)

    c1 = [self.vocab.sos_index] + [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c1.split()] + [self.vocab.eos_index]
    c2 = [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c2.split()] + [self.vocab.eos_index]
    
    # ... (rest is identical to dataset.py)
```

### Test Results:

✅ Same data loading structure
✅ Same is_next logic for CWP and DUP
✅ Same MLM masking for both CFG and DFG
✅ Same sequence pair format
✅ Position information correctly removed

### Usage:

```python
from palmtree.dataset.dataset_baseline import BaselineDataset

dataset = BaselineDataset(
    dfg_corpus_path='/path/to/dfg_train.txt',
    cfg_corpus_path='/path/to/cfg_train.txt',
    vocab=vocab,
    seq_len=512,
    corpus_lines=None,  # Load all lines
    on_memory=True
)
```

### Comparison:

| Feature | dataset.py | dataset_baseline.py |
|---------|-----------|---------------------|
| Structure | ✓ | ✓ Same |
| CFG_next (CWP) | ✓ | ✓ Same |
| DFG_next (DUP) | ✓ | ✓ Same |
| MLM on CFG | ✓ | ✓ Same |
| MLM on DFG | ✓ | ✓ Same |
| Position Info | ✓ Preserved | ✗ Masked (baseline) |
| Address Types | ✓ daddr/address | ✗ All → "address" |
| Var Format | ✓ var(0xOFFSET) | ✗ var_0xOFFSET |

The baseline is now a **perfect control** for testing the impact of position information!
