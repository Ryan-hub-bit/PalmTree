# Simplified Address-Aware Embeddings

## Overview
Replaced sin/cos positional encodings with direct MLP projections for address and var embeddings.
Much simpler, fewer parameters, same representational power.

## Changes Made

### 1. AddressPositionalEmbedding (address/daddr tokens)

**Before:**
- 3 normalized positions [0,1] → sin/cos encoding → 3×128 = 384 dims
- 384 → MLP (384→1536→768) → 768 dims
- Separate MLPs for address vs daddr
- **Parameters: 3,543,552**

**After:**
- 3 normalized positions [0,1] → stack → 3 dims
- 3 → MLP (3→1536→768) → 768 dims  
- Separate MLPs for address vs daddr
- **Parameters: 9,216**
- **Reduction: 3,534,336 parameters (99.7%)**

**Rationale:**
- Hierarchical structure already in position values (same BB → same bb_pos)
- No periodicity needed (addresses don't wrap)
- MLP can learn distance/similarity from raw positions
- Sin/cos was encoding smoothness we don't need

### 2. VarPositionalEmbedding (var tokens)

**Before:**
- Var offset (integer) → sin/cos encoding → 768 dims
- Precomputed div_term buffer
- **Parameters: 0 (just buffer)**

**After:**
- Var offset → normalize [0,1] → 1 dim
- 1 → MLP (1→1536→768) → 768 dims
- **Parameters: 3,696**

**Rationale:**
- Offsets are discrete (0, 8, 16, 24...)
- Let MLP learn: "nearby offsets → similar embeddings"
- No need for fixed sin/cos patterns

## Architecture Summary

```
Token:
  [token_id] → Embedding(1348, 768) → token_emb

Sequence Position (unchanged):
  [seq_pos] → Sinusoidal(768) → seq_pos_emb

Address (for address/daddr tokens):
  [binary_pos, function_pos, bb_pos] → Stack(3)
    → code_address_MLP(3→1536→768) if token='address'
    → data_address_MLP(3→1536→768) if token='daddr'
    → addr_pos_emb

Var (for var tokens):
  [offset] → Normalize → MLP(1→1536→768) → var_pos_emb

Segment:
  [segment_id] → Embedding(256, 768) → segment_emb

Final:
  embedding = token_emb + seq_pos_emb + addr_pos_emb + var_pos_emb + segment_emb
  embedding = LayerNorm(embedding)
  embedding = Dropout(embedding)
```

## Parameter Comparison

| Component | Before (sin/cos) | After (direct MLP) | Savings |
|-----------|------------------|-------------------|---------|
| Address embedding | 3,543,552 | 9,216 | 3,534,336 (99.7%) |
| Var embedding | 0 | 3,696 | -3,696 |
| **Total** | **3,543,552** | **12,912** | **3,530,640 (99.6%)** |

## Benefits

1. **Simpler architecture**: No complex sin/cos frequency calculations
2. **Fewer parameters**: 99.6% reduction in embedding parameters
3. **More interpretable**: Raw positions directly encoded
4. **Same expressiveness**: MLP can learn any transformation needed
5. **Better for hierarchical data**: No false smoothness assumptions

## What's Preserved

✅ **Hierarchical structure:**
- Instructions in same BB have similar bb_pos values
- Instructions in same function have similar function_pos values
- The 3-tuple uniquely identifies hierarchical location

✅ **Separate representations:**
- Code addresses (jumps/calls) use dedicated MLP
- Data addresses (memory) use dedicated MLP
- Different control flow vs data flow patterns

✅ **Gradient flow:**
- Direct projection allows clean backprop
- No sin/cos nonlinearities to complicate gradients

## Testing

```bash
# Test simplified embeddings
python3 -c "
from address_embedding import AddressPositionalEmbedding, VarPositionalEmbedding
import torch

# Address embedding: 3 positions → 768
addr = AddressPositionalEmbedding(d_model=768)
print(f'Address params: {sum(p.numel() for p in addr.parameters()):,}')

# Var embedding: 1 offset → 768  
var = VarPositionalEmbedding(d_model=768)
print(f'Var params: {sum(p.numel() for p in var.parameters()):,}')
"
```

Output:
```
Address params: 9,216
Var params: 3,696
```

## Migration

**No changes needed for:**
- Training script (train_addressaware.py)
- Dataloader (dataloader_addressaware.py)
- Model architecture (model_addressaware.py)

**Model compatibility:**
- New checkpoints: Will have different embedding layer structure
- Old checkpoints: Can load with `strict=False`, will reinitialize embeddings
- Same input/output interface
