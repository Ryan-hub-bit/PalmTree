# Dual MLP Architecture for Address Tokens

## Overview
Implemented separate MLP projections for `address` (code addresses) and `daddr` (data addresses) tokens to allow the model to learn distinct representations for control flow vs data flow.

## Architecture Changes

### AddressPositionalEmbedding
**File:** `address_embedding.py`

**Before (Single MLP):**
- One `self.projection` MLP used for all address tokens
- No distinction between code and data addresses

**After (Dual MLP):**
- `self.code_address_projection`: MLP for code addresses (jump/call targets)
- `self.data_address_projection`: MLP for data addresses (memory operands)
- Each MLP: 384 → 1536 (GELU) → 768
- Total parameters: 3,543,552 (1.77M per MLP)

### Implementation Details

#### 1. MLP Architecture
```python
# Code address MLP (for control flow)
self.code_address_projection = nn.Sequential(
    nn.Linear(384, 1536),  # 3 * intermediate_size → d_model * 2
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(1536, 768),  # d_model * 2 → d_model
    nn.Dropout(dropout)
)

# Data address MLP (for data flow)
self.data_address_projection = nn.Sequential(
    nn.Linear(384, 1536),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(1536, 768),
    nn.Dropout(dropout)
)
```

#### 2. Forward Pass Logic
```python
def forward(self, binary_pos, function_pos, bb_pos, token_ids, vocab_stoi):
    # ... encode positions with sin/cos ...
    
    # Determine token types
    address_token_id = vocab_stoi.get('address', -1)
    daddr_token_id = vocab_stoi.get('daddr', -1)
    
    is_code_address = (token_ids == address_token_id).unsqueeze(-1)
    is_data_address = (token_ids == daddr_token_id).unsqueeze(-1)
    
    # Apply separate MLPs
    code_embedding = self.code_address_projection(concatenated)
    data_embedding = self.data_address_projection(concatenated)
    
    # Combine based on token type
    embedding = code_embedding * is_code_address.float() + \
                data_embedding * is_data_address.float()
    
    # Zero out non-address tokens
    embedding = embedding * address_mask.float()
    
    return embedding
```

#### 3. Integration with Model

**AddressAwareBERTEmbedding:**
- Added `vocab_stoi` parameter to constructor
- Passes `token_ids` and `vocab_stoi` to `address_position` module

**Model Creation (train_addressaware.py):**
```python
# Create model
model = create_addressaware_model(...)

# Set vocab_stoi for address/daddr distinction
model.bert.embeddings.vocab_stoi = vocab.stoi

model = model.to(device)
```

## Rationale

### Why Separate MLPs?

1. **Semantic Distinction:**
   - `address`: Control flow addresses (jumps, calls)
     - Critical for understanding program structure
     - Jump Target Prediction (JTP) task focuses on these
     - Should encode function/basic block relationships
   
   - `daddr`: Data flow addresses (memory operands)
     - Critical for understanding data dependencies
     - Should encode memory layout patterns
     - Different optimization goals than control flow

2. **Task-Specific Learning:**
   - Code address MLP can specialize for JTP task
   - Data address MLP can specialize for MLM task (predicting memory operands)
   - Independent parameter spaces prevent interference

3. **Representational Capacity:**
   - Each MLP learns domain-specific features
   - Code MLP: Function boundaries, BB structure, call graphs
   - Data MLP: Memory access patterns, data structure layouts

## Verification

### Test Results (test_dual_mlp.py)

```
✓ SUCCESS: address and daddr produce different embeddings (separate MLPs working!)

Embedding differences:
  address vs daddr (pos 2 vs 5): 1.9853  ← Large difference (different MLPs)
  address vs address (pos 2 vs 7): 0.8119  ← Small difference (same MLP, similar positions)

Parameter counts:
  Code address MLP: 1,771,776 parameters
  Data address MLP: 1,771,776 parameters
  Total: 3,543,552 parameters
```

### Key Observations:
- ✅ Separate MLPs are instantiated correctly
- ✅ Different embeddings for address vs daddr tokens
- ✅ Similar embeddings for same token type with same positions
- ✅ Zero embeddings for non-address tokens
- ✅ Equal parameter counts (architectural symmetry)

## Impact on Training

### Model Size Increase:
- Previous: 1 MLP = 1.77M parameters
- Current: 2 MLPs = 3.54M parameters
- **+1.77M parameters total** (~0.15% for a 768-dim 12-layer model)

### Expected Benefits:
1. Better JTP accuracy (code addresses optimized for jump target prediction)
2. Better MLM accuracy on memory operands (data addresses specialized)
3. Clearer separation of control flow and data flow representations
4. More interpretable embeddings for downstream tasks

### Training Compatibility:
- ✅ No changes to dataloader
- ✅ No changes to loss computation
- ✅ No changes to training loop
- ✅ Backward compatible with existing checkpoints (if loaded with `strict=False`)

## Files Modified

1. **address_embedding.py:**
   - `AddressPositionalEmbedding.__init__()`: Added dual MLPs
   - `AddressPositionalEmbedding.forward()`: Added token type distinction logic
   - `AddressAwareBERTEmbedding.__init__()`: Added `vocab_stoi` parameter
   - `AddressAwareBERTEmbedding.forward()`: Pass token_ids to address_position

2. **model_addressaware.py:**
   - Added `vocab_stoi=None` parameter to AddressAwareBERTEmbedding initialization

3. **train_addressaware.py:**
   - Set `model.bert.embeddings.vocab_stoi = vocab.stoi` after model creation

4. **test_dual_mlp.py:**
   - New test script to verify dual MLP functionality

## Usage Notes

### For Training:
- No changes required to training script
- Vocabulary must have 'address' and 'daddr' tokens
- Model will automatically use appropriate MLP based on token type

### For Inference:
- Same forward pass interface
- Token type determines which MLP is used
- No special handling required

### For Checkpoints:
- New checkpoints will have separate MLPs
- Old checkpoints can be loaded with `strict=False` if needed
- Will need to initialize new MLP parameters randomly

## Future Work

### Potential Extensions:
1. **Learnable Gating:** Mix both MLPs with learned weights
2. **Shared Lower Layers:** Share input layer, split only at output
3. **Triple MLP:** Add separate MLP for `var` tokens
4. **Attention-Based Routing:** Use attention to dynamically select MLP

### Evaluation Ideas:
1. Probe representations to verify control vs data flow separation
2. Compare JTP accuracy with dual vs single MLP
3. Analyze gradient flow to each MLP during training
4. Visualize embedding spaces for address vs daddr tokens
