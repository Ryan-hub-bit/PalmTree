# Address-Aware Pretraining Tasks

## Motivation

While the current address-aware model includes position embeddings at three hierarchical levels (binary, function, basic block), the model needs explicit supervision to learn what these positions **mean**. The existing NSP tasks help with control flow and data flow, but don't directly teach the model about address relationships.

## New Pretraining Tasks

### 1. Address Distance Prediction (ADP)

**Goal**: Teach the model to understand how far apart two instruction segments are.

**Task**: Given two instruction segments (sent1 and sent2), predict the relative distance between them.

**Categories** (5 classes):
- **Very Close** (distance < 0.01): Instructions are consecutive or nearly consecutive
- **Close** (0.01 ≤ distance < 0.05): Instructions are in the same local region
- **Medium** (0.05 ≤ distance < 0.2): Instructions are in nearby regions
- **Far** (0.2 ≤ distance < 0.5): Instructions are in different parts of the function/binary
- **Very Far** (distance ≥ 0.5): Instructions are far apart in the binary

**Distance Calculation**:
```python
# For binary level
distance = abs(sent1_binary_pos - sent2_binary_pos)

# For function level (when both in same function)
distance = abs(sent1_function_pos - sent2_function_pos)

# For BB level (when both in same BB)
distance = abs(sent1_bb_pos - sent2_bb_pos)
```

**Implementation**:
- Use the [CLS] token representation (contains global context of both segments)
- Output layer: 5-way classification
- Loss: Cross-entropy loss

**Benefits**:
- Model learns that position values encode proximity
- Helps model understand that close addresses have related functionality
- Enables better understanding of code locality

### 2. Address Order Prediction (AOP)

**Goal**: Teach the model to understand the sequential order of instructions.

**Task**: Given two instruction segments (sent1 and sent2), predict if sent1 comes before sent2 in the binary.

**Classes** (2 classes):
- **Before** (1): sent1 comes before sent2 (sent1_pos < sent2_pos)
- **After** (0): sent1 comes after sent2 (sent1_pos > sent2_pos)

**Order Determination**:
```python
# For binary level
label = 1 if sent1_binary_pos < sent2_binary_pos else 0

# For function level (within same function)
label = 1 if sent1_function_pos < sent2_function_pos else 0

# For BB level (within same BB)
label = 1 if sent1_bb_pos < sent2_bb_pos else 0
```

**Implementation**:
- Use the [CLS] token representation
- Output layer: 2-way classification (binary)
- Loss: Cross-entropy loss

**Benefits**:
- Model learns that positions encode execution order
- Complements NSP_CFG (which checks control flow validity)
- NSP_CFG: Are these two blocks connected in the CFG? (semantic relationship)
- AOP: Does instruction A come before B in memory? (positional relationship)

## Training Integration

### Multi-Task Learning

The model will be trained with multiple objectives:

```python
total_loss = (
    α * MLM_loss +           # Masked Language Model (existing)
    β * NSP_CFG_loss +       # CFG Next Sentence Prediction (existing)
    γ * NSP_DFG_loss +       # DFG Next Sentence Prediction (existing)
    δ * ADP_loss +           # Address Distance Prediction (NEW)
    ε * AOP_loss             # Address Order Prediction (NEW)
)
```

**Suggested weights**:
- α = 1.0 (MLM is primary task)
- β = 1.0 (NSP_CFG is important for control flow)
- γ = 1.0 (NSP_DFG is important for data flow)
- δ = 0.5 (ADP provides auxiliary supervision)
- ε = 0.5 (AOP provides auxiliary supervision)

### Data Preparation

For each training sample with two segments (sent1, sent2):

1. **Extract position information**:
   - sent1_binary_pos = mean(binary_pos for tokens in sent1)
   - sent2_binary_pos = mean(binary_pos for tokens in sent2)
   - Similarly for function_pos and bb_pos

2. **Generate ADP label**:
   ```python
   distance = abs(sent1_binary_pos - sent2_binary_pos)
   if distance < 0.01:
       adp_label = 0  # very_close
   elif distance < 0.05:
       adp_label = 1  # close
   elif distance < 0.2:
       adp_label = 2  # medium
   elif distance < 0.5:
       adp_label = 3  # far
   else:
       adp_label = 4  # very_far
   ```

3. **Generate AOP label**:
   ```python
   aop_label = 1 if sent1_binary_pos < sent2_binary_pos else 0
   ```

### Training Loop Modification

```python
# Forward pass
mlm_out, nsp_out, adp_out, aop_out = model(
    token_ids, segment_labels, 
    binary_pos, function_pos, bb_pos,
    corpus_type='cfg',
    return_address_tasks=True  # Enable address tasks
)

# Compute losses
mlm_loss = criterion_mlm(mlm_out, masked_tokens)
nsp_loss = criterion_nsp(nsp_out, nsp_labels)
adp_loss = criterion_adp(adp_out, adp_labels)
aop_loss = criterion_aop(aop_out, aop_labels)

# Combined loss
total_loss = mlm_loss + nsp_loss + 0.5 * adp_loss + 0.5 * aop_loss
```

## Expected Benefits

1. **Better Position Understanding**:
   - Model explicitly learns that position values encode distance and order
   - Address embeddings become more meaningful

2. **Improved Probing Results**:
   - Linear/non-linear probes should show stronger correlation between embeddings and positions
   - Model should better recover position information from embeddings

3. **Better Downstream Performance**:
   - Code similarity tasks benefit from understanding locality (similar code is often nearby)
   - Binary diffing benefits from understanding address order
   - Function boundary detection benefits from understanding distance patterns

## Implementation Status

✅ Model architecture updated with ADP and AOP heads
⏳ Data preparation script needed (generate ADP/AOP labels)
⏳ Training script modification needed (add new losses)
⏳ Evaluation script for address tasks

## Next Steps

1. Create data preparation script to add ADP/AOP labels to training data
2. Modify training script to include new losses
3. Train model with address-aware tasks
4. Re-run probing experiments to validate improvement
5. Evaluate on downstream tasks
