# Address-Aware Tasks: Regression Approach (CFG ONLY)

## Important: CFG Only, Not DFG

**Address-based tasks (ADP/AOP) are applied ONLY to CFG, not DFG.**

**Rationale**:
- **CFG sequences**: Follow control flow → instructions ARE continuous/nearby in memory
- **DFG sequences**: Follow data dependencies → instructions CAN BE scattered across memory
- **Conclusion**: Address distance/order is meaningful for CFG, meaningless for DFG

### Example:

**CFG trace** (continuous in memory):
```
0x401000: mov rax, 0      (bnorm=0.21, fnorm=0.00)
0x401004: test rax, rax   (bnorm=0.21, fnorm=0.15)  ← 4 bytes later
0x401008: je 0x401020     (bnorm=0.21, fnorm=0.30)  ← 4 bytes later
```
→ **ADP makes sense**: Distance between instructions ≈ 0.15 function units

**DFG trace** (scattered, following def-use chains):
```
0x401000: mov rax, [rbx]  (defines rax)
0x401234: add rbx, 8      (uses rbx, defines rbx)  ← 564 bytes later!
0x401008: test rax, rax   (uses rax)               ← back to 0x401008
```
→ **ADP is meaningless**: Addresses jump around following data dependencies, not memory layout

## Why Regression Instead of Classification?

### Problem with 5-Category Classification:
1. **Too Coarse**: Only 5 buckets for entire address space
2. **Loss of Information**: Can't distinguish "2 steps ahead" from "10 steps ahead" 
3. **Mismatch with Encoding**: Our distance encoding uses:
   - `fnorm` clipped to `[-3, +3]` (continuous)
   - `bbnorm` clipped to `[-10, +10]` (continuous)
4. **Ignores Sign**: Absolute distance loses direction information

### Benefits of Regression:
1. **Exact Distance Learning**: Model learns precise numerical relationships
2. **Matches Encoding**: Directly predicts the values used in position encoding
3. **Fine-Grained**: Can distinguish small differences in distance
4. **Preserves Sign**: Learns direction (forward/backward jumps)

## Task Definitions

### 1. Address Distance Prediction (ADP) - **REGRESSION**

**Goal**: Predict the signed distance between two instruction segments at 3 hierarchical levels.

**Output**: 3 continuous values
- `binary_dist`: Distance in binary-normalized space [0, 1]
- `fnorm_dist`: Distance in function-normalized units [-3, +3] (clipped)
- `bbnorm_dist`: Distance in BB-normalized units [-10, +10] (clipped)

**Architecture**:
```python
self.adp_head = nn.Sequential(
    nn.Linear(hidden, hidden),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden, 128),
    nn.ReLU(),
    nn.Linear(128, 3)  # 3 regression outputs
)
```

**Loss Function**: MSE (Mean Squared Error)
```python
# Ground truth labels
binary_dist_true = abs(sent2_binary_pos - sent1_binary_pos)
fnorm_dist_true = clip((sent2_func_pos - sent1_func_pos), -3, +3)
bbnorm_dist_true = clip((sent2_bb_pos - sent1_bb_pos), -10, +10)

# Compute loss
adp_loss = F.mse_loss(adp_output, torch.stack([binary_dist_true, fnorm_dist_true, bbnorm_dist_true], dim=1))
```

**Label Generation**:
```python
def generate_adp_labels(sent1_positions, sent2_positions):
    """
    sent1_positions: (binary_pos, function_pos, bb_pos) for sent1
    sent2_positions: (binary_pos, function_pos, bb_pos) for sent2
    
    Returns: [binary_dist, fnorm_dist, bbnorm_dist]
    """
    b1, f1, bb1 = sent1_positions
    b2, f2, bb2 = sent2_positions
    
    # Binary: absolute distance (always positive)
    binary_dist = abs(b2 - b1)
    
    # Function: signed distance, clipped
    fnorm_dist = max(-3.0, min(3.0, f2 - f1))
    
    # BB: signed distance, clipped
    bbnorm_dist = max(-10.0, min(10.0, bb2 - bb1))
    
    return [binary_dist, fnorm_dist, bbnorm_dist]
```

### 2. Address Order Prediction (AOP) - **BINARY CLASSIFICATION**

**Goal**: Predict if sent1 comes before sent2 in the binary.

**Output**: 2 classes
- Class 1: sent1 comes **before** sent2 (sent1_pos < sent2_pos)
- Class 0: sent1 comes **after** sent2 (sent1_pos >= sent2_pos)

**Architecture**:
```python
self.aop_head = nn.Sequential(
    nn.Linear(hidden, hidden),
    nn.Tanh(),
    nn.Linear(hidden, 2)  # 2-way classification
)
```

**Loss Function**: Cross-Entropy Loss
```python
aop_label = 1 if sent1_binary_pos < sent2_binary_pos else 0
aop_loss = F.cross_entropy(aop_output, aop_label)
```

**Label Generation**:
```python
def generate_aop_label(sent1_binary_pos, sent2_binary_pos):
    """
    Returns: 1 if sent1 before sent2, else 0
    """
    return 1 if sent1_binary_pos < sent2_binary_pos else 0
```

## Training Integration

### Multi-Task Loss (CFG-only for ADP/AOP)

```python
# Forward pass with address tasks
# === CFG: All tasks (MLM + NSP + ADP + AOP) ===
cfg_mlm, cfg_nsp, cfg_adp, cfg_aop = model(
    cfg_token_ids, cfg_segment_labels, 
    cfg_binary_pos, cfg_function_pos, cfg_bb_pos,
    corpus_type='cfg',
    return_address_tasks=True  # Returns ADP/AOP for CFG
)

# === DFG: NSP only (NO MLM, NO ADP, NO AOP) ===
_, dfg_nsp = model(
    dfg_token_ids, dfg_segment_labels,
    dfg_binary_pos, dfg_function_pos, dfg_bb_pos,
    corpus_type='dfg',
    return_address_tasks=False  # DFG doesn't use address tasks
)

# Compute individual losses
mlm_loss = F.cross_entropy(cfg_mlm.view(-1, vocab_size), cfg_masked_tokens.view(-1))
nsp_cfg_loss = F.cross_entropy(cfg_nsp, cfg_nsp_labels)
nsp_dfg_loss = F.cross_entropy(dfg_nsp, dfg_nsp_labels)
adp_loss = F.mse_loss(cfg_adp, cfg_adp_labels)  # MSE for regression (CFG only)
aop_loss = F.cross_entropy(cfg_aop, cfg_aop_labels)  # CFG only

# Combined loss
total_loss = (
    1.0 * mlm_loss +       # CFG: Primary task
    1.0 * nsp_cfg_loss +   # CFG: Control flow semantics
    1.0 * nsp_dfg_loss +   # DFG: Data dependency semantics
    0.5 * adp_loss +       # CFG only: Distance regression
    0.5 * aop_loss         # CFG only: Order classification
)
```

### Data Preparation Example

```python
# For a training sample with two segments
sent1_tokens = [...]  # First segment tokens
sent2_tokens = [...]  # Second segment tokens

# Extract average positions for each segment
sent1_binary_pos = sent1_positions[:, 0].mean()  # Average binary position
sent1_func_pos = sent1_positions[:, 1].mean()    # Average function position
sent1_bb_pos = sent1_positions[:, 2].mean()      # Average BB position

sent2_binary_pos = sent2_positions[:, 0].mean()
sent2_func_pos = sent2_positions[:, 1].mean()
sent2_bb_pos = sent2_positions[:, 2].mean()

# Generate ADP labels (regression targets)
adp_labels = generate_adp_labels(
    (sent1_binary_pos, sent1_func_pos, sent1_bb_pos),
    (sent2_binary_pos, sent2_func_pos, sent2_bb_pos)
)
# adp_labels = [binary_dist, fnorm_dist, bbnorm_dist]

# Generate AOP label (classification target)
aop_label = generate_aop_label(sent1_binary_pos, sent2_binary_pos)
# aop_label = 1 (before) or 0 (after)
```

## Expected Benefits

### 1. **Direct Alignment with Encoding**
- Model learns exact distance values used in position encoding
- No information loss from bucketing into categories
- Learns the clipping behavior ([-3, +3] and [-10, +10])

### 2. **Better Gradient Flow**
- MSE regression provides smooth gradients
- Model can learn fine-grained distance differences
- Classification would have discrete boundaries

### 3. **Improved Probing Results**
- After training with regression task, linear probes should show:
  - **Higher R² values** (stronger linear relationship)
  - Better recovery of position from embeddings
  - Model explicitly supervised to encode position numerically

### 4. **Interpretable Predictions**
- Can directly interpret model's distance predictions
- Easy to evaluate: "Is predicted distance close to actual distance?"
- Can visualize prediction errors vs. actual distances

## Evaluation Metrics

### For ADP (Regression):
```python
# Mean Absolute Error
mae_binary = torch.abs(adp_pred[:, 0] - adp_true[:, 0]).mean()
mae_fnorm = torch.abs(adp_pred[:, 1] - adp_true[:, 1]).mean()
mae_bbnorm = torch.abs(adp_pred[:, 2] - adp_true[:, 2]).mean()

# R² Score (correlation)
r2_binary = 1 - ((adp_pred[:, 0] - adp_true[:, 0])**2).sum() / ((adp_true[:, 0] - adp_true[:, 0].mean())**2).sum()
```

### For AOP (Classification):
```python
# Accuracy
accuracy = (aop_pred.argmax(dim=1) == aop_true).float().mean()
```

## Comparison with Classification

| Aspect | 5-Category Classification | Regression |
|--------|--------------------------|------------|
| **Granularity** | Coarse (5 buckets) | Fine-grained (continuous) |
| **Information Loss** | High (buckets lose precision) | None (exact values) |
| **Alignment with Encoding** | Poor (doesn't match clipping) | Perfect (matches encoding) |
| **Gradient Quality** | Discrete boundaries | Smooth gradients |
| **Interpretability** | Category names | Actual distance values |
| **Probing Performance** | Likely weak | Likely strong |

## Implementation Status

✅ Model updated with regression ADP head (3 outputs)
✅ AOP head added (2-class classification)
⏳ Data preparation: Generate continuous distance labels
⏳ Training script: Use MSE loss for ADP
⏳ Evaluation: Track MAE and R² for distance prediction

## Next Steps

1. **Modify dataloader** to compute distance labels (regression targets)
2. **Update training loop** to use MSE loss for ADP
3. **Train model** with new regression task
4. **Evaluate** distance prediction accuracy (MAE, R²)
5. **Re-run probing** to validate that position information is better encoded
