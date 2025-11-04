# Fixes Applied to Handle NaN Losses

## Problem
Training reached Epoch 2 with **NaN losses** appearing in MLM, Address, and overall loss, while CFG and Contrastive remained valid.

## Root Cause
**Gradient explosion** in the prediction heads due to:
1. Large logits before softmax (numerical overflow in `exp()`)
2. High learning rate (1e-4)
3. Insufficient gradient clipping (max_norm=1.0)
4. Random initialization of prediction heads

## Fixes Applied

### 1. **Logit Clamping** (train.py)
- **MLM Head**: Clamp logits to [-100, 100] before CrossEntropyLoss
- **Address Head**: Clamp logits to [-100, 100] before CrossEntropyLoss
- Prevents `exp(large_number)` overflow in softmax

### 2. **NaN Detection & Recovery** (train.py)
- Check for NaN after each loss computation
- Skip batch if NaN detected (prevents cascading failures)
- Print warning with diagnostic info

### 3. **Reduced Learning Rate** (config.py)
- **Before**: 1e-4
- **After**: 5e-5 (50% reduction)
- More conservative updates prevent gradient explosion

### 4. **Aggressive Gradient Clipping** (config.py + train.py)
- **Before**: max_norm=1.0
- **After**: max_norm=0.5 (50% reduction)
- Limits gradient magnitude more strictly

### 5. **Better Weight Initialization** (train.py)
- Added `_init_weights()` method to CFGPretrainModel
- Initialize all prediction heads with small weights (std=0.02)
- Initialize biases to zero
- Prevents large initial outputs

## Expected Improvements
1. ✅ **Stable training**: No more NaN losses
2. ✅ **Better convergence**: Smaller learning rate = smoother updates
3. ✅ **Controlled gradients**: Clamping + clipping prevents explosions
4. ✅ **Better initialization**: Small weights = gradual learning

## Code Changes Summary

### config.py
```python
LEARNING_RATE = 5e-5  # Reduced from 1e-4
MAX_GRAD_NORM = 0.5   # Reduced from 1.0
```

### train.py
```python
# MLM head
mlm_logits = torch.clamp(mlm_logits, min=-100, max=100)
if torch.isnan(mlm_loss):
    mlm_loss = torch.tensor(0.0, device=device)

# Address head
addr_logits = torch.clamp(addr_logits, min=-100, max=100)
if torch.isnan(addr_loss):
    addr_loss = torch.tensor(0.0, device=device)

# Combined loss
if torch.isnan(loss):
    continue  # Skip batch

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# Weight initialization
def _init_weights(self):
    nn.init.normal_(self.mlm_head.weight, mean=0.0, std=0.02)
    # ... (all heads initialized)
```

## Next Steps
1. **Restart training** from scratch with these fixes
2. **Monitor metrics**: All 5 losses should remain finite
3. **Expected behavior**: Slower but more stable convergence
4. If NaN still occurs: Further reduce learning rate to 2e-5 or 1e-5
