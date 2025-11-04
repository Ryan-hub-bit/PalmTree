# Task Switches - Flexible Pretraining Configuration

## Overview

You can now **enable/disable any pretraining task** by editing the `ENABLE_TASKS` dictionary in `config.py`.

## Configuration

### Location: `config.py`

```python
ENABLE_TASKS = {
    'mlm': True,            # Masked Language Modeling
    'cfg_prediction': True, # CFG edge prediction
    'addr_prediction': True,# Address type classification
    'contrastive': True     # Contrastive loss (stay close to PalmTree)
}

TASK_WEIGHTS = {
    'mlm': 1.0,
    'cfg_prediction': 1.0,
    'addr_prediction': 0.5,
    'contrastive': 0.3
}
```

## Available Tasks

### 1. **MLM (Masked Language Modeling)**
- **What it does:** Randomly masks 15% of tokens, model predicts original tokens
- **Purpose:** Learn token representations and context
- **When to disable:** If you only care about CFG structure, not token semantics

### 2. **CFG Prediction**
- **What it does:** Binary classification - predicts if BB2 follows BB1 in control flow
- **Purpose:** Learn control flow patterns
- **When to disable:** If you only care about token embeddings, not control flow

### 3. **Address Prediction**
- **What it does:** 4-class classification for address tokens (addr_start/end/code/data)
- **Purpose:** Learn to distinguish address types
- **When to disable:** If you don't care about address token types

### 4. **Contrastive Learning**
- **What it does:** Keeps non-address token embeddings close to PalmTree's semantic space
- **Purpose:** Preserve semantic knowledge from PalmTree
- **When to disable:** If you want embeddings to drift freely from PalmTree

## Common Configurations

### Standard BERT-style (MLM only)
```python
ENABLE_TASKS = {
    'mlm': True,
    'cfg_prediction': False,
    'addr_prediction': False,
    'contrastive': False
}
```

### CFG-focused (MLM + CFG)
```python
ENABLE_TASKS = {
    'mlm': True,
    'cfg_prediction': True,
    'addr_prediction': False,
    'contrastive': False
}
```

### Address-focused (MLM + Address)
```python
ENABLE_TASKS = {
    'mlm': True,
    'cfg_prediction': False,
    'addr_prediction': True,
    'contrastive': True  # Keep semantic space
}
```

### Free embeddings (no PalmTree constraint)
```python
ENABLE_TASKS = {
    'mlm': True,
    'cfg_prediction': True,
    'addr_prediction': True,
    'contrastive': False  # Let embeddings drift
}
```

### All tasks (default)
```python
ENABLE_TASKS = {
    'mlm': True,
    'cfg_prediction': True,
    'addr_prediction': True,
    'contrastive': True
}
```

## How It Works

When training starts, you'll see:

```
Starting training for 20 epochs...

================================================================================
Enabled Pretraining Tasks:
================================================================================
  mlm                 : ✓ ENABLED    (weight=1.0)
  cfg_prediction      : ✓ ENABLED    (weight=1.0)
  addr_prediction     : ✗ DISABLED   
  contrastive         : ✓ ENABLED    (weight=0.3)
================================================================================
```

### During Training:

- **Disabled tasks:** Loss set to 0.0, not computed, no gradient updates
- **Enabled tasks:** Computed normally with specified weights
- **Combined loss:** `total_loss = Σ(enabled_weight × enabled_loss)`

### Example:

If only MLM and CFG are enabled:
```python
loss = (1.0 × mlm_loss) + (1.0 × cfg_loss) + (0 × addr_loss) + (0 × contra_loss)
     = mlm_loss + cfg_loss
```

## Performance Impact

| Configuration | Speed | Memory | Use Case |
|--------------|-------|--------|----------|
| All tasks | Slowest | Highest | Best performance, full training |
| MLM only | Fastest | Lowest | Quick baseline, semantic only |
| MLM + CFG | Medium | Medium | Balance speed and CFG learning |
| No contrastive | Slightly faster | Same | When PalmTree constraint not needed |

## Tips

1. **Start with all tasks** to establish baseline
2. **Disable contrastive** if you want to explore different embedding spaces
3. **Disable CFG** if your dataset has poor CFG labels
4. **Disable Address** if you don't use address tokens in downstream tasks

## Validation Metrics

Disabled tasks will show `0.0000` in metrics:

```
Epoch 1 Summary:
  Train Loss: 8.3719 | MLM: 7.9647 | CFG: 0.6928 | Addr: 0.0000 | Contra: 0.1332
                                                        ↑
                                              addr_prediction disabled
```

This is **normal and expected** behavior! ✅
