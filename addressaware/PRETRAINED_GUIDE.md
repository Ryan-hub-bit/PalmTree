# Using Pre-trained PalmTree with Address-Aware BERT

This guide explains how to initialize Address-Aware BERT with pre-trained PalmTree weights.

## Overview

The Address-Aware BERT can be initialized with pre-trained PalmTree weights:
- **Token Embeddings**: Use PalmTree's learned token representations
- **Transformer Blocks**: Use PalmTree's pre-trained transformer layers
- **Address Positional Embeddings**: NEW component (trained from scratch)

This approach allows you to leverage PalmTree's learned representations while adding address-aware capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Address-Aware BERT Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: Tokens + (binary_pos, function_pos, bb_pos)                │
│     ↓                                                                │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Token Embedding (from pre-trained PalmTree) [OPTIONAL FREEZE]│   │
│  └────────────────────────────────────────────────────────────┘   │
│     +                                                                │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Address Positional Embedding (NEW - train from scratch)    │   │
│  │   - Sin/Cos encoding on binary_pos (8dp)                   │   │
│  │   - Sin/Cos encoding on function_pos (6dp)                 │   │
│  │   - Sin/Cos encoding on bb_pos (4dp)                       │   │
│  └────────────────────────────────────────────────────────────┘   │
│     +                                                                │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Segment Embedding                                           │   │
│  └────────────────────────────────────────────────────────────┘   │
│     ↓                                                                │
│  Layer Norm + Dropout                                               │
│     ↓                                                                │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Transformer Blocks x N (from PalmTree) [OPTIONAL FREEZE]   │   │
│  └────────────────────────────────────────────────────────────┘   │
│     ↓                                                                │
│  ┌──────────────┐  ┌──────────────┐                               │
│  │  MLM Head    │  │  NSP Head    │                               │
│  └──────────────┘  └──────────────┘                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Training Strategies

### Strategy 1: Fine-tune Everything (Default)
Train all components including pre-trained weights.

```bash
python3 train.py \
    --cfg_train data.txt \
    --dfg_train data.txt \
    --vocab vocab.pkl \
    --palmtree_checkpoint ../pre-trained_model/palmtree/transformer.ep19 \
    --cuda
```

**Pros**: Maximum flexibility, can adapt fully to address-aware task
**Cons**: Requires more data, risk of forgetting pre-trained knowledge

### Strategy 2: Freeze Token Embeddings
Keep token embeddings fixed, only train address components and transformer.

```bash
python3 train.py \
    --palmtree_checkpoint ../pre-trained_model/palmtree/transformer.ep19 \
    --freeze_token_emb \
    ...
```

**Pros**: Preserves token semantics learned by PalmTree
**Cons**: Less flexibility in token representations

### Strategy 3: Freeze Transformer Blocks
Keep transformer frozen, only train embeddings and task heads.

```bash
python3 train.py \
    --palmtree_checkpoint ../pre-trained_model/palmtree/transformer.ep19 \
    --freeze_transformer \
    ...
```

**Pros**: Very fast training, good when data is limited
**Cons**: Cannot adapt transformer to address patterns

### Strategy 4: Freeze Both (Feature Extraction)
Use PalmTree as feature extractor, only train address embeddings and heads.

```bash
python3 train.py \
    --palmtree_checkpoint ../pre-trained_model/palmtree/transformer.ep19 \
    --freeze_token_emb \
    --freeze_transformer \
    ...
```

**Pros**: Fastest training, minimal data needed
**Cons**: Least flexibility

## Quick Start

### 1. Prepare Data
Generate address-aware training data:
```bash
cd ../scripts
./generate_data.sh
```

### 2. Train with Pre-trained PalmTree
```bash
cd ../addressaware
./train_with_palmtree.sh
```

Edit `train_with_palmtree.sh` to configure:
- Path to PalmTree checkpoint
- Freezing options
- Hyperparameters

### 3. Monitor Training
Training logs show:
- Which components are frozen
- Number of trainable vs frozen parameters
- MLM and NSP losses
- Validation metrics (if validation set provided)

## Expected Behavior

### Loading Pre-trained Weights
```
[INFO] Loading PalmTree checkpoint from: ../pre-trained_model/palmtree/transformer.ep19
[INFO] Checkpoint from epoch: 19
[INFO] Extracted token embedding: torch.Size([50000, 768])
[INFO] Extracted 144 transformer parameters
[INFO] Loading pre-trained token embeddings: torch.Size([50000, 768])
[INFO] Loading pre-trained transformer blocks
```

### With Freezing
```
[INFO] Freezing token embeddings
[INFO] Freezing transformer blocks
[INFO] Model Statistics:
  Total parameters: 123,456,789
  Trainable parameters: 12,345,678
  Frozen parameters: 111,111,111
```

## Files

- **`load_pretrained.py`**: Utility to load PalmTree checkpoints
- **`train_with_palmtree.sh`**: Example training script with pre-trained weights
- **`model.py`**: Updated to accept pre-trained components
- **`address_embedding.py`**: Updated to accept pre-trained token embeddings

## Troubleshooting

### Vocabulary Mismatch
If vocabularies don't match:
```
Error: Token embedding size mismatch
```
Solution: Use the same vocabulary as PalmTree or rebuild vocabulary.

### Shape Mismatch
If model dimensions don't match:
```
Error: Hidden size mismatch
```
Solution: Set `--hidden`, `--layers`, `--attn_heads` to match PalmTree checkpoint.

### Checkpoint Not Found
```
Error: PalmTree checkpoint not found
```
Solution: Check path to transformer.ep19 file.

## Best Practices

1. **Start with Strategy 1** (fine-tune everything) if you have sufficient data
2. **Use Strategy 3 or 4** if data is limited
3. **Monitor validation loss** to detect overfitting
4. **Compare with from-scratch training** to measure benefit of pre-training
5. **Save checkpoints frequently** to recover from training issues

## Performance Tips

- Use `--multi_gpu` if available for faster training
- Adjust `--batch_size` based on GPU memory
- Use `--num_workers 4` for faster data loading
- Lower `--lr` when fine-tuning to preserve pre-trained weights

## Next Steps

After training:
1. Evaluate on downstream tasks (function similarity, vulnerability detection)
2. Compare embeddings with/without address awareness
3. Analyze what address patterns the model learns
4. Fine-tune for specific applications
