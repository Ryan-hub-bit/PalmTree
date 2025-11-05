# Address-Aware BERT: Final Architecture

## Overview

This model **EXTENDS PalmTree** by adding hierarchical address-aware positional encodings while keeping all PalmTree components frozen.

## Architecture Summary

```
Input Tokens + Address Positions
         ↓
┌─────────────────────────────────────┐
│  FROZEN PALMTREE COMPONENTS         │
├─────────────────────────────────────┤
│  • Token Embeddings                 │
│  • Sequence Positional Encoding     │
│  • Segment Embeddings               │
│  • Transformer Blocks (×12)         │
└─────────────────────────────────────┘
         +
┌─────────────────────────────────────┐
│  NEW TRAINABLE COMPONENTS           │
├─────────────────────────────────────┤
│  • Address Positional Encoding      │
│    - Binary level (8dp)             │
│    - Function level (6dp)           │
│    - Basic block level (4dp)        │
│  • MLM Head                         │
│  • NSP Head                         │
└─────────────────────────────────────┘
```

## Embedding Components

### 1. Token Embedding (FROZEN - from PalmTree)
- Learned vocabulary representations
- Size: [vocab_size, hidden_dim]
- **Status: Frozen** - preserves PalmTree's token semantics

### 2. Sequence Positional Encoding (FROZEN - from PalmTree)
- Standard sinusoidal encoding: sin/cos(position/10000^(2i/d_model))
- Encodes token order in sequence: [0, 1, 2, 3, ...]
- **Status: Frozen** - standard transformer positional encoding

### 3. Address Positional Encoding (NEW - TRAINABLE)
- Hierarchical three-level encoding:
  - **Binary level**: Position in entire binary (8 decimal places)
  - **Function level**: Position within function (6 decimal places)
  - **Basic block level**: Position within BB (4 decimal places)
- Each level uses sin/cos encoding
- Learnable weights to balance importance of three levels
- **Status: Trainable** - this is the NEW component we're learning

### 4. Segment Embedding (FROZEN - from PalmTree)
- For NSP task: distinguishes two segments
- **Status: Frozen** - uses PalmTree's learned segment embeddings

## What Gets Trained?

### FROZEN (from PalmTree):
```python
✗ embedding.token_embedding.weight          # Token representations
✗ embedding.sequence_position.pe            # Sequence order encoding  
✗ embedding.segment_embedding.weight        # Segment labels
✗ transformer_blocks[*]                     # All transformer layers
```

### TRAINABLE (NEW):
```python
✓ embedding.address_position.level_weights  # Balance 3 address levels
✓ embedding.address_position.*              # All address encoding params
✓ mlm_head.*                                # Masked language model head
✓ nsp_head.*                                # Next sentence prediction head
```

## Training Process

1. **Load Pre-trained PalmTree** checkpoint
2. **Freeze** all PalmTree components automatically
3. **Initialize** address positional encodings randomly
4. **Train** only address encodings + task heads
5. **Result**: PalmTree + Address-Awareness

## Example Usage

```bash
# Train with frozen PalmTree
python3 train.py \
    --cfg_train data/cfg_output.txt \
    --dfg_train data/dfg_output.txt \
    --vocab vocab.pkl \
    --palmtree_checkpoint ../pre-trained_model/palmtree/transformer.ep19 \
    --cuda
```

## Expected Output

```
Loading pre-trained PalmTree from: ../pre-trained_model/palmtree/transformer.ep19
[INFO] Checkpoint from epoch: 19
[INFO] Extracted token embedding: torch.Size([50000, 768])
[INFO] Extracted segment embedding: torch.Size([2, 768])
[INFO] Extracted 144 transformer parameters

[INFO] Loading pre-trained token embeddings: torch.Size([50000, 768])
[INFO] Token embeddings FROZEN
[INFO] Sequence positional embeddings (standard sin/cos) - FROZEN
[INFO] Address positional embeddings (3-level) - TRAINABLE
[INFO] Loading pre-trained segment embeddings: torch.Size([2, 768])
[INFO] Loading pre-trained transformer blocks
[INFO] Transformer blocks FROZEN

Model Statistics:
  Total parameters: 123,456,789
  Trainable parameters: 5,678,910  (~4-5% of total)
  Frozen parameters: 117,777,879   (~95-96% of total)
```

## Key Differences from Original PalmTree

| Component | PalmTree | Address-Aware BERT |
|-----------|----------|-------------------|
| Token Embedding | Learned | **Reused (frozen)** |
| Sequence Position | Sin/cos on token index | **Reused (frozen)** |
| Address Position | ❌ None | **✓ NEW (3-level sin/cos)** |
| Segment Embedding | Learned | **Reused (frozen)** |
| Transformer | Learned | **Reused (frozen)** |
| MLM Head | Learned | **Re-initialized** |
| NSP Head | Learned | **Re-initialized** |

## Data Format

Input requires both sequence order AND address information:

```
mov(0x401008:0.21068702:0.296296:0.4444) rax qword [...]
│           │         │        │        │
│           │         │        │        └─ BB position (4dp)
│           │         │        └─ Function position (6dp)
│           │         └─ Binary position (8dp)
│           └─ Actual address
└─ Opcode token
```

Model sees:
- **Token**: `mov` → PalmTree token embedding (frozen)
- **Sequence**: position 0 → PalmTree sequence encoding (frozen)
- **Address**: (0.21068702, 0.296296, 0.4444) → NEW address encoding (trainable)
- **Segment**: 0 or 1 → PalmTree segment embedding (frozen)

## Benefits

1. **Leverage PalmTree's Knowledge**: Pre-trained token and transformer representations
2. **Add Structural Awareness**: Learn where instructions appear in binary hierarchy
3. **Efficient Training**: Only ~5% of parameters need training
4. **Fast Convergence**: Start from strong pre-trained representations
5. **Better Generalization**: Frozen PalmTree prevents overfitting on small datasets

## Files

- `address_embedding.py`: Four-component embedding (token + seq_pos + addr_pos + segment)
- `model.py`: Address-aware BERT with automatic freezing
- `load_pretrained.py`: Load and extract PalmTree components
- `train.py`: Training script with PalmTree support
- `train_with_palmtree.sh`: Example training command

## Next Steps

After training, you'll have a model that:
- Understands tokens (from PalmTree)
- Understands sequence order (from PalmTree)
- **NEW**: Understands address hierarchy (binary → function → basic block)
- Can be used for address-aware downstream tasks

This creates a **NEW MODEL** that combines PalmTree's language understanding with structural address awareness!
